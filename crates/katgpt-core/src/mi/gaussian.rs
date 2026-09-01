//! Closed-form Gaussian MI arm, gated by the shipped `sketched_gaussianity`
//! probe (Plan 583 T3.1/T3.2).
//!
//! For jointly Gaussian `(X, Y)` the MI has the textbook closed form
//! `I = ½[ln det Σ_x + ln det Σ_y − ln det Σ]` with
//! `Σ = [[Σ_x, Σ_xy], [Σ_yx, Σ_y]]` — the only arm in the module with a
//! ground-truth-accurate value. The price is the distributional assumption,
//! so the arm is **gated**: the joint population must pass the shipped
//! projection-normality probe (`data_probe::gaussianity::sketched_gaussianity`
//! — consumed, not re-implemented; the Mardia alternative is the documented
//! `[-]` defer). A failed gate returns [`NotGaussian`], which routes the
//! caller to the critic/perm arms — it is NEVER silently swallowed into a
//! number.
//!
//! The gate is load-bearing by construction and pinned by T3.2: the `Y = X²`
//! control (strictly dependent, zero correlation, non-Gaussian joint) and a
//! heavy-tail fixture MUST return `NotGaussian`; Gaussian fixtures must
//! reproduce the analytic value to 1e-3 nats.

use super::MiNats;
use crate::data_probe::gaussianity::{GaussianityReport, GaussianityScratch, sketched_gaussianity};

/// Gate threshold on the `GaussianityReport.score ∈ (0,1)` sigmoid aggregate.
/// Calibrated by T3.2: Gaussian populations at audit sizes (n ≥ 4096) score
/// well above; the `Y = X²` and heavy-tail controls score well below.
pub const GAUSSIAN_GATE_THRESHOLD: f32 = 0.5;

/// Why the Gaussian arm refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NotGaussian {
    /// The gaussianity gate fired (`score ≤ threshold`).
    GateFired { score: f32, threshold: f32 },
    /// Population too small for a non-singular covariance (n ≤ d + 1).
    TooFewSamples { n: usize, d: usize },
    /// The sample covariance was not positive definite (degenerate
    /// population — also routed here rather than silently swallowed).
    NotPositiveDefinite,
}

impl NotGaussian {
    /// The gate score if the refusal came from the gate (diagnostic for the
    /// tuple).
    #[must_use]
    pub fn score(&self) -> Option<f32> {
        match *self {
            Self::GateFired { score, .. } => Some(score),
            _ => None,
        }
    }
}

/// Streaming covariance accumulator (Welford mean + outer-product M2) —
/// O(N·d²) streaming; allocations only at construction (G4).
#[derive(Clone, Debug)]
pub struct CovAccumulator {
    n: u64,
    d: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
    delta: Vec<f64>,
}

impl CovAccumulator {
    #[must_use]
    pub fn new(d: usize) -> Self {
        assert!(d > 0, "d must be positive");
        Self {
            n: 0,
            d,
            mean: vec![0.0; d],
            m2: vec![0.0; d * d],
            delta: vec![0.0; d],
        }
    }

    /// Dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Rows pushed so far.
    #[must_use]
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Push one row (first `d` entries consumed). Three passes over d with
    /// the stored `delta` scratch — the classic vector Welford update
    /// `M2 += δ_old ⊗ δ_new` (all δ_new vs the POST-update mean).
    pub fn push(&mut self, row: &[f32]) {
        assert!(row.len() >= self.d, "row shorter than d");
        let d = self.d;
        self.n += 1;
        let n = self.n as f64;
        for ((dv, rv), mv) in self.delta.iter_mut().zip(row.iter()).zip(self.mean.iter()) {
            *dv = f64::from(*rv) - *mv;
        }
        for (mv, dv) in self.mean.iter_mut().zip(self.delta.iter()) {
            *mv += dv / n;
        }
        for (i, doi) in self.delta.iter().enumerate() {
            let m2r = &mut self.m2[i * d..i * d + d];
            for ((mj, rv), mv) in m2r.iter_mut().zip(row.iter()).zip(self.mean.iter()) {
                *mj += doi * (f64::from(*rv) - *mv);
            }
        }
    }

    /// Write the sample covariance (row-major d×d, denominator n−1) into
    /// `out`. Requires n ≥ 2.
    pub fn covariance_into(&self, out: &mut [f64]) {
        assert!(self.n >= 2, "covariance needs n ≥ 2");
        assert!(out.len() >= self.d * self.d, "out too small");
        let denom = (self.n - 1) as f64;
        for (o, &m) in out.iter_mut().zip(self.m2.iter()) {
            *o = m / denom;
        }
    }

    /// Closed-form Gaussian MI from THIS accumulator's covariance:
    /// `I = ½[ln det Σx + ln det Σy − ln det Σ]`. The eigendecomposition
    /// workspaces live in `arm` — no allocation in steady state.
    ///
    /// # Errors
    /// [`NotGaussian::TooFewSamples`] / [`NotGaussian::NotPositiveDefinite`].
    pub fn mi_gaussian(
        &self,
        dx: usize,
        dy: usize,
        arm: &mut GaussianArmScratch,
    ) -> Result<MiNats, NotGaussian> {
        let d = self.d;
        assert_eq!(d, dx + dy, "accumulator dim must equal dx+dy");
        if self.n <= (d as u64) + 1 {
            return Err(NotGaussian::TooFewSamples {
                n: self.n as usize,
                d,
            });
        }
        let denom = (self.n - 1) as f64;
        arm.ensure(d);
        // Stage Σx.
        for i in 0..dx {
            for j in 0..dx {
                arm.sub[i * dx + j] = self.m2[i * d + j] / denom;
            }
        }
        let lx = logdet_staged(arm, dx)?;
        // Stage Σy.
        for i in 0..dy {
            for j in 0..dy {
                arm.sub[i * dy + j] = self.m2[(dx + i) * d + (dx + j)] / denom;
            }
        }
        let ly = logdet_staged(arm, dy)?;
        // Stage the full joint.
        for v in 0..d * d {
            arm.sub[v] = self.m2[v] / denom;
        }
        let ljoint = logdet_staged(arm, d)?;
        Ok(MiNats::from_nats((0.5 * (lx + ly - ljoint)) as f32))
    }
}

/// Eigendecomposition workspaces for the logdets (grow-once, G4).
pub struct GaussianArmScratch {
    pub(crate) eigvals: Vec<f64>,
    pub(crate) eigvecs: Vec<f64>,
    pub(crate) sub: Vec<f64>,
    pub(crate) eig: crate::linalg::SymmetricEigScratch,
}

impl Default for GaussianArmScratch {
    fn default() -> Self {
        Self {
            eigvals: Vec::new(),
            eigvecs: Vec::new(),
            sub: Vec::new(),
            eig: crate::linalg::SymmetricEigScratch::new(),
        }
    }
}

impl GaussianArmScratch {
    /// Grow to handle a joint dimension of `d` (idempotent).
    pub fn ensure(&mut self, d: usize) {
        if self.eigvals.len() < d {
            self.eigvals.resize(d, 0.0);
        }
        if self.eigvecs.len() < d * d {
            self.eigvecs.resize(d * d, 0.0);
        }
        if self.sub.len() < d * d {
            self.sub.resize(d * d, 0.0);
        }
        self.eig.ensure_capacity(d);
    }
}

/// `ln det` of the symmetric PSD d×d matrix currently staged in
/// `arm.sub[..d²]`, via the shipped `linalg::symmetric_eig`:
/// `Σ ln(clamp(λᵢ, λ_max·1e−12))`. Non-finite / non-positive spectra beyond
/// the floor → [`NotGaussian::NotPositiveDefinite`].
fn logdet_staged(arm: &mut GaussianArmScratch, d: usize) -> Result<f64, NotGaussian> {
    arm.ensure(d);
    // symmetric_eig requires EXACTLY-length slices (it debug-asserts
    // len == n) — slice the grown workspaces down to d / d².
    crate::linalg::symmetric_eig(
        &mut arm.eigvals[..d],
        &mut arm.eigvecs[..d * d],
        &arm.sub[..d * d],
        &mut arm.eig,
        d,
        64,
    );
    let mut lambda_max = f64::NEG_INFINITY;
    for &l in arm.eigvals.iter().take(d) {
        if !l.is_finite() {
            return Err(NotGaussian::NotPositiveDefinite);
        }
        lambda_max = lambda_max.max(l);
    }
    if lambda_max <= 0.0 {
        return Err(NotGaussian::NotPositiveDefinite);
    }
    let floor = (lambda_max * 1e-12).max(f64::MIN_POSITIVE);
    let mut acc = 0.0f64;
    for &l in arm.eigvals.iter().take(d) {
        acc += l.max(floor).ln();
    }
    Ok(acc)
}

/// Closed-form Gaussian MI from an explicit joint (dx+dy)×(dx+dy) sample
/// covariance: `I = ½[ln det Σx + ln det Σy − ln det Σ]`.
///
/// # Errors
/// [`NotGaussian::NotPositiveDefinite`] on a singular/indefinite covariance.
pub fn mi_from_cov(
    cov: &[f64],
    dx: usize,
    dy: usize,
    arm: &mut GaussianArmScratch,
) -> Result<MiNats, NotGaussian> {
    let d = dx + dy;
    assert!(cov.len() >= d * d, "cov too small");
    arm.ensure(d);
    for i in 0..dx {
        for j in 0..dx {
            arm.sub[i * dx + j] = cov[i * d + j];
        }
    }
    let lx = logdet_staged(arm, dx)?;
    for i in 0..dy {
        for j in 0..dy {
            arm.sub[i * dy + j] = cov[(dx + i) * d + (dx + j)];
        }
    }
    let ly = logdet_staged(arm, dy)?;
    arm.sub[..d * d].copy_from_slice(&cov[..d * d]);
    let ljoint = logdet_staged(arm, d)?;
    Ok(MiNats::from_nats((0.5 * (lx + ly - ljoint)) as f32))
}

/// The gated Gaussian arm: gate the joint population with the shipped
/// `sketched_gaussianity` probe, then evaluate the closed form.
///
/// `joint_buf` must have capacity `n·(dx+dy)` (scratch — filled with the
/// `[xᵢ ‖ yᵢ]` rows); `g_scratch` must be built for `(n, dx+dy, seed)`. The
/// covariance is (re)accumulated inside this call unless the accumulator
/// already holds exactly `n` rows (steady-state reuse). A fired gate returns
/// [`NotGaussian::GateFired`] — the caller routes to the critic/perm arms.
///
/// # Errors
/// [`NotGaussian`] — never silently swallowed.
#[allow(clippy::too_many_arguments)]
pub fn mi_gaussian_gated(
    x: &[f32],
    y: &[f32],
    n: usize,
    dx: usize,
    dy: usize,
    cov: &mut CovAccumulator,
    joint_buf: &mut [f32],
    g_scratch: &mut GaussianityScratch,
    arm: &mut GaussianArmScratch,
) -> Result<MiNats, NotGaussian> {
    let d = dx + dy;
    assert!(
        x.len() >= n * dx && y.len() >= n * dy,
        "population too small"
    );
    assert!(joint_buf.len() >= n * d, "joint_buf too small");
    assert_eq!(g_scratch.n_samples(), n, "g_scratch n mismatch");
    assert_eq!(g_scratch.dim(), d, "g_scratch dim mismatch");
    assert_eq!(cov.dim(), d, "cov accumulator dimension");
    for i in 0..n {
        joint_buf[i * d..i * d + dx].copy_from_slice(&x[i * dx..(i + 1) * dx]);
        joint_buf[i * d + dx..(i + 1) * d].copy_from_slice(&y[i * dy..(i + 1) * dy]);
    }
    let report: GaussianityReport = sketched_gaussianity(joint_buf, g_scratch);
    if report.score <= GAUSSIAN_GATE_THRESHOLD {
        return Err(NotGaussian::GateFired {
            score: report.score,
            threshold: GAUSSIAN_GATE_THRESHOLD,
        });
    }
    if cov.n() as usize != n {
        // (Re)accumulate for exactly this population.
        *cov = CovAccumulator::new(d);
        for i in 0..n {
            cov.push(&joint_buf[i * d..(i + 1) * d]);
        }
    }
    cov.mi_gaussian(dx, dy, arm)
}

/// Analytic Gaussian MI: `dep · −½·ln(1−ρ²)` — the ground truth for the
/// gates.
#[must_use]
pub fn mi_gaussian_analytic(rho: f32, dep: usize) -> f64 {
    let r = f64::from(rho);
    -0.5 * dep as f64 * (1.0 - r * r).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mi::test_support::{gaussian_pairs, gaussian_pairs_dep, splitmix};

    #[test]
    fn gaussian_fixture_passes_gate_and_matches_analytic() {
        // 2-D joint (dx=dy=1, ρ=0.5): truth ≈ 0.1438 nats. The sample-MI
        // standard error is ≈ (1−ρ²)/√n ≈ 0.001 at n = 524288 — the plan's
        // 1e-3 accuracy claim needs that scale (measured deviation was
        // 0.0044 at n = 8192, exactly 1 SE of the smaller fixture).
        let n = 524_288;
        let rho = 0.5f32;
        let (x, y) = gaussian_pairs(rho, n, 2024);
        let mut cov = CovAccumulator::new(2);
        let mut joint_buf = vec![0.0f32; n * 2];
        let mut g = GaussianityScratch::new(n, 2, 11);
        let mut arm = GaussianArmScratch::default();
        let mi = mi_gaussian_gated(&x, &y, n, 1, 1, &mut cov, &mut joint_buf, &mut g, &mut arm)
            .expect("Gaussian fixture must pass the gate");
        let truth = mi_gaussian_analytic(rho, 1);
        eprintln!(
            "gaussian arm: MI = {:.6} vs analytic {truth:.6} (err = {:.6})",
            mi.nats(),
            (f64::from(mi.nats()) - truth).abs()
        );
        assert!(
            (f64::from(mi.nats()) - truth).abs() < 1e-3,
            "MI {} vs analytic {truth}",
            mi.nats()
        );
        // The gate score recorded on the passing fixture (calibration
        // evidence for the 0.5 threshold).
        let score = sketched_gaussianity(&joint_buf, &mut g).score;
        eprintln!("gaussian fixture gate score = {score} (threshold {GAUSSIAN_GATE_THRESHOLD})");
    }

    #[test]
    fn t32_yx2_control_must_fire_the_gate() {
        let mut rng = splitmix(4242);
        let n = 8192;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let gx = rng.normal();
            x.push(gx);
            y.push(gx * gx);
        }
        let mut cov = CovAccumulator::new(2);
        let mut joint_buf = vec![0.0f32; n * 2];
        let mut g = GaussianityScratch::new(n, 2, 12);
        let mut arm = GaussianArmScratch::default();
        let err = mi_gaussian_gated(&x, &y, n, 1, 1, &mut cov, &mut joint_buf, &mut g, &mut arm)
            .expect_err("Y = X² must be refused by the gate");
        assert!(
            matches!(err, NotGaussian::GateFired { .. }),
            "expected GateFired, got {err:?}"
        );
        assert!(err.score().unwrap_or(1.0) < GAUSSIAN_GATE_THRESHOLD);
    }

    #[test]
    fn t32_heavy_tail_control_must_fire_the_gate() {
        // x Gaussian, y = 0.5·x + 0.5·Pareto-tail noise (1/u): the joint has
        // a heavy tail the projection-KS probe must catch.
        let mut rng = splitmix(9090);
        let n = 8192;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let gx = rng.normal();
            let u = ((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
            let heavy = 1.0 / u - 1.0; // Pareto(1) tail
            x.push(gx);
            y.push(0.5 * gx + 0.5 * heavy as f32);
        }
        let mut cov = CovAccumulator::new(2);
        let mut joint_buf = vec![0.0f32; n * 2];
        let mut g = GaussianityScratch::new(n, 2, 13);
        let mut arm = GaussianArmScratch::default();
        let err = mi_gaussian_gated(&x, &y, n, 1, 1, &mut cov, &mut joint_buf, &mut g, &mut arm)
            .expect_err("heavy-tail fixture must be refused by the gate");
        assert!(matches!(err, NotGaussian::GateFired { .. }), "got {err:?}");
        eprintln!(
            "heavy-tail gate score = {:.4} (threshold {GAUSSIAN_GATE_THRESHOLD})",
            err.score().unwrap_or(f32::NAN)
        );
    }

    #[test]
    fn multi_dim_gaussian_matches_analytic() {
        // d = 8, dep = 4 dependent dims at ρ = 0.6: truth = 4·(−½ln(0.64)).
        // Sample-MI SE ≈ dep·ρ/√(n(1−ρ²)) ≈ 0.009 at n = 65536 ⇒ tolerance
        // 0.02 (fixed seed, deterministic).
        let n = 65_536;
        let rho = 0.6f32;
        let dep = 4;
        let (x, y) = gaussian_pairs_dep(rho, n, 8, dep, 3141);
        let mut cov = CovAccumulator::new(16);
        let mut joint_buf = vec![0.0f32; n * 16];
        let mut g = GaussianityScratch::new(n, 16, 14);
        let mut arm = GaussianArmScratch::default();
        let mi = mi_gaussian_gated(&x, &y, n, 8, 8, &mut cov, &mut joint_buf, &mut g, &mut arm)
            .expect("multi-dim Gaussian must pass the gate");
        let truth = mi_gaussian_analytic(rho, dep);
        assert!(
            (f64::from(mi.nats()) - truth).abs() < 0.02,
            "MI {} vs analytic {truth}",
            mi.nats()
        );
    }

    #[test]
    fn cov_accumulator_matches_two_pass_covariance() {
        let n = 1000;
        let (x, y) = gaussian_pairs(0.3, n, 77);
        let mut cov = CovAccumulator::new(2);
        for i in 0..n {
            cov.push(&[x[i], y[i]]);
        }
        let mut out = vec![0.0f64; 4];
        cov.covariance_into(&mut out);
        let mx = x.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;
        let my = y.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;
        let sxx = x.iter().map(|v| (f64::from(*v) - mx).powi(2)).sum::<f64>() / (n - 1) as f64;
        let syy = y.iter().map(|v| (f64::from(*v) - my).powi(2)).sum::<f64>() / (n - 1) as f64;
        let sxy = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (f64::from(*a) - mx) * (f64::from(*b) - my))
            .sum::<f64>()
            / (n - 1) as f64;
        assert!((out[0] - sxx).abs() < 1e-9);
        assert!((out[3] - syy).abs() < 1e-9);
        assert!((out[1] - sxy).abs() < 1e-9);
        assert!((out[2] - sxy).abs() < 1e-9);
    }

    #[test]
    fn too_few_samples_is_reported_not_guessed() {
        let n = 4;
        let (x, y) = gaussian_pairs(0.5, n, 5);
        // d = 16 joint from 4 rows ⇒ singular ⇒ the covariance path refuses
        // (gate may pass on tiny n — either way it must not return Ok).
        let mut cov = CovAccumulator::new(16);
        let mut joint_buf = vec![0.0f32; n * 16];
        let mut g = GaussianityScratch::new(n, 16, 15);
        let mut arm = GaussianArmScratch::default();
        let mut x8 = vec![0.0f32; n * 8];
        let mut y8 = vec![0.0f32; n * 8];
        for i in 0..n {
            for j in 0..8 {
                x8[i * 8 + j] = x[i];
                y8[i * 8 + j] = y[i];
            }
        }
        let r = mi_gaussian_gated(
            &x8,
            &y8,
            n,
            8,
            8,
            &mut cov,
            &mut joint_buf,
            &mut g,
            &mut arm,
        );
        assert!(r.is_err());
    }

    #[test]
    fn singular_covariance_refuses_not_positive_definite() {
        // Perfectly collinear 2-D joint ⇒ singular ⇒ NotPositiveDefinite.
        let n = 256;
        let x = gaussian_pairs(0.999, n, 8).0;
        let mut cov = CovAccumulator::new(2);
        let mut joint_buf = vec![0.0f32; n * 2];
        let mut g = GaussianityScratch::new(n, 2, 16);
        let mut arm = GaussianArmScratch::default();
        // y = x exactly (degenerate Gaussian joint) — the gate may pass (the
        // joint IS Gaussian), then the covariance is exactly singular.
        let y_exact: Vec<f32> = x.clone();
        let r = mi_gaussian_gated(
            &x,
            &y_exact,
            n,
            1,
            1,
            &mut cov,
            &mut joint_buf,
            &mut g,
            &mut arm,
        );
        match r {
            Err(NotGaussian::NotPositiveDefinite | NotGaussian::GateFired { .. }) => {}
            other => panic!("expected refusal, got {other:?}"),
        }
    }
}
