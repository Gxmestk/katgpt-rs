//! Certified frontier — a monotone, provably-sound safe-set expansion operator.
//!
//! Source: [Research 510](../../../.research/510_ActFlow_Certified_Frontier_Expansion.md)
//! distilling De Santi et al., *Active Flow Expansion for Out-of-Distribution
//! Discovery*, [arXiv:2606.08802](https://arxiv.org/abs/2606.08802). The
//! operator itself is the SAFEOPT lineage (Sui et al. 2015/2018) — known art,
//! shipped here the way this repo ships bandits: the operator is standard, the
//! fusion (grow-then-navigate with [`crate::viable_manifold_graph`]) and the
//! modelless framing are the contribution.
//!
//! Plan 580 Phase 1. Feature `certified_frontier`, std-only, zero deps.
//!
//! # What this is
//!
//! Given (a) a buffer of **binary verifier outcomes** per latent cell, (b) a
//! closed-form uncertainty model, and (c) an a-priori **Lipschitz budget**,
//! grow a set of cells that provably satisfy `p(z) >= h`, and answer two
//! questions the caller cannot answer alone:
//!
//! - *Where do I look next?* — [`CertifiedFrontier::acquire_frontier_target`]
//! - *When do I stop looking?* — [`should_advance`]
//!
//! Everything is GD-free: no training, no backprop. Weight-free by
//! construction — the "model" is a Beta-Bernoulli count per cell plus an
//! optional linear-kernel posterior variance.
//!
//! # Phase 0 measured this before it was built
//!
//! [Bench 687](../../../.benchmarks/687_certified_frontier_phase0_poc.md):
//! zero soundness violations, monotone growth, and **51.4x** separation of
//! frontier acquisition over passive sampling at an identical query budget.
//!
//! The same PoC measured something the plan did not ask, and it shapes this
//! API (T0.3): **the Lipschitz dilation is conditional, not free.** A hop is
//! admissible iff `best_cb - h >= L * spacing`, so on a coarse lattice
//! [`CertifiedFrontier::reachability_dilation`] relaxes and certifies
//! *nothing*, silently. That is why [`CertifiedFrontier::dilation_feasibility`]
//! is a first-class, cheap predicate rather than a debug aid: a caller must be
//! able to see a dead dilation without instrumenting a run. Measured crossover
//! (dense world, 6 000 queries, 0 violations throughout): 16x16 and 32x32
//! certify 0 cells by dilation; 64x64 certifies 6; 96x96 certifies 30 of 113
//! (27%). The predicted and observed crossovers agree on all four points.
//!
//! The cause is that a *global* Lipschitz constant charges a plateau hop the
//! steepest-cliff price — and the paper's `L = L_s * L_g` is global in exactly
//! the same way, so this is not an artifact of the Beta substitute. Hence
//! [`FrontierCell::lipschitz`]: a caller with a tighter **a-priori** bound for
//! a region supplies it per cell, and hops pay `max(L_from, L_to)`.
//!
//! # Soundness contract (read before setting `lipschitz`)
//!
//! `cfg.lipschitz` and `FrontierCell::lipschitz` MUST be **a-priori upper
//! bounds** on the local Lipschitz constant of `p` in probability space. They
//! are the one input this module cannot check. Estimating `L` from the same
//! observations that drive expansion is **unsound** — a too-small `L` makes
//! [`CertifiedFrontier::reachability_dilation`] certify cells that are not
//! valid, and no test in this module can see it. The uncertainty model is
//! conservative; the Lipschitz budget is the caller's proof obligation.

use core::f32;

/// Lipschitz constant of the logistic sigmoid — `sup |s'(z)| = s'(0) = 1/4`.
///
/// The `L_s` of the paper's `beta_t` schedule and of `L = L_s * L_g`.
pub const SIGMOID_LIPSCHITZ: f32 = 0.25;

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

#[inline]
fn dot<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn sq_dist<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ── configuration ──────────────────────────────────────────────────────────

/// Static configuration for one certified-frontier run.
///
/// `lipschitz` is a proof obligation, not a tuning knob — see the module-level
/// soundness contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontierConfig {
    /// Ridge regulariser of the kernel posterior (`lambda` in `(K + lambda I)`).
    pub lambda: f32,
    /// Failure probability of the confidence schedule.
    pub delta: f32,
    /// RKHS-norm bound `B` on the latent score field `g`.
    pub b_rkhs: f32,
    /// Validity threshold: a cell is valid iff `p(z) >= h`.
    pub h: f32,
    /// Global a-priori Lipschitz bound on `p` in probability space.
    pub lipschitz: f32,
    /// Acquisition inflation factor (paper's Eq 14 `alpha`); `1.0` = Eq 33.
    pub alpha: f32,
    /// Nearest-neighbour spacing of the cell lattice. Used ONLY by
    /// [`CertifiedFrontier::dilation_feasibility`] to price a representative
    /// hop; the dilation itself uses exact pairwise distances.
    pub cell_spacing: f32,
    /// Cells within this distance of a certified cell are acquisition
    /// candidates. Set to `0.0` to sample strictly inside the certified set.
    pub acquire_radius: f32,
    /// Target certified-bound precision, the `epsilon` of the halting law.
    pub epsilon: f32,
}

impl Default for FrontierConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            delta: 0.05,
            b_rkhs: 1.0,
            h: 0.6,
            lipschitz: 1.0,
            alpha: 1.0,
            cell_spacing: 1.0,
            acquire_radius: 1.5,
            epsilon: 0.05,
        }
    }
}

/// Why a dilation pass will (or will not) admit anything — the T0.3 predicate.
///
/// A dead dilation is otherwise invisible: `reachability_dilation` returns `0`
/// both when the frontier is complete and when every hop is unaffordable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DilationFeasibility {
    /// `max(cb - h)` over certified cells: the best bound headroom available
    /// to pay for a hop. Negative/`-inf` when nothing is certified.
    pub best_headroom: f32,
    /// `L * cell_spacing`: what one representative lattice hop costs.
    pub hop_cost: f32,
    /// `best_headroom >= hop_cost`.
    pub feasible: bool,
    /// `hop_cost - best_headroom`. Positive means the dilation is a no-op and
    /// the certified set can only grow by querying.
    pub deficit: f32,
}

// ── cells ──────────────────────────────────────────────────────────────────

/// One latent cell: its feature, its verifier tally, and its certified bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontierCell<const D: usize> {
    /// Latent coordinate. Distances between these drive the dilation.
    pub feat: [f32; D],
    /// Count of `true` verifier outcomes.
    pub valid: u32,
    /// Count of `false` verifier outcomes.
    pub invalid: u32,
    /// Monotone non-decreasing lower bound on `p(z)`. Never assigned a smaller
    /// value — this is what makes the certified set monotone (Lemma E.2).
    pub cb: f32,
    /// `cb >= cfg.h` has held at least once.
    pub certified: bool,
    /// Local a-priori Lipschitz bound. `NaN` (the default) falls back to
    /// `cfg.lipschitz`. Must be an upper bound; see the soundness contract.
    pub lipschitz: f32,
    /// Externally supplied posterior sd, e.g. from [`PosteriorBuffer`]. `NaN`
    /// (the default) uses the Beta-Bernoulli sd.
    pub sigma_override: f32,
    /// `true` once the cell was admitted by a Lipschitz hop rather than by its
    /// own observations. Pure bookkeeping — this is the T0.3 attribution, and
    /// it must be read at the moment `cb` crosses `h`, never at end-state.
    pub by_dilation: bool,
}

impl<const D: usize> FrontierCell<D> {
    /// A fresh, unobserved, uncertified cell at `feat`.
    #[must_use]
    pub fn new(feat: [f32; D]) -> Self {
        Self {
            feat,
            valid: 0,
            invalid: 0,
            cb: 0.0,
            certified: false,
            lipschitz: f32::NAN,
            sigma_override: f32::NAN,
            by_dilation: false,
        }
    }
}

impl<const D: usize> Default for FrontierCell<D> {
    fn default() -> Self {
        Self::new([0.0; D])
    }
}

// ── closed-form pieces (plan functions 2, 3, 7) ────────────────────────────

/// Beta-Bernoulli posterior mean and **variance** from a verifier tally.
///
/// Laplace prior `Beta(1, 1)`, so `a = valid + 1`, `b = invalid + 1`:
/// `mu = a / (a + b)`, `var = a b / ((a+b)^2 (a+b+1))`.
///
/// This is the plan's honest substitute for the paper's kernel-logistic
/// `mu_t`, which needs a convex solve. Exact, allocation-free, and monotone in
/// the tally — the properties the soundness proof actually uses.
#[inline]
#[must_use]
pub fn beta_mean_variance(valid: u32, invalid: u32) -> (f32, f32) {
    let a = valid as f32 + 1.0;
    let b = invalid as f32 + 1.0;
    let n = a + b;
    let mean = a / n;
    let var = (a * b) / (n * n * (n + 1.0));
    (mean, var)
}

/// Maximum information gain of a linear kernel in `d` dimensions after `t`
/// observations: `gamma_t = d * ln(1 + t / (d * lambda))`.
///
/// Sub-linear in `t` — the plateau that bounds how long the halting law can
/// keep a caller querying one cell.
#[inline]
#[must_use]
pub fn linear_information_gain(t: u32, d: usize, lambda: f32) -> f32 {
    let d = d.max(1) as f32;
    d * (1.0 + t as f32 / (d * lambda.max(f32::EPSILON))).ln()
}

/// The paper's Eq 31/37 confidence width
/// `beta_t = 4 L_s B + 2 L_s sqrt(2 kappa / lambda * (gamma_t + ln(1/delta)))`
/// with `L_s = 1/4` and `kappa = 1 / (s(B) (1 - s(B)))` closed-form for the
/// sigmoid link.
///
/// Monotone non-decreasing in `t` (pinned by test) — that monotonicity is what
/// lets the union bound cover every round, which in turn is what lets `cb` be
/// a running max without breaking soundness.
#[inline]
#[must_use]
pub fn confidence_schedule(t: u32, delta: f32, lambda: f32, b_rkhs: f32, d_eff: usize) -> f32 {
    let s_b = sigmoid(b_rkhs);
    let kappa = 1.0 / (s_b * (1.0 - s_b)).max(f32::EPSILON);
    let gamma = linear_information_gain(t, d_eff, lambda);
    let delta = delta.clamp(f32::EPSILON, 1.0);
    let inner = 2.0 * kappa / lambda.max(f32::EPSILON) * (gamma + (1.0 / delta).ln());
    4.0 * SIGMOID_LIPSCHITZ * b_rkhs + 2.0 * SIGMOID_LIPSCHITZ * inner.max(0.0).sqrt()
}

/// The halting law: a certified hop is guaranteed once `sigma <= eps / (2 beta)`.
///
/// Answers *when do I stop looking at this cell* — the counterpart to
/// [`CertifiedFrontier::acquire_frontier_target`]'s *where do I look next*.
#[inline]
#[must_use]
pub fn should_advance(sigma: f32, beta: f32, epsilon: f32) -> bool {
    sigma <= epsilon / (2.0 * beta.max(f32::EPSILON))
}

/// Round budget implied by the halting law: `T ~ 8 alpha^2 beta^2 gamma / eps^2`.
///
/// A planning figure — how many queries a caller should expect to spend before
/// [`should_advance`] fires.
#[inline]
#[must_use]
pub fn advance_horizon(alpha: f32, beta: f32, gamma: f32, epsilon: f32) -> f32 {
    let e = epsilon.max(f32::EPSILON);
    8.0 * alpha * alpha * beta * beta * gamma / (e * e)
}

// ── Prop 1 design bounds (ship beside, per plan) ───────────────────────────

/// Fraction of the unit sphere in `m` dimensions inside a cap of half-angle
/// `phi`, in the exponential form the plan pre-registers:
/// `exp(-(m - 1) cos^2(phi) / 2)`.
///
/// This is the design law behind Phase 0's measured 51.4x: a narrow valid
/// corridor is exponentially hard to hit by passive sampling, so targeted
/// acquisition separates exponentially in the ambient dimension.
#[inline]
#[must_use]
pub fn spherical_cap_bound(m: usize, phi_rad: f32) -> f32 {
    let m = m.max(1) as f32;
    let c = phi_rad.cos();
    (-(m - 1.0) * c * c / 2.0).exp()
}

/// Laurent-Massart chi-square deviation radius:
/// `sqrt(d + 2 sqrt(d ln(1/delta)) + 2 ln(1/delta))`.
///
/// The concentration radius of a `d`-dimensional isotropic Gaussian — the
/// honest "how far out does a sample land" companion to
/// [`spherical_cap_bound`].
#[inline]
#[must_use]
pub fn laurent_massart_radius(d: usize, delta: f32) -> f32 {
    let d = d as f32;
    let l = (1.0 / delta.clamp(f32::EPSILON, 1.0)).ln();
    (d + 2.0 * (d * l).sqrt() + 2.0 * l).max(0.0).sqrt()
}

// ── diversity / coverage scoreboards (plan function 8) ─────────────────────

/// Capacity of [`sphere_exclusion_coverage`]'s alloc-free center list.
pub const SPHERE_EXCLUSION_MAX_CENTERS: usize = 256;

/// Outcome of a sphere-exclusion scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SphereExclusion {
    /// Number of accepted centers.
    pub centers: usize,
    /// `true` when the scan hit [`SPHERE_EXCLUSION_MAX_CENTERS`] and stopped
    /// accepting. A saturated count is a floor, not a measurement — raise the
    /// threshold or subsample rather than comparing two saturated runs.
    pub saturated: bool,
}

/// Greedy sphere-exclusion cluster count at `threshold`.
///
/// Order-pinned: the greedy scan runs in slice order, so a fixed input order
/// gives a bit-identical count. That determinism is the point — this is a
/// scoreboard for A/B runs, not a clustering algorithm.
///
/// Alloc-free, so the center list is capped at
/// [`SPHERE_EXCLUSION_MAX_CENTERS`]; saturation is reported rather than
/// silently truncating the count.
#[must_use]
pub fn sphere_exclusion_coverage<const D: usize>(
    samples: &[[f32; D]],
    threshold: f32,
) -> SphereExclusion {
    let t2 = threshold * threshold;
    let mut centers = 0usize;
    // Indices of accepted centers, tracked in-place over `samples`.
    let mut accepted = [0usize; SPHERE_EXCLUSION_MAX_CENTERS];
    for (i, s) in samples.iter().enumerate() {
        let covered = accepted[..centers]
            .iter()
            .any(|&c| sq_dist(s, &samples[c]) <= t2);
        if covered {
            continue;
        }
        if centers == SPHERE_EXCLUSION_MAX_CENTERS {
            return SphereExclusion {
                centers,
                saturated: true,
            };
        }
        accepted[centers] = i;
        centers += 1;
    }
    SphereExclusion {
        centers,
        saturated: false,
    }
}

/// Vendi score `exp(-sum lambda_i ln lambda_i)` on a kernel's eigenvalues.
///
/// Eigenvalues are normalised to sum 1 first; zero/negative entries are
/// skipped (`0 ln 0 = 0`). Returns `0.0` for an empty or degenerate spectrum.
#[must_use]
pub fn vendi_diversity(eigs: &[f32]) -> f32 {
    let total: f32 = eigs.iter().filter(|e| **e > 0.0).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &e in eigs {
        if e > 0.0 {
            let p = e / total;
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}

// ── exact Eq-10 posterior variance (plan function 1) ───────────────────────

/// Fixed-capacity observation buffer carrying an **incremental Cholesky** of
/// `(K + lambda I)` for the linear kernel `k(a, b) = <a, b>`.
///
/// `append_observation` extends the factor by one row in `O(n^2)`; there is
/// never a re-solve and never an explicit inverse.
///
/// # Size
///
/// The factor is `MAX_OBS * MAX_OBS` floats — 256 KiB at `MAX_OBS = 256`. That
/// is fine on a main thread but will overflow a small spawned-thread stack;
/// box it (`Box::new(PosteriorBuffer::new(..))`) when `MAX_OBS > 256`.
#[derive(Debug, Clone)]
pub struct PosteriorBuffer<const MAX_OBS: usize, const D: usize> {
    feats: [[f32; D]; MAX_OBS],
    y: [f32; MAX_OBS],
    /// Lower-triangular Cholesky factor of `(K + lambda I)`.
    chol: [[f32; MAX_OBS]; MAX_OBS],
    /// `(K + lambda I)^-1 y`, refreshed on append.
    alpha: [f32; MAX_OBS],
    scratch: [f32; MAX_OBS],
    n: usize,
    lambda: f32,
}

impl<const MAX_OBS: usize, const D: usize> PosteriorBuffer<MAX_OBS, D> {
    /// Empty buffer with ridge `lambda`.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        Self {
            feats: [[0.0; D]; MAX_OBS],
            y: [0.0; MAX_OBS],
            chol: [[0.0; MAX_OBS]; MAX_OBS],
            alpha: [0.0; MAX_OBS],
            scratch: [0.0; MAX_OBS],
            n: 0,
            lambda,
        }
    }

    /// Number of stored observations.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` when no observation has been appended.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Solve `L v = rhs` in place over `scratch[..n]` (forward substitution).
    #[inline]
    fn forward_substitute(chol: &[[f32; MAX_OBS]; MAX_OBS], out: &mut [f32], n: usize) {
        for i in 0..n {
            let mut acc = out[i];
            for j in 0..i {
                acc -= chol[i][j] * out[j];
            }
            out[i] = acc / chol[i][i];
        }
    }

    /// Solve `L^T v = rhs` in place over `out[..n]` (back substitution).
    #[inline]
    fn back_substitute(chol: &[[f32; MAX_OBS]; MAX_OBS], out: &mut [f32], n: usize) {
        for i in (0..n).rev() {
            let mut acc = out[i];
            for j in (i + 1)..n {
                acc -= chol[j][i] * out[j];
            }
            out[i] = acc / chol[i][i];
        }
    }

    /// Append one observation, extending the Cholesky factor by a rank-1 row.
    ///
    /// Returns `false` (and changes nothing) when the buffer is full.
    pub fn append_observation(&mut self, feat: &[f32; D], y: f32) -> bool {
        if self.n >= MAX_OBS {
            return false;
        }
        let n = self.n;
        // w = L^-1 k(X, x_new)
        let feats = &self.feats;
        for (s, f) in self.scratch[..n].iter_mut().zip(feats.iter()) {
            *s = dot(f, feat);
        }
        Self::forward_substitute(&self.chol, &mut self.scratch, n);
        let mut rem = self.lambda + dot(feat, feat);
        let scratch = &self.scratch;
        for (c, w) in self.chol[n][..n].iter_mut().zip(scratch.iter()) {
            *c = *w;
            rem -= *w * *w;
        }
        // `lambda > 0` keeps this strictly positive; the max is a numerical
        // floor for the near-duplicate-feature case, not a correctness patch.
        self.chol[n][n] = rem.max(self.lambda * 1e-6).sqrt();
        self.feats[n] = *feat;
        self.y[n] = y;
        self.n = n + 1;
        self.refresh_alpha();
        true
    }

    fn refresh_alpha(&mut self) {
        let n = self.n;
        self.alpha[..n].copy_from_slice(&self.y[..n]);
        Self::forward_substitute(&self.chol, &mut self.alpha, n);
        Self::back_substitute(&self.chol, &mut self.alpha, n);
    }

    /// Eq 10 exactly:
    /// `sigma^2(x) = k(x,x) - k(x,X) (K + lambda I)^-1 k(X,x)`.
    ///
    /// One forward substitution, `O(n^2)`, no allocation.
    #[must_use]
    pub fn posterior_variance_linear(&self, x: &[f32; D], scratch: &mut [f32]) -> f32 {
        let n = self.n;
        debug_assert!(scratch.len() >= n, "scratch must hold at least len() floats");
        let k_self = dot(x, x);
        if n == 0 {
            return k_self;
        }
        for (s, f) in scratch[..n].iter_mut().zip(self.feats.iter()) {
            *s = dot(f, x);
        }
        Self::forward_substitute(&self.chol, scratch, n);
        let quad: f32 = scratch[..n].iter().map(|v| v * v).sum();
        (k_self - quad).max(0.0)
    }

    /// Ridge posterior mean `k(x, X) (K + lambda I)^-1 y`, `O(n D)` off the
    /// cached `alpha`.
    #[must_use]
    pub fn ridge_mean(&self, x: &[f32; D]) -> f32 {
        self.feats[..self.n]
            .iter()
            .zip(self.alpha[..self.n].iter())
            .map(|(f, a)| dot(f, x) * a)
            .sum()
    }
}

// ── the frontier (plan functions 4, 5, 6 + the type) ───────────────────────

/// Fixed-capacity certified cell set. Zero-allocation by construction.
#[derive(Debug, Clone)]
pub struct CertifiedFrontier<const MAX_CELLS: usize, const D: usize> {
    cells: [FrontierCell<D>; MAX_CELLS],
    /// Pre-hop `cb` snapshot, so one `reachability_dilation` pass is exactly
    /// one hop and cannot chain through cells certified within the same pass.
    hop_cb: [f32; MAX_CELLS],
    len: usize,
    certified: u32,
    /// Cells certified by a Lipschitz hop rather than by their own tally.
    dilated: u32,
}

impl<const MAX_CELLS: usize, const D: usize> Default for CertifiedFrontier<MAX_CELLS, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_CELLS: usize, const D: usize> CertifiedFrontier<MAX_CELLS, D> {
    /// An empty frontier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cells: [FrontierCell::default(); MAX_CELLS],
            hop_cb: [0.0; MAX_CELLS],
            len: 0,
            certified: 0,
            dilated: 0,
        }
    }

    /// Register a cell. Returns its index, or `None` when at capacity.
    pub fn push_cell(&mut self, feat: [f32; D]) -> Option<usize> {
        if self.len >= MAX_CELLS {
            return None;
        }
        let i = self.len;
        self.cells[i] = FrontierCell::new(feat);
        self.len = i + 1;
        Some(i)
    }

    /// Number of registered cells.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when no cell has been registered.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Registered cells, in insertion order.
    #[inline]
    #[must_use]
    pub fn cells(&self) -> &[FrontierCell<D>] {
        &self.cells[..self.len]
    }

    /// Mutable access to one cell — for `lipschitz` / `sigma_override`.
    #[inline]
    pub fn cell_mut(&mut self, i: usize) -> Option<&mut FrontierCell<D>> {
        (i < self.len).then(|| &mut self.cells[i])
    }

    /// How many cells are certified.
    #[inline]
    #[must_use]
    pub fn certified_count(&self) -> u32 {
        self.certified
    }

    /// How many certified cells were admitted by a Lipschitz hop.
    ///
    /// Counted at the moment `cb` crosses `h`. Do **not** re-derive this at
    /// end-state from "certified but never queried": a frontier policy hands
    /// maximum posterior sigma to a freshly dilated cell and queries it moments
    /// later, so the end-state count reads `0` even where dilation did 27% of
    /// the work (Bench 687 T0.3).
    #[inline]
    #[must_use]
    pub fn dilated_count(&self) -> u32 {
        self.dilated
    }

    /// Seed a cell as certified from caller-side a-priori knowledge.
    ///
    /// Sets `cb` to `h` — the weakest bound consistent with "known valid", so
    /// a seed buys no free dilation headroom it did not earn.
    pub fn seed_certified(&mut self, i: usize, cfg: &FrontierConfig) -> bool {
        if i >= self.len || self.cells[i].certified {
            return false;
        }
        self.cells[i].cb = self.cells[i].cb.max(cfg.h);
        self.cells[i].certified = true;
        self.certified += 1;
        true
    }

    /// Record one binary verifier outcome against a cell.
    pub fn observe(&mut self, i: usize, valid: bool) -> bool {
        if i >= self.len {
            return false;
        }
        match valid {
            true => self.cells[i].valid += 1,
            false => self.cells[i].invalid += 1,
        }
        true
    }

    /// Posterior sd used for this cell: `sigma_override` when finite, else the
    /// Beta-Bernoulli sd.
    #[inline]
    #[must_use]
    pub fn sigma(&self, i: usize) -> f32 {
        let c = &self.cells[i];
        match c.sigma_override.is_finite() {
            true => c.sigma_override,
            false => beta_mean_variance(c.valid, c.invalid).1.sqrt(),
        }
    }

    /// Lower confidence bound `mu - beta * sigma`, clamped to `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn lcb(&self, i: usize, beta: f32) -> f32 {
        let c = &self.cells[i];
        let (mean, _) = beta_mean_variance(c.valid, c.invalid);
        (mean - beta * self.sigma(i)).clamp(0.0, 1.0)
    }

    /// Upper confidence bound `mu + beta * sigma`, clamped to `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn ucb(&self, i: usize, beta: f32) -> f32 {
        let c = &self.cells[i];
        let (mean, _) = beta_mean_variance(c.valid, c.invalid);
        (mean + beta * self.sigma(i)).clamp(0.0, 1.0)
    }

    /// Refresh every cell's `sigma_override` from a kernel posterior.
    ///
    /// Opt-in: the default path is the Beta sd, which is what Phase 0 gated on.
    pub fn refresh_kernel_sigma<const MAX_OBS: usize>(
        &mut self,
        buf: &PosteriorBuffer<MAX_OBS, D>,
        scratch: &mut [f32],
    ) {
        for i in 0..self.len {
            let feat = self.cells[i].feat;
            self.cells[i].sigma_override = buf.posterior_variance_linear(&feat, scratch).sqrt();
        }
    }

    /// **Eq 32 — the certified-set update.** Raise every `cb` to its LCB and
    /// certify whatever crosses `h`. Returns the number newly certified.
    ///
    /// `cb` moves by `max`, so the certified set is monotone across *any*
    /// query sequence (T2.3). Soundness rests on `beta` covering every round,
    /// which is what [`confidence_schedule`]'s monotonicity in `t` buys.
    pub fn expand_certified(&mut self, cfg: &FrontierConfig, beta: f32) -> u32 {
        let mut newly = 0;
        for i in 0..self.len {
            let lcb = self.lcb(i, beta);
            let c = &mut self.cells[i];
            if lcb > c.cb {
                c.cb = lcb;
            }
            if !c.certified && c.cb >= cfg.h {
                c.certified = true;
                newly += 1;
            }
        }
        self.certified += newly;
        newly
    }

    /// Effective Lipschitz cost of the hop `i -> j`: `max(L_i, L_j)`, falling
    /// back to `cfg.lipschitz` for any cell without a local bound.
    #[inline]
    fn hop_lipschitz(&self, i: usize, j: usize, cfg: &FrontierConfig) -> f32 {
        let li = match self.cells[i].lipschitz.is_finite() {
            true => self.cells[i].lipschitz,
            false => cfg.lipschitz,
        };
        let lj = match self.cells[j].lipschitz.is_finite() {
            true => self.cells[j].lipschitz,
            false => cfg.lipschitz,
        };
        li.max(lj)
    }

    /// **The T0.3 predicate.** Can a hop be afforded at all right now?
    ///
    /// Cheap (`O(n)`) and meant to be called before every dilation: a coarse
    /// lattice makes [`Self::reachability_dilation`] a silent no-op, and the
    /// return value of that call cannot distinguish "nothing left to admit"
    /// from "nothing was ever affordable".
    ///
    /// `feasible` is **necessary, not sufficient**: it prices the single best
    /// headroom against one representative lattice hop, so `!feasible`
    /// guarantees a dilation admits nothing, while `feasible` only means some
    /// hop is affordable *if* an uncertified cell sits that close to the cell
    /// holding the headroom.
    #[must_use]
    pub fn dilation_feasibility(&self, cfg: &FrontierConfig) -> DilationFeasibility {
        let mut best = f32::NEG_INFINITY;
        let mut min_l = f32::INFINITY;
        for i in 0..self.len {
            if self.cells[i].certified {
                best = best.max(self.cells[i].cb - cfg.h);
                let li = match self.cells[i].lipschitz.is_finite() {
                    true => self.cells[i].lipschitz,
                    false => cfg.lipschitz,
                };
                min_l = min_l.min(li);
            }
        }
        let l = match min_l.is_finite() {
            true => min_l,
            false => cfg.lipschitz,
        };
        let hop_cost = l * cfg.cell_spacing;
        DilationFeasibility {
            best_headroom: best,
            hop_cost,
            feasible: best >= hop_cost,
            deficit: hop_cost - best,
        }
    }

    /// **Eq 15 — one Lipschitz reachability hop per iteration.**
    ///
    /// Admits `z` when some certified `z'` has `cb(z') - L d(z, z') >= h`. The
    /// relaxed bound `cb(z') - L d` is written into `cb(z)` by `max`, so
    /// dilation is monotone exactly like [`Self::expand_certified`].
    ///
    /// `hop_budget` passes are run; each pass reads a pre-hop snapshot, so a
    /// cell certified in pass `k` can only extend the set in pass `k + 1`.
    /// Returns the number newly certified across all passes.
    ///
    /// Cost is `O(hop_budget * certified * uncertified * D)`. Call
    /// [`Self::dilation_feasibility`] first — on a coarse lattice this whole
    /// loop is guaranteed to admit nothing.
    pub fn reachability_dilation(&mut self, cfg: &FrontierConfig, hop_budget: u32) -> u32 {
        let mut newly = 0;
        for _ in 0..hop_budget {
            for j in 0..self.len {
                self.hop_cb[j] = self.cells[j].cb;
            }
            let mut admitted = 0;
            for j in 0..self.len {
                if self.cells[j].certified {
                    continue;
                }
                let mut best = self.hop_cb[j];
                for i in 0..self.len {
                    if !self.cells[i].certified {
                        continue;
                    }
                    let d = sq_dist(&self.cells[i].feat, &self.cells[j].feat).sqrt();
                    let cand = self.hop_cb[i] - self.hop_lipschitz(i, j, cfg) * d;
                    if cand > best {
                        best = cand;
                    }
                }
                if best > self.cells[j].cb {
                    self.cells[j].cb = best;
                }
                if best >= cfg.h {
                    self.cells[j].certified = true;
                    self.cells[j].by_dilation = true;
                    admitted += 1;
                }
            }
            self.certified += admitted;
            self.dilated += admitted;
            newly += admitted;
            if admitted == 0 {
                break;
            }
        }
        newly
    }

    /// **Eq 33 — safe uncertainty sampling.** `argmax sigma` over certified
    /// cells and cells within `cfg.acquire_radius` of one.
    ///
    /// Ties break to the lowest index, so a fixed cell order gives a
    /// deterministic query sequence. `cfg.acquire_radius = 0.0` restricts the
    /// search to the certified set itself (the strict Eq-33 reading); the
    /// wider default is the policy that measured 51.4x in Phase 0, because
    /// restricting to certified cells makes growth depend entirely on a
    /// dilation that a coarse lattice cannot afford.
    ///
    /// `cfg.alpha` scales the returned cell's sigma threshold only through
    /// [`should_advance`]; acquisition itself is scale-free.
    #[must_use]
    pub fn acquire_frontier_target(&self, cfg: &FrontierConfig) -> Option<usize> {
        let r2 = cfg.acquire_radius * cfg.acquire_radius;
        let mut best: Option<(usize, f32)> = None;
        for j in 0..self.len {
            let candidate = self.cells[j].certified
                || (r2 > 0.0
                    && (0..self.len).any(|i| {
                        self.cells[i].certified
                            && sq_dist(&self.cells[i].feat, &self.cells[j].feat) <= r2
                    }));
            if !candidate {
                continue;
            }
            let s = self.sigma(j);
            if best.is_none_or(|(_, bs)| s > bs) {
                best = Some((j, s));
            }
        }
        best.map(|(j, _)| j)
    }

    /// Straddling gate: is querying this cell decision-relevant at all?
    ///
    /// `true` only when the threshold lies inside the cell's confidence band
    /// after paying for one hop — deep-inside and far-outside cells prune to
    /// zero queries. The EVPI-shaped companion to acquisition (Plan 580 T4.2).
    #[must_use]
    pub fn query_is_decision_relevant(&self, i: usize, cfg: &FrontierConfig, beta: f32) -> bool {
        if i >= self.len {
            return false;
        }
        let lcb = self.lcb(i, beta);
        let ucb = self.ucb(i, beta);
        let l = match self.cells[i].lipschitz.is_finite() {
            true => self.cells[i].lipschitz,
            false => cfg.lipschitz,
        };
        lcb - l * cfg.cell_spacing < cfg.h && cfg.h <= ucb
    }
}
