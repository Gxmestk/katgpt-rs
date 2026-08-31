//! Donsker–Varadhan bound evaluation over fixed critic scores.
//!
//! `I(X;Y) ≥ E_P[T] − log E_Q[e^T]` with `P` = the joint (identity pairing)
//! and `Q` = the product (permuted pairing). Three λ-family members from ONE
//! score pass:
//!
//! - **λ = 0 — plug-in** ([`DvReport::l0`]): `mean(joint) − logmeanexp(perm)`.
//!   The classic form; biased **upward** on the null at small N
//!   (`bias ≲ C·dof/N`, see the `mi` module docs).
//! - **leave-one-out** ([`DvReport::loo`]): each joint score is compared
//!   against the logmeanexp of the permutation scores **excluding itself**
//!   (`total − self`, O(1) per i after one O(N) sum) — removes the dominant
//!   self-inclusion bias term. The default reported value.
//! - **λ = 1 — NWJ form** ([`DvReport::l1`]): `mean(joint) + 1 − mean(e^perm)`
//!   (Nguyen–Wainwright–Jordan, arXiv:0809.0853). Lower variance, higher
//!   bias at high MI (dies ≳ 3 nats per the follow-up literature); the λ = 1
//!   member of the DV↔NWJ interpolation (Poole et al., arXiv:1905.06922).
//!
//! All log-mean-exp terms are **max-subtracted** — no overflow, no softmax
//! over the batch. Permutation draws come from the scratch's BLAKE3-seeded
//! RNG for run-to-run bit-determinism ([`dv_bound_perm_average`]).
//!
//! [`dv_report`] also exports **spread**: the between-fold standard deviation
//! (8 contiguous folds) of the per-fold λ=0 estimate. Large spread is the
//! honest signal that the DV variance pathology has taken over (per-dependent-
//! dim ρ > 1/√3 ⇒ the Q-term has divergent variance); ship it with the value.

use super::MiScratch;

/// Number of contiguous folds used for the between-fold spread estimate.
pub const DV_FOLDS: usize = 8;

/// DV-family bound report. Fields are f32 per plan; internal math is f64.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DvReport {
    /// λ = 0 plug-in bound: `mean(joint) − logmeanexp(perm)`.
    pub l0: f32,
    /// Leave-one-out bound — the default headline value (the self-inclusion
    /// bias is removed).
    pub loo: f32,
    /// λ = 1 NWJ form: `mean(joint) + 1 − mean(e^perm)`.
    pub l1: f32,
    /// Between-fold std-dev of the per-fold λ=0 estimate (8 folds). The
    /// "is this number trustworthy" axis — ship it, never a bare value.
    pub spread: f32,
}

/// λ = 0 plug-in DV bound from pre-computed scores.
#[must_use]
pub fn dv_plug_in(scores_joint: &[f64], scores_perm: &[f64]) -> f32 {
    let n = scores_joint.len().min(scores_perm.len());
    assert!(n > 0, "empty scores");
    (mean(scores_joint, n) - logmeanexp(scores_perm, n)) as f32
}

/// Leave-one-out DV bound from pre-computed scores.
///
/// `loo = mean_i[joint_i − logmeanexp_{j≠i}(perm_j)]` where the exclusion is
/// `lse_all + ln(1 − e^{p_i − m}) − ln(n−1)` in the max-subtracted domain —
/// O(1) per element after one O(N) pass. The (pathological, but guarded)
/// case where sample *i* carries the entire exponent mass clamps to the f64
/// tiny floor instead of producing −inf.
#[must_use]
pub fn dv_loo(scores_joint: &[f64], scores_perm: &[f64]) -> f32 {
    let n = scores_joint.len().min(scores_perm.len());
    assert!(n > 1, "LOO needs n ≥ 2");
    let m = max_of(scores_perm, n);
    let mut total = 0.0f64;
    for &p in &scores_perm[..n] {
        total += (p - m).exp();
    }
    let lm = f64::ln(n as f64 - 1.0);
    let mut acc = 0.0f64;
    for i in 0..n {
        let e_i = (scores_perm[i] - m).exp();
        let rest = (total - e_i).max(f64::MIN_POSITIVE);
        let lse_i = m + rest.ln() - lm;
        acc += scores_joint[i] - lse_i;
    }
    (acc / n as f64) as f32
}

/// λ = 1 NWJ form from pre-computed scores.
#[must_use]
pub fn nwj(scores_joint: &[f64], scores_perm: &[f64]) -> f32 {
    let n = scores_joint.len().min(scores_perm.len());
    assert!(n > 0, "empty scores");
    let mut q = 0.0f64;
    for &p in &scores_perm[..n] {
        q += p.exp();
    }
    (mean(scores_joint, n) + 1.0 - q / n as f64) as f32
}

/// Full report (all three λ members + fold spread) from one score pass.
#[must_use]
pub fn dv_report(scores_joint: &[f64], scores_perm: &[f64]) -> DvReport {
    DvReport {
        l0: dv_plug_in(scores_joint, scores_perm),
        loo: dv_loo(scores_joint, scores_perm),
        l1: nwj(scores_joint, scores_perm),
        spread: fold_spread(scores_joint, scores_perm),
    }
}

/// Between-fold std-dev of the per-fold λ=0 estimate over [`DV_FOLDS`]
/// contiguous folds (deterministic — no RNG, no extra allocation).
#[must_use]
pub fn fold_spread(scores_joint: &[f64], scores_perm: &[f64]) -> f32 {
    let n = scores_joint.len().min(scores_perm.len());
    if n < DV_FOLDS * 2 {
        return 0.0;
    }
    let fold = n / DV_FOLDS;
    let mut vals = [0.0f64; DV_FOLDS];
    for (f, v) in vals.iter_mut().enumerate() {
        let lo = f * fold;
        let hi = if f == DV_FOLDS - 1 { n } else { lo + fold };
        *v = f64::from(dv_plug_in(&scores_joint[lo..hi], &scores_perm[lo..hi]));
    }
    let mu = vals.iter().sum::<f64>() / DV_FOLDS as f64;
    let var = vals.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / DV_FOLDS as f64;
    var.sqrt() as f32
}

/// Average the LOO bound over `draws` fresh uniform permutations (each draw
/// re-scores the permuted pairs — the score cost multiplies, the bound math
/// stays O(N) per draw). With `antithetic` each drawn σ is also applied as
/// σ⁻¹ (a negatively-correlated partner draw — the classic variance-reduction
/// pairing for the Q-term); the effective draw count rounds up to even.
///
/// Returns `(mean_loo, std_loo_across_draws)` — the multi-draw mean is the
/// recommended headline at moderate-to-high MI where a single permutation
/// draw is noisy; the across-draw std is an independent, stronger spread
/// signal than [`fold_spread`].
#[allow(clippy::too_many_arguments)]
pub fn dv_bound_perm_average(
    critic: super::Critic,
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    draws: usize,
    antithetic: bool,
    scratch: &mut MiScratch,
) -> (f32, f32) {
    use super::PermSource;
    assert!(n > 1, "LOO needs n ≥ 2");
    scratch.reseed();
    let pairs = if antithetic {
        (draws.max(1) + 1).div_ceil(2)
    } else {
        draws.max(1)
    };
    scratch.score_joint(critic, x, y, n, d);
    let mut acc = 0.0f64;
    let mut acc2 = 0.0f64;
    let mut count = 0.0f64;
    for _ in 0..pairs {
        scratch.next_perm(n);
        scratch.score_perm(critic, x, y, n, d, PermSource::Current);
        let v = f64::from(dv_loo(&scratch.joint, &scratch.perm));
        acc += v;
        acc2 += v * v;
        count += 1.0;
        if antithetic {
            scratch.score_perm(critic, x, y, n, d, PermSource::Inverse);
            let v = f64::from(dv_loo(&scratch.joint, &scratch.perm));
            acc += v;
            acc2 += v * v;
            count += 1.0;
        }
    }
    let mu = acc / count;
    let var = (acc2 / count - mu * mu).max(0.0);
    (mu as f32, var.sqrt() as f32)
}

/// SMILE-style variance-controlled DV (Clément et al., arXiv:1906.03309):
/// clip BOTH score vectors at the `[tau, 1−tau]` empirical quantiles of the
/// permutation scores, then evaluate the plug-in + LOO forms on the clipped
/// values.
///
/// WHY: for unbounded critics the Q-term `E_Q[e^T]` has divergent moments
/// (for the dot critic at every ρ > 0; per-dependent-dim ρ > 1/√3 for the
/// matched quadratic critic) — a single extreme permutation score dominates
/// the logmeanexp and the value collapses. Clipping trades a small bias for
/// finite variance: the stable instrument for high-MI regimes and for
/// high-dimension fixed critics (the IB path). **In-place**: the score
/// slices are clipped in the scratch (callers re-score each pass anyway);
/// `sort_buf` needs length ≥ n (the scratch's `stat_buf`).
/// Returns `(plug_in_clipped, loo_clipped)`.
pub fn dv_smile_in_place(
    scores_joint: &mut [f64],
    scores_perm: &mut [f64],
    tau: f64,
    sort_buf: &mut [f64],
) -> (f32, f32) {
    let n = scores_joint.len().min(scores_perm.len());
    assert!(n > 1, "SMILE needs n ≥ 2");
    assert!(sort_buf.len() >= n, "sort_buf too small");
    assert!((0.0..0.5).contains(&tau), "tau must be in [0, 0.5)");
    sort_buf[..n].copy_from_slice(&scores_perm[..n]);
    sort_buf[..n].sort_unstable_by(f64::total_cmp);
    let lo_i = (tau * (n - 1) as f64) as usize;
    let hi_i = ((1.0 - tau) * (n - 1) as f64) as usize;
    let lo = sort_buf[lo_i];
    let hi = sort_buf[hi_i];
    for v in &mut scores_joint[..n] {
        *v = v.clamp(lo, hi);
    }
    for v in &mut scores_perm[..n] {
        *v = v.clamp(lo, hi);
    }
    (
        dv_plug_in(scores_joint, scores_perm),
        dv_loo(scores_joint, scores_perm),
    )
}

/// Multi-draw SMILE-LOO average with the deterministic quadratic critic —
/// the accuracy instrument for the gates (finite variance ⇒ a mean over
/// permutation draws is meaningful, unlike the raw dot-critic case).
/// Returns `(mean, std)` across `draws` fresh uniform permutations.
#[allow(clippy::too_many_arguments)]
pub fn quadratic_dv_smile_average(
    q: &QuadraticCritic,
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    dep: usize,
    draws: usize,
    tau: f64,
    scratch: &mut MiScratch,
) -> (f32, f32) {
    assert!(n > 1, "needs n ≥ 2");
    scratch.reseed();
    scratch.ensure(n, d.max(1));
    q.score_dependent_into(x, y, n, d, dep, None, &mut scratch.joint);
    let mut acc = 0.0f64;
    let mut acc2 = 0.0f64;
    for _ in 0..draws.max(1) {
        scratch.next_perm(n);
        q.score_dependent_into(x, y, n, d, dep, Some(&scratch.perm_idx), &mut scratch.perm);
        let (v, loo) = dv_smile_in_place(
            &mut scratch.joint,
            &mut scratch.perm,
            tau,
            &mut scratch.stat_buf,
        );
        let _ = v;
        let v = f64::from(loo);
        acc += v;
        acc2 += v * v;
    }
    let c = draws.max(1) as f64;
    let mu = acc / c;
    let var = (acc2 / c - mu * mu).max(0.0);
    (mu as f32, var.sqrt() as f32)
}

// ────────────────────────────────────────────────────────────────────────
// QuadraticCritic — the deterministic near-optimal gate critic (T1.5)
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic quadratic-feature critic
/// `T(x, y) = Σ_i [ a·x_i·y_i + b·(x_i² + y_i²) ]` over the first `dep`
/// dimensions (dependent dims only — null dims carry zero features).
///
/// For standardized jointly-Gaussian pairs with per-dim correlation ρ, the
/// coefficients `a = ρ/(1−ρ²)`, `b = −ρ²/(2(1−ρ²))` ([`QuadraticCritic::matched`])
/// reproduce the exact Gaussian log-density-ratio up to a constant shift — the
/// DV bound is then exact in infinite samples, which makes this the gate
/// critic for the estimator-accuracy GOAT (T1.5): it tests the **estimator
/// math** (LOO, logmeanexp, permutation), not the zero-parameter critic
/// family. Independent (null) dimensions are excluded via `dep`, so the
/// structured fixtures measure exactly the dependent subspace.
///
/// Not part of the shipped [`super::Critic`] enum on purpose: it encodes
/// knowledge of the data geometry (ρ) and exists for gates and for callers
/// who genuinely know their geometry. The blind-instrument answer for
/// unknown geometry is the K-ladder + permutation tuple, not an oracle critic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadraticCritic {
    pub a: f32,
    pub b: f32,
}

impl QuadraticCritic {
    /// ρ-matched coefficients for standardized jointly-Gaussian pairs.
    #[must_use]
    pub fn matched(rho: f32) -> Self {
        let r2 = (rho * rho).min(0.999_999);
        let denom = 1.0 - r2;
        Self {
            a: rho / denom,
            b: -r2 / (2.0 * denom),
        }
    }

    /// The DV bound VALUE this critic achieves analytically on standardized
    /// Gaussian data with per-dim correlation ρ over `dep` dependent dims:
    /// `dep · [ a·ρ + 2b + ½·ln((1−2b)² − a²) ]` — equals
    /// `dep · (−½·ln(1−ρ²))` at the matched coefficients.
    #[must_use]
    pub fn analytic_bound(&self, rho: f32, dep: usize) -> f64 {
        let a = f64::from(self.a);
        let b = f64::from(self.b);
        let per_dim =
            a * f64::from(rho) + 2.0 * b + 0.5 * (((1.0 - 2.0 * b) * (1.0 - 2.0 * b)) - a * a).ln();
        per_dim * dep as f64
    }

    /// Score the pair set `(i, idx[i])` over the FIRST `dep` dimensions of
    /// each row. Pure — no scratch needed.
    #[allow(clippy::too_many_arguments)]
    pub fn score_dependent_into(
        &self,
        x: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
        dep: usize,
        idx: Option<&[u32]>,
        out: &mut [f64],
    ) {
        assert!(dep <= d, "dep > d");
        assert!(x.len() >= n * d && y.len() >= n * d, "population too small");
        assert!(out.len() >= n, "out too small");
        if let Some(sigma) = idx {
            assert!(sigma.len() >= n, "sigma too small");
        }
        for i in 0..n {
            let j = idx.map_or(i, |s| s[i] as usize);
            let xr = &x[i * d..i * d + dep];
            let yr = &y[j * d..j * d + dep];
            let mut acc = 0.0f64;
            for k in 0..dep {
                acc += f64::from(self.a * xr[k] * yr[k]);
                acc += f64::from(self.b * (xr[k] * xr[k] + yr[k] * yr[k]));
            }
            out[i] = acc;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// f64 scalar helpers (module-private math core; re-used by bounds.rs)
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn mean(s: &[f64], n: usize) -> f64 {
    let mut acc = 0.0;
    for &v in &s[..n] {
        acc += v;
    }
    acc / n as f64
}

pub(crate) fn max_of(s: &[f64], n: usize) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in &s[..n] {
        if v > m {
            m = v;
        }
    }
    m
}

/// Max-subtracted log-mean-exp: `m + ln(mean(e^{s − m}))`. No overflow, no
/// softmax (the batch is reduced, not normalized).
pub(crate) fn logmeanexp(s: &[f64], n: usize) -> f64 {
    let m = max_of(s, n);
    let mut acc = 0.0;
    for &v in &s[..n] {
        acc += (v - m).exp();
    }
    m + (acc / n as f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mi::test_support::{gaussian_pairs, splitmix};

    // ── T1.3: shift invariance ──────────────────────────────────────────────

    #[test]
    fn t13_shift_invariance_bit_exact_anchor() {
        // Small half-integer scores + power-of-two shifts: every intermediate
        // value is exactly representable in f64, so the DV closed forms are
        // exact at the bit level. NOTE: only the DV members (l0, loo) and the
        // fold spread are shift-invariant — the NWJ form (l1) is
        // gauge-dependent (its E_Q[e^T] term picks up e^c), which is why l1
        // must always ship together with the critic that produced it.
        let joint: Vec<f64> = (0..16).map(|i| (i % 5) as f64 - 1.5).collect();
        let perm: Vec<f64> = (0..16).map(|i| ((i * 7) % 11) as f64 - 4.0).collect();
        let base = dv_report(&joint, &perm);
        for &c in &[2.0f64, -4.0, 8.0] {
            let sj: Vec<f64> = joint.iter().map(|v| v + c).collect();
            let sp: Vec<f64> = perm.iter().map(|v| v + c).collect();
            let shifted = dv_report(&sj, &sp);
            assert_eq!(base.l0.to_bits(), shifted.l0.to_bits(), "l0, c = {c}");
            assert_eq!(base.loo.to_bits(), shifted.loo.to_bits(), "loo, c = {c}");
            assert_eq!(
                base.spread.to_bits(),
                shifted.spread.to_bits(),
                "spread, c = {c}"
            );
            // NWJ shift-covariance identity:
            // l1(T+c) = l1(T) + c − (e^c − 1)·E_Q[e^T].
            // (f32 storage — 1e-2 abs at the |l1| ≈ 490 magnitude of this
            // fixture is ~f32 epsilon scale.)
            let eq = perm.iter().map(|v| v.exp()).sum::<f64>() / perm.len() as f64;
            let expect = f64::from(base.l1) + c - (c.exp() - 1.0) * eq;
            assert!(
                (f64::from(shifted.l1) - expect).abs() < 1e-2,
                "l1 gauge shift, c = {c}: {} vs {expect}",
                shifted.l1
            );
        }
    }

    #[test]
    fn t13_shift_invariance_random_tolerance() {
        // Random-magnitude scores: the closed form is shift-invariant exactly;
        // fp summation introduces ≤ ~1e-12 relative wobble in f64 — assert at
        // 1e-6 nats (5 orders of magnitude inside the f64 noise floor scale).
        let mut rng = splitmix(11);
        let joint: Vec<f64> = (0..256).map(|_| f64::from(rng.normal())).collect();
        let perm: Vec<f64> = (0..256).map(|_| f64::from(rng.normal())).collect();
        let base = dv_report(&joint, &perm);
        let c = 37.317;
        let sj: Vec<f64> = joint.iter().map(|v| v + c).collect();
        let sp: Vec<f64> = perm.iter().map(|v| v + c).collect();
        let shifted = dv_report(&sj, &sp);
        for (a, b, name) in [
            (base.l0, shifted.l0, "l0"),
            (base.loo, shifted.loo, "loo"),
            (base.spread, shifted.spread, "spread"),
        ] {
            assert!(
                (a - b).abs() < 1e-6,
                "{name} drift {a} vs {b} under shift {c}"
            );
        }
    }

    // ── T1.4: null calibration (bias ≤ C·dof/N, curve recorded) ─────────────

    #[test]
    fn t14_null_calibration_curve() {
        // ρ = 0 ⇒ true MI = 0. With a FIXED informative critic (the ρ=0.3
        // matched coefficients — what a caller would carry under the
        // alternative), the systematic null bias is the log-Jensen term
        // ≈ Var_Q(e^T)/(2·E_Q[e^T]²·N) ≈ 0.049/N (for this critic
        // E_Q[e^T] = 0.9539, Var = 0.0898 measured exactly) — well inside the
        // plan's C·dof/N heuristic, which additionally covers adapted
        // critics. The dominant error at finite N is fixture noise
        // (std_T ≈ 0.358 ⇒ SE = σ/√(N·runs)), so the gate is a proper
        // statistical bound: |bias| ≤ 4·SE + 2·dof/N.
        let q = QuadraticCritic::matched(0.3);
        let null_value = q.analytic_bound(0.0, 1);
        let mut rows: Vec<(usize, f64, f64)> = Vec::new();
        for &n in &[100usize, 1_000] {
            let mut l0_acc = 0.0f64;
            let mut loo_acc = 0.0f64;
            let runs = 32;
            for r in 0..runs {
                let (x, y) = gaussian_pairs(0.0, n, 1_000 + r);
                let mut sj = vec![0.0; n];
                let mut sp = vec![0.0; n];
                q.score_dependent_into(&x, &y, n, 1, 1, None, &mut sj);
                let mut sc = crate::mi::MiScratch::new(n, 1, 700 + r);
                sc.next_perm(n);
                q.score_dependent_into(&x, &y, n, 1, 1, Some(&sc.perm_idx), &mut sp);
                let rep = dv_report(&sj, &sp);
                l0_acc += f64::from(rep.l0);
                loo_acc += f64::from(rep.loo);
            }
            rows.push((
                n,
                l0_acc / runs as f64 - null_value,
                loo_acc / runs as f64 - null_value,
            ));
        }
        for (n, l0, loo) in &rows {
            let se = 0.358 / (*n as f64 * 32.0).sqrt();
            let bound = 4.0 * se + 2.0 * 3.0 / *n as f64;
            assert!(l0.abs() <= bound, "plug-in bias {l0} > {bound} at N={n}");
            assert!(loo.abs() <= bound, "LOO bias {loo} > {bound} at N={n}");
        }
        // The LOO form sits at/below the plug-in form on average (the
        // self-inclusion term it removes is an upward bias).
        assert!(
            rows[0].1 <= rows[0].2 + 1e-3,
            "LOO bias {} should sit at/below plug-in {}",
            rows[0].1,
            rows[0].2
        );
        eprintln!("t14 null-calibration curve (n, plugin_bias, loo_bias): {rows:?}");
    }

    // ── estimator accuracy (module-scale smoke for the GOAT grid) ───────────

    #[test]
    fn dv_loo_tracks_gaussian_truth_module_scale() {
        // 1-D, ρ = 0.3, matched quadratic critic, N = 2e4 (module scale).
        // Truth: −½·ln(1−ρ²) ≈ 0.0465 nats. Tolerance 0.01 at this N.
        let rho = 0.3f32;
        let n = 20_000;
        let truth = -0.5 * (1.0 - f64::from(rho) * f64::from(rho)).ln();
        let (x, y) = gaussian_pairs(rho, n, 777);
        let q = QuadraticCritic::matched(rho);
        let mut sj = vec![0.0; n];
        let mut sp = vec![0.0; n];
        q.score_dependent_into(&x, &y, n, 1, 1, None, &mut sj);
        let mut sc = crate::mi::MiScratch::new(n, 1, 42);
        sc.next_perm(n);
        q.score_dependent_into(&x, &y, n, 1, 1, Some(&sc.perm_idx), &mut sp);
        let rep = dv_report(&sj, &sp);
        assert!(
            (f64::from(rep.loo) - truth).abs() < 0.01,
            "LOO {} vs truth {truth} (report {rep:?})",
            rep.loo
        );
        // The plug-in form sits at/above LOO here (its self-inclusion term is
        // an upward bias at this scale).
        assert!(
            rep.l0 >= rep.loo - 1e-3,
            "plug-in {} should sit at/above LOO {}",
            rep.l0,
            rep.loo
        );
    }

    #[test]
    fn analytic_bound_matches_closed_form_truth() {
        for &rho in &[0.1f32, 0.3, 0.5, 0.7] {
            let q = QuadraticCritic::matched(rho);
            let truth = -0.5 * (1.0 - f64::from(rho) * f64::from(rho)).ln();
            let analytic = q.analytic_bound(rho, 1);
            assert!(
                (analytic - truth).abs() < 1e-6,
                "analytic {analytic} vs closed form {truth} at ρ={rho}"
            );
        }
    }

    #[test]
    fn multi_draw_average_lands_within_spread_at_moderate_mi() {
        // At ρ = 0.7 (inside the finite-variance regime for the MATCHED
        // quadratic critic — ρ < 1/√3 ≈ 0.577 is where the RAW Q-term's
        // variance diverges, but with SMILE clipping even 0.7 is stable) the
        // 16-draw SMILE-LOO average must land within 2× its across-draw
        // spread + 0.05 nats of truth. NOTE the raw dot-critic DV is NOT the
        // instrument here: its Q-term variance diverges at every ρ > 0.
        let rho = 0.7f32;
        let n = 8_000;
        let truth = -0.5 * (1.0 - f64::from(rho) * f64::from(rho)).ln();
        let (x, y) = gaussian_pairs(rho, n, 4242);
        let q = QuadraticCritic::matched(rho);
        let mut s = crate::mi::MiScratch::new(n, 1, 9);
        let (mean16, std16) = quadratic_dv_smile_average(&q, &x, &y, n, 1, 1, 16, 0.01, &mut s);
        eprintln!("multi-draw ρ=0.7: mean={mean16:.5} ± {std16:.5} vs truth {truth:.5}");
        assert!(
            (f64::from(mean16) - truth).abs() <= f64::from(std16) * 2.0 + 0.05,
            "multi-draw mean {mean16} ± {std16} vs truth {truth}"
        );
    }

    #[test]
    fn smile_tames_the_high_mi_regime() {
        // ρ = 0.9: the RAW DV's Q-term has divergent variance (the value is
        // dominated by one extreme permutation score); the SMILE-clipped
        // value must be finite, stable, and a valid lower bound direction
        // (it may sit below truth — that bias is the documented trade).
        let rho = 0.9f32;
        let n = 8_000;
        let truth = -0.5 * (1.0 - f64::from(rho) * f64::from(rho)).ln();
        let (x, y) = gaussian_pairs(rho, n, 5150);
        let q = QuadraticCritic::matched(rho);
        let mut s = crate::mi::MiScratch::new(n, 1, 10);
        let (a, sa) = quadratic_dv_smile_average(&q, &x, &y, n, 1, 1, 8, 0.01, &mut s);
        let (b, sb) = quadratic_dv_smile_average(&q, &x, &y, n, 1, 1, 8, 0.01, &mut s);
        assert_eq!(a, b, "SMILE multi-draw must be deterministic");
        assert!(
            sa < 0.1 && sb < 0.1,
            "SMILE spread {sa}/{sb} should be tame"
        );
        assert!(
            f64::from(a) <= truth + 0.05,
            "bound must not exceed truth materially"
        );
        eprintln!("smile ρ=0.9: mean={a:.5} ± {sa:.5} vs truth {truth:.5} (clipped-bias trade)");
    }
}
