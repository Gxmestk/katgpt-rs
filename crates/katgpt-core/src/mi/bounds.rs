//! Bound ladder from ONE score pass: NWJ / InfoNCE-K / JS + the K-ladder
//! tightness diagnostic (Plan 583 T2.1/T2.2).
//!
//! All four bounds consume the same scratch-resident score pair
//! (`scratch.joint` / `scratch.perm`) — one scoring pass, every bound. The
//! ladder evaluates InfoNCE at increasing negative counts K and exports the
//! **saturation gap** (`critic_headroom`): how much tighter the bound gets as
//! K grows. Large headroom ⇒ the critic family is still hungry — the reported
//! magnitude is ceiling-limited, not converged (InfoNCE ≤ log K hard ceiling;
//! Poole et al. arXiv:1905.06922).
//!
//! Coherence (checked by the GOAT gates, T2.4): InfoNCE(K) is monotone
//! non-decreasing in K in expectation; at zero/low MI every bound sits at or
//! below the true MI (bounds are lower bounds — on the null they may be
//! NEGATIVE, which is correct, not a bug); DV ≥ NWJ typically (NWJ trades
//! variance for bias at high MI).

use super::MiScratch;
use super::dv::{dv_plug_in, mean, nwj};

/// Maximum K-ladder rungs (fixed-size — zero-alloc by construction).
pub const MAX_K_LADDER: usize = 8;

/// Default K ladder (T2.2).
pub const DEFAULT_K_LADDER: [u32; 5] = [4, 16, 64, 256, 1024];

/// One InfoNCE-K rung.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KRecord {
    pub k: u32,
    /// The InfoNCE-K estimate in nats.
    pub infonce: f32,
}

/// The full bound set from one score pass.
#[derive(Clone, Debug)]
pub struct BoundLadder {
    /// λ = 0 plug-in DV (see `dv`).
    pub dv: f32,
    /// NWJ (λ = 1 of the DV family; same value as `dv::DvReport::l1`).
    pub nwj: f32,
    /// Jensen–Shannon variational bound
    /// `E_P[T] + ln 2 − E_Q[softplus(T)]` (f-GAN conjugate
    /// `f*_JS(t) = softplus(t) − ln 2`; the bound value is ≤ ln 2 since
    /// JS ≤ ln 2).
    pub js: f32,
    /// InfoNCE at the LARGEST ladder rung.
    pub infonce_kmax: f32,
    /// K-ladder rungs (ascending K; unused tail entries are zeroed).
    pub ladder: [KRecord; MAX_K_LADDER],
    /// Number of valid `ladder` entries.
    pub n_rungs: usize,
    /// **Saturation gap**: `infonce(K_max) − infonce(K_min)`. Large ⇒ the
    /// critic family is still hungry (bound not converged); ~0 ⇒ saturated at
    /// the log-K ceiling or at the truth. The "how much MI can this critic
    /// family even see" diagnostic. With an external ground truth available,
    /// the caller computes `truth − infonce_kmax` for the residual gap.
    pub critic_headroom: f32,
}

/// InfoNCE-K estimate from pre-computed scores.
///
/// Partitions the n pairs into consecutive blocks of K; anchor i's candidate
/// set is `{joint_i} ∪ {perm_j : j in block(i), j ≠ i}` (negatives drawn from
/// the same permutation vector — one pass, no N×K matrix):
/// `Î_K = mean_i[ ln|B(i)| + joint_i − ln( e^{joint_i} + Σ_{j≠i∈B(i)} e^{perm_j} ) ]`.
/// K must be ≥ 2 and n ≥ K; the final partial block uses its actual size
/// (≥ 2). Under H0 the value is ≤ 0 (it is a lower bound on I = 0) — a small
/// negative InfoNCE on independent data is correct behavior, not a bug.
#[must_use]
pub fn infonce_k(scores_joint: &[f64], scores_perm: &[f64], k: u32) -> f32 {
    let n = scores_joint.len().min(scores_perm.len());
    let k = k as usize;
    assert!(k >= 2 && n >= k, "InfoNCE-K needs k ≥ 2 and n ≥ k");
    let mut acc = 0.0f64;
    let mut count = 0.0f64;
    let mut start = 0;
    while start < n {
        let end = (start + k).min(n);
        let block = end - start;
        if block >= 2 {
            // One max + one sum over the block's perm scores, then per-anchor
            // exclusion via the O(1) rest-identity (the dv::dv_loo pattern).
            let mut m = f64::NEG_INFINITY;
            for &p in &scores_perm[start..end] {
                if p > m {
                    m = p;
                }
            }
            let mut sum = 0.0f64;
            for &p in &scores_perm[start..end] {
                sum += (p - m).exp();
            }
            for i in start..end {
                let e_i = (scores_perm[i] - m).exp();
                let rest = (sum - e_i).max(f64::MIN_POSITIVE);
                let hi = scores_joint[i].max(m);
                let lse_i = hi + ((scores_joint[i] - hi).exp() + rest * (m - hi).exp()).ln();
                acc += (block as f64).ln() + scores_joint[i] - lse_i;
                count += 1.0;
            }
        }
        start = end;
    }
    (acc / count) as f32
}

/// Jensen–Shannon variational bound:
/// `E_P[T] + ln 2 − E_Q[softplus(T)]` with the numerically-stable
/// `softplus(t) = max(t, 0) + ln(1 + e^{−|t|})`.
#[must_use]
pub fn js_bound(scores_joint: &[f64], scores_perm: &[f64]) -> f32 {
    let n = scores_joint.len().min(scores_perm.len());
    assert!(n > 0, "empty scores");
    let mut q = 0.0f64;
    for &p in &scores_perm[..n] {
        q += softplus(p);
    }
    (mean(scores_joint, n) + std::f64::consts::LN_2 - q / n as f64) as f32
}

/// Numerically-stable softplus: `ln(1 + e^t)` without overflow.
#[must_use]
pub fn softplus(t: f64) -> f64 {
    t.max(0.0) + (-t.abs()).exp().ln_1p()
}

/// Full ladder from the scratch's current score pass.
pub fn bounds_all(scratch: &MiScratch, k_ladder: &[u32]) -> BoundLadder {
    let n = scratch.joint.len().min(scratch.perm.len());
    assert!(n > 0, "scratch has no scores");
    assert!(!k_ladder.is_empty() && k_ladder.len() <= MAX_K_LADDER);
    let mut ladder = [KRecord { k: 0, infonce: 0.0 }; MAX_K_LADDER];
    for (i, &k) in k_ladder.iter().enumerate() {
        let v = if n < k as usize {
            // Not enough samples for this rung — record NaN so the ladder
            // shape stays visible instead of silently truncating.
            f32::NAN
        } else {
            infonce_k(&scratch.joint, &scratch.perm, k)
        };
        ladder[i] = KRecord { k, infonce: v };
    }
    let first = ladder[0].infonce;
    let last = ladder[k_ladder.len() - 1].infonce;
    BoundLadder {
        dv: dv_plug_in(&scratch.joint, &scratch.perm),
        nwj: nwj(&scratch.joint, &scratch.perm),
        js: js_bound(&scratch.joint, &scratch.perm),
        infonce_kmax: last,
        ladder,
        n_rungs: k_ladder.len(),
        critic_headroom: last - first,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mi::dv::QuadraticCritic;
    use crate::mi::test_support::{gaussian_pairs, gaussian_pairs_dep};

    /// Score pass with an INFORMATIVE fixed quadratic critic (the critic a
    /// caller would carry under the alternative ρ), so the bounds have
    /// Q-term variance to work with.
    fn score_pass(rho_data: f32, critic_rho: f32, n: usize, seed: u64) -> (MiScratch, f64) {
        let (x, y) = gaussian_pairs(rho_data, n, seed);
        let q = QuadraticCritic::matched(critic_rho);
        let mut s = MiScratch::new(n, 1, seed ^ 0xA5A5);
        q.score_dependent_into(&x, &y, n, 1, 1, None, &mut s.joint);
        s.next_perm(n);
        q.score_dependent_into(&x, &y, n, 1, 1, Some(&s.perm_idx), &mut s.perm);
        let truth = if rho_data > 0.0 {
            -0.5 * (1.0 - f64::from(rho_data) * f64::from(rho_data)).ln()
        } else {
            0.0
        };
        (s, truth)
    }

    #[test]
    fn infonce_monotone_in_k_at_low_mi() {
        // ρ = 0.2 (low MI): the ladder must be monotone non-decreasing within
        // a small finite-sample tolerance.
        let (s, _truth) = score_pass(0.2, 0.2, 20_000, 123);
        let ladder = bounds_all(&s, &DEFAULT_K_LADDER);
        let mut prev = f32::NEG_INFINITY;
        for r in &ladder.ladder[..ladder.n_rungs] {
            assert!(
                r.infonce.is_finite() && r.infonce >= prev - 0.02,
                "InfoNCE not monotone at K={}: {} < {prev}",
                r.k,
                r.infonce
            );
            prev = r.infonce;
        }
    }

    #[test]
    fn bounds_sit_at_or_below_truth_at_low_mi() {
        let rho = 0.2f32;
        let (s, truth) = score_pass(rho, rho, 50_000, 321);
        let ladder = bounds_all(&s, &[64]);
        assert!(
            f64::from(ladder.dv) <= truth + 0.02,
            "dv {} > truth {truth}",
            ladder.dv
        );
        assert!(
            f64::from(ladder.nwj) <= truth + 0.05,
            "nwj {} > truth {truth}",
            ladder.nwj
        );
        assert!(
            f64::from(ladder.js) <= truth + 0.02,
            "js {} > truth {truth}",
            ladder.js
        );
        assert!(
            f64::from(ladder.infonce_kmax) <= truth + 0.03,
            "infonce {} > truth {truth}",
            ladder.infonce_kmax
        );
        assert!(
            ladder.js <= std::f32::consts::LN_2 + 1e-4,
            "js bound exceeds ln 2"
        );
    }

    #[test]
    fn null_bounds_are_negative_and_match_the_critics_own_null_value() {
        // ρ_data = 0 with the ρ=0.3-matched critic: the critic's own DV bound
        // value on the null is analytically 2b + ½ln((1−2b)²−a²) ≈ −0.052
        // (bounds may be negative on the null — correct, they bound I = 0
        // from below). The estimator must reproduce that value within
        // finite-sample noise, and InfoNCE must stay ≤ 0 + tiny.
        let (s, _t) = score_pass(0.0, 0.3, 50_000, 555);
        let q = QuadraticCritic::matched(0.3);
        let null_value = q.analytic_bound(0.0, 1);
        let ladder = bounds_all(&s, &[64, 1024]);
        assert!(
            (f64::from(ladder.dv) - null_value).abs() < 0.02,
            "dv on null {} vs analytic null bound {null_value}",
            ladder.dv
        );
        assert!(
            ladder.infonce_kmax <= 0.01,
            "InfoNCE on null must sit at/below I = 0, got {}",
            ladder.infonce_kmax
        );
        assert!(
            ladder.critic_headroom.abs() < 0.02,
            "headroom on null = {}",
            ladder.critic_headroom
        );
    }

    #[test]
    fn structured_null_dims_do_not_inflate_bounds() {
        // d = 16 with dep = 2 dependent dims (ρ = 0.4): the bounds on the
        // dependent subspace must match the 2-D value, NOT scale with d —
        // the quadratic critic scores only the dependent dims.
        let (x, y) = gaussian_pairs_dep(0.4, 30_000, 16, 2, 9001);
        let q = QuadraticCritic::matched(0.4);
        let mut s = MiScratch::new(30_000, 16, 77);
        q.score_dependent_into(&x, &y, 30_000, 16, 2, None, &mut s.joint);
        s.next_perm(30_000);
        q.score_dependent_into(&x, &y, 30_000, 16, 2, Some(&s.perm_idx), &mut s.perm);
        let truth = 2.0 * -0.5 * (1.0 - 0.16f32).ln();
        let ladder = bounds_all(&s, &[64]);
        assert!(
            (f64::from(ladder.dv) - f64::from(truth)).abs() < 0.02,
            "dv {} vs 2-dim truth {truth}",
            ladder.dv
        );
    }

    #[test]
    fn infonce_h0_is_nonpositive_and_identification_works() {
        // Hand-checkable: constant scores ⇒ lse = ln K ⇒ Î = ln K + 0 − ln K = 0.
        let joint = vec![0.0f64; 64];
        let perm = vec![0.0f64; 64];
        assert!(infonce_k(&joint, &perm, 8).abs() < 1e-12);
        // A perfect anchor (joint ≫ block negatives) ⇒ Î → ln K.
        let mut j = vec![-10.0f64; 64];
        let mut p = vec![-10.0f64; 64];
        for i in 0..64 {
            j[i] = 5.0;
            p[i] = -10.0;
        }
        let v = infonce_k(&j, &p, 8);
        assert!(
            (f64::from(v) - 8f64.ln()).abs() < 1e-5,
            "perfect identification {v}"
        );
    }

    #[test]
    fn softplus_stable_at_extremes() {
        assert_eq!(softplus(800.0), 800.0);
        assert_eq!(softplus(-800.0), 0.0); // underflows: e^(-800) == 0 in f64
        assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-12);
    }
}
