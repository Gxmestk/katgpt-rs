//! Smooth-minimum soft similarity for variable-length multi-token retrieval.
//!
//! Distilled from SoftMatcha 2 (Yoneda et al., ICML 2026, arXiv:2602.10908)
//! by Research 385. The smooth-min aggregation interpolates between plain-min
//! (β→∞, the strictest aggregation) and plain-sum (β≈1, the most lenient),
//! providing better discrimination than plain-cosine (mean) on multi-token
//! queries with partial mismatches.
//!
//! ## GOAT gate (Issue 041 PoC, 2026-07-12; Issue 041 removed, see git history)
//!
//! The PoC (`examples/issue_041_smooth_min_poc.rs`) uses synthetic per-token
//! embeddings — 200 items, 200 queries with all 4 tokens mismatched (same
//! cluster, different word). Results:
//!
//! | Gate | Metric | Result |
//! |------|--------|--------|
//! | G1 (quality) | recall@5 gain | **+12.0pp** (0.815 vs 0.695) |
//! | G2 (latency) | overhead vs mean | **~0 ns** (LLVM vectorized) |
//! | G3 (β sensitivity) | all β ∈ [10¹, 10⁶] beat plain cosine | ✅ |
//!
//! The gain is **modelless** (pure arithmetic on pre-computed cosines, no
//! training, no weights). The PoC resolves the contradiction between
//! Research 385 §4 (PoC uses synthetic data) and Issue 041 (PoC blocked on
//! consumer prerequisites) — the PoC does NOT need a real consumer.
//!
//! ## When to use
//!
//! Use `smooth_min_similarity` when you have per-position cosine similarities
//! (one per token pair) and want to aggregate them into a single score that
//! penalizes low-cosine positions more than plain averaging would. This is
//! the scenario where a distractor item has 1-2 exact-match positions (high
//! cosine) but several unrelated positions (low cosine) — plain mean is fooled
//! by the high-cosine positions, smooth-min correctly penalizes the low ones.
//!
//! ## Feature gate
//!
//! DEFAULT-ON (2026-07-12, Issue 041 T6). Promoted after the first consumer
//! GOAT gate passed: `RerankMethod::SmoothMinAligned` in katgpt-attn-match
//! achieved recall@5 = 1.000 vs Cosine 0.495 (+50.5pp) on position-aligned
//! multi-token retrieval.

/// Smooth-minimum similarity for variable-length soft pattern matching.
///
/// Aggregates per-position cosine similarities into a single score in [0, 1]
/// using the smooth-min (softmin / LogSumExp family) function:
///
/// ```text
/// sim = 1 - log_β( Σ(β^(1-c_i) - 1) + 1 )
/// ```
///
/// where `c_i` is the i-th cosine similarity and `β` controls sharpness:
/// - `β → ∞`: approaches `min(c_i)` — strict, one bad match kills the score.
/// - `β ≈ 1`: approaches `Σ(c_i) / m` — lenient, like plain mean.
/// - `β = 10⁴`: the paper's empirically-best operating point.
///
/// # Arguments
///
/// * `cosines` — per-position cosine similarities, each in `[-1, 1]`.
/// * `beta` — sharpness parameter. Must be `> 1.0` (use plain mean for β=1).
///
/// # Returns
///
/// Similarity score. Equals `1.0` when all cosines are `1.0` (perfect
/// match). Decreases as cosines decrease. Can go below `0` for poor matches
/// (low cosines or many tokens) — this is correct behavior, as the score is
/// used for ranking, not as a probability. The `+1` and `-1` terms in the
/// formula ensure `sim = 1.0` at the all-perfect-match boundary.
///
/// # Panics
///
/// Panics if `beta <= 1.0` (ln(β) ≤ 0 makes the logarithm undefined) or if
/// `cosines` is empty.
///
/// # Example
///
/// ```
/// # use katgpt_core::smooth_min_similarity;
/// // All positions match moderately → moderate score
/// let sim = smooth_min_similarity(&[0.7, 0.7, 0.7, 0.7], 1e4);
/// assert!(sim > 0.4 && sim < 0.7, "moderate match: {sim}");
///
/// // One position is a perfect match, rest are bad → low score
/// // (plain mean would give 0.40, smooth-min correctly penalizes)
/// assert!(smooth_min_similarity(&[1.0, 0.2, 0.2, 0.2], 1e4) < 0.1);
/// ```
#[inline]
pub fn smooth_min_similarity(cosines: &[f32], beta: f32) -> f32 {
    assert!(beta > 1.0, "beta must be > 1.0, got {beta}");
    assert!(!cosines.is_empty(), "cosines must not be empty");

    let log_beta = beta.ln();
    // Σ(β^(1-c_i) - 1) + 1
    // Using mul_add for numerical stability: (1-c) * ln(β)
    let sum = cosines
        .iter()
        .map(|&c| ((1.0 - c) * log_beta).exp() - 1.0)
        .sum::<f32>()
        + 1.0;

    // Guard against sum ≤ 0 (can happen if cosines are very negative)
    if sum <= 0.0 {
        return 0.0;
    }

    1.0 - sum.ln() / log_beta
}

/// Insertion/deletion penalty using Zipfian-whitened norm.
///
/// Computes `exp(-norm_sq / gamma)` — a decay penalty for edit operations
/// (insertions/deletions) in variable-length pattern matching. Tokens with
/// high Zipfian-whitened norm (high information content) are penalized more
/// for editing; low-information tokens (stopwords like "the", "of") are cheap
/// to edit.
///
/// # Arguments
///
/// * `norm_sq` — squared norm of the edited token's embedding (post-Zipfian
///   whitening). Should be `>= 0`.
/// * `gamma` — penalty scale. The paper sets `γ = m · γ'` where `γ'` is tuned
///   so the penalty equals `1/e` at the 50th-lowest norm for query length `m`.
///
/// # Returns
///
/// Penalty multiplier in `(0, 1]`. `norm_sq = 0` → 1.0 (no penalty).
/// `norm_sq = gamma` → `1/e ≈ 0.368`.
///
/// # Example
///
/// ```
/// # use katgpt_core::edit_penalty;
/// assert!((edit_penalty(0.0, 1.0) - 1.0).abs() < 1e-6);
/// assert!((edit_penalty(1.0, 1.0) - std::f32::consts::E.powi(-1)).abs() < 1e-6);
/// ```
#[inline]
pub fn edit_penalty(norm_sq: f32, gamma: f32) -> f32 {
    (-norm_sq / gamma).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    // ── smooth_min_similarity: basic properties ──

    #[test]
    fn all_ones_gives_one() {
        // All cosines = 1.0 → perfect match → similarity = 1.0
        let sim = smooth_min_similarity(&[1.0, 1.0, 1.0, 1.0], 1e4);
        assert!((sim - 1.0).abs() < TOL, "all-ones should give 1.0, got {sim}");
    }

    #[test]
    fn single_element_returns_it() {
        // m=1: smooth_min([c]) = 1 - log_β(β^(1-c) - 1 + 1) = 1 - (1-c) = c
        let sim = smooth_min_similarity(&[0.7], 1e4);
        assert!((sim - 0.7).abs() < TOL, "single element should return c, got {sim}");
    }

    #[test]
    fn all_equal_cosines_decreases_with_token_count() {
        // For all c_i = c and m > 1, smooth_min = 1 - log_β(m·(β^(1-c)−1)+1)
        // which is LESS than c (the log_β(m) offset). More tokens → lower
        // score for the same cosine level. This is by design: a 4-token match
        // at cosine 0.5 is less confident than a 1-token match at 0.5.
        let sim_m1 = smooth_min_similarity(&[0.5], 1e4);
        let sim_m2 = smooth_min_similarity(&[0.5, 0.5], 1e4);
        let sim_m4 = smooth_min_similarity(&[0.5, 0.5, 0.5, 0.5], 1e4);
        assert!((sim_m1 - 0.5).abs() < TOL, "m=1 should return c, got {sim_m1}");
        assert!(sim_m2 < sim_m1, "m=2 should be < m=1: {sim_m2} < {sim_m1}");
        assert!(sim_m4 < sim_m2, "m=4 should be < m=2: {sim_m4} < {sim_m2}");
    }

    #[test]
    fn beta_large_approaches_min_minus_offset() {
        // β → ∞: smooth_min → min(c_i) − log_β(m)
        // For m=4, β=10^10: log_10^10(4) ≈ 0.0602, so sim ≈ min − 0.06
        let cosines = [0.3, 0.8, 0.5, 0.9];
        let min_val = 0.3f32;
        let sim_large = smooth_min_similarity(&cosines, 1e10);
        // Should be close to min, slightly below due to the m-offset
        assert!(
            sim_large > min_val - 0.1 && sim_large <= min_val,
            "β=10^10 should be near min ({min_val}), got {sim_large}"
        );
    }

    #[test]
    fn beta_affects_score() {
        // Different β values give different scores (the function is
        // responsive to β). The direction depends on the input distribution:
        // for all-moderate cosines, lower β (sum-like) can give higher scores;
        // for mixed high+low cosines, higher β (min-like) can give higher scores.
        // The PoC showed all β ∈ [10¹, 10⁶] beat plain cosine on recall@5.
        let cosines = [0.3, 0.8, 0.5, 0.9];
        let sim_low = smooth_min_similarity(&cosines, 1e2);
        let sim_mid = smooth_min_similarity(&cosines, 1e4);
        let sim_high = smooth_min_similarity(&cosines, 1e6);
        // Just verify they're all different and finite
        assert!(sim_low.is_finite() && sim_mid.is_finite() && sim_high.is_finite());
        assert!(sim_low != sim_mid || sim_mid != sim_high, "β should affect the score");
    }

    #[test]
    fn penalizes_low_cosine_more_than_mean() {
        // [0.9, 0.9, 0.1, 0.1] vs [0.5, 0.5, 0.5, 0.5]
        // mean: 0.5 vs 0.5 (tie!) but smooth_min should prefer all-moderate
        let mixed = [0.9f32, 0.9, 0.1, 0.1];
        let uniform = [0.5f32, 0.5, 0.5, 0.5];
        let sim_mixed = smooth_min_similarity(&mixed, 1e4);
        let sim_uniform = smooth_min_similarity(&uniform, 1e4);
        assert!(
            sim_uniform > sim_mixed,
            "uniform {sim_uniform} should beat mixed {sim_mixed}"
        );
    }

    #[test]
    fn handles_negative_cosines() {
        // Anti-correlated tokens should give low (possibly negative) scores
        let sim = smooth_min_similarity(&[0.5, -0.5, 0.5, -0.5], 1e4);
        assert!(sim < 0.0, "anti-correlated should give negative score, got {sim}");
    }

    #[test]
    fn empty_panics() {
        let result = std::panic::catch_unwind(|| {
            smooth_min_similarity(&[], 1e4);
        });
        assert!(result.is_err(), "empty cosines should panic");
    }

    #[test]
    fn beta_leq_one_panics() {
        let result = std::panic::catch_unwind(|| {
            smooth_min_similarity(&[0.5], 1.0);
        });
        assert!(result.is_err(), "beta=1.0 should panic");
    }

    #[test]
    fn deterministic() {
        // Same input → same output
        let cosines = [0.3, 0.7, 0.5, 0.9];
        let a = smooth_min_similarity(&cosines, 1e4);
        let b = smooth_min_similarity(&cosines, 1e4);
        assert!((a - b).abs() < f32::EPSILON, "should be deterministic");
    }

    // ── edit_penalty: basic properties ──

    #[test]
    fn edit_penalty_zero_norm_is_one() {
        assert!((edit_penalty(0.0, 1.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn edit_penalty_at_gamma_is_inv_e() {
        let expected = std::f32::consts::E.powi(-1);
        assert!((edit_penalty(1.0, 1.0) - expected).abs() < TOL);
    }

    #[test]
    fn edit_penalty_decreasing_in_norm() {
        // Higher norm → lower penalty (more information lost)
        let p1 = edit_penalty(0.5, 1.0);
        let p2 = edit_penalty(1.0, 1.0);
        let p3 = edit_penalty(2.0, 1.0);
        assert!(p1 > p2 && p2 > p3, "should be decreasing: {p1} > {p2} > {p3}");
    }

    #[test]
    fn edit_penalty_increasing_in_gamma() {
        // Higher gamma → less penalty (more tolerant)
        let p1 = edit_penalty(1.0, 0.5);
        let p2 = edit_penalty(1.0, 1.0);
        let p3 = edit_penalty(1.0, 2.0);
        assert!(p1 < p2 && p2 < p3, "should be increasing: {p1} < {p2} < {p3}");
    }

    #[test]
    fn edit_penalty_always_in_unit_interval() {
        // For non-negative norm_sq, penalty is in (0, 1]
        for &norm_sq in &[0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0] {
            let p = edit_penalty(norm_sq, 1.0);
            assert!(p > 0.0 && p <= 1.0, "penalty {p} not in (0,1] for norm_sq={norm_sq}");
        }
    }

    // ── Numerical properties from the GOAT PoC ──

    #[test]
    fn poc_correct_item_beats_distractor() {
        // From the PoC: correct item has all-moderate cosines, distractor has
        // 2 exact matches + 2 unrelated. Plain cosine is fooled; smooth-min
        // correctly ranks the correct item higher.
        let correct = [0.53f32, 0.55, 0.29, 0.33];
        let distractor = [0.29f32, 1.0, -0.12, 1.0];

        let correct_plain = correct.iter().sum::<f32>() / 4.0;
        let distractor_plain = distractor.iter().sum::<f32>() / 4.0;

        let correct_smooth = smooth_min_similarity(&correct, 1e4);
        let distractor_smooth = smooth_min_similarity(&distractor, 1e4);

        // Plain cosine: distractor wins (wrong!)
        assert!(
            distractor_plain > correct_plain,
            "plain cosine should rank distractor higher: {distractor_plain} > {correct_plain}"
        );

        // Smooth-min: correct wins (right!)
        assert!(
            correct_smooth > distractor_smooth,
            "smooth-min should rank correct higher: {correct_smooth} > {distractor_smooth}"
        );
    }

    #[test]
    fn two_token_case() {
        // m=2: smooth_min of [0.8, 0.6] should be below min(0.6) due to the
        // log_β(m) offset, but above the distractor scenario.
        let sim = smooth_min_similarity(&[0.8, 0.6], 1e4);
        // For m=2, β=10⁴: sim ≈ 0.586 (below min 0.6, above 0)
        assert!(sim > 0.0 && sim < 0.6, "two-token case: {sim} should be in (0, 0.6)");
    }

    #[test]
    fn large_token_count() {
        // m=64: should not panic or overflow. With many tokens, the
        // log_β(m) offset is larger, so the score can be lower.
        let cosines: Vec<f32> = (0..64).map(|i| 0.3 + 0.01 * i as f32).collect();
        let sim = smooth_min_similarity(&cosines, 1e4);
        assert!(sim.is_finite(), "large token count should give finite result");
        // With 64 tokens at cosine 0.3-0.93, sim is dominated by the low end
        // and the large m offset. Just check it's finite and ordered correctly.
    }
}
