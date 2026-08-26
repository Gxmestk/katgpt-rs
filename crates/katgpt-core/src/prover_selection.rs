//! Prover-selection statistics + cross-state advantage centering
//! (katgpt-rs Issue 692, from Research 509 — arXiv:2410.08146 Setlur et al.).
//!
//! The paper's result: per-step supervision should be **advantages under a
//! complementary prover policy**, and the right way to *select* a
//! prover/critic/verifier is not strength but distinguishability +
//! alignment (Theorem 3.1: improvement ≳ γ·(D + Al)). This module hosts the
//! modelless kernels; T1 ships the D/Al selection statistics + the theorem
//! bound + its sigmoid-gated exposure, T2 the changepoint kernel. The K\*
//! law gate (T3) lands beside them.
//!
//! T1 — [`distinguishability`] / [`alignment`] / [`theorem_bound`] /
//! [`selection_gate`]: offline-computable from logged Bernoulli outcomes,
//! they form a **predicted-gain pre-gate** — skip wiring a prover when the
//! bound says ≈0 gain, before any runtime cost. The paper's selector
//! inverts our strength-only rankings everywhere (drafters by mean
//! acceptance, rules by Elo, critics by head quality): a *weaker but
//! complementary* prover beats a stronger one (Prop F.1: complementarity η
//! ⇒ Ω(η²) gain — this is NOT distillation; the prover's ceiling does not
//! bound the gain).
//!
//! T2 — `first_pit`: the first index where a Q̂ sequence pits (drops below
//! ε). Consumer: riir-clippy's PAV data curation (riir-train Plan 356 A1)
//! segments incorrect edit sequences at their first pit — the prefix before
//! it is the (prefix, Q̂) training pair's boundary. riir-clippy's
//! `pav_data::first_pit` is a twin of this fn (identical signature +
//! semantics); the twin swaps to this import when its crate bumps the
//! katgpt-core dep.
//!
//! Pure modelless arithmetic, no deps, no allocs, zero-cost-unless-invoked.

/// Distinguishability D(μ) = E_s Var_{a~π}[A^μ(s,a)] — how much the prover's
/// **advantage** varies across the base policy's actions, averaged over
/// states (Theorem 3.1's first term).
///
/// A prover that succeeds from any prefix (too strong) or fails from any
/// prefix (too weak) has Q^μ flat within every state ⇒ A^μ ≡ 0 ⇒ D = 0 —
/// the paper's core prediction, pinned by tests here. Selection by D alone
/// already inverts strength ranking: the *intermediate* prover wins.
///
/// # Estimator
///
/// Both logs are ragged 2-D: `base_outcomes[s][a]` = Q̂^π(s,a) and
/// `prover_outcomes[s][a]` = Q̂^μ(s,a), the Bernoulli success means logged at
/// the (state, action) pairs the base policy visited. The empirical action
/// distribution **is** the π-weight: uniform over a state's logged entries,
/// with entry multiplicity encoding π(a|s) (duplicate an entry to weight
/// it). V^μ(s) = mean_a Q^μ(s,a) and A^μ(s,a) = Q^μ(s,a) − V^μ(s); because
/// the advantages are exactly centered under those weights,
/// Var_{a~π}[A^μ] = mean_a (A^μ)². `base_outcomes` defines the support the
/// prover must be evaluated on (shape contract, `debug_assert`ed; states
/// with no logged entries are skipped; an empty log yields 0.0).
///
/// Non-negative by construction (mean of squares), ≤ 0.25 for Q̂ ∈ [0,1]
/// (Popoviciu). NaN entries propagate — screening is the caller's contract
/// (the same totality stance as [`first_pit`]).
#[must_use]
pub fn distinguishability(base_outcomes: &[&[f32]], prover_outcomes: &[&[f32]]) -> f32 {
    debug_assert_eq!(
        base_outcomes.len(),
        prover_outcomes.len(),
        "prover log must cover the base policy's states"
    );
    let mut sum = 0.0_f32;
    let mut states = 0_usize;
    for (base_s, prover_s) in base_outcomes.iter().zip(prover_outcomes) {
        debug_assert_eq!(
            base_s.len(),
            prover_s.len(),
            "prover log must cover the state's logged actions"
        );
        let n = prover_s.len();
        if n == 0 {
            continue;
        }
        // V^μ(s) under the empirical π weights (entry multiplicity = π(a|s)).
        let mut v = 0.0_f32;
        for &q in *prover_s {
            v += q;
        }
        v /= n as f32;
        // Var_{a~π}[A^μ] = mean_a (Q^μ − V^μ)² — the mean of A^μ is exactly 0
        // under these weights, so this IS the variance (two-pass form: the
        // numerically-stable one).
        let mut var = 0.0_f32;
        for &q in *prover_s {
            let a = q - v;
            var += a * a;
        }
        sum += var / n as f32;
        states += 1;
    }
    if states == 0 {
        0.0
    } else {
        sum / states as f32
    }
}

/// Alignment Al(μ) = E_s E_{a~π}[A^μ(s,a)·A^π(s,a)] — the inner product of
/// the prover's advantage with the base policy's own advantage (Theorem
/// 3.1's second term; a dot product, house style).
///
/// Positive = the prover's progress signal points along the base's;
/// negative = anti-aligned (rewarding what the base already penalizes).
/// Cauchy–Schwarz pins the magnitude: |Al(μ)| ≤ √(D(μ)·D(π)) — asserted on
/// the exhaustive Bernoulli grid in the tests. At μ = π the inner product
/// degenerates to the mean square, so Al(π) == D(π) exactly (the paper's
/// μ=π collapse, pinned by test).
///
/// Same data model + shape contract as [`distinguishability`]: per-(s,a)
/// Bernoulli means, entry multiplicity = π(a|s), empty states skipped,
/// empty log → 0.0, NaN propagates.
#[must_use]
pub fn alignment(base_outcomes: &[&[f32]], prover_outcomes: &[&[f32]]) -> f32 {
    debug_assert_eq!(
        base_outcomes.len(),
        prover_outcomes.len(),
        "prover log must cover the base policy's states"
    );
    let mut sum = 0.0_f32;
    let mut states = 0_usize;
    for (base_s, prover_s) in base_outcomes.iter().zip(prover_outcomes) {
        debug_assert_eq!(
            base_s.len(),
            prover_s.len(),
            "prover log must cover the state's logged actions"
        );
        let n = base_s.len();
        if n == 0 {
            continue;
        }
        // V^π(s) and V^μ(s) in one pass.
        let mut v_pi = 0.0_f32;
        let mut v_mu = 0.0_f32;
        for (&qb, &qm) in base_s.iter().zip(*prover_s) {
            v_pi += qb;
            v_mu += qm;
        }
        v_pi /= n as f32;
        v_mu /= n as f32;
        // E_{a~π}[A^μ·A^π].
        let mut inner = 0.0_f32;
        for (&qb, &qm) in base_s.iter().zip(*prover_s) {
            inner += (qm - v_mu) * (qb - v_pi);
        }
        sum += inner / n as f32;
        states += 1;
    }
    if states == 0 {
        0.0
    } else {
        sum / states as f32
    }
}

/// Theorem 3.1's predicted-gain lower bound: improvement ≳ γ·(D + Al).
///
/// γ is the prover-mediated improvement discount (paper notation; pass the
/// consumer's effective horizon/discount). Al may be negative — a
/// distinguishable but anti-aligned prover predicts net harm, which is the
/// point of the pre-gate: skip wiring the prover when the bound is ≤ 0,
/// before any runtime cost. Pure arithmetic, no clamping.
#[must_use]
pub fn theorem_bound(d: f32, al: f32, gamma: f32) -> f32 {
    gamma * (d + al)
}

/// Sigmoid-gated exposure of [`theorem_bound`] — the bounded (0,1) selection
/// surface (house rule: sigmoid, never softmax).
///
/// Monotone in the bound, so ranking provers by the gate is order-identical
/// to ranking by the raw bound (compression is harmless for ordering);
/// threshold at 0.5 ⇔ threshold at bound = 0. Aligned provers gate above
/// 0.5, anti-aligned below (pinned by tests).
#[must_use]
pub fn selection_gate(d: f32, al: f32, gamma: f32) -> f32 {
    crate::sigmoid(theorem_bound(d, al, gamma))
}

/// First index where `q_seq[i] < eps`; `None` when the sequence never pits.
///
/// The changepoint kernel of the PAV curation path (Issue 692 T2): on an
/// incorrect rollout the first pit is where the process-advantage estimate
/// first collapses — everything before it is a viable prefix label, at/after
/// it the sequence is failing. Strict `<` (the twin's semantics): a value
/// exactly at ε has not yet pitted.
///
/// Deterministic, allocation-free, O(n) short-circuit on the first hit.
#[must_use]
pub fn first_pit(q_seq: &[f32], eps: f32) -> Option<usize> {
    q_seq.iter().position(|&q| q < eps)
}

#[cfg(test)]
mod tests {
    use super::{alignment, distinguishability, first_pit, selection_gate, theorem_bound};

    #[test]
    fn empty_sequence_never_pits() {
        assert_eq!(first_pit(&[], 0.5), None);
    }

    #[test]
    fn pits_at_first_strict_crossing() {
        // the FIRST crossing only — later recoveries never mask it
        let seq = [0.9, 0.8, 0.2, 0.9, 0.1];
        assert_eq!(first_pit(&seq, 0.5), Some(2));
    }

    #[test]
    fn exact_threshold_has_not_pitted() {
        // strict <: a value exactly at eps is not yet a pit
        assert_eq!(first_pit(&[0.5, 0.5], 0.5), None);
        assert_eq!(first_pit(&[0.5, 0.49], 0.5), Some(1));
    }

    #[test]
    fn first_element_can_pit() {
        assert_eq!(first_pit(&[0.1, 0.9], 0.5), Some(0));
    }

    #[test]
    fn all_above_never_pits() {
        assert_eq!(first_pit(&[0.9, 0.8, 0.7], 0.5), None);
    }

    #[test]
    fn zero_and_negative_eps() {
        // eps = 0: only negative Q̂ pits (Q̂ ∈ [0,1] normally — the guard
        // for adversarial/NaN-free inputs is the caller's)
        assert_eq!(first_pit(&[0.0, -0.1], 0.0), Some(1));
        assert_eq!(first_pit(&[0.0, 0.1], 0.0), None);
    }

    #[test]
    fn nan_compares_false_so_never_pits() {
        // IEEE: NaN < eps is false — a NaN element never triggers; the
        // kernel stays total (the PAV pipeline's estimator is L2-validated,
        // but the kernel does not assume it)
        assert_eq!(first_pit(&[f32::NAN, 0.9], 0.5), None);
        assert_eq!(first_pit(&[0.9, f32::NAN, 0.1], 0.5), Some(2));
    }

    #[test]
    fn matches_the_riir_clippy_twin_bit_for_bit() {
        // the twin's exact semantics (riir-clippy src/pav_data/mod.rs):
        // q_seq.iter().position(|&q| q < eps) — same expression, same
        // strictness. Pins the swap-to-substrate contract.
        let twin = |q_seq: &[f32], eps: f32| q_seq.iter().position(|&q| q < eps);
        let seqs: [&[f32]; 6] = [
            &[],
            &[0.9],
            &[0.5],
            &[0.2, 0.9, 0.1],
            &[0.9, 0.8, 0.7],
            &[f32::NAN, 0.0, -1.0, 0.9],
        ];
        for seq in seqs {
            for eps in [0.0_f32, 0.25, 0.5, 1.0] {
                assert_eq!(first_pit(seq, eps), twin(seq, eps));
            }
        }
    }

    // ── T1: distinguishability + alignment + theorem bound + gate ─────────

    const EPS: f32 = 1e-6;

    #[test]
    fn two_action_state_hand_computed_d_and_al() {
        // quarters are exact in f32: V = 0.5, A = ±0.25, every product exact.
        let base: [&[f32]; 1] = [&[0.25, 0.75]];
        let anti: [&[f32]; 1] = [&[0.75, 0.25]];
        // D: mean of A² = (0.0625 + 0.0625)/2
        assert_eq!(distinguishability(&base, &anti), 0.0625);
        // Al: mean of A^μ·A^π = (−0.0625 + −0.0625)/2
        assert_eq!(alignment(&base, &anti), -0.0625);
    }

    #[test]
    fn mu_equals_pi_collapses_alignment_onto_distinguishability() {
        // the paper's μ=π degeneracy, exact: Al(π) = E[(A^π)²] = D(π)
        let base: [&[f32]; 2] = [&[0.25, 0.75], &[0.1, 0.9]];
        let d = distinguishability(&base, &base);
        let al = alignment(&base, &base);
        assert_eq!(d, al);
        // state 1: A = ±0.4 → E[A²] = 0.16; state 0: 0.0625 → mean 0.11125
        assert!((d - 0.11125).abs() < EPS);
    }

    #[test]
    fn too_strong_prover_has_zero_distinguishability() {
        // succeeds from any prefix ⇒ Q^μ flat at 1 ⇒ A^μ ≡ 0 (paper §1.2)
        let base: [&[f32]; 2] = [&[0.25, 0.75], &[0.1, 0.6, 0.9]];
        let strong: [&[f32]; 2] = [&[1.0, 1.0], &[1.0, 1.0, 1.0]];
        assert_eq!(distinguishability(&base, &strong), 0.0);
        assert_eq!(alignment(&base, &strong), 0.0);
    }

    #[test]
    fn too_weak_prover_has_zero_distinguishability() {
        // fails from any prefix ⇒ Q^μ flat at 0 ⇒ A^μ ≡ 0
        let base: [&[f32]; 2] = [&[0.25, 0.75], &[0.1, 0.6, 0.9]];
        let weak: [&[f32]; 2] = [&[0.0, 0.0], &[0.0, 0.0, 0.0]];
        assert_eq!(distinguishability(&base, &weak), 0.0);
        assert_eq!(alignment(&base, &weak), 0.0);
    }

    #[test]
    fn constant_within_state_prover_zero_d_any_level() {
        // flat within the state at ANY level (strength without spread) —
        // the exact selector difference vs strength ranking
        let base: [&[f32]; 2] = [&[0.25, 0.75], &[0.4, 0.6]];
        let flat: [&[f32]; 2] = [&[0.3, 0.3], &[0.9, 0.9]];
        assert_eq!(distinguishability(&base, &flat), 0.0);
        assert_eq!(alignment(&base, &flat), 0.0);
    }

    #[test]
    fn d_averages_over_states() {
        let base: [&[f32]; 2] = [&[0.25, 0.75], &[0.5, 0.5]];
        let prover: [&[f32]; 2] = [&[0.25, 0.75], &[0.5, 0.5]];
        // state 0: 0.0625, state 1: 0 (flat) → mean 0.03125 (exact: dyadic)
        assert_eq!(distinguishability(&base, &prover), 0.03125);
    }

    #[test]
    fn entry_multiplicity_is_the_policy_weight() {
        // duplicating an entry re-weights π(a|s), shifting V^μ and D — the
        // documented weighting mechanism, exercised on both sides.
        let two: [&[f32]; 1] = [&[0.0, 1.0]];
        assert_eq!(distinguishability(&two, &two), 0.25); // V=.5, A=±.5
        // the failing action duplicated: π-weight 2/3 on 0.0 → V=1/3,
        // A = [−1/3, −1/3, 2/3], mean square = 2/9
        let dup: [&[f32]; 1] = [&[0.0, 0.0, 1.0]];
        assert!((distinguishability(&dup, &dup) - 2.0 / 9.0).abs() < EPS);
        assert!(distinguishability(&dup, &dup) < 0.25);
    }

    #[test]
    fn empty_logs_and_empty_states_yield_zero() {
        let empty: [&[f32]; 0] = [];
        assert_eq!(distinguishability(&empty, &empty), 0.0);
        assert_eq!(alignment(&empty, &empty), 0.0);
        let one_empty: [&[f32]; 1] = [&[]];
        assert_eq!(distinguishability(&one_empty, &one_empty), 0.0);
        assert_eq!(alignment(&one_empty, &one_empty), 0.0);
    }

    #[test]
    fn single_action_states_contribute_nothing() {
        // one action ⇒ no spread ⇒ A = 0 exactly
        let base: [&[f32]; 2] = [&[0.7], &[0.25, 0.75]];
        let prover: [&[f32]; 2] = [&[0.9], &[0.75, 0.25]];
        // only state 1 contributes: 0.0625 / 2 states
        assert_eq!(distinguishability(&base, &prover), 0.03125);
        assert_eq!(alignment(&base, &prover), -0.03125);
    }

    #[test]
    fn exhaustive_bernoulli_grid_invariants() {
        // synthetic Bernoulli grid (Q̂ ∈ {0, 0.1, …, 1}², single state):
        // D ≥ 0 and ≤ 0.25 (Popoviciu on [0,1] values), and the
        // Cauchy–Schwarz cap |Al| ≤ √(D(μ)·D(π)) everywhere.
        let base: [&[f32]; 1] = [&[0.25, 0.75]];
        let d_pi = distinguishability(&base, &base);
        let vals = [0.0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        for q0 in vals {
            for q1 in vals {
                let prover: [&[f32]; 1] = [&[q0, q1]];
                let d = distinguishability(&base, &prover);
                let al = alignment(&base, &prover);
                assert!(
                    d.is_finite() && d >= 0.0 && d <= 0.25,
                    "D out of range at [{q0}, {q1}]: {d}"
                );
                let cs_cap = (d * d_pi).sqrt();
                assert!(
                    al.abs() <= cs_cap + EPS,
                    "Cauchy-Schwarz violated at [{q0}, {q1}]: |{al}| > {cs_cap}"
                );
            }
        }
    }

    #[test]
    fn theorem_bound_is_gamma_times_the_sum() {
        assert_eq!(theorem_bound(0.0625, 0.0625, 0.9), 0.9 * (0.0625 + 0.0625));
        assert_eq!(theorem_bound(0.0, -0.0625, 1.0), -0.0625);
        assert_eq!(theorem_bound(0.25, 0.25, 0.0), 0.0);
        // anti-alignment dominating the bound predicts net harm — the
        // pre-gate's skip condition
        assert!(theorem_bound(0.0625, -0.125, 1.0) < 0.0);
    }

    #[test]
    fn selection_gate_is_bounded_and_order_preserving() {
        // bounded strictly inside (0,1)
        let g_hi = selection_gate(0.25, 0.25, 1.0);
        let g_lo = selection_gate(0.0, -0.25, 1.0);
        assert!(g_hi > 0.5 && g_hi < 1.0);
        assert!(g_lo < 0.5 && g_lo > 0.0);
        // monotone in the bound: ranking by gate == ranking by bound
        let a = selection_gate(0.2, 0.0, 1.0);
        let b = selection_gate(0.1, 0.0, 1.0);
        let c = selection_gate(0.0, 0.0, 1.0);
        assert!(a > b && b > c);
        assert!((c - 0.5).abs() < EPS); // zero bound gates at the midpoint
        // aligned above 0.5, anti-aligned below — at the same D (the
        // anti-aligned bound must go strictly negative: Al dominating D)
        assert!(selection_gate(0.0625, 0.0625, 0.9) > 0.5);
        assert!(selection_gate(0.0625, -0.125, 0.9) < 0.5);
    }

    #[test]
    fn nan_propagates_screening_is_the_callers_contract() {
        // the kernels stay total: NaN inputs surface as NaN outputs (never
        // as a plausible silent number) — same stance as first_pit. Note D
        // reads only the prover log (the base log defines the support), so a
        // base-side NaN surfaces through alignment, not D.
        let clean: [&[f32]; 1] = [&[0.5, 0.5]];
        let nan_prover: [&[f32]; 1] = [&[f32::NAN, 0.5]];
        let nan_base: [&[f32]; 1] = [&[f32::NAN, 0.5]];
        assert!(distinguishability(&clean, &nan_prover).is_nan());
        assert!(alignment(&nan_base, &clean).is_nan());
        assert!(alignment(&clean, &nan_prover).is_nan());
    }
}
