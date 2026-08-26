//! Prover-selection statistics + cross-state advantage centering
//! (katgpt-rs Issue 692, from Research 509 — arXiv:2410.08146 Setlur et al.).
//!
//! The paper's result: per-step supervision should be **advantages under a
//! complementary prover policy**, and the right way to *select* a
//! prover/critic/verifier is not strength but distinguishability +
//! alignment (Theorem 3.1: improvement ≳ γ·(D + Al)). This module hosts the
//! modelless kernels; T2 ships the changepoint kernel first — the D/Al
//! primitives (T1) and the K\* law gate (T3) land beside it.
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
    use super::first_pit;

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
}
