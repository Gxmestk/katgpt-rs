//! Candidate-enumeration policies for SipIt inversion.
//!
//! The [`InversionPolicy`] enum selects how the driver enumerates the
//! vocabulary `V` at each position. The default [`RandomPolicy`] is
//! uniform-without-replacement: worst case `|V|` trials per position
//! (amortized `|V|/2`), but the random ordering makes the worst case
//! astronomically unlikely on real transformers (paper §E.1 reports
//! gradient-guided finds the token in <0.25% of |V|; uniform-random does
//! noticeably worse but is still correct in the limit).
//!
//! Phase 2 adds [`GradientGuidedPolicy`] behind the `grad_policy` sub-feature:
//! ranks candidates by L∞ distance of `F(v; π, t)` from `h̆_t`, using the
//! caller-supplied gradient hook (paper Alg 3). The Phase 1 driver never
//! reaches this branch.

use fastrand::Rng;

/// Policy selector. See module docs.
#[derive(Clone, Copy, Debug, Default)]
pub enum InversionPolicy {
    /// Uniform-without-replacement. Worst case `T · |V|` trials.
    #[default]
    Random,
    /// Gradient-guided ranking (paper Alg 3). Phase 2 — caller must supply
    /// `crate::inversion::InversionGradient`.
    #[cfg(feature = "grad_policy")]
    GradientGuided { step_size: f32, grad_clip: f32 },
}

/// Uniform-without-replacement vocabulary enumeration.
///
/// Generated lazily via a Fisher-Yates shuffle on a `Vec<u32>` of length
/// `|V|`; the shuffle is allocation-free on subsequent positions (the
/// permutation buffer is reused). The Rng is held by the policy so that
/// two consecutive calls within the same driver run produce different
/// orderings.
pub struct RandomPolicy {
    rng: Rng,
    permutation: Vec<u32>,
    cursor: usize,
}

impl RandomPolicy {
    /// Construct for a vocabulary of size `vocab_size`. Allocates one
    /// `Vec<u32>` of length `vocab_size`; this is a one-time setup cost,
    /// not a per-position allocation.
    pub fn new(vocab_size: u32, seed: u64) -> Self {
        let permutation: Vec<u32> = (0..vocab_size).collect();
        Self {
            rng: Rng::with_seed(seed),
            permutation,
            cursor: 0,
        }
    }

    /// Reset for the next position: re-shuffle the remaining-vocabulary
    /// permutation and reset the cursor. Allocation-free.
    ///
    /// Per AGENTS.md hot-loop rules, we use a Fisher-Yates shuffle in-place
    /// on the existing buffer.
    pub fn reset(&mut self) {
        let n = self.permutation.len();
        for i in (1..n).rev() {
            let j = self.rng.usize(0..=i);
            self.permutation.swap(i, j);
        }
        self.cursor = 0;
    }

    /// Return the next candidate token, or `None` if the vocabulary is
    /// exhausted. Allocation-free.
    #[inline]
    pub fn next_candidate(&mut self) -> Option<u32> {
        if self.cursor < self.permutation.len() {
            let v = self.permutation[self.cursor];
            self.cursor += 1;
            Some(v)
        } else {
            None
        }
    }

    /// Number of candidates already returned by `next_candidate` since the
    /// last `reset`.
    #[inline]
    pub fn candidates_tried(&self) -> usize {
        self.cursor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn random_policy_visits_each_token_once_per_reset() {
        let mut p = RandomPolicy::new(8, 0);
        p.reset();
        let mut seen: HashSet<u32> = HashSet::new();
        let mut count = 0;
        while let Some(v) = p.next_candidate() {
            assert!(seen.insert(v), "token {v} returned twice in one pass");
            count += 1;
        }
        assert_eq!(count, 8);
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn random_policy_two_passes_can_differ() {
        // With seed 0 on vocab 8, the two passes should differ in ordering
        // at least once (probability of identical orderings is 1/8! ≈ 2.5e-5).
        let mut p = RandomPolicy::new(8, 0);
        p.reset();
        let first: Vec<u32> = std::iter::from_fn(|| p.next_candidate()).collect();
        p.reset();
        let second: Vec<u32> = std::iter::from_fn(|| p.next_candidate()).collect();
        assert_eq!(first.len(), 8);
        assert_eq!(second.len(), 8);
        // Same set, possibly different order.
        let s1: HashSet<u32> = first.iter().copied().collect();
        let s2: HashSet<u32> = second.iter().copied().collect();
        assert_eq!(s1, s2);
    }

    #[test]
    fn random_policy_candidates_tried_advances() {
        let mut p = RandomPolicy::new(4, 1);
        p.reset();
        assert_eq!(p.candidates_tried(), 0);
        let _ = p.next_candidate();
        assert_eq!(p.candidates_tried(), 1);
        let _ = p.next_candidate();
        let _ = p.next_candidate();
        assert_eq!(p.candidates_tried(), 3);
    }

    #[test]
    fn random_policy_returns_none_after_exhaustion() {
        let mut p = RandomPolicy::new(2, 0);
        p.reset();
        let _ = p.next_candidate();
        let _ = p.next_candidate();
        assert!(p.next_candidate().is_none());
    }
}
