//! Local verifier — acceptance region + per-candidate check.
//!
//! Implements the local test from paper §3: candidate `v` is accepted at
//! position `t` under prefix `π` iff `‖h̆_t − F(v; π, t)‖_∞ ≤ ε`. The L∞ norm
//! is a sufficient (conservative) proxy for the L2 acceptance region used in
//! the paper's pseudocode; on toy transformers in f32 it produces identical
//! verdicts at the same ε up to float precision.
//!
//! Per AGENTS.md hot-loop rules: the check is allocation-free. The
//! `_into` variant writes the residual into a caller-supplied buffer
//! (useful for gradient-guided ranking in Phase 2).

use crate::inversion::InversionError;

/// Acceptance region `A_{π,t}(v; ε)` for SipIt.
///
/// Parameterized by the tolerance `ε`. The acceptance decision is
/// `max_i |h̆_t[i] − F(v; π, t)[i]| ≤ ε`.
#[derive(Clone, Copy, Debug)]
pub struct AcceptanceRegion {
    pub tolerance: f32,
}

impl AcceptanceRegion {
    #[inline]
    pub fn new(tolerance: f32) -> Self {
        Self { tolerance }
    }

    /// Check whether `candidate_state` falls in the acceptance region around
    /// `observed_state`. Allocation-free.
    #[inline]
    pub fn contains(&self, observed_state: &[f32], candidate_state: &[f32]) -> bool {
        observed_state.len() == candidate_state.len()
            && observed_state
                .iter()
                .zip(candidate_state.iter())
                .all(|(o, c)| (*o - *c).abs() <= self.tolerance)
    }
}

/// Convenience: check acceptance and return the residual L∞ norm.
///
/// Writes the per-coord residuals into `residual_scratch` (length `d`); the
/// returned scalar is `max_i |residual_scratch[i]|`. Useful for Phase 2's
/// gradient-guided ranking, which needs the residual vector.
///
/// Returns `Err(InversionError::ScratchLenMismatch)` if the scratch buffer
/// is the wrong length.
#[inline]
pub fn accept_observation_into(
    observed_state: &[f32],
    candidate_state: &[f32],
    residual_scratch: &mut [f32],
) -> Result<f32, InversionError> {
    if residual_scratch.len() != observed_state.len() {
        return Err(InversionError::ScratchLenMismatch {
            expected: observed_state.len(),
            got: residual_scratch.len(),
        });
    }
    let mut max_abs: f32 = 0.0;
    for ((o, c), r) in observed_state
        .iter()
        .zip(candidate_state.iter())
        .zip(residual_scratch.iter_mut())
    {
        let diff = *o - *c;
        let abs = diff.abs();
        *r = diff;
        if abs > max_abs {
            max_abs = abs;
        }
    }
    Ok(max_abs)
}

/// Convenience: check whether `candidate_state` is within `tolerance`
/// (L∞ norm) of `observed_state`. Returns the L∞ residual (so the caller can
/// short-circuit on the first pass without recomputing). Allocation-free.
#[inline]
pub fn accept_observation(
    observed_state: &[f32],
    candidate_state: &[f32],
    tolerance: f32,
) -> (bool, f32) {
    debug_assert_eq!(observed_state.len(), candidate_state.len());
    let mut max_abs: f32 = 0.0;
    for (o, c) in observed_state.iter().zip(candidate_state.iter()) {
        let abs = (*o - *c).abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    (max_abs <= tolerance, max_abs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_observation_zero_residual_passes() {
        let o = [1.0_f32, 2.0, 3.0];
        let c = [1.0_f32, 2.0, 3.0];
        let (accepted, residual) = accept_observation(&o, &c, 1e-3);
        assert!(accepted);
        assert_eq!(residual, 0.0);
    }

    #[test]
    fn accept_observation_within_tolerance_passes() {
        let o = [1.0_f32, 2.0, 3.0];
        let c = [1.0_f32 + 5e-4, 2.0 - 5e-4, 3.0];
        let (accepted, residual) = accept_observation(&o, &c, 1e-3);
        assert!(accepted, "residual={residual} should be within 1e-3");
        assert!((residual - 5e-4).abs() < 1e-6);
    }

    #[test]
    fn accept_observation_above_tolerance_fails() {
        let o = [1.0_f32, 2.0, 3.0];
        let c = [1.0_f32 + 2e-3, 2.0, 3.0];
        let (accepted, residual) = accept_observation(&o, &c, 1e-3);
        assert!(!accepted);
        assert!((residual - 2e-3).abs() < 1e-6);
    }

    #[test]
    fn accept_observation_into_writes_residuals() {
        let o = [1.0_f32, 2.0, 3.0];
        let c = [1.5_f32, 2.5, 2.5];
        let mut scratch = [0.0_f32; 3];
        let max_abs = accept_observation_into(&o, &c, &mut scratch).unwrap();
        assert_eq!(max_abs, 0.5);
        assert_eq!(scratch, [-0.5, -0.5, 0.5]);
    }

    #[test]
    fn accept_observation_into_rejects_wrong_scratch_len() {
        let o = [1.0_f32, 2.0, 3.0];
        let c = [1.0_f32, 2.0, 3.0];
        let mut scratch = [0.0_f32; 2];
        let err = accept_observation_into(&o, &c, &mut scratch).unwrap_err();
        assert_eq!(
            err,
            InversionError::ScratchLenMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn acceptance_region_contains_handles_unequal_lengths() {
        let r = AcceptanceRegion::new(1e-3);
        // Unequal lengths: `contains` returns false (no panic).
        assert!(!r.contains(&[1.0_f32, 2.0, 3.0], &[1.0_f32, 2.0]));
    }
}
