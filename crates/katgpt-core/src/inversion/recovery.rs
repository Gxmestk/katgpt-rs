//! Inversion driver — outer T-loop × inner |V|-loop, prefix-conditioned.
//!
//! The driver maintains the invariant that at position `t`, the prefix `π`
//! already contains `t` verified tokens (the recovered prefix). For each
//! candidate `v` produced by the policy, it calls `InversionForward::hidden_at_into`
//! to get `F(v; π, t)`, then checks acceptance via [`crate::inversion::verifier::accept_observation`].
//! On acceptance, the candidate is appended to the prefix and the outer
//! loop advances. On exhaustion of the policy (or hitting `max_trials_per_position`),
//! the driver returns [`InversionResult::Failed`] with the position and
//! number of candidates tried.
//!
//! Per AGENTS.md hot-loop rules: zero per-iteration allocation after setup.
//! The prefix `π` grows by one `u32` per outer iteration (amortized via
//! `Vec::push`); the candidate-state scratch is a single `&mut [f32]` of
//! length `d` that the caller supplies once. The [`RandomPolicy`]'s
//! permutation buffer is allocated once in `new` and reused.

use crate::inversion::{
    InversionConfig, InversionError, InversionForward, InversionResult, InversionPolicy,
    ObservedStates, RandomPolicy,
};
use crate::inversion::verifier::accept_observation;

/// Run SipIt inversion with a caller-supplied scratch buffer.
///
/// This is the allocation-free hot path. `scratch` must have length
/// `observed.d_len` (one row of hidden state); the driver writes the
/// candidate state `F(v; π, t)` into it on each trial. `seed` seeds the
/// `RandomPolicy`'s RNG; two runs with the same `(observed, vocab_size,
/// forward, config, seed)` produce bit-identical results (Phase 1 G4
/// determinism property).
///
/// Returns the recovered token sequence on success, or `Failed` if no
/// candidate was accepted at some position within `max_trials_per_position`.
#[allow(clippy::missing_errors_doc)]
pub fn invert_sequence_into<F: InversionForward>(
    observed: &ObservedStates<'_>,
    vocab_size: u32,
    forward: &F,
    config: &InversionConfig,
    scratch: &mut [f32],
    seed: u64,
) -> Result<InversionResult, InversionError> {
    if vocab_size == 0 || observed.t_len == 0 {
        return Err(InversionError::EmptyInput);
    }
    if scratch.len() != observed.d_len {
        return Err(InversionError::ScratchLenMismatch {
            expected: observed.d_len,
            got: scratch.len(),
        });
    }

    let mut prefix: Vec<u32> = Vec::with_capacity(observed.t_len);
    let mut random_policy = RandomPolicy::new(vocab_size, seed);

    for t in 0..observed.t_len {
        let observed_state = observed.row(t);
        let trials = run_one_position(
            observed_state,
            t,
            &prefix,
            forward,
            config,
            scratch,
            &mut random_policy,
        )?;
        match trials {
            PositionOutcome::Accepted(token) => prefix.push(token),
            PositionOutcome::Exhausted { candidates_tried } => {
                return Ok(InversionResult::Failed {
                    failed_position: t,
                    candidates_tried,
                });
            }
        }
    }

    Ok(InversionResult::Recovered(prefix))
}

/// Convenience: same as [`invert_sequence_into`] but allocates the scratch
/// buffer internally. Useful for one-off inversions; hot paths should reuse
/// a caller-owned scratch via `_into`.
#[allow(clippy::missing_errors_doc)]
pub fn invert_sequence<F: InversionForward>(
    observed: &ObservedStates<'_>,
    vocab_size: u32,
    forward: &F,
    config: &InversionConfig,
    seed: u64,
) -> Result<InversionResult, InversionError> {
    let mut scratch = vec![0.0_f32; observed.d_len];
    invert_sequence_into(observed, vocab_size, forward, config, &mut scratch, seed)
}

enum PositionOutcome {
    Accepted(u32),
    Exhausted { candidates_tried: usize },
}

#[inline]
fn run_one_position<F: InversionForward>(
    observed_state: &[f32],
    position: usize,
    prefix: &[u32],
    forward: &F,
    config: &InversionConfig,
    scratch: &mut [f32],
    random_policy: &mut RandomPolicy,
) -> Result<PositionOutcome, InversionError> {
    debug_assert_eq!(scratch.len(), observed_state.len());

    // Phase 1 only supports the Random policy. Phase 2 will branch on
    // `config.policy` and use the gradient hook when `GradientGuided` is set.
    match config.policy {
        InversionPolicy::Random => {}
        #[cfg(feature = "grad_policy")]
        InversionPolicy::GradientGuided { .. } => {
            // Phase 2 — not implemented in Phase 1. Fall through to Random
            // so the API is callable without panicking; Phase 2 will replace
            // this stub with the actual gradient-guided ranking.
        }
    }

    random_policy.reset();
    let mut candidates_tried = 0_usize;
    while let Some(candidate) = random_policy.next_candidate() {
        candidates_tried += 1;
        forward.hidden_at_into(prefix, candidate, position, scratch)?;
        let (accepted, _residual) = accept_observation(observed_state, scratch, config.tolerance);
        if accepted {
            return Ok(PositionOutcome::Accepted(candidate));
        }
        if candidates_tried >= config.max_trials_per_position {
            return Ok(PositionOutcome::Exhausted { candidates_tried });
        }
    }
    Ok(PositionOutcome::Exhausted { candidates_tried })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inversion::{InversionConfig, InversionResult};

    /// Trivial forward: state at position `t` is `[v as f32 * t, v as f32]`.
    /// Different `(v, t)` → different state, so the "true" prompt is exactly
    /// recoverable.
    struct ToyForward;

    impl InversionForward for ToyForward {
        fn hidden_at_into(
            &self,
            _prefix: &[u32],
            candidate: u32,
            position: usize,
            out: &mut [f32],
        ) -> Result<(), InversionError> {
            // Length-2 hidden state: deterministic in (v, t).
            out[0] = candidate as f32 * position as f32;
            out[1] = candidate as f32;
            Ok(())
        }
    }

    fn make_observed(prompt: &[u32]) -> Vec<f32> {
        let d = 2_usize;
        let mut out = Vec::with_capacity(prompt.len() * d);
        for (t, &v) in prompt.iter().enumerate() {
            out.push(v as f32 * t as f32);
            out.push(v as f32);
        }
        out
    }

    #[test]
    fn invert_recovers_simple_prompt() {
        let prompt: Vec<u32> = vec![3, 1, 4, 1, 5];
        let buf = make_observed(&prompt);
        let observed = ObservedStates::from_row_major(&buf, prompt.len(), 2).unwrap();
        let cfg = InversionConfig::default();
        let result = invert_sequence(&observed, 8, &ToyForward, &cfg, 0).unwrap();
        match result {
            InversionResult::Recovered(recovered) => assert_eq!(recovered, prompt),
            other => panic!("expected Recovered, got {other:?}"),
        }
    }

    #[test]
    fn invert_into_reuses_scratch() {
        let prompt: Vec<u32> = vec![2, 7, 1];
        let buf = make_observed(&prompt);
        let observed = ObservedStates::from_row_major(&buf, prompt.len(), 2).unwrap();
        let cfg = InversionConfig::default();
        let mut scratch = [0.0_f32; 2];
        let result =
            invert_sequence_into(&observed, 8, &ToyForward, &cfg, &mut scratch, 42).unwrap();
        match result {
            InversionResult::Recovered(recovered) => assert_eq!(recovered, prompt),
            other => panic!("expected Recovered, got {other:?}"),
        }
    }

    #[test]
    fn invert_fails_on_wrong_observed() {
        // Wrong observed states → no candidate will match.
        let bogus_observed: Vec<f32> = (0..8).map(|i| i as f32 * 100.0).collect();
        let observed = ObservedStates::from_row_major(&bogus_observed, 4, 2).unwrap();
        let cfg = InversionConfig::default();
        let result = invert_sequence(&observed, 8, &ToyForward, &cfg, 0).unwrap();
        match result {
            InversionResult::Failed {
                failed_position: 0, ..
            } => {}
            other => panic!("expected Failed at position 0, got {other:?}"),
        }
    }

    #[test]
    fn invert_fails_on_max_trials_per_position() {
        // Same wrong observed as above, but cap trials at 2 → fails fast.
        let bogus_observed: Vec<f32> = (0..8).map(|i| i as f32 * 100.0).collect();
        let observed = ObservedStates::from_row_major(&bogus_observed, 4, 2).unwrap();
        let cfg = InversionConfig {
            max_trials_per_position: 2,
            ..InversionConfig::default()
        };
        let result = invert_sequence(&observed, 8, &ToyForward, &cfg, 0).unwrap();
        match result {
            InversionResult::Failed {
                failed_position: 0,
                candidates_tried,
            } => assert_eq!(candidates_tried, 2),
            other => panic!("expected Failed with 2 candidates, got {other:?}"),
        }
    }

    #[test]
    fn invert_empty_vocabulary_errors() {
        let observed_buf: Vec<f32> = vec![0.0; 4];
        let observed = ObservedStates::from_row_major(&observed_buf, 2, 2).unwrap();
        let cfg = InversionConfig::default();
        let err = invert_sequence(&observed, 0, &ToyForward, &cfg, 0).unwrap_err();
        assert_eq!(err, InversionError::EmptyInput);
    }

    #[test]
    fn invert_empty_observed_errors() {
        let observed = ObservedStates::from_row_major(&[], 0, 2).unwrap();
        let cfg = InversionConfig::default();
        let err = invert_sequence(&observed, 8, &ToyForward, &cfg, 0).unwrap_err();
        assert_eq!(err, InversionError::EmptyInput);
    }

    #[test]
    fn invert_into_wrong_scratch_len_errors() {
        let prompt: Vec<u32> = vec![3, 1];
        let buf = make_observed(&prompt);
        let observed = ObservedStates::from_row_major(&buf, 2, 2).unwrap();
        let cfg = InversionConfig::default();
        let mut scratch = [0.0_f32; 3]; // wrong length (should be 2)
        let err =
            invert_sequence_into(&observed, 8, &ToyForward, &cfg, &mut scratch, 0).unwrap_err();
        assert_eq!(
            err,
            InversionError::ScratchLenMismatch {
                expected: 2,
                got: 3
            }
        );
    }
}
