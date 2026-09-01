//! ThinkingController integration — Plan 195 T5.
//!
//! Extension functions to integrate chain folding with the ThinkingController
//! feedback loop. Converts fold statistics into thinking feedback.
//!
//! _Root-resident by design (Issue 033 §C, Option C)._ Integrates with
//! root-only `ThinkingController` (Plan 194) for thinking-mode feedback.

use super::types::{FoldResult, FoldStats, ThinkingFoldFeedback};

/// Convert a fold result into thinking feedback for the controller.
pub fn fold_thinking_feedback(result: &FoldResult, fold_budget: f32) -> ThinkingFoldFeedback {
    ThinkingFoldFeedback {
        tokens_saved: result.tokens_saved,
        steps_folded: result.folded_steps,
        fold_budget,
    }
}

/// Convert accumulated fold stats into a summary feedback.
pub fn fold_stats_feedback(stats: &FoldStats) -> ThinkingFoldFeedback {
    ThinkingFoldFeedback {
        tokens_saved: stats.total_tokens_saved,
        steps_folded: stats.total_steps_folded,
        fold_budget: 0.0, // Stats don't track per-query budget
    }
}

/// Calculate the effective token reduction ratio from fold stats.
///
/// Returns 0.0 if no queries have been folded.
pub fn token_reduction_ratio(stats: &FoldStats) -> f32 {
    if stats.queries_folded == 0 {
        return 0.0;
    }
    stats.total_tokens_saved as f32 / stats.queries_folded as f32
}

/// Calculate the effective step reduction ratio from fold stats.
///
/// Returns 0.0 if no queries have been folded.
pub fn step_reduction_ratio(stats: &FoldStats) -> f32 {
    if stats.queries_folded == 0 {
        return 0.0;
    }
    stats.total_steps_folded as f32 / stats.queries_folded as f32
}

// ── Issue 699 T2: structural CoT halt observer (opt-in) ────────────────

/// Opt-in answer-stream observer for the thinking trace (Issue 699 T2;
/// TRACE, arXiv:2510.07880).
///
/// `ThinkingController` decides WHETHER to think and has no answer-text
/// stream of its own (verified at Issue 699 T2 — it consumes confidences
/// and rewards, never step answers), so per the issue's plug-point rule the
/// structural monitor rides ALONGSIDE as an explicit observer: the caller
/// feeds each answer-bearing reasoning step and maps the returned halt
/// decision onto its own loop control. Reshaping the controller is
/// explicitly out of scope; wiring this into the controller's mode
/// selection is the T4 PoC's job.
///
/// Note: this file is `chain_fold`-gated (the fold module's own gate), so
/// the observer additionally requires `chain_fold` — an artifact of the
/// hook's sanctioned location, recorded here for feature-matrix honesty.
/// The monitor itself lives in `katgpt-core/structural_cot_halt` with no
/// chain_fold coupling.
#[cfg(feature = "structural_cot_halt")]
pub struct ThinkingTraceHaltObserver {
    monitor: katgpt_core::structural_cot_halt::StructuralTraceMonitor,
}

#[cfg(feature = "structural_cot_halt")]
impl ThinkingTraceHaltObserver {
    /// Construct with an explicit halting policy.
    pub fn new(policy: katgpt_core::structural_cot_halt::HaltPolicy) -> Self {
        Self {
            monitor: katgpt_core::structural_cot_halt::StructuralTraceMonitor::new(policy),
        }
    }

    /// Construct with the pattern-conditional fusion ([`HaltPolicy::Auto`]).
    pub fn auto() -> Self {
        Self {
            monitor: katgpt_core::structural_cot_halt::StructuralTraceMonitor::auto(),
        }
    }

    /// Feed one answer-bearing reasoning step; returns the halt decision.
    /// Zero-alloc, deterministic (see the monitor's contract).
    pub fn observe(
        &mut self,
        answer: &str,
    ) -> katgpt_core::structural_cot_halt::StructuralHaltDecision {
        self.monitor.step(answer)
    }

    /// Shared monitor access for composition (classify_prefix, counters,
    /// compose_votes against the numeric family).
    pub fn monitor(&mut self) -> &mut katgpt_core::structural_cot_halt::StructuralTraceMonitor {
        &mut self.monitor
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_thinking_feedback() {
        let result = FoldResult {
            total_steps: 10,
            kept_steps: 7,
            folded_steps: 3,
            tokens_saved: 50,
            retention_ratio: 0.7,
            verification_passed: true,
        };

        let feedback = fold_thinking_feedback(&result, 0.7);
        assert_eq!(feedback.tokens_saved, 50);
        assert_eq!(feedback.steps_folded, 3);
        assert!((feedback.fold_budget - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fold_stats_feedback() {
        let stats = FoldStats {
            total_tokens_saved: 500,
            total_steps_folded: 30,
            queries_folded: 10,
            verification_pass_rate: 0.9,
        };

        let feedback = fold_stats_feedback(&stats);
        assert_eq!(feedback.tokens_saved, 500);
        assert_eq!(feedback.steps_folded, 30);
    }

    #[test]
    fn test_token_reduction_ratio_empty() {
        let stats = FoldStats::default();
        assert_eq!(token_reduction_ratio(&stats), 0.0);
    }

    #[test]
    fn test_token_reduction_ratio() {
        let stats = FoldStats {
            total_tokens_saved: 100,
            total_steps_folded: 10,
            queries_folded: 5,
            verification_pass_rate: 1.0,
        };
        let ratio = token_reduction_ratio(&stats);
        assert!((ratio - 20.0).abs() < f32::EPSILON); // 100 / 5
    }

    #[test]
    fn test_step_reduction_ratio_empty() {
        let stats = FoldStats::default();
        assert_eq!(step_reduction_ratio(&stats), 0.0);
    }

    #[test]
    fn test_step_reduction_ratio() {
        let stats = FoldStats {
            total_tokens_saved: 100,
            total_steps_folded: 15,
            queries_folded: 3,
            verification_pass_rate: 0.8,
        };
        let ratio = step_reduction_ratio(&stats);
        assert!((ratio - 5.0).abs() < f32::EPSILON); // 15 / 3
    }

    // ── Issue 699 T2: structural halt observer ────────────────────

    #[cfg(feature = "structural_cot_halt")]
    #[test]
    fn test_thinking_trace_halt_observer_backtrack() {
        use katgpt_core::structural_cot_halt::StructuralHaltDecision;
        let mut obs = ThinkingTraceHaltObserver::auto();
        assert_eq!(obs.observe("42"), StructuralHaltDecision::Continue);
        assert_eq!(obs.observe("43"), StructuralHaltDecision::Continue);
        assert_eq!(
            obs.observe("42"),
            StructuralHaltDecision::Halt {
                reason: katgpt_core::structural_cot_halt::StructuralHaltReason::BacktrackRevisit,
                step: 3
            },
            "Auto resolves the explorer cycle → backtrack halt"
        );
        // Frozen episode.
        assert_eq!(
            obs.observe("44"),
            StructuralHaltDecision::Halt {
                reason: katgpt_core::structural_cot_halt::StructuralHaltReason::BacktrackRevisit,
                step: 3
            }
        );
    }

    #[cfg(feature = "structural_cot_halt")]
    #[test]
    fn test_thinking_trace_halt_observer_monitor_access() {
        use katgpt_core::structural_cot_halt::HaltPolicy;
        let mut obs = ThinkingTraceHaltObserver::new(HaltPolicy::Never);
        for answer in ["a", "b", "c", "a"] {
            assert!(!obs.observe(answer).is_halt());
        }
        assert_eq!(obs.monitor().revisit_count(), 1);
        assert_eq!(obs.monitor().step_count(), 4);
    }
}
