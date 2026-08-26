//! Issue 690 regression gates: `ActionSpaceLog` f64 accumulator.
//!
//! The f32 running sum under-reported `avg_action_space()` once the total
//! passed ~2^24 (one f32 ULP > 1, so every `+= n` rounds — systematically
//! downward). Measured on a 7×7 go arena arm (2,260,000 records × 50
//! actions): 48.593884 vs true 50 = **−2.81%**. The defect surfaced as a
//! phantom "0.5% pruned on an unconstrained control arm" in the Plan 348
//! constraint-DSL sweep (f32 "before" vs f64 "after" accumulators).
//!
//! Fix: `total_sum: f64` + `PlayerAgg::sum: f64` — exact integer sums to
//! 2^53, unreachable at any arena scale. `f32` return types preserved
//! (cast at the boundary).

use super::ActionSpaceLog;
use crate::traits::GameState;

/// Constant action-space dummy — the go-arena shape (50 legal moves).
#[derive(Clone)]
struct ConstState {
    n: usize,
}

impl GameState for ConstState {
    type Action = usize;

    fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
        vec![0; self.n]
    }

    fn tick(&self) -> u32 {
        0
    }

    fn is_terminal(&self) -> bool {
        false
    }

    fn reward(&self, _player_id: u8) -> f32 {
        0.0
    }

    fn action_space_size(&self, _player_id: u8) -> usize {
        self.n
    }
}

/// The issue's measured regime: 2.26M records × 50 actions → total 1.13e8,
/// ~6.7× past the 2^24 f32 exact-integer ceiling. The mean must be EXACT.
#[test]
fn avg_exact_past_2p24_issue690() {
    const RECORDS: usize = 2_260_000;
    const N: usize = 50;
    let state = ConstState { n: N };
    let mut log = ActionSpaceLog::with_capacity(RECORDS);
    for _ in 0..RECORDS {
        log.record(&state, 0);
    }
    assert_eq!(log.len(), RECORDS);
    // Exact — no tolerance. f64 holds 1.13e8 exactly; the division by
    // 2.26e6 is exact in f64 and the f32 cast of 50.0 is exact.
    assert_eq!(log.avg_action_space(), 50.0f32);
    assert_eq!(log.avg_action_space_for(0), 50.0f32);
    assert_eq!(log.peak_action_space(), N);
}

/// The old f32 running sum over the SAME stream must measurably drift —
/// the A/B proving the defect was real (and the fix load-bearing).
/// Mirrors the pre-fix `total_sum: f32` accumulation shape exactly.
#[test]
fn old_f32_shape_fails_issue690() {
    const RECORDS: usize = 2_260_000;
    const N: usize = 50;
    let mut f32_sum = 0.0f32;
    for _ in 0..RECORDS {
        f32_sum += N as f32;
    }
    let f32_avg = f32_sum / RECORDS as f32;
    // Measured −2.81% at this scale; require the drift is clearly present
    // (>1%) so the gate stays falsifiable if f32 semantics ever change.
    let rel_err = ((f32_avg - 50.0f32) / 50.0f32).abs();
    assert!(
        rel_err > 0.01,
        "old f32 shape should drift >1% at 2.26M records, got {f32_avg} (rel {rel_err})"
    );
}

/// Small-count behavior unchanged: exact at the scales the existing
/// katgpt-pruners tests cover.
#[test]
fn small_counts_still_exact() {
    let state = ConstState { n: 3 };
    let mut log = ActionSpaceLog::new();
    log.record(&state, 0);
    log.record(&state, 0);
    log.record(&state, 1);
    assert_eq!(log.avg_action_space(), 3.0);
    assert_eq!(log.avg_action_space_for(0), 3.0);
    assert_eq!(log.avg_action_space_for(1), 3.0);
    assert_eq!(log.avg_action_space_for(99), 0.0);
    log.clear();
    assert_eq!(log.avg_action_space(), 0.0);
    assert_eq!(log.len(), 0);
}

/// Mixed action spaces with a large per-step count: f64 stays exact where
/// per-entry values themselves already exceed 2^24 (f32 represents only
/// EVEN integers there — so the fixtures use even values whose mean is
/// exactly representable, making the assertion tolerance-free).
#[test]
fn large_single_action_space_exact() {
    let big = ConstState { n: (1 << 24) + 8 };
    let bigger = ConstState { n: (1 << 24) + 32 };
    let mut log = ActionSpaceLog::new();
    log.record(&big, 0);
    log.record(&bigger, 0);
    // mean = 2^24 + 20 — even, exactly representable in f32.
    assert_eq!(log.avg_action_space(), (1u64 << 24) as f32 + 20.0);
}
