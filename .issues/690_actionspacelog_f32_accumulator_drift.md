# Issue 690 — `ActionSpaceLog::avg_action_space()` drifts low: f32 running sum

Filed 2026-08-25. Found while building the Plan 348 Item C constraint DSL
(riir-train), which consumes `ActionSpaceLog` as its branching-factor
instrument.

## Symptom

`ActionSpaceLog::avg_action_space()` under-reports the mean action space once
the accumulated total passes ~2²⁴. Measured on a 7×7 `GoState`:

| records | true per-step | `avg_action_space()` | error |
|---|---|---|---|
| 2,260,000 | 50 | 48.593884 | **−2.81%** |

2.26M records is not a stress figure — it is one arm-seed of the Plan 348 go
arena (300 steps × 24 tasks × K=8, one record per policy move).

## Cause

`ActionSpaceLog` keeps `total_sum: f32` (`traits/mod.rs`), incremented once per
`record()`:

```rust
self.total_sum += n as f32;
...
pub fn avg_action_space(&self) -> f32 {
    if self.entries.is_empty() { 0.0 } else { self.total_sum / self.entries.len() as f32 }
}
```

At 2.26M × 50 the running total reaches ~1.13e8, where one f32 ULP is 8. Each
`+= 50` then rounds, and the rounding is systematically downward, so the sum
lags the true total and the mean with it. `PlayerAgg::sum` is also `f32`, so
`avg_action_space_for()` has the identical defect.

`peak_action_space()` is a `usize` and is **not** affected.

## How it surfaced (why this is worth fixing, not just documenting)

A constraint-DSL sweep reported **0.5% of the action space pruned on an
unconstrained control arm** — an arm with no constraint active, where the
correct answer is exactly 0. The "before" mean came from
`avg_action_space()` (f32, drifting) and the "after" mean from an f64
accumulator; the gap between the two accumulators looked like a real pruning
effect. A reader would have taken it for one.

## Fix directions

1. **`total_sum: f64` + `PlayerAgg::sum: f64`** (keep the `f32` return type by
   casting at the boundary). One-line-ish, no API change, no allocation
   change; `f64` holds exact integer sums to 2⁵³, which this can never reach.
2. Or compute the mean from `entries` on demand — but `entries` is already
   `Vec<(usize, u32, u8)>` and the O(1) read is the point of the running sum,
   so (1) is preferred.

Either way, add a test that records >2²⁴ worth of actions and asserts the mean
is exact — the current tests all run at small counts, which is why this
survived.

## Consumer status

`riir-train`'s `arena_constraints::BranchingProfile` (Plan 348 Item C) now
accumulates BOTH halves in f64 itself and treats `mean_before()` as
authoritative; it still records into `ActionSpaceLog` and still consumes
`peak_action_space()` (exact), and exposes `log_mean_before()` so the drift
stays visible rather than silently averaged away. That consumer-side guard can
be removed once this is fixed.
