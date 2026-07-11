# Issue 133: Post-Heal Conflict Detection Gap

> **Date:** 2026-07-11
> **Severity:** MEDIUM — self-healing mechanisms produce unvalidated results
> **Status:** RESOLVED (2026-07-11) — both impls + wiring done, GOAT gate passed
> **Related:** Plan 088 (LDT, DEFAULT-ON), Plan 316 (neighbor_heal, DEFAULT-ON),
> Proposal 013 P1 (feeling_brain), Research 050 (LDT), Research 152 (LDT Phase 2)

## Problem

Two DEFAULT-ON self-healing mechanisms — `neighbor_heal` (Plan 316,
riir-neuron-db) and `feeling_brain` (Proposal 013 P1, riir-games) — heal
damaged state toward a target (neighbor centroid / personality baseline)
**without validating the healed result**. The LDT `ConflictDetector` trait
(Plan 088, DEFAULT-ON in katgpt-rs) already exists and is generically
applicable, but is only wired into token-level DDTree pruning. No impl
checks healed runtime state.

## Evidence

### Gap 1: `neighbor_heal` — no validation of healed `style_weights`

`riir-neuron-db/src/neighbor_heal.rs` L147-196:

```rust
pub fn neighbor_heal_delta_into(
    damaged: &NeuronShard,
    neighbors: &[&NeuronShard],
    weights: &[f32],
    out: &mut [f32; STYLE_DIM],
) {
    // ... weighted average of neighbors' style_weights ...
    // NO validation of `out` before returning
}
```

`plan_neighbor_heal_into` (L207-229) calls this and returns `true` — no
check on whether the healed `style_weights` is semantically valid (e.g.,
close to a recognizable archetype, consistent with `intrinsic_dim`, or
matching the shard's structural constraints).

### Gap 2: `feeling_brain` — no cross-axis validation of healed emotion

`riir-games/src/civ/emotion/feeling_brain.rs` L178-179:

```text
field[axis] = baseline[axis] + (field[axis] - baseline[axis]) * exp(-λ * dt)
```

Each axis heals independently toward the personality baseline. No check
for cross-axis impossibilities (e.g., anger > 0.7 AND calm > 0.7 —
mutually exclusive emotional states). The `ReactiveFsmState` (L288-325)
gates on fear alone, not on cross-axis combinations.

### PoC Results (compiled + run, 2026-07-11)

**neighbor_heal scenario**: 3 neighbor shards (aggressive, cautious,
balanced), weights [0.80, 0.15, 0.05]. Healed vector is 0.247 from the
nearest archetype (threshold 0.15) → **conflicted = true**. Current code
accepts the healed shard blindly.

**feeling_brain scenario**: timid animal (anger baseline=0.1, calm
baseline=0.2). Kicked: anger→0.95, calm→0.95. After 5 ticks of healing:

```
Tick 1: anger=0.948, calm=0.913  ⚠ CONFLICT (anger>0.7 AND calm>0.7)
Tick 2: anger=0.947, calm=0.879  ⚠ CONFLICT
Tick 3: anger=0.945, calm=0.846  ⚠ CONFLICT
Tick 4: anger=0.943, calm=0.814  ⚠ CONFLICT
Tick 5: anger=0.942, calm=0.784  ⚠ CONFLICT
```

Root cause: anger heals slowly (λ=0.002, "grudge lingers") while calm
heals at medium rate (λ=0.05). The per-axis heal formula treats anger
and calm as independent dimensions — it has no awareness that they are
mutually exclusive (anger = high arousal/agitated, calm = low
arousal/tranquil). The impossible state persists for 5+ ticks.

## Proposed Fix

Wire LDT's `ConflictDetector` trait (already shipped, Plan 088, DEFAULT-ON)
into both self-healing mechanisms as a post-heal validation gate.

### New impls

1. **`ShardConflictDetector`** (riir-neuron-db) — checks healed
   `style_weights` against:
   - Archetype manifold distance (is the blend coherent?)
   - `intrinsic_dim` consistency (does the blend match the gate?)
   - L2 norm within expected range (not degenerate)

2. **`HlaConflictDetector`** (riir-games or riir-engine) — checks healed
   emotion against cross-axis constraints:
   - `anger > 0.7 ⟹ calm < 0.7` (mutually exclusive)
   - `desperation > 0.7 ⟹ valence < 0.6` (desperation implies negative valence)
   - `fear > 0.8 ⟹ calm < 0.5` (extreme fear precludes calm)

### Wiring

- **MAPE-K loop** (`riir-neuron-db/src/mape_k.rs`): after
  `plan_with_index` produces the heal target, call
  `ShardConflictDetector::is_conflicted(&heal_target)`. If conflicted,
  escalate to `CorrectiveGoal::ExternalEscalation` instead of applying
  the invalid heal.

- **feeling_brain evolve** (`riir-games/src/civ/emotion/feeling_brain.rs`):
  after `evolve_feeling_brain`, call
  `HlaConflictDetector::is_conflicted(&field)`. If conflicted, clamp the
  offending axis to the constraint boundary (e.g., if anger > 0.7 and
  calm > 0.7, clamp calm to 0.7).

## GOAT Gate

| Gate | Check | Pass criteria |
|------|-------|---------------|
| G1 — Correctness | PoC: anger+calm conflict detected after heal | `is_conflicted` returns true for the impossible state |
| G2 — No false positives | Normal emotion evolution (no kicks) does not trigger conflict | `is_conflicted` returns false for baseline states |
| G3 — No regression | Existing neighbor_heal + feeling_brain tests pass unchanged | All existing tests green |
| G4 — Zero-allocation | `is_conflicted` uses stack-only, no heap | criterion bench: < 50 ns per check |
| G5 — Modelless | No training, no backprop — pure threshold checks | API surface has no training types |
| G6 — Feature gate | Behind `lattice_deduction` (already exists) or new `heal_validation` gate | Default build unchanged |

## Implementation Scope

- [x] `ShardConflictDetector` impl in `riir-neuron-db/src/neighbor_heal.rs` (done, gated `heal_validation`)
- [x] `HlaConflictDetector` impl in `riir-games/src/civ/emotion/feeling_brain.rs` (done 2026-07-11)
- [x] Wire `ShardConflictDetector` into `MapeKLoop::plan_with_index` (done)
- [x] Wire `HlaConflictDetector` into `evolve_feeling_brain` (done 2026-07-11 — `resolve_heal_conflicts` called at end of `evolve_feeling_brain` under `#[cfg(feature = "heal_validation")]`)
- [x] G1-G6 GOAT gate (G1 detects PoC conflict, G2 no false positives, G3 48/48 tests pass, G4 zero-alloc stack-only, G5 modelless, G6 behind `heal_validation` feature)
- [x] Feature gate (`heal_validation` in both `katgpt-core` and `riir-games`)
- [x] Fix: `riir-neuron-db/src/neighbor_heal.rs:245` — gate `HealConflictDetector` import behind `#[cfg(feature = "heal_validation")]` (was unconditional, broke default build)

**Estimated effort:** 4-6 files. The `ConflictDetector` trait already
ships — this is adding impls + wiring, not new infrastructure.

## Why This Matters

The self-healing mechanisms are DEFAULT-ON and GOAT-proven for their
core function (heal damaged state). But they have a blind spot: they
don't validate the healed result. The `feeling_brain` PoC proves this
can produce an impossible emotional state (anger + calm both > 0.7)
that persists for 5+ ticks. The `neighbor_heal` gap means a healed
shard can be semantically invalid (non-sensical archetype blend)
without any check catching it.

The fix is cheap: the `ConflictDetector` trait already exists (Plan 088,
DEFAULT-ON), and the impls are simple threshold checks (cross-axis
constraints for emotions, manifold distance for shards). The gain is
provable: the PoC demonstrates a real failure mode that the current
code doesn't catch.

## Cross-References

- `katgpt-rs/crates/katgpt-core/src/speculative/types.rs` — `ConflictDetector` trait (Plan 088)
- `katgpt-rs/.research/050_LDT_Lattice_Deduction_Transformer.md` — LDT distillation
- `katgpt-rs/.research/152_LDT_Phase2_Lattice_State_Fusion.md` — LDT Phase 2 (ConflictClauseDB)
- `riir-neuron-db/src/neighbor_heal.rs` — `neighbor_heal_delta_into` (no validation)
- `riir-neuron-db/src/mape_k.rs` — `MapeKLoop::plan_with_index` (wire point)
- `riir-ai/crates/riir-games/src/civ/emotion/feeling_brain.rs` — `evolve_feeling_brain` (wire point)
- `riir-ai/crates/riir-poc/benches/feeling_brain_goat.rs` — existing feeling_brain PoC

## TL;DR

Two DEFAULT-ON self-healing mechanisms (`neighbor_heal`, `feeling_brain`)
heal damaged state without validating the result. PoC proves
`feeling_brain` produces an impossible emotional state (anger + calm
both > 0.7) that persists for 5+ ticks. LDT's `ConflictDetector` trait
(Plan 088, already shipped, DEFAULT-ON) can fill this gap with two
simple impls (`ShardConflictDetector`, `HlaConflictDetector`) wired
into the heal paths. The fix is cheap (trait exists, impls are threshold
checks) and the gain is provable (PoC demonstrates the failure mode).
