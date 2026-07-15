# Issue 149 — Full Guided-PIBT Flow Direction Assignment

**Status:** RESOLVED (2026-07-15) — mechanism implemented correctly, but has near-zero effect on real maps due to corridor-definition mismatch
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Prior issue:** [148](148_real_movingai_maps.md) (real MovingAI maps — confirmed genuine ~4× algorithmic gap)
**Benchmark:** [440 GOAT](../.benchmarks/440_lllg_paper_repro_goat.md)

## Context

Issue 148 closed the map-fidelity hypothesis: the real MovingAI ht_chantry map
gives ratio 0.27 — 3× better than the synthetic approx (0.09), but still below
the 0.30 PASS threshold. A genuine ~4× algorithmic gap remains (4.51 vs paper
~17).

Issue 147 had already implemented `CounterFlowHindrance` (a dynamic, agent-aware
penalty for anti-aligned goal directions), but it had **zero effect** because
it's the 3rd PIBT tiebreak (only fires when guidance_mismatch AND goal_dist are
tied).

This issue implemented **full Guided-PIBT flow direction assignment**: a static,
topology-aware flow field that assigns one-way directions to corridor cells,
inserted as a new PIBT cost term between `guidance_mismatch` and `goal_dist`.

## What was done

1. **New module: `flow.rs`** — `FlowField<P>` trait + `NoFlow` default +
   `GridFlowField` impl. Corridor detection: a cell is a corridor cell if it
   has exactly 2 passable neighbors on opposite sides (1-wide passage).

2. **Modified `pibt.rs`** — added `flow_mismatch: u8` field to `Candidate`.
   New cost tuple: `(guidance_mismatch, flow_mismatch, goal_dist, hindrance, ε)`.
   The flow_mismatch term is computed via `flow_field.mismatch(from, next)` for
   each candidate move.

3. **Modified `mod.rs`** — `LifelongLaCam` gains a `flow_field:
   Option<Box<dyn FlowField<P> + Send + Sync>>` field and a `with_flow_field`
   builder method. When `None`, uses `NoFlow` (paper-faithful).

4. **11 unit tests** — NoFlow correctness, open-map (no corridors), horizontal
   corridor detection, vertical corridor detection, flow mismatch (aligned/
   against/wait), junction/dead-end/cornexclusion, orchestrator integration,
   determinism.

5. **Benchmark updated** — `run_simulation` computes `GridFlowField::from_map`
   and passes it via `.with_flow_field()`. Map topology summary now prints
   corridor counts.

## Result: mechanism correct, but near-zero effect on real maps

| Map | Issue 148 (no flow) | Issue 149 (with flow) | Corridors | Change |
|---|---|---|---|---|
| empty-48-48 | 18.52 (0.68) | 18.52 (0.68) | **0** | identical ✅ |
| random-64-64-10 | 14.37 (0.68) | 14.65 (0.69) | **63** | +2% (noise) |
| warehouse | 7.34 (0.41) | 7.34 (0.41) | **0** | identical |
| ht_chantry | 4.51 (0.27) | 4.61 (0.27) | **8** | +2% (noise) |

**The flow direction assignment mechanism is correctly implemented and tested,
but it has near-zero effect on the real benchmark maps.** Root cause:

### The corridor-definition mismatch

The flow field detects **strict 1-wide corridors** — cells with exactly 2
passable neighbors on opposite sides (a 1-wide passage between two walls). This
is the right definition for classic 1-wide grid mazes.

**But the real MovingAI maps don't have 1-wide corridors:**
- `ht_chantry` (Dragon Age: Origins): only **8 corridor cells** out of 7461
  passable (0.1%). Game-map corridors are typically 2-wide or wider.
- `warehouse`: **0 corridor cells**. The warehouse layout uses wide aisles
  between shelf blocks, not narrow passages.
- `empty/random`: 0 and 63 corridors respectively (expected).

The mechanism works correctly on synthetic 1-wide corridors (11 unit tests
prove it), but real game maps need a broader corridor definition — likely
**2-wide passage detection** or **passage-width-aware flow assignment**.

### No regression on open maps

The flow field has zero corridors on open maps, so `flow_mismatch` is always 0,
and the cost tuple degenerates to the paper-faithful ordering. This confirms
the "safe promotion" design: empty-48-48 throughput is identical (18.52 → 18.52).

## GOAT gate status (unchanged from Issue 148)

| Gate | Status | Detail |
|---|---|---|
| **G1** | **PARTIAL 2/4** | empty 0.68 ✅, random 0.69 ✅, warehouse 0.41 ❌, ht_chantry 0.27 ❌. Unchanged. |
| **G2** | **FAIL** | Warm-start non-consumable (ratio 1.00). Unchanged. |
| **G3** | **PASS** | 1601 tests pass (11 new flow field tests). Clippy clean. |
| **G4** | **PASS** | 467ms median at 1000 agents (<500ms target). |

**Promotion decision: KEEP OPT-IN** (unchanged).

## What was accomplished

- `FlowField<P>` trait + `GridFlowField` impl shipped — a new pluggable seam
  for Guided-PIBT direction assignment. Correct, tested, modelless.
- `flow_mismatch` cost term inserted in the PIBT candidate tuple at position 2
  (between guidance_mismatch and goal_dist).
- `with_flow_field` builder on `LifelongLaCam` — consumers with 1-wide corridor
  maps can opt in.
- The "safe promotion" design verified: zero regression on open maps.

## What remains (honest next steps)

1. **Broaden corridor detection to 2-wide passages** — the real ht_chantry
   corridors are 2-wide, not 1-wide. A 2-wide passage detector would identify
   pairs of adjacent cells where both have a wall on the same side. This is
   more complex but necessary for real game maps.

2. **Warehouse still needs a different fix** — 0 corridors means the flow field
   can't help warehouse at all. The warehouse gap is task-completion rate on
   shelf-aisle structure, not corridor deadlock.

3. **riir-ai/489 fusion** — still pragmatic for open/random topologies (2/4
   maps pass). The substrate works correctly via the four pluggable seams.

4. **G2 warm-start** — still FAIL, unchanged.

## Files changed

| File | Change |
|---|---|
| `crates/katgpt-core/src/multi_agent_path/flow.rs` | New: FlowField trait + NoFlow + GridFlowField (+263 lines) |
| `crates/katgpt-core/src/multi_agent_path/pibt.rs` | flow_mismatch field + FlowField param (+49 lines) |
| `crates/katgpt-core/src/multi_agent_path/mod.rs` | with_flow_field builder + FlowFieldBox (+43 lines) |
| `crates/katgpt-core/src/multi_agent_path/tests.rs` | 11 new flow field unit tests (+189 lines) |
| `crates/katgpt-core/benches/bench_440_lllg_paper_repro.rs` | Flow field integration + corridor count in summary |
| `.benchmarks/440_lllg_paper_repro_goat.md` | Issue 149 section |
| `.issues/.highwater` | 148 → 149 |

## Acceptance criteria

- [x] Implement `FlowField<P>` trait + `NoFlow` default + `GridFlowField` impl
- [x] Add `flow_mismatch: u8` field to `Candidate` in `pibt.rs`
- [x] Update `lexicographic_cmp` to include flow_mismatch at position 2
- [x] Thread `FlowField` through `pibt_step` → `greedy_pibt_pass`
- [x] Add `with_flow_field` builder to `LifelongLaCam` orchestrator
- [x] Update benchmark to compute `GridFlowField` from each map
- [x] Unit tests for corridor detection + direction assignment + mismatch (11 tests)
- [x] Run G1 gate — verify no regression on empty/random; measure ht_chantry
- [x] Document results honestly in `.benchmarks/440_lllg_paper_repro_goat.md`
- [x] Update `.issues/.highwater` 148 → 149
