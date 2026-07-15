# Issue 150 — 2-Wide Corridor Detection for Guided-PIBT Flow Field

**Status:** RESOLVED (2026-07-15) — 2-wide detection works correctly (4920 corridors on warehouse vs 0 before), but throughput unchanged because the flow_mismatch tiebreak position is too weak to enforce directional flow
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Prior issue:** [149](149_guided_pibt_flow_direction_assignment.md) (1-wide corridor detection — near-zero effect on real maps)
**Benchmark:** [440 GOAT](../.benchmarks/440_lllg_paper_repro_goat.md)

## Context

Issue 149 shipped the `FlowField<P>` pluggable seam + `GridFlowField` impl with
1-wide corridor detection. The mechanism is correct (11 unit tests prove it), but
it has **near-zero effect on real benchmark maps**:

| Map | Corridor cells (1-wide) | % of passable |
|---|---|---|
| empty-48-48 | 0 | 0.0% |
| random-64-64-10 | 63 | 1.7% |
| warehouse | 0 | 0.0% |
| ht_chantry | **8** | **0.1%** |

**Root cause:** Real game-map corridors (Dragon Age: Origins `ht_chantry`) are
**2-wide or wider**, not 1-wide. The strict "exactly 2 opposite passable
neighbors" detector misses 2-wide passages where each cell has 3 passable
neighbors (left, right, and the partner cell on the other row/column of the pair).

## What this issue does

Broaden `GridFlowField::from_map` to detect **both 1-wide and 2-wide corridors**.

### 2-wide corridor definition

A pair of adjacent cells forms a 2-wide passage when they're flanked by walls on
both sides perpendicular to the adjacency axis:

- **2-wide horizontal corridor:** cells `(x, y)` and `(x, y+1)` are both passable,
  with walls at `(x, y-1)` and `(x, y+2)`. The corridor runs left-right (flow axis
  = Horizontal).
- **2-wide vertical corridor:** cells `(x, y)` and `(x+1, y)` are both passable,
  with walls at `(x-1, y)` and `(x+2, y)`. The corridor runs up-down (flow axis
  = Vertical).

Both cells in a pair get the same `FlowDirection` (sign=+1).

### Overlap handling

- A 1-wide corridor cell (exactly 2 opposite passable neighbors) is detected
  first and takes priority.
- A cell classified as BOTH a 2-wide horizontal and 2-wide vertical corridor cell
  is a junction — left unclassified (`None`).
- 3+ wide passages are NOT detected (they're wide enough for free passing).

## Acceptance criteria

- [x] Add `width: u8` field to `FlowDirection` (1 or 2)
- [x] Extend `GridFlowField::from_map` with 2-wide detection pass
- [x] Add `corridor_1wide_count()` and `corridor_2wide_count()` diagnostic methods
- [x] Unit tests for 2-wide horizontal/vertical corridor detection
- [x] Unit tests for 2-wide flow mismatch (aligned/against/wait)
- [x] Unit test: 3-wide passage NOT detected as 2-wide
- [x] Unit test: junction cell (both H and V 2-wide) NOT classified
- [x] Unit test: no regression on open maps (0 corridors)
- [x] Update benchmark to print 1-wide vs 2-wide corridor counts
- [x] Run G1 gate — measure corridor counts and throughput on all 4 maps
- [x] Document results honestly
- [x] Update `.issues/.highwater` 149 → 150

## Result

### Corridor detection: dramatically improved

The 2-wide detector finds **significantly more corridors** on real game maps:

| Map | 1-wide (Issue 149) | 2-wide (Issue 150) | Total | Coverage |
|---|---|---|---|---|
| empty-48-48 | 0 | 0 | 0 | 0.0% |
| random-64-64-10 | 63 | 182 | 245 | 6.6% |
| warehouse | 0 | **4920** | **4920** | **50.3%** |
| ht_chantry | 8 | 102 | 110 | 1.5% |

**The 2-wide detection successfully identifies the corridor topology of real
maps.** Warehouse went from 0 to 4920 corridor cells — 50% of passable cells.
This confirms that the warehouse aisles are exactly 2-wide, as suspected.

### Throughput: unchanged (the real finding)

Despite the massive increase in corridor coverage, throughput is **unchanged**
(within noise) on all 4 maps:

| Map | Issue 149 | Issue 150 | Change |
|---|---|---|---|
| empty-48-48 | 18.52 (0.68) | **18.52 (0.68)** | identical |
| random-64-64-10 | 14.65 (0.69) | **14.56 (0.69)** | -0.6% (noise) |
| warehouse | 7.34 (0.41) | **7.33 (0.41)** | -0.1% (noise) |
| ht_chantry | 4.61 (0.27) | **4.66 (0.27)** | +1% (noise) |

### Root cause: the tiebreak position is too weak

The `flow_mismatch` cost term sits at position 2 in the 5-tuple:

```text
⟨ guidance_mismatch, flow_mismatch, goal_dist, hindrance, ε ⟩
```

It only breaks ties between candidates with the same `guidance_mismatch`. In
practice, the guidance source (space-time A*) steers agents well enough that
`guidance_mismatch` is almost always 0 for the preferred move, and the
collision-checking loop picks the first collision-free candidate regardless of
flow. The flow direction is a **soft hint**, not a hard constraint.

**The bottleneck is not corridor detection — it's the enforcement mechanism.**
To actually improve throughput, the flow direction needs to influence the
**guidance source** (space-time A* should prefer routes aligned with the flow
field), not just the PIBT tiebreak.

## GOAT gate status (unchanged)

| Gate | Status | Detail |
|---|---|---|
| **G1** | **PARTIAL 2/4** | empty 0.68 ✅, random 0.69 ✅, warehouse 0.41 ❌, ht_chantry 0.27 ❌ |
| **G2** | **FAIL** | Warm-start non-consumable (ratio 1.00) |
| **G3** | **PASS** | 1611 tests (7 new 2-wide tests). Clippy clean. |
| **G4** | **PASS** | 225ms median at 1000 agents |

**Promotion decision: KEEP OPT-IN** (unchanged).

## What was accomplished

- 2-wide corridor detection correctly identifies real game-map topology.
- `width: u8` field on `FlowDirection` enables width-aware diagnostics.
- 7 new unit tests covering all 2-wide detection edge cases.
- Zero regression on open maps (the "safe promotion" design holds).
- Honest negative result documented: corridor detection was the wrong hypothesis
  for the throughput gap.

## What remains (honest next steps)

1. **Flow-aware guidance** — make the space-time A* guidance source prefer routes
   aligned with the flow field. This is where the flow direction actually matters:
   it should influence path planning, not just the PIBT tiebreak.
2. **Warehouse-specific fix** — 4920 corridors (50% coverage) but no effect means
   the warehouse gap is NOT corridor deadlock. It's task-completion rate on
   shelf-aisle structure. Needs intersection reservation or shelf-aware goal
   assignment.
3. **riir-ai/489 fusion** — still pragmatic for open/random topologies (2/4 maps
   pass).
4. **G2 warm-start** — still FAIL.
