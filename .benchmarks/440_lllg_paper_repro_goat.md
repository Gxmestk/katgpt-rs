# Benchmark 440: LLLG Paper Reproduction GOAT Gate

**Date:** 2026-07-15 (updated Issue 143)
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Research:** [424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md)
**Paper:** [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, AAAI 2026
**Issue:** [143](../.issues/143_lacam_escalation_full_pibt.md) — LaCAM escalation (greedy PIBT + priority shuffle retry)
**Prior issue:** [142](../.issues/142_full_space_time_astar_guidance_upgrade.md) — full space-time A* upgrade
**Prior issue:** [140](../.issues/140_pibt_priority_inheritance_and_warmstart_integration.md) — PIBT PI + warm-start investigation

---

## TL;DR

**G3 (no-regression): PASS. G4 (latency): PASS. G1 (throughput): PARTIAL
(3/4 maps). G2 (congestion): FAIL (warm-start not consumable — confirmed
by Issue 142 even with full A*).**

Issue 143 added LaCAM escalation to the greedy PIBT: when ≥ 20 agents are
stuck (systemic congestion), retry with shuffled priority orderings (up to 2
retries). Warehouse throughput improved +8.3% (ratio 0.39 → 0.42). The
escalation has minimal overhead on open maps (threshold prevents unnecessary
retries). Recursive PIBT (full priority inheritance with eviction) was also
tested but **REJECTED — collapses throughput -92%** (empty-48-48: 18.6 → 1.5).

| Map | A* (Issue 142) | **+ LaCAM retry (Issue 143)** | Change |
|---|---|---|---|
| empty-48-48 | 18.63 (ratio 0.68) | **18.52 (ratio 0.68)** | -0.6% (noise) |
| random-64-64-10 | 14.57 (ratio 0.69) | **14.57 (ratio 0.69)** | no change |
| warehouse | 6.99 (ratio 0.39) | **7.57 (ratio 0.42)** | **+8.3%** |
| ht_chantry | 0.15 (ratio 0.01) | **0.15 (ratio 0.01)** | no change |

G1 still passes on **3/4 maps** (unchanged from Issue 142). ht_chantry
remains at 0.01 — the maze topology requires **global routing**
(Guided-PIBT), not local priority-shuffle retry. The paper's own caveat #1
documents this as LLLG's known limitation.

Latency at 1000 agents: 234ms median (was 88ms pre-retry, now 234ms with
retry overhead). Still under the 500ms target. The retry triggers at high
density (1000 agents on empty-48-48 = 43% density → frequent systemic
stalls). On the G1 benchmark (800 agents), median latency is 60-70ms on
open maps.

**Promotion decision: KEEP OPT-IN.** Unchanged from Issue 142. The substrate
passes G3/G4 and partially passes G1 (3/4 maps), but the G1 ht_chantry
failure and G2 warm-start non-consumption prevent full GOAT validation.

---

## Issue 143 results (2026-07-15) — LaCAM escalation

### What changed

1. **LaCAM escalation** added to `pibt.rs`: when the greedy PIBT produces
   ≥ 20 stuck agents (systemic congestion), retry with shuffled priority
   orderings (up to 2 retries). The stuck agents are elevated to high
   priority, and non-stuck agents are randomly perturbed. The result with
   the fewest stuck agents is returned.

2. **Recursive PIBT tested and REJECTED.** The full recursive priority
   inheritance (agent i evicts undecided agent j from its cell, recursively)
   was implemented and benchmarked. Result: **throughput collapses -92%**
   (empty-48-48: 18.6 → 1.5). The eviction forces agents to move away from
   their goals, creating cascading stalls. The greedy PIBT — which lets
   agents compromise by taking their next-best cell — has dramatically
   higher collective throughput in the lifelong MAPF setting. The recursive
   variant is right for one-shot MAPF (finding ANY solution), wrong for
   lifelong MAPF (sustained throughput).

3. **Stuck-agent threshold** (MIN_STUCK_FOR_RETRY = 20): prevents retry
   overhead on open maps where occasional stuck agents (1-5) resolve
   naturally next tick. Retries only fire on genuinely congested maps.

### Latency analysis

The LaCAM retry adds overhead on dense maps:
- **800 agents, empty/random**: median 60-70ms (retries rarely trigger — too
  few stuck agents). Comparable to pre-retry baseline (~88ms).
- **800 agents, warehouse**: median 134ms (retries trigger — warehouse has
  systemic shelf-aisle congestion). The +8.3% throughput gain justifies the
  cost.
- **1000 agents, empty-48-48 (G4)**: median 234ms (retries trigger at 43%
  density). Under the 500ms target but above the 100ms stretch goal.

### ht_chantry — why local retry doesn't help

The maze topology creates head-on corridor conflicts: two agents meet in a
1-wide passage, both want to pass. Neither can move forward (vertex
conflict), neither can wait (the other is blocking), and backing up requires
one agent to reverse direction — which the greedy PIBT doesn't consider (it
prefers goal-directed moves). The priority shuffle doesn't help because the
issue isn't WHO goes first, it's that SOMEONE must back up, and no priority
ordering makes that happen with local decisions.

The fix is **global routing** (Guided-PIBT from the paper): pre-compute
flow directions for corridors and route agents accordingly. This is a
significantly larger implementation and is the paper's own recommended
approach for long-corridor maps (caveat #1).

---

## Issue 142 results (2026-07-15) — full space-time A*

Issue 142 replaced the greedy rollout with a proper priority-queue space-time
A* and fixed the broken multi-round refinement. This is the upgrade that Issue
140 identified as the real blocker.

### What changed

1. **`astar_for_agent` rewritten** from greedy rollout to proper A* over
   `(position, depth)` state space with BFS-distance heuristic. Priority queue
   (BinaryHeap), g/h/f scores, came_from path reconstruction. The A* has w-step
   lookahead with proper cost accumulation — it can plan multi-step detours
   around collisions.

2. **Multi-round refinement fixed** (unrecord/re-record). Previously each
   round called `clear_occupancy()`, making rounds no-ops (agent 0 always saw
   an empty map). Now each agent unrecords its previous path before recomputing,
   so round 1 agent 0 sees round-0 paths of agents 1..n-1.

3. **Dead code removed**: `step_cost_bfs` and `cycle_penalty` free functions
   (greedy rollout helpers) deleted. `collision_count` `#[allow(dead_code)]`
   removed (now used by the A*).

### Warm-start consumption — TRIED AND CONFIRMED HARMFUL

Occupancy-seeding with warm-start forecasts was implemented and benchmarked:

| Config | empty-48-48 throughput |
|---|---|
| A* without warm-start seeding | **18.60** |
| A* with warm-start seeding | 14.73 |

Seeding HURTS by 21%. The forecast is invalidated when PIBT deviates from the
guidance (common on dense maps), creating misleading phantom collision
constraints that the A* routes around. This confirms Issue 140's finding but
with the full A* — the problem isn't the greedy rollout, it's that warm-start
forecasts are too stale on dense maps without LaCAM escalation.

**Decision:** warm-start data is consumed (taken/cleared) but NOT seeded into
the occupancy. LllgPi = LllgEmpty with the current implementation. Positive
warm-start consumption likely requires LaCAM escalation (Phase 5) to keep
forecasts accurate.

---

## Issue 140 investigation results (2026-07-15)

The Issue 140 analysis identified two blocking items for G1/G2 full pass:
1. PIBT priority inheritance upgrade
2. Warm-start integration

Both were implemented, benchmarked, and found to be **blocked by a deeper
architectural gap**: the greedy guidance rollout.

### PIBT priority inheritance — implemented, benchmarked, REVERTED

The full recursive PIBT with priority inheritance was implemented (~200 lines
of recursive backtracking in `pibt.rs`):

- `pibt_recursive` function: when agent `i` wants cell `u` occupied by
  undecided agent `j`, recursively push `j` to move before committing.
- `in_chain: HashSet<usize>` for cycle prevention.
- `MAX_PIBT_DEPTH = 64` cap for pathological recursion.
- `blocked_cell: Option<&P>` swap prevention (pushed agent can't take
  pusher's cell).

**Benchmark result:** Throughput COLLAPSED on all maps:
- empty-48-48: 17.32 → 0.47 (ratio 0.63 → 0.02)
- random-64-64-10: 13.15 → 0.36
- All maps: mean_stops 170-300 out of 300 steps (agents barely moving)

**Root cause:** The recursive push is too conservative for dense maps
without LaCAM-level search escalation. In the paper's design, PIBT is
combined with LaCAM — when PIBT fails to resolve a conflict, LaCAM
escalates to a full search. Without LaCAM, the recursive push requires
occupants to vacate before committing, causing cascading stalls. The greedy
PIBT (take first collision-free candidate, let later agents adapt) has
higher throughput in the lifelong MAPF setting without LaCAM.

**Decision:** Reverted to the greedy PIBT (Phase 2 code). The recursive PIBT
is deferred until LaCAM is added (Phase 5). Documented in `pibt.rs` module
docs.

### Warm-start integration — infrastructure landed, consumption DEFERRED

The warm-start integration was plumbed end-to-end:

- `set_warm_start(Vec<Vec<P>>)` method added to `LocalGuidanceSource` trait
  (default no-op).
- `SpaceTimeGuidance` stores the data in a new `warm_start: Option<Vec<Vec<P>>>`
  field.
- `LifelongLaCam::tick` calls `guidance.set_warm_start(warm)` before
  `compute_guidance`.
- Solution recording fixed: prepends the executed PIBT action so suffix
  extraction correctly skips the executed step (T2.6 complete).

**Consumption attempt 1 (occupancy seeding, all paths):**
Throughput COLLAPSED: empty-48-48 17.32 → 0.47. The warm-start forecast
seeded ALL agents' paths into the occupancy map, causing agents to avoid
forecast cells (including their own forecast — self-collision). G2 "passed"
(ratio 0.48) but G1 was destroyed.

**Consumption attempt 2 (occupancy seeding, self-collision removal):**
Throughput still collapsed. Removing each agent's own forecast before
processing helped slightly but the forecast from OTHER agents still created
too many collision constraints for the greedy rollout.

**Consumption attempt 3 (weak bias, -0.5 per matching step):**
No effect on grid maps. BFS distances are integers (1.0 apart); the 0.5
bonus is too weak to break ties that don't exist. LllgPi = LllgEmpty.

**Root cause:** The paper's warm-start is designed for **full space-time A***,
where it provides an initial bound for A* pruning. The greedy rollout doesn't
benefit from warm-start — it always picks the locally-best step, ignoring the
forecast. Occupancy-seeding makes the greedy rollout MORE conservative (avoiding
forecast cells), which hurts throughput. Weak bias is too weak to matter on
integer-distance grids.

**Decision:** Warm-start infrastructure is in place (stored, consumed one-shot)
but NOT consumed by the greedy rollout. The `compute_guidance` method clears
the data (preventing stale leaks) but doesn't seed the occupancy. Consumption
is deferred to the full A* upgrade. LllgPi and LllgEmpty produce identical
results with the greedy rollout.

---

## G1 — Throughput (correctness)

**800 agents, 300 steps, 4 maps.** (Updated Issue 142 — full space-time A*)

| Map | Our throughput | Paper target | Ratio | Verdict |
|---|---|---|---|---|
| empty-48-48 | 18.63 | 27.3 | 0.68 | **PASS** |
| random-64-64-10 | 14.57 | 21.1 | 0.69 | **PASS** |
| warehouse-10-20-10-2-2 | 6.99 | 18.0 | 0.39 | **PASS** |
| ht_chantry-approx | 0.15 | 17.0 | 0.01 | **FAIL** |

**Pass criterion:** ratio ≥ 0.30 (within reasonable range, system works).

**Improvement vs Issue 140 (greedy rollout):**
- empty: 0.63 → 0.68 (+7.6%)
- random: 0.62 → 0.69 (+10.8%)
- warehouse: 0.35 → 0.39 (+11.0%) — **crossed the 0.30 threshold**
- ht_chantry: 0.01 → 0.01 (marginal)

**Analysis:**
- 3/4 maps now PASS (was 2/4). The A* with BFS-distance heuristic and
  multi-round refinement improves throughput across the board.
- ht_chantry FAILS because the maze topology with narrow bottlenecks requires
  LaCAM-level search escalation. The w_Φ=5 window can't see far enough through
  the maze to plan detours. This is the paper's known limitation (caveat #1:
  long one-cell-wide corridors).

**Root cause of ht_chantry failure:** The maze structure creates long
one-cell-wide corridors where agents meet head-on. Without LaCAM escalation
(which can reorder agents globally), the greedy PIBT can't resolve these
conflicts. Priority inheritance alone doesn't help (Issue 140 showed it
*collapses* throughput without LaCAM). This is Phase 5 work.

---

## G2 — Congestion mitigation

**empty-48-48, 1000 agents, 100 steps. LLLG_Π vs LllgEmpty baseline.**
(Updated Issue 142)

| Scheme | max_stops/cell | mean_stops | throughput |
|---|---|---|---|
| LLLG_Π | 56 | 6.1 | 18.60 |
| LllgEmpty | 56 | 6.1 | 18.60 |

**Ratio: 1.00 (identical). FAIL.**

**Root cause:** Issue 142 confirmed (with the full A*) that occupancy-seeding
with warm-start forecasts HURTS throughput. The forecast is invalidated when
PIBT deviates from the guidance. The data is consumed (cleared) but NOT seeded
into the occupancy, so LllgPi = LllgEmpty.

**Fix required:** LaCAM escalation (Phase 5) to keep PIBT deviations rare, so
warm-start forecasts stay accurate. Alternatively, a different warm-start
consumption method (e.g. soft bias, f-bound pruning) might work but was not
found effective on integer-distance grids (Issue 140 attempt 3).

---

## G3 — No-regression

```
cargo clippy -p katgpt-core --all-features --lib: clean
cargo test -p katgpt-core --lib: 1556/1556 pass
cargo test -p katgpt-core --features multi_agent_path --lib: 22/22 pass
```

**PASS.** Existing tests unaffected.

---

## G4 — Latency

**empty-48-48, 1000 agents, 100 steps.** (Updated Issue 142)

| Metric | Value |
|---|---|
| Median per-tick | 87.83 ms |
| Max per-tick | 285.36 ms |
| Paper (M1 Ultra) | 210–260 ms |
| Issue 140 (greedy) | 82.17 ms |

**PASS (target < 500ms). Stretch < 100ms: PASS.**

The A* is ~7% slower than the greedy rollout (87.83ms vs 82.17ms) due to the
priority queue and hash map overhead, but still well within the stretch goal
and 2.4× faster than the paper's M1 Ultra result. The A* explores a narrow
cone around the BFS gradient thanks to the admissible heuristic.

---

## Honest caveats

1. **Maps are synthetic approximations.** We don't have the exact MovingAI
   MAPF benchmark map files for warehouse-10-20-10-2-2 and ht_chantry.

2. **PIBT priority inheritance requires LaCAM.** The recursive push is too
   conservative without LaCAM-level escalation. See Issue 140 investigation.

3. **Warm-start requires full A*.** The greedy rollout can't consume the
   warm-start forecast. Occupancy-seeding collapses throughput; weak bias is
   too weak on integer-distance grids.

4. **Agent count is 800 not 1000.** Paper throughput targets are at 1000
   agents. We run at 800.

5. **The ht_chantry approximation is more extreme than the real map.**

---

## What was accomplished (Issue 142)

1. **Full space-time A* landed.** Replaced the greedy rollout with a proper
   priority-queue A* over `(position, depth)` state space with BFS-distance
   heuristic. The A* has w-step lookahead and can plan multi-step detours.

2. **Multi-round refinement fixed.** The broken `clear_occupancy()`-each-round
   pattern (making rounds no-ops) replaced with unrecord/re-record: each agent
   removes its previous path before recomputing, so round 1+ actually improves
   on round 0.

3. **Throughput improved on all 4 maps.** 3/4 maps now pass G1 (was 2/4).
   Warehouse crossed the 0.30 threshold. ht_chantry remains at 0.01 (needs
   LaCAM).

4. **Warm-start consumption confirmed harmful.** Occupancy-seeding with
   warm-start forecasts was implemented, benchmarked, and found to HURT
   throughput (18.60 → 14.73 on empty-48-48). The forecast is invalidated when
   PIBT deviates from the guidance. Confirmed with the full A* that this is
   not a greedy-rollout-specific problem — it's a forecast-accuracy problem
   that likely needs LaCAM to fix.

---

## Next steps

1. **LaCAM escalation** (Phase 5) — when PIBT fails to resolve a conflict,
   escalate to LaCAM-level search. This is the single blocker for both ht_chantry
   (G1) and warm-start consumption (G2). With LaCAM:
   - ht_chantry: LaCAM resolves corridor conflicts that PIBT can't.
   - Warm-start: LaCAM keeps PIBT deviations rare, so forecasts stay accurate.
   - PIBT priority inheritance: recursive push works with LaCAM fallback.

2. **Download real MovingAI maps** — for exact paper reproduction. The
   synthetic approximations may differ from the real warehouse/ht_chantry
   topology.

3. **Phase 3** (Fusion hooks documentation) — document the four pluggable seams
   (`CostFn`, `LocalGuidanceSource`, `WarmStartScheme`, `HindranceEstimator`)
   with stub examples.

4. **riir-ai/489** (private runtime fusion) — can now consume `multi_agent_path`
   via the four seams. Works correctly on open/moderate maps (3/4 G1 maps pass);
   the consumer should be aware of ht_chantry/maze limitations.

The substrate is functional and well-performing on the 3/4 maps it handles.
The ht_chantry failure and G2 warm-start non-consumption both point to the same
fix: LaCAM escalation (Phase 5).

---

## Promotion Decision (Phase 5, T5.1/T5.2)

**Decision: KEEP OPT-IN.** Reaffirms the T2.6 decision recorded in the TL;DR above.

### T5.1 Considerations (all four weighed)

1. **Modelless?** ✅ Yes — the substrate is entirely heuristic (PIBT greedy
   selection + BFS distance field + warm-start suffix reuse + blocking-count
   hindrance). No training, no backprop, no gradient descent. **Promotion is
   *allowed* by AGENTS.md's modelless mandate** — the rule permits default-on
   for modelless gains.

2. **Heavy / leaf-clean?** ❌ Multi-agent pathfinding is **not** a leaf-clean
   primitive. The substrate is ~1000 LOC across 8 files (mod, config, pibt,
   local_guidance, warm_start, hindrance, position, tests) plus the bench
   harness. Consumers that don't need crowd pathfinding would pay the compile
   cost. Keeping it opt-in avoids bloating the default build — mirrors the
   `cgsp` (Plan 274) and `induced_cwm` (Plan 296) precedent for heavier
   substrates.

3. **GOAT gate status?** ❌ **Not fully passed.** G3 (no-regression) and G4
   (latency) PASS, but G1 (throughput) is only PARTIAL (2/4 maps) and G2
   (congestion) FAILS. The modelless mandate's promotion rule requires a
   *modelless gain* — a perf gain on a biased/incorrect result is explicitly
   NOT a modelless gain (AGENTS.md Feature Flag Discipline). G1's warehouse/maze
   failures mean the substrate produces measurably wrong answers on those map
   classes; promoting a 2/4-correct primitive to default-on would violate the
   quality-gate rule even though the primitive itself is modelless.

4. **Super-GOAT claim validated?** ❌ **No.** The Super-GOAT selling point
   rests on the riir-ai/489 fusion gates G5–G7 (HLA projection per-NPC
   personality modulation, Crowd MCGS physical layer, P350 multi-agent
   closure). Those gates have **not run yet**. Promoting the substrate to
   default before the fusion is validated is premature — if G5–G7 fail, the
   substrate stays a standalone pathfinder with no Super-GOAT upside, and the
   default-on promotion would have been wasted build cost.

### Rationale (why opt-in is correct now)

The substrate is shipped, documented (Phase 3 fusion hooks with compile-checked
rustdoc examples), and available to any consumer that wants it via the
`multi_agent_path` feature flag. Promotion to default-on is deferred until
**both** of these hold:

- **G1/G2 unblocked** — the Phase 5 full space-time A* upgrade (replacing the
  greedy rollout) is the single change that unblocks both gates: A* benefits
  from warm-start (G2) and produces better paths through warehouse/maze
  corridors (G1). The latency budget has ample headroom (82ms current vs 500ms
  target).
- **Super-GOAT validated** — riir-ai/489 G5–G7 pass, confirming the HLA ×
  Crowd MCGS × P350 fusion produces emergent crowd coordination beyond what
  either primitive alone achieves.

Until then, the substrate is opt-in and the Super-GOAT claim is conditional.
