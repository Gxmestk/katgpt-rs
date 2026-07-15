# Benchmark 440: LLLG Paper Reproduction GOAT Gate

**Date:** 2026-07-15
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Research:** [424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md)
**Paper:** [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, AAAI 2026
**Issue:** [140](../.issues/140_pibt_priority_inheritance_and_warmstart_integration.md) — PIBT PI + warm-start investigation

---

## TL;DR

**G3 (no-regression): PASS. G4 (latency): PASS. G1 (throughput): PARTIAL (2/4 maps).
G2 (congestion): FAIL (warm-start not consumable by greedy rollout).**

The substrate works correctly on open and moderately obstructed maps
(empty-48-48 ratio 0.63, random-64-64-10 ratio 0.62). It fails on
warehouse (ratio 0.35) and maze-like maps (ht_chantry ratio 0.01) because
the greedy PIBT lacks priority inheritance — agents jam in narrow
corridors and cannot push each other through. This is consistent with the
paper's known limitation (caveat #1: long one-cell-wide corridors).

The latency gate G4 is excellent: 80ms median at 1000 agents, 2.6× faster
than the paper's M1 Ultra result (210-260ms). The simpler algorithm (greedy
vs full A*) trades throughput for speed.

**Promotion decision: KEEP OPT-IN.** The substrate passes G3/G4 and
partially passes G1, but the G1 warehouse/maze failures and G2 warm-start
non-consumption prevent full GOAT validation.

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

**800 agents, 300 steps, 4 maps.**

| Map | Our throughput | Paper target | Ratio | Verdict |
|---|---|---|---|---|
| empty-48-48 | 17.32 | 27.3 | 0.63 | **PASS** |
| random-64-64-10 | 13.15 | 21.1 | 0.62 | **PASS** |
| warehouse-10-20-10-2-2 | 6.30 | 18.0 | 0.35 | **FAIL** |
| ht_chantry-approx | 0.11 | 17.0 | 0.01 | **FAIL** |

**Pass criterion:** ratio ≥ 0.30 (within reasonable range, system works).

**Analysis:**
- Open/moderate maps PASS. The BFS distance field guidance navigates agents
  around obstacles correctly.
- Warehouse FAILS because the shelf-aisle structure creates long narrow
  corridors where greedy PIBT (without priority inheritance) causes permanent
  jams.
- ht_chantry FAILS because the maze structure with bottlenecks is
  fundamentally hard for greedy guidance.

**Root cause:** G1 warehouse/maze failures require BOTH priority inheritance
PIBT AND LaCAM escalation to resolve. The priority inheritance alone (without
LaCAM) is too conservative and collapses throughput (see Issue 140
investigation above).

---

## G2 — Congestion mitigation

**empty-48-48, 1000 agents, 100 steps. LLLG_Π vs LllgEmpty baseline.**

| Scheme | max_stops/cell | mean_stops | throughput |
|---|---|---|---|
| LLLG_Π | 52 | 7.2 | 16.73 |
| LllgEmpty | 52 | 7.2 | 16.73 |

**Ratio: 1.00 (identical). FAIL.**

**Root cause:** The warm-start data is plumbed through (stored in
`SpaceTimeGuidance`, cleared one-shot per tick) but NOT consumed by the greedy
rollout. The greedy rollout doesn't benefit from warm-start — it always picks
the locally-best step, ignoring the forecast. Occupancy-seeding collapses
throughput (see Issue 140 investigation).

**Fix required:** Full space-time A* guidance, where warm-start provides the
initial bound for A* pruning. This is the Phase 5 upgrade.

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

**empty-48-48, 1000 agents, 100 steps.**

| Metric | Value |
|---|---|
| Median per-tick | 82.17 ms |
| Max per-tick | 293.30 ms |
| Paper (M1 Ultra) | 210–260 ms |

**PASS (target < 500ms). Stretch < 100ms: PASS.**

Our impl is 2.6× faster than the paper's M1 Ultra result because our
algorithm is simpler (greedy guidance vs full space-time A*; greedy PIBT vs
priority inheritance). The perf headroom exists to upgrade to a proper A*
within the latency budget.

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

## What was accomplished (Issue 140)

Despite the G1/G2 gates not improving, the Issue 140 investigation produced
valuable results:

1. **Warm-start infrastructure landed.** `set_warm_start` trait method,
   `SpaceTimeGuidance` stores the data, `tick()` threads it through. Ready
   for consumption when full A* lands. No more `let _ = warm_start()` discard.

2. **Solution recording fixed (T2.6).** `tick()` now prepends the executed
   PIBT action to the guidance path before storing in `WarmStartCache`. The
   suffix extraction correctly skips the executed step (not the guidance's
   preferred step, which may differ).

3. **PIBT priority inheritance fully evaluated.** Implemented, benchmarked,
   and honestly documented why it needs LaCAM. The recursive PIBT code is
   preserved in git history for future reference.

4. **The real blocker identified.** Both G1 warehouse/maze and G2 congestion
   require **full space-time A* guidance** (not greedy rollout). The greedy
   rollout is the bottleneck — it can't benefit from priority inheritance
   (needs LaCAM) or warm-start (needs A* pruning). The honest path forward is
   the A* upgrade, not more PIBT/warm-start plumbing.

---

## Next steps

1. **Full space-time A* guidance** — replace the greedy rollout with proper
   priority-queue A* on Eq. 1 cost. This is the single upgrade that unblocks
   both G1 (better paths through corridors) and G2 (warm-start provides A*
   pruning bound). The latency budget (82ms current, 500ms target) has ample
   headroom.

2. **LaCAM escalation** — when PIBT fails to resolve a conflict, escalate to
   LaCAM-level search. This unblocks the recursive PIBT priority inheritance
   for warehouse/maze maps. Phase 5.

3. **Download real MovingAI maps** — for exact paper reproduction.

The substrate is functional and well-performing on the 2/4 maps it handles.
The failures are honest, explained, and the path forward (A* upgrade) is clear.
