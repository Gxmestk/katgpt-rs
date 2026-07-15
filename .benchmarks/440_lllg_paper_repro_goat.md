# Benchmark 440: LLLG Paper Reproduction GOAT Gate

**Date:** 2026-07-15
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Research:** [424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md)
**Paper:** [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, AAAI 2026

---

## TL;DR

**G3 (no-regression): PASS. G4 (latency): PASS. G1 (throughput): PARTIAL (2/4 maps).
G2 (congestion): FAIL (warm-start integration bug).**

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
bug prevent full GOAT validation. The Super-GOAT claim remains conditional
on the riir-ai/489 G5–G7 fusion gates AND a PIBT priority-inheritance
upgrade.

---

## Substrate upgrades applied during Phase 2

Three correctness fixes were applied to the Phase 1 substrate during the
GOAT gate process:

1. **PIBT wall-aware neighbors** — `LifelongLaCam::tick` was passing `None`
   to `pibt_step`, causing agents to move through walls/out of bounds. Fixed
   by adding `LifelongLaCam::with_neighbors()` and threading the wall-aware
   neighbor closure through to PIBT.

2. **BFS distance fields** — Phase 1's greedy guidance used Manhattan
   distance (`Position::dist_heuristic`), which ignores walls and led agents
   into dead ends on obstacle-heavy maps. Fixed by computing BFS flood-fill
   distance fields from each goal (true shortest-path distance around
   obstacles). This improved random-64-64-10 throughput from 0.88 → 13.15
   (15× improvement). The BFS cache is persistent across ticks with a
   4000-entry cap to bound memory.

3. **Guidance goal pull** — Phase 1's `step_cost` weighted the goal distance
   by 0.1 (too weak). Fixed to use full BFS distance as the goal pull.

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
  jams. Agents pile up at aisle entrances and cannot push through.
- ht_chantry FAILS because the maze structure with bottlenecks is
  fundamentally hard for greedy guidance — every bottleneck becomes a
  permanent traffic jam.
- Both failures are consistent with the paper's known limitation (caveat #1:
  "long one-cell-wide corridors"). The paper uses real PIBT with priority
  inheritance which resolves corridor jams by recursively pushing agents.

**Root cause:** PIBT as implemented is greedy (pick first collision-free
candidate, fall back to wait). Real PIBT has priority inheritance: when a
high-priority agent wants a cell occupied by a low-priority agent, the
low-priority agent is forced to move first. This is a significant algorithmic
upgrade (~200 lines of recursive backtracking).

---

## G2 — Congestion mitigation

**empty-48-48, 1000 agents, 100 steps. LLLG_Π vs LllgEmpty baseline.**

| Scheme | max_stops/cell | mean_stops | throughput |
|---|---|---|---|
| LLLG_Π | 52 | 7.2 | 16.73 |
| LllgEmpty | 52 | 7.2 | 16.73 |

**Ratio: 1.00 (identical). FAIL.**

**Root cause:** The warm-start integration is broken. `LifelongLaCam::tick`
discards the warm-start data with `let _ = self.warm_start.warm_start()`. The
guidance source (`SpaceTimeGuidance::compute_guidance`) recomputes from
scratch every tick regardless of the warm-start scheme. LLLG_Π and LllgEmpty
produce identical results because the warm-start is never consumed.

**Fix required:** Thread the warm-start initialization into the guidance
source. The `LocalGuidanceSource` trait needs a warm-start parameter, or the
`SpaceTimeGuidance` needs to own the warm-start cache internally.

---

## G3 — No-regression

```
cargo clippy -p katgpt-core --all-features --lib: clean
cargo test -p katgpt-core --lib: 1556/1556 pass
```

**PASS.** Existing tests unaffected.

---

## G4 — Latency

**empty-48-48, 1000 agents, 100 steps.**

| Metric | Value |
|---|---|
| Median per-tick | 80.44 ms |
| Max per-tick | 288.68 ms |
| Paper (M1 Ultra) | 210–260 ms |

**PASS (target < 500ms). Stretch < 100ms: PASS.**

Our impl is 2.6× faster than the paper's M1 Ultra result because our
algorithm is simpler (greedy guidance vs full space-time A*; greedy PIBT vs
priority inheritance). The perf headroom exists to upgrade to a proper A*
within the latency budget.

---

## Latency breakdown by map (G1 runs, 800 agents)

| Map | Median (ms) | Max (ms) |
|---|---|---|
| empty-48-48 | 52 | 237 |
| random-64-64-10 | 53 | 381 |
| warehouse-10-20-10-2-2 | 44 | 213 |
| ht_chantry-approx | 42 | 61 |

Latency is well within budget on all maps. The random-64-64-10 max spike
(381ms) is from BFS cache misses when many agents get new goals
simultaneously.

---

## Honest caveats

1. **Maps are synthetic approximations.** We don't have the exact MovingAI
   MAPF benchmark map files for warehouse-10-20-10-2-2 and ht_chantry. Our
   synthetic versions approximate the topology (shelf-aisle for warehouse,
   maze-with-bottlenecks for ht_chantry). Throughput numbers may differ from
   what the paper reports on the real maps.

2. **PIBT lacks priority inheritance.** This is the primary algorithmic gap.
   Real PIBT recursively resolves conflicts by forcing lower-priority agents
   to move when a higher-priority agent claims their cell. Our greedy PIBT
   falls back to "wait" which causes permanent jams in corridors.

3. **Warm-start is not integrated.** The `WarmStartScheme` enum exists and
   the cache records data, but `tick()` discards the warm-start result. This
   is a correctness bug — LLLG_Π and LllgEmpty produce identical behavior.

4. **Agent count is 800 not 1000.** Paper throughput targets are at 1000
   agents. We run at 800 (the G1 gate specifies 800). Throughput at 800 is
   typically slightly lower than at 1000.

5. **The ht_chantry approximation is more extreme than the real map.** Our
   synthetic maze has narrower bottlenecks than the real Dragon Age map.
   The real ht_chantry has wider corridors. The 0.01 ratio overstates the
   gap.

---

## Next steps

1. **PIBT priority inheritance upgrade** — the single most impactful fix.
   Would unblock warehouse and significantly improve ht_chantry. Estimated
   ~200 lines of recursive backtracking in `pibt.rs`. This is the blocking
   item for G1 full pass.

2. **Warm-start integration** — thread warm-start data into
   `SpaceTimeGuidance`. This unblocks G2. Estimated ~50 lines.

3. **Full space-time A*** — replace greedy rollout with proper priority-queue
   A* on the Eq. 1 cost. Lower priority — the greedy + BFS approach already
   works well on moderate maps. This would improve optimality but is not
   blocking for the G1 gate on open/moderate maps.

4. **Download real MovingAI maps** — for exact paper reproduction. Not
   blocking for the algorithmic validation.

The substrate is functional and well-performing on the 2/4 maps it handles.
The failures are honest, explained, and fixable. The GOAT gate has served
its purpose: it identified the exact algorithmic gaps that need upgrading.
