# Issue 142: Full Space-Time A* Guidance Upgrade (Plan 440 Phase 2 real blocker)

**Date:** 2026-07-15
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Benchmark:** [440](../.benchmarks/440_lllg_paper_repro_goat.md)
**Prior issue:** [140](../.issues/140_pibt_priority_inheritance_and_warmstart_integration.md) — identified this as the real blocker
**Status:** RESOLVED

---

## Problem

Issue 140 identified that both G1 (warehouse/maze throughput) and G2 (congestion
mitigation) are blocked by the same root cause: `astar_for_agent` in
`local_guidance.rs` is **not actually A*** — it is a greedy rollout that commits
to the locally-best step at each depth without backtracking.

The greedy rollout:
- Cannot plan multi-step detours around collisions (myopic 1-step lookahead).
- Cannot consume warm-start data (Issue 140 found occupancy-seeding collapses
  throughput because the greedy rollout over-avoids forecast cells).
- Gets stuck following the BFS gradient into corridor dead-ends.

Additionally, the multi-round refinement (`m=2`) is **broken**: each round clears
the occupancy map, making rounds 1+ identical to round 0 (agent 0 always sees an
empty map). The rounds are no-ops.

## Fix

Three coupled changes to `local_guidance.rs`:

### T1: Rewrite `astar_for_agent` as proper space-time A*

Replace the greedy rollout with a priority-queue A* over `(position, depth)` state
space:
- **State**: `(P, u8)` — position and depth (steps taken, 0..=w_phi).
- **Start**: `(start, 0)`. **Goal test**: `depth == w_phi`.
- **g(n)**: accumulated transition cost = `Σ (1 + α·χ)` over transitions.
- **h(n)**: BFS distance from position to goal (admissible — never overestimates
  the true remaining cost since each remaining transition costs ≥ 1).
- **f(n)**: `g + h`. Min-heap by `(f, depth)`.
- **Transition cost**: `1.0 + alpha * chi(neighbor, depth)` — base 1 per move
  plus collision penalty.
- **Path reconstruction**: follow `came_from` from goal-depth state back to start.

This gives the A* w-step lookahead with proper cost accumulation — it can see
that accepting 1 collision now leads to a clear path, while the greedy rollout
only sees the immediate collision cost.

### T2: Fix multi-round refinement (unrecord/re-record)

Currently each round calls `clear_occupancy()`, making rounds no-ops. Fix:
- Clear occupancy **once** at the start.
- Seed warm-start forecasts **once**.
- Each round, for each agent: **unrecord** previous path → compute A* → **record**
  new path. This way round 1 agent 0 sees round-0 paths of agents 1..n-1 (a real
  improvement over round 0 where agent 0 saw only warm-start forecasts).

### T3: Consume warm-start via occupancy seeding with self-collision removal

Issue 140 found occupancy-seeding collapses throughput for the greedy rollout.
Hypothesis: A* handles it because the collision cost is a soft penalty weighed
against total path cost, not a myopic per-step avoidance.

Implementation:
- Seed all agents' warm-start forecasts into occupancy before round 0.
- Before processing agent `i`, `unrecord_path(warm_start[i])` to remove
  self-collision (agent doesn't avoid its own forecast).
- After A*, record agent `i`'s new path.

## Acceptance

- [x] `cargo clippy -p katgpt-core --features multi_agent_path --lib`: clean.
- [x] `cargo test -p katgpt-core --features multi_agent_path --lib`: 25/25 pass.
- [x] `cargo test -p katgpt-core --lib`: 1556/1556 pass (G3 no-regression).
- [x] Re-run GOAT gate benchmark. G1 improved (3/4 maps pass, was 2/4).
      G2 still fails (warm-start confirmed harmful even with A*).
- [x] Update `.benchmarks/440_lllg_paper_repro_goat.md` with honest re-run results.

## Status: RESOLVED

Full space-time A* landed. Throughput improved on all 4 maps (+7-11%). 3/4
G1 maps now pass (was 2/4). Warehouse crossed the 0.30 threshold. ht_chantry
remains at 0.01 (needs LaCAM, Phase 5). Warm-start consumption confirmed
harmful (occupancy-seeding creates phantom collisions from stale forecasts).

The remaining G1/G2 blockers both point to **LaCAM escalation** (Phase 5):
- ht_chantry needs LaCAM to resolve corridor conflicts.
- Warm-start needs LaCAM to keep PIBT deviations rare so forecasts stay accurate.

## References

- [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, AAAI 2026
- [Issue 140](../.issues/140_pibt_priority_inheritance_and_warmstart_integration.md) — prior investigation
- [Research 424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md) §1.3 mechanism (a)

## TL;DR

The single highest-leverage upgrade that unblocks G1 (warehouse/maze throughput),
G2 (warm-start consumption), and sets up PIBT priority inheritance (needs A* +
LaCAM fallback). Replaces the greedy rollout with proper space-time A*, fixes the
broken multi-round refinement, and consumes warm-start via occupancy seeding.
