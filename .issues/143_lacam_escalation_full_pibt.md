# Issue 143: LaCAM Escalation — Full Recursive PIBT + Priority Shuffle Fallback

**Date:** 2026-07-15
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Prior issues:** [140](140_pibt_priority_inheritance_and_warmstart_integration.md) (PIBT PI + warm-start), [142](142_full_space_time_astar_guidance_upgrade.md) (A* guidance)
**Status:** RESOLVED

## Context

Issue 142 resolved the A* guidance upgrade (throughput improved +7-11% on all
maps, warehouse crossed G1 threshold at ratio 0.39). Two GOAT gates remain
blocked:

1. **G1 ht_chantry (ratio 0.01)** — maze topology with narrow bottlenecks.
   The w_Φ=5 window can't see far enough; agents deadlock in corridors.
2. **G2 warm-start** — forecast is invalidated by PIBT deviations on dense
   maps.

Both point to the same fix: **LaCAM escalation**. The current `pibt_step` is
greedy (take first collision-free candidate, let later agents adapt). When
agents meet head-on in a corridor, neither can move — the greedy variant has
no eviction mechanism.

## Acceptance Criteria

- [x] Implement full recursive PIBT with priority inheritance (Okumura et
      al. 2022): when agent i wants a cell occupied by undecided agent j, i
      recursively evicts j (j must move first).
      **RESULT:** Implemented, benchmarked, and **REJECTED**. Recursive PIBT
      collapses throughput -92% (empty-48-48: 18.6 → 1.5). The eviction
      forces agents to move away from their goals, creating cascading stalls.
      The greedy PIBT + bounded retry is the right approach for lifelong MAPF.
- [x] Cycle prevention in the recursion chain (i evicts j, j evicts k, ...).
      **RESULT:** Implemented (chain set), but the recursive approach itself
      was rejected. The greedy PIBT doesn't need cycle prevention.
- [x] LaCAM escalation fallback: when PIBT deadlocks, retry with shuffled
      priority orderings (limited retries).
      **RESULT:** Implemented. When ≥ 20 agents are stuck, retry with
      shuffled orders (up to 2 retries). Warehouse improved +8.3%.
- [x] All existing tests pass (25+ multi_agent_path + 1556 base).
      **RESULT:** 27 multi_agent_path tests pass (25 original + 2 new),
      1556 base tests pass.
- [x] Benchmark: ht_chantry throughput improves (target ratio > 0.05, ideally
      > 0.10). No regression on empty/random/warehouse.
      **RESULT:** ht_chantry unchanged (ratio 0.01 — needs global routing,
      not local retry). Warehouse improved +8.3% (0.39 → 0.42). No regression
      on empty/random. ht_chantry target NOT met — the maze topology is an
      algorithmic ceiling for local-search approaches (paper caveat #1).
- [x] Clippy clean on all-features.
      **RESULT:** Clean.

## Root Cause Analysis

The greedy PIBT (current implementation) processes agents in priority order,
each taking the first collision-free candidate. It has a critical flaw:

**Corridor head-on deadlock:** Agents A (going north) and B (going south)
meet in a 1-wide corridor. With greedy PIBT:
- A (higher priority) wants B's cell → blocked → waits.
- B (lower priority) wants A's cell → edge collision → waits.
- Both wait forever. No agent ever moves.

**Full PIBT with priority inheritance solves this:**
- A (higher priority) wants B's cell → B is undecided → evict B.
- Recursively try to place B: B tries A's cell → edge collision (A decided
  to move to B's current). B tries the cell behind it (south). If that cell
  is free, B backs up.
- A takes B's old cell. Conflict resolved.

The key insight from Issue 140: recursive PIBT alone can be too conservative
on dense maps (cascading stalls). The solution is to pair it with LaCAM
escalation: when PIBT deadlocks, retry with different priority orderings.

## Implementation

### Phase A — Full recursive PIBT (`pibt.rs`)

Replace the greedy loop with recursive priority inheritance:
- Agent processing in priority order (main loop).
- For each undecided agent, call `pibt_recursive()` which:
  1. Generates candidates sorted by lexicographic cost.
  2. For each candidate cell u:
     a. If u collides with a decided agent → skip.
     b. If u is occupied by undecided agent j (and j not in chain, i has
        priority) → recursively evict j. If j moved, take u.
     c. If u is free → take it.
  3. If no candidate works → wait in place.

### Phase B — LaCAM escalation (`pibt.rs`)

When recursive PIBT produces stuck agents, retry with shuffled priorities:
- Up to N retries (configurable, default 3).
- Each retry randomizes the priority order of stuck agents (elevating them).
- Return the result with the fewest stuck agents.

## TL;DR

LaCAM escalation landed as greedy PIBT + bounded priority-shuffle retry.
Warehouse throughput improved +8.3% (ratio 0.39 → 0.42). Recursive PIBT
(full priority inheritance) was tested and REJECTED (-92% throughput).
ht_chantry remains at ratio 0.01 — needs global routing (Guided-PIBT), not
local retry. This is the paper's own known limitation (caveat #1).

**Net result:** G1 PARTIAL 3/4 maps (unchanged map count, warehouse ratio
improved), G2 FAIL (unchanged), G3/G4 PASS. Promotion decision: KEEP OPT-IN
(unchanged).

This was the last local-search upgrade available. The remaining blockers
(ht_chantry + warm-start) require either global routing (Guided-PIBT) or the
full LaCAM configuration tree search — both are significantly larger
implementations. The substrate is now at its local-search algorithmic ceiling.

The substrate works correctly on 3/4 map types and is ready for the
riir-ai/489 private runtime fusion (which can consume it via the four
pluggable seams). The ht_chantry limitation should be documented as a known
constraint for consumers with maze-heavy topologies.
