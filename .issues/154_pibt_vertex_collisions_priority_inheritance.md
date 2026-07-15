# Issue 154 — PIBT Vertex Collisions on Congested Maps (Lacks Priority Inheritance)

**Date:** 2026-07-15
**Status:** OPEN (poc/optimization)
**Discovered by:** Proposal 023 / Issue 516 G6c measurement
**Severity:** Medium (substrate guarantee is overstated; throughput workaround exists)

## Problem

The `pibt_step` function in `crates/katgpt-core/src/multi_agent_path/pibt.rs`
produces **vertex collisions** on congested maps. The "Collision-free by
construction" guarantee documented in
`crates/katgpt-core/src/multi_agent_path/config.rs:83-84` does NOT hold
empirically.

On the riir-ai G6c benchmark scenario (60 NPCs, 20×20 grid, 6-cell bottleneck
gap, 200 ticks), the LLLG substrate has:
- **125/200 ticks (62.5%) with vertex collisions** — two agents on the same cell
- **0/200 edge collisions** — PIBT's swap prevention works correctly
- **75/200 ticks (37.5%) collision-free**

Full measurement: `riir-ai/.benchmarks/516_g6c_collision_freedom_delta.md`

## Root Cause

### The stuck-agent forced-wait bug

In `greedy_pibt_pass` (line ~453-457), when an agent is "stuck" (can't find a
collision-free move AND can't wait because its current cell is committed by
another agent), the code **forces the agent to wait at its current position**:

```rust
for agent in &stuck {
    let i = usize::from(*agent);
    moves[i] = Some(config.pos(*agent).clone());
}
```

But the stuck agent's current position IS committed by another agent (that's
exactly why it's stuck — `can_wait = false`). Forcing it to wait there creates
a vertex collision.

The same issue exists in `pibt_step` (line ~270-274) after LaCAM retries.

### Why greedy PIBT produces stuck agents

The current implementation is a **simplified PIBT without priority inheritance
(PI)**. The original PIBT paper (Okumura et al. 2019) describes PI as a key
mechanism: when agent A wants to move to cell P (occupied by agent B), A
"pushes" B — B's priority is raised above A's, B is processed first, and if B
can move, A gets P. Without PI, agent A commits to P, then B can't move and
can't wait (P is committed) → B is stuck → forced-wait → collision.

This is the same root cause documented in
`.benchmarks/440_lllg_paper_repro_goat.md`: "greedy PIBT lacks priority
inheritance." It causes G1 PARTIAL (2/4 maps fail).

## The all-wait fix (tested and rejected)

A straightforward fix — fall back to all-wait when any agent is stuck — was
implemented and tested:

```rust
let all_wait: Vec<P> = config.positions.clone();
Ok(JointAction::new(all_wait))
```

**Results with the all-wait fix:**

| Metric | Before fix | After fix |
|---|---|---|
| Collision-free rate | 37.5% | **100.0%** (fixed) |
| G6a convergence (riir-ai) | 26.7% | **0.0%** (killed) |
| G7a crossing time (riir-ai) | 12 ticks | **200 ticks** (G7 BROKEN) |

The all-wait fix makes the substrate collision-free but **destroys throughput**
on congested maps. It was reverted; the current code (collisions on congested
ticks) is the lesser evil vs zero throughput.

The revert adds documentation comments in `pibt.rs` explaining the tradeoff.

## Proposed Fix: PIBT Priority Inheritance

Implement the PI mechanism from Okumura et al. 2019 §4:

1. When agent A wants to move to cell P (occupied by agent B), A "pushes" B.
2. B's priority is raised above A's (priority inheritance).
3. B is processed before A and must find a collision-free move.
4. If B can move, A gets P. If B can't, A doesn't get P (stays or tries elsewhere).

This resolves stuck agents WITHOUT all-wait. With PI, both collision-free
(G6c) AND convergence (G6a) can pass simultaneously.

### Expected impact

- **G1 (substrate gate):** 2/4 maps → potentially 3-4/4 (PI resolves the
  greedy PIBT failures on warehouse and ht_chantry)
- **G6c (consumer gate):** 0.360 → ~0.98+ (LLLG collision-free → 1.0)
- **G6a (consumer gate):** 26.7% → potentially higher (agents can push through
  bottlenecks without collisions)
- **G7 (consumer gate):** stays PASS (12 ticks — PI doesn't hurt open maps)

### Complexity

PI is a non-trivial algorithmic change. The `greedy_pibt_pass` function needs
to be restructured to support dynamic priority updates during the pass (not
just a fixed priority order). The `pibt_step` function's retry loop may also
need adjustment. Estimated effort: a dedicated plan (Phase 1: implement PI,
Phase 2: benchmark, Phase 3: GOAT gate re-run).

## References

- Okumura et al. 2019, "Priority Inheritance with Backtracking for Iterative
  Multi-agent Path Finding" (PIBT paper)
- Arita & Okumura 2026, arXiv:2605.16855 (LLLG source paper)
- riir-ai Proposal 023 (the proposal that motivated this measurement)
- riir-ai Issue 516 (the consumer-side issue tracking G6c)
- riir-ai Benchmark 516 G6c (the full measurement + root-cause analysis)
- `katgpt-rs/.benchmarks/440_lllg_paper_repro_goat.md` (substrate G1-G4 gate,
  documents the PI gap)

## TL;DR

PIBT has vertex collisions on 62.5% of congested-map ticks because stuck agents
are forced to wait at positions committed by others. The all-wait fix corrects
collisions but kills throughput (rejected). The proper fix is PIBT priority
inheritance (Okumura 2019). This is a dedicated plan, not a quick patch.
