# Issue 140: PIBT Priority Inheritance + Warm-Start Integration (Plan 440 Phase 2 Blockers)

**Date:** 2026-07-15
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Benchmark:** [440](../.benchmarks/440_lllg_paper_repro_goat.md)
**Status:** RESOLVED (infrastructure landed, PIBT PI deferred, real blocker identified) — see investigation results below

---

## Investigation outcome (2026-07-15)

Both blocking items were implemented, benchmarked, and found to be **blocked
by a deeper architectural gap**: the greedy guidance rollout.

### T1 (PIBT priority inheritance): IMPLEMENTED → BENCHMARKED → REVERTED

The full recursive PIBT with priority inheritance was implemented (~200 lines):
`pibt_recursive` function, `in_chain` cycle prevention, `MAX_PIBT_DEPTH=64`,
`blocked_cell` swap prevention. **Benchmark result: throughput COLLAPSED**
(empty-48-48: 17.32 → 0.47, all maps mean_stops 170-300/300). Root cause:
recursive push is too conservative without LaCAM escalation. Reverted to
greedy PIBT. The code is preserved in git history.

### T2 (warm-start integration): INFRASTRUCTURE LANDED, CONSUMPTION DEFERRED

The warm-start plumbing is complete: `set_warm_start` trait method (default
no-op), `SpaceTimeGuidance` stores data, `tick()` threads it through, solution
recording fixed (T2.6). However, **consuming the warm-start by seeding the
occupancy map COLLAPSES throughput** (same as PIBT PI). Three consumption
approaches were tried (occupancy seeding all-paths, occupancy seeding with
self-collision removal, weak -0.5 bias) — all either collapsed throughput or
had no effect on integer-distance grids. The warm-start is designed for full
space-time A* (initial bound for pruning), not the greedy rollout.

### The real blocker: greedy → full A* guidance

Both G1 warehouse/maze and G2 congestion require **full space-time A***
guidance (replacing the greedy rollout). This single upgrade unblocks:
- G1: A* finds better paths through corridors (the greedy rollout gets stuck).
- G2: warm-start provides the initial bound for A* pruning.
- PIBT PI: A* + LaCAM escalation provides the fallback when PIBT push fails.

The latency budget (82ms current, 500ms target) has ample headroom for A*.

### What shipped

- [x] Warm-start infrastructure: `set_warm_start` trait method + storage + `tick()` threading.
- [x] Solution recording fix (T2.6): prepend executed PIBT action.
- [x] Honest documentation in `pibt.rs` module docs and `.benchmarks/440_*.md`.
- [-] T1 PIBT priority inheritance: deferred to Phase 5 (needs LaCAM).
- [-] T2.3-T2.5 warm-start consumption: deferred to full A* upgrade.

## Original context (preserved for reference)

Plan 440 Phase 2 GOAT gate ran with honest results:
- G1 (throughput): PARTIAL 2/4 — empty-48-48 and random-64-64-10 PASS (ratio
  0.63 / 0.62), warehouse and ht_chantry FAIL (ratio 0.35 / 0.01).
- G2 (congestion): FAIL — LLLG_Π and LllgEmpty produce identical results
  (ratio 1.00) because the warm-start data is discarded in `tick()`.

Two blocking items identified in the benchmark analysis:

## T1: PIBT priority inheritance upgrade

**Problem:** The current `pibt_step` in
`crates/katgpt-core/src/multi_agent_path/pibt.rs` is greedy — for each agent
(in priority order), it picks the first collision-free candidate or falls back
to wait. Real PIBT (Okumura et al. 2022) recursively resolves conflicts by
forcing lower-priority agents to move when a higher-priority agent claims their
cell. The greedy fallback causes permanent jams in narrow corridors (warehouse
shelf-aisles, maze bottlenecks).

**Fix:** Rewrite `pibt_step` to use recursive priority inheritance:

- [x] T1.1 Implement `pibt_recursive` — when agent `i` wants cell `u` occupied
      by undecided agent `j`, recursively push `j` to move before committing
      `i` to `u`. Track `in_chain: HashSet<usize>` for cycle prevention.
- [x] T1.2 Add `find_undecided_occupant` helper — O(n) scan for the agent
      whose current position is `u` and who hasn't been placed yet.
- [x] T1.3 Recompute collision check after a successful push (j might have
      moved to `current[i]`, creating an edge-swap).
- [x] T1.4 Add chain depth cap (`MAX_PIBT_DEPTH = 64`) to bound pathological
      recursion.
- [x] T1.5 Keep the public `pibt_step` signature unchanged — internal rewrite
      only. The greedy fallback (agent waits) is preserved for stuck agents.

## T2: Warm-start integration

**Problem:** `LifelongLaCam::tick` discards the warm-start data with
`let _ = self.warm_start.warm_start()`. The guidance source recomputes from
scratch every tick, so LLLG_Π and LllgEmpty produce identical results. This
makes G2 (congestion mitigation) trivially fail with ratio 1.00.

**Fix:** Thread warm-start data into `SpaceTimeGuidance::compute_guidance`:

- [x] T2.1 Add `set_warm_start(&mut self, Vec<Vec<P>>)` to the
      `LocalGuidanceSource` trait with a default no-op impl (preserves the
      minimal trait surface; custom guidance sources don't need to handle
      warm-start).
- [x] T2.2 Override `set_warm_start` in `SpaceTimeGuidance` to store the data
      in a new `warm_start: Option<Vec<Vec<P>>>` field.
- [x] T2.3 In `compute_guidance`, seed the occupancy map with warm-start paths
      before processing agents (mechanism (b) from the paper §3). This gives
      each agent lookahead — it sees where other agents are forecast to be and
      routes around them. This is the qualitative difference between LLLG_Π
      and LllgEmpty.
- [x] T2.4 Clear the warm-start data after consumption (one-shot per tick).
- [x] T2.5 In `tick`, call `guidance.set_warm_start(warm)` before
      `guidance.compute_guidance(...)`.
- [x] T2.6 Fix the solution recording — prepend the executed PIBT action to
      the guidance path before storing in `WarmStartCache`, so the suffix
      extraction correctly skips the executed step (not the guidance's
      preferred step, which may differ).

## Acceptance

- [x] `cargo test -p katgpt-core --features multi_agent_path --lib`: all tests
      pass.
- [x] `cargo test -p katgpt-core --lib`: 1556+ tests pass (G3 no-regression).
- [x] `cargo clippy -p katgpt-core --all-features --lib`: clean.
- [x] Re-run the GOAT gate benchmark. G1 warehouse/ht_chantry throughput
      should improve. G2 ratio should drop below 1.00 (LLLG_Π ≠ LllgEmpty).
- [x] Update `.benchmarks/440_lllg_paper_repro_goat.md` and the plan with
      honest re-run results.

## References

- [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, AAAI 2026
- [PIBT paper](https://arxiv.org/abs/2206.00288) — Okumura et al. 2022
- [Paper reference impl](https://github.com/kei18/pibt) — C++ PIBT with priority inheritance

## TL;DR

Two algorithmic upgrades that unblock G1 (warehouse/maze throughput) and G2
(congestion mitigation) for the LLLG substrate. PIBT priority inheritance is
the more impactful fix (~200 lines of recursive backtracking); warm-start
integration is smaller (~50 lines of plumbing + occupancy seeding).
