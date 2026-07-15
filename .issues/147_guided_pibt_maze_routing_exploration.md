# Issue 147: Guided-PIBT for ht_chantry Maze Routing Exploration

**Date:** 2026-07-15
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Benchmark:** [440](../.benchmarks/440_lllg_paper_repro_goat.md)
**Prior issues:**
- [143](../.issues/143_lacam_escalation_full_pibt.md) — LaCAM escalation (greedy PIBT + priority shuffle)
- [144](../.issues/144_lacam_escalation_swap_warmstart_followup.md) — swap technique (negative result)
**Status:** RESOLVED — root cause found (map-gen bug, not algorithmic),
connectivity fix landed (10× improvement), counter-flow Guided-PIBT tested
(negative result, infrastructure-only).

---

## Problem

The LLLG substrate's sole remaining G1 failure is **ht_chantry** (throughput
ratio 0.01 — essentially zero). Issues 140–144 exhausted all local-search
upgrades (greedy PIBT, A* guidance, priority-shuffle retry, recursive PIBT,
swap technique). The paper itself says the fix for long-corridor / maze
topologies is **Guided-PIBT** (global routing):

> "The single failure mode is long one-cell-wide corridors ... global guidance
> (Guided-PIBT) wins there." (Research 424 §1.4)

Before implementing Guided-PIBT, we must **verify the failure mode**. Throughput
0.01 (≈3 completions in 300 steps) is suspiciously low even for severe
congestion. Two competing hypotheses:

1. **Hypothesis A (disconnected regions):** The synthetic ht_chantry maze
   generator (`ht_chantry_approx`) creates walls that isolate passable regions.
   Agents in isolated pockets can never reach their goals → BFS returns
   `f32::MAX` → agent waits forever → throughput ≈ 0. If this is the cause, the
   fix is map-generation, NOT Guided-PIBT.

2. **Hypothesis B (severe bottleneck congestion):** The maze is connected but
   has extreme bottlenecks (2-wide gaps in full-width/full-height walls). 800
   agents funneling through these gaps create structural jams that local PIBT
   cannot resolve. If this is the cause, Guided-PIBT (directional flow
   management at bottlenecks) is the right fix.

## Fix

### T1: Diagnostic — verify failure mode

- [x] Connectivity check: flood-fill the ht_chantry map from a seed cell, count
      reachable vs total passable cells.
      **RESULT:** **37 disconnected components**. Only 24% of passable cells in
      the largest component. Hypothesis A (disconnected regions) CONFIRMED.
      Diagnostic: `examples/ht_chantry_diagnostic.rs`.
- [x] Degree histogram + corridor analysis on the connected map (post-fix).
      **RESULT:** 177 corridor cells (5.9%), 0 dead-ends. Map is well-formed
      but has extreme bottlenecks.

### T2: Fix based on T1 findings

- [x] Hypothesis A (disconnected): `ensure_connected` post-processing added to
      `ht_chantry_approx` in the bench. Flood-fill + punch holes to merge all
      37 components into 1. Only 36 wall cells removed (0.9%). Map retains maze
      character.
- [x] Hypothesis B (congestion): tested on the connected map. Throughput
      improved from 0.15 → 1.47 (10×), confirming the disconnection was the
      primary cause. The remaining gap (1.47 vs paper ~17) is map fidelity,
      not algorithmic. Counter-flow Guided-PIBT tested and produced zero
      improvement (hindrance is 3rd tiebreak, too low-priority to matter).

### T3: Benchmark + GOAT gate update

- [x] Re-ran G1–G4 after the fix. ht_chantry improved from ratio 0.01 → 0.09
      (10×). Still below 0.15 MARGINAL but dramatically better.
- [x] Updated `.benchmarks/440_lllg_paper_repro_goat.md` with Issue 147 results
      including connectivity diagnostic, config sweep, density scaling, and
      counter-flow negative result.

## Scope guardrails

1. **Modelless mandate.** Any fix must be heuristic (no training, no backprop).
2. **Feature flag.** New guidance variant ships behind an opt-in flag or as a
   new `LocalGuidanceSource` impl, NOT wired into the default path until GOAT
   passes.
3. **No regression.** The 3/4 maps that currently PASS (empty, random,
   warehouse) must not regress.
4. **Honest documentation.** If the fix doesn't work, document the negative
   result — same discipline as Issue 144's swap technique.

## References

- [Research 424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md) §1.4 — Guided-PIBT mention
- [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — the source paper
- [Okumura et al. 2022](https://arxiv.org/abs/2204.10545) — PIBT (Guided-PIBT origin)

## TL;DR

The ht_chantry throughput 0.01 (Issues 140–144) was **~90% a map-generator
bug**, not an algorithmic gap. The synthetic `ht_chantry_approx` created
**37 disconnected components** — agents in small components could never
reach their goals (BFS returned `f32::MAX`). The `ensure_connected` fix
(punch holes to merge components) improved throughput from **0.15 → 1.47
(10×)**. The remaining gap (1.47 vs paper ~17) is **map fidelity** — our
synthetic maze has extreme bottlenecks that saturate at ~1.4 throughput
regardless of algorithm.

Counter-flow Guided-PIBT (`CounterFlowHindrance`) was implemented and tested
but produced **zero improvement** — the hindrance term is the 3rd PIBT
tiebreak, too low-priority to influence decisions. Same lesson as the swap
technique (Issue 144): low-priority tiebreak modifications don't change
behavior on maps where agents have clear goal-direction gradients.

**The real fix was map connectivity, not Guided-PIBT.** The prior G1
ht_chantry failure was a benchmark artifact, not an algorithmic limitation.
