# Issue 144: LaCAM Escalation Follow-Up — Swap Technique + Warm-Start Re-Eval

**Date:** 2026-07-15
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Benchmark:** [440](../.benchmarks/440_lllg_paper_repro_goat.md)
**Prior issues:**
- [140](../.issues/140_pibt_priority_inheritance_and_warmstart_integration.md) — PIBT PI collapsed without LaCAM fallback; warm-start harmful
- [142](../.issues/142_full_space_time_astar_guidance_upgrade.md) — RESOLVED: full A* landed, +7-11% throughput, warm-start still harmful
- [143](../.issues/143_lacam_escalation_full_pibt.md) — RESOLVED: full recursive PIBT + priority shuffle fallback (covers PIBT PI + retry)
**Status:** ACTIVE — supplements Issue 143 with the swap technique (T1) and
warm-start re-evaluation (T2) not covered by 143's recursive PIBT + shuffle.

---

## Problem

Issue 143 landed recursive PIBT with priority inheritance + priority shuffle
fallback. This covers two of the three escalation mechanisms the paper uses.
The remaining gaps:

1. **Swap technique** — Issue 143's recursive PIBT handles general push
   conflicts but does not specifically detect the two-agent corridor-exchange
   pattern (agent `i` at `u` wants `v`, agent `j` at `v` wants `u`). The swap
   technique (Okumura 2023a, arXiv:2309.02425) detects this pattern and reverses
   both agents' preferences, enabling a clean exchange without deep recursion.
   This directly targets the ht_chantry corridor deadlock.

2. **Warm-start re-evaluation** — Issue 142 found occupancy-seeding HURTS
   throughput because PIBT deviations invalidate the forecast. Issue 143's
   recursive PIBT should reduce deviations (agents are pushed rather than
   waiting), so warm-start forecasts should be more accurate. This needs
   re-testing.

### Relationship to Issue 143

Issue 143 (concurrent agent) covers:
- Full recursive PIBT with priority inheritance (the Issue 140 mechanism, now
  with A* guidance reducing cascade risk)
- LaCAM priority shuffle fallback (bounded retry with randomized orderings)

This issue (144) covers the remaining mechanisms:
- T1: Swap technique (corridor-exchange detection)
- T2: Warm-start consumption re-evaluation (post-PIBT-PI)

The paper's three escalation mechanisms from the LLLG paper and its local
guidance companion (arXiv:2510.19072):

1. **PIBT priority inheritance** (Okumura et al. 2022) — when agent `i` wants a
   cell occupied by undecided agent `j`, recursively push `j` to move before
   committing `i`. Issue 140 implemented this (~200 lines) but it **collapsed
   throughput** because the recursive push is too conservative without a fallback.
   Issue 142's A* guidance should make conflicts rarer (the guidance steers agents
   away from congested cells), which may make PIBT PI viable now.

2. **Swap technique** (Okumura 2023a) — detect two-agent corridor deadlock (agent
   `i` at `u` wants `v`, agent `j` at `v` wants `u`) and reverse both agents'
   PIBT preferences so they exchange positions. The paper's local-guidance paper
   (arXiv:2510.19072 §"Integration with Swap") says: "if the swap situation is
   identified for an agent, discard the guidance term and use the reverse scoring
   ⟨0, −dist(v, g_i), ε⟩." This directly targets the ht_chantry corridor deadlock.

3. **Bounded configuration retry** — when PIBT (with PI + swap) still can't place
   an agent, retry with a different priority ordering before falling back to wait.
   This is a lightweight analog of LaCAM's DFS backtracking: instead of a full
   depth-first search over configuration space (which the LLLG paper's §4.C Fig. 9
   showed degrades lifelong throughput via anytime refinement), do a **bounded
   retry** (e.g. 3-5 random priority shuffles) to escape pathological orderings.

### Why now (post-Issue 142)?

Issue 140's PIBT PI failed because the greedy guidance produced congestion-heavy
paths → every cell was contested → PIBT PI cascaded into deep recursion that
stalled everyone. Issue 142 replaced the greedy rollout with proper space-time A*,
which improves throughput by 7-11% across all maps by finding better collision-
avoiding paths. With fewer contested cells, PIBT PI recursion should be shallower
and less likely to cascade.

The hypothesis is testable: re-enable PIBT PI (from Issue 140's git history,
commit preserved) against the current A* guidance and benchmark. If throughput
holds or improves, PIBT PI ships. If it still collapses, PIBT PI is deferred
again and only the swap technique + bounded retry ship.

## Fix

### T1: Swap technique (targets ht_chantry corridor deadlock)

**Note:** This is the one mechanism from the original plan that Issue 143 does
NOT cover. The swap technique is complementary to recursive PIBT — it catches
the specific two-agent exchange pattern that recursive PIBT may resolve
suboptimally (deep recursion vs. O(1) swap detection).

Add swap detection to `pibt_step` in `pibt.rs`:

Before processing agent `i`, check if there exists an undecided agent `j` such
that:
- `j` is at agent `i`'s preferred next cell (`config.pos(j) == preferred_i`)
- `j`'s preferred next cell is agent `i`'s current position
  (`preferred_j == config.pos(i)`)

If both hold, this is a **swap deadlock**. Resolve it by:
1. Processing agent `j` first with reversed preference: ⟨0, −dist(v, g_j), ε⟩
   (move toward `i`'s current cell, i.e. exchange positions).
2. Then processing agent `i` normally (move toward `j`'s current cell).

This unblocks the corridor exchange without full LaCAM search. The swap
detection is O(n) per agent (scan for the paired agent), negligible cost.

**Implementation note:** The paper says to "discard the guidance term" for the
swapped agent. This means: for the swapped agent `j`, set `guidance_mismatch = 0`
for ALL candidates (guidance is ignored), and reverse the `goal_dist` sort
(descending instead of ascending) so `j` moves toward `i`'s cell. The `ε`
tiebreak is preserved for determinism.

### T2: PIBT priority inheritance — covered by Issue 143

**Superseded.** Issue 143 landed full recursive PIBT with priority
inheritance. See [Issue 143](../.issues/143_lacam_escalation_full_pibt.md)
for implementation details and benchmark results.

### T3: Bounded configuration retry — covered by Issue 143

**Superseded.** Issue 143 landed LaCAM priority shuffle fallback (bounded retry
with randomized priority orderings). See Issue 143 for implementation details.

### T4: Re-evaluate warm-start consumption (post-Issue 143)

After T1 (swap) lands on top of Issue 143's recursive PIBT + shuffle,
PIBT deviations from guidance should be rarer (swap resolves corridor
deadlocks without deep recursion; PI resolves push conflicts with minimal
deviation). This means warm-start forecasts should be more accurate.

Re-test the warm-start occupancy-seeding from Issue 142:
- Seed warm-start forecasts into occupancy before round 0 (as Issue 142
  originally tried).
- Benchmark: if throughput holds (within 5% of non-seeded) AND LllgPi ≠
  LllgEmpty (G2 ratio < 1.0), warm-start consumption ships.
- If throughput still drops > 5%, warm-stay deferred again.

### T5: GOAT gate re-run

Re-run the full G1-G4 benchmark after T1 (swap) + T4 (warm-start re-eval):
- **G1**: ht_chantry ratio should improve from 0.01 → ≥ 0.15 (MARGINAL) or
  ≥ 0.30 (PASS). Warehouse should hold at ≥ 0.39 or improve.
- **G2**: LllgPi ratio should drop below 1.00 (warm-start consumed + beneficial).
- **G3**: 1556+ tests pass (no regression).
- **G4**: median latency ≤ 500ms (target), stretch ≤ 100ms. The swap adds
  O(n) per agent, warm-start seeding adds zero cost. Should fit in the
  82-93ms budget.

## Acceptance

- [ ] `cargo clippy -p katgpt-core --features multi_agent_path --lib`: clean.
- [ ] `cargo test -p katgpt-core --features multi_agent_path --lib`: all pass.
- [ ] `cargo test -p katgpt-core --lib`: 1556+ pass (G3 no-regression).
- [ ] G1 ht_chantry ratio ≥ 0.15 (MARGINAL) or ≥ 0.30 (PASS).
- [ ] G1 warehouse ratio holds ≥ 0.39 (no regression from A* baseline).
- [ ] G2 LllgPi ratio < 1.00 (warm-start consumed and beneficial) OR honest
      documentation of why it still fails.
- [ ] G4 latency ≤ 500ms (stretch ≤ 100ms).
- [ ] Update `.benchmarks/440_lllg_paper_repro_goat.md` with honest re-run results.

## Scope guardrails

1. **No full LaCAM DFS.** The LLLG paper's §4.C Fig. 9 showed LaCAM* anytime
   refinement degrades lifelong throughput. Issues 143 + 144 implement
   lightweight escalation (recursive PIBT + shuffle + swap), not a complete
   search. If the lightweight escalation fails to reach G1/G2 targets, the
   honest conclusion is that the substrate needs the full LaCAM DFS for
   one-shot MAPF (initial goal assignment) — but that's a separate, larger issue.

2. **This issue is additive to Issue 143.** T2 (PIBT PI) and T3 (bounded
   retry) are DONE by Issue 143. This issue only adds T1 (swap technique) and
   T4 (warm-start re-eval). If Issue 143's benchmark already passes G1/G2,
   this issue may be closed as moot.

3. **File budget.** Swap technique adds ~50 LOC to `pibt.rs`. Warm-start
   re-eval modifies ~20 LOC in `local_guidance.rs` (re-enable occupancy
   seeding behind a config flag). No new files.

4. **Determinism.** All random tiebreaks use the seeded `fastrand::Rng`
   passed through `tick()`. Replay is bit-identical.

## References

- [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, AAAI 2026 (LLLG)
- [arXiv:2510.19072](https://arxiv.org/abs/2510.19072) — Local Guidance for Configuration-Based MAPF (swap technique §"Integration with Swap")
- [Okumura et al. 2022](https://arxiv.org/abs/2204.10545) — PIBT: priority inheritance with backtracking
- [Okumura 2023a](https://arxiv.org/abs/2309.02425) — Improving LaCAM (swap technique)
- [Issue 140](../.issues/140_pibt_priority_inheritance_and_warmstart_integration.md) — PIBT PI collapsed without fallback
- [Issue 142](../.issues/142_full_space_time_astar_guidance_upgrade.md) — RESOLVED: A* landed, warm-start still harmful
- [Research 424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md) §1.3 mechanism (a), §1.5 anytime-refinement negative result

## TL;DR

Supplements Issue 143 (recursive PIBT + shuffle) with two remaining mechanisms:
the swap technique (O(1) corridor-exchange detection, targets ht_chantry) and
warm-start consumption re-evaluation (post-PIBT-PI, may unblock G2). If Issue
143's benchmark already passes G1/G2, this issue is closed as moot.
