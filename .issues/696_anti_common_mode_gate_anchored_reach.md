# Issue 696: RVM modelless extraction — anti-common-mode scalar gate + anchored signed-reach blend operator

**Status:** T1+T2 DONE (2026-08-28, feat `89f78b09`, bench [690](../.benchmarks/690_anti_common_mode_anchored_reach_goat.md): G1 9+8 unit gates PASS, G2 blend 234 ns / score 6023 ns @ N=1000 — the sub-µs ask honestly missed on the score axis (two exact selection passes, sub-µs at N ≲ 160), G3 default 1978/0/7 + feature-on = default + module tests, G4 0 allocs; both features opt-in — promotion only via T3. T3/T4/T5 OPEN.

## Source pattern (why these two primitives)

Two closed-form shapes extracted from the RVM paper's reward/anchor design, both grep-verified absent from the 15-repo corpus (2026-08-28):

1. **`anti_common_mode`** — the DT2 dynamic-tracking reward shape: a scalar signal that resists capture by the population-dominant degenerate mode, via 4 composable steps:
   - **peak-quantile statistic** — summarize a distribution by its fastest 5%, not its mean (`select_nth_unstable` partition at the P95);
   - **context-scaled threshold** — τ proportional to the observable's dynamic range (paper: `τ = 6·min(H,W)/256`);
   - **median subtraction over the hack-carrying population** — `m' = m − median(population)` cancels the common mode exactly and is robust to the outliers being detected (the population is the unit the hack operates through — pixels-per-frame-pair in the paper; NPCs-per-zone for crowd fear);
   - **band window** — `clip((m'−τlo)/(τmid−τlo)) · clip((τhi−m')/(τhi−τmid))`: both extremes score 0, only the genuine middle band earns signal.

2. **`anchored_reach`** — the RVM Eq. 7 update family as a single branch-free operator: `out = anchor + A·(candidate − anchor)` with signed reach `A ∈ ℝ` sweeping five regimes: A=0 clamp / (0,1) blend / 1 adopt / >1 overshoot / <0 repel. Plus reward-modulated reach schedules `A(r)`: linear `r`, house-sigmoid `2σ(kr)−1`, sign-flip `(2r−1)/β̄`. Sigmoid not softmax by construction (RVM itself never normalizes across the group — signed unnormalized weights suffice).

**Signal-diff vs incumbents (coverage check, 2026-08-28):** `katgpt-core::flow::steering::blend_steering(flow, avoidance, flow_weight)` consumes a hand-set convex weight over two candidate directions — no anchor-relative form, no signed reach, no A(r) modulation, no overshoot/repel. riir-engine `ReestimationScheduler`'s `MultiTickBandWindow` is a *temporal* observation window for band-edge triggers, not a value-band + median-subtract gate. Reward-hack defenses in the corpus (DeltaFilter 6-stage, ReviewMetrics benefit-ratio, path_consistency, length/BLEU penalties) are detection/gating, none cancels the population-dominant mode by median subtraction.

- [x] T1 — `anti_common_mode` primitive in `katgpt-core` (behind opt-in feature): peak-quantile + median-subtract + band-window over `&[f32]`, zero-alloc, `select_nth_unstable` based. Unit gates: median cancels a population-dominant mode exactly; both band tails score 0; peak-quantile ≠ mean on a minority-active distribution.
- [x] T2 — `anchored_reach` operator in `katgpt-core`: `out = anchor + A·(cand − anchor)`, per-axis or scalar A; A(r) schedule constructors (linear / `2σ(kr)−1` / sign-flip). Unit gates: the 5 regimes produce the predicted outputs; bit-identical at A=1 with plain adoption.
- [ ] T3 — **consumer PoC (the falsifiable headline): CLR crowd-panic re-enable.** `tick_swarm_emotions_collective` (opt-in `clr_collective_threat`) was demoted because one monster panics the entire swarm to the borders (riir-mmorpg-examples Plan 019). Re-enable WITH the T1 gate on per-NPC threat exposure (median over the zone's NPC population; band-window the genuine middle). 3 arms in `riir-poc` or the mmorpg harness: panic+gate / panic-only / baseline. PASS = border-band occupancy at baseline AND Bench 010's distributed-threat detection retained (200-NPC cluster, ~60/200 direct observers → detection ≈ 100%).
- [ ] T4 — anchored-reach consumers (each its own falsifiable A/B, file separately if adopted): (a) think-brain belief update with overshoot reach `A = 1 + λ·target_speed` as lead prediction on moving platforms (belief lead-error < fixed-A at equal observation cadence); (b) negative reach on below-group-mean outcomes as grudge/fear contrastive aversion (repeated-failure acceptance rate drops under A<0, unchanged at A=0); (c) guidance-as-anchor — expensive planner output as followers' frozen anchor (path-adherence within ε at ≥4× less think-cost than followers re-planning).
- [ ] T5 — group-definition A/B sub-note: same raw scores standardized per-zone vs globally produce different signed weights → different emergent competition structure (local relative fear vs world-scale ranking). One-line change, document the behavioral consequence; fold into whichever consumer lands first.

## GOAT gate shape (per primitive)

G1 correctness (unit gates above, bit-identical where claimed); G2 perf (zero-alloc, sub-µs at N=1000 — both are O(N) scalar ops); G3 no-regression (feature-off byte-identical; T3's baseline arm); G8 emergent-behavior falsifiable A/B (T3 is the headline). Promotion only via T3's PASS — the primitive exists to kill a measured failure, and its value claim IS that PoC.

## References

- Research 433 (riir-train) — full Path 0 decomposition + training-half mapping to Plan 360
- arXiv:2608.23664 §C.3 (DT/DT2 reward construction) + Eq. 7 (anchored regression family)
- CLR records: riir-mmorpg-examples Plans 018/019, Bench 010 (detection G8), Bench 011 (composition)
- Sibling precedent: katgpt-rs Issue 695 (`bounded_target` + `realization_gap`, the Research 432 extraction — same paper family, complementary halves)
- Bench 690 — T1+T2 GOAT-shape record (`.benchmarks/690_anti_common_mode_anchored_reach_goat.md`): gates + measured G2 + honest scope notes
