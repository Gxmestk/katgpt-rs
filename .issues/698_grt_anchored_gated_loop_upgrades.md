# Issue 698: GRT-anchored gated loop upgrades (modelless runtime corollaries of arXiv:2608.15062)

**Status:** OPEN — filed 2026-08-30 from `../.research/519_GRT_Gated_Recurrent_Transformers.md` (No-GD-advocate pass merged; substrate all shipped: Plans 108/136/428/304)

## Problem / opportunity

The GRT paper (5th in the looped-transformer family) supplies measured constants, orderings, and laws that upgrade our shipped loop stack (`forward_looped` + `ResidualGate` + `LoopStabilityMode` + halters) without any training. Three headline facts from the paper:

1. **The loop anchor must be FIXED, not drifting** — trained-gate anchor ablation: frozen prelude output 2.68 vs drifting `h(r-1)` 3.38 (**+0.70 nats**) vs raw embedding 3.73 vs zeros 8.08. Our Plan 428 FLT_res variant distributes the *drifting* prev state; GRT's evidence points at the frozen variant.
2. **The gate should be convex, not additive** — `h(r) = g⊙h(r-1) + (1−g)⊙o` bounds ‖h‖ ≤ max(‖prev‖, ‖out‖) for free; our additive `h̃ + ρ⊙h_prev` has no bound (the exact instability Plan 428 fights). And the gate schedule is transcribable: write-early → copy-late (openness 0.18 → 0.07 monotone; copy-saturated dims 19.7% → 28.8%).
3. **Exit is a quality rule with a floor** — 77% of loss reduction lands in the first 2 steps; per-step gain concave; extending R past training-R *degrades*. Feeds GainCostLoopHalter an `l_min=2` default + concavity assertion, and motivates an offline gain-spectrum table.

## Tasks (priority order — T1 de-risks the rest)

- [ ] T1 **Gain-spectrum measurement bench**: for a pinned fixture + existing weights, measure Δloss(r) for r = 0..R → committed lookup table (loop budget → expected quality); asserts monotonicity + records where the gain actually concentrates. G1 bit-identical re-measurement; fixture hash pinned in the same commit. **If our model puts 40% of gain in step 2, the paper's l_min=2 does not transfer — that is a finding, not a failure.**
- [ ] T2 **`LoopStabilityMode::FixedAnchor`**: hoist `ctx.prev_h` at `tau==0` into a dedicated anchor buffer; inject per-loop. A/B ordering test (fixed vs drifting vs none vs raw-embed) — falsifiable content is the *ordering*, not the absolute numbers. G3 flag-off byte-identical; G4 one hoisted buffer, zero alloc. Caveat honestly recorded: paper numbers are from anchor-trained weights; on vanilla-trained weights the anchor is OOD input — the ordering is the structural claim under test.
- [ ] T3 **Convex copy-late gate schedule interpretant** for `ResidualGate`: `ResidualGate::copy_late_schedule(loop_count, g0, gR)` + convex-blend application path; free-theorem spec test (‖h(r)‖ ≤ max(‖prev‖, ‖out‖) every loop, the HLA-boundedness class). Form-mismatch caveat: existing checkpoints trained under the additive form — fixture A/B (convex-schedule vs additive vs none) arbitrates; sweep 2–3 schedules, commit the sweep report. Note the direction tension with 414 Fusion B's stability decay (ρ→0.8): on shared weights the fixture decides; paper evidence (convergence by R) supports copy-late.
- [ ] T4 **Halter floors** (`gain_cost_halt.rs`): `l_min = 2` default + concavity floor rule — gated on T1's measured spectrum; keep the cos-θ oscillation detector armed as the non-convergent fallback (contraction is measured, not proven, for arbitrary weights).
- [ ] T5 **KV mean-across-steps**: `CacheStrategy::Mean` running mean of K/V across loop iterations (fixed loop order = deterministic f32 sum). **Lossy-surface promotion rule in full**: gate on argmax stability + max_abs band (the Bench-773 lesson — max_rel with a floor denominator cannot certify lossy numerics), never bit-identity; per-family conditional retention on the deployed path. tf_loop side: replaces the dedicated stash pass with the loop's own running mean — a whole window-forward deleted (measure end-to-end wall-clock).
- [ ] T6 **Per-step state noise** (deferred, afternoon-cost): BLAKE3-seeded per-(pos, loop) Gaussian on the loop input; expect a wash (+0.018 nats is the paper's smallest ablation, measured on noise-trained weights).
- [ ] T7 **Per-token difficulty routing** (deferred, gated): only if a T1-side probe shows ≥2× marginal-gain separation by step-1 entropy/margin; per-token branching breaks batched prefill — decode-only first. PABEE-class prior art [unverified — quota] covers the mechanism; our content is the separation measurement.
- [ ] T8 **Hand conditional gate** `σ(β·(cos(h, h_pre)−θ)+b)` (deferred, coin-flip): mechanism direction (open on divergence) extracted from the paper's contrastive projection finding; honest coin-flip on weights that never learned contrastive reading; requires T1 mechanism gate (gate-open events co-locate with high marginal gain, rank correlation threshold) + T2 anchor + inter-loop norm prerequisite.

## Dependencies

- T2/T3/T8 compose with Plan 428's `InterLoopNorm` (GRT's gate consumes LN inputs — normalization is a prerequisite, not a competitor); composition ablated in order: norm → anchor → gate schedule.
- Training-side companion lives in `riir-train/.plans/364` (depth sampling, gate-init recipes, GRT-micro arms) — do not duplicate here; this issue is runtime/modelless only.

## References

- `../.research/519_GRT_Gated_Recurrent_Transformers.md` — full distillation, Path-0 table, signal-diff vs our `ResidualGate`
- arXiv:2608.15062v4 (Hegazy, Alanwar, Elhoushi); ablations §5 + Appendix B/E; anchor ablation Table 11
- Plans 108 (LT2 loop), 136 (training-free loop / damped Euler), 428 (loop-stability PoC, `LoopStabilityMode`), 304 (GainCost halting); Research 414 (loop-stability gap table)
