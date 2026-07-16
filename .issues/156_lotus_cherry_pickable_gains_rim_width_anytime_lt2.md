# Issue 156 — LOTUS Cherry-Pickable Gains (RiM Width + Any-Time LT2 Validation)

**Filed:** 2026-07-16
**Priority:** P2 (Gain — two actionable items + two noted fusions; one is a config default audit, one is a PoC validation)
**Origin:** Research 442 (LOTUS, arxiv 2606.31779) — initial Pass verdict was correct on prior-art but lazy on value extraction. The §1.55 value-extraction scan (added to the research skill this session) surfaced these cherry-pickable gains. See `katgpt-rs/.research/442_LOTUS_Looped_Parallel_CoT_Supervision_PASS.md` §2 "Cherry-pickable gains" for the full analysis.

**Related:**
- `.research/442_LOTUS_Looped_Parallel_CoT_Supervision_PASS.md` (parent — PASS-with-gains note)
- `.research/073_LT2_Linear_Time_Looped_Transformers.md` + `.plans/108_lt2_looped_inference_pipeline.md` (LT2 — shipped, DEFAULT-ON, GOAT 8/8)
- `.plans/172_rim_reasoning_buffer_slots.md` (RiM slots — shipped, DEFAULT-ON feature, zero-decode-cost GOAT)
- `.research/273_ELT_Elastic_Looped_Transformers_Any_Time_Inference.md` (ELT — Gain verdict, Any-Time claim unvalidated)
- `.plans/428_loop_stability_poc.md` (loop stability — addresses the residual explosion LOTUS's `L_step` exposes)

## The gains

Four cherry-pickable gains surfaced from the LOTUS §1.55 scan. Two are actionable (T1, T2); two are noted fusions for future reference (F1, F2).

### T1 — RiM slot width: document the M≥5 floor for reasoning tasks (config audit)

**The claim:** LOTUS Table 7 sweeps per-block width `c ∈ {1, 5, 10, 25, 30}` at K=6 fixed:

| c (tokens/block) | Total latent positions (Kc) | GSM8K acc (%) |
|---|---|---|
| 1 | 6 | 49.7 |
| 5 | 30 | 67.5 |
| 10 | 60 | 68.4 |
| 25 | 150 | 70.0 |
| 30 | 180 | 70.0 |

There is a **cliff at c=1→5** (+17.8pp) and **saturation at c≥25**. Our `Config::rim_tokens_per_block` default is **M=2** (from the RiM paper, which was about pause tokens, not reasoning). M=2 is in the cliff regime — if a future caller enables RiM for reasoning tasks at M=2, they're leaving ~15pp of quality on the floor vs M=5.

**Current state (verified this session):** all `Config` presets ship with `rim_block_count: 0` (RiM disabled). The feature flag `rim_slots` is DEFAULT-ON but no preset turns the field on. So this is NOT a current misconfiguration — it's a **latent trap for future callers**. The M=2 default is fine for the RiM paper's pause-token use case; it's wrong for the LOTUS-style reasoning use case.

**Action:** document the M≥5 floor in the `rim_tokens_per_block` doc comment, and add a `rim_tokens_per_block_reasoning` preset helper (or a doc note pointing callers to M≥5 for reasoning tasks). No default change needed since no preset enables RiM today.

### T2 — Any-Time LT2 validation: prove `loop_count` is elastic at inference (PoC)

**The claim:** LOTUS Table 6 sweeps inference-time loop count R ∈ {1..7} on a model trained at R=6:

| R (inference) | GSM8K acc (%) |
|---|---|
| 1 | 22.7 |
| 2 | 40.0 |
| 3 | 55.0 |
| 4 | 63.5 |
| 5 | 68.7 |
| 6 | 70.0 (trained) |
| 7 | 69.3 |

Accuracy climbs monotonically to the trained R=6, then dips slightly at R=7. This is the **Any-Time inference** property: one trained artifact serves any compute budget R ∈ [R_min, R_max].

**Our gap:** Research 273 ELT §2.3 claimed this as a Gain-tier property of our LT2 (`LoopMode::WeightShared { loop_count }` is per-Config; per-dispatch elastic L is a small missing piece). But we **never validated** that our LT2 actually exhibits the Any-Time property — we only claimed it by architectural analogy. LOTUS provides the empirical template; we need to run it on our stack.

**Action:** PoC at `/tmp/issue156_anytime_lt2/` (standalone, `CARGO_TARGET_DIR=/tmp/issue156_anytime_lt2/target`, clean up when done). Three competitors minimum:
1. **Baseline** — single-pass (no loop), R=1.
2. **Trained R_max** — LT2 with `loop_count = R_max` (the static default).
3. **Elastic R** — same artifact, run at R ∈ {1, R_max/2, R_max} at inference.

**Measurement:** output stability (logit KL divergence across R values) + latency per R. The Any-Time property holds iff the elastic-R outputs are monotonically more stable as R → R_max AND latency scales linearly with R (no superlinear cost).

**Verdict rule:** if elastic-R output is chaotic (no monotonic stability), the Any-Time claim FAILS for our LT2 — Research 273 ELT's Gain verdict needs revision. If it holds, we have empirical evidence to open a plan for per-dispatch elastic `loop_count` driven by existing budget signals (`ReestimationScheduler::set_active_budget`, per-NPC tier).

## Noted fusions (no code change, recorded for future reference)

### F1 — PCL factorization as a modelless design principle for screening composition

LOTUS §3.3.2 introduces the **Parallel Chain Likelihood (PCL)**: the step loss factorizes the chain as `∏_p(Tᵢⱼ|Q)` — conditionally independent readouts — but the latent states are jointly computed via shared looped computation. Two complementary roles: `L_step` provides support coverage (per-position mass on right tokens); `L_ans` provides global joint selection (answer is decoded from the jointly computed latent configuration).

**Transferable inference principle (modelless):** when you have a jointly-computed latent workspace (LT2+RiM loops), apply `ConstraintPruner` / `ScreeningPruner` **per-position independently** — don't couple them to global state. The joint coherence comes from the looped substrate, not from the screener. This is a correctness-preserving simplification of our `CLR vote` × `SalienceTriGate` × `BoMSampler` composition on top of RiM slots.

**Why noted, not actioned:** this is a design principle, not a code change. It informs how future screening compositions should be structured on top of LT2+RiM, but our current screening stack doesn't currently violate it (we don't couple per-position screeners to global HLA state in a way the PCL lens would flag). Recorded in Research 442 §2.3 for future reference.

### F2 — Per-iteration vs post-loop readout schedule for BoMSampler vs CLR

LOTUS found: auxiliary decoder (LOTUS-aux) works best reading `h^(t)` at each iteration `t` (per-iteration readout, shorter gradient path); direct LM-head readout (LOTUS) works best reading only `h^(R)` post-loop (lets early blocks refine fully).

**Transferable composition schedule:**
- **BoMSampler** (Plan 281 — auxiliary, samples K hypotheses) → read **per-iteration** (shorter path, more diverse hypotheses across R depths).
- **CLR vote** (riir-ai Plan 316 — direct, votes on the final action) → read **post-loop only** (let latents fully refine before voting).

Currently both read post-loop. This is a tunable composition gain for a future plan that wires BoMSampler to consume LT2+RiM latents.

**Why noted, not actioned:** requires a plan that fuses BoMSampler with LT2+RiM readout, which doesn't exist yet. Recorded here + in Research 442 §2.3 for the future fusion plan.

## What is NOT cherry-pickable (the honest gap)

**LOTUS `L_step` supervision recipe** — genuinely requires training (gradient descent through R iterations with gold CoT data). §3.5 modelless paths all fail:
1. Freeze/thaw — N/A (no systematic bias correctable by snapshot; requires gold-token supervision signal).
2. Raw/lora hot-swap — N/A (no deterministic construction aligns latent positions to gold CoT tokens).
3. Latent-space correction — N/A (the "correction" IS the gold CoT data, which is data not a projection).

→ riir-train. This is the one piece of LOTUS we genuinely lack and cannot recover modellessly.

## Tasks

- [x] **T1** Audit `rim_tokens_per_block` doc comment — add the M≥5 floor note for reasoning tasks. **DONE 2026-07-16** (`crates/katgpt-types/src/config.rs` L158-167). Doc-only; cargo clippy clean.
- [ ] **T2** Run the Any-Time LT2 PoC at `/tmp/issue156_anytime_lt2/` — record raw numbers below. If Any-Time property holds → open plan for per-dispatch elastic `loop_count`. If it fails → revise Research 273 ELT's Gain claim.
- [ ] **F1** (noted, no action) PCL design principle — referenced from Research 442 §2.3 for future screening-composition plans.
- [ ] **F2** (noted, no action) Per-iter vs post-loop readout schedule — referenced from Research 442 §2.3 for future BoMSampler × LT2 fusion plan.

## PoC Results (T2)

_To be filled in after T2 runs._

---

## TL;DR

LOTUS (arxiv 2606.31779) initial verdict was Pass — architecture shipped as LT2+RiM, training → riir-train. The §1.55 value-extraction scan (added to the research skill this session) surfaced four cherry-pickable gains. Two are actionable: T1 (RiM M≥5 floor doc — low-risk doc audit) and T2 (Any-Time LT2 validation PoC — closes Research 273 ELT's unvalidated Gain claim). Two are noted fusions: F1 (PCL design principle for screening composition) and F2 (per-iter vs post-loop readout schedule for BoMSampler vs CLR). The `L_step` supervision recipe itself is the one piece that genuinely needs riir-train.
