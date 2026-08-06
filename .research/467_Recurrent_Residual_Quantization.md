# Research 467: Recurrent Residual Quantization (RRQ) — Single-Checkpoint Multi-Precision Weights

> **Source:** Yu Luo, Bo Dong, Wenhua Cheng, Haihao Shen (Intel). "Recurrent Residual Quantization: A Progressive Multi-Precision Representation for LLMs" [arXiv:2608.04048](https://arxiv.org/abs/2608.04048), Aug 2026.
> **Date:** 2026-08-06
> **Status:** Done — Gain (novel for our stack; no concrete consumer today; ship behind feature flag, default-off, re-evaluate when a multi-precision weight use case lands).
> **Related Research:** 020 (TurboQuant — our closest shipped quant codec, residual at activation level not weight), 065 (RotorQuant / PlanarQuant / IsoQuant — block-diagonal rotation quantization with QJL residual), 110 (Ciot — Plasma ternary SIMD + Cold tier Q4_K dequant-on-read), 159 (KVarN — variance-normalized KV), 200 (Quant outlier collapse + KS D-statistic detector — load-time outlier flag, RRQ's PMR selector is the natural sibling), 202 (QAT Infusion — selective layer precision, per-token activation scaling), 265 (b-posit — alternative bounded-range precision format), 418 (StreamDQ → SIMD LUT DeQuant — fused dequant+dot, the stage kernel fusion target), 439 (NVFP4 RL — Pass, format-specific to FP4 we don't ship), 463 (Moka quant_error_lora — the closest cousin: SVD-based low-rank compensation of the SAME `E = W − dequant(W_q)` error; RRQ is the additive-quantization-chain alternative)
> **Related Plans:** 568 (RRQ plan, this note), 100 (RotorQuant / PlanarQuant / IsoQuant — QJL residual correction reused as inspiration for stage-aware residual), 431/452 (SIMD LUT DeQuant — stage kernel substrate), 565 (quant_error_lora defend-wrong PoC — Moka negative result; RRQ's residual coding is the alternative mechanism that doesn't suffer the small-kernel parameter paradox)
> **PASS-Redirects (synthesis):** *n/a — Gain verdict, not Pass.*
> **Classification:** Public

---

## TL;DR

RRQ represents an LLM weight matrix as `W̃(t) = Ŵ0 + Σ_{k=1..t} R̂k` — a low-bit quantized base plus a sequence of quantized residual corrections. Each prefix of stages is a usable model at a distinct effective bit-width: 2-bit base → 2 bits, +1 stage → 4 bits, +2 stages → 6 bits, +3 stages → 8 bits. Construction is **calibration-free, all-RTN, no Hessian, no joint multi-bit optimization** (3.3× faster than MatGPTQ on Qwen3-8B). Inference exploits linearity: `A·W̃(t) = A·Ŵ0 + Σ A·R̂k` — every precision prefix is just a partial sum of stage GEMMs.

**Verdict: Gain — novel for our stack (zero prior art for multi-precision weight checkpoints), useful pattern with strong fusion hooks, but no concrete consumer today.** Our shipped quant codecs (TurboQuant, RotorQuant, KVarN, OCTOPUS, b-posit, hybrid_oct_pq, spectral) are all **KV-cache-oriented**; we ship **no multi-precision weight representation**, and the only "multi_precision" code in the stack is a *failed training-time LoRA experiment* (MPNS, negative arena result, now in riir-train). The natural consumer — per-NPC expert precision routing (`quant_expert_goat.rs`) — exists but each expert currently ships independently with no pain point forcing a single-checkpoint multi-precision solution. Ship behind feature flag `rrq_quant`, default-off, with a benchmark proving storage reduction + 4/6/8-bit quality parity, and revisit when (a) we serve a multi-precision LLM at runtime, (b) per-NPC expert routing wants to share a multi-precision base, or (c) a freeze/thaw versioning need emerges for incremental precision upgrades.

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitives, stripped of training setup (RRQ is PTQ — no training at all):

1. **Additive residual weight reconstruction** — `W̃(t) = Σ stages`. Pure inference math; each stage is a self-contained 2-bit quantized tensor with its own scale/zero-point. No nested bit layout (MatQuant/MatGPTQ MSB slicing); just `+`.
2. **Matmul linearity decomposition** — `A·W̃(t) = Σ A·(stage)`. Trivially true; the value is that each stage's GEMM can reuse the *same* SIMD dequant kernel (Plan 452 `simd_lut_dequant`) with a different scale, summed in registers.
3. **Peak-to-Mean Ratio (PMR) outlier threshold** — paper §3 derives a closed-form condition `K > r·(2^(n1+1) − 1)` for when a 2-stage RRQ beats direct fixed-bit quantization (K = outlier magnitude, r = inlier radius, n1 = first-stage bits). This is a **load-time modelless selector** that picks per-layer: RRQ vs direct quantization. Natural sibling to the KS D-statistic outlier detector (Research 200) — both flag outlier-heavy layers, but PMR is a *quantization-strategy selector* while KS is a *security anomaly detector*.

---

## 1. Paper Core Findings

### 1.1 The representation

Weights are stored as `S+1` quantized tensors: one base stage `Ŵ0` + `S` residual stages `R̂1..R̂S`. The k-th residual is the quantization error of the (k−1)-th reconstructed approximation:

```
Q_j^0   = Q_{b0}(x, z_0)              # base: 2-bit RTN
r_j^0   = x_j − x̂_j^0                 # residual = full − base dequant
Q_j^k   = Q_{bk}(r_j^{k-1}, z_k)      # stage k: 2-bit RTN on previous residual
r_j^k   = r_j^{k-1} − r̂_j^k           # next residual = prev residual − stage dequant
W̃(t)   = Ŵ0 + Σ_{k=1..t} R̂k           # prefix-t reconstruction (additive)
```

Default config: `b0 = b1 = b2 = b3 = 2` → 2/4/6/8-bit prefixes. **All stages use the same group size (128)** and the same per-stage quantizer (RTN). The paper also reports a stronger `SignRoundV2-base` variant — the *only* place a learned rounding operator enters; the all-RTN variant still tracks it within 0.1 Task Avg at 6/8 bits.

### 1.2 Outlier threshold analysis (the load-time selector)

For a uniform quantizer on a group with inlier radius `r` and one outlier of magnitude `K`:

| Quantity | Direct B-bit | RRQ (n1 base + n2 residual) |
|---|---|---|
| Step size | `(K+r)/(2^B − 1)` | base: as above; residual: `2r/(2^n2 − 1)` (if base captures K) |
| Mean abs error | `(K+r)/(4·(2^B−1))` | `r/(2·(2^n2−1))` |
| RRQ-better threshold | — | `K > r·(2·(2^B−1)/(2^n2−1) − 1)` ≈ `K > r·(2^(n1+1) − 1)` |

Concrete numbers (paper Table 3, B=4, r=0.5):
- 1-bit base + 3-bit residual: K > 3.29r (lowest threshold, but 1-bit base rarely useful as operating point)
- 2-bit base + 2-bit residual: K > 9r (the paper's default)
- 3-bit base + 1-bit residual: K > 29r (too much capacity in base, too little in residual)
- Direct 4-bit: N/A

The paper validates this with **Peak-to-Mean Ratio (PMR)** = `max|x| / mean|x|` per group. Qwen3-14B has mean group-max PMR 27.8 (severe outliers) → RRQ 4-bit is competitive with direct RTN. Llama-3.1-8B has max group-max K/MAE ≈ 64 but mean ≈ 26.5 (mild) → direct RTN wins at 4-bit. **PMR is a load-time, modelless, per-layer decision variable.**

### 1.3 Construction cost

Qwen3-8B on A100, all-RTN 2/4/6/8 package: **1293 s total** (412 s for 4 × 2-bit RTN passes, 383 s QDQ-model save + residual compute, rest I/O). vs MatGPTQ 4239 s → **3.3× faster**. No Hessian, no calibration data, no joint multi-bit objective. Each stage is independent — adding a precision prefix requires only "configure stage format", not "re-run joint optimization".

### 1.4 Quality at a glance (Qwen3-8B-B Task Avg, 5-task macro)

| Method | 2-bit | 4-bit | 6-bit | 8-bit |
|---|---|---|---|---|
| GPTQ | — | 73.62 | 73.62 | 73.50 |
| MatGPTQ | — | 73.02 | 73.56 | 73.27 |
| RRQ (RTN, all stages) | 61.58 | 72.07 | 73.27 | 73.39 |
| RRQ (SignRoundV2 base) | 65.73 | 72.84 | 72.92 | 73.53 |

**At 6/8 bits, near-BF16 parity.** At 4 bits, model-dependent — RRQ wins on outlier-heavy models (Qwen3-14B, Phi-3-Med), loses on flat-distribution models (Llama-3.1-8B). The standalone 2-bit operating point is genuinely useful only with the SignRoundV2 base (the gap shrinks dramatically once residual stages are added).

### 1.5 Package size (Appendix G)

| Package | Qwen3-8B | Phi-3-Med | Notes |
|---|---|---|---|
| Separate checkpoints (2+4+6+8 est) | 17.4 GB | 34.1 GB | 4× independent |
| RRQ (2+4+6+8 single package) | **7.3 GB** | **14.3 GB** | 2-bit base + 3× 2-bit residual |
| MatGPTQ (3+4+8) | 7.0 GB | 13.7 GB | 3 prefixes, nested bits |

RRQ is ~4–5% larger than MatGPTQ (per-stage scale overhead) but supports 4 prefixes vs 3, AND supports a standalone 2-bit operating point MatGPTQ cannot.

### 1.6 Prefill/decode split (Appendix B+C)

Because the GEMM decomposes linearly, prefill can run at high precision (sum all stages) while decode runs at low precision (base only) — same checkpoint. Qwen2.5-72B GSM8K: BF16-prefill + INT2-decode = 0.9045 (vs BF16/BF16 0.9037, vs INT2/INT2 0.8666). **The split-precision profile is a side-effect of the additive representation, not a separate mechanism.**

---

## 2. Distillation

### 2.1 What is genuinely new for our stack

Vocabulary translation (paper → codebase), grep BOTH sets:

| Paper term | Codebase equivalent (verified by grep) | New for us? |
|---|---|---|
| Residual quantization / residual correction | `quant_error_lora.rs` (SVD of `E = W − dequant(W_q)`); QJL residual in TurboQuant (activation level) | **YES at weight level as a quantization chain** — `quant_error_lora` is *one* correction layer (closed-form SVD), RRQ is *N iterated* corrections (each itself quantized). Different mechanism, same problem. |
| Multi-precision / single checkpoint | `multi_precision_npc` (riir-train, **FAILED** arena, negative result, training-time LoRA) | **YES modelless** — the failed cousin was model-based (dual-objective LoRA training); RRQ is pure PTQ with no training. Zero shipped modelless multi-precision weight representation. |
| Matryoshka / nested bit slicing | none — we don't ship MatQuant or MatGPTQ | **YES** (as alternative to it — RRQ explicitly replaces nested MSB slicing with additive residuals) |
| Outlier threshold / PMR | KS D-statistic detector (Research 200, shipped as load-time outlier flag) | **YES as quantization-strategy selector** — KS is a security anomaly flag; PMR is a "should this layer use RRQ or direct quant?" decision. Sibling, not duplicate. |
| Matmul decomposition `A·W̃ = Σ A·stage` | trivial linearity, no primitive exploits it | **YES as a tiered-dispatch primitive** — `simd_lut_dequant` (Plan 452) is the natural per-stage kernel; the sum is a register-level accumulation |
| Stage-wise codec reuse | none — every shipped codec is single-precision | **YES** — paper §5.4 explicitly notes heterogeneous stage formats are representationally supported (base = GPTQ, residual = RTN); we don't exploit this anywhere |

**Confirmed zero hits:** `residual quantiz`, `RRQ`, `recurrent residual`, `Matryoshka`, `MatGPTQ`, `MatQuant`, `PMR`, `peak_to_mean`, `prefix_precision`, `stage_dequant`, `additive_recon` — all return zero matches in shipped `*.rs`.

### 2.2 Closest shipped cousins

| Cousin | Repo / location | Relation |
|---|---|---|
| **`quant_error_lora.rs`** (Issue 565 / Research 463) | `katgpt-rs/crates/katgpt-core/src/quant_error_lora.rs` | **Strongest cousin.** Same problem (`E = W − dequant(W_q)`), different mechanism (closed-form SVD vs iterated RTN). Strategy A (weight-space SVD), Strategy B (output-space SVD), Strategy D (top-K sparse). RRQ is essentially Strategy R: "iteratively quantize E at 2-bit until exhausted". Failed PoC on Moka (small CNN, parameter paradox) — RRQ inherits the same small-kernel problem but is *cheaper per correction* (no SVD, just RTN). |
| **QJL residual in TurboQuant** (Research 020 / Plan 043) | `katgpt-rs/crates/katgpt-quant/src/turboquant/` | Residual at the *activation* level (1-bit QJL sketch on top of MSE quantizer for unbiased attention scores). Same "encode the error cheaply" philosophy; RRQ moves it to the *weight* level and iterates. |
| **QJL residual in RotorQuant** (Plan 100) | `katgpt-rs/crates/katgpt-quant/src/iso_quant/` | Same residual concept, KV cache target, single correction. |
| **`simd_lut_dequant`** (Plan 452 / Research 418) | `katgpt-rs/crates/katgpt-core/src/simd_lut_dequant.rs` | **The natural per-stage kernel substrate.** Each RRQ stage is a 2-bit RTN quantized tensor → dequant via the LUT path → sum in registers. The fused-dequant+dot variant is the natural "stage GEMM" primitive. |
| **KS D-statistic detector** (Research 200 / OAQG) | shipped as load-time outlier flag | Sibling: KS flags outlier-tampered layers for *security*; PMR flags outlier-heavy layers for *quantization-strategy selection*. Both run once at load, both compose. |
| **`quant_expert_goat.rs`** (per-expert precision routing) | `riir-ai/crates/riir-games/tests/` | The closest *consumer* — currently each expert ships at a fixed precision. A future RRQ-backed variant could share one multi-precision base across experts. |
| **`multi_precision_npc`** (FAILED, riir-train) | `riir-train/crates/riir-train-engine/src/multi_precision_npc.rs` | Negative-result cousin — model-based (LoRA training) attempt at multi-precision that collapsed in the arena. RRQ is the modelless PTQ alternative that doesn't touch training. |

### 2.3 Latent-space reframing (mandatory per workflow §1 step 3)

RRQ's residual chain is a **progressive reconstruction in weight space**. The latent-space reframing is straightforward: each stage's residual `r^k` lives in a progressively narrower subspace of the original weight manifold (the paper's analysis §3.4 — residual radius `B_{r,t}` shrinks stage by stage). The reconstruction `W̃(t) = Σ stages` is a **prefix-of-stages view of the weight**, exactly analogous to:

- **Matryoshka embeddings** — but for *weights* not activations; prefix-of-dimensions becomes prefix-of-stages
- **`CommittedFieldBlend`** (Plan 321) — `π · apply_blended(π_max)` where `π_max` is the contribution cap; RRQ's prefix-t is the depth-of-correction cap
- **Latent functor k-selector** (Plan 303) — choose k of N functors; RRQ chooses t of S stages

The reframing that gives RRQ a Super-GOAT angle would be: **freeze/thaw versioning of residual stages as independent shards** — each stage is its own `NeuronShard`, the prefix-t view is a runtime composition. This is the cleanest fusion into the freeze/thaw pillar (riir-neuron-db `MerkleFrozenEnvelope`). **But this is a fusion idea, not a Super-GOAT claim** — it needs a concrete consumer to justify the work, and the consumer does not exist today.

### 2.4 §3.5 modelless unblock protocol check (mandatory before any riir-train redirect)

**Is RRQ training-only?** NO — RRQ is explicitly **post-training quantization** (PTQ). The paper's main result uses RTN for all stages; the only "training" is the optional SignRoundV2 base (a learned rounding operator, applied at quantization time, not as a training loop). The §3.5 check is moot — there's nothing to redirect to riir-train. **RRQ is modelless by construction.**

**Path 0 (training-target decomposition):** not applicable — RRQ has no training target to decompose. It's pure inference-time representation.

**Dual-track contribution:** RRQ contributes only to the **modelless track**. No riir-train follow-up is implied by this paper. (A *separate* question — "can SignRoundV2-style learned rounding be re-implemented modellessly via freeze/thaw or latent correction?" — is out of scope for this note; RRQ's all-RTN result shows the learned base is not load-bearing for the headline 6/8-bit parity.)

---

## 3. Fusion (the Super-GOAT hunt — novelty TBD, not committed)

### 3.1 RRQ × `simd_lut_dequant` — fused multi-stage dequant+dot kernel

Each RRQ stage is a 2-bit RTN quantized tensor. The natural kernel: **one fused dequant+dot per stage, summed in registers**. This is a direct extension of `dequant_dot_via_lut` (Plan 452) to a 4-stage sum. The LUT is per-stage (each stage has its own scale/zero-point), the accumulator is shared. **Hypothesis:** at 8-bit prefix, the 4-stage LUT path is at parity with a single 8-bit LUT path because the LUT cost is amortized across the same SIMD gather, only the scale+sum differs. Untestable without the kernel — tracked as Plan 568 Phase 3.

### 3.2 RRQ × KS D-statistic — load-time layer-strategy router

Run KS D-statistic (Research 200 OAQG) AND PMR (paper §3) at model load. KS detects tampered/compromised layers; PMR selects quantization strategy:

- PMR > 9r AND KS < 0.1 → RRQ (outlier-heavy, untampered)
- PMR < 9r AND KS < 0.1 → direct 4-bit RTN (flat distribution, no benefit from RRQ)
- KS > 0.25 → flag for security review regardless of quantization choice

This is the **load-time quant-strategy router** — a per-layer dispatch table built once at model load. Composes with existing `quant_expert_goat.rs` per-expert routing at the layer depth.

### 3.3 RRQ × `quant_error_lora` — iterated quantization as an alternative to SVD correction

`quant_error_lora` (Research 463) approximates `E = W − dequant(W_q)` as `A·B` (rank-r SVD) or as top-K sparse. RRQ approximates the same `E` as `Σ R̂k` where each `R̂k` is itself 2-bit quantized. **The two are dual representations of the same residual.** Hybrid: SVD for the dominant low-rank structure + RRQ for the high-frequency residual of the SVD approximation. This is unexplored and would need a PoC to settle.

### 3.4 RRQ × freeze/thaw — versioned residual stages as independent shards (the Super-GOAT angle)

Each RRQ stage is a self-contained `NeuronShard` candidate. The prefix-t view is a runtime composition of t shards. Implications:

- **Incremental precision upgrades** — ship a 4-bit checkpoint; users who want 6-bit thaw one more shard, no re-download of the base. This is a genuine product capability (chain commitment of incremental precision = the canonical RTDC use case from Research 280).
- **Per-NPC personality divergence via stage selection** — different NPCs thaw different numbers of residual stages, producing genuine behavioral divergence from one shared base. (Sibling to `CommittedFieldBlend`'s π_max cap, but on the precision axis instead of the contribution axis.)
- **Cross-tier transport** — Plasma tier (2-bit base only, hot path) → Hot tier (+1 stage, 4-bit) → Warm tier (+2 stages, 6-bit). Same checkpoint, three tiers, the tier transition is "include one more shard in the sum".

**This is a Super-GOAT candidate IF a consumer needs incremental precision upgrades.** Today no consumer does — `quant_expert_goat.rs` ships each expert at a fixed precision, the per-NPC personality divergence story is handled by `CommittedFieldBlend` (a different axis), and our LLM serving uses Kimi-K3 as a dev fixture (not multi-precision production). **Without a consumer, this stays a fusion idea, not a committed Super-GOAT.** Re-evaluate when one of those consumers materializes.

---

## 4. Verdict

**Gain.** Ship behind feature flag `rrq_quant`, default-off, with a benchmark proving (a) storage reduction vs separate checkpoints, (b) 6/8-bit quality parity with direct quantization on a fixture, (c) PMR-based load-time selector correctly classifies Llama vs Qwen outlier profiles. Promote to default-on only if a concrete consumer lands (multi-precision LLM serving, per-NPC expert base sharing, or incremental precision upgrade via freeze/thaw).

**Reasoning:**

- **Q1 (no prior art):** PASS — zero shipped multi-precision weight representation. Closest cousins (`quant_error_lora`, QJL residual, MPNS) cover related but distinct problems.
- **Q2 (new class of behavior):** PARTIAL PASS — "single checkpoint → multiple precisions at runtime" is a new capability for our stack, but it's a *refinement of an existing capability* (we can already quantize to any bit-width), not a new capability class. The genuinely new class would be §3.4 (incremental precision upgrades via freeze/thaw), but that needs a consumer.
- **Q3 (product selling point):** FAIL today — "multi-precision weight checkpoint" is not a customer-facing selling point when we don't serve multi-precision LLMs in production. The selling point emerges only with §3.4's incremental-upgrade fusion.
- **Q4 (force multiplier across pillars):** PARTIAL — connects to quant codecs (could feed RRQ stages from TurboQuant/OCTOPUS), SIMD LUT DeQuant (Plan 452), KS detector (Research 200), freeze/thaw (riir-neuron-db `MerkleFrozenEnvelope`). But none of these connections is load-bearing for an existing pillar today.

**Not Super-GOAT.** The honest verdict is Gain — novel mechanism with strong fusion hooks and no current consumer. Q3 and Q4 fail; the Super-GOAT angle (§3.4) is contingent on a future consumer.

**MOAT gate (per domain §1.6):** `katgpt-rs` is the correct repo — generic modelless inference primitive (weight quantization math), no game/chain/shard IP. A future Super-GOAT promotion (if §3.4 lands) would split: open primitive stays in `katgpt-rs`, the freeze/thaw versioning guide goes to `riir-neuron-db/.research/` (shard residency), the per-NPC personality divergence guide goes to `riir-ai/.research/` (game runtime). For now, all in `katgpt-rs`.

**§3.6 defend-wrong PoC:** NOT REQUIRED — verdict is Gain with explicit "no concrete consumer today" caveat, no parity claim against any shipped primitive (the closest cousin `quant_error_lora` was already PoC-refuted for the Moka small-kernel case in Research 463). If Plan 568 lands and a future session claims RRQ achieves parity with direct quantization on a real model, THAT claim needs a PoC.

---

## 5. What we take (and what we don't)

| Take | Target | Type | Status |
|---|---|---|---|
| Additive residual weight representation (`W̃ = Σ stages`) | katgpt-rs | Modelless inference primitive | Plan 568 Phase 1 |
| PMR-based load-time quant-strategy selector | katgpt-rs | Modelless infra | Plan 568 Phase 2 |
| Fused multi-stage LUT dequant+dot kernel | katgpt-rs | SIMD kernel (extends Plan 452) | Plan 568 Phase 3 |
| Prefix-t view as runtime precision tier | katgpt-rs | Tier dispatch | Plan 568 Phase 4 (stretch) |

| Don't take | Why not |
|---|---|
| SignRoundV2 learned base | Training-method artifact (learned rounding operator); the all-RTN variant is the modelless path and tracks SignRoundV2-base within 0.1 Task Avg at 6/8 bits. |
| GPTQ / AWQ / OmniQuant stage variants | Already explored in our stack via other codecs; the paper §5.4 says heterogeneous stage formats are representationally supported but doesn't empirically evaluate them. Speculative. |
| Matryoshka / MatGPTQ nested bit slicing | RRQ explicitly *replaces* this with additive residuals. We don't ship MatQuant/MatGPTQ; RRQ is the additive alternative if we ever need multi-precision. |
| Hardware-specific NVFP4 / FP4 stages | Format-specific to NVIDIA Blackwell tensor cores. We're CPU/SIMD/ANE. Already verdicted Pass in Research 439. |

---

## 6. Implementation priority (in Plan 568)

| Phase | Scope | Gate | Priority |
|---|---|---|---|
| 1 | Skeleton: `RrqWeights` struct (base + N residual stages, each `Vec<u8>` codes + per-stage scale/zero), `prefix_reconstruct_into(t, &mut [f32])`, `prefix_dot_into(t, &[f32], &mut [f32])` | G1 bit-exact reconstruction matches reference (sum of stages); G4 alloc-free hot path | P0 |
| 2 | Load-time PMR + KS selector: `RrqStrategyRouter` decides per-layer RRQ vs direct quant | G1 selector classifies Llama vs Qwen correctly (synthetic + real fixture) | P1 |
| 3 | Fused 4-stage LUT dequant+dot kernel (extends `simd_lut_dequant`) | G2 latency at parity with single-stage 8-bit LUT path at 8-bit prefix | P2 |
| 4 | Stretch — prefix-t as tier dispatch (Plasma=2-bit, Hot=4-bit, Warm=6-bit) | G3 no-regression on existing tier tests | P3 (deferred until consumer) |

**Default-off rationale:** no concrete consumer today. The benchmark in Phase 1 proves the primitive works; promotion to default-on waits for a consumer (multi-precision LLM serving, per-NPC expert base sharing, or incremental precision upgrade via freeze/thaw).

---

## TL;DR (final)

RRQ is a clean, calibration-free, modelless PTQ framework that produces multiple precision prefixes from a single additive-residual weight checkpoint. It's genuinely novel for our stack (zero shipped multi-precision weight representation; the failed MPNS cousin was model-based). The math decomposes into three transferable primitives: additive residual reconstruction, matmul linearity for stage-wise GEMM, and PMR-based load-time quant-strategy selection. **Verdict: Gain** — ship behind `rrq_quant` feature flag with a benchmark, default-off, revisit when a consumer materializes. The Super-GOAT angle (incremental precision upgrades via freeze/thaw, §3.4) is a fusion idea tracked in Plan 568 Phase 4, not a committed claim.
