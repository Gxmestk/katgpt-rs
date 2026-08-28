# Behavior Before Perplexity — BTL-3 Compact 27B AVQ2 Recipe (Behavior-Gated Compression)

**Paper:** Bad Theory Labs, "Behavior Before Perplexity: A failure-driven recipe for compressing a 27B agentic model to 8.39 GB", July 2026. https://www.badtheorylabs.com/papers/behavior-before-perplexity/paper.pdf (not on arXiv). Notably cites PrismML Bonsai 27B as ref [12] — the exact model family of our preferred `riir-train/data/Ternary-Bonsai-27B-Q2_0.gguf`.

**Status:** DISTILLED → **MEASURED 2026-08-28 (PoC RESOLVED — [riir-ai Bench 776](../../riir-ai/.benchmarks/776_issue750_behavior_gate_t1_t2.md))**: the behavior-first gate + nested-prefix bisection ran end-to-end on gemma-2-2b Q4_K — **first behavior flip at prefix k=1** (layer 0 alone flips the sealed 66-item subset; restoring it costs 106.7 MiB, priced by the override probe); the lossy-surface promotion rule (behavior per-family, not bit-identity/ppl alone) is now adopted in katgpt-rs AGENTS.md §Feature Flag Discipline, cross-linking Research 502 + Bench 696 + Bench 776 as the three independent arrivals.

**TL;DR:** A lab compresses a Qwen3.6-27B-derived agentic model (BTL-3) to 8.39 GB (≈2.0–2.1 effective bits/weight) and documents that **every proxy metric failed as a promotion gate**: candidates with acceptable perplexity / KL / teacher-forced token agreement / reconstruction loss could not make a single valid tool call (0.8B ternary: 0/7 tools at 1.7177 bits; 4B magnitude-recovery: 34% token agreement, 0% valid calls). The fix was methodological — *change the unit of optimization from a tensor to a deployed behavior*: nested-prefix behavioral bisection, behavior-proven BF16 precision islands priced in bytes, a rank-8 LoRA correction trained on the exact packed (quantized) forward graph, selectively rescued embedding rows, a rank-32 closed-form-style output-head residual, and a sealed 100-turn tool-contract gate authored only after everything was frozen (92.2% conditional retention; 100% on 4 of 5 behavior families, 30% on parallel-multiple — reported honestly). The stack-relevant transfer is **not** the quantization math (UniSVQ owns that) but the **promotion protocol**: lossy compression may only be promoted by free-running behavior on the exact artifact, with per-family conditional retention — the same lesson we learned independently from the opposite end in Issue 717 / Bench 696 ("the corpus metric mis-measures target-conditioned drafters").

## What the paper establishes

1. **Proxy-metric blindness, with negative controls.** Perplexity, KL, token agreement, and reconstruction error stayed useful as *diagnostics* but none could *promote* a candidate. Most instructive artifact: a 4B PTQ-recovery model earned "accidental credit on irrelevance" — a model that cannot call a tool also cannot make an *irrelevant* call, so precision-style metrics inflate for exactly the most broken models. 60% of its generations failed to stop.
2. **The recipe (ordered; changing the order changes the method):** freeze source + family-disjoint splits → sequential **packed-prefix calibration** (each layer calibrated on FP64 input second moments captured through the *already-quantized* prefix, never clean FP inputs) → 128-wide seeded Hadamard + **4-weight affine-lattice codes with block-LDLQ error propagation** + scale search {0.8, 1.0, 1.2} → gated MLP (up/gate/down) fitted as one functional unit → **behavioral bisection**: replay nested packed prefixes until a behavior first flips, override one module/group, keep the override only if the *same emitted behavior* returns, price it in physical bytes → BF16 islands placed only where an intervention was behavior-proven (never by a rule like "attention 4-bit, MLP 2-bit") → rank-8 LoRA (α16, 100 AdamW steps, layers 40–63, four module families) trained **on the frozen packed graph**, composed at runtime not merged → embedding quantized with 4,096 rescued rows (structural tokens first, then frequency) → rank-32 activation-weighted head residual `W ≈ Q + UVᵀ` → **byte ledger solved exactly**; when over budget, demote only previously *measured* islands → sealed gate authored post-freeze.
3. **Result:** 8,392,369,600-byte GGUF, 2,416 byte-verified tensors, loader asserts no dense decoder matrix survives. Teacher 90/100 on the sealed gate; compact retains 83/90 = 92.2% conditional. Per-family: single/parallel/sequential/abstention 20/20 each; **parallel-multiple 3/10 (30%)** — the weak cell is reported, not hidden. All generations stopped; the 7 malformed outputs were "fluent over-abstention", not parser artifacts.
4. **Byte-exchange discipline:** two BF16 islands were demoted to INT4 *only after* their recovery had been measured, recovering 123.86 MB. The precision map is "a record of observed cliffs, not a general claim about Qwen sensitivity."

## Prior-art check (novelty gate Q1 — fails; the paper is honest about this)

| Claim | Verdict | Prior art |
|---|---|---|
| Affine-lattice 2-bit vector core (AVQ2) | Not new — the paper says so | UniSVQ arXiv:2606.10520 (affine-lattice codewords; ICML'26 code released); VPTQ, QTIP, additive quantization lineage |
| Curvature-weighted assignment + propagated error | Not new | GPTQ 2210.17323, SparseGPT 2301.00774 |
| Hadamard incoherence | Not new | QTIP et al. — and **ships in our stack** (`katgpt-kv/src/kvarn/hadamard.rs`, Research 159) |
| "Compression breaks agentic behavior while ppl looks fine" (the thesis) | Not new as a *finding* | "Can Compressed LLMs Truly Act?" arXiv:2505.19433 (ICML 2025, ACBench — function calling/embodied/problem-solving degradation under quantization+pruning); "Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression" (EMNLP-F 2024) |
| Low-rank quantization-error residual / quant-aware adapter | Not new | LQ-LoRA 2311.12023, QLoRA; our riir-train/.research/087 QAT-LoRA fusion |
| **The promotion protocol**: behavior as the unit of optimization DURING compression (bisection → byte-priced behavior-proven islands → correction on the packed graph → sealed post-freeze gate → exact ledger) | New as a *combination* (their claim, credible — no prior paper promotes/demotes precision by emitted-behavior deltas priced in bytes) | — |

Super-GOAT gate: Q1 ✗ (published prior art on both core and thesis), Q2 ✗ (no new game-behavior class; this is compression QA), Q3 ~ (moderate local-inference product story), Q4 ✓ (connects quant kernels, Bonsai, GOAT discipline). → **Gain, not Super-GOAT.**

## Path 0 decomposition (§3.5 — training-target split)

| Component | Extractable without GD? | Ships here? | Disposition |
|---|---|---|---|
| AVQ2 codes + block-LDLQ + scale search | Yes (combinatorial assignment) | No (nearest: ternary GEMV kernels, Q4_K dequant) | riir-train/gpu pipeline candidate — see Path 0.5 below |
| Seeded Hadamard incoherence | Yes | **Yes** — `kvarn/hadamard.rs` | Covered |
| FP64 second-moment curvature calibration | Yes (moment accumulation) | **Partial** — SpectralQuant `calibration.rs` (covariance + eigenbasis rotation, bench-only); `covmatch_second_moment_into` (different purpose) | Signal-diff: SpectralQuant rotates for outlier suppression; BTL fits *codes* under curvature — different consumers, shared substrate |
| Packed-prefix sequential calibration | Yes | No as a calibration rule; **the principle ships as eval discipline** (Issue 718: gate the deployed handle path, not the idealized one) | Adopt as a written rule |
| Nested-prefix behavioral bisection | Yes (pure procedure) | No (`prefix_replay` in riir-train = trajectory-collection window shifting, Plan 348 — name collision only) | **Open — Issue 750 T2** |
| Behavior-proven precision islands | Yes (search procedure) | **Partial** — q8kv f32 sink sidecar (Issue 716) = position-granularity rescue; `norms_only` (Issue 720) = live-fields-only; no behavior-driven placement | **Open — Issue 750 (protocol)** |
| Gated-MLP joint 3-matrix fitting | Yes | No (fused GeGLU kernels exist; joint *quantization fitting* doesn't) | Pipeline candidate only |
| Rank-8 behavior LoRA on packed graph | No (GD, but bounded: 100 steps) | **Partial** — QAT-LoRA fusion plans (riir-train 087) cover adapter-aware-of-quantization; the delta is "train on the REAL packed path, not STE simulation" | Fold into 087-lineage plans when next opened |
| Rank-32 head residual `W ≈ Q + UVᵀ` (activation-weighted) | **Yes** — closed-form SVD/least-squares of the error under activations; no GD required | No (grep-verified: no `head_residual`/rank-k quant-error correction) | **Open primitive candidate** — deterministic-construction LoRA (constraint #3 compliant); no consumer today (we consume GGUFs, we don't produce quantized weights) → deferred with reason |
| Embedding row rescue (structural + frequency) | Yes (selection rule) | **Partial** — sink-*position* rescue ships; row-granularity doesn't | Notes into Issue 716 lineage |
| Sealed behavior gates (post-freeze, per-family, conditional retention) | Yes (eval protocol) | **Partial** — GOAT gates are correctness-first; Bench 696 found the deployment-metric principle independently; cargo-heal `score_bench` is already behavior-first for healing | **Open — Issue 750 T1** |
| Exact byte ledger + demote-only-measured-islands | Yes (bookkeeping) | **Partial** — Batch 56 `superseded-format-constructor-live-fields-only` + Issue 720 dead-weight tripwires are the same family | Covered as discipline |

**Path 0.5 disposition (training-efficiency re-evaluation):** building our own AVQ2 PTQ pipeline end-to-end is **deferred with reason, not lazily redirected** — the ~2-bit 27B deployment slot is already occupied by born-ternary `Ternary-Bonsai-27B-Q2_0.gguf` (user-designated preferred model), and the paper's own negative controls (PTQ ternary at 1.72 bits: 0/7 tools; PTQ scalar 2-bit pair: 27/63) are evidence **for** the born-ternary branch we already chose and **against** PTQ-ing dense checkpoints for agentic use. Reopen trigger: a dense-only checkpoint becomes a required local deployment target (no ternary/QAT edition available) — then the recipe above is the plan skeleton, with UniSVQ's released code as the fitting core.

## Distillation — what transfers to this stack

### 1. Behavior-gated promotion for anything lossy (the headline transfer)

Protocol, extracted: (a) **promotion requires free-running behavior on the exact artifact** — ppl/KL/bit-agreement stay as diagnostics; (b) **the gate is sealed after freeze** — authored once the candidate is fixed, on a fresh first-party task family, so dev-gate overfitting cannot promote garbage; (c) **per-family breakdown** — aggregate retention hid a 30% parallel-multiple cliff behind four 100% families; (d) **conditional retention** — score only on turns the full-precision reference solved, so neither reference failures nor compact flukes pollute the number; (e) **watch the irrelevance-credit artifact** — a system that cannot act also cannot act wrongly; precision-style metrics inflate for the most broken candidates (our healer already defends this axis: `score_bench` reports heal_rate AND created_rate separately, so a propose-nothing healer cannot win).

We discovered (a) independently and late — Issue 717 / Bench 696: "on TRUE acceptance (vs the target's own greedy — the deployment metric) the trained drafter wins decisively; on the corpus-proxy protocol the modelless table wins — the corpus metric mis-measures target-conditioned drafters", and Bench 694's chain-seam wall-clock (0.326× at acceptance 0.000 — the proxy said fine, the deployment metric said dead). This paper generalizes those finds into a standing rule: **every lossy promotion (quant format, drafter, compressed artifact) gates on the deployed metric, per family, conditionally.**

### 2. Nested-prefix behavioral bisection (diagnostic harness — open)

When a quantized model misbehaves: replay nested packed prefixes (layers 0..k quantized) until a behavior first flips; bisect the flip to one module or interacting module group by override; keep an override only if the *same emitted behavior* returns; price every intervention in physical bytes. Two paper findings make this non-obvious: (i) the **module with worse token metrics was the behaviorally correct override** (MLP beat attention at prefix 9 — aggregate metrics mislocalize, behavior localizes); (ii) after installing a fix you must **recapture downstream activations through the new packed prefix** (the inputs every later layer actually sees). Consumer here: the next KVarN/Q8KV/sink-guard re-gate, or diagnosing a misbehaving GGUF (Bonsai/Kimi/Gemma) at the layer level.

### 3. Closed-form rank-k quantization residual (open primitive candidate, no consumer)

`W ≈ Q + UVᵀ` with U,V fitted by activation-weighted SVD/least-squares of the error matrix — deterministic, no gradient descent, exactly constraint #3's "deterministically constructed LoRA overlay". Distinct from riir-train 087 (QAT trains the adapter; this *solves* it). No consumer ships (we consume quantized GGUFs; we don't produce weight-quantized artifacts; KV-cache quant error is dynamic, not a static weight) → filed as a deferred task in Issue 750, reopen when we produce our own quantized weights.

### 4. Byte-priced rescue ledger (generalizes two shipped patterns)

Issue 716's f32 sink sidecar (position-granularity rescue) and Issue 720's `norms_only` (live-fields-only constructor) are both members of the paper's general pattern: **a rescue budget ledger where every precision promotion/demotion is individually measured in behavior and priced in bytes, and budget overruns are resolved by demoting only previously-measured islands** (their two BF16→INT4 demotions recovered 123.86 MB with behavior re-proven). The unifying discipline for future lossy work: never allocate precision by rule ("attention 4-bit, MLP 2-bit"); allocate by measured behavioral cliff, and keep the ledger exact.

### 5. Born-ternary validation (zero action, moat line)

The paper's negative controls are the strongest published argument yet for our Bonsai branch: PTQ ternary/vector routes needed the entire behavior-bisection apparatus to reach 92.2% tool retention, while born-ternary (trained-ternary) Bonsai ships at the same size class without the rescue machinery. One-line positioning: *PTQ extreme compression spends its precision budget re-buying behavior the dense model already had; born-ternary never sells it.*

### 6. Packed-prefix calibration ≈ deployed-path validation (rule, not code)

"Calibrating every layer from full-precision hidden states was rejected because later layers would be fitted to inputs they never receive at inference time" — the calibration twin of our Issue 718 lesson (the handle path lacked the delta_routing fallback the forward path had: gate the path that ships, not the path that's easy to run). Write it once, apply everywhere: **calibrate, gate, and verify on the exact deployed artifact and path.**

## Fusion

- **× Research 487 + Issue 716 (sink-aware KV quant):** 487's f32 sink sidecar is a *position*-granularity rescue with a byte cost; this paper adds *row* (embedding) and *module* (BF16 island) granularities plus the ledger that prices them together. Fusion candidate: one **rescue-budget ledger type** (position | row | module entries, each `{bytes, behavior_proof}`) that any future lossy-quant GOAT reports — makes "spend precision only where an intervention works" checkable in review.
- **× riir-train/.research/087 (QAT-LoRA fusion):** 087's adapters train against simulated quantization (STE); BTL trains the rank-8 correction against the **real packed graph** with every quantized weight frozen. When a 087-lineage plan next opens, add the real-packed-path variant as the default with STE as the cheap approximation — same shape as our "bench the deployed path" rule.
- **× Issue 717 / Bench 696 + cargo-heal `score_bench`:** three independent arrivals at deployment-metric-over-proxy from three directions (speculative drafting, code healing, model compression). Fusion: elevate it to a named GOAT-gate rule for **lossy** promotions — G1 for lossy surfaces is the behavior gate; bit-identity is only available for lossless ones.
- **× Batch 56 / Issue 720 (dead-weight discipline):** their exporter "asserts that no compatible dense decoder matrix survives" is our zero-size-placeholder tripwire pattern applied at artifact scale — fold into the artifact-proof step of any future compression work.

## Routing / GOAT verdict

- **Gain.** Note here (katgpt-rs — consistent with the quantization cluster 110/159/200/418/439/487); actionable PoC as `riir-ai/.issues/750` (behavior-gate + bisection harness on an in-tree small model); closed-form residual deferred-with-reason inside the same issue; full AVQ2 pipeline deferred-with-reason (Bonsai occupies the slot, see Path 0.5).
- If Issue 750 T1/T2 land and a dense-only deployment target appears, the pipeline plan belongs in `riir-train/.plans/` per Path 0.5, consuming UniSVQ's released fitting core; kernels land in riir-gpu beside the existing Q4_K/Q8KV/ternary family.
- Latent/raw boundary: untouched — this is inference-asset QA; no sync-surface implications.
