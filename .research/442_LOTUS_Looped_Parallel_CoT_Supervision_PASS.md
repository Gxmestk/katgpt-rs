# Research 442: LOTUS — Looped Transformers with Parallel Supervision on Latents

> **Source:** [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779) — Ying Fan (MSR), Anej Svete (ETH), Kangwook Lee (KRAFTON / Ludo Robotics). v2, 13 Jul 2026.
> **Code:** https://github.com/yingfan-bot/lotus
> **Date:** 2026-07-16
> **Status:** Done
> **Related Research:** 073 (LT2 — architecture we ship), 097 (Training-Free Loop), 158 (MUX multiplexed latents), 192 (NextLat / RiM buffer slots), 241 (SwiR switch-thinking), 250 (Latent Recursion Policy Improvement), 273 (ELT elastic looped — canonical precedent), 277 (DiffusionGemma — scratch tokens already shipped), 295 (AC-GPT position-aware prefix), 414 (Fully Looped + Readout Blind Spot — LOTUS empirically demonstrates the blind spot)
> **Related Plans:** 108 (LT2 — shipped, DEFAULT-ON, GOAT 8/8), 136 (Training-Free Loop — shipped), 172 (RiM Reasoning Buffer Slots — shipped, DEFAULT-ON), 275 (SwiR), 283 (Self-Advantage Gate), 428 (Loop Stability PoC)
> **Classification:** Public

---

## TL;DR

LOTUS is a **training recipe** for latent CoT: a looped padded Transformer processes K learnable latent blocks of c tokens in parallel for R iterations, with a per-position cross-entropy loss (`L_step`) on each latent position's gold CoT-step token routed through the base LM head. At 3B scale it is the first latent method to bridge the gap to explicit CoT on GSM8K (within 1.5pp) while cutting thought-phase latency 2.5× (compact math) to 6.9× (natural-language CoT). Post-loop latents recover gold CoT tokens (70.9% top-1) AND surface unseen-but-valid alternative chains (15.3% top-1).

**Verdict: Pass** — the architecture is shipped, the supervision recipe → riir-train, and the empirical findings validate existing design decisions rather than add a new inference primitive.

**Distilled for katgpt-rs (modelless, inference-time):** Nothing new. LOTUS's three ingredients map to shipped primitives with prior research notes covering each.

---

## 1. Paper Core Findings

### 1.1 Architecture — looped padded Transformer (we already ship this)

LOTUS places K learnable padded latent blocks of c tokens between question and answer, delimited by `<BoT>` / `<EoT>`. The base model `f_θ` iterates R times over the latent region:

```
h^(0) = f_θ(E | C_pre)
h^(t) = f_θ(E + h^(t-1) | C_pre),    t = 1, ..., R
```

where `C_pre` is the question-prefix KV cache (computed once, reused), `E ∈ R^{Kc×d}` are the learnable latent embeddings, and the brackets denote subsequence concatenation. This is a finite-unroll, input-injected recurrence over a looped Transformer.

**Our shipped equivalent:** `LoopMode::WeightShared { loop_count }` (Plan 108 / Research 073, GOAT 8/8, DEFAULT-ON) + RiM Reasoning Buffer Slots (Plan 172, `rim_block_count: K`, `rim_tokens_per_block: c`, DEFAULT-ON after GOAT proof of zero decode cost). LOTUS's K=6 blocks × c=25 tokens = Kc=150 latent positions is exactly the RiM slot configuration.

### 1.2 Parallel cross-entropy supervision on gold CoT tokens (→ riir-train)

The training objective has two losses:

```
L_step = (1/N_step) Σ_i Σ_j CE(f_head(h^(R)_{i,j}), T_{i,j})    // per-position step loss
L_ans  = (1/|A|)  Σ_m  CE(f_head(z_m), A_{m+1})                  // answer suffix next-token
L      = L_ans + λ_step · L_step,    λ_step = 0.05
```

Training: AdamW, bf16 (Llama) / fp32 (GPT-2), gradient clipping 1.0, batch 128, 30 epochs, staged curriculum (convert one CoT step into a latent block per `E_stage=1` epochs). Gradient propagated through all R iterations. λ_step = 0.05 for Llama, 0.033 for natural-language stress test.

**This is gradient descent through R loop iterations.** Per §3.5 modelless-unblock check: no deterministic construction produces the per-position gold-token alignment — it requires the gold CoT data + backprop. → **riir-train**.

### 1.3 The PCL factorization insight (the conceptual contribution)

§3.3.2 introduces the **Parallel Chain Likelihood (PCL)**: the step loss factorizes the chain as `p^PCL_θ(T|Q) = ∏_{i,j} p_θ(T_{i,j}|Q)` — conditionally independent readouts — but the latent states themselves are NOT independent because the looped Transformer computes them jointly via shared computation. Two complementary roles:

- **`L_step` provides support coverage** — every gold chain lies inside `∏_i ∏_j supp(q(T_{i,j}|Q))`. Per-position mass on right tokens.
- **`L_ans` provides global joint selection** — answer is decoded from the jointly computed latent configuration, so gradients favor jointly-coherent hidden states.

This is a **theoretical lens**, not a new inference primitive. It explains why independent per-position supervision can produce globally coherent answers — but the mechanism (jointly computed latents via looped shared weights) is exactly LT2 + RiM slots.

### 1.4 Empirical findings (insights, not new primitives)

- **First latent CoT to bridge at 3B** — 70.0% GSM8K vs 71.5% explicit CoT (within 1.5pp); surpasses CoT on out-of-domain average.
- **2.5×–6.9× thought-phase latency reduction** — 133ms vs 338ms (math), 140.8ms vs 963.6ms (NL).
- **Post-loop latents are CoT-aligned** — 70.9% top-1 recovery of gold CoT tokens through base LM head.
- **Latents carry unseen-but-valid alternative chains** — Section 5.2: only-U intermediates surface at 15.3% top-1 / 64.0% top-5, despite never being trained on.
- **R tunable at inference without retraining** — Table 6: trained at R=6, accuracy climbs monotonically from R=1 (22.7%) to R=6 (70.0%), dips at R=7 (69.3%).
- **Looped backbone + parallel supervision are both essential** — ablations Table 4 + 5 confirm.
- **Direct LM-head routing is robust across scale** — LOTUS-aux (auxiliary decoder) matches at 3B but degrades at GPT-2 (35.5% vs 44.1%).

---

## 2. Distillation

### 2.1 What ships in our stack (do NOT reimplement)

| LOTUS component | Our shipped equivalent | Status |
|---|---|---|
| Looped padded Transformer (R iterations over Kc latent positions) | `LoopMode::WeightShared { loop_count }` (Plan 108) + RiM slots (Plan 172) | ✅ DEFAULT-ON, GOAT 8/8 |
| K learnable latent blocks × c tokens between Q and A | `rim_block_count: usize` (K) + `rim_tokens_per_block: usize` (c) | ✅ DEFAULT-ON, GOAT-proven zero decode cost |
| Prefix KV cache `C_pre` reused across R iterations | Standard LT2 forward pass in `forward_looped()` | ✅ Shipped |
| Per-position logit readout from latent positions | RiM T3 "Logit Readout from Buffer End" at index `n_prompt + K*M - 1` | ✅ Shipped |
| Per-loop residual gate ρ_τ (zero-init) | `ResidualGate` (zero-init) from Plan 108 | ✅ Shipped |
| R tunable at inference | `loop_count` is per-`Config`; per-dispatch elastic L is a small missing piece (Research 273 §2.3, Gain-tier) | 🟡 Mostly shipped (static), per-dispatch dynamic is open |
| Loop-stability fixes (inter-loop RMSNorm, FLA, Attention Injection) | Plan 428 PoC (Research 414) — addresses the residual explosion LOTUS's `L_step` empirically exposes | 🔬 Active PoC |

### 2.2 What's the training recipe (→ riir-train, NOT distilled here)

| LOTUS training component | Why it requires gradient descent |
|---|---|
| `L_step` parallel cross-entropy on gold CoT tokens | Requires gold CoT data + backprop through R iterations + LM head gradient |
| Staged curriculum (`K_e = min(⌊e/E_stage⌋, K)`) | Training schedule, not an inference primitive |
| `λ_step` weighting (0.05 / 0.033) | Training hyperparameter |
| LOTUS-aux auxiliary decoder `g_ϕ` (full-size deep copy of base) | Trained jointly with base LM; training-time only |
| AdamW + bf16 + grad-clip 1.0 + batch 128 + 30 epochs | Standard fine-tuning recipe |

**§3.5 modelless-unblock check:**
1. **Freeze/thaw snapshot correction** — N/A. The recipe is not a systematic bias correctable by a frozen snapshot; it requires gold-token supervision signal flowing into the latent positions.
2. **Raw/lora reader-writer hot-swap** — N/A. No deterministic construction aligns latent positions to gold CoT tokens; the alignment is learned via backprop through the looped shared weights.
3. **Latent-space correction** — N/A. The "correction" here is the gold CoT signal itself, which is data not a deterministic projection.

All three paths fail. → **riir-train** (genuine training dependency).

### 2.3 Latent-space reframing (the Super-GOAT angle was searched; not found)

Per the workflow §1 step 3, the seven Super-GOAT factory modules were checked for a latent-space reframe of LOTUS:

- **HLA per-NPC latent state**: LOTUS's Kc-position latent workspace maps to a per-NPC HLA state with K=6 "sub-goal blocks" of c=25 inner tokens. But the per-NPC sub-goal compaction primitive (`riir-ai/.research/155_Per_NPC_Sub_Goal_Compaction_Guide.md`) already ships this — K blocks of compacted sub-goal state, refined across cycles. The mapping is direct, not novel.
- **`latent_functor/` operations**: LOTUS's R iterations of weight-shared refinement IS functor application. We ship this as `latent_functor/reestimation.rs` (coherence-driven re-estimation across cycles). The "intermediate states are valid belief states" framing is exactly Research 273 ELT's Any-Time claim — recorded, Gain-tier.
- **`cgsp_runtime/` curiosity**: LOTUS's "unseen-but-valid alternative chains surface in latents" (§5.2) aligns with CGSP's curiosity-driven exploration of valid-but-unseen directions. But CGSP already explores this space via the Learning-Potential score; LOTUS's finding is *evidence for* our existing design, not a new mechanism.
- **`NeuronShard` / freeze envelope**: LOTUS's latents are session-scoped, not committed. The freeze/thaw cadence for personality versioning is already covered by `riir-ai/.research/158_Per_NPC_Committed_Personality_Blend_Guide.md` (committed personality blend) + freeze/thaw envelope.
- **DEC Stokes operators**: N/A — LOTUS is not a manifold-geometry paper.
- **LatCal fixed-point commitment**: N/A — LOTUS is not a chain-commitment paper.
- **`BoMSampler` K-hypothesis**: LOTUS's "latents carry unseen-but-valid alternatives" is conceptually adjacent to BoMSampler's K-hypothesis sampling, but BoMSampler samples K explicit hypotheses while LOTUS refines a single jointly-computed workspace. Different mechanism; the multi-path insight is already captured.

**No Super-GOAT reframe found.** Every angle maps to a shipped primitive with a prior research note.

### 2.4 Fusion (for the record — none planned)

The closest cousin fusion candidates, all of which produce capabilities that already ship:

| Fusion | Sources | Capability | Ships as |
|---|---|---|---|
| LOTUS × LT2 × RiM | Architecture | K-block × c-token × R-iteration looped workspace | Plan 108 + 172 (DEFAULT-ON) |
| LOTUS × Self-Advantage Gate | Per-position logit supervision × dead-compute detector | Halt R early when post-loop latent stabilizes | Plan 283 (DEFAULT-ON) |
| LOTUS × PathwayTracker | Loop-depth R × stability-based early exit | Elastic R per dispatch | Plan 231 (DEFAULT-ON) |
| LOTUS × BoMSampler | Post-loop multi-path latents × K-hypothesis sampling | Diverse trajectory hypotheses from one looped workspace | Plan 281 (DEFAULT-ON) |
| LOTUS × SwiR | Latent reasoning mode × explicit↔latent switch | Mode controller chooses between LOTUS-style and explicit CoT | Plan 275 |
| LOTUS PCL × CLR vote | Per-position independent scoring × nonlinear reliability gating | "(mean(v))^M" reliability vote IS the L_ans-side joint selection pressure | riir-ai Plan 316 (DEFAULT-ON) |

**None of these produce a new capability class.** Each cell either ships or is a Gain-tier coordination layer (per Research 273 ELT §2.3).

---

## 3. Verdict

**Tier: Pass**

**One-line reasoning:** LOTUS's three ingredients — (1) looped padded Transformer architecture, (2) parallel cross-entropy supervision on gold CoT tokens, (3) empirical findings on CoT-aligned latents — decompose as: architecture shipped (Plan 108 LT2 + Plan 172 RiM, both DEFAULT-ON, GOAT 8/8 + zero-decode-cost), supervision recipe → riir-train (genuine gradient descent through R iterations, §3.5 modelless paths all fail), and findings validate existing design decisions rather than introduce a new inference primitive.

**Tiers (high → low):**

| Tier | Criteria | Routing |
|---|---|---|
| **Super-GOAT** | Novel mechanism + new capability class + selling point + force multiplier | (not met — every angle maps to shipped primitive) |
| **GOAT** | Provable gain over existing approach | (not met — no new inference primitive) |
| **Gain** | Incremental improvement | (close — R-tunable-at-inference is Gain-tier per Research 273 §2.3, but already noted there) |
| **Pass** | Training-only (→ riir-train) OR mechanism already shipped OR LLM-orchestration class | **← this** — architecture shipped, supervision → riir-train, insights validate existing design |

**§3.5 modelless-unblock check:** All three paths fail for the supervision recipe. → riir-train (genuine gradient descent dependency, documented above).

**§3.6 defend-wrong PoC requirement:** N/A. This verdict makes no quality parity claim ("already ships at parity"). The architectural-coverage claim is grep-verified (Plan 108 + 172 ship the architecture; Plan 428 addresses the loop stability). No PoC needed because no parity claim — only architectural redirect.

**MOAT gate (katgpt-rs):** Out of scope for moat contribution. The architecture is shipped; the supervision recipe is training (riir-train); the insights don't add a new inference primitive. Recording the verdict for prior-art hygiene only.

### Routing decisions

- **LOTUS architecture (K blocks × c tokens × R iterations)** → already shipped (Plan 108 + 172). No action.
- **LOTUS training recipe (`L_step` parallel CE, λ_step curriculum, LOTUS-aux auxiliary decoder, AdamW config)** → **riir-train**. Out of scope for this workflow — one-line note recorded here, no files created in riir-train this session.
- **PCL factorization insight (§3.3.2)** → conceptual lens; documents why RiM slots + LT2 + per-position screening works. No code change.
- **Empirical findings (post-loop latents carry unseen alternatives; R tunable at inference)** → validates existing design. The R-tunable-at-inference property is the same Any-Time framing recorded in Research 273 ELT (Gain-tier, open as a possible small katgpt-rs optimization issue).
- **Loop stability exposure** → already addressed by Plan 428 (PoC for parameter-free fixes for the Readout Blind Spot that LOTUS's `L_step` empirically demonstrates).

---

## 4. Related work — what this verdict rules in and out

**Rules in (validates):**
- LT2 + RiM slot architecture is the right substrate for parallel-padded latent reasoning (Plan 108 + 172 DEFAULT-ON).
- The Self-Advantage Gate (Plan 283) and PathwayTracker (Plan 231) compose naturally as the inference-time analog of LOTUS's R-tunable depth.
- BoMSampler (Plan 281) is the inference-time consumer of LOTUS-style multi-path latent workspaces.
- Plan 428 (Loop Stability PoC) is the correct response to the residual-explosion failure mode that LOTUS's `L_step` supervision is partially mitigating at training time.

**Rules out (does NOT add):**
- A new katgpt-rs inference primitive. Architecture is shipped; the recipe is training.
- A Super-GOAT moat claim. Every latent-space reframe maps to a shipped primitive.
- A riir-ai game-runtime guide. The per-NPC mappings (sub-goal compaction, committed personality, CGSP curiosity) are already covered.

**Canonical failure mode prevented:** A paper-vocabulary grep for `LOTUS|padded latent|parallel CoT supervision` would return zero hits and could falsely suggest novelty. The codebase-vocabulary grep (`LoopMode::WeightShared`, `rim_block_count`, `forward_looped`, `ResidualGate`) returns the shipped primitives. The notes-layer grep (`LT2`, `RiM buffer slots`, `ELT`, `Fully Looped Transformer`, `Readout Blind Spot`) returns the prior research corpus. Three-layer check confirms: covered on all three layers.

---

## TL;DR

LOTUS = LT2 (Plan 108, DEFAULT-ON, GOAT 8/8) + RiM Reasoning Buffer Slots (Plan 172, DEFAULT-ON, zero-decode-cost GOAT) + a training recipe (`L_step` parallel CE on gold CoT tokens, λ_step curriculum, LOTUS-aux auxiliary decoder) → riir-train. The PCL factorization insight (§3.3.2) is a conceptual lens explaining why per-position independent supervision + jointly-computed looped latents produces globally coherent answers — it documents the mechanism behind our shipped RiM + LT2 + per-position screening composition. Empirical findings (first latent CoT to bridge at 3B; 2.5-6.9× latency reduction; post-loop latents carry unseen-but-valid alternatives; R tunable at inference) validate existing design decisions rather than introduce new inference primitives. Verdict: **Pass** — architecture shipped, supervision → riir-train, insights validate. The loop-stability exposure (residual explosion under `L_step`) is the same Readout Blind Spot addressed by Plan 428's parameter-free fixes.
