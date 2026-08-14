# Research 480: FLARE — Diffusion for Hybrid Language Model (Dual-Trust Decode + Proposal-Consistency Exactness)

> **Source:** [FLARE: Diffusion for Hybrid Language Model](https://arxiv.org/abs/2606.01774) — Zhu, Shi, Ge, Tan, Xu, Zhu, Kuen, Goswami, Jain, Chen, Tao, Gu (Adobe Research + Georgia Tech), arXiv:2606.01774v2, 2026-06-01 (v2 2026-08-04)
> **Date:** 2026-08-14
> **Status:** Implemented — Issue 587 closed (all T1–T7 done, T8 deferred); GOAT G1–G4 PASS, `SoftmaxArgmax` promoted to default per [Bench 634](../.benchmarks/634_self_spec_acceptance_policy_goat.md). Audit follow-up: Issue 651 (FlashAR Cold/Warm paths) — **RESOLVED 2026-08-15** (Bench 637): Warm/Cold run Eq 21 against the slot-aligned law (also fixed FlashAR's own instance of the 587 off-by-one — H-win positions had auto-accepted); Plasma/Hot stay skip-biased by design.
> **Related Research:** 055 (Nemotron TriMode — closest shipped cousin), 034 (D2F), 072 (DMax SPD), 149 (FlashAR), 154 (DFlare — **name collision, different paper**), 376 Phase 4 (SetDiffusion), 070 (Gated DeltaNet 2)
> **Related Plans:** 066 (D2F), 089 (D2fDrafterVerifier / tri_mode), 109 (DMax SPD), 116 (DiffusionSampler), 166 (FlashAR consensus)
> **Cross-ref (riir-ai / riir-train):** riir-ai Research 036 (Luce hybrid DeltaNet megakernel), riir-train Research 003 (D2F training)
> **Classification:** Public
> **Issue:** [587 — distribution-preserving self-speculation acceptance](../.issues/587_self_spec_exact_acceptance_policy.md)

---

## TL;DR

FLARE converts hybrid-attention (softmax + Gated DeltaNet) AR checkpoints into dLLMs under a ~10B-token budget, with one checkpoint serving two decode regimes: **AR-Trust** (noisy/diffusion stream drafts → clean/AR stream verifies via speculative rejection) and **Diffusion-Trust** (parallel block denoising with confidence commits). For us the headline is NOT the conversion recipe — our tri_mode lineage already ships dual-trust decode — but the **proposal-consistency exactness analysis**: our shipped `D2fDrafterVerifier` uses greedy argmax prefix-match acceptance, which FLARE's Eq (8)/(21)/(22) taxonomy shows is distribution-biasing, with a ~20-line distribution-preserving fix (Softmax-Argmax policy) and a memory-lean streaming verify (Gumbel-max correction, no `[(K+1)×V]` materialization).

**Distilled for katgpt-rs (modelless, inference-time):**
Speculative acceptance of parallel (non-causal) drafts cannot reuse the AR proposal law q recomputed at verify time — parallel drafts under a masked block context break the Leviathan proposal-consistency condition. Three policies span the exactness/overhead trade-off: (1) **Exact-Truncated** — store the draft-time top-k proposal `(ids, probs)` and run true `min(1, p/q)` rejection; (2) **Softmax-Argmax** (Eq 21) — argmax drafts, accept `⟺ u ≤ p_i(d_i)`, correction `~ p \ {d_i}` — exact w.r.t. the target at near-zero overhead; (3) **Truncated-Argmax** (Eq 22) — both sides top-k truncated, approximate. Correction tokens can be sampled by **Gumbel-max over `[p−q]₊` without materializing the correction distribution**, and `p_i(d_i)` can be gathered from a streaming `(max, sum-exp)` pair over logits — eliminating the full per-position softmax/prob tensors.

---

## 1. Paper Core Findings

### 1.1 Framework

- **Goal:** convert strong hybrid-attention AR checkpoints (Qwen3.5-2B/4B/9B, softmax + GDN layers + ShortConv) into serving-efficient dLLMs with a modest (~10B token, 9000-step, bs256, L4096) SFT budget — vs 50–200B in prior conversion work (SDAR, Efficient-dLM).
- **Objective (token-equal AR+diffusion):** `L = L_AR + L_diff`. Clean stream = causal next-token loss. Noisy stream = block diffusion where each block's masked set `M_b` and complement `M_b^c` form **two complementary noisy views** — every token contributes exactly one AR signal + one diffusion signal at unit weight, by construction (vs Nemotron TriMode's α=0.3 weighting). Complementary views double as **antithetic mask sampling** (variance reduction).
- **Mask:** document-packed clean/noisy — clean stream token-causal and isolated from noisy; noisy blocks bidirectional within block, attend to preceding clean context only; document boundaries reset states. No cross-document leakage, no padding overhead.
- **Logit shift** on noisy-stream diffusion terms aligns block-diffusion logits with inference-time token positions (avoids a wasted block-boundary prediction). Small benchmark effect; kept for decoding compatibility.

### 1.2 Headline empirical finding — transfer data dominates

Controlled ablations (Qwen3-1.7B seed, fixed budget, 12-task suite):
- Pure block-diffusion transfer loses −21.8 pts avg vs AR-SFT. Restoring a **token-causal AR clean stream** recovers +14.0 pts (the single largest step); adding the clean-stream NTP loss restores Math to parity; logit shift saturates.
- Once objective/mask are aligned, **residual variation is governed by transfer-data mix**, not algorithmic choice. Mix 4 (Long-CoT+Math+IF, weights 0.4/0.4/0.2) won.
- **AR-SFT is a faithful low-cost proxy** for screening data mixes before expensive dLLM conversion (converted dLLMs track their AR-SFT counterparts under the same mix).
- Aggressive instance-level filtering (IFD scoring + cluster-balanced sampling, App C.5) did NOT close the residual gap to the source model — the bottleneck is distribution mismatch with the seed's own post-training data, not filtering quality.

### 1.3 Hybrid-backbone training machinery (Appendix A)

Non-causal visibility on linear-attention layers is a **state-scheduling problem, not a mask**: a noisy block must seed from the clean block-boundary state `S_(b−1)B`, expose its tokens to one another (block-end readout `õ_ℓ = S̃_bB^T q̃_ℓ` gives in-block bidirectionality), and never leak across blocks/documents (state resets at doc starts, cross-doc lag masking in ShortConv).
- **Route I (chunk-then-refine):** materialize every block-boundary clean state in HBM (`L/B` scaling — ~22 GiB/layer at B=1), then run independent block-local recurrences. Wins at B≥16 (dense chunk matmul saturates tensor cores).
- **Route II (fused two-stream):** one program per chunk, replay block-boundary states in registers from **strided checkpoints** (stride S, `N_ckpt = M/S` per chunk, ~128 MiB vs ~22 GiB at B=1/S=16), fused backward emitting only `dh^inject_[c]`. Wins at B<16 — a structural requirement at diffusion block sizes (FLARE uses B=4).
- ShortConv analog: noisy lags read in-block, clean lags read across the boundary; fused kernel always wins.
- End-to-end: FLARE-2B MFU 24.81% at B=4 vs pure-AR 24.04% — kernel stack absorbs the ~2× two-stream overhead (4× per-token attention work counting complementary views).

### 1.4 Unified inference (Appendix B)

One checkpoint, two decode paths, one SGLang stack:
- **AR-Trust:** anchor + K verify rows (causal) + N−1 draft rows (bidirectional) in one forward. Clean-stream logits verify noisy-stream drafts left-to-right. Draft policies + exactness (see TL;DR). Fused verify kernels: streaming `(max, sum-exp)` → `p_i(d_i)` gather; Gumbel-max correction; sparse k×k verify for truncated policies; tiled top-k LM head (never materialize `[M,V]`).
- **Diffusion-Trust:** parallel denoising with confidence-threshold commits (`A_s = {i ∈ R_s : c_i ≥ γ_s}`, all-commit at s=S). **Denoise passes read the recurrent state but do NOT write it back** — only a final token-causal replay commits the block (intermediate contents may be revised; early writes would contaminate the state trajectory).
- **Recurrent-state rewind for partial acceptance:** accepting r of K speculative tokens on a delta-rule layer is not a KV tail-trim. **Cache-and-scatter:** record `S^(t)` for every verify position as the recurrent kernel's epilogue (stores only, no FLOPs), then one fused gather-scatter at offset r across all GDN layers — vs replay (extra per-layer matmuls).
- Serving details: prefix-tile fast path (skip per-score mask predicates on prefix-only KV tiles), CUDA-graph replay eligibility pinned on 4 invariants (block size, mask mode, state-update mode, logits mode) — one graph per draft policy.
- Throughput: FLARE-2B @ C=8 on 1×A100: 2087 tok/s GSM8K (2.2× LLaDA-2.1-mini, 4.8× SDAR-1.7B). FLARE-9B retains 95–99% of Qwen3.5-9B on math/MMLU-Pro; Diffusion-Trust consistently weaker on long structured code outputs (commits without left-to-right syntactic verification).

## 2. Distillation

### 2.1 What already ships here (honest mapping)

| FLARE component | Our code | Status |
|---|---|---|
| AR-Trust (diffusion drafts → AR verifies) | `D2fDrafterVerifier` (Plan 089, `tri_mode`) + `DecodeStrategy::SelfSpeculation` | ✅ ships — **but prefix-match acceptance (see 2.2)** |
| Dual-path / consensus acceptance | `FlashARConsensusVerifier` (Plan 166, `flashar_consensus`) | ✅ ships (different acceptance scheme — audit for the same bias) |
| Diffusion-Trust (confidence block denoising) | `d2f_decode_block` (Plan 066, `dllm`) | ✅ ships |
| Trained sampler | `DiffusionSampler` (Plan 116) | ✅ ships |
| Soft-parallel / set variants | `DiscreteDiffusionSoft` (Plan 109), `SetDiffusion` (Research 376) | ✅ ships |
| Exact p/q rejection (AR drafter) | `LeviathanVerifier` | ✅ ships — **not wired to the D2F drafter** |
| Hybrid GDN backbone + diffusion | — | ❌ no consumer today (riir-ai has the linear-attention lineage: HLA, Research 036 Luce, KDA refs) |

### 2.2 The actionable delta — proposal-consistency exactness (→ Issue 587)

`d2f_verifier.rs` Phase 3 currently: *"simple prefix matching: compare draft[i] with argmax of target p_dist[i+1]; accept longest matching prefix + bonus token at first mismatch."* Two defects per FLARE's analysis:
1. **Not distribution-preserving.** Accepting only `d_i == argmax(p_i)` and correcting with the argmax collapses the output toward the target mode — temperature/sampling semantics are destroyed (every rejection emits the greedy token). FLARE Eq (21) (Softmax-Argmax): accept `⟺ u ≤ min(1, p_i^full(d_i))`, correction `y* ~ p^full \ {d_i}` — exact w.r.t. the target, ~20 lines in Phase 3, reusing the already-materialized `p_dist`.
2. **`[(K+1)×V]` materialization.** `p_distributions_flat` holds full-vocab distributions per verify position (5×256K f32 ≈ 5 MB/round at Gemma-2 vocab, plus full softmax each). FLARE's streaming verify: pass 1 computes `(max, sum-exp)` over logits and gathers `p_i(d_i)` directly; on rejection, pass 2 Gumbel-max selects the correction from `[p−q]₊` — never forming prob tensors.
3. **Exact-Truncated (Eq 8)** needs the draft-time proposal law `q_i` stored (top-k ids+probs). Subtlety FLARE makes explicit: parallel drafts under a masked block context are NOT sampled from the AR factorization, so recomputing `q_i` at verify time breaks equivalence — either draft argmax (point-mass q) or plumb `q` out of `d2f_decode_block`. Our D2F drafter samples with temperature, so exactness requires the plumb (or argmax drafting).

### 2.3 Secondary extractions (cross-repo, recorded not planned)

- **D3 — recurrent-state cache-and-scatter rewind** (riir-ai): partial-accept rewind on delta-rule layers = record per-verify-position states as kernel epilogue + fused gather-scatter at offset r. Only relevant if speculative decode ever runs on a hybrid/linear-attention model (HLA / Luce / KDA lineage). No consumer today → defer.
- **D4 — antithetic complementary-view masking** (riir-train, Research 003 cross-ref): mask `M` and complement `M^c` as an antithetic pair halves mask-sampling gradient variance for free in any stochastic-masking trainer.
- **D5 — AR-SFT proxy screening** (riir-train methodology): before an expensive conversion/distillation run, screen data mixes with cheap AR-SFT — converted models track their AR-SFT counterparts under the same mix.
- **D6 — confirmation, not new (vocabulary bridge):** FLARE's serving discipline — *denoise passes read state but never write; only the finalized block's causal replay commits* — is the same law as our two-brain one-way gate (think brain never writes the info brain) and freeze/thaw atomic commit. Independently derived in the kernel domain; reinforces the invariant, no code change.

### Fusion

- **FLARE Eq (21) × our `LeviathanVerifier` × `D2fDrafterVerifier`** → distribution-preserving self-speculation with zero new kernels: the p/q machinery already exists for AR drafters; the D2F path needs only the policy swap (+ optional q-plumb for Eq 8). GOAT framing: G1 = exactness test (accepted-token empirical distribution == target sampling distribution; prefix-match baseline FAILS by construction), G2 = acceptance rate + verify latency, G4 = steady-state alloc-free preserved.
- **FLARE state-rewind × riir-ai HLA/Luce × our KV snapshot/restore** → a linear-attention speculative state-rollback primitive. Deferred until a hybrid-backbone inference consumer exists.

## 3. Verdict

**Tier: Gain** — mechanism largely ships (tri_mode lineage), but the paper exposes a concrete correctness gap (prefix-match acceptance is distribution-biasing) plus a verify-path memory optimization, both actionable now → Issue 587. Not GOAT (no gain proven yet — that's the issue's job). Not Super-GOAT: Q1 fails (SDD arXiv:2408.05636, DiffuSpec, TiDAR arXiv:2511.08923, I-DLM arXiv:2604.11035, Nemotron TriMode — dense prior art; FLARE itself cites TiDAR/I-DLM as closest), Q3 fails (no game-stack selling point), Q4 partial.

**MOAT gate:** katgpt-rs speculative slot — generic inference mechanics, no game/chain/shard IP → correct repo, public. Training-recipe aspects cross-ref'd to riir-train; hybrid-backbone aspects to riir-ai. No sibling guides (not Super-GOAT).

**Name-collision warning:** FLARE (Adobe, this note) ≠ DFlare (Research 154, layer-wise fusion block diffusion speculative decoding). Future greps for "flare" must disambiguate by arxiv ID.

## What we do NOT extract

| Aspect | Why not |
|---|---|
| Triton Route I/II kernels, strided-checkpoint details | We're CPU/Metal/CubeCL, not Triton; no hybrid-backbone training consumer |
| SGLang serving stack, CUDA-graph 4-invariant replay | We ship a library, not a serving framework |
| Transfer-data mixes (Nemotron pools, IFD pipeline) | No AR→dLLM conversion campaign; recorded as methodology cross-ref only |
| Document-packing cu_seqlens guards | Training-side; no consumer |
| MoE-dLLM upcycling, ELBO-surrogate RL (future work) | Out of scope |

## Cross-References

- `.research/055_Nemotron_TriMode_Diffusion.md` — tri-mode + self-speculsion prior art (our closest cousin; its "MISSING D2F Drafter Verifier" row was filled by Plan 089 — this note fills the *acceptance policy* gap it left)
- `.research/034_D2F_Discrete_Diffusion_Forcing.md` + Plan 066 — Diffusion-Trust path
- `.research/149_FlashAR_Diagonal_Parallel_Decoding.md` + Plan 166 — consensus acceptance (audit for mode-bias)
- `.research/154_DFlare_Layer_Wise_Fusion_Block_Diffusion.md` — **different paper, similar name**
- `.research/070_Gated_DeltaNet_2_Decoupled_Erase_Write_Linear_Attention.md` — GDN background for §1.3
- riir-ai `.research/036_Luce_Megakernel_Hybrid_DeltaNet_Attention.md` — hybrid-attention consumer side
- riir-train `.research/003_D2F_Discrete_Diffusion_Forcing.md` — D2F training (antithetic masking cross-ref)
