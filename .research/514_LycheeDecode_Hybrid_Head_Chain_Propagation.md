# Research 514: LycheeDecode — Hybrid-Head Chain Propagation for Sparse Decode

> **Source:** [LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](https://arxiv.org/abs/2602.04541) — Lin, Li, Chen, Shi, Chen, Hu, Zhang (HIT Shenzhen), ICLR 2026, Feb 2026
> **Date:** 2026-08-27
> **Status:** DISTILLED — pending owner decision (actionable delta gated behind the Issue 694 overlap PoC)
> **Related Research:** 086 (RTPurbo — closest cousin; head calibration SHIPS as `rt_turbo`), 362 (HydraHead — causal calibration, `CalibrationMode::CausalNecessity`), 436 (FlashMemory — the ACTIVE sparse path; τ-step time amortization), 071 (DashAttention — α-entmax block routing base), 225 (MSA — sparse-attention distillation training), 399 (HiLS), 213 (StillKV)
> **Related Plans:** 126 (rt_turbo), 358 (causal head importance), 44 (PFlash window+sinks)
> **Classification:** Public

---

## TL;DR

LycheeDecode partitions attention heads into a few **retrieval heads** (full attention over the whole context → `argsTopK` of the attention row they already computed → propagate the top-k index set to the *same-index head in the next layer*) and a majority of **sparse heads** (attend ONLY to the inherited set; propagate it unchanged). Head-index-level cross-layer sharing — vs TidalDecode/OmniKV's layer-level sharing (one set for ALL heads in subsequent layers). Head roles are learned via a **HardKuma** (hard-concrete-style) near-binary differentiable gate with a closed-form expected-L0 budget under Lagrangian control. A **workload-pooling kernel** (aggregate every head's block work into one pool, partition into uniform "splits" across thread blocks) fixes the dense-retrieval/sparse-head imbalance. 2.7× e2e at 128K; up to 7× kernel-level at batch 8; quality at parity or above full attention (denoising hypothesis: sparse heads filter distractor tokens).

**For our stack the honest split:** head specialization + per-head calibration + dynamic top-p **already ship** (`rt_turbo`, Plan 126 + Plan 358, GOAT 6/6). The two unshipped deltas are (1) **cross-layer selection reuse along per-head chains** — the *depth* amortization axis, composable with `flashmemory_sparse`'s *time* amortization (τ-step refresh) on the active 256K-Bonsai-on-4090 path — and (2) the **workload-pooling kernel schedule**, which is directly relevant because FlashMemory's per-head *variable* block counts (1–20 blocks/head) create exactly the imbalance it solves. Both are gated behind a cheap falsifiable measurement: **adjacent-layer same-index-head top-k overlap on real weights** (the paper's Fig. 2 shows it varies 0–100% per head on Llama3 — chain reuse is head-conditional, not universal).

**Distilled for katgpt-rs (modelless, inference-time):**
1. **Cross-layer top-k overlap statistic** — offline calibration signal: `overlap((l,h), (l-1,h)) = |topK(A_h^l) ∩ topK(A_h^{l-1})| / k` from ONE calibration forward pass. Decides which heads may join a reuse chain (high overlap) vs must refresh per layer (low overlap). Extends `HeadCalibration` with a third axis (retrieval score / causal IE / **chain affinity**).
2. **Chain-propagation decode schedule** — sparse-chain heads gather the inherited k-token set instead of re-selecting per layer. Selection cost drops by the chain length; KV read per sparse head drops to k tokens (vs rt_turbo local heads' fixed w-token window).
3. **Workload-pooling split scheduling** (kernel guide; consumer is riir-ai's GPU decode) — heterogeneous per-head work (dense retrieval heads, 1–20-block sparse heads) pooled per batch item into uniform splits with online-softmax partials + a flash-decode-style combine.

NOT distilled: HardKuma training (→ riir-train recipe line, §2.5), the TileLang kernel itself, cache correction (periodic dense re-prefill of polluted tokens — TidalDecode/RSA's technique; the modelless analog already ships as FlashMemory's τ-step refresh, same "periodic dense repair" family).

---

## 1. Paper Core Findings

### 1.1 The mechanism (head-level hybrid decode)

- **Retrieval head** `h ∈ H_R` at layer l: full attention `A = softmax(qKᵀ/√d)`, output `AV`, and `S_h^(l+1) = argsTopK(A_h^(l), k)` — the selection is a byproduct of attention it had to compute anyway (near-zero marginal cost).
- **Sparse head** `h ∈ H_S` at layer l: `O = softmax(q·K[S_h^(l)]ᵀ/√d)·V[S_h^(l)]` — gather-only; set propagated unchanged (`S^(l+1)_h = S^(l)_h`).
- Layer 0: ALL heads retrieval (chain initialization).
- Roles are per-(layer, head) — head index h may be retrieval at layer 5, sparse at 6–11, retrieval again at 12 (refresh point for that chain).
- Budget in the paper: **32 retrieval head-slots total** across a 32-layer × 32-Q-head (8-KV-head GQA) Llama3-8B — ~3% of head-slots, matching TidalDecode's 2-full+2-selector-layer × 8-KV-head budget. GQA handled by average-pooling Q heads onto KV heads for selection.
- At inference, Q/K/V projection output channels are **permuted once** so retrieval and sparse heads form two contiguous clusters (kernel-friendly).

### 1.2 The motivating measurement (Fig. 2 — why head-level, not layer-level)

Top-k (k=5) overlap between same-index heads of ADJACENT LAYERS on Llama3 varies from **0% to 100% per head**. Layer-level sharing forces the 0%-overlap heads to consume a foreign set; per-head chains let high-overlap heads reuse (cheap) while low-overlap heads refresh (accurate). This statistic is itself the cheapest modelless distillate — it is the calibration signal that decides chain membership.

### 1.3 HardKuma head identification (the training half)

- Construction: `u~U(0,1)` → `s = (1 − u^{1/β})^{1/α}` (Kumaraswamy inverse CDF; Kuma CDF `F(x)=1−(1−x^α)^β` closed-form, unlike Beta) → stretch `s' = s(q−p)+p` with (p,q)=(−0.1,1.1) → rectify `z = min(1, max(0, s'))`. Point masses at exactly 0 and 1, continuous in between, differentiable a.e.
- Training: hybrid map `Ã = z·A_R + (1−z)·A_S` (both branches computed per head) under logit distillation from a full-attention teacher (shared teacher KV cache); Lagrangian min–max `L_distill + λ(E‖z‖₀ − N_target)` where **`E‖z‖₀ = Σ_{l>0,h} (1 − F(−p/(q−p); α, β))` in closed form** — the expected retrieval-head count is exactly controllable; λ ascends/descends on budget violation.
- 3000 steps, single A100-80G, passkey-retrieval data (BookSum + 10×32-word needles), lr 0.01, init α=β=1 (uniform). Inference: `E[z] > 0.5` → retrieval (no rounding gap — the training heatmap polarizes to 0/1 where DuoAttention's continuous gate stays grey 0.4–0.6, Fig. 8).

### 1.4 The workload-pooling kernel (Appendix C)

Naive per-head CTA assignment: sparse-head threads finish fast and idle; dense retrieval heads set the critical path. Fix: **aggregate ALL heads' block computations (full-attention blocks + each sparse head's selected blocks) into one work pool per batch item; partition into many uniform "splits"; distribute splits homogeneously across thread blocks**; per-split online-softmax accumulators `(o, m, ℓ)` + a flash-decode-style combine pass. TileLang, block 64, 90% sparsity on sparse heads: ≥6/8 sparse ratios beat FlashAttention-2 at all lengths; peak 7× at 128K batch 8 (I/O-bound regime — KV bandwidth saturated).

### 1.5 Results + load-bearing ablations

- LongBench: best avg among Quest/DuoAttention/TidalDecode/SeerAttention-R at both 1024/4096 budgets; occasionally above full attention (Qwen3-8B PRe 93.25 vs 89.08).
- Reasoning (AIME24, DeepSeek-R1-Distill-Llama-8B): 40.0 vs full-attention 23.3 — but ONLY with **cache correction** (every 32 decoded tokens, dense re-prefill of those "polluted" tokens to rebuild their KV). Without it: 26.7.
- **Ratio selection (budget ∝ seq length) generally beats top-k at equal sparsity**; top-p robust at low sparsity, collapses at extreme; threshold worst.
- **25% retrieval heads > 50%** at large budgets — more retrieval heads imports MORE noise (the denoising hypothesis inverted: selection is also filtering).
- RULER at fixed 4096 budget: near-parity at 4–8K, −6.8 avg at 64K (acceptable for a 16× smaller effective budget).

### 1.6 Limitations (paper's own)

Fixed per-head budget (no dynamic cross-head allocation — cites Ada-KV); text-only; no vLLM integration; HotpotQA-based identification has high gradient variance (short answers).

---

## 2. Distillation

### 2.1 Prior-art surface — what already ships (must not duplicate)

| Paper mechanism | Our shipped equivalent | Coverage |
|---|---|---|
| Retrieval/local head partition + offline calibration | `rt_turbo` (R086, Plan 126): `HeadCalibration`, `calibrate_from_scores` (observational needle mass), `calibrate_from_causal_scores` (R362 Plan 358, activation-patching IE — strictly stronger than observational), `CalibrationMode::{AttentionMass, CausalNecessity, AdaptiveCausal}` | ✅ Ships, GOAT 6/6 (Bench 035), 91 tests. **Unconsumed by riir-\* (stall risk per goat-audit)** |
| Dynamic token selection (top-p / ratio) | `select_top_p` / `select_top_p_blockwise` in rt_turbo; DashAttention α-entmax block routing | ✅ Ships |
| Local heads = window + sinks | PFlash (Plan 044) | ✅ Ships |
| Per-head variable block selection + periodic refresh | `flashmemory_sparse` (R436, Issue 584): per-head sigmoid ≥0.5 block selection on MLA latent-KV centroids, τ-step refresh — **per-head granularity already** (1–20 blocks/head), amortized over TIME | ✅ Ships — the ACTIVE long-context path (riir-ai Bench 671: 1.8× decode @64K, 256K Bonsai on 4090) |
| Cache correction (periodic dense repair) | FlashMemory τ-step refresh (same "periodic dense repair" family) | ⚠️ Concept ships; exact 32-token re-prefill variant untested |
| **Per-head cross-layer token-set propagation (chains)** | — | ❌ **Does not ship.** No cross-layer selection reuse anywhere (grep-verified: `propagat.*token`, `selector layer`, `cross-layer shar` → zero attention-path hits) |
| **Workload pooling for heterogeneous per-head work** | — | ❌ Does not ship. Our GPU decode kernels (attention_cubecl splitgqa, cudarc flash-decode) assign uniform per-head work |
| HardKuma near-binary gate + closed-form E‖z‖₀ | — (our gates are sigmoid thresholds; nothing trains a binary head mask) | ❌ Training-side → riir-train recipe line |

### 2.2 The gap, precisely

Our active sparse path (`flashmemory_sparse`) amortizes selection over **time** (τ steps) but recomputes per-head selection at **every layer**: per decode step, per layer, per head → score blocks against centroids, sigmoid-select. LycheeDecode's delta is the **depth** axis: head h's selection at layer l is reused by head h at l+1..(next refresh). The axes compose: selection cost ÷ (τ × chain_length). The falsifiable gate is whether our models' heads have high adjacent-layer same-index overlap — the paper says head-conditional (0–100% on Llama3), so the honest design is heterogeneous: high-overlap heads chain, low-overlap heads keep per-layer refresh. That heterogeneity is exactly what the overlap statistic decides — modelless, one calibration forward pass, extends `HeadCalibration`.

Bandwidth note (back-of-envelope, decode, n=64K, d=128/head, k=4096, w=8192): chain-sparse heads read k tokens/head; rt_turbo local heads read a w=8192 window/head regardless of need. For heads that DO need remote tokens, chains cut per-head KV read ~2× at these params and eliminate per-layer rescoring; for truly-local heads window+sinks is already minimal. Populations differ per model — which is why calibration (not a fixed rule) assigns membership.

### 2.3 Fusion — `R436 (FlashMemory) × R086/362 (calibration) × LycheeDecode`

**Per-head chain-refresh schedule on the FlashMemory substrate:** FlashMemory already produces per-head selected block sets every τ steps per layer. Add (a) the **overlap statistic** to calibration (chain affinity axis), (b) a **chain assignment** — high-overlap head-groups share one selection point (the group's "retrieval" head at layer l selects; the rest gather), refreshed on the existing τ cadence, (c) heads below the overlap threshold keep today's per-layer refresh. Selection compute drops by the average chain length at zero quality change IF overlap is high (the PoC's question). None of the three notes alone has the depth axis; the paper supplies the schedule, we supply the calibration machinery and the active substrate.

**Kernel side (riir-ai consumer):** FlashMemory's 1–20 blocks/head already creates the variable-work imbalance the paper's kernel solves. When heterogeneous budgets reach our GPU decode path (flashmemory on 4090), workload pooling (aggregate blocks across heads → uniform splits → online-softmax partials → combine) is the scheduling answer. This is a kernel-design guide line for riir-ai, not katgpt-rs code.

### 2.4 Latent-to-latent + game-context reframing (mandatory; honest)

At the transformer-KV level this is inference infra (raw KV indices), not latent-op territory. The game-level analog — "a few sentinels scan the full field; the crowd reuses the sentinels' findings" — **already ships in a stronger, decision-gated form**: CLR collective threat (`tick_swarm_emotions_collective` — reliable-scout observers propagate to non-observing NPCs), the EVPI-gated active perception gate (Issue 738 — scout only when the plausible set straddles the decision boundary, 10.6× success-per-cost), and Research 281's per-tick Speak/Silent/Delegate tri-gate. No new game capability class. This honestly kills the Super-GOAT angle.

### 2.5 Not modellessly distillable (→ riir-train)

- **HardKuma head-identification training** — needs gradient descent (distillation loss, α/β optimization). Recipe line for riir-train: IF head-identification training ever runs (DuoAttention/RAT+/MSA-style), use HardKuma gates, not continuous-then-round gates — closed-form E‖z‖₀ gives exact budget control and eliminates the train-inference rounding gap (the paper's Fig. 8: DuoAttention stays grey 0.4–0.6; HardKuma polarizes). The distribution math itself (Kuma inverse CDF + stretch + rectify, closed-form endpoint masses) is ~30 LOC if ever needed.
- **Denoising quality claims** ("sparse beats full attention") — need a quality PoC on our weights, not architectural reasoning (§3.6 discipline).

---

## 3. Verdict: **Gain**

**One-line:** head-level specialization ships; the depth-amortization axis (per-head chain propagation) and the workload-pooling kernel are real unshipped deltas on an ACTIVE path (`flashmemory_sparse` → riir-ai 256K Bonsai), but they are efficiency improvements gated behind a cheap falsifiable measurement — not a new capability class.

| Novelty gate | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **Partial** | Partition + calibration + top-p ship (rt_turbo/Plan 126/358); per-head cross-layer propagation + workload pooling do NOT (grep-verified, code + notes, both vocabularies). Published art check: the paper itself is the art for chains; ACL-2026 "token importance dynamics" and TROPE are adjacent-but-distinct. |
| Q2 New behavior class? | **No** | Efficiency on an existing capability. Game-level analog ships stronger (CLR collective + EVPI gate). |
| Q3 Product selling point? | **No** | Perf optimization of long-context decode; not a game-AI selling point. |
| Q4 Force multiplier? | **Partial** | Connects FlashMemory (R436) + rt_turbo calibration (R086/362) + riir-ai kernel work + one riir-train recipe line — but through one axis (depth amortization), not ≥2 new pillar links. |

All-4-NO on Super-GOAT → **Gain** (no "candidate" hedging). MOAT gate: katgpt-rs transformer-stack slot = attention/KV/sparse-decode — the note + any future primitive land correctly here; the kernel guide line routes to riir-ai.

**Actionable (the reason this is Gain, not Pass):** the overlap measurement is cheap, falsifiable, and decides a real design choice on the active 256K serving path. Filed as **Issue 694** (PoC: adjacent-layer same-head top-k overlap on real weights — Kimi-K3-0.40B locally, Bonsai-27B on 4090 when exclusive). Kill criteria: median overlap below ~0.5 across heads on BOTH models → chains dead, close issue, keep this note as the negative record. Pass criteria: a majority head-population above ~0.7 → a chain-mode plan on `flashmemory_sparse` (feature flag `fm_chains`, GOAT gate: G1 selection-cosine ≥ 0.95 vs per-layer refresh at equal budget — the flashmemory G1 pattern; G2 selection-compute reduction; G3 no-regression on its 169 tests; G4 alloc-free chain reuse).

**Reverse-grep hits (why not Pass):** `flashmemory_sparse`'s G5 PARTIAL ("~50% synthetic at 4K-256K / 74% real ≤4K; the paper's 90% needs real long-context attention patterns" — overview.md) is a documented quality gap that better selection (chain-refreshed, overlap-aware) could narrow; rt_turbo's GOAT Proof 4 measures routing efficiency only against dense, not against a depth-amortized variant.

> **Outcome addendum (2026-08-27, Issue 694 / [Bench 686](../.benchmarks/686_lychee_chain_overlap_poc.md)):** measured — **KILL for `fm_chains` on Kimi-K3-0.40B; no chain plan.** Two findings. (1) *Architectural:* the issue's premise was wrong — Kimi-K3-0.40B is a hybrid (`kda_layers [1,2,3,5,6,7]` linear/delta-rule, `full_attn_layers [4,8]` MLA): only 2 of 8 layers are full attention and they are NOT adjacent, so the paper's adjacent-layer mechanism is structurally inapplicable; the only hostable chain is L3→L7 (span 4). (2) *Measured:* that span-4 chain is dead — same-head top-k overlap sits at/below the k/n chance baseline at every budget (raw512 0.135 vs 0.140; raw1024 0.236 vs 0.280; raw2048 0.527 vs 0.559; block-64 variants same), same-head margin is negative for 6/8 heads, and selections cluster by layer (within-layer off-diag 0.57–0.68) not by head identity — while the same harness shows the TIME axis strongly alive (L7 step-to-step 0.753 vs 0.280 chance), the axis `flashmemory_sparse` already amortizes. Generality unproven (T3 [-]: 4090 held a sibling compute process + no attention-prob tap on the Bonsai Rust path); the paper's adjacent-layer claim on dense models is NOT refuted — reopening requires a dense full-attention model and a retrieval-capable calibration set.

---

## 4. References

- DuoAttention (Xiao et al., ICLR 2025) — continuous-gate retrieval/streaming partition (the training baseline HardKuma beats)
- TidalDecode (Yang et al., ICLR 2025b) / OmniKV (Hao et al., ICLR 2025) — layer-level sharing (what head-level sharing replaces)
- RazorAttention (Tang et al., ICLR 2025) — training-free retrieval-head KV compression (eviction family)
- Quest (Tang et al., ICML 2024) — query-aware page-level sparsity
- SeerAttention-R (Gao et al., 2025) — trainable gating network for sparse attention
- Wu et al., ICLR 2025 — "Retrieval Head Mechanistically Explains Long-Context Factuality" (the retrieval-head foundation)
- Ada-KV (Feng et al., 2024) — adaptive budget allocation across heads (the paper's stated future work)
- Bastings et al., ACL 2019 — HardKuma / differentiable binary variables (the distribution's origin)
- ACL 2026 "Leveraging Token Importance Dynamics for Efficient LLM Decoding" — importance propagation across DECODING STEPS (time axis; distinct from cross-LAYER chains)

> **Cross-ref addendum (2026-08-27):** closes nothing; opens Issue 694.
