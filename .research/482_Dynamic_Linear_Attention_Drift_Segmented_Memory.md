# Research 482: Dynamic Linear Attention — Drift-Segmented Multi-State Memory

> **Source:** "Dynamic Linear Attention" ([arXiv:2606.10650](https://arxiv.org/abs/2606.10650)), Xin Wang, Hui Shen, Boyuan Zheng et al. (OSU + UMich + ByteDance Seed, Jun 2026; ICML 2026 poster)
> **Date:** 2026-08-14
> **Status:** Validated — PoC PASS ([Bench 635](../.benchmarks/635_drift_segment_goat.md), 2026-08-15: training-free drift segmentation beats fixed-LFU by +46pp needle recall on change-point streams / +75pp stationary at matched budget; feature `drift_segment` stays opt-in, promotion unblocked at next re-gate; F2 → riir-ai Issue 677, F3 → riir-neuron-db Issue 595)
> **Related Research:** 199 (Memory Caching — closest cousin, shipped as Plan 223b), 435 (Temporal Derivative Kernel — the I_t analog), 378 (HOLA capacity-W evict cache), 024 (δ-Mem + surprise write gate), 070 (GDN2), 028 (HLA), 006 (Raven slots); riir-ai 084 (MC game AI), 007 (four-tier memory), 161 (ICT branching snapshots)
> **Related Plans:** 223b (`katgpt-kv/segment_checkpoint`), 277 (temporal_deriv), 395 (HOLA), 053 (δ-Mem MSW), 105 (GDN2)
> **Cross-ref (riir-neuron-db):** `hope_compactor.rs` (Plan 321 — global lowest-J_merge pair merge, offline)
> **Classification:** Public

---

## TL;DR

DLA replaces fixed-schedule multi-state linear attention (Log-Linear's Fenwick tree, arXiv:2506.04761) with **content-adaptive segmentation**: a per-token *State Information Score* (relative drift of the token's rank-1 contribution vs the current memory state) opens a **new memory state** at semantic transitions, and a capacity-K cache **merges the adjacent pair with lowest information density** when full. For us the value is the policy, not the training: at inference the whole mechanism is a threshold gate + a greedy argmin merge — **modelless-validable** (Path 0: every component has a shipped analog). The published open lane (per prior-art search) is the **training-free post-hoc variant** — nobody has applied dynamic state merging to a pretrained/incremental memory without co-training, which is exactly the variant our substrate can test cheaply.

**Distilled for katgpt-rs (modelless, inference-time):**
A bounded multi-slot memory where **surprise opens a slot, density closes one**: slot allocation gated by a drift score (our `TemporalDerivativeKernel::surprise_norm()` on the key stream — O(d), vs the paper's O(d²) state-Frobenius delta), capacity enforced by merging the chronologically **adjacent** pair with lowest mean information score, readout via existing query-dependent gating (GRM/sigmoid — never softmax). This composes three shipped primitives (temporal_deriv × segment_checkpoint × hope_compactor's pair-merge pattern) into a capability none has alone: **adaptive-resolution memory at fixed budget**.

---

## 1. Paper Core Findings

**Mechanism 1 — Information-Aware Dynamic State Merging.** Per token: `s_t = φ(k_t)v_t^T`; score `I_t = ‖S_t − S_{t−1}‖_F / (‖S_{t−1}‖_F + ε)` (RMSNorm-stabilized). Boundary `b_t = 1[I_t ≥ τ]` (τ=0.6, robust 0.5–0.7): low-drift tokens merge into the current state; high drift appends a new state. Soft/differentiable gating during pre-training; hard segmentation at inference.

**Mechanism 2 — Capacity-Bounded Memory Modeling.** Cache `M = {(S_i, n_i, Ī_i)}, m ≤ K` (K=30, robust 20–40) in chronological order. When full: merge the adjacent pair `argmin_i (Ī_i + Ī_{i+1})/(n_i + n_{i+1})` (density = mean per-token score), then insert. Readout `o_t = Σ_i λ_{t,i} q_t^T S_i` with learned per-slot λ (linear over query, reused from Log-Linear's design).

**Theorem 3.1 + Corollary 3.2.** Block-summarization deviation is bounded by ‖q‖·Σᵢ √|Cᵢ|·√(Σ_{t∈Cᵢ}‖u_t − ūᵢ‖²) — i.e. **within-block heterogeneity dominates the error**; on non-stationary sequences (two distinct regimes) any fixed blocking that straddles the change point is strictly worse than change-point-aligned adaptive blocking. Classical echo: MRL-99-lineage histogram/change-point optimality.

**Results** (50B-token from-scratch pre-training, 4×A100; Mamba-2-780M + Gated DeltaNet-1.3B): beats Log-Linear on all 16 datasets — commonsense avg 31.0→34.2 (Mamba-2; also beats a full-attention Transformer 32.9), in-context retrieval avg 20.3→28.8 (GDN), RULER 4K S-NIAH-3 10.6→37.1 and MQ-NIAH +67% (Mamba-2 vs Log-Linear), consistent LongBench gains. Efficiency: higher throughput + lower runtime memory than Log-Linear at all batch sizes/context lengths. Ablation: merging-only (DLA(I)) already beats Log-Linear; both mechanisms add independently.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent (shipped) |
|---|---|
| State Information Score / information variation | `surprise_norm()` (temporal_deriv dual-EMA), HOLA `β·‖e‖` write magnitude, latent_functor coherence decay |
| State merging / summarize | consolidation `sleep()`, `shard_compactor`, `hope_compactor` pair merge |
| Capacity-bounded state cache | `SegmentStore::max_segments`, HOLA capacity-W, `mux_latent` budget |
| Multi-state linear attention | δ-Mem MSW (`MultiDomainMemory`), `segment_checkpoint`, Raven slots |
| Semantic transition / boundary | change point, collapse trigger, `tau_reest` threshold, band-edge onset |

### 2.2 Substrate map — what ships vs the gap

| DLA component | Closest shipped substrate | What it actually does | Gap |
|---|---|---|---|
| `I_t` drift score | `TemporalDerivativeKernel` (katgpt-types::temporal, Plan 277); HOLA score; HLA band-edge | Key-space dual-EMA ‖fast−slow‖; instantaneous write magnitude | **None** — key-space proxy is O(d) vs paper's O(d²) state delta (cheaper, already consumed by δ-Mem) |
| τ exceeded → NEW state slot | δ-Mem surprise gate (θ=0.10, **inverted polarity**: suppresses low-surprise writes into ONE state); `tau_reest`; entropy collapse→inject; ICT `snapshot_threshold` | Threshold actions suppress/re-estimate/explore/snapshot — never **allocate a memory slot** | **Full gap** — no drift-triggered state allocation anywhere |
| Capacity-K + adjacent-density merge | HOLA min-heap (**evict** min β·‖e‖); `SegmentStore` LFU (**evict**); `hope_compactor` (**global** lowest-J_merge pair merge, offline, shards) | Bounded caches everywhere, but evict-not-merge or offline-global-not-adjacent | **Full gap** — no online adjacency-restricted density-ordered merge |
| Query-weighted readout λ | GRM gating (`segment_checkpoint/gating.rs`), SSC top-k, `BanditWeighted` | Fully exists (modelless variants) | None — use sigmoid/bandit, not learned λ |
| "Fixed blocking suboptimal" | Our own record agrees: Research 199 measured constant-size beating Fenwick-log — but both are fixed | — | — |

### 2.3 The modelless adaptation (our delta over the paper)

1. **Score in key space, not state space.** The paper's `I_t` needs the d×d state delta every token (cost ≈ the update itself). Our temporal_deriv `surprise_norm()` on the key embedding is O(d) and is the same signal class (regime change ⇒ key distribution shift). For small-rank states (δ-Mem rank 8) the Frobenius form is also affordable — both arms belong in the PoC.
2. **Polarity composition.** δ-Mem already gates *writes* by surprise (suppress low-info). DLA gates *slot boundaries* by surprise (split on high). The two compose: suppress-merge when low, split when high — a two-sided surprise policy over one score.
3. **Modelless readout.** Learned λ → GRM-style sigmoid gating or `BanditWeighted` (constraint: sigmoid, never softmax).

### 2.4 Fusion (paper × shipped primitives)

- **F1 — DriftSegmentStore (katgpt-kv, this repo; Issue 652 — resolved + removed 2026-08-15, record: [Bench 635](../.benchmarks/635_drift_segment_goat.md)).** `segment_checkpoint` sibling: boundaries from `surprise_norm() ≥ τ` instead of fixed 128-token tiles; LFU evict → adjacent-density merge; readout via existing GRM gating. Bench: needle recall at matched budget vs SegmentStore-policy and single-state accumulator on synthetic change-point streams. This is the **Area-7 training-free variant** — no published prior art found.
- **F2 — per-NPC episodic belief memory (riir-ai).** `evolve_belief` already carries the temporal_deriv channel (Plan 277 Fusion F1). Game reframe: an NPC's day is non-stationary (patrol → ambush → trade); surprise opens an episodic slot, stable spans merge, K bounded per NPC at 20 Hz. Selling-point shape: "NPCs remember the moments that matter, at bounded cost" (Living World / four-tier memory, Research 007/084/161). File only after F1's PoC.
- **F3 — consolidation policy (riir-neuron-db).** Wake-buffer uniform average → density-aware adjacent merge (online cousin of `sleep_diverse`, Plan 005); `hope_compactor` could adopt adjacency restriction where shard order is meaningful. Cold-path, policy transfer only.

### 2.5 Prior-art landscape (honest, from §4 search)

- **AMD** (arXiv:2605.06946, May 2026, concurrent): learned per-token per-level decay on Log-Linear — content-adaptive *weighting*, no boundaries/merge, requires training. Narrows any "first content-adaptive multi-state" claim.
- **Titans** (arXiv:2501.00663): surprise-gated test-time memory writes — conceptual ancestor of `I_t`.
- **Classical streaming:** Min-Merge histograms / wavelet synopses (merge adjacent pair minimizing loss under budget) — the merge policy shape is ~2000s DB literature; DLA's Corollary echoes change-point optimality.
- MoM (2502.13685), SSE (2507.16577): routing/partition multi-state, not temporal segmentation. RAT (2507.04416): fixed 16-chunks.
- **Area 7 — post-hoc, training-free dynamic state merging on pretrained linear attention: nothing found.** Open lane AND our main risk: the paper's gains are co-trained (soft gating during pre-training); the training-free transfer is unproven — hence the PoC is the load-bearing gate (defend-wrong §3.6: architectural parity is proven by grep, quality parity only by head-to-head).

---

## 3. Verdict

**Tier: Gain** — research note (this file) + Issue 652 (PoC/proof task; **resolved + removed 2026-08-15** — PoC PASS via [Bench 635](../.benchmarks/635_drift_segment_goat.md), stays opt-in, F2/F3 follow-ups filed to riir-ai Issue 677 + riir-neuron-db Issue 595).

**Reasoning:** Doesn't ship + modelless-validable (Path 0: score ✓ / threshold ✓ / greedy merge ✓ / readout ✓ — all components have analogs; no riir-train deferral). Not Super-GOAT: Q1 (no prior art) is weakened — AMD is concurrent published content-adaptive multi-state, Titans owns surprise-gated writes, classical min-merge owns the merge-policy shape, and DLA itself is the published art for the composite; Q3 (selling point) is plausible ("adaptive-resolution memory at bounded cost, no training") but unproven until the PoC. Per the no-candidate rule this is filed as **fusion idea, novelty TBD** — the issue is the deciding experiment.

**MOAT gate (katgpt-rs):** In scope — generic attention/KV memory-management primitive, pure math + policy, no game/chain/shard semantics. Consumer wiring (F2/F3) routes to riir-ai / riir-neuron-db later. ✓

**Constraints honored:** sigmoid gates (never softmax) for boundary + readout; zero-alloc fixed arrays (K-bounded ring, O(K) argmin scan); feature-flagged opt-in (`drift_segment`); not UQ-bearing (recall metric, no distribution claim — conformal floor N/A); quality claims deferred to the PoC per §3.6.
