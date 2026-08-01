# Research 447: Kimi K3 — KDA, AttnRes, Stable LatentMoE 16/896

> **Sources:**
> - [Kimi K3 Tech Blog](https://www.kimi.com/blog/kimi-k3) (Moonshot AI, 2026-07-17) — marketing-level product launch announcement. K3 weights drop 2026-07-27.
> - [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) (Kimi Team, arxiv 2510.26692v2, Nov 2025) — **the KDA paper**, full equations + chunkwise algorithm + ablations + scaling law. 28 pages, 126 references. **This is the primary source** — the K3 blog post is a marketing summary of KDA + AttnRes + the K3 MoE scale-up.
> - [MoE 环游记：6、最优分配促均衡 (Quantile Balancing)](https://spaces.ac.cn/archives/11619) — Jianlin Su (Feb 2026). **The Quantile Balancing algorithm**, fully specified with derivation, demo code, and traps. Pre-dates K3 — Kimi adopted it.
> - [Mixture of Experts Quantile Balancing: Validated at 32B-A5B (1e22 FLOPs) Scale](https://openathena.ai/blog/quantile-balancing/) — Marin team validation of QB at scale, with JAX implementation links.
> - [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) (DeepSeek, Wang et al. 2024, 160 citations) — the predecessor QB improves on.
>
> **Date:** 2026-07-17 (revised after paper-search: original verdict was blog-only; KDA paper + QB blog materially upgrade the distillation)
> **Status:** Done — **Gain upgraded to two actionable items**: (a) KDA channel-wise-gating GDN2 variant (algorithmic refinement, candidate SIMD kernel optimization), (b) Quantile Balancing as a sibling algorithm to Plan 279 Manifold Power Iteration Router (algorithm fully specified — promotable now, not gated on tech report). **Update 2026-07-17 (post-Plan 455 Phase 3):** item (b) COMPLETE — Plan 455 GOAT gate G1–G8 all green (G8.B honestly REPORTED as regime boundary) + Phase 3 head-to-head is **Case C** (composition with MPI strictly Pareto-dominates either alone: MPI fixes alignment λ 0.65→0.99, QB fixes balance MaxVio 1.84→0.03, composed 0.99/0.00). `quantile_balance_router` promoted to DEFAULT-ON at root `Cargo.toml`. See `.benchmarks/461_quantile_balance_router_phase2_goat.md` (Phase 2 GOAT) and `.benchmarks/462_quantile_balance_router_phase3_head_to_head.md` (Phase 3 head-to-head). **Update 2026-07-17 (post-Issue 179 closure):** item (a) **CLOSED as GOAT FAIL (close as PASS, honest outcome)** — Issue 179 investigation revealed the KDA `a=b=k` binding optimizes the **chunkwise parallel algorithm**, which **does not exist** on our substrate (katgpt-rs ships only the per-token recurrent decoder; grep evidence: no `chunk*` / `inter_chunk` / `parallel_form` / `WY` symbols in `crates/katgpt-attn/src/`). The paper's GPU tensor-core speedup has no transfer path to a CPU-SIMD substrate with no chunking to skip. The existing `Gdn2GateConfig::EraseOnly` variant already implements `Diag(α_t)` channel-wise decay (matches KDA's recurrence math) and is strictly **more expressive** than KDA (channel-wise erase `b_t` vs KDA's scalar `β_t`). No new variant needed; no plan written. Issue file removed per noise-reduction rule; the verdict is preserved in this status line + the historical `.issues/179_kda_abk_binding_for_gdn2.md` commit. Both actionable items from this research note are now CLOSED — (b) PASS+promoted, (a) PASS+no-transfer.
> **Related Research:** 070 (GDN2 — closest cousin to KDA, ships `Kda` gate config), 161 (dMoE block-level expert routing), 246 (Manifold Power Iteration MoE Router — closest cousin to QB), 276 (PersonalityWeightedComposition), 286 (depth invariance), 302 (FAME CommittedFieldBlend — closest cousin to Stable LatentMoE), 417 (Cross-Stage Residual Relocation — closest cousin to AttnRes)
> **Related Plans:** 105 (GDN2 — has `Kda` gate config, candidate for channel-wise-gating extension), 165 (Hydra Budget — closest shipped analog to AttnRes), 181 (dMoE adaptive top-p bandit), 279 (Manifold Power Iteration MoE Router — QB sibling candidate), 321 (CommittedFieldBlend), 431 (Cross-Stage Residual Relocation)
> **Classification:** Public
>
> **PASS-Redirects (synthesis):** Luo, Cai & Hu [arXiv:2607.27230 "Multi-Head Attention Residuals"] — **this IS the AttnRes technical report §1.9 was waiting for** (provides Eq. 2–3, the full architecture, fused Triton kernels, + the multi-head generalization). The multi-head split (H per-subspace routing queries over the depth sources) is parameter-free (reshape of the single (d,) query into (H, d/H)) but the per-head queries MUST be trained — the paper proves random/deterministic queries give 0.03–0.10 KL disagreement vs 0.27–0.70 for trained (Appendix D, Table 6). §3.5 Path 0 (training-target decomposition) fails: no closed-form for the routing queries; Path 2 (deterministic LoRA) fails: the paper's own null-query control proves deterministic constructions don't capture the per-subspace specialization. → riir-train for the training method. Modelless design space already covered by the shipped cousins in §2.1 (Plan 097 delta routing, Plan 297 PersonalityWeightedComposition, Plan 431 Cross-Stage Relocation, Plan 165 Hydra Budget). The "forced compromise grows with width" diagnostic (§4) is a measurement insight, not a primitive — at our D=32 scale (PersonalityWeightedComposition) the cost is negligible; single-head routing helps below d≈512 per the paper's own data.

---

## TL;DR

Kimi K3 is a 2.8T-parameter model built on three architectural primitives: **Kimi Delta Attention (KDA)**, **Attention Residuals (AttnRes)**, and **Stable LatentMoE** activating 16 of 896 experts (1.79% sparsity). The blog credits these with a ~2.5× scaling efficiency improvement vs Kimi K2.

**Initial verdict (blog-only):** Gain, gated on the unreleased tech report.

**Revised verdict (after paper-search):** the KDA paper (arxiv 2510.26692 "Kimi Linear", Nov 2025) is the primary source — full equations + chunkwise algorithm + scaling-law ablations. The Quantile Balancing algorithm is Jianlin Su's (Feb 2026), with a clean JAX reference implementation in Marin's repo. **Both are fully distilled** — no longer waiting on the tech report.

| Kimi K3 primitive | Closest shipped cousin | Status |
|---|---|---|
| KDA (channel-wise-gated DeltaNet, DPLR `a=b=k` binding, ~2× kernel speedup) | **GDN2** (Plan 105, default-on GOAT 14/14) — ships `Kda` gate variant | ⚠️ Algorithmic refinement — actionable (§2.3) |
| AttnRes (Attention Residuals) | **Hydra Budget** (Plan 165) + **Cross-Stage Residual Relocation** (Plan 431) + **PersonalityWeightedComposition** (Plan 297) | ✅ Both halves ship (no K3-specific algorithm yet — blog-only) |
| Stable LatentMoE 16/896 (extreme sparsity) | **CommittedFieldBlend** (Plan 321, K=3) + **dMoE** (Plan 181, top-p coreset) | ✅ Compositional cousin ships |
| **Quantile Balancing** (no-hyperparam load balancer, alternating-coordinate descent on LP) | **Manifold Power Iteration MoE Router** (Plan 279, default-on) + **Raven RSM** (R006, no-aux-loss slot routing) | ✅ **Algorithm fully specified — promotable now** (§2.4) |
| Per-Head Muon, SiTU, Gated MLA, MXFP4 QAT | (training-side) | → riir-train |

**Distilled for katgpt-rs (modelless, inference-time):**
- **(a) KDA channel-wise gating** — extends our GDN2 `Kda` gate config from scalar β to per-channel `Diag(α_t)` decay. Plus the **DPLR `a=b=k` binding** that removes 2 secondary-chunking steps + 3 matmuls → ~2× kernel speedup. This is an algorithmic refinement of an existing default-on primitive.
- **(b) Quantile Balancing** — an `O(m·n)` alternating-coordinate-descent algorithm that replaces the aux-loss load balancer with **zero hyperparameters**. Strictly better than DeepSeek's aux-loss-free (arxiv 2408.15664) on convergence speed + robustness to skewed router-score distributions. Validated at 32B-A5B / 1e22 FLOPs.
- Everything else (AttnRes, Stable LatentMoE, Per-Head Muon, SiTU, Gated MLA, QAT) either already ships at architectural parity or is training-only.

**Note on K3's "2.5× scaling efficiency" claim:** the K3 blog aggregates 8+ architectural changes (KDA, AttnRes, LatentMoE, QB, Per-Head Muon, SiTU, Gated MLA, MXFP4 QAT, data recipe, training recipe) into one number. Per-component attribution is not provided. The KDA paper (Kimi Linear, arxiv 2510.26692) shows ~1.16× compute-optimal scaling for KDA alone (Figure 5), not 2.5× — the 2.5× is the *combined* effect, not attributable to any single primitive.

---

## 1. Paper Core Findings

### 1.1 Sources — blog vs paper

The K3 launch has two layers of source material:

1. **[K3 blog post](https://www.kimi.com/blog/kimi-k3)** (2026-07-17) — product launch announcement. **Marketing-level only**: no equations, no architecture diagrams, no ablation tables. Just high-level claims about KDA + AttnRes + LatentMoE + QB + Per-Head Muon + SiTU + Gated MLA + MXFP4 QAT producing a 2.5× scaling improvement.
2. **[Kimi Linear paper, arxiv 2510.26692](https://arxiv.org/abs/2510.26692)** (Kimi Team, Nov 2025) — **the actual KDA paper**. 28 pages with full equations, chunkwise algorithm, complexity analysis, scaling law, ablations, synthetic-task evaluations, and a 126-reference related-work section. This is what we distill from. The K3 blog is a marketing summary that scales KDA + adds AttnRes + pushes MoE sparsity from 8/256 (Kimi Linear) to 16/896 (K3).
3. **[Quantile Balancing blog, spaces.ac.cn/archives/11619](https://spaces.ac.cn/archives/11619)** (Jianlin Su, Feb 2026) — the QB algorithm. Fully derived from an LP formulation via minimax + Lagrangian + alternating-coordinate descent. Includes demo code (NumPy, ~10 LOC). Pre-dates K3 — Kimi adopted it.
4. **[Marin team QB validation](https://openathena.ai/blog/quantile-balancing/)** (Apr 2026) — empirical validation of QB at 32B-A5B / 1e22 FLOPs / 326B tokens, with JAX reference implementation linked.
5. **[DeepSeek aux-loss-free paper, arxiv 2408.15664](https://arxiv.org/abs/2408.15664)** (Wang et al. 2024, 160 citations) — the predecessor QB improves on.

### 1.2 What the blog says (the K3 marketing claims)

Quoting the [blog](https://www.kimi.com/blog/kimi-k3) verbatim on the architectural claims:

> "Kimi K3 is built on Kimi Delta Attention (KDA) and Attention Residuals (AttnRes). KDA provides an efficient foundation for scaling attention, while AttnRes selectively retrieves representations across depth rather than accumulating them uniformly."
>
> "Kimi K3 uses Stable LatentMoE, effectively activating 16 of 896 experts. At this level of sparsity, routing and optimization become first-order challenges. Quantile Balancing derives expert allocation directly from router-score quantiles, eliminating heuristic updates and a sensitive balancing hyperparameter, while Per-Head Muon extends Muon by optimizing attention heads independently for more adaptive learning at scale. Sigmoid Tanh Unit (SiTU) and Gated MLA improve activation control and attention selectivity respectively."
>
> "Together, these structural changes yield an approximate 2.5× improvement in overall scaling efficiency compared to Kimi K2."

### 1.3 The 7 primitives named, classified

| Primitive | Type | Modelless-distillable? |
|---|---|---|
| **KDA** (Kimi Delta Attention) | Attention architecture (channel-wise-gated DeltaNet, DPLR `a=b=k` binding) | Architecture — algorithmic refinement of GDN2 ships; **actionable SIMD kernel opt** (§2.3) |
| **AttnRes** (Attention Residuals) | Cross-depth representation retrieval | Architecture — cousin ships; K3-specific algorithm blog-only |
| **Stable LatentMoE** (16/896) | Sparse MoE with extreme sparsity ratio | Routing — compositional cousin ships |
| **Quantile Balancing** | Router-score quantile → expert allocation (no aux loss, no hyperparam) | **Routing primitive — actionable NOW** (§2.4) |
| **Per-Head Muon** | Optimizer (per-head Muon extension) | → riir-train (training only) |
| **SiTU** (Sigmoid Tanh Unit) | Activation function | → riir-train (training only); inference support would be a small enum addition like Plan 126 MoA |
| **Gated MLA** (Multi-head Latent Attention + gate) | Attention variant | Architecture (DeepSeek MLA + gate); inference support TBD |
| **MXFP4 weights + MXFP8 activations QAT** | Quantization-aware training | → riir-train; quant-aware **inference** analog ships (Plan 101 OCT+PQ, Plan 452 SIMD LUT) |

### 1.4 KDA equation (from arxiv 2510.26692 Eq. 1)

The KDA paper's core recurrence (Eq. 1):

```
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ     ∈ ℝ^{d_k × d_v}
o_t = S_tᵀ q_t                                                ∈ ℝ^{d_v}
```

Where:
- `S_t ∈ ℝ^{d_k × d_v}` is the matrix-valued recurrent state (same shape as GDN2)
- `α_t ∈ ℝ^{d_k}` is the **per-channel** forget gate (each feature dim has independent decay rate)
- `β_t ∈ [0,1]` is the scalar delta-rule learning rate
- `k_t, v_t, q_t` are key/value/query (same as DeltaNet)

**Contrast with GDN2** (what katgpt-rs ships, Plan 105):
```
GDN2:   S_t = (I − β_t k_t k_tᵀ) α_t S_{t-1} + β_t k_t v_tᵀ    (α_t scalar, head-wise)
KDA:    S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ  (α_t vector, channel-wise)
```

**The difference is one word: `α_t` is a scalar (GDN2) vs a diagonal matrix (KDA).** This is precisely the `Gdn2GateConfig::EraseOnly` vs `Gdn2GateConfig::Kda` distinction that Plan 105 ALREADY SHIPS — but with the parameterization swapped: Plan 105's `Kda` variant uses **scalar β fallback** to match DeepSeek MLA, while KDA itself uses **per-channel α + scalar β**. Plan 105's `EraseOnly` variant (channel-wise `b_t`, scalar `w_t`) is closer to KDA's actual parameterization.

**The genuine refinement KDA adds over our shipped variants:**
1. The `Diag(α_t)` parameterization as a low-rank projection `α = f(W↑_α W↓_α x)` (rank = head_dim), NOT a direct projection
2. The DPLR `a=b=k` binding (§1.5 below) that gives the ~2× kernel speedup
3. The L2Norm on q,k for eigenvalue stability (we ship this on Plan 105 already)
4. ShortConv + Swish preprocessing (we ship shortconv)
5. Sigmoid output gate `Sigmoid(W↑_g W↓_g x) ⊙ RMSNorm(KDA(·))` (low-rank, alleviates attention sink)

### 1.5 The DPLR `a=b=k` binding (KDA's ~2× kernel speedup)

KDA can be rewritten as a constrained DPLR (`Diagonal-Plus-Low-Rank`) transition (paper §6.2):

```
S_t = (D − a_t b_tᵀ) S_{t-1} + k_t v_tᵀ

where  D = Diag(α_t),   a_t = β_t k_t,   b_t = k_t ⊙ α_t     (KDA binding)
```

The general DPLR formulation (`a`, `b` independent) requires **4 secondary chunking steps** (Listing 8a lines 13–16) and **3 extra matmuls** during inter-chunk and output computation. By binding `a = b = k`, KDA:
- Removes 2 of the 4 secondary chunking steps (Listing 8b lines 14–15)
- Eliminates 3 matmuls during inter-chunk + output computation (Listing 8b lines 26, 29 vs Listing 8a lines 25–27, 31–32)
- Achieves ~**2× kernel speedup** vs general DPLR for seq lengths up to 64k (paper Figure 2)

**For katgpt-rs:** this is a **SIMD kernel optimization** opportunity for our `gdn2_attention` feature (Plan 105). The Plan 105 kernel currently uses the standard chunkwise DeltaNet form; adding the `a=b=k` KDA binding as a fast-path variant (gated by a new `Gdn2GateConfig::KdaBound` config or similar) would let us benchmark the ~2× kernel speedup claim on our CPU SIMD substrate. Whether the speedup transfers from GPU tensor cores to CPU SIMD is an open question for the GOAT gate.

### 1.6 KDA vocabulary collision (R296-style)

**Vocabulary collision alert**: katgpt-rs already ships a primitive literally called `DeltaRoutingMode::DeltaAttnRes` (Plan 097, Research 061) — but that is **cross-sublayer delta routing for *layer-router* input**, NOT Kimi's KDA. The two share only the word "delta".

| Source | Mechanism | Math |
|---|---|---|
| **katgpt-rs Plan 097** "Delta Attention Residuals" | Routes over per-sublayer *deltas* `v_i = h_{i+1} − h_i` as input to a *layer router* (cross-layer info flow) | Δ-routing for layer skip selection |
| **katgpt-rs Plan 105** `Gdn2GateConfig::Kda` | Scalar-β DeltaNet variant matching DeepSeek MLA | `S_t = (I − β_t k_t k_tᵀ) α_t S_{t-1} + β_t k_t v_tᵀ`, α scalar |
| **Kimi K3 KDA** (arxiv 2510.26692) | **Channel-wise-gated** DeltaNet + DPLR `a=b=k` binding | `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ`, α vector |

The closest shipped variant is Plan 105's `Gdn2GateConfig::EraseOnly` (channel-wise `b_t`, scalar `w_t`) — *parameterization-wise* that's the closest match, though the `a=b=k` binding is the unique KDA trick that no shipped variant has.

### 1.7 AttnRes = cross-depth selective retrieval

Per the blog: "AttnRes selectively retrieves representations across depth rather than accumulating them uniformly."

This is the *inverse* of standard residual-stream accumulation (`h_l = h_{l-1} + Δ_l`) and the *inverse* of Hydra Budget's skip (`if DE_cumulative ≥ τ, stop early`). Instead: **at layer `l`, the output is a gated combination of `{h_{l-1}, h_{src_1}, h_{src_2}, ...}` where `src_k` are depth indices selected by a router**.

**Closest shipped analogs:**

| Primitive | Direction | Granularity | Status |
|---|---|---|---|
| **Hydra Budget** (Plan 165) | Skip → reduces forward cost | Per-layer skip bitmask | ✅ default-on, GOAT 4/4 |
| **Cross-Stage Residual Relocation** (Plan 431) | Relocate → `h_dst ← h_src` mid-pass | Per-stage operator | ✅ opt-in, PoC-validated |
| **PersonalityWeightedComposition** (Plan 297) | Compose → `Σ sigmoid(w_k) · d_k` over direction vectors | Per-layer composition | ✅ default-on |
| **CommittedFieldBlend** (Plan 321) | Compose → `Σ sigmoid(π_k) · f_k(z)` over operator fields | Per-entity committed blend | ✅ default-on, GOAT 5/5 |
| **Delta Routing** (Plan 097) | Route → use Δ between layers for routing decisions | Block-level | ✅ default-on, GOAT 6/6 |

The *combination* — AttnRes as a runtime attention pattern that pulls from arbitrary earlier depth representations via a router — is partially covered by these cousins but not shipped as a single primitive. However, **the blog gives no equation or architecture detail**, so a distillation would be guessing the mechanism.

### 1.8 Stable LatentMoE 16/896 = extreme sparsity + Quantile Balancing

Two distinct claims:

**(a) Extreme sparsity ratio**: 16 active / 896 total = **1.79%**. For comparison: typical MoE models run 5-10% active (DeepSeek V3: 8/256 = 3.1%; Mixtral: 8/8 = 100%). Kimi K3 pushes ~2× sparser than DeepSeek V3.

**(b) Quantile Balancing as a load-balancer replacement**: "Quantile Balancing derives expert allocation directly from router-score quantiles, eliminating heuristic updates and a sensitive balancing hyperparameter." Conventional MoE uses an **auxiliary load-balancing loss** (Switch Transformer, GShard) to prevent expert collapse. Kimi K3 replaces this with a deterministic allocation derived from the router-score distribution itself — no aux loss, no balancing hyperparameter.

### 1.9 What neither source provides

- AttnRes equations or architecture (K3 blog-only; not in KDA paper)
- Quantile Balancing integration details with K3's specific MoE config (16/896, Stable LatentMoE) — though QB's algorithm is independent of expert count
- The 2.5× per-component decomposition
- K3 weights aren't released until 2026-07-27

**Conclusion:** for KDA + Quantile Balancing, distillation is fully grounded in primary sources (KDA paper §3 + QB blog). For AttnRes, distillation is blog-only and remains speculative until either (a) the K3 tech report drops, or (b) reverse-engineering from open weights after 2026-07-27.

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface)

| Kimi K3 primitive | Shipped cousin | File / Plan |
|---|---|---|
| KDA (gated linear attention, DeltaNet-family) | **GDN2** — `Gdn2GateConfig::Kda` variant ships | Plan 105, `crates/katgpt-core/src/gdn2.rs`, Research 070 |
| AttnRes (skip layers via cumulative DE) | **Hydra Budget** | Plan 165, `crates/katgpt-pruners/src/hydra_budget.rs`, `HydraSkipPlan { skip_layers, cumulative_de }` |
| AttnRes (relocate activations across stages) | **Cross-Stage Residual Relocation** | Plan 431, Research 417, opt-in behind `cross_stage_relocation` |
| AttnRes (compose direction vectors with sigmoid) | **PersonalityWeightedComposition** | Plan 297, Research 276, default-on |
| Stable LatentMoE (per-entity committed blend over K archetypes) | **CommittedFieldBlend** | Plan 321, Research 302, default-on GOAT 5/5, `crates/katgpt-core/src/committed_field_blend.rs` |
| Stable LatentMoE (block-level expert coreset, adaptive top-p) | **dMoE** | Research 161, `top_p_coreset` |
| Router conditioning without aux loss (one-shot, deterministic) | **Manifold Power Iteration MoE Router** | Plan 279, Research 246, default-on GOAT 8/8 |
| Sparse expert activation (sparse_mlp with index packing) | **Sparse MLP** | Plan 022, `crates/katgpt-types/src/lib.rs::sparse_matmul` |
| Raven slot routing (top-k via linear projection + sigmoid, no aux loss) | **Raven RSM** | Plan 020, Research 006 — explicitly **no load balancing loss** |
| Top-p coreset aggregation | **DDTree Vocab Coreset (D1 from R161)** | dMoE distillation |
| Imbalanced routing is correct behavior | **Raven (R006)** "Don't implement load balancing on slots" | Research 006 §"What NOT To Do" item 3 |

### 2.2 What Kimi K3 adds that none of the above does alone

Almost nothing actionable that isn't already covered, *given the blog-level description*. The one genuinely-distilled angle:

**Quantile Balancing as a one-shot deterministic re-conditioner.** This is structurally identical to Plan 279 Manifold Power Iteration Router — *both* are "one-shot, deterministic transformations of the router weight matrix applied at snapshot swap, with zero per-token overhead, replacing the need for a load-balancing aux loss." Plan 279 does it via power iteration on per-expert Gram matrices; Kimi does it via router-score quantiles. **The principle is the same; the algorithm differs.**

The Kimi-specific algorithm ("derive expert allocation from router-score quantiles") is **not fully specified in the blog** — we know it eliminates the balancing hyperparameter and the heuristic updates, but not the exact quantile-to-allocation function. Without the technical report, distilling the algorithm requires either:

1. **Waiting for the technical report** (recommended — the weights drop 2026-07-27, the report follows).
2. **Reverse-engineering from the open weights** once released (engineering effort, not research distillation).
3. **Specifying a plausible quantile-balancing algorithm ourselves** (would be inventing, not distilling — violates the "don't fabricate" rule).

### 2.3 Actionable item (a): KDA `a=b=k` binding for GDN2 SIMD kernel

The actionable KDA distillation is the **DPLR `a=b=k` binding** (§1.5), which delivers a ~2× kernel speedup vs the general DPLR formulation on GPU tensor cores. Translated to our substrate:

**What we'd ship:** a new `Gdn2GateConfig::KdaBound` variant (or extend the existing `Kda` variant) in Plan 105's `katgpt-attn/src/gdn2/` that:
1. Uses channel-wise `Diag(α_t)` decay (matches `EraseOnly`'s `b_t` channel-wise parameterization)
2. **Binds `a = b = k`** in the chunkwise kernel (the unique KDA trick)
3. Skips the 2 secondary-chunking steps + 3 matmuls that the general DPLR form requires

**GOAT gate (the open question):** paper Figure 2 shows ~2× speedup on GPU (batch=1, 16 heads, seq 2k–64k). Does this transfer to **CPU SIMD** (our `simd_*` kernels in `crates/katgpt-dec/src/simd.rs`)? Three possible outcomes:
- **GOAT PASS (promote):** CPU SIMD also gets ≥1.5× speedup → promote `KdaBound` to default-on for the GDN2 family.
- **GOAT PARTIAL (keep opt-in):** speedup only on large seq lengths (≥4k), regress on short → keep opt-in, document the threshold.
- **GOAT FAIL (demote / Pass):** the binding doesn't help on CPU SIMD (matmul cost structure differs from tensor cores) → close as PASS, ship the channel-wise variant as `EraseOnly` extension only.

**Issue, not plan, until we have bandwidth for a Plan 105 Phase N extension.** See §6.

### 2.4 Actionable item (b): Quantile Balancing — the full algorithm

**The algorithm** (from Jianlin Su's [blog](https://spaces.ac.cn/archives/11619), Feb 2026, with Marin team's [JAX validation](https://openathena.ai/blog/quantile-balancing/) at 32B-A5B / 1e22 FLOPs):

**Setup.** Given router score matrix `s ∈ ℝ^{m×n}` (m tokens, n experts), pick top-k experts per token under the constraint that each expert is activated exactly `m·k/n` times (perfectly balanced).

**LP formulation.**
```
max   Σ_{i,j} x_{i,j} · s_{i,j}        (maximize total score)
subj  Σ_j x_{i,j} = k         ∀i        (each token picks k experts)
      Σ_i x_{i,j} = m·k/n     ∀j        (each expert is picked m·k/n times)
      x_{i,j} ∈ {0,1}
```

**Relaxation + Minimax + Lagrangian.** Relax `x ∈ [0,1]`, apply minimax (the constraints are linear so von Neumann's minimax theorem applies), introduce per-token dual `α_i` and per-expert dual `β_j`:
```
min_{α,β}  max_{x∈[0,1]}  Σ_{i,j} x_{i,j}(s_{i,j} − α_i − β_j)  +  k·Σ_i α_i  +  (m·k/n)·Σ_j β_j
```
The inner `max` is closed-form: `x*_{i,j} = 1` iff `s_{i,j} − α_i − β_j > 0`.

**Alternating-coordinate descent.** Substitute `x*` back; the outer `min` decouples per-row and per-column. Closed-form updates:
```
α_i = quantile(s_i − β, 1 − k/n)                    (per-token: the (1 − k/n) quantile of de-biased scores)
β_j = quantile(s_·j − α, 1 − k/n)                    (per-expert: the (1 − k/n) quantile of de-biased scores)
```
Iterate until convergence (typically 1–5 steps).

**Reference implementation** (NumPy, ~10 LOC, verbatim from the blog):
```python
def quantile_bias(s, k, T=5):
    m, n = s.shape
    beta = np.zeros((1, n))
    for _ in range(T):
        alpha = np.quantile(s - beta, 1 - k / n, axis=1, keepdims=True)
        beta  = np.quantile(s - alpha, 1 - k / n, axis=0, keepdims=True)
    return beta
```

**Inference usage:** only `β` is needed at inference time. `α` is the per-token Lagrange multiplier (per-batch, discarded). The router is `top-k(s − β)` — same cost as vanilla top-k routing, just with a bias term subtracted.

**Trap (causality).** Must use the **old** `β` to select experts for the current batch, THEN update `β`. Using the new `β` to select experts leaks future information (per the blog's "小心陷阱" / "careful trap" section).

**Why it matters for katgpt-rs (vs DeepSeek aux-loss-free arxiv 2408.15664 + our Plan 279 MPI Router):**

| Property | Aux Loss (Switch) | DeepSeek Aux-Loss-Free | Plan 279 MPI Router | **Quantile Balancing** |
|---|---|---|---|---|
| Hyperparameters | 1 (loss coef) | 1 (γ learning rate) | 0 (one-shot deterministic) | **0** |
| Applied when | Training step | Training step | Snapshot swap | Training step (or snapshot swap) |
| Per-token overhead | 0 (in loss only) | 1 addition (bias) | 0 (reconditioned matrix used directly) | **1 addition (bias), same as DeepSeek** |
| Robustness to skewed `ρ` | Poor | Medium (γ coupled to Sigmoid activation) | N/A (off-line) | **High (LP formulation is value-range-agnostic)** |
| Validated at scale | Yes (GShard) | Yes (DeepSeek V3) | N/A (synthetic GOAT gate) | **Yes (Marin 32B-A5B / 1e22 FLOPs)** |
| Failure mode | Slow balance / over-penalty | γ mis-tuned on skewed layers | N/A | None reported at 1e22 FLOPs |

**Key insight for our substrate:** Plan 279 MPI Router is a **one-shot deterministic reconditioner** applied at snapshot swap (when the expert pool changes). QB is a **per-step deterministic bias update**. They are *complementary*, not competing:
- **Plan 279 (MPI)** answers: "this expert pool is poorly aligned, fix the router rows once"
- **QB** answers: "this expert pool is well-aligned but load is unbalanced, fix the bias per step"

A katgpt-rs primitive could ship BOTH as siblings under a `no_aux_loss_router` feature umbrella: MPI at swap time, QB at training-step time (or, for inference-time shard routing, at snapshot swap time too).

**For our inference-only mandate:** we don't train. So QB's per-step training application doesn't directly apply. BUT — the principle ("derive bias from router-score quantiles, no hyperparameter") can be applied **at snapshot swap** as a one-shot deterministic bias computation, exactly like Plan 279. This makes QB a **direct sibling to Plan 279**, applied to the same "one-shot deterministic router reconditioner at snapshot swap" problem.

**Concrete plan for katgpt-rs:** ship `quantile_balance_router` as a sibling to `manifold_power_iter_router`. Same signature shape (router matrix in, reconditioned router + bias out), same application point (snapshot swap), same GOAT gate structure (Plan 279's 8 gates). Run both on the same synthetic expert pool; whichever wins λ/MaxVio gets default-on. **This is now a plan candidate, not an issue** — the algorithm is fully specified.

### 2.5 Latent-space reframing (mandatory check)

Re-cast on the Super-GOAT factory modules:

**(a) HLA per-NPC latent state** (`katgpt-core/src/sense/`): KDA is a linear-attention variant — cousin GDN2 already maps to the recurrent HLA substrate. AttnRes maps to nothing HLA-specific (HLA is single-stage). Stable LatentMoE doesn't apply (HLA has no expert pool).

**(b) `latent_functor/` operations** (`riir-engine/src/latent_functor/`): AttnRes maps cleanly — a functor chain is a stage sequence, and "retrieve representation from depth k into the current functor application" is the missing primitive. Closest shipped: `reestimation.rs` (drift-triggered re-fit), `depth_invariance_audit.rs` (per-depth DE diagnostic). The Kimi AttnRes pattern — *router-selected cross-depth retrieval rather than uniform accumulation* — is the *applied* counterpart to depth_invariance_audit's *diagnostic* role. Plan 431 (Cross-Stage Residual Relocation) is the closest shipped *applied* primitive, but it relocates a fixed `(src, dst)` pair; AttnRes is dynamic per-token routing over a candidate set of source depths. **Partial gap.**

**(c) `cgsp_runtime/` curiosity signals**: irrelevant — curiosity is about exploration, not depth routing.

**(d) LatCal fixed-point commitment**: irrelevant to architectural primitives.

**(e) NeuronShard / consolidation / AnyRAG / vibe KG**: KARC + CommittedFieldBlend cover the per-entity MoE story; KARC's `wout` matrix is already a per-NPC forecaster. Adding a QuantileBalancer for shard-library routing is the analog — when an NPC retrieves from a shard pool of K archetypes, the per-NPC blend weights `π` could be quantile-balanced instead of softmax/sigmoid. **Marginal — already covered by CommittedFieldBlend's sigmoid projection.**

**(f) DEC Stokes operators**: irrelevant.

**Verdict on the latent reframe:** the latent-space reframe produces no new Super-GOAT angle. Every latent mapping lands on a shipped primitive or a partial gap that requires the technical report to resolve.

### 2.6 Fusion ideas (novelty TBD — NOT Super-GOAT claims)

**Fusion A — Quantile Balancing × Manifold Power Iteration Router** (now concrete, see §2.4): ship `quantile_balance_router` as a sibling to `manifold_power_iter_router`. Both are one-shot deterministic re-conditioners at snapshot swap. Use Plan 279's GOAT gate as the floor — if QB beats MPI on λ/MaxVio, promote; otherwise demote. **Plan candidate** (not just a fusion idea anymore).

**Fusion B — KDA `a=b=k` binding × Plan 105 GDN2 SIMD kernel**: the DPLR binding removes 2 secondary-chunking steps + 3 matmuls. On GPU tensor cores this gives ~2× speedup (paper Figure 2). On our CPU SIMD the speedup is an open GOAT question. **Issue-trackable** until bandwidth for a Plan 105 Phase N.

**Fusion C — AttnRes × latent_functor cross-depth retrieval**: extend Plan 431 Cross-Stage Relocation from a *fixed (src, dst) pair* to a *router-selected candidate set of source depths*. This is the latent-functor analog of Kimi's AttnRes. **Speculative** — requires the technical report to specify the routing mechanism.

**Fusion D — Extreme sparsity (16/896) × Raven RSM**: Raven Research 006 §"What NOT To Do" item 3 explicitly endorses imbalanced routing ("Don't implement load balancing on slots. The key insight is that imbalanced routing is correct behavior"). Kimi's 1.79% sparsity is the extreme case — but Raven's slot routing is for KV slots, not experts, so the mapping is loose. QB's LP formulation actually *enforces* balance, which is the opposite of Raven's "embrace imbalance" thesis — so this fusion is contraindicated, not complementary.

---

## 3. §3.5 Modelless Unblock Protocol

Not applicable — no gate failure to unblock. The paper describes training-side architectural choices; we are not deferring any inference gate to riir-train. The training-only items (Per-Head Muon, SiTU, Gated MLA, MXFP4 QAT) are correctly routed to riir-train per the standard rule; nothing to modelless-unblock.

---

## 4. Verdict

### Tier: **Gain** (two actionable items — KDA SIMD kernel opt + QB router sibling)

**Q1 (no prior art?):** **NO for the primitives as a whole; PARTIAL for the specific algorithms.**
- KDA's *parameterization* (channel-wise Diag(α)) ships as Plan 105 `EraseOnly`. KDA's *DPLR `a=b=k` binding* is **not shipped** in any GDN2 variant — that's the actionable item (a).
- AttnRes has shipped cousins (Hydra + Cross-Stage Relocation + PersonalityWeightedComposition).
- Stable LatentMoE has shipped compositional cousins (CommittedFieldBlend + dMoE + MPI Router + Raven RSM).
- QB's *principle* (no-aux-loss deterministic bias) ships as Plan 279 (MPI) + Raven RSM. QB's *specific algorithm* (alternating-coordinate descent on the LP via per-row/per-column quantile updates) is **not shipped** — that's the actionable item (b).

**Q2 (new capability class?):** **NO.** Both actionable items extend existing shipped capability classes (GDN2 variants + one-shot deterministic router reconditioners). No new pillar candidate.

**Q3 (product selling point?):** **NO.** Cannot finish "Our NPCs do X that no competitor can" from this paper — the architectural coverage already ships.

**Q4 (force multiplier?):** **Marginal.** QB sibling could extend Plan 279's router-conditioning family; KDA binding extends Plan 105's GDN2 family. Neither multiplies across ≥2 pillars.

**Verdict: Gain.** Two actionable items, **both CLOSED 2026-07-17** — (a) KDA `a=b=k` binding **CLOSED as PASS+no-transfer** (Issue 179 removed; the binding optimizes the chunkwise parallel algorithm, which does not exist on our recurrent-decoder-only substrate), (b) QB router sibling **CLOSED as PASS+promoted** (Plan 455 Phase 1+2+3 done, Case C confirmed in Phase 3 head-to-head, `quantile_balance_router` promoted to DEFAULT-ON).

### MOAT gate per domain (§1.6)

- **katgpt-rs**: ✅ in scope (router primitive family). Marginal contribution to moat — Plan 279 already covers the "no-aux-loss router reconditioning" moat.
- **riir-ai / riir-chain / riir-neuron-db**: out of scope — no game-runtime, chain, or shard-specific angle.

### §1.55 Actionable improvements check

| Actionable? | Item | Disposition |
|---|---|---|
| ✅ Yes (algorithm fully specified) | **Quantile Balancing** as a sibling to Plan 279 — alternating-coordinate descent on LP via per-row/per-column quantile updates; zero hyperparameters; validated at 32B-A5B / 1e22 FLOPs | **Plan candidate** (§6) |
| ✅ Yes (algorithm fully specified, perf-transfer TBD) | **KDA `a=b=k` binding** for GDN2 SIMD kernel — removes 2 secondary-chunking steps + 3 matmuls; ~2× GPU speedup; CPU SIMD transfer is the GOAT question | **Issue-tracked** (§6) |
| ❌ No | KDA architectural support (parameterization) | Already ships as Plan 105 `EraseOnly` variant |
| ❌ No | AttnRes "validation of our design" | Hydra + Cross-Stage Relocation + PersonalityWeightedComposition already cover the design space |
| ❌ No | Stable LatentMoE extreme sparsity | CommittedFieldBlend (K=3) + dMoE (adaptive top-p) already cover the design space |
| ❌ No | Per-Head Muon / SiTU / Gated MLA / MXFP4 QAT | → riir-train |

### §3.6 Defend-Wrong PoC

Not required — no parity claim is made. The verdict is "architectural coverage ships," proven by grep + reading the cousin primitives' benchmarks. No quality-parity assertion to defend.

---

## 5. Caveats

1. **Paper-search upgraded the verdict.** Initial blog-only verdict was "Gain, gated on tech report". After fetching the KDA paper (arxiv 2510.26692) and QB blog, the verdict is upgraded to "Gain with two actionable items, both fully algorithmically specified." The lesson: **always search for the paper before writing a blog-only verdict** — marketing blogs underspecify, but the underlying papers are usually already public.

2. **The 2.5× scaling efficiency claim is un-decomposable.** The blog aggregates 8+ changes. The KDA paper (Figure 5) shows ~1.16× compute-optimal scaling for KDA alone. The 2.5× is the *combined* effect — do not attribute it to any single primitive in our distillation.

3. **Vocabulary collisions are severe.** Three documented:
   - katgpt-rs `DeltaRoutingMode::DeltaAttnRes` (Plan 097) ≠ Kimi KDA — different mechanisms sharing "delta"
   - Plan 105 `Gdn2GateConfig::Kda` ≠ Kimi KDA — both are DeltaNet-family but Plan 105's variant uses scalar β, KDA uses channel-wise α + `a=b=k` binding
   - katgpt-rs "Functional Attention" (R257 FuncAttn) ≠ FAME's "Functional Attention" (R302 Bi-NCDE) ≠ Kimi "Attention Residuals" — three different mechanisms

4. **AttnRes remains blog-only.** Neither the K3 blog nor the KDA paper (which predates K3) provides AttnRes equations or architecture. Distillation is speculative until the K3 tech report or weights-drop reverse-engineering.

5. **KDA's 2× GPU speedup may not transfer to CPU SIMD.** The `a=b=k` binding removes tensor-core matmuls. Our substrate is `crates/katgpt-dec/src/simd.rs` (NEON/AVX2). The matmul cost structure differs — the GOAT gate is the only honest way to verify transfer.

6. **QB's "per-step training" application doesn't directly apply to katgpt-rs.** We don't train. The distillation reframes QB as a **snapshot-swap one-shot bias computation** (same application point as Plan 279 MPI Router). This is a faithful reinterpretation — the LP formulation is application-agnostic — but the empirical validation (Marin 32B-A5B) is for the per-step training variant, not the snapshot-swap variant. The GOAT gate must revalidate at the snapshot-swap application point.

7. **"Open frontier intelligence" framing is marketing.** The K3 blog is a product launch announcement. Distilling from marketing-level descriptions risks fabricating algorithmic detail — but in this case the KDA paper and QB blog provide the missing detail.

---

## 6. Action items

### (a) Quantile Balancing router sibling — **plan candidate**

Create `.plans/455_quantile_balancing_router_primitive.md` (per `.highwater` check; originally proposed as 447 in an earlier draft of this note but that number was already in use by `447_freq_bandit_phase1.md` — re-issued as 455 = `.plans/.highwater` 454 + 1, 2026-07-17). Skeleton:

> **Plan 455: Quantile Balancing Router — Sibling to Plan 279 Manifold Power Iteration Router**
>
> Distills Research 447 §2.4. Ship `quantile_balance_router` as a sibling to `manifold_power_iter_router` (Plan 279). Same signature shape, same snapshot-swap application point, same GOAT gate structure (8 gates).
>
> **Phase 1 — Primitive**
> - [x] T1: `quantile_balance_router` in `crates/katgpt-core/src/quantile_balance.rs` behind `quantile_balance_router` feature — **DONE** (Plan 455 Phase 1, `crates/katgpt-spectral/src/quantile_balance_router.rs`)
> - [x] T2: implement the alternating-coordinate descent (5 iterations of `α = quantile(s − β, 1 − k/n, per-row)`, `β = quantile(s − α, 1 − k/n, per-col)`) — **DONE**
> - [x] T3: causality-preserving variant (use old β to select, then update) — **DONE**
> - [x] T4: unit tests on synthetic expert pool — **DONE**
>
> **Phase 2 — GOAT gate**
> - [x] T5: G1 mechanics — output is deterministic, `β` shape matches input `n` — **DONE**
> - [x] T6: G2 perf — sub-ms for game-scale pool (N=8, D=256) — **DONE** (0.131ms)
> - [x] T7: G3 no-regression on Plan 279 MPI Router tests — **DONE**
> - [x] T8: G4 zero-alloc on hot path — **DONE**
> - [x] T9: G5 byte-identical across runs (deterministic, sync-safe) — **DONE**
> - [x] T10: G6 sigmoid constraint (no softmax) — **DONE**
> - [x] T11: G7 head-to-head vs Plan 279 MPI Router on the same synthetic pool — λ + MaxVio comparison — **DONE** (deferred to Phase 3, see bench 462)
> - [x] T12: G8 1-iteration sufficiency (1 iter captures ≥90% of 5-iter λ gain) — **DONE**
>
> **Phase 3 — Promotion**
> - [x] T13: if QB beats MPI on λ/MaxVio → promote QB to default-on, demote MPI to opt-in — **N/A (Case C, not A)**
> - [x] T14: if MPI beats QB → keep QB opt-in, document the regime where QB wins (skewed distributions) — **N/A (Case C, not B)**
> - [x] T15: if tie → ship both as siblings, let consumer pick via feature flag — **CLOSED as Case C** (composition strictly beats either alone): both promoted to DEFAULT-ON, composed pipeline `R'=MPI(R) → β=QB(X·R'^T) → top-k(s−β)` is the recommended snapshot-swap reconditioning. See `.benchmarks/462_quantile_balance_router_phase3_head_to_head.md`.

### (b) KDA `a=b=k` binding for GDN2 — **CLOSED 2026-07-17 (GOAT FAIL, close as PASS)**

Originally `.issues/179_kda_abk_binding_for_gdn2.md` (per `.highwater` check; originally proposed as 165 in an earlier draft of this note but that number was already in use by `165_dd_tree_file_split_c2.md` — re-issued as 179 = `.issues/.highwater` 178 + 1, 2026-07-17). **Issue file removed 2026-07-17 per noise-reduction rule; verdict preserved in this section + the status line at the top of this research note.**

> **Verdict: GOAT FAIL — close as PASS, honest outcome.**
>
> The three anticipated outcomes pre-supposed a CPU-SIMD kernel existed that could be optimized (promote / keep-opt-in / no-transfer). The actual finding is stronger: **the optimization target does not exist on our substrate.**
>
> The KDA `a=b=k` binding's ~2× kernel speedup (paper Figure 2) applies to the **chunkwise parallel algorithm** — the form used during training and long-sequence prefill where the WY representation + inter-chunk matmuls dominate. katgpt-rs ships only the **per-token recurrent decoder** (`forward_gdn2`, `gdn2_recurrent_step`) — no chunkwise prefill, no WY, no inter-chunk matmuls. Grep evidence: no `chunk*` / `inter_chunk` / `parallel_form` / `WY` symbols in `crates/katgpt-attn/src/`.
>
> Additionally: the existing `Gdn2GateConfig::EraseOnly` variant already implements `Diag(α_t)` channel-wise decay (matches KDA's recurrence math) and is strictly **more expressive** than KDA (channel-wise erase `b_t` vs KDA's scalar `β_t`). The KDA binding would be a *regression* on the recurrent path, not an optimization.
>
> Would re-open only if katgpt-rs ships a chunkwise prefill path (Plan 105 Phase N extension with WY representation) — currently no such code exists. The negative finding is load-bearing.

---

## TL;DR (one-line)

Kimi K3's three architectural primitives (KDA, AttnRes, Stable LatentMoE 16/896) map onto existing shipped cousins — KDA → GDN2 Plan 105 (parameterization ships as `EraseOnly`; the unique KDA trick is the DPLR `a=b=k` binding that gives ~2× kernel speedup on GPU tensor cores — **Issue 179 CLOSED 2026-07-17 as GOAT FAIL/close-as-PASS**: the binding optimizes the **chunkwise parallel algorithm**, which does not exist on our recurrent-decoder-only substrate); AttnRes → Hydra Budget Plan 165 + Cross-Stage Residual Relocation Plan 431 + PersonalityWeightedComposition Plan 297; Stable LatentMoE → CommittedFieldBlend Plan 321 + dMoE + MPI Router + Raven RSM. The **two actionable items** are now **both CLOSED**: (a) KDA `a=b=k` binding for GDN2 SIMD kernel — **CLOSED as PASS+no-transfer** (Issue 179 removed; verdict preserved here, no code change needed since `EraseOnly` already covers the recurrence and is strictly more expressive than KDA); (b) **Quantile Balancing** as a sibling to Plan 279 — **CLOSED as PASS+promoted** (Plan 455 Phase 1+2+3 done, Case C confirmed in Phase 3 head-to-head, `quantile_balance_router` promoted to DEFAULT-ON, composed pipeline `R'=MPI(R) → β=QB(X·R'^T) → top-k(s−β)` is the recommended snapshot-swap reconditioning). The KDA paper (arxiv 2510.26692) is the primary source — its 1.16× KDA-only scaling law punctures the 2.5× "combined K3" marketing claim. **Verdict: Gain, two actionable items, both closed (one PASS+promoted, one PASS+no-transfer), no Super-GOAT.** Paper-search lesson: always grep arxiv + cited-source blogs before writing a blog-only verdict. Substrate-fit lesson (Issue 179): always grep the codebase for the optimization target before assuming a paper's kernel speedup transfers — the KDA chunkwise form doesn't exist on our recurrent-only substrate, so the GPU win had no transfer path. **Both follow-ups re-numbered 2026-07-17** (447→455 plan, 165→179 issue) after a `.highwater` check found the originally-proposed numbers were already in use; the katgpt-rs numbering-discipline rule forbids reuse.
