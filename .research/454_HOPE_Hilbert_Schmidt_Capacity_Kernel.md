# Research 454: HOPE — Hilbert-Schmidt Capacity Kernel + Optimal Rank-1 Parent

> **Source:** Hossein Mobahi & Peter L. Bartlett, *Hilbert Operator for Progressive Encoding (HOPE): A Mathematical Framework for Deconstructing Learned Representations in Deep Networks* — Google DeepMind / UC Berkeley — [arXiv:2607.21366](https://arxiv.org/abs/2607.21366) — 2026-07-24
> **Date:** 2026-07-24
> **Status:** Active — Super-GOAT candidate, open-primitive layer
> **Related Research:** [katgpt-rs/.research/233](233_Attention_Matching_KV_Compaction.md) (AM closest shipped compaction cousin), [katgpt-rs/.research/257](257_Functional_Attention_Spectral_Transport_Operator.md) (FUNCATTN rank-1 operator), [katgpt-rs/.research/302](302_FAME_Sampling_Invariant_Per_Entity_MoE.md) (FAME committed blend), [katgpt-rs/.research/303](303_Transolver_Physics_Attention_FUNCATTN_Predecessor.md) (rank-1 latent functor), [katgpt-rs/.research/306](306_Galerkin_Transformer_FUNCATTN_Grandparent_Predecessor.md) (closest Hilbert framing)
> **Related Plans:** [katgpt-rs/.plans/469](../.plans/469_hilbert_schmidt_capacity_kernel_primitive.md), [riir-neuron-db/.plans/321](../../riir-neuron-db/.plans/321_hope_shard_capacity_metric_compaction.md)
> **Cross-ref (riir-neuron-db):** [riir-neuron-db/.research/302](../../riir-neuron-db/.research/302_HOPE_Shard_Capacity_Metric_SuperGOAT_Guide.md) — **the Super-GOAT private guide** (selling-point owner: shard compaction pillar)
> **Cross-ref (riir-ai):** Plan 515 (swarm extraction), MAG direction mining (Plan 418), Committed Personality Runtime (Plan 336) — runtime consumer cross-refs
> **Classification:** Public — open-primitive layer (generic math, no game/chain/shard IP)

---

## TL;DR

HOPE gives us a **data-free, hyperparameter-free, scale-invariant capacity metric**
for any PH-1 (positively-homogeneous-of-degree-1, e.g. ReLU) "neuron" — modeled as a
**rank-1 Hilbert-Schmidt operator** `f_i = g_i ⊗ w_out,i` in `L²(X, P_X; ℝᶜ)` — plus a
**closed-form optimal rank-1 parent** for merging two neurons via principal eigenvector
of a rank-2 matrix `A = w_out,i (w_in,i)ᵀ + w_out,j (w_in,j)ᵀ`, and a **structural
slack/core partition** with severed cross-connections for continual learning.

The distilled open primitive is **three pieces of closed-form math**:

1. **Self-kernel** `K(i,i) = (γᵢ²+βᵢ²)·Φ(βᵢ/|γᵢ|) + βᵢ·|γᵢ|·φ(βᵢ/|γᵢ|)` (closed form
   for ReLU; requires only `(γ, β)` = pre-activation scale + shift from a BN-style
   calibration snapshot).
2. **Cross-kernel** `K(i,j) ≈ (1/π)·(√(1-ρ̂ᵢⱼ²) + (π - arccos ρ̂ᵢⱼ)·ρ̂ᵢⱼ)·√(K(i,i)·K(j,j))`
   with warped correlation `ρ̂ᵢⱼ = 2κ/(1+√(1+4κ²))` — Arc-Cosine kernel order 1.
3. **Optimal parent** direction `û = principal eigenvector of Aᵀ·A` (rank-2 SVD),
   scale `s* = (a+b·E_rem)/(2·E_rem+b)` in closed form, where `a,b` are cached
   pairwise constants and `E_rem` is the residual layer capacity.

**Distilled for katgpt-rs (modelless, inference-time):**
The math is substrate-independent. It applies to ANY rank-1 operator parameterized
by `(w_in, w_out, γ, β)` — which is exactly the shape of our `latent_functor` rank-1
operators (R123), our `CommittedFieldBlend` archetype fields (R302), and — most
urgently — our `ShardCompactor` AM compaction (Plan 233/326), whose single-query
mode has a documented **rank-1 collapse failure** (Issue 001, "AM algorithm
collapses any ensemble to rank-1 with a single query"; Plan 319 T5.6 G5 FAIL at
1.015× vs target 1.5×).

---

## 1. Paper Core Findings

### 1.1 The neuron as a rank-1 Hilbert-Schmidt operator (§3–§5)

A PH-1 neuron (ReLU, Leaky-ReLU, PReLU, linear) is the function:

```
f_i(x) = w_out,i · Ψ((w_eff_in,i)ᵀ·x + b_i)         where Ψ(c·z) = c·Ψ(z) for c ≥ 0
```

This is the outer product `f_i ≜ g_i ⊗ w_out,i` of:
- a **continuous scalar landscape** `g_i(x) = Ψ((w_eff_in,i)ᵀ·x + b_i)` in `H_in = L²(X, P_X; ℝ)`, and
- a **finite-dim output vector** `w_out,i ∈ H_out = ℝᶜ`.

So `f_i ∈ H = H_in ⊗ H_out` is a **rank-1 Hilbert-Schmidt operator**. Capacity is
`‖f_i‖_H = ‖w_out,i‖₂ · √K(i,i)`. The tensor structure factors the inner product:

```
⟨f_i, f_j⟩_H = ⟨g_i, g_j⟩_H_in · ⟨w_out,i, w_out,j⟩_ℝᶜ = K(i,j) · ⟨w_out,i, w_out,j⟩
```

**Scale invariance (the key property):** for PH-1 activations, scaling `w_eff_in,i`
by λ > 0 and `w_out,i` by 1/λ leaves `f_i` unchanged but rescales raw weights. The
Hilbert norm `‖f_i‖_H = ‖w_out,i‖₂·√K(i,i)` cancels this exactly because positive
homogeneity scales the kernel `K` by λ while `‖w_out,i‖` shrinks by 1/λ.

**Shape invariance:** capacity depends on the input space `X` **only through the
scalar kernel** `K(i,i) = E[Ψ²(y_i)]`. Fan-in width `n` inflates raw `Var(w_raw·x)`
but BN's absorbed parameters `w_eff_in,i = (γ/√(σ²+ε))·w_raw,i` cancel this — the
variance of the pre-activation `y_i` is bounded entirely by the learned scale
`γ²`. **Capacity is decoupled from input tensor dimension.**

### 1.2 Data-free surrogate distribution via BN + Maximum Entropy (§4)

For networks with Batch Normalization, HOPE constructs a **multivariate Gaussian
surrogate** `P_X = N(μ̂_x, Σ̂_x)` constrained only by the BN moving statistics
`(μ_i, σ²_i)`. Two justifications:

1. **CLT + Diaconis-Freedman:** as fan-in grows, every 1-D linear projection of
   `x` seen by a neuron converges to Gaussian, regardless of the true (non-
   Gaussian) data manifold. Neurons are "oblivious" to the data shape.
2. **Maximum Entropy:** if every linear projection of a random vector is Gaussian,
   the vector itself must be multivariate Gaussian.

Optimal surrogate parameters:
- `μ̂_x = W⁺_raw · μ_BN` (Moore-Penrose pseudo-inverse).
- `Σ̂_x` chosen to maximize differential entropy `H(x) ∝ log det(Σ_x)` subject to
  the BN variance constraints `w_raw,iᵀ Σ_x w_raw,i = σ²_i`.

For non-BN networks (LayerNorm, GroupNorm, raw networks): a one-time calibration
pass over a small unlabeled data batch measures `(μ_i, σ²_i)`, then the same
formulation applies (Appendix E footnote).

**For our codebase:** `NeuronShard` doesn't ship BN statistics directly, but it
**does** ship `style_weights[64]` + `intrinsic_dim` + `spectral_flatness` —
analogous sufficient statistics. See the riir-neuron-db guide for the bridge.

### 1.3 Closed-form ReLU kernels (§5 + Appendix E)

Under `y_i ~ N(β_i, γ²_i)`, the ReLU self-kernel reduces to a 1-D integral
independent of input dimensionality:

```
K(i,i) = (γ²_i + β²_i)·Φ(β_i/|γ_i|) + β_i·|γ_i|·φ(β_i/|γ_i|)
```

where `φ, Φ` are the standard Normal PDF/CDF. The cross-kernel requires a bivariate
normal CDF for the exact form; for large-scale networks, a zero-bias approximation
yields the **Arc-Cosine kernel order n=1** (Cho & Saul 2009):

```
K(i,j) ≈ (1/π)·(√(1-ρ̂²) + (π - arccos ρ̂)·ρ̂)·√(K(i,i)·K(j,j))
```

with warped correlation `ρ̂_ij = 2κ/(1+√(1+4κ²))`, where
`κ = (ρ_eff / (1-ρ²_eff)) · (|γ_i|/‖w_eff_in,i‖) · (|γ_j|/‖w_eff_in,j‖)`.

### 1.4 Optimal parent neuron via principal eigenvector (§7)

Merging two neurons `(i, j)` finds the rank-1 parent `f_p = s·ψ` minimizing
`J_merge = √(‖f_i - f_p‖²_H + ‖f_j - f_p‖²_H) / (E_a - ‖f_i‖ - ‖f_j‖ + ‖f_p‖)`.
The optimal direction is found in closed form via a **rank-2 eigendecomposition**
of `Aᵀ·A` where `A ≜ w_out,i · (w̃_in,i)ᵀ + w_out,j · (w̃_in,j)ᵀ`. The rank-2
subspace bypasses the ambient dimension entirely — **O(1) per pair**.

The optimal scale `s*` has a closed form: `s* = (a + b·E_rem) / (2·E_rem + b)`,
where `a = ‖f_i‖² + ‖f_j‖²`, `b = ⟨ψ, f_i + f_j⟩_H`, `E_rem = E_a - ‖f_i‖ - ‖f_j‖`.

### 1.5 Macro block eviction (§8)

For residual pathways `Y = X + F(X)`, forcing `F(X) → 0` collapses the block to
identity. Cost:

```
J_evict = Σ_l (N_active^(l) · E_active^(l)) / E_identity,    E_identity = Σ_k √(γ²_k + β²_k)
```

where `E_identity` is the RMS energy of the surviving skip pathway. This generalizes
granular cost to macro structures under a **single unified rate-distortion objective**
with distortion rate `DR = J / ΔP_init` and Dantzig greedy selection.

### 1.6 DEFT — Dispersed Elastic Fine-Tuning (§11.2)

For continual learning: rank all neurons by pruning cost `J_prune` at the percentile
threshold `P`, freeze the top-percentile as **Universal Core**, sever cross-
connections from upstream **Plastic Slack** to downstream Core (structural mask),
apply elasticity map `E ∈ {0,1}` to gradients. Provable bounds:

- **Static Initialization Shock** ≤ `|N_slack|·τ` (Theorem H.1).
- **Dynamic Decoupling** — frozen core experiences zero interference during
  fine-tuning (Theorem H.2, structural induction on layers).
- **Unified Cumulative Bound** — total distortion bounded by linear sum of slack
  capacities + merge projection errors (Corollary H.5).

H-Score on CIFAR-100 → SVHN transfer: **DEFT 65.82 ± 3.96** vs Full FT 13.88,
Head-Only 45.79, EWC 12.54, PEFT 10.18.

### 1.7 Encoding loop (§10)

Receding-horizon greedy: at each step, scan all L·N² candidate pairs, compute DR =
`J / ΔP_init` in O(1) per pair from cached scalars, execute single best action,
localized O(N) update. Initialization is O(L·N²) one-time.

---

## 2. Distillation

### 2.1 The transferable primitive

Three pieces of **closed-form math** that ship as a generic open primitive in
`katgpt-rs/crates/katgpt-core/`:

1. `relu_self_kernel(gamma, beta) -> f32` — Eq 3.
2. `relu_cross_kernel_approx(w_eff_in_i, w_eff_in_j, gamma_i, gamma_j, w_out_i, w_out_j) -> f32`
   — Eq 5, Arc-Cosine order 1 approximation.
3. `optimal_rank1_parent(w_eff_in_i, w_eff_in_j, w_out_i, w_out_j, gamma_i, gamma_j,
   beta_i, beta_j, E_rem) -> (u_hat, v_hat, s_star, K_self)` — Eq 12–14.

Plus a thin orchestration layer:

4. `hope_capacity(operator: &Rank1Operator) -> f32` — `‖f‖_H = ‖w_out‖·√K(i,i)`.
5. `hope_prune_cost(victim, layer_state) -> f32` — Eq 6 left.
6. `hope_merge_cost(pair, layer_state) -> f32` — Eq 6 right, using `optimal_rank1_parent`.
7. `hope_block_eviction_cost(block, identity_capacity) -> f32` — Eq 20.
8. `hope_greedy_step(layer_states) -> Action` — Dantzig greedy with distortion rate.

**Generic over `Rank1Operator { w_in: &[f32], w_out: &[f32], gamma: f32, beta: f32 }`** —
no shard/HLA/game semantics. The kernel math is the IP; the orchestration is thin.

### 2.2 Fusion candidates — what this primitive combines WITH

| Fusion | Combines with | What it produces | Where it lands |
|---|---|---|---|
| **HOPE × AM ShardCompactor** | `riir-neuron-db/src/shard_compactor.rs` (Plan 233) + Plan 319 wedge G5 FAIL | Fixes the AM single-query rank-1 collapse (Issue 001) — HOPE's optimal parent preserves subspace geometry instead of collapsing to mean | riir-neuron-db |
| **HOPE × CommittedFieldBlend** | `katgpt-rs/crates/katgpt-core/src/committed_field_blend.rs` (Plan 321) | Replaces the FAME sigmoid-on-dot blend computation with HOPE's Hilbert-Schmidt-optimal parent for the **commit phase**; keeps FAME's sampling-invariance for the apply phase | katgpt-rs + riir-ai |
| **HOPE × MAG Direction Mining** | `riir-ai/crates/riir-engine/src/mag_direction_mining.rs` (Plan 418) | HOPE's cross-kernel `K(i,j)` IS the redundancy signal MAG needs — mined directions that are redundant (high `K(i,j)/√(K(i,i)K(j,j))`) get merged, novel ones (low cross-kernel) are promoted | riir-ai |
| **HOPE × Sleep-Time Anticipator** | `katgpt-rs/crates/katgpt-core/src/sleep_time_anticator.rs` (Plan 334) | DEFT's slack/core partition is a modelless freeze/thaw percentile — gives Sleep-Time a principled capacity threshold for "which shards to consolidate vs freeze" | katgpt-rs + riir-ai |
| **HOPE × DEC Stokes (boundary-vs-volume)** | `katgpt-rs/crates/katgpt-core/src/dec/` | J_evict for a region = N·E/E_identity mirrors boundary-flux mass (Codifferential δ). The capacity metric is a 0-form; pruning a neuron = exterior derivative d. **Caveat (R296):** boundary wins only for d ≤ 3 — for high-dim shards (HLA, KG embeddings) the interior sum wins. | katgpt-rs |
| **HOPE × Newton-Schulz Blocked Matmul** | `katgpt-rs/.plans/421` | The rank-2 SVD in optimal_rank1_parent can use Newton-Schulz iteration for the principal eigenvector — same primitive, different application | katgpt-rs |

### 2.3 Latent-space reframing (MANDATORY per skill §1.3)

The HOPE primitive re-casts naturally onto all seven Super-GOAT factory modules:

- **HLA per-NPC latent state** (`katgpt-rs/crates/katgpt-core/src/sense/`): each HLA
  scalar (valence/arousal/desperation/calm/fear) IS a rank-1 operator — the input
  landscape `g_i` is the projection onto the i-th direction vector, the output
  vector `w_out,i` is the singleton `{1.0}`. Capacity `‖f_i‖_H = √K(i,i)` is the
  activation energy of that direction under the HLA's belief-state distribution.
  This gives a **principled, scale-invariant "direction importance"** for the
  HLA's 8-dim state — currently we have no such metric.
- **`latent_functor/`**: rank-1 operators between latent spaces (R123) are the
  k=1 special case of HOPE's framework. HOPE generalizes to rank-k via the
  Aᵀ·A eigendecomposition.
- **`cgsp_runtime/`**: curiosity signals are scalar projections; their capacity
  is `√K(i,i)`. HOPE gives a **curiosity capacity threshold** — below it, the
  direction is "slack" (exploration), above it, "core" (exploitation).
- **LatCal fixed-point commitment**: the capacity metric is a single f32 per
  neuron — trivially LatCal-committable as a raw scalar. The optimal parent's
  `u_hat, v_hat, s_star` are finite-precision and bit-identical across nodes
  (deterministic closed-form math).
- **`NeuronShard` style_weights**: 64-element direction vector. The shard IS a
  rank-1 operator `(w_in = style_weights, w_out = commitment_scalar, γ = intrinsic_dim,
  β = spectral_flatness)`. HOPE capacity = `‖commitment‖·√K(shard,shard)` gives a
  **principled shard importance** for compaction/retrieval (replaces cosine
  fidelity as the compaction quality metric).
- **DEC Stokes operators**: J_evict's `E_active/E_identity` ratio is a 0-form
  capacity / 0-form identity ratio — a scalar quotient of two scalar cochains.
  Not a Stokes-type boundary trick; it's a Hilbert-space volume quotient.

### 2.4 Connection map — what this multiplies

| Existing system | HOPE multiplier |
|---|---|
| `ShardCompactor` (Pillar 2 component) | Fixes Issue 001 rank-1 collapse; principled capacity metric replaces AM cosine fidelity |
| `CommittedFieldBlend` (R302 Super-GOAT) | Optimal parent replaces sigmoid-on-dot blend at commit time |
| `MAG Direction Mining` (Plan 418) | Cross-kernel IS the redundancy signal; principled novelty gate |
| `Sleep-Time Anticipator` (Plan 334) | DEFT percentile = principled freeze threshold |
| `Committed Personality Runtime` (Plan 336) | DEFT structural mask severs cross-connections cleanly |
| `freeze/thaw envelope` (riir-neuron-db) | Capacity-aware freeze — record `‖f_i‖_H` alongside BLAKE3 hash |
| `PersonalityWeightedComposition` (Plan 297) | HOPE rank-1 parent as a one-shot composition primitive |
| HLA direction vectors (sense/) | Principled per-direction capacity (currently no metric) |

### 2.5 Latent vs raw boundary

**Stays latent / local:**
- `K(i,i)`, `K(i,j)` kernel values — computed from `(w_eff_in, γ, β)` only.
- `u_hat`, `v_hat` parent directions — finite-precision but only meaningful in the
  shard's own latent space.
- The capacity metric `‖f_i‖_H` — a scalar, but its *interpretation* is local
  (per-shard, per-NPC).

**Crosses sync boundary as raw scalar (LatCal-committable):**
- `‖f_i‖_H` itself — one f32 per shard/direction. Bit-identical across nodes.
- The slack/core partition `E_i ∈ {0,1}` — one bit per shard.
- `s_star` optimal scale — one f32 per merge operation.

**Never crosses sync boundary:**
- The full parent direction `u_hat ∈ ℝⁿ` — too large, only locally meaningful.
- `K(i,j)` for all pairs — O(N²) — local-only cache.

### 2.6 What stays private vs open

- **Open (katgpt-rs/crates/katgpt-core/):** the kernel math (`relu_self_kernel`,
  `relu_cross_kernel_approx`, `optimal_rank1_parent`), the generic capacity +
  prune/merge/evict cost functionals, the greedy orchestration. Generic over
  `Rank1Operator`.
- **Private (riir-neuron-db):** the `NeuronShard`-to-`Rank1Operator` bridge
  (mapping `style_weights[64]` + `intrinsic_dim` + `spectral_flatness` to the
  `(w_in, w_out, γ, β)` shape), the integration into `ShardCompactor`, the
  DEFT-style slack/core partition for the freeze envelope.
- **Private (riir-ai):** runtime consumers — MAG direction mining fusion, Sleep-
  Time capacity threshold, Committed Personality structural mask, HLA per-
  direction capacity.

---

## 3. Verdict

### Tier: **Super-GOAT** (all 4 novelty-gate questions YES)

| Question | Answer | Evidence |
|---|---|---|
| **Q1 No prior art?** | **YES.** No codebase hit on `hilbert.schmidt`, `capacity_metric`, `functional_capacity`, `block_eviction` (network-compression sense), `structural.mask`, `slack.core`, `elasticity_map`. Closest cousins: AM (Plan 233) uses cosine fidelity — NOT scale-invariant, NOT data-free; FAME (R302) uses sigmoid-on-dot blend — NOT rank-1-optimal; Galerkin (R306) is Hilbert framing for *attention*, not *compression*. Rank-1 operator vocabulary ships in `latent_functor` (R123) but as *inference operators*, not as *capacity metrics*. | grep across 7 repos + read TL;DRs |
| **Q2 New class of behavior?** | **YES.** Currently we have **no principled, scale-invariant, data-free capacity metric** for shards/directions/personalities. AM cosine fidelity is scale-sensitive (Issue 001 rank-1 collapse is the symptom). FAME blend is per-entity MoE, not per-neuron importance. HOPE's capacity metric + optimal parent is a new capability: principled rank-1 merge that respects subspace geometry. | Issue 001 (AM single-query rank-1 collapse); Plan 319 T5.6 G5 FAIL |
| **Q3 Product selling point?** | **YES.** "Our NPC personality shards have a scale-invariant Hilbert-Schmidt capacity metric, so freeze/thaw cycles + sleep-time consolidation can identify which directions are load-bearing core vs plastic slack without any calibration pass — and the merge step preserves subspace geometry instead of collapsing to rank-1 like Attention-Matching does." This is a **pillar amplifier** for Pillar 2 (riir-neuron-db). | selling-point sentence constructed |
| **Q4 Force multiplier?** | **YES.** Connects ≥4 pillars/systems: riir-neuron-db ShardCompactor (fixes Issue 001), katgpt-rs CommittedFieldBlend (better parent computation), riir-ai MAG direction mining (cross-kernel as redundancy signal), riir-ai Sleep-Time Anticipator (DEFT freeze percentile), riir-ai Committed Personality Runtime (DEFT structural mask). | connection map §2.4 |

### MOAT gate (per domain, per skill §1.6)

| Domain | MOAT contribution | In scope? |
|---|---|---|
| **katgpt-rs** (public engine) | **Fundamental primitive** — closed-form Hilbert-Schmidt kernel math. Generic over `Rank1Operator`, no game/chain/shard semantics. **Promotes to default if G1–G4 PASS.** | ✅ in scope — open primitive |
| **riir-neuron-db** (private shards) | **Pillar 2 amplifier** — fixes the AM rank-1 collapse (Issue 001); gives ShardCompactor a principled capacity metric + optimal parent. **Private guide created.** | ✅ in scope — Super-GOAT guide in riir-neuron-db/.research/302 |
| **riir-ai** (private runtime) | **Cross-pillar multiplier** — MAG redundancy signal, Sleep-Time freeze threshold, Committed Personality structural mask, HLA per-direction capacity. **Cross-ref from riir-neuron-db guide.** | ✅ in scope — runtime consumers |
| **riir-chain** (private chain) | **Sync-boundary bridge** — capacity metric is a single f32, LatCal-committable. Trivial bridge. | optional — defer to consumer pull |
| **riir-train** | **NO.** HOPE compression framework is fully modelless (closed-form analytic, no training). DEFT's *gradient step* is training-only → riir-train, but the structural mask + slack/core partition is modelless and stays here. | ✗ out of scope — only DEFT's gradient update is → riir-train |

### §3.5 Modelless unblock protocol — HOPE passes Path 0

Decompose HOPE's training-target math:

| Component | Math | Modelless analog (shipped) | Status |
|---|---|---|---|
| BN statistics → Gaussian surrogate | `(μ̂_x, Σ̂_x)` from BN moving averages + MaxEnt | `NeuronShard.spectral_flatness` + `intrinsic_dim` + new `calibration_snapshot` field | needs bridge |
| ReLU self-kernel | Eq 3 closed form | None shipped — **new primitive** | new math |
| Cross-kernel (Arc-Cosine order 1) | Eq 5 closed form | None shipped — **new primitive** | new math |
| Optimal parent direction | Principal eigenvector of rank-2 `Aᵀ·A` | `Newton-Schulz` (Plan 421), `MANCE SVD caching` (Plan 427) ship principal-eigenvector primitives | compose |
| Optimal parent scale | `s* = (a+b·E_rem)/(2·E_rem+b)` closed form | None shipped — **new primitive** | new math |
| Block eviction identity mapping | `Y = X + F(X) → Y = X` | Freeze/thaw with zero residual — `MerkleFrozenEnvelope` already supports identity projection | compose |
| Structural mask (slack/core severance) | `M_ij = 0 if E_in=1 ∧ E_out=0` | None shipped — **new primitive** (DEFT's structural mask) | new |
| DEFT gradient elasticity | `g_t = E_out ⊙ ∇L_target` | Training-only — → riir-train | redirect |

**All training-target math decomposes into modelless analogs + 4 new closed-form
primitives.** HOPE compression framework is **MODELLESS-VALIDABLE** as a fusion.

Only DEFT's gradient step is genuine riir-train territory. The structural mask +
slack/core partition + capacity-aware freeze/thaw percentile (DEFT's static side)
are modelless and ship in katgpt-rs + riir-neuron-db + riir-ai.

---

## 4. Honest caveats and risks

1. **BN assumption.** HOPE's data-free path requires BN moving averages. Our
   `NeuronShard` does not ship BN statistics directly. The bridge (riir-neuron-db
   guide §3) maps `(style_weights, intrinsic_dim, spectral_flatness)` to the
   `(w_in, γ, β)` shape — but this is an **approximation**, not the paper's exact
   data-free guarantee. For LayerNorm/GroupNorm networks (most LLMs), the paper
   notes a one-time calibration pass is needed. **Risk:** the bridge may need
   empirical validation (G5 quality gate) before claiming the data-free property
   holds for shards.

2. **Cross-kernel approximation.** The Arc-Cosine order-1 form assumes zero bias
   `β ≈ 0`. The exact form requires bivariate normal CDF — computationally
   prohibitive for large N. **Risk:** the zero-bias approximation may under-
   estimate redundancy for high-bias neurons. The paper notes this is acceptable
   for "highly correlated neuron pairings" (which is what the greedy optimizer
   naturally selects).

3. **Issue 001 may not be fully fixed.** HOPE's optimal parent is principled, but
   the G5 post-compaction gate (Plan 319) tests retrieval diversity on a specific
   workload. **Risk:** the capacity metric may improve retrieval diversity but
   not pass the 1.5× threshold on every domain. The riir-neuron-db plan ships
   the G5 re-gate as the Super-GOAT confirmation step.

4. **Compute cost.** Initialization is O(L·N²) for pairwise geometry caching.
   For 1000-NPC swarms with N=64-dim shards, this is 64² = 4096 pairs per NPC
   = 4M pairs total. **Risk:** may need spatial hashing or block-diagonal
   approximation for crowd scale. The paper's locality property (Corollary C.6)
   helps — only the modified subspace is recomputed per step.

5. **DEFT is training-side.** The structural mask + slack/core partition is
   modelless, but the actual continual-learning benefit (H-Score 65.82) requires
   target fine-tuning. **Risk:** claiming "continual learning without training"
   would be dishonest — only the *partition* is modelless; the *adaptation* is
   training-side.

6. **Block eviction at the personality-branch level is speculative.** The paper
   proves it for ResNet residual blocks. We'd be applying it to dendritic LoRA
   branches / cognitive branches — a different architectural shape. **Risk:** the
   identity-mapping robustness may not hold for non-residual branch topologies.

---

## 5. Implementation priority

| Phase | Scope | Gate | Priority |
|---|---|---|---|
| 1 | Open primitive: `relu_self_kernel`, `relu_cross_kernel_approx`, `optimal_rank1_parent` in `katgpt-rs/crates/katgpt-core/src/hope/` | G1 (bit-exact vs reference), G2 (< 1µs per kernel), G4 (0 allocs) | **P0** — blocks everything |
| 2 | Orchestration: `hope_capacity`, `hope_prune_cost`, `hope_merge_cost`, `hope_block_eviction_cost`, `hope_greedy_step` | G3 (no regression on katgpt-core tests) | **P0** |
| 3 | riir-neuron-db bridge: `NeuronShard → Rank1Operator`, integrate into `ShardCompactor` as alt compaction path | G5 (post-compaction retrieval diversity ≥ 1.5× vs AM single-query, the Issue 001 failure threshold) | **P1** — Super-GOAT confirmation |
| 4 | riir-ai runtime: MAG direction mining fusion (cross-kernel as redundancy signal) | G6 (mining novelty preservation) | **P2** |
| 5 | riir-ai runtime: Committed Personality structural mask + Sleep-Time capacity threshold | G7 (personality drift bounded) | **P2** |
| 6 | DEFT gradient elasticity → riir-train | (out of scope for this workflow) | **P3** — note redirect |

---

## 6. References

- **Source paper:** [arXiv:2607.21366](https://arxiv.org/abs/2607.21366) — Mobahi & Bartlett, HOPE, 2026-07-24
- **Plan:** [katgpt-rs/.plans/469](../.plans/469_hilbert_schmidt_capacity_kernel_primitive.md)
- **Private Super-GOAT guide:** [riir-neuron-db/.research/302](../../riir-neuron-db/.research/302_HOPE_Shard_Capacity_Metric_SuperGOAT_Guide.md)
- **Closest shipped cousins:**
  - AM ShardCompactor: [katgpt-rs/.research/233](233_Attention_Matching_KV_Compaction.md), Plan 233, Plan 326
  - FAME CommittedFieldBlend: [katgpt-rs/.research/302](302_FAME_Sampling_Invariant_Per_Entity_MoE.md), Plan 321
  - Rank-1 latent functor: [katgpt-rs/.research/303](303_Transolver_Physics_Attention_FUNCATTN_Predecessor.md) §2.1
  - Galerkin Hilbert framing: [katgpt-rs/.research/306](306_Galerkin_Transformer_FUNCATTN_Grandparent_Predecessor.md)
  - Newton-Schulz principal eigenvector: [katgpt-rs/.plans/421](../.plans/421_newton_schulz_blocked_matmul.md)
  - MANCE SVD caching: [katgpt-rs/.plans/427](../.plans/427_mance_svd_caching.md)
  - Wall Attention block eviction (KV-cache sense): [katgpt-rs/.research/431](431_Wall_Attention_Diagonal_Gate_RoPE_Replacement.md) §Fusion 2
