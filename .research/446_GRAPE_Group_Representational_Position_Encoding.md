# Research 446: Group Representational Position Encoding (GRAPE)

> **Source:** Yifan Zhang, Zixiang Chen, Yifeng Liu, Zhen Qin, Huizhuo Yuan, Kangping Xu, Yang Yuan, Quanquan Gu, Andrew Chi-Chih Yao — *Group Representational Position Encoding* — [arXiv:2512.07805](https://arxiv.org/abs/2512.07805), ICLR 2026.
> **Date:** 2026-07-17
> **Status:** Done — verdict GAIN; all four actionable items CLOSED. Trilogy landed opt-in on `develop` — GRAPE-M Rodrigues (originally Issue 159, removed 2026-07-17 per noise rule; verdict in [Benchmark 457](../.benchmarks/457_grapem_rodrigues_goat.md)) + unified `PositionGroupAction` trait (Issue 160, [Benchmark 458](../.benchmarks/458_position_group_action_goat.md)) + GRAPE-AP vector gates (Issue 161, [Benchmark 459](../.benchmarks/459_grape_ap_vector_goat.md)) + GL(d+2) joint lift (Issue 163, [Benchmark 460](../.benchmarks/460_grape_joint_lift_goat.md)). All four promoted-deferred (no hot-path consumer today; re-evaluate when a transformer attention path lands).
> **Related Research:** 028 (HLA — higher-order linear attention), 070 (GDN2 — diagonal decay), 086 (RTPurbo — pre-RoPE retrieval projection), 305 (phase rotation — UFO's per-channel 2D rotation), 314 (group invariance + f-divergences), 355 (LieFlow — `GroupAction` trait), 392 (attention dilution SSMax), 431 (Wall Attention — diagonal forget gates as RoPE replacement)
> **Related Plans:** 173 (Wall Attention — `WallDiagonalGate`), 233 (attention matching / `PositionFreeCompactor`), 322 (phase rotation), 397 (HGA — RoPE-aware summarizer)
> **Classification:** Public

---

## TL;DR

GRAPE unifies **all** mainstream transformer positional encodings (RoPE, ALiBi, FoX, LieRE, Wall, NoPE) under a single group-action view `G(n) = exp(n·ω·L)`. The multiplicative family (RoPE, LieRE) lives in `SO(d)` with rank-2 skew generators `L = ab^T - ba^T`; the additive family (ALiBi, FoX, Wall) lives in a homogeneous lift of `GL(d+2)` with rank-1 nilpotent generators `A² = 0`. Both reduce to **closed-form `O(d)` kernels** (Rodrigues formula for multiplicative, `I + n·ω·A` for additive), enabling **learned rotation planes** without the `O(d³)` matrix exponential LieRE needs. Position encoding becomes a **group-action trait** — the same abstraction our `crates/katgpt-core/src/group_invariance_probe.rs::GroupAction` already ships for symmetry discovery.

**Distilled for katgpt-rs (modelless, inference-time):**

> The codebase ships RoPE (canonical coordinate pairs + log-uniform spectrum → `PositionFreeCompactor`) and Wall (scalar prefix-sum gates → `WallDiagonalGate`), but treats them as **two separate alternatives** without a shared abstraction. GRAPE's contribution that lands here is three-fold: (1) a **closed-form rank-2 Rodrigues exponential** `exp(L) = I + sin(s)/s·L + (1−cos(s))/s²·L²` for arbitrary `L = ab^T − ba^T` (generalizes `phase_rotation.rs`'s scalar 2D broadcast to arbitrary learned planes); (2) a **unified `PositionGroupAction` trait** that subsumes RoPE (`SO(d)` action) + ALiBi/FoX/Wall (`GL(d+2)` unipotent lift) as one family, letting compaction tools (`PositionFreeCompactor`, `apply_rope_phase_shift`) speak the same vocabulary as decay gates (`WallDiagonalGate`); (3) the **GRAPE-AP path-integral** form — vector-similarity-gated gates `ψ_h(t,ℓ) = α·g(⟨p_t, R_ℓ·p_ℓ⟩/d)` — strictly extends Wall's scalar gates with content-aware decay at `O(t)` per step.

---

## 1. Paper Core Findings

### 1.1 The unification — two families, one group-action view

GRAPE reframes position encoding as `G(n) = exp(n·ω·L)` for a generator `L` chosen from a Lie algebra:

| Family | Group | Generator | What it produces | Special cases |
|---|---|---|---|---|
| **Multiplicative (GRAPE-M)** | `SO(d)` | `L = ab^T − ba^T` rank-2 skew | Orthogonal rotation of `q` and `k`; norm-preserving | RoPE (canonical planes, log-uniform θ), LieRE (dense skew) |
| **Additive (GRAPE-A)** | `GL(d+2)` homogeneous lift | `A² = 0` rank-1 nilpotent | Additive logit bias `(j−i)·ω·Λ` (Q-K gated) | ALiBi (constant slope β_h), FoX (per-token forget gate f_t), Wall (per-channel prefix-sum gates) |
| **Path-integral Additive (GRAPE-AP)** | Same as GRAPE-A but endpoint-dependent factors | `ψ_h(t,ℓ) = α·g(⟨p_t, R_ℓ·p_ℓ⟩/d)` | Content-similarity-gated cumulative bias | Strict superset of GRAPE-A; subsumes FoX when edges are endpoint-independent |

**Key structural facts:**
- For GRAPE-M: `L² = −s²·P_U` where `U = span{a,b}`, `s = √(αβ − γ²)`. The minimal polynomial `λ(λ² + s²)` collapses the exponential to a closed form (Rodrigues-type):
  ```
  exp(L) = I + (sin s / s)·L + ((1 − cos s) / s²)·L²
  ```
- For GRAPE-A: `A² = 0` makes `exp(n·ω·A) = I + n·ω·A` exactly — no truncation needed.
- Both produce one-parameter subgroups `G(n+m) = G(n)·G(m)`, so `eq_i^T ek_j = q_i^T G(j−i) k_j` depends only on offset — the **exact relative law** holds for arbitrary `L`, not just canonical RoPE.
- Scoring uses `G^−⊤` on the key side (the inverse-transpose), which **cancels all multiplicative distortion** for GRAPE-A and yields a pure additive logit term `q^T k + (j−i)·ω·Λ`.

### 1.2 RoPE is recovered exactly (Proposition 3.1)

Set `d/2` mutually orthogonal vectors `{a_i}`, `b_i = J·a_i` (where `J` is the canonical 90° operator), and per-plane angles `θ_i`. The commuting MS-GRAPE generator `L_RoPE = Σ θ_i·L(a_i, J·a_i)` block-diagonalizes into per-plane rotations — **exactly** the standard RoPE map. The canonical log-uniform spectrum is one specific `θ_i` schedule; GRAPE allows learning the spectrum or tying it across heads/layers.

### 1.3 ALiBi, FoX, Wall as GRAPE-A / GRAPE-AP instances

- **ALiBi** (Eq. 4.7): the rank-1 nilpotent `A_h = −β_h·e_{d+2}·e_{d+1}^T` in `GL(d+2)` produces the `(j−i)·β_h` additive bias exactly.
- **FoX** (Appendix C): per-token forget gate `f_t ∈ (0,1]` → `a_ℓ = log(f_ℓ)` → `b_h(t,j) = Σ_{ℓ=j+1}^t log(f_ℓ) = D_{ij,h}` (FoX's `D`). The path product of unipotent factors collapses to a single `I + D_{ij}·E` because `E² = 0`.
- **Wall Attention** (Research 431 / Plan 173): per-channel prefix-sum gates `P_t = Σ log(g_u)` — this is GRAPE-AP with **endpoint-independent edges** and **per-channel** `a_ℓ` instead of per-head scalar. Wall's "rescale `q̃ = exp(P)⊙q`, `k̃ = exp(−P)⊙k`" is precisely the homogeneous-lift scoring (Eq. 4.5).

### 1.4 Closed-form rank-2 application (Section 2.3 + Appendix I)

For a single rank-2 plane, `y = G(n)·x` is `O(d)` via **two inner products**:
```
u = ⟨a, x⟩,  v = ⟨b, x⟩
y = x + f_1(n)·(a·v − b·u) + f_2(n)·[γ·(a·v + b·u) − β·a·u − α·b·v]
```
where `(α, β, γ)` are plane scalars (`α=‖a‖²`, `β=‖b‖²`, `γ=a^T b`) and `f_1, f_2` are trigonometric scalars with series guards as `s → 0`. **No matrix materialization.** This is the basis of the `O(d)` per-head cost that beats LieRE's `O(d³)` `torch.matrix_exp`.

### 1.5 GRAPE-AP path-integral: vector similarity gates

The genuinely new mechanism (Section 5): for each head `h` and endpoint `t`, define edge potentials
```
ψ_h(t, ℓ) = α_h · g(⟨p_{t,h}, R_ℓ·p_{ℓ,h}⟩ / d)    ≤ 0,    ℓ < t
```
where `p_{·,h}` are positional embeddings (linear projection + RMSNorm of token features), `R_ℓ = exp(ℓ·J)` is a fixed commuting rotation, and `g` is monotone increasing + 1-Lipschitz (e.g. `g = log sigmoid`). The bias is `b_h(t, j) = Σ_{ℓ=j+1}^t ψ_h(t, ℓ)`. This gives **content-similarity-aware decay** — tokens that match the query's positional embedding decay slower, mismatching ones faster. Wall is the special case `ψ_h(t, ℓ) ≡ −θ_h · a_ℓ` (endpoint-independent).

### 1.6 Empirical results (Section 6, FineWeb-Edu 100B, Llama arch)

- 770M models: **GRAPE-AP** beats RoPE by **+1.15 avg** (56.91 vs 55.76), beats FoX by +0.61.
- Training stability: RoPE suffers a training instability spike around 30B tokens that GRAPE-AP does not.
- All GRAPE variants compose cleanly with the KV-shift module FoX uses.
- PaTH Attention (Yang et al. 2025b) is shown to be **contractive and near-singular** (Appendix J.4) — `det(P_{j→i}) = Π(1−β_s)` decays exponentially, potentially impairing long-context. GRAPE-M is in `SO(d)` so all singular values are 1 (volume-preserving).

---

## 2. Distillation

### 2.1 Vocabulary translation (paper ↔ codebase)

| Paper term | Codebase term / shipped location |
|---|---|
| RoPE canonical planes | `PositionFreeCompactor` (`crates/katgpt-kv/src/still_kv/position_free.rs`) |
| RoPE un-rotate / re-rotate | `undo_rope` / `reapply_rope` (`crates/katgpt-kv/src/shard_kv/rope.rs`); `apply_rope_phase_shift` (`crates/katgpt-attn-match/src/chunked.rs`) |
| ALiBi additive bias | (no direct impl; GRAPE-A would be the substrate) |
| FoX per-token forget gate | (no direct impl) |
| **Wall per-channel prefix-sum gates** | **`WallDiagonalGate`** (`crates/katgpt-attn/src/diagonal_gate.rs`, Plan 173, Research 431) — exact GRAPE-AP instance |
| Multiplicative rotation (single plane) | `phase_rotation_gate_into` (`crates/katgpt-core/src/phase_rotation.rs`, Plan 322) — scalar broadcast `cos·a + sin·b`, NOT general rank-2 Rodrigues |
| `GroupAction` trait (Lie group acts on `ℝᵈ`) | `GroupAction` (`crates/katgpt-core/src/group_invariance_probe.rs`, Research 355) |
| Skew generator `L = ab^T − ba^T` | (referenced in `crates/katgpt-core/src/linalg/geometric_product.rs` notes re OFT but no impl) |
| Matrix exponential `exp(n·ω·L)` | `crates/katgpt-dec/src/nonlinear_heat_kernel.rs` uses `exp(t·L)` form for heat equation; no general Lie-group exponential |
| `exp(−β·‖x‖)` forget decay | `elasticity_gated_update.rs::neighborhood_weight` (different mechanism — RBF, not nilpotent) |

### 2.2 What ships vs what GRAPE adds

| Mechanism | Ships in katgpt-rs? | GRAPE's added value |
|---|---|---|
| RoPE (canonical planes, log-uniform θ) | ✅ `PositionFreeCompactor` | None — RoPE is a GRAPE-M special case |
| RoPE position-free compaction | ✅ `apply_rope_phase_shift`, `PositionFreeBridge` | None |
| Wall per-channel prefix-sum gates | ✅ `WallDiagonalGate` (Plan 173) | None — Wall is a GRAPE-AP instance |
| **Rank-2 Rodrigues `exp(L)` for arbitrary `L = ab^T − ba^T`** | ❌ (`phase_rotation.rs` only does scalar-broadcast 2D rotation between two named halves; not the general plane) | **NEW: closed-form `O(d)` kernel** |
| **Unified `PositionGroupAction` trait** (RoPE + ALiBi + FoX + Wall as one family) | ❌ (RoPE and Wall are separate modules with no shared abstraction) | **NEW: trait + closed-form exp for both `SO(d)` and `GL(d+2)`** |
| **GRAPE-AP vector-similarity gates** (`ψ_h(t,ℓ) = α·g(⟨p_t, R_ℓ·p_ℓ⟩/d)`) | ❌ (Wall uses key-projected scalar gates, not vector positional-embedding similarity) | **NEW: content-similarity-aware decay** |
| **Composed `GL(d+2)` block-diagonal** (rotary + additive in one transform) | ❌ (Wall *replaces* RoPE; they are not composed) | **NEW: block-diagonal joint lift** (Appendix E) |

### 2.3 Fusion candidates (cross-repo)

The latent-space reframing — required by §1 of the workflow — yields these candidates:

1. **GRAPE-M × HLA per-NPC belief state** (`riir-engine/src/hla/`): HLA already rotates between `[valence,arousal,desperation,calm]` halves via `phase_rotation_gate_into` (Plan 322). GRAPE-M's rank-2 Rodrigues with **per-NPC learned planes** would let each NPC's belief state rotate in a personality-specific plane — a modelless analog of OFT (Research 020, redirects to riir-train). **Latent-to-latent operation on HLA state.** → `riir-ai` Super-GOAT candidate, tracked separately.

2. **GRAPE-AP × Wall Attention × `group_invariance_probe`** (`katgpt-core`): use the existing `GroupAction` trait to parameterize a position-encoding family. The vector similarity `⟨p_t, R_ℓ·p_ℓ⟩` in GRAPE-AP is exactly the kind of dot-product projection the codebase already uses for HLA direction vectors — but applied to positional embeddings instead of belief state. **Latent-to-latent** (positional embedding similarity, not raw offset).

3. **GRAPE-M × `MerkleFrozenEnvelope` freeze/thaw** (`riir-neuron-db`): a frozen snapshot could carry a **learned rank-2 rotation plane** `(a, b, ω)` per shard, applied at retrieval time via the closed-form Rodrigues. This is the modelless analog of a per-shard RoPE tuning — `O(d)` per retrieval, BLAKE3-committable (the plane vectors are fixed-size `[d]` floats). **Latent-to-latent** (operates on shard `style_weights[d]`).

4. **GRAPE-A composition × LatCal commitment** (`riir-chain`): the additive logit bias `(j−i)·ω·Λ` is **already a raw scalar** — it crosses the sync boundary trivially. The homogeneous lift `GL(d+2)` produces a deterministic, committable bias that LatCal could encode without breaking the sync-boundary rule. **Raw, deterministic, sync-safe.**

### 2.4 What stays public vs private

- **Public (`katgpt-rs`):** the closed-form Rodrigues rank-2 exponential primitive, the unified `PositionGroupAction` trait, the GRAPE-AP path-integral kernel. These are generic transformer math — no game IP, no chain IP.
- **Private (`riir-ai`):** per-NPC HLA plane learning, personality-specific rotation planes fused with committed-personality runtime (Plan 336).
- **Private (`riir-neuron-db`):** per-shard rotation plane in `MerkleFrozenEnvelope`.
- **Private (`riir-chain`):** LatCal commitment of the GRAPE-A additive bias.

---

## 3. Verdict

**Verdict: GAIN.**

The unified group-action view is **architecturally illuminating** but does not produce a new capability class for our stack — RoPE already ships (`PositionFreeCompactor`) and Wall already ships (`WallDiagonalGate`); both are GRAPE special cases. The novel transferable pieces are three small leaf-clean primitives:

1. **Closed-form rank-2 Rodrigues exponential** `exp(L) = I + sin(s)/s·L + (1−cos(s))/s²·L²` — currently missing; `phase_rotation.rs` only does scalar broadcast.
2. **Unified `PositionGroupAction` trait** — currently RoPE and Wall are separate modules; GRAPE shows they are the same group-action family.
3. **GRAPE-AP vector-similarity gates** — currently Wall uses scalar key-projected gates; GRAPE-AP generalizes to vector positional-embedding similarity.

These are engine-layer improvements (MIT-appropriate, generic transformer math), not Super-GOAT territory because:
- **No new capability class** — same position-encoding function, parametrically richer.
- **No product selling point** — modelless inference; the model's position encoding is fixed by the upstream checkpoint. We don't retrain; the learned `(a, b, ω)` parameters are part of the upstream weights.
- **No force multiplier across pillars** — touches RoPE/Wall only, not HLA/functor/shard/LatCal.

### Why not PASS

Per §1.55 of the research skill — PASS requires "no actionable improvements". GRAPE produces three concrete actionable items (above). The closed-form Rodrigues is the strongest: it would let `phase_rotation.rs` generalize from scalar-broadcast 2D rotation to **arbitrary-plane rank-2 rotation** in `O(d)`, which is a real substrate gap (currently if you want a rotation in a non-canonical plane, you have to materialize the full `d×d` rotation matrix — `O(d²)` storage + `O(d²)` application).

### Why not Super-GOAT (novelty gate Q1–Q4)

| Q | Answer |
|---|---|
| Q1 No prior art? | **NO** — RoPE and Wall ship; the unified group-action view is the novel part but not a new mechanism. |
| Q2 New class of behavior? | **NO** — same position-encoding class. |
| Q3 Product selling point? | **NO** — engine-layer primitive, no game-AI moat. |
| Q4 Force multiplier? | **NO** — touches one pillar (transformer substrate), not ≥2. |

→ All 4 NO → GAIN, no Super-GOAT guide created.

### MOAT gate per domain (§1.6)

- **`katgpt-rs` (this repo):** **Strengthens moat** — closed-form Rodrigues is a fundamental / principle primitive the adoption funnel benefits from. ✅ In scope.
- **`riir-ai` fusion candidate:** HLA per-NPC learned rotation planes — a possible Super-GOAT, but needs its own novelty gate before claiming. Track separately.
- **`riir-neuron-db` fusion candidate:** per-shard rotation plane in `MerkleFrozenEnvelope` — possible Gain, track separately.

---

## 4. Actionable follow-ups — ALL CLOSED (2026-07-17)

Four issues opened (per AGENTS.md — "Create issue at .issues for poc, proof, optimization or refactor task, do not create plan"). All four landed opt-in on `develop`; promotion to default-on deferred pending a hot-path consumer. Issue files removed 2026-07-17 per noise rule; verdicts preserved in their GOAT benchmarks + git history.

1. **GRAPE-M rank-2 Rodrigues exponential** (originally Issue 159) — new primitive in `katgpt-core`, gated `grapem_rodrigues` (opt-in). `O(d)` per application via 2 inner products. Generalizes `phase_rotation.rs` from scalar broadcast to arbitrary plane. **GOAT gate:** G1 bit-identical to materialized `expm(L)` on random `(a,b,ω)`; G2 latency `< 2× phase_rotation_gate_into`; G4 zero-alloc. **Verdict:** PASS — see [Benchmark 457](../.benchmarks/457_grapem_rodrigues_goat.md).

2. **Unified `PositionGroupAction` trait** (originally Issue 160) — abstract trait in `katgpt-core`, gated `position_group_action` (opt-in). Subsuming `PositionFreeCompactor` (RoPE) + `WallDiagonalGate` (Wall) + future ALiBi/FoX under one `G(n) = exp(n·ω·L)` interface. Enables a unified `apply_phase_shift` / `un_rotate` API. **GOAT gate:** G3 no-regression (existing RoPE/Wall paths unchanged when feature off). **Verdict:** PASS — see [Benchmark 458](../.benchmarks/458_position_group_action_goat.md).

3. **GRAPE-AP vector-similarity gates** (originally Issue 161) — extension of `WallDiagonalGate` to vector positional-embedding similarity `ψ_h(t,ℓ) = α·g(⟨p_t, R_ℓ·p_ℓ⟩/d)`, gated `grape_ap_vector` (opt-in). **GOAT gate:** G2 latency overhead `< 1.5×` Wall's scalar path; G4 alloc-free after scratch init. **Verdict:** PASS — see [Benchmark 459](../.benchmarks/459_grape_ap_vector_goat.md).

4. **`GL(d+2)` block-diagonal joint lift** (Appendix E; originally Issue 163, deferred until 159+160+161 landed) — composes GRAPE-M rotary with GRAPE-A additive bias into a single block-diagonal action. Gated `grape_joint_lift` (opt-in, implies `grapem_rodrigues`). **GOAT gate:** G1 bit-identical to manual composition + relativity; G2 latency smoke; G4 alloc-free after `new`. **Verdict:** PASS — see [Benchmark 460](../.benchmarks/460_grape_joint_lift_goat.md).

### Cross-repo follow-ups (for downstream guides, not this note)

- `riir-ai`: HLA per-NPC learned rotation planes — fusion candidate, needs separate novelty gate.
- `riir-neuron-db`: per-shard rotation plane in `MerkleFrozenEnvelope` — possible Gain.
- `riir-chain`: LatCal commitment of GRAPE-A additive bias — possible Gain (the bias is already a raw scalar).

---

## 5. References

- **Paper:** [arXiv:2512.07805](https://arxiv.org/abs/2512.07805) — Zhang et al., ICLR 2026.
- **Project page:** https://github.com/model-architectures/GRAPE
- **Prior art in stack:**
  - [`crates/katgpt-kv/src/still_kv/position_free.rs`](../crates/katgpt-kv/src/still_kv/position_free.rs) — `PositionFreeCompactor` (RoPE special case).
  - [`crates/katgpt-attn/src/diagonal_gate.rs`](../crates/katgpt-attn/src/diagonal_gate.rs) — `WallDiagonalGate` (GRAPE-AP scalar instance).
  - [`crates/katgpt-core/src/phase_rotation.rs`](../crates/katgpt-core/src/phase_rotation.rs) — `phase_rotation_gate_into` (scalar-broadcast 2D rotation; precursor to rank-2 Rodrigues).
  - [`crates/katgpt-core/src/group_invariance_probe.rs`](../crates/katgpt-core/src/group_invariance_probe.rs) — `GroupAction` trait (Lie-group action abstraction).
  - [`.research/431_Wall_Attention_Diagonal_Gate_RoPE_Replacement.md`](431_Wall_Attention_Diagonal_Gate_RoPE_Replacement.md) — Wall distillation (the existing GRAPE-AP instance).
  - [`.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md`](355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md) — LieFlow distillation + `GroupAction` trait origin.
  - [`.research/305_Phase_Modulated_Cross_Domain_Coupling.md`](305_Phase_Modulated_Cross_Domain_Coupling.md) — phase rotation lineage.
- **Sibling work cited in paper:**
  - LieRE (Ostmeier et al., ICML 2025) — dense skew generator + `O(d³)` `matrix_exp`; GRAPE-M beats it with rank-2 closed form.
  - PaTH Attention (Yang et al., 2025b) — shown contractive + near-singular (Appendix J.4); GRAPE-M is volume-preserving.
  - Forgetting Transformer / FoX (Lin et al., ICLR 2025) — exact GRAPE-A instance.

## TL;DR

GRAPE unifies RoPE/ALiBi/FoX/Wall/LieRE under one group-action view `G(n) = exp(n·ω·L)`. The codebase already ships RoPE (`PositionFreeCompactor`) and Wall (`WallDiagonalGate`) as **two separate alternatives** — both are GRAPE special cases. The novel transferable pieces are three small leaf-clean primitives: (1) closed-form rank-2 Rodrigues exponential for arbitrary `L = ab^T − ba^T` (currently missing — `phase_rotation.rs` only does scalar broadcast), (2) a unified `PositionGroupAction` trait, (3) GRAPE-AP vector-similarity gates. **Verdict: GAIN** — engine-layer primitive, no Super-GOAT (no new capability class, no game-AI selling point, no force multiplier across pillars). Three follow-up `.issues/` entries identified but no plan opened. Cross-repo fusion candidates (per-NPC HLA learned planes, per-shard rotation in `MerkleFrozenEnvelope`, LatCal commitment of additive bias) are noted for separate novelty gates.
