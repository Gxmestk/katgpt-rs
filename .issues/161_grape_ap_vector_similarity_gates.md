# Issue 161 — GRAPE-AP Vector-Similarity Gates (Content-Aware Path-Integral Decay)

> **Source:** [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — Zhang et al., *GRAPE* ([arXiv:2512.07805](https://arxiv.org/abs/2512.07805), ICLR 2026), §5.
> **Opened:** 2026-07-17
> **Type:** POC / primitive task (per AGENTS.md — issue, not plan)
> **Verdict that opened it:** Gain (Research 446 §3). Extends existing `WallDiagonalGate`.

---

## TL;DR

GRAPE-AP (Section 5) strictly extends Wall Attention's scalar prefix-sum gates with **vector-similarity-gated** decay. For each head `h` and decoding step `t`, the bias from key position `j` to query `t` is a path integral of edge potentials:

```
b_h(t, j) = Σ_{ℓ=j+1}^{t} ψ_h(t, ℓ)
ψ_h(t, ℓ) = α_h · g( ⟨p_{t,h}, R_ℓ·p_{ℓ,h}⟩ / d )    ≤ 0,    ℓ < t
```

where `p_{·,h}` are per-head positional embeddings (linear projection + RMSNorm of token features), `R_ℓ = exp(ℓ·J)` is a fixed commuting rotation, and `g` is monotone increasing + 1-Lipschitz (default: `g = log sigmoid`). Tokens whose positional embedding matches the query's decay slower; mismatching tokens decay faster.

**Wall Attention** (Plan 173 / Research 431, our existing `WallDiagonalGate`) is the special case where `ψ_h(t, ℓ) ≡ −θ_h · a_ℓ` (endpoint-independent edges) and the gate is a scalar per channel. GRAPE-AP makes the gate **vector** and **endpoint-dependent**.

Empirically (paper §6), GRAPE-AP beats RoPE by **+1.15 avg** on 770M FineWeb-Edu — the largest single-mechanism gain in the paper.

## Deliverable

Extension of `WallDiagonalGate` (or new sibling primitive), feature-gated `grape_ap_vector` (opt-in):

```rust
/// GRAPE-AP path-integral gate with vector positional embeddings.
///
/// For each (query_t, key_ℓ) pair, computes ψ_h(t, ℓ) = α · g(⟨p_t, R_ℓ·p_ℓ⟩/d).
/// Maintains a per-head prefix sum of ψ along the causal path.
pub struct GrapeApGate {
    head_dim: usize,
    alpha: f32,
    /// Positional embedding projection: token_features -> p_h.
    pos_proj: Vec<f32>, // [d × d] or factorized
    /// Rotation schedule R_ℓ = exp(ℓ·J) — precomputed or on-the-fly.
    rotations: RotationSchedule,
    /// Link function g (default: log_sigmoid).
    link: fn(f32) -> f32,
    /// Per-head prefix sum buffer (length L).
    prefix: Vec<f32>,
}

impl GrapeApGate {
    /// Observe a new token at position ℓ with features x_ℓ.
    /// Updates the prefix sum: prefix[ℓ] = prefix[ℓ−1] + ψ_h(t, ℓ).
    pub fn observe(&mut self, x_key: &[f32], x_query: &[f32], ell: usize);

    /// Compute the bias b_h(t, j) for all j ≤ t.
    /// Returns a slice into the prefix sum: b_h(t, j) = prefix[t] − prefix[j].
    pub fn bias_row(&self, t: usize) -> &[f32];
}
```

## GOAT gate

- **G1 (correctness):** for the special case where `ψ_h(t, ℓ)` is made endpoint-independent (e.g., `p_t` is constant), the bias reduces to the existing `WallDiagonalGate` formula. Bit-identical on that special case.
- **G2 (perf):** per-step overhead `< 1.5×` the existing `WallDiagonalGate::compute_gate_from_projection` at `d=64`.
- **G3 (no-regression):** all existing Wall tests pass unchanged.
- **G4 (alloc-free):** after `GrapeApGate::new`, no allocations in `observe` or `bias_row` (the prefix buffer is pre-sized to `L_max`).
- **G5 (dilution sanity):** on a synthetic "two-cluster" workload (queries from cluster A, keys from cluster B with mismatching positional embeddings), the bias correctly diverges from the matched-cluster case by `> 2×` the noise floor.

## Tasks

- [ ] **T1** Implement `GrapeApGate` in `katgpt-core/src/grape_ap.rs` or `katgpt-attn/src/grape_ap.rs`.
- [ ] **T2** Implement `RotationSchedule` (caches `R_ℓ = exp(ℓ·J)` lazily; uses the rank-2 Rodrigues from Issue 159 if available, else direct sin/cos).
- [ ] **T3** Default link function: `g(z) = log_sigmoid(z)` (the paper's choice).
- [ ] **T4** GOAT gate tests G1–G5.
- [ ] **T5** Add the feature gate `grape_ap_vector` to `Cargo.toml`.
- [ ] **T6** Document the math: the path-integral construction + the unipotent GL lift + the Wall special-case reduction.

## Acceptance criteria

- Primitive ships behind `grape_ap_vector` feature gate.
- GOAT gate G1–G5 PASS documented in `.benchmarks/NNN_grape_ap.md`.
- A worked example showing vector-similarity-aware decay on a synthetic two-cluster workload.

## Non-goals

- **NOT** training the positional-embedding projection — that's `→ riir-train`. The projection weights are user-supplied (modelless application).
- **NOT** integrating with `WallDiagonalGate`'s deployment path — `WallDiagonalGate` stays as-is for users who want the scalar specialization. GRAPE-AP is the strict generalization.
- **NOT** the multiplicative side — that's Issue 159 (Rodrigues) + Issue 160 (trait).

## Dependencies

- Soft-depends on Issue 159 (rank-2 Rodrigues) for the `R_ℓ` rotation schedule. A direct sin/cos fallback works if Issue 159 is not landed.
- Independent of Issue 160 (the trait can be added later).

## Risks

- **Latency risk:** the per-step `O(t)` similarity sweep could be a hot-path concern at long context. Mitigation: cap `t` window (sliding prefix sum) and benchmark the actual `L_max` we care about (HLA `d=8`, shard `d=64`, transformer `d=128`).
- **Semantic risk:** the link function `g = log_sigmoid` saturates; if the positional embeddings cluster tightly, all gates converge to the same value and the mechanism reduces to Wall. The G5 dilution-sanity test catches this.

## Cross-references

- [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — parent research note.
- [Research 431](../.research/431_Wall_Attention_Diagonal_Gate_RoPE_Replacement.md) — Wall distillation (the scalar special case).
- [Plan 173](../.plans/173_wall_attention_diagonal_gate.md) — Wall Attention plan.
- [`crates/katgpt-attn/src/diagonal_gate.rs`](../crates/katgpt-attn/src/diagonal_gate.rs) — `WallDiagonalGate` (the existing scalar instance being generalized).
- [Issue 159](159_grapem_rank2_rodrigues_exponential.md) — closed-form Rodrigues (soft dep for `R_ℓ`).
- [Issue 160](160_position_group_action_trait.md) — unified trait (independent).
