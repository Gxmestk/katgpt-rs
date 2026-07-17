# Issue 159 — Closed-form Rank-2 Rodrigues Exponential for Arbitrary Plane

> **Source:** [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — Zhang et al., *GRAPE* ([arXiv:2512.07805](https://arxiv.org/abs/2512.07805), ICLR 2026), §2.3 + Appendix I.
> **Opened:** 2026-07-17
> **Type:** Optimization / primitive task (per AGENTS.md — issue, not plan)
> **Verdict that opened it:** Gain (Research 446 §3). Engine-layer primitive, no Super-GOAT.

---

## TL;DR

GRAPE-M shows that for any rank-2 skew-symmetric generator `L = ab^T − ba^T ∈ so(d)`, the matrix exponential admits the closed-form Rodrigues formula

```
exp(L) = I + (sin s / s)·L + ((1 − cos s) / s²)·L²
```

where `s = √(αβ − γ²)`, `α=‖a‖²`, `β=‖b‖²`, `γ=a^T b`. Application `y = exp(n·ω·L)·x` is `O(d)` via two inner products — no `d×d` matrix materialization (beats LieRE's `O(d³)` `torch.matrix_exp`).

Our `phase_rotation.rs::phase_rotation_gate_into` (Plan 322) only does **scalar-broadcast 2D rotation** `out = cos(α)·a + sin(α)·b` between two named halves. It cannot express a rotation in an **arbitrary learned plane** `U = span{a, b}` where `a, b` are not orthogonal coordinate selectors. Closing this gap unlocks: learned rotation planes for HLA per-NPC personality-specific rotation (riir-ai fusion candidate §2.3 of Research 446), per-shard rotation in `MerkleFrozenEnvelope` (riir-neuron-db fusion candidate), and a principled generalization of RoPE to non-canonical bases.

## Deliverable

New primitive in `katgpt-core`, feature-gated `grapem_rodrigues` (opt-in).

```rust
/// Compute `y = exp(n·ω·L)·x` where `L = a·b^T − b·a^T` (rank-2 skew).
/// O(d) via 2 inner products. Writes to `out`; `out` may not alias `x`.
pub fn grapem_apply_into(
    a: &[f32], b: &[f32], x: &[f32], n: f32, omega: f32,
    out: &mut [f32],
) -> Result<(), GrapemError>;

/// Pre-compute plane scalars (α, β, γ, s) once; apply repeatedly with different (n, ω, x).
pub struct Rank2Plane { alpha: f32, beta: f32, gamma: f32, s: f32, /* ... */ }
impl Rank2Plane {
    pub fn new(a: &[f32], b: &[f32]) -> Self;
    pub fn apply_into(&self, x: &[f32], n: f32, omega: f32, out: &mut [f32]);
}
```

## GOAT gate

- **G1 (correctness):** bit-identical (within f32 epsilon) to materialized `expm(n·ω·L)·x` on random `(a, b, ω, n)` for `d ∈ {8, 16, 32, 64}`. Use a reference `expm` implementation (e.g., the scaling-squaring algorithm) as ground truth.
- **G2 (perf):** latency `< 2×` the existing `phase_rotation_gate_into` scalar-broadcast path at `d=8` (the HLA scale).
- **G3 (no-regression):** all existing lib tests pass; no behavior change when feature is off.
- **G4 (alloc-free):** `apply_into` performs 0 allocations after `Rank2Plane::new` (which stores only the 4 scalars, not the vectors).

## Tasks

- [x] **T1** Implement `grapem_apply_into` + `Rank2Plane` in `katgpt-core/src/grapem.rs` (or extend `phase_rotation.rs`).
- [x] **T2** Implement the reference `expm` (scaling-squaring) for test comparison — keep it test-only, not in the public API.
- [x] **T3** Write the GOAT gate tests (G1–G4 above).
- [x] **T4** Add the feature gate `grapem_rodrigues` to `katgpt-core/Cargo.toml`.
- [x] **T5** Document the math in the module doc-comment (Rodrigues formula + the `O(d)` application derivation).
- [-] **T6** GOAT verdict: promote to default-on if G1–G4 all PASS. — **DEFERRED**: G1–G4 all PASS (see [.benchmarks/457](../.benchmarks/457_grapem_rodrigues_goat.md)), but promotion is deferred because the gain is a new capability (arbitrary-plane rotation), not a perf/quality gain on an existing primitive. Re-evaluate when a concrete consumer lands.

## Acceptance criteria

- Primitive ships behind `grapem_rodrigues` feature gate.
- GOAT gate G1–G4 PASS documented in `.benchmarks/NNN_grapem_rodrigues.md`.
- If G1 FAILs (numerical drift from the closed form vs `expm`), document the regime where it fails and demote to permanent opt-in.

## Non-goals

- **NOT** learning the plane `(a, b)` — that requires training (→ riir-train). The primitive applies a user-supplied plane modellessly.
- **NOT** composing multiple planes — that's the multi-subspace GRAPE-M extension (Issue 161 follow-up).
- **NOT** integrating with `PositionFreeCompactor` (RoPE) yet — that's Issue 160 (unified trait).

## Cross-references

- [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — parent research note.
- [`crates/katgpt-core/src/phase_rotation.rs`](../crates/katgpt-core/src/phase_rotation.rs) — the existing scalar-broadcast 2D rotation that this generalizes.
- [`.research/305_Phase_Modulated_Cross_Domain_Coupling.md`](../.research/305_Phase_Modulated_Cross_Domain_Coupling.md) — phase rotation lineage.
- [`.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md`](../.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md) — `GroupAction` trait origin (sibling abstraction).
- riir-ai fusion candidate (Research 446 §2.3): per-NPC HLA learned rotation planes — separate novelty gate needed.
- riir-neuron-db fusion candidate (Research 446 §2.3): per-shard rotation plane in `MerkleFrozenEnvelope`.
