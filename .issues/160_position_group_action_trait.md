# Issue 160 — Unified `PositionGroupAction` Trait (RoPE + ALiBi + FoX + Wall)

> **Source:** [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — Zhang et al., *GRAPE* ([arXiv:2512.07805](https://arxiv.org/abs/2512.07805), ICLR 2026), §2.2 + §4.1 + Appendix E.
> **Opened:** 2026-07-17
> **Type:** Refactor / unification task (per AGENTS.md — issue, not plan)
> **Verdict that opened it:** Gain (Research 446 §3). Architectural cleanup, no Super-GOAT.

---

## TL;DR

GRAPE shows that RoPE (`SO(d)` multiplicative action), ALiBi / FoX / Wall (`GL(d+2)` unipotent lift), and NoPE (trivial `L = 0`) are all instances of one group-action family `G(n) = exp(n·ω·L)` obeying the exact relative law `G(t−s) = G(s)^T·G(t)`. Today in katgpt-rs:

- `PositionFreeCompactor` (RoPE) lives in `crates/katgpt-kv/src/still_kv/position_free.rs` — its own module, its own API (`un_rotate_keys`, `un_rotate_f32`).
- `WallDiagonalGate` (Wall = GRAPE-AP scalar instance) lives in `crates/katgpt-attn/src/diagonal_gate.rs` — different trait (`DiagonalGate`), different vocabulary (`compute_gate`, `apply`, `apply_inverse`).
- `apply_rope_phase_shift` (`crates/katgpt-attn-match/src/chunked.rs`) does the un-rotate / re-rotate dance **specifically for RoPE** — it cannot be reused for an ALiBi or FoX path.
- `GroupAction` trait (`crates/katgpt-core/src/group_invariance_probe.rs`, Research 355) exists for symmetry discovery, but is not used for position encoding.

The result: any tool that wants to be position-encoding-agnostic (KV compaction, attention matching, attention dilation) has to special-case RoPE vs Wall. GRAPE's contribution is the unified abstraction that lets all of them speak the same vocabulary.

## Deliverable

A new trait in `katgpt-core`, feature-gated `position_group_action` (opt-in):

```rust
/// A positional encoding as a one-parameter group action `G(n) = exp(n·ω·L)`.
///
/// All mainstream position encodings are instances:
/// - RoPE: multiplicative `SO(d)` action, rank-2 skew generators.
/// - ALiBi / FoX / Wall: additive `GL(d+2)` homogeneous lift, rank-1 nilpotent generators.
/// - NoPE: trivial `L = 0` (identity action).
///
/// The exact relative law `G(t−s) = G(s)^{-1}·G(t)` holds for all implementations.
pub trait PositionGroupAction {
    /// Apply `G(n)` to `x`, writing to `out`. `out.len() == x.len()`.
    fn apply_at(&self, n: f32, x: &[f32], out: &mut [f32]);

    /// Apply `G(n)^{-1}` (the group inverse) to `x`. Default: rotate by `-n`.
    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]);

    /// Dimension of the vector being acted on.
    fn dim(&self) -> usize;
}

/// Reference impls:
pub struct RopeAction { /* theta, head_dim, freqs */ }
pub struct AlibiAction { /* beta_h */ }
pub struct FoxAction { /* per-token forget gates */ }
pub struct WallAction { /* per-channel prefix sums */ }
```

Plus adapters that let `PositionFreeCompactor` and `WallDiagonalGate` be constructed via the trait.

## GOAT gate

- **G1 (correctness):** `RopeAction` produces bit-identical results to `PositionFreeCompactor` on the same `(theta, head_dim, pos)` inputs.
- **G2 (perf):** the trait dispatch overhead is `< 5ns` per call (measured against direct calls).
- **G3 (no-regression):** all existing RoPE/Wall tests pass unchanged when the trait is feature-off. When feature-on, the trait is additive — no existing path is forced through it.
- **G4 (alloc-free):** `apply_at` and `apply_inverse_at` perform 0 allocations.

## Tasks

- [x] **T1** Define `PositionGroupAction` trait in `katgpt-core/src/position_group_action.rs` (or extend `group_invariance_probe.rs`).
- [x] **T2** Implement `RopeAction` — wraps `PositionFreeCompactor`'s math. (Direct implementation of the per-pair 2D rotation; `GrapeMAction` wraps `Rank2Plane` for the general rank-2 case.)
- [x] **T3** Implement `AlibiAction`, `FoxAction`, `WallAction` (the additive family — straightforward from the nilpotent closed form). Plus `NopeAction` (trivial action) and `GrapeMAction` (GRAPE-M bridge).
- [x] **T4** GOAT gate tests G1–G4. (19 unit tests in-crate; gate results in [.benchmarks/458](../.benchmarks/458_position_group_action_goat.md).)
- [x] **T5** Add the feature gate `position_group_action` to `katgpt-core/Cargo.toml`. (Implies `grapem_rodrigues`.)
- [x] **T6** Document the math: the `G(n) = exp(n·ω·L)` unification + the homogeneous-lift construction for the additive family. (Module doc + per-impl doc comments.)
- [-] **Promotion**: deferred — no hot-path consumer today. Re-evaluate when a position-encoding-agnostic tool lands.

## Acceptance criteria

- Trait ships behind `position_group_action` feature gate.
- All four reference impls (RoPE/ALiBi/FoX/Wall) pass G1 bit-identical-to-specialized-impl tests.
- A short `.docs/` note (or extension to `phase_rotation.md`) documents the unification.

## Non-goals

- **NOT** rewriting `PositionFreeCompactor` or `WallDiagonalGate` to use the trait internally — they stay as-is for hot-path performance. The trait is a **vocabulary bridge**, not a hot-path replacement.
- **NOT** the rank-2 Rodrigues closed form — that's Issue 159.
- **NOT** GRAPE-AP vector-similarity gates — that's Issue 161.
- **NOT** composing multiplicative + additive in one transform — that's a follow-up after Issues 159 + 160 + 161 land.

## Dependencies

- Soft-depends on Issue 159 (rank-2 Rodrigues) for the multiplicative side's general implementation. The RoPE special case can be implemented directly without Issue 159; the general rank-2 case requires it.

## Cross-references

- [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — parent research note.
- [Research 431](../.research/431_Wall_Attention_Diagonal_Gate_RoPE_Replacement.md) — Wall distillation.
- [Research 355](../.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md) — `GroupAction` trait origin.
- [`crates/katgpt-kv/src/still_kv/position_free.rs`](../crates/katgpt-kv/src/still_kv/position_free.rs) — `PositionFreeCompactor` (RoPE).
- [`crates/katgpt-attn/src/diagonal_gate.rs`](../crates/katgpt-attn/src/diagonal_gate.rs) — `WallDiagonalGate`.
- [Issue 159](159_grapem_rank2_rodrigues_exponential.md) — closed-form Rodrigues (multiplicative-side generalization).
- [Issue 161](161_grape_ap_vector_similarity_gates.md) — GRAPE-AP vector gates (additive-side generalization).
