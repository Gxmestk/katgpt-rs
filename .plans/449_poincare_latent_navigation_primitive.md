# Plan 449: Poincaré Adapter — Closed-Form Latent Navigation Primitive

**Date:** 2026-07-18
**Research:** [katgpt-rs/.research/449_SeeSE3_Poincare_Adapter_Primitive.md](../.research/449_SeeSE3_Poincare_Adapter_Primitive.md)
**Private guide:** [riir-ai/.research/319_SeeSE3_Latent_Imagination_Game_Runtime_Guide.md](../../riir-ai/.research/319_SeeSE3_Latent_Imagination_Game_Runtime_Guide.md)
**Source paper:** [arXiv:2607.14228](https://arxiv.org/abs/2607.14228) — Chen et al., *SeeSE3: Emergence of 3D Space in Vision Features* (DeepMind, 15 Jul 2026). Headline theorems: 3 (local decodability always exists), 5 (global obstruction = manifold curvature; nonlinear φ unrolls), 7 (rotation easier than translation — depth-dependence).
**Target:** `katgpt-rs/crates/katgpt-core/src/poincare.rs` (new module) + Cargo feature `poincare_navigator` (opt-in)
**Status:** Active — Phase 1 IN-FLIGHT

---

## Goal

Ship the minimal concrete prototype of the Poincaré Adapter primitive: a frozen `PoincareAdapter` Pod holding the offline-fit triple `(φ, W, W†)`, plus the closed-form navigator `poincare_navigate_into(z_src, delta_target, adapter, z_out)` and the multi-step variant `poincare_multi_step_into`. Prove the Super-GOAT candidate (Research 449) via 7 GOAT gates (G1–G7) measuring local decodability, global unrolling, inverse-navigation round-trip, zero-alloc steady state, sub-µs latency, multi-step coherence, and latent-vs-raw boundary.

The primitive is the **adoption hook** for the private game-runtime selling point in `riir-ai/.research/319` — NPCs imagine observations at unvisited positions. That selling point is private IP; this plan ships only the generic math.

---

## Phase 1 — Skeleton: PoincareAdapter Pod + navigator fns (CORE)

### Tasks

- [ ] **T1.1** Define `PoincareAdapter` Pod in `crates/katgpt-core/src/poincare.rs`:
  - `#[repr(C)]` struct: `{ phi_w1: [f32; PHID1], phi_b1: [f32; PHIHID], phi_w2: [f32; PHID2], phi_b2: [f32; PHIOUT], W: [f32; TARGET_DIM * PHIOUT], W_pinv: [f32; PHIOUT * TARGET_DIM], target_dim: u8, phi_hidden: u16, phi_out: u16, blake3: [u8; 32] }`
  - Constants: `PHIHID = 64`, `PHIOUT = 20`, `TARGET_DIM_LE_MAX = 8` (target space ≤ 8D).
  - `bytemuck Pod + Zeroable`. BLAKE3 commitment over all weights.
  - Constructors: `new_from_fit(...) -> Result<Self, FitError>` (offline ridge + closed-form φ), `from_bytes(...)`, `canonical_bytes() -> Vec<u8>` (for `MerkleFrozenEnvelope`).
  - Validation: roundtrip serialize/deserialize; BLAKE3 verify.

- [ ] **T1.2** Implement `poincare_navigate_into(z_src: &[f32], delta_target: &[f32], adapter: &PoincareAdapter, z_out: &mut [f32])`:
  - Compute `g_src = phi(z_src)` (2-layer MLP, SIMD via existing `simd_dot_f32`).
  - Compute `g_dest = g_src + W_pinv · delta_target` (one matvec, `TARGET_DIM_LE_MAX × PHIOUT`).
  - If `phi` is invertible: `z_out = phi_inv(g_dest)` (closed-form for PCA-tanh φ; for general φ, leave `z_out = g_dest` in chart space and document the consumer-side retrieval pattern).
  - Zero-allocation. Single function call. ≤ 1µs SIMD target.

- [ ] **T1.3** Implement `poincare_multi_step_into(z_src, delta_target, n_steps, adapter, z_out)`:
  - Split `delta_target / n_steps`. Iterate `poincare_navigate_into(z, delta_target/n_steps, adapter, z)` `n_steps` times, writing to `z_out`.
  - Open-loop integrator. No correction.
  - Tests: 4-step path stays within chart valid region; deterministic given seed.

- [ ] **T1.4** Implement offline fit helpers (modelless, no gradient descent):
  - `fit_poincare_adapter(z_pairs: &[(&[f32], &[f32])], target_pairs: &[(&[f32], &[f32])], config: FitConfig) -> Result<PoincareAdapter, FitError>`
  - **Closed-form ridge for W**: `W = (ZᵀZ + αI)⁻¹·ZᵀY` where Z = stacked `φ(z₂) − φ(z₁)`, Y = stacked `target₂ − target₁`. α = 1.0 default (matches paper).
  - **Deterministic closed-form φ**: PCA on the z-pairs (top-`PHIOUT` components via existing `thin_svd_into` from Plan 301), then `tanh` activation. This is the modelless-unblock path per research skill §3.5 — gradient φ fit is a riir-train follow-up IF G2 fails.
  - **Pseudo-inverse**: `W_pinv = Wᵀ(W·Wᵀ)⁻¹` (rank-`TARGET_DIM` assumption).
  - Tests: round-trip on a known linear map (identity + random projection) achieves R² > 0.95.

- [ ] **T1.5** Wire Cargo feature `poincare_navigator` (opt-in) in `crates/katgpt-core/Cargo.toml`:
  - `[features] poincare_navigator = ["dep:blake3", "dep:bytemuck"]` (likely already transitively enabled).
  - Export from `crates/katgpt-core/src/lib.rs`: `pub mod poincare; pub use poincare::{PoincareAdapter, poincare_navigate_into, poincare_multi_step_into, fit_poincare_adapter, FitConfig, FitError};`

- [ ] **T1.6** Update `katgpt-rs/README.md` Feature Showcase with a new entry for Plan 449 (Poincaré Adapter). Update `.docs/` index if a relevant folder exists.

---

## Phase 2 — GOAT gate G1–G7 (PROOF)

Bench: `crates/katgpt-core/benches/bench_449_poincare_goat.rs`. Run with `cargo bench -p katgpt-core --features poincare_navigator --bench bench_449_poincare_goat -- --nocapture`.

### Tasks

- [ ] **T2.1** **G1 — Local decodability (Theorem 3 analog).**
  - Construct known smooth map `f: R⁶ → R^d` with rank-6 Jacobian: `f(x) = tanh(W_rand · x)` where `W_rand ∈ R^{d×6}` has rank 6, `d = 64`.
  - Sample 1000 random `x` pairs; compute `y_i = f(x_i)`.
  - Fit adapter on `(y_i, x_i)` pairs.
  - Assert: `W · (φ(y₂) − φ(y₁)) ≈ (x₂ − x₁)` to within `O(‖·‖²)` for small displacements. Max abs diff < 1e-3.
  - **PASS threshold:** max abs diff < 1e-3 on small-displacement test set.

- [ ] **T2.2** **G2 — Global unrolling (Theorem 5c analog).**
  - Construct deliberately-curved manifold: `f(g) = MLP(g)` with 2 hidden layers, known to have non-constant Jacobian.
  - Sample region `K` of `g` values; compute features.
  - Fit adapter on `(f(g_i), g_i)` pairs.
  - Assert: test R² > 0.5 over `K`.
  - Assert: linear-only baseline (no φ, just ridge `W` on raw `Δf`) achieves R² < 0.
  - **PASS threshold:** adapter R² > 0.5; linear-only R² < 0.

- [ ] **T2.3** **G3 — Inverse navigation round-trip.**
  - Construct 1000-point embedding table `{(z_i, target_i)}`.
  - Pick held-out `(z_src, delta_target)`; compute `z_dest = z_src + W_pinv · delta_target` (in chart space).
  - Retrieve nearest neighbor of `z_dest` in embedding table (cosine sim).
  - Assert: retrieved `target_retrieved` matches `target_src + delta_target` within Hit@ε.
  - **PASS threshold:** Hit@0.3 > 0.5 on synthetic target space (6D Lie-algebra-like).

- [ ] **T2.4** **G4 — Zero-alloc steady state.**
  - `TrackingAllocator` audit: call `poincare_navigate_into` 1000 times.
  - Assert: 0 allocations after the first warmup call.

- [ ] **T2.5** **G5 — Latency.**
  - Bench: single `poincare_navigate_into` call, `d = 64`, `target_dim = 6`, `phi_hidden = 64`, `phi_out = 20`.
  - Assert: mean latency < 1µs SIMD on Apple Silicon.
  - Use existing `criterion` bench harness pattern (matches Plan 357 bench).

- [ ] **T2.6** **G6 — Multi-step coherence.**
  - 4-step open-loop trajectory: split `delta_target / 4`, iterate navigator 4 times.
  - Assert: trajectory stays within chart's valid region (norm bound); R² of retrieved vs ground-truth > 0.3 at step 4.
  - Determinism: same seed → same trajectory bit-identically.

- [ ] **T2.7** **G7 — Latent-vs-raw boundary.**
  - Static audit: `poincare_navigate_into` signature takes only `&[f32]` + `&PoincareAdapter`; no `SyncBlock`/`ChainConsensus`/`MapPos` references.
  - Assert: no `katgpt-rs/crates/katgpt-core/src/poincare.rs` import of sync/chain/game types.

- [ ] **T2.8** Document G1–G7 results in `.benchmarks/449_poincare_goat.md`. Verdict table. PASS/FAIL per gate. Honest record (PoC §3.6 — a refuted gate is informative, not a failure).

---

## Phase 3 — Promotion decision

### Tasks

- [ ] **T3.1** If all G1–G7 PASS: **STAY OPT-IN** by design. This is a primitive, not a default-on capability. The default-on promotion happens only after riir-ai/.research/319 G8 (imagination R²) passes and a real game-runtime consumer exists. Document the opt-in status in `Cargo.toml` comment.
- [ ] **T3.2** If any gate FAILS: diagnose. The most likely failure modes:
  - G2 FAIL (adapter doesn't unroll): closed-form PCA-tanh φ insufficient → escalate to gradient fit (riir-train follow-up).
  - G3 FAIL (inverse navigation misses): W_pinv rank-deficient → check `target_dim` vs `phi_out` ratio.
  - G5 FAIL (latency): reduce `phi_hidden` or `phi_out`; SIMD-fy the MLP evaluation.
- [ ] **T3.3** Update Research 449 with the G1–G7 results. Note any honest revision to the verdict.

---

## Phase 4 — Downstream consumer wiring (riir-ai follow-up)

> Not in this plan's scope. Filed as the riir-ai implementation of `riir-ai/.research/319` Integration 1. This plan only ships the open primitive; the consumer wiring is a separate riir-ai plan.

- [ ] **T4.1** (DEFERRED to riir-ai plan) HLA ↔ MapPos adapter fit per `riir-ai/.research/319` Integration 1.
- [ ] **T4.2** (DEFERRED to riir-ai plan) Two-brain imagination loop per Integration 1.
- [ ] **T4.3** (DEFERRED to riir-ai plan) MCTS over imagined HLA per Integration 2.
- [ ] **T4.4** (DEFERRED to riir-ai plan) sleep_time → SpatialAnticipatedHla per Integration 3.

---

## Phase 5 — Fusion (speculative, post Phase 3)

- [ ] **T5.1** Fusion with SE(2) equivariant features (R166/Plan 354): adapter fits tighter on SE(2) features than raw features. Verify with a head-to-head G2 gate.
- [ ] **T5.2** Fusion with Motor-Gated DEC (R168/Plan 357): compose imagined HLA (this primitive) with imagined spatial field (Motor-Gated DEC) for full sensorimotor imagination.
- [ ] **T5.3** Fusion with InducedCwmKernel (Plan 296): the adapter triple as a frozen Cwm parameter set.
- [ ] **T5.4** Fusion with Spherical Geodesic Steering (Plan 405): Poincaré for linear-chart segments; Slerp for geodesic corrections. Hybrid navigator.

---

## Validation summary

- [ ] **Plan complete** when Phase 1 + Phase 2 land and a verdict is recorded in `.benchmarks/449_poincare_goat.md`. Phase 3 is the promotion decision (likely "stay opt-in"). Phase 4 is downstream and not in scope.
- [ ] **Commit** with `feat: poincare navigator primitive (Plan 449 Phase 1)` after Phase 1 + Phase 2 PASS.
- [ ] **Run clippy** before commit: `cargo clippy -p katgpt-core --features poincare_navigator --all-targets`. Fix all warnings.

---

## Notes

- **Modelless-first per research skill §3.5**: the open primitive ships a closed-form ridge solver for `W` and a deterministic PCA-tanh φ. The gradient fit (AdamW + 10 epochs per paper) is a riir-train follow-up ONLY IF G2 fails. Path 0 (training-target decomposition) applies: the paper's "value" is the math (`ΔP ≈ W·Δz` + the unrolling), NOT the training loop.
- **Theorem 7 design constraint**: expect rotation/facing-direction targets to fit tighter than translation-magnitude targets. Document this in the module doc; design consumers to lean on the easier component.
- **Sibling primitives**: LFS (Plan 309, forward) + VMG (Plan 312, graph) + Poincaré (this plan, closed-form linear) cover the three navigation regimes. Selection rule documented in Research 449 §2.3.
- **No backprop at inference** (modelless mandate constraint #1). Adapter is fit once offline, frozen, BLAKE3-committed, atomic hot-swap.
- **Per AGENTS.md**: `CARGO_TARGET_DIR=/tmp/plan449` for isolated builds during Phase 2 GOAT gate runs; clean up when done.
