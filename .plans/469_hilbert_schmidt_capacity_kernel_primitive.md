# Plan 469: Hilbert-Schmidt Capacity Kernel — Open Primitive

**Date:** 2026-07-24
**Research:** [katgpt-rs/.research/454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md](../.research/454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md)
**Source paper:** [arXiv:2607.21366](https://arxiv.org/abs/2607.21366) — Mobahi & Bartlett, HOPE, 2026-07-24
**Target:** `katgpt-rs/crates/katgpt-core/src/hope/` (new module) + Cargo feature `hope_capacity`
**Status:** Active — Phase 1-3 implemented (commit 7964f210); G2 latency bench + G4 alloc-free CountingAllocator audit pending.

---

## Goal

Ship the **three pieces of closed-form HOPE math** as a generic open primitive in
`katgpt-core`:

1. `relu_self_kernel(γ, β) -> f32` — Eq 3 of the paper, the ReLU self-kernel
   under Gaussian pre-activation `y ~ N(β, γ²)`.
2. `relu_cross_kernel_approx(...) -> f32` — Eq 5, Arc-Cosine order-1
   approximation of the ReLU cross-kernel.
3. `optimal_rank1_parent(...) -> Rank1Parent` — Eq 12–14, the optimal rank-1
   parent neuron for merging two rank-1 operators via principal eigenvector of
   rank-2 `Aᵀ·A`.

Plus a thin orchestration layer (capacity, prune cost, merge cost, block eviction
cost, greedy step) generic over a `Rank1Operator` trait.

**GOAT gate:** G1 bit-exact vs analytic reference, G2 latency (< 1µs per kernel,
< 200ns for optimal parent), G3 no-regression on katgpt-core tests, G4 alloc-free.
Promotion to **default-on** if all gates pass — the math is closed-form, generic,
and unblocks the riir-neuron-db Super-GOAT integration (Plan 321).

---

## Phase 1 — Skeleton + Self-Kernel (P0, unblocking)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/hope/mod.rs` with module
      doc + feature gate `hope_capacity`. Re-export the public API at
      `katgpt_core::hope::*`.
- [ ] **T1.2** Define the `Rank1Operator` trait (generic over input borrowed
      slices — no allocation):

      ```rust
      pub trait Rank1Operator {
          fn w_in(&self) -> &[f32];   // effective input weights (after BN absorption)
          fn w_out(&self) -> &[f32];  // output weights
          fn gamma(&self) -> f32;     // pre-activation scale (γ > 0)
          fn beta(&self) -> f32;      // pre-activation shift (β)
      }
      ```

- [ ] **T1.3** Implement `relu_self_kernel(gamma: f32, beta: f32) -> f32`:

      ```text
      K(i,i) = (γ² + β²)·Φ(β/|γ|) + β·|γ|·φ(β/|γ|)
      ```

      Use `libm::erf` for Φ (no `std` dep on no_std targets) and `libm::exp` for φ.
      Document the closed form (paper Eq 3 + Appendix E Theorem E.2).

- [ ] **T1.4** Implement `relu_phi_pdf(x: f32) -> f32` and `relu_phi_cdf(x: f32) -> f32`
      helpers (standard normal PDF/CDF). Reuse from `statrs` or hand-roll with
      `libm::erf`. **MUST be std-free for no_std/wasm32 targets.**

- [ ] **T1.5** Add unit tests in `hope/tests.rs`:
      - `relu_self_kernel(1.0, 0.0) ≈ 0.5` (standard normal, half-wave rectified energy).
      - `relu_self_kernel(1.0, 1.0)` matches analytic value `(1+1)·Φ(1) + 1·φ(1)`.
      - `relu_self_kernel(γ, β)` is invariant to `sign(γ)` (paper §3.1 normalization invariance).
      - `relu_self_kernel(γ, β) ≥ 0` for all `γ, β` (Cauchy-Schwarz on self).

### Gate

- [x] **T1.G1** G1 bit-exact: tests pass with analytic reference values (closed-form).
- [ ] **T1.G2** G2 latency: criterion bench `relu_self_kernel` < 10 ns/call. **Deferred to T4.1 bench.**
- [x] **T1.G4** G4 alloc-free: smoke test passes (signature analysis — `relu_self_kernel` takes `(f32, f32)` and returns `f32`).

---

## Phase 2 — Cross-Kernel + Optimal Parent (P0)

### Tasks

- [ ] **T2.1** Implement `warped_correlation(w_eff_in_i, w_eff_in_j, gamma_i, gamma_j) -> f32`:

      ```text
      ρ_eff = dot(w_eff_in_i, w_eff_in_j) / (‖w_eff_in_i‖·‖w_eff_in_j‖)
      κ = (ρ_eff / (1 - ρ_eff²)) · (|γ_i|/‖w_eff_in_i‖) · (|γ_j|/‖w_eff_in_j‖)
      ρ̂_ij = 2κ / (1 + √(1 + 4κ²))
      ```

      (Paper Eq 4 + Appendix E.3 Proposition E.3.)

- [ ] **T2.2** Implement `relu_cross_kernel_approx(...) -> f32`:

      ```text
      K(i,j) ≈ (1/π)·(√(1 - ρ̂²) + (π - arccos(ρ̂))·ρ̂)·√(K(i,i)·K(j,j))
      ```

      (Paper Eq 5, Arc-Cosine kernel order 1, zero-bias approximation.)

- [ ] **T2.3** Implement `optimal_parent_direction(w_eff_in_i, w_eff_in_j, w_out_i, w_out_j) -> (u_hat, v_hat)`:
      - Build the rank-2 matrix `A = w_out_i·(w̃_in_i)ᵀ + w_out_j·(w̃_in_j)ᵀ` where
        `w̃_in = [w_eff_in; β]` (augmented).
      - Compute `û = principal eigenvector of Aᵀ·A` via **rank-2 closed-form SVD**
        (avoid ambient-dim operations — restrict to the 2D span of `w̃_in_i, w̃_in_j`).
      - Compute `v̂ = (K(û, w̃_in_i)·w_out_i + K(û, w̃_in_j)·w_out_j) / ‖...‖`.
      - Resolve sign ambiguity by evaluating both `±û` in the exact objective
        (paper Eq 11).

- [ ] **T2.4** Implement `optimal_parent_scale(a, b, E_rem) -> f32`:

      ```text
      s* = (a + b·E_rem) / (2·E_rem + b)
      ```

      where `a = ‖f_i‖²_H + ‖f_j‖²_H`, `b = ⟨ψ, f_i + f_j⟩_H`, `E_rem = E_a - ‖f_i‖ - ‖f_j‖`.

- [ ] **T2.5** Bundle into `Rank1Parent { u_hat: Vec<f32>, v_hat: Vec<f32>, s_star: f32, K_self: f32 }`.
      Add `optimal_rank1_parent(op_i: &impl Rank1Operator, op_j: &impl Rank1Operator, E_rem: f32) -> Rank1Parent`.

- [ ] **T2.6** Add tests:
      - Cross-kernel diagonal consistency: `K(i,i) ≈ cross_kernel(i, i)`.
      - Cauchy-Schwarz: `|K(i,j)| ≤ √(K(i,i)·K(j,j))` for random inputs.
      - Optimal parent on identical inputs returns the input itself (trivial merge).
      - Optimal parent on orthogonal inputs returns the principal component (rank-2 → rank-1).

### Gate

- [x] **T2.G1** G1 bit-exact: cross-kernel matches analytic Arc-Cosine order-1 reference. Cauchy-Schwarz test pins the bound.
- [ ] **T2.G2** G2 latency: deferred to T4.1 bench.
- [x] **T2.G4** G4 alloc-free: `optimal_rank1_parent_into_scratch` variant takes `&mut [f32]` scratches — zero-alloc by signature.

---

## Phase 3 — Orchestration Layer (P0)

### Tasks

- [ ] **T3.1** Implement `hope_capacity(op: &impl Rank1Operator) -> f32`:

      ```text
      ‖f‖_H = ‖w_out‖₂ · √K(i,i)
      ```

- [ ] **T3.2** Implement `hope_prune_cost(victim: &impl Rank1Operator, N_active: usize, E_a: f32) -> f32`:

      ```text
      J_prune = N · ‖f_victim‖_H / (E_a - ‖f_victim‖_H)
      ```

      (Paper Eq 6 left.)

- [ ] **T3.3** Implement `hope_merge_cost(pair, parent, N_active, E_a) -> f32`:

      ```text
      J_merge = N · √(‖f_i - f_p‖²_H + ‖f_j - f_p‖²_H) / (E_a - ‖f_i‖ - ‖f_j‖ + ‖f_p‖)
      ```

      (Paper Eq 6 right.)

- [ ] **T3.4** Implement `hope_block_eviction_cost(N_active_per_layer, E_active_per_layer, E_identity) -> f32`:

      ```text
      J_evict = Σ_l (N_active^(l) · E_active^(l)) / E_identity
      ```

      (Paper Eq 20.)

- [ ] **T3.5** Implement `HopeAction` enum (`Prune { victim_idx }`, `Merge { i_idx, j_idx }`,
      `Evict { block_id }`) and `hope_greedy_step(layer_states: &[LayerState]) -> HopeAction`.
      Uses O(1) per-pair cached scalar lookup + Dantzig greedy selection (paper Eq 23).

- [ ] **T3.6** Implement `HopeLayerState { operators: Vec<impl Rank1Operator>, E_rem: f32, pair_cache: HashMap<(usize,usize), PairCache>> }`
      with `update_after_action` for the localized O(N) recompute (paper §10 phase 3).

- [ ] **T3.7** Add integration tests:
      - Greedy step on a 4-neuron synthetic layer produces a known sequence.
      - Block eviction on a 2-layer residual structure terminates at identity.
      - Locality: only the modified subspace is recomputed (Corollary C.6).

### Gate

- [ ] **T3.G1** G1: greedy sequence matches a Python reference implementation (NumPy). **Deferred — needs Python reference.**
- [ ] **T3.G2** G2: `hope_greedy_select` on a 64-neuron layer < 5 µs/step (cached). **Deferred to T4.1 bench.**
- [x] **T3.G3** G3: `cargo test -p katgpt-core --lib` passes 1766 tests (default features, 0 failures); `--all-features` 30 HOPE tests pass.
- [ ] **T3.G4** G4: 0 allocations in 100 steady-state greedy steps after warmup. **Deferred to T4.1 bench — CountingAllocator audit.**

---

## Phase 4 — GOAT Gate + Promotion (P0)

### Tasks

- [ ] **T4.1** Add `benches/hope_kernel_goat.rs` covering all gates (G1, G2, G4).
- [ ] **T4.2** Run `cargo test -p katgpt-core --features hope_capacity --lib` — all green.
- [ ] **T4.3** Run `cargo clippy -p katgpt-core --features hope_capacity --all-targets` — zero warnings.
- [ ] **T4.4** Update `katgpt-rs/crates/katgpt-core/Cargo.toml`: add `hope_capacity = []` to `[features]`.
- [ ] **T4.5** Update `katgpt-rs/README.md` Feature Showcase section with a HOPE entry.
- [ ] **T4.6** **Promotion decision:** if G1–G4 all PASS, promote `hope_capacity` to
      `default = ["hope_capacity", ...]`. Document in the README + Cargo.toml.
- [ ] **T4.7** Run `cargo check --workspace --all-features` to confirm no combo regression.

### Gate

- [ ] **T4.G5** (deferred to riir-neuron-db Plan 321 Phase 3) — Super-GOAT G5
      compaction quality gate runs in riir-neuron-db, not here. The open primitive
      only owns G1–G4.

---

## Phase 5 — Documentation + Lean Spec Self-Test (P1, post-promotion)

### Tasks

- [ ] **T5.1** Add `katgpt-rs/.benchmarks/468_hope_kernel_goat.md` with the G1–G4 results.
- [ ] **T5.2** Add a Lean spec self-test in `katgpt-rs/.proofs/KatgptProof/Hope/SpecTests.lean`
      (mirrors Plan 441 convention):
      - `relu_self_kernel_standard_normal = 1/2` (concrete instance).
      - `relu_cross_kernel_diagonal = relu_self_kernel` (concrete instance).
      - `cauchy_schwarz_cross_kernel` (universal property).
- [ ] **T5.3** Cross-link from `katgpt-rs/.research/233` (AM closest cousin),
      `katgpt-rs/.research/302` (FAME), `katgpt-rs/.research/306` (Galerkin).

---

## Out of Scope (redirects)

- **DEFT gradient elasticity** (`g_t = E_out ⊙ ∇L_target`) — training-only, → riir-train.
- **`NeuronShard → Rank1Operator` bridge** — riir-neuron-db IP, → Plan 321.
- **MAG direction mining fusion** — riir-ai runtime IP, separate plan.
- **Committed Personality structural mask** — riir-ai runtime IP, separate plan.
- **HLA per-direction capacity** — application of the open primitive, no new code here.

---

## References

- **Source paper:** [arXiv:2607.21366](https://arxiv.org/abs/2607.21366) — Mobahi & Bartlett, HOPE, 2026-07-24
- **Research note:** [katgpt-rs/.research/454](../.research/454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md)
- **Private Super-GOAT guide:** [riir-neuron-db/.research/302](../../riir-neuron-db/.research/302_HOPE_Shard_Capacity_Metric_SuperGOAT_Guide.md)
- **Integration plan:** [riir-neuron-db/.plans/321](../../riir-neuron-db/.plans/321_hope_shard_capacity_metric_compaction.md)
- **Closest shipped cousins:**
  - AM (Plan 233) — `katgpt-rs/.research/233_Attention_Matching_KV_Compaction.md`
  - FAME (R302) — `katgpt-rs/.research/302_FAME_Sampling_Invariant_Per_Entity_MoE.md`
  - Galerkin (R306) — `katgpt-rs/.research/306_Galerkin_Transformer_FUNCATTN_Grandparent_Predecessor.md`
  - Newton-Schulz (Plan 421) — for principal eigenvector primitive
  - MANCE SVD caching (Plan 427) — for SVD reuse pattern
