# Plan 438: FORE — Fitted Occupancy-Ratio Estimator Primitive

**Date:** 2026-07-14
**Research:** [katgpt-rs/.research/423_Adjoint_Bellman_KL_Contraction_Occupancy_Ratio.md](../.research/423_Adjoint_Bellman_KL_Contraction_Occupancy_Ratio.md)
**Source paper:** [arxiv:2607.05375](https://arxiv.org/abs/2607.05375) — van der Laan & Kallus, *Fitted Occupancy-Ratio Evaluation without Bellman Completeness*, 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/occupancy/` (new module) + Cargo feature `occupancy_ratio`
**Verdict:** GOAT (Research 423 §3.1) — novel + modelless + three fusion targets; not Super-GOAT (Q2/Q3 fail the novelty gate).
**Status:** 🟡 Phase 1 ✅ COMPLETE (T1.1–T1.7). Phase 2 🔲 BLOCKED on paper Algorithm 1 verification.

---

## Goal

Ship the open primitive distilled from Research 423: a generic, modelless
**fitted occupancy-ratio estimator** that converges under realizability alone
(no Bellman completeness required). The substrate-independent contribution is
the **adjoint Bellman KL contraction** (paper Lemma 3.1): the operator

    B^γ_π ω = (1−γ)ω_0 + γ · d((ων)P_π)/dν

contracts relative entropy by factor γ per fitted iteration. FORE = repeated
KL projection of a normalized exponential class onto the adjoint-Bellman
image of the previous iterate.

**Why here, why now:** zero prior art across the 5-repo quintet (Research 423
§3.2 Q1 = YES). The primitive unlocks three downstream fusions — (A) CLR
re-estimation stabilization, (B) freeze/thaw convergence guarantee, (C)
cheaper-than-bisimulation state abstraction — but each fusion requires its own
PoC and lives in a sibling repo. This plan ships **only the engine primitive**
in `katgpt-core`; fusions are tracked as out-of-scope follow-ups.

**GOAT gate (per Research 423 §4):**

| Gate | Requirement |
|---|---|
| **G1 correctness** | Baird-style MRP from paper §6.1: FORE converges to known `ω_π,γ(upper) = 0.2211`, `ω_π,γ(lower) = 15.7987` within 1% relative error after K=20 iterations on n=10000 transitions. |
| **G2 perf** | FORE fit on n=10000, state_dim=8, K=20 < 100 ms on Apple Silicon (cold-tier budget). Linear log-ratio class only for the perf gate. |
| **G3 no-regression** | `cargo clippy --workspace --all-features` + `cargo test -p katgpt-core --lib` pass unchanged. Feature is opt-in (`occupancy_ratio = []`). |
| **G4 alloc-free** | Inner KL-projection loop is zero-allocation in steady state (pre-allocated scratch buffers, `Vec::with_capacity` + `clear()` reuse). Outer `fit()` may allocate the output `Vec<f32>`. |
| **G5 modelless-ness** | No gradient descent through any base weight. `LogRatioClass::fit_kl_projection` may use GD on its *own* parameters (the supervised learner), but must not touch `NeuronShard`, `LoRAWeightVersion`, or `SenseModule` weights. |
| **G6 floor (UQ)** | N/A — ratio estimator, not a forecaster. Triggers if a downstream value-estimation app is added (then must beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`). |

**Promotion path:** ship opt-in. Promote to default-on only if a downstream
consumer (Fusion A CLR stabilization in `riir-poc`) demonstrates the gain —
otherwise stays opt-in as an engine primitive consumers can opt into.

---

## Phase 1 — Module Skeleton + Trait Surface

### Tasks

- [x] **T1.1** Create `katgpt-core/src/occupancy/mod.rs`. Gate behind
  `occupancy_ratio = []` feature in `katgpt-core/Cargo.toml`. Add to
  `[features]` block alongside other opt-in primitives (e.g.
  `cochain_point_sampler`).
- [x] **T1.2** Define `OccupancyRatioEstimator<H: LogRatioClass>` struct:
  `log_ratio_class: H`, `gamma: f32`, `k_iterations: usize`. Constructor
  `new(h, gamma, k_iterations) -> Self` with `gamma ∈ [0, 1)` asserted.
- [x] **T1.3** Define `LogRatioClass` trait (generic over the supervised
  learner — substrate-independent):
  ```rust
  pub trait LogRatioClass {
      type Params;
      fn evaluate(&self, params: &Self::Params, x: &[f32]) -> f32;  // h(x)
      fn fit_kl_projection(
          &self,
          transitions: &TransitionBatch<'_>,
          initial_moments: &InitialMoments<'_>,
          current_ratio: &[f32],   // ω̂^(k)(X_i)
          gamma: f32,
          scratch: &mut KlProjectionScratch,
      ) -> Self::Params;
  }
  ```
- [x] **T1.4** Define `TransitionBatch<'a>` (borrow-only, zero-copy):
  `states: &'a [f32]` (flattened `[n * state_dim]`), `successors: &'a [f32]`,
  `rewards: Option<&'a [f32]>`, `n: usize`, `state_dim: usize`. Also added
  `state(i)` / `successor(i)` slice accessors for ergonomic per-transition reads.
- [x] **T1.5** Define `InitialMoments<'a>` (the `P̂_0 h` estimator input —
  empirical initial-state distribution moments). Kept as simple borrow-only
  container: `initial_states`, `initial_ratio`, `n_init`, `state_dim`.
  Fields may be refined in Phase 2 once Algorithm 1 `P̂_0 h` is verified.
- [x] **T1.6** Define `KlProjectionScratch` (pre-allocated work buffers:
  `target_weights: Vec<f32>`, `design_rows: Vec<f32>`, `normal_eq_rhs: Vec<f32>`).
  Reused across iterations via `clear()` — never grown inside the loop.
  Constructor `new(n, feature_dim)` + `clear()` both shipped.
- [x] **T1.7** Define the theorem-statement module `pub mod kl_contraction`
  (doc-only, no impl) documenting Lemma 3.1:
  ```
  D_ν(B^γ_π ω ∥ B^γ_π ω̃)  ≤  γ · D_ν(ω ∥ ω̃)
  ```
  Cross-reference the candidate Lean 4 formalization target (deferred per
  Research 423 §5 caveat #4 — isomorphism is a hypothesis, not a theorem).

## Phase 2 — Linear Log-Ratio Class + KL-Projection Fit Loop

The paper instantiates the supervised learner as any class rich enough to
realize `log ω_π,γ`. For the G1/G2 gates we ship a **linear** class
`h_θ(x) = θ · φ(x)` with configurable feature map `φ`. The KL projection
then reduces to a weighted least-squares problem solved via normal equations
(closed-form, no iterative GD — keeps G5 trivially satisfied and G2 in
budget).

### Tasks

- [ ] **T2.1** Define `LinearLogRatioClass { feature_dim: usize }` implementing
  `LogRatioClass` with `type Params = Vec<f32>` (the θ vector).
- [ ] **T2.2** Default feature map `phi_identity(x) = x` (state_dim =
  feature_dim). Plug-in point for nonlinear feature maps (Fourier features,
  Random Kitchen Sinks) — out of scope for this plan, but the trait allows it.
- [ ] **T2.3** Implement `fit_kl_projection` for `LinearLogRatioClass`:
  1. Compute target weights `w_i = (1−γ) · P̂_0(X_i) + γ · ω̂^(k)(X^+_i) · ν(X^+_i|X_i) / ν(X_i)`
     — the adjoint-Bellman image of the current ratio (paper Eq. pre-Algorithm 1).
  2. Build the weighted normal equations `(Φᵀ W Φ) θ = Φᵀ W (log w_i)` where
     `Φ` is the `[n × feature_dim]` design matrix.
  3. Solve via Cholesky decomposition (reuse `katgpt-core/src/linalg/` if a
     Cholesky impl exists; otherwise add a minimal `cholesky_solve` helper in
     `occupancy/linalg.rs` — do NOT pull an external linear-algebra dep).
- [ ] **T2.4** Implement `OccupancyRatioEstimator::fit`:
  ```rust
  pub fn fit(&self, transitions: &TransitionBatch<'_>, initial_moments: &InitialMoments<'_>)
      -> Vec<f32>  // ω_fit(X_i) at each transition
  ```
  Loop K times: (a) evaluate current ratio at each `X_i`, (b) call
  `fit_kl_projection` to get new θ, (c) renormalize via
  `ω̂^(k+1)(X_i) = exp(h_θ(X_i)) / (1/n) Σ_j exp(h_θ(X_j))` (the log-partition
  `Λ_ν(h)` is the sample mean of exponentiated scores — paper Eq. for the
  normalized exponential class). Reuse `KlProjectionScratch` across iterations.
- [ ] **T2.5** Implement `value_estimate(ratio: &[f32], rewards: &[f32]) -> f32`:
  `V̂^π = (1/n) Σ ω(X_i) · r_i` (the doubly-robust-friendly downstream quantity).
- [ ] **T2.6** Verify G5 modelless-ness by inspection: `LinearLogRatioClass`
  mutates only its own `Params` (`Vec<f32>` θ); no `NeuronShard`, no
  `LoRAWeightVersion`, no `SenseModule` handle appears anywhere in the module.

## Phase 3 — Baird-MRP Test Fixture (G1 Known-Answer)

The paper §6.1 validates FORE on a Baird-style MRP with analytically known
occupancy ratios. This is the G1 correctness anchor — encode the MRP exactly
as specified.

### Tasks

- [ ] **T3.1** Construct the Baird-style MRP state space and transition kernel
  in `tests/occupancy_baird_mrp.rs`. State space: `state_dim = 8`, two
  absorbing-like regions ("upper" and "lower") with the paper's transition
  probabilities. Use a fixed seed (`StdRng::seed_from_u64(423)`) for
  reproducibility.
- [ ] **T3.2** Compute the analytical `ω_π,γ(upper) = 0.2211` and
  `ω_π,γ(lower) = 15.7987` independently in the test (solve the linear system
  `(I − γ P_π) d^π = (1−γ) d_0` directly) — this cross-checks the paper's
  numbers against our own MRP construction.
- [ ] **T3.3** Sample `n = 10000` transitions `(X_i, X^+_i)` from the
  behavior policy `ν` over the constructed MRP.
- [ ] **T3.4** Run `OccupancyRatioEstimator::fit` with `K = 20`, `gamma = 0.9`
  (paper's setting). Assert the fitted ratios at the upper/lower anchor states
  are within 1% relative error of the analytical values.

## Phase 4 — GOAT Gate

### Tasks

- [ ] **T4.1 (G1)** `cargo test -p katgpt-core --features occupancy_ratio
  --test occupancy_baird_mrp` passes (T3.4 assertion). Record the achieved
  relative error in `.benchmarks/438_occupancy_ratio_goat.md`.
- [ ] **T4.2 (G2)** Add `benches/occupancy_ratio_fit.rs` benchmarking
  `OccupancyRatioEstimator::fit` on n=10000, state_dim=8, K=20. Gate: p99
  wall-clock < 100 ms on Apple Silicon. Record in the benchmark doc.
- [ ] **T4.3 (G4)** Zero-alloc audit on the inner KL-projection loop using
  `CountingAllocator` (mirror Plan 422 T3.4 pattern): after warmup, 0
  allocations across 100 iterations. The outer `fit()` may allocate the
  output `Vec<f32>` and the initial `KlProjectionScratch`. Record in the
  benchmark doc.
- [ ] **T4.4 (G5)** Code-review checklist sign-off: no GD through base weights.
  `LinearLogRatioClass::fit_kl_projection` solves normal equations in closed
  form (no gradient steps). Document this in the module doc-comment.

## Phase 5 — No-Regression + Docs + Softmax Carve-Out

### Tasks

- [ ] **T5.1** `cargo clippy -p katgpt-core --features occupancy_ratio
  --all-targets` passes clean (per global rule: clippy before commit).
- [ ] **T5.2** `cargo clippy --workspace --all-features` passes (the
  `merkle_root`/`can_freeze` lesson — audit all feature combos).
- [ ] **T5.3** `cargo test -p katgpt-core --lib` passes unchanged (no
  regressions in the default feature set).
- [ ] **T5.4** Module doc-comment in `occupancy/mod.rs` describes the primitive
  as **generic off-policy evaluation math** — no game/chain/shard/NPC semantics.
  Cross-reference Research 423 for the fusion targets.
- [ ] **T5.5** Document the **softmax-vs-sigmoid carve-out** explicitly in the
  module doc (per Research 423 §3.4 + caveat #2): FORE's normalized exponential
  class is structurally softmax over the offline sample. This is **density-ratio
  normalization** (the correct mathematical operation — the log-partition is
  the cumulant-generating function of the empirical distribution), NOT a
  direction-vector projection. Cite the `product_key_memory.rs` precedent
  ("Deviation from the global sigmoid rule — convex-combination coefficients,
  not a probability/UQ claim"). The global sigmoid rule applies to semantic-
  domain projections onto learned directions; it does not apply here.
- [ ] **T5.6** Add a "Honest limitations" section to the module doc covering
  Research 423 §5 caveats #3 (offline transition data requires
  engram/delta_mem instrumentation for `(state, target-policy-action)` pairs)
  and #5 (continuous high-dim state spaces are the binding constraint —
  feasible for 8-dim HLA / 64-dim style_weights, infeasible for raw pixels).
- [ ] **T5.7** Re-export through `katgpt-core/src/lib.rs` under
  `#[cfg(feature = "occupancy_ratio")] pub mod occupancy;`.

---

## Out of Scope

- **Fusion A (CLR re-estimation stabilization)** — lives in `riir-ai`. Requires
  a PoC in `riir-poc` per Research 423 §3.6 (3 competitors: FORE-weighted CLR
  vs. frozen baseline vs. coherence-gated CLR; print value RMSE / KL-from-
  target / wall-clock / alloc-count). Track as a separate `riir-ai/.issues/NNN`
  when ready. Do NOT promote `occupancy_ratio` to default-on until Fusion A's
  PoC validates the gain.
- **Fusion B (freeze/thaw convergence guarantee)** — Lean 4 theorem candidate
  in `riir-neuron-db/.proofs/` or `riir-ai/.proofs/RiirAiProof/Runtime/`.
  Deferred until runtime wiring PoC confirms γ-contraction holds under float
  precision (Research 423 §5 caveat #7 — depends on personality-evolution
  operator being a Markov kernel, which holds for archetype blends but not
  arbitrary LLM-steered updates).
- **Fusion C (FORE-ratio state equivalence)** — `katgpt-rs` follow-up plan
  candidate. Benchmark bisimulation quotient size vs. FORE-ratio quotient size
  vs. ground-truth OPE error on a toy MDP. Not part of this plan.
- **Nonlinear log-ratio classes** (Fourier features, neural log-ratio) — the
  trait supports them, but only `LinearLogRatioClass` ships here. Adding a
  nonlinear class is a follow-up if a consumer needs it.
- **Behavioral policy estimation (`ν`-from-data)** — FORE assumes `ν` is given.
  For NPCs, `ν` is the empirical engram distribution; plumbing that into the
  primitive is consumer-side (riir-ai), not engine-side.
- **The backward-regression variant (paper Appendix F)** — requires adjoint
  Bellman completeness, defeating the point. Skip (Research 423 §2.4).
- **Continuous high-dimensional state spaces** — the paper's acknowledged
  limitation (§7). Documented as a limitation; no mitigation attempted here.

---

## Promotion Decision (pre-filled, pending gate)

**Stays opt-in** (`occupancy_ratio = []`) regardless of G1–G5 outcome. The
primitive's value proposition is a *guarantee multiplier* on downstream
consumers (Fusion A/B/C), none of which ship in this plan. Promotion to
default-on requires a downstream consumer (typically Fusion A in riir-ai) to
demonstrate the gain empirically in `riir-poc` — per the GOAT-gate promotion
rule ("Promotion requires modelless gain"; a primitive with no consumer has
no demonstrated gain). Demote nothing — there is no incumbent (no prior OPE
primitive exists in the corpus).
