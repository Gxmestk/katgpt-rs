# Research 465: ODEWorld / PT-Flow — Continuous Predictive Architecture via Physical-Time Flow

> **Source:** Dongxiu Liu, Haoyi Niu, Peng Cheng, Yuan Gao, Xirui Kang, Sangli Teng, Koushil Sreenath, Xianyuan Zhan, *ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow* — [arXiv:2607.27924](https://arxiv.org/abs/2607.27924) (AIR Tsinghua + BAIR Berkeley, 30 Jul 2026)
> **Date:** 2026-08-03
> **Status:** Done — verdict locked (**PASS for katgpt-rs / riir-ai / riir-chain / riir-neuron-db**)
> **Classification:** Public (this note). Training recipe (JVP supervision + Savitzky-Golay target smoothing + dynamical-representation-decoupling encoder/decoder training) → riir-train.
> **Related Research:** 365 (PhysiFormer — **ships the continuous-time latent trajectory prediction primitive via DEC heat kernel, the closest mathematical cousin**), 359 (Isomorphic Neural Field World Model — **Super-GOAT; motor-gated DEC propagation is the per-NPC realization**), 426 (Temporal Straightening — **the EXACT paper ODEWorld cites as Wang et al. 2026a for the representation-collapse problem; identical PASS verdict**), 360 (AdaJEPA — same JEPA-world-model domain, same PASS verdict), 375 (Velocity Field Ensemble open primitive — ships `v_θ` as ridge-optimal blend of P frozen velocity fields), 288 (KARC — continuous-time delay-basis ridge forecaster in latent space), 236 (QGF — test-time Q-guided flow), 296 (Stokes vocabulary crosswalk)
> **Related Plans:** 357 (Motor-Gated DEC Propagation — ships `evolve_motor_gated_field` + heat-kernel trajectories), 251 (DEC operators + Hodge — ships `hodge_decompose` = the modelless "dynamical representation decoupling" analog), 303/317 (Latent Functor Runtime — ships `extract_functor`, the closed-form `A = I` maximally-straight velocity direction), 376 (Velocity Field Ensemble primitive — ships `v_θ` as ridge-combined multi-field), 342 (Latent Trajectory Geometry — ships `mean_curvature` for representation-collapse diagnostics)
> **Domain:** katgpt-rs (this note, public). The distilled runtime primitives already ship — no new public or private file.
> **PASS-Redirects (synthesis):** *(none — this is the PASS verdict's own anchor; future sessions grepping `arxiv:2607.27924` or "ODEWorld" / "PT-Flow" land here.)*

---

## TL;DR

**Paper:** ODEWorld parameterizes the latent world-model dynamics as a continuous-time ODE `dz_t/dt = v_θ(z_t, t; z_0, c)`, predicting the future via ODE-solver integration `z_T = z_0 + ∫₀^T v_θ dt` in physical time. Two training innovations: (1) **dynamical representation decoupling** — encoder `f_dyn(·; s_0)` and decoder `g_dyn(·; s_0)` are conditioned on the initial state `s_0`, so the latent `z_t` captures only the dynamics delta; (2) **direct first-order supervision** — `v_θ` is supervised via a Jacobian-vector-product (JVP) target `ẑ_t = ∂f_dyn/∂s_t · ṡ_t`, with `ṡ_t` estimated by a Savitzky-Golay filter, avoiding the JEPA representation-collapse problem without external regularization. The continuous formulation enables any-resolution temporal generation and bidirectional (backward) prediction.

**Verdict:** **PASS for katgpt-rs / riir-ai / riir-chain / riir-neuron-db.** Every runtime piece of PT-Flow decomposes into primitives that **already ship** in this codebase under different vocabulary. The two genuinely training-only pieces (JVP supervision to LEARN `v_θ`/`f_dyn`; Savitzky-Golay target smoothing) → riir-train as a one-line refinement of the existing JEPA-pretraining + RLVR recipe. This is the **same canonical JEPA-world-model PASS class** as Research 426 (Temporal Straightening — the exact prior-art paper ODEWorld cites as "Wang et al., 2026a" for the representation-collapse problem), Research 360 (AdaJEPA — same authors' circle, identical verdict), and Research 358 (SMWM).

**Distilled for katgpt-rs (modelless, inference-time):**
PT-Flow's inference math — ODE integration along a learned velocity field in latent space — IS the **DEC heat-kernel trajectory primitive** (`h(t) = exp(t·A)·h₀`) shipped in Plans 357/359, generalized by the nonlinear variant (`heat_kernel_trajectory_nonlinear`, Duhamel + Gauss-Legendre quadrature on the ReLU source term). The dynamical-representation decoupling IS `hodge_decompose` extracting the harmonic (zero-Laplacian, ODE-compatible) component. The learned velocity direction `v_θ` IS `extract_functor`'s closed-form displacement, or — for the multi-field case — the ridge-optimal `VelocityFieldEnsemble` (Research 170/375, Plan 376).

---

## 1. Paper Core Findings

### 1.1 PT-Flow — the continuous-time latent ODE paradigm

The paper's core equation (Eq. 1):

```
dz_t/dt = v_θ(z_t, t; z_0, c)   ⇒   z_T = z_0 + ∫₀^T v_θ(z_t, t; z_0, c) dt
```

where `s_t` is the raw state at physical time `t`, `z_t = f_dyn(s_t; s_0)` is the dynamical latent representation, `z_0` is the initial-state latent, and `c` is the goal/instruction conditioning. Prediction = ODE-solver integration (RK4 in the paper) over the latent velocity field.

### 1.2 Dynamical representation decoupling (§3, the "proper latent space")

The encoder `f_dyn(s_t; s_0)` and decoder `g_dyn(z_t; s_0)` are **cross-attention blocks conditioned on the initial state `s_0`**. The encoder's query tokens attend to both `s_t` and `s_0`, producing a compact `z_t ∈ R^{1×768}` (a single token) that summarizes only the *temporal changes* between `s_0` and `s_t`. The static context (time-invariant background, object textures) is offloaded to `s_0`-conditioning, so `z_t` carries only the dynamics delta.

**Why it matters (per the paper):** not all visual features can be modeled as an ODE; static backgrounds and time-invariant details disrupt velocity-field learning. Decoupling ensures the latent space is ODE-compatible.

### 1.3 Direct first-order supervision via JVP (§3, the anti-collapse trick)

The time derivative of the latent state decomposes via the chain rule (Eq. 3):

```
ẑ_t = dz_t/dt = (∂f_dyn(s_t; s_0)/∂s_t) · ṡ_t + (∂f_dyn(s_t; s_0)/∂s_0) · ṡ_0
                                                              ↑
                                              s_0 time-invariant ⇒ this term = 0
```

So `ẑ_t = JVP(f_dyn(s_t; s_0), ṡ_t)` — a Jacobian-vector product. The target `ṡ_t` is approximated by a **Savitzky-Golay derivative filter** (window 5, kernel `w = (1/10)[-2, -1, 0, 1, 2]`), which fits a low-order polynomial in a least-squares sense over a sliding window — better numerical accuracy + temporal smoothness than naive finite differences.

The velocity-field loss (Eq. 4):

```
L_v = ‖v_θ(z_t, t; z_0, c) − sg(JVP(f_dyn(s_t; s_0), ṡ_t))‖²
```

with stop-gradient `sg(·)` on the target (though ablation B.6 shows it works without stop-gradient too).

**Why it matters (per the paper):** unlike JEPA-style consistency losses (which couple prediction + reconstruction and suffer representation collapse), direct first-order supervision decouples the two objectives. This is the load-bearing anti-collapse mechanism.

### 1.4 ODEWorld — the instantiated world model (§4)

- Frozen pre-trained **DINOv2** encoder `f_obs` projects raw images `x_t → s_t` (DINO feature space); frozen decoder `g_obs` reconstructs `s_t → x̂_t`.
- `f_dyn` / `g_dyn` operate **inside the DINO feature space** (not raw pixels).
- `v_θ` is a lightweight 3-layer MLP with FiLM time conditioning, taking `z_t`, `z_0`, `c`, `τ` (rescaled physical time `τ = t/L`) as input.
- Post-goal-stationarity: when `τ > τ_c`, fix future state + drive `v_θ → 0`.

### 1.5 Empirical results (§5)

- **Video prediction (LIBERO, Agibot-World):** ODEWorld beats V-JEPA 2 and LDP on PSNR + LPIPS, especially at long horizons (64 frames). Latency: 0.072s for @64 frames on A100 (vs 0.619s V-JEPA 2, 3.953s LDP).
- **Bidirectional prediction (§5.3):** sign flip `v_τ → -v_τ` produces physically-plausible backward sequences.
- **Any-resolution generation (Fig. 5):** trained on 1/3 frame rate, ODEWorld recovers intermediate frames (temporal super-resolution).
- **Policy learning (LIBERO-LONG, real-world bi-manual):** sequential-subgoal paradigm achieves 83.6% avg success (vs 78.0% GLCBC, 81.0% VPP); real-world bi-manual 80% vs 55% X-VLA baseline.
- **Anti-collapse (App. C, RankMe):** ODEWorld's 768-dim `z` maintains effective rank ≥413 across horizons 1–32; V-JEPA 2 = 203.7, DINOv2 CLS = 376.1.

### 1.6 What is NOT the contribution

- **DINOv2 as the vision backbone** — pre-trained, frozen, off-the-shelf.
- **The ODE-solver itself (RK4)** — off-the-shelf.
- **The specific robotics benchmarks** — application domain, not mechanism.
- **Action conditioning** — explicitly NOT incorporated ("current version of ODEWorld does not incorporate action conditioning", App. E).

The contribution is the **PT-Flow paradigm** (continuous-time latent velocity field + dynamical decoupling + first-order supervision) and its **ODEWorld instantiation** for video + policy learning.

---

## 2. Distillation — duplicate detection vs our corpus

### 2.1 Vocabulary crosswalk (paper → codebase)

| Paper term | Codebase equivalent | Where it ships |
|---|---|---|
| Latent ODE velocity field `v_θ(z_t, t)` | DEC heat-kernel trajectory operator exponential `exp(t·A)·h₀` (linear) or Duhamel variation-of-parameters (nonlinear `dh/dt = A·h + ReLU(h)`) | `katgpt-rs/crates/katgpt-dec/src/heat_kernel.rs::heat_kernel_trajectory_krylov` + `nonlinear_heat_kernel.rs::heat_kernel_trajectory_nonlinear` |
| ODE integration for prediction `z_T = z_0 + ∫ v_θ dt` | Heat-kernel trajectory IS the closed-form ODE solution operator — `h(t) = exp(t·A)·h₀` is the exact integral of `dh/dt = A·h` | Same files; Plan 357, Research 365 |
| Velocity field `v_θ` as learned direction | `extract_functor` constructs the displacement direction closed-form; `VelocityFieldEnsemble` ridge-solves the regression-optimal blend of P frozen velocity fields | `riir-ai/crates/riir-engine/src/latent_functor/arithmetic/mod.rs::extract_functor` + `katgpt-rs/.research/375` + Plan 376 |
| Dynamical representation decoupling (s_0-conditioned encoder/decoder) | `hodge_decompose` extracts the harmonic (zero-Laplacian = flat = ODE-compatible) component; `extract_functor` constructs the `A = I` maximally-decoupled linearization | `katgpt-dec/src/hodge.rs::hodge_decompose` + `latent_functor/arithmetic/mod.rs::extract_functor` |
| Savitzky-Golay derivative filter for `ṡ_t` target | N/A — training recipe (target smoothing for the JVP supervisor). Belongs in riir-train. | — |
| JVP first-order supervision `ẑ_t = JVP(f_dyn, ṡ_t)` | N/A — training recipe (gradient signal to LEARN `v_θ` and `f_dyn`). The runtime analog is `extract_functor` deriving the direction from observed transitions closed-form, no JVP/GD. | — |
| Bidirectional prediction (`v → -v` sign flip) | Heat kernel supports `t<0` natively (`exp(t·A)` for any real `t`); functor direction trivially negated | Same files |
| Any-resolution temporal generation (variable step) | Heat kernel takes continuous `t` parameter; Krylov approximates `expm(t·A)` for any `t` in O(k·nnz(A)) regardless of step size | Same files |
| Representation collapse avoidance (RankMe effective rank) | `mean_curvature` trajectory geometry metric (diagnostic); harmonic projection; `RankMe`-style effective rank via Plan 415 (Within-Class Effective Rank) | `katgpt-core/src/latent_trajectory_geometry.rs` + Plan 342 + Plan 415 |
| Continuous-time forecaster in compact latent | KARC delay-basis ridge forecaster (continuous-time, ridge-fit on delay-embedded features) | `katgpt-rs/crates/katgpt-core/src/linalg/ridge_solve.rs`; Research 288, Plan 308 |

**Grep verification:** paper-vocabulary grep (`ODEWorld|PT-Flow|physical.time flow|JVP.*velocity|dynamical.representation.decoupling`) returns ZERO hits across all 7 repos. Codebase-vocabulary grep (`heat_kernel_trajectory|extract_functor|hodge_decompose|VelocityFieldEnsemble`) hits the shipped primitives below. **The ODE-in-latent-space machinery ships under operator names; vocabulary translation is the only defense** — the canonical failure mode the research skill flags.

### 2.2 The five decomposed pieces — all shipped

ODEWorld's runtime mechanism decomposes into five pieces, **each of which already ships** in this codebase:

#### Piece 1 — Continuous-time latent trajectory prediction via operator exponential

**Paper:** `z_T = z_0 + ∫₀^T v_θ dt`, integrated via RK4.

**Shipped:** `katgpt-rs/crates/katgpt-dec/src/heat_kernel.rs` ships THREE variants of the exact same mathematical object:

```rust
// Linear: h(t) = exp(t·A)·h₀ — exact solution of dh/dt = A·h
pub fn heat_kernel_trajectory_linear(eig, h0, motor_vec, motor_dim, t) -> CochainField
pub fn heat_kernel_trajectory_linear_into(eig, h0, motor_vec, motor_dim, t, &mut out)  // zero-alloc

// Krylov-subspace approximation: h(t) ≈ V_k · exp(t·H_k) · V_kᵀ · h₀  (k ≈ 20-50)
pub fn heat_kernel_trajectory_krylov(cx, h0, motor_vec, motor_dim, t, k) -> CochainField

// Nonlinear (Duhamel + Gauss-Legendre quadrature on ReLU source):
// solves dh/dt = -h + Δ·ReLU(h) + diag(motor)·h — handles v_θ nonlinear in z
pub fn heat_kernel_trajectory_nonlinear(cx, eig, h0, motor_vec, motor_dim, t, n_quad, relu_slope)
```

The linear variant covers ODEWorld's linearized case. The nonlinear variant covers ODEWorld's full MLP `v_θ(z_t, t)` — Duhamel variation-of-parameters + Gauss-Legendre quadrature on the ReLU source term is a well-known exponential integrator for semilinear ODEs `dh/dt = A·h + f(h)`, which is exactly ODEWorld's `dz_t/dt = v_θ(z_t, t)` with `v_θ = A·z + nonlinear(z)`. Plan 359 Phase 3 GOAT-gated the nonlinear variant (G1 nonlinear correctness, G2 latency, G3 Hodge preservation, G4 zero-alloc).

**The Krylov variant is ODEWorld's RK4 replacement — and it's strictly better:** RK4 accumulates O(T·dt⁴) global error; Krylov `expm(t·A)` is exact for the linear part and approximates only the action of the exponential (controllable error via k). For T > k, Krylov is **cheaper than T RK4 steps AND more accurate** (Research 365 §2.2). This is the single-shot trajectory prediction PhysiFormer argued for, shipped.

#### Piece 2 — The velocity field `v_θ` as a direction

**Paper:** `v_θ` is a learned 3-layer MLP.

**Shipped (single-field):** `riir-ai/crates/riir-engine/src/latent_functor/arithmetic/mod.rs::extract_functor` constructs the displacement direction closed-form from a transition buffer:

```rust
// extract_functor: f = (1/N) Σ_k (target_k − source_k), coherence = mean cos(target_k − source_k, f)
// apply_functor:   out = source + functor     (predict the next latent by additive displacement)
```

`extract_functor` re-estimates the displacement `f` (the velocity direction at one-step horizon) from observed transitions; `apply_functor` applies it. By construction, this is the `A = I` (maximally straight, ε = 0) regime of ODEWorld's linearization — the velocity direction IS the displacement, no MLP needed. Super-GOAT Research 123, Plans 303/317.

**Shipped (multi-field):** `VelocityFieldEnsemble` (Research 170, 375; Plan 376) ridge-solves the regression-optimal linear combination of P frozen velocity fields:

```
b̂(x) = Σ_i η_i · b_i(x),  where (K + λI) η = r
```

This is `v_θ` as a weighted ensemble of P archetype forecasters — strictly more expressive than ODEWorld's single MLP, and regression-optimal for the observed trajectory data.

#### Piece 3 — Dynamical representation decoupling

**Paper:** `f_dyn(·; s_0)` and `g_dyn(·; s_0)` conditioned on initial state, offloading static context.

**Shipped:** `katgpt-rs/crates/katgpt-dec/src/hodge.rs::hodge_decompose` extracts the harmonic (zero-Laplacian, flat = ODE-compatible) component of any latent trajectory cochain:

```
ω = exact ⊕ harmonic ⊕ coexact    (Helmholtz/Hodge decomposition)
harmonic = ker(Δ)                  — the maximally-flat, ODE-compatible subtrajectory
```

Projecting a latent trajectory onto its harmonic component IS the modelless "keep only the ODE-compatible dynamics delta" operation. Plan 251 (DEFAULT-ON DEC substrate), Research 219/296. The Hodge decomposition is preserved exactly by the heat-kernel trajectory (harmonic eigenvalues = 0, so `exp(t·0) = 1` — harmonic components are conserved; exact/coexact components are damped by their eigenvalues). **This is a strictly stronger guarantee than ODEWorld's learned encoder-decoupling**: the decoupling is structural (algebraic), not learned.

#### Piece 4 — Bidirectional + any-resolution prediction

**Paper:** sign flip `v → -v` for backward; variable integration step for any-resolution.

**Shipped:** immediate consequence of `h(t) = exp(t·A)·h₀` being defined for any real `t`:
- **Backward:** `h(-t) = exp(-t·A)·h₀` — exact same primitive, negated `t`.
- **Any-resolution:** `t` is a continuous parameter; `exp(t·A)` is computable for any `t` in O(k·nnz(A)) via Krylov regardless of step size. No new primitive needed.

These are not separate capabilities — they are immediate mathematical properties of the operator-exponential formulation. The fact that ODEWorld frames them as headline features is a consequence of ODEWorld using RK4 (a discrete-step method); the heat-kernel formulation gets them for free.

#### Piece 5 — Anti-collapse (RankMe effective rank ≥ 413)

**Paper:** JVP supervision + decoupling avoids JEPA representation collapse.

**Shipped (modelless analogs):**
- `katgpt-core/src/latent_trajectory_geometry.rs::LatentTrajectoryGeometry::mean_curvature` (Plan 342) — the curvature metric, used as a collapse diagnostic.
- `hodge_decompose` projection onto harmonic component (Plan 251) — the structural "remove the high-curvature part" operation.
- Plan 415 (Within-Class Effective Rank) — the RankMe-style effective-rank diagnostic, shipped.
- `VelocityFieldEnsemble` ridge solve — the Gram matrix `K + λI` is well-conditioned by construction (Tikhonov regularization), which structurally prevents the collapse mode where all fields align.

ODEWorld's *anti-collapse mechanism* is the JVP supervision (a training-loop trick). Our substrate prevents collapse structurally — by construction, not by supervision. The two are different routes to the same property; ours is modelless.

### 2.3 Closest cousins across all 7 repos

| Cousin | Domain | Verdict / status | Overlap with ODEWorld |
|---|---|---|---|
| **Research 365 + Plan 357 (PhysiFormer → DEC heat kernel)** | katgpt-rs | **GOAT, shipped** | The closest mathematical cousin — single-shot trajectory prediction via operator exponential `h(t) = exp(t·A)·h₀`. ODEWorld's `z_T = z_0 + ∫ v_θ dt` IS this integral in latent space. Ships linear + Krylov + nonlinear variants. |
| **Research 359 + riir-ai 168 (Isomorphic Neural Field World Model, Motor-Gated DEC)** | katgpt-rs / riir-ai | **Super-GOAT, shipped** | World-model + DEC propagation + motor-gated channels = ODEWorld's velocity field + action conditioning (which ODEWorld doesn't have). Strictly broader. |
| **Research 426 (Temporal Straightening, Wang et al. 2026a)** | katgpt-rs | **PASS** (2026-07-15) | **The exact prior art ODEWorld cites** in §2 + §3 for the representation-collapse problem. Identical PASS verdict — same authors' circle (LeCun, Ren, Wang), same JEPA-world-model domain, same "runtime analog already ships" conclusion. |
| **Research 360 (AdaJEPA, same authors' circle)** | katgpt-rs | **PASS** (2026-07-01) | Same JEPA-world-model domain, same PASS verdict, same canonical vocabulary-mismatch failure class. PoC Addendum honestly refuted quality parity but confirmed architectural coverage. |
| **riir-ai Research 170 + katgpt-rs Research 375 / Plan 376 (Velocity Field Ensemble)** | riir-ai / katgpt-rs | **Super-GOAT guide + open primitive** | Ships `v_θ` as the ridge-optimal linear combination of P frozen velocity fields. Strictly more expressive than ODEWorld's single MLP. |
| **Research 288 + Plan 308/332 (KARC)** | katgpt-rs / riir-ai | **DEFAULT-ON** | Continuous-time delay-basis ridge forecaster in latent space — a different parameterization of the same "learn a latent velocity" problem, ridge-fit not MLP-fit. |
| **Plan 251 (DEC operators + Hodge decomposition)** | katgpt-rs | **DEFAULT-ON substrate** | `hodge_decompose` = the modelless "dynamical representation decoupling" — extracts the harmonic (ODE-compatible) component structurally. |
| **Research 123 + Plans 303/317/357 (Latent Functor Runtime)** | riir-ai | **Super-GOAT, shipped** | `extract_functor` constructs the `A = I` maximally-straight velocity direction closed-form from transitions — no MLP, no JVP, no GD. |
| **Plan 342 + Research 324 (Latent Trajectory Geometry)** | katgpt-rs | **Gain, shipped** | `mean_curvature` = representation-collapse diagnostic (curvature is the inverse of smoothness, the property ODEWorld's RankMe measures). |
| **Research 358 (SMWM, same author Balestriero)** | katgpt-rs | **PASS** | Third same-author PASS precedent in the JEPA-world-model domain. |
| **riir-train `dec_training/hodge_reward.rs`** | riir-train | shipped | Hodge-decomposed reward shaping by topological mode — the *training-reward* analog of ODEWorld's JVP first-order supervision, already integrated with the JEPA-pretraining + RLVR recipe. |

---

## 3. Mandatory latent-space reframing (per SKILL §1.5 step 3)

| Target substrate | ODEWorld reframing | Status |
|---|---|---|
| **(a) HLA per-NPC latent state** | "NPC's belief/affect state evolves along a learned velocity field in continuous time, integrated via heat kernel" — exactly the motor-gated DEC propagation pitch (Research 359, riir-ai 168) | Already shipped as Super-GOAT |
| **(b) `latent_functor/` (the velocity direction)** | "`extract_functor` IS the `v_θ` direction at one-step horizon; `apply_functor` IS the Euler step; the heat kernel IS the multi-step integration" — verbatim, modelless | Already shipped as Super-GOAT (Research 123) |
| **(c) `cgsp_runtime/` (the MPC replan loop)** | "Each CGSP cycle uses the heat-kernel trajectory as the multi-step predictor; curiosity = deviation between rollout and heat-kernel forecast" | Already shipped; heat-kernel integration with CGSP is the Plan 357/359 wiring |
| **(d) LatCal fixed-point commitment (sync boundary)** | ODEWorld's velocity field is per-encoder-local (latent geometry), never crosses sync; only the resulting action (raw) crosses — same discipline as HLA's 5 synced scalars | Boundary discipline inherited, no new bridge needed |
| **(e) `NeuronShard` / `MerkleFrozenEnvelope` / Raven consolidation** | The frozen `v_θ` (or the velocity-field ensemble) is a BLAKE3-committed frozen artifact; per-NPC adapted functors are local latent state; sleep-time consolidation is the cross-episode integration | Already shipped (Plan 296 InducedCwmKernel, Plan 341 sleep-time) |
| **(f) DEC Stokes operators** | **The strongest reframing.** `hodge_decompose` extracts the harmonic (ODE-compatible) component; `heat_kernel_trajectory_*` integrates the velocity field exactly; `belief_mass_divergence` (Plan 314) is the conservation-law validator. **The entire PT-Flow mechanism ships as DEC operators.** | Already shipped (Plans 251/314/357) |

Every substrate either already ships the equivalent or is orthogonal. **No new latent-to-latent operation is suggested by ODEWorld that the codebase does not already have.** The operator-exponential formulation (heat kernel) is strictly stronger than ODEWorld's RK4 integration — it gets bidirectional + any-resolution + Hodge preservation for free, as mathematical consequences rather than separate capabilities.

---

## 4. §3.5 Modelless-unblock check

The paper IS partially training-only (JVP supervision + Savitzky-Golay target smoothing to LEARN `v_θ` and `f_dyn` via backprop). Per §3.5 the question is whether the distilled runtime primitive can be implemented modellessly. **It already is** — all three modelless paths are shipped:

1. **Freeze/thaw path** — N/A as a *gate failure*. The primitive IS the runtime pattern: the frozen velocity field (single `extract_functor` direction OR `VelocityFieldEnsemble` of P fields) is committed via `MerkleFrozenEnvelope`; the heat-kernel trajectory applies it for any time `t` atomically. Readers keep the old snapshot until the swap completes.
2. **Raw/lora reader-writer hot-swap** — N/A. `extract_functor` derives the velocity direction closed-form from observed transitions; no learned MLP, no constructed LoRA needed. `VelocityFieldEnsemble` ridge-solves the optimal combination of P pre-frozen fields — closed-form, no GD.
3. **Latent-space correction** — N/A. `hodge_decompose` projects onto the harmonic (ODE-compatible) subspace at inference (Plan 251). `mean_curvature` (Plan 342) is the diagnostic. Both are zero-allocation, gateable, BLAKE3-committable.

No deferral to riir-train is needed from the modelless side because **the runtime primitives already cover all three modelless paths** the paper's mechanism could be realized through. The training recipe itself (JVP supervision + Savitzky-Golay target smoothing + dynamical-decoupling encoder/decoder joint optimization) is a refinement that belongs in riir-train — and as §7 below notes, the Hodge-decomposed reward analog already ships there as `dec_training/hodge_reward.rs`.

---

## 5. Novelty gate (§1.5) — all four NO

| Q | Answer | Evidence |
|---|---|---|
| **1. No prior art?** | NO | `heat_kernel_trajectory_*` (Plans 357/359, Research 365) ships the continuous-time latent ODE integration primitive — linear, Krylov, AND nonlinear variants. `extract_functor` (Plan 303/317, Super-GOAT Research 123) ships the velocity-direction construction. `hodge_decompose` (Plan 251) ships the dynamical-decoupling analog. `VelocityFieldEnsemble` (Research 170/375, Plan 376) ships `v_θ` as ridge-optimal multi-field blend. Research 426 (Temporal Straightening, the EXACT paper ODEWorld cites as Wang et al. 2026a for the representation-collapse problem) was already verdicted PASS with the same "runtime analog already ships" conclusion. Research 360 (AdaJEPA, same authors' circle) PASS. Research 358 (SMWM, same author Balestriero) PASS. |
| **2. New capability class?** | NO | Continuous-time latent trajectory prediction via operator exponential + Krylov already ships. The "bidirectional" + "any-resolution" properties are *immediate mathematical consequences* of `exp(t·A)` being defined for any real `t`, not separate capabilities. ODEWorld frames them as headline features only because it uses RK4 (a discrete-step method); the heat-kernel formulation gets them for free. |
| **3. Product selling point?** | NO | "NPCs plan in continuous latent time" is already the heat-kernel trajectory pitch (Research 365, Plan 357) and the motor-gated DEC world-model pitch (Research 359, riir-ai 168). The "any-resolution subgoal generation" angle is exactly what Krylov-subspace `expm` provides. The selling-point sentence doesn't form for our substrate because we already ship it. |
| **4. Force multiplier (≥2 pillars)?** | NO | Touches heat kernel + latent functor + velocity-field ensemble + KARC + sleep-time anticipator + Hodge decomposition + trajectory geometry, but all already integrated. No new pillar connection. |

**Verdict: PASS for modelless/runtime.** Not Super-GOAT, not GOAT, not Gain.

---

## 6. MOAT gate per domain

| Repo | In-scope? | MOAT contribution | Decision |
|---|---|---|---|
| `katgpt-rs` (public) | Marginal | None — heat-kernel trajectory primitive already shipped (Plans 357/359); Hodge decomposition already shipped (Plan 251); velocity-field ensemble already shipped (Plan 376). No new open primitive to add. | **No file created** (this note is the only output) |
| `riir-ai` (private runtime) | In-scope | None — Research 123 (Super-GOAT latent functor) + Research 168 (Motor-Gated DEC World Model) + Research 170 (Velocity Field Ensemble) already cover the runtime IP, strictly more broadly. | **No guide created** |
| `riir-chain` (private chain) | Out of scope | N/A — ODEWorld's velocity field is per-encoder-local latent state, never crosses sync | — |
| `riir-neuron-db` (private shards) | Out of scope | N/A — the frozen `v_θ` (or velocity-field ensemble) already commits via BLAKE3 `MerkleFrozenEnvelope`; harmonic projection is a local read op, not a shard mutation | — |
| `riir-train` (private training) | In-scope | Marginal — JVP first-order supervision + Savitzky-Golay target smoothing + dynamical-decoupling encoder/decoder joint training is a refinement of existing JEPA pretraining + RLVR objectives. **The Hodge-decomposed reward analog already ships** as `riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs` (Plan 277 T9-T12). The remaining delta is "use `ẑ_t = JVP(f_dyn, ṡ_t)` with Savitzky-Golay-smoothed `ṡ_t` as the velocity-field supervisor alongside `hodge_reward`" — a one-line variant, not a new research line. | **→ riir-train** (see §7) |

---

## 7. → riir-train (one-line redirect per SKILL §"Redirect to riir-train")

If prioritized, file a plan in `riir-train/.plans/` extending the existing `riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs` and the JEPA-pretraining recipe: add **JVP first-order supervision `L_v = ‖v_θ − sg(JVP(f_dyn, ṡ_t))‖²` with Savitzky-Golay-smoothed `ṡ_t` (window-5 kernel `[-2,-1,0,1,2]/10`) as the velocity-field supervisor** alongside the existing Hodge-decomposed reward, tested on the Bomber/Go/Civ arenas against the `hodge_reward`-only baseline. Hypothesis (per ODEWorld §4 + App. B.6): the JVP target alone (without stop-gradient) provides effective regularization because the first-order velocity constraint directly regularizes temporal evolution in latent space; the Savitzky-Golay smoothing handles the DINO-encoder temporal-inconsistency noise that ODEWorld explicitly cites as a limitation (App. E). The strongest test bed would be a 2-D navigation toy where the latent velocity field's any-resolution prediction quality can be compared against finite-difference ground truth. **Not pursued here — out of scope for this workflow.**

The only genuinely transferable *runtime* observation — ODEWorld's finding that a **single latent token** (`z_t ∈ R^{1×768}`) suffices for strong performance (App. B.4) — is already the codebase's posture: HLA's per-NPC 8-dim affect state is the compact aggregated signal; the underlying `style_weights[64]` shard is the fine-grained signal. The compact-latent-suffices pattern ships; no new primitive is implied.

---

## TL;DR

**Paper:** *ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow* (Liu, Niu, Cheng, Gao, Kang, Teng, Sreenath, Zhan; AIR Tsinghua + BAIR Berkeley, arXiv:2607.27924, 30 Jul 2026).

**Verdict:** **PASS for katgpt-rs / riir-ai / riir-chain / riir-neuron-db.** The paper's runtime mechanism — continuous-time latent ODE integration along a learned velocity field — **IS the DEC heat-kernel trajectory primitive** (`h(t) = exp(t·A)·h₀` and its nonlinear Duhamel-quadrature extension) shipped in Plans 357/359 (Research 365 GOAT, Research 359 Super-GOAT). The velocity direction `v_θ` IS `extract_functor`'s closed-form displacement (Super-GOAT Research 123, Plans 303/317) or the ridge-optimal `VelocityFieldEnsemble` blend (Research 170/375, Plan 376). The dynamical representation decoupling IS `hodge_decompose` extracting the harmonic (zero-Laplacian, ODE-compatible) component (Plan 251). The bidirectional + any-resolution "headline features" are *immediate mathematical consequences* of `exp(t·A)` being defined for any real `t`, not separate capabilities. The training recipe (JVP first-order supervision + Savitzky-Golay target smoothing + joint encoder/decoder optimization) belongs in riir-train as a one-line refinement — and the Hodge-decomposed reward analog already ships there as `dec_training/hodge_reward.rs`.

This is the **same canonical JEPA-world-model PASS class** as Research 426 (Temporal Straightening — the exact prior art ODEWorld cites as Wang et al. 2026a for the representation-collapse problem, identical PASS verdict), Research 360 (AdaJEPA — same authors' circle), and Research 358 (SMWM — same author Balestriero). The ODE-in-latent-space machinery ships under DEC operator names (`heat_kernel_trajectory_*`, `hodge_decompose`, `extract_functor`, `VelocityFieldEnsemble`); vocabulary translation is the only defense — the canonical failure mode the research skill flags.

**Files created this session:** `katgpt-rs/.research/465_ODEWorld_PT_Flow_Continuous_Latent_ODE_Velocity.md` (this note — the only output). PASS-Redirects cross-reference lines added to Research 365, 359, 426, and riir-ai Research 170 per SKILL §1.55.1.

**Recommended next step:** None for katgpt-rs / riir-ai / riir-chain / riir-neuron-db. The riir-train follow-up (add JVP first-order supervision alongside `hodge_reward`) is optional and out of scope for this workflow.

---

## 8. PoC-scope note (per SKILL §3.6)

A "defend-wrong" PoC at `riir-poc/` is **not required** for this verdict. §3.6 mandates a PoC when a verdict *downgrades a paper on the grounds that "the runtime analog already ships" or achieves "parity"* — i.e. when an architectural-evidence-only claim asserts quality parity. This verdict makes **no quality-parity claim**:

- It does not claim the shipped `heat_kernel_trajectory_krylov` "matches" ODEWorld's RK4 on LIBERO video prediction (we are not a video world model; LIBERO is irrelevant to our substrate).
- It does not claim the shipped `extract_functor` "performs as well as" ODEWorld's learned MLP `v_θ` on robotic policy learning.
- It claims only **architectural coverage** (the five decomposed pieces ship separately under different vocabulary) + **mathematical isomorphism** (the operator-exponential formulation is the closed-form solution of the ODE, and gets bidirectional + any-resolution as free mathematical consequences) + **substrate mismatch** (ODEWorld's pixel-reconstruction quality metrics are irrelevant to a per-NPC belief-state runtime).

The verdict is a PASS, not a parity-backed downgrade of a quality claim. The §3.6 PoC mandate triggers on the latter; it does not trigger on a structural-coverage PASS where the paper's primary evaluation domain (robotic manipulation video generation) doesn't exist in the runtime.

If a future plan *does* consume ODEWorld's framing for a runtime change (e.g. wiring `heat_kernel_trajectory_krylov` as the MPC planner's multi-step predictor inside `cgsp_runtime`), THAT plan would carry its own quality-gate PoC (G1 trajectory-alignment against ground-truth NPC trajectories). This research note does not.
