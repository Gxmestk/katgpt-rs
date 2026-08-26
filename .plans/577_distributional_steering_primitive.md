# Plan 577: Distributional Steering Primitive — FK Weights + First-Variation Table + Picard Ψ̇ Solver

**Date:** 2026-08-24 (landed 2026-08-26)
**Research:** [katgpt-rs/.research/505_Mean_Field_Distributional_Steering.md](../.research/505_Mean_Field_Distributional_Steering.md)
**Source paper:** [arXiv:2608.08770](https://arxiv.org/abs/2608.08770) — Howard & Nüsken, "A Mean-Field Framework for Inference-Time Distributional Control of Diffusion Models" (SPIGM @ ICML 2026)
**Target:** `crates/katgpt-core/src/distributional_steering.rs` (new module) + Cargo feature `distributional_steering` (opt-in)
**Status:** COMPLETE — G1 FAIL (partial) ⇒ stays opt-in; record [Bench 682](../.benchmarks/682_distributional_steering_goat.md)

> **Renumber note (binding):** the bench/test targets hardcode
> `.benchmarks/577_*` / `tests/bench_577_*` in this plan — renumbered to
> **682** (577 was already allocated to `577_emotion_direction_rank`; the
> monotonic never-reuse rule, highwater 680). All Phase-3 file references
> below read as 682.

---

## Goal

Ship the generic, modelless population-steering primitive: given a weighted particle population `(states, weights)` and a measure-defined reward `R(μ)` with closed-form first variation, (a) compute the ∇Ψ steering term, (b) accumulate Feynman-Kac log-weights with the mean-field correction `Ψ̇` solved by damped Picard, and (c) expose the weighted empirical measure `μ̂ = Σ w_i δ_{X_i}` — the object the paper's Theorem 3.4 proves converges to the implicit tilted target `μ* ∝ e^{Ψ(x,μ*)} p`. Opt-in feature; promotes to default only on G1 targeting PASS. Consumers: BoM hypothesis weighting (katgpt-rs), crowd affect-distribution targeting (riir-ai, Guide 344 — their plan deferred until this passes G1–G2).

**GOAT gate (the falsifiable harness is the paper's own low-dim experiment):** on a 1D bimodal-GMM population with MMD reward toward a reweighted target and known analytic `μ*`, the optimality gap `J(μ̂) − J(μ*)` must be minimized at steering coefficient `λ = λ*` — vs gradient-only steering (gap minimum elsewhere/uncharacterized) and no-steer arms. Quality axis unproven until this gate runs (Research 505 §3 caveat 1).

---

## Phase 1 — Reward Table + Weighted Measure (CORE)

### Tasks

- [x] **T1.1** `MeasureReward` trait: `first_variation_into(&self, x, pop, out)` + `second_variation` + `reward()` + `as_any()` (closed-table downcast for the hot loops). `RewardKind` is `#[repr(u8)]`. — *Single-position form writes Ψ to `out[0]` (the slice signature kept for batch headroom); `second_variation` non-zero only for MMD (`−2k`).*
- [x] **T1.2** Closed-form rows: `LinearReward{dir}` (Ψ = a·x), `MomentReward{gain, phi}` (Ψ = F'(m)·(p·x), `MomentGain` `#[repr(u8)]` NegSq/Sq/Identity), `MmdReward{gamma, target, dim}`. — *Kernel is a local twin of `mag/transfer.rs::rbf_kernel` (same `fast_exp(−γ‖a−b‖²)` formula — that helper is private to mag; no cross-feature dep). **Sign correction:** Research 505's Table-2 transcription (`Ψ = 2∫k(μ−ν)`) has a sign slip against its own `R = −MMD²`; the module ships `Ψ = 2[emb_ν − emb_μ]` (δR/δμ's x-dependent kernel — pinned by the finite-difference tests + the BoM adapter test which caught the wrong sign empirically: the target hypothesis was down-weighted to 6.7e-5 before the flip).*
- [x] **T1.3** `WeightedPopulation` borrowed view (states + unnormalized log weights), `weights_into` via log-sum-exp (f64 accumulators; degenerate → uniform, never NaN). Zero alloc.
- [x] **T1.4** `gradient_steering_into` (cold path, per-row analytic ∇Ψ) + the stepper's cached-kernel hot path; `clamp_steering_norm` helper for the ≤10%-of-|b| config. — *Hot ≡ cold pinned by test (the MMD gradient is `λ·4γ·[S_pop − S_ν]`, attraction toward the target).*
- [x] **T1.5** Unit tests: MMD + Moment first variations vs numerical functional differentiation via probe differences (the mass-preserving finite difference is δR/δμ + an x-independent const; probe differences cancel the constant — that constant is invisible to the tilt anyway). Linear exactness. All pass.
- [x] **T1.6** Feature wired (pre-wired by the coordinator scaffold; `cargo check --features distributional_steering` verified green).

## Phase 2 — FK Log-Weights + Picard Ψ̇ Solver

### Tasks

- [x] **T2.1** `FkStepper::begin_step`/`finish_step` two-phase API: `A_i += (b_i·∇Ψ_i + Ψ̇_i)·δt` with `clip_log_delta` (paper clips at 1.0); the ≤10%-of-|b| clamp ships as `clamp_steering_norm` (config on the consumer side — it needs |b| which only the consumer holds at begin time). — *Key contract discovery (Bench 682): `b` = the FULL simulated drift (base + steering) — the `b·∇Ψ` term carries the position transport / Girsanov overshoot correction; and Ψ̇ must be pure MEASURE drift (both Ψ terms at the advanced positions — evaluating the second at the old positions imports a `λ²|∇Ψ|²δt` transport term that explodes; max|Ψ̇| measured up to 230 before the fix).*
- [x] **T2.2** Picard Ψ̇ inside `finish_step`: warm-started, damped, K_FP config; kernel matrix built ONCE per step (symmetric fill + one `simd_exp_inplace` pass) and reused across all iterations + the gradient + the Ψ evaluations. Zero-alloc steady state (G4-pinned). — *Stability finding (Bench 682): the iteration Jacobian norm ≈ `2λ·E_w|k−emb|` ≈ 0.2λ for bandwidth-5 kernels — **divergent for λ≳5 at damping 1.0 regardless of K_FP**; consumers need damping O(1/λ) (the G1 harness uses α=min(1, 2/λ), K_FP=8). Encoded in the stepper docs.*
- [x] **T2.3** `residual_resample_into` (systematic-within-residuals) + `systematic_resample_into` (deterministic), both documented NOT for persistent-agent use per Research 505 caveat 2. — *Weights-only is the persistent-agent mode; the G1 harness uses the resampling protocol (the paper's own sampling-consumer mode) because weights-only degenerates to ESS→1 by λ≈7.5 over 30 steps (a real property, documented — bounded by the clip, harmless for crowd salience consumers).*
- [x] **T2.4** `tilt_residual`: one more Picard update from the current state, L1 weight gap — the cheap convergence certificate. Weak-tilt settled residual < 0.05 (pinned).
- [x] **T2.5** Unit tests: Picard Ψ̇ vs the Alg-3 dense linear system `(I−MW+(Mw)wᵀ)Ψ̇=(MW−(Mw)wᵀ)c` with `M = −2λk` (max_diff/scale < 5e-2 @ δt=1e-3, K_FP=200); weights sum to 1 across K_FP ∈ {1,3,5,10}; K_FP=1-vs-3 bias pinned in BOTH start regimes — *honest caveat: a zero-drift consumer makes Ψ̇ ≡ 0 identically (candidate weights = old weights → no measure drift — correct math) and every K_FP vacuous*; damping-0.5 residual ≤ damping-1.0 on the λ=40 strong-tilt fixture.

## Phase 3 — GOAT Gate (falsifiable targeting harness)

### Tasks

- [x] **T3.1** `tests/bench_682_distributional_steering_goat.rs` — the paper's 1-D experiment as specified (base GMM −1/+1 unit-var 1:3; target 3:1; MMD² reward RBF bandwidth 5.0; `J = λ*·MMD² + KL` with leave-one-out KL at RBF 0.2; λ grid {0,1.25,2.5,5,7.5,10,12.5,15}) plus: analytic-μ\* grid fixed point (damped), 4096-particle stratified reference evaluated through the SAME estimators (same-footing — estimator bias cancels in the λ-argmin), CRN across arms and λ, 8 seeds, 2 σ schedules, ESS-guard systematic resampling (the paper's sampling protocol), and adaptive Picard damping α=min(1, 2/λ).
- [x] **T3.2** **G1 (targeting): FAIL (partial)** — λ\*=5: **2/2 schedules FK min at λ=5** ✓✓ (the headline signature reproduces — clean V-shaped gap curves); λ\*=10: 1/2 (σ=1.0 ✓ at 10; σ=0.5 lands at 5, curve flat to within the 8-seed noise floor, Δgap ≈ 0.003–0.013); **separation claim NOT reproduced** (gradient-only ≈ FK, gaps agree to the 3rd decimal — in the 1-D broad-kernel Langevin regime the position steering does the work and the FK weights are a small correction). ⇒ primitive does NOT promote; the refutation + four harness-bug findings are recorded in [Bench 682](../.benchmarks/682_distributional_steering_goat.md).
- [x] **T3.3** **G2 (perf): FAIL at the literal gate** — 9045 ns/particle/step @ N=1000 d=1 (15420 @ d=8); FK/gradient-only ratio 3.91× (the marginal FK machinery cost bounded < 10× — the only perf assertion that binds). The sub-µs threshold is structurally infeasible for exact O(N²) MMD at N=1000 (the kernel build alone is 10⁶ fast_exp ≈ 1 µs/particle); the paper's 0.036–0.24% figure is relative to network evals a modelless stack doesn't have. Breakdown + the N≲300 sub-µs crossover + the approximate-kernel reopen path in Bench 682 §G2.
- [x] **T3.4** **G3 PASS** — default lib **1951/0/7i** (module compiles out — opt-in); feature-on **1969/0** (+18 module tests).
- [x] **T3.5** **G4 PASS** — **0 allocs** over 1000 steps @ N=1000 (release; debug-ignored with reason: the debug run exceeds 60 s at which point libtest's slow-warning itself allocates +2 on the shared global counter — measured at step 481, harness noise, not module allocation).
- [x] **T3.6** Determinism PASS — two-run bit-identity of states + weights (pinned in-test); index loops only, no HashMap in the path.
- [x] **T3.7** Bench doc [`.benchmarks/682_distributional_steering_goat.md`](../.benchmarks/682_distributional_steering_goat.md) — verdict table, honest findings (incl. the Research 505 sign-slip correction), stays-opt-in disposition + three reopen paths.

## Phase 4 — Consumers + Docs

### Tasks

- [x] **T4.1** BoM adapter — `distributional_steering::bom` behind `#[cfg(all(feature = "bom_sampling", feature = "distributional_steering"))]`: `hypothesis_weights_into` (static tilt fixed point over the K hypotheses — the Picard loop without the time dimension) + `select_best_fk` (argmax-weight alternative to the trait's argmax `select_best`). — *Composes cleanly with **NO Cargo.toml change**: `bom_sampling` is default-on and implies `micro_belief`, so the cfg resolves `crate::micro_belief`; the composition test drives a real `LeakyIntegrator::sample_k_states`. No UQ claim made (Research 505 caveat 5 in the module docs).*
- [x] **T4.2** `examples/distributional_steering_demo.rs` — 2-D GMM 1:3 → 3:1 dial: **MMD² 0.331 → 0.011 (3.4% of before)**, weighted cluster shares 0.24/0.76 → **0.67/0.33** toward the 0.75/0.25 target, ESS 522/600, 1 resample, top-10 weights 2.3× uniform (the "who carries it" read-out). — *Deviation: the plan's Sinkhorn-divergence print was replaced by the reward's own MMD² metric (no new dependency for a demo; noted in Bench 682).*
- [x] **T4.3** Docs: `.docs/09_feature_catalog/opt_in_features.md` entry added. — *`crates/katgpt-core/README.md` carries no per-feature opt-in table (module tables only — inspected); skipped per the plan's conditional and noted here.*
- [x] **T4.4** riir-ai signal: **REPORT-ONLY — DO NOT FILE YET.** G1 is FAIL-partial and G2 is over-budget at crowd scale; the crowd-affect-targeting plan (riir-ai Guide 344 P0) should wait for the diffusion-sampler-shaped harness reopen path (Bench 682 §Disposition). Handoff note recorded in the bench doc.

## Non-goals

- No entropy-reward row (needs density estimation; approximate via MMD-to-uniform on a manifold instead — Research 505 risk 3).
- No training, no amortized steering networks (that is the fine-tuning quadrant — Santi 2511.22640 / Smith 2510.10020 — riir-train cross-ref only).
- No persistent-NPC resampling mode (weights-only is the correct consumer shape).
- No game semantics in katgpt-rs (crowd targeting lives in riir-ai per Guide 344).

## Landing record (2026-08-26)

**Files:** `src/distributional_steering.rs` (~1.6k lines incl. 18 unit tests),
`tests/bench_682_distributional_steering_goat.rs`,
`tests/bench_682_distributional_steering_alloc_check.rs`,
`examples/distributional_steering_demo.rs`,
`.benchmarks/682_distributional_steering_goat.md`, this plan, and the
`.docs/09_feature_catalog/opt_in_features.md` entry. No Cargo.toml / lib.rs
changes (the coordinator scaffold pre-wired them).

**Verdict:** GOAT G1 **FAIL (partial)** + G2 **FAIL at the literal gate** ⇒
`distributional_steering` **stays opt-in**. What genuinely reproduced: the
λ\*=5 targeting minimum in both noise schedules (the theory's V-curve), the
J trade-off structure (MMD² ↓ / KL ↑ in exactly the predicted balance), and
the 2-D dial demo (29× MMD² reduction, shares 0.24/0.76 → 0.67/0.33 toward
0.75/0.25). What didn't: λ\*=10 at one of two schedules (flat curve at the
noise floor), the gradient-only separation claim (regime-dependent), and
the sub-µs perf gate (structurally infeasible for exact O(N²) MMD at
N=1000).

**Reopen paths** (Bench 682 §Disposition): (a) a diffusion-sampler-shaped
harness to reproduce the separation claim — the prerequisite for the
riir-ai crowd plan; (b) approximate kernel features (random features /
Nyström) for the G2 threshold; (c) N≲300 populations are already sub-µs
per particle.
