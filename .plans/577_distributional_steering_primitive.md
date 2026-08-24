# Plan 577: Distributional Steering Primitive — FK Weights + First-Variation Table + Picard Ψ̇ Solver

**Date:** 2026-08-24
**Research:** [katgpt-rs/.research/505_Mean_Field_Distributional_Steering.md](../.research/505_Mean_Field_Distributional_Steering.md)
**Source paper:** [arXiv:2608.08770](https://arxiv.org/abs/2608.08770) — Howard & Nüsken, "A Mean-Field Framework for Inference-Time Distributional Control of Diffusion Models" (SPIGM @ ICML 2026)
**Target:** `crates/katgpt-core/src/distributional_steering.rs` (new module) + Cargo feature `distributional_steering` (opt-in)
**Status:** Active — Phase 1 not started

---

## Goal

Ship the generic, modelless population-steering primitive: given a weighted particle population `(states, weights)` and a measure-defined reward `R(μ)` with closed-form first variation, (a) compute the ∇Ψ steering term, (b) accumulate Feynman-Kac log-weights with the mean-field correction `Ψ̇` solved by damped Picard, and (c) expose the weighted empirical measure `μ̂ = Σ w_i δ_{X_i}` — the object the paper's Theorem 3.4 proves converges to the implicit tilted target `μ* ∝ e^{Ψ(x,μ*)} p`. Opt-in feature; promotes to default only on G1 targeting PASS. Consumers: BoM hypothesis weighting (katgpt-rs), crowd affect-distribution targeting (riir-ai, Guide 344 — their plan deferred until this passes G1–G2).

**GOAT gate (the falsifiable harness is the paper's own low-dim experiment):** on a 1D bimodal-GMM population with MMD reward toward a reweighted target and known analytic `μ*`, the optimality gap `J(μ̂) − J(μ*)` must be minimized at steering coefficient `λ = λ*` — vs gradient-only steering (gap minimum elsewhere/uncharacterized) and no-steer arms. Quality axis unproven until this gate runs (Research 505 §3 caveat 1).

---

## Phase 1 — Reward Table + Weighted Measure (CORE)

### Tasks

- [ ] **T1.1** `MeasureReward` trait in `distributional_steering.rs`: `first_variation_into(&self, x: &[f32], pop: &WeightedPopulation, out: &mut [f32])` + `second_variation(&self, x, y, pop) -> f32` (second needed only for the linear solver; Picard needs finite diffs of the first). Zero game semantics; `#[repr(u8)]` enum dispatch closed over a small table.
- [ ] **T1.2** Closed-form rows: `Linear(f)` (Ψ = f(x) — degenerates to pointwise; unit-test equivalence against plain steering), `Moment(F, φ)` (Ψ = F'(∫φ dμ)·φ(x)), `Mmd(kernel, target_particles)` (Ψ(x,μ) = 2∫k(x,y)(μ−ν)(dy); second variation = 2·mean-centered kernel). Reuse `rbf_mmd_sq` conventions (`mag/transfer.rs`) for the kernel; targets are weighted particle sets (no density objects).
- [ ] **T1.3** `WeightedPopulation` POD: `[f32]` states (flat N×d), `log_weights: &mut [f32]`, `weights_into(&mut [f32])` via **log-sum-exp** normalization. Fixed-capacity scratch struct; zero alloc.
- [ ] **T1.4** `gradient_steering_into`: `∇_x Ψ` per particle — for the closed-form rows this is analytic (MMD: `2∫∇_x k(x,y)(μ−ν)(dy)`); RBF kernels make this the same kernel matrix evaluated once. Compose with `LatentField`-style application (consumer passes the increment to its own integrator).
- [ ] **T1.5** Unit tests: variation rows verified against numerical functional differentiation (`(R((1−ε)μ+εδ_x) − R(μ))/ε`); MMD first variation matches finite difference at ε→0.
- [ ] **T1.6** Feature flag `distributional_steering = []` wired in katgpt-core `Cargo.toml` (opt-in, no default change) + module gated in lib.rs + `cargo check --features distributional_steering` green.

## Phase 2 — FK Log-Weights + Picard Ψ̇ Solver

### Tasks

- [ ] **T2.1** `FkStepper::step`: `A_i += (b_i·∇Ψ_i + Ψ̇_i)·δt` — `b_i` supplied by the consumer (their own per-tick drift), `Ψ̇` from T2.2. Clamp per-step log-weight delta (paper clips at 1.0) + steering-norm clamp (≤10% of |b|) as config.
- [ ] **T2.2** `psi_dot_picard`: damped fixed point per paper Alg 4 — candidate next-weights → candidate next-measure → `Ψ̇ ≈ [Ψ(x, μ̃_{t+δt}) − Ψ(x, μ_t)]/δt` → weight update; `k_fp: u8` (default 3), `damping: f32` (default 1.0; 0.5 for strong tilts). Kernel/reward evaluations computed once per step and reused across iterations (paper: Picard = 0.036–0.24% runtime). Fixed scratch; zero alloc steady-state.
- [ ] **T2.3** `residual_resample_into` (optional, sampling consumers only — `#[cfg]`-documented as NOT for persistent-agent use; Research 505 caveat 2). Systematic variant as the deterministic alternative.
- [ ] **T2.4** Self-consistency residual gate `tilt_residual(pop, reward)`: fixed-point check `μ̂ ≈ (1/Z)e^{Ψ(·,μ̂)}p`-formulated as the Picard fixed-point gap at convergence — cheap convergence certificate for consumers.
- [ ] **T2.5** Unit tests: Ψ̇ Picard matches the implicit linear-system solution (Alg 3 form, small N) to tolerance; weights sum to 1 across K_FP settings; K_FP=1 vs 3 bias visible on a strong-tilt fixture (documents the paper's own observation); damping rescues the diverging strong-λ case.

## Phase 3 — GOAT Gate (falsifiable targeting harness)

### Tasks

- [ ] **T3.1** `tests/bench_577_distributional_steering_goat.rs` (feature-gated): the paper's 1D experiment — base = bimodal GMM (means −1/+1, unit var, weights 1:3); reward = MMD² toward reweighted GMM (3:1), RBF kernel bandwidth 5.0; analytic `μ*` via GMM tilt; objective `J(μ) = λ*·MMD²(μ,ν) + KL(μ‖p₁)` (leave-one-out KL estimator, RBF 0.2). Three arms: no-steer / gradient-only / FK+Picard, λ swept around λ* ∈ {5, 10}.
- [ ] **T3.2** **G1 (targeting)**: optimality gap minimized at λ = λ* for the FK arm across ≥2 noise schedules; gradient-only arm's minimum elsewhere (the paper's headline separation, reproduced in Rust). FAIL ⇒ primitive does NOT promote; note records the refutation.
- [ ] **T3.3** **G2 (perf)**: per-particle per-step cost of the full FK+Picard path sub-µs at N=1000 (release, M3) — kernel matrix + Picard arithmetic only; document the fixed per-step setup. Bench vs gradient-only arm.
- [ ] **T3.4** **G3 (no-regression)**: default features untouched (opt-in only) — `cargo check` + `cargo test -p katgpt-core --lib` green without the feature; with the feature, full lib suite green.
- [ ] **T3.5** **G4 (alloc-free)**: tracking-allocator harness over 1000 steps at N=1000 — zero steady-state allocs (scratch pre-sized).
- [ ] **T3.6** Determinism: seeded runs bit-identical (fixed iteration order, no HashMap in the path); two-run equality test.
- [ ] **T3.7** Benchmark doc `.benchmarks/577_distributional_steering_goat.md` with the verdict table + GOAT outcome. Promotion decision recorded here; if G1 PASS → promote to default in the same commit series; if FAIL → stays opt-in with the negative result documented.

## Phase 4 — Consumers + Docs

### Tasks

- [ ] **T4.1** BoM adapter (opt-in composition, feature `bom_sampling` + `distributional_steering`): FK weights over K hypotheses against an `Mmd`/moment reward as an alternative to `select_best` argmax. NOTE: makes no UQ claim — if any future gate claims calibrated uncertainty from this, the "Report the Floor" rule attaches there (Research 505 caveat 5).
- [ ] **T4.2** Example `examples/distributional_steering_demo.rs`: 2-D GMM population steered to a target histogram — print before/after Sinkhorn-divergence + per-particle weight distribution (the "who carries it" read-out).
- [ ] **T4.3** `.docs/` entry + README feature-gate row (opt-in until promotion).
- [ ] **T4.4** Signal to riir-ai: on G1–G2 PASS, file the crowd-targeting plan there (Guide 344's P0) referencing this primitive's commit.

## Non-goals

- No entropy-reward row (needs density estimation; approximate via MMD-to-uniform on a manifold instead — Research 505 risk 3).
- No training, no amortized steering networks (that is the fine-tuning quadrant — Santi 2511.22640 / Smith 2510.10020 — riir-train cross-ref only).
- No persistent-NPC resampling mode (weights-only is the correct consumer shape).
- No game semantics in katgpt-rs (crowd targeting lives in riir-ai per Guide 344).
