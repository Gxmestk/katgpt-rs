# Plan 581: twist_smc — Opaque-Reward SMC Steering + Modelless Twist Amortization

**Date:** 2026-08-28
**Status:** Active — Phase 0 (not started)
**Research:** [katgpt-rs/.research/517_CDM_Amortized_Twist_SMC_Discrete_Diffusion.md](../.research/517_CDM_Amortized_Twist_SMC_Discrete_Diffusion.md)
**Source paper:** [arXiv:2605.23346](https://arxiv.org/abs/2605.23346) — CDM: Contrastive Distribution Matching for Amortized SMC in Discrete Diffusion (Kim et al. 2026)
**Target:** `crates/katgpt-core/src/distributional_steering.rs` (extend) + `crates/katgpt-core/src/twist_cache.rs` (new) + Cargo feature `twist_smc` (opt-in, implies `distributional_steering`)
**Trained-head counterpart:** `riir-train/.plans/361_cdm_contrastive_twist_head.md` — same GOAT gate, different arm

---

## Goal

Extend the shipped distributional-steering substrate (Plan 577/Bench 682) from closed-form measure-rewards to **opaque black-box rewards** with **modelless amortization**: steer a particle population toward `p* ∝ p·ψ` where `ψ ∝ exp(β·V̂)` and `V̂` is estimated WITHOUT gradient descent — (a) an x̂₀ posterior-mean reward proxy (1 reward query per particle-step instead of `M` rollouts), (b) a state-keyed value memo so resampled particles never re-rollout, (c) a one-shot ridge readout table fit offline from cached (features, R) pairs. GOAT gate: downstream reward uplift at **matched reward-query budget** vs BoM floor and full-M SMC; the trained head (Plan 361) must beat this arm to promote.

Consistency footing (No-GD advocate row 1): self-normalized twisted SMC is consistent for ANY positive ψ — every amortization below is variance reduction, never correctness.

## Phase 1 — Opaque pointwise reward + consistency docs (CORE)

### Tasks

- [ ] **T1.1** `ClosureReward` row: pointwise `Ψ(x) = r(x)` via a caller-supplied closure (`Fn(&[f32]) -> f32`), second variation 0; document it as the degenerate `R(μ) = ∫r dμ` case (R505 Prop 3.1) that recovers plain per-state reward steering — the row that admits black-box scorers the Table-2 rows can't express.
- [ ] **T1.2** Module docs: consistency-for-any-positive-ψ note + "amortization = variance reduction" contract; cross-ref Research 517 + Plan 361.
- [ ] **T1.3** Unit tests: closure row vs `LinearReward` equivalence on affine `r`; finite/NaN rejection at the boundary (house `is_finite` discipline).

## Phase 2 — x̂₀ posterior-mean reward proxy

- [ ] **T2.1** `X0ProxyReward`: consumes caller-provided per-particle marginals `p(x₀|x_t)` (the same tensor `dllm_solver`/`ppot` consumers already produce), computes `x̂₀ = argmax` (or expectation for scalar domains), evaluates `r(x̂₀)` ONCE per (particle, step). Feature `twist_smc`.
- [ ] **T2.2** Cost contract pinned by test: reward-call count == particle count per step (vs `M·K` for full MC twist), measured via a counting closure.
- [ ] **T2.3** Proxy-quality gate helper: Spearman rank correlation of proxy vs true terminal reward on a held-out set (caller-supplied) — exported as a diagnostic, not a gate (the end-to-end gate is Phase 4).

## Phase 3 — Value memo + one-shot ridge table

- [ ] **T3.1** `twist_cache.rs`: state-keyed value memo — `papaya::HashMap<BLAKE3(state bytes + t), CachedValue>`; hit ⇒ lookup, miss ⇒ caller rollout + insert; `clear()` + capacity cap + staleness TTL (tick-keyed) for persistent-agent reuse. Zero-alloc steady state (`get` returns `Option<&f32>`-equivalent via entry API).
- [ ] **T3.2** One-shot ridge readout: offline fit `ψ(features) ≈ log(1+R)` by closed-form normal equations (`+λI`) over a cached `(features, R)` buffer — reuse `katgpt-core` `linalg` (Newton-Schulz/Cholesky house pattern); deterministic, no iterations.
- [ ] **T3.3** β/KL-budget selection: reuse `entropic_tilt::solve_beta` to pick `β` under a KL budget (anti-mode-collapse knob, R517 §1.5); exported as `select_beta_by_budget`.
- [ ] **T3.4** G1 determinism: two-run bit-identity with fixed seed stream (papaya is read-path lock-free; iteration order never enters results — house rule).

## Phase 4 — GOAT gate

- [ ] **T4.1** Harness `tests/bench_780_twist_smc_goat.rs` (number per `.benchmarks/.highwater` at write time): controlled 1-D/2-D toy + one real-ish domain (quest-grammar acceptance as the black-box `r`). Arms at **matched reward-query budget**: (a) BoM floor (R248 — reward at end only), (b) full-M SMC (ground truth, M=8), (c) proxy-only (T2), (d) memo+ridge (T3), (e) no-steer base.
- [ ] **T4.2** Metrics: downstream reward (primary), ESS trajectory, distinct-n / cluster count (diversity — R517 §1.5 axis), reward-calls-per-step (budget axis), wall-clock. UQ discipline: weighted measure stays a ranking/steering signal; any distribution/coverage claim must first beat the conformal-naive floor (Plan 340 rule).
- [ ] **T4.3** Verdict rules: promote `twist_smc` to default only if (d) ≥ (a) on reward at equal budget AND diversity non-regression; record which amortization tier carries the win; demote-if-loser vs arms (a)/(b). Report to `.benchmarks/`; wire the winner into the Bench 682 gate family as a standing row.
- [ ] **T4.4** Handoff: publish the (features, R) cache + gate harness as the shared eval for Plan 361's trained head (the head must beat arm (d) at matched budget to promote — single gate, two arms).

## Non-goals

- No training in this repo (Plan 361 owns the head). No continuous-diffusion port (no Tweedie substrate). No persistent-agent resampling (R505 caveat 2 stands — resampling is sampling-consumers-only).
