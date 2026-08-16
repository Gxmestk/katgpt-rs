# Bench 659 — UGC certified-unmasking-schedule PoC (Issue 664 / Research 485 / arXiv:2608.13520)

**Date:** 2026-08-17
**Verdict:** **G1 / G1-cert / G1(exact) / G4 ALL PASS — T5 G1b FAIL (honest negative result). Estimator + schedule construction land as always-on diagnostic substrate; NO `ugc_schedule` feature flag; GOAT-track CLOSED.**

## What ran

| Gate | Harness | Result |
|---|---|---|
| G1 closed forms | `katgpt-core` `tests/ugc_664_poc.rs` (13 tests, release) | q_rep (Eq 24a) + q_par (Eq 24b) reproduced to rel < 1e-9; q_par(λ) = q_rep(−λ) reflection exact; H(0,1) = ((d−1)/(d+1))·ln2 via direct integration AND the TSE identity (Eq 62a) at d ∈ {8, 16, 40} |
| G1 paper Ratios (exact path) | same | Noisy repeated bit d=128: **4.511 / 2.165 / 1.653** vs paper {4.51, 2.16, 1.65} — 3-digit reproduction via the Bernstein representation (Eq 65a/65b) + Gauss–Legendre |
| G1 paper Ratios (MC path) | same | Discrete mixtures (η=0.02, M=2^{d/4}): d=32 **2.230** (1.8%), d=48 **3.373** (7.8%, 2 center seeds avg) vs paper {2.19, 3.13}; d=64 **4.039** (4.9%, 3 seeds avg, m=48, `#[ignore]`d recorded run `g1_mixture_ratio_d64_recorded`) vs paper 3.85 |
| G1 machinery cross-check | same | Profile-increments Ratio vs exact quadrature on the noisy bit (d=64, η=0.1, m=512): C=4.91 vs 4.80, P=1.64 vs 1.70, R=3.00 vs 2.82 (6%) |
| G1-cert coverage | same | Empirical frequency of {measured KL ≤ 4Ĉ/N} across 32 seeds × 2 cells (noisy bit d=6, η ∈ {0.2, 0.35}, K=2 blocks, N=6): **32/32 = 1.000 both cells** (bar 0.813 binomial-aware). KL measured by exact enumeration of the sampler's output law (3^d states) with zero-cost init+completion (sequential exact conditionals) |
| T1 interval estimator | same | Sandwich validated: raw Q̄=0.0397 ≤ H=0.0745 ≤ 2·Q̄=0.0794 (Lemma 1, d=48, η=0.15, [0.2,0.8], m=64); truncated Ĥ + r̂ upper-bound holds via the Bernstein radius |
| T3 construction | same | equal-√q grid: all 8 steps within ±10% of mean √q mass on the exact profile; DP ≤ uniform partition cost (1.232 ≤ 1.284) |
| G4 alloc | `tests/ugc_alloc_check.rs` (CountingAllocator) | **0 allocations** across 50 `estimate_interval` + 50 `bernoulli_unmask_with_grid` calls after scratch warm-up |
| **G1b promotion gate** | root `tests/ugc_g1b_gate.rs` (dllm, release, 5 cells) | **FAIL — negative result** (details below) |

## G1b — the falsifiable gate and why it failed

Setup: real `d2f_decode_block` path, `Config::micro_dllm()` trained 150 epochs on the pattern dataset, block_size ∈ {8, 15}, τ_conf ∈ {0.3, 0.7, 0.8}, 3 model seeds. ε(N) = held-out two-side-smoothed KL of per-position marginals + pairwise joints vs a 48-step reference decode (S=1024/512 per point). Ĉ from the UGC profile estimated THROUGH the real transformer (one forward per posterior; the estimator worked — C ∈ {7.0, 11.4, 17.7}, ratio 1.45–2.12: real geometry signal).

| Cell | ε(N) shape | Preset-8 actual passes | N* (UGC) | Reduction | Verdict |
|---|---|---|---|---|---|
| bs=8 s=42 τ=0.3 | flat at MC-noise floor (~4e-4) for all N ≥ 3 | 2.49 (early exit) | 48 (formula saturates at ε≈0) | **−8.6%** | FAIL |
| bs=15 s=42 τ=0.3 | flat (~8e-4) for N ≥ 3 | 2.84 | 48 | **+1.4%** | FAIL |
| bs=8 s=1337 τ=0.3 | flat (~5e-4) for N ≥ 3 | 2.23 | 48 | **−0.3%** | FAIL |
| bs=8 s=42 τ=0.7 | ε(2)=0.21, **exactly 0.0000 for all N ≥ 4** | 3.25 | 6 (scan fallback) | **+0.9%** | FAIL |
| bs=15 s=42 τ=0.8 | **exactly 0.0000 at every N ≥ 2** (N-invariant outputs) | 8.00 (no exit) | 2 (scan fallback) | **+75.0%** | degenerate "PASS" |

**Structural findings (the negative result's content):**

1. **The confidence-threshold loop is already pass-adaptive.** At τ ≤ 0.7 the loop early-exits at its convergence step (2.2–3.2 passes); the fixed preset wastes nothing. The reveal-time schedule axis the paper optimizes (where along the reveal path each step lands) does not exist in this loop — only `max_steps × early-exit` exists.
2. **The certificate N* formula does not transfer to this loop's semantics.** At the measured quality scale ε(8) ≈ 0 (the preset's output law is step-count-invariant once past minimal convergence — measured EXACTLY zero at τ=0.7/0.8), `N = 8Ĉ/ε` is undefined/saturated; the theory's object (KL of a random-order Bernoulli unmasking grid) is not this loop's error. This is Research 485 caveat 1 confirmed empirically.
3. **The one 75% cell is not a UGC win.** At bs=15/τ=0.8 the outputs are IDENTICAL at every N ≥ 2 (all revealing completes by step 2; the rest never crosses τ=0.8) — N*=2 came from the empirical scan fallback, not the certificate. The saving is available by lowering `denoise_steps`, no UGC needed (a preset-tuning observation).
4. **UGC estimation itself works on the real decode path** — C/ratio estimates through the transformer are stable and show real conditional structure. The substrate is a working diagnostic; the promotion claim (≥20% fewer forward passes at equal measured quality) has no purchase on this loop.

**Negative-result disposition (per Issue 664 T5):** no `ugc_schedule` feature flag; T1–T4 land as always-on diagnostic substrate in `katgpt-core::ugc_schedule`; T6 (freeze/thaw artifact) does not land (gated on T5 PASS); T7 (GPU consumer) stays deferred. Re-open trigger: adopting the paper's random-order reveal in a d2f variant — the reveal-time schedule axis would then exist and the certificate would have a semantics to bind to.

## Debugging record (what the harness caught — the reason the numbers are trustworthy)

Five real bugs were caught and fixed by the exact-check harness during development, each of which would have produced plausible-looking wrong numbers:

1. **Posterior orientation sign bug** (`1/(1+odds)` vs `odds/(1+odds)`): invisible to every entropy-based check (H_b is p↔1−p symmetric), corrupts sequential z-sampling and KL enumeration (measured KL 2.39 vs true ~0.3; MC h 2.79 vs 6.1 hand-check).
2. **Wrong m₀ law in the exact path** (Bin(j, ½) instead of the ½Bin(j,1−η)+½Bin(j,η) mixture): the correlated-through-U reveal values have a much wider m₀ law; fixed via the H_b-symmetry reduction to a single Bin(j, 1−η) average.
3. **Shared rolling posterior across coordinates** in `trajectory_q` (the T1 Q-statistic walk): coordinate i's "previous" posterior was coordinate d−1's. Fixed with per-coordinate rows.
4. **Heavy-tail noise in consecutive-KL increments** (the transition fires in one interval per trajectory): replaced by coupled h-curve differences (common random numbers) + the **integration-by-parts increment form** `ΔH = [t(1−t)h]ₚ^q + ∫(2t−1)h dt` — identically zero for constant h, killing the dominant correlated noise (measured: C inflated +70% at m=48 by midpoint weighting).
5. **2× prefactor in `coarse_complexity`**: the paper's `2ℓ_d` IS the full log-odds range `ln(ψ(T)/ψ(t0)) = 2ln(d−1)`; the code multiplied the range by 2 again. This single character explained the mixture Ratio being 2× the paper's (4.68 → 2.23 after fix).

Plus the T5 harness itself: one-sided smoothing against a near-deterministic reference is a ln(1/α)-per-cell artifact floor (~140 nats of pure floor in the first run) — fixed with two-sided smoothing; and the unused z-pool loop would hang forever at τ where 48-step decodes never fully converge (removed — `estimate_profile` draws its own z through the adapter).

## Substrate landed

`katgpt-core/src/ugc_schedule.rs` (always-on, zero-dep beyond `crate::types::Rng`):
- `UgcDenoiser` trait + `UgcScratch` (zero-alloc steady state)
- `estimate_interval` (Eq 32–34 dyadic truncated empirical-Bernstein), `estimate_profile` (coupled h-curve)
- `equal_sqrt_mass_grid`, `dp_partition`, `certified_block_plan`, `certified_iteration_count`, `reveal_grid_from_plan`
- `bernoulli_unmask_with_grid` (Eq 11 sampler, zero-defect init/completion)

Documented deviations from the paper: `B_α` estimated from the same trajectories (aggregate Minkowski bound) instead of assumed known — validated empirically by the 32/32 coverage; per-block `Ĥ_k + r̂_k` uppers use point estimates of the block masses.

## Run commands

```bash
# G1 + G1-cert + T3 + G4 (26 s release):
cargo test -p katgpt-core --release --test ugc_664_poc --test ugc_alloc_check -- --nocapture
# d=64 mixture recorded cell (~3 min):
cargo test -p katgpt-core --release --test ugc_664_poc g1_mixture_ratio_d64_recorded -- --ignored --nocapture
# T5 gate (5 cells, ~2 s after train):
cargo test --release --features dllm --test ugc_g1b_gate -- --nocapture
```
