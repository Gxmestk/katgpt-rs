# Issue 664: UGC certified-unmasking-schedule PoC (arXiv:2608.13520)

> **Research:** [katgpt-rs/.research/485_UGC_Certified_Unmasking_Schedules.md](../.research/485_UGC_Certified_Unmasking_Schedules.md)
> **Source paper:** [arXiv:2608.13520](https://arxiv.org/abs/2608.13520) — Wainwright, "The data geometry of masking diffusion: Certified-optimal schedules via unmasking growth complexity" (2026-08-13)
> **Verdict that opened this:** Gain (GOAT-track pending PoC) — modelless-validable, zero shipped substrate, but gains confined to the D2F decode path at block sizes below the paper's asymptotic regime. This issue is the falsifiable gate.
> **Opened:** 2026-08-16
> **CLOSED:** 2026-08-17 — **G1/G1-cert/G4 PASS, G1b FAIL (honest negative result).** Record: [Bench 659](../.benchmarks/659_ugc_certified_schedule_poc.md). Estimator + schedule construction landed as always-on diagnostic substrate (`katgpt-core::ugc_schedule`); no `ugc_schedule` feature flag; GOAT-track closed. Re-open trigger: a d2f variant adopting the paper's random-order reveal (the reveal-time schedule axis would then exist for the certificate to bind to).

## Goal

Prove or refute, on controlled toys AND real decode shapes, that UGC-based certified schedules (a) reproduce the paper's published numbers, (b) validate their high-probability KL certificate empirically, and (c) beat the hardcoded D2F step presets (8/12/4) at equal measured quality. Promotion to a feature-flag plan happens ONLY on T5 PASS; a negative T5 keeps the estimator as a diagnostic + closes the GOAT-track.

All components are modelless (Monte Carlo over a frozen denoiser's posteriors; closed-form schedule math; truncated empirical Bernstein). No training, no riir-train dependency.

## Tasks

- [x] **T1 — UGC increment estimator** (`katgpt-core`, new `ugc_schedule.rs` sibling to `set_diffusion_schedule.rs`): forced-mask coupled reveal trajectories; reveal-odds-dyadic grid (ψ(v_j) = min{2^j·ψ(p), ψ(q)}); trajectory statistic Q(ℓ) (paper Eq 32b); truncated estimator Ĥ_m + empirical-Bernstein radius r̂_m (Eq 33/34). Zero-alloc: fixed-size dyadic arrays + scratch buffers; `f32` in, `f32` out. Landed always-on (`pub mod ugc_schedule`, no gate — same rationale as `set_diffusion_schedule`). Debug-driven fixes: per-coordinate posterior rows (shared rolling buffer corrupted the Q statistic); documented B̂-from-same-trajectories deviation (Minkowski aggregate), validated by T4's 32/32 coverage.
- [x] **T2 — Exact-check harness** (`tests/ugc_664_poc.rs`): Ratios reproduced — noisy bit d=128 η∈{0.01,0.30,0.45} → **4.511/2.165/1.653** (paper 4.51/2.16/1.65, 3 digits, exact Bernstein+GL path); mixtures d∈{32,48,64} → **2.230/3.373/4.039** (paper 2.19/3.13/3.85; 1.8%/7.8%/4.9%, MC path; d=64 is the `#[ignore]`d recorded run). Closed forms q_rep/q_par (Eq 24a/24b) rel < 1e-9; H(0,1) = ((d−1)/(d+1))·ln2 via direct integration AND the TSE identity (NOT the mangled rendered fraction — confirmed). Two harness-caught math bugs on the way: posterior orientation sign (entropy-invisible, sampling-fatal) and the m₀ mixture law (Bin(j,1−η), not Bin(j,½)).
- [x] **T3 — Schedule construction:** equal-√q-mass N-step grid (all steps within ±10% of mean √q mass on the exact profile), K-block multipliers ρ_k (Prop 1 Eq 28a), DP block boundaries (edge cost √(S·H), Eq 39; DP 1.232 ≤ uniform 1.284). Hosted in `katgpt-core::ugc_schedule` (`ScheduleKind`-adjacent constructors returning the same `Vec<f32>` step-grid shape); NO feature flag — per the negative T5, no production d2f consumer wiring lands.
- [x] **T4 — Certificate validation:** 32 seeds × 2 cells (noisy bit d=6, η∈{0.2,0.35}, K=2 blocks, N=6): empirical frequency of {KL ≤ 4Ĉ/N} = **32/32 = 1.000 both** (binomial-aware bar 0.813). KL measured by exact enumeration of the sampler's output law over 3^d states with zero-cost init/completion. Coverage measured, never asserted — Report-the-Floor discipline held.
- [x] **T5 — Falsifiable promotion gate (G1b): FAIL (honest negative result).** 5 cells (bs∈{8,15} × τ∈{0.3,0.7,0.8} × 3 model seeds) on the real `d2f_decode_block` path with a trained micro-dLLM. Reductions −8.6%…+1.4% on non-degenerate cells (bar ≥20%); the single +75% cell is degenerate (outputs N-invariant, N*=2 from the scan fallback, not the certificate). Structural cause: the confidence-threshold loop early-exits at convergence (preset wastes nothing) and the certificate's N=8Ĉ/ε is undefined at the measured ε(8)≈0 — Research 485 caveat 1 (random-order vs confidence-threshold reveal) confirmed empirically. Full evidence: [Bench 659](../.benchmarks/659_ugc_certified_schedule_poc.md). Negative-result disposition executed: negative result recorded, estimator kept diagnostic-only, this issue closed.
- [-] **T6 — Freeze/thaw artifact:** not landed — gated on T5 PASS.
- [-] **T7 — GPU consumer** (`riir-gpu/gemma2_d2f`): deferred (was already; moot after T5 FAIL).

## Gate summary (final)

| Gate | Content | Threshold | Result |
|---|---|---|---|
| G1 | paper-number reproduction (Ratios, closed forms) | within ~5% of published values | **PASS** (exact path: 3 digits; MC path ≤ 7.8%) |
| G1-cert | certificate coverage frequency | ≥ 1−η empirically | **PASS** (32/32, 1.000) |
| G1b | real-shape step-count reduction at equal quality | ≥ 20% fewer passes | **FAIL** (−8.6%…+1.4%; one degenerate +75% not UGC-derived) |
| G2 | estimator amortization | once per model (T6 artifact) | [-] moot (T5 FAIL) |
| G4 | allocator-free steady state | existing `CountingAllocator` pattern | **PASS** (0 allocs / 100 calls) |

**Companion G4 test:** `tests/ugc_alloc_check.rs` (separate CountingAllocator binary).
