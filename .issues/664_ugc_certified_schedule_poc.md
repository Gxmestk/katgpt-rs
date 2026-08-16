# Issue 664: UGC certified-unmasking-schedule PoC (arXiv:2608.13520)

> **Research:** [katgpt-rs/.research/485_UGC_Certified_Unmasking_Schedules.md](../.research/485_UGC_Certified_Unmasking_Schedules.md)
> **Source paper:** [arXiv:2608.13520](https://arxiv.org/abs/2608.13520) — Wainwright, "The data geometry of masking diffusion: Certified-optimal schedules via unmasking growth complexity" (2026-08-13)
> **Verdict that opened this:** Gain (GOAT-track pending PoC) — modelless-validable, zero shipped substrate, but gains confined to the D2F decode path at block sizes below the paper's asymptotic regime. This issue is the falsifiable gate.
> **Opened:** 2026-08-16

## Goal

Prove or refute, on controlled toys AND real decode shapes, that UGC-based certified schedules (a) reproduce the paper's published numbers, (b) validate their high-probability KL certificate empirically, and (c) beat the hardcoded D2F step presets (8/12/4) at equal measured quality. Promotion to a feature-flag plan happens ONLY on T5 PASS; a negative T5 keeps the estimator as a diagnostic + closes the GOAT-track.

All components are modelless (Monte Carlo over a frozen denoiser's posteriors; closed-form schedule math; truncated empirical Bernstein). No training, no riir-train dependency.

## Tasks

- [ ] **T1 — UGC increment estimator** (`katgpt-core`, new `ugc_schedule.rs` sibling to `set_diffusion_schedule.rs`): forced-mask coupled reveal trajectories; reveal-odds-dyadic grid (ψ(v_j) = min{2^j·ψ(p), ψ(q)}); trajectory statistic Q(ℓ) (paper Eq 32b); truncated estimator Ĥ_m + empirical-Bernstein radius r̂_m (Eq 33/34). Zero-alloc: fixed-size dyadic arrays + scratch buffers; `f32` in, `f32` out.
- [ ] **T2 — Exact-check harness** (paper's own ensembles): noisy repeated bit η ∈ {0.01, 0.30, 0.45} (target Ratios 4.51 / 2.16 / 1.65), discrete mixtures d ∈ {32, 48, 64} (2.19 / 3.13 / 3.85), parity/repeated-bit closed forms q_rep/q_par (Eq 24a/24b), H(0,1) = ((d−1)/(d+1))·log2 via the TSE identity (NOT the mangled rendered fraction — see Research 485 §1 fetch caveat). A tiny exact posterior toy (repeated bit has analytic μ_i) avoids needing a trained model for G1.
- [ ] **T3 — Schedule construction:** equal-√q-mass N-step grid; K-block geometric multipliers ρ_k = min(1, 4√(C/N)·√(S_k/H_k)); DP block-boundary selection (edge cost e(i,j) = √(S·H), paper Eq 39). Host as a `ScheduleKind`-adjacent constructor in `katgpt-forward` — but NO feature flag yet (see T5).
- [ ] **T4 — Certificate validation:** across ≥ 32 seeds × several (N, η, ensemble) cells, the empirical frequency of {KL ≤ 4Ĉ/N + init + completion} must be ≥ 1−η (binomial check). Report-the-Floor discipline: coverage measured, never asserted.
- [ ] **T5 — Falsifiable promotion gate (G1b):** on real decode shapes (block_size 8–16, an actual d2f decode path from `katgpt-forward`), UGC-adaptive step counts reach equal *measured* sample quality (held-out KL or proxy) with **≥ 20% fewer forward passes** than the fixed presets. PASS → open the `ugc_schedule` feature-flag plan (GOAT G1–G4 per Research 485 §4). FAIL → record the negative result in `.benchmarks/`, keep T1–T4 estimator as diagnostic-only substrate, close this issue.
- [ ] **T6 — Freeze/thaw artifact:** if T5 passes, the estimated schedule serializes as a committed (BLAKE3) versioned artifact reused across decode calls — estimator cost paid once per model.
- [-] **T7 — GPU consumer** (`riir-gpu/gemma2_d2f`): deferred until T5 passes. Includes the honest caveat that the paper's certificate covers random-order reveal, not the loop's confidence-threshold reveal (Research 485 caveat 1) — needs the random-subset decode variant or a bound extension.

## Gate summary

| Gate | Content | Threshold |
|---|---|---|
| G1 | paper-number reproduction (Ratios, closed forms) | within ~5% of published values |
| G1-cert | certificate coverage frequency | ≥ 1−η empirically |
| G1b | real-shape step-count reduction at equal quality | ≥ 20% fewer passes |
| G2 | estimator amortization | once per model (T6 artifact) |
| G4 | allocator-free steady state | existing `CountingAllocator` pattern |
