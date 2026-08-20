# Issue 675: `tether` — closed-form outcome-fit estimator blend primitive (ρ* + EV + λ^(1/L) + admissibility gate)

**Filed:** 2026-08-20
**Source:** [Research 426](../../riir-train/.research/426_Le_Critique_PVF_TETHER.md) (riir-train) — arXiv:2608.16739 "Le Critique: Privileged Value Functions for LLM Reinforcement Learning" (TETHER baseline). Plan consumer: riir-train [Plan 345](../../riir-train/.plans/345_tether_pvf_loss_grpo.md).
**Class:** open primitive (Tier-0 public math, no game IP — the `entropic_tilt` precedent)

## Problem

The stack blends two estimators in at least four places, always with a **fixed or threshold rule**, never fit against realized outcomes:

| Site | Current mechanism | Hazard |
|---|---|---|
| `flow/cache.rs::get_or_compute_dual{,_postmax}` (`DualLeoMixer`) | fixed `alpha: f32` param; cache docs assume α fixed per goal | Bench 553: fixed α + harmful student measured **−100%** |
| `KvRoutingConfig::blend_factor` | fixed confidence thresholds | EVPO-style hard rule — dominated by soft blend in-sample |
| `entropic_tilt` consumers (`loss_grpo`) | pure LOO/group baseline, no value-side endpoint | Plan 336 T3.4: constant-reward groups → **bit-exact zero advantage** → no gradient at p ∈ {0,1} |
| `committed_field_blend` | weights frozen at commit (by design — personality, NOT a fix target) | n/a (different semantics; listed for signal-diff completeness) |

arXiv:2608.16739 ships the cure as closed-form math: blend `b(ρ) = (1−ρ)·p1 + ρ·p2`, fit ρ per window by OLS against realized outcomes,

```
ρ* = clip[0,1]( Σ A_B·Δ / Σ Δ² ),  A_B = R − p1,  Δ = p2 − p1
```

with the **exact in-sample guarantee** `SSE(ρ*) ≤ min(SSE(p1), SSE(p2))` (convexity + both endpoints feasible — no conditions), EMA smoothing `ρ_k = d·ρ_{k−1} + (1−d)·ρ̂_k`, and the **lag law**: ρ fit on window k applies only to window k+1 (same-window application makes the baseline a function of its own returns = bias by construction — the admissibility condition).

Prior art honesty: forecast combination (Bates–Granger) is 50+ years old; EVPO (arXiv:2604.19485) hard-switches on explained variance. The primitive's value is the **never-worse guarantee + lag/EMA contract encoded as an API shape** (same-window application unrepresentable), not new math.

## Proposed surface (katgpt-core, feature `tether`, opt-in)

1. **`TetherBlend`** — 3-accumulator online fit: `observe(r, p1, p2)`, `rho_hat()` (batch ρ*), `rho()` (EMA-smoothed, lagged), `blend(p1, p2) -> f32`. Degenerate guard `ΣΔ² < ε` keeps previous ρ. Zero heap.
2. **Lag law as API shape** — `publish(window)` ⇒ applies to windows > t; a same-window blend call is a compile-time or debug-assert rejection (test-pinned).
3. **`EvAccumulator`** — one-pass explained variance `EV = 1 − Var(R−V)/Var(R)` (5 running sums, Welford-style stable path) + the control-variate check `Var(R−V) < Var(R) iff 2Cov(R,V) > Var(V)`. Ships as a telemetry pair with ρ̂.
4. **`horizon_decay(c, L) -> f32` LUT** — λ = c^(1/L) ("retain fraction c after horizon L"); the paper's near-1 λ finding (0.4^(1/8192) ≈ 0.9999 beats 1.0 at 8k).
5. **Admissibility vocabulary** — doc-level per-channel rule (own-future inadmissible; independent-instance sibling outcomes admissible; contested shared encounters violate it). Monte-Carlo fixture: mean advantage preserved under admissible z, measurably biased under a leaking z.

Consumers (≥2 required):

1. riir-train `loss_grpo` TETHER baseline (Plan 345) — the p1=LOO / p2=value-head endpoint pair.
2. riir-clippy **selection blend** (riir-clippy Issue 033) — `select_best_candidate`'s `W_EVO·evo + W_RATE·reliability` is literally `b(ρ)=(1−ρ)p1+ρp2` with ρ hand-pinned at 0.4 (`const W_EVO: f32 = 0.6; const W_RATE: f32 = 0.4;`) and never swept; the `EvolveRecorder::record_outcome` stream is the realized-outcome feed. A/B on the existing Issue-026 harness. (User-challenge find 2026-08-20 — promoted from the original R2 deferral, which had conflated this with the retrieval-layer fusion.)
3. `DualLeoMixer` adaptive-α variant for future consumers (seal, quest) — NOT a civ reopen (riir-ai 322 stop rule honored; see Research 426 §6 R1).
4. (Fusion candidates, unfunded) KARC×conformal-floor regime blending; deliberation route-scoring; cohort LOO credit.

## Report-the-Floor hazard (pinned in-source)

Blending a UQ primitive WITH the conformal-naive floor does **not** discharge the Issue-010 promotion gate — the primitive itself must beat the floor unblended. Runtime blending is orthogonal to the promotion gate; a floor+noise blend must not be citable as "beats the floor".

## Tasks

- [ ] T1: `tether` feature + `TetherBlend` (ρ* closed form, EMA, lag contract; degenerate guards) + unit fixtures: brute-force grid argmin equivalence, exact in-sample never-worse asserts (both endpoints), complementary/identical/anti-complementary error regimes, determinism given identical outcome streams
- [ ] T2: lag-law API shape + test that same-window application is rejected; known-answer EMA fixtures (constant ρ̂ ⇒ geometric convergence, exact pinned values)
- [ ] T3: `EvAccumulator` (stable one-pass, two-pass reference cross-check) + control-variate iff-check helper
- [ ] T4: `horizon_decay` LUT (`λ^L == c` within 1e-9, one exp/log path) + out-of-sample holdout fixture: `SSE_out(soft) ≤ SSE_out(best fixed endpoint)` on a stationary process + drift fixture asserting EMA tracking
- [ ] T5: admissibility doc-rule + Monte-Carlo mean-preservation/bias fixture pair
- [ ] T6: GOAT gate — G1 (above fixtures, bit-identical two runs), G2 (<100ns per observe at K≤64), G3 (opt-in, default surface untouched), G4 (counting allocator: 0 steady-state allocs); report-the-floor hazard comment in-source
- [ ] T7: promote/demote ruling recorded in Research 426 + katgpt-rs research note follow-up if promoted
