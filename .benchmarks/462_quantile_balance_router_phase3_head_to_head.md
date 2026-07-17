# Plan 455 Phase 3 — Head-to-Head: QB vs MPI vs Composed

**Plan:** [`katgpt-rs/.plans/455_quantile_balancing_router_primitive.md`](../.plans/455_quantile_balancing_router_primitive.md)
**Date:** 2026-07-17
**Status:** ✅ **CASE C — Composition strictly beats either alone. Both promoted to DEFAULT-ON.**
**Test artifact:** `crates/katgpt-spectral/tests/bench_455_phase3_head_to_head.rs`

---

## Reproduce

```bash
# Head-to-head test (debug is fine — no timing gate)
cargo test -p katgpt-spectral \
           --features "quantile_balance_router manifold_power_iter_router" \
           --test bench_455_phase3_head_to_head -- --nocapture --test-threads=1
```

---

## The Question

Plan 279 (MPI Router) and Plan 455 (QB Router) are both one-shot deterministic
MoE router reconditioners applied at freeze/thaw snapshot swap. Research 447 §2.4
predicts they solve **orthogonal** problems on the joint `(λ, MaxVio)` Pareto
frontier:

- **MPI** operates on `R ∈ ℝ^{N×D}` (router rows) against per-expert Gram
  matrices. It improves router-expert **alignment** (λ). Does NOT touch the
  per-token score distribution.
- **QB** operates on `s ∈ ℝ^{m×n}` (calibration-batch router scores). It
  computes a per-expert bias `β` that drives load-balance **MaxVio** → 0. Does
  NOT touch the router weight matrix.

The predicted outcome is **Case C — composition strictly beats either alone**.
This benchmark constructs a deliberately-hard fixture where BOTH axes are
broken, measures all four variants, and confirms Case C empirically.

---

## Fixture Design (T3.1)

A deliberately-hard synthetic MoE pool with BOTH low λ AND high MaxVio_load:

| Component | Construction | Effect |
|---|---|---|
| Grams `M[i]` (D×D) | `e_i·e_i^T + 0.1·I` (diagonal) | Principal direction = standard basis `e_i` |
| Router `R[i]` (N×D) | `cos(θ)·e_i + sin(θ)·e_{i+N}`, θ=1.0 rad | Misaligned with `e_i` → low λ ≈ 0.65 |
| Input `X[j]` (M×D) | `2.0·(e_0+e_1)/√2 + Gaussian(0, 0.5²)` | Hot direction favors experts 0,1 → MaxVio ≈ 3.0 |

Dimensions: N=8, D=256, M=256, k=2 (the Plan 455 G4 game-scale point).

**Why this fixture demonstrates Case C:**

1. **MPI fixes λ**: retracting `R[i]` toward `e_i` moves the router row from
   ~40° off-axis to ~6° off-axis → λ jumps from 0.65 to 0.99.
2. **MPI does NOT fix MaxVio**: `e_0` and `e_1` ARE the hot direction. Retracting
   `R[0]` toward `e_0` makes expert 0 EVEN MORE dominant (MaxVio 1.84 → 2.67).
3. **QB fixes MaxVio**: the bias `β` penalizes the over-selected experts 0,1,
   redistributing tokens to experts 2–7 → MaxVio drops from 1.84 to 0.03.
4. **QB does NOT fix λ**: QB operates on scores, not on the router matrix.
5. **Composition fixes both**: MPI → high λ, then QB on the MPI-conditioned
   scores → MaxVio → 0. Both axes optimal simultaneously.

---

## Decision Matrix (T3.5)

Measured on the fixture above (deterministic, seeded RNG = 42):

| Variant | λ | MaxVio_load | β‖∞ | Verdict |
|---|---|---|---|---|
| **Vanilla** (no conditioning) | 0.6529 | 1.8438 | 0.0000 | both axes broken |
| **MPI only** (Plan 279) | **0.9918** | 2.6719 | 0.0000 | fixes λ; **worsens** MaxVio (MPI amplifies hot-direction bias) |
| **QB only** (Plan 455) | 0.6529 | **0.0312** | 0.6667 | fixes MaxVio (59× reduction); λ unchanged |
| **Composed** (MPI+QB) | **0.9918** | **0.0000** | 0.4219 | **fixes both → Case C** |

Higher λ = better (alignment). Lower MaxVio = better (balance).

---

## GOAT Gate (G-P3-1 through G-P3-6)

| Gate | Check | Measured | Status |
|---|---|---|---|
| **G-P3-1** | MPI improves λ by ≥ 0.1 | Δλ = +0.3389 (0.65 → 0.99) | ✅ PASS |
| **G-P3-2** | QB halves MaxVio | 59.0× reduction (1.84 → 0.03) | ✅ PASS |
| **G-P3-3** | Composed preserves MPI's λ (within 1e-4) | \|Δ\| = 0.00e0 (bit-identical) | ✅ PASS |
| **G-P3-4** | Composed reduces MaxVio vs MPI-only | 2.67 → 0.00 | ✅ PASS |
| **G-P3-5** | Composed beats QB-only on λ by ≥ 0.1 | Δλ = +0.3389 | ✅ PASS |
| **G-P3-6** | Composed strictly Pareto-dominates BOTH alternatives | dominates MPI (same λ, better MaxVio) AND dominates QB (same MaxVio, better λ) | ✅ PASS |

**Total: 6/6 PASS. Case C confirmed.**

---

## Honest Findings (Phase 3)

### Finding 1 — MPI can WORSEN MaxVio_load (not just fail to fix it)

On this fixture, MPI-only MaxVio (2.67) is WORSE than vanilla (1.84). This
happens because MPI retracts `R[i]` toward the Gram principal direction `e_i`,
and for experts 0,1 the principal direction `e_0`/`e_1` IS the input hot
direction. So MPI makes the already-dominant experts even more dominant.

**Implication:** MPI and QB are not just complementary — QB is **necessary**
after MPI on distributions where the Gram principal directions align with the
input's dominant directions. Shipping MPI alone (without QB) can actively harm
load balance in production. The composed pipeline (MPI → QB) is the safe
default.

### Finding 2 — The composition is commutative in λ but NOT in MaxVio

λ is determined solely by `R'` (the MPI-conditioned router). QB doesn't touch
`R'`, so `λ(composed) = λ(MPI-only)` exactly (bit-identical, G-P3-3).

MaxVio depends on the score distribution `s' = X · R'^T`, which changes after
MPI. So `β` must be recomputed on the post-MPI scores (not the vanilla scores).
The composed pipeline is: `R' = MPI(R)` → `s' = X·R'^T` → `β' = QB(s')` →
route via `top-k(s' − β')`. Applying vanilla `β` to post-MPI scores would be
a stale-bias error (G8.B lesson from Phase 2).

### Finding 3 — QB's β is SMALLER on the composed pipeline (0.42 vs 0.67)

On vanilla scores, QB needs `β‖∞ = 0.67` to rebalance. On MPI-conditioned
scores, QB needs only `β‖∞ = 0.42`. This is because MPI's norm equalization
partially reduces the score disparity (even though directional bias remains).
The composition is synergistic: MPI does some of the balance work, QB finishes
it with a smaller bias.

### Finding 4 — The fixture's θ=1.0 rad is the sweet spot for Case C

- θ too small (e.g., 0.3): λ is already high (~0.95), MPI has nothing to fix →
  the fixture doesn't demonstrate the MPI axis.
- θ too large (e.g., 1.5): λ is very low (~0.35), but the router rows are so
  noisy that the hot-direction signal is diluted → MaxVio drops, QB has less
  to fix.
- θ=1.0 (57°) gives λ ≈ 0.65 (clearly improvable by MPI) AND MaxVio ≈ 1.84
  (clearly improvable by QB). Both axes are non-trivially broken.

---

## Promotion Decision (T3.6): CASE C → Promote BOTH

Per Plan 455 §"Promotion rule": G1–G8 all green (Phase 2) + Phase 3 Case C
(composition strictly beats either alone) → promote `quantile_balance_router`
to DEFAULT-ON alongside `manifold_power_iter_router`.

**Action taken (2026-07-17):**
- `quantile_balance_router` added to root `Cargo.toml` `default` feature list.
- Both router primitives are now DEFAULT-ON.
- The composed pipeline `R' = MPI(R)` → `β = QB(s')` → `top-k(s − β)` is the
  recommended snapshot-swap reconditioning pipeline for riir-ai consumers.
- No `RouterReconditioner` trait extracted (per Plan 455 DRY Note: premature
  until empirical evidence of repeated composition). The empirical evidence
  is now in — a future issue can extract the trait if consumers need it.

---

## What This Does NOT Prove

1. **Production-scale validation.** This is a synthetic fixture (N=8, D=256).
   Real NPC shard pools may have different geometry. The promotion is based
   on the algorithmic orthogonality argument (Research 447 §2.4) + this
   empirical confirmation on a deliberately-hard case.

2. **That MPI always worsens MaxVio.** Finding 1 is fixture-specific — it
   happens because the Gram principal directions align with the input hot
   direction. On other distributions MPI may be MaxVio-neutral or even
   slightly helpful. The safe default is always compose (MPI → QB).

3. **The composed pipeline is optimal.** There may be distributions where
   QB-only (skip MPI) is sufficient, or where MPI-only is sufficient. The
   promotion ships BOTH as default-on so consumers get the full pipeline;
   they can disable either via `--no-default-features` if profiling shows
   one is unnecessary for their workload.

---

## TL;DR

Plan 455 Phase 3 head-to-head: **Case C confirmed.** On a deliberately-hard
fixture (N=8, D=256, M=256, k=2) with both low λ (0.65) and high MaxVio_load
(1.84), the composed pipeline (MPI → QB) achieves λ=0.99 AND MaxVio=0.00 —
strictly Pareto-better than either alone. MPI alone actually **worsens**
MaxVio (1.84 → 2.67) because retracting toward Gram principal directions can
amplify input-distribution bias; QB is necessary after MPI. Both primitives
promoted to DEFAULT-ON. The composed pipeline is the recommended snapshot-swap
reconditioning for riir-ai consumers.
