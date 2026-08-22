# Benchmark 573 — CCE Simplex Solver at Scale (Plan 572 GOAT Gate)

**Date:** 2026-08-06
**Plan:** [572](../.plans/572_cce_simplex_solver_at_scale.md)
**Feature:** `cce_moderator` (already default-on; Plan 572 extends it)
**Test:** `tests/cce_simplex_goat.rs`

## TL;DR

The bounded two-phase primal simplex + constraint-generation wrapper ships as
the production solver for CCE LPs too large for BFS enumeration. The 2-player
RPS case at NA=81 (the production case from Issue 575 that the BFS solver
cannot solve) solves in **~2 ms** via constraint generation (4 active
constraints at convergence). All 5 GOAT gates PASS.

## What shipped

### Layer 1 — `simplex.rs` (bounded two-phase primal simplex)

`pub(crate) fn solve_simplex(mat, rhs, obj, n_vars, na) -> Option<Vec<f64>>`

Drop-in replacement for `enumerate_bfs`:
- Phase I: artificial variables to find an initial feasible basis.
- Phase II: minimize the original objective from the Phase I basis.
- Bland's rule (smallest-index entering variable with negative reduced cost;
  smallest-index tiebreak on min-ratio leaving variable) — guarantees finite
  termination, fully deterministic, no RNG.
- 8 unit tests: trivial, infeasible, redundant constraint, chicken parity,
  emission parity, 2-var intersection, determinism, large LP (NA=81 shape).

### Layer 2 — auto-selection in `solve` / `solve_heterogeneous`

`solve_lp_auto()` estimates `C(n_vars, n_cons)` and selects:
- BFS enumeration if `C(n_vars, n_cons) ≤ 50_000` (exact, fast for small LPs).
- Two-phase simplex otherwise.

This is transparent — same signature, same output contract. All existing CCE
tests pass bit-identically (auto-select picks BFS for every existing test).

### Layer 3 — constraint generation wrapper

`solve_heterogeneous_cg()` + `solve_heterogeneous_cg_with_tolerance()`:

Starts with no deviation constraints (just sum-to-1), solves the relaxed LP,
finds the most-violated deviation via the external-regret separation oracle,
adds it, and re-solves. Converges in `O(|active set|)` iterations — far fewer
than `Σ_i |D_i|` because only the binding deviations end up active.

## GOAT gate (T4) — 2-player RPS at NA=81

The production case from Issue 575: N=9 (joint recommendation), A=9 (joint
play), 3 deviations per player (6 total). The full LP has `n_vars = 87,
n_cons = 7`, giving `C(87, 7) ≈ 3.6 × 10^10` BFS candidates — intractable for
enumeration.

| Gate | Target | Result |
|---|---|---|
| G1 — correctness at scale | `solve_heterogeneous_cg` returns a valid CCE on NA=81 | ✅ PASS — `is_heterogeneous_cce(ε=1e-4) == true` |
| G2 — non-regression | all existing CCE tests pass bit-identically | ✅ PASS — 57 CCE lib tests pass (55 existing + 2 CG) |
| G3 — artifact rejection | solver does NOT return the RPS trivial artifact | ✅ PASS — γ₀(P1) ≥ 0 (artifact would be -1.0); mass on (P,R) < 0.99 |
| G4 — determinism | two runs produce bit-identical ρ | ✅ PASS — `rho1.entries == rho2.entries` |
| G5 — modelless | pure linear algebra, no training, no deps | ✅ PASS — zero new Cargo.toml deps |

## Perf characterization (T5 — informational)

| Metric | Value |
|---|---|
| `solve_heterogeneous_cg` wall-clock at NA=81 | ~2 ms (debug mode, Apple Silicon) |
| Constraint-generation iterations to converge | 4 |
| Inner LP size at convergence | 1 + 4 = 5 constraints |
| Full LP (for comparison) | 1 + 6 = 7 constraints, `C(87,7) ≈ 3.6×10^10` BFS candidates |

The constraint-generation wrapper converges in 4 iterations (well under the
6-deviation worst case), meaning only 4 of the 6 deviations are binding at the
optimum. The inner LPs are tiny (5 constraints at most), so the per-iteration
cost is dominated by the separation oracle (6 deviation evaluations), not the
LP solve.

## What CCE does the solver find?

The 2-player RPS is a zero-sum game. The Nash equilibrium is the uniform
honest joint distribution (each `(s₁,s₂)` gets mass 1/9, play = recommendation),
giving γ₁ = γ₂ = 0. The CG solver finds a CCE with γ₀ ≥ 0 — consistent with
the Nash equilibrium value. For a zero-sum game, the set of CCEs includes the
Nash equilibrium, and the moderator objective γ₀ = 0 at Nash.

## Non-goals (reaffirmed)

- No general LP solver library (simplex handles CCE LP shape only).
- No interior-point method (simplex returns a vertex BFS, matching output contract).
- No revised simplex with explicit basis inverse (deferred until a concrete consumer needs it).
- No parallel simplex (pivot loop is sequential for determinism).

## Cross-references

- [Plan 572](../.plans/572_cce_simplex_solver_at_scale.md) — the execution plan.
- Issue 575 (removed) — RPS artifact closure (verify-only at NA=81; this plan adds solve). Verdict captured in Research 468 §9.
- [Plan 569](../.plans/569_transition_kernel_cce.md) — transition-kernel CCE (predecessor).
- `.docs/04_calibration/cce_moderator.md` — API reference (updated by T6).
