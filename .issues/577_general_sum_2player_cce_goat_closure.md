# Issue 577 — General-Sum 2-Player CCE GOAT Closure

**RESOLVED + removed per noise-reduction rule.** Full narrative below.

## Context

Plan 572 shipped `solve_heterogeneous_cg` (constraint-generation simplex solver)
and closed the 2-player RPS CCE at NA=81. The GOAT gate
(`tests/cce_simplex_goat.rs`) tested G1–G5 on **zero-sum** RPS only.

The substrate is **already general-sum by construction**:
- `HeterogeneousPayoff<N, A>` accepts per-player cost tensors via
  `reward_follow(player, state, action)`.
- The default moderator objective `gamma0` averages per-player welfare
  `(1/P) Σ_i γ_i(ρ)` — no zero-sum assumption.
- The LP deviation constraints are "no player can profit by deviating" —
  fully general.

## Substrate check (substrate-first skill)

- Searched for: `general_sum`, `GeneralSum`, `asymmetric`, `two_player`,
  `HeterogeneousPayoff`, `solve_heterogeneous_cg`, `is_heterogeneous_cce`
- Found: substrate IS general-sum by construction. No new code needed —
  only a GOAT test gap.
- Decision: **verify-only** (add general-sum test to existing GOAT gate).

## Tasks (ALL DONE)

- [x] T1: Added general-sum 2-player Chicken + PD tests to
      `tests/cce_simplex_goat.rs`. 6 new tests:
      - `g1_gen_chicken_cg_matches_bfs` — CG-vs-BFS parity (strongest correctness)
      - `g1_gen_chicken_valid_cce` — convergence + valid CCE + sums-to-1
      - `g3_gen_chicken_no_profitable_deviation` — direct CCE constraint check
      - `g4_gen_chicken_deterministic` — bit-identical two runs
      - `g1_gen_pd_cg_matches_bfs` — CG-vs-BFS parity on PD
      - `g1_gen_pd_valid_cce` — convergence + valid CCE
- [x] T2: PD included in T1 (general-sum with asymmetric payoffs).
- [x] T3: Full CCE test suite — 57 lib tests + 11 GOAT gate tests, zero regression.
- [x] T4: `cce_moderator.md` Limitation #1 updated — general-sum closure noted.
- [x] T5: Committed. Highwater updated. This issue removed.

## Key finding: model inconsistency (not a solver bug)

The individual-action model with shared `ρ` has a known inconsistency for
multi-player general-sum games: `γ_i` assumes the OTHER player follows their
recommendation, so both players' costs can be very negative simultaneously
(e.g., PD gamma0 = -5.0, which would imply both players get reward 5 —
physically impossible since that requires (D,C) and (C,D) simultaneously).

This is a MODEL property, not a solver bug. The GOAT tests verify SOLVER
correctness via CG-vs-BFS parity (both solvers agree on the inflated objective)
+ CCE validity (the constraints ARE satisfied). Welfare bounds from naive
gamma0 interpretation don't hold in this model; a proper joint-action welfare
computation would require a different formulation.

## References

- `tests/cce_simplex_goat.rs` — GOAT gate (now covers zero-sum RPS + general-sum Chicken/PD)
- `crates/katgpt-core/src/cce/heterogeneous.rs:207` — existing PD verify-only test
- `crates/katgpt-core/src/cce/external_regret.rs:623` — existing Chicken verify-only test
- Plan 572 — constraint-generation solver
