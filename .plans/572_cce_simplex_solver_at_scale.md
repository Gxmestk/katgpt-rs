# Plan 572 — CCE Simplex Solver at Scale (Constraint Generation + Two-Phase Primal Simplex)

**Status:** complete (T1–T6 all done)
**Branch:** `develop`
**Feature gate:** `cce_moderator` (already default-on; this plan extends it, no new gate)
**Predecessors:**
- Plan 295 (LP-CCE moderator primitive — BFS enumeration solver)
- Plan 300 (heterogeneous CCE + ExternalRegret separation oracle)
- Plan 569 (transition-kernel-constrained CCE — `solve_with_dynamics`)
- Issue 575 (RPS artifact CLOSED via 2-player CCE — verify-only at NA=81)

**Motivation:** Issue 575 Part B proved the 2-player CCE path (`is_heterogeneous_cce` with N=9, A=9 joint-action model) correctly rejects the RPS trivial artifact. But the existing BFS solver cannot **solve** the LP at NA=81 — `C(87, 7) ≈ 3.6 × 10^10` BFS candidates is intractable. The substrate can VERIFY closure but not SOLVE for the optimal CCE on larger games. This is the pre-existing limitation #3 in `.docs/04_calibration/cce_moderator.md`. Production use (riir-ai Plan 325 runtime routing zero-sum games to the 2-player path) needs a real solver.

**Substrate check (substrate-first skill):**
- Searched for: `simplex`, `linear_program`, `lp_solve`, `solve_heterogeneous`, `is_heterogeneous_cce`, `HeterogeneousPayoff`, `gaussian_elimination`, `lu_decomposition`, `matrix_inverse`, `solve_linear_system`, `partial_pivot`.
- Found:
  - `CceLp::solve_heterogeneous` (lp.rs:269) — BFS enumeration, exact for `N·A + |D| ≤ ~25`.
  - `solve_square_system` (lp.rs:451) — Gaussian elimination with partial pivoting (reusable as the inner linear-algebra kernel).
  - `ExternalRegret::best_deviation` (external_regret.rs:63) — the **separation oracle** for constraint generation (finds the most-violated deviation).
  - `solve_linear_system` in `tests/occupancy_baird_mrp.rs:118` — test-only, not reusable substrate.
  - No real simplex / LP solver exists anywhere in the 7-repo workspace.
- Decision: **build new** substrate (a bounded two-phase primal simplex + constraint-generation wrapper). No existing substrate to consume.
- Architectural rules checked: modelless (pure linear algebra, no training); generic over `<const N, const A>` (no game semantics); determinism-preserving (Bland's rule for anti-cycling — no RNG, no tie-breaking by pointer address).

---

## Design

### The CCE LP shape (always)

```text
min  c·x        where x = (ρ[0..NA], s[0..K]),  c = (gamma0_coeff[0..NA], 0, ..., 0)
s.t. row 0:    Σ ρ = 1                              (sum-to-1)
     row 1..K: g_κ · ρ + s_κ = 0  for each active κ  (regret constraints)
     x ≥ 0
```

`K` = number of active deviation constraints. For the homogeneous case `K = |D|`. For the heterogeneous case `K = Σ_i |D_i|`. For constraint generation, `K` grows from 0 to the active set size (typically << `Σ_i |D_i|`).

### Two-layer solver

**Layer 1 — `simplex.rs` (new):** A bounded two-phase primal simplex that solves any LP in the CCE shape above. Replaces `enumerate_bfs` as the inner solver when `C(n_vars, n_cons)` is too large for enumeration.

- Phase I: minimize the sum of artificial variables to find an initial feasible basis. If the Phase I objective > 0, the LP is infeasible.
- Phase II: minimize the original objective `c·x` starting from the Phase I feasible basis.
- Anti-cycling: **Bland's rule** (choose the smallest-index eligible entering variable) — guarantees termination, deterministic, no RNG. Worst-case exponential but never observed in practice on CCE-sized LPs.
- Pivot tolerance + degeneracy handling: eps = 1e-9 for reduced-cost sign; rows with RHS within eps of zero are flagged degenerate but still pivot normally under Bland's rule.
- Reuses `solve_square_system`'s partial-pivoting Gaussian elimination as the basis-solve kernel (DRY with the BFS path).

**Layer 2 — constraint generation wrapper (in `lp.rs`):** `solve_heterogeneous_cg` + `solve_with_constraint_generation`:

```text
active = {}  // no deviation constraints
loop:
    rho = simplex.solve(active)   // tiny LP: just sum-to-1 + active constraints
    for each player i:
        κ* = ExternalRegret::best_deviation_for_player(rho, i)
        if γ_i(rho) - γ_dev_i(rho, κ*) > ε:
            active.add((i, κ*))   // add the most-violated constraint
    if no constraint was added: return rho  // converged
```

This is the textbook cutting-plane / constraint-generation method for CCEs. Converges in O(|active set|) iterations, which is typically much smaller than `Σ_i |D_i|` (only the binding deviations end up active). Each inner LP is small (1 + |active| constraints), so even BFS would work for the inner solve — but we use the simplex for uniformity + to handle the case where the active set grows large.

### Auto-selection (BFS vs simplex)

`solve` / `solve_heterogeneous` keep their current signatures. Internally:
- Compute `n_candidates = C(n_vars, n_cons)`.
- If `n_candidates <= BFS_CUTOFF` (e.g. 50_000): use `enumerate_bfs` (proven exact, fast for small LPs).
- Else: use the simplex (either directly for the full LP, or via constraint generation for the heterogeneous case).

This preserves bit-identical behavior on all existing tests (which are all small enough for BFS) while unblocking the large-LP case.

### Why not just simplex-without-constraint-generation?

The full heterogeneous LP at NA=81 with 6 deviations has `n_vars = 87, n_cons = 7`. A simplex handles this fine. But for a hypothetical 4-player game with N=16, A=4, and 12 deviations per player: `n_vars = 64 + 48 = 112, n_cons = 1 + 48 = 49`. Still fine for simplex. Constraint generation is a **perf optimization** (smaller inner LPs) + a **robustness hedge** (the active set is often tiny — 2-3 constraints even when `Σ|D_i|` is large). Both layers ship; auto-selection picks the right one.

### Determinism contract

The simplex is deterministic: Bland's rule chooses the smallest-index entering variable, and the leaving variable is chosen by min-ratio with smallest-index tiebreak. No RNG, no hash-order dependence, no parallelism in the pivot loop. Two runs on the same LP produce bit-identical `ρ`. This matches the existing BFS determinism contract.

---

## Tasks

### T1 — `simplex.rs` substrate (the bounded two-phase primal simplex)
- [x] Implement `pub(crate) fn solve_simplex(mat, rhs, obj, n_vars, na) -> Option<Vec<f64>>` with the same signature as `enumerate_bfs` (drop-in replacement).
- [x] Phase I (artificial variables) + Phase II (original objective).
- [x] Bland's rule for entering variable (smallest-index with negative reduced cost).
- [x] Min-ratio test with smallest-index tiebreak for leaving variable.
- [x] Reuse `solve_square_system` as the basis-solve kernel OR inline a revised-simplex-style basis update (decide based on perf in T3).
- [x] Unit tests: (a) identity basis, (b) singular submatrix handling, (c) infeasible LP (Phase I obj > 0), (d) chicken game matches BFS, (e) emission game matches BFS, (f) RPS-sized random LP feasible.

### T2 — Auto-selection in `solve` / `solve_heterogeneous`
- [x] Add `BFS_CUTOFF` constant (50_000 candidates).
- [x] In `solve`: compute `C(n_vars, n_cons)`; if ≤ cutoff use `enumerate_bfs`, else use `solve_simplex`.
- [x] In `solve_heterogeneous`: same auto-selection. (For very large heterogeneous LPs, T4's constraint generation is the production path; this is the fallback.)
- [x] Verify all existing tests still pass bit-identically (the auto-select must pick BFS for every existing test — they're all small).

### T3 — Constraint generation wrapper
- [x] `solve_with_constraint_generation(d, p)` for the homogeneous case.
- [x] `solve_heterogeneous_cg(game)` for the heterogeneous case (uses per-player `best_deviation` separation oracle).
- [x] Convergence: loop until no violated constraint is found (ε = 1e-6).
- [x] Iteration cap: `Σ_i |D_i| + 1` (worst case: every deviation becomes active). Error if exceeded (would indicate a bug).
- [x] Unit tests: (a) emission game matches direct solve, (b) PD matches direct solve (homogeneous 2-player).

### T4 — GOAT gate: 2-player RPS at NA=81 (the production case from Issue 575)
- [x] Construct the N=9, A=9 joint-action RPS game (reuse the Issue 575 PoC setup).
- [x] `solve_heterogeneous_cg(game)` returns a valid occupation measure in < 60 s (actual: ~2 ms).
- [x] The result passes `is_heterogeneous_cce(ε = 1e-4)`.
- [x] The result's `γ₀` is ≥ the Nash equilibrium's `γ₀` (γ₀ = 0; result P1 γ₀ ≥ 0).
- [x] Compare against the Issue 575 artifact: the solver does NOT return the artifact `ρ((P,R),(P,R))=1`.
- [x] Benchmark file: `.benchmarks/573_cce_simplex_goat.md`.

### T5 — Perf characterization (informational, not a gate)
- [x] Bench: solve_heterogeneous_cg wall-clock on NA=81 (~2 ms debug mode).
- [x] Bench: number of constraint-generation iterations to converge (4).
- [x] Bench: inner-LP size at convergence (1 + 4 = 5 constraints).
- [x] Document in the benchmark file. No promotion gate on perf — correctness at scale is the gate (T4). Perf is informational for riir-ai Plan 325 routing decisions.

### T6 — Doc sync
- [x] `.docs/04_calibration/cce_moderator.md` §Limitations #3: updated from "needs a real simplex" to "simplex shipped (Plan 572); auto-selected for large LPs; constraint generation for heterogeneous".
- [x] `.docs/04_calibration/cce_moderator.md` §Limitations #1: updated from "BFS solver cannot SOLVE at NA=81" to "solver ships; 2-player CCE path is now solve+verify".
- [x] `CceLp` API table: documented `solve_heterogeneous_cg` + `solve_heterogeneous_cg_with_tolerance`.
- [x] Add Plan 572 reference.

---

## Non-goals

- **General LP solver library.** This simplex handles the CCE LP shape only (equality constraints + non-negativity). Not a general-purpose LP solver. No external deps (good_lp, minilp, etc.) — pure Rust, modelless, zero-dep.
- **Interior-point method.** Simplex is exact (returns a vertex BFS) and matches the existing BFS-output contract. Interior-point returns an interior point that would need crossover to match. Not worth the complexity for CCE-sized LPs.
- **Revised simplex with explicit basis inverse.** The naive simplex recomputes the basis solve each iteration via `solve_square_system`. For CCE-sized LPs (n_cons ≤ ~50) this is fast enough. A revised simplex with incremental basis updates is a perf optimization for very large LPs — defer until a concrete consumer needs it.
- **Parallel simplex.** The pivot loop is inherently sequential. Parallelism would break the determinism contract.
- **General-sum n-player CCE.** The constraint-generation wrapper is generic over `H: HeterogeneousPayoff<N, A>`, so it handles any number of players. But the GOAT gate (T4) only tests 2-player zero-sum (RPS). General-sum n-player correctness is a consumer concern (riir-ai Plan 325).

---

## GOAT gate (T4)

| Gate | Target | Method |
|---|---|---|
| G1 — correctness at scale | `solve_heterogeneous_cg` returns a valid CCE on NA=81 | `is_heterogeneous_cce(ε=1e-4) == true` |
| G2 — non-regression | all existing CCE tests pass bit-identically | auto-select picks BFS for small LPs |
| G3 — artifact rejection | solver does NOT return the RPS trivial artifact | γ₀(result) ≠ γ₀(artifact); result passes 2-player CCE check |
| G4 — determinism | two runs produce bit-identical ρ | Bland's rule, no RNG |
| G5 — modelless | pure linear algebra, no training, no deps | audit (no new Cargo.toml deps) |

No perf gate (T5 is informational). The value prop is **capability** (solve at scale) not **speed** (BFS is already fast for small LPs).

---

## Promotion

`cce_moderator` is already default-on. This plan extends it with a new module (`simplex.rs`) + new methods (`solve_with_constraint_generation`, `solve_heterogeneous_cg`). No new feature gate — the simplex ships as part of the existing `cce_moderator` surface. The auto-selection in `solve` / `solve_heterogeneous` is transparent (same signature, same output contract, just a different inner solver for large LPs).

If T4 PASSes, update `.docs/04_calibration/cce_moderator.md` to reflect the new capability. No Cargo.toml promotion needed.

---

## References

- Issue 575 (RPS artifact closure PoC) — the immediate predecessor; proved verify-only at NA=81.
- Plan 569 (TransitionKernelCce) — the prior CCE substrate extension; same shape (new solver variant).
- `.docs/04_calibration/cce_moderator.md` §Limitations — the doc to update.
- Bertsimas & Tsitsiklis, *Introduction to Linear Optimization* — the two-phase simplex reference.
- Bland, R.G. (1977), "New finite pivoting rules for the simplex method," *Mathematics of Operations Research* 2(2) — the anti-cycling rule.
