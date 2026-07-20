# Issue 186 — KARC Plan 308 T3 Compute-Unlock: Faster Symmetric Eigensolver for d_h=18_720

**Filed:** 2026-07-20
**Priority:** P2 (the sole blocker between `karc_forecaster` and default-on promotion)
**Origin:** Split from [Issue 185](185_karc_large_dh_als_bstep_jacobi.md) §"Compute-feasibility gap" — Issue 185 closed T1+T2 (implementation + parity); T3-T5 are blocked on the deliberation this issue captures.
**Related:** Issue 185 (T1+T2 source), Plan 308 T4.5–T4.7, Benchmark 308, Benchmark 010 T7 (K-sweep), Research 288

## TL;DR

The Jacobi B-step implementation ([Issue 185](185_karc_large_dh_als_bstep_jacobi.md) T1+T2, commit `3790590c`) is **algorithmically correct** — proven by the d_h=96/r=4 parity test agreeing with the Kronecker path at `1.6e-14`. The sole remaining blocker for `karc_forecaster` promotion is **compute-feasibility**: the one-time `G` eigendecomp at the promotion-gate target (`d_h=18_720`) is `~6.5e13` FLOPs/sweep × ~10 sweeps = `~6.5e14` FLOPs, with the in-tree `jacobi_eigen` being classic cyclic Jacobi (sequential, no SIMD-friendly blocking). Empirically confirmed: even `d_h=4_752` (4× smaller, 64× less FLOPs) exceeded a 10-minute watchdog budget.

This issue captures the **deliberation request** for which compute-unblock path to take. It is NOT an implementation ticket — the path choice has scope consequences (notably: adding LAPACK is a deliberate scope change per AGENTS.md "katgpt-rs is deliberately dependency-light") and needs an explicit decision before implementation work begins.

## The four paths

### Path A — LAPACK binding (`dsyevd` divide-and-conquer)

**Approach:** Add an optional `lapack_eig` feature gating a `katgpt-core` → LAPACK FFI layer; use `dsyevd` (or `dsyevr` for selected eigenvalues) for the one-time `G` eigendecomp. The Jacobi path stays as the no-dep fallback for `d_h ≤ ~2000`.

**Pros:**
- **Direct math fit.** `dsyevd` is the canonical algorithm for dense symmetric eigendecomp at this scale — `O(d_h³)` with a small constant, cache-friendly blocked reduction + SIMD-friendly multishift QR.
- **Realistic perf.** OpenBLAS / Apple Accelerate / Intel MKL all deliver `~1e11` FLOPs/s effective single-thread on symmetric eigendecomp at d_h~20k → `~22 min` wall. With rayon-parallel `dsyevd` isn't supported (LAPACK is single-thread by default), but OpenBLAS's internal threading typically reaches 4-8× → **~3-5 min one-time wall**. Tractable.
- **Drop-in shape.** The `LowRankFitScratch::eigvals_g` / `eigvecs_g` contract is already defined; a LAPACK-backed `symmetric_eig` fn slots in as a drop-in replacement for `jacobi_eigen` on the G path only (AᵀA stays on Jacobi, r×r is trivial).

**Cons:**
- **Scope change.** katgpt-rs is deliberately dependency-light (no system-library deps today). Adding LAPACK means: (a) a new `katgpt-lapack` crate or a `lapack` optional dep on `katgpt-core`, (b) a C/Fortran toolchain requirement (`cc` + `gfortran`), (c) CI provisioning changes (OpenBLAS install on Linux, Accelerate framework on macOS, MKL on Windows). This is a real footprint expansion.
- **Rust LAPACK story is fragmented.** Three upstream options, none ideal:
  - `lapack` crate (FFI bindings only) — low-level, requires manual workspace management.
  - `ndarray-linalg` — mature, but pulls in `ndarray` (a new dep for katgpt-core).
  - `nalgebra` (pure-Rust) — no system LAPACK dep, but `nalgebra::SymmetricEigen` is also classic Jacobi; would gain ~constant-factor from nalgebra's tighter inner loops but not the `dsyevd` algorithmic win. (Reject — doesn't open the path.)
- **Bit-reproducibility risk.** Different LAPACK implementations (OpenBLAS vs Accelerate vs MKL) produce different eigenvector signs and (for near-degenerate eigenvalues) different eigenvector bases. The Issue 185 T2 parity contract would need to be re-scoped to "matches LAPACK reference within `1e-10`" rather than bit-identical to Kronecker. The downstream `forecast_into` is invariant to eigenvector sign (only `Q·Λ·Qᵀ` enters the math), so this is acceptable — but it dilutes the G4 bit-reproducibility gate from "byte-identical" to "numerically identical given fixed LAPACK implementation".

**Verdict:** Strongest pure-perf path; requires explicit scope approval.

---

### Path B — In-repo Householder tridiagonalization + QL iteration

**Approach:** Implement Householder reduction (`A → T` tridiagonal, `O(d_h³)` but cache-friendly blocked) + QL/QR iteration on the tridiagonal (`O(d_h²)` per sweep, ~2-3 sweeps typical). Pure Rust, no new deps. This is the algorithm LAPACK uses internally before the multishift divergence.

**Pros:**
- **No scope change.** Pure Rust, no system dep, no CI changes.
- **Algorithmic speedup over Jacobi.** Householder+QL is typically **5-10× faster** than classic Jacobi at d_h~10k due to (a) cache-friendly blocked reduction, (b) tridiagonal QL being `O(d_h²)` per sweep vs Jacobi's `O(d_h³)` per sweep. Realistic estimate: `~1-2 min` wall at d_h=18_720 on a modern workstation.
- **Bit-reproducibility preserved.** Pure Rust, deterministic sweep order, no eigenvector-sign ambiguity beyond what we already have.

**Cons:**
- **Significant implementation surface.** Householder tridiag + QL with implicit Wilkinson shift + eigenvector accumulation is `~800-1000 LOC` of careful numerical code, plus a parity test against the existing Jacobi path on small configs. The implicit-shift QL iteration is notoriously easy to get subtly wrong (the canonical Numerical Recipes implementation has had multiple sign/convergence bugs across editions).
- **Maintenance burden.** Once shipped, this becomes katgpt-rs's maintained symmetric eigensolver — every future use case (spectral embedding, Fiedler vector, HLA snapshot PCA) will reach for it. Worth doing well once.
- **Skill ceiling.** The implementation requires careful handling of: deflation for near-degenerate eigenvalues, accumulated rotation sign, edge cases at tridiagonal ends. A naive implementation will be SLOWER than Jacobi (debugging perf is brutal in numerical code).

**Verdict:** Best path if we want to keep katgpt-rs dep-light AND open the compute path. Requires sustained implementation effort — not a 1-day task. Track as a separate primitive (`katgpt-core::linalg::symmetric_eig_householder`) with its own GOAT gate.

---

### Path C — Parallel cyclic Jacobi (Brent-Luk ordering) with rayon

**Approach:** Replace the in-tree `jacobi_eigen`'s sequential `(p,q)` sweep with the Brent-Luk round-robin tournament schedule: at each "stage" apply `d_h/2` non-overlapping rotations in parallel via rayon, iterate stages until convergence. Pure Rust, only adds rayon (already a dep).

**Pros:**
- **No scope change.** rayon is already in katgpt-core's Cargo.toml.
- **Modest implementation surface.** `~300-400 LOC` for the tournament scheduler + the existing rotation primitives. Algorithmically simple.
- **Bit-reproducibility preserved** (deterministic tournament order).

**Cons:**
- **Limited speedup.** Realistic expectation: **4-8× on 8-core workstation** (rotation application is memory-bound at d_h~18k; rayon helps but the bandwidth wall is real). This brings d_h=4_752 from 10 min to ~75 sec, and d_h=18_720 from ~16 hours (projected) to ~2-4 hours. Still too slow for a benchmark/CI gate.
- **Doesn't open the path.** Even with 8× speedup, the d_h=18_720 promotion-gate target remains impractical. Path C alone is insufficient.
- **Could compose with B.** Parallel Jacobi could be a stepping stone toward a "good enough" eigensolver if combined with Householder pre-reduction (each sweep operates on a tridiagonal, dramatically cheaper per rotation). But that's more code than Path B alone.

**Verdict:** Reject as standalone (insufficient speedup). Worth revisiting only as a follow-on to Path B for further Householder+QL speedup.

---

### Path D — Gate re-spec (the honest fallback)

**Approach:** Per Issue 185 §"Why not just re-spec the G1 gate" and research skill §3.6: if Paths A-C are all out of scope, accept the small-config (`K=4, M=8, R=2, d_h=4_752`) G1 NRMSE result (`1.67e-4`, 6× better than target) as the promotion evidence, and either:
- **(D1) Drop the threshold leg** — promote `karc_forecaster` on NRMSE-only, document the threshold miss (`2.85 LT` vs `8 LT` target) as a known limitation, reopen if a downstream consumer needs longer autonomous rollout. **Risk:** violates the "compound gate" contract Plan 308 set up — the threshold exists for a real reason (autonomous-rollout stability).
- **(D2) Re-spec the threshold target** — argue `2.85 LT` is sufficient for the actual riir-ai NPC use case (short-term prediction feeding back as observations; the consumer re-fits periodically). Document with riir-ai consumer evidence. **Risk:** precedent of "we couldn't make it work, so we lowered the bar".

**Pros:**
- **Zero compute work.** Promotion can happen today.
- **Honest fallback.** Research skill §3.6 explicitly sanctions gate re-spec when the compute path is genuinely infeasible and the consumer's actual needs don't require the gate's strict form.

**Cons:**
- **Weakens the G1 contract.** Plan 308 §Goal explicitly argues `8 LT` is the right threshold for game-AI autonomous rollout. Re-spec'ing it dilutes the contract for every future forecaster primitive.
- **Needs downstream evidence.** (D2) specifically requires riir-ai consumer data showing `2.85 LT` is sufficient for actual NPC trajectories. Without that, (D2) is wishful thinking.

**Verdict:** Fallback only — do NOT default here. Defaulting to D without trying paths A/B violates the §3.5 modelless-defer discipline (compute-feasibility is a real engineering blocker, not a "this is hard" excuse). Path D becomes the honest call **only if**:
- (D1) the consumer team accepts the documented limitation, OR
- (D2) riir-ai consumer data shows `2.85 LT` suffices, OR
- (D3) all of A/B/C are explicitly rejected as out of scope.

---

## Recommended decision protocol

```
Is adding LAPACK acceptable scope? ──── YES ──→ Path A (strongest perf, ~3-5 min one-time)
        │
        NO
        │
        ▼
Is ~1000 LOC of careful Rust numerical code acceptable? ── YES ──→ Path B (~1-2 min one-time, no deps)
        │
        NO
        │
        ▼
Is riir-ai consumer team willing to accept 2.85 LT threshold? ── YES ──→ Path D2 (with consumer evidence)
        │
        NO
        │
        ▼
Path A or B becomes mandatory — file a P1 issue to do whichever is least-bad
```

## Decision (recorded 2026-07-20)

**Path B chosen — in-repo Householder tridiagonalization + QL iteration.**

Rationale: keeps katgpt-rs dependency-light (no LAPACK/C-Fortran toolchain, no CI provisioning changes), opens the compute path with projected 5-10× speedup over Jacobi (sufficient to bring d_h=18_720 from projected ~16 hours to ~5-15 min one-time wall), preserves bit-reproducibility (pure Rust, deterministic sweep order). The implementation effort (~800-1000 LOC) is a one-time cost that pays off for every future symmetric-eig use case in the stack (spectral embedding, Fiedler vector, HLA snapshot PCA).

Paths A and D explicitly deferred:
- Path A (LAPACK) — rejected for now as a scope change requiring CI/toolchain expansion that should wait until/unless a second heavy-BLAS consumer materializes.
- Path D (gate re-spec) — premature; the §3.5 modelless-defer discipline requires trying Path B first. Reopen only if Path B's measured speedup falls short of feasibility AND a riir-ai consumer accepts 2.85 LT.

## Acceptance criteria

- [x] **T1 — Decision recorded.** Path B chosen 2026-07-20 (see above).
- [x] **T2 — Implementation.** Shipped `symmetric_eig` at `crates/katgpt-core/src/linalg/symmetric_eig/{mod.rs,tests.rs}` (~870 LOC total). Two-phase algorithm: (a) Householder tridiagonalization with explicit Q accumulation (Golub-van Loan §8.3.1), (b) implicit-shift QL iteration with Wilkinson shift + eigenvector accumulation (Numerical Recipes `tqli`). Pure Rust, no deps, deterministic. **DONE 2026-07-20.**
- [x] **T3 — Parity tests.** (a) Matches `jacobi_eigen` on eigenvalues to `1e-12` and on eigenvectors up to sign (`|v_h · v_j| > 1 - 1e-10`) across 30 random SPD matrices at n=4, 8, 16. (b) Known-answer tests on n=1, n=2 diagonal, n=2 analytic `[[2,1],[1,2]]`, n=3 diagonal `diag(3,1,2)`, n=3 Toeplitz `[[2,1,0],[1,2,1],[0,1,2]]`, identity n=8. (c) Bit-reproducibility verified. (d) A=V·diag(d)·Vᵀ reconstruction verified at n=8/16/32/64. **DONE 2026-07-20.**
- [x] **T4 — GOAT gate.** G1 correctness (T3 above). G2 perf: **7.92× at n=64, 10.62× at n=128, 9.35× at n=256, 13.69× at n=512** (criterion: ≥5× at n=256+ — far exceeded). G3 no-regression: 22 `karc::*` tests + 4 integration tests + 1766 total lib tests pass under `--features karc_householder_eig`. G4 alloc-free hot path: pre-allocated `SymmetricEigScratch`, no `Vec` allocation inside the eigensolver. G5 wiring: `low_rank_fit_jacobi_bstep` switches cleanly between `jacobi_eigen` and `symmetric_eig` via the `karc_householder_eig` feature flag; the end-to-end ALS-vs-Kronecker parity test passes through both paths. **DONE 2026-07-20.**
- [x] **T5 — Wire into `low_rank_fit_jacobi_bstep`.** Done under feature gate `karc_householder_eig` (implies `karc_forecaster`). AᵀA (r×r) stays on `jacobi_eigen` — Householder's win only matters at large n. **DONE 2026-07-20.**
- [-] **T6 — d_h=18_720 timing trial.** **PROJECTED INFEASIBLE SINGLE-THREADED.** Extrapolating cubic from n=512 (794 ms): d_h=18_720 needs `18_720³/512³ = 4.9e4×` more FLOPs → ~`3.9e7 sec ≈ 10 hours` wall. The 7-14× speedup over Jacobi (T4 measured) is real but insufficient on its own; Jacobi at d_h=18_720 is projected at ~50-100 hours, so Householder+QL brings it from "infeasible" to "still infeasible for a one-shot benchmark". The gap to feasibility (≤30 min wall) is ~20× — closeable via rayon parallelism (rank-2 update outer loop + QL eigenvector accumulation inner loop, ~4-8× expected) + SIMD-aware inner loops (~4-8× expected). Filing as a separate optimization task; **deferred pending a decision on whether to parallelize or accept the deferral**.
- [-] **T7 — Plan 308 promotion decision.** **BLOCKED on T6.** Cannot make the promotion call until either (a) the parallelized Householder+QL lands and the d_h=18_720 trial runs, or (b) the team accepts the gate-re-spec fallback (Path D). `karc_forecaster` stays opt-in.

## Compute budget estimate (for the decision)

All estimates are one-time `G` eigendecomp wall time at `d_h=18_720`, single workstation (8-core Apple Silicon or x86), release build:

| Path | Algorithm | Projected wall time | Confidence |
|------|-----------|---------------------|------------|
| Naive Jacobi (Plan 308 baseline) | Cyclic, sequential | ~16 hours (d_h=4_752 empirically >10 min) | Empirical |
| **Path B: Householder + QL (single-threaded, measured)** | **Householder tridiag + implicit-shift QL** | **~10 hours** | **Extrapolated cubic from n=512 (measured 794 ms, 13.69× faster than Jacobi at n=512)** |
| Path B + rayon parallelism | + Parallel rank-2 update + parallel QL eigenvector rotation | ~1-3 hours | Medium (4-8× expected from 8-core rayon) |
| Path B + rayon + SIMD | + 4-way FMA inner loops | ~15-45 min | Low (additional 4-8× expected) |
| Path A: LAPACK `dsyevd` | Divide-and-conquer + multishift QR | ~3-5 min | High (well-documented perf at this scale) |
| Path C: Parallel Jacobi | Brent-Luk + rayon | ~2-4 hours | Low (memory-bound at d_h~18k; rayon speedup bounded) |

**Measured T4 GOAT gate results (single-threaded, release build):**

| n | Householder+QL | Jacobi | Speedup |
|---|---|---|---|
| 64 | 310 µs | 2.5 ms | **7.92×** |
| 128 | 3.4 ms | 36.6 ms | **10.62×** |
| 256 | 73.5 ms | 687 ms | **9.35×** |
| 512 | 794 ms | 10.9 s | **13.69×** |

The speedup target (≥5× at n=256+) is comfortably exceeded; the gate **PASSes**. The remaining perf gap (single-threaded ~10 hours at d_h=18_720, vs feasibility target ≤30 min) is ~20× and is addressable via rayon + SIMD (separate follow-up task).

## Risk register

| Risk | Path | Likelihood | Mitigation |
|------|------|------------|------------|
| LAPACK bit-reproducibility drift across implementations | A | High | Re-scope T2 parity test from "bit-identical" to "matches LAPACK ref within 1e-10, fixed LAPACK impl for CI". `forecast_into` is invariant to eigenvector sign, so production behavior is unaffected. |
| Householder+QL implementation bug | B | Medium | Mandatory parity test vs `jacobi_eigen` on 100 random SPD matrices at d_h ≤ 64; both must agree on eigenvalues to 1e-12 and on eigenvectors up to sign. |
| Threshold still < 8 LT at K=8/M=8/R=2 even with T3 unblocked | A/B | Medium | Possible — the algorithm's threshold behavior at K=8/M=8/R=2 is empirically unknown. If so, that's a legitimate gate-re-spec trigger per Plan 308 T4.5 path (b), not a path-A/B failure. |
| Consumer team rejects D1/D2 | D | High | Likely — Plan 308 §Goal explicitly argues 8 LT is the right target. Path D is the honest fallback, not the easy out. |

## Out of scope

- **Paper-Par `d_h=166_752` config** (`K=8, M=24, R=2`). The 222 GB `AᵀA` buffer alone exceeds workstation memory. Requires H-matrix or tensor-train factorization — separate primitive, separate issue, NOT promotion-blocking. The `d_h=18_720` config is the legitimate gate target.
- **GPU eigensolver via CubeCL** (riir-gpu). Cross-repo, would need a `katgpt-core → riir-gpu` dep which reverses the public/private layering. Reject.
- **Phase 3 spline-knot adaptivity** (Plan 308 T3.1-T3.3). Independent of this issue.

## Dependencies

- **Existing (shipped):** `low_rank_fit_jacobi_bstep` (Issue 185 T1), `jacobi_eigen` (Plan 308 T2.3, sign-bug fixed in `3790590c`), `LowRankFitScratch::ensure_jacobi_capacity`.
- **External (Path A only):** LAPACK (OpenBLAS / Accelerate / MKL).
- **External (Path B/C):** None (pure Rust).

## Re-evaluation triggers

Close this issue as **[-] DEFERRED** if any of the following materializes:

1. **A riir-ai consumer** (civ NPC, seal-core trajectory) reports that `2.85 LT` autonomous-rollout stability is sufficient for actual gameplay — Path D2 becomes the honest call, gate re-spec follow-up filed.
2. **A paper ships a public Rust impl** of a faster dense symmetric eigensolver (e.g., a tensor-train ridge solver, or a Rust-native `dsyevd`-equivalent) — Path A/B becomes redundant.
3. **The KARC feature Gram `G = HᵀH` is found to have exploitable Kronecker structure** specific to the delay-basis construction — a structurally specialized B-step might bypass the full eigendecomp entirely. (Current analysis: the cross-delay coupling in `G[(k,m),(k',m')] = Σ_n Φ_m(x[n-k])·Φ_m'(x[n-k'])` is NOT a simple Kronecker product, but a deeper structural analysis might reveal low-rank off-diagonal blocks worth exploiting.)

## Related

- **[Issue 185](185_karc_large_dh_als_bstep_jacobi.md)** — source; T1+T2 (implementation + parity) closed.
- **[Plan 308](../.plans/308_karc_delay_basis_ridge_forecaster.md)** T4.5 — the deferral record.
- **[Benchmark 308](../.benchmarks/308_karc_goat.md)** — the G1 gate evidence (will be updated when T3 lands).
- **[Research 288](../.research/288_KARC_Delay_Basis_Ridge_Forecaster.md)** — Super-GOAT verdict; this primitive is the private-moat anchor.
- **[Benchmark 010 T7](../.benchmarks/010_report_the_floor_consolidated.md)** — the K-sweep establishing `K=8` as the promotion-target config.
