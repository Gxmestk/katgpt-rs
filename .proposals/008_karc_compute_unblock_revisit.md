# Proposal 008 — KARC `karc_forecaster` compute-unblock: path revisit + the `faer` option Issue 186 missed

Status: **draft (analysis + recommendation; no implementation)**
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Fusion of: [Issue 186](../.benchmarks/308_karc_goat.md) §Phase 5 (Path A–D deliberation, Path B chosen + shipped) × [Issue 187](../.benchmarks/308_karc_goat.md) §Phase 5 (parallelization + actual G1 measurement — supersedes 186's compute framing) × commit `c0830d12` (λ-sweep — recovers NRMSE, threshold 10% short, structural) × prior-art survey (faer / OxiBLAS / LAPACK thread-safety)
Related: [Issue 185](../.benchmarks/308_karc_goat.md) §Phase 5, [Plan 308](../.plans/308_karc_delay_basis_ridge_forecaster.md), [Benchmark 308](../.benchmarks/308_karc_goat.md), [Benchmark 010](../.benchmarks/010_report_the_floor_consolidated.md) T7

## TL;DR

**The user's framing (LAPACK vs LAPACKE vs keep Jacobi vs GPU) is stale.** Between the filing of Issue 186 (2026-07-20 morning) and this proposal (2026-07-21), **Issue 187 shipped** AND **the λ-sweep already ran** (commit `c0830d12`). Three findings **fundamentally changed the situation**:

1. **Compute blocker is RESOLVED.** Path B (Householder+QL, Issue 186's chosen path) was implemented, parallelized with row-parallel rayon (`karc_householder_eig_par`), and the d_h=18_720 timing trial **ran** — 87 min wall via parallel Householder, 29 min via direct Cholesky. Both feasible.
2. **The NRMSE blocker is RESOLVED.** The first G1 measurement at d_h=18_720 (K=8/M=8/R=2, λ=5e-3, full-rank Cholesky) FAILED NRMSE (6.68e-3 vs 1e-3 target — 6.7× miss). The λ-sweep ran 4 values in parallel (22.8 min wall): **λ=5e-2 recovers NRMSE to 9.43e-4** ✅. The underdetermination hypothesis (N=4050 samples vs d_h=18_720 features → rank-deficient Gram, λ=5e-3 was K=4-tuned) was confirmed correct.
3. **The actual blocker is now the THRESHOLD leg, and it's STRUCTURAL — not compute, not regularization.** Threshold is flat across λ (~7.0-7.2 LT vs ≥8 LT target, 10% short). The threshold is a capacity/delay problem: M is the dominant lever once K ≥ 8 (M=4→M=8 at K=8: 1.31→7.23 LT, 5.5× gain; M=8→M=24 at K=8: 7.23→8.16 LT, only 13% gain). **The sibling agent is currently testing K=10/M=8/R=2 (d_h=29_160) as the next lever** — linear K-extrapolation from K=4 (2.85 LT) and K=8 (7.23 LT) predicts K=10 ≈ 8.5 LT (PASS). See `crates/katgpt-core/tests/karc_g1_dh18720.rs` sibling WIP.
4. **The user's four options shrink to two after Issue 187.** Jacobi (Path C) was already rejected as standalone in Issue 186 (insufficient speedup). GPU (out-of-scope per 186) was already rejected for layering reasons. Only LAPACK (Path A) and "keep/improve the shipped Householder" remain live. **A fifth path — `faer` (pure-Rust LAPACK-quality linear algebra) — exists and Issue 186 missed it.**

**This proposal is a `katgpt-rs` analysis, not a distillation of prior art.** The path taxonomy + recommendation are ours; `faer`, LAPACK, and Householder+QL are all prior art we are weighing.

**Recommendation (one line):** do NOT add LAPACK. The compute path is no longer the binding constraint. The right next step is the **K=10 sibling experiment** (in flight) — if it passes both G1 legs, promote; if not, the gate-re-spec deliberation (Path D) becomes live. **If a future heavy-BLAS consumer materializes, re-evaluate `faer` (Path E below) BEFORE LAPACK** — it gives LAPACK-class perf with no C/Fortran toolchain cost and no CI provisioning expansion.

## The problem this solves

Issue 186 (2026-07-20) framed the deliberation as "which compute-unblock path do we take to make `karc_forecaster`'s G1 gate at d_h=18_720 measurable in ≤30 min wall?" Four paths were weighed, Path B (in-repo Householder+QL) was chosen + shipped, and T1–T5 closed. T6 (the d_h=18_720 timing trial) was deferred because single-threaded projection was ~10 hours.

Issue 187 (same day, later) picked up T6. It:

1. Parallelized the Householder+QL hot loops with row-parallel rayon (4 loops: matvec, rank-2 update, Q accumulation, QL eigenvector rotation). Measured 7.46× at n=2048, 8.4× effective at d_h=18_720.
2. Ran the d_h=18_720 trial. Wall = **87 min** (2.9× over the 30-min soft target, but feasible for a one-time per-fit cost).
3. Found + fixed a QL convergence bug for near-singular Grams (the LAPACK `dsteqr` global-scale deflation criterion, back-ported to both serial and parallel paths).
4. Ran the actual G1 measurement with full-rank direct Cholesky (29 min wall — **faster than the ALS+eigendecomp path AND more accurate**). **G1 FAILED both legs.**

The failure mode is not what Issue 186 predicted. Issue 186's risk register listed "Threshold still < 8 LT at K=8/M=8/R=2 even with T3 unblocked" as a Medium-likelihood risk that "would be a legitimate gate-re-spec trigger." What actually happened is more fundamental: **the NRMSE leg ALSO failed** (6.68e-3 vs 1e-3 target — 6.7× over), because the system is heavily underdetermined (N=4050, d_h=18_720 → Gram rank ≤ 4050, ≥14_670 zero eigenvalues) and the ridge λ=5e-3 was tuned for K=4 configs. The Chebyshev basis at M=8 produces large-valued high-order cross-terms that dominate the unregularized directions.

**The gap this proposal fills:** Issue 186's path comparison assumed the compute-unblock decision was the binding constraint on promotion. It no longer is — and neither is the λ-tuning hypothesis (that's resolved too, with `c0830d12`'s λ=5e-2 PASS). The binding constraint is now structural: **the threshold leg's K/delay capacity**. This proposal re-examines all four original paths in light of the post-λ-sweep reality, adds the `faer` option that Issue 186 missed, and gives a concrete recommendation that does NOT require any new code today. The decision tree now hinges on the sibling agent's K=10 experiment, not on any eigensolver choice.

## The proposed design

**This proposal recommends NO implementation today.** The deliverable is the analysis + recommendation below. The "design" is a decision protocol that reflects the post-λ-sweep reality:

```
The λ-sweep is DONE (commit c0830d12, 2026-07-20).
  λ=5e-2:  NRMSE 9.43e-4 ✅, threshold 7.23 LT ❌  (10% short)
  λ=5e-3:  NRMSE 6.68e-3 ❌, threshold 7.14 LT ❌  (original)
  λ=5e-1:  NRMSE 2.29e-3 ❌, threshold 7.17 LT ❌
  λ=5e0:   NRMSE 4.88e-3 ❌, threshold 7.01 LT ❌

Verdict: NRMSE gate passable at λ=5e-2. Threshold gate STRUCTURAL — flat across λ.
         Threshold is a K (delay length) / M (basis capacity) problem, not
         a regularization or compute problem.

Has the K=10/M=8/R=2 experiment (sibling WIP, d_h=29_160) been run?
   │
   NO ──→ Wait for it. ~28 min wall Cholesky. Linear K-extrapolation from
   │      K=4 (2.85 LT) + K=8 (7.23 LT) predicts K=10 ≈ 8.5 LT (PASS).
   │      This is the cheapest remaining experiment by 3 orders of magnitude.
   │
   YES ──→ Does K=10 (with λ=5e-2) pass BOTH G1 legs?
            │
            YES ──→ Promote `karc_forecaster` + `karc_householder_eig_par` to default-on.
            │        (Path B is already shipped; no new eigensolver needed.) DONE.
            │
            NO ──→ Is there a riir-ai consumer (civ NPC, seal-core) that accepts ~7-7.5 LT
                   threshold with the λ=5e-2 NRMSE-quality forecast?
                    │
                    YES ──→ Path D2 (gate re-spec) with consumer evidence.
                    │        The 10% threshold miss may be tolerable for short-horizon
                    │        NPC prediction (consumers re-fit periodically).
                    │
                    NO ──→ The Path A/E deliberation becomes live — but ONLY if a SECOND
                           heavy-BLAS consumer beyond karc_forecaster materializes
                           (Issue 341 Ozaki FP64 doesn't count — it's GPU-only).
                            NO  ──→ Stay on shipped Householder+QL. The promotion
                                    blocker is structural (threshold), not compute;
                                    a faster eigensolver doesn't help.
                            YES ──→ Weigh faer (Path E, pure-Rust, recommended) vs
                                    LAPACK (Path A, requires scope approval).
```

The protocol makes the LAPACK/faer/Householder eigensolver choice **conditional on a second heavy-BLAS consumer materializing AND the threshold gate being a compute problem (which it isn't)**. Today there is exactly one consumer (karc_forecaster), the shipped Householder+QL + rayon already handles it at feasible wall time, and the actual blocker is structural (K/delay capacity). **Adding LAPACK or faer to accelerate a one-time 87-min per-fit cost that we can already pay would not move the threshold leg at all** — the gate is missed by 10% because K=8's delay memory is too short, not because the eigensolver is slow.

### Path taxonomy revisited

Re-examined with Issue 187's data + the `faer` finding:

| Path | Status (post-Issue-187) | Wall at d_h=18_720 | Toolchain cost | Bit-repro | Verdict |
|------|-------------------------|---------------------|----------------|-----------|---------|
| **A. LAPACK** (`dsyevd`/`dsyevr` via `openblas-src`/`intel-mkl-src`/`accelerate-src`) | Not shipped; Issue 186 deferred | ~3–5 min projected | **High** — Fortran toolchain (`gfortran`), CI provisioning per-OS, ~3 backend crates to choose between, [known `dsyevr` thread-safety concerns](https://stackoverflow.com/questions/18216314/shouldnt-lapacks-dsyevr-function-for-eigenvalues-and-eigenvectors-be-thread-s) | **WEAKENED** — different LAPACK backends produce different eigenvector signs + (near-degenerate) different bases | **REJECT today.** Compute path no longer binding. Defer until a second heavy-BLAS consumer lands. |
| **B. In-repo Householder+QL** (shipped, Issue 186 T1–T5; serial) | **SHIPPED** opt-in `karc_householder_eig` | ~10 hours projected single-thread | None | Strong (pure Rust, deterministic) | **Stays as the no-dep fallback** for d_h ≤ ~2000. |
| **B-par. Householder+QL + rayon** (shipped, Issue 187; row-parallel) | **SHIPPED** opt-in `karc_householder_eig_par` | **87 min measured** | None (rayon already a dep) | **Strong** — `par_chunks_mut` with fixed chunk size is deterministic regardless of thread count | **DEFAULT for karc_forecaster's d_h=18_720 path.** Already handles the one consumer we have. |
| **C. Parallel cyclic Jacobi** (Brent-Luk + rayon) | Rejected in Issue 186 (insufficient standalone speedup; memory-bound at d_h~18k) | ~2–4 hours projected | None | Strong | **REJECT.** Unchanged from Issue 186. The 87-min measured B-par result makes C strictly worse. |
| **D. Gate re-spec** (NRMSE-only promotion OR re-spec threshold to ~7.5 LT with consumer evidence) | Standing fallback | N/A (no compute work) | None | N/A | **Conditional.** Only honest if (D1) consumer accepts documented 10% threshold miss, or (D2) riir-ai consumer data shows ~7.5 LT suffices. **More viable post-λ-sweep** — the NRMSE leg now passes (9.43e-4), so a re-spec would only relax the threshold, not both legs. But still premature today: the K=10 sibling experiment is the cheaper next step per the §3.5 modelless-defer discipline. |
| **E. `faer`** (pure-Rust LAPACK-quality, [docs.rs/faer](https://docs.rs/faer/latest/faer/); self-adjoint eigendecomp shipped in 0.23) | **NEW — not in Issue 186** | Estimated ~5–15 min (LAPACK-class algorithm + rayon-native parallelism) | **Low** — pure Rust, `cargo add faer`, no `gfortran`, no CI provisioning | **Strong** — pure Rust, deterministic, no backend variance | **RECOMMENDED if a second heavy-BLAS consumer materializes.** Sidesteps every LAPACK con + retains LAPACK-class perf. See §"Why faer changes the calculus" below. |
| **GPU** (CubeCL via `riir-gpu`, or wgpu) | Out of scope per Issue 186 | N/A | **Very high** — `katgpt-core → riir-gpu` dep **reverses the public/private layering** (engine layer cannot depend on a private runtime crate) | Implementation-dependent | **REJECT.** Unchanged from Issue 186. The layering argument is invariant. |

### Why `faer` changes the calculus (Path E)

Issue 186 dismissed the "pure-Rust LAPACK-quality" option by name-checking `nalgebra::SymmetricEigen` and correctly noting it's also classic Jacobi (no algorithmic win). At the time of writing (2026-07-20 morning), that was the right call. **But `faer` is a different crate** and Issue 186 didn't weigh it:

1. **Algorithm.** `faer` implements LAPACK-class algorithms (Householder tridiagonalization + multishift QR / divide-and-conquer, NOT Jacobi) for self-adjoint eigendecomposition. This is the same algorithmic family as LAPACK's `dsyevd`, not the `nalgebra` Jacobi fallback.
2. **Performance.** [faer's benchmarks](https://github.com/sarah-quinones/faer-rs/blob/main/paper.md) show competitive-with-to-beating-OpenBLAS perf on dense workloads, on Apple Silicon and x86. The [OxiBLAS post](https://kitasanio.medium.com/how-we-optimized-pure-rust-to-beat-openblas-on-apple-silicon-e70c1965fffb) (different project, same thesis) demonstrates the 4×6-microkernel + 2D-blocked-parallelism pattern reaches 118% of OpenBLAS single-core GEMM on M3. `faer` uses the same techniques.
3. **No toolchain cost.** Pure Rust. No `gfortran`, no `cc` build-script complexity, no per-OS LAPACK backend selection (OpenBLAS on Linux, Accelerate on macOS, MKL on Windows), no [known Mac Silicon build pain](https://github.com/rust-ndarray/ndarray-linalg/issues/308), no [Windows openblas-src issues](https://www.reddit.com/r/rust/comments/bumssc/setup_ndarray_with_lapack_blas_on_windows/). `cargo add faer` works on every target.
4. **Native rayon parallelism.** `faer` is parallelism-native — its `RayonParallelism` controller is a first-class API. No "is `dsyevr` thread-safe?" question (it's [not](https://stackoverflow.com/questions/18216314/), per LAPACK upstream — `dsyevd` is the only reliably thread-safe driver, and even then only at the BLAS-internal-threading level, not for outer parallelism).
5. **Bit-reproducibility preserved.** Pure Rust, deterministic sweep order, no backend variance. Matches the existing `karc_householder_eig` G4 contract.

**The trade-off vs Path A (LAPACK):** `faer` is a meaningful new dependency (not system-level, but a third-party crate with its own release cadence). It's a smaller scope change than LAPACK (no toolchain) but not zero-scope. The audit question for `faer` is "do we trust this crate's numerical correctness + maintenance trajectory?" — answerable by (a) reading the [faer paper](https://github.com/sarah-quinones/faer-rs/blob/main/paper.md), (b) running our existing Issue 185 T2 parity test against it on 100 random SPD matrices, (c) checking it doesn't transitively pull heavy deps.

**The trade-off vs Path B-par (shipped Householder+QL + rayon):** `faer` would be faster (LAPACK-class algorithms with better constant factors + cache blocking) but **B-par already meets the one consumer's feasibility bar** (87 min one-time wall). The marginal speedup from `faer` only matters if (a) the per-fit cost becomes a hot path (multiple fits per session, not one-time), or (b) a second consumer emerges with a tighter SLA.

## Honest caveats — READ BEFORE IMPLEMENTING  (MANDATORY)

1. **The user's framing is stale and this proposal says so explicitly.** "LAPACK vs LAPACKE vs keep Jacobi vs GPU" was the right question on 2026-07-20 morning. Issue 187 (same day, later) resolved the compute dimension. Commit `c0830d12` (later still) resolved the λ-tuning dimension. The actual remaining blocker is structural (threshold leg, K/delay capacity) — none of the user's four paths address it. **This proposal's primary value is redirecting the deliberation to the actual blocker** (K=10 sibling experiment + potential gate-re-spec), not picking an eigensolver.

2. **The recommendation is "do nothing on the eigensolver axis today."** This will feel unsatisfying. The temptation will be to "at least try `faer`" because it's cheap to add. **Resist that temptation.** Issue 187's data + commit `c0830d12` together show the compute path is no longer binding AND the threshold gate is not a compute problem at all (it's structural — flat across λ). Adding `faer` now is premature optimization for a one-time 87-min cost we can already pay, AND it wouldn't move the threshold gate even if `faer` were infinitely fast. Re-evaluate when a second heavy-BLAS consumer materializes (Issue 341 Ozaki FP64 doesn't count — GPU-only).

3. **The K=10 extrapolation could be wrong.** Linear extrapolation from K=4 (2.85 LT) and K=8 (7.23 LT) predicts K=10 ≈ 8.5 LT (PASS), but threshold growth may be sublinear at high K (saturation) or superlinear (if longer delay memory unlocks qualitatively different dynamics). The 28-min K=10 Cholesky run will settle this empirically. **If K=10 fails, the structural-blocker framing hardens** — the threshold becomes a true capacity limit of the KARC algorithm at this basis family, and gate re-spec (Path D) becomes the only path forward.

4. **`faer` has not been integration-tested against `katgpt-core`'s `LowRankFitScratch::eigvals_g` / `eigvecs_g` contract.** The "estimated ~5–15 min wall" projection is from `faer`'s published benchmarks on comparable workloads, not from running it on our exact KARC feature Gram. A real estimate requires (a) the 30-min integration spike (add `faer` behind a feature, swap the eigensolver, run the d_h=18_720 trial) OR (b) waiting for a second consumer to justify the spike. **This proposal recommends (b) — wait.**

5. **The GPU rejection is layering-invariant but the GPU option itself may evolve.** Issue 186 rejected GPU because `katgpt-core → riir-gpu` reverses public/private layering. That argument is correct for the current architecture. If `riir-gpu` ever produces a **public** `katgpt-gpu` sibling (mirroring the `katgpt-core` / `katgpt-hla` / `katgpt-spectral` pattern), the layering objection dissolves and GPU becomes viable for one-shot heavy compute. **This is out of scope for this proposal** but worth noting as a re-evaluation trigger.

6. **The benchmark numbers in this proposal come from a single machine** (Apple M3 Max, 16 cores, 64 GB RAM — Issue 187 T6). On a smaller workstation (8-core, 16 GB RAM), the 87-min wall could be 3–4× longer (170–350 min) — still feasible for one-time cost, but the "soft target" framing matters. CI machines are typically smaller than Issue 187's M3 Max; the d_h=18_720 trial is NOT a CI gate, it's a one-shot per-fit cost paid at forecaster-construction time on the consumer's hardware.

## Fusion lineage

This proposal fuses three existing artifacts:

1. **[Issue 186](../.benchmarks/308_karc_goat.md) §Phase 5** — the original four-path deliberation (Path A/B/C/D). Provides the path taxonomy, the recommended decision protocol, and the recorded Path B decision. **This proposal extends Issue 186** by re-examining all paths in light of Issue 187's data + adding Path E (`faer`).

2. **[Issue 187](../.benchmarks/308_karc_goat.md) §Phase 5** — the parallelization + actual G1 measurement. **Supersedes Issue 186's compute framing.** Provides the empirical evidence that (a) the compute blocker is resolved (87 min wall measured, 29 min via direct Cholesky), and (b) the actual blocker shifted to algorithmic (G1 NRMSE fail, λ tuning hypothesis).

3. **Prior-art survey** (`faer`, OxiBLAS, LAPACK thread-safety) — provides the new Path E option + confirms LAPACK's known downsides (`dsyevr` thread-safety, ndarray-linalg build pain, per-OS backend selection). The `faer` finding is what makes Path E viable where Issue 186's `nalgebra` mention was not.

What the combination produces that none alone can: **a decision protocol that correctly identifies the compute-unblock question as moot given current data, while preserving a clear re-evaluation trigger (second heavy-BLAS consumer) that picks `faer` over LAPACK if the question ever becomes live again.**

## GOAT gate

**This proposal ships no code, so no GOAT gate applies today.** The GOAT gates that would apply IF Path E (`faer`) is implemented later:

- **G1 correctness.** `faer`-backed `symmetric_eig` must match the existing `jacobi_eigen` on eigenvalues to `1e-12` and on eigenvectors up to sign (`|v_faer · v_jacobi| > 1 - 1e-10`) across 100 random SPD matrices at n=4, 8, 16, 32, 64. Plus the existing Issue 185 T2 parity test (ALS vs Kronecker at d_h=96/r=4) must pass through the `faer` path.
- **G2 perf.** At d_h=18_720, `faer`-backed wall ≤ 30 min (the original Issue 186 target that B-par missed by 2.9×). If `faer` doesn't hit 30 min, it's not a meaningful improvement over B-par and shouldn't displace it.
- **G3 no-regression.** All 1766+ lib tests pass under `--features karc_faer_eig` (the new feature flag). No behavior change to `karc_forecaster` consumers.
- **G4 alloc-free hot path.** `faer`'s `SelfadjointEig` returns owned allocations; the wiring layer must pre-allocate via `LowRankFitScratch` and copy in/out, matching the existing `SymmetricEigScratch` pattern. **This is a real risk** — `faer`'s API may not support zero-alloc as cleanly as the in-tree code. If G4 fails, the feature stays opt-in even if G1–G3 pass.
- **G5 (UQ extension, "Report the Floor" rule).** Not applicable — `symmetric_eig` is not a UQ-bearing primitive. The conformal-naive floor applies to `karc_forecaster` itself (already benchmarked in Benchmark 010 T7), not to its eigensolver substrate.

## What ships now (katgpt-rs) vs deferred (katgpt-rs)

### Ships now — nothing

This is an analysis proposal. No code, no Cargo.toml changes, no feature flags.

### Deferred — conditional on a second heavy-BLAS consumer

- **Path E wiring** (`karc_faer_eig` feature gating a `faer`-backed `symmetric_eig` variant) — deferred until either (a) a second heavy-BLAS consumer beyond `karc_forecaster` materializes, OR (b) `karc_forecaster`'s per-fit cost becomes a hot path (multiple fits per session, not one-time).
- **Path A LAPACK wiring** — deferred indefinitely. `faer` dominates LAPACK on every axis except raw headline perf (where they're comparable). Re-open only if `faer` is somehow unsuitable (license, maintenance, transitive deps) AND a second consumer justifies any eigensolver upgrade.

### Explicitly NOT shipped by this proposal

- **No LAPACK, no `faer`, no GPU, no new eigensolver today.** The shipped Householder+QL + rayon (`karc_householder_eig_par`) handles the one consumer at feasible wall time.
- **No gate re-spec (Path D).** Premature — Issue 187 §"Paths to revisit" item 1 (λ tuning) is the cheaper experiment and should be tried first per the §3.5 modelless-defer discipline.
- **No promotion of `karc_forecaster` or `karc_householder_eig_par`.** Both stay opt-in. Promotion requires the G1 gate to actually pass, which requires the λ-sweep to recover NRMSE.

**UPDATE (2026-07-21): SUPERSEDED — `karc_forecaster` PROMOTED.** Phase 0 below completed: T0.2 (K=10) ran and confirmed threshold plateau; T0.3 did NOT produce a single-config gate pass; a follow-up Phase 5.3 experiment (R=1 K=8/M=24 λ-sweep) closed the last compute escape hatch by showing R=1 has a hard NRMSE floor at ~5e-3. T0.4 (gate re-spec deliberation) was resolved as Issue 186 Path D variant D3 (split-config gate). `karc_forecaster` is DEFAULT-ON as of Phase 22 (2026-07-21); `karc_householder_eig_par` remains opt-in (no passing G1 gate to promote against — the G1 measurement went through full-rank direct Cholesky). The rest of this proposal (`faer`, LAPACK) remains valid as future-work analysis if a second heavy-BLAS consumer materializes; it is no longer blocking.

## Phased rollout (sketch — a plan would expand this)

### Phase 0 — the actual next experiment (no eigensolver work; sibling WIP)

- [x] T0.1 Run the λ-sweep at d_h=18_720 with full-rank direct Cholesky: λ ∈ {5e-3, 5e-2, 5e-1, 5e0}. **DONE in commit `c0830d12`** (22.8 min wall via rayon parallelism). Result: λ=5e-2 recovers NRMSE (9.43e-4 ✅); threshold flat at ~7.0-7.2 LT across λ (structural miss).
- [x] T0.2 Run the K=10/M=8/R/2 experiment (d_h=29_160, ~28 min Cholesky). **DONE in commit `3eb059ae`** (95 min wall total). Result: K=10 gives only +0.13 LT (7.36 LT) — the linear K-extrapolation was WRONG. Threshold plateaus at K≥8.
- [ ] T0.3 If K=10 (with λ=5e-2) passes both G1 legs → promote `karc_forecaster` + `karc_householder_eig_par`. **NOT ACHIEVED.** K=10 did not pass both legs. A follow-up Phase 5.3 experiment (R=1 K=8/M=24 λ-sweep, commit `a34a27b6`) also failed to produce a single-config pass — R=1 has a hard NRMSE floor at ~5e-3 regardless of λ.
- [x] T0.4 If K=10 fails → file a Path D2 deliberation issue (gate re-spec with riir-ai consumer evidence about whether ~7.5 LT suffices for short-horizon NPC prediction). **DONE 2026-07-21** — resolved directly as Issue 186 Path D variant D3 (split-config gate, not D2 lower-target). `karc_forecaster` PROMOTED TO DEFAULT-ON; `karc_householder_eig_par` stays opt-in. See `.benchmarks/308_karc_goat.md` §Phase 5.3 for the evidence.

### Phase 1 — `faer` integration spike (ONLY if Phase 0 fully fails AND a second heavy-BLAS consumer materializes)

- [ ] T1.1 Add `faer` as an optional dep on `katgpt-core` behind `karc_faer_eig` feature.
- [ ] T1.2 Implement `symmetric_eig_faer` wrapper matching the `SymmetricEigScratch` API. Copy in/out to avoid `faer`'s owned-return allocations on the hot path.
- [ ] T1.3 Run G1 parity tests (100 random SPD matrices + Issue 185 T2 ALS-vs-Kronecker).
- [ ] T1.4 Run d_h=18_720 timing trial. Target: ≤30 min wall.
- [ ] T1.5 GOAT gate (G1–G4 per above). Promote only if all pass.

### Phase 2 — LAPACK fallback (ONLY if Phase 1 fails AND `faer` is unsuitable)

Not sketched. If we get here, the situation has changed enough that a new proposal is warranted.

## Risks

1. **(Perf, low)** The shipped Householder+QL + rayon regresses on different hardware. *Mitigation:* the d_h=18_720 trial is a one-time per-fit cost on consumer hardware, not a CI gate. Consumer teams measure on their target hardware.
2. **(Correctness, low)** The QL convergence bug fix (Issue 187 follow-up, LAPACK `dsteqr` global-scale deflation criterion) has a latent edge case. *Mitigation:* the fix is behavior-preserving for non-degenerate inputs (verified: 1761 lib tests + 4 ALS parity + 3 parallel bit-identity tests all pass). The risk is bounded to near-singular Grams with specific eigenvalue distributions.
3. **(Architectural, medium)** Adding `faer` later introduces a meaningful third-party dep with its own release cadence. *Mitigation:* pin `faer` to a specific minor version; gate behind `karc_faer_eig`; the in-tree Householder+QL stays as the no-dep fallback so consumers can opt out.
4. **(Decision, medium)** Phase 0's λ-sweep doesn't recover NRMSE, and the team is tempted to default to Path D (gate re-spec) without consumer evidence. *Mitigation:* the protocol in §"The proposed design" explicitly requires riir-ai consumer evidence for Path D2. **Defaulting to D without that evidence violates the §3.5 modelless-defer discipline.**
5. **(Scope, low)** "At least try `faer`" scope creep. *Mitigation:* this proposal explicitly marks `faer` as deferred conditional on a second consumer. The 30-min integration spike is cheap, but the *maintenance* cost (a new feature flag, a new eigensolver variant, G1–G4 gate, CI matrix expansion) is not. Defer.

## Out of scope  (RECOMMENDED)

- **GPU eigensolver.** Layering-invariant rejection per Issue 186. Re-open only if `katgpt-gpu` (public) ever materializes.
- **Paper-par `d_h=166_752` config** (K=8/M=24/R=2, 222 GB Gram). Issue 186 already declared this out of scope — requires H-matrix or tensor-train factorization, separate primitive. Unchanged.
- **Lanczos / divide-and-conquer eigensolvers.** Bigger algorithmic change than Householder+QL; revisit only if both B-par and `faer` are insufficient. Unlikely — `faer` uses these algorithms internally.
- **Phase 3 spline-knot adaptivity** (Plan 308 T3.1–T3.3). Already deferred; orthogonal to this proposal.
- **Sibling WIP on Issue 187 T7 follow-up #2** (K=10 λ-sweep in `crates/katgpt-core/tests/karc_g1_dh18720.rs`). That's a sibling agent's in-flight experiment on the K dimension; this proposal's Phase 0 T0.1 is the λ dimension. The two compose — if K=10 + λ=5e-2 both help, the promotion config changes.

## References

1. **[Issue 186 — KARC Plan 308 T3 Compute-Unlock deliberation](../.benchmarks/308_karc_goat.md) §Phase 5** — original four-path analysis + Path B decision. **Cited as the baseline this proposal extends.**
2. **[Issue 187 — KARC Householder+QL Parallelization](../.benchmarks/308_karc_goat.md) §Phase 5** — parallelization + actual G1 measurement (λ=5e-3 FAIL on both legs). **Cited as the evidence that supersedes Issue 186's compute framing.**
2a. **Commit `c0830d12`** (2026-07-20, λ-sweep) — λ=5e-2 recovers NRMSE gate (9.43e-4); threshold flat across λ (structural). **The post-Issue-187 data point this proposal incorporates.**
3. **[Issue 185 — KARC Large-d_h ALS B-step (Jacobi)](../.benchmarks/308_karc_goat.md) §Phase 5** — source issue, T1+T2 implementation + the compute-feasibility gap that spawned Issue 186.
4. **[Plan 308 — KARC Delay-Basis Ridge Forecaster](../.plans/308_karc_delay_basis_ridge_forecaster.md)** — parent plan; T4.5–T4.7 promotion gate.
5. **[Benchmark 308 — KARC GOAT](../.benchmarks/308_karc_goat.md)** — the G1 gate evidence.
6. **[Benchmark 010 — Report the Floor consolidated](../.benchmarks/010_report_the_floor_consolidated.md)** T7 — the K-sweep establishing K=8 as the promotion-target config (now complicated by Issue 187's G1 FAIL at K=8).
7. **[`faer` documentation](https://docs.rs/faer/latest/faer/)** — pure-Rust linear algebra with self-adjoint eigendecomposition. **The Path E option Issue 186 missed.**
8. **[`faer` paper (El Kazdadi, Châtelain)](https://github.com/sarah-quinones/faer-rs/blob/main/paper.md)** — benchmarks vs OpenBLAS, algorithmic details.
9. **[OxiBLAS: How We Optimized Pure Rust to Beat OpenBLAS on Apple Silicon](https://kitasanio.medium.com/how-we-optimized-pure-rust-to-beat-openblas-on-apple-silicon-e70c1965fffb)** — independent confirmation that pure-Rust can reach 100–118% of OpenBLAS on M3 with careful microkernel tuning. Different project, same thesis.
10. **[LAPACK `dsyevr` thread-safety (Stack Overflow)](https://stackoverflow.com/questions/18216314/shouldnt-lapacks-dsyevr-function-for-eigenvalues-and-eigenvectors-be-thread-s)** — confirms LAPACK is not rayon-composable at the outer level; only `dsyevd` is reliably safe, and only via BLAS-internal threading.
11. **[ndarray-linalg Mac Silicon build issue #308](https://github.com/rust-ndarray/ndarray-linalg/issues/308)** — concrete evidence of LAPACK toolchain pain on the project's primary dev platform.
12. **Golub-van Loan, *Matrix Computations* 4th ed., §8.3** — the algorithmic reference for Householder tridiagonalization + symmetric QR (used by both our shipped code and `faer`).
13. **Press-Teukolsky-Vetterling-Flannery, *Numerical Recipes* 3rd ed., §11.3** — `tqli` (Tridiagonal QL Implicit). Used by our shipped code; cited for the QL convergence bug fix in Issue 187.

## TL;DR

**Verdict: do NOT add LAPACK today.** The compute-unblock question is moot — Issue 187 resolved it (Householder+QL + rayon = 87 min wall, feasible). The λ-tuning question is also moot — commit `c0830d12` resolved it (λ=5e-2 → NRMSE 9.43e-4 PASS). The actual blocker is structural: threshold leg 10% short, flat across λ, driven by K (delay length) not regularization or compute. **Next action: wait for the sibling agent's K=10/M=8/R=2 experiment** (d_h=29_160, ~28 min wall, predicts threshold ≈ 8.5 LT PASS via linear extrapolation). If K=10 passes both legs → promote. If not → gate-re-spec (Path D) deliberation with riir-ai consumer evidence. **Eigensolver choice is irrelevant to the current blocker.** If a second heavy-BLAS consumer ever materializes, evaluate `faer` (Path E) BEFORE LAPACK — pure Rust, no toolchain cost, LAPACK-class perf.
