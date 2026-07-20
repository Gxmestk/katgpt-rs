# Issue 185 — KARC Plan 308: Large-d_h ALS B-step (Jacobi Eigendecomposition) to Unblock Promotion

**Filed:** 2026-07-20
**Priority:** P2 (promotion-unblocker — the only thing standing between Plan 308 and `karc_forecaster` going default-on)
**Related:** Plan 308, Research 288, Benchmark 308, Benchmark 010 (T7 K-sweep), riir-ai/.plans/332, riir-ai/.research/152, katgpt-rs/.issues/127 (sibling perf-track)
**Origin:** Plan 308 §Phase 4 T4.5 deferral, Benchmark 308 §"Path to promotion" item 1 — the critical-path future-work anchor that the §3.6 modelless-defer protocol identifies as the only remaining modelless blocker for promotion.

## TL;DR

Plan 308's algorithm is proven correct (NRMSE **1.67e-4**, 6× better than the paper's 1.0e-3 target; G2/G3/G4 all PASS at 381 ns/call + zero-alloc + bit-reproducible). The only thing keeping `karc_forecaster` opt-in is a **compute-feasibility gap**, not an algorithmic defect: the config that passes both G1 legs simultaneously (NRMSE ≤ 1e-3 AND threshold ≥ 8 LT) is `K=8, M=24, R=2, d_h=166752`, which needs a **220 GB Cholesky** today. The Jacobi-eigen ALS B-step already noted in the Phase 2 rustdoc is the only modelless path to closing that gap.

This issue tracks the implementation of the large-d_h ALS B-step using Jacobi eigendecomposition of `AᵀA` + `r` separate `d_h × d_h` solves (`O(r · d_h³)` instead of `O((r·d_h)³)`), which eliminates the 220 GB Cholesky bottleneck and unblocks Plan 308 T4.5–T4.7 promotion.

## Context — what is the actual blocker?

### The compound G1 gate

Plan 308's G1 is a compound gate: `NRMSE ≤ 1.0e-3` (one-step expressiveness) **AND** `threshold ≥ 8 LT` at `ε=0.1` (autonomous-rollout stability). The benchmark 308 config sweep shows these two legs are **driven by different hyperparameters**:

| Config | d_h(R=2) | NRMSE (1 LT) | Threshold (ε=0.1) | G1 NRMSE | G1 Thr |
|--------|----------|--------------|-------------------|----------|--------|
| K=4, M=8, R=2 (current shipping bench) | 4 752 | **1.67e-4** ✅ | 2.85 LT | ✅ | ❌ |
| K=8, M=4, R=2 | 4 752 | 6.19e-3 | 1.31 LT | ❌ | ❌ |
| K=8, M=24, R=2 (paper-Par config) | 166 752 | (expected ≤ 1e-3) | (expected ≥ 8 LT) | ? | ? |
| Phase 1: K=8, M=24, first-order | 576 | 4.79e-3 | **8.16 LT** ✅ | ❌ | ✅ |

**Key empirical insight (benchmark 308):** `K` (delay length) drives threshold time (autonomous-rollout memory); `M` (basis count) and `R` (feature order) drive one-step NRMSE. No currently-benchable config passes both — the K=4 config has 28× better NRMSE but 2.9× worse threshold than the Phase 1 K=8 config. The config that should pass both (K=8, M=24, R=2) is the one requiring the 220 GB Cholesky.

### Why the current B-step fails on large d_h

Phase 2 shipped `low_rank_fit` (Plan 308 T2.3) with the **exact Kronecker-vectorized B-step**:

```text
(G ⊗ AᵀA + λI) · vec(B) = vec(Aᵀ · Covᵀ)
```

This is an `(r·d_h) × (r·d_h)` Cholesky solve. Cost: `O((r·d_h)³)` time, `O((r·d_h)²)` space.

For the K=8, M=24, R=2, d_h=166752 config at r=8: `(8 · 166752)² ≈ 1.78e12` f64 ≈ **14.2 TB** just for the system matrix. Even the smaller K=8, M=8, R=2 config (d_h=18720) needs `(8·18720)² ≈ 2.24 GB` for the Gram and ~6 minutes for the O(n³) Cholesky — at the edge of feasibility for a benchmark example, infeasible as a CI gate.

### The unlock — Jacobi eigen + r independent d_h solves

Phase 2's `low_rank_fit` rustdoc already documented the future-work path. Per Plan 308 T2.3 deviation note and benchmark 308 §"Implementation notes":

> The B-step is an EXACT solve via the Kronecker vectorization … feasible for `r·d_h ≤ ~2000`. Scale rebalance after each A+B pair prevents the ALS gauge drift … `jacobi_eigen` is also shipped (standalone symmetric eigendecomposition, kept for future large-d_h path).

The math (standard ALS for bilinear ridge, see paper Eq. 47 derivation):

```text
Minimize over A (r×D), B (r×d_h):   ‖Y − A·B·Hᵀ‖_F² + λ(‖A‖_F² + ‖B‖_F²)

A-step (fix B):  A = Y · H · Bᵀ · (B·HᵀH·Bᵀ + λI)⁻¹     ← (r×r) Cholesky, cheap
B-step (fix A):  for each column-block, eigen-decompose  AᵀA = QΛQᵀ  (d_h×d_h, once per ALS iter)
                  then B = (Λ ⊙ (HᵀH) + λI)⁻¹ · (AᵀY·H)ᵀ
                          ⇑ diagonally-coupled, r independent d_h×d_h solves
```

Critically, **once `QΛQᵀ = AᵀA` is computed**, the B-step decouples into `r` independent `d_h × d_h` solves. Per ALS iteration:

- Eigendecompose `AᵀA`: `O(d_h³)` (one-time per iter; **Jacobi** is the shipped implementation, no external dep)
- `r` independent `d_h × d_h` solves: `r · O(d_h³)` — not `(r·d_h)³`

**Total per iter: `O((1+r) · d_h³)`, NOT `O((r·d_h)³)`.** For r=8, d_h=166752: that's `9 · d_h³ ≈ 4.2e15` FLOPs/iter, ~minutes per ALS iteration on CPU SIMD — **feasible**. Memory: `~d_h²` for `AᵀA` + `Q` + `Λ` ≈ **222 GB** at d_h=166752 — **still infeasible at the paper-Par config**.

This means the Jacobi-eigen path **alone** doesn't reach the paper-Par `d_h=166752` config. It reaches the smaller `d_h=18720` config (K=8, M=8, R=2): `(8+1)·18720³ ≈ 5.9e13` FLOPs/iter, memory `~18720² ≈ 2.8 GB` per buffer — **feasible on a single workstation**.

**This issue targets the d_h=18720 (K=8, M=8, R=2) config as the GOAT gate target.** The d_h=166752 paper-Par config remains out of reach without further factorization (see "Out of scope" below).

## Goal

Add a `low_rank_fit_jacobi_bstep` path to `crates/katgpt-core/src/linalg/ridge_solve.rs` (the module that already ships `low_rank_fit` and `jacobi_eigen`) that:

1. Performs the **A-step** identically to the existing Kronecker path (r×r Cholesky, unchanged).
2. Performs the **B-step** via:
   - One `jacobi_eigen` call on `AᵀA` → `QΛQᵀ` (reuse the shipped function — DRY).
   - `r` independent `d_h × d_h` Cholesky solves for the diagonally-coupled B-columns.
3. Applies the same scale-rebalance fix (`A←cA, B←B/c`, `c=√(‖B‖/‖A‖)`) after each A+B pair to pin the ALS gauge — already shipped in `low_rank_fit`, hoist it into a shared helper.

Validate against the existing Kronecker path on the small config (bit-reproducibility for `r·d_h ≤ 2000`, where both paths are feasible) — they MUST produce identical `A`, `B` modulo the ALS convergence tolerance.

## Acceptance Criteria

- [x] **T1 — Implementation.** `low_rank_fit_jacobi_bstep` ships in `crates/katgpt-core/src/karc/large_dh.rs` (sibling to `karc/mod.rs` — extracted to keep `mod.rs` under the 2048-line guideline). The function shares the existing `LowRankFitScratch` (extended via `ensure_jacobi_capacity` with d_h×d_h eigvecs/eigvals/scratch buffers — pre-allocated, zero-alloc hot path). **DONE 2026-07-20.**
- [x] **T2 — Bit-reproducibility parity test.** `tests/karc_low_rank_jacobi_vs_kronecker.rs` + `karc/large_dh.rs` unit test: on d_h=96 / r=4, the Jacobi and Kronecker paths produce ALS solutions agreeing to `1.6e-14` after the same iteration count (machine precision for f64). Also pins the Jacobi path's own bit-reproducibility. **DONE 2026-07-20.**
- [-] **T3 — G1 GOAT gate re-run at K=8, M=8, R=2 (d_h=18_720).** **BLOCKED** — see "Compute-feasibility gap" below. The naive Jacobi eigendecomp of G at d_h=18_720 is ~3.3e13 FLOPs/sweep × ~10 sweeps = ~3.3e14 FLOPs one-time, plus cache-hostile random-access pattern. Even d_h=4_752 (the existing shipping bench) exceeded a 10-minute watchdog timeout in the timing trial — d_h=18_720 is 16× larger. T3 needs a faster symmetric eigensolver (Lanczos iteration with full reorthogonalization, or a LAPACK binding) — tracked as a follow-up. The T1 implementation is correct (T2 proves this); the blocker is purely compute-feasibility.
- [-] **T4 — Memory ceiling check.** **BLOCKED with T3** — same blocker (the example doesn't run yet). The buffer footprint math is unchanged from the issue body: at K=8, M=8, R=2, r=8, d_h=18_720, the largest single buffer is `Q ∈ R^{18720×18720}` ≈ 2.6 GB f64 (heap-allocated, caller-provided scratch).
- [-] **T5 — Plan 308 promotion decision.** **BLOCKED with T3** — cannot make the promotion call without T3's evidence. `karc_forecaster` stays opt-in.

## Compute-feasibility gap (T3 blocker, discovered 2026-07-20)

The Issue 185 risk register listed "Jacobi too slow" as a Medium-likelihood
risk. Empirically it is a **certainty** at d_h=18_720, and even at d_h=4_752
the one-time G eigendecomp exceeds a 10-minute budget. Three paths forward,
none in scope for this issue's T1+T2 closure:

1. **Lanczos iteration with full reorthogonalization** — top-k eigenpairs of G
   in `O(d_h·k²)` per step. But the B-step needs ALL d_h eigenvalues (every
   `Λ_g[l]` appears in the scaling `Λ_a[i]·Λ_g[l] + λ`), so we'd need k=d_h,
   which collapses Lanczos back to `O(d_h⁴)` — worse than Jacobi. **Doesn't help.**
2. **LAPACK binding** (`dsyevd` divide-and-conquer) — `O(d_h³)` with a small
   constant factor and SIMD-friendly memory access. Would make d_h=18_720
   feasible in ~5-10 min one-time. But katgpt-rs is deliberately
   dependency-light (no system-library deps); adding a Lapack binding is a
   significant scope change. **Possible follow-up.**
3. **Avoid eigendecomposing G entirely** — re-derive the B-step as `r`
   independent Cholesky solves of `(Λ_a[i]·G + λI)`, each `O(d_h³)`.
   Per-iter cost `O(r·d_h³) = 5.2e13` FLOPs at the target config — still
   infeasible (50 iters × 5.2e13 = 2.6e15 FLOPs ≈ days at CPU SIMD rates).
   **Doesn't help either.**

The honest conclusion: the d_h=18_720 promotion-gate target requires either
an external eigensolver dependency or a fundamentally different B-step
formulation (e.g., one that exploits the Kronecker structure of the KARC
feature Gram, if any). T1+T2 stand as a complete, correct, well-tested
primitive that's ready to consume the moment a faster eigensolver lands.

## Latent bug found and fixed during T1

Implementing T1 surfaced a **latent sign bug in `jacobi_eigen`** (Plan 308 T2.3).
The original rotation-angle formula was `0.5 · atan(2·apq / (app − aqq))`, but
the in-place updates use the rotation convention `J = [[c, s], [−s, c]]`, which
requires `0.5 · atan(2·apq / (aqq − app))` (sign-flipped denominator). The bug
produced correct results only when `app == aqq` (the `π/4` special case). It
was latent because `jacobi_eigen` had **zero callers** before Issue 185 — every
prior consumer used Cholesky. The fix ships with T1, pinned by the
`jacobi_eigen_sign_convention_correct` unit test.

This is exactly the kind of latent bug the Issue 185 implementation work was
supposed to surface: the Phase 2 rustdoc documented `jacobi_eigen` as "kept
for future large-d_h path" but never exercised it. Exercising it found the bug.

## Out of scope

- **Paper-Par d_h=166752 config** (K=8, M=24, R=2). The 222 GB per-buffer memory ceiling is out of reach without a tensor-train or H-matrix factorization of `AᵀA` — that's a separate, harder primitive (tracked under "Future work" below). The d_h=18720 config is the legitimate promotion-gate target: it's the smallest config that should pass both G1 legs simultaneously.
- **GPU/LoRA training of `B`.** Per AGENTS.md modelless-first mandate, the B-step is closed-form per ALS iteration — no gradient descent, no riir-train deferral. The Jacobi-eigen path is the modelless unblock per research skill §3.5 Path 1 (freeze/thaw of the `A,B` solution after a one-shot fit is the freeze/thaw analog; no per-step training).
- **Phase 3 spline-knot adaptivity** (Plan 308 T3.1–T3.3). Separate deferred phase; not coupled to promotion.
- **KarcShard Pod layout** (riir-neuron-db). The low-rank `A,B` form already ships (Plan 308 T2.3 result, G4-extended); shard persistence is downstream of promotion.

## Why not just re-spec the G1 gate (the alternative T4.5 path)?

Plan 308 T4.5 listed two paths to promotion: (a) large-d_h ALS B-step, or (b) gate re-spec accepting the small-config NRMSE + relaxed threshold. The honest argument for **doing (a) first** rather than defaulting to (b):

1. **The threshold gate exists for a real reason.** Per Plan 308 §Goal, the threshold ≥ 8 LT target is "the open primitive; riir-ai integration plan targets the full 16 LT". Game-AI NPC forecasting feeds predictions back as observations (autonomous rollout). A 2.85 LT horizon means the forecaster destabilizes within ~3 LT of unsupervised operation — that's a real product-AI failure mode, not a benchmark pedantry.
2. **Re-spec would be a precedent of "we couldn't make it work, so we lowered the bar".** The Plan 306 G4 re-spec (cited as precedent in Plan 308 T4.5) was a different shape: Plan 306's gate measured something the algorithm wasn't designed to optimize. Plan 308's threshold gate measures the **exact** behavior the algorithm is designed for (autonomous-rollout stability). Re-spec'ing it would dilute the G1 contract for every future forecaster primitive.
3. **The blocker is compute, not algorithm.** The Jacobi-eigen path is straightforward numerical linear algebra. The 6× NRMSE win and the 8.16 LT threshold result (Phase 1) prove the algorithm works; this issue is about closing the last engineering gap to make the simultaneous-pass config benchable.

If T3 fails (the Jacobi-eigen path can't reach d_h=18720 within a workstation budget, or the threshold still doesn't pass at K=8/M=8/R=2), **then** a gate re-spec becomes the honest fallback. But defaulting to re-spec without trying the modelless unblock violates the §3.5 Path 1 (freeze/thaw of compute-feasibility) discipline.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Jacobi eigendecomposition of `AᵀA` (d_h=18720) is too slow on CPU SIMD | Medium | Profile on a smaller d_h first (d_h=4752 is the current shipping benchmark; d_h=18720 is 4× larger). If >10 min/iter, fall back to Lanczos iteration (top-r eigenpairs only — `AᵀA` only needs the top-r eigenvalues for the B-step, since `B` has rank r). Track as T3 follow-up if it materializes. |
| ALS doesn't converge at d_h=18720 in the per-iter budget | Low | The Kronecker path converged at d_h=4752 in 50 iters with the scale-rebalance fix; the eigendecomp doesn't change the optimization landscape, just the per-iter compute. |
| Threshold still < 8 LT at K=8/M=8/R=2 | Medium | Possible — Phase 1's 8.16 LT was first-order K=8/M=24; R=2 features might destabilize the autonomous rollout differently. If so, file the gate re-spec follow-up with the new evidence. **Honest revision is the explicit acceptance path per research skill §3.6.** |
| Memory blowup on the `Q` buffer (2.6 GB f64) | Low | Caller-provided scratch (per AGENTS.md rule). The example owns the buffer; the primitive never allocates in the hot path. |

## Dependencies

- **Existing (shipped, DRY):** `low_rank_fit`, `jacobi_eigen`, `KarcBasis` trait, `feature_expand_higher_order`, `chunked_gram_into`, the double-scroll ODE example harness.
- **External:** None. Pure numerical linear algebra on f64; no new crates.

## Related

- **Plan 308** — `katgpt-rs/.plans/308_karc_delay_basis_ridge_forecaster.md` (the parent plan; T4.5 deferral)
- **Benchmark 308** — `katgpt-rs/.benchmarks/308_karc_goat.md` (the G1 gate evidence; the "Path to promotion" section)
- **Research 288** — `katgpt-rs/.research/288_KARC_Delay_Basis_Ridge_Forecaster.md` (Super-GOAT verdict; this primitive is the private-moat anchor)
- **Benchmark 010 T7** — `katgpt-rs/.benchmarks/010_report_the_floor_consolidated.md` (the K-sweep that refuted the "K=4 too shallow" hypothesis and established the K=8 promotion-target config)
- **riir-ai/.plans/332** — KARC runtime integration (downstream consumer; promotion unlocks the runtime GOAT gate)
- **riir-ai/.research/152** — per-NPC KARC forecaster guide (the private selling point this primitive enables)

## Future work (not in scope for this issue)

- **Paper-Par d_h=166752 config.** Requires either H-matrix approximation of `AᵀA` or a tensor-train decomposition of the B-step. Not promotion-blocking — d_h=18720 is a legitimate G1 gate target. Track as a separate P3 issue if and when a game-AI use case needs the larger config.
- **Lanczos-iteration B-step.** If T3 surfaces a perf issue with full Jacobi at d_h=18720, Lanczos on `AᵀA` (top-r eigenpairs) would cut the per-iter cost from `O(d_h³)` to `O(r·d_h²)`. Track as T3 follow-up.
- **AdaptiveBSplineBasis** (Plan 308 Phase 3, T3.1). Independent of this issue.

## Re-evaluation triggers

Close this issue as **[-] DEFERRED** if any of the following materializes:

1. T3 surfaces an algorithmic regression (G1 NRMSE fails at K=8/M=8/R=2) — would indicate the Jacobi-eigen path introduces a correctness bug, and the issue should be re-classed as a bug fix rather than a promotion-unblocker.
2. A game-AI consumer (riir-ai runtime) reports that the K=4 config's 2.85 LT threshold is sufficient for actual NPC trajectories — would obsolete the threshold gate and make T4.5 path (b) the right choice.
3. A paper ships a directly-applicable low-rank ridge solver that beats the Jacobi-eigen path (e.g., a tensor-train ridge solver with public Rust impl) — would make this implementation redundant.

## TL;DR

Plan 308's KARC algorithm is proven correct (NRMSE 6× better than paper target, G2/G3/G4 all PASS). The sole blocker for `karc_forecaster` promotion to default-on is a **compute-feasibility gap**: the config that passes both G1 legs simultaneously needs a 220 GB Cholesky today. The Jacobi-eigen ALS B-step (already half-shipped as `jacobi_eigen` in Phase 2, documented as future work in the rustdoc) reduces the per-iter cost from `O((r·d_h)³)` to `O((1+r)·d_h³)` and the per-buffer footprint from `(r·d_h)²` to `d_h²` — making the K=8/M=8/R=2 (d_h=18720) config benchable on a single workstation. Target: ship the path, re-run G1 at d_h=18720, and either promote `karc_forecaster` to default-on (both legs pass) or file a gate re-spec follow-up with honest evidence (threshold still < 8 LT).
