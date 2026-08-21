# Bench 671 — `spectral_pencil` primitive GOAT (Issue 676 T10, Research 495)

**Date:** 2026-08-22 · **Box:** M3 Max (Apple Silicon, release builds, quiet
box — no sibling compute during measurement) · **Feature:** `spectral_pencil`
(opt-in, implies `hebbian_kernel_memory` for `SeedRng`).

## G1 — correctness (property gates, all green)

| Gate | Tier | Result |
|---|---|---|
| sym isometry (`‖sym(v)‖_F == ‖v‖₂`, inner product) | 1–2 / 4 ulp | PASS (`sym::tests`) |
| pack/unpack round-trip | ≤1 ulp | PASS |
| Jacobi: known spectra, `A·V == V·Λ`, bit-reproducible | exact/≤1e-4 | PASS |
| Sturm count == Jacobi full-solve count | **10⁶ matrices (release, 2.82 s, ~7M midpoint checks)** + 10k always-on | PASS, zero mismatches |
| Sturm bisect == Jacobi values (d=10, 200 seeds) | ≤1e-4 rel | PASS |
| γk ≥ ½ on ‖x‖∞ ≤ 5 (dense seeded init, Lemma 2) | 8 seeds × 4 k × 64 pts | PASS |
| γk ≥ ½ (tridiag seeded init, ε = 1/(12Rn)) | 8 × 3 × 64 | PASS |
| Non-commutativity ‖[A₀,Aᵢ]‖_F > 0 | all features | PASS |
| Same-seed bit-identical construction | exact | PASS |
| Mirror duality λk(−A) = −λ_{d−k+1}(A) | 64 × all k, ≤1e-5 | PASS |
| k=1 concave / k=d convex (Claim 1) | midpoint sweeps | PASS |
| Growth envelope dominates \|f(x)\| | 200 × all k | PASS |
| Attribution vs central FD | **10⁵ probes (release)** ≤5e-3 | PASS |
| \|vᵀAᵢv\| ≤ ‖Aᵢ‖₂ (Lemma 1) | 512 × all features | PASS |
| PSD ⇒ non-decreasing, NSD ⇒ non-increasing (Loewner) | mixed pencil, 3 k, 40 steps | PASS |
| Rank-one fast path == dense quadratic | 256 trials ≤1e-4 rel | PASS |
| Gauge: f-invariance under random conjugation | 64 trials, all k | PASS |
| Gauge: canonical form preserves f + idempotent | ≤1e-4 / ≤1e-5 | PASS |
| Gauge: canonicalization bit-deterministic | exact | PASS |
| Warp: g⁻¹(g(x)) == x | **10⁵ constructions (release)** ≤1e-3 | PASS |
| Warp strictly increasing (B ≻ 0) | 4 k × 50 steps | PASS |
| T10 GOAT binary: G1 cross-construction bit-identical folds + headroom shapes (d∈{8,16,32}) + Jacobi converged at d=32 | 1 test (serial) | PASS |

## G2 — latency (criterion, median)

| Path | d=8 | d=16 | d=32 |
|---|---|---|---|
| dense eval (pinned Jacobi) | 3.95 µs | 21.1 µs | 166.7 µs |
| tridiag eval (Sturm bisection) | 748 ns | 2.05 µs | 3.71 µs |
| Sturm `count_below` (O(d) integer predicate) | — | **51 ns** | — |

**10k NPC × 20 Hz headroom arithmetic** (200k evals/s, printed per the issue):

- tridiag d=16: 2.05 µs × 200k = **0.41 core-seconds/s (~41% of one P-core)** — the production path at NPC scale.
- tridiag d=8: 748 ns × 200k = **~15% of one core**.
- Sturm predicate: 51 ns × 200k = **~1%** — free.
- dense d=16: 21.1 µs × 200k = **~4.2 cores — NOT affordable per-tick at 10k NPCs** (honest verdict: dense is the low-cardinality path — per-NPC personality readouts at spawn/GM-inspect cadence and GM-tool surfaces, not the 20 Hz swarm tick; that is exactly why the tridiagonal family exists, paper §9).

Note: the paper's §7.3 arithmetic (dense d=16 ≈ 8K FLOPs/eval) is the
theoretical op count; this implementation carries the determinism policy's
cost — f64 accumulation everywhere + `to_full` materialization per eval +
the pinned schedule's full sweeps. An f32-accumulated fast path is possible
future work if a hot consumer appears (none today — promotion gate is a
consumer GOAT, per the pattern).

## G3 — no-regression

- Default features: katgpt-core lib **1904 passed / 0 failed / 7 ignored** (module compiles out — opt-in).
- `--all-features` lib: compiles clean; spectral_pencil feature state: 38 unit tests green + GOAT binary green.
- `cargo clippy --features spectral_pencil --lib`: **0 warnings**.

## G4 — alloc-free steady state

GOAT binary (`tests/bench_676_spectral_pencil_goat.rs`, CountingAllocator,
single serial test fn): **0 allocs / 0 deallocs over 4×1000 hot-path calls**
(dense eval, tridiag eval, Sturm count, attribution) after 8-iteration warmup.

**En-route lesson (recorded in the test's doc comment):** the first version
had three parallel test fns in the binary and measured 12 phantom "leaks" —
thread-spawn bookkeeping from the sibling test landed in the counter window.
Per-path bisection showed every path individually at zero. The
`analytic_lattice_alloc_check` convention ("all checks in ONE function =
serial by construction") is load-bearing, not stylistic.

## En-route bug caught by the T8 gates (fixed before landing)

Raw off-diagonal writes into `SymPacked.data` silently halve in `to_full()`
(the `1/√2` packing convention) — `canonicalize` and `MonotoneWarp`'s B build
both had it; the invariance/idempotency tests failed with 0.003–0.07 drift
and the round-trip caught the rest. All writes now go through
`pack_from_full`/`set`. Diagonal-only writes are convention-safe.

## Verdict

**GOAT G1–G4 PASS (modelless).** Feature stays **opt-in** — promotion
requires a consumer GOAT (riir-ai Issue 736 is the armed converter; per the
house pattern, promotion is the consumer's gate to win).

## T12 — UQ follow-through: scope-limited (not shipped)

The eigengap-confidence sigmoid readout `σ(−γk/τ)` and the monotone
box→interval (`A(lo) ⪯ A(x) ⪯ A(hi)` ⇒ 2-solve certified interval) are
UQ-bearing claims — under the "Report the Floor" rule both must beat
`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`
(`katgpt-core/src/conformal/floor_harness.rs`) before shipping. **Neither is
exposed** by this landing: the raw eigengap γk ships as a structural
certificate (a trust flag threshold, not a calibrated probability). The
sigmoid/interval readouts + their floor benchmarks reopen with the first
consumer that needs them (riir-ai 736's Davis–Kahan stability surfaces) —
the KARC precedent: scope-limit honestly rather than ship unmeasured
confidence.
