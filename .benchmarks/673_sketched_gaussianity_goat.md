# 673 — Sketched Gaussianity Probe GOAT (Issue 681, Research 498)

**Landed:** 2026-08-22
**Feature:** `katgpt-core/gaussianity_probe` (opt-in) + root forwarder + katgpt-spectral `gaussianity_agreement` (test-only)
**Source:** Research 498 (LeVLJEPA arXiv:2607.00784 — SIGReg distilled from training loss to inference-time diagnostic); the paper's K=8-16 fixed 1D projections + per-projection normality statistic, applied as a measured guard.

## What shipped

`katgpt-core/src/data_probe/gaussianity.rs` — the Cramér-Wold sketch for d-dimensional embedding populations:

- **16 fixed directions**: 4 coordinate-axis anchors (`e_0..e_3`) + 12 BLAKE3-derived Rademacher ±1 rows (seedable, exact in f32). The axis anchors are the honest fix for the issue's own motivating case: a purely random sketch dilutes an axis-aligned bimodal mixture by |cos| ≈ 1/√d (at d=64, μ=2σ separation shrinks to 0.5σ in projection — invisible); the anchor catches it at any d.
- **Per direction**: KS-vs-fitted-Gaussian D statistic — a VERBATIM port of `katgpt_spectral::spectral::ks_d_statistic` (same sort, same f64 accumulation order, same A&S `normal_cdf`), made because the katgpt-core leaf must not depend on katgpt-spectral (`rrq_quant`). Public as `ks_d_vs_fitted_gaussian`; the port is pinned bit-identical by `katgpt-spectral/tests/gaussianity_agreement.rs` (2 tests: identical 1D samples → identical bits; full-probe `per_direction[a]` vs `ks_d_statistic` on f64-reconstructed projections → identical bits).
- **Aggregate**: `score = sigmoid(ln(p_min / 0.01))` where `p_min` is the NR Kolmogorov complementary CDF at the worst D (n-aware — KS critical ∝ 1/√n; saturates at 1.0 below λ=0.3 where the series diverges). The log-multiple margin is scale-free; an earlier linear-p margin draft was caught in design (sigmoid(−κ·p₀) bottoming at 0.48 for hard rejects — the p ∈ [0, p₀] span is too small to saturate).
- **Zero-alloc** (G4): `GaussianityScratch { directions, projections }` allocated once; probe steady-state 0 allocs/100 calls (in-module TrackingAllocator test, the latent_confounder_audit pattern).

## GOAT verdicts

| Gate | Verdict | Evidence |
|---|---|---|
| G1 fixtures | **PASS** | (i) Gaussian d=64 n=1024: score **0.9756** (accept), erank 63.6/64. (ii) Bimodal e_0 μ=3σ: score **2.4e-23** (hard reject), worst_dir=0 (anchor), D_0=0.167. (iii) Radial 5%@10: score **1.0e-28**, **16/16** directions D>0.1 (margin-wide). (iv) Lattice {0,1} d=8: score **1.0e-28**. In-module: 11/11 tests. |
| G1 non-redundancy pin | **PASS** | `effective_rank` on the SAME fixtures: (ii) **53.3/64 = 83.3%** ("healthy" to the rank metric) while the probe scores 2.4e-23 — the blind spot pinned. (iii) 57.3/64 = 89.5%. (iv) 7.97/8. The μ=3σ operating point is the tradeoff sweet spot: the mixture spike consumes exactly one covariance eigenvalue (theory 80% of d; empirical 83%), still far above collapse-scale (1-10%). |
| G2 latency | **PASS** | n=1024 d=64: probe **697.7 µs** vs `effective_rank` **2928.0 µs** (probe **4.20× faster**; erank pays the O(d³) Jacobi sweep + per-call row allocation). 50-rep mean, release. |
| G3 no-regression | **PASS** | default lib 1904/0/7i (unchanged), feature-on 1915/0/7i (+11), `--no-default-features` clean, `--all-features` clean, root forwarder compiles, clippy `-D warnings` clean (lib + bench + spectral tests in their feature states). |
| G4 alloc | **PASS** | 0 allocations / 100 steady-state probe calls (in-module TrackingAllocator, sentinel-guarded). |
| G5 determinism | **PASS** | 3 runs bit-identical (per_direction array + score + p_min bits). |
| Cross-crate agreement | **PASS** | katgpt-spectral `gaussianity_agreement`: 2/2 bit-identical vs `ks_d_statistic`. |

Run: `cargo bench -p katgpt-core --features gaussianity_probe,sink_aware_attn --bench bench_681_gaussianity_goat` (prints the verdict table).

## Honest findings (recorded in the module docs too)

1. **The blind spot is real and bounded**: a non-axis-aligned, moderate-strength (μ ≲ 3σ) bimodal departure in high d is missed by all 16 directions (a random Rademacher direction at d=64 has |cos| ≈ 1/8 with the mixture axis). The sketch is the cheap always-on audit; `ica_lens` (katgpt-spectral, FastICA) is the optimizing locator a consumer runs when the sketch trips. Complements, not competitors — and the leaf constraint forbids consuming ica_lens from core anyway.
2. **CLT smoothing hides per-coordinate idiosyncrasy at high d**: projections are sums; per-coordinate discreteness/non-Gaussianity washes out for d ≳ 32. The lattice fixture uses d=8 where sums stay visibly discrete. Margin-WIDE departures (mixtures across samples, radial heavy tails) are caught at any d — that is the probe's designed target.
3. **Two-point KS D is scale-invariant at exactly 0.5 − Φ(−1) ≈ 0.341** (fitted σ = the point magnitude c) — pinned by test for c ∈ {1, 5}.
4. **Dead-guard quirk faithfully ported**: the original `ks_d_statistic`'s `if std < 1e-10 return 0.0` can never fire (std is clamped `.max(1e-10)` the line before) — a constant sample returns 0.5, not 0. The port keeps this bit-identical (agreement trumps cleanup); noted for the spectral-side owner.
5. **Fixture-threshold corrections during calibration** (both caught by the falsifiable fixtures, which is the point of having them): the true population D for a μ=2σ bimodal vs its fitted Gaussian is ≈0.096 (the mixture CDF at x=μ is 0.75, not 0.5 — a first-principles arithmetic slip), hence the μ=3σ operating point; two-point D is 0.341, not 0.5.

## Promotion verdict

**Stays opt-in** (`gaussianity_probe = []`, not in default) per the issue's own T5: promotion is a consumer decision. The three waiting consumers: band_conditioner Fisher-z precondition guard (advisory field), riir-ai Issue 743 edge_lora hidden-space monitor (Phase 1 T1-T2 consume erank today; this probe is the deeper blind-spot detector), riir-neuron-db freeze-gate advisory (`FreezeGateReport` additive field — the bimodal-two-styles-before-freeze case).

## Substrate check (substrate-first skill, Mode 1)

- Searched: gaussianity, normality, ks_d_statistic, kolmogorov, epps/pulley, rademacher, sketched, cramer, projection test (paper + operator vocabulary, 3+ variants each).
- Found: `ks_d_statistic` (katgpt-spectral — the port source), `effective_rank`/`within_class_effective_rank` (geometry.rs — second-moment blind spot + G2 comparison target), `ica_lens` (FastICA optimizing locator — complement), `types::Rng` (fixtures), qmc `ks_uniform` test helper (NR Kolmogorov CDF precedent), `counting_allocator!`/TrackingAllocator (G4).
- Decision: build new (the multi-direction Cramér-Wold sketch for populations does not exist; the issue filed it in the right repo — modelless inference primitive, no riir deps, katgpt-rs upstream).
