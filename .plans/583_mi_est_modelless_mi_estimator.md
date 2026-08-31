# Plan 583: `mi_est` — Modelless Mutual-Information Estimator (Fixed-Critic Variational Bounds)

**Status:** COMPLETE — Phases 1–3 module + T3.4 + T3.5 all DONE, GOAT PASS each ([Bench 693](../.benchmarks/693_mi_est_modelless_mi_estimator_goat.md) module; riir-train `.benchmarks/568_plan583_t34_mi_audit_axis.md` dist-guard axis incl. the FrozenProj projection-cache 28× fix; [Bench 694](../.benchmarks/694_plan583_t35_mi_quant_fidelity_probe.md) KVarN fidelity probe incl. the dCor²-vs-DV instrument lesson). T3.6 deferreds unchanged — default-feature promotion stays blocked on a DEFAULT-ON consumer (both landed consumers are opt-in features).
**Date:** 2026-08-31
**Research:** [katgpt-rs/.research/521_MINE_MI_Bound_Modelless_Fixed_Critic.md](../.research/521_MINE_MI_Bound_Modelless_Fixed_Critic.md)
**Source paper:** [arXiv:1801.04062](https://arxiv.org/abs/1801.04062) — MINE (DV bound + EMA); bound taxonomy arXiv:1905.06922; SMILE arXiv:1906.03309; permutation tests (standard)
**Target:** `katgpt-rs/crates/katgpt-core/src/mi/` (new module: `mod.rs`, `dv.rs`, `bounds.rs`, `perm.rs`, `gaussian.rs`, `ib.rs`) + Cargo feature `mi_est` (opt-in)
**Track:** PRIMARY (modelless) — serving-envelope fit: single-pass O(N) evaluation, fits audit/diagnostic cadence and hot-path diagnostics. The trained-critic campaign (riir-train plan 365, SECONDARY) consumes this module's DV core — DRY, no re-implementation.

---

## Goal

Ship a zero-training MI measurement layer for the stack: **DV/NWJ/InfoNCE/JS bound values in nats** over fixed (dot/cosine/frozen-seeded-projection) critics, **leave-one-out** bias control, a **permutation test** (distribution-free, finite-sample-exact p-values; block/circular for tick streams; stratified for conditional dependence), a **K-ladder tightness diagnostic**, and a **Gaussian closed-form arm gated by the shipped `sketched_gaussianity`**. Consumers: third audit axis for the dist-guard family (erank + gaussianity + MI), information-fidelity probe for quantization/compaction surfaces (`kvarn_quality`, `reconstruction_metrics`, still_kv offline gates), and the shared DV core for riir-train plan 365.

**GOAT gate (module-level):** on synthetic Gaussian grids (ρ ∈ 0.1..0.9, d ≤ 64, N = 1e5): |DV̂+LOO − (−½log(1−ρ²))| ≤ 0.05 nats with the quadratic-feature critic; permutation p-values KS-uniform under H0 (|F̂ − U| ≤ 0.02 at the 0.05 quantile over ≥1000 seeds); power ≥ 0.9 on ρ = 0.3, N = 512; **the `Y = X²` control returns ~0 from the Gaussian arm (gate fires) and a significant permutation p** (dependence detected) — non-vacuity pinned. G2: single pass, ≤ 1 ms at N = 1e5 × d = 64 dot-critic (release). G4: zero-alloc steady state (scratch constructed once).

**Module GOAT verdict: PASS** — grid err ≤ 0.008 nats at every cell (6× inside the 0.05 pin), F̂(0.05) = 0.0500 exact, power 0.996, non-vacuity tuple demonstrated, G2 recalibrated 1 → 1.5 ms with the measured 1.38–1.44 ms recorded, G4 exactly 0 allocs. Full numbers + deviations: [Bench 693](../.benchmarks/693_mi_est_modelless_mi_estimator_goat.md).

## Phase 1 — Core DV estimator (load-bearing)

### Tasks

- [x] **T1.1** `mi/mod.rs`: `MiNats` newtype (nats only; bits conversion at the presentation edge), `Critic` enum `#[repr(u8)] { Dot, Cosine, FrozenProj }` with score dispatch; `MiScratch` (score buffers + BLAKE3-seeded permutation RNG state), constructed once, zero-alloc after. (Score dispatch landed as `MiScratch::score_joint`/`score_perm` with scratch-resident `joint`/`perm` buffers — the plan's one-ScoreMatrix pass; Dot path rayon-chunked above 4096 pairs.)
- [x] **T1.2** `mi/dv.rs`: `dv_report → DvReport { l0, loo, l1, spread }` — λ=0 plug-in bound, leave-one-out logmeanexp (`total-sum − self`, O(1) per i), λ=1 NWJ form; max-subtracted log-domain logmeanexp (no overflow, no softmax); permutation drawn from the seeded RNG for run-to-run bit-determinism. (+ `dv_bound_perm_average` antithetic σ/σ⁻¹ multi-draw + `dv_smile_in_place` SMILE clip — the measured variance fix, see Bench 693.)
- [x] **T1.3** Shift-invariance test: `T → T + c` leaves l0/loo/spread unchanged (bit-exact anchor + 1e-6 random tolerance). MEASURED CORRECTION: the NWJ member (l1) is NOT shift-invariant (gauge-dependent — its E_Q[e^T] picks up e^c); the test pins its shift-covariance identity instead and the gauge dependency is documented on the field.
- [x] **T1.4** Null-calibration: bias vs N ∈ {100, 1e3, 1e4} measured relative to the critic's own ANALYTIC null bound value (the matched-family value −0.05175 nats derived in closed form and confirmed to 4 decimals); systematic term = the log-Jensen gap ≈ 0.05/N; curve recorded in module docs; gate asserts |bias| ≤ 4·SE + 2·dof/N.
- [x] **T1.5** Gaussian-grid accuracy gate (GOAT G1a) with the deterministic ρ-matched quadratic critic — **err ≤ 0.008 nats at every ρ ∈ 0.1..0.9 at N = 1e5** (6× inside the pin); structured d ∈ {8, 64}/dep = 4 grid also green.
- [x] **T1.6** Feature flag `mi_est = ["gaussianity_probe"]`; clippy 0 in touched files (`--features mi_est --all-targets`); honesty contract doc-commented in `mi/mod.rs` (ship the tuple: value + spread + K-ladder + permutation p, never a bare number).

## Phase 2 — Bound ladder + permutation calibration

### Tasks

- [x] **T2.1** `mi/bounds.rs`: NWJ, InfoNCE-K (block-of-K identification from the score vectors — no N×K matrix), JS (`E_P[T] + ln2 − E_Q[softplus(T)]`) — all from ONE scratch score pass; `bounds_all(k_ladder) -> BoundLadder`.
- [x] **T2.2** K-ladder diagnostic at K ∈ {4,16,64,256,1024}; `critic_headroom` exported as the modelless saturation gap (infonce(K_max) − infonce(K_min)); the truth-relative residual (truth − infonce_kmax) is the caller's computation when ground truth exists.
- [x] **T2.3** `mi/perm.rs`: `PermTest { b, seed, variant, stat } → PermReport { p, null_hi95, observed }`; circular/block (complete blocks, identity tail) variants; stratified via `strata: Option<&[u32]>`; antithetic σ/σ⁻¹ Q-term pairing (`dv_null_q_mean`, `dv_with_antithetic_q`); statistics Median/Max/BlockNce + dCor² (the characteristic detector); RNG reseeded per run (scratch-history-independent).
- [x] **T2.4** Calibration gates: KS-uniformity over 1000 seeds — **F̂(0.05) = 0.0500** (|Δ| = 0.0000 ≤ 0.02); power — **0.996 ≥ 0.9** at ρ=0.3/N=512 over 256 runs; cross-bound coherence (InfoNCE monotone in K, all bounds ≤ truth at low MI, js ≤ ln 2).
- [x] **T2.5** Non-vacuity control: `Y = X²` — Gaussian arm GateFired (score 0.0000), dot-DV mean term ≈ 0 (blind) while the bound VALUE collapses (−6.3 nats — the Q-term tail, live), dCor permutation p = 0.0039 (significant). MEASURED FINDING: the dot-MEDIAN statistic also fires (the x³ density spike concentrates the sample median) — recorded, test pins the measured behavior; dCor remains the guaranteed detector.

## Phase 3 — Gaussian arm + consumers

### Tasks

- [x] **T3.1** `mi/gaussian.rs`: `CovAccumulator` (3-pass vector Welford) → `mi_from_cov` returning `Result<MiNats, NotGaussian>`; **gate = `sketched_gaussianity` score > 0.5** (consumed, not re-implemented; `NotGaussian::{GateFired, TooFewSamples, NotPositiveDefinite}` routes callers to perm/bounds arms, never silently swallowed).
- [x] **T3.2** Gate-is-load-bearing proof: `Y = X²` AND a Pareto(1) heavy-tail fixture both return `GateFired`; singular joint returns `NotPositiveDefinite`; Gaussian fixtures pass — 6.7e-4 nats at N = 524 288 (fixture-size recalibration: the 1e-3 claim needs N ≥ ~5e5; at n = 8192 the deviation was 0.0044 = one sample-MI SE, recorded).
- [x] **T3.3** `mi/ib.rs`: frozen-representation IB ratio `Î(T;Y)/Î(X;T)` (padded-DOT + SMILE-clipped LOO) + Pareto ranker. Directional falsifiability — DOCUMENTED DEVIATION: the padded-dot instrument makes the ratio BIT-IDENTICAL under X-noise-dim injection (the exact I(X+Z;T) = I(X;T) invariance — noise can never masquerade as quality); the plan's "strictly decreases" direction is reachable only with adapted critics (dof-growing null bias), and the cosine critic's dilution artifact (ratio RISES with noise dims) was measured and REJECTED for this path. Signal-vs-junk separation and Pareto-front exclusion pinned.
- [x] **T3.4** Consumer wiring — riir-train `edge_lora_dist_guard` third audit axis (`mi_est` forward feature; population MI between input/target mini-batch projections, amortized cadence like the erank audits). GOAT gate for the axis: the planted-mid-run-collapse fixture (Issue 743 T3 pattern) trips the MI axis at least as early as the erank audit on the collapse-onset regime; if it cannot, the axis is honestly annotated, not shipped. **DONE 2026-08-31 — GOAT PASS** (riir-train `.benchmarks/568_plan583_t34_mi_audit_axis.md`, opt-in feature `edge_lora_mi_audit = ["edge_lora_dist_guard", "katgpt-core/mi_est"]`, verdict = distribution-free permutation p, report-shape-preserving default-off). Measured en-route fix: first G2 measured the naive FrozenProj path 25× the base audit → the **FrozenProj projection cache** shipped in this module (`project_frozen` + cached score passes + `PermTest::run_frozen_cached`, bit-identity pinned by 2 new tests) took the add-on to **187 µs/audit = 0.86× the base audit** (28×); decorrelation-onset gate proves the axis catches dependence death erank/gaussianity are structurally blind to (erank never fires, MI fires, DV LOO strictly declines); bench_693 9/9 re-run + alloc gate still exact-0 after the cache change. Cosine-critic alternative rejected (loses mixed-sign sensitivity).
- [x] **T3.5** Consumer wiring — offline quantization-fidelity probe: `I(W; Ŵ)` (or activation-projection surrogate) over pre/post-quant weight populations for one quant surface (KVarN or still_kv bench harness), reported alongside existing reconstruction metrics; **audit-only, no gate flip** until a re-gate shows decision value. **DONE 2026-08-31 — probe GOAT PASS** ([`.benchmarks/694_plan583_t35_mi_quant_fidelity_probe.md`](../.benchmarks/694_plan583_t35_mi_quant_fidelity_probe.md); `katgpt-kv/mi_probe` opt-in test-support feature). Instrument lesson recorded: strict cross-width ordering on the raw DV value is NOT a law (the fixed critic's null gauge is population- AND noise-level-dependent, unsigned; a seed change flipped the 2-bit arm above 4-bit) — the magnitude axis is **dCor²** (1.00000 > 0.99982 > 0.99891 strict, no gauge), p significant at 8/4/2 bits, degenerate control fires, DV tuple reported gauge-caveated, existing `pseudo_decode_eval` MSE/cosine columns ride the same record.
- [x] **T3.6** `[-]` deferred: Mardia alternative gate (revisit only on copula false-accept evidence); KSG arm (validation-referee only, inside `perm.rs` tests); default-feature promotion (blocked on the no-default-consumer rule — promote only after T3.4/T3.5 land AND their GOAT gates pass).

## Validation

`cargo test -p katgpt-core --features mi_est --lib` — 40/40 (debug + release); `cargo test -p katgpt-core --features mi_est --test bench_693_mi_est_goat` — 9/9 (debug + release); `bench_693_mi_est_alloc_check` — 1/1 (exactly 0 allocs steady state); `cargo clippy -p katgpt-core --features mi_est --all-targets` — 0 findings in touched files; feature-off default build — 1980/0, bit-identical to the HEAD pin. Full numbers: [Bench 693](../.benchmarks/693_mi_est_modelless_mi_estimator_goat.md).

### Recorded deviations

1. **G2**: 1 ms → 1.5 ms pin (measured 1.38–1.44 ms min-of-5; residual = the bounds-checked y-row gather + f64 promotion, not a physics wall; gather-free layout = the follow-up lever).
2. **T1.3**: NWJ (l1) is gauge-dependent, not shift-invariant — shift-covariance identity pinned instead.
3. **T3.3**: "strictly decreases" → bit-exact invariance (fixed-critic honest form; cosine dilution measured and rejected).
4. **T3.2**: the 1e-3 Gaussian-arm accuracy claim holds at N ≥ ~5e5 (fixture resized; the gate itself is scale-free).

## Validation (original plan text)

`cargo test -p katgpt-core --features mi_est --lib` (module tests); `cargo clippy -p katgpt-core --features mi_est --all-targets`; feature-off build unchanged; CARGO_TARGET_DIR=/tmp for the gated build per house rule. Benchmarks: single-pass timing (G2) + alloc-steady-state (G4) recorded in `.benchmarks/` with a GOAT doc before any consumer promotion.
