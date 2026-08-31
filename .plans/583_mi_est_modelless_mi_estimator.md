# Plan 583: `mi_est` — Modelless Mutual-Information Estimator (Fixed-Critic Variational Bounds)

**Status:** Active — Phase 1 not started.
**Date:** 2026-08-31
**Research:** [katgpt-rs/.research/521_MINE_MI_Bound_Modelless_Fixed_Critic.md](../.research/521_MINE_MI_Bound_Modelless_Fixed_Critic.md)
**Source paper:** [arXiv:1801.04062](https://arxiv.org/abs/1801.04062) — MINE (DV bound + EMA); bound taxonomy arXiv:1905.06922; SMILE arXiv:1906.03309; permutation tests (standard)
**Target:** `katgpt-rs/crates/katgpt-core/src/mi/` (new module: `mod.rs`, `dv.rs`, `bounds.rs`, `perm.rs`, `gaussian.rs`, `ib.rs`) + Cargo feature `mi_est` (opt-in)
**Track:** PRIMARY (modelless) — serving-envelope fit: single-pass O(N) evaluation, fits audit/diagnostic cadence and hot-path diagnostics. The trained-critic campaign (riir-train plan 365, SECONDARY) consumes this module's DV core — DRY, no re-implementation.

---

## Goal

Ship a zero-training MI measurement layer for the stack: **DV/NWJ/InfoNCE/JS bound values in nats** over fixed (dot/cosine/frozen-seeded-projection) critics, **leave-one-out** bias control, a **permutation test** (distribution-free, finite-sample-exact p-values; block/circular for tick streams; stratified for conditional dependence), a **K-ladder tightness diagnostic**, and a **Gaussian closed-form arm gated by the shipped `sketched_gaussianity`**. Consumers: third audit axis for the dist-guard family (erank + gaussianity + MI), information-fidelity probe for quantization/compaction surfaces (`kvarn_quality`, `reconstruction_metrics`, still_kv offline gates), and the shared DV core for riir-train plan 365.

**GOAT gate (module-level):** on synthetic Gaussian grids (ρ ∈ 0.1..0.9, d ≤ 64, N = 1e5): |DV̂+LOO − (−½log(1−ρ²))| ≤ 0.05 nats with the quadratic-feature critic; permutation p-values KS-uniform under H0 (|F̂ − U| ≤ 0.02 at the 0.05 quantile over ≥1000 seeds); power ≥ 0.9 on ρ = 0.3, N = 512; **the `Y = X²` control returns ~0 from the Gaussian arm (gate fires) and a significant permutation p** (dependence detected) — non-vacuity pinned. G2: single pass, ≤ 1 ms at N = 1e5 × d = 64 dot-critic (release). G4: zero-alloc steady state (scratch constructed once).

## Phase 1 — Core DV estimator (load-bearing)

### Tasks

- [ ] **T1.1** `mi/mod.rs`: `MiNats` newtype (nats only; bits conversion at the presentation edge), `Critic` enum `#[repr(u8)] { Dot, Cosine, FrozenProj }` with a `score_into(joint, perm_pairs, scratch)` dispatch; `MiScratch` (score buffers + BLAKE3-seeded permutation RNG state), constructed once, zero-alloc after.
- [ ] **T1.2** `mi/dv.rs`: `dv_bound(&scores_joint, &scores_perm, mode) -> DvReport` where `DvReport { l0: f32, loo: f32, l1: f32, spread: f32 }` — λ=0 plug-in bound, leave-one-out logmeanexp (`total-sum − self`, O(1) per i), λ=1 form; max-subtracted log-domain logmeanexp (no overflow, no softmax); permutation drawn from the seeded RNG for run-to-run bit-determinism.
- [ ] **T1.3** Shift-invariance test at bit level: `T → T + c` leaves the DV report unchanged (exact closed form) — free correctness canary.
- [ ] **T1.4** Null-calibration test: on ρ=0 data, measured bias vs N over {1e2..1e6}; pin `bias(N) ≤ C·dof/N` as a recorded curve in the module docs (the "detects MI on the null" trap made visible, not hidden).
- [ ] **T1.5** Gaussian-grid accuracy gate (module-level GOAT G1a) with a deterministic quadratic-feature critic; CI/eval-only (no runtime dep).
- [ ] **T1.6** Feature flag `mi_est = []` in katgpt-core; clippy 0 in touched files; doc-comment the honesty contract: "reports the bound VALUE, not I; the gap is the critic's approximation error — always ship the tuple (value, spread, K-ladder, permutation p), never a bare number."

## Phase 2 — Bound ladder + permutation calibration

### Tasks

- [ ] **T2.1** `mi/bounds.rs`: NWJ (`E_P[T] + 1 − E_Q[e^T]`), InfoNCE-K (`log K − CE`, one-of-K identification over the score matrix), JS — all from ONE `ScoreMatrix` scratch pass; `bounds_all(k_ladder) -> BoundLadder`.
- [ ] **T2.2** K-ladder diagnostic: evaluate at K ∈ {4,16,64,256,1024}; the `Î(K)` saturation gap vs the best available ground truth is exported as `critic_headroom: f32` (how much MI this critic family can even see).
- [ ] **T2.3** `mi/perm.rs`: `PermTest { b, seed } -> PermReport { p, null_hi95 }` wrapping any arm; **circular/block** variant flag for serially-dependent (tick) data; **stratified** variant (shuffle within Z-strata) for conditional dependence I(X;Y|Z); antithetic pairing (σ and σ⁻¹ averaged) for the Q-term.
- [ ] **T2.4** Calibration gates: KS-uniformity of p under H0 over ≥1000 seeds; power ≥ 0.9 at ρ=0.3/N=512; cross-bound coherence on the Gaussian grid (InfoNCE(K) monotone in K; ordering vs DV consistent with theory; all ≤ truth at low-MI regime).
- [ ] **T2.5** Non-vacuity control: `Y = X²` fixture — Gaussian arm refuses (gate fires), permutation p significant, DV report near-zero (bilinear-blind) — the tuple demonstrates WHY the report ships all fields.

## Phase 3 — Gaussian arm + consumers

### Tasks

- [ ] **T3.1** `mi/gaussian.rs`: `CovAccumulator` (Welford means + outer-product sums, O(N·d²) streaming) → `mi_from_cov` returning `Result<MiNats, NotGaussian>`; **gate = `sketched_gaussianity` score > threshold** (consume the existing primitive — no Mardia re-implementation; `NotGaussian` routes callers to perm/bounds arms, never silently swallowed).
- [ ] **T3.2** Gate-is-load-bearing proof: `Y = X²` (and one heavy-tail fixture) MUST return `NotGaussian`; Gaussian fixtures pass to 1e-3 nats vs the analytic value.
- [ ] **T3.3** `mi/ib.rs`: frozen-representation IB ratio `Î(T;Y)/Î(X;T)` diagnostic + Pareto ranker over candidate representations. Directional falsifiability test: injecting independent noise dims into X strictly decreases the ratio at fixed Î(T;Y).
- [ ] **T3.4** Consumer wiring — riir-train `edge_lora_dist_guard` third audit axis (`mi_est` forward feature; population MI between input/target mini-batch projections, amortized cadence like the erank audits). GOAT gate for the axis: the planted-mid-run-collapse fixture (Issue 743 T3 pattern) trips the MI axis at least as early as the erank audit on the collapse-onset regime; if it cannot, the axis is honestly annotated, not shipped.
- [ ] **T3.5** Consumer wiring — offline quantization-fidelity probe: `I(W; Ŵ)` (or activation-projection surrogate) over pre/post-quant weight populations for one quant surface (KVarN or still_kv bench harness), reported alongside existing reconstruction metrics; **audit-only, no gate flip** until a re-gate shows decision value.
- [ ] **T3.6** `[-]` deferred: Mardia alternative gate (revisit only on copula false-accept evidence); KSG arm (validation-referee only, inside `perm.rs` tests); default-feature promotion (blocked on the no-default-consumer rule — promote only after T3.4/T3.5 land AND their GOAT gates pass).

## Validation

`cargo test -p katgpt-core --features mi_est --lib` (module tests); `cargo clippy -p katgpt-core --features mi_est --all-targets`; feature-off build unchanged; CARGO_TARGET_DIR=/tmp for the gated build per house rule. Benchmarks: single-pass timing (G2) + alloc-steady-state (G4) recorded in `.benchmarks/` with a GOAT doc before any consumer promotion.
