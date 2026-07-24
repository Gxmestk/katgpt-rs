# Benchmark 468: HOPE Hilbert-Schmidt Capacity Kernel GOAT Gate (G1–G4)

**Date:** 2026-07-24
**Plan:** [katgpt-rs/.plans/469_hilbert_schmidt_capacity_kernel_primitive.md](../.plans/469_hilbert_schmidt_capacity_kernel_primitive.md) — Phase 4 T4.1
**Research:** [katgpt-rs/.research/454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md](../.research/454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md)
**Source paper:** [arXiv:2607.21366](https://arxiv.org/abs/2607.21366) — Mobahi & Bartlett, HOPE, Google DeepMind, 2026-07-24
**Bench:** `crates/katgpt-core/benches/bench_469_hope_kernel_goat.rs`
**Commits:** `bdd403d2` (bench + zero-alloc fix), promotion in this commit
**Hardware:** Apple Silicon (M-series), release build
**Verdict:** ✅ **ALL GATES PASS** → `hope_capacity` promoted to `default` (Phase 23).

---

## Goal

Validate the four GOAT gates (G1 correctness, G2 latency, G3 no-regression, G4 alloc-free)
for the open primitive layer of HOPE — the Hilbert-Schmidt Capacity Kernel distilled from
arXiv:2607.21366. This is the open half of the HOPE Super-GOAT; the private half (NeuronShard
integration, Super-GOAT G5 compaction-quality gate) lives in riir-neuron-db Plan 321.

---

## Gate Results

### G1 — Correctness sanity ✅ PASS

Re-verifies the 4 load-bearing analytic invariants the latency loop exercises (the full
30-test G1 bit-exact suite lives in `hope::tests`; this bench only guards against silent
fixture changes that would make the latency numbers meaningless):

| Invariant | Measured | Expected |
|---|---|---|
| `relu_self_kernel(1, 0)` (half-wave rectified energy) | 0.5000 | 0.5 |
| Scale invariance `cap(λ=1) − cap(λ=2)` | 0.00e0 | < 1e-4 |
| Cauchy-Schwarz margin `√(K_ii·K_jj) − |K_ij|` | 4.83e-1 | > 0 |
| Optimal scale `s*` on non-degenerate fixture | 1.1014 | > 0, finite |

### G2 — Latency ✅ PASS

Mean ns/call over 10,000 × 256 = 2,560,000 calls per kernel at D=8 (HLA scale, the primary
riir-ai consumer). Sub-ns precision via f64 mean (NOT per-batch u64 median, which rounds
sub-ns kernels to 0). Inner loop of BATCH=256 amortizes the ~40 ns `Instant::now()` floor
on macOS `mach_absolute_time`.

| Kernel | Mean ns/call | Target | Headroom |
|---|---:|---:|---:|
| `relu_self_kernel` | 0.32 | ≤ 10 | 31× under |
| `warped_correlation` | 0.32 | ≤ 50 | 154× under |
| `relu_cross_kernel_approx` | 1.89 | ≤ 80 | 42× under |
| `hope_capacity` | 0.61 | ≤ 80 | 131× under |
| `hope_prune_cost` | 0.93 | ≤ 100 | 108× under |
| `hope_merge_cost` | 20.73 | ≤ 200 | 10× under |
| `optimal_rank1_parent_into_scratch` | 253.77 | ≤ 400 | 1.6× under |
| `hope_greedy_select(32 candidates)` | 5.86 | ≤ 100 | 17× under |
| `hope_block_eviction_cost(2-layer)` | 0.31 | ≤ 50 | 161× under |

All 9 kernels comfortably under target. The two "heaviest" paths are exactly the ones the
paper predicts: `optimal_rank1_parent_into_scratch` (rank-2 eigendecomp + sign resolution
+ 2 cross-kernel evals) and `hope_merge_cost` (2 capacity calls + parent metrics).

### G3 — No-regression ✅ PASS (informational; verified externally)

- `cargo test -p katgpt-core --lib` → **1796 passed; 0 failed** (was 1766 before HOPE;
  the +30 are the HOPE unit tests now in the default surface post-promotion).
- `cargo check -p katgpt-core --all-features` → clean.
- `cargo check -p katgpt-core --no-default-features` → clean (HOPE is `[]`, no deps).

### G4 — Alloc-free hot path ✅ PASS

`CountingAllocator` audit over 100 steady-state calls per kernel (after a 10-call warmup):

| Kernel | Allocs / 100 calls |
|---|---:|
| `relu_self_kernel` | 0 |
| `warped_correlation` | 0 |
| `relu_cross_kernel_approx` | 0 |
| `hope_capacity` | 0 |
| `hope_prune_cost` | 0 |
| `hope_merge_cost` | 0 |
| `hope_greedy_select` | 0 |
| `hope_block_eviction_cost` | 0 |
| `optimal_rank1_parent_into_scratch` | 0 |

All 9 hot-path kernels: **0 allocations**. The owned `optimal_rank1_parent` is intentionally
NOT measured — it's caller-controlled allocation (returns `Rank1Parent` with two `Vec`s);
the zero-alloc contract is on the `_into_scratch` variant.

---

## The zero-alloc bug the bench caught

The first run of this bench FAILED G4: `optimal_rank1_parent_into_scratch` was allocating
**2 `Vec<f32>`s per call** (200 allocs / 100 calls). Root cause: `compute_alignment_objective`
returned an owned `Vec<f32>` for the polarity-comparison snapshot, and it was called twice
(once per ±û polarity).

This was a real contract violation — the function was named `_into_scratch` and documented
as zero-alloc, but the sign-resolution step silently broke that. The bench caught it; the
fix (commit `bdd403d2`) replaces the owned `Vec` with a fixed-size stack buffer of
`RANK1_PARENT_MAX_OUT_DIM = 64` (comfortably above HLA's 8 scalars and style_weights' 64
dims; larger output dims fall back to the owned variant). All 30 `hope::tests` still pass
after the refactor — the fix is behavior-preserving.

**Lesson:** the G4 CountingAllocator audit is load-bearing. Signature analysis ("takes
`&mut [f32]` scratches → must be zero-alloc") is necessary but NOT sufficient — a function
can take scratches AND still allocate internally for a sub-step. Only the runtime audit
catches this. Mirrors the Issue 354 lesson (torn-read invariant held by construction only
after the refactor; the stress test was the empirical complement).

---

## Modelless verification

Per AGENTS.md §"MANDATORY: exhaust modelless paths before deferring to riir-train":

- **`relu_self_kernel`**: closed-form `erf` approximation (A&S 7.1.26) + arithmetic. ✅ modelless.
- **`relu_cross_kernel_approx`**: closed-form Arc-Cosine order-1 + `arccos` + arithmetic. ✅ modelless.
- **`optimal_rank1_parent_*`**: rank-2 closed-form eigendecomp (2×2 characteristic polynomial)
  + sign resolution via direct objective evaluation. ✅ modelless.
- **`hope_capacity` / `hope_prune_cost` / `hope_merge_cost` / `hope_block_eviction_cost`**:
  compose the above + arithmetic. ✅ modelless.
- **`hope_greedy_select`**: linear scan for minimum. ✅ modelless.

No training, no backprop, no gradient descent, no learned parameters anywhere in the
primitive. The only HOPE mechanism that requires training is DEFT's gradient elasticity
(`g_t = E_out ⊙ ∇L_target`) — explicitly out of scope, redirected to riir-train per
Research 454 §3.5.

---

## Promotion

`hope_capacity` promoted to `default` (Phase 23, 2026-07-24) per AGENTS.md rule "GOAT pass →
promote to default":

1. **G1–G4 all PASS** with modelless gain (no quality gate depends on training).
2. **Zero runtime cost unless invoked** — `hope_capacity = []` (no extra deps); the module
   compiles into the lib but no code runs unless a caller invokes a `hope_*` function.
3. **Pattern match** — `manifold_bandit` P370 (G2 FAIL but default-on), `ac_prefix` P313
   (modelless unblock), `poincare_navigator` P449, `karc_forecaster` Phase 22.

The Super-GOAT G5 compaction-quality gate (riir-neuron-db Plan 321 Phase 3) runs against
the AM multi-query baseline (1.5011×) on the Plan 319 wedge workload. **The Super-GOAT
claim is provisional until that gate runs.** If G5 fails, the verdict honestly downgrades
to GOAT — but the open primitive stays default-on regardless (it's the substrate the
Super-GOAT depends on; demoting it would block the very integration that would re-elevate
it).

---

## Run

```bash
# Default features (post-promotion)
cargo bench -p katgpt-core --bench bench_469_hope_kernel_goat -- --nocapture

# Or the direct-binary workaround for the macOS dyld/trustd stall
CARGO_TARGET_DIR=/tmp/hope_goat cargo build --release -p katgpt-core \
    --bench bench_469_hope_kernel_goat
/tmp/hope_goat/release/deps/bench_469_hope_kernel_goat-* --nocapture
```

---

## See also

- [Plan 469](../.plans/469_hilbert_schmidt_capacity_kernel_primitive.md) — the open primitive plan
- [Research 454](../.research/454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md) — Super-GOAT research note
- [riir-neuron-db/.research/302](../../riir-neuron-db/.research/302_HOPE_Shard_Capacity_Metric_SuperGOAT_Guide.md) — private Super-GOAT guide
- [riir-neuron-db/.plans/321](../../riir-neuron-db/.plans/321_hope_shard_capacity_metric_compaction.md) — integration plan (Super-GOAT G5 gate)
- Closest shipped cousins: AM (Plan 233), FAME (R302), Galerkin (R306), Newton-Schulz (Plan 421)
