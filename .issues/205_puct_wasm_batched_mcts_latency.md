# Issue 205 — PUCT WASM Batched MCTS Latency Optimization

## Status
**RESOLVED (2026-07-31) — negative result, code kept opt-in.**
See [Benchmark 205](../.benchmarks/205_puct_wasm_batched_mcts_latency.md).

The batched MCTS implementation is correct (12 tests pass, G1 bit-identical
forward pass verified) but the measured gain is **1.09× at K=8** (33.7ms →
30.8ms/move at b50) — far below the estimated 3-5×. The forward pass is
compute-bound, not cache-bound. The K=1 sequential path remains the
default (preserves wasmi parity); batched is opt-in via
`PuctPlayer::with_batch_k`. Full analysis in the benchmark doc.

## Problem

The PUCT WASM build measures **29.8 ms/move** at budget=50 under V8 JIT
(same engine as Chrome). The prior session (commit `bb45f7cc`) eliminated
~800 heap allocations/move (zero-alloc `Board`, stack `neighbors()`,
early-exit `has_liberty`), but measured under V8 JIT the improvement was
within noise: **29.6 ms vs 29.8 ms**.

**Root cause (measured):** the 29.8 ms is **forward-pass-dominated**.
~0.50 ms/pass × 50 nodes = 25 ms (84%). Tree overhead is ~15%,
allocations are ~0.2%. Tree-side micro-optimizations cannot move the
needle — the only path to dramatic improvement is reducing the
forward-pass count or the per-pass cost.

## Hypothesis

**Batched MCTS** (the AlphaZero production technique) evaluates K leaves
in ONE batched forward pass instead of K separate passes. With K=8,
forward-pass count drops 50 → 7. IF the batched forward pass achieves
weight-cache locality (each weight slice loaded once, used K times), the
per-batch cost grows sub-linearly with K — e.g. 2-3× a single pass
instead of 8×. Estimated outcome: **b50 from 29.8 ms → ~8-12 ms/move**.

The win is **contingent on cache locality**. If the batched pass is just
K sequential calls to the existing `conv2d_into`, there is no win — the
total FLOPs are identical. The restructure must reuse weight reads
across the K samples in the inner loop.

## Non-goals (per AGENTS.md global rules)

- **Rayon** — out of scope. WASM is single-threaded in the browser.
  Multi-threaded WASM needs SharedArrayBuffer + `-atomics` + web workers
  (different deployment model).
- **More SIMD** — the dot kernel (`wasm32_simd128_dot_f32`) already uses
  SIMD128 with 4 accumulators × 4 lanes = 16-wide unroll. Saturates the
  SIMD unit per call. Further SIMD wins come only through restructuring
  the conv2d loop nesting (which the batched impl does).

## Tasks

- [x] T1. `forward_batch_with_scratch` — DONE. Processes K feature tensors
      through the network in one call, structured for weight-cache
      locality. Bit-identical to K sequential calls (G1 PASS).
- [x] T2. Batched conv2d primitive — DONE. `conv2d_batched_into` reuses
      the weight slice for each out_channel across all K samples.
- [x] T3. PUCT search loop restructure — DONE.
  - [x] T3.1 Leaf queue — collects up to K leaves during selection.
  - [x] T3.2 Virtual loss — `virtual_loss` field on `PuctNode`;
        `effective_mean_value` penalizes in-flight paths.
  - [x] T3.3 Batched expansion — encodes features for all K leaves, runs
        ONE `forward_batch_with_scratch`.
  - [x] T3.4 Batched backprop — `backprop_clearing_virtual_loss`.
- [x] T4. Correctness gate — DONE. `g1_batched_forward_matches_sequential`
      + 6 batched PUCT tests all PASS.
- [x] T5. Latency gate — DONE (negative result). K=8 gives 1.09× at b50
      (33.7→30.8ms). K=50 gives 1.19× (33.7→28.2ms). Far below the 3-5×
      estimate. See [Benchmark 205](../.benchmarks/205_puct_wasm_batched_mcts_latency.md).
- [x] T6. wasmi parity — DONE. K=1 path is bit-identical to pre-batch code.
      `wasmi_arena_init` takes a `batch_k` param (0/1 = sequential).
- [-] T7. Promotion decision — NOT PROMOTED. The ~10% gain doesn't justify
      the complexity. K=1 stays default; batched is opt-in via
      `PuctPlayer::with_batch_k`. Honest negative result documented.

## Out of scope

- im2col + GEMM restructure of conv2d — a larger refactor that could
  give another 1.5-2× on top of batching, but doubles the
  implementation surface. Track separately if T5 falls short.
- Tree-level parallelism — N/A in WASM (single-threaded).
- Neural network architecture changes — out of scope (the win is in the
  search, not the net).

## References

- Prior session: `bb45f7cc` (zero-alloc board/neighbors/has_liberty).
- Native PUCT parity: `katgpt_pruners::go::moka_net::GoPuctMokaPlayer`
  (Bench 205).
- WASM PUCT port: Issue 204 (resolved + removed).
- Bench harness: `crates/katgpt-moka-wasm/bench/bench_puct.js` +
  `bench_k_sweep.js` (committed in-tree by the hardening follow-up;
  was `/tmp/moka-puct-204/bench_final.js` during the initial Issue 204
  investigation). Build: `./scripts/build-moka-wasm.sh`.
