# Benchmark 205 — PUCT WASM Batched MCTS Latency

**Date:** 2026-07-31
**Issue:** [.issues/205](../.issues/205_puct_wasm_batched_mcts_latency.md)
**Status:** Negative result (gain too small to justify promotion)

## TL;DR

Batched MCTS (K=8) gives **1.09× speedup** (33.7ms → 30.8ms/move at b50)
under V8 JIT — far below the estimated 3-5×. The forward pass is
**compute-bound, not cache-bound**: the Moka net (~100KB weights) fits in
L2 cache, so sequential passes already benefit from cache residency.
The SIMD dot kernel saturates the FPU per call, and batching K samples
through the same weight slice doesn't reduce the total FLOP count.

## Setup

- **Hardware:** Apple Silicon (M-series)
- **Runtime:** Node.js v24.10.0 (V8 JIT — same engine as Chrome)
- **Build:** `./scripts/build-moka-wasm.sh` (encodes the full SIMD pipeline:
  `RUSTFLAGS='-C target-feature=+simd128'` + `cargo build --target
  wasm32-unknown-unknown` + `wasm-bindgen --target nodejs` + `wasm-opt -Oz
  --enable-simd`. Without the SIMD flags the scalar fallback runs ~16×
  slower — ~500ms/move instead of ~30ms. The script exists because this
  regression was diagnosed the hard way during Issue 205.)
- **Method:** setup-subtracted (reset+replay outside timing), N=10 samples
  per config, mid-game position (8 stones played).
- **Harness:** `crates/katgpt-moka-wasm/bench/bench_puct.js` (sequential vs
  batched at b50/b100/b200) + `bench_k_sweep.js` (K=1..50 at b50). Both
  load the wasm-bindgen nodejs output via `require()` + V8 JIT.
- **Harness:** `crates/katgpt-moka-wasm/bench/bench_puct.js` +
  `bench_k_sweep.js` (committed; was `/tmp/moka-puct-205/` during the
  initial investigation, moved in-tree by the hardening follow-up).
  Build with `./scripts/build-moka-wasm.sh` before running.

## K sweep at b50

| K (batch size) | ms/move | Speedup vs K=1 |
|---|---|---|
| 1 (sequential) | 33.7 | 1.00× |
| 2 | 32.4 | 1.04× |
| 4 | 32.0 | 1.05× |
| 8 | 30.8 | 1.09× |
| 16 | 29.9 | 1.13× |
| 25 | 28.9 | 1.17× |
| 50 (single batch) | 28.2 | 1.19× |

Monotonic improvement with diminishing returns. The gain flattens —
even K=50 (one giant batch covering the entire budget) only reaches 1.19×.

## Why the estimate was wrong

The original hypothesis: "the Moka trunk block is ~100KB, spills L1
(32KB), sequential passes reload from L2 K times, batched loads once per
out_channel." This was **wrong** because:

1. **L2 latency is ~10ns, FLOP latency is ~1ns.** Even if sequential
   passes reload weights from L2 every pass, the reload cost (~1µs for
   100KB at 10ns/line) is tiny compared to the compute cost (~500µs for
   3M MACs at the SIMD throughput). The forward pass is FPU-bound.
2. **The SIMD dot kernel already saturates the 128-bit FPU.** `simd_dot_f32`
   uses 4 independent accumulators × 4 lanes = 16-wide unroll. The FPU
   can't go faster by restructuring the loop nesting — it's already at
   peak throughput per cycle.
3. **Total FLOPs are identical.** Batching K samples through one conv2d
   call does the same K × positions × out_channels × patch_len MACs as
   K sequential calls. Without reducing FLOPs (via im2col + GEMM with
   weight reuse across the batch dimension), there's no compute win.

## What WOULD help (not pursued)

1. **im2col + batched GEMM** — restructure conv2d as a matmul where the
   batch dimension multiplies the same weight matrix. This DOES reduce
   weight reloads (each weight row used K times in one GEMM call). But
   it's a substantial refactor (~500 LOC) with memory tradeoffs (im2col
   expansion costs memory). Estimated 1.5-2× on top of batching.
2. **Smaller/faster network** — the real bottleneck is the net itself
   (12 residual blocks, 32 channels). A distilled/smaller net would cut
   FLOPs proportionally. Out of scope (changes the model, not the search).
3. **GPU/WebGPU** — the only path to dramatic speedup. Out of scope for
   this crate (CPU-only WASM by design).

## Correctness verification

- **G1 (batched forward ≈ sequential):** `g1_batched_forward_matches_sequential`
  PASSES — max policy diff 0.0000e0, max value diff 0.0000e0 across 8
  random boards. The batched forward pass is bit-identical (within f32
  epsilon) to K sequential forward passes.
- **G1 (PUCT determinism):** `batched_puct_search_is_deterministic_given_same_board`
  PASSES — same board → same move across two runs.
- **G1 (virtual loss diversity):** `batched_puct_explores_diverse_leaves_via_virtual_loss`
  PASSES — at K=8, budget=8, ≥2 root children receive visits (virtual
  loss drives diverse exploration within a batch).
- **G3 (no-regression):** 12/12 lib tests pass (6 pre-existing + 6 new
  batched tests). Clippy clean.

## Decision

**Do NOT promote batched MCTS to default.** The ~10% gain at K=8 doesn't
justify the complexity (virtual loss, leaf queueing, root-first expansion,
terminal-leaf handling, partial-batch edge cases). The K=1 sequential
path remains the default:

- Preserves the wasmi parity guarantee (bit-identical move choices).
- Simpler code (no virtual loss, no batch scratch, no leaf queue).
- The ~3ms savings (33.7 → 30.8) is within run-to-run variance for most
  real workloads.

The batched code stays in-tree as opt-in via `PuctPlayer::with_batch_k(budget, c_puct, top_k, batch_k)`.
Consumers who need the extra 10-19% (e.g., a real-time deployment at the
edge of the tick budget) can opt in; everyone else gets the simpler path.

## Reproducibility

To reproduce these numbers:

```bash
./scripts/build-moka-wasm.sh                                    # build with SIMD
node crates/katgpt-moka-wasm/bench/bench_puct.js                 # b50/b100/b200, seq vs batched
node crates/katgpt-moka-wasm/bench/bench_k_sweep.js              # K=1..50 at b50
```

Fresh re-run on this machine (2026-07-31, after the build script landed):

| Config | This doc (prior session) | Fresh re-run |
|---|---|---|
| b50 K=1 | 33.7 ms | 33.8 ms |
| b50 K=8 | 30.8 ms | 30.7 ms |
| b50 K=50 | 28.2 ms | 28.1 ms |

All within run-to-run noise — the finding reproduces cleanly.
