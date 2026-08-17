# Issue 660 — Single-submit (graph-fused) Metal forward

**Status:** Resolved (all 5 tasks) — removed per the noise-reduction rule;
resolution record: Bench 661 (tasks 1–4) + Bench 662 (task 5) + this file's
git history.
**Opened:** 2026-08-16
**Owner:** katgpt-rs
**Related Benchmarks:** 656 (MTP Metal batch-width floor)
**Blocks:** any MTP / speculative work on Metal

## Problem

`crates/katgpt-backend/src/gpu.rs` allocates **9 command buffers and issues 9
`wait_until_completed()` CPU syncs per forward pass**, 8 of them *inside* the
per-layer loop (lines 538–737) — roughly `8 × n_layer` CPU↔GPU round trips per
token.

That is precisely the anti-pattern behind llama.cpp's Metal MTP loss. Benchmark
656 identified it as **failure mode 1**, and it is an ARTIFACT rather than a
hardware limit: MLX fuses verification into the same compute graph as the
forward pass and the identical speculative idea "flips from a loss to a large
win" (MTPLX: 1.6× on M4 Mac mini, 2.24× on M5 Max).

Building MTP on this backend as it stands would reproduce llama.cpp's failure.

## Cross-repo substrate (checked 2026-08-16)

This issue targets **katgpt-rs's own raw-Metal backend** (`katgpt-backend/gpu.rs`,
`metal` crate). It does not duplicate riir-ai, but must not ignore it either:

- `riir-ai/crates/riir-gpu/` owns a separate wgpu/CubeCL GPU forward path with
  `forward_flashprefill.rs`, `ternary_deltanet_gpu_forward.rs`,
  `weaver_gpu_dflash.rs`, `weaver_gpu_corrector.rs`.
- **`forward/batched_dispatch.rs` already implements full-sequence batched GPU
  dispatch** behind `gpu_batched_forward` (Plan 363 / Issue 017). The batched
  width-N forward listed as a follow-up below should consume or extend that
  rather than being re-derived here.
- riir-ai's Metal GEMV/matmul benches (599, 611, 619, 656, 666) are prior art for
  kernel-level Metal work on this hardware.

**Measurement protocol is mandatory.** riir-ai adopted an interleaved protocol
after two sign-flipping corrections (1.19× → **0.87×** in Bench 666; 1.24× →
**0.95×** in Issue 658). Any before/after claim here must use it: 2 warmup pairs
discarded + 5 measure pairs, alternating A→B / B→A within each pair, per-pair
ratio as the primary metric. Sequential A-then-B measurement on this box has
demonstrably produced wins that were actually losses.

## Why do it regardless of MTP

Collapsing the per-layer syncs should speed up **ordinary width-1 decode** too.
The dispatch overhead is paid on every token today, speculative or not. This is
the highest-value item on the MTP prerequisite list and the only one that pays
off even if the MTP pivot is abandoned.

## Tasks

- [x] Encode the whole forward into **one** command buffer; one
      `wait_until_completed()` at the end. (2026-08-17)
- [x] Remove per-layer CPU syncs; keep GPU-side ordering via encoder order.
      (2026-08-17 — 9 syncs → 1; the mid-forward `x→xr2` CPU copy became a
      blit-encoder copy, the one real CPU dependency that forced the syncs)
- [x] Verify bit-identical logits vs the current path (G1). (2026-08-17 —
      exact f32 bit patterns over the sequence [0,1,3,7,5], identical)
- [x] Measure width-1 decode tok/s before and after — this is the standalone
      win. (2026-08-17 — interleaved protocol, 5 pairs: median ratio 0.1272
      → **7.86× faster** on micro, 2019-2063 → 255-282 µs/token; Bench 661)
- [x] Only then: batched width-N forward (`InferenceBackend::forward` is
      single-token: `token: usize, pos: usize`). (2026-08-17 — `GpuBackend::
      forward_batch(tokens, pos)`: N tokens in ONE command buffer, per-slot
      buffers for pos/seq_len/embeddings/logits, shared per-token encode body;
      G1 bit-identical vs sequential incl. KV-cache follow-up probe, G2
      1.35×/2.19×/2.37× at widths 2/5/8 — Bench 662)

## Resolution (2026-08-17)

**Tasks 1-4 DONE, GOAT PASS (Bench 661).** The whole forward is now ONE
command buffer with one commit + wait. Encoder order within a command
buffer provides the GPU-side ordering (dispatches within a compute encoder
serialize with memory visibility — the same guarantee the old per-block
waits provided; the code already relied on it for matmul→relu→matmul).
The `n_head` attention heads share one encoder (shared `scores_buf`
ordering preserved by encoder serialization). G1 bit-identical; G2 median
interleaved ratio 0.1272 (7.86×) on the overhead-dominated micro config;
G3 23/23 lib tests + clippy clean.

## Resolution addendum (2026-08-17, task 5)

**Task 5 DONE, GOAT PASS (Bench 662) — the issue is now fully resolved.**
`GpuBackend::forward_batch(tokens, pos)` processes N tokens in ONE command
buffer with a single commit + wait, returning logits for EVERY position
(the MTP verification shape). Per-slot buffers carry each token's
pos/seq_len/embedding/logits (shared buffers cannot work — CPU writes all
land before GPU execution, so every dispatch would see the last value);
the per-token dispatch body is shared verbatim with `forward` via
`GpuFrame::encode_token`. G1: bit-identical vs sequential forwards incl. a
follow-up KV-cache probe. G2 (interleaved): 1.35× / 2.19× / 2.37× at widths
2/5/8 — width-8 runs at 139.4 µs/token, 2.5× below the post-661 single-token
path. The width-2 ceiling is structural (`N(E+W)/(N·E+W)` amortization of
one wait); raising it further needs GEMM-shaped matmuls (M=N — new MSL
kernels, out of scope). The `InferenceBackend` trait is untouched; promote
`forward_batch` to a trait method when an MTP consumer needs
backend-agnostic batching. MTP/speculative work on Metal is unblocked.
