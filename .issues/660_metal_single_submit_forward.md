# Issue 660 — Single-submit (graph-fused) Metal forward

**Status:** Open
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

- [ ] Encode the whole forward into **one** command buffer; one
      `wait_until_completed()` at the end.
- [ ] Remove per-layer CPU syncs; keep GPU-side ordering via encoder order.
- [ ] Verify bit-identical logits vs the current path (G1).
- [ ] Measure width-1 decode tok/s before and after — this is the standalone win.
- [ ] Only then: batched width-N forward (`InferenceBackend::forward` is
      single-token: `token: usize, pos: usize`).
