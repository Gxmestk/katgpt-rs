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
