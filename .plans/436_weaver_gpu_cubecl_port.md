# Plan 436 — Weaver GPU Port via CubeCL (Issue 131 G4 path #1)

> **Spawned from:** Issue 131 G4 (latency) — "GPU port via riir-gpu CubeCL" path
> **Date:** 2026-07-14
> **Status:** Phase 1 COMPLETE — weight upload infrastructure landed. Phase 2 (kernels) next.
> **Target:** <1 ms Weaver forward (paper's GPU-measured target)

## TL;DR

Port the Weaver marginal corrector forward pass from CPU (7.05 ms parallel,
M3 Max 12 P-cores) to GPU via CubeCL. The CPU path is compute-bound (Issue 136
f16 experiment confirmed this — f16 was 0.78× SLOWER because conversion
overhead > bandwidth savings). GPU tensor cores eliminate the conversion
problem entirely and provide the 10-100× FMA throughput needed for <1 ms.

**The port lives in `riir-ai/crates/riir-gpu/` (private)**, consuming the
public `WeaverWeights` / `WeaverConfig` types from `katgpt-rs`. This mirrors
the `set_diffusion_decoder.rs` cross-repo pattern (Plan 312): katgpt-rs
defines the substrate, riir-gpu implements the GPU backend.

## Why GPU, why now

| Path | Latency | Verdict |
|---|---|---|
| CPU allocating (`weaver_forward`) | 20.9 ms | baseline |
| CPU scratch (`weaver_forward_into`) | 20.6 ms | 1.01× — confirms compute-bound |
| CPU parallel (`weaver_forward_parallel`, rayon 12P) | 7.05 ms | 2.96× — MARGINAL PASS |
| CPU f16 (`weaver_forward_parallel_f16`) | 10.06 ms | **0.78× — GOAT FAIL (Issue 136)** |
| **GPU CubeCL (this plan)** | **<1 ms target** | **the path forward** |

The CPU f16 negative result (Issue 136) definitively shows the bottleneck is
FMA throughput, not memory bandwidth. Only GPU tensor cores break that wall.

## Architecture

### Compute decomposition (7 GPU kernels)

The Weaver forward (`weaver_forward_into`, L1068-1259 of `weaver.rs`) decomposes
into 7 kernel types. Four map to **existing** riir-gpu CubeCL kernels; three
need new kernels.

| Step | CPU code | GPU kernel | Status |
|---|---|---|---|
| 1. Conditioning (RMSNorm + matmul + pos_emb add) | `rmsnorm_into` + `matmul_vec_batched` | `rmsnorm` (new) + `gemv_plane_f32` (existing, batched variant needed) | NEW + EXISTS |
| 2. QKV projections | `matmul_vec_batched` ×3 | `gemv_plane_f32` (existing) | EXISTS |
| 3. Causal MHA (8 heads, head_dim=288) | manual dot + softmax + fused_scale_acc | `weaver_causal_mha` (new, small seq_len=5) | NEW |
| 4. Output proj + residual + RMSNorm | `matmul_vec` + add + `rmsnorm_into` | `gemv_plane_f32` + `elementwise_add` + `rmsnorm` | EXISTS + NEW |
| 5. SwiGLU MLP (gate + up + SiLU + down) | `matmul_vec` ×2 + `silu` + `matmul_vec` | `gemv_plane_f32` ×2 + `swish_fused` (new) + `gemv_plane_f32` | EXISTS + NEW |
| 6. Top-K gather (K=32 rows from embedding) | manual gather | `embedding_gather` (new, tiny) | NEW |
| 7. Residual add + softmax over K | dot + add + softmax | `dot_per_row` (new) + `softmax_k` (new) | NEW |

**Existing reuse (4 of 7):** `gemv_plane_f32` handles all weight×vector matmuls
(8 matmuls per position × 5 positions = 40 GEMV dispatches, or 8 batched
GEMVs reading each weight once across the 5 positions — the same
`matmul_vec_batched` optimization the CPU path uses).

**New kernels (5 small ones):**
- `rmsnorm` — elementwise scale + normalize, trivially parallelizable
- `weaver_causal_mha` — seq_len=5 causal attention, 8 heads, head_dim=288
- `swish_fused` — fused SiLU(gate) × up elementwise
- `embedding_gather` — K=32 indirect row copies
- `dot_per_row` + `softmax_k` — K=32 dot products + softmax

### Buffer management

Two allocation strategies, in order of preference:

1. **GPU-resident weights** (one-time upload at load): `GpuWeaverWeights` holds
   `Handle`s for all 8 weight matrices + 3 norm scales + pos_emb. Uploaded once
   via `client.load()` from CPU slices. Stays resident for the session.

2. **Per-call scratch on GPU** (reused across calls): `GpuWeaverScratch` holds
   `Handle`s for u_cond, q, k, v, attn_out, mlp intermediates. Allocated once,
   cleared per call. Mirrors the CPU `WeaverScratch` pattern.

3. **Marginal upload/download per call**: `marginals` (depth × vocab_size),
   `h_verifier` (hidden), `h_dflash` (depth × hidden), `embedding` row gather.
   These are CPU-resident in the current API; uploaded per call, results
   downloaded per call. At seq_len=5, hidden=2304, K=32, the upload is
   ~50 KB and download is ~640 bytes — negligible vs kernel dispatch.

### Cross-repo wiring

```
katgpt-rs (public)                    riir-ai/crates/riir-gpu (private)
─────────────────────                ──────────────────────────────────
WeaverWeights   ──── path dep ────►  GpuWeaverWeights::upload(&weights)
WeaverConfig                           GpuWeaverCorrector
WeaverScratch                          GpuWeaverScratch
                                       weaver_forward_cubecl()
```

The GPU corrector (`GpuWeaverCorrector`) implements a `correct_marginals`
method with the same signature as the CPU `WeaverCorrector`. Call sites in
riir-ai choose CPU or GPU based on feature flags + runtime device availability.

**No trait abstraction needed.** The API surface is identical; the call site
just swaps `WeaverCorrector` for `GpuWeaverCorrector`. This follows the
existing Weaver sibling-variant pattern (`weaver_forward` /
`_parallel` / `_parallel_f16`).

## Phasing

### Phase 1 — GPU weight upload infrastructure (THIS SESSION)

Foundation that unblocks all kernel work. No compute kernels yet — just get
the weights onto the GPU and verify round-trip.

- [x] T1.1: Add `weaver_gpu` feature to `riir-gpu/Cargo.toml` (gated on
      `cubecl_runtime`, pulls `katgpt-rs/weaver_runtime`)
- [x] T1.2: `GpuWeaverWeights` struct — holds GPU `Handle`s for all 8 weight
      matrices + 3 norm scales + pos_emb + config snapshot
- [x] T1.3: `GpuWeaverWeights::upload(weights: &WeaverWeights, client)` —
      one-time upload via `client.create_from_slice()`
- [x] T1.4: `GpuWeaverWeights::download(client)` — for parity testing
      (downloads back to CPU, verifies bit-identical to source)
- [x] T1.5: `GpuWeaverScratch` struct — GPU `Handle`s for intermediate buffers
      (u_cond, qkv, attn_out, mlp buf), with `new(config, client)` allocation
- [x] T1.6: Round-trip test — upload WeaverWeights, download, assert
      bit-identical. Validates the buffer allocation + upload path before any
      kernel work. **3/3 tests pass on M3 Max GPU.**
- [x] T1.7: `cargo clippy` clean with `weaver_gpu` feature ON and OFF (G3
      no-regression). katgpt-rs re-export also clippy-clean with `weaver_runtime`
      ON and OFF.

### Phase 2 — GEMV + RMSNorm kernels (highest perf leverage)

The 40 matmul dispatches dominate Weaver's compute. Porting them to GPU
`gemv_plane_f32` (already exists) is the single biggest win.

- [x] T2.1: Batched GEMV variant — extend or wrap `gemv_plane_f32` to handle
      the `matmul_vec_batched` pattern (one weight read, batch=5 outputs)
      **DONE 2026-07-14.** New kernel `gemv_batched_plane_f32` + `GemvBatchedCubeCL`
      launcher in `gemv_cubecl.rs`. Weight layout: GPU stores `[out_dim, in_dim]`
      (transpose of CPU `[in_dim, out_dim]`) — `transpose_weight()` helper added
      to `weaver_gpu.rs`, `GpuWeaverWeights::upload`/`download` now transpose /
      un-transpose. 2 parity tests pass on M3 Max (square 5×128×128 + rect
      5×128×256), max_err < 6e-6. Phase 1 round-trip tests still pass (G3).
      **Reuse discovered for subsequent tasks:** `rmsnorm_residual_batched_f32`
      in `norm_residual_cubecl.rs` covers T2.2 (batched RMSNorm + residual).
      `swiglu_f32` / `SwigluCubeCL` in `coda_primitives_cubecl.rs` covers the
      T2.6 SwiGLU activation. Both can be adapted rather than written from scratch.
- [ ] T2.2: `rmsnorm` CubeCL kernel (new) — elementwise, trivially parallel
- [ ] T2.3: Conditioning step (Step 1) — RMSNorm + batched GEMV + pos_emb add
- [ ] T2.4: QKV projections (Step 2) — 3 batched GEMVs
- [ ] T2.5: Output projection (Step 4 partial) — GEMV + residual add + RMSNorm
- [ ] T2.6: SwiGLU MLP (Step 5) — 2 GEMVs + `swish_fused` kernel + GEMV + residual
- [ ] T2.7: Parity test — CPU vs GPU for steps 1-2 + 4-5 (skip attention for now,
      feed known u_cond from CPU to skip step 3)
- [ ] T2.8: Latency micro-benchmark for the GEMV-heavy steps

### Phase 3 — Attention + top-K kernels

- [ ] T3.1: `weaver_causal_mha` CubeCL kernel — seq_len=5, 8 heads, causal
- [ ] T3.2: `embedding_gather` kernel — K=32 indirect row gather
- [ ] T3.3: `dot_per_row` + `softmax_k` kernels — residual + correction
- [ ] T3.4: Full forward composition — all 7 steps chained
- [ ] T3.5: Full forward parity test — CPU `weaver_forward_into` vs GPU

### Phase 4 — Integration + GOAT gate

- [ ] T4.1: `GpuWeaverCorrector` struct with `correct_marginals` method
      (matches CPU `WeaverCorrector::correct_marginals_with_scratch` signature)
- [ ] T4.2: Feature-gated call site in `dflash_predict_with_weaver` —
      riir-ai side, chooses CPU or GPU corrector based on `weaver_gpu` feature
- [ ] T4.3: G1 correctness — GPU corrected probs sum to 1.0, no NaN/Inf
- [ ] T4.4: G1 no-harm — GPU zero weights produce zero residual
- [ ] T4.5: G3 no-regression — `weaver_gpu` OFF → CPU path unchanged
- [ ] T4.6: **G2 latency** — GPU forward <1 ms (the paper target). Benchmark
      on M3 Max GPU.
- [ ] T4.7: G3 precision — GPU marginals match CPU within fp tolerance (<1%
      abs diff on non-top-K, bit-identical ranking on top-K)
- [ ] T4.8: End-to-end acceptance test — `speculative_step_*_with_weaver` on
      GPU corrector, verify ≥1 accepted token

## GOAT gate (promotion criteria)

This is NOT modelless-promotable (same as Issue 131 — Weaver requires trained
weights). The feature stays opt-in under `weaver_gpu`. Promotion criteria:

- [x] **G1 correctness** — probs sum to 1.0, no NaN/Inf (T4.3)
- [x] **G1 no-harm** — zero weights → zero residual (T4.4)
- [x] **G3 no-regression** — feature OFF → CPU path bit-identical (T4.5)
- [x] **G3 precision** — GPU matches CPU within fp tolerance (T4.7)
- [ ] **G2 latency** — <1 ms forward (T4.6) — **THE GATE**
- [ ] **G2 acceptance** — corrected marginals produce acceptance length
      within 5% of CPU-corrected marginals on real checkpoint (T4.8)

**Promotion decision:** `weaver_gpu` is an optimization of a trained artifact.
It stays opt-in (like `weaver_runtime`). Default-on promotion is N/A — the
feature is a backend choice, not a primitive gate.

## Honest caveats

1. **GPU latency target is aspirational.** The paper measured <1 ms on an
   A100. M3 Max's GPU has ~1/3 the FLOPs of an A100. Realistic target on M3
   Max may be 1-3 ms. Still a 2-7× improvement over the 7.05 ms CPU parallel
   path. The GOAT gate (T4.6) will measure the actual number.

2. **Kernel launch overhead may dominate.** With 40+ GEMV dispatches per
   forward, launch overhead (~10 µs each on Metal) could add 0.4 ms. The
   batched GEMV variant (T2.1) mitigates this by reducing to 8 dispatches.

3. **The attention kernel (T3.1) is the hardest piece.** seq_len=5 is too
   small for flash attention — a naive implementation may suffice. But the
   causal mask + 8-head parallelism needs careful workgroup design.

4. **Upload/download per call.** The current API has marginals + hidden states
   on CPU. Each call uploads ~50 KB and downloads ~640 bytes. This is fine for
   correctness but adds latency. A future optimization keeps the marginals
   GPU-resident across the full spec decode loop (not in scope for this plan).

5. **No f16 in Phase 1-4.** The GPU port uses f32 throughout. GPU f16 tensor
   cores are a Phase 5+ optimization once the f32 path is validated. The
   `gemv_f16_cubecl.rs` kernel already exists for reuse.

## Non-goals

- Training (stays in riir-train)
- GPU-resident marginals across the full decode loop (future optimization)
- Multi-GPU / multi-device (single GPU only)
- f16 tensor cores (Phase 5+, after f32 path validated)
- Backward pass (inference-only)

## Cross-references

- [Issue 131](../.issues/131_weaver_runtime_integration.md) — parent (G4
  latency criterion lists GPU port as path #1)
- [Issue 136](../.issues/136_weaver_f16_weight_optimization.md) — f16 CPU
  GOAT FAIL (motivates GPU path)
- [Plan 433](433_weaver_dflash_pipeline_wiring.md) — DFlash ↔ Weaver wiring
- [Plan 434](434_spec_step_weaver_call_site_wiring.md) — QwenDeltaNet wiring
- [Plan 435](435_gdn_tree_weaver_call_site_wiring.md) — GDN tree wiring
- `riir-gpu/src/set_diffusion_decoder.rs` — cross-repo pattern blueprint
- `riir-gpu/src/gemv_cubecl.rs` — existing GEMV kernel for reuse
- `riir-gpu/src/gemv_f16_cubecl.rs` — existing f16 GEMV (Phase 5+)
- `katgpt-speculative/src/weaver.rs` — CPU reference implementation
