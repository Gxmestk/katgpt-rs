# Issue 567 — CNN→Transformer Code-Path Unification (Path A: Shared NN Primitives)

> **Filed:** 2026-08-01
> **Origin:** User direction after Issue 566 (CNN→Transformer Latent Bridge)
> closed negative on both phases. The user clarified the original task was
> "convert CNN to transformer" = **unify the code path** (option #2), not
> "bridge Moka features into Gemma 2" (what Research 464 formalized).
> **Type:** Refactor (extract shared NN primitives)
> **Status:** DONE — G1 bit-identical PASS, wasm32 verified.

## Context

Before this issue, the CNN engine (`katgpt-moka-wasm/src/moka.rs`) and the
transformer engine (`katgpt-forward`/`katgpt-transformer`) shared exactly
ONE primitive: `simd_dot_f32` from `katgpt-types`. Everything else was
duplicated — Moka had its own `conv2d_into`, `linear_into`,
`global_mean_max_into`, `relu_inplace`; the transformer had its own
`matmul`/`rmsnorm` paths.

This issue is **Path A** of the unification: extract the shared NN
operations into a new `katgpt-nn` crate so both engines import from the
same source. Path B (unified `forward()` dispatch with `Layer` enum) builds
on top of Path A's primitives later.

## Design rule (for Path B compatibility)

All functions take `&mut [f32]` scratch buffers as parameters (caller-
allocated), NOT owning their own `ForwardContext`. This keeps them
composable for a future unified `Layer` enum dispatch where a single
`forward()` routes to these primitives per layer type.

## What was extracted

| Function | Purpose | Former location |
|---|---|---|
| `dot_lanes` | Bias-aware dot product (`init + simd_dot_f32`) | `moka.rs` L155 |
| `conv2d_into` | 2D conv (k×k, zero-padded, HWC layout) | `moka.rs` L161 |
| `conv2d_batched_into` | Batched 2D conv (K samples, weight reuse) | `moka.rs` L246 |
| `linear_into` | Bias-bearing fully-connected layer | `moka.rs` L219 |
| `linear_batched_into` | Batched linear (K samples, weight reuse) | `moka.rs` L326 |
| `global_mean_max_into` | Spatial mean+max pool | `moka.rs` L386 |
| `global_mean_max_batched_into` | Batched spatial pool | `moka.rs` L347 |
| `relu_inplace` | In-place ReLU activation | `moka.rs` L378 |

## What stayed in `katgpt-moka-wasm`

- `MokaWeights`, `ResidualBlock`, `GlobalBranch`, `Wb` — Moka-specific weight types
- `MokaScratch`, `MokaBatchScratch` — Moka-specific scratch buffers
- `forward_with_scratch`, `forward_corrected_with_scratch` — Moka's forward graph
- Weight loading (manifest parsing, dequantization)
- `moka_int8.rs` int8 kernels (own SDOT/vmull/wasm-extmul paths)
- `research.rs` corrected/collecting/scalar variants (Moka-specific)

## Migration approach

- `moka.rs` imports the shared primitives via `use katgpt_nn::{conv2d_into, linear_into, ...}`
- `relu_inplace` + `global_mean_max_into` re-exported via `pub(crate) use katgpt_nn::{relu_inplace, global_mean_max_into}` so sibling modules (`moka_int8.rs`, `research.rs`) keep their `crate::moka::relu_inplace` import paths unchanged
- Zero changes to `moka_int8.rs` or `research.rs`

## Tasks

- [x] **T1** — Create `katgpt-nn` crate (Cargo.toml + src/lib.rs + 5 unit tests)
- [x] **T2** — Add to workspace members
- [x] **T3** — Add `katgpt-nn` dep to `katgpt-moka-wasm/Cargo.toml`
- [x] **T4** — Extract 8 functions from `moka.rs` to `katgpt-nn`
- [x] **T5** — G1 bit-identical: all 17 Moka tests PASS (including
      `g1_int8_matches_f32_baseline`, `g1_batched_forward_matches_sequential`)
- [x] **T6** — wasm32 verified: both `katgpt-nn` + `katgpt-moka-wasm` compile clean
- [x] **T7** — `research` feature compiles clean
- [x] **T8** — clippy clean (zero new warnings)

## GOAT Gate

- **G1** (correctness): **PASS** — 17/17 Moka tests pass. The extraction is
  a pure code move; no computation changed. `g1_int8_matches_f32_baseline`
  + `g1_batched_forward_matches_sequential` verify bit-identical output.
- **G2** (perf): **N/A** — identical compiled code, different crate. No
  perf change possible.
- **G3** (no-regression): **PASS** — default build unaffected, `research`
  feature compiles clean, wasm32 target compiles clean.
- **G4** (alloc-free): **PASS** — the extracted functions don't allocate.

## Next steps (Path B — separate issue)

- Define `Layer` enum: `Conv(ConvConfig)` | `Attention(AttnConfig)` | `Mlp(MlpConfig)` | `Pool` | `Activation(ActKind)`
- Define `Model` struct holding a sequence of layers + weights
- Unified `forward(model, input) -> output` that dispatches per layer
- Both Moka and Gemma expressed as `Model` instances

Path A established the primitive API surface. Path B builds the dispatch
layer on top of it.
