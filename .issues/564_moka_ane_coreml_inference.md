# Issue 564 — Moka v1 on Apple Neural Engine via CoreML (`katgpt-backend::ane`)

**Filed:** 2026-07-29
**Status:** ❌ CLOSED — negative result, confirmed by direct measurement (2026-07-29, same day). Do not pursue further without new evidence (e.g. a much larger model, or batched inference across many positions).
**Source:** Plan 563 (Go Modelless-vs-Moka Baseline) latency follow-up — user asked to grep harder for unused primitives before assuming CPU SIMD was the ceiling.
**Severity:** Optimization investigation (not a correctness bug). Outcome is negative.
**Scope:** `crates/katgpt-backend/src/ane.rs` (existing CoreML/ANE backend, currently transformer-only) + `crates/katgpt-pruners/src/go/moka_net.rs::ane_probe` (new, scoped probe — stem + one residual block only, NOT the full network).

## Result

**ANE is slower than CPU for this network, not faster — confirmed by direct measurement, not assumption.**

Built the scoped probe exactly as planned below: stem conv + one full non-global residual block (9 layers: 5 convs, 3 ReLU, 1 residual add) as a real CoreML graph, loaded via `coreml-native`, run via `model.predict()`, timed against a matched CPU reference running the **identical** 9 layers (same weights, same input, same primitives — not a proportional estimate from the full network).

| | Latency (9-layer slice) |
|---|---|
| CPU (`conv2d_into` + `simd_dot_f32`, from Plan 563) | **56 µs** |
| ANE (CoreML `predict()`, `ComputeUnits::All`) | **261 µs** |
| Ratio | **ANE is 4.66× slower** |

(First run measured 831 µs for ANE — order of magnitude noisier, likely cold-start/contention on this machine, as the SIMD-tuning work in Plan 563 already documented for this same box. Even the *faster*, likely more representative 261 µs reading is still ~4.7× slower than CPU.)

**Correctness was proven, for what it's worth:** `max_abs_diff = 5.96×10⁻⁸` (f32 machine epsilon) — the CoreML output and CPU reference agree to the bit, confirming the `[out,kh,kw,in]`↔`[out,in,kh,kw]` weight transpose and HWC↔CHW activation transpose are both implemented correctly. The layer-mapping work is not wasted even though the performance verdict is negative — if this is ever revisited (larger model, batched inference), the transpose logic in `ane_probe` is a correct, reusable starting point.

**Why, honestly:** exactly the risk flagged before writing any code — a model this small (105K params) is dominated by CoreML's fixed per-call dispatch/marshalling overhead (FFI bridging, tensor copy in/out, Neural Engine driver handoff), not by compute. That overhead is roughly constant regardless of how many layers are in one `predict()` call, so authoring the remaining ~50 layers into a single larger graph would reduce the overhead-per-layer share but is very unlikely to close a 4.7× gap when the *whole* CPU network only costs ~450 µs to begin with — there isn't enough total compute in this model for a fixed per-call overhead of a couple hundred microseconds to amortize away.

**Decision: do not build the remaining layers.** Per the plan below (written before this result was known), a probe-first negative result stops the investigation rather than continuing to a full 40+-layer graph for a technique already shown to lose at this scale.

## Context

Plan 563 got `GoMokaSearchPlayer` from 119 ms/move to ~2.8 ms/move (~42×) via CPU kernel work (cache-friendly loop order, `katgpt_types::simd::simd_dot_f32`, tuned search depth/beam). Asked directly whether any unused repo primitive could help further (after AHLA/LEO/MUX/plasma-tier were checked and found not applicable), a deeper grep surfaced `crates/katgpt-backend/src/ane.rs` — a real CoreML/Apple Neural Engine execution backend, currently used only for one transformer linear layer (`lm_head`), completely unused for Moka.

This is qualitatively different from every other lever pulled in Plan 563: it is not "swap a function call and measure," it is "author a new multi-layer inference graph against an external protobuf schema, on a new optional dependency, gated to one OS, with a real chance of zero payoff." Filing this as an issue rather than silently spending hours on it.

## What exists today (verified by reading, not assumed)

- `AneBackend::compile()` builds a CoreML `Model` protobuf spec at runtime from in-memory weights (no `.mlmodelc` file), loads it via `coreml_native::Model::load_from_bytes()`, and runs `predict()`. Real hardware dispatch, not a cost model.
- `validate_residency()` times a micro-prediction: ANE < 1 ms vs CPU fallback > 5 ms. If the graph doesn't land on the ANE, CoreML silently falls back to CPU/GPU — the module says so explicitly.
- `build_conv2d_linear_model_spec()` — the only "conv" builder that exists — is actually a **single** 1×1-conv-shaped linear layer (no bias), standing in for the transformer's `lm_head`. It is not a general multi-layer graph builder.
- Layer types already imported and used somewhere in `ane.rs`: `Convolution`, `InnerProduct`, `Add`, `Activation` (ReLU), `Scale`, `Multiply`, `Dot`, `Softmax`. **No `Pooling`/`Reduce` layer type is used anywhere in this file** — unconfirmed whether `coreml-proto` (v0.1.0, `prost`-generated from Apple's `NeuralNetwork.proto`) even exposes one.
- No dependency cycle: `katgpt-backend` depends on `katgpt-forward`/`katgpt-transformer`/`katgpt-types`/`katgpt-core`, not `katgpt-pruners` — `katgpt-pruners` could safely add `katgpt-backend` as a new dependency.
- Gating: `ane = ["dep:coreml-native", "dep:coreml-proto", "dep:prost"]` in `katgpt-backend`, forwarded by the root `ane` feature. `#[cfg(target_os = "macos")]` throughout — this machine qualifies (Darwin/Apple Silicon), but the feature is not portable.

## What Moka's topology needs that doesn't exist yet

~40 layers to hand-assemble (vs. the existing single-layer builder):
- Stem: `Conv(12→32, 3×3, pad 1) + ReLU`
- 12× residual block: `Conv(32→16,1×1)+ReLU → Conv(16→16,3×3,pad1)+ReLU → [3 of 12: GlobalMeanMaxPool(16ch) → Linear(32→8)+ReLU → Linear(8→16) → broadcast-add] → Conv(16→16,3×3,pad1)+ReLU → Conv(16→32,1×1) → Add(residual) → ReLU`
- Policy head: `Conv(32→4,1×1)+ReLU → flatten → Linear(324→82)`
- Value head: `Conv(32→2,1×1)+ReLU → flatten → Linear(162→32)+ReLU → Linear(32→1)+tanh`

The global mean+max pooling step is the one piece with no confirmed CoreML binding in this codebase. If `coreml-proto` has no `Pooling`/`Reduce` layer, it would need to be synthesized from primitives that do exist (e.g. a fixed-weight `InnerProduct` averaging matrix for the mean; max has no obvious linear-algebra substitute and might force a CPU/ANE hybrid split at exactly the 3 blocks that need it).

## Honest risk — this may deliver zero speedup

Two independent failure modes, both real:

1. **Residency failure.** The ANE is tuned for larger batched workloads; a 16–32-channel conv net this small is an unusual shape for it. `validate_residency()` may simply report CPU fallback, in which case all the graph-authoring work produces no latency change (CoreML dispatch/marshalling overhead could even make it *slower* than the current in-process Rust CPU path, which has zero FFI/serialization cost per call).
2. **Missing pooling primitive.** If `coreml-proto` truly has no reduce/pool layer, the 3 global-residual blocks either need a hand-built linear-algebra workaround or a CPU/ANE split mid-graph — either adds real complexity for a piece that's already cheap on CPU (global pooling is O(81×16), negligible next to the convs).

## Plan

1. **Scoped probe before full build.** Compile a minimal graph — stem conv + ONE non-global residual block — and run `validate_residency()` on it. This answers the residency question for a representative slice of the topology without authoring all 40 layers first.
2. **Check `coreml-proto`'s layer enum directly** (not by grepping `ane.rs`, which only shows layers someone happened to use) for `Pooling`/`Reduce` availability.
3. **If residency holds and pooling exists (or is worked around cheaply):** author the full graph, validate against the CPU reference the same way Plan 563 did (`optimized_conv_matches_naive_reference`-style equivalence test, ANE output vs `forward_with_scratch` CPU output, same tolerance).
4. **If residency fails on the probe:** stop here, record the negative result in this issue and in `.docs/06_game_arenas/go_arena.md`, do not build the remaining 38 layers for a graph already known not to help.

## Status

Investigation in progress — probe step running.
