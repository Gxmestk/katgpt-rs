# katgpt-backend

Device inference backends (CPU / Apple Neural Engine / GPU) for the
transformer forward pass — the `InferenceBackend` trait + provider impls
(Issue 413, resolved).

## Overview

Defines the [`InferenceBackend`] trait that decouples the high-level generate
loop from the concrete compute backend (CPU, Apple Neural Engine, GPU). The
default [`CpuBackend`] delegates to [`katgpt_forward::forward()`].

This leaf crate previously lived in the root crate per Issue 033 §C's
circular-dependency argument ("the trait cannot move without its providers; the
providers cannot move without root's forward"). That argument became stale when
`forward` + `ForwardContext` moved to the `katgpt-forward` leaf (Plan 385 /
Issue 007 Phase F, 2026-07-05): every type the backends import now lives in a
leaf crate, so this crate sits above `katgpt-forward` / `katgpt-transformer` /
`katgpt-types` with zero circular deps.

## Key types / modules

- `InferenceBackend` — the trait abstracting a forward-pass backend.
- `CpuBackend` — default impl that delegates to `katgpt_forward::forward()`.
- `auto_backend()` — auto-selects the best available device at runtime.
- `CompileError` — backend weight compilation error type.
- `AneBackend` / `AneError` / `build_conv2d_linear_model_spec` /
  `validate_residency` — Apple Neural Engine backend (Plan 176), gated
  `ane` (CoreML + CoreML protobuf).
- `GpuBackend` — Metal compute pipelines (Plan 176), gated `gpu_inference`.

## Feature flags

`default = []`. macOS-only device backends are opt-in.

| Feature | Default | Description |
|---|---|---|
| `ane` | no | Apple Neural Engine backend (CoreML). Pulls `coreml-native` + `coreml-proto` + `prost`. |
| `gpu_inference` | no | GPU inference backend (Metal compute). Pulls `metal`. |

Both device features are only available on `target_os = "macos"`.

## Dependencies

- `katgpt-forward` — `forward()`, `ForwardContext`.
- `katgpt-transformer` — `TransformerWeights`, `MultiLayerKVCache`.
- `katgpt-types` — `Config`, `kv_dim`.
- `log` — `auto_backend()` logs the selected device.
- `metal` *(optional, macOS)* — GPU compute pipelines.
- `coreml-native` / `coreml-proto` / `prost` *(optional, macOS)* — CoreML
  runtime + protobuf spec builder.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
