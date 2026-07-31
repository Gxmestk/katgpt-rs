# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/katopz/katgpt-rs/compare/katgpt-types-v0.1.1...katgpt-types-v0.2.0) - 2026-07-31

### Added

- *(types)* f16xf16->f32 SIMD GEMV kernel via ARMv8.2-A FP16 widening FMA (Issue 201)
- *(katgpt-forward)* add f16 weight quantization forward path (Issue 200, G2 FAIL — opt-in only)
- binary plasma tier (Issue 145 Phase 0+1, GOAT gate PASSED)
- add loop_stability_fix feature for inter-loop RMSNorm in forward_looped
- add gated_mlp (SwiGLU) feature flag for aHLA MLP training (Issue 377)
- *(lt2)* modelless loop-stable residual gate (Plan 483 T2.1+T2.2)

### Fixed

- *(clippy)* mark raw-pointer-deref fn unsafe + fix CLR call sites missed by signature change
- *(katgpt-types)* allow unused_unsafe for __cpuid on Rust 1.93+
- *(katgpt-types)* gate SimdLevel import behind target_arch for wasm32
- deprecate Plan 230 ShardEmbedding/JlProjectionMatrix (Issue 139)
- Issue 145 Phase 3 — binary_plasma feature-combination bug + consumer migration

### Other

- rename per-NPC 'HLA' → 'belief' across katgpt-rs (Issue 195)
- *(clippy)* use RangeInclusive::contains for [0,1] range assertions
- *(simd)* add simd_tanh_inplace (NEON/AVX2/WASM Padé [2/2] kernel)
- *(simd)* add fast_tanh (Padé [2/2]) + adopt in activation hot paths
- *(simd)* add fast_exp + adopt SIMD softmax in attention hot paths
- *(simd)* use Cephes exp in entropy_f32 (DDTree hot path)
- *(cgsp)* delegate cgsp::types::sigmoid to fast_sigmoid (Cephes)
- *(simd)* use Cephes polynomial exp in fast_sigmoid (1.7× vs libm)
- *(features)* feature-gate-audit wider sweep — 5 stale comment surfaces
- apply rustfmt import-sorting drift pass
- *(config)* document RiM slot width M≥5 floor for reasoning tasks (Issue 156 T1)
- fix stale .rs doc-comments after numbering collision renumber (Issue 153)
- Issue 136 — Weaver f16 weight path (GOAT FAIL, honest negative result)
- fix all 504 cargo doc warnings in remaining workspace crates
- fix stale issue references (002, 024, 037, 043) in source code

## [0.1.1](https://github.com/katopz/katgpt-rs/compare/katgpt-types-v0.1.0...katgpt-types-v0.1.1) - 2026-07-11

### Fixed

- add README.md for 7 published crates — crates.io page was blank

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-types-v0.1.0) - 2026-07-11

### Added

- Plan 410 3A.2 — Clone derive on LoraAdapter
- *(types,pruners)* Issue 019 Phase C.1+C.3 upstream promotions
- *(core)* 004 adaptive causal calibration open primitive — cheap-proxy escalate (Phase 1+2)
- *(calibration)* Plan 358 Phase 4 — RTPurbo wiring + promote/demote (causal head-importance)
- *(katgpt-types)* co-extract MerkleOctree + MerkleProof to leaf (Plan 338 Phase 2.5)
- *(katgpt-types)* co-extract TemporalDerivativeKernel<N> to leaf (Plan 338 Phase 2)
- *(katgpt-types)* co-extract ScaleBoundary to leaf (Plan 338 Phase 1)
- *(katgpt-micro-belief)* [**breaking**] promote micro-belief kernel to its own public crate (Issue 007 Phase E Tier 1 #3)
- *(katgpt-types)* [**breaking**] promote types+simd substrate to its own public crate (Issue 007 Phase E Tier 1 #2)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- resolve all cargo clippy warnings/errors across crates
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)
- 2 remaining test-surface clippy warnings (useless_vec + needless_range_loop)
- scrub private code-symbol leaks from public doc comments (issue 360 class A)
- *(docs+hla)* resolve Issue 009 — ahla_step math divergence was a bug, not a variant
- *(clippy)* needless_range_loop + assertions_on_constants in small crates

### Other

- derive Copy on 200+ primitive-field structs
- repo-wide rustfmt pass (import/module reorder + line wrapping)
- workspace-wide optimization sweep across 13 crates
- extract katgpt-kv + katgpt-spectral crates (Issue 015)
- *(simd)* simd_l_inf_distance_f32 + blocked argmax_pair (riir-neuron-db Issue 003)
- hot-path optimizations across katgpt-{core,dec,hla,micro-belief,personality,transformer,types}
