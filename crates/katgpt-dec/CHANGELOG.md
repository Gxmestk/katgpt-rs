# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3](https://github.com/katopz/katgpt-rs/compare/katgpt-dec-v0.1.2...katgpt-dec-v0.1.3) - 2026-08-22

### Added

- *(680)* signed_coupling_dynamics — signed-graph Glauber dynamics + the three crowd order parameters (GOAT G1-G4 PASS, opt-in)

### Fixed

- clippy heal — member-crate benches/examples slice (canon/dec/speculative/spectral benches + types example, 6 files, 65 edits)
- clippy heal — crates src slice 1 (spectral/dec/kv/attn-match, 42 files, 515 edits)
- katgpt-dec manual_checked_ops warning (unblocks -D warnings gate)

### Other

- hot-path optimization sweep across crates/ + src/ (39 files)
- *(todo)* correct bogus/stale plan references in 4 inline TODOs

## [0.1.2](https://github.com/katopz/katgpt-rs/compare/katgpt-dec-v0.1.1...katgpt-dec-v0.1.2) - 2026-07-31

### Added

- *(dec)* SE(2)-equivariant lift primitive (Plan 560, Research 457)
- *(dec)* Plan 454 G1b modelless fix — crowding-death unblocks branched morphology, promote grid_3d to default
- *(dec)* Plan 454 T5 — argmax_block_type raw→categorical bridge + T6 gating
- stochastic_birth_death_step NCA growth + SplitMix64 PRNG (plan 454 T4)
- graph_laplacian_grid_3d_into 7-point stencil fast path (plan 454 T3)
- CellComplex::grid_3d 3D cubical constructor + B1/B2/B3 (plan 454 T2)
- GridDims enum + 3D back-compat accessors (plan 454 T1)

### Other

- *(features)* feature-gate-audit source .rs — 17 stale lib.rs surfaces
- *(features)* feature-gate-audit wider sweep — 5 stale comment surfaces
- *(dec)* hoist expm scratch buffers to krylov_expmv_into call site
- *(dec)* hoist matmul scratch out of krylov expm_small Taylor loop
- apply rustfmt import-sorting drift pass
- *(benches)* extract shared counting_allocator! macro for katgpt-dec
- Issue 176 — 5 mechanical test-extraction splits for missed soft-limit files
- sync README + .docs with Plan 454 (grid_3d DEFAULT-ON), Issue 139 (Plan 230 deprecation), Research 442/444
- *(dec)* Plan 454 T9 scope-hallucination correction — CIV_SPECS is HLA goal-direction labels, not cochains
- *(dec)* Plan 454 G4b optimization — fuse 4 field passes + logit gate (123.7% → 55.2% overhead)
- *(triage)* close Issue 155 — impl done via Plan 454, GOAT G1b/G4b fail, grid_3d opt-in
- clippy fixes across workspace
- fix all 504 cargo doc warnings in remaining workspace crates

## [0.1.1](https://github.com/katopz/katgpt-rs/compare/katgpt-dec-v0.1.0...katgpt-dec-v0.1.1) - 2026-07-11

### Fixed

- add README.md for 7 published crates — crates.io page was blank

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-dec-v0.1.0) - 2026-07-11

### Added

- add bench_422_cochain_point_sampler GOAT perf gates (G4+G5 PASS)
- implement cochain point sampler primitive (Plan 422, Research 404)
- *(dec)* Plan 413 — multi-scale V-cycle primitive (htno_v_cycle)
- *(dec)* P407 Phase 3 — sheaf ADMM amplification (T3.1+T3.2+T3.3 all PASS)
- *(dec)* P407 Phase 2 sheaf_admm GOAT gate G1-G6 + promote to default ([#407](https://github.com/katopz/katgpt-rs/pull/407))
- *(dec)* P407 Phase 1 sheaf_admm skeleton — SheafMaps + LocalObjective + sheaf_admm_step ([#407](https://github.com/katopz/katgpt-rs/pull/407))
- Plan 370 Phase 4 — DEC-cochain fusion exploration (T4.1-T4.3)
- *(dec)* Plan 359 Phase 4 — BoM trajectory sampler (near-harmonic perturbation)
- *(dec)* Plan 359 Phase 3 — nonlinear exponential integrator (Duhamel + Gauss-Legendre)
- Plan 359 Phase 5 GOAT — heat_kernel_trajectory PROMOTED to DEFAULT-ON
- Plan 359 Phase 2 — Krylov expmv heat kernel trajectory (online path)
- *(dec)* Plan 359 Phase 1 — DEC heat kernel trajectory predictor (linear path)
- *(dec)* add Clone+Debug derives to CochainField
- motor-gated DEC field primitive (Plan 357, Research 359)
- *(katgpt-dec)* [**breaking**] promote DEC substrate to its own public crate (Issue 007 Phase E Tier 1 #1)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- resolve all remaining clippy warnings — clone-on-Copy, needless_range_loop, ptr_arg, redundant_field_names, too_many_arguments, manual_div_ceil across 6 crates (katgpt-core, katgpt-dec, katgpt-kv, katgpt-pruners, katgpt-spectral, katgpt-speculative, katgpt-attn)
- *(clippy+feat)* katgpt-dec sheaf_admm clippy + fastrand feature forwarding for RngLite
- *(clippy)* resolve all clippy warnings in katgpt-rs
- clean clippy warnings across workspace
- resolve all cargo clippy warnings/errors across crates
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)
- clippy-clean all 13 modified crates (--tests)
- *(clippy)* needless_range_loop + assertions_on_constants in small crates
- *(clippy)* auto-fix batch for katgpt-{core,dec,types,sleep,transformer}

### Other

- *(dec)* line_integral O(P×E) → O(P+|E|) via one-shot vertex-pair lookup
- clippy cleanup across katgpt-core + katgpt-dec (iterator zips, allow attrs, unused-import removal)
- total_cmp conversions in remaining forward/dec/percepta/kv paths
- hot-path micro-optimizations across crates
- repo-wide rustfmt pass (import/module reorder + line wrapping)
- *(dec)* Issue 037 — extract duplicated test helpers into tests/common/
- grid-stencil fast path closes Plan 357 G5 (120µs → 29µs, 4.1× speedup)
- workspace-wide optimization sweep across 13 crates
- SIMD reductions + zero-alloc scratch variants (round 2)
- hot-path optimizations across katgpt-{core,dec,hla,micro-belief,personality,transformer,types}
- *(dec)* migrate eggshell IP out of public katgpt-dec → riir-neuron-db (Issue 008)
- *(katgpt-dec)* spatially-pruned splat for SafetyCochain::from_projectile_threat
