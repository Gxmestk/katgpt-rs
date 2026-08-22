# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1](https://github.com/katopz/katgpt-rs/compare/katgpt-sense-v0.2.0...katgpt-sense-v0.2.1) - 2026-08-22

### Fixed

- clippy heal — crates src slice 2 (speculative/attn/transformer/percepta/types + 16 small-tail crates, 91 files, 411 edits)

## [0.2.0](https://github.com/katopz/katgpt-rs/compare/katgpt-sense-v0.1.1...katgpt-sense-v0.2.0) - 2026-07-31

### Other

- rename per-NPC 'HLA' → 'belief' across katgpt-rs (Issue 195)
- *(sense)* micro-optimizations in octree + schema_centroid
- *(simd)* adopt fast_exp in katgpt-sense reconstruction
- fix all 504 cargo doc warnings in remaining workspace crates
- hoist allocations out of hot loops, eliminate redundant Vec in topo sort

## [0.1.1](https://github.com/katopz/katgpt-rs/compare/katgpt-sense-v0.1.0...katgpt-sense-v0.1.1) - 2026-07-11

### Fixed

- add README.md for 7 published crates — crates.io page was blank

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-sense-v0.1.0) - 2026-07-11

### Added

- *(katgpt-sense)* [**breaking**] promote sense substrate to standalone crate (Plan 338 Phase 3)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- resolve clippy warnings — clone-on-Copy, loop-variable indexing, too_many_arguments allow
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)
- clear all-features clippy errors + warnings across 7 files
- clippy-clean all 13 modified crates (--tests)

### Other

- derive Copy on 200+ primitive-field structs
- repo-wide rustfmt pass (import/module reorder + line wrapping)
- workspace-wide optimization sweep across 13 crates
