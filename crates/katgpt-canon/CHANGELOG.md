# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-canon-v0.1.0) - 2026-07-31

### Added

- Bench 562 — katgpt-canon GOAT G1/G2/G4 gates (all 17 PASS)
- katgpt-canon API extension for Issue 385 T1 migration
- Proposal 009 P0 — ship katgpt-canon crate (CanonicalIntent + 3 ModelAdapters)

### Fixed

- unbreak `--all-targets` clippy across the workspace

### Other

- *(katgpt-attn-match, katgpt-canon)* simplify dot_8wide to auto-vectorizing loop — 1.26× faster on NEON
- *(canon)* 8-wide FMA dot product for ProcrustesAdapter project_into
