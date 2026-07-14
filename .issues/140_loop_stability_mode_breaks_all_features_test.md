# Issue 140: `loop_stability_mode` field breaks `bench_ldt_lattice_deduction` test under `--all-features`

**Discovered:** 2026-07-14 (during Plan 438 Phase 5 T5.2 workspace-wide clippy sweep)
**Severity:** Medium — blocks `cargo clippy --workspace --all-features`; does not affect default build
**Root cause commit:** Plan 428 Phase 2 (`loop_stability_fix` feature) — see `.plans/428_loop_stability_poc.md`

## Symptom

```
$ cargo clippy --workspace --all-features --all-targets

error[E0063]: missing field `loop_stability_mode` in initializer of `katgpt_rs::types::Config`
  --> tests/bench_ldt_lattice_deduction.rs:33:9
   |
33 |         Config {
   |         ^^^^^^ missing `loop_stability_mode`
```

## Root Cause

Plan 428 added the `loop_stability_mode` field to `Config` in
`crates/katgpt-types/src/config.rs`, gated behind
`#[cfg(feature = "loop_stability_fix")]`:

```rust
#[cfg(feature = "loop_stability_fix")]
pub loop_stability_mode: super::LoopStabilityMode,
```

All 13 `Config` constructors (`micro`, `game`, `game_go`, `game_fft`, `draft`,
`small_target`, `gqa_draft`, `bpe`, `bpe_draft`, `gemma2_2b`, `qwen_deltanet`,
etc.) correctly initialize the field behind the same cfg gate. The
`goat_428_loop_stability.rs` test also handles it correctly.

However, `tests/bench_ldt_lattice_deduction.rs` (gated by
`#[cfg(feature = "lattice_deduction")]`) constructs `Config` via a **raw struct
literal** at line 33:

```rust
fn make_config(vocab_size: usize, draft_lookahead: usize, tree_budget: usize) -> Config {
    Config {
        vocab_size,
        block_size: 256,
        // ... ~30 fields ...
        // MISSING: #[cfg(feature = "loop_stability_fix")] loop_stability_mode: ...
    }
}
```

Under `--all-features`, both `lattice_deduction` and `loop_stability_fix` are
enabled simultaneously, so the field exists but the literal doesn't provide it.
This is the same bug class as the `merkle_root` lesson — no single feature
turns on both, so `cargo hack --each-feature` misses it; only `--all-features`
catches it.

## Fix

Either:
1. **Preferred:** switch `make_config()` to use `Config::micro()` (or another
   constructor) and override only the fields the test needs — constructors handle
   all feature-gated fields automatically.
2. **Alternative:** add the feature-gated field to the struct literal:
   ```rust
   #[cfg(feature = "loop_stability_fix")]
   loop_stability_mode: katgpt_rs::types::LoopStabilityMode::None,
   ```

Option 1 is more robust against future feature-gated Config fields (this will
keep happening as long as fields are added behind feature gates).

## Scope

- **Affected:** only `tests/bench_ldt_lattice_deduction.rs` (1 test file)
- **Not affected:** default build, any single-feature build, any lib/example/bench
- **Pre-existing:** introduced by Plan 428; not caused by Plan 438 (occupancy_ratio)

## Verification (after fix)

```
cargo clippy --workspace --all-features --all-targets
```
should complete with zero errors (the 68 warnings in `katgpt-pruners` and
`examples/recos_goat.rs` are separate pre-existing issues, not this bug).
