# Issue 167 — parallax_attn.rs File Split (Issue 162 soft-limit band)

> **Source:** [Issue 162](../.issues/162_code_smell_audit.md) §"High-band soft-limit files"
> **Opened:** 2026-07-17
> **Type:** refactor (file-size hygiene)
> **Status:** DONE

---

## TL;DR

Split the monolithic `crates/katgpt-core/src/parallax_attn.rs` (**2524 lines**,
23% over the 2048 `.rs` soft limit) into a `parallax_attn/` module folder.
The implementation stays in `mod.rs` (973 lines, well under 2048 ✓); the test
suite moved to `tests.rs` (1559 lines — tests are exempt from the soft limit
per Issue 162). The public API surface is preserved 1:1. GOAT gate
**G1 + G3 PASS**: 1558/1558 katgpt-core default lib tests pass; 24/24
parallax_attn feature tests pass (across 4 feature-gate combinations); clippy
clean with `parallax_attn,sink_aware_attn,ssmax_temperature` features.

Mirrors Issues 164/165/166 — tests extraction is the principled minimal split
for a file whose implementation is cohesive (Parallax is one attention
mechanism with 3 composable extensions: sigmoid activation, sink-aware gate,
SSMax temperature).

## Motivation

Issue 162 §"High-band soft-limit files" listed `parallax_attn.rs` (2524 lines)
in the soft-limit band. The file is 62% tests (1560 of 2524 lines) — the
implementation is only 965 lines. The test extraction is the cleanest win in
the remaining soft-limit band: biggest ratio of tests-to-impl, bringing the
impl well under the 2048 target.

## Module layout

| File | Lines | Contents |
|------|-------|----------|
| `parallax_attn/mod.rs` | 973 | Implementation: `ParallaxConfig`, `ParallaxActivation`, `ParallaxScratch`, `apply_parallax_*`, `forward_parallax_*`, SSMax integration, sink-aware integration |
| `parallax_attn/tests.rs` | 1559 | 4 test sections: main tests + `sink_aware_tests` + `ssmax_composition_tests` + `ssmax_sink_aware_tests` |

`mod.rs` at 973 lines is **well under the 2048 soft limit** ✓ (62% reduction).
`tests.rs` at 1559 lines exceeds the soft limit but tests are explicitly exempt
per Issue 162.

## Design notes

- **Four test sections with different cfg gates:** the original file had:
  - `mod tests` — always-on (`#[cfg(test)]`)
  - `mod sink_aware_tests` — `#[cfg(all(test, feature = "parallax_attn", feature = "sink_aware_attn"))]`
  - `mod ssmax_composition_tests` — `#[cfg(all(test, feature = "parallax_attn", feature = "ssmax_temperature"))]`
  - `mod ssmax_sink_aware_tests` — `#[cfg(all(test, feature = "parallax_attn", feature = "sink_aware_attn", feature = "ssmax_temperature"))]`

  All four moved to `tests.rs`. The outer `#[cfg(test)] mod tests;` in mod.rs
  gates the whole file; each submodule retains its feature gate (with `test`
  dropped from the `cfg(all(...))` since the outer gate subsumes it).

- **Visibility adjustment (the `pub(super)` → `fn` fix):** the original
  `build_sink_case` and `lcg_fill` helpers were `pub(super)` so the sibling
  `sink_aware_tests` module could call them via `super::tests::build_sink_case`.
  After the split, `sink_aware_tests` is a CHILD of `tests` (not a sibling of
  the `tests` module). Two changes were needed:
  1. `pub(super) fn build_sink_case` → `fn build_sink_case` (callers are now
     in child modules, which can access parent's private items)
  2. `super::tests::build_sink_case` → `super::build_sink_case` (6 call sites
     in `sink_aware_tests` + `ssmax_sink_aware_tests`)

  Same fix for `lcg_fill` (6 call sites). This is the only logic change in the
  split — everything else is mechanical.

- **Test path change (minor):** before the split, the feature-gated test
  modules were at `parallax_attn::{sink_aware,ssmax_composition,ssmax_sink_aware}_tests::*`
  (siblings of `tests`). After the split, they're at
  `parallax_attn::tests::{sink_aware,ssmax_composition,ssmax_sink_aware}_tests::*`
  (nested). Doesn't affect correctness or CI filtering.

## GOAT gate

| Gate | Target | Result |
|------|--------|--------|
| **G1** (correctness) | Bit-identical behavior: all existing tests pass | **PASS** — 1558/1558 katgpt-core default lib tests pass; 24/24 `parallax_attn,sink_aware_attn,ssmax_temperature` feature tests pass |
| **G3** (no-regression) | No clippy warnings, no compile errors | **PASS** — `cargo clippy -p katgpt-core --features parallax_attn,sink_aware_attn,ssmax_temperature --lib` clean |

G2 (perf) and G4 (alloc-free) are **N/A** for a pure file split.

## Validation

```
cargo clippy -p katgpt-core --features parallax_attn --lib                          → clean
cargo clippy -p katgpt-core --features parallax_attn,sink_aware_attn,ssmax_temperature --lib  → clean
cargo test -p katgpt-core --lib                                                     → 1558/1558 PASS (3 ignored)
cargo test -p katgpt-core --features parallax_attn --lib parallax_attn              → 16/16 PASS
cargo test -p katgpt-core --features parallax_attn,sink_aware_attn,ssmax_temperature --lib parallax_attn  → 24/24 PASS
```

Hardware: Apple M3 Max (aarch64, NEON).

## References

- [Issue 162](../.issues/162_code_smell_audit.md) — the code-smell audit.
- [Issue 164](../.issues/164_transformer_file_split_c1.md) — transformer.rs split.
- [Issue 165](../.issues/165_dd_tree_file_split_c2.md) — dd_tree.rs split.
- [Issue 166](../.issues/166_dllm_file_split.md) — dllm.rs split (same session).
