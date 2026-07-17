# Issue 165 — dd_tree.rs File Split (Issue 162 C2)

> **Source:** [Issue 162](../.issues/162_code_smell_audit.md) §C2 (Critical — >3200-line hard-limit violation)
> **Opened:** 2026-07-17
> **Type:** refactor (file-size hygiene)
> **Status:** DONE

---

## TL;DR

Split the monolithic `crates/katgpt-speculative/src/dd_tree.rs` (**4207 lines**,
31% over the 3200 hard limit) into a `dd_tree/` module folder. The dominant
chunk — the `TreeBuilder` struct + its **2094-line impl block** — is extracted to
`tree_builder.rs`. Public API surface preserved 1:1 via `mod.rs` re-exports.
GOAT gate **G1 + G3 PASS**: 305/305 katgpt-speculative tests pass; 200/200
katgpt-rs tests pass; 1079/1079 `--all-features` speculative tests pass; clippy
clean workspace-wide.

## Module layout

| Module | Lines | Contents |
|--------|-------|----------|
| `mod.rs` | 2125 | Data types + all `build_dd_tree_*` builder functions + helpers (SDE, lodestar, width-scale, cross-scale, manifold, domino, speculative) |
| `tree_builder.rs` | 2091 | `TreeBuilder` struct + impl (the pre-allocated buffer pool + zero-alloc build methods) |
| `tests.rs` | 902 | In-module test suite (moved from `dd_tree_tests.rs`) |

Both implementation files are now **under the 3200 hard limit**. They remain in
the 2048–3200 soft-limit band (High, not Critical). Further soft-limit reduction
would require splitting the builder functions or the TreeBuilder impl further,
which offers diminishing returns on a 2091-line cohesive impl block.

## Design notes

- **`tree` field visibility:** the `TreeBuilder.tree` field was made `pub(crate)`
  because 4 builder functions in `mod.rs` (`build_dd_tree`, `build_dd_tree_pruned`,
  `build_dd_tree_balanced`, `build_dd_tree_screened`) extract the built tree via
  `std::mem::take(&mut builder.tree)` (they reuse the builder afterward, so
  `into_tree()` which consumes the builder doesn't fit).

- **Test file relocation:** `dd_tree_tests.rs` (sibling file, referenced via
  `#[path]`) moved into the module folder as `tests.rs`, and the `#[path]`
  attribute was removed (standard module resolution now applies). The test file
  uses `use super::*;` which resolves to `mod.rs` — all items re-exported there
  remain accessible.

## GOAT gate

| Gate | Target | Result |
|------|--------|--------|
| **G1** (correctness) | Bit-identical behavior: all existing tests pass | **PASS** — 305/305 `katgpt-speculative` lib tests; 200/200 `katgpt-rs` lib tests; 1079/1079 `--all-features` speculative tests |
| **G3** (no-regression) | No clippy warnings, no compile errors | **PASS** — `cargo clippy -p katgpt-speculative` clean; `cargo clippy --all-features -p katgpt-speculative --lib --tests` clean; `cargo clippy --workspace` clean |

G2 (perf) and G4 (alloc-free) are **N/A** for a pure file split.

## Validation

```
cargo clippy -p katgpt-speculative                        → clean
cargo clippy --all-features -p katgpt-speculative --lib --tests → clean
cargo clippy --workspace                                  → clean
cargo test -p katgpt-speculative --lib                    → 305/305 PASS
cargo test --all-features -p katgpt-speculative --lib     → 1079/1079 PASS
cargo test -p katgpt-rs --lib                             → 200/200 PASS
```

Hardware: Apple M3 Max (aarch64, NEON).
