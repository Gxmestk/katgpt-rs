# Issue 168 — Split `crates/katgpt-forward/src/dd_tree.rs` (2566 lines)

> **Source:** Issue 162 high-band soft-limit file hygiene (4th of 11).
> **Opened:** 2026-07-17
> **Type:** Refactor (file-size hygiene, single-axis tests extraction)
> **Parent:** [`.issues/162_code_smell_audit.md`](162_code_smell_audit.md) — High section, `crates/katgpt-forward/src/dd_tree.rs` (2566)

---

## TL;DR

`crates/katgpt-forward/src/dd_tree.rs` is **98% tests** — only ~175 lines of impl
(the two feature-gated wrappers `build_dd_tree_screened_with_schedule` and
`build_dd_tree_gdsd` + the `pub use katgpt_speculative::dd_tree::*` re-export
glob + gated imports). The remaining ~2389 lines are a single `#[cfg(test)]
mod tests { ... }` block exercising the full dd_tree + dflash_predict
pipeline.

This is the trivial case from Issue 162's "high-band soft-limit" list — a
mechanical single-axis tests extraction. No functional split, no helper
visibility changes (no `pub(super)` test helpers — unlike Issue 167's
parallax_attn split).

---

## Task

- [x] Convert `crates/katgpt-forward/src/dd_tree.rs` (2566 lines) into a
      module folder `crates/katgpt-forward/src/dd_tree/` with:
  - `mod.rs` — impl (the two feature-gated wrappers + the re-export glob +
    gated imports). **179 lines** (well under 2048 ✓).
  - `tests.rs` — the entire `mod tests { ... }` body. **2387 lines** (tests
    exempt).
- [x] GOAT gate:
  - **G1** (correctness): katgpt-forward lib — 109/109 default + 112/112
    under `thinking_prune,sr2am_configurator,gdsd_distill` (49/49 in
    `dd_tree::tests::*`, including the 3 gated `test_goat_*` tests that
    exercise the resident feature-gated impl fns). Root lib also passes
    200/200 under the same features.
  - **G3** (no-regression): `cargo clippy -p katgpt-forward --all-targets
    --features thinking_prune,sr2am_configurator,gdsd_distill` clean;
    `cargo clippy --workspace` clean.
- [x] Update parent Issue 162 — mark this entry DONE with strikethrough.
- [x] Commit.

## Pre-existing wiring note (not caused by this split)

The katgpt-forward `thinking_prune` feature alone is `[]` (empty) and does
NOT forward `sr2am_configurator` to `katgpt-pruners`. The impl fn
`build_dd_tree_screened_with_schedule` references `katgpt_pruners::PrunerSchedule`,
which is gated behind `katgpt-pruners/sr2am_configurator`. Verified by
reproducing the same `PrunerSchedule not found` error against the **unsplit**
original file — the error is pre-existing, caused by the root crate being
the only place that wires `thinking_prune → sr2am_configurator` forwarding.

Running `cargo test -p katgpt-forward --features thinking_prune` in
isolation fails both before and after the split. The GOAT gate therefore
runs through the root crate (where the feature forwarding is complete) OR
with `sr2am_configurator` added explicitly. This is documented; no fix
required for the split itself.

## Non-goals

- No functional changes. No behavior change. No new tests. No new modules
  beyond `dd_tree/{mod.rs, tests.rs}`.
- No touching the leaf `katgpt_speculative::dd_tree` (already split by
  Issue 165).
- No touching the root `src/speculative/dd_tree.rs` (23-line shim that
  re-exports from katgpt-forward).

## Context

The file's structure (per `read_file` outline):

- L1-176: impl — module doc, gated imports, `pub use katgpt_speculative::dd_tree::*`,
  two `#[cfg(feature = "...")] pub fn`s.
- L177: `#[cfg(test)] mod tests {`
- L178-2565: tests body
- L2566: closing `}`

`use super::*` inside `mod tests` resolves to `dd_tree::*` which brings in
the glob from `katgpt_speculative::dd_tree`. After extraction:

- `mod.rs` ends with `#[cfg(test)] mod tests;`
- `tests.rs` starts with the body (the existing `use super::*;` resolves to
  `dd_tree::*` — same as before, since `tests.rs` is still a child of
  `dd_tree`).

No `pub(super)` helpers, no path corrections needed.

## Cross-references

- Issue 162 — parent audit.
- Issue 165 — the *other* `dd_tree.rs` split (katgpt-speculative leaf, 4207
  lines, C2 critical). Distinct file.
- Issue 166, Issue 167 — same split pattern (tests extraction).
