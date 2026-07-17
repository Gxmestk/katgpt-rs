# Issue 169 — Split `crates/katgpt-core/src/karc.rs` (2597 lines)

> **Source:** Issue 162 high-band soft-limit file hygiene (5th of 11).
> **Opened:** 2026-07-17
> **Type:** Refactor (file-size hygiene, single-axis tests extraction)
> **Parent:** [`.issues/162_code_smell_audit.md`](162_code_smell_audit.md) — High section, `crates/katgpt-core/src/karc.rs` (2597)

---

## TL;DR

`crates/katgpt-core/src/karc.rs` is **22% tests** — ~2005 lines of impl (KARC
delay-basis-ridge forecaster, gated `karc_forecaster`) + ~591 lines of tests
in a single `#[cfg(test)] mod tests { ... }` block.

This is the **only remaining split that lands under 2048 after test
extraction** — `mod.rs` will be ~2007 lines (impl 2005 + the new
`#[cfg(test)] mod tests;` directive). The other candidate (`weaver.rs`
2817/26% tests) leaves 2058 — *stays over* the soft limit, so per Issue 162
verdict (diminishing returns on splits that don't land under) we skip it.

Same single-axis tests-extraction pattern as Issues 166/167/168. No
`pub(super)` helpers, no path corrections (verified by grep — only one
`use super::*;` at the top of `mod tests`, no other `super::` references
inside the test body).

---

## Task

- [x] Convert `crates/katgpt-core/src/karc.rs` (2597 lines) into a module
      folder `crates/katgpt-core/src/karc/` with:
  - `mod.rs` — impl (KarcBasis trait + Fourier/Chebyshev/BSpline impls +
    DelayRing + feature_expand + chunked_gram + KarcScratch + LowRankFitScratch
    + jacobi_eigen + low_rank_fit + KarcForecaster + FitError). **2008 lines**
    (under 2048 ✓ — exactly as predicted).
  - `tests.rs` — the entire `mod tests { ... }` body. **589 lines** (tests
    exempt).
- [x] GOAT gate (under `karc_forecaster` feature):
  - **G1** (correctness): 19/19 `karc::tests::*` pass + 1587/1587 katgpt-core
    lib under `karc_forecaster` (vs 1558 default — 29 extra gated tests).
  - **G3** (no-regression): `cargo clippy -p katgpt-core --lib --features
    karc_forecaster` clean; `cargo clippy -p katgpt-core --tests --features
    karc_forecaster` clean; `cargo clippy --workspace` clean (default + lib
    under `karc_forecaster`).
- [x] Update parent Issue 162 — mark this entry DONE with strikethrough.
- [x] Commit.

## Pre-existing unrelated build errors (NOT caused by this split)

Two unrelated failures surface when running `cargo clippy --workspace
--features karc_forecaster --all-targets`:

1. `examples/ht_chantry_diagnostic.rs` — requires `multi_agent_path`
   feature (separate opt-in). Reproduces against the unsplit original.
2. `tests/issue_156_anytime_lt2_poc.rs` — pre-existing argument count
   mismatch on `forward_looped` (the `elastic_override` parameter issue
   documented in the prior session as "pre-existing unrelated build error").
   Reproduces against the unsplit original.

Neither is caused by the split. The karc module + its tests + its clippy
all pass cleanly under `karc_forecaster`.

## Non-goals

- No functional changes. No behavior change. No new tests. No new modules
  beyond `karc/{mod.rs, tests.rs}`.
- No touching `karc_dp` (sibling module, gated `karc_forecaster`).
- No promotion of `karc_forecaster` to default-on (separate decision, not
  in scope for a file-hygienic refactor).

## Context

File structure (per `read_file` outline + grep):

- L1-2005: impl
- L2006: `#[cfg(test)]`
- L2007: `mod tests {`
- L2008-2596: tests body (591 lines, 22%)
- L2597: closing `}`

After extraction:
- `mod.rs` ends with `#[cfg(test)] mod tests;`
- `tests.rs` starts with the body. The existing `use super::*;` resolves to
  `karc::*` — same as before, since `tests.rs` is still a child of `karc`.

No `pub(super)` helpers, no path corrections needed (verified).

## Cross-references

- Issue 162 — parent audit.
- Issues 166/167/168 — same split pattern (single-axis tests extraction).
- Plan 308 — KARC primitive plan (where the file originated).
