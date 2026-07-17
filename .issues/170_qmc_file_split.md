# Issue 170 — Split `crates/katgpt-core/src/speculative/qmc.rs` (2516 lines)

> **Source:** Issue 162 code-smell audit soft-limit follow-on. The prior
> session's summary misclassified this file as "would need functional split"
> — that verdict was incorrect. qmc.rs is **57% tests** (1433 test lines,
> 1083 impl lines), so a single-axis test extraction lands `mod.rs` at
> **1085 lines** — well under the 2048 soft limit.
> **Opened:** 2026-07-17
> **Type:** Refactor (file-size hygiene, single-axis tests extraction)

---

## TL;DR

Mechanical test extraction. `crates/katgpt-core/src/speculative/qmc.rs`
(2516 lines) → `qmc/mod.rs` (1085 ✓) + `qmc/tests.rs` (1430, tests exempt).
Single `#[cfg(test)] mod tests` block at line 1084, single `use super::*;`
at top, zero `pub(super)`/`pub(crate)` helpers. No path corrections. GOAT
gate G1 + G3 PASS.

## Task

- [x] Verify single test block + no `pub(super)` helpers
- [x] Create `qmc/` module folder
- [x] Extract `mod tests` body to `qmc/tests.rs`
- [x] Replace test block in `qmc/mod.rs` with `#[cfg(test)] mod tests;`
- [x] Remove original `qmc.rs`
- [x] G1 — tests pass under `qmc_sampling`
- [x] G3 — clippy clean (lib + tests + workspace)
- [x] Update Issue 162 entry, advance `.highwater`

## GOAT gate

| Gate | Check | Result |
|---|---|---|
| **G1** | qmc tests pass under `qmc_sampling` | **119 passed / 0 failed / 2 ignored** (qmc + qmc_halter + sampling qmc-descend tests) |
| **G1** | full katgpt-core lib under `qmc_sampling` | **1558 passed / 0 failed / 3 ignored** |
| **G3** | clippy lib + tests under feature | clean |
| **G3** | clippy workspace lib | clean |

## Pre-existing unrelated failures (NOT caused by this split)

- `examples/ht_chantry_diagnostic.rs` requires `multi_agent_path` feature
  (documented in Issue 169, reproduces against unsplit original).

## Prior-session verdict correction

The prior session's Issue 162 summary listed `qmc.rs` under "would need
functional split". That was wrong — the file is 57% tests, and test
extraction leaves `mod.rs` at 1085 lines. This issue corrects that
verdict and closes the gap.
