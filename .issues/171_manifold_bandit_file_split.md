# Issue 171 — Split `crates/katgpt-core/src/manifold_bandit.rs` (2196 lines)

> **Source:** Issue 162 code-smell audit soft-limit follow-on. The prior
> session's summary misclassified this file as "would need functional split"
> — that verdict was incorrect. manifold_bandit.rs is **41% tests** (909 test
> lines, 1287 impl lines), so a single-axis test extraction lands `mod.rs`
> at **1290 lines** — well under the 2048 soft limit.
> **Opened:** 2026-07-17
> **Type:** Refactor (file-size hygiene, single-axis tests extraction)

---

## TL;DR

Mechanical test extraction. `crates/katgpt-core/src/manifold_bandit.rs`
(2196 lines) → `manifold_bandit/mod.rs` (1290 ✓) + `manifold_bandit/tests.rs`
(906, tests exempt). Single `#[cfg(test)] mod tests` block at line 1288,
single `use super::*;` at top, zero `pub(super)`/`pub(crate)` helpers.
No path corrections. GOAT gate G1 + G3 PASS.

## Task

- [x] Verify single test block + no `pub(super)` helpers
- [x] Create `manifold_bandit/` module folder
- [x] Extract `mod tests` body to `manifold_bandit/tests.rs`
- [x] Replace test block in `manifold_bandit/mod.rs` with `#[cfg(test)] mod tests;`
- [x] Remove original `manifold_bandit.rs`
- [x] G1 — tests pass under `manifold_bandit`
- [x] G3 — clippy clean (lib + tests + workspace)
- [x] Update Issue 162 entry, advance `.highwater`

## GOAT gate

| Gate | Check | Result |
|---|---|---|
| **G1** | manifold_bandit tests under `manifold_bandit` | **34 passed / 0 failed / 0 ignored** |
| **G1** | full katgpt-core lib under `manifold_bandit` | **1558 passed / 0 failed / 3 ignored** |
| **G3** | clippy lib + tests under feature | clean |
| **G3** | clippy workspace lib | clean |

## Prior-session verdict correction

The prior session's Issue 162 summary listed `manifold_bandit.rs` under
"would need functional split". That was wrong — the file is 41% tests,
and test extraction leaves `mod.rs` at 1290 lines. This issue corrects
that verdict and closes the gap. Same class of error as Issue 170 (qmc.rs).
