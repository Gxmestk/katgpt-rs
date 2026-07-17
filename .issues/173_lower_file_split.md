# Issue 173 — Split `crates/katgpt-percepta/src/wasm/lower.rs` (2248 lines)

> **Source:** Issue 162 code-smell audit soft-limit follow-on. The prior
> session's summary misclassified this file as "would need functional split"
> — that verdict was incorrect. lower.rs is **17% tests** (378 test lines,
> 1870 impl lines), so a single-axis test extraction lands `mod.rs` at
> **1873 lines** — under the 2048 soft limit.
> **Opened:** 2026-07-17
> **Type:** Refactor (file-size hygiene, single-axis tests extraction)

---

## TL;DR

Mechanical test extraction. `crates/katgpt-percepta/src/wasm/lower.rs`
(2248 lines) → `wasm/lower/mod.rs` (1873 ✓) + `wasm/lower/tests.rs`
(375, tests exempt). Single `#[cfg(test)] mod tests` block at line 1871,
single `use super::*;` at top, zero `pub(super)`/`pub(crate)` helpers.
No path corrections. GOAT gate G1 + G3 PASS.

## Task

- [x] Verify single test block + no `pub(super)` helpers
- [x] Create `wasm/lower/` module folder
- [x] Extract `mod tests` body to `wasm/lower/tests.rs`
- [x] Replace test block in `wasm/lower/mod.rs` with `#[cfg(test)] mod tests;`
- [x] Remove original `lower.rs`
- [x] G1 — tests pass under `percepta_wasm`
- [x] G3 — clippy clean (lib + tests + workspace)
- [x] Update Issue 162 entry, advance `.highwater`

## GOAT gate

| Gate | Check | Result |
|---|---|---|
| **G1** | wasm::lower tests under `percepta_wasm` | **22 passed / 0 failed / 0 ignored** |
| **G1** | full katgpt-percepta lib under `percepta_wasm` | **225 passed / 0 failed / 0 ignored** |
| **G3** | clippy lib + tests under feature | clean |
| **G3** | clippy workspace lib | clean |

## Module gate note

`wasm` is gated behind the `percepta_wasm` feature in katgpt-percepta.
Tests do NOT run under default features (same as unsplit original).

## Prior-session verdict correction

The prior session's Issue 162 summary listed `lower.rs` under "would need
functional split". That was wrong — the file is 17% tests, and test
extraction leaves `mod.rs` at 1873 lines. This issue corrects that
verdict and closes the gap. Same class of error as Issues 170 (qmc.rs),
171 (manifold_bandit.rs), and 172 (d2f.rs).
