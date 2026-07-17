# Issue 172 — Split `crates/katgpt-forward/src/d2f.rs` (2268 lines)

> **Source:** Issue 162 code-smell audit soft-limit follow-on. The prior
> session's summary misclassified this file as "would need functional split"
> — that verdict was incorrect. d2f.rs is **21% tests** (485 test lines,
> 1783 impl lines), so a single-axis test extraction lands `mod.rs` at
> **1783 lines** — under the 2048 soft limit.
> **Opened:** 2026-07-17
> **Type:** Refactor (file-size hygiene, single-axis tests extraction)

---

## TL;DR

Mechanical test extraction. `crates/katgpt-forward/src/d2f.rs` (2268 lines)
→ `d2f/mod.rs` (1783 ✓) + `d2f/tests.rs` (485, tests exempt). Single
`#[cfg(test)] mod tests` block at line 1781, single `use super::*;` at
top, zero `pub(super)`/`pub(crate)` helpers. No path corrections. GOAT
gate G1 + G3 PASS.

## Task

- [x] Verify single test block + no `pub(super)` helpers
- [x] Create `d2f/` module folder
- [x] Extract `mod tests` body to `d2f/tests.rs`
- [x] Replace test block in `d2f/mod.rs` with `#[cfg(test)] mod tests;`
- [x] Remove original `d2f.rs`
- [x] G1 — tests pass under `dllm` (d2f is dllm-gated)
- [x] G3 — clippy clean (lib + tests + workspace)
- [x] Update Issue 162 entry, advance `.highwater`

## GOAT gate

| Gate | Check | Result |
|---|---|---|
| **G1** | d2f tests under `dllm` | **18 passed / 0 failed / 0 ignored** |
| **G1** | full katgpt-forward lib under `dllm` | **127 passed / 0 failed / 0 ignored** (109 default + 18 d2f) |
| **G1** | root crate lib under `dllm` | **200 passed / 0 failed / 0 ignored** |
| **G3** | clippy lib + tests under feature | clean (3 pre-existing warnings in dd_tree/tests.rs, unrelated) |
| **G3** | clippy workspace lib under feature | clean |

## Module gate note

`d2f` is gated behind the `dllm` feature in katgpt-forward. Tests do NOT
run under default features (this is true of the unsplit original too —
verified via baseline worktree at `/tmp/katgpt-d2f-baseline`).

## Prior-session verdict correction

The prior session's Issue 162 summary listed `d2f.rs` under "would need
functional split". That was wrong — the file is 21% tests, and test
extraction leaves `mod.rs` at 1783 lines. This issue corrects that
verdict and closes the gap. Same class of error as Issues 170 (qmc.rs)
and 171 (manifold_bandit.rs).
