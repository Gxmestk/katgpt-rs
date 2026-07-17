# Issue 174 — Split `crates/katgpt-core/src/traits.rs` (2203 lines)

> **Source:** Issue 162 code-smell audit soft-limit follow-on. The prior
> session's summary misclassified this file as "would need functional split"
> — that verdict was incorrect. traits.rs is **32% tests** (708 test lines
> across 5 modules, 1480 impl lines), so a single-axis test extraction lands
> `mod.rs` at **1495 lines** — under the 2048 soft limit.
> **Opened:** 2026-07-17
> **Type:** Refactor (file-size hygiene, single-axis tests extraction)

---

## TL;DR

Mechanical test extraction. `crates/katgpt-core/src/traits.rs` (2203 lines)
→ `traits/mod.rs` (1495 ✓) + 5 sibling test files (708 total, tests exempt).
Five independent test modules each extracted to its own file. Zero
`pub(super)`/`pub(crate)` helpers. No path corrections. GOAT gate G1 + G3
PASS.

## Task

- [x] Verify 5 test modules + no `pub(super)` helpers
- [x] Create `traits/` module folder
- [x] Extract each `mod tests_*` body to a sibling `.rs` file
- [x] Replace test blocks in `traits/mod.rs` with `mod tests_*;` declarations
- [x] Remove original `traits.rs`
- [x] G1 — tests pass (default + under `recursion_logits`)
- [x] G3 — clippy clean (lib + tests + workspace)
- [x] Update Issue 162 entry, advance `.highwater`

## Test modules extracted

| File | Lines | Original location | cfg gate |
|---|---|---|---|
| `tests_leo.rs` | 359 | lines 1481-1841 | (none — module always compiled, inner `#[test]` gates emission) |
| `tests_spec_gen.rs` | 50 | lines 1845-1897 | `#[cfg(test)]` |
| `tests_best_buddies.rs` | 73 | lines 1901-1976 | `#[cfg(test)]` |
| `tests_reject_confidence.rs` | 95 | lines 1985-2082 | `#[cfg(test)]` |
| `recursion_logits_tests.rs` | 115 | lines 2085-2203 | `#[cfg(all(test, feature = "recursion_logits"))]` |

**Note on tests_leo:** unlike the other four, `tests_leo` has NO `#[cfg(test)]`
at the module level — it's always compiled. This is a pre-existing quirk
(intentional or not) that this split preserves exactly. The inner `#[test]`
attributes still gate test emission to test builds.

## GOAT gate

| Gate | Check | Result |
|---|---|---|
| **G1** | traits tests default | **36 passed / 0 failed / 0 ignored** |
| **G1** | traits tests under `recursion_logits` | **39 passed / 0 failed / 0 ignored** (36 + 3 recursion) |
| **G1** | full katgpt-core lib under `recursion_logits` | **1561 passed / 0 failed / 3 ignored** (1558 default + 3) |
| **G3** | clippy lib + tests under feature | clean |
| **G3** | clippy workspace lib | clean |

## Prior-session verdict correction

The prior session's Issue 162 summary listed `traits.rs` under "would need
functional split". That was wrong — the file is 32% tests across 5 modules,
and test extraction leaves `mod.rs` at 1495 lines. This issue corrects that
verdict and closes the gap. Same class of error as Issues 170 (qmc.rs),
171 (manifold_bandit.rs), 172 (d2f.rs), and 173 (lower.rs).
