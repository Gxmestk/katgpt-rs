# Issue 176 — Missed soft-limit files: 5 mechanical test-extraction splits

## Context

Issue 162's High-band audit listed ~11 files but stated "~20 files sit in this
band." A direct `find ... | wc -l` sweep for `.rs` files ≥ 2048 lines revealed
**8 library files over the soft limit that were entirely absent from the
audit**. This is the same class of miss as Issues 170–175 (prior sessions
called files "out of scope" without verifying structural seams).

This issue covers the **5 mechanical test-extraction splits** — files whose
`#[cfg(test)] mod tests` block is the dominant fraction of their size and
extracting it lands the impl well under 2048.

| File | Total | Tests start | Test lines | Impl lands at | Test % |
|---|---|---|---|---|---|
| `crates/katgpt-pruners/src/vocab_channel_pruner.rs` | 2053 | 1174 | 1912 (incl. test-only helper at L142) | **141** | 93% |
| `crates/katgpt-percepta/src/legacy.rs` | 2124 | 909 | 1216 | **908** | 57% |
| `crates/katgpt-core/src/funcattn.rs` | 2086 | 981 | 1106 | **980** | 53% |
| `crates/katgpt-dec/src/sheaf_admm.rs` | 2109 | 1220 | 890 | **1219** | 42% |
| `crates/katgpt-percepta/src/graph/types.rs` | 2055 | 1331 | 725 | **1330** | 35% |

All 5 files have:
- Single contiguous `#[cfg(test)] mod tests` block
- Zero `pub(super)` helpers (no path corrections needed)
- Standard `use super::*;` inside tests

## Tasks

- [x] T1. Extract `vocab_channel_pruner.rs` tests → `vocab_channel_pruner/tests.rs` (93% tests — biggest win). **PASS:** 176/176 katgpt-pruners tests under `vocab_channel_pruner`; mod.rs lands at 1183 (was 2053).
- [x] T2. Extract `legacy.rs` tests → `legacy/tests.rs`. **PASS:** 40/40 katgpt-percepta default + 339/339 under `percepta_compile`; mod.rs lands at 910 (was 2124).
- [x] T3. Extract `funcattn.rs` tests → `funcattn/tests.rs`. **PASS:** 22/22 `funcattn::tests::*` under `funcattn`; mod.rs lands at 983 (was 2086).
- [x] T4. Extract `sheaf_admm.rs` tests → `sheaf_admm/tests.rs`. **PASS:** 20/20 `sheaf_admm::tests::*` (DEFAULT-ON feature); mod.rs lands at 1222 (was 2109).
- [x] T5. Extract `graph/types.rs` tests → `graph/types/tests.rs`. **PASS:** 54/54 `graph::types::tests::*` under `percepta_graph`; mod.rs lands at 1333 (was 2055).
- [x] T6. GOAT G1+G3. **PASS:** workspace `cargo clippy --lib` clean; workspace `cargo test --lib` = 6519 passed / 0 failed under default features. katgpt-core lib alone = 1880 passed under default + funcattn feature.
- [x] T7. Issue 162 updated with "8 additional soft-limit files found post-audit" note + this issue recorded.

## Notes

- The `sigmoid` test-only helper in `vocab_channel_pruner/mod.rs` (L142, `#[cfg(test)]`) stays in mod.rs — it's correctly `#[cfg(test)]`-gated and visible to the sibling `tests.rs` via `use super::*;` when both compile under test mode.
- `graph/types.rs` (T5) was nested: split `graph/types.rs` → `graph/types/{mod.rs, tests.rs}`. Parent `graph/mod.rs` declares `pub mod types;` which resolves to either `types.rs` or `types/mod.rs` — Rust accepts both.
- Discovered the `graph::types::tests` were always gated behind `percepta_graph` feature (not a regression — pre-existing). All 54 tests appear when the feature is enabled.
- Same class of miss as Issues 170–175: prior sessions' audit listed ~11 files but said "~20" without enumerating the rest.

## Out of scope (separate issues)

- `bandit.rs` (2178) — functional split (Issue 177): 14 `// ──` delimiters mark natural seams (Strategy / Stats / Pruner / Beta sampling / Environment / Session / SharedBanditStats / RandOpt)
- `dd_tree/mod.rs` (2125) + `tree_builder.rs` (2091) — post-Issue 165 split results; investigation (Issue 178)
- `weaver.rs` (2817) — user-explicit skip (stands)

## References

- Issue 162 (parent audit)
- Issues 164–175 (prior mechanical + functional splits)
