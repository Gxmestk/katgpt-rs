# Issue 709: three agents' WIP swept into one commit — one regression fixed, one still open

**Status:** T1 FIXED (2026-09-02, same commit as this file). T2 OPEN and NOT
mine to fix — it belongs to the in-flight wasmi upgrade (riir-ai Plan 563).
Filed so the second one is visible rather than discovered by a CI red.

## What happened

`b2527521` ("feat: kNN differential-entropy estimator … Issue 708 P1") committed
six files. Two of them were **other agents' uncommitted work**, picked up by a
whole-tree `git add`:

| swept content | owner | effect |
|---|---|---|
| `tpr = []` + the `bench_707` block in `crates/katgpt-core/Cargo.toml` | Issue 707 (this session) | harmless — the rest of 707 landed in `1f7a96d4` |
| removal of `usage_rate_eviction = []` + the `bench_697` block | a stale-snapshot revert of Plan 585 | **regression, T1 below** |
| `wasmi = "1.0"` → `"2"` in the root `Cargo.toml` | riir-ai Plan 563 | **half-landed, T2 below** |

The 585 removal was never anybody's *edit*: the worktree copies of
`crates/katgpt-core/Cargo.toml`, `src/lib.rs` and `.plans/585_*.md` were exact
**ancestor blobs** (`4aa3de01`, `b50db0ef`, `6f1095a0`) — an older snapshot
restored over `HEAD`, not work in progress. A blob-vs-ancestor check separates
the two in one pass and is the cheap defence here; file mtimes do not (all
three carried the same 00:29 timestamp as a batch of genuine sibling edits).

## Tasks

- [x] **T1** Restore `usage_rate_eviction = []` and the `bench_697` `[[bench]]`
  block from `b2527521^`, and restore the worktree's ancestor-blob copies of
  `src/lib.rs` + `.plans/585_*.md` so a future whole-tree `add` cannot re-land
  the removal. Verified: `--features usage_rate_eviction` compiles the lib AND
  `bench_697_usage_rate_eviction_goat` again. Before the fix the feature did not
  exist (`error: the package 'katgpt-core' does not contain this feature`) while
  `src/kv_eviction/mod.rs` and the bench file were still tracked — 1002 lines of
  `lib.rs`-declared module gated on a flag nobody could turn on, and a GOAT
  bench that could not be run.
- [-] **T2** `develop` does not build `--locked`. HEAD's root manifest requires
  `wasmi ^2`; HEAD's `Cargo.lock` carries only `1.0.9` (4 entries). The
  uncommitted worktree lock already has `2.0.0`, and
  `crates/katgpt-moka-wasm` + `crates/katgpt-pruners` still say `1.0` at HEAD —
  i.e. the upgrade is mid-flight. **Unblock:** the Plan 563 owner lands the lock
  + the remaining manifests together. Committing the lock alone from here would
  publish a partially-migrated tree, so this is deliberately left alone.
- [-] **T3** Consider a pre-commit guard that refuses a commit whose staged set
  contains a file the committing session never touched. **Unblock:** owner call
  — it trades a real class of loss against friction on every legitimate
  multi-file commit, and the cheaper habit (`git -C <repo> add <named files>`,
  never `-A`) already exists in AGENTS.md.

## Why this keeps happening

`git add -A` from a repo root is indistinguishable, to the tool, from intent.
Concurrent sessions in one worktree make it a routine hazard rather than an
occasional one — three agents were writing into `katgpt-rs` within the same
hour here. The two habits that actually work: stage **named files**, and when a
shared file is dirty, build the staged blob from `HEAD` plus your own hunk
(`git hash-object -w` + `git update-index --cacheinfo`) instead of staging the
worktree copy. That is how `1f7a96d4` committed its `lib.rs` change without
carrying the 585 removal along with it.
