# Issue 709: three agents' WIP swept into one commit — one regression fixed, one still open

**Status:** T1, T2, T3a, T3c all DONE — `scripts/staged_set_audit.py` ships
four signals and the fourth (rustfmt round-trip, `--fmt`) refuted this
session's own standing belief about its worktree on first run. **T3b (the
refusing pre-commit hook) is DECIDED: won't ship** — see its row. T1 FIXED,
T2 **RESOLVED by the Plan 563 owner** (verified 2026-09-02
21:0x — the root manifest, `katgpt-moka-wasm`, `katgpt-pruners` and
`Cargo.lock` all read wasmi 2 at HEAD with none of them dirty, and
`cargo metadata --locked` exits 0). T3 **half-landed**: the measuring half
ships as `scripts/staged_set_audit.py`; the refusing pre-commit hook stays an
owner call and is deliberately not shipped.

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
- [x] **T2** ~~`develop` does not build `--locked`.~~ **RESOLVED by the owner**,
  not from here. Re-measured at HEAD: root manifest, `katgpt-moka-wasm` and
  `katgpt-pruners` all say `wasmi = "2"`, `Cargo.lock` carries `wasmi 2.0.0`
  (+ `wasmi_collections` / `wasmi_core` / `wasmi_ir`), none of those four files
  is dirty, and `cargo metadata --locked` exits 0 — i.e. resolution matches the
  committed lock, which is exactly what T2 said it did not. Leaving it alone
  was the right call: the migration landed as one tree.

  Original text: `develop` does not build `--locked`. HEAD's root manifest requires
  `wasmi ^2`; HEAD's `Cargo.lock` carries only `1.0.9` (4 entries). The
  uncommitted worktree lock already has `2.0.0`, and
  `crates/katgpt-moka-wasm` + `crates/katgpt-pruners` still say `1.0` at HEAD —
  i.e. the upgrade is mid-flight. **Unblock:** the Plan 563 owner lands the lock
  + the remaining manifests together. Committing the lock alone from here would
  publish a partially-migrated tree, so this is deliberately left alone.
- [x] **T3a** Ship the **measurement**, which is not an owner call:
  `scripts/staged_set_audit.py` (report, always exit 0). Single-linkage
  clustering over the staged files' worktree mtimes — the technique that caught
  this by hand twice — plus a second, independent signal (a staged path that
  still has unstaged changes, i.e. a concurrent editor mtime clustering cannot
  see). `selftest()` pins six shapes on every invocation, including the chaining
  case, because the failure mode is degrading to "1 episode, always" and
  printing a confident verdict indistinguishable from a real one.

  **Canaried on this repo's live state**, not asserted: staging one file from a
  sibling's 20:04:39 rustfmt sweep beside a 21:07:17 file of mine reported
  `2 editing episodes` + REVIEW; staging mine alone reported `✓ one editing
  episode`. Two-sided, so neither a dead nor an always-on verdict passes.

  A **third signal** was added after the first run found a hazard the other two
  are structurally blind to: a dirty-or-staged file that LACKS substantive lines
  the newest commit on its own path added, i.e. committing it reverts them. A
  whole-repo sweep is ONE episode and its files are not also-dirty, so neither
  earlier signal can see it. Live: `tpr/als.rs` sat dirty from the 20:04:39
  rustfmt sweep while `0ef7f078` landed a 22-line Issue 712 correctness fix in
  the same file at 21:08 — committing the sweep would have silently reverted it.
  Two-stage, because `mtime < commit time` alone flags the commonest shape there
  is (edit at 21:03, commit at 21:04 → the newest commit on that path is your
  own edit; it false-positived on two files), so line-set containment is the
  confirmation. Swept all 19 contract repos: exactly one hazard, two other repos
  dirty-but-clean — not always-on.

  **The signal beat its own author's attribution.** The sweep's owner had
  already reasoned "those dirty files are a sibling's work" and moved on —
  wrong twice over: they *were* the author, and one of the files carried a
  pending revert of someone else's landed fix. So this is not only a
  did-you-stage-someone-else's-work check; it catches a session **mis-attributing
  its own** churn away from itself, which no amount of care about `git add -A`
  would have found. Note also that `git log --author` cannot help at all here:
  every concurrent session commits as the same git user, so authorship does not
  distinguish sessions and mtime is the only signal that does.

  **Resolved 2026-09-02** by the owner (`katgpt-rs-be`), not from here. Cause:
  `cargo fmt -p katgpt-core -- <files>` — everything after `--` goes to rustfmt
  as *options*, not as a file selection, so each of three runs formatted the
  whole package. 189 of 204 reverted, `als.rs` included; audit now reports 0
  stale, and `cargo test -p katgpt-core --features tpr --lib tpr::` passes 27/27
  with the Issue 712 guard intact.
  Documented in AGENTS.md beside the `hash-object` + `update-index` recipe for
  committing your blob out of a file a sibling is editing.
- [x] **T3c** Add the **rustfmt round-trip** as a fourth signal:
  `git show HEAD:$f | rustfmt --edition 2024 --emit stdout | diff - $f`. If
  identical, the worktree copy is exactly "HEAD, formatted" and *provably*
  carries zero content, so reverting it cannot lose anyone's work. It splits
  "dirty because churn" from "dirty because work" with no mtime heuristic and
  no false positives from line re-wrapping — which an `--ignore-all-space` diff
  cannot do (measured: it called 214/215 files non-whitespace in another repo,
  because rustfmt re-wraps tokens *across* lines). This is what made the
  204-file revert provably rather than probably safe: 188 pure churn, 16 real,
  and `als.rs` matched the round-trip of `0ef7f078^` — i.e. the pre-guard
  version formatted, carrying nothing of its own. **Unblock:** none; it is
  Rust-only and needs a toolchain in PATH, so it is a separate signal rather
  than a replacement for the three that ship. Credit: `katgpt-rs-be`.

  **LANDED 2026-09-03** behind `--fmt`, opt-in because it shells out to
  rustfmt per file and needs a toolchain. It is the only signal here that
  yields a **proof** rather than evidence, and it earned that on its first
  run by refuting the session's own standing belief: the 15 files that four
  consecutive summaries called "a third session's pre-existing rustfmt
  churn" classified **0 pure churn, 15 carry content** — every one a real
  lint fix (`map_or(false, ·)` → `is_some_and`, `needless_borrow`,
  `repeat().take()` → `repeat_n`, unused `mut`, `needless_late_init`).
  Landed as `989f1bdf` after the two that collided with incoming commits
  were verified already-resolved upstream.

  It does one thing beyond the original formulation: diffing against
  `rustfmt(HEAD)` **isolates the content**, because formatting cancels. That
  is the review you actually want on a file whose real edit is buried under
  a whole-file reformat, and no whitespace-ignoring diff can produce it —
  rustfmt re-wraps tokens *across* lines, so `-w` still reports deletions.

  The `skip` verdict is load-bearing and is pinned by `selftest()`: a
  non-Rust or unparseable file must never classify as `churn`, because that
  verdict authorises a revert. Every other signal here only over- or
  under-warns; this one can destroy work if it is wrong in that direction.
- [x] **T3b** The **refusing** pre-commit hook — **DECIDED 2026-09-03: WON'T
  SHIP.** Closed as a decision, not deferred further; a permanently-open
  owner row is indistinguishable from a forgotten one.

  The evidence T3a and T3c were built to produce now argues *against* the
  hook. A refusing gate must fire on a predicate cheap enough to evaluate at
  commit time, and the only such predicates here are the three heuristics —
  each of which has a legitimate-use false positive: a whole-repo sweep is
  one honest episode, a partial-blob commit is deliberately also-dirty, and
  a rebase leaves files legitimately older than HEAD. The one signal with no
  false positives, the round-trip, costs a rustfmt subprocess per file and
  is Rust-only — unshippable as a blocking hook.

  So the hook would block honest work on heuristics while the sound signal
  stays advisory, and it would have blocked this very session's `989f1bdf`,
  which knowingly landed another session's work *after proving it safe*.
  Measured instead of argued: over four workspace sweeps the report produced
  exactly one true hazard and it was acted on both times it mattered —
  including once against this session's own tree. A report that is read
  beats a gate that is bypassed.

  **Revisit if** the report is ever observed being ignored on a true
  positive; that, not friction, is the condition that would justify the
  trade.

## Why this keeps happening

`git add -A` from a repo root is indistinguishable, to the tool, from intent.
Concurrent sessions in one worktree make it a routine hazard rather than an
occasional one — three agents were writing into `katgpt-rs` within the same
hour here. The two habits that actually work: stage **named files**, and when a
shared file is dirty, build the staged blob from `HEAD` plus your own hunk
(`git hash-object -w` + `git update-index --cacheinfo`) instead of staging the
worktree copy. That is how `1f7a96d4` committed its `lib.rs` change without
carrying the 585 removal along with it.
