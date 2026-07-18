# Issue 180 — Remaining Documentation Path Drift (Cross-Repo Scope)

> **Date:** 2026-07-18
> **Status:** OPEN (substantially reduced — session 9 cut stale count 74%)
> **Scope:** All 7 private repos + katgpt-rs
> **Origin:** Discovered during session 8 of the cross-repo benchmark/doc cleanup task.
> **Updated:** Session 9 (2026-07-18) — bulk mechanical resolution phase.

## TL;DR

Session 8 of the benchmark-cleanup umbrella task expanded scope from
bench-only DRY consolidation (sessions 1-7) to **documentation path drift**
— `.benchmarks/*.md` / `.docs/*.md` / `.plans/*.md` files referencing
file paths that no longer exist due to crate-split refactors.

Session 9 (this session) executed **8 distinct mechanical strategies**
plus iterative passes to resolve stale refs from ~9004 down to ~2328
(**74% reduction**, **6676 refs fixed**).

Three classes of drift remain **open** in this issue (require manual
review or per-file decisions):

- **A. Genuinely ambiguous** — basename matches multiple unrelated files
  in multiple repos. Resolving requires per-reference context inspection.
- **B. Suffix-mismatch ambiguous** — basename uniquely matches but the
  candidate path's prefix differs from the source ref's prefix. Some
  are confident (parent dir matches crate name), some are not.
- **C. Truly-gone files** — refs to files that don't exist anywhere,
  including design docs describing files that were never implemented.

## Session 9 strategies (what worked)

| # | Strategy | Count fixed | Notes |
|---|---|---:|---|
| 1 | **Iterative crate-double-nest collapse** (`crates/X/crates/X/` → `crates/X/`) | ~600 | Fixed session-8 regression. Iterative passes catch triple-nesting. |
| 2 | **Cross-repo unprefixed** (add `<repo>/` prefix when file exists in exactly one other repo) | ~250 | Conservative: only when unambiguous. |
| 3 | **AMBIGUOUS_SUFFIX** (basename matches one candidate by full-suffix) | ~2400 | The biggest single win. Boundary-aware regex prevents double-prefix bug. |
| 4 | **AMBIGUOUS segment match** (try last N segments as suffix) | ~400 | Catches cases where the doc dropped a leading `src/` or `crates/X/`. |
| 5 | **PREFIXED_INVALID resolution** (right repo + wrong path → unique candidate) | ~460 | Combines 4 sub-strategies. |
| 6 | **File→directory refactor** (`X.rs` → `X/mod.rs`) | ~380 | Files that became directory modules. |
| 7 | **Module-name to crate-name** (`pruners/X.rs` → `crates/katgpt-pruners/src/X.rs`) | ~385 | Parent-dir matches crate-suffix heuristic. |
| 8 | **Git rename history** (`git log --diff-filter=R`) | ~100 | Last-resort: ask git what the file was renamed to. |

**Boundary-aware regex pattern** (key insight that avoided session-8's
double-prefix regression):

```python
pattern = re.compile(
    r'(^|[^\w/\-])'           # start-of-text OR non-path char
    + r'(' + re.escape(old) + r')'  # the old ref
    + r'(?![\w/\-])'          # not followed by path char
)
```

This prevents `src/foo.rs` from matching inside `crates/X/src/foo.rs`
(which would otherwise produce `crates/X/src/src/foo.rs`).

## Current state (post-session-9)

| Repo | SAME_REPO | Cross-repo fixed | Stale remaining | Notes |
|---|---:|---:|---:|---|
| katgpt-rs | 4155 | 656+717 | 392+111+160+161+119 | Issue 180 lives here |
| riir-ai | 2546 | 262+1162 | 162+124+125+154+94 | Mostly cross-repo (consumes katgpt-rs + sibling crates) |
| riir-chain | 104 | 59+23 | 1+1+0+4+2 | Small surface |
| riir-neuron-db | 126 | 55+66 | 1+0+4+8+10 | Small surface |
| riir-game-sdk | 33 | 0+9 | 2+3+0+9+0 | Facade crate — most refs are re-exports |
| riir-train | 336 | 36+449 | 38+137+55+50+9 | Sibling-repo refs dominate |
| seal-online-remaster | 1050 | 0+3 | 117+41+16+188+0 | Big consumer; 188 TRULY_GONE |
| poc-maxman | 9 | 2+7 | 3+0+0+1+1 | Tiny |
| **TOTAL** | **8359** | **1070+2436** | **716+417+360+575+235 = 2328** | |

Total stale dropped from ~9004 (session-8 end) → 2328 (session-9 end).

## What remains (3 classes)

### A. AMBIGUOUS (716 refs)

Basename matches multiple files in different crates. Resolving requires
per-reference context inspection — the surrounding markdown's domain
keywords (ShardIndex, NpcClr, KARC, etc.) disambiguate.

Example: `mod.rs` referenced in a doc that discusses `transformer` code
likely means `crates/katgpt-transformer/src/.../mod.rs`, but multiple
`mod.rs` files exist in `katgpt-transformer` itself.

### B. Suffix-mismatch AMBIGUOUS_RESOLVED (777 refs)

Basename uniquely matches one file, but the path leading to it differs
from what the doc wrote. Some are confidently fixable (parent dir
matches crate name — strategy 7 above caught the easy ones), the rest
need manual judgment.

### C. TRULY_GONE (575) + PREFIXED_INVALID-gone (235)

References to files that don't exist anywhere:
- Files deleted without doc update
- Design docs describing files that were never implemented as described
- Files moved to other repos without doc update (some caught by git
  rename history, the rest need manual research)

## Recommended strategy for session 10

1. **Per-file annotation pass** for the 61-ref doc
   (`seal-online-remaster/.plans/009_layer7_client.md`) — these are
   design-plan refs to files that were never implemented as named.
   Annotate the doc with a note explaining the design evolved.

2. **Domain-keyword disambiguation** for AMBIGUOUS — spawn subagents
   to read each doc, identify domain keywords, and pick the right
   candidate. Time-intensive but high-value.

3. **Truly-gone annotation** — for refs that genuinely point at deleted
   files, add a `> **Note:** The file ... no longer exists; kept as
   historical record.` blockquote at the doc head per the noise-reduction
   rule.

## Verification

Session 9 includes:
- **Zero double-prefix regressions** — verified via
  `((?:crates|src)/[\w\-]+)/\1/` regex after every commit.
- **CRLF preservation** — all writes use binary mode + CRLF detection
  (seal-online-remaster still uses CRLF for some files).
- **No `.rs` source files touched** — only `.benchmarks/`, `.docs/`,
  `.plans/`, `.issues/`, `.research/` paths modified (verified via
  `git diff --name-only` after every commit).

## Commits landed in session 9

32 commits across 8 repos. Highlights:

| Repo | Strategy | Replacements | Commit(s) |
|---|---|---:|---|
| katgpt-rs | Double-nest collapse (iter) | 441 | 1e2092c5, 8d56c5b4 |
| riir-train | Double-nest collapse | 156 | ca3c9f7 |
| riir-chain | Double-nest collapse | 6 | e4d024b |
| katgpt-rs | Cross-repo prefix | 31 | cf1b22e3 |
| riir-ai | Cross-repo prefix | 96 | 6e132712 |
| riir-neuron-db | Cross-repo prefix | 2 | 561924d |
| riir-train | Cross-repo prefix | 107 | 303c100 |
| poc-maxman | Cross-repo prefix | 5 | 16ee550 |
| All 7 | AMBIGUOUS_SUFFIX resolution | ~2400 | fe4c0fea, 3760ee21, ... |
| All 7 | AMBIGUOUS segment match | ~400 | 964ec7c4, ... |
| All 7 | PREFIXED_INVALID resolution | ~460 | b0995098, e10583ef, ... |
| All 7 | File→dir refactor | ~380 | 48c38395, f5ac3653, ... |
| All 7 | Module→crate name match | ~385 | 8a148554, ced344f4, ... |
| All 7 | Parent-dir→crate-suffix match | ~260 | 2107af8f, 6c80b9b1, ... |
| All 7 | Git rename history | ~100 | d186e211, 41ae7a81, ... |

(Full per-commit list available via `git log --oneline --since="1 day ago"`
in each repo.)

## See also

- Session 7 summary: `riir-neuron-db` commit `d69a84c` splitmix64 BenchRng.
- Session 8 summary: 12 commits, ~2,700 path renames (the regression
  session 9 had to clean up).
- AGENTS.md "Numbering Discipline" rule: numbers are monotonic and never
  reused. Issue 180 is the next free number after `.highwater` = 179.
