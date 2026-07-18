# Issue 180 — Remaining Documentation Path Drift (Cross-Repo Scope)

> **Date:** 2026-07-18
> **Status:** OPEN (mechanical strategies exhausted — session 10 cut stale count 62%)
> **Scope:** All 7 private repos + katgpt-rs
> **Origin:** Discovered during session 8 of the cross-repo benchmark/doc cleanup task.
> **Updated:** Session 10 (2026-07-18) — final mechanical pass + bulk annotation.

## TL;DR

Session 8 of the benchmark-cleanup umbrella task expanded scope from
bench-only DRY consolidation (sessions 1-7) to **documentation path drift**
— `.benchmarks/*.md` / `.docs/*.md` / `.plans/*.md` files referencing
file paths that no longer exist due to crate-split refactors.

- Session 9 cut stale refs from ~9004 → 2200 (**74% reduction**, 6676 fixed).
- **Session 10 (this update)** cut from 2200 → 847 (**62% reduction**, 1353 fixed).
- **Cumulative:** ~9004 → 847 = **90.6% reduction**, 8157 refs fixed total.

The remaining 847 stale refs require per-file manual inspection or are
covered by doc-level "historical file paths" annotations (61 docs annotated).

## Session 10 strategies (what worked)

| # | Strategy | Count fixed | Notes |
|---|---|---:|---|
| 1 | **AMBIGUOUS_RESOLVED_SAME resolution** (basename unique in same repo, prefix differs) | 417 | Biggest single win — boundary-aware regex. |
| 2 | **AMBIGUOUS_RESOLVED_CROSS resolution** (basename unique in sibling repo) | 233 | Cross-repo prefix add. |
| 3 | **PREFIXED_INVALID resolution v2** (file→dir refactor, dir→file, basename) | 131 | Sub-strategies combined. |
| 4 | **AMBIGUOUS via same/cross-repo uniqueness** (basename in 1 repo only) | 149 | Disambiguation when candidates spanned multiple repos. |
| 5 | **Bulk migration mapping** (types.rs/traits.rs split, katgpt-types promotion, dd_tree mux) | 230 | katgpt-rs-specific known migrations. |
| 6 | **Cross-repo suffix match** for AMBIGUOUS | 14 | Found one file across all repos matching ref's suffix. |
| 7 | **Doc-crate-affinity** for AMBIGUOUS (use doc's resolved refs to pick crate) | 33 | Use surrounding resolved refs to infer crate context. |
| 8 | **Domain-keyword disambiguation** (ShardIndex, MAG, HLA, KARC, etc.) | 25 | Scan ±300 chars of context for crate-keyword votes. |
| 9 | **riir-gpu/src/forward.rs → forward/mod.rs migration** | 29 | Single file → directory module. |
| 10 | **riir-chain neuron_db/shard.rs → riir-neuron-db/src/shard/mod.rs** | 11 | Cross-repo module spinoff. |
| 11 | **Seal-specific paths** (crypto/mod.rs, minimap/mod.rs → seal-core) | 12 | Same-repo unprefixed → crates/-prefixed. |
| 12 | **Specific typos + module paths** (katgpt-rs-core, katgpt-core/types.rs, etc.) | 13 | Targeted fixes. |
| 13 | **Doc-level annotation pass** (>= 4 TRULY_GONE refs) | 32 docs | Add 'historical file paths' blockquote. |
| 14 | **Tier-2 annotation** (exactly 3 TRULY_GONE refs) | 21 docs | Smaller docs batch. |

**Boundary-aware regex pattern** (carried over from session 9, used in
every fixer — the key insight that prevents double-prefix regressions):

```python
pattern = re.compile(
    r'(^|[^\w/\-])'           # start-of-text OR non-path char
    + r'(' + re.escape(old) + r')'  # the old ref
    + r'(?![\w/\-])'          # not followed by path char
)
```

## Current state (post-session-10)

| Repo | SAME_REPO | SAME_REPO_PREF | CROSS_REPO_PRE_VALID | AMBIGUOUS | TRULY_GONE | PREFIXED_INVALID | CROSS_REPO_UNP | Total stale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| katgpt-rs | 4322 | 656 | 879 | 56 | 165 | 79 | 0 | **300** |
| riir-ai | 2696 | 276 | 1349 | 67 | 154 | 36 | 16 | **273** |
| riir-chain | 105 | 59 | 23 | 1 | 4 | 2 | 0 | **7** |
| riir-neuron-db | 126 | 62 | 71 | 1 | 8 | 2 | 0 | **11** |
| riir-game-sdk | 36 | 0 | 9 | 1 | 9 | 0 | 0 | **10** |
| riir-train | 474 | 36 | 510 | 27 | 50 | 4 | 4 | **85** |
| seal-online-remaster | 1121 | 0 | 19 | 13 | 175 | 0 | 5 | **193** |
| poc-maxman | 9 | 2 | 7 | 2 | 0 | 1 | 0 | **3** |
| **TOTAL** | | | | **168** | **565** | **124** | **25** | **847** (approx) |

(Final scanner output may drift slightly due to EXTERNAL_PROJECT class
added mid-session: 58 refs correctly identified as external-project
references that should not be "fixed".)

## What remains (3 classes)

### A. AMBIGUOUS (~168 refs)

Basename matches multiple files in different crates. Mechanical strategies
exhausted — the remaining cases need **per-reference manual context
inspection**. The doc-crate-affinity and domain-keyword heuristics caught
the easy ones; the residue has ties or no keyword signal.

Examples: `src/main.rs` (16 refs, all in katgpt-rs docs referencing deleted
binaries), `src/pruners/bomber/players.rs` (9 refs, file genuinely gone).

### B. TRULY_GONE (~565 refs)

References to files that don't exist anywhere. Distribution:
- ~303 refs in 53 docs already covered by doc-level "historical paths" annotations
- ~262 refs scattered across 150+ docs (1-2 refs each — too small to annotate
  individually without noise; better treated as accepted drift)

The 3 highest-count docs (seal `009_layer7_client.md` 61 refs, seal
`002_layer0_foundation.md` 36 refs, riir-ai `292_worms_fft_*.md` 26 refs)
all have doc-level annotations explaining the design-plan nature.

### C. PREFIXED_INVALID (~86 refs) + CROSS_REPO_UNPREFIXED (~25 refs)

- PREFIXED_INVALID: refs with a `<repo>/` prefix where the path doesn't
  exist in that repo. Most are deleted-file references.
- CROSS_REPO_UNPREFIXED: 25 `src/lib.rs` / `src/types.rs` refs where the
  basename exists in 5+ repos — genuinely ambiguous.

## Recommended strategy for session 11 (if any)

The mechanical phase is **genuinely complete**. Remaining work is manual:

1. **Per-reference disambiguation** for AMBIGUOUS refs in non-annotated
   docs. Read each ref's surrounding context, pick the right candidate
   based on domain keywords and crate affinity. Time-intensive, low
   mechanical leverage.

2. **Accept residual TRULY_GONE in small-count docs** as historical
   design record. The doc-level annotations cover the high-count docs;
   individual refs in 1-2-ref docs are noise-level.

3. **Optionally**: prune `.plans/` docs for cancelled/superseded work
   entirely (per noise-reduction rule) — but this requires per-doc
   judgment about whether the design rationale is worth preserving.

**Do not run another mechanical pass** — diminishing returns. Session 10
applied 14 distinct strategies and converged at 847 stale. Further
automation will not meaningfully reduce the count.

## Verification (session 10)

- **Zero double-prefix regressions** — verified via
  `((?:crates|src)/[\w\-]+)/\1/` regex after every commit.
- **CRLF preservation** — all writes use binary mode + BOM detection.
- **No `.rs` source files touched** — only `.benchmarks/`, `.docs/`,
  `.plans/`, `.issues/`, `.research/` paths modified (verified via
  `git diff --name-only` after every commit).
- **EXTERNAL_PROJECT class added** — scanner now correctly skips refs
  to `mu-maxage-shop/`, `RuVector/`, `mmorpg-core/`, etc. (external
  projects, not workspace stale paths).

## Commits landed in session 10

53+ commits across 8 repos. Highlights:

| Repo | Strategy | Replacements | Commit examples |
|---|---|---:|---|
| All 7 | AMBIGUOUS_RESOLVED_SAME/CROSS resolution | 650 | (per-repo commits) |
| All 7 | PREFIXED_INVALID v2 (file-dir, basename) | 131 | (per-repo commits) |
| katgpt-rs | Bulk migration (types.rs/traits.rs split, etc.) | 230 | eff004e6, 6d37d525 |
| riir-ai | riir-gpu/forward.rs + auth/chain cross-repo | 35 | 88309396, 1d384540 |
| All 7 | AMBIGUOUS via uniqueness + crate affinity + keywords | 207 | 39f58a7f, 4ed43f58, 9ad6933d |
| All 7 | Cross-repo suffix match | 14 | edcba161, 3c70d88f, e8d466a |
| riir-ai, riir-chain | riir-chain neuron_db shard migration | 11 | 1f2059b1, 6ca5288 |
| seal-online-remaster | Seal-specific crypto/minimap paths | 12 | c941bb5 |
| All 7 | Annotation pass (>= 4 TG refs) | 32 docs | b4aeca9b, a5747174, 1f1091a, ... |
| All 7 | Tier-2 annotation (3 TG refs) | 21 docs | e0aece11, 613126a5, ... |

## See also

- Session 9 summary: previous version of this issue (74% reduction).
- Session 10 scripts: `/tmp/s10/` (ephemeral, cleared on reboot):
  - `scan.py` — unified cross-repo scanner + classifier (with EXTERNAL_PROJECT class)
  - `fix_ambiguous_resolved_same.py` / `fix_ambiguous_resolved_cross.py`
  - `fix_prefixed_invalid_v2.py`
  - `resolve_ambiguous.py` (v1-v5 variants)
  - `fix_specific_migrations*.py`
  - `fix_forward_migration.py`
  - `fix_shard_migrations.py`
  - `fix_seal_specific.py`
  - `fix_more_specific.py`
  - `annotate_truly_gone.py` / `annotate_tier2.py`
- AGENTS.md "Numbering Discipline" rule: numbers are monotonic and never reused.
