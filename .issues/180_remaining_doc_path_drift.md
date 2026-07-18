# Issue 180 — Remaining Documentation Path Drift (Cross-Repo Scope)

> **Date:** 2026-07-18
> **Status:** OPEN
> **Scope:** All 7 private repos + katgpt-rs
> **Origin:** Discovered during session 8 of the cross-repo benchmark/doc cleanup task.

## TL;DR

Session 8 of the benchmark-cleanup umbrella task expanded scope from
bench-only DRY consolidation (sessions 1-7) to **documentation path drift**
— `.benchmarks/*.md` / `.docs/*.md` / `.plans/*.md` files referencing
file paths that no longer exist due to crate-split refactors.

Two large classes of drift were detected and **bulk-fixed** in this session:

1. **Issue 002 4-layer split in riir-ai** — `crates/riir-games/{src,tests,benches}/`
   → `crates/riir-games-{civ,quest,shared}/{src,tests,benches}/`.
   **213 path renames** across 96 .md files in riir-ai.

2. **katgpt-rs crate split** — `src/<X>/Y.rs` → `crates/katgpt-<X>/src/Y.rs`
   where the X→katgpt-X mapping is known. **568 path renames** across 401
   .md files in katgpt-rs, plus smaller batches in riir-chain (8),
   riir-neuron-db (3), riir-game-sdk (1), riir-train (284), and
   seal-online-remaster (171).

Three classes of drift remain **open** in this issue:

- **A. Cross-repo references** — `.md` in repo X references paths that
  live in repo Y (e.g. riir-ai docs referencing `crates/riir-chain/src/...`).
  Resolution requires building a unified file index across all 8 repos.
- **B. Ambiguous basename refs** — paths like `src/foo/types.rs` or
  `src/foo/mod.rs` where the basename matches multiple files. Need
  per-reference context inspection (e.g. reading surrounding markdown)
  to disambiguate.
- **C. Truly-gone files** — refs to files that don't exist anywhere
  (deleted without doc update). Need manual annotation or doc removal.

## Counts (per repo, post-session-8)

| Repo | Remaining stale refs | Notes |
|---|---:|---|
| katgpt-rs | 338 | Mix of cross-repo (crates/katgpt-core/src/... from sibling docs), ambiguous (mod.rs/types.rs), and truly-gone (e.g. `src/pruners/bomber.rs`, `src/blue_bear_pruner.rs`). |
| riir-ai | 883* | Mostly cross-repo (`crates/katgpt-core/src/...`, `crates/riir-chain/src/...`) + 371 truly-gone (e.g. `src/catchup/*` moved to riir-chain, `riir-chain/src/encoding/latcal.rs` likely moved). *The 883 count is inflated by cross-repo refs that DO exist, just not in riir-ai. |
| riir-chain | 5 | Truly-gone (e.g. `riir-chain/crates/riir-chaind/src/ring.rs`, `riir-chain/crates/riir-chaind/src/harness.rs`). |
| riir-neuron-db | 15 | Mix; most look like `src/catchup/*` which moved to riir-chain. |
| riir-game-sdk | 4 | Small residue. |
| riir-train | 229 | Mostly `src/<crate>/...` (engine/gpu split); needs same crates/-prefix strategy as seal-online-remaster. |
| seal-online-remaster | 215 | Residue from non-crates path conventions (e.g. `<crate>/...` without `src/`). |
| poc-maxman | 8 | Small; mostly `src/<module>/mod.rs` ambiguity. |

## Recommended strategy for session 9

### Phase 1 — Cross-repo resolver (highest impact)

Build a unified file index across all 8 repos on disk:

```python
all_files = {}  # rel_path -> repo
for repo in REPOS:
    for root, _, names in os.walk(repo):
        for n in names:
            if n.endswith('.rs'):
                rel = os.path.relpath(...)
                all_files.setdefault(rel, []).append(repo)
```

Then for each stale ref in repo X's .md files:
1. Check if the ref exists verbatim in any other repo (the doc was probably
   written from the perspective of that other repo).
2. If yes, either rewrite with explicit repo prefix (e.g.
   `katgpt-rs/crates/katgpt-core/src/...`) or add an annotation noting the
   cross-repo location.

### Phase 2 — Ambiguous refs

For each ambiguous basename ref (e.g. `src/foo/types.rs`), inspect the
surrounding markdown context (the lines around the ref) and use
domain-specific keywords (e.g. "ShardIndex", "NpcClr", "KARC") to pick
the correct candidate.

### Phase 3 — Truly-gone

For each truly-gone ref, decide per-file:
- If the .md is a historical record (like the Tucker or blob_store docs),
  leave alone (intentional).
- If the .md claims current behavior, add an artifact note at the top
  explaining the referenced file is gone.
- If the .md is itself stale and the work it describes is dead, remove
  the .md per the noise-reduction rule.

## Verification

Each phase should:
1. Re-run the stale-ref detector to confirm count drops.
2. Spot-check 10 random renames per phase for correctness.
3. Verify line-endings preserved (seal-online-remaster uses CRLF for
   some files; detect with `file <path>` and write back accordingly).
4. Verify no `.rs` files touched (`git diff --name-only | grep -v
   '^\.benchmarks/\|^\.docs/\|^\.plans/\|^\.issues/'` should be empty).

## Commits landed in session 8 (for reference)

| Repo | Commit | Description |
|---|---|---|
| katgpt-rs | `2f319bd3` | 3 dangling bench refs (005_g_zero, 014_epiplexity, 036_asymmetric) |
| katgpt-rs | `b5e6c81b` | 401 .md files, 1538 path renames (crate split sync) |
| riir-ai | `3357a59e` | 75 .md files, 229 path renames (Issue 002 4-layer split) |
| riir-ai | `9550acfe` | 24 .md files, file→dir module refactors + admitted_ops merge note |
| riir-ai | `92520a1a` | 1 .md file, riir-games-shared -> riir-games path fix |
| riir-train | `4d0bd1c` | 264_lclm_lora_goat missing-artifact annotation |
| riir-train | `eecbd4e` | 86 .md files, 284 path renames |
| riir-chain | `d02a5d6` | 4 .md files, 8 path renames |
| riir-neuron-db | `c52ae7e` | 3 .md files, 3 path renames |
| riir-game-sdk | `462158b` | 1 .md file, 1 path rename |
| seal-online-remaster | `90d31ed` | 27 .md files, 171 path renames (CRLF preserved) |

## See also

- Session 7 summary (the prior umbrella): `riir-neuron-db` commit `d69a84c`
  splitmix64 BenchRng extraction.
- AGENTS.md "Numbering Discipline" rule: numbers are monotonic and never
  reused, even after a file is removed. Issue 180 is the next free number
  after `.highwater` = 179.
