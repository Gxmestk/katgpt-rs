# Issue 703 — every instrument that enumerates the repo set by hand has gone blind

Status: **OPEN (1 proposal, 4 instances already fixed)** — filed 2026-09-01 from
three independent hits in one session. The fixes landed; what is open is whether
to gate the *class*, because nothing stops the fifth instance.

## The class

An instrument (skill, script, doc paragraph) that needs "all the repos" writes
the list out. Repos gain contracts; the list does not. The instrument keeps
reporting clean over a set that no longer matches the workspace — and a clean
result over a partial set is indistinguishable from a clean result over the
whole one.

The workspace has **18** repos with a root `BOUNDARY.md` (measured 2026-09-01;
15 at the 2026-08-21 run). Derived, never typed:

```bash
cd /Users/katopz/git && for d in */; do
  [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
done
```

`-d "$d/.git"`, not `-e`: a `git worktree` has a `.git` **file**, and including
one duplicates every hit of the repo it shadows.

## Measured instances (all in one session, none looked for on purpose)

| instrument | claimed | actual | consequence | fixed in |
|---|---|---|---|---|
| `substrate-first` SKILL Step 2 grep | hard-coded 7-repo brace list | **7 of 18** | the anti-duplication gate could not see `riir-armageddon` (which consumes `GenericSpatialBelief` in 3 files) or `riir-dapps` — both **product-set** repos. A gate that cannot see a repo cannot say whether it consumes substrate or duplicates it | `60655c48`, audited clean in `b39fdb9d` |
| `doc-sync` SKILL repo table | header "14", table 12 rows | **12 of 18** | the table IS the work list, so 6 repos were never synced | `87f28fc0` |
| `AGENTS.md` §"Repo count" | "the workspace is **15 repos**" | **15 of 18** | the paragraph ends *"Read a count in prose as a claim, not a fact"* | `b3718986` |
| `proposal` / `goat-audit` / `feature-gate-audit` / `research` SKILLs | "the 7-repo stack" | **7 of the 8 product repos** | each kept a private copy of the product set omitting `riir-dapps` (added 2026-08-20), so settlement-**composition** work routes to `riir-chain`, which owns only value/authority primitives | `60655c48` |

Two axes, and both drifted independently: the **product/distillation set** (8 —
routing targets) and the **workspace** (18 — search scope). Canonical home for
both: `AGENTS.md` §"Repo count".

## Why a gate is not obvious (the reason this is a proposal, not a fix)

The naive check — grep every instrument for an `N repos` claim and compare
against the derived count — **false-positives on history**. `doc-sync` and
`boundary-guard` keep run-log tables whose rows correctly say *"all 7 repos"*
about a run that did cover 7. Rewriting those would destroy the record. A gate
that cries wolf on an accurate historical row is one somebody loosens.

Two shapes that avoid it, in cost order:

1. **Grep for the mechanical defect, not the count.** A hard-coded brace list
   (`/Users/katopz/git/{a,b,c}`) or a multi-repo path enumeration inside a
   *fenced command block* in `.agents/skills/*/SKILL.md`. Run-log rows are
   prose, so they do not match. Cheap, and it catches the exact failure all four
   instances share. **Recommended.**
2. **Have each instrument print its derived set**, and gate on the *absence* of
   a derivation rather than on a number — the `.issues/842` liveness rule
   (a check that examined zero repos must not read like one that examined 18)
   applied to skills instead of scripts.

Not proposed: rewriting historical run-log counts. They are measurements.

## Related

- `.issues/702` — the doc-drift auditors run in one repo of eight. **Same
  narrow-scope class one level up:** its own sweep for sibling copies of
  `bench_doc_audit.py` covered "the 7 siblings". Re-checked 2026-09-01 across
  all 18: still zero copies (only `katgpt-rs/scripts/` has them), so 702's
  conclusion holds — but it held by luck, not by coverage. Note `riir-burner`
  carries 15 scripts of its own and was outside that sweep.
- `riir-ai/.issues/842` — the liveness rule this borrows (print the population).
- `riir-chain/BOUNDARY.md` D2 (closed) — the same shape in a lockfile: two build
  roots, one invisible to the checker that existed.
