# Issue 703 — every instrument that enumerates the repo set by hand has gone blind

Status: **CLOSED (gate shipped, shape 1)** — filed 2026-09-01 from three
independent hits in one session; the gate that was the open question shipped the
same day as `scripts/skill_repo_set_gate.py`, wired into the per-push docs gate.
On its first real run it found **4 more instances nobody had looked for**,
including one in the file that had just been "fixed".

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
| `substrate-first` SKILL Step 2 grep | hard-coded 7-repo brace list | **7 of 18** | the anti-duplication gate could not see `riir-armageddon` (which consumes `GenericSpatialBelief` in 2 files / 6 sites — re-measured 2026-09-01; this row said 3 files, which was itself an unverified count inside an issue about unverified counts) or `riir-dapps` — both **product-set** repos. A gate that cannot see a repo cannot say whether it consumes substrate or duplicates it | `60655c48`, audited clean in `b39fdb9d` |
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

## What shipped, and what it found (2026-09-01)

Shape **1**. `scripts/skill_repo_set_gate.py`, in `scripts/docs_gate.sh` (4th
check, +0.03 s on a 3.08 s gate) and `.github/workflows/docs_gate.yml`.

It flags, inside a **fenced block**, two or more repos used as **path
components** (`riir-ai/…`) or a brace list (`git/{a,b}`). Prose naming a repo
does not match, because prose does not put a slash after the name — which is
what keeps the corrected `substrate-first` block (whose *comment* names
`riir-armageddon` and `riir-dapps` while its *command* derives the set) from
firing. A deliberately narrow block declares itself:
`<!-- repo-set-ok: <reason> -->`.

### Four instances found on the first run

| instrument | typed | of | consequence, measured |
|---|---|---|---|
| `substrate-first` **"Audit Step 2"** | 2 (brace list ×5) | 18 | **The same file `60655c48` had just fixed.** That commit corrected the step *named* "Step 2" and left this one — the step that actually hunts duplicate implementations. Deriving the set: **256 hits vs 6**, 42×, 1.0 s |
| `proposal` Layers A–E | 4 | 18 | prior-art grep blind to **784 of 2,509 documents** (31%); `.issues` worst at 21 of 145 (**85% invisible**). Prose above it claimed "all seven repos" while the commands covered four |
| `feature-gate-audit` §Scope | 8 rows under a "**7** repos" header | 18 | listed `seal-online-remaster/`, **which does not exist** (the dirs are `seal-online-remaster-unity/` and `seal-game-editor/`) — audits a workspace that is not this one |
| `goat-audit` Layers 1–2 | 3 | 18 | **not drift** — it matches the reasoned §"Repos in scope". Kept, marked `repo-set-ok`, and the marker was verified load-bearing (removing it re-fires the gate). Its `riir-ai/**/*.toml` globstar *was* a real bug: bash without `shopt -s globstar` expands `**` as `*`, so it checked one level. Now `-r --include` |

### Two bugs found in the gate itself, by its own canaries

Both would have produced exactly the vacuous green the issue is about.

1. **Naive fence toggle mis-phases.** Flipping state on every ` ``` ` line
   inverts after an unterminated fence, scanning the *complement* — prose as
   code, code as prose — and reporting clean either way.
   `rust-optimize/SKILL.md` had an unclosed ` ```text ` at line 511 (43 lines
   swallowed, mis-rendered) and it ate the gate's first canary, which is how
   this was found. Now: a closer must be a bare run ≥ the opener's, and an
   unterminated fence is **reported**, not dropped. Fence closed in this commit.
2. **The vocabulary was the derived set, so CI was structurally vacuous.** In a
   lone checkout only `katgpt-rs` is derivable, so `riir-ai/src riir-chain/src`
   is not recognisable as repo names at all — a permanent green. Fixed by
   splitting **vocabulary** (committed `scripts/repo_set.txt`, 18 names) from
   **population** (the SKILL.md actually visible: 12 locally, 8 in CI). Both are
   printed; the workstation run re-derives and **fails on snapshot drift**.

Verified by canary, exit codes recorded: detector fires (1), marker suppresses
(0), unterminated fence fires (1), stale snapshot fires (1), CI-shaped lone
checkout still detects (1) and passes clean (0).

### What this gate deliberately does NOT see

A fenced **scope table** listing repo names *without* paths (`goat-audit`
§"Repos in scope"). That is prose-shaped; catching it is the false-positive-prone
case declined above. `feature-gate-audit`'s was found by human read, not by the
gate. Shape 2 remains unbuilt.

## Related

- `.issues/702` — the doc-drift auditors run in one repo of eight. **Same
  narrow-scope class one level up:** its own sweep for sibling copies of
  `bench_doc_audit.py` covered "the 7 siblings". Re-checked 2026-09-01 across
  all 18: still zero copies (only `katgpt-rs/scripts/` has them), so 702's
  conclusion holds — but it held by luck, not by coverage. Note `riir-burner`
  carries 15 scripts of its own and was outside that sweep.
- **A 5th and 6th instance, found 2026-09-01 while re-measuring Issue 701 R2:**
  (a) R2's own survey table covered **12 of 18** repos, so 9 repos with no CI at
  all were counted as 5; it is now produced by `scripts/ci_gate_coverage.py`
  rather than typed. (b) `riir-clippy/.github/workflows/ops_dashboard.yml` is
  built around a hard-coded **"the 11 repos"** generator list against a
  workspace of 18. Not fixed from here — riir-clippy owns it and an agent was
  active in it. This is the class continuing to produce instances *after* the
  gate shipped, which is expected: the gate covers `SKILL.md` command blocks
  only, not workflows or Rust source.
- `riir-ai/.issues/842` — the liveness rule this borrows (print the population).
- `riir-chain/BOUNDARY.md` D2 (closed) — the same shape in a lockfile: two build
  roots, one invisible to the checker that existed.
