# Sibling doc-drift auditors — design record (Issue 702, closed 2026-09-01, file removed 2026-09-03)

Status: **historical record, RESOLVED.** Every closing condition of Issue 702 was
met and verified by real runs; the issue file was removed under the
noise-reduction rule and its reusable content lives here. The living usage doc
for the two auditors is [`doc_status_auditors.md`](doc_status_auditors.md); this
file is the *why* behind their current shape. Recover the full narrative with
`git log --all -- '.issues/702_*.md'` (last revision `467994d0`).

**Fix commits, in order:** `47ddd7df` (filed) · `0470742d` (riir-chain dialect
tokenizer) · `b6a5695e` (namespaced `crate/feature` tokens) · `89432b38`
(mirror blind spot bounded) · `3a57b1cf` (`(package, feature)` reachability
closure; found 2 stale labels in katgpt-rs itself) · `a90dd631` (docs gate
wiring + 185× walk speedup) · `d2228161` (`scripts/docs_drift_sweep.py`) ·
`7bb438e3` + `ed5f4865` (CI/workstation manifest-population fixes) ·
`467994d0` (cadence CLOSED). Sibling-side fixes: riir-neuron-db `c06133e`,
riir-clippy `7736e30` + `1df3599`, riir-ai `95354cd62` + `2b14457cb`,
riir-chain `08eb27d9`, riir-neuron-db `b9f87b4`.

## The finding, in one sentence

`scripts/bench_doc_audit.py` and `scripts/cargo_comment_audit.py` accept a repo
path and audit any repo, and for months nothing pointed them at any repo but
katgpt-rs — so a sibling with stale labels was indistinguishable from a sibling
nobody had checked. Four confirmed stale GOAT-record labels were found across
three siblings (riir-ai ×2, riir-clippy, riir-neuron-db), plus two in katgpt-rs
once the closure was fixed. All six corrected.

## Design decisions the auditors now embody

### 1. A "0 labels" verdict must be explained, not trusted

The auditor could not read riir-chain's dialect at all: `(` and `*` were
missing from the status-boundary lookahead, so the lazy status group never
terminated and 26 benchmark docs audited as "0 labels, 0 mismatches". Fixing the
boundary class added **+27 labels** workspace-wide — 21 of them in katgpt-rs's
own corpus, never read before.

Every remaining 0-label repo was then chased to the end: 7 have zero `Feature:`
headers (correct zero), riir-game-sdk writes scope notes rather than status
claims (correct skip). **A zero is a hypothesis until its cause is named.**

### 2. Namespaced labels were three stacked defects

`` `crate/feature` `` tokens were unreadable for three independent reasons, each
of which alone produced silently fewer labels: `/` absent from the token name
class; a lazy status capture stopping at the first comma of a compound
parenthetical; `+` absent from the status class. Fixed with a **monotone**
widened retry (`widened_status`) that fires only where the tight capture
returned `unknown` — it can promote unknown → known and can never change a
verdict already reached. Of the 16 residual `unknown` namespaced tokens, 15 are
correct (forwarding *targets* inside another feature's parenthetical).

### 3. Widening reach adds false positives as readily as findings

Two were created and caught before shipping:

- **Forwarded-only defaults.** A feature off in its owning crate's `default`
  but on in a default build via an ancestor's forwarding edge. Both readings are
  defensible, so the opt-in-vs-default mismatch is suppressed — **but only for a
  label that scopes its claim** (a namespaced token). A BARE token in a
  repo-level doc is a deployed claim and gets no suppression; a blanket rule
  would have excused the two katgpt-rs drifts below.
- **Cross-crate name collapse.** Collapsing `pkg/feat` to `feat` is right for
  the deployed model and wrong for the own-crate model (riir-engine had a
  default `se2_equivariant = ["katgpt-core/tropical_algebra"]` *and* a local
  non-default `tropical_algebra`). Only a bare entry activates a same-crate
  feature; `local_default_closure` enforces that.

A third suppression is textual (`SCOPED_CLAIM_RE`: "opt-in in this crate").
Every suppressed bucket is **counted in the `[skipped: …]` tail**, never
silently dropped.

### 4. The default closure was wrong in BOTH directions

Per-manifest closure + union **under**-approximated (could not follow
`riir-games-civ/default → osc_emotion → riir-games-shared/osc_emotion → osc_npc`
across a manifest boundary) and **over**-approximated (`pkg/feat` collapse; 47
bogus names in riir-ai alone). Replaced by reachability over `(package,
feature)` nodes from every member's `default`, not following an edge whose
target package is outside the repo. Measured across 18 repos × both auditors it
changed exactly one verdict (a riir-ai false positive) and exposed two stale
katgpt-rs labels reachable only in three hops through a `pkg/feat` edge
(`hla_eigenbasis_recovery` via `ica_lens`; `still_kv` via `chain_fold`).

### 5. Three status words, not two

"opt-in" / "default-on" cannot describe a feature with no bare `default[]` entry
that a default build still enables through a chain. riir-neuron-db's README
already had a §"Transitive default" section, so **`on by transitive default`**
is now transition vocabulary, matched as the FULL phrase — a looser
`transitive.*default` escaped the negation guard by matching six characters
past a "NOT".

### 6. Pins are vacuous until canaried

Two of the first four `selftest()` pins passed with their fix reverted: one was
guarding a **duplicate** of the status logic (selftest re-implemented it
inline), the other was the wrong inverse. A second DRY defect surfaced later —
`parse_terminal_transition` applied outside the shared `classify_token`. Both
fixed by routing every status rule through the one path. All seven pins now
exit 1 naming the shape when their fix is reverted; an `own_default ⊆
deployed_default` invariant warns rather than mis-suppresses.

### 7. The mirror blind spot was bounded, not caveated

The `pkg/feat` collapse inflates the deployed default set, which could hide the
"doc says DEFAULT, manifest says no" branch. Measured by intersecting
externally-collapsed names with docs claiming DEFAULT for one: katgpt-rs **0**,
riir-ai 1 (a correct doc naming the owning repo), everyone else 0. Real in
structure, **empty in population** — a re-measure only re-runs the
intersection.

## Cadence — three tiers, none subsuming another

| instrument | where | cadence | scope |
|---|---|---|---|
| `docs_gate.yml` / `docs_gate.sh` | CI + workstation | per-push | katgpt-rs only |
| `sibling_docs_drift.yml` (`workflow_call`) | sibling CI | caller's choice | one caller |
| `scripts/docs_drift_sweep.py` | workstation | on demand | every contract repo |

Wired and **dispatched** (logs read, figures matching the local sweep) for
riir-ai, riir-chain, riir-neuron-db; riir-clippy runs both auditors by path in
its weekly `rust.yml`. This was blocked until `.issues/704` moved default
branches to `develop` — before that, three workflow files would have been inert
decoration. riir-train, riir-mmorpg-examples and riir-dapps ship no CI at all
(an Actions-budget owner call, recorded as the "or records why not" branch)
and are covered by the workstation sweep.

**Running the dispatch caught two bugs review had not**, and they are the same
bug pointed opposite ways: the auditors' own checkout inside the audited tree
folded a foreign repo's manifests IN (riir-neuron-db read 396 Cargo comments —
katgpt-rs's count; fixed structurally, sibling checkouts); untracked local
manifests (a `.container-src/` copy in riir-chain) fed the default closure
locally only (fixed by reading `git ls-files`). Two instruments that must agree
is what found both.

## Not done from here, and why

Sibling docs and CI are owned per `BOUNDARY.md`; every sibling-side fix above
was landed by or coordinated with that repo, and this record does not claim
them. `scripts/docs_drift_sweep.py` is deliberately NOT in `docs_gate.sh`'s
`CHECKS`: CI has a single checkout and would print a confident green over zero
repos.
