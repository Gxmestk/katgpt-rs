# Issue 702 — the doc-drift auditors run in ONE repo of eighteen, and three siblings carry confirmed stale labels

Status: **RESOLVED 2026-09-01 — drift CLOSED, cadence CLOSED and verified by
real runs.** `.issues/704` unblocked it the same day: with the default branch
moved to `develop`, `riir-ai`, `riir-chain` and `riir-neuron-db` each carry a
weekly `docs_drift.yml` calling katgpt-rs's reusable workflow, and all three
were **dispatched and inspected**, not merely committed. The label-bearing
repos with no CI at all record why not (below). Kept rather than removed under
the noise-reduction rule: it is the canonical record of the auditors' design
decisions and AGENTS.md cites it. (All 4 original drift rows CLOSED — riir-neuron-db `c06133e`,
riir-clippy `7736e30` (+ weekly docs-drift CI in `1df3599`), riir-ai
`osc_emotion` + `band_edge_trigger` in riir-ai `95354cd62`. Reopened in scope,
not in status: replacing the per-manifest closure with a `(package, feature)`
reachability graph exposed **2 previously invisible drifts in katgpt-rs
itself**, both fixed here. Every one of the 18 contract repos now audits at 0
mismatches. Remaining: the cadence condition for the other label-bearing
siblings)** —
filed from katgpt-rs because the tooling lives here; the fixes belong in the
owning repos. Re-measured 2026-09-01 across all **18** contract repos (the
original table covered 8), which turned up something the wiring gap was hiding:
**the auditor could not read one sibling's dialect at all**, and a repo it
cannot read reports "0 labels, 0 mismatches" — indistinguishable from clean.

`scripts/bench_doc_audit.py` and `scripts/cargo_comment_audit.py` accept a repo
path and audit any repo. As of `a90dd631` katgpt-rs runs them per-push via
`.github/workflows/docs_gate.yml`. **No other repo runs either one**, and until
that commit neither ran anywhere — both were RED in katgpt-rs itself on false
positives (case-sensitive `Opt-in` regex; first-status-wins label parsing).

## The auditor was blind to a whole dialect (found 2026-09-01, fixed here)

`riir-chain` writes its labels as

    **Feature:** `chain_block_producer` — **default-OFF** (implies `chain_catchup` …)

The *vocabulary* was fine — `parse_status_phrase("default-OFF")` already
returned `opt-in`. The **tokenizer** was not: `(` and `*` were missing from the
status-boundary lookahead, so the lazy status group could never terminate and
the line produced **no token at all**. 26 riir-chain benchmark docs audited as
"0 labels, 0 mismatches".

Fixed by adding `(` and `*` to the boundary class. Measured before → after:

| repo | labels before | after | mismatches |
|---|---|---|---|
| katgpt-rs | 72 | **91** | 0 → 0 |
| riir-ai | 59 | **62** | 1 → 1 |
| riir-neuron-db | 8 | **10** | 0 → **1** (new, verified real — see below) |
| riir-train | 4 | **5** | 0 → 0 |
| riir-mmorpg-examples | 1 | **2** | 0 → 0 |
| riir-chain | 0 | **1** | 0 → 0 |

+27 labels, and **the bug was in this repo's own corpus too** — 21 katgpt-rs
tokens of the form `` `bar` (**DEFAULT-ON** since …) `` had never been read.
All 21 classify as `default` and all agree with the manifests.

A `selftest()` now runs on **every** invocation and pins four real line shapes
(both dialects, plus the `opt-in, NOT default-on` negation case). Canaried:
reverting the boundary class makes it exit 1 with the offending shape. Without
it a regex regression stays silent — the audit keeps printing "0 mismatches"
over fewer labels, which is exactly how riir-chain went unnoticed.

### "0 labels" now has a taxonomy — most of it is NOT blindness

Of the 8 repos still reporting 0 labels, **7 have zero `Feature:` headers**:
their docs simply do not use the convention, so 0 is the right answer.
`riir-game-sdk` has 9 headers but writes `` `game_e2e` (chain-free) `` — a
scope note, not a status claim, so skipping it is also correct. **No remaining
blindness was found**; 18 namespaced `crate/feature` tokens (e.g.
`` `riir-wallet/siwr` ``) are still unreadable and are the one known gap left.

Good news next: there are **no sibling copies** of these scripts (re-checked
2026-09-01 across all 18 contract repos — the first pass covered 7, and held by
luck rather than by coverage; note `riir-burner` carries 15 scripts of its own
and was outside it). They are invoked by path, so the fixes in
`a90dd631` apply workspace-wide with no sync problem. The gap is purely that
nothing *invokes* them outside this repo.

## Measurement (2026-09-01, post-fix auditors, read-only)

| Repo | bench-doc labels | mismatches | Cargo comments | mismatches |
|---|---|---|---|---|
| katgpt-rs | 91 | 0 | 396 | 0 |
| riir-ai | 62 | **1** | 10 | 0 |
| riir-clippy | 9 | **2** | 0 | 0 |
| riir-neuron-db | 10 | **1** | 0 | 0 |
| riir-train | 5 | 0 | 0 | 0 |
| riir-mmorpg-examples | 2 | 0 | 0 | 0 |
| riir-chain | 1 | 0 | 4 | 0 |
| riir-dapps | 1 | 0 | 0 | 0 |
| katgpt-web, riir-armageddon, riir-auth, riir-burner, riir-dao, riir-deployer, riir-game-sdk, riir-unity, riir-viewbridge, seal-game-editor | 0 | 0 | 0 | 0 |

(The original version of this table listed 8 repos — the product set — not the
18 contract repos. The 10 added rows are all clean, so the conclusion held, but
it held without having been checked. Same class as `.issues/703`.)

**Superseded 2026-09-01 (second pass).** The counts above were taken with a
tokenizer that could not read a `` `crate/feature` `` token at all. After the
three fixes in §"Namespaced labels" the label counts are katgpt-rs **92**,
riir-ai **77**, riir-chain **2**, riir-neuron-db **11**; riir-ai's mismatch
count went 1 → **2**. Every other row is unchanged. Treat the table above as
the first-pass record, not the current state.

A zero above is not automatically a clean bill of health — it can mean no label
matched the recognised forms at all. That caveat was written when it was only a
suspicion; it has now been chased to the end for every zero row, and the answer
is in §"'0 labels' now has a taxonomy" above: one real tokenizer bug (fixed),
and the rest are repos that legitimately do not use the convention.

## Confirmed drift — verified against the manifests, not taken from the auditor

Both were re-checked by walking every `Cargo.toml` and grepping the actual
`default` arrays, because a union-closure verdict can be over-broad:

1. **riir-ai** — `.benchmarks/516_osc_emotion_bridge_goat.md:7` says
   `` `osc_emotion` (opt-in; implies `civ_emotion` + L0 `osc_npc` substrate) ``.
   `osc_emotion` IS in `crates/riir-games-civ/Cargo.toml` `default[]`.
2. **riir-clippy** — `.benchmarks/045_fix_compile_goat_corpus212.md:5` and
   `.benchmarks/041_rustc_errors_cargo_fix_floor.md:6` both say
   `` `rustc_errors` (opt-in, default-off) ``. `rustc_errors` IS in the ROOT
   `Cargo.toml` `default[]`.

3. **riir-neuron-db** — `.docs/03_freeze_thaw/frozen_envelope.md:9` says
   `` **Feature gate:** `merkle_freeze` — **opt-in** ``. It is not: the
   default-on `experience_graph` lists `merkle_freeze` directly
   (`experience_graph -> merkle_freeze`, one hop), so it ships enabled. Surfaced
   only by the tokenizer fix above, then verified by walking `default[]` and
   computing the closure by hand rather than trusting the auditor's verdict.

   **FIXED 2026-09-01 in riir-neuron-db `c06133e`** (repo was clean and idle, so
   editing it collided with nobody). The README was drifted the same way and in
   the more telling manner: it filed `merkle_freeze` under §"Opt-in" while the
   repo *already has* a §"Transitive default" section built for exactly this
   case, holding one row. Row moved there and the `\*` footnote generalised to
   cover both. Five other features also imply `merkle_freeze`; only
   `experience_graph` is itself default-on. riir-neuron-db now audits 11 labels
   / **0 mismatches**.

This is the stale-flag-state class: the feature was promoted and the benchmark
doc that gated the promotion never got updated. It matters because those docs
are the GOAT record — a reader deciding whether a primitive is production
default reads the benchmark, not the manifest.

## Namespaced labels: three more layers of the same blindness (2026-09-01)

The first pass closed with "18 namespaced `crate/feature` tokens are the one
known gap left". Closing it turned up that the gap was three stacked defects,
each of which alone produced *silently fewer labels and still "0 mismatches"*:

1. **`/` was absent from the token name class.** `` `katgpt-core/gaussianity_probe` ``
   yielded no token. The *vocabulary* for it already existed
   (`_parse_feature_spec` has normalised `crate/foo` since the first version) —
   only the tokenizer lacked it. Exactly the shape of the riir-chain dialect bug
   above, one layer down.
2. **The status capture is lazy and stops at the first comma.** A compound
   parenthetical — `` `riir-wallet/siwr` (client + RP kit, default-OFF) `` —
   captured `"client + RP kit"` → `unknown` → discarded, with the real status
   word one comma further on. Now retried over the remainder of the enclosing
   clause (`widened_status`). The retry fires **only** where the tight capture
   already returned `unknown`, so it is monotone by construction: it can
   promote unknown → default/opt-in and can never change a verdict the
   tokenizer already reached.
3. **`+` was absent from the status class**, which is why (2)'s line produced no
   token to retry in the first place.

Measured effect: **+6 labels** from (1) alone, **+21 labels** with (2) and (3)
— riir-ai 62 → 77, katgpt-rs 91 → 92, riir-chain 1 → 2, riir-neuron-db 10 → 11.

### Most `unknown` verdicts were correct, and that had to be checked too

Of the namespaced tokens, 16 still parse as `unknown` — and for 15 of them that
is the **right** answer: they are forwarding *targets* named inside another
feature's parenthetical (`` `ruliology` (opt-in, root forwards to
`katgpt-ruliology/ruliology` + …) ``), where the bare feature is the actual
label and is already read. Only the riir-chain siwr line was a primary label.

## Two false positives this work created, caught before shipping

Both were produced by the new coverage, and both would have red-flagged a doc
that is **correct**. They are recorded because the near-miss is the lesson:
widening an auditor's reach adds false positives as readily as findings, and the
script's own comments already warn that false positives are how a gate earns a
reputation for noise and gets ignored.

1. **Forwarded-only defaults.** `riir-wallet/siwr` is off in `riir-wallet`'s own
   `default`, and on in a default build only because `riir-chaind`'s
   `default -> chain_siwr -> riir-wallet/siwr` pulls it in. Both readings are
   defensible, so "default-OFF" is not drift. Since own-crate-default ⊆
   deployed-default always, this collapses to one rule: **suppress the
   opt-in-vs-default mismatch when a feature is default only by forwarding.**
   New `find_own_crate_defaults` / `local_default_closure` compute the strict
   subset; the count is reported, never silently dropped.
2. **Cross-crate name collapse.** `_parse_feature_spec` collapses `pkg/feat` to
   `feat` — correct for the deployed model, wrong for the own-crate model.
   riir-engine has `se2_equivariant = ["katgpt-core/tropical_algebra"]` in its
   `default` (Cargo.toml:503) *and* a separate local
   `tropical_algebra = [...]` (:2085) that is **not** in default. Collapsing
   credited the local flag as default-on and reported confident drift on
   `` `tropical_algebra` (riir-engine, opt-in) ``, which is right. Only a BARE
   entry activates a same-crate feature; `local_default_closure` now enforces
   that, and it is pinned (see below).

A third suppression is textual: a label may scope its own claim
(`` `npc_sleep_time` (sleep_time_catalog, opt-in in this crate — see npc.md
§Feature gate landscape for the layered gate split) ``). That doc is describing
a layered split it is fully aware of; the flat repo-wide model cannot adjudicate
it. `SCOPED_CLAIM_RE` skips and counts it.

`checked N labels, M mismatches` now carries a `[skipped: …]` tail naming every
suppressed bucket (cross-repo / forwarded-only / crate-scoped), so none of the
three is invisible.

### The pins were vacuous until canaried — two of four

`selftest()` gained the namespaced + compound shapes and an own-default case.
Canarying all four by reverting each fix in a copy, **two passed anyway**:

- The widened-retry pin was guarding a **duplicate**: `selftest` re-implemented
  the status logic inline, so breaking the iterator's copy left it green. Both
  now call one `classify_token`.
- The own-default canary was the wrong inverse — the defect is the *collapse*
  via `_parse_feature_spec`, not merely admitting namespaced entries. Re-run
  against the real pre-fix code it fires.

All four now exit 1 naming the shape. An uncanaried pin is an unknown, not a
pass — the same rule this repo's AGENTS.md applies to uninvoked assertions.

## New confirmed drift (4th), and one non-finding

4. **riir-ai** — `.benchmarks/498_hla_band_edge_quality.md:6` says
   `` `riir-engine/band_edge_trigger` (Plan 498 Phase 1, opt-in) ``.
   `band_edge_trigger` IS a bare entry in `crates/riir-engine/Cargo.toml`
   `default[]`. Corroborated inside riir-ai's own corpus: `498_hla_band_edge_burn_in.md`
   records "**promoted to DEFAULT-ON by this gate**" and adds it to the default
   list, and `551_band_edge_level_playing_field.md:7` says "default-on since
   Plan 498 Phase 4". The Phase-1 doc was accurate when written and was never
   updated — the same stale-GOAT-record class as items 1–3, and the reason the
   in-line `parse_terminal_transition` rule is not enough: here the promotion is
   recorded in a *different* document.

**Not a finding:** `tropical_algebra` (riir-ai) — see false positive 2 above.
Verified by hand against riir-engine's manifest; the doc is correct.

### The mirror blind spot, measured rather than left as a caveat

The `pkg/feat` -> `feat` collapse is only *wrong* when the qualifier names a
crate outside the repo. It is also asymmetric: it INFLATES the deployed default
set, which suppresses the other branch — "doc says DEFAULT but the feature is in
no default array". So the fix above could have left real drift hidden.

Measured across the repos that have labels, by counting names that reach the
deployed set ONLY through an external qualifier, then intersecting with docs
that claim DEFAULT for one of them:

| repo | externally-collapsed names | docs claiming DEFAULT for one | real drift |
|---|---|---|---|
| katgpt-rs | **0** | 0 | 0 |
| riir-ai | 47 | 1 | **0** (doc is correct) |
| riir-neuron-db | 4 | 0 | 0 |
| riir-chain | 1 (`esp-println/jtag-serial`) | 0 | 0 |
| riir-clippy | 0 | 0 | 0 |

katgpt-rs having **zero** is the load-bearing row: this repo's "0 mismatches
over 92 labels" is not resting on the collapse. The single riir-ai hit is
`.benchmarks/153_karc_g3_anticipation_salience.md:7`, which writes
`` `salience_tri_gate` (katgpt-rs, default-on) `` — it names the owning *repo*
explicitly, and the claim checks out (own-crate default-on in katgpt-core).
A correct doc, not a suppressed finding.

So the residual blind spot is real in structure and **empty in population**.
Recording the bound rather than the caveat: a future re-measure only needs to
re-run the intersection, not re-derive the argument.

## The closure was wrong in BOTH directions — and hid drift in this repo

`find_cargo_defaults` computed the default closure **per manifest**, then unioned.
That is two errors, not one, and they point opposite ways:

* **UNDER-approximation — it could not follow a chain that leaves a manifest.**
  `riir-games-civ/default -> osc_emotion -> riir-games-shared/osc_emotion ->
  osc_npc` crosses a manifest boundary, so `osc_npc` read as opt-in while a
  default build ships it on. Found the hard way: correcting a riir-ai doc to say
  the true thing made the audit go RED against the corrected doc.
* **OVER-approximation — collapsing `pkg/feat` to `feat`.** Right for the
  deployed question (target crate is in-repo) and wrong when the qualifier names
  an external crate. 47 bogus names in riir-ai alone.

Both are gone: reachability over `(package, feature)` nodes from every member's
`default` as a root, not following an edge whose target package is outside the
repo. **Measured across 18 repos x both auditors, it changed exactly one
verdict** (the riir-ai false positive) and left the other 17 bit-identical —
including katgpt-rs's 92 labels and 396 Cargo comments.

### It found two stale labels in katgpt-rs's own benchmarks

The old model literally could not see these — both are reached in three hops
through a `pkg/feat` edge, so the per-manifest closure stopped short:

5. `.benchmarks/001_hla_eigenbasis_recovery_goat.md:5` said
   `` `hla_eigenbasis_recovery` (opt-in) ``.
   `katgpt-rs/default -> ica_lens -> katgpt-spectral/ica_lens ->
   hla_eigenbasis_recovery`.
6. `.benchmarks/245_still_kv_goat_metric_fix.md:5` said
   `` `still_kv` (opt-in, Plan 245) ``.
   `katgpt-rs/default -> chain_fold -> katgpt-speculative/chain_fold`, which
   lists both `still_kv` and `katgpt-kv/still_kv`.

Both verified by hand against the manifests (root `default[]`,
`katgpt-spectral/Cargo.toml:139`, `katgpt-speculative/Cargo.toml:132`) rather
than taken from the auditor. This is the same class as items 1-4 — and it
landed in the repo that owns the auditor, which is the argument against reading
"my own gate is green" as "my own docs are right".

### The forwarded-only rule was too broad, and the fix narrowed it

Suppressing every forwarded-only opt-in label would have excused both of the
above. The rule is right only for a label that **scopes its claim**: a
namespaced token (`riir-wallet/siwr`) is a per-crate claim, so accept it when
either model agrees; a BARE token in a repo-level doc is a deployed claim and
gets no suppression. Narrowed accordingly — which is precisely what turned
these two from silent skips into findings.

### A third status word, because two states were not enough

`opt-in` / `default-on` cannot describe a feature with no bare `default[]` entry
that a default build still enables through a chain. The workspace already had a
name for it — riir-neuron-db's README carries a §"Transitive default" section —
so `on by transitive default` is now transition vocabulary, and both katgpt-rs
labels use it.

Deliberately matched as the FULL phrase. A looser `transitive.*default` escaped
the negation guard (which only inspects text immediately before the match) by
matching at "transitive" in `(opt-in, NOT on by transitive default)`, six
characters past the "NOT", and read that label as default-on. The selftest's
negation case caught it.

### Seven pins, all canaried, and the second DRY defect it exposed

Adding the transitive vocabulary made the new pin fail while the production path
was correct — because `parse_terminal_transition` was applied in
`iter_bench_doc_labels`, *outside* the shared `classify_token`. The pin was
right and the factoring was wrong; the transition now lives in the one status
path. This is the second time in this issue that a status rule outside the
shared path made a pin unable to reach it (the first was the widened retry).

All seven pins were re-canaried by reverting each fix in a copy; each exits 1
naming the shape. An `own_default ⊆ deployed_default` invariant check now warns
rather than silently mis-suppressing — it holds in all 18 repos.

## NOT fixed from here

Each repo owns its own docs and CI per `BOUNDARY.md`, and agents held both
riir-ai and riir-clippy at the time of writing. Editing a sibling's
`.benchmarks/` mid-session is how two agents' work collides, and claiming a
number from a sibling's `.issues/.highwater` is the documented collision path.

## Closing conditions

- [x] riir-ai: correct the `osc_emotion` label (or demote the feature, if the
      doc reflects the intended state and the manifest is what drifted — check
      which is wrong before editing the doc). **DONE 2026-09-01, riir-ai
      `95354cd62`** — the manifest was right; the doc was corrected via the
      transition form so the promotion history survives. Correcting it is what
      made the audit go RED against the corrected doc and exposed the
      per-manifest-closure bug (see §"The closure was wrong in BOTH directions").
- [x] riir-clippy: same for `rustc_errors` in both benchmark docs. **DONE
      2026-09-01, riir-clippy `7736e30`** — both labels now read "default-ON
      since 2026-08-29 by owner call" with the driver-shaped inertness note;
      the manifest was confirmed right before editing (`Cargo.toml:120` lists
      `rustc_errors` in `default`), and `bench_doc_audit.py` re-run over the
      repo: 9 labels / **0 mismatches**.
- [x] riir-neuron-db: correct the `merkle_freeze` label, or drop it from
      `experience_graph` if default-on was not intended. **DONE 2026-09-01,
      riir-neuron-db `c06133e`** — recorded in full in item 3 above; the row
      moved into that README's existing §"Transitive default" section.
- [x] Each sibling either runs the two auditors on some cadence or records why
      not. `.github/workflows/docs_gate.yml` is portable: pure Python, ~3s,
      ubuntu-latest, no cfg surface — the only katgpt-rs-specific part is the
      `count_features.py` step and its README claim table.
      **DONE 2026-09-01 for every sibling with CI, and VERIFIED BY RUNNING IT.**
      `riir-ai` `2b14457cb`, `riir-chain` `08eb27d9`, `riir-neuron-db` `b9f87b4`
      — a weekly `docs_drift.yml` whose entire body is a three-line call to
      katgpt-rs's `sibling_docs_drift.yml`. Dispatched each and read the logs:
      riir-neuron-db 11 labels / 0 comments, riir-ai 78 / 10, riir-chain 2 / 2 —
      every figure matching the local sweep. A private `gist-rs` repo calling a
      public `katopz` reusable workflow works with the default token, which had
      been an assumption until it was run.

      **Running it is what caught the bug review had not.** The first run
      reported riir-neuron-db at **396** inline Cargo comments; that repo has
      **zero**, and 396 is exactly katgpt-rs's count. The auditors' own checkout
      sat in `.auditors/` INSIDE the audited tree, so the manifest walk descended
      into it. The tree-name guard passed throughout, because the top-level
      directory genuinely was the caller — the contamination came in through a
      different door than the one guarded. Fixed structurally (sibling
      checkouts, `7bb438e3`) and re-verified: 396 -> 0.

      That in turn exposed the mirror defect on the workstation side: the
      auditors read manifests **git has never seen**. riir-chain's
      `cloudflare/edge-wallet-container/.container-src/` is an untracked copy of
      the repo, which is why local said 4 comments and CI said 2. Untracked
      manifests feed the DEFAULT CLOSURE, so they can flip a verdict — and only
      locally, where it is then reported as the repo's state. `ed5f4865` reads
      `git ls-files` with a walk fallback; local and CI now agree exactly.

      **These two are the same bug pointed in opposite directions:** one folded
      a foreign repo's files IN, the other folded local-only files in. Neither
      was visible from one side alone. Two instruments that must agree is what
      found both.

      **The remaining label-bearing repos have no CI at all** — `riir-train` (5
      labels), `riir-mmorpg-examples` (2), `riir-dapps` (1) ship no
      `.github/workflows/` whatsoever. Recorded as the "or records why not"
      branch rather than fixed: standing up a repo's FIRST workflow to run a
      ~1-minute docs audit is an Actions-budget decision for its owner, and
      those owners are on record constraining that budget (the main-only push
      triggers). `riir-dapps` additionally still defaults to `main` while its
      work lands on `develop`, so a workflow there would be inert — the
      `.issues/704` shape, outside that issue's approved set. All three are
      covered by `scripts/docs_drift_sweep.py`, which is not a cadence and does
      not pretend to be one, but does make them measured rather than unknown.

      **Superseded note (kept for the reasoning):** before 704 landed, wiring
      these three would have shipped **three inert files** — default branch
      `main`, no `.github/workflows/` on it, and `schedule`/`workflow_dispatch`
      firing only from the default branch. Adding a fourth decoration to each
      would have ticked this checkbox while changing nothing, which is the exact
      failure this issue is about.
      Wiring a `docs-drift` job into riir-ai / riir-chain / riir-neuron-db was
      the obvious next step and would have shipped **three inert files**: all
      three have default branch `main`, `main` carries no `.github/workflows/`
      at all, and `schedule` + `workflow_dispatch` fire only from the default
      branch. Their existing CI has not run since 2026-07-31 for exactly this
      reason. Adding a fourth decoration to each would have satisfied this
      checkbox while changing nothing — the precise failure this issue is about.
      Blocked on 704's owner decision (default branch vs promote-to-main).

      **Covered in the meantime by `scripts/docs_drift_sweep.py` (`d2228161`),**
      the workstation tier: both auditors over every derived contract repo in
      one command, population derived and expectations committed. It is not a
      CI cadence and does not pretend to be one; it is what makes the blocked
      repos measured rather than unknown. Measured on landing: 18 repos,
      0 mismatches, 8 label-bearing.

      **The reusable caller is ready for the moment 704 unblocks:**
      `.github/workflows/sibling_docs_drift.yml` (`workflow_call`), so a sibling
      writes three lines instead of copying riir-clippy's job — the auditors
      default to auditing *katgpt-rs*, so a copy that omits the path argument
      passes forever against the wrong repo. That is now structurally
      unreachable, and the workflow asserts the audited tree is the caller's.

      **riir-clippy DONE 2026-09-01 (`1df3599`):** a `docs-drift` job in its
      existing weekly `rust.yml` (the repo's small-trigger Actions-budget
      decision forbids a per-push develop trigger — the recorded "why not"
      for per-push), running BOTH auditors by path from a katgpt-rs checkout,
      each at `.` — never unported, because two of docs_gate.sh's four checks
      are katgpt-rs-rooted by construction (`count_features.py` takes no repo
      argument; `skill_repo_set_gate.py` gates katgpt-rs's own SKILL set) and
      would be a vacuous green here.
- [x] Cadence wired AND verified by dispatch for every sibling with CI;
      why-not recorded for the three with none.
- [-] Remove this file — DEFERRED. AGENTS.md and both auditors cite it as the
      canonical record of their design decisions; removing it would leave
      dangling refs, which is worse noise than the file.

Refs: `a90dd631` (docs gate + auditor fixes + 185x walk speedup),
`scripts/docs_gate.sh`, `.issues/701` R2 (the same one-repo-of-twelve shape for
the compile/lint full gate).
