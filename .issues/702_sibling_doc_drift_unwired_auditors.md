# Issue 702 — the doc-drift auditors run in ONE repo of eighteen, and three siblings carry confirmed stale labels

Status: **OPEN (0/3 fixes; measurement re-done and a 4th condition added)** —
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

This is the stale-flag-state class: the feature was promoted and the benchmark
doc that gated the promotion never got updated. It matters because those docs
are the GOAT record — a reader deciding whether a primitive is production
default reads the benchmark, not the manifest.

## NOT fixed from here

Each repo owns its own docs and CI per `BOUNDARY.md`, and agents held both
riir-ai and riir-clippy at the time of writing. Editing a sibling's
`.benchmarks/` mid-session is how two agents' work collides, and claiming a
number from a sibling's `.issues/.highwater` is the documented collision path.

## Closing conditions

- [ ] riir-ai: correct the `osc_emotion` label (or demote the feature, if the
      doc reflects the intended state and the manifest is what drifted — check
      which is wrong before editing the doc).
- [ ] riir-clippy: same for `rustc_errors` in both benchmark docs.
- [ ] riir-neuron-db: correct the `merkle_freeze` label, or drop it from
      `experience_graph` if default-on was not intended.
- [ ] Each sibling either runs the two auditors on some cadence or records why
      not. `.github/workflows/docs_gate.yml` is portable: pure Python, ~3s,
      ubuntu-latest, no cfg surface — the only katgpt-rs-specific part is the
      `count_features.py` step and its README claim table.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `a90dd631` (docs gate + auditor fixes + 185x walk speedup),
`scripts/docs_gate.sh`, `.issues/701` R2 (the same one-repo-of-twelve shape for
the compile/lint full gate).
