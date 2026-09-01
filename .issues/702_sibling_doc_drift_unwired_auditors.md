# Issue 702 — the doc-drift auditors run in ONE repo of eight, and two siblings carry confirmed stale labels

Status: OPEN (0/3) — filed from katgpt-rs because the tooling lives here; the
fixes belong in the owning repos.

`scripts/bench_doc_audit.py` and `scripts/cargo_comment_audit.py` accept a repo
path and audit any repo. As of `a90dd631` katgpt-rs runs them per-push via
`.github/workflows/docs_gate.yml`. **No other repo runs either one**, and until
that commit neither ran anywhere — both were RED in katgpt-rs itself on false
positives (case-sensitive `Opt-in` regex; first-status-wins label parsing).

Good news first: there are **no sibling copies** of these scripts (checked
2026-09-01 across the 7 siblings). They are invoked by path, so the fixes in
`a90dd631` apply workspace-wide with no sync problem. The gap is purely that
nothing *invokes* them outside this repo.

## Measurement (2026-09-01, post-fix auditors, read-only)

| Repo | bench-doc labels | mismatches | Cargo comments | mismatches |
|---|---|---|---|---|
| katgpt-rs | 72 | 0 | 396 | 0 |
| riir-ai | 59 | **1** | 10 | 0 |
| riir-clippy | 9 | **2** | 0 | 0 |
| riir-chain | 0 | 0 | 4 | 0 |
| riir-neuron-db | 8 | 0 | 0 | 0 |
| riir-dapps | 1 | 0 | 0 | 0 |
| riir-train | 4 | 0 | 0 | 0 |
| riir-game-sdk | 0 | 0 | 0 | 0 |

Note the two zero-label repos (riir-chain, riir-game-sdk): a zero there is not a
clean bill of health, it means no label in `.benchmarks`/`.docs` matched the
recognised forms at all. Worth a look before reading those rows as green.

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
- [ ] Each sibling either runs the two auditors on some cadence or records why
      not. `.github/workflows/docs_gate.yml` is portable: pure Python, ~3s,
      ubuntu-latest, no cfg surface — the only katgpt-rs-specific part is the
      `count_features.py` step and its README claim table.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `a90dd631` (docs gate + auditor fixes + 185x walk speedup),
`scripts/docs_gate.sh`, `.issues/701` R2 (the same one-repo-of-twelve shape for
the compile/lint full gate).
