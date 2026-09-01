# Issue 701 — the three surfaces `scripts/full_gate.sh` still does NOT cover

Status: OPEN (0/3 closed) — filed alongside the gate itself so its limits are
recorded rather than implied. `scripts/full_gate.sh` closes the
compile+lint-surface hole that left `develop` red for weeks (Issue 700 →
`c284dbb2`, `3e58e821`). It does not close everything, and a gate whose
boundaries are undocumented gets read as total coverage.

Per the "no silent caps" rule: what follows is what the gate *cannot* see.

## R1 — per-feature isolation is not gated (537 flags)

`--all-features` proves the union compiles. It says nothing about a feature
compiling **on its own**. Every primitive in this repo ships behind an opt-in
flag (AGENTS.md "Feature Flag Discipline"), so "feature X alone is broken" is a
live failure mode that the current gate is blind to by construction.

The standard tool is `cargo hack --each-feature`, and a sibling already runs it:
`riir-neuron-db/.github/workflows/rust.yml` chains
`--each-feature` → `--all-features --all-targets` → `cargo test --all-features`.

**Why it is not simply adopted here.** Measured by the repo's own
`scripts/count_features.py`: **537 total flags, 189 default-on**. `--each-feature`
means 537 workspace compiles. At even 2 min each that is ~18 h — infeasible as
a gate, weekly or otherwise. riir-neuron-db can afford it because it is far
smaller. Naively copying that workflow here would produce a gate that always
times out, which is worse than the documented gap.

Candidate designs, none validated:
- **Changed-flags-only.** At PR time, `--each-feature` restricted to flags whose
  definition the diff touches. Bounded by the diff, not the manifest.
- **Sampled rotation.** N flags per weekly run, cursor persisted, so the full
  set is covered over weeks. Catches rot, not regressions.
- **Default-on subset.** 189 rather than 537 — still ~6 h; probably also too slow.

## R2 — 11 of 12 sibling repos have no full-gate workflow

Surveyed 2026-09-01 by listing `.github/workflows/*.yml` and grepping for the
`--all-targets`/`--all-features` pair:

| Repo | Workflows | Runs the full gate |
|---|---|---|
| riir-neuron-db | 2 | **yes** (`rust.yml`) |
| riir-ai | 1 | no |
| riir-chain | 3 | no |
| riir-clippy | 1 | no |
| riir-dao | 1 | no |
| riir-deployer | 1 | no |
| riir-game-sdk | 2 | no |
| riir-dapps / riir-train / riir-unity / riir-viewbridge / riir-mmorpg-examples | 0 | no |

The class this gate catches is therefore live and unmeasured across the
workspace. riir-ai Issue 830 is the precedent for the magnitude: **99 errors /
50 targets / 9 crates** at a HEAD where every existing gate was green.

NOT fixed from here: each repo owns its own CI per `BOUNDARY.md`, and three of
them had agents actively working at the time of writing. `scripts/full_gate.sh`
is written to be portable — the only katgpt-rs-specific parts are the macOS
platform layer and the AGENTS.md parity check.

## R3 — the warning surface is not gated

The gate reports the warning count as information and gates only on errors.
Measured on the first green run: **138 warning lines** across the workspace at
`--all-targets --all-features`, of which **62 come from `katgpt-core`'s lib test
alone**. So `-D warnings` is not a one-line change and is not attempted here.
Anything adopted should be measured further first: a per-lint histogram, then a
decision on whether to gate the whole surface or a named subset.

Note the count is *warning lines*, not distinct findings — cargo emits a
per-target "generated N warnings" summary line that this figure includes. Get
the histogram before treating 138 as the size of the job.

Note the interaction with R1's tooling and with `cargo heal`: mechanical classes
should be healed (see the `cargo-heal` skill), not hand-fixed, and the healer's
own `--verify` is a `cargo check` — blind to the clippy-only classes that the
full gate exists to catch.

## Closing conditions

- [ ] R1: a per-feature-isolation check runs on some cadence, with its sampling
      or scoping stated in the workflow (a silent top-N is the thing being
      avoided).
- [ ] R2: the remaining repos either run the gate or record why not; the
      measured error count per repo is reported, not assumed to be zero.
- [ ] R3: the warning surface is measured and a gate decision recorded.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `scripts/full_gate.sh`, `.github/workflows/full_gate.yml`,
`.benchmarks/695_simd_len_guard_goat.md` §G3, riir-ai Issue 830.
