# Issue 701 — the three surfaces `scripts/full_gate.sh` still does NOT cover

Status: **OPEN — 3 of 6 closing conditions done** (R1a `4c12c4e8`/`baf300fa`;
R1b-cadence `feature_isolation_weekly.yml` 2026-09-02; R3a measured). R2
re-measured 2026-09-01 and its answer changed: not "11 of 12"
but **1 of 18**, because the survey itself had the Issue 703 defect. Filed
alongside the gate itself so its limits are
recorded rather than implied. `scripts/full_gate.sh` closes the
compile+lint-surface hole that left `develop` red for weeks (Issue 700 →
`c284dbb2`, `3e58e821`). It does not close everything, and a gate whose
boundaries are undocumented gets read as total coverage.

Per the "no silent caps" rule: what follows is what the gate *cannot* see.

## R1 — per-feature isolation is not gated (568 flags)

`--all-features` proves the union compiles. It says nothing about a feature
compiling **on its own**. Every primitive in this repo ships behind an opt-in
flag (AGENTS.md "Feature Flag Discipline"), so "feature X alone is broken" is a
live failure mode that the current gate is blind to by construction.

The standard tool is `cargo hack --each-feature`, and a sibling already runs it:
`riir-neuron-db/.github/workflows/rust.yml` chains
`--each-feature` → `--all-features --all-targets` → `cargo test --all-features`.

**Why it is not simply adopted here.** Measured by the repo's own
`scripts/count_features.py`: **568 total flags, 197 default-on** across 29
feature-defining manifests. `--each-feature` means one workspace compile per
flag.

### Measured cost, 2026-09-01 — this section's original estimate was WRONG

The first version of this issue said "at even 2 min each that is ~18 h" and
dismissed the default-on subset as "still ~6 h; probably also too slow". Both
numbers were guesses, and both were pessimistic. Actually measured on an M3 Max
with a warm target dir, `cargo check -p katgpt-rs --no-default-features
--features <flag>`, seeded random sample of 6 from the 280 root-manifest opt-in
flags (seed 701, reproducible):

| flag | wall-clock |
|---|---|
| `monopoly` | 110.2s |
| `thicket_variance_probe` | 45.8s |
| `dflare_kv_routing` | 33.5s |
| `delta_mem` | 22.0s |
| `kimi_k3` | 21.2s |
| `vocab_coreset` | 4.2s |

**mean 39.5s** → 568 flags ≈ **6.2 h**; 197 default-on ≈ **2.2 h**.

So the default-on subset is *viable as a weekly gate* — `full_gate.yml` already
carries a 180-minute ceiling and 2.2 h fits under it. That is a different
conclusion from the one this issue shipped with.

Read the extrapolation as a point estimate, not a bound:
- n=6 of 280, and the range is 26× (4.2s → 110.2s). The mean is not tight.
- **Warm** target dir sharing already-built deps. The one-time
  `--no-default-features` baseline cost 60s and **~11 GiB** of disk; a cold CI
  runner pays that first.
- An M3 Max is faster than any GitHub runner, and macOS minutes bill at a
  multiple of Linux — but this gate has no `cfg(target_os)` surface, so unlike
  the full gate it can run on ubuntu.
- **Disk is a real second axis, not accounted for before.** Marginal cost was
  ~0.09 GiB per flag. Even sub-additive, a 568-flag sweep is tens of GiB, and
  this box was at 60 GiB free with 117 GiB already in `target/`.

### Is isolation currently green? Sampled before wiring anything

21 of the 280 root opt-in flags (7.5%), three seeded batches, 2026-09-01:
**21 pass / 0 fail**. Read that as a low failure rate, NOT as zero — by the rule
of three, 0/21 bounds the rate at roughly **<=14% at 95% confidence**. Enough to
wire a PR gate without expecting it to red every flag-touching PR; not enough to
claim the flag surface is clean.

One harness note, because it nearly misread as a finding: the sampling loop
exited 1 with all 21 green. The last command in the block was
`[ -n "$fails" ] && echo ...`, and `[ -n "" ]` returns 1 — the failing exit was
the harness, not the sample.

Revised candidate designs:
- **Changed-flags-only (best per-PR).** `--each-feature` restricted to flags
  whose *definition* the diff touches. Bounded by the diff, not the manifest: a
  typical 1-3 flag PR costs ~40-120s. This is the design to build first.
- **Default-on subset, weekly.** 197 flags ≈ 2.2 h measured. Now plausible;
  needs one real cold run on a runner before committing to it.
- **Sampled rotation.** N flags per weekly run, cursor persisted. Still the
  fallback for full 568 coverage; catches rot, not regressions.

## R2 — 1 of 18 repos gates the full surface (re-measured, derived)

**The two earlier hand surveys of this were both wrong, in three separate
ways.** That is the argument for `scripts/ci_gate_coverage.py`, which now
produces the table below; do not re-type it.

| defect | effect on the old answer |
|---|---|
| the repo list was **typed** ("11 of 12 sibling repos") | covered **12 of 18** — the Issue 703 class. `riir-armageddon`, `riir-auth`, `riir-burner`, `seal-game-editor` and `katgpt-web` were simply absent, so "5 repos with 0 workflows" was really **9** |
| only workflow **YAML** was grepped | the gate usually lives in a **script the workflow calls**. `katgpt-rs/full_gate.yml` runs `./scripts/full_gate.sh`; `riir-neuron-db/rust.yml` runs `./scripts/ci_feature_guard.sh`. Grepping YAML alone reports both as ungated |
| the grep **included comments** | several workflows discuss `--all-features --all-targets` in a preamble while running nothing of the kind. This alone flipped `riir-neuron-db` and `katgpt-rs` in an intermediate run |

A fourth defect appeared in the *fix*: scoring signals over a concatenated blob
rates a repo "full" when `--all-targets` is in one script and `--all-features`
in another. AGENTS.md's entire point is that those axes are **independent**, so
a green on each separately says nothing about their combination. Signals are now
scored **per command**, with continuations joined.

### Measured 2026-09-01 by `scripts/ci_gate_coverage.py`

| repo | workflows | scripts followed | signals | full surface |
|---|---|---|---|---|
| `katgpt-rs` | 5 | `scripts/docs_gate.sh`, `scripts/full_gate.sh`, `scripts/proof_gate.sh` | `clippy` `--workspace` `--all-targets` `--all-features` `--keep-going` | **yes** |
| `katgpt-web` | 1 | — | — | no |
| `riir-ai` | 1 | `scripts/proof_gate.sh` | — | no |
| `riir-armageddon` | 0 | — | — | no CI |
| `riir-auth` | 0 | — | — | no CI |
| `riir-burner` | 0 | — | — | no CI |
| `riir-chain` | 3 | `scripts/proof_gate.sh`, `scripts/proof_negative_test.sh`, `scripts/spec_match_gate.sh`, `scripts/standalone_dep_gate.sh`, `scripts/dockerfile_heal_gate.sh`, `scripts/clippy_gate.sh`, `scripts/test_gate.sh`, `scripts/replay_negative_test.sh`, `scripts/feature_pair_gate.sh`, `scripts/frost_client_only_gate.sh`, `scripts/action_slot_policy_gate.sh`, `scripts/settlement_policy_gate.sh`, `scripts/settlement_apply_sites_gate.sh`, `scripts/block_pipeline_reachability_gate.sh`, `scripts/wasm_import_gate.sh` | `--each-feature` `--keep-going` <br>*(scattered across commands: `clippy` `--all-features` `--each-feature` `--keep-going`)* | **needs a human read** — `--all-targets` live in data, not in a command |
| `riir-clippy` | 1 | — | — | **needs a human read** — `clippy` live in data, not in a command |
| `riir-dao` | 1 | `scripts/direction_gate.sh` | `clippy` `--all-targets` | partial |
| `riir-dapps` | 0 | — | — | no CI |
| `riir-deployer` | 1 | — | — | no |
| `riir-game-sdk` | 2 | `scripts/ci_wasm32_guard.sh` | — | no |
| `riir-mmorpg-examples` | 0 | — | — | no CI |
| `riir-neuron-db` | 2 | `scripts/proof_gate.sh`, `scripts/standalone_dep_gate.sh`, `scripts/ci_feature_guard.sh` | `--all-targets` `--all-features` <br>*(scattered across commands: `--all-targets` `--all-features` `--each-feature`)* | partial |
| `riir-train` | 0 | — | — | no CI |
| `riir-unity` | 0 | — | — | no CI |
| `riir-viewbridge` | 0 | — | — | no CI |
| `seal-game-editor` | 0 | — | — | no CI |

**1/18 statically full · 2 need a human read · 3 partial · 9 have no CI at all.**

### The two "human read" rows, read

- **`riir-chain` — effectively FULL, in matrix form.** `scripts/clippy_gate.sh`
  holds a `CONFIGS` table of `pkg:features:--all-targets` rows (with `__ALL__`
  meaning all-features) that a loop expands into `args=(clippy -p "$pkg" …)`.
  The flags live in **data**, so no static scorer can see them. This is arguably
  *stronger* than katgpt-rs's single command — a real feature matrix — though it
  is per-package rather than `--workspace`. The script is wired from
  `toolchain_drift.yml`.
- **`riir-clippy` — NOT gated.** Its one workflow, `ops_dashboard.yml`, is a
  dashboard generator; it installs the `clippy` *component* but runs no lint
  gate. **Cross-repo finding while reading it:** that workflow is built around
  a hard-coded **"the 11 repos"** generator list against a workspace of 18 —
  another Issue 703 instance, in a sibling's CI. Not fixed from here (riir-clippy
  owns it, and an agent was active in it); recorded in `.issues/703` §Related.

### Correction to two rows the old table asserted

- `riir-neuron-db` was called **yes**. It is **partial**: its full-surface
  command is `cargo check --all-features --all-targets` — **`check`, not
  `clippy`**. That is a documented blind spot in this repo's own AGENTS.md (two
  `cargo heal` escape classes are rejected by clippy's typeck and accepted by
  `check`), so the distinction is not pedantic.
- `riir-chain` was called **no**. See above — it is the most thorough gate in
  the workspace after this repo's.

NOT fixed from here: each repo owns its own CI per `BOUNDARY.md`, and agents
were active in several at the time of writing. `scripts/full_gate.sh` is written
to be portable — the only katgpt-rs-specific parts are the macOS platform layer
and the AGENTS.md parity check.

**Addendum 2026-09-02 — the three `hand_only` rows are closed.** The delegated
idle worker applied `.issues/706`'s recommendation (1): the `riir-clippy`-shape
weekly `schedule`, which fires from the default branch (`develop` since the
`.issues/704` flip) while `main` stays frozen and the no-develop-push owner
call stands. `riir-chain` `b4a9b6e7` (Tue 04:13 UTC), `riir-neuron-db`
`9d041d1` (04:29), `riir-dao` `9848811` (04:43 — whose workflow also stopped
hand-mirroring its guard layers and runs `scripts/ci_feature_guard.sh`; with
one feature total, default + `advisory_transport` IS the full matrix, so the
dao's "partial" coverage verdict above was a static-scoring artifact, not a
real gap). The dormancy had already cost a real red: `riir-neuron-db`'s
standalone-dep gate pin went stale nine days earlier (`29af2b0` changed the
katgpt-rs patch set to `katgpt-device-verify` without re-pinning `EXPECTED`)
and nobody could see it because nothing ran the gate — fixed `97e5161`. R2
remains open for the 8 no-CI rows and the not-full coverage rows above.

**Addendum 2026-09-02 (second fan-out slice) — `riir-viewbridge` wired.**
`6098972` (its repo): `scripts/ci_feature_guard.sh` (6 layers — check, clippy
-D default + net feature, tests with a 13-binary floor, wasm32 core check,
net-feature tests incl. the real-QUIC loopback round-trips),
`scripts/standalone_dep_gate.sh` (ONE escaping dep pinned: the optional
`riir-net` path dep into private riir-ai; injection-verified), and
`.github/workflows/rust.yml` (weekly Sundays 03:41 UTC — default branch is
`develop`, verified via ls-remote symref, so the rot check audits the right
branch; standalone job secret-free; feature-guard provisions riir-ai via
`SIBLING_REPOS_TOKEN`, fail-loud when absent). Baselines measured before
wiring, not assumed: clippy clean both feature states, 107 passed / 3 ignored
across 13 test binaries, wasm32 green.ubuntu was correct for this repo where
it was wrong for the isolation sweep: viewbridge has NO cfg(target_os)
surface, so every layer compiles the same real code on Linux. The
`SIBLING_REPOS_TOKEN` secret must exist for this repo (org-level or added
once) — if absent, the feature-guard job fails LOUD with provisioning
instructions rather than silently skipping, and the standalone job still runs.
Remaining in R2: riir-dapps, riir-train, riir-auth, riir-unity, and the repos
outside the surveyed set's reach (riir-burner, katgpt-web — neither was in
this workspace's checkout set), plus seal-game-editor (read-only repo,
owner-owned — record-why-not only).

**Addendum 2026-09-02 (third fan-out slice) — `riir-dapps` wired + default
branch flipped.** `2a31a17` (its repo): `scripts/ci_feature_guard.sh`
(L1 direction gate / L2 default tests / L3 clippy -D / L4 chain_backend /
L5 all-features clippy -D / L6 all-features tests — 502 green at wiring),
`scripts/standalone_dep_gate.sh` (3 entries pinned: riir-chain,
riir-chain-sdk, riir-neuron-db — both siblings private, NO public sibling at
all; injection-verified), `.github/workflows/rust.yml` (weekly Saturdays
03:53 UTC; standalone job secret-free; feature-guard provisions both
siblings via `SIBLING_REPOS_TOKEN`, fail-loud). **Its default branch moved
`main` -> `develop` in the same landing** — `riir-dapps` existed at the
2026-09-01 flip and was missed; main was strictly behind develop (no unique
commits); AGENTS.md §Branch corrected in the same commit. The `cloudflare/`
standalone crates (warm-tier-do, kat-service) are deliberately NOT gated by
this workflow — own workspace roots, own deploy surfaces, own CI when they
earn it. Baselines measured before wiring: default 111/0, chain_backend
149/0, all-features 502/0 across 32 binaries; clippy clean both states.
Remaining in R2: riir-train (active sibling lane — next cycle), riir-auth,
riir-unity, out-of-checkout-set repos (riir-burner, katgpt-web),
seal-game-editor (read-only — record-why-not only).

## R3 — the warning surface is not gated

The gate reports the warning count as information and gates only on errors.

**Measured 2026-09-01** (`--workspace --all-targets --all-features`, both
message formats, cross-reconciled). Three quantities live within 23 of each
other here, and the first green run's headline reported none of them:

| Quantity | Value | What it is |
|---|---|---|
| `^warning` lines | 138 | findings **plus** cargo's 20 per-target tallies — the old headline |
| emitted warnings | 141 | JSON count; also the exact sum of those 20 tallies |
| **distinct findings** | **118** | 141 − 23 duplicates (same site compiled in `lib` *and* `lib test`) |

The 23 duplicates are 21 in `katgpt-pruners` plus two singletons; deduplicating
the JSON by (lint, file, line, column) yields 118, matching the human-format
render exactly. The gate now reports 118 findings across 20 targets —
`scripts/full_gate.sh` counted lines before this, which is why "138" appears in
this issue's history.

Deduplicated histogram — 24 distinct lints, 118 findings:

| n | lint | class |
|---|---|---|
| 37 | `clippy::needless_range_loop` | mechanical |
| 20 | `clippy::needless_borrows_for_generic_args` | mechanical |
| 19 | `clippy::unnecessary_map_or` | mechanical (38 emitted, 19 distinct) |
| 6 | `unused_mut` | mechanical |
| 4 | `unused_variables` | needs judgement (may indicate dead logic) |
| 3 each | `unusual_byte_groupings`, `dead_code`, `manual_repeat_n` | mixed |
| ≤2 each | 16 further lints incl. `if_same_then_else`, `too_many_arguments`, `question_mark` | needs judgement |

By crate: `katgpt-core` 70, `katgpt-pruners` 21, root benches 13, root tests 8,
`katgpt-attn` 3, `katgpt-speculative`/`katgpt-types`/`src` 1 each.

**Decision recorded: heal, then gate — not `-D warnings` now.** The top three
lints are **76 of 118 (64%)** and are exactly `cargo heal`'s mechanical domain
(global rule; `cargo-heal` skill). Flipping `-D warnings` first would red the
gate on 118 findings at once and the pressure would be to disable the gate, not
fix the code. Order that works:

1. `cargo heal --fix --write --verify` the three mechanical classes — but note
   `--verify` is a `cargo check`, blind to the clippy-only classes this gate
   exists to catch, so re-run the full gate after, not the healer's own verify.
   `katgpt-pruners` alone offers 19 machine-applicable suggestions.
2. Re-measure. The residual should be ~30-40, mostly judgement calls.
3. Then decide `-D warnings` wholesale vs a named `-D` subset of the settled
   lints — with the histogram above as the before-picture.

### Baseline moved: 118 -> 119 after a 52-edit heal sweep

Re-measured 2026-09-01 after sibling commit `91891c2c` healed 26 files / 52
edits: **119 findings across 20 targets**, i.e. +1. Fifty-two mechanical edits
reduced this surface by nothing and added one, which is consistent with the
finding below — the healer's lint set and this gate's lint set barely intersect.
Take it as confirmation that R3b cannot be delegated to a heal sweep.

Two process facts from that run, both of which cost time:

- `scripts/full_gate.sh` **deleted the log on success**, including when the
  caller had named a path via `$FULL_GATE_LOG`. The count survives in the
  summary; the histogram behind it does not, and it is R3b's only input. Fixed
  in `613c88d3` — a named path is now honoured on pass.
- A warm re-run **did** reproduce the full diagnostics (119/20, log intact), so
  the histogram was recoverable without a second cold run. Worth recording
  because the opposite is often true of `cargo clippy` and was assumed here.

### R3b first slice, measured 2026-09-01 — the healer does not target this surface

Attempted on `crates/katgpt-pruners` (21 findings, 19 of them
`unnecessary_map_or`, 17 in one file — the tightest available slice):

    cargo heal --fix --write --verify --verify-args "--all-features" crates/katgpt-pruners

Result: **8 edits across 4 files, all `manual_let_else`** (plus one behaviour-
neutral statement reorder). `manual_let_else` appears **0 times** in the gate's
141 emitted diagnostics — it is outside the lint set the gate reports. So the
run reduced the 118-finding surface by **zero**, and none of the top three
classes was touched:

| lint | findings | healed |
|---|---|---|
| `clippy::needless_range_loop` | 37 | 0 |
| `clippy::needless_borrows_for_generic_args` | 20 | 0 |
| `clippy::unnecessary_map_or` | 19 | 0 |

Cost: ~25 min, because `--verify-args "--all-features"` makes the healer pay
**two workspace-wide all-features builds** (baseline + re-check) to validate 8
edits. Budget for that before scoping a heal on this repo.

Consequences for R3b:
- **`cargo heal` is the wrong tool for these three classes.** The global rule
  names `map_or` as mechanical-and-healable, but `clippy::unnecessary_map_or`
  (suggesting `is_some_and` / `is_none_or`) was not recognised. Route the top
  three through `cargo clippy --fix` instead, then re-measure.
- **riir-clippy intake** (per the repo rule on observed misses): the healer's
  corpus has no rule for the single largest lint class in this repo. Recorded
  here rather than filed in riir-clippy because an agent held that tree at the
  time; move it across when free.
- The 8 edits are kept — compile-verified, semantically identical (every
  let-else `else` diverges via `return`), and re-checked with the FULL gate
  rather than the healer's own `cargo check`.

Not attempted: the remaining ~76 sites across four crates in one sweep.

## Closing conditions

- [x] R1a: a per-feature-isolation check runs on a stated cadence with its
      scoping in the workflow — `scripts/feature_isolation_gate.py` +
      `.github/workflows/feature_isolation.yml`, diff-bounded, macos-latest,
      pull_request only (push has no base ref and would pass vacuously).
- [x] R1b-measure: broader coverage than the diff — **DONE 2026-09-01**, and
      the estimate it was blocked on was wrong by 32x. All **228** default-on
      (package, flag) pairs built in isolation in **4.1 min** (mean 1.08s,
      median 0.50s), not the extrapolated ~2.2 h. **2 real failures found and
      fixed** (`b50db0ef`): `katgpt-core/hebbian_kernel_memory` and
      `velocity_field_ensemble` both consume `linalg::ridge_solve` without
      joining the `cfg(any(...))` that gates `pub mod linalg`. Full numbers,
      the per-package table and the category error behind the old estimate:
      `.benchmarks/696_default_on_feature_isolation_sweep.md`.
- [x] R1b-cadence: wire it, and at what cadence — **DONE 2026-09-02** as
      `.github/workflows/feature_isolation_weekly.yml`: Mondays 04:47 UTC
      (after full_gate's Monday 04:17 slot; off-hour; clear of the siblings'
      Tuesday weekly slots), `--scope default-on`, `workflow_dispatch` for
      manual runs, `cancel-in-progress: false` (the schedule is the rot
      check). Local pre-push verification: the script's `--list` walk
      reproduced the Bench-696 population exactly (228 pairs / 197 unique
      names) and the scope self-test passed on the current tree; YAML parsed.
      **Platform deviation from the recommendation below, recorded:** macOS,
      not ubuntu. The recommendation's "this gate has no `cfg(target_os)`
      surface" is true of the SCRIPT and false of the CODE it compiles — the
      device backends are `cfg(all(target_os = "macos", feature = ...))`, so on
      a Linux runner a device-backed flag switches on and compiles to NOTHING:
      the vacuous green feature_isolation.yml's preamble rejects. Whether any
      of the 228 default-on pairs is device-backed was not worth resolving
      when the decision does not need it — at 4.1 min warm the
      runner-multiplier cost is immaterial either way, so the sweep runs where
      every isolation claim is a claim about real compiled code.
      It catches a class NOTHING else does — `--all-features` compiles
      the union where some other consumer always supplies `linalg`, and the
      per-PR gate is diff-bounded so it never looks at a flag whose own
      definition did not change. Both flags above were invisible to every
      existing gate and were found by this sweep's first run.
- [ ] R2: the remaining repos either run the gate or record why not; the
      measured error count per repo is reported, not assumed to be zero.
- [x] R3a: the warning surface is measured (118 findings / 24 lints / per-crate
      + per-lint histogram above) and the gate decision recorded: heal the
      mechanical 64% first, re-measure, then choose the `-D` scope.
- [ ] R3b: execute that order — heal, re-measure, flip the chosen `-D` set.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `scripts/full_gate.sh`, `.github/workflows/full_gate.yml`,
`.benchmarks/695_simd_len_guard_goat.md` §G3, riir-ai Issue 830.
