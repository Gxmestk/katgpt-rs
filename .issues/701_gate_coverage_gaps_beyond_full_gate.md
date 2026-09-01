# Issue 701 — the three surfaces `scripts/full_gate.sh` still does NOT cover

Status: OPEN (0/3 closed; R3 measured, execution pending) — filed alongside the gate itself so its limits are
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

Revised candidate designs:
- **Changed-flags-only (best per-PR).** `--each-feature` restricted to flags
  whose *definition* the diff touches. Bounded by the diff, not the manifest: a
  typical 1-3 flag PR costs ~40-120s. This is the design to build first.
- **Default-on subset, weekly.** 197 flags ≈ 2.2 h measured. Now plausible;
  needs one real cold run on a runner before committing to it.
- **Sampled rotation.** N flags per weekly run, cursor persisted. Still the
  fallback for full 568 coverage; catches rot, not regressions.

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

- [ ] R1: a per-feature-isolation check runs on some cadence, with its sampling
      or scoping stated in the workflow (a silent top-N is the thing being
      avoided).
- [ ] R2: the remaining repos either run the gate or record why not; the
      measured error count per repo is reported, not assumed to be zero.
- [x] R3a: the warning surface is measured (118 findings / 24 lints / per-crate
      + per-lint histogram above) and the gate decision recorded: heal the
      mechanical 64% first, re-measure, then choose the `-D` scope.
- [ ] R3b: execute that order — heal, re-measure, flip the chosen `-D` set.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `scripts/full_gate.sh`, `.github/workflows/full_gate.yml`,
`.benchmarks/695_simd_len_guard_goat.md` §G3, riir-ai Issue 830.
