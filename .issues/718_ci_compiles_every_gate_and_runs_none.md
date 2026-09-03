# Issue 718 — CI compiles every gate in this repo and EXECUTES none

**Status:** OPEN — T1 DONE, filed 2026-09-03. T2 (the cheap selftest lane) is
ungated and actionable now; T3 (a full test job) is an
owner cost call. Found sideways, while closing a *different* instance of the
same class in seal-remake (`seal-remake` `e1ead85`).

## The measurement

`cargo clippy` and `cargo check` **compile** test targets. Neither runs one.
Measured 2026-09-03 on `develop`:

```
grep -rnE "cargo (test|nextest|bench)" scripts/ .github/
```

returns **two** hits, and both are prose — a docstring and a report string
inside `scripts/cfg_gated_target_audit.py`. There is no test-executing
command anywhere in `scripts/` or `.github/`.

Per workflow, the command each one actually reaches:

| workflow | trigger | what it runs | executes tests? |
|---|---|---|---|
| `full_gate.yml` | weekly + dispatch | `scripts/full_gate.sh` | **no** |
| `docs_gate.yml` | per-push | python auditors | no (n/a) |
| `feature_isolation.yml` | per-push (diff-bounded) | `feature_isolation_gate.py` → `cargo check` | **no** |
| `feature_isolation_weekly.yml` | weekly | same, `--scope default-on` | **no** |
| `lean_proofs.yml` | dispatch | `proof_gate.sh` (Lean) | no (n/a) |
| `sibling_docs_drift.yml` | `workflow_call` | python auditors | no (n/a) |
| `release-plz.yml` | dispatch | release | no (n/a) |

And `scripts/full_gate.sh`'s six layers are compile-only by construction:

| layer | command |
|---|---|
| 1 | cargo present |
| 2 | platform coverage |
| 3 | `GATE_ARGS` = `cargo clippy --workspace --all-targets --all-features --keep-going` |
| 3b | liveness — did the run examine anything |
| 4 | zero errors |
| 5 | doc/script parity |
| 6 | `REL_ARGS` = `cargo check --workspace --all-targets --all-features --keep-going --release` |

**Scope of what that leaves unexecuted** (`cargo metadata`, not a guess):
**477 integration-test targets, 31 lib targets (unit tests), 176 bench
targets, over 32 packages.** Zero are run by any automatic trigger.

## Why this is the next rung on a ladder this repo already climbed

AGENTS.md documents three rungs and stops one short of this:

1. **"a workflow file is identical on disk whether or not it can execute"**
   → workflows on a non-default branch are inert (`.issues/704`).
2. **"can fire is not does fire"** → a `workflow_dispatch`-only gate is a
   button, not a schedule (`.issues/706`).
3. **"a green test count can be a count of nothing"** → a `#![cfg]`-gated
   file compiles empty and prints `ok. 0 passed` (`.docs/10_audits/cfg_gated_silent_zero_pass.md`).
4. **← THIS: "compiles is not runs."** A gate CI compiles and never executes
   is in exactly the state rung 3 warns about, except the count is not zero
   — there is no count at all, because nothing produced one.

By this repo's own standard — *"Treat an uninvoked assertion as unknown, not
as passing"* — every Rust assertion here is currently **unknown**.

## This makes Issue 713 T3's arming half-complete

713 T3 added `required-features` rows to 39 GOAT gates (`180be9c5`) so that
naming a target without its features errors with exit 101 instead of
reporting a green zero. AGENTS.md records the safety argument for that
change as:

> Adding the rows is safe and does **not** red an existing CI: `cargo test
> --workspace` silently *skips* a target whose required-features are off.

That is true, and read one way it is the other edge of the same fact: the
rows make a **named** run honest, and **nothing names them**. The arming
fixed the failure mode where a green zero is cited as evidence; it did not
create a path by which the 39 gates ever run. AGENTS.md also says *"All 39
pass there"* under `--release` — that was a **workstation** measurement, and
nothing repeats it.

Not a criticism of 713 T3, which did what it set out to do. The point is
that "armed" and "run" are separate axes and only the first was closed.

## The same class, different mechanism, already fixed once (the cross-check)

`seal-remake` had the sibling instance and it is closed (`e1ead85`, this
session): that repo's guard DOES run `cargo test --workspace`, but all three
`texture_vessel` test targets carry `required-features = ["texture_vessel"]`
(default-OFF), so the run built 7 test executables and **none** of them.
Its Issue 001 G1/G2/G4 gates and Issue 002 instrument selftests were claimed
DONE while nothing automatic had executed one assertion — and layer 2's
`--all-features` clippy **compiled** them, which is exactly why it stayed
invisible. Fixed by a `layer 3b` that names each target and floors its
assertion count per target.

Two mechanisms, one class:

| repo | `cargo test` in CI? | required-features targets reached? | result |
|---|---|---|---|
| seal-remake (before `e1ead85`) | yes | no — silently skipped | 13 assertions unexecuted |
| **katgpt-rs (now)** | **no** | n/a — nothing runs | **508 targets unexecuted** |

## Tasks

- [x] **T1 — document the axis (free, and mandatory regardless of T3).** DONE — AGENTS.md now carries a sixth **compile vs EXECUTE** row in the full-gate blind-spot table plus the paragraph that tells a reader what a green gate does and does not claim. The same edit removed a live instance of the drift that table exists to catch: the preamble said "three **independent** axes" while the table had five, so the count is now not written at all.
  AGENTS.md's full-gate section enumerates five blind spots of the gate
  command and every one of them is a *compilation* axis; a reader finishes
  it believing a green full gate is a strong whole-repo claim. Add the sixth
  row: the gate does not execute anything. State plainly that CI is
  compile-only, so a green is never evidence that a test passes. Cheap,
  ungated, and it stops the misreading immediately.
- [ ] **T2 — a per-push lane for the INSTRUMENT selftests.** The auditors
  that gate this repo carry `selftest()` functions precisely because a
  tokenizer regression is silent, and several are Python (already run by
  `docs_gate.yml`). The Rust-side equivalents — e.g. the quantile /
  parser / classifier selftests — are fast, have no GPU or platform
  surface, and are the highest-value-per-second tests in the repo, because
  they guard the gates that DO run. Enumerate them, then run exactly those
  per-push. Must be a DERIVED set, not a hand-typed list (the workspace rule:
  a hand-typed population is what makes a cross-repo gate permanently
  green), and must assert a per-target floor so a target that compiles to
  nothing FAILS rather than reporting `ok. 0 passed` — the seal-remake
  `layer 3b` shape.
- [ ] **T3 — a full test job (OWNER COST CALL, do not self-authorize).**
  `cargo test --workspace --all-features --release` is the complete fix and
  is expensive: AGENTS.md already records the full gate at >13 min for
  compile alone and says per-push was *deliberately* declined on measured
  cost, and `--release` is required because four gates false-RED in debug
  and `fast_bpe_goat` is 388 s debug vs 15.6 s release. Price it first
  (one dispatch run, wall-clock + Actions minutes), then let the owner
  choose weekly / dispatch-only / a subset. **Do not add a per-push job for
  this.**
- [ ] **T4 — sweep the other 17 contract repos for both mechanisms.**
  `scripts/ci_gate_coverage.py` answers "does anything automatically start
  the compile/lint surface" and deliberately does not ask "does anything
  execute a test" — the two questions have different answers, as this issue
  shows. Extend it (or add a sibling report) to cross **compile coverage ×
  test execution**, the way it already crosses coverage × reachability. A
  report, not a gate, for the reason the others are.

## Gates

| Gate | Criterion |
|---|---|
| G1 | T1's AGENTS.md row exists and the docs gate stays green |
| G2 | T2's selftest lane FAILS on a target that reports `ok. 0 passed` — canaried by making one report zero, not argued |
| G3 | T2's population is derived from the tree, never typed; the derivation is asserted by a `selftest()` |
| G4 | T4's report distinguishes "compiles, not executed" from "not compiled" — pooling them reproduces the pooled-total mistake `.docs/10_audits/cfg_gated_silent_zero_pass.md` corrects twice |

## Honest caveats

- **Compile-only CI may be a deliberate cost decision**, and if so this
  issue is mostly T1: the gap is that the decision is written down nowhere,
  so a green full gate reads stronger than it is. T3 exists to make the
  choice explicit rather than implicit.
- The tests are not unrun in an absolute sense — agents run them on the
  workstation constantly, and AGENTS.md records workstation results
  (the 39 armed gates at `--release`, the 704-test `--workspace` count).
  The claim here is narrower and is the one that matters for rot: **no
  automatic trigger executes any of them**, so a regression is found when
  somebody happens to look.
- `--all-features` on a test RUN is not the same claim as on a compile: it
  is one configuration of many, and the `-p` vs `--workspace` axis
  AGENTS.md documents applies to execution too. T3 should not be sold as
  total coverage.

## References

- `.issues/713` T3 / T4 + `.docs/10_audits/cfg_gated_silent_zero_pass.md` — the arming half of this
- `.issues/704` (inert workflows) + `.issues/706` ("can fire is not does fire") — rungs 1-2
- `scripts/full_gate.sh` layers 1-6; `scripts/ci_gate_coverage.py` (T4's home)
- `seal-remake` `e1ead85` — the sibling instance, closed: guard `layer 3b`,
  per-target named floors, and `.benchmarks/002_png_vs_ktx2_host_cpu_rss.md`
  for the work that surfaced it
