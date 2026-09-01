# Issue 704 — seven sibling workflows cannot fire, and this repo's advertised weekly gate is one of the casualties

Status: **OPEN** — measurement complete and the instrument is wired
(`scripts/ci_gate_coverage.py` grew the axis, `d2228161`+); the two false
documentation claims in katgpt-rs are corrected; the *fixes* are each repo's own
call and two of them are outward-facing repo-settings changes, so they are NOT
made from here. Filed from katgpt-rs because the instrument lives here.

## The finding

`.issues/701` R2 asked which repos gate their full compile+lint surface in CI,
and `ci_gate_coverage.py` answered it by reading each repo's working tree. It
never asked the prior question: **can those workflows run at all?**

They largely cannot. Measured 2026-09-01 across the derived contract repos:

- **7 workflows have no live trigger whatsoever.**
- **6 more run, but lost a trigger they declare** — including the weekly full
  gate this repo's own AGENTS.md advertises.

A workflow that never executes is decoration. This repo already states the rule
one level down — *"treat an uninvoked assertion as unknown, not as passing"* —
and `scripts/docs_gate.sh`'s preamble is a monument to it (three assertions
existed, nothing invoked any of them, two were red). Issue 704 is that same
rule applied one level up: to the workflows themselves.

## The mechanism — two GitHub rules that differ, and the difference is the bug

| trigger | fires from | consequence when the file lives only on `develop` |
|---|---|---|
| `schedule` | **default branch only** | never fires |
| `workflow_dispatch` | **default branch only** | never fires, and never appears in the Actions UI |
| `push` / `pull_request` | the **pushed ref** / the PR merge commit | fine — *if* the branch filter names a branch that carries the file |
| `workflow_call` | the ref the **caller pins** | unaffected |

Four repos (`riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-dao`) plus
`katgpt-rs` and `riir-game-sdk` have **default branch `main`**, and `main` in
the first four carries **zero** `.github/` files — frozen at the 2026-07-31
`release: promote develop to main (v0.1.1)` commit while develop has moved on
by 2088 / 829 / 173 / 20 commits respectively.

So their workflows declare `push: branches: [main]` (an owner's Actions-budget
call, main-only by design) and the file that would answer that push does not
exist on `main`. Nothing fires. `workflow_dispatch` cannot rescue it for the
same default-branch reason. **These repos have had no CI running at all since
at least 2026-07-31.**

## Measured (verbatim, `scripts/ci_gate_coverage.py`)

```
▸ reachability (default branch vs where the workflow file lives)
  katgpt-rs              default=main     docs_gate.yml[pull_request?,push]  feature_isolation.yml[pull_request?]  full_gate.yml[pull_request?,push]  lean_proofs.yml[pull_request?,push,workflow_dispatch]  release-plz.yml[workflow_dispatch]  sibling_docs_drift.yml[workflow_call]
  katgpt-web             default=?        deploy.yml[UNREACHABLE]
  riir-ai                default=main     lean_proofs.yml[pull_request?,push]
  riir-chain             default=main     lean_proofs.yml[UNREACHABLE]  rust.yml[UNREACHABLE]  toolchain_drift.yml[UNREACHABLE]
  riir-clippy            default=develop  ops_dashboard.yml[schedule,workflow_dispatch]  rust.yml[schedule,workflow_dispatch]
  riir-dao               default=main     rust.yml[UNREACHABLE]
  riir-deployer          default=develop  release.yml[push,workflow_dispatch]
  riir-game-sdk          default=main     nightly.yml[UNREACHABLE]  wasm32.yml[pull_request?,push]
  riir-neuron-db         default=main     lean_proofs.yml[UNREACHABLE]  rust.yml[UNREACHABLE]

  7 workflow(s) cannot fire from any trigger — a gate that never runs is decoration, not coverage:
    - riir-chain/lean_proofs.yml
    - riir-chain/rust.yml
    - riir-chain/toolchain_drift.yml
    - riir-dao/rust.yml
    - riir-game-sdk/nightly.yml
    - riir-neuron-db/lean_proofs.yml
    - riir-neuron-db/rust.yml
  schedule/workflow_dispatch need the file on the DEFAULT branch; push/pull_request
  need it on a branch their filter names.

  6 workflow(s) RUN but lost a declared trigger — the file fires on one
  trigger while another it declares (and its docs may advertise) never does:
    ! katgpt-rs/docs_gate.yml: declared workflow_dispatch — never fires
    ! katgpt-rs/feature_isolation.yml: declared workflow_dispatch — never fires
    ! katgpt-rs/full_gate.yml: declared schedule workflow_dispatch — never fires
    ! katgpt-rs/sibling_docs_drift.yml: declared workflow_dispatch — never fires
    ! riir-ai/lean_proofs.yml: declared workflow_dispatch — never fires
    ! riir-clippy/rust.yml: declared push — never fires

  2 workflow(s) NOT classified — no remote refs to read a default branch from, or PR-only in a repo whose
  policy this script cannot see. Unmeasured, not clean:
    ? katgpt-rs/feature_isolation.yml (PR-only)
    ? katgpt-web/deploy.yml
```

## What this does NOT claim

Three states, deliberately kept apart, because collapsing them is how this
instrument would earn either a false green or a false red:

- **`katgpt-web/deploy.yml` is unmeasured, not dead.** That repo has no remote
  refs to read a default branch from. Scoring an unmeasured repo as a finding is
  the confident-green-over-nothing inversion, pointed the other way.
- **`pull_request` is conditional, not dead.** A `pull_request` run uses the
  workflow file from the PR's *merge commit*, so a develop-only workflow does
  fire on a PR. Whether a PR is ever opened is workflow policy git cannot see —
  several repos here land work directly on `develop` and never open one. Calling
  it live over-reports; calling it dead over-claims. Reported as its own state.
  `katgpt-rs/feature_isolation.yml` is PR-only and this repo does not use PRs,
  so in practice it does not run — but that is a policy fact, not a git fact.
- **`riir-clippy/rust.yml`'s dead `push` is benign and already recorded.** Its
  default branch IS `develop`, so its `schedule` and `workflow_dispatch` both
  work and its weekly gate genuinely runs. The `push: branches: [main]` line is
  a no-op consistent with that repo's own documented decision not to trigger per
  push on develop. The instrument showing a deliberate choice is not a defect.

## The casualty in this repo

`katgpt-rs/full_gate.yml` declares `schedule: '17 4 * * 1'` and
`workflow_dispatch`. **Neither has ever fired** — default branch is `main`,
which carries only `lean_proofs.yml` and `release-plz.yml`. Its own comment
reads *"never cancel a scheduled run mid-flight in favour of a push — the
schedule is the rot check."* The rot check had rotted.

AGENTS.md said `full_gate.yml` *"runs it weekly and on demand"*. Both halves
were false. Corrected in this commit to state the declaration and the measured
reality; the weekly full gate — the one gate that covers all four independent
axes AGENTS.md is built around — **has not run since it was written**.

`sibling_docs_drift.yml` (added `d2228161`) inherited the same trap: its
`workflow_dispatch` smoke-test entry point is dead for the identical reason.
Its `workflow_call` path is unaffected, because callers pin the ref. The
caveat is now written into the file rather than left as an unstated assumption.

## Why the fixes are not made from here

Each repo owns its own CI per `BOUNDARY.md`, and both available fixes are
outward-facing:

1. **Move the default branch to `develop`** (a GitHub repository setting).
   Correct for `riir-ai` / `riir-chain` / `riir-neuron-db` / `riir-dao` /
   `katgpt-rs`, whose work all lands on `develop` — it is what `riir-clippy` and
   `riir-deployer` already do, and both of their CI setups work. Changes how
   clones, PRs, and the repo landing page behave.
2. **Promote `develop` to `main`**, carrying the workflows. Restores the
   main-only push triggers as designed, but is a release action.

Neither is a code change and both are the owner's call. Recommended: (1) for the
develop-workflow repos, since it fixes the cause rather than resetting a clock
that will re-expire at the next 32-day gap.

## Interaction with `.issues/701`

701 R2's table is measured by this same script and its **existing columns are
unchanged** — the reachability output is appended as a separate section
precisely so 701's quoted rows stay valid. But 701 R2's *question* ("N of 18
repos gate the full surface") now has a prior qualifier: several of the repos it
scores are running nothing. Coverage and reachability are independent axes, and
reachability dominates. 701 is another agent's lane; this is noted, not edited.

## Closing conditions

- [x] Measure the axis, with the three states kept distinct.
- [x] Wire it into `scripts/ci_gate_coverage.py` (additive; report, not gate).
- [x] Correct AGENTS.md's `full_gate.yml` weekly/on-demand claim.
- [x] Record the caveat inside `sibling_docs_drift.yml`.
- [ ] Owner decision on default branch vs promote-to-main, per repo.
- [ ] Re-run `scripts/ci_gate_coverage.py`; expect 0 dead workflows.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `d2228161` (the sweep + reusable workflow), `.issues/701` R2 (same
one-repo-of-many shape for the compile/lint surface), `.issues/702` (same shape
for the doc auditors), `.issues/703` (same shape for hand-typed repo sets).
