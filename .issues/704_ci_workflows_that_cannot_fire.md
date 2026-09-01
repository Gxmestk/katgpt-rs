# Issue 704 — seven sibling workflows cannot fire, and this repo's advertised weekly gate is one of the casualties

Status: **RESOLVED 2026-09-01 for the approved set; ONE row open.** Default
branch moved `main` -> `develop` on `riir-ai`, `riir-chain`, `riir-neuron-db`,
`riir-dao` (org `gist-rs`) and `katgpt-rs` (owner `katopz`), on the owner's
call — the reversible settings fix, chosen over promote-to-main so the frozen
v0.1.1 `main` and the promote process are both untouched. **Dead workflows went
7 -> 1**; the remaining one is `riir-game-sdk/nightly.yml`, outside the approved
set and a different shape (see below). Original status: measurement complete and the instrument is wired
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

## After the fix — re-measured, same instrument

```
▸ reachability (default branch vs where the workflow file lives)
  katgpt-rs              default=develop  docs_gate.yml[pull_request?,push,workflow_dispatch]  feature_isolation.yml[pull_request?,workflow_dispatch]  full_gate.yml[pull_request?,push,schedule,workflow_dispatch]  lean_proofs.yml[pull_request?,push,workflow_dispatch]  release-plz.yml[workflow_dispatch]  sibling_docs_drift.yml[workflow_call,workflow_dispatch]
  katgpt-web             default=?        deploy.yml[UNREACHABLE]
  riir-ai                default=develop  lean_proofs.yml[pull_request?,push,workflow_dispatch]
  riir-chain             default=develop  lean_proofs.yml[workflow_dispatch]  rust.yml[workflow_dispatch]  toolchain_drift.yml[schedule,workflow_dispatch]
  riir-clippy            default=develop  ops_dashboard.yml[schedule,workflow_dispatch]  rust.yml[schedule,workflow_dispatch]
  riir-dao               default=develop  rust.yml[workflow_dispatch]
  riir-deployer          default=develop  release.yml[push,workflow_dispatch]
  riir-game-sdk          default=main     nightly.yml[UNREACHABLE]  wasm32.yml[pull_request?,push]
  riir-neuron-db         default=develop  lean_proofs.yml[workflow_dispatch]  rust.yml[workflow_dispatch]

  1 workflow(s) cannot fire from any trigger — a gate that never runs is decoration, not coverage:
    - riir-game-sdk/nightly.yml
  schedule/workflow_dispatch need the file on the DEFAULT branch; push/pull_request
  need it on a branch their filter names.

  6 workflow(s) RUN but lost a declared trigger — the file fires on one
  trigger while another it declares (and its docs may advertise) never does:
    ! riir-chain/lean_proofs.yml: declared push — never fires
    ! riir-chain/rust.yml: declared push — never fires
    ! riir-clippy/rust.yml: declared push — never fires
    ! riir-dao/rust.yml: declared push — never fires
    ! riir-neuron-db/lean_proofs.yml: declared push — never fires
    ! riir-neuron-db/rust.yml: declared push — never fires

  1 workflow(s) NOT classified — no remote refs to read a default branch from, or PR-only in a repo whose
  policy this script cannot see. Unmeasured, not clean:
    ? katgpt-web/deploy.yml
```

**katgpt-rs's weekly full gate now fires.** `full_gate.yml` went from
`[pull_request?, push]` to `[pull_request?, push, schedule, workflow_dispatch]`
— the >13-minute all-axes gate that AGENTS.md has advertised as weekly since it
was written runs for the first time. `docs_gate.yml`, `feature_isolation.yml`
and `sibling_docs_drift.yml` regained their dispatch entry points; the last of
those was dead in the same commit that introduced it.

**riir-chain's `toolchain_drift.yml` weekly cron is live**, and the four
siblings' `rust.yml` / `lean_proofs.yml` are now at minimum manually
dispatchable — previously they had no live trigger at all.

### The six `declared push — never fires` rows are NOT a regression

They are the same `push: branches: [main]` lines as before, still no-ops because
work lands on `develop` and `main` carries no workflow files. That is each
repo's **recorded Actions-budget decision** ("main-only, the account is at its
spending limit"), not drift. Adding `develop` to those filters would trigger a
full build on every push — precisely the cost those owners declined. Left alone
deliberately: the fix restored the schedule/dispatch paths without touching
anyone's budget stance. Their gates run weekly or on demand, not per push.

### The one remaining dead workflow is out of scope here

`riir-game-sdk/nightly.yml` (daily cron) still cannot fire: that repo's default
is `main`, and unlike the four above its `main` *does* carry a workflow
(`wasm32.yml`, whose push trigger is live), so it is not the frozen-main shape
and flipping its default is a different judgement call. Not included in the
approved set; left for that repo's owner.

## Downstream: what the fix unblocked, and what running it then caught

`.issues/702`'s cadence condition was blocked on this issue and closed the same
day. `riir-ai`, `riir-chain` and `riir-neuron-db` now each run a weekly
`docs_drift.yml`, **dispatched and log-inspected** rather than assumed. Two
defects surfaced only because two instruments had to agree:

- The reusable workflow checked katgpt-rs out **inside** the audited tree, so
  the manifest walk descended into it — riir-neuron-db, which has zero inline
  Cargo comments, reported katgpt-rs's 396. Fixed `7bb438e3`.
- The mirror defect locally: the auditors read manifests **git has never seen**
  (riir-chain's untracked in-repo container-source copy), so the workstation
  said 4 where CI said 2. Untracked manifests feed the default closure, so this
  could flip a verdict on the workstation only. Fixed `ed5f4865`.

Both were invisible from one side alone, and both are the reason a green CI
wiring is worth nothing until someone reads a real run's log.

## Closing conditions

- [x] Measure the axis, with the three states kept distinct.
- [x] Wire it into `scripts/ci_gate_coverage.py` (additive; report, not gate).
- [x] Correct AGENTS.md's `full_gate.yml` weekly/on-demand claim.
- [x] Record the caveat inside `sibling_docs_drift.yml`.
- [x] Owner decision on default branch vs promote-to-main, per repo. **DONE
      2026-09-01** — default branch, applied to the five repos above.
- [x] Re-run `scripts/ci_gate_coverage.py`. **7 dead -> 1**, verified against
      the API *and* a re-derived `origin/HEAD` rather than the PATCH response.
- [ ] `riir-game-sdk/nightly.yml` — that repo's owner call, different shape.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `d2228161` (the sweep + reusable workflow), `.issues/701` R2 (same
one-repo-of-many shape for the compile/lint surface), `.issues/702` (same shape
for the doc auditors), `.issues/703` (same shape for hand-typed repo sets).
