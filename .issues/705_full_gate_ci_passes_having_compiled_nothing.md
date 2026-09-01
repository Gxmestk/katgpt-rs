# Issue 705 — the full gate's first two CI runs passed having compiled zero units

Status: **FIXED, verification run in flight.** Liveness sentinel added to
`scripts/full_gate.sh` (Layer 3b) and the cache split in
`.github/workflows/full_gate.yml` (`ad0b7b19`). Awaiting the first conclusive
CI run to confirm the numbers converge with the workstation's.

## What happened

`.issues/704` moved the default branch to `develop`, which made the weekly full
gate live for the first time — its `schedule` and `workflow_dispatch` had never
fired. Its first two CI runs then both reported:

```
✓ full gate PASSED — 0 errors, 0 unbuildable targets (0 warning finding(s) across 0 target(s), not gated)
```

having compiled **zero units**.

| run | trigger | wall-clock | cache | Compiling/Checking lines | verdict |
|---|---|---|---|---|---|
| `33523516143` | push | 2m30s | `full match: true` | **0** | PASSED |
| `33529923375` | dispatch | 2m26s | `full match: true` | **0** | PASSED |

The same command on the workstation the same day reports **119 findings across
20 targets** (`.issues/701` R3b). A green built from nothing — the exact vacuous
pass this gate exists to catch, arrived at by the gate itself.

This is the sharpest instance yet of the rule this repo keeps restating: *treat
an uninvoked assertion as unknown, not as passing.* Here the assertion was
invoked, exited 0, printed a checkmark, and still verified nothing.

## Why it was invisible

Three separate things had to line up, and each is worth fixing on its own:

1. **The count that would have exposed it was never printed.** The summary
   reported warnings and targets but not units compiled. `0 warning finding(s)
   across 0 target(s)` reads as "clean" to anyone who does not already know the
   workstation number is 119/20.
2. **The evidence was destroyed with the runner.** `Upload gate log` was
   conditioned on `failure()`, so on a pass the log — the only place the
   absence of `Compiling` lines is visible — was unreachable. A pass whose
   working is unavailable is a claim, not a result.
3. **The script already documented the trap and did not guard it.** Its own
   comment reads: *"a second run against a now-warm target dir emits almost
   nothing, because cargo does not replay diagnostics for crates it considers
   fresh."* Knowing a failure mode is not the same as asserting against it.

## The fix

**Layer 3b, a liveness sentinel on two independent signals.** Either alone has a
blind spot:

- `UNITS` — cargo actually built something (`Compiling` / `Checking` lines).
- `TALLIES` — cargo **replayed** cached diagnostics without rebuilding, which a
  warm *local* re-run does; `.issues/701` R3b measured 119/20 exactly that way.

Zero of both means the run is reporting on its cache rather than the code, and
may no longer call itself a pass. A genuinely warning-free workspace with a
fully warm target dir lands here too — and "I cannot distinguish clean from
unmeasured" is the honest thing to say about that state, not a pass.

`UNITS` is now printed on **every** pass, not only when interesting.

**Canaried in place**, three shapes: no-op run → INCONCLUSIVE exit 1;
replay-only → PASS; real compile → PASS. *In place* because `full_gate.sh`
derives `REPO_ROOT` from its own path, so a `/tmp` copy resolves it to `/` and
dies at the platform layer before ever reaching the code under test — the
copy-vs-in-place trap, which cost two confusing canary rounds here.

**The cache split.** `cargo clean -p` on each workspace member before the gate:
dependencies are the expensive part and the gate makes **no claim** about them;
workspace members are the entire subject of the assertion and must be re-linted
every run. Deleting the whole cache would also be correct but pays a full cold
dependency build weekly for no added coverage. Members come from
`cargo metadata`, never typed — a hand-typed list silently stops covering a new
crate (`.issues/703`'s shape). 32 members at time of writing.

**Log uploaded on `always()`.**

## Closing conditions

- [x] Liveness sentinel, canaried in place across three run shapes.
- [x] Workspace artifacts dropped so the gate re-lints its actual subject.
- [x] Gate log uploaded on success as well as failure.
- [ ] One conclusive CI run: `UNITS > 0`, and the warning census within reach of
      the workstation's 119/20. **Until that lands, the gate's CI history
      contains no run that verified anything.**
- [ ] Read the wall-clock off that run — `full_gate.yml`'s preamble sets it as
      the promotion criterion for per-push and says "do not guess". The two
      vacuous runs' 2m30s is NOT that number and must not be quoted as it.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `ad0b7b19` (the fix), `.issues/704` (made the gate live at all),
`.issues/701` R3b (the 119/20 workstation baseline this is measured against),
`.issues/702` (the same vacuous-green class in the doc auditors).
