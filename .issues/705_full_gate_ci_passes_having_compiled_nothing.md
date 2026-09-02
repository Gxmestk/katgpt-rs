# Issue 705 — the full gate's first two CI runs passed having compiled zero units

Status: **ROOT CAUSE FOUND AND FIXED — and it was not what the first fix
assumed.** The vacuous pass was never about the build cache. Every counter in
`scripts/full_gate.sh` is `^`-anchored, and the workflow sets
`CARGO_TERM_COLOR: always`, so every line began with an escape sequence and
**every count matched zero — including the error count**. The gate was
structurally incapable of FAILING in CI. Proven by revert probe: the original
script reports `✓ full gate PASSED — 0 errors` on a log containing a colourised
`error[E0425]`. Fixed by stripping ANSI before any counting (`ad0b7b19` added
the sentinel that caught it; the strip lands in this commit).

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

## The real root cause — the first hypothesis was wrong, and how it was caught

The initial fix assumed a restored build cache made cargo consider everything
fresh. That was **wrong**, and the thing that disproved it was the second half
of the fix: uploading the gate log on `always()`. With the log finally
retrievable from a green-turned-red run, the artifact from `33530563741` shows

```
    Checking katgpt-types v0.2.1 (/Users/runner/work/katgpt-rs/...)
warning: using `chunks_exact` with a constant chunk size
```

— 3,471 lines of real output. The build was never the problem. Measured on that
exact artifact:

| counter | as the gate read it | ANSI stripped |
|---|---|---|
| units compiled | **0** | **32** |
| warning lines | **0** | 369 (297 findings across 72 targets) |
| error diagnostics | **0** | 0 |

`CARGO_TERM_COLOR: always` is set in the workflow env, so a line arrives as
`ESC[1m ESC[92m    Checking ESC[0m katgpt-types …` — it does not start with
whitespace, `warning`, or `error`, and every `^`-anchored grep in the script
misses it. Locally the same script works *by accident*: cargo suppresses colour
when stdout is not a TTY, and there it is redirected to `$LOG`.

**The serious half is the error counter.** `DIAGS` was defeated identically, so
`✗ full gate FAILED` was unreachable in CI. A completely broken workspace would
have produced the same green checkmark. Revert-probed against the pre-fix
script to make sure this is a fact and not an inference:

```
ORIGINAL gate, colourised error: exit=0 :: ✓ full gate PASSED — 0 errors, 0 unbuildable targets
```

**The sentinel was right for the wrong reason, and that is the argument for it.**
Layer 3b refused to certify a census it could not perform, so the run went
INCONCLUSIVE rather than green — while its printed *diagnosis* blamed the cache.
A guard that fails closed on "I cannot measure this" beats one that has to
predict the cause correctly. Its message now names both causes and tells the
reader how to tell them apart: check whether the log HAS `Checking`/`warning`
lines.

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
- [x] ANSI stripped before every count, canaried in place: colourised
      compile+warnings PASSES with correct nonzero counts, colourised ERROR
      FAILS (it passed before), plain error still FAILS.
- [x] One conclusive CI run. **Run `33531188553`, 2026-09-01:**
      `✓ full gate PASSED — 0 errors, 0 unbuildable targets (297 warning
      finding(s) across 72 target(s), not gated; 32 unit(s) compiled)`, 2m17s.
      The first run in this gate's CI history that verified anything.
- [x] Reconcile the census gap. **Not a defect — different compilers.**
      Workstation `rustc 1.93.0` (clippy 0.1.93); CI `rustc 1.98.0` via
      `dtolnay/rust-toolchain@stable`. Five releases apart, and clippy gains
      lints across them: the dominant finding in the CI artifact is
      `chunks_exact_to_as_chunks`, whose help URL is literally
      `rust-clippy/rust-1.98.0/...` and which does not exist in 1.93. So
      **119/20 is a 1.93 number and 297/72 is a 1.98 number; they are not
      comparable and neither is wrong.** The gate's own preamble anticipates
      this ("a weekly run that reds on a toolchain bump is the gate working").
      `.issues/701` R3b's 119/20 baseline is implicitly toolchain-pinned —
      do not compare it to a CI figure without stating both `rustc -V`.
- [ ] Owner call, now that the number exists: `full_gate.yml`'s promotion
      criterion for per-push is satisfied on TIME (2m17s, vs the >13 min the
      estimate assumed) but is a BILLING question — macOS bills at a multiple
      of Linux, so ~23 Linux-equivalent minutes per push with several agents
      pushing daily. Measurement recorded in the workflow preamble; the
      decision is not taken from here.
- [x] Read the wall-clock off that run — **DONE 2026-09-02**: run
      `33531570320` (push, success, 2026-09-01 16:24→16:26:33 UTC) is the
      first verifying run: **2m17s, 297/72, 0 errors** — and the number is
      quoted in `full_gate.yml`'s preamble as the per-push cost, which is
      exactly where this condition said it must land.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `ad0b7b19` (the fix), `.issues/704` (made the gate live at all),
`.issues/701` R3b (the 119/20 workstation baseline this is measured against),
`.issues/702` (the same vacuous-green class in the doc auditors).
