# Issue 667 — dual-crate test invocation breaks: `test_prediction_scorer_basic` not gated behind `partial_scoring`

Filed + resolved same session (2026-08-17). Found during the Issue 029 T3/T4
heal sweeps (previous session) while running the dual-crate gate.

## Symptom

```
cargo test -p katgpt-speculative -p katgpt-pruners --lib
```

fails:

```
error[E0433]: cannot find type `EchoPredictionScorer` in this scope
error: could not compile `katgpt-pruners` (lib test) due to 1 previous error
```

Single-crate invocations (`-p katgpt-speculative` alone, `-p katgpt-pruners`
alone) and workspace-wide invocations do NOT fail — verified at HEAD
`4ccb2f280` via detached worktree (pre-existing, not introduced by the
Issue 029 heals).

## Root cause

Feature-unification gap between two different gates on one item:

1. `katgpt-speculative` dev-dep:
   `katgpt-pruners = { path = "../katgpt-pruners", features = ["echo_env_predictor"] }`
   — active only when katgpt-speculative is a TEST TARGET.
2. Under the dual invocation BOTH crates are test targets → dev-dep features
   unify → `katgpt-pruners/echo_env_predictor` = ON, `partial_scoring` = OFF.
3. `echo_env_integration` module compiles (gated on `echo_env_predictor`),
   its `#[cfg(test)] mod tests` compiles, and the un-gated
   `test_prediction_scorer_basic` references `EchoPredictionScorer` — which
   is `#[cfg(feature = "partial_scoring")]`-gated (it impls the
   `katgpt-core/partial_scoring`-gated `PartialScorer` trait) → E0433.

Why other invocations pass:
- `-p katgpt-pruners --lib` alone: default features = `[]` →
  `echo_env_predictor` OFF → module not compiled at all.
- `-p katgpt-speculative --lib` alone: katgpt-pruners is built as a dev-dep
  LIBRARY — its `#[cfg(test)]` module never compiles.
- workspace-wide: another member's dep set unifies `partial_scoring` ON, so
  the test compiles.

## Tasks

- [x] Reproduce at HEAD (dual invocation `--no-run` → E0433, exact message captured).
- [x] Fix: gate `test_prediction_scorer_basic` behind `#[cfg(feature = "partial_scoring")]`
      (the test exercises a partial_scoring-gated item; the gate mirrors the
      item gate — the layer-split convention).
- [x] Verify dual invocation compiles clean (`--no-run`).
- [x] Verify the test still RUNS when the feature is on
      (`-p katgpt-pruners --features echo_env_predictor,partial_scoring --lib test_prediction_scorer_basic`).
- [x] Verify the production surface unchanged (`cargo check -p katgpt-pruners --features echo_env_predictor`).

## Resolution

One-line cfg gate on the test fn. No behavior change in any shipped surface —
`echo_env_integration` lib code is untouched; only the test module's feature
honesty fixed.

Record: git history (this file removed per the noise-reduction rule).
