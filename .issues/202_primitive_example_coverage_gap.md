# Issue 202 — Primitive Example Coverage Gap

**Filed:** 2026-07-29
**Origin:** Plan 561 second-session re-audit (the prior session's "lone exception" claim was incorrect).
**Severity:** Documentation gap (no behavior impact).

## Problem

Multiple public primitives in `katgpt-core` ship **zero example harnesses**. The
prior session (commit `dfecdeef`) claimed `transformer_inversion` was "the lone
exception" — that claim was wrong. An independent re-audit (`grep -rl
'katgpt_core::<mod>' examples/` + broad alternate-name grep) found at least 9
genuine user-facing primitives with no example coverage at all.

Every public primitive should have at least one reference example showing its
API surface + intended use case — this is established practice (CNA has 3
examples, EGA has 4, MUX has 4, ATTN_MATCH has 5; 224 total example files in
the repo).

## Affected primitives (verified zero example coverage)

### DEFAULT-ON primitives (highest priority — shipped but undocumented)

| Primitive | Feature gate | Plan | Why it matters |
|---|---|---|---|
| `conformal` | `conformal_predictive_intervals` (DEFAULT-ON) | Plan 340 | **The mandated UQ baseline** — AGENTS.md requires every UQ-bearing primitive to benchmark against `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`. Zero examples for a mandated baseline is the worst gap. |
| `best_belief` | `best_belief` (DEFAULT-ON) | Plan 336 | ε-quantile Beta lower bound for conservative selection. Grandfathered UQ primitive per Issue 010. |
| `ssmax` | `ssmax_temperature` (DEFAULT-ON) | Plan 411 | Length-aware log-N attention temperature (SSMax). |
| `poincare` | `poincare_navigator` (DEFAULT-ON) | — | Poincaré navigator (hyperbolic geometry). |

### Opt-in primitives (lower priority — not in default build)

| Primitive | Feature gate | Plan |
|---|---|---|
| `newton_schulz` | `newton_schulz` | Plan 152 (Muon optimizer) |
| `qgf` | `qgf` | Q-gradient field |
| `faithfulness` | `faithfulness_probe` | Plan 244 (FaithfulnessProbe) |

## Proposed resolution

Build example harnesses prioritized by:

1. **`conformal`** (BLOCKING) — the mandated UQ baseline. An example here
   would show how to construct `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`,
   fit it on calibration data, and query intervals — the exact pattern every
   UQ primitive author needs. This is also the reference for Issue 010's
   "Report the Floor" rule.
2. **`best_belief`** — pairs naturally with `conformal` (both UQ primitives).
3. **`ssmax`** + **`poincare`** — DEFAULT-ON, deserve at least a minimal demo.
4. Opt-in primitives as time permits.

Each example should follow the established pattern: module doc comment with
"what this proves / what this does NOT prove", runnable demonstration of the
core API, honest scope note.

## Not included (internal substrate — examples not needed)

`alloc`, `linalg`, `traits`, `freeze`, `dec_freeze`, `proof_cache`,
`delta_mem`, `mcts_state_action_cache`, `thinking_mode`, `shard_embedding`,
`simd_lut_dequant`, `set_diffusion_schedule`, `content_store`, etc. — these
are infrastructure consumed by other primitives/modules, not standalone
user-facing primitives.

## Verification method

```bash
# For each module, check if any example imports it:
for mod in conformal best_belief ssmax poincare newton_schulz qgf faithfulness; do
  count=$(grep -rl "katgpt_core::$mod" examples/ 2>/dev/null | wc -l)
  echo "$mod: $count examples"
done
```

## Out of scope

- Building ALL examples in one pass — each is a separate focused task.
- Modifying any primitive's API — examples document existing API, don't change it.
- Plan 561 (`transformer_inversion`) — already has its example (commit `dfecdeef`).
