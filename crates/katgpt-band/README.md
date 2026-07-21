# katgpt-band

Band-Conditioned KV Segment Selector cluster — band conditioning set, BCKVSS,
Collider-Consistency `ConstraintPruner`, Adaptive CoT stopper (Plan 265,
arXiv:2605.12733). Extracted from `katgpt-rs` root per Proposal 003 Phase 11
(2026-07-04).

## Overview

Four tightly inter-coupled modules built around band conditioning +
conditional independence tests. `bckvss` and `collider_pruner` both build on
`band_conditioner::ComputeTarget` and `BandConditioningSet`; they share paper
origin (Plan 265) and the conditional-independence-test substrate. Splitting
would either duplicate the substrate or force awkward one-way deps, so they
live as one crate.

## Key types / modules

- `band_conditioner` — the cluster's foundation (`BandConditioningSet`,
  `ComputeTarget`, `conditional_dependence_fisher_z`). Gates
  `band_conditioner`. **DEFAULT-ON** (Plan 265 T5.3, 2026-07-02): G0a/G0b
  pass.
- `bckvss` — Fusion A: Band-Conditioned KV Segment Selector. Gates `bckvss`.
  Opt-in.
- `collider_pruner` — Fusion C: `ColliderConsistency` `ConstraintPruner` for
  DDTree (impls `katgpt_core::{ConstraintPruner, PreservationScorer}`). Gates
  `collider_consistency`. **DEFAULT-ON** (Plan 265 T5.3, 2026-07-02): G7–G9
  pass.
- `adaptive_cot_stopper` — Fusion D: theory-backed adaptive CoT stopping
  criterion. Gates `adaptive_cot_identifiability`. Opt-in (pending GOAT G10).

## Feature flags

`default = ["band_conditioner", "collider_consistency"]`.

| Feature | Default | Description |
|---|---|---|
| `band_conditioner` | yes | Band conditioning set + CI test primitives (Plan 265 Phase 0). |
| `bckvss` | no | Band-Conditioned KV Segment Selector (Plan 265 Phase 1). Implies `band_conditioner`. |
| `collider_consistency` | yes | `ColliderConsistency` `ConstraintPruner` for DDTree (Plan 265 Phase 3). Implies `band_conditioner` + `katgpt-core/local_branch_routing`. |
| `adaptive_cot_identifiability` | no | Theory-backed adaptive CoT stopper (Plan 265 Phase 4). Implies `bckvss`. |

## Dependencies

- `katgpt-core` — `sigmoid` (all four modules), `ConstraintPruner` +
  `PreservationScorer` traits (collider_pruner).
- `fastrand` — deterministic RNG for `bckvss::SyntheticScm`'s AR(1) sampler.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
