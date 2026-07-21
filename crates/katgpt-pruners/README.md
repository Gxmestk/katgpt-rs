# katgpt-pruners

Domain-specific constraint pruners for the DDTree search engine. Extracted
from `katgpt-rs/src/pruners/` (Plan 005, 2026-06-29).

## Overview

Constraint pruners sit between the speculative drafter and the verifier. They
look at the partial token sequence produced so far and decide which branches
of the draft tree are dead (prune now) vs which deserve further expansion
(keep). Different game domains have different constraint shapes — this crate
ships the dungeon/tactical/bomber implementations plus the substrate they
share.

The `bomber` sub-module stays in the `katgpt-rs` root crate (depends on
main-crate-only `transformer` / `inference_router` / `trigger_gate`); bomber
consumes this crate's `arena` + `game_state` modules via the path dep in the
root `Cargo.toml`.

## Key types / modules

Always-compiled:

- `pathfinder` — `Target`, `enumerate_targets`, `find_distance`, `find_path`,
  `reachable_positions`.
- `dungeon_pathfinder` — `DungeonAction`, `MultiFloorBlocked`,
  `MultiFloorTarget`, multi-floor pathfinding.
- `dungeon_pruner` — `DungeonMap`, `DungeonPruner`, `DungeonState`,
  `FloorGrid`, `StairConnection`.
- `map_generator` — `GeneratedDungeon`, `GeneratedMap`, `MapGenerator`.
- `tactical_pruner` — `GameState`, `TacticalPruner`.
- `game_state` — `GameState` trait (always compiled — no `bevy_ecs` dep, G1
  fix Plan 065).
- `freeze` — freeze/thaw pruner support.
- `emotion_vector` — emotion-direction pruner.
- `feature_class` — `FeatureClass` re-export shim (Plan 292 Phase 1).
- Re-export: `katgpt_core::thinking_mode::ThinkingMode` — canonical enum
  consumed by `katgpt_speculative::thinking_controller` and
  `efficiency_reward` (Plan 388 Phase 3 unified the duplicate enums).

Feature-gated:

- `future_probe` — frozen direction vector for forecasting future behavior
  probability (Plan 292 Phase 2, Research 267). **DEFAULT-ON** since
  2026-07-03 (all 4 real-model GOAT gates PASS on Gemma 2 2B).
- `subterranean` — Subterranean RNG-samurai bandit game domain.
- `spec_compile` — `SpecPruner` (compiled speculative pruner).

## Feature flags

`default = []`. The crate has ~50 opt-in features mirroring the historical
`katgpt-rs/src/pruners/` feature surface. Each gates one domain or auxiliary
module. See `Cargo.toml` for the full list — headline entries:

| Feature | Description |
|---|---|
| `bandit` | `bandit::Bandit` multi-armed bandit substrate. |
| `bandit_top_p` | Top-p truncation bandit variant. |
| `cna_steering` | CNA contrastive neuron attribution modulator (Plan 087). |
| `g_zero` | G-Zero GOAT-search bandit arena. |
| `dreamer` | Dreamer-style rollout policy. |
| `delta_mem` | Delta-memory pruner. |
| `tes_loop` | TES loop pruner. |
| `federation` / `federation_composer` | Federation pruners. |
| `hydra_budget` | Hydra adaptive layer budget (Research 148, Plan 165). |
| `data_gate` | Task-level gating for self-play stability (Plan 111). |
| `sr2am_configurator` | SR²AM Configurator context types (Plan 112). |
| `closure_instrument` | PTG + motif mining + PRI/CDG/TaR. |
| `ruliology` | Wolfram ruliology enumeration (Plan 188, Research 168). |
| `spechop` | SpecHop stage-specialized decode paths (Plan 102). |
| `questbench` | Underspecification scoring. |
| `mech_attribution` | Mechanistic-attribution pruner. |
| `nexus_elo` / `nexus_elo_proxy` | Nexus ELO ranking pruners. |
| `gepa_reflective` | GEPA reflective pruner. |

Plus `complexity_prior_sampler` / `mcts_k_prior` / `bandit_k_prior` /
`spec_k_prior` — K-prior sampling modules.

## Dependencies

- `katgpt-core` — shared traits + primitives.
- `katgpt-types` — shared types.
- `katgpt-transformer` — `TransformerWeights`.
- `katgpt-speculative` — DDTree substrate.
- `katgpt-percepta` — Percepta 2D hard-attention integration.
- `fastrand`, `rayon`, `log`, `blake3`, `serde`, `serde_json`, `postcard`,
  `bytemuck`, `half` — always-on.
- `bevy_ecs` *(optional)* — entity-component-system integration.
- `wasmi` *(optional)* — WASM validator runtime.
- `papaya` *(optional)* — lock-free hash map.
- `rustfft` *(optional)* — FFT-based pruners.
- `reqwest` *(optional)* — HTTP-based pruners (e.g. federation composer).

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
