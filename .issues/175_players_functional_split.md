# Issue 175 — `players.rs` Functional Split

## Context

Continuation of Issue 162 code-smell audit. The prior session marked
`src/pruners/bomber/players.rs` (2828 lines) as "out of scope for
mechanical extraction" claiming it had no natural seam. **Re-verification
contradicts that verdict.**

The file's own comments explicitly delimit 7 player types
(`// ── P1: Random ──`, `// ── P2: Greedy ──`, etc.), matching the
decomposition already used by sibling modules (`go/players.rs`,
`fft/players.rs`, `monopoly/players.rs` all ship alongside separate
`*_player.rs` files). This IS a natural seam — the prior session missed it
for the same reason it missed Issues 170–174: no actual line-count +
structure analysis was done.

## Numbers

- Before: `players.rs` = **2828 lines**
- Tests: only 97 lines (3%)
- 7 distinct player types + shared helpers + factory

## Plan

Functional split into a `players/` directory:

| File | Scope | Est. lines |
|---|---|---|
| `players/mod.rs` | imports + trait `BomberPlayer` + constants + types + factory functions + re-exports | ~250 |
| `players/helpers.rs` | shared utilities (`move_target`, `action_index`, `manhattan`, `in_blast_zone`, `update_bombs`, `update_powerups`, `update_opponents`, `predict_direction`, `count_escape_routes`, `trap_score`, `intercept_score`, `has_escape_route`, `is_safe_action`, `should_place_bomb`, `is_reverse`, `escape_distance`, `score_action`, `wall_density`, `has_adjacent_wall`, `sigmoid_scores`, `count_walkable`, `lora_score_actions`) | ~770 |
| `players/random_player.rs` | `RandomPlayer` | ~60 |
| `players/greedy_player.rs` | `GreedyPlayer` | ~115 |
| `players/validator_player.rs` | `ValidatorPlayer` | ~135 |
| `players/hl_player.rs` | `HLPlayer` (incl. `update_arm_q`, `mark_compressed`, `update_outcome`, `compress_cycle`, `freeze`, `thaw`, `check_safety`) | ~960 |
| `players/lora_player.rs` | `LoraPlayer` | ~180 |
| `players/lora_wasm_player.rs` | `LoraWasmPlayer` | ~230 |
| `players/nn_player.rs` | `NNPlayer` | ~160 |
| `players/tests.rs` | tests | ~97 |

All target files land under 2048. `hl_player.rs` is the biggest at ~960.

## External API surface (must be preserved from `players::`)

Verified by `grep -rn "players::"` across `src/pruners/bomber/`:

- Trait: `BomberPlayer`
- Player types: `RandomPlayer`, `GreedyPlayer`, `ValidatorPlayer`, `HLPlayer`,
  `LoraPlayer`, `LoraWasmPlayer`, `NNPlayer`
- Factory: `create_players_with_wasm`
- Helpers (consumed by 9+ external files): `in_blast_zone`, `score_action`,
  `should_place_bomb`, `is_safe_action`
- Constants/types: `ACTION_COUNT`, `ALL_ACTIONS`, `KnownBomb`

All re-exported from `players/mod.rs` via `pub use` / `pub(crate) use`.

## Tasks

- [x] Bump `.issues/.highwater` to 175
- [x] Move `players.rs` → `players/mod.rs` (verify build)
- [x] Extract `tests` block → `players/tests.rs`
- [x] Extract `helpers` (free functions) → `players/helpers.rs`
- [x] Extract `RandomPlayer` → `players/random_player.rs`
- [x] Extract `GreedyPlayer` → `players/greedy_player.rs`
- [x] Extract `ValidatorPlayer` → `players/validator_player.rs`
- [x] Extract `HLPlayer` → `players/hl_player.rs`
- [x] Extract `LoraPlayer` → `players/lora_player.rs`
- [x] Extract `LoraWasmPlayer` → `players/lora_wasm_player.rs`
- [x] Extract `NNPlayer` → `players/nn_player.rs`
- [x] Verify `players/mod.rs` < 2048 lines (155 lines ✓)
- [x] GOAT G1: `cargo test -p katgpt-core --lib` (default + bandit + bomber-wasm + contextual_bandit + binned_blend + kernel_blend + bomber-agent)
- [x] GOAT G3: `cargo clippy --workspace` clean
- [x] Final workspace sweep
- [x] Update Issue 162 to mark this entry DONE
- [x] Commit

## Final line counts

| File | Lines |
|---|---|
| `players/mod.rs` | 155 |
| `players/helpers.rs` | 783 |
| `players/random_player.rs` | 72 |
| `players/greedy_player.rs` | 125 |
| `players/validator_player.rs` | 148 |
| `players/hl_player.rs` | 1016 |
| `players/lora_player.rs` | 190 |
| `players/lora_wasm_player.rs` | 237 |
| `players/nn_player.rs` | 178 |
| `players/tests.rs` | 102 |
| **Total** | **3006** |

(Original `players.rs` was 2828 lines; +178 lines overhead from module
headers + import boilerplate — expected for a directory split.)

## GOAT evidence

- **G1** (correctness): 1558/1558 katgpt-core lib tests (default features); 374/374 katgpt-rs lib tests under `bomber-wasm`; full workspace sweep 1879+ passing across all crates (only pre-existing flaky latency test `jacobian_svd_r8x8_latency_gate` failed once on loaded hardware, passes in isolation)
- **G3** (no-regression): `cargo clippy --workspace --lib` clean across the whole workspace
- All feature combinations compile: default + `bomber-wasm` + `contextual_bandit` + `binned_blend` + `kernel_blend` + `bomber-agent` + `--all-features`

## Pre-existing unrelated failures (NOT caused by this split)

- `examples/ht_chantry_diagnostic.rs` needs `multi_agent_path` feature (reproduces against unsplit original)
- `tests/issue_156_anytime_lt2_poc.rs` has `elastic_override` argument mismatch (carried over from prior session)
- `subspace_phase_gate::tests::jacobian_svd_r8x8_latency_gate` is a flaky perf gate (passes in isolation; fails under loaded hardware)
