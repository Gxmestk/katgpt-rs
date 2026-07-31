# katgpt-percepta

Percepta-style **O(log N) 2D Attention** via Convex Hull KV Cache +
transformer-vm RIIR (Plan 064).

## Overview

Standard transformer attention computes `Q·K` for all N past keys → `O(N)` per
step. Percepta restricts attention heads to `d=2`, making the dot product a 2D
geometric projection. When keys form a convex hull, finding the maximum
attention score becomes ternary search over a unimodal (bitonic) sequence →
`O(log N)`.

Integration points with the rest of `katgpt-rs`:

- **DDTree branch pruning**: validate drafted tokens before target verification.
- **Deterministic Validator**: encode state-machine rules as 2D key embeddings.
- **"Free embedding" bridge**: project hidden states to 2D for fast retrieval.

## Key types / modules

- `legacy` — original `KVCache2D` (Graham Scan + ternary search), `Sudoku9x9`,
  `StreamingSolver`. Always compiled.
- `types` — shared types for the CHT hull vertices (`HullMeta`, `TieBreak`,
  `Vec2` f64). Gates `percepta`.
- `cht` — Dynamic Convex Hull Trick / `LineContainer` for `O(log h)`
  max-envelope queries. Gates `percepta`.
- `hull` — `HullHalf`, `HardAttentionHead`, `BruteAttentionHead` (O(log N) 2D
  hard attention). Gates `percepta`.
- `encoding` — parabolic key encoding helpers for 2D attention. Gates
  `percepta`.
- `cumsum` — cumulative sum via uniform attention (fetch_sum equivalent). Gates
  `percepta`.
- `transformer` — `VanillaTransformer` with ReGLU FFN, autoregressive
  generation. Gates `percepta`.
- Gate DSL (`gates`, `graph`, `wasm`, `compile`) — cumulative Percepta gate /
  expression / WASM / MILP scheduling ladder (TG-A through TG-J).

## Feature flags

`default = []`. The Percepta feature ladder is cumulative — each tier is a
strict superset of the previous; consumers enable the highest tier they need.

| Feature | Tier | Description |
|---|---|---|
| `percepta` | TG-A | CHT hull cache — upper+lower, `HullMeta`, tie-break, cumsum. Pulls `ordered-float`. |
| `percepta_gates` | TG-B | + ReGLU, stepglu, multiply, persist gate primitives. Implies `percepta`. |
| `percepta_graph` | TG-C | + Expression/Dimension DSL, `ProgramGraph`. Implies `percepta_gates`. |
| `percepta_wasm` | TG-E+F | + WASM decoder + lowering + interpreter — pure Rust, NOT wasmtime. Implies `percepta_graph`. |
| `percepta_compile` | TG-D+G-J | + MILP + weights + transformer + Futamura + CLI. Implies `percepta_wasm`. Pulls `good_lp` (highs + microlp). |

## Dependencies

- `ordered-float` *(optional)* — `Ord` wrapper for f64 (CHT breakpoints, TG-A).
  Gated by `percepta`.
- `good_lp` *(optional)* — MILP solver for scheduling (TG-D, highs primary +
  microlp fallback). Gated by `percepta_compile`.
- `log` — used by scheduler + specialize (both behind `percepta_compile`).

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
