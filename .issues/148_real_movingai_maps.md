# Issue 148 — Real MovingAI Benchmark Maps (Map-Fidelity Hypothesis Test)

**Status:** RESOLVED (2026-07-15)
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Prior issue:** [147](147_guided_pibt_maze_routing_exploration.md) (ht_chantry connectivity fix)
**Benchmark:** [440 GOAT](../.benchmarks/440_lllg_paper_repro_goat.md)

## Context

Issue 147 concluded that the remaining ht_chantry G1 gap (ratio 0.09 on the
synthetic `ht_chantry_approx`) was "map fidelity" — the synthetic maze had
5.9% corridor cells (2× the real map's 2.9%), and the recommendation was to
download the real MovingAI map to close the gap. Issue 148 tests that
hypothesis by downloading all 4 real MovingAI benchmark maps and re-running
the G1 gate against them.

## Acceptance criteria

- [x] Download real MovingAI maps for all 4 paper scenarios
- [x] Add a reusable `GridMap::from_movingai(text)` parser (leaf primitive)
- [x] Unit tests for the parser (5 tests: basic, obstacle chars, short rows,
      malformed rejection, lenient type line)
- [x] Update the G1 gate to run against the real maps
- [x] Keep synthetic approximations as diagnostic comparisons
- [x] Document the result honestly (whether the hypothesis held or not)
- [x] Update the GOAT gate status and benchmark doc

## What was done

1. Downloaded `mapf-map.zip` from https://movingai.com/benchmarks/mapf/ and
   extracted `empty-48-48.map`, `random-64-64-10.map`,
   `warehouse-10-20-10-2-2.map`, `ht_chantry.map` into
   `crates/katgpt-core/benches/data/`.

2. Added `GridMap::from_movingai(text)` to `position.rs` — parses the
   standard MovingAI format (`type octile / height H / width W / map / rows`).
   Per the MAPF benchmark convention, only `.` is passable; all other chars
   (`@`, `O`, `T`, `W`, `S`, ...) are walls. Returns `Option<Self>` so callers
   can fall back to synthetic generators if a download fails.

3. Added 5 unit tests in `tests.rs` covering: basic parsing, all-obstacle-char
   handling, short-row edge case, malformed-input rejection, lenient type line.

4. Updated `generate_maps()` in the bench to load the 4 real maps via
   `include_str!`, keeping the 4 synthetic approximations as diagnostic
   comparisons (printed but not counted toward the gate).

5. Updated `paper_targets` and the G1 gate loop to use the `-real` suffixed
   names; maps not in `paper_targets` are now skipped by the gate (previously
   they ran against a default target of 15.0, which was misleading).

## Result: hypothesis partially confirmed

### ht_chantry — 3× improvement, but genuine gap remains

| Metric | Synthetic (Issue 147) | Real (Issue 148) |
|---|---|---|
| Throughput | 1.47 | **4.51** |
| Ratio vs paper (~17) | 0.09 | **0.27** |
| Completions (300 steps) | 442 | **1353** |
| Corridor cell density | 5.9% | 2.9% |

The map-fidelity hypothesis was **partially correct**: 3× of the prior ~12× gap
was indeed map fidelity (the synthetic's denser maze capped throughput). But
**0.27 is still below the 0.30 PASS threshold**, and a genuine ~4× algorithmic
gap remains (4.51 vs paper ~17).

**Implication:** Full Guided-PIBT (flow direction assignment) is now genuinely
warranted for maze topology — the map-fidelity excuse is exhausted. The
`CounterFlowHindrance` infrastructure from Issue 147 is available but needs to
be promoted from the 3rd PIBT tiebreak to a higher-priority cost term.

### warehouse — unexpected: size doesn't help

| Metric | Synthetic | Real |
|---|---|---|
| Passable cells | 1971 | **9776 (5×)** |
| Throughput | 7.57 | **7.34** |
| Ratio vs paper (~18) | 0.42 | **0.41** |
| Max stops/cell | 190 | **8** |

The real warehouse is 5× larger with wider aisles, yet throughput is
**unchanged**. This proves warehouse congestion is **not size-limited** — it's
a genuine algorithmic limit on shelf-aisle topology. The low max_stops=8 on
the real map (vs 190 on the synthetic) confirms agents aren't deadlock-stuck;
they just complete tasks too slowly.

**Implication:** Warehouse needs a different fix than map size — likely
intersection reservation or shelf-aware goal assignment, not global routing.

### empty/random — sanity check passed

- empty-48-48: identical (synthetic is exact by construction).
- random-64-64-10: -1.4% (17 fewer passable cells in real map; negligible).

## GOAT gate status (updated)

| Gate | Status | Detail |
|---|---|---|
| **G1** | **PARTIAL 2/4** | empty 0.68 ✅, random 0.68 ✅, warehouse 0.41 ❌, ht_chantry 0.27 ❌ (MARGINAL). All 4 maps now real MovingAI files. |
| **G2** | **FAIL** | Warm-start non-consumable (ratio 1.00). Unchanged. |
| **G3** | **PASS** | 1590 tests pass (5 new parser tests added). Clippy clean. |
| **G4** | **PASS** | 222ms median at 1000 agents (<500ms target). |

## Files changed

| File | Change |
|---|---|
| `crates/katgpt-core/src/multi_agent_path/position.rs` | Added `GridMap::from_movingai(text)` constructor (+66 lines) |
| `crates/katgpt-core/src/multi_agent_path/tests.rs` | Added 5 parser unit tests (+82 lines) |
| `crates/katgpt-core/benches/data/*.map` | 4 real MovingAI map files (new, ~43KB total) |
| `crates/katgpt-core/benches/bench_440_lllg_paper_repro.rs` | `generate_maps()` loads real maps; gate loop skips diagnostic-only maps; verdict text updated |
| `.benchmarks/440_lllg_paper_repro_goat.md` | Issue 148 section + updated TL;DR + GOAT table |
| `.issues/.highwater` | 147 → 148 |

## Commit

`feat: real MovingAI maps for LLLG G1 gate (Issue 148) — ht_chantry 0.09→0.27, warehouse unchanged`
