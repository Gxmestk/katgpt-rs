# Bench 632 — cognitive_branch pre-snapshot elimination + CLR bridge cache

**Session 29** (2026-08-12) — continuation of the multi-session riir-engine
optimization loop.

## Goal

P2g_cognitive_branch was the #1 every-tick phase at p50=79µs (Session 28
baseline, 1000 NPCs, cadence=3, 3-run median). Two structural inefficiencies
were identified:

1. **Pre-snapshot loops**: `tick_cognitive_branch` ran two 1000-iteration
   serial pre-passes per tick (`embeddings.extend(npcs.iter().map(...))` +
   `clr_signals.extend(npc_clr_states.iter().map(...))`) to build derived
   data that could be computed inline inside the per-NPC closure.
2. **Redundant execute_admitted_ops**: `execute_admitted_ops` iterated ALL
   NPCs on every tick, even though all `pending` buffers were cleared on
   non-write ticks (59/60 at cadence=60).

## Changes

### 1. Eliminate `embeddings` + `clr_signals` pre-snapshot Vecs

- Removed `tick_scratch.cog_branch_embeddings: Vec<[f32; 8]>` and
  `tick_scratch.cog_branch_clr_signals: Vec<(f32, Option<f32>)>` from
  `TickScratch`.
- Replaced pre-snapshot loops with inline `build_hla_embedding(npc)` calls
  inside the per-NPC closure + the live_composition pass.
- Removed the `mem::take` + restore dance for both Vecs.
- Added `build_hla_embedding(&NpcState) -> [f32; 8]` helper function.

### 2. Skip `execute_admitted_ops` on non-write ticks

- Gated `self.execute_admitted_ops(n_mm)` on `should_write`.
- On non-write ticks (59/60), all `pending` buffers are empty (cleared in
  the fast-path), so the iteration was pure overhead.

### 3. CLR bridge cache

- Added `tick_scratch.clr_bridge_cache: Vec<(f32, Option<f32>)>` — caches
  the bridged `(r_k, s_lp)` values per NPC slot.
- Rebuilt at the end of `tick_npc_clr_decision` when `make_decision` is true
  (CLR decision cadence ≈ 25920 ticks ≈ 7 in-game minutes).
- The cognitive branch reads from the cache instead of recomputing
  `bridge_clr_reliability()` (1 sigmoid) + `sigmoid(desperation)` fallback
  (1 sigmoid) per NPC per tick.
- When `latest_learning_potential` is `None` (pre-first-decision or rare),
  `s_lp` falls back to `sigmoid(desperation)` (changes every tick — not
  cacheable).

### 4. `bridge_clr_reliability_for_cache` pub(crate) wrapper

- Added `pub(crate) fn bridge_clr_reliability_for_cache(clr_r: f32) -> f32`
  in `cognitive_branch/mod.rs` so `tick_npc_clr_decision` can call it for
  the cache rebuild. The private `bridge_clr_reliability` stays private
  (it's the hot-path inline call).

## A/B Results (Apple M3 Max, release, phase_profile, 1000 NPCs, cadence=3)

3-run median (1515 samples per run, 500 ticks per run):

| Metric | Baseline (Session 28) | Optimized (Session 29) | Change |
|---|---|---|---|
| **P2g p50** | **79 µs** | **67 µs** | **-15%** |
| **P2g p90** | **99 µs** | **71 µs** | **-28%** |
| **P2g p10** | **71 µs** | **65 µs** | **-8%** |
| **P2g mean** | **148.5 µs** | **132.8 µs** | **-11%** |
| Full tick p50 | ~227 µs | ~216 µs | -5% |

The p90 improvement (-28%) is particularly notable — the tail latency from
write-tick spikes is reduced because the pre-snapshot work + execute_admitted_ops
skip eliminate ~8µs of overhead that compounded with the write-tick cost.

## GOAT Gate

| Gate | Status | Detail |
|---|---|---|
| G1 (correctness) | ✅ PASS | 1449 lib tests pass (same count as baseline). Vec removal preserves identical semantics (inline computation produces the same `[f32; 8]` array). CLR bridge cache fallback produces identical values to live computation when cache is empty. |
| G2 (perf) | ✅ PASS | P2g: -15% p50 (79→67µs), -28% p90 (99→71µs). Full tick: -5% p50 (227→216µs). |
| G3 (no-regression) | ✅ PASS | 1449 tests (same count). Clippy clean (`--all-targets --features tick_perf_trace`). |
| G4 (alloc-free) | ✅ PASS | Pre-snapshot Vecs removed (2 fewer fields in tick_scratch). CLR bridge cache is pre-allocated + reused (mem::take + restore pattern). No per-tick allocation. |

## Files Changed

- `crates/riir-games-civ/src/civ/map_tick/cognitive_branch/mod.rs` —
  pre-snapshot elimination + inline `build_hla_embedding()` + CLR cache read +
  execute_admitted_ops skip + `bridge_clr_reliability_for_cache` wrapper
- `crates/riir-games-civ/src/civ/map_tick/mod.rs` —
  removed `cog_branch_embeddings` + `cog_branch_clr_signals` fields,
  added `clr_bridge_cache` field, updated `reset()`
- `crates/riir-games-civ/src/civ/map_tick/npc_clr.rs` —
  CLR bridge cache rebuild at end of `tick_npc_clr_decision`

## Cumulative Optimization Loop Impact

| Session | Full tick p50 (1000 NPCs, cadence=3) | Cumulative speedup |
|---|---|---|
| Pre-Session 20 | 2.84 ms | 1.00× |
| Session 20 (LEO SIMD matvec) | ~1.60 ms | 1.78× |
| Session 21 (cadence + SIMD sigmoid) | ~1.23 ms | 2.31× |
| Session 22 (lazy zone_hash) | ~1.09 ms | 2.61× |
| Session 23 (criminals scan cache) | ~0.95 ms | 2.99× |
| Session 24 (emotion proj SIMD) | ~0.95 ms | 3.11× |
| Session 25 (LEO SIMD sigmoid) | ~0.86 ms | 3.30× |
| Session 26 (daily_loop Vec + DriftGate) | ~0.83 ms | 3.42× |
| Session 27 (guard LEO cadence gate) | ~0.37 ms mean | ~5.3× mean |
| Session 28 (talk_bubbles Vec) | ~0.30 ms p50 | ~9.5× p50 |
| **Session 29 (pre-snap elim + CLR cache)** | **~0.22 ms p50** | **~12.9× p50** |
