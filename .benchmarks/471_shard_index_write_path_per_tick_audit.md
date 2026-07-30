# Audit 471 — ShardIndex Write-Path Per-Tick Caller Verification

> **Source:** Session 15 challenge of session 14's "rust-optimize pass genuinely
> complete" verdict on `riir-neuron-db`.
> **Date:** 2026-07-29
> **Verdict:** Session 14's "complete" verdict is **confirmed accurate**. No new
> benchmark warranted — this audit documents the trace so future sessions do not
> re-walk these paths.
> **No code changed.** No new bench shipped. This is a verification note only.

## TL;DR

Session 14 flagged two "potential future audit angles":

1. Whether `ShardIndex::insert` (write path, called at zone setup) or
   `ShardIndex::update_shard` (called on shard updates) has a per-tick caller
   beyond cold setup/update cadence.
2. Whether the `neuron_vessel_runtime` path (which also imports `ShardIndex`)
   has a per-tick consumer beyond `ZoneEggshellRuntime`.

**Both angles traced + closed — NO gap found.** Session 14's verdict holds.
Detailed traces below.

## Angle 1: ShardIndex write paths

### `ShardIndex::insert` — cold-path only ✅

Grep across the 7-repo stack for `shard_index\.insert\b` + `ShardIndex::insert`:

| Call site | Repo | Cadence |
|---|---|---|
| `build_runtime` helper in `zone_eggshell_goat.rs`, `zone_eggshell.rs` tests, `zone_eggshell_goat_gates.rs` | riir-engine | bench/test setup |
| `build_shard` helper in `neuron_vessel_runtime/tests.rs` | riir-engine | bench/test setup |
| `shard_read_retrieves_from_real_shard_index` in `metamemory_tests.rs` | riir-games-civ | test setup |
| `g5_modelless_deterministic` in `bench_476_sleep_transfer_g8_validation.rs` | riir-games-civ | bench setup |
| In-crate tests + benches | riir-neuron-db | test/bench setup |

**Zero production per-tick callers.** Every call site is in a `build_runtime`
/ `build_shard` / test / bench helper that constructs the index once before
measurement. This matches session 14's claim.

### `ShardIndex::update_shard` — per-tick code path but DEAD IN PRODUCTION ✅

This is the subtle case. Grep for `\.(update_shard|update_shards_batch)\s*\(`:

| Call site | Repo | Cadence |
|---|---|---|
| **`execute_admitted_ops`** in `riir-games-civ/.../cognitive_branch/mod.rs:1161-1166` | riir-games-civ | **per-tick code path BUT DEAD IN PROD** (see below) |
| `ConsolidationPipeline::consolidate` in `riir-neuron-db/src/consolidation/mod.rs:1297, 1607` | riir-neuron-db | Night-phase (once per in-game day) per the doc contract |
| `bench_012_fast_knn_update.rs`, `bench_457_scan_floor_profiler.rs`, `bench_476_sleep_transfer_g8_validation.rs` | various | benches |
| `neuron-db-sdk/src/lib.rs::update` | riir-neuron-db (CRUD layer) | **zero production callers** (see below) |
| In-crate tests + benches | riir-neuron-db | test/bench |

**The `execute_admitted_ops` site is the only candidate for a per-tick write
path.** It lives inside the civ map tick. BUT the body is:

```rust
MemoryOp::ShardDelta { delta, alpha } => {
    // `apply_delta` requires `delta.len() >= STYLE_DIM`.
    // The current write provider never produces `ShardDelta`
    // (8-dim HLA ≠ 64-dim style weights), but handle it defensively.
    if delta.len() >= STYLE_DIM {  // STYLE_DIM = 64; HLA = 8 → always false
        shard_index.update_shard(&zone_hash, |shard| {
            shard.apply_delta(&delta, alpha);
            shard.commitment
        }, &guard);
    }
}
```

**The branch is never taken in production** because the write provider
(verified at `cognitive_branch/mod.rs:895-917`) only ever emits
`MemoryOp::KgTripleEmit` or `MemoryOp::EngramAdmit` — never `ShardDelta`.
HLA is 8-dim; `STYLE_DIM` is 64; the guard `delta.len() >= STYLE_DIM` always
evaluates false. This is **dead code in the per-tick path today** — the same
pattern as `ExperienceGraph::latent_seeded_ns_traversal` (session 14 finding):
a substrate wired into the per-tick path but never actually invoked because no
producer feeds it.

**If a future feature starts producing `ShardDelta` per-tick, this becomes a
hot path with zero benchmark coverage.** The function body
(`papaya::HashMap::compute` write lock + shard clone + conditionally 4
`RwLock` acquisitions + O(n) `hull_index` rebuild) is genuinely hot-path-shaped.
That day is not today.

### `ShardIndex::update_shards_batch` — benches only ✅

Grep returned only `bench_012_fast_knn_update.rs` + in-crate doc examples. Zero
production callers.

### `neuron_db_sdk::update` (CRUD wrapper) — zero production callers ✅

The SDK CRUD `update<T>` function wraps `ShardIndex::update_shard`. Grep for
`neuron_db_sdk::update` across the stack:

| Call site | Repo | Cadence |
|---|---|---|
| `game_sync/inventory.rs` lines 29, 54 | riir-games | **doc comment only** (`//!`), not actual code |
| `bench_458_scan_projection_goat.rs` | riir-neuron-db | bench |
| `issue_021_projection.rs` | riir-neuron-db | test |
| `crud_in_mem_vs_on_disk.rs`, `game_data_crud.rs` | riir-neuron-db | example |

**Zero production callers.** The Warm-tier persistence path (Plan 013 in
riir-mmorpg-examples) is the only consumer that would call it, and that path
goes through the chain, not the SDK CRUD layer directly.

## Angle 2: `neuron_vessel_runtime` per-tick consumer

Grep for `ShardIndex` across
`riir-ai/crates/riir-engine/src/neuron_vessel_runtime/{bridge,cold_source,mod,runtime}.rs`:

**Zero matches in production runtime files.** `ShardIndex` appears ONLY in
`neuron_vessel_runtime/tests.rs` (test helper `build_shard`).

The production runtime uses `NeuronShard` directly — passed by value via
`load_zone(zone, &shard)`. There is no `ShardIndex` field on
`NeuronVesselRuntime`.

### `NeuronVesselRuntime` itself — zero production instantiations ✅

Grep for `NeuronVesselRuntime::(new|with_default_config|with_cold_source)`:

**Every call site is in tests, benches, or doc examples.** Zero production
instantiations. The `neuron_vessel_runtime` is a substrate without a
production consumer today — same status as
`ExperienceGraph::latent_seeded_ns_traversal` (session 14 finding).

## Bonus: `enter_zone` third-layer check

Re-read `riir-engine/src/zone/eggshell.rs:189-242` to verify session 14's
claim that `enter_zone` has exactly two papaya layers (no hidden third lock):

```rust
pub fn enter_zone(&self, zone_hash: ZoneHash, ...) -> io::Result<Arc<ValidatedZoneView>> {
    let guard = self.shard_index.guard();                       // papaya op #1 (ShardIndex)
    let shard = self.shard_index.get(&zone_hash, &guard)?;      // papaya op #2 (ShardIndex)
    // ... capture raw_state fields ...
    self.cache.get_or_regen(zone_hash, topology_version, || {   // papaya op #3 (ZoneGeometryCache — Bench 469)
        regen_zone_to_mmap(shard, ...)
    })
}
```

**Exactly two papaya HashMaps touched** (`shard_index` + `cache`), as session 14
documented. No third lock layer. The comment at L194-198 confirms the guard is
held across the regen closure intentionally so the borrowed `&NeuronShard`
stays valid for `regen_from_shard_and_raw`.

## Bonus: `ShardIndex::query` per-tick caller check

Grep for `ShardIndex::query` consumers across the stack (filtering out
unrelated `PkmIndex::query`, `bevy::query_filtered`, etc.):

| Call site | Repo | Cadence |
|---|---|---|
| In-crate tests (`index/tests.rs`, `neuron_db_fv.rs`) | riir-neuron-db | tests |
| In-crate benches (`bench_011`, `bench_337`, `neuron_db_crud_bench.rs`) | riir-neuron-db | benches |
| `katgpt-rs/examples/recos_goat.rs`, `similarity.rs` | katgpt-rs | **comments only** — no actual call |
| `riir-poc/src/experience_graph_poc.rs` | riir-poc | comment only |
| `riir-games/examples/quest_yaml_demo.rs` | riir-games | comment only |

**Zero production per-tick callers.** Session 14's claim that Bench 011/012/337
cover the k-NN query path holds — the substrate is well-benchmarked, and no
production code exercises it today.

## Final verdict

Session 14's "genuinely complete" verdict on the `rust-optimize` pass for
`riir-neuron-db` is **confirmed accurate**.

The per-tick hot path is covered end-to-end across six layers (MAPE-K,
substrate, vibe, freeze gate, zone cache leaf primitives, ShardIndex papaya
read — Benches 464-470). The write paths (`insert`, `update_shard`,
`update_shards_batch`, `query`) have zero production per-tick callers — they
are cold-path (init), Night-phase (once/day), or dead code in the per-tick
path (the `ShardDelta` branch that no producer feeds).

### When to re-open this audit

Re-open if ANY of the following materializes:

1. **A producer starts emitting `MemoryOp::ShardDelta` per-tick** in civ or any
   other consumer. The `update_shard` body
   (`papaya::HashMap::compute` + 4 `RwLock` + O(n) `hull_index` rebuild) is
   genuinely hot-path-shaped; the dead-code status is a producer-side accident,
   not an inherent property.
2. **`neuron_db_sdk::update` gains a production caller** (e.g. the Warm-tier
   persistence path in riir-mmorpg-examples starts using the CRUD layer
   directly instead of going through the chain).
3. **`NeuronVesselRuntime` gains a production instantiation** (currently
   tests/benches/doc-examples only).
4. **`ShardIndex::query` gains a production per-tick caller** (currently
   in-crate tests/benches only).

Until then, no further optimization work is warranted on `riir-neuron-db`.
