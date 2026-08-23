# Issue 683 — `MultiLayerKVCache` has no model-agnostic all-sliding constructor

**Filed:** 2026-08-23
**Status:** OPEN
**Owner repo:** katgpt-rs (modelless inference primitive — correct home)
**Consumer blocked:** riir-train [Plan 343](https://github.com/gist-rs/riir-train) T1.6 (Maglev drafter G4 cache invariant)

## What exists

`crates/katgpt-transformer/src/kv_cache.rs` already ships the hard part — a
bounded sliding-window KV cache with a **mirrored ring buffer**:

- `MultiLayerKVCache::sliding_capacity: Vec<usize>` — per-layer logical window
  (`0` = unbounded).
- `new_gemma4_sliding_bounded(...)` — allocates `2 · sliding_window · kvd` per
  sliding layer. Writes go to `pos % capacity` **and** are mirrored to
  `pos % capacity + capacity`, so any window of ≤ `capacity` positions is always
  contiguous in the buffer and the attention code needs **zero wrap-around
  handling**. Reads remap `t_start → t_start % capacity`.
- `sliding_capacity(layer_idx)`, `fill_pos`, `advance_pos`, `snapshot`,
  `restore` — the accessors a speculative/draft seam needs.

This is good substrate and nothing about it needs redesigning.

## The gap

The only constructor that turns sliding-bounding **on** is
`new_gemma4_sliding_bounded`, which is model-shaped twice over:

1. It is named for, and derives its layer pattern from, **Gemma-4's alternating
   sliding/full** schedule.
2. There is no way to say "**every** layer is sliding at window W" without
   going through that pattern, and no public setter for `sliding_capacity`.

So a consumer whose architecture is uniformly sliding cannot use the cache
without either borrowing a Gemma-4 config shape or reimplementing the ring.

## Why it matters (the concrete consumer)

riir-train Plan 343's Maglev drafter is **SSSS** — every layer sliding-window,
`W = 512` hard-pinned. That is the *strictly simpler* case than Gemma-4's
alternating pattern, and it currently has no constructor. Plan 343 T1.6's G4
gate is precisely "drafter KV cache == W per layer, zero growth across a run
≫ W tokens", i.e. an assertion over `sliding_capacity(layer_idx)`. Without this
constructor the likely outcome is a **parallel ring-buffer implementation in
riir-train**, which is the exact drift the substrate-first rule exists to stop.

## Proposed shape

```rust
/// Every layer sliding-bounded at `window`. Mirrored-ring allocation, same
/// invariant as `new_gemma4_sliding_bounded` (physical 2*window*kvd per layer).
pub fn new_all_sliding_bounded(config: &Config, window: usize) -> Self
```

Alternative (smaller, but weaker): make `sliding_capacity` settable per layer
after construction. Rejected as the primary because the *allocation* must agree
with the capacity — a setter that does not resize is a footgun that silently
produces out-of-range reads, and the mirrored-write contract is exactly the kind
of invariant that should be established once, at construction.

Prefer the constructor; it keeps "capacity implies allocation" true by
construction.

## Tasks

- [ ] **T1** — add `new_all_sliding_bounded(config, window)`; assert
      `window > 0` and reuse the existing mirrored-ring allocation path rather
      than duplicating it.
- [ ] **T2** — test the invariant the consumer will gate on: run ≫ `window`
      positions and assert per-layer buffer length is exactly
      `2 · window · kvd` and `sliding_capacity(l) == window` for every `l`
      (zero growth).
- [ ] **T3** — test the property that makes the mirror worth having: for any
      `t_start` with a window of ≤ `window` positions, the read slice is
      contiguous and equals the logically-expected positions — including across
      at least two wraps, which is where an off-by-one in the mirror shows up.
- [ ] **T4** — vocabulary: add `sliding`/`ring`/`window` aliases to the doc
      comment. `RingKvCache`, `SlidingWindowCache`, `WindowedKvCache`,
      `kv_ring`, `circular cache` currently all return **zero** grep hits
      across the workspace, so a consumer searching English names concludes the
      substrate does not exist. Cheap fix, prevents the re-implementation.

## Note

No new crate, no new dep, no new type — this is one constructor plus tests on a
primitive that already ships. Filed before any code per the substrate-first
consume-vs-build rule.
