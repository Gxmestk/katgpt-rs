# Issue 683 — `MultiLayerKVCache` has no model-agnostic all-sliding constructor

**Filed:** 2026-08-23
**Status:** OPEN — ⚠ reframed 2026-08-24: the cache is UNWIRED (doc-only write contract) and has a working plain-modulo twin in riir-ai; adjudicate the convention (T0) before adding the constructor (T1)
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

## ⚠ CORRECTION (2026-08-24, same session) — the cache is UNWIRED, and it has a twin

Deeper grep after filing. Two findings that change this issue's framing, and the
first one invalidates part of my own reasoning above:

**1. `MultiLayerKVCache`'s sliding path has no implementation and no consumer.**
`sliding_capacity` has **zero** references outside `kv_cache.rs` anywhere in the
workspace, and `new_gemma4_sliding_bounded` is called **only by `kv_cache.rs`'s
own unit tests** (lines 747 / 789 / 881). The mirrored-write contract —
*"the forward code writes K/V at `pos % capacity` AND mirrors to
`pos % capacity + capacity`"* — exists **only as a doc comment**. No forward path
in any of the 15 repos does that write. So the design's sole advantage (windows
always contiguous ⇒ zero wrap-around handling in attention) is **unrealized**,
and the 2× memory is currently paid for nothing.

This is a doc-lie of the same class the substrate-first rule warns about, and it
is worse than a missing feature: a doc comment asserting what "the forward code"
does invites a consumer to trust an unimplemented contract. It is exactly what I
did one message earlier in riir-train Plan 343 T1.6 ("CONSUME, do NOT build") —
recorded there as a correction.

**2. A working twin ships in riir-ai.**
`riir-ai/crates/riir-gpu/src/gemma4_cubecl/kv_cache.rs::Gemma4CpuKVCache` has its
own `sliding_capacity: Vec<usize>` and a **real** `store()` that ring-writes at
`pos % sliding_capacity` with capacity `sw * stride` — **plain modulo, not
mirrored, 1× memory** — and it is exported (`pub use kv_cache::Gemma4CpuKVCache`)
and consumed.

| | katgpt-rs `MultiLayerKVCache` | riir-ai `Gemma4CpuKVCache` |
|---|---|---|
| allocation | mirrored `2·W·kvd` | plain `W·stride` |
| write | doc'd only, **unimplemented** | real `store()`, `pos % W` |
| consumers | its own unit tests | exported + used |

So the two repos disagree on the ring convention, and the *working* one is the
plain-modulo one in the downstream repo — while the upstream primitive repo holds
the unwired 2× design. That is the parallel-system drift this repo's rules exist
to catch, and it should be resolved before a new constructor is bolted onto
either side.

## Revised task ordering

T1 (the constructor) is no longer the first thing to do — adding an API to an
unwired cache is low value and would deepen the duplication. Decide the
convention first:

- [ ] **T0 (NEW, blocking) — adjudicate the duplication.** Either (a) implement
      the mirrored write in a katgpt-rs forward path and make riir-ai's Gemma-4
      GPU cache consume the upstream primitive, or (b) accept plain-modulo as the
      house convention, delete the unrealized 2× mirroring from
      `MultiLayerKVCache`, and fix the doc. **(b) is cheaper and matches the only
      code that actually runs**; (a) is only worth it if the contiguous-read
      saving in attention is measured, which it never has been.
- [ ] **T0b (NEW, cheap, do regardless) — fix the doc-lie now.** Reword the
      `sliding_capacity` comment from "the forward code writes…" to an explicit
      *"contract not yet implemented by any forward path; see Issue 683"*, so no
      future consumer trusts it. This is the highest value-per-byte item here.

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
