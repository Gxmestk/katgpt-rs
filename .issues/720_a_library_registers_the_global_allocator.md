# Issue 720 — the root crate registers a `#[global_allocator]` **as a library**, so every downstream alloc gate is either a conflict or a silent zero

**Status:** OPEN — measured, not yet fixed. The fix has real cross-repo blast
radius and is an owner call on sequencing, not on whether. Found 2026-09-03
while closing riir-ai `.issues/855` Class 3.

## The defect

`src/lib.rs:257`:

```rust
#[cfg(debug_assertions)]
#[global_allocator]
static GLOBAL_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;
```

That is a **library** choosing the process allocator for every binary that
links it. Rust permits exactly one `#[global_allocator]` per binary, and the
choice belongs to the binary crate. `katgpt-core` gets this right two lines of
reasoning away — its own copy is `#[cfg(all(test, debug_assertions))]` and its
comment says why:

> Downstream consumers (katgpt-rs root, riir-engine, etc.) install their OWN
> `#[global_allocator]`; this static is `cfg(test)` so it does not exist when
> katgpt-core is consumed as a library — no double-declare conflict.

The root crate's is **not** `cfg(test)`, and cannot be: integration tests link
the lib as a dependency, where `cfg(test)` is false, and
`tests/kimi_k3_g4_alloc_free.rs` documents that it relies on exactly this
static. So the current shape is load-bearing *and* wrong, which is why it has
survived.

## What it costs downstream — two failure modes, opposite directions

**1. A conflict, visible only in debug.** Any downstream test that installs its
own allocator fails to compile — but only when `debug_assertions` is on AND the
feature that pulls the root crate is enabled. Measured instances:

| repo | target | shape |
|---|---|---|
| riir-ai | `riir-games-quest/tests/issue847_tpr_goat.rs` | FIXED in `b35f8b901` (riir-ai `.issues/855` Class 3) |
| riir-train | `riir-train-engine/tests/xhc_train_phase7.rs` | OPEN — identical shape, `#[cfg(debug_assertions)] #[global_allocator]` at line 320 |

The riir-ai fix guards on `not(feature = "quest_compression_draft")` because
that repo has exactly ONE feature pulling `dep:katgpt-rs`. **That fix does not
generalise:** `riir-train-engine` has at least five (`kimi_k3_train`,
`go-latent-steering`, `go-data-tools`, `bonsai-go`, and the `katgpt-rs/go`
edge at line 1122), so the equivalent guard is a five-term disjunction that
goes stale the next time someone adds a feature. That is the argument for
fixing it at the source rather than once per consumer.

**2. A silent zero, in every profile.** A consumer that does NOT install one
and does NOT link the root crate calls `get_alloc_stats()` against no tracking
allocator and gets a real `0` — indistinguishable from an allocation-free hot
path. riir-ai `.issues/856` was exactly this (a release stub returning a
fabricated `0`); this is the unobserved-zero sibling of it.

## Blast radius — measured 2026-09-03, not estimated

Across the 4 repos that consume `katgpt_core::alloc::{get,reset}_alloc_stats`:

```
consumers = 67    register their own = 49    RELY ON THE LIBRARY'S = 18
```

Per-FILE grep, so it under-counts consumers whose allocator lives in a
`tests/common/` module linked into the same target — treat 49 as a floor and
18 as a ceiling. Of the 18, most are `#[cfg(test)]` blocks inside
`katgpt-core/src/*.rs`, which are served by katgpt-core's own `cfg(all(test,
debug_assertions))` static and are unaffected. The ones that are not:

- `katgpt-rs/examples/kimi_k3_hello_world.rs` — links the root lib, so it is
  served by the static this issue proposes to remove.
- `riir-ai/crates/riir-poc/src/behavior_gate_poc.rs`
- `riir-ai/crates/riir-games-civ/src/civ/map_tick/mod.rs`
- `riir-train/crates/riir-train-gpu/tests/bench_558_issue490_t2_incremental_staging_goat.rs`
- `riir-train/crates/riir-train-gpu/tests/bench_490_anchor_accumulation_goat.rs`

**Those last four are the ones to check FIRST, and not because of this fix.**
A unit test inside `riir-games-civ` linking `katgpt-core` as a plain dependency
gets **no** tracking allocator today — katgpt-core's is `cfg(test)` on
*katgpt-core*, not on the consumer. If any of them asserts `allocs == 0`
without installing an allocator, it is passing over nothing **right now**, and
removing the root static changes nothing for it.

## Tasks

- [ ] **T1** — audit the five files above: does each install an allocator
      (directly or via a linked `tests/common`), and does each assert on the
      counter? Any that asserts without one is a live vacuous gate, independent
      of T2/T3, and should be fixed first.
- [ ] **T2** — the non-negotiable half, cheap and independent of the rest:
      every consumer that asserts on `get_alloc_stats()` gets a **liveness
      sentinel** — force a known heap allocation, assert the counter saw it,
      *before* the measurement. `riir-games-quest/tests/issue847_tpr_goat.rs`
      (riir-ai `b35f8b901`) is the reference implementation. This makes both
      failure modes loud and is worth doing even if T3 never happens.
- [ ] **T3** — owner call on sequencing: move the registration out of
      `src/lib.rs` and into each target that needs it. Correct, and the only
      thing that removes the conflict class rather than guarding it per
      consumer. Blocked on T2 across all repos first — without sentinels, T3
      converts a compile error into a silent zero, which is strictly worse and
      is the trade riir-ai `.issues/856` already refused once.
- [ ] **T4** — `riir-train-engine/tests/xhc_train_phase7.rs` needs *something*
      today; it is the one known live conflict. If T3 is deferred, it needs the
      five-term feature guard plus a sentinel. Prefer T3.

## Related

- riir-ai `.issues/855` Class 3 — the first instance, fixed `b35f8b901`
- riir-ai `.issues/856` — the fabricated-zero sibling (release stub returning 0)
- `.docs/10_audits/debug_release_profile_axis.md` — why this is invisible in one profile
- `.docs/10_audits/cfg_gated_silent_zero_pass.md` — the PROFILE dimension; the 3 debug-only
  load-bearing targets it reports are all alloc gates, i.e. this same tension
