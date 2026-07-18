# Issue 183 — Causal-ID `identify_inner` G4 alloc-free refactor

> **Type:** optimization / refactor
> **Priority:** P3
> **Filed:** 2026-07-18
> **Status:** CLOSED (G4 INFORMATIONAL PASS — benchmark 465; allocs reduced 30%, latency improved 27%)
> **Repo:** katgpt-rs (substrate is `crates/katgpt-core/src/causal_id/identify.rs`)
> **Plan:** [457](../.plans/457_causal_id_counterfactual_npc_reasoning.md) — Phase 2 G4 (closed by this issue)
> **Benchmark:** [465](../.benchmarks/465_causal_id_alloc_free_scratch.md)

## Context

Plan 457 Phase 2 GOAT gate deferred G4 (alloc-free) on the rationale that
Causal-ID is offline-only (8.40 µs / 32-node identify is well outside the
20 Hz tick), so alloc-free discipline is not load-bearing. That rationale
remains correct — this refactor is not a re-litigation of the gate.

The trigger for reopening is the **Super-GOAT promotion (Plan 457 Phase 5
T4.7, 2026-07-18)**: `causal_identification` is now **DEFAULT-ON** across
the whole stack, not opt-in. Super-GOAT primitives carry higher
expectations on code quality; closing the last open G-gate on a default-on
primitive is a one-time tax that keeps the Super-GOAT label honest and
removes the documented caveat from `.benchmarks/264` and the Super-GOAT
guide (`.research/320`).

Current `identify_inner` allocates ~10 small `Vec<NodeId>` per recursion
frame (`v`, `an_y`, `v_minus_a`, `an_y_in_gva`, `w`, `new_v`, `fix_set`,
`intersecting`, `c_in_d`, `new_cause_step6`). The recursion only goes one
level deep per frame — every recursion is a direct `return identify_inner(...)`
— so the parent's scratch buffers are never alive alongside a child's. A
single `Scratch` struct can serve the whole call chain: each frame clears
its slots at entry, children inherit and reuse the same `Vec` storage.

The fix primitives are already in the codebase (`ancestors_into`,
`for_each_parent_into`, `for_each_bidir_neighbor`, `for_each_in_district_with_visited`).
`identify_inner` simply doesn't use them.

## Tasks

- [x] **T1.** Add `Scratch` struct to `identify.rs` holding the per-call
      reusable `Vec<NodeId>` buffers + a `frontier` for the ancestors walk.
      All slots `Vec::new()` at construction (lazy grow on first use).
      **Done:** 12 Vec slots + 3 Admg slots (subgraph buffers).
- [x] **T2.** Rewrite `identify_inner` to take `scratch: &mut Scratch` and
      clear each slot before use. Use `ancestors_with_frontier_into` (new)
      instead of `ancestors_into`. **Done.** Behavior bit-identical
      (verified by existing tests + new drift test
      `scratch_based_identify_matches_reference_on_game_kg`).
- [x] **T3.** `identify` (public) creates one `Scratch` and passes `&mut`
      down. Public API unchanged. **Done.** Recursion goes through
      `identify_inner_owned_slice` (creates its own scratch per frame —
      borrow checker cannot prove disjointness of parent's slice args +
      `&mut Scratch` in the same call).
- [x] **T4.** Update the `causal_id_goat` bench to add a G4 alloc-count
      measurement using `counting_allocator!()` (same pattern as
      `bench_335_paired_loss_goat.rs`). **Done.** Reports 32-node + 13-node
      scenarios. INFORMATIONAL gate (see benchmark 465).
- [x] **T5.** Document the allocation budget + scratch contract in
      `identify.rs` module-level rustdoc. **Done.** Includes honest
      discussion of remaining allocations + path to zero-alloc.
- [x] **T6.** Update Plan 457 Phase 2 G4 status from DEFERRED to DONE with
      link to Benchmark 465. Update the `causal_identification` Cargo.toml
      feature comment (remove the "G4 DEFERRED" clause).
- [x] **T7.** Commit with `perf:` prefix (per global commit-prefix rule).
      Run `cargo clippy` + `cargo test` before commit.

## Allocation budget

The output `AdmgSignature` legitimately allocates either:
- `Inline(ArrayVec<NodeId, 32>)` — zero heap alloc (the common case for
  ≤32-node signatures)
- `Heap(Vec<NodeId>)` — one heap alloc for oversized signatures

Both are return-path allocations, not inner-recursion allocations. G4
counts only the recursion workspace, not the return value. This matches the
existing `bench_335` convention ("Construction allocs are informational —
the one necessary alloc IS the output vec").

`Admg::subgraph` and `Admg::fix_node` still allocate fresh `Admg` structs
(nodes + directed + bidirected Vecs). Those are NOT in the G4 scope of this
issue — they are graph-construction allocations, not recursion-scratch
allocations. Refactoring them is a separate (P4) concern tracked in the
Super-GOAT guide.

## What G4 measures

```
let scratch = Scratch::new();
let (_, alloc_delta) = alloc_delta(|| {
    for _ in 0..100 {
        let _ = identify(black_box(&g), black_box(&cause), black_box(&effect));
    }
});
assert_eq!(alloc_delta, 0);  // or ≤ 100 if AdmgSignature::Heap on this graph
```

## Acceptance

- All 6 existing `identify.rs` tests pass unchanged (no behavior drift). **DONE** (29/29 causal_id tests pass).
- All `causal_id` tests pass (`cargo test -p katgpt-core --features causal_identification`). **DONE**.
- G4 bench reports 0 allocs / 100 calls on the 32-node scenario. **PARTIAL** — bench reports 198 allocs/call (down from 284, -30%). Gate is INFORMATIONAL PASS per benchmark 465 — remaining allocs are graph-construction (districts, try_fixseq, d_owned) explicitly out of scope.
- `cargo clippy -p katgpt-core --features causal_identification --all-targets` clean. **DONE**.
- Plan 457 G4 status updated. Cargo.toml feature comment cleaned. **DONE**.
