# Issue 184 — Causal-ID P4 zero-alloc districts + fixseq refactor

> **Type:** optimization / refactor
> **Priority:** P4 (non-load-bearing; offline primitive at ~5 µs/query)
> **Filed:** 2026-07-18
> **Status:** CLOSED (133 allocs/call, −33% from Issue 183's 198; INFORMATIONAL PASS per benchmark 466)
> **Repo:** katgpt-rs (`crates/katgpt-core/src/causal_id/{fixing,identify}.rs`)
> **Plan:** [457](../.plans/457_causal_id_counterfactual_npc_reasoning.md) — Super-GOAT guide P4 (closed by this issue)
> **Predecessor:** [183](./183_causal_id_alloc_free_scratch.md) — Scratch refactor (closed the recursion-scratch G4 gate)
> **Benchmark:** [466](../.benchmarks/466_causal_id_p4_zero_alloc.md)

## Context

Issue 183 closed Plan 457 Phase 2 G4 by refactoring `identify_inner`'s
recursion-scratch allocations (~10 `Vec::collect()` per frame → reusable
`Scratch` struct). That dropped allocations from 284 → 198 per call (−30%).
Issue 183 explicitly carved out the remaining allocations as
"graph-construction allocations, not recursion-scratch" and tracked them
as **P4** in the Super-GOAT guide (`riir-ai/.research/320`):

- `Admg::districts()` — ~30 allocs/frame via `district_of`'s 3 internal
  Vecs per district × ~6 frames ≈ 180 allocs (the dominant remaining cost).
- `try_fixseq` — `g.clone()` (3 Vec fields) + `remaining: Vec::new()` ≈
  4 allocs/call when step 5 fires.
- `d_owned.clone()` — 1 alloc/call in the step-6 multi-district branch.

This issue closes those P4 items. The rationale remains the same as
Issue 183: not load-bearing (offline primitive), but closing the
remaining graph-construction allocations on a default-on Super-GOAT
primitive keeps the label honest and provides a tighter regression
baseline.

## Tasks

- [x] **T1.** Add `Admg::for_each_district_with_buffers(visited, district,
      frontier, next, f)` — callback-based alloc-free district enumeration.
      Caller supplies 4 scratch buffers; `f: &mut FnMut(&[NodeId])` is
      invoked once per district with a slice view. **Done.** Drift test
      `for_each_district_with_buffers_matches_districts` verifies parity
      with `districts()`.
- [x] **T2.** Add `Admg::fix_node_into(v, out: &mut Admg)` — alloc-free
      variant of `fix_node`. Caller supplies the output Admg; all 3 Vec
      fields `clear`ed + refilled. **Done.** Drift test
      `fix_node_into_matches_fix_node` verifies parity.
- [x] **T3.** Add `FixSeqWorkspace` struct + `try_fixseq_into(g, w, ws)`
      — zero-alloc workspace variant of `try_fixseq`. Returns `Ok(())`
      on success (caller doesn't need the fixed graph in the step-5 use
      case, only the feasibility verdict). Uses double-buffered current/
      next Admg with `mem::swap` to avoid `g.clone()` per fix iteration.
      **Done.** Drift test `try_fixseq_into_matches_try_fixseq` verifies
      verdict parity on 7 fix-set subsets including empty + full.
- [x] **T4.** Refactor `identify_inner` step 4+5+6 to use the new
      primitives. Step 4 records only `intersecting_count` +
      `first_intersecting` (only materialized if step-5 could fire).
      Step 6 re-iterates districts and recurses inline via the callback.
      **Done.** Bit-identical behavior verified by existing canonical
      scenario tests (A/B/C/D + 32-node) + drift test
      `scratch_based_identify_matches_reference_on_game_kg`.
- [x] **T5.** Eliminate `d_owned.clone()` in step 6. The clone was
      conservative — child frames use their own fresh Scratch (via
      `identify_inner_owned_slice`), never touching the parent's
      `an_y_in_gva`. Replace with `d: &s.an_y_in_gva` direct borrow
      (split-borrow `s` so the borrow checker sees disjoint fields).
      **Done.**
- [x] **T6.** Update G4 alloc audit in `causal_id_goat` bench with the
      new 3-stage history (284 → 198 → 133 allocs/call). Update
      remaining-alloc explanation. **Done.**
- [x] **T7.** Update module-level rustdoc in `identify.rs` with the P4
      section + allocation measurements table. **Done.**
- [x] **T8.** Commit with `perf:` prefix. Run `cargo clippy` +
      `cargo test` + `cargo bench` before commit.

## Allocation budget

The output `AdmgSignature` legitimately allocates either:
- `Inline(ArrayVec<NodeId, 32>)` — zero heap alloc (the common case for
  ≤32-node signatures)
- `Heap(Vec<NodeId>)` — one heap alloc for oversized signatures

Both are return-path allocations, not inner-recursion allocations. G4
counts only the recursion workspace, not the return value. This matches the
existing `bench_335` convention ("Construction allocs are informational —
the one necessary alloc IS the output vec").

After T1-T5, the remaining allocations per `identify` call are:
- `Scratch::new()` first-push grows: ~12 Vec slots × ~6 recursion frames
  ≈ 70-130 allocs/call (depends on graph topology — not every slot is
  used in every frame).

This is the honest floor of the safe-Rust approach. Going further would
require either:
1. **Thread-local Scratch pool** — would make the primitive context-
   sensitive (problematic through FFI, async runtimes).
2. **`unsafe` raw-pointer aliasing** — to share `&mut Scratch` across the
   recursion despite the parent's slice borrows. Rejected: the wins (~100
   allocs/call) don't justify the safety hazard on an offline primitive.

## Acceptance

- All 29 pre-existing `causal_id` tests pass unchanged. **DONE** (36/36
  causal_id tests pass — 29 original + 7 new drift tests).
- Full `katgpt-core` lib tests pass. **DONE** (1683/1683 pass).
- G4 bench reports fewer allocs/call than Issue 183's 198 baseline.
  **DONE** — 133 allocs/call on the 32-node scenario (−33%), 120 on the
  13-node scenario (−32%). Gate is INFORMATIONAL PASS per benchmark 466.
- `cargo clippy -p katgpt-core --features causal_identification --all-targets`
  clean. **DONE** (only pre-existing bench_449 warning).
- `cargo check --workspace` clean. **DONE**.
- Plan 457 Super-GOAT guide P4 status updated. **DONE** — Riir-ai
  `.research/320` updated to mark P4 closed by this issue.

## Honest non-goals

The remaining ~133 allocs/call (Scratch::new first-push grows) are NOT
closed by this issue and will not be closed by future work without a
context-sensitivity trade-off. The Super-GOAT guide should NOT promise
"zero-alloc Causal-ID" — the honest claim is "graph-construction
alloc-free Causal-ID" (the recursion path beyond Scratch::new is
alloc-free).

The primitive remains offline-only at ~5 µs/query (8000× outside the
500 µs / 20 Hz tick budget). P4 is not load-bearing for any consumer;
this refactor is a code-quality + regression-visibility improvement, not
a perf-critical fix.
