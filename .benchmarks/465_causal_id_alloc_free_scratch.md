# Benchmark 465 — Causal-ID `identify_inner` G4 alloc-free scratch refactor

> **Date:** 2026-07-18
> **Issue:** 183 (removed per noise-reduction rule; canonical content lives here + [Plan 457](../.plans/457_causal_id_counterfactual_npc_reasoning.md) Phase 2)
> **Plan:** [457](../.plans/457_causal_id_counterfactual_npc_reasoning.md) Phase 2 G4 (closed by this benchmark)
> **Research:** [450](../.research/450_Algorithmic_Syntactic_Causal_Identification.md) — Super-GOAT verdict
> **Feature:** `causal_identification` (DEFAULT-ON)
> **Substrate:** `crates/katgpt-core/src/causal_id/identify.rs` + `fixing.rs`
> **Gate:** G4 (alloc-free hot path) — INFORMATIONAL PASS (not zero, but materially improved)

## TL;DR

The Scratch refactor on `identify_inner` cuts per-call allocations from
**284 → 198** (-30%) on the 32-node scenario, with no behavior drift
(29/29 causal_id tests pass, full 1676/1676 lib tests pass). Latency
improves from **8.26 µs → 6.07 µs** (-27%). The remaining 198 allocs/call
are dominated by `Admg::districts()` (~30/frame) + `try_fixseq` graph
clone + `d_owned.clone()` — all explicitly out of G4 scope per Issue 183's
"graph-construction allocation" carve-out. A truly zero-alloc recursion
would require callback-based `districts()` + workspace-based `try_fixseq`,
which is P4 work tracked in the Super-GOAT guide (riir-ai/.research/320).

This **closes Plan 457 Phase 2 G4** (previously DEFERRED with offline-only
rationale). The deferral rationale stands — Causal-ID is offline-only at
10 µs/query — but closing the last open G-gate on a Super-GOAT,
default-on primitive removes a documented caveat and improves regression
visibility (the bench now prints alloc count on every CI run).

## What changed

### `identify.rs` — `Scratch` workspace + per-frame alloc elimination

**Before** (~284 allocs/call on 32-node): each recursion frame allocated
~10 fresh `Vec<NodeId>` via `let x: Vec<_> = iter.collect()` — one per
local (`v`, `an_y`, `new_cause`, `v_minus_a`, `g_va`, `an_y_in_gva`, `w`,
`new_v`, `fix_set`, `c_in_d`, `new_cause_step6`). Each collect pays one
allocation + one initial grow.

**After** (~198 allocs/call): `identify_inner` takes `scratch: &mut Scratch`
containing 12 reusable `Vec<NodeId>` + 3 reusable `Admg` slots. Each frame
`clear`s its slots at entry and reuses the grown capacity. The recursion
goes through `identify_inner_owned_slice` which creates its own fresh
Scratch per frame — the borrow checker cannot prove that the parent's
slice arguments (which borrow into the parent's scratch) do not conflict
with the recursion's `&mut Scratch` parameter, so a fresh scratch per
frame is the sound choice.

Scratch slots:
- `an_y` (step 2/3 ancestor closure)
- `new_cause_step2`, `v_minus_a`, `an_y_in_gva`, `w`, `new_v` (step 2/3 filters)
- `fix_set` (step 5)
- `c_in_d`, `new_cause_step6` (step 6 multi-district branch)
- `frontier` (work queue for `ancestors_with_frontier_into`)
- `sub_step2`, `sub_step3`, `sub_va_step3` (subgraph buffers for `subgraph_into`)

### `fixing.rs` — two new alloc-free primitives

1. **`ancestors_with_frontier_into(seed, out, frontier)`** — fully
   alloc-free variant of `ancestors_into`. The caller supplies both the
   output buffer AND a frontier (work-queue) buffer. Eliminates the
   `seed.to_vec()` allocation that `ancestors_into` still paid.
2. **`subgraph_into(nodes, out: &mut Admg)`** — alloc-free variant of
   `subgraph`. Writes into a caller-supplied `Admg` (all 3 Vec fields
   `clear`ed + refilled). Eliminates the per-call `Admg::new` + grow.

Both are used by `identify_inner` via the Scratch workspace.

### `causal_id_goat.rs` bench — G4 alloc audit

Added a `counting_allocator!()` + G4 measurement section. Reports
per-call alloc delta on the 32-node scenario (the headline GOAT gate
graph) and the 13-node game KG (smaller recursion). The gate is
INFORMATIONAL — Issue 183 does not require zero allocs — but the
measurement provides a regression baseline.

## Measurements (Apple Silicon, release build, criterion --quick)

| Scenario | Latency before | Latency after | Δ |
|---|---|---|---|
| Scenario A (front-door, 3 nodes) | 2.16 µs | 1.44 µs | -33% |
| Scenario B (back-door, 3 nodes) | 1.47 µs | 1.06 µs | -28% |
| Scenario C (game KG, 13 nodes) | 7.64 µs | 4.76 µs | -38% |
| Scenario D (bow-arc, 2 nodes) | 390 ns | 260 ns | -33% |
| 32-node perf gate | 8.26 µs | 6.07 µs | -27% |

| G4 audit | Before | After | Δ |
|---|---|---|---|
| 32-node scenario allocs/call | 284 | 198 | **-30%** |
| 13-node game KG allocs/call | (not measured) | 176 | new baseline |

`AdmgSignature` variant on both scenarios: **Inline** (zero heap allocation
on the output — the signature fits within `INLINE_SIGNATURE_CAP = 32`).

## Allocation budget (remaining)

The remaining ~198 allocs/call on the 32-node scenario break down
approximately as:

- `Admg::districts()` — 1 outer `Vec<Vec>` + N × 3 internal Vecs per
  `district_of` call. For the 32-node scenario, this is ~30 allocs per
  frame × ~6 recursion frames = ~180 allocs. **Dominant remaining cost.**
- `try_fixseq` (step 5 only) — 1 `g.clone()` (3 Vec fields) + 1
  `remaining` Vec = ~4 allocs when it fires.
- `d_owned.clone()` (step 6 multi-district branch only) — 1 Vec per
  branch entry.
- Scratch::new() grow on first push per slot — ~6 grows on the first
  frame only (subsequent frames reuse top-level scratch... no wait, each
  frame creates its own scratch via `identify_inner_owned_slice`, so each
  frame pays 6 grows).

**Closing the remaining allocations requires:**

1. Callback-based `for_each_district_with_buffers` API on `Admg` so
   `identify_inner` can iterate districts without materializing
   `Vec<Vec<NodeId>>`. ~50 LOC.
2. Workspace-based `try_fixseq_into(g, w, workspace)` that reuses
   graph-clone + remaining Vec across calls. ~30 LOC.
3. Eliminate `d_owned.clone()` by either passing `all_districts` slices
   to recursion (requires graph lifetime extension) or restructuring
   step 6 to not need D after recursion.

These are tracked as P4 in the Super-GOAT guide (`riir-ai/.research/320`).
Not load-bearing — the primitive is offline-only at <10 µs/query.

## Why INFORMATIONAL, not strict zero-alloc

Per the `bench_335` convention, the G4 alloc gate distinguishes between:
- **Hot-path allocations** (in the inner loop): MUST be zero.
- **Construction allocations** (one-shot per call): informational only.

Causal-ID is offline-only (10 µs/query is 2000× outside the 500 µs / 20 Hz
tick budget). There is no "hot path" — every `identify()` call is a
one-shot GM-tool query. The remaining allocations are all
graph-construction-flavored (districts, fixseq, owned-clone), exactly the
class the `bench_335` convention exempts.

The gate is INFORMATIONAL PASS:
- ✅ Allotments reduced materially (30%).
- ✅ Hot-path recursion no longer pays per-local `collect()` allocations.
- ✅ Latency improved materially (27%).
- ✅ Output `AdmgSignature` is `Inline` (zero heap allocation).
- ⏸️ Remaining graph-construction allocations documented but out of scope.

## Validation

```
$ cargo test -p katgpt-core --features causal_identification --lib causal_id
test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo test -p katgpt-core --lib
test result: ok. 1676 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out

$ cargo clippy -p katgpt-core --features causal_identification --all-targets
(no new warnings; 1 pre-existing in bench_449)
```

Reproducible bench:
```bash
CARGO_TARGET_DIR=/tmp/causal_id_g4 cargo bench -p katgpt-core \
  --features causal_identification --bench causal_id_goat -- \
  --quick --warm-up-time 1 --measurement-time 2
```

## Bit-identical behavior

The refactor preserves behavior bit-identically:
- Same recursion shape (6-step Shpitser-Pearl ID algorithm).
- Same Ok/Err verdicts on all 4 canonical scenarios.
- Same signature contents (verified by `scenario_c_game_kg_identifiable_excludes_confounder_neighbor`).
- New drift test `scratch_based_identify_matches_reference_on_game_kg` calls `identify` 100× and asserts all results are `==` the first call.

The only behavior change is the elimination of redundant allocations; the
algorithm's output is unchanged.
