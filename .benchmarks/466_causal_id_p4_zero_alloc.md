# Benchmark 466 — Causal-ID P4 zero-alloc districts + fixseq

> **Primitive:** `causal_id::identify` (Plan 457, Super-GOAT)
> **Issue:** 184 (removed per noise-reduction rule; canonical content lives here + [Plan 457](../.plans/457_causal_id_counterfactual_npc_reasoning.md) Super-GOAT guide P4)
> **Predecessor:** [465](./465_causal_id_alloc_free_scratch.md) — Issue 183 Scratch refactor
> **Date:** 2026-07-18
> **Machine:** Apple Silicon (M-series), release build, `criterion --quick`
> **Repro:** `CARGO_TARGET_DIR=/tmp/causal_id_p4 cargo bench -p katgpt-core --features causal_identification --bench causal_id_goat -- --quick --warm-up-time 1 --measurement-time 2`

## What this measures

The G4 alloc-count audit on `identify()` for two scenarios:

- **32-node perf gate** — the headline GOAT gate (synthesized layered
  faction→resource→NPC→encounter→outcome cascade with 3 cross-layer
  bidirected confounders).
- **13-node game KG** — Scenario C from the Issue 545 PoC (faction→NPC→
  encounter→outcome with one `NPC1 ↔ NPC2` confounder).

Both runs use the `counting_allocator!()` pattern (per-call alloc counter)
over a 100-call steady-state loop.

## Three-stage allocation history

| Stage | 32-node allocs/call | 32-node latency | Δ allocs | Δ latency |
|---|---|---|---|---|
| Pre-Issue-183 baseline | 284 | 8.26 µs | — | — |
| [Bench 465] Issue 183 Scratch refactor | 198 | 6.07 µs | −30% | −27% |
| **[Bench 466] P4 zero-alloc districts + fixseq** | **133** | **5.10-5.22 µs** | **−33% more** | **−14% more** |
| **Cumulative (284 → 133)** | | | **−53%** | **−37%** |

## 13-node game KG measurements

| Stage | Allocs/call | Latency |
|---|---|---|
| Pre-Issue-183 (inferred) | ~176 | 7.64 µs |
| [Bench 465] Issue 183 Scratch refactor | 176 | 4.76 µs |
| **[Bench 466] P4** | **120** | **4.03-4.17 µs** |
| **Δ from 465** | **−32%** | **−12 to −15%** |

## All five canonical scenarios (this bench)

| Scenario | Latency (mean) |
|---|---|
| Scenario A (front-door, 3 nodes) | 1.35 µs |
| Scenario B (back-door, 3 nodes) | 0.92 µs |
| Scenario C (game KG, 13 nodes) | 4.03-4.17 µs |
| Scenario D (bow-arc, 2 nodes, NotIdentifiable) | 258 ns |
| 32-node perf gate | 5.10-5.22 µs |

`AdmgSignature` variant on both audit scenarios: **Inline** (zero heap
allocation on output — both signatures fit within the 32-node inline cap).

## What was eliminated (the P4 work)

The Issue 183 close-out doc explicitly tracked three graph-construction
allocation sources as P4 (out of G4 scope, future work):

| Source | Where | Old allocs | P4 fix |
|---|---|---|---|
| `Admg::districts()` | `identify_inner` step 4 + 6 | ~30/frame × ~6 frames ≈ 180 allocs (dominant) | `Admg::for_each_district_with_buffers` callback API — caller supplies 4 scratch buffers |
| `try_fixseq` | `identify_inner` step 5 | `g.clone()` (3 Vec fields) + `remaining: Vec::new()` ≈ 4 allocs/call | `try_fixseq_into(g, w, &mut FixSeqWorkspace)` — double-buffered Admg + workspace scratch |
| `d_owned.clone()` | `identify_inner` step 6 multi-district branch | 1 alloc/branch | Eliminated — direct borrow `d: &s.an_y_in_gva` (split-borrow proves disjointness) |

## Remaining allocations (the honest floor)

```
── G4: alloc audit (Issue 183 + P4, 100-call steady-state, 32-node scenario) ──
   total allocs / 100 calls: 13300
   per-call average:          133
   AdmgSignature variant:     Inline (zero heap)
   Gate: INFORMATIONAL — Issue 183 does not require zero allocs.
   Remaining: Scratch::new() first-push grows (~12 slots × ~6 frames)

── G4: alloc audit (13-node game KG, single-step recursion) ──
   total allocs / 100 calls: 12000
   per-call average:          120
   AdmgSignature variant:     Inline (zero heap)
```

The remaining ~133 allocs/call (32-node) and ~120 allocs/call (13-node) are
the `Scratch::new()` first-push grow cost:

- `Scratch` has ~12-15 `Vec<NodeId>` slots.
- Each recursion frame creates its own fresh Scratch (per
  `identify_inner_owned_slice`).
- Empty `Vec::new()` is zero-cost; the first `push` in each slot triggers
  a `Vec::grow` (one alloc).
- ~6 recursion frames × ~12-15 slots × 1 grow each ≈ ~70-90 grows
  theoretical; observed 133 (some slots grow twice due to capacity
  doubling on larger graphs).

This is the honest floor of the safe-Rust approach. Going further would
require:

1. **Thread-local Scratch pool** — would make the primitive context-
   sensitive (problematic through FFI, async runtimes). The primitive is
   used from `riir-engine` cognition paths; a thread-local could surprise
   consumers under tokio/async executors.
2. **`unsafe` raw-pointer aliasing** — to share `&mut Scratch` across the
   recursion despite the parent's slice borrows (which the borrow checker
   conservatively rejects). Rejected: the wins (~100 allocs/call) don't
   justify the safety hazard on an offline primitive.

Neither is planned. The honest claim is "graph-construction alloc-free
Causal-ID" — the recursion path beyond `Scratch::new` is alloc-free.

## Bit-identical behavior verification

The refactor preserves behavior bit-identically. Verified by:

- **Existing canonical scenarios** (A/B/C/D + 32-node) all pass unchanged.
  See `causal_id::identify::tests::*`.
- **Drift test** `scratch_based_identify_matches_reference_on_game_kg`
  (added by Issue 183) — 100× call equality check on the 13-node game KG.
- **New drift tests** (this issue):
  - `for_each_district_with_buffers_matches_districts` — verifies the
    callback API returns the same districts as `districts()`.
  - `fix_node_into_matches_fix_node` — verifies the alloc-free variant
    produces the same graph as `fix_node` for all 3 nodes in `build_chain()`.
  - `try_fixseq_into_matches_try_fixseq` — verifies the workspace variant
    returns the same Ok/Err verdict as `try_fixseq` on 7 fix-set subsets
    (including empty + full + 3 straddle-the-district cases).

## Gate verdict

**INFORMATIONAL PASS** — the gate was always informational per Issue 183
(offline primitive, not load-bearing). The P4 work materially reduces
allocations (−33% from 465's 198, −53% cumulative from the pre-refactor
284) and improves latency (−14% from 465, −37% cumulative).

The remaining ~133 allocs/call are the `Scratch::new()` first-push grow
cost, which is the honest floor of safe Rust without thread-local pooling
or `unsafe` aliasing. Closing them is NOT planned and NOT promised by the
Super-GOAT label.
