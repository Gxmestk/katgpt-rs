# GOAT Gate — Plan 457 Causal-ID (Syntactic Causal Identification)

**Date:** 2026-07-18
**Plan:** [457](../.plans/457_causal_id_counterfactual_npc_reasoning.md) — Phase 2
**Research:** [450](../.research/450_Algorithmic_Syntactic_Causal_Identification.md) (Gain → PoC-confirmed)
**PoC:** [Issue 545](../../../riir-ai/.issues/545_causal_id_defend_wrong_poc.md) — DONE, GAIN PROVEN
**Source paper:** [arXiv:2403.09580](https://arxiv.org/abs/2403.09580) Cakiqi & Little 2024
**Feature gate:** `causal_identification` (opt-in)
**Bench:** `crates/katgpt-core/benches/causal_id_goat.rs`

## TL;DR

| Gate | Verdict | Evidence |
|---|---|---|
| **G1** soundness (4 PoC scenarios) | ✅ PASS | A/B/C identifiable, D NotIdentifiable, C signature size = 5 with NPC1 correctly excluded (matches Issue 545 ground truth) |
| **G2** perf (≤100µs identify on 32 nodes) | ✅ PASS | **8.40 µs** on the 32-node synthesized subgraph — **12× headroom** |
| **G3** no-regression | ✅ PASS | 1647 lib tests pass under default features; 28 causal_id tests pass under `--features causal_identification` + `--all-features` |
| **G4** alloc-free steady state | ⚠️ **DEFERRED** | Per-recursion allocations documented below. **Acceptable for an offline-only primitive** (8 µs/call budget; not on tick path). Mandatory refactor if Phase 4 puts this on a hot path. |

**Phase 2 verdict:** G1+G2+G3 PASS, G4 deferred with rationale. Feature stays **opt-in** per the Plan 457 T2.6 promotion gate — promotion to default REQUIRES Phase 4 consumer validation (≥30% Ok rate with actionable insight not available from Canvas reachability). **Phase 3 unblocked.**

## G1 — Soundness (4 PoC scenarios)

Reproduces the Issue 545 defend-wrong PoC verdict. All four scenarios are asserted inside the bench setup (panic on mismatch):

| Scenario | Topology | Query | Ground truth | Got |
|---|---|---|---|---|
| **A** front-door (3 nodes) | `A→M→Y, A↔Y` | `identify(Y, do(A))` | IDENTIFIABLE (front-door adj.) | ✅ Ok, signature size = 2 (`{M, Y}`) |
| **B** back-door (3 nodes) | `Z→A→Y, Z→Y` | `identify(Y, do(A))` | IDENTIFIABLE (back-door on Z) | ✅ Ok, signature size = 2 (`{Z, Y}`) |
| **C** game KG (13 nodes) | full graph + `NPC1 ↔ NPC2` | `identify(Outcome, do(E1))` | IDENTIFIABLE, **NPC1 excluded** | ✅ Ok, signature size = 5 (`{F2, R2, NPC2, E2, Outcome}`) — NPC1 correctly **absent** |
| **D** bow-arc (2 nodes) | `A→Y, A↔Y` | `identify(Y, do(A))` | NOT IDENTIFIABLE (canonical hedge) | ✅ Err(NotIdentifiable) |
| 32-node subgraph | layered cascade + 3 confounders | `identify(Outcome, do(Encounter0))` | IDENTIFIABLE | ✅ Ok, signature size = 6 |

**Load-bearing finding (Scenario C):** the 5-node signature `{F2, R2, NPC2, E2, Outcome}` correctly **excludes NPC1** — the bidirected-confounder neighbor. Canvas FlowGraph reachability yields only a boolean `reaches=true` and would mis-attribute NPC1 as a cause. This is the empirical Gain proof from Issue 545 reproduced bit-identically here.

## G2 — Latency (≤100µs identify on 32 nodes)

Criterion bench, release build, Apple Silicon:

| Bench | Latency (median) |
|---|---|
| `scenario_a_frontdoor_3node` | 1.90 µs |
| `scenario_b_backdoor_3node` | 1.37 µs |
| `scenario_c_game_kg_13node` | 6.95 µs |
| `scenario_d_bowarc_2node` | 354 ns |
| **`scenario_32node_perf_gate`** | **8.40 µs** |

**Verdict:** 32-node identify at **8.40 µs** vs target **≤100 µs** → **12× headroom**. PASS.

**Reproduce:**

```bash
CARGO_TARGET_DIR=/tmp/causal_id_goat cargo bench \
    -p katgpt-core --features causal_identification \
    --bench causal_id_goat -- --nocapture
```

## G3 — No-regression

| Check | Result |
|---|---|
| `cargo test -p katgpt-core --lib` (default features) | **1647 tests pass**, 5 ignored, 0 failed |
| `cargo test -p katgpt-core --features causal_identification --lib causal_id` | 28 tests pass, 0 failed |
| `cargo test -p katgpt-core --all-features --lib causal_id` | 28 tests pass, 0 failed |
| `cargo clippy -p katgpt-core --features causal_identification --all-targets` | clean (4 design-justified `#[allow]`s; 1 pre-existing warning in `bench_449_poincare_goat.rs` is unrelated) |
| `cargo test -p katgpt-core --features causal_identification --doc causal_id` | 2 doctests pass |

PASS.

## G4 — Alloc-free steady state (DEFERRED)

### Honest finding

The current `identify_inner` recursion **does allocate** per call. Allocation audit by code inspection (counted allocations per recursion level):

| Site | Allocation |
|---|---|
| `let v: Vec<NodeId> = g.nodes.clone();` | 1 Vec clone |
| `let an_y = g.ancestors(effect);` | 1 Vec |
| `let sub = g.subgraph(&an_y);` | 1 new Admg (3 Vecs internally) |
| `let new_cause: Vec<NodeId> = ...filter().collect();` | 1 Vec per recursion branch |
| `let v_minus_a: Vec<NodeId> = ...filter().collect();` | 1 Vec |
| `let g_va = g.subgraph(&v_minus_a);` | 1 new Admg |
| `let an_y_in_gva = g_va.ancestors(effect);` | 1 Vec |
| `let w: Vec<NodeId> = ...filter().collect();` | 1 Vec |
| `let all_districts = g.districts();` | Vec of Vecs |
| `let intersecting: Vec<...> = ...filter().collect();` | 1 Vec |
| `let fix_set: Vec<NodeId> = ...filter().collect();` | 1 Vec |
| Recursive calls | own stack of the above |

**Estimated ~15-20 allocations per top-level `identify()` call** on a 32-node subgraph.

### Why this is acceptable (deferred, not failed)

1. **Offline-only primitive.** Plan 457 design constraint #3: "S2 runs in ~24µs on 13 nodes but cannot fit the 20Hz tick." This is a GM tool / sleep-cycle / quest-authoring primitive — not a tick-path primitive.
2. **Allocation budget is bounded.** Recursion depth is bounded by graph depth (max 64). Total allocations per call are `O(depth × per-level-allocs)` = `O(64 × 20)` = ~1280 allocations worst-case, but in practice ~50-100 on realistic 32-node subgraphs.
3. **G2 latency has 12× headroom.** Even with allocations, identify is at 8.40 µs vs 100 µs target. Eliminating allocations would improve latency by maybe 2-3× (1-3 µs est.) but the gate already passes.
4. **The return type is already alloc-free.** `AdmgSignature` uses `ArrayVec<NodeId, 32>` inline — no heap allocation on the signature itself for ≤32-node signatures. This was the load-bearing design choice for the consumer-facing read path.
5. **Alloc-free `for_each_*` primitives already ship.** `for_each_parent`, `for_each_bidir_neighbor`, `for_each_in_district_with_visited`, `ancestors_into` — these are ready for a future refactor that passes scratch buffers through the recursion.

### When this becomes mandatory

If Plan 457 Phase 4 lands a consumer that runs `identify()` on a hot path (tick, frame, render), the alloc-free refactor becomes mandatory. The right approach:

1. Define `IdentifyScratch { visited: Vec<NodeId>, frontier: Vec<NodeId>, an_buf: Vec<NodeId>, ... }` — owned by the caller, `clear()`ed per call.
2. Pass `&mut IdentifyScratch` through `identify_inner` instead of allocating fresh Vecs.
3. Replace `g.subgraph()` (which builds a new Admg) with `SubgraphView<'_>` — a borrowed view over the parent Admg + a node-set, no cloning.
4. Cache `g.districts()` once at the top-level identify, pass `&[Vec<NodeId>]` down.

Estimated effort: ~3-4 hours. **Not done in Phase 2** because the offline-only context makes it lower priority than Phase 3 (ADMG construction layer — the load-bearing research work).

### Test that pins the G4 status

The unit test `signature_inline_below_cap_heap_above` in `types.rs` pins the ArrayVec spill behavior (≤32 inline, >32 heap). The bench `causal_id_goat.rs` measures the end-to-end latency including allocations — if allocations ever push G2 over budget, the bench will fail.

## Promotion decision (Plan 457 T2.6)

**Stay opt-in.** Per Plan 457 T2.6:

> Promotion to default REQUIRES a modelless gain (Phase 4 consumer shows the primitive is consumed). Per AGENTS.md, the gain was proven empirically in PoC Issue 545 — but the *consumer* gain (does anyone actually call this in prod?) must be demonstrated by Phase 4 before promotion.

The Phase 1+2 ship is the *capability*. Phase 3+4 ship the *consumer*. Until Phase 4 demonstrates ≥30% Ok rate with actionable insight not available from Canvas reachability, the primitive stays opt-in.

## Phase 3 unblocked

Phase 3 (ADMG construction layer in riir-ai) is the **load-bearing research work** per Plan 457. Three confounder sources to wire:

1. **GM-authored hidden variables** — `HiddenConfounder { a, b, reason }` declarations from the GM tool.
2. **System-detected confounders** — sigmoid-gated co-occurrence in `experience_graph` above a threshold.
3. **Designer-authored zone/faction confounders** — static config (e.g., "all NPCs in zone Z share an unobserved mood vector").

See Plan 457 §"Phase 3" task list.
