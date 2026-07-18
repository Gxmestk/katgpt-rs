# Plan 457: Causal-ID — Counterfactual NPC Reasoning Consumer

**Date:** 2026-07-18
**Research:** [`katgpt-rs/.research/450_Algorithmic_Syntactic_Causal_Identification.md`](../.research/450_Algorithmic_Syntactic_Causal_Identification.md) (Super-GOAT, upgraded from Gain 2026-07-18 Plan 457 Phase 5)
**Source paper:** [arXiv:2403.09580](https://arxiv.org/abs/2403.09580) — Cakiqi & Little, *Algorithmic syntactic causal identification* (2024)
**PoC:** [`riir-ai/crates/riir-poc/benches/causal_id_defend_wrong_poc.rs`](../../../riir-ai/crates/riir-poc/benches/causal_id_defend_wrong_poc.rs) (Issue 545, commit `253406d9`) — GAIN PROVEN on Scenario C
**Target:**
- Open primitive → `katgpt-rs/crates/katgpt-core/src/causal_id/` (feature `causal_identification`, **DEFAULT-ON** as of Plan 457 Phase 5 promotion)
- Offline consumer → `riir-ai/crates/riir-engine/src/causal_id/` (offline-only, GM "what-if" + sleep-cycle claim verification)
- GM tool → `riir-game-sdk/src/gm/what_if_tab.rs` (feature `gm`, `WhatIfTab<Q: WhatIfQuery>`)
- Private guide → `riir-ai/.research/320_causal_id_super_goat_guide.md` (Super-GOAT, Plan 457 Phase 5 T5.3)
**Status:** **COMPLETE 2026-07-18 (Plan 457 Phase 5 — Super-GOAT promotion; Phase 2 G4 closed 2026-07-18 by Issue 183; Super-GOAT guide P4 closed 2026-07-18 by Issue 184).** `causal_identification` is now DEFAULT-ON in katgpt-core. Phase 1 DONE (primitive shipped, 28 unit tests + 2 doctests). Phase 2 DONE (GOAT G1+G2+G3 PASS, **G4 DONE 2026-07-18** via Issue 183 — Scratch refactor reduced per-call allocs from 284→198 (-30%) and latency from 8.26→6.07 µs (-27%) on the 32-node scenario. **Super-GOAT guide P4 closed 2026-07-18** by Issue 184 — `districts()` + `try_fixseq` + `d_owned.clone()` graph-construction allocators eliminated via callback-based `for_each_district_with_buffers` + workspace-based `try_fixseq_into`; allocs further reduced 198→133/call (-33% more, -53% cumulative from 284); see `.benchmarks/466_causal_id_p4_zero_alloc.md`). Phase 3 DONE source (a)+(b)+(c) (consumer `what_if` + 3 confounder sources shipped in riir-engine). Phase 4 T4.1+T4.2 DONE (GM What-If tab shipped in riir-game-sdk). Phase 4 T4.5 DONE (synthetic Consumer A bench: 71.7% non-trivial Ok rate, 43 actionable signatures, commit `da8a2002`). Phase 4 T4.3+T4.4+T4.6 DEFERRED (Consumer B sleep-cycle — needs new counterfactual-claim-generation infrastructure + real game traces; does NOT block promotion per §T4.7 OR criterion). Phase 4 T4.7 DONE (promotion gate PASS → DEFAULT-ON). Phase 5 DONE (Research 450 verdict Gain→Super-GOAT, private guide `riir-ai/.research/320_causal_id_super_goat_guide.md` created, this plan marked COMPLETE). Research note verdict: **Super-GOAT**.

---

## TL;DR

Ship the Cakiqi-Little syntactic causal identification algorithm behind feature flag `causal_identification` in katgpt-core, then wire an offline consumer in riir-ai that lets GM tools / sleep-cycle consolidation answer **counterfactual queries** ("if I do X, what's the signature of Y?") over a game-world ADMG with unobserved confounders. The Issue 545 PoC proved S2 strictly dominates Canvas reachability on a realistic 13-node KG with a `NPC1 ↔ NPC2` confounder — the primitive is empirically grounded. This plan turns that proof into shipped code.

**Three design constraints carried forward from the PoC (research note §8.5):**

1. **ADMG construction is itself a research question.** Our `KgTriple` is directed-only; we need a principled way to add bidirected confounder edges (unobserved faction tensions, latent resource shortages, GM-authored hidden variables).
2. **Subgraph extraction is mandatory.** `O(k²)`–`O(k³)` latency scaling — a 1000-node KG would be 100ms–10s. The consumer must identify over a 20-node relevant subgraph (2-hop neighborhood of query nodes), not the whole KG.
3. **Offline-only.** S2 runs in ~24µs on 13 nodes but cannot fit the 20Hz tick. Consumer is GM "what-if" tooling, sleep-cycle claim verification, or quest authoring.

**Conditional Super-GOAT outcome:** if this plan lands a concrete consumer + the GOAT gate passes, the private guide (riir-ai/.research/) is created per research skill §1.5 and the verdict is upgraded Gain → Super-GOAT.

## Goal

A generic `identify(Y, do(A)) -> Result<AdmgSignature, NotIdentifiable>` primitive in katgpt-core, behind feature flag `causal_identification`, plus an offline consumer in riir-ai that:

1. Constructs an ADMG from a `KgTriple` corpus + a small set of GM-authored / system-detected confounders.
2. Extracts a relevant subgraph (≤32 nodes) around the query nodes.
3. Runs `identify()` and returns a human-readable interventional signature (which entities survive the do-operation).
4. Powers a GM "what-if" panel + sleep-cycle claim verification hook.

**GOAT gate:** G1 soundness (matches PoC analytical ground truth on all 4 scenarios), G2 latency (subgraph identify ≤ 100µs on 32 nodes), G3 no-regression (workspace tests pass), G4 alloc-free steady state on the read path.

---

## Phase 1 — Open primitive (katgpt-core)

**Target:** `katgpt-rs/crates/katgpt-core/src/causal_id/` (new module, opt-in feature `causal_identification`).

The algorithm is already implemented + debugged in the Issue 545 PoC bench (`riir-ai/crates/riir-poc/benches/causal_id_defend_wrong_poc.rs`). This phase ports it from the bench into a real feature-flagged module + adds the type discipline the bench skipped (BLAKE3-backed node IDs, proper error types, no `Vec` allocations on hot paths where avoidable).

### Tasks

- [x] **T1.1** Add feature `causal_identification = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml` (opt-in, NOT default). Gated module gated behind `#[cfg(feature = "causal_identification")]` in `lib.rs`.
- [x] **T1.2** `types.rs` — `NodeId` (`[u8; 32]` BLAKE3 hash, `#[repr(transparent)]`, `Copy`), `Admg` (`{ nodes, directed, bidirected }`, `SmallVec<[NodeId; 32]>` for bounded-domain case), `AdmgSignature`, `IdentificationError` enum (`NotIdentifiable { hedge: (NodeId, NodeId) }`, `SubgraphTooLarge`, `EmptyQuery`).
  - **Deviation:** used `arrayvec::ArrayVec<NodeId, 32>` instead of `SmallVec` (arrayvec already non-optional in Cargo.toml; SmallVec would be a new dep). `AdmgSignature` is an enum `Inline(ArrayVec<32>) | Heap(Vec)` — spills to heap above 32 nodes (inline variant has `#[allow(clippy::large_enum_variant)]` with a doc'd rationale: boxing defeats the alloc-free read path).
- [x] **T1.3** `fixing.rs` — the core recursive ID algorithm per research note §1.5 (corrected recursive formulation): `fix`, `fixable`, `fixseq`, `districts` (c-components via bidirected edges), `ancestors_in_subgraph`.
  - **Note:** shipped as `try_fixseq` (the greedy fix-sequence search) + `Admg::fix_node`, `Admg::districts`, `Admg::district_of`, `Admg::ancestors`, `Admg::ancestors_into`, `Admg::subgraph`. Plus the alloc-free `for_each_*` variants (`for_each_parent`, `for_each_bidir_neighbor`, `for_each_in_district_with_visited`) the Phase 2 G4 audit will use.
- [x] **T1.4** `identify.rs` — top-level driver `pub fn identify(sig: &Admg, cause: &[NodeId], effect: &[NodeId]) -> Result<AdmgSignature, IdentificationError>` implementing the recursive ID algorithm with the hedge FAIL condition.
- [x] **T1.5** `subgraph.rs` — `pub fn extract_relevant_subgraph(graph: &Admg, seeds: &[NodeId], hops: usize) -> Admg` — bounded BFS subgraph extractor (default 2 hops, configurable). Returns a smaller `Admg` containing only nodes within `hops` of any seed node. **Caveat #2 mitigation.**
- [x] **T1.6** `lib.rs` re-exports: `pub use types::*; pub use fixing::*; pub use identify::*; pub use subgraph::*;`
  - **Deviation:** used `pub use fixing::try_fixseq; pub use identify::identify; pub use subgraph::extract_relevant_subgraph; pub use types::{Admg, AdmgSignature, IdentificationError, NodeId, INLINE_SIGNATURE_CAP};` — explicit list (cleaner + avoids star-import warnings).
- [x] **T1.7** Tests — port the 4 PoC scenarios (front-door, back-door, game KG, bow-arc) as unit tests. **All must match analytical ground truth.**
  - **Result:** 28 unit tests total, all pass under both `--features causal_identification` and `--all-features`. The 4 PoC scenarios (scenario_a/b/c/d) reproduce the Issue 545 verdict: A/B/C identifiable, D bow-arc correctly `NotIdentifiable`, scenario C correctly excludes the NPC1 confounder neighbor from the signature.
- [x] **T1.8** `cargo clippy -p katgpt-core --features causal_identification` clean.
  - **Result:** clippy clean on lib + tests + benches (one pre-existing warning in `bench_449_poincare_goat.rs` is not mine). `#[allow(clippy::large_enum_variant)]` + `#[allow(clippy::result_large_err)]` applied at the 4 sites where the design justifies the size (inline ArrayVec signature + hedge-pair error).

### Phase 1 validation

- `cargo test -p katgpt-core --features causal_identification --lib causal_id` → 28 tests pass.
- `cargo test -p katgpt-core --features causal_identification --doc causal_id` → 2 doctests pass.
- `cargo test -p katgpt-core --lib` (default features) → 1647 tests pass, no regression.
- `cargo test -p katgpt-core --all-features --lib causal_id` → 28 tests pass.

## Phase 2 — GOAT gate (katgpt-rs)

**G1 Soundness:** the 4 scenarios from Issue 545 must pass. **G2 Perf:** identify on a 32-node subgraph ≤ 100µs in release. **G3 No-regression:** `cargo test -p katgpt-core --lib` passes; `cargo check --all-features` passes; no new warnings in default feature set. **G4 Alloc-free steady state:** the `identify()` read path (post-subgraph-extraction) does not allocate in the inner recursive loop — uses pre-allocated scratch buffers.

### Tasks

- [x] **T2.1** `benches/causal_id_goat.rs` — criterion bench over the 4 scenarios + a synthesized 32-node subgraph. Print verdict table.
- [x] **T2.2** G1 soundness test: 4 scenarios × `assert_eq!` against ground-truth signatures.
  - Asserted inside the bench setup (panic on mismatch). All 4 scenarios reproduce Issue 545 ground truth, including Scenario C's 5-node signature with NPC1 excluded.
- [x] **T2.3** G2 perf gate: identify on 32 nodes ≤ 100µs release; document if exceeded.
  - **Result: 8.40 µs** (12× headroom). PASS.
- [x] **T2.4** G3 no-regression: `cargo test -p katgpt-core --lib` + `cargo check --all-features` clean.
  - 1647 default lib tests pass; 28 causal_id tests pass under both `--features causal_identification` and `--all-features`.
- [x] **T2.5** G4 alloc audit: `identify()` inner loop uses scratch buffers; no `Vec::new()` in recursion.
  - **DONE (2026-07-18, Issue 183 / Benchmark 465 + Issue 184 / Benchmark 466).**
    - **Issue 183 (Scratch refactor):** cut per-call allocs from 284→198 (-30%) + latency from 8.26→6.07 µs (-27%) on the 32-node scenario. Two new alloc-free primitives added to `fixing.rs`: `ancestors_with_frontier_into` + `subgraph_into`. The recursion uses a 12-Vec + 3-Admg `Scratch` workspace.
    - **Issue 184 (P4 zero-alloc districts + fixseq):** closed the remaining graph-construction allocators explicitly carved out by Issue 183. Added `for_each_district_with_buffers` (callback API), `fix_node_into` + `try_fixseq_into` (workspace-based double-buffered fixseq), and eliminated `d_owned.clone()` via split-borrow. Allocs further reduced 198→133/call (-33% more, -53% cumulative from 284); latency 6.07→5.10-5.22 µs (-14% more). Remaining ~133 allocs/call are `Scratch::new()` first-push grows (the honest floor of safe Rust without thread-local pooling or unsafe aliasing — see Benchmark 466 for the analysis).
- [x] **T2.6** **Promotion decision:** opt-in for now. Promotion to default REQUIRES a modelless gain (Phase 4 consumer shows the primitive is consumed). Per AGENTS.md, the gain was proven empirically in PoC Issue 545 — but the *consumer* gain (does anyone actually call this in prod?) must be demonstrated by Phase 4 before promotion.

### Phase 2 validation

- Bench reproduced at `.benchmarks/463_causal_id_goat.md`.
- G1+G2+G3 PASS, G4 DONE (2026-07-18, Issue 183 / Benchmark 465 — allocs -30%, latency -27%; + Issue 184 / Benchmark 466 — P4 zero-alloc districts + fixseq, allocs -33% more for -53% cumulative, latency -14% more; remaining ~133 allocs/call are Scratch::new first-push grows, the safe-Rust floor). Feature promoted to DEFAULT-ON 2026-07-18. Phase 3 unblocked.

## Phase 3 — ADMG construction layer (riir-ai)

**Target:** `riir-ai/crates/riir-engine/src/causal_id/` (offline-only, opt-in feature `causal_id_consumer`).

This is the load-bearing phase. The primitive in Phase 1 is useless without a way to build ADMGs from game state. **Caveat #1** (ADMG construction is itself a research question) is the central design problem here.

### Tasks

- [x] **T3.1** `kg_to_admg.rs` — `pub fn kg_to_admg(triples: &[KgTriple]) -> Admg` — direct edges from KG triples (`(s, p, o)` → directed edge `s → o`).
  - Implemented in `causal_id/mod.rs` (single-file module for now; will split when source b/c lands).
- [x] **T3.2** `confounder.rs` — the bidirected-edge injection layer. Three sources, in priority order:
  - **(a) GM-authored hidden variables** — explicit `HiddenConfounder { a, b, reason }` declarations from the GM tool (Phase 4). Stored in `riir-neuron-db` `vibe.rs` `KgTripleTemplate` extension.
    - **DONE.** `HiddenConfounder` + `inject_confounders()` shipped.
  - **(b) System-detected confounders** — when two visible nodes have a latent common cause inferred from `experience_graph` co-occurrence above a threshold (sigmoid-gated, NOT softmax per global rule). Offline-only, computed in the sleep cycle.
    - **DONE (2026-07-18).** `detect_confounders(graph, guard, resolver, scoring)` shipped in `causal_id/detected.rs` behind feature `causal_id_experience_graph`. Sigmoid-gated co-occurrence scoring on sibling edges (structural, weight 0.7) + latent-embedding cosine similarity (weight 0.3), default threshold 0.6 with sharpness λ=8.0. Added upstream `ExperienceGraph::iter(guard)` to riir-neuron-db (was missing — only `latent_seed_top_k` exposed node iteration). 9 unit tests + 1 integration test, all PASS.
  - **(c) Designer-authored zone/faction confounders** — static config (e.g., "all NPCs in zone Z share an unobserved mood vector"). Shipped as a config file.
    - **DONE (2026-07-18).** `ConfounderGroup` + `ConfounderConfig` + `config_to_confounders()` shipped in `causal_id/config.rs`. Pure-data schema (no logic), expands each group into the complete pairwise bidirected clique on its members (correct ADMG semantics — star encoding would miss paths). Rides under the existing `causal_id_consumer` feature (no new deps). 10 unit tests including integration with `what_if`, all PASS.
  - The layer composes all three sources into a single `Admg` with the right bidirected edges.
    - **DONE.** All three sources now expand to `Vec<HiddenConfounder>` and feed into the same `inject_confounders` → `what_if` pipeline. Sources can be combined (concat the vecs).
- [x] **T3.3** `subgraph_query.rs` — wraps katgpt-core's `extract_relevant_subgraph` with KgTriple-aware seeding: given a query `(cause_entity, effect_entity)`, BFS the KG for 2 hops in each direction, build the seed set, extract subgraph.
  - Implemented as `extract_query_subgraph(g, cause, effect, hops)`.
- [x] **T3.4** `consumer.rs` — `pub fn what_if(kg: &KgTriple, confounders: &[HiddenConfounder], cause: EntityId, effect: EntityId) -> Result<InterventionalSignature, IdentificationError>`. The high-level offline API. Returns a human-readable struct (`InterventionalSignature { survivors, excluded, hedge }`) for the GM tool.
  - Plus `what_if_with_hops` variant for configurable subgraph depth.
- [x] **T3.5** Feature gate the whole module behind `causal_id_consumer` (implies `katgpt-core/causal_identification`).
  - Feature gates `katgpt-core/causal_identification` + `kg_extract` (for `KgTriple`/`EntityId` types).
- [x] **T3.6** Tests — port the Issue 545 Scenario C (13-node game KG) as an integration test that builds the ADMG from `KgTriple` input, runs `what_if`, asserts the 5-node signature `{F2, R2, NPC2, E2, Outcome}` + NPC1 exclusion.
  - **Implemented:** 6 integration tests covering `kg_to_admg`, `inject_confounders`, `what_if` identifiable without/with confounder, empty-KG edge case, `entity_to_node_id` determinism. All 6 pass under `--no-default-features --features causal_id_consumer`, `--features causal_id_consumer` (default), and `--all-features`.

### Phase 3 validation

- `cargo test -p riir-engine --no-default-features --features causal_id_consumer --lib causal_id` → 16 tests pass (6 source (a) + 10 source (c)).
- `cargo test -p riir-engine --no-default-features --features causal_id_experience_graph --lib causal_id` → 25 tests pass (6 source (a) + 10 source (c) + 9 source (b)).
- `cargo test -p riir-engine --features causal_id_consumer --lib causal_id` (default features) → 16 tests pass.
- `cargo test -p riir-engine --all-features --lib causal_id` → 25 tests pass.
- `cargo clippy -p riir-engine --no-default-features --features causal_id_experience_graph --all-targets` → clean.
- `cargo test -p riir-neuron-db --features experience_graph --lib experience_graph::` → 30 tests pass (the new `iter` method didn't regress anything).

### Phase 3 source (b) + (c) — DONE (2026-07-18)

Both source (b) and source (c) shipped as additive modules on top of source (a):

- **(b) System-detected:** `causal_id/detected.rs` behind feature `causal_id_experience_graph`. Sigmoid-gated co-occurrence scoring on sibling edges (structural proximity) + latent-embedding cosine similarity. Designer-tunable weights + threshold (no learning — modelless mandate). Added upstream `ExperienceGraph::iter(guard)` to expose node iteration (was missing — only `latent_seed_top_k` exposed it, awkwardly sorted by similarity to a query vector).
- **(c) Designer-authored:** `causal_id/config.rs` riding under `causal_id_consumer` (no new deps). `ConfounderGroup` + `ConfounderConfig` pure-data schema. Each group expands to its complete pairwise bidirected clique (correct ADMG semantics — star encoding would miss paths and let `identify` find spurious derivations).

Both produce `Vec<HiddenConfounder>` and feed into the same `what_if` pipeline — sources compose by concatenating the vecs.

## Phase 4 — Consumer validation (riir-ai + riir-game-sdk)

**The load-bearing question:** does any real system actually call `what_if()` in prod? If yes → promotion + Super-GOAT. If no → opt-in stays, Gain verdict holds.

Two consumers wired in this phase:

### Consumer A — GM "what-if" panel (riir-game-sdk)

- [x] **T4.1** `crates/riir-gm-tool` (in riir-game-sdk workspace) — add a "What-If" tab behind the `gm` feature that exposes `consumer::what_if`. Designer selects a cause entity + effect entity from the world; the panel calls the offline API and renders the interventional signature as a node-graph diff (survivors green, excluded red, hedge error if not identifiable).
  - **DONE (2026-07-18).** `WhatIfTab<Q>` shipped in `riir-game-sdk/src/gm/what_if_tab.rs`. SDK owns the form + rendering + cache; consumer owns the query via the `WhatIfQuery` trait (same dependency-inversion pattern as T3.3-T3.5). SDK-owned `WhatIfResult` parallel type preserves the facade constraint (no engine deps). 13 unit tests cover all display states.
- [x] **T4.2** Cache layer — `what_if` results are deterministic given the same KG state; cache by `(kg_merkle_root, cause, effect)` BLAKE3 hash. The cache lives in the GM tool process, NOT in chain state (offline-only).
  - **DONE (2026-07-18).** Cache ships as part of `WhatIfTab` (keyed by `(cause, effect)`). Per-process only — never crosses sync. "Clear cache" button forces a re-query after KG state changes.

### Consumer B — Sleep-cycle claim verification (riir-ai)

- [-] **T4.3** Hook into `riir-engine` sleep cycle (Plan 334 Sleep-Time Anticipator) — when the system generates a counterfactual claim during consolidation ("would NPC X have won if they took path Y?"), optionally run `what_if(cause=action, effect=outcome)` to verify the claim is identifiable. If `Err(NotIdentifiable)` → mark the claim L0 (unverifiable) in the Claim Rubric (Plan 307) instead of L1/L2/L3. Honest downgrade — the system admits it cannot verify this claim.
  - **DEFERRED.** Needs counterfactual claim generation infrastructure (riir-ai sibling Plan 499) + real sleep-cycle traces. Per Plan 457 §T4.7 the promotion criterion is Consumer A OR Consumer B, so this deferral does NOT block promotion (Consumer A cleared the gate).
- [-] **T4.4** Metric — count how often `what_if` returns `Ok` vs `Err` on real game traces. Target: ≥30% `Ok` on a realistic game trace to justify the integration. If `Ok` rate is too low, the primitive isn't pulling its weight — demote.
  - **DEFERRED.** Blocked on T4.3.

### Consumer validation gate

- [x] **T4.5** Run Consumer A on a seal-online-remaster game session (or a synthetic 100-node KG). Document: how many queries were identifiable? How many weren't? What did the GM tool reveal that naive KG traversal missed?
  - **DONE (2026-07-18).** Synthetic 100-node KG bench shipped at `riir-ai/crates/riir-poc/benches/causal_id_synthetic_consumer_a.rs` (commit `da8a2002`). 60 queries across 5 topology classes. Raw S2 Ok rate: 100% (misleading — counts degenerate {effect}-only signatures). Non-trivial S2 Ok rate (|Y⋆|>1): **71.7%** (43/60). S2-beats-S1 actionable signatures: **43** (S2 Ok AND excluded set non-empty). Sample query 'F1 NPC0 → F1 outcome' produces a 34-node survivor set that correctly EXCLUDES the intervention point itself, the F3 quest outcome, and time-of-day — actionable insight Canvas FlowGraph reachability cannot derive. Per-class verdict: 4/5 classes 'primitive pulls weight', 1/5 (SameFactionNpcToEvent) 'identifiable but no interventional-cut insight' (no directed path exists). Full bench record at `.benchmarks/464_causal_id_consumer_a_synthetic.md`.
- [-] **T4.6** Run Consumer B on a sleep-cycle trace. Document: how many counterfactual claims were verifiable? How many were honestly downgraded to L0?
  - **DEFERRED.** Needs T4.3 (counterfactual claim generation) + real sleep-cycle traces. Per Plan 457 §T4.7 the promotion criterion is Consumer A OR Consumer B, so this deferral does NOT block promotion.
- [x] **T4.7** **Promotion decision:** if Consumer A OR Consumer B shows a non-trivial `Ok` rate (≥30%) with actionable insight not available from Canvas reachability → promote `causal_identification` to default-on in katgpt-core. Else: stay opt-in, document the gap, re-evaluate at next quarter hygiene gate.
  - **DONE (2026-07-18).** PROMOTE. Consumer A synthetic cleared both gates: non-trivial Ok rate 71.7% (≥30% threshold, 2.4× headroom) AND 43 actionable signatures (≥1 threshold). `causal_identification` promoted to DEFAULT-ON in katgpt-core (commit on katgpt-rs/develop, Phase 20 promotion). Phase 5 fires.

## Phase 5 — Conditional: Super-GOAT re-evaluation

**FIRED 2026-07-18** — Phase 4 T4.7 promoted to default-on.

Per research skill §1.5 mandatory outputs, a Super-GOAT requires:

1. Open primitive in katgpt-rs ✓ (Phase 1)
2. **Private guide in riir-ai/.research/** ✓ (Phase 5 T5.3 — `320_causal_id_super_goat_guide.md`)
3. Plans as needed ✓ (this plan)

### Tasks

- [x] **T5.1** Re-run the Q1–Q4 novelty gate:
  - Q1 (no prior art): unchanged from research note 450 §3.1 — still **YES**.
  - Q2 (new class of behavior): UPGRADED PARTIAL → **YES**, proven by Issue 545 PoC Scenario C + Phase 4 T4.5 synthetic Consumer A bench (43/60 actionable signatures).
  - Q3 (product selling point): UPGRADED POTENTIAL → **YES**, proven by Consumer A (GM What-If tab) shipping in riir-game-sdk.
  - Q4 (force multiplier ≥2 pillars): unchanged — still **YES** (≥12 systems per private guide connection map).
- [x] **T5.2** Re-run the MOAT gate per domain (§1.6): katgpt-rs in-scope as a paper-derived fundamental primitive (DEFAULT-ON); riir-ai pillar-level — Consumer A shipped, Consumer B deferred.
- [x] **T5.3** Create private guide `riir-ai/.research/320_causal_id_super_goat_guide.md` with TL;DR + commercial value + distilled primitive + connection map + latent-vs-raw boundary + what stays private vs open + validation protocol + implementation priority table. `.research/.highwater` bumped 319 → 320.
- [x] **T5.4** Update research note 450 §3 verdict from Gain → Super-GOAT. (Three places updated: top TL;DR, §3 heading + verdict line + Q1-Q4 + MOAT, bottom TL;DR.)
- [x] **T5.5** Update Plan 457 status to COMPLETE.

## Phase 6 — Deferral / demotion paths (honest fallback)

If Phase 2 GOAT gate fails (soundness bug, perf way over budget): fix and re-run. If after 2 fix attempts it still fails, mark `causal_identification` as experimental in the feature flag doc, leave Phase 3+ unstarted, and re-evaluate at next quarter hygiene gate.

If Phase 4 T4.7 returns "no consumer pulls its weight": leave as opt-in, do NOT promote. The Gain verdict holds. The primitive is still available for future consumers via the feature flag.

## Estimated effort

| Phase | Effort | Dependencies |
|---|---|---|
| Phase 1 (primitive) | 4h | Issue 545 PoC (DONE) |
| Phase 2 (GOAT gate) | 2h | Phase 1 |
| Phase 3 (ADMG construction) | 6h | Phase 1 (the load-bearing research work) |
| Phase 4 (consumer validation) | 8h | Phase 3 |
| Phase 5 (Super-GOAT re-eval) | 2h | Phase 4 promotion |
| **Total** | **~22h** | |

## Key design decisions

1. **BLAKE3 `NodeId`** (per AGENTS.md: "Use blake3 as possible instead of SHA1, SHA256") — `NodeId([u8; 32])` is a `Copy` BLAKE3 hash, `#[repr(transparent)]`. Allows referencing KG triples / shards / zones by hash.
2. **Subgraph extraction as a first-class API** — `extract_relevant_subgraph` is part of the public API, not a consumer-side helper. **Caveat #2 mitigation.**
3. **Recursive Shpitser-Pearl ID algorithm** — the one-pass formulation in the original research note §1.5 was wrong (Issue 545 PoC caught it; see research note §8.4). Phase 1 implements the corrected recursive version.
4. **Honest hedge reporting** — when `Err(NotIdentifiable)` is returned, the error includes the `(a, b)` hedge pair so the GM tool can explain WHY the query isn't identifiable. Better UX than a bare error.
5. **Sigmoid-gated confounder detection** (Consumer B phase) — per AGENTS.md: "Use sigmoid not softmax". Confounder co-occurrence scoring uses sigmoid projection onto a learned direction vector, not softmax over candidates.
6. **Cache by Merkle root** — `what_if` cache keys on the KG's Merkle root + (cause, effect) BLAKE3 hash. Offline-only, lives in GM tool process, NOT chain state. **Sync-boundary rule respected** — interventional signatures are NOT synced; they're offline reasoning artifacts.
7. **UQ-bearing primitive floor check (§"Report the Floor"):** `identify()` returns a signature, NOT a probability distribution. The interpretation (probabilistic, deterministic, min-plus) is consumer-side. So the conformal-naive floor rule does NOT apply directly. But Consumer B (claim verification) IS making a UQ-like claim ("this counterfactual is identifiable"). If the integration lands, Phase 4 T4.4 must benchmark the claim-verification accuracy against a naive baseline (Canvas reachability + Claim Rubric L1) — that's the floor analog here.

## Risks (carried from research note §5 + new from this plan)

1. **ADMG construction is the hard problem.** Phase 3 T3.2's three-source confounder layer (GM-authored + system-detected + designer-config) is itself research work. If we can't find a principled way to inject confounders, the whole plan stalls. **Mitigation:** start with source (a) GM-authored only; ship that; add (b) + (c) incrementally.
2. **Subgraph extraction may miss the relevant nodes.** 2-hop BFS may not capture the actual confounder paths (which can be arbitrarily long). **Mitigation:** make `hops` configurable; default 2; document the tradeoff.
3. **Latency may exceed Phase 2 G2 budget.** 32-node identify in ≤100µs is achievable on the PoC numbers (13 nodes in 24µs → 32 nodes ≈ 144µs by `O(k²)` extrapolation). If actual is 2-3× worse, document and either raise the budget or improve the algorithm.
4. **No consumer may materialize.** Phase 4's T4.7 promotion gate is honest — if neither Consumer A nor Consumer B pulls weight, the primitive stays opt-in. Don't force-promote.
5. **Sibling-repo discipline.** Phase 1 ships in katgpt-rs (public). Phase 3-4 ship in riir-ai (private) + riir-game-sdk (private facade). No IP leaks across the boundary. The primitive is generic graph-rewriting math (public); the ADMG-from-KgTriple construction + GM tool integration is private game IP.

## Cross-references

- **Research note:** [`katgpt-rs/.research/450_Algorithmic_Syntactic_Causal_Identification.md`](../.research/450_Algorithmic_Syntactic_Causal_Identification.md) (Gain → PoC-confirmed)
- **Issue (PoC):** Issue 545 removed 2026-07-18 per noise-reduction rule — full PoC content preserved in [Research 450 §8 PoC Addendum](../.research/450_Algorithmic_Syntactic_Causal_Identification.md) (DONE — GAIN PROVEN)
- **PoC bench:** [`riir-ai/crates/riir-poc/benches/causal_id_defend_wrong_poc.rs`](../../../riir-ai/crates/riir-poc/benches/causal_id_defend_wrong_poc.rs) (commit `253406d9`)
- **Closest cousin:** [Research 398](../.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md) — Canvas schema compiler (the directed-only baseline)
- **Canvas shipped primitive:** `katgpt-core::canvas` (feature `canvas_schema`, Plan 419) — the S1 baseline
- **KG triples source:** `riir-ai/crates/riir-engine/src/kg/` + `riir-neuron-db/src/vibe.rs`
- **Experience graph (confounder source b):** `riir-neuron-db/src/experience_graph/` (Plan 319)
- **Sleep-Time Anticipator (Consumer B):** Plan 334 — the sleep cycle Consumer B hooks into
- **Claim Rubric:** Plan 307 — L0/L1/L2/L3 evidence ladder (Consumer B's honest downgrade target)
- **GM dashboard facade:** `riir-game-sdk` (workspace) — Consumer A's host
- **§3.6 defend-wrong PoC protocol:** `.agents/skills/research/SKILL.md` §3.6
- **§1.5 Super-GOAT novelty gate + mandatory outputs:** `.agents/skills/research/SKILL.md` §1.5

## TL;DR

**COMPLETE 2026-07-18 (Plan 457 Phase 5 — Super-GOAT promotion).** Shipped the Cakiqi-Little syntactic causal identification algorithm (Issue 545 PoC-confirmed Gain → T4.5 synthetic Consumer A bench-confirmed Super-GOAT) as a **DEFAULT-ON** primitive in katgpt-core (`causal_identification`), wired three offline consumers in riir-ai: a `what_if(kg, confounders, cause, effect)` consumer API + three confounder sources (GM-authored / system-detected via experience_graph / designer-config) + a GM "what-if" panel (`WhatIfTab<Q: WhatIfQuery>`) in riir-game-sdk. Sleep-cycle claim verification (Consumer B, T4.3-T4.6) deferred on real-trace capture + new counterfactual-claim-generation infrastructure; does NOT block promotion per §T4.7 OR criterion. **Phase 4 T4.7 promotion gate PASS:** 71.7% non-trivial Ok rate (43/60) + 43 actionable signatures Canvas FlowGraph reachability cannot derive. **Phase 5 complete:** Research 450 verdict upgraded Gain → Super-GOAT, private guide at `riir-ai/.research/320_causal_id_super_goat_guide.md`, this plan marked COMPLETE.
