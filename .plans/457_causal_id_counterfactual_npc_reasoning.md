# Plan 457: Causal-ID — Counterfactual NPC Reasoning Consumer

**Date:** 2026-07-18
**Research:** [`katgpt-rs/.research/450_Algorithmic_Syntactic_Causal_Identification.md`](../.research/450_Algorithmic_Syntactic_Causal_Identification.md) (Gain → PoC-confirmed, §8)
**Source paper:** [arXiv:2403.09580](https://arxiv.org/abs/2403.09580) — Cakiqi & Little, *Algorithmic syntactic causal identification* (2024)
**PoC:** [`riir-ai/crates/riir-poc/benches/causal_id_defend_wrong_poc.rs`](../../../riir-ai/crates/riir-poc/benches/causal_id_defend_wrong_poc.rs) (Issue 545, commit `253406d9`) — GAIN PROVEN on Scenario C
**Target:**
- Open primitive → `katgpt-rs/crates/katgpt-core/src/causal_id/` (feature `causal_identification`, opt-in)
- Offline consumer → `riir-ai/crates/riir-engine/src/causal_id/` (offline-only, GM "what-if" + sleep-cycle claim verification)
**Status:** Active — Phase 0 (planning complete, implementation not started)

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

- [ ] **T1.1** Add feature `causal_identification = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml` (opt-in, NOT default). Gated module gated behind `#[cfg(feature = "causal_identification")]` in `lib.rs`.
- [ ] **T1.2** `types.rs` — `NodeId` (`[u8; 32]` BLAKE3 hash, `#[repr(transparent)]`, `Copy`), `Admg` (`{ nodes, directed, bidirected }`, `SmallVec<[NodeId; 32]>` for bounded-domain case), `AdmgSignature`, `IdentificationError` enum (`NotIdentifiable { hedge: (NodeId, NodeId) }`, `SubgraphTooLarge`, `EmptyQuery`).
- [ ] **T1.3** `fixing.rs` — the core recursive ID algorithm per research note §1.5 (corrected recursive formulation): `fix`, `fixable`, `fixseq`, `districts` (c-components via bidirected edges), `ancestors_in_subgraph`.
- [ ] **T1.4** `identify.rs` — top-level driver `pub fn identify(sig: &Admg, cause: &[NodeId], effect: &[NodeId]) -> Result<AdmgSignature, IdentificationError>` implementing the recursive ID algorithm with the hedge FAIL condition.
- [ ] **T1.5** `subgraph.rs` — `pub fn extract_relevant_subgraph(graph: &Admg, seeds: &[NodeId], hops: usize) -> Admg` — bounded BFS subgraph extractor (default 2 hops, configurable). Returns a smaller `Admg` containing only nodes within `hops` of any seed node. **Caveat #2 mitigation.**
- [ ] **T1.6** `lib.rs` re-exports: `pub use types::*; pub use fixing::*; pub use identify::*; pub use subgraph::*;`
- [ ] **T1.7** Tests — port the 4 PoC scenarios (front-door, back-door, game KG, bow-arc) as unit tests. **All must match analytical ground truth.**
- [ ] **T1.8** `cargo clippy -p katgpt-core --features causal_identification` clean.

## Phase 2 — GOAT gate (katgpt-rs)

**G1 Soundness:** the 4 scenarios from Issue 545 must pass. **G2 Perf:** identify on a 32-node subgraph ≤ 100µs in release. **G3 No-regression:** `cargo test -p katgpt-core --lib` passes; `cargo check --all-features` passes; no new warnings in default feature set. **G4 Alloc-free steady state:** the `identify()` read path (post-subgraph-extraction) does not allocate in the inner recursive loop — uses pre-allocated scratch buffers.

### Tasks

- [ ] **T2.1** `benches/causal_id_goat.rs` — criterion bench over the 4 scenarios + a synthesized 32-node subgraph. Print verdict table.
- [ ] **T2.2** G1 soundness test: 4 scenarios × `assert_eq!` against ground-truth signatures.
- [ ] **T2.3** G2 perf gate: identify on 32 nodes ≤ 100µs release; document if exceeded.
- [ ] **T2.4** G3 no-regression: `cargo test -p katgpt-core --lib` + `cargo check --all-features` clean.
- [ ] **T2.5** G4 alloc audit: `identify()` inner loop uses scratch buffers; no `Vec::new()` in recursion.
- [ ] **T2.6** **Promotion decision:** opt-in for now. Promotion to default REQUIRES a modelless gain (Phase 4 consumer shows the primitive is consumed). Per AGENTS.md, the gain was proven empirically in PoC Issue 545 — but the *consumer* gain (does anyone actually call this in prod?) must be demonstrated by Phase 4 before promotion.

## Phase 3 — ADMG construction layer (riir-ai)

**Target:** `riir-ai/crates/riir-engine/src/causal_id/` (offline-only, opt-in feature `causal_id_consumer`).

This is the load-bearing phase. The primitive in Phase 1 is useless without a way to build ADMGs from game state. **Caveat #1** (ADMG construction is itself a research question) is the central design problem here.

### Tasks

- [ ] **T3.1** `kg_to_admg.rs` — `pub fn kg_to_admg(triples: &[KgTriple]) -> Admg` — direct edges from KG triples (`(s, p, o)` → directed edge `s → o`).
- [ ] **T3.2** `confounder.rs` — the bidirected-edge injection layer. Three sources, in priority order:
  - **(a) GM-authored hidden variables** — explicit `HiddenConfounder { a, b, reason }` declarations from the GM tool (Phase 4). Stored in `riir-neuron-db` `vibe.rs` `KgTripleTemplate` extension.
  - **(b) System-detected confounders** — when two visible nodes have a latent common cause inferred from `experience_graph` co-occurrence above a threshold (sigmoid-gated, NOT softmax per global rule). Offline-only, computed in the sleep cycle.
  - **(c) Designer-authored zone/faction confounders** — static config (e.g., "all NPCs in zone Z share an unobserved mood vector"). Shipped as a config file.
  - The layer composes all three sources into a single `Admg` with the right bidirected edges.
- [ ] **T3.3** `subgraph_query.rs` — wraps katgpt-core's `extract_relevant_subgraph` with KgTriple-aware seeding: given a query `(cause_entity, effect_entity)`, BFS the KG for 2 hops in each direction, build the seed set, extract subgraph.
- [ ] **T3.4** `consumer.rs` — `pub fn what_if(kg: &KgTriple, confounders: &[HiddenConfounder], cause: EntityId, effect: EntityId) -> Result<InterventionalSignature, IdentificationError>`. The high-level offline API. Returns a human-readable struct (`InterventionalSignature { survivors, excluded, hedge }`) for the GM tool.
- [ ] **T3.5** Feature gate the whole module behind `causal_id_consumer` (implies `katgpt-core/causal_identification`).
- [ ] **T3.6** Tests — port the Issue 545 Scenario C (13-node game KG) as an integration test that builds the ADMG from `KgTriple` input, runs `what_if`, asserts the 5-node signature `{F2, R2, NPC2, E2, Outcome}` + NPC1 exclusion.

## Phase 4 — Consumer validation (riir-ai + riir-game-sdk)

**The load-bearing question:** does any real system actually call `what_if()` in prod? If yes → promotion + Super-GOAT. If no → opt-in stays, Gain verdict holds.

Two consumers wired in this phase:

### Consumer A — GM "what-if" panel (riir-game-sdk)

- [ ] **T4.1** `crates/riir-gm-tool` (in riir-game-sdk workspace) — add a "What-If" tab behind the `gm` feature that exposes `consumer::what_if`. Designer selects a cause entity + effect entity from the world; the panel calls the offline API and renders the interventional signature as a node-graph diff (survivors green, excluded red, hedge error if not identifiable).
- [ ] **T4.2** Cache layer — `what_if` results are deterministic given the same KG state; cache by `(kg_merkle_root, cause, effect)` BLAKE3 hash. The cache lives in the GM tool process, NOT in chain state (offline-only).

### Consumer B — Sleep-cycle claim verification (riir-ai)

- [ ] **T4.3** Hook into `riir-engine` sleep cycle (Plan 334 Sleep-Time Anticipator) — when the system generates a counterfactual claim during consolidation ("would NPC X have won if they took path Y?"), optionally run `what_if(cause=action, effect=outcome)` to verify the claim is identifiable. If `Err(NotIdentifiable)` → mark the claim L0 (unverifiable) in the Claim Rubric (Plan 307) instead of L1/L2/L3. Honest downgrade — the system admits it cannot verify this claim.
- [ ] **T4.4** Metric — count how often `what_if` returns `Ok` vs `Err` on real game traces. Target: ≥30% `Ok` on a realistic game trace to justify the integration. If `Ok` rate is too low, the primitive isn't pulling its weight — demote.

### Consumer validation gate

- [ ] **T4.5** Run Consumer A on a seal-online-remaster game session (or a synthetic 100-node KG). Document: how many queries were identifiable? How many weren't? What did the GM tool reveal that naive KG traversal missed?
- [ ] **T4.6** Run Consumer B on a sleep-cycle trace. Document: how many counterfactual claims were verifiable? How many were honestly downgraded to L0?
- [ ] **T4.7** **Promotion decision:** if Consumer A OR Consumer B shows a non-trivial `Ok` rate (≥30%) with actionable insight not available from Canvas reachability → promote `causal_identification` to default-on in katgpt-core. Else: stay opt-in, document the gap, re-evaluate at next quarter hygiene gate.

## Phase 5 — Conditional: Super-GOAT re-evaluation

**Only fires if Phase 4 T4.7 promotes to default-on.**

Per research skill §1.5 mandatory outputs, a Super-GOAT requires:

1. Open primitive in katgpt-rs ✓ (Phase 1)
2. **Private guide in riir-ai/.research/** — must be created in this phase
3. Plans as needed ✓ (this plan)

### Tasks

- [ ] **T5.1** Re-run the Q1–Q4 novelty gate:
  - Q1 (no prior art): unchanged from research note 450 §3.1 — still YES.
  - Q2 (new class of behavior): UPGRADED from PARTIAL → YES, proven by PoC Issue 545 + Phase 4 consumer validation.
  - Q3 (product selling point): UPGRADED from POTENTIAL → YES, proven by Consumer A (GM tool) shipping.
  - Q4 (force multiplier ≥2 pillars): unchanged — still YES.
- [ ] **T5.2** Re-run the MOAT gate per domain (§1.6): katgpt-rs in-scope as a paper-derived fundamental primitive; riir-ai pillar-level if Consumer B (sleep-cycle claim verification) lands.
- [ ] **T5.3** Create private guide `riir-ai/.research/NNN_Causal_Id_Super_GOAT_Guide.md` with TL;DR + commercial value + distilled primitive + connection map + latent-vs-raw boundary + what stays private vs open + validation protocol + implementation priority table. **Bump `.research/.highwater` first.**
- [ ] **T5.4** Update research note 450 §3 verdict from Gain → Super-GOAT.
- [ ] **T5.5** Update Plan 457 status to COMPLETE.

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
- **Issue (PoC):** [`riir-ai/.issues/545_causal_id_defend_wrong_poc.md`](../../../riir-ai/.issues/545_causal_id_defend_wrong_poc.md) (DONE — GAIN PROVEN)
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

Ship the Cakiqi-Little syntactic causal identification algorithm (Issue 545 PoC-confirmed Gain) as a feature-flagged primitive in katgpt-core (`causal_identification`), then wire two offline consumers in riir-ai: a GM "what-if" panel (riir-game-sdk) and a sleep-cycle claim verification hook. The hard problem is ADMG construction from our directed-only `KgTriple` (caveat #1) — Phase 3 T3.2 ships a three-source confounder injection layer (GM-authored + system-detected + designer-config). The promotion gate (Phase 4 T4.7) is honest: if neither consumer pulls weight, the primitive stays opt-in and the Gain verdict holds. If both pass → promote to default + create the private Super-GOAT guide (Phase 5) + upgrade the verdict Gain → Super-GOAT.
