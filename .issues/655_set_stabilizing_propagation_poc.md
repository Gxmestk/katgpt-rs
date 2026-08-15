# Issue 655: POC — Set-Stabilizing Importance Propagation Selector vs BFS-Decay vs Single-Hop

> **Source:** katgpt-rs [.research/483](../.research/483_KEEP_KV_Centric_Memory_Management.md) (KEEP, arXiv:2602.23592, DAC 2026)
> **Filed:** 2026-08-15
> **Type:** POC / proof task (defend-wrong §3.6 — quality claims need head-to-head, architectural reasoning insufficient)
> **Status:** Open

## Claim to test

**Falsifiable claim:** a query-seeded importance propagation iterated until the *selected set* stabilizes (KEEP's multi-hop recompute selection; HippoRAG's PPR class) **beats** the shipped BFS k-hop + inverse-sigmoid hop-decay traversal (`KgTripleIndex::k_hop_neighbors` + `riir-rag fuse_graph_candidates`) on **multi-hop chain recall at equal selection budget**, and beats single-hop lookup trivially.

Every operator ingredient ships in katgpt-core; only the membership-fixpoint loop is missing. This is the one genuinely-unshipped composition identified by the Research 483 substrate audit.

## Three competitors (defend-wrong minimum)

1. **Single-hop** (baseline): query→memory dot-product top-k. The engram latent-lookup shape.
2. **BFS k-hop + hop-decay** (shipped): uniform expansion, inverse-sigmoid decay by distance — the `fuse_graph_candidates` shape.
3. **Set-stabilizing propagation** (new): `scores = query_seed; loop { selected = top_r(scores); scores' = edge_avg(selected, weights × reliability); stop when selected stabilizes or max_iters }` — CLR-reliability-weighted, sigmoid-gated membership.

## Toy domain

Synthetic memory chains in KEEP Fig-6 shape (ground-truth action requires a 2–3-hop chain, e.g. *locked door → key → table*): N segments, planted chains at known hop distance, calibrated distractors (high query-similarity, zero chain-relevance — the case BFS-decay under-ranks). Vary: hop distance 1–4, distractor density, edge-weight noise, budget k. Optionally a second harness on the riir-rag G5 transitive-caller corpus shape (2-hop, zero lexical overlap).

## Operator shape (house rules)

- Zero-alloc: caller-owned scratch (`scores`, `next`, `selected` bitset/fixed array), `into` suffix convention.
- `max_iters` + membership-stability stop (mirror `recall_to_fixed_point(tol, max_sweeps)`).
- Sigmoid gate for membership (never softmax).
- No new deps; pure linear algebra over an adjacency/edge-weight matrix.
- Feature flag on the operator; stays opt-in until the gate passes.

## Gates

- **G1 (quality, load-bearing):** chain-completion recall@budget — propagation ≥ BFS-decay on ≥2-hop chains; report where it loses (1-hop should tie).
- **G2 (perf):** µs-scale at N ≤ 1024 memories, budget k ≤ 32; early-stop vs full BFS cost (BFS is O(degree^k) — propagation may be cheaper at equal recall).
- **G3:** no regression — operator is additive/feature-gated.
- **G4:** alloc-free steady state.
- **Failure protocol:** if propagation loses or ties everywhere, record raw numbers in Research 483 §"PoC Addendum", close as refuted-on-quality (architectural/latency axes stand), keep the POC as regression check. No silent revision.

## Tasks

- [ ] T1 — Synthetic chain-corpus generator (deterministic seed, planted chains, distractor calibration)
- [ ] T2 — `propagate_selection_to_fixpoint` operator behind feature flag (zero-alloc, membership-stability stop)
- [ ] T3 — Three-way head-to-head harness (single-hop / BFS-decay / propagation) + verdict table
- [ ] T4 — G1 sweep (hop distance × distractor density × budget) + G2 latency + G4 alloc check
- [ ] T5 — Verdict recorded in Research 483; if PASS → route consumers (riir-rag graph fusion F1, engram chain recall F2) as issues in their repos; if FAIL → record + close

## Consumers if it passes

- riir-rag `fuse_graph_candidates` (GraphRAG quality; extends the shipped G5 test)
- riir-ai engram conditional-memory chain recall (game side — only on PASS)
- katgpt-kv `cs_kv_probe` alternative scorer path (speculative)
