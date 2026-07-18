# Plan 457 — Causal-ID Synthetic Consumer A T4.5 Benchmark

**Plan:** [`katgpt-rs/.plans/457_causal_id_counterfactual_npc_reasoning.md`](../.plans/457_causal_id_counterfactual_npc_reasoning.md)
**Research:** [`katgpt-rs/.research/450_Algorithmic_Syntactic_Causal_Identification.md`](../.research/450_Algorithmic_Syntactic_Causal_Identification.md)
**Bench:** [`riir-ai/crates/riir-poc/benches/causal_id_synthetic_consumer_a.rs`](../../riir-ai/crates/riir-poc/benches/causal_id_synthetic_consumer_a.rs)
**Commit:** `da8a2002` on `riir-ai/develop`
**Date:** 2026-07-18
**Defend-wrong protocol:** research skill §3.6 — raw numbers recorded honestly regardless of outcome

## TL;DR

**Verdict: PROMOTE.** Synthetic Consumer A clears the Plan 457 §T4.7 promotion gate on both metrics:

| Gate | Metric | Result | Threshold | Verdict |
|---|---|---|---|---|
| 1 | S2 non-trivial Ok rate (\|Y⋆\|>1) | **71.7%** (43/60) | ≥30% | **PASS** |
| 2 | S2-beats-S1 actionable signatures | **43** (71.7%) | ≥1 | **PASS** |

The shipped `what_if` API produces interventional signatures that directed-only Canvas reachability (S1) cannot derive. Sample query (F1 NPC0 → F1 outcome) yields a 34-node survivor set that correctly EXCLUDES the intervention point itself, the F3 quest outcome, and time-of-day — exactly the kind of counterfactual reasoning Canvas's boolean reachability cannot support.

**Phase 5 (Super-GOAT re-evaluation) fires** — Consumer A synthetic alone is sufficient per the plan's "Consumer A OR Consumer B" promotion criterion. Consumer B (T4.6 sleep-cycle trace) remains BLOCKED on real-trace capture + new counterfactual-claim infrastructure (T4.3); the absence of Consumer B does NOT block promotion.

## Synthetic KG topology (100 nodes)

| Component | ID range | Count | Role |
|---|---|---|---|
| Faction 1 NPCs | 100–109 | 10 | Reasoning entities |
| Faction 2 NPCs | 200–209 | 10 | Reasoning entities |
| Faction 3 NPCs | 300–309 | 10 | Reasoning entities |
| F1 events | 130–139 | 10 | Per-NPC event emissions |
| F2 events | 230–239 | 10 | Per-NPC event emissions |
| F3 events | 330–339 | 10 | Per-NPC event emissions |
| F1/F2/F3 resources | 50–64 | 15 | Causal ancestors of NPCs |
| Quest outcomes | 1000–1002 | 3 | One per faction |
| Mood proxies | 700–719 | 20 | Faction-shared mediators |
| World-state | 800–801 | 2 | Weather + time-of-day |
| **Total** | | **100** | |

**Directed edges:** ~125 (resource→npc, npc→event, event→outcome, world→npc, rumor bridges, mood→npc)
**Bidirected confounder edges:** 136 (3 × 45-edge intra-faction cliques + 1 world-state pair)

The intra-faction confounder cliques encode the realistic game-world assumption that all NPCs in a faction share a hidden "faction mood" common cause (commander influence, supply-line state). This is the structure Canvas FlowGraph reachability cannot see — its directed-only projection collapses all 45 edges per faction into nothing.

## 60-query test set (5 classes × ~10-20 queries each)

| Class | Count | Topology tested |
|---|---|---|
| IntraFactionNpcToOutcome | 20 | NPC → own faction's quest outcome (confounder path exists) |
| CrossFactionNpcToOutcome | 10 | NPC → another faction's outcome (multi-hop via rumor bridge) |
| ResourceToOutcome | 10 | Resource → outcome (long causal chain, 3 hops) |
| SameFactionNpcToEvent | 10 | NPC → another NPC's event (confounder-mediated) |
| WorldStateToOutcome | 10 | World-state → outcome (environment-mediated, 3 hops) |

Subgraph hop counts: 2 (default) for IntraFactionNpcToOutcome + SameFactionNpcToEvent; 3 for the longer chains (ResourceToOutcome, WorldStateToOutcome, CrossFactionNpcToOutcome).

## Competitors

- **S1** — Directed-only reachability (Canvas FlowGraph equivalent). Returns boolean.
- **S2** — Full `what_if(kg, confounders, cause, effect)` with designer-authored confounders (Plan 457 Phase 3 source (c)). Returns `InterventionalSignature` (survivor set + excluded set + optional hedge).

## Per-class results

| Class | Total | S2 Ok | S2 Hedge | S2 Empty | S2>S1 (actionable) | Verdict |
|---|---|---|---|---|---|---|
| IntraFactionNpcToOutcome | 20 | 20 | 0 | 0 | **20** | primitive pulls weight |
| CrossFactionNpcToOutcome | 10 | 10 | 0 | 0 | **8** | primitive pulls weight |
| ResourceToOutcome | 10 | 10 | 0 | 0 | **9** | primitive pulls weight |
| SameFactionNpcToEvent | 10 | 10 | 0 | 0 | 0 | no interventional-cut insight (S1 already says "no path") |
| WorldStateToOutcome | 10 | 10 | 0 | 0 | **6** | primitive pulls weight |
| **Total** | **60** | **60** | **0** | **0** | **43 (71.7%)** | |

## Honest caveats

### 1. Raw Ok rate is misleadingly 100%

The raw S2 Ok rate is 100%, but 17 of the 60 queries are degenerate: when there's no causal path from cause to effect, the algorithm correctly returns `Ok(signature = {effect})` — mathematically `Σ_{Y|do(X)} = P(Y)` when X has no causal influence on Y. This is technically correct but **not actionable insight** beyond S1's "no path" verdict. The non-trivial Ok metric (|Y⋆| > 1) is the honest one: **71.7%**.

### 2. SameFactionNpcToEvent: 0 actionable

All 10 queries in this class have `S1 = —` (no directed path; the NPC has no directed edge to another NPC's event) and `S2 = Ok(|Y⋆|=1)` (trivial {effect} signature). The bidirected confounder between the two NPCs doesn't help here because the algorithm's interventional cut severs the cause's directed ancestry, and there is no directed ancestry to speak of. **The primitive correctly produces no actionable insight when there's no causal path** — this is the honest "no signal" case.

### 3. Subgraph size at the G2 perf budget edge

Signature sizes range 1–49 nodes (mean 28.6). The 49-node maximum brushes against the Plan 457 Phase 2 G2 perf budget (32-node subgraph target). At 3 hops on this topology, the subgraph extractor captures nearly the entire faction — the interventional cut is doing real work (excluded sets of 9+ nodes) but at the cost of larger subgraphs. Real game deployments should tune `hops` per query type.

### 4. Synthetic topology only

This is a synthetic 100-node KG with hand-crafted structure. It is NOT a substitute for:
- **Consumer B (T4.6)** — real sleep-cycle traces. The synthetic KG doesn't model the latent-embedding cosine-similarity scoring of Plan 457 Phase 3 source (b) (system-detected confounders); it uses source (c) (designer-authored cliques) exclusively.
- **Real game traces from seal-online-remaster** — those would test the primitive on actual KG structure, not the idealized faction-mood topology encoded here.

The synthetic topology is designed to be **favorable to the primitive**: the confounder structure is known, dense, and aligned with the query classes. Real game KGs may be sparser or noisier, which could lower the actionable rate.

## Sample signature (IntraFactionNpcToOutcome, first actionable query)

**Query:** `F1 NPC0 (entity 100) → F1 outcome (entity 1000)`

**Survivors (|Y⋆|=34):**
```
50 51 52 53 54    # F1 resources (causal ancestors of all F1 NPCs)
101 102 103 104 105 106 107 108 109    # OTHER F1 NPCs (in confounder clique with cause)
130 131 132 133 134 135 136 137 138 139    # all F1 events
238    # F2 event 8 (rumor bridge → F1 NPC5)
700 701 702 703 704 705 706    # F1 mood proxies
800    # weather (world-state, ancestor of F1 NPC0)
1000    # F1 outcome (the effect)
```

**Excluded (|exc|=9):**
```
100    # the cause itself — do(NPC0) severs it from the signature
200 300 304 305 308    # F2/F3 NPCs (no causal path through NPC0)
330    # F3 event 0 (no causal path)
801    # time-of-day (only F3 NPCs reference it)
1002    # F3 outcome (different faction's quest)
```

**S1 verdict:** `reaches = true` (boolean, no signature).
**S2 verdict:** `Σ_{1000|do(100)}` is identifiable via the survivor set above; 9 nodes excluded.

The 9 excluded nodes are the actionable insight. Canvas reachability says "yes, NPC0 reaches outcome 1000" but cannot tell the GM that:
- NPC0 itself is severed from the signature (the intervention point)
- F3's outcome is unaffected by intervening on NPC0
- Time-of-day is unaffected
- F3's NPCs 304/305/308 are unaffected

This is the kind of counterfactual reasoning a GM needs to answer "if I intervene on NPC0, what's the causal signature of outcome 1000?" — and Canvas cannot do it.

## Promotion decision (Plan 457 §T4.7)

**PROMOTE.** Both gates pass on Consumer A synthetic:
- Gate 1 (≥30% non-trivial Ok rate): 71.7% — **PASS** (2.4× headroom over threshold)
- Gate 2 (≥1 actionable insight S1 cannot derive): 43 queries — **PASS**

**Promotion action:**
1. Move `causal_identification` from opt-in to default-on in `katgpt-rs/crates/katgpt-core/Cargo.toml`.
2. Update the feature-flag doc comment in `katgpt-core/src/causal_id/mod.rs` to reflect the promotion.
3. Update `riir-engine` to forward the feature as default-on.
4. Update Research 450 §3 verdict: Gain → Super-GOAT (Phase 5 T5.4).
5. Create private guide `riir-ai/.research/NNN_Causal_Id_Super_GOAT_Guide.md` (Phase 5 T5.3).
6. Mark Plan 457 COMPLETE (Phase 5 T5.5).

**Re-evaluation triggers (per Plan 457 §"Deferral / demotion paths"):**
- If Consumer B (T4.6) later lands on real sleep-cycle traces and shows <30% non-trivial Ok rate: do NOT demote. The OR criterion was already satisfied by Consumer A. Document Consumer B's finding as a known limitation.
- If a soundness bug is found in `identify()`: revert promotion, fix, re-run Plan 457 Phase 2 GOAT gate.
- If a real game trace from seal-online-remaster shows <10% non-trivial Ok rate across 100+ queries: reopen the promotion decision at the next quarter hygiene gate.

## Methodology

### Reproducibility

```bash
CARGO_TARGET_DIR=/tmp/causal_id_consumer_a cargo bench \
  -p riir-poc --bench causal_id_synthetic_consumer_a -- --nocapture
```

The KG construction is fully deterministic (no RNG). The query set is hard-coded. Running this command produces bit-identical numbers on every invocation.

### Limitations

1. **Synthetic topology** — hand-crafted to be favorable to the primitive. Real game KGs may produce lower actionable rates.
2. **Source (c) only** — designer-authored confounder cliques. Source (b) (system-detected via `experience_graph`) is not exercised; the `causal_id_experience_graph` feature is not enabled for this bench.
3. **No latency measurement** — Plan 457 Phase 2 G2 already proved 8.40µs/32-node identify latency; this bench doesn't repeat the measurement (the focus is correctness/actionability, not perf).
4. **60 queries is small** — the per-class breakdown is statistically thin (10 queries per class). A larger sweep would tighten the per-class confidence intervals.

## References

- **Plan:** [`katgpt-rs/.plans/457_causal_id_counterfactual_npc_reasoning.md`](../.plans/457_causal_id_counterfactual_npc_reasoning.md)
- **Research note:** [`katgpt-rs/.research/450_Algorithmic_Syntactic_Causal_Identification.md`](../.research/450_Algorithmic_Syntactic_Causal_Identification.md)
- **Issue 545 PoC (Gain proven):** [`riir-ai/.issues/545_causal_id_defend_wrong_poc.md`](../../riir-ai/.issues/545_causal_id_defend_wrong_poc.md) (DONE)
- **Issue 545 PoC bench:** [`riir-ai/crates/riir-poc/benches/causal_id_defend_wrong_poc.rs`](../../riir-ai/crates/riir-poc/benches/causal_id_defend_wrong_poc.rs) (commit `253406d9`)
- **This bench:** [`riir-ai/crates/riir-poc/benches/causal_id_synthetic_consumer_a.rs`](../../riir-ai/crates/riir-poc/benches/causal_id_synthetic_consumer_a.rs) (commit `da8a2002`)
- **Paper:** [arXiv:2403.09580](https://arxiv.org/abs/2403.09580) (Cakiqi & Little 2024)
- **Closest cousin:** Research 398 — Canvas schema compiler (the directed-only S1 baseline)
