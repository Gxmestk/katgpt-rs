# Research 483: KEEP — KV-Cache-Centric Memory Management for Embodied Planning

> **Source:** [KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning](https://arxiv.org/abs/2602.23592) — Zebin Yang, Tong Xie, Baotong Lu, Shaoshan Liu, Bo Yu, Meng Li (Peking Univ. AI Institute / AIRS Shenzhen / Microsoft Research), DAC 2026, v2 2026-03-17. Code: [PKU-SEC-Lab/KEEP_Embodied_Memory](https://github.com/PKU-SEC-Lab/KEEP_Embodied_Memory) (vLLM-based, ~2.8K LOC Python).
> **Date:** 2026-08-15
> **Status:** Gain — actionable (Issue 655 filed for the missing primitive POC)
> **Classification:** Public (katgpt-rs KV/serving substrate angle)
> **Related Research:** 436 (FlashMemory — closest structural cousin: block-KV selection + cold/hot pools + sigmoid threshold), 213 (Still Perceiver), 109 (Shard Drop-In), 249 (DecentMem — PageRank-not-novel precedent), 060 (MeMo — HippoRAG2 as baseline), 482 (drift_segment — change-rate segmentation), 39 (SpectralQuant)
> **Cross-ref (riir-ai / riir-neuron-db):** riir-ai Research 007 (Four-Tier Memory), 147/278 (Engram conditional memory — the game-side consumer); riir-neuron-db kg_triple `k_hop_neighbors` + ExperienceGraph (the BFS traversal the POC compares against)
> **Related Issues:** 655 (POC: set-stabilizing importance propagation selector)

---

## TL;DR

KEEP stores an embodied agent's memory **as reusable KV-cache blocks instead of text**, then solves the two problems that follow: (1) memory updates invalidate KV (→ **Static-Dynamic Memory Construction**: group memory by update frequency; static groups are KV-computed as fused units preserving intra-group cross-attention, dynamic groups per-segment), and (2) independently-cached blocks lose cross-block interactions (→ **Multi-hop Memory Re-computation**: iteratively propagate importance from query to memories to related memories until the selected set stabilizes, recompute only that set). Plus (3) **Layer-balanced Memory Loading**: a skip-monotonicity invariant (not-recomputed-early ⇒ never-recomputed) licenses pre-loading the guaranteed residue during idle early-layer windows. Results on ALFRED: 2.68× vs text memory; vs CacheBlend (EuroSys'25 Best Paper): +4.13% SR, 1.90× TTFT reduction.

**Distilled for katgpt-rs (modelless):** Mechanisms 1–2 largely ship here (`PagedKVCache.fork()` COW prefix sharing, `SegmentStore` LFU segment caching, wake-sleep freeze tiering, `drift_segment` change-rate segmentation). The **genuinely missing composition is M3's selection fixpoint**: iterate a query-seeded, edge-weighted importance propagation until the *selected set* stabilizes. Every ingredient ships (power iteration, CLR reliability weighting, `recall_to_fixed_point` tolerance loop, `k_hop_neighbors` adjacency) but no shipped loop iterates a **selection membership** to a fixpoint — all shipped fixpoints stabilize state, consensus, or eigenvectors. That composition is modelless (pure linear algebra + sigmoid), falsifiable against the shipped BFS-decay traversal, and is the subject of Issue 655.

---

## 1. Paper Core Findings

### 1.1 The setting — memory-as-KV-store for long-horizon agents

Embodied planners (ALFRED, WAH-NL; LoTA-style pipeline) re-prefill a prompt of retrieved memory segments (object states, task history) every step. Generated actions are <10 tokens; the memory prompt can be tens of thousands of tokens → **prefill is >90% of planning latency**. The same segments are re-accessed across steps (the table's location matters for many actions) → cache their KV. But memory mutates: after "pick milk from table", the milk AND table segments change → prefix reuse dies at the first updated segment; fixed-size block reuse invalidates the tail of the updated block; smaller blocks preserve update-locality but sever cross-block attention (measured SR drop).

### 1.2 Mechanism 1 — Static-Dynamic Memory Construction

Cluster memory segments semantically (sentence encoder, `all-mpnet-base-v2`). A group is **static** iff no member changed in the last `t=10` steps (≈ one task): compute its KV **as one fused unit** (full intra-group cross-attention preserved, cached as a unit). A group is **dynamic** if any member updated within `t` steps: compute **per segment** (an update invalidates only its own segment). Groups transition static↔dynamic at low frequency (~once per tens of steps); transition triggers one re-layout. Retrieval granularity follows tier: static groups retrieved at group level, dynamic at segment level.

### 1.3 Mechanism 2 — Multi-hop Memory Re-computation

Independently cached blocks lose cross-block attention. Prior fixes: EPIC recomputes heuristic fixed positions (block start/end); CacheBlend selects tokens by KV discrepancy — but **static w.r.t. the query** (decided by prefix composition only). Embodied relevance is query- and context-dependent: the "table" segment is critical for "unlock the door" only via the chain *Locked Door → Key → Table*. KEEP propagates importance over the memory-attention graph:

- **Init:** importance = mean query→memory attention.
- **Propagate:** take top-`r_i` memories as the relevant set; update all importance scores by averaging cross-attention weights *from the relevant set* (important memories propagate to their own important neighbors).
- **Converge:** repeat until the relevant-set composition stabilizes.
- **Recompute:** at layer `i+1`, recompute KV only for the stabilized set. First layer recomputes all; ratio `r_i` decreases with depth (CacheBlend schedule). Robust across ratios (Fig 10).

**System advantage (the access-pattern insight):** selection/recompute at **segment granularity** (not token) ⇒ non-recomputed KV stays **contiguous** and loads as whole blocks; token-granularity methods must load everything and patch in-place. Propagation cost hides inside the same layer's MLP.

### 1.4 Mechanism 3 — Layer-balanced Memory Loading

KV lives in CPU RAM; per-layer loads to GPU overlap next-layer compute. Imbalance: early layers recompute much (little to load), late layers recompute little (much to load) ⇒ bubbles on both ends. **Invariant:** a memory not recomputed at an earlier layer can never be recomputed later (its hidden state was discarded at the skip decision) ⇒ its KV is *guaranteed* loadable for all subsequent layers. So: during idle early-layer load windows, pre-load the guaranteed-not-recomputed residue for future layers. Moves loading work from starved late layers to idle early ones.

### 1.5 Results (ALFRED / WAH-NL, Qwen-2.5-14B & 32B-INT4, 1× A6000)

| Comparison | SR / Sub-SR | TTFT |
|---|---|---|
| vs LoTA full recompute (14B / 32B) | −0.33% / −0.31% (negligible) | **1.74× / 1.91× faster** |
| vs Full Reuse (PromptCache-class) | +9.89% / +9.78% | ~2.5× faster |
| vs CacheBlend | +4.94% / +4.13% | 1.54× / 1.90× faster |
| vs text-based memory (40 segments) | — | 2.19× / 2.68× faster |

Ablations (14B): −static-dynamic = **−6.94% SR** + 1.54× TTFT (biggest pillar); −multi-hop = −2.52% SR; −layer-balanced = 1.20× TTFT.

---

## 2. Distillation — What Ships Here (audit 2026-08-15)

Full-workspace grep sweep (paper vocab + operator-name translation, `src/**/*.rs` across katgpt-rs / riir-ai / riir-neuron-db / riir-mmorpg-examples):

### 2.1 Mapping table

| KEEP concept | Shipped analog | Location | Status |
|---|---|---|---|
| KV-block reuse across queries (PromptCache/RagCache class) | `PagedKVCache::fork()` — refcounted COW prefix page sharing (Issue 053); `SegmentStore`/`SegmentCheckpoint` — KVarN-compressed segment KV + LFU + GRM sigmoid gating (Plan 223b); `KvSegmentPool` + `RollingHash` — O(n) variable-length reuse *detection*; `EmbeddingRouter` — anyrag retrieval → "KV cache priming"; `preload_kv_cache` (drafter←target) | `katgpt-transformer/src/kv_cache.rs`, `katgpt-kv/src/{segment_checkpoint,cache_prune}`, `riir-router/src/embedding.rs` | ✅ **Strong** |
| Update-frequency tiering, granularity per tier | Wake-sleep consolidation + two-sided `can_freeze` gate (dynamic per-event capture → frozen committed shard); CCE "cold tier per moderator refresh, not per tick" vs hot per-tick; ARG offline sleep-cycle; `drift_segment` (per-token drift opens states at semantic transitions — segmentation BY change rate); KARC LOD tiers; mmorpg `static_data` (boot-committed BLAKE3 ShardIndex) | `riir-neuron-db/src/consolidation/`, `riir-engine/src/{cce_runtime,arg_runtime}`, `katgpt-kv/src/drift_segment/` | ✅ **Strong at persistence level** — NOT as a KV-layout split (no static-prefix/dynamic-suffix KV layout exists; no literal `update_frequency`/`hot_cold`/`frozen_prefix` identifiers) |
| Query-seeded importance scoring for KV selection | `CsKvProbe` (Lasso over ablation masks → `KvGroupRanking`), SP-KV `utility_predictor` (MLP), HOLA `β·‖e‖` surprise, `select_highest_attn_keys`/OMP (katgpt-attn-match), causal head importance | `katgpt-kv/src/{cs_kv_probe,sp_kv}`, `katgpt-core/src/hippocampal_cache.rs` | ✅ Single-pass scorers ship |
| **Multi-hop propagation until selection stabilizes** | `power_iteration_deflate` (zone_manifold / peira / civ-emotion — eigvec fixpoints); `recall_to_fixed_point(tol, max_sweeps)` (cp_hopfield — STATE fixpoint); ADMM `k_admm` consensus (sheaf_coordination — fixed-K); CLR `clr_weighted_set_attention_into` (reliability-weighted, **one step per tick**); `k_hop_neighbors` BFS + `fuse_graph_candidates` hop-decay (riir-rag `graph_rag`) | katgpt-core, riir-engine, riir-neuron-db/riir-rag | ⚠️ **Partial — ingredients only.** Zero shipped loops iterate a *selection set* to a fixpoint (grep-verified: no `pagerank|multi_hop|iterate_until` hits workspace-wide) |
| Layer-granular KV loading + skip monotonicity | `DoubleBuffer` async Q/DQ (chunk-granular load/compute overlap), `preload_kv_cache` idle-window preload, per-layer sliding windows (`MultiLayerKVCache`), `read_batched` sync batching; monotonic-tick invariants ship only in the sync/anti-cheat layer (`stale_tick_limit`, `remote_tick > last_remote_tick`) | `katgpt-kv/src/async_qdq.rs`, `katgpt-transformer/src/kv_cache.rs`, `riir-gpu` | ⚠️ Partial — overlap ships chunk/op-granular; **no layer-indexed KV scheduler, no skip monotonicity** |

### 2.2 The missing primitive — selection-set fixpoint propagation

KEEP's M3 expressed in our operator vocabulary:

```text
scores  = query_seed(query, memories)          // mean attention — ships (attn-match family)
loop {
    selected = top_r(scores)                    // relevant set — ships (DensityBudget/TopK)
    scores'  = edge_avg(selected, cross_attn)   // propagate — ships as ONE step
              (optionally × clr_reliability)    // ships (Plan 570)
    if selected' == selected { break }          // ← MEMBERSHIP fixpoint — NOT shipped
    scores = scores'
}
recompute_or_retrieve(selected)                 // sigmoid-gated apply — ships
```

Every arrow ships as an operator; the loop with a **membership-stability stopping rule** does not. Shipped fixpoints stabilize *state* (Hopfield), *consensus* (ADMM), or *eigenvectors* (power iteration). This is personalized-PageRank-with-early-stop re-expressed in house primitives — which is exactly why it's cheap to POC and falsify (Issue 655).

### 2.3 The access-pattern lesson (segment- over token-granularity)

Coarser selection granularity beats finer when it keeps the **non-selected residue contiguous** — whole-block loads instead of load-everything-patch-in-place. This is the same lesson as the B40 host-scratch slot-order coalescing rule and the mirrored-ring contiguity rule (B34): *the unit of selection should equal the unit of transfer*. Worth remembering wherever KV/weight selection meets I/O.

### 2.4 What does NOT distill modellessly

- The vLLM 3-thread layer pipeline itself (serving infrastructure; our serving paths are router + katgpt-kv).
- Nothing here is training — the whole paper is inference-time systems work (sentence-encoder clustering is off-the-shelf). No riir-train redirect needed.

---

## 3. Fusion Ideas

### F1: riir-rag graph fusion — BFS-decay → set-stabilizing propagation

`fuse_graph_candidates` (graph_rag, Plan 526) reaches transitive callers at 2 BFS hops with inverse-sigmoid hop decay — uniform expansion, no re-weighting. Replace/augment with query-seeded propagation over the `KgTripleIndex` adjacency until membership stabilizes: HippoRAG's PPR result (and KEEP's independent validation in the KV domain) says weighted propagation recovers chains BFS-decay under-ranks at equal budget. Consumer: GraphRAG quality (G5 test exists to extend). This is Issue 655's primary head-to-head.

### F2: engram chain recall — single-hop lookup → multi-hop via KG adjacency

riir-ai's engram conditional memory (Research 147/278) retrieves by latent lookup. KEEP's exact motivating example is a game-shaped quest chain: *locked door → key → table*. Seeding propagation from the NPC's query/goal direction over the engram-KG adjacency, CLR-reliability-weighted (the NPC's own observation reliability), bounded by a tick-budget max_iters, yields "the NPC recalls the whole chain, not the fragment". Game selling-point direction — pursue only if Issue 655's POC shows a real recall gap vs single-hop.

### F3: SegmentStore static-group commit — update-frequency-tiered KV layout

Port M1 to `katgpt-kv/segment_checkpoint`: segments stable for `t` accesses fuse into a committed group (one BLAKE3-committed unit, intra-group cross-attention preserved — the freeze/thaw philosophy applied at KV-segment granularity); recently-updated segments stay individual. Consumers: `EmbeddingRouter` KV priming with dynamic context. Defer until a serving path with mutating memory prompts actually exists (honest: none today runs frequently-mutating memory prefill).

### F4: skip-monotonicity prefetch pattern

"Decision X made once (skip) is never revisited ⇒ the residue is guaranteed-loadable now" — a generalizable prefetch license. Fits `async_qdq::DoubleBuffer` (prefetch the guaranteed tail during idle) and the GPU host-orchestration corpus family (sibling-dispatches-before-readback). Recorded here; let batch mining formalize it if it recurs in our kernels.

---

## 4. Verdict: Gain

**One-line reasoning:** Mechanisms 1–2 ship strongly here under operator names (`PagedKVCache.fork`, `SegmentStore`, wake-sleep freeze gates, `drift_segment`) — KEEP validates the tier-by-update-frequency philosophy we already run at the persistence layer; the one genuinely missing composition (iterate importance propagation until the *selected set* stabilizes) is modelless, cheap to build from shipped operators, and falsifiable against the shipped BFS-decay traversal — actionable, hence Gain + Issue 655.

**Not Super-GOAT because:**
- **Q1 partial:** the propagation concept has published prior art — HippoRAG (NeurIPS 2024, PPR over memory KG; HippoRAG2 already sits as a baseline in Research 060's tables) — and notably **KEEP does not cite it**; Research 249 (DecentMem) already logged PageRank teleportation as "not novel" for this stack. The unclaimed part is only the combination (attention-weighted edges × KV-recompute target × segment granularity).
- **Q2 no:** no new capability class — transitive recall at k hops already ships (riir-rag G5); the delta is selection quality at equal budget.
- **Q3 weak:** cannot finish "our NPCs do X no competitor can" from this paper alone.

**Prior-art landscape (for future greps):** CacheBlend (EuroSys'25 Best Paper, arXiv:2405.16444 — token-granularity KV discrepancy recompute, the direct ancestor) · PromptCache (MLSys'24) · RagCache (TOCS'25) · EPIC (ICML'25, fixed-position recompute) · ReCA (ASPLOS'25, embodied agents + KV systems co-design) · KVFlow (NeurIPS'25, agent-DAG prefix reuse) · MemArt (OpenReview, concurrent — KV-as-agent-memory, no tiering/propagation) · HippoRAG/HippoRAG2 (NeurIPS'24/25 — PPR memory selection). Update-frequency-tiered KV granularity: **no published prior art found** (KEEP's strongest pillar).

**MOAT gate (§1.6):** katgpt-rs bar = fundamental/base primitive via fusion — the propagation operator is generic math (no game semantics) → note + issue correctly live here. Game-side consumers (F2) stay riir-ai if the POC pays.

---

## 5. What Stays Where (7-Repo Discipline)

| Component | Repo | Why |
|---|---|---|
| Selection-set fixpoint propagation operator (query-seeded, edge-weighted, membership-stable stop, sigmoid-gated) | katgpt-core (open) | Generic math, no game semantics |
| POC harness + head-to-head vs BFS-decay / single-hop | katgpt-rs `.issues/655` | poc/proof task |
| GraphRAG fusion upgrade (F1) | riir-ai (riir-rag) | Retrieval consumer |
| Engram chain recall (F2) | riir-ai | Game cognition consumer |
| Static-group KV commit (F3) | katgpt-kv | Serving substrate — deferred pending a mutating-memory consumer |
| Layer-loading pipeline (M3 proper) | not routed | vLLM-scale serving infra; pattern recorded (§2.3/§3-F4) only |

---

## 6. Limitations and Honest Caveats

1. **HippoRAG omission** — KEEP doesn't cite the closest concept-level prior art; treat its multi-hop novelty claims as combination-novel, not concept-novel.
2. **Concurrent MemArt** (OpenReview, unpublished) covers "KV cache AS the agent memory store" — details restricted (403); if our claims ever overlap that framing, verify manually.
3. **Our hot path is modelless** — game NPC cognition does not re-prefill LLM memory per tick, by design. KEEP's value here concentrates in the serving paths (riir-router priming, katgpt-kv segment reuse, FlashMemory-class long-context) and in the *selection* primitive (F1/F2), not in per-tick game loops.
4. **Dense-memory failure mode** (inherited from 436): tasks needing global attention collapse under any sparse selection — KEEP's own Full-Reuse rows show the floor; any propagation selector needs a dense fallback.
5. **BFS is O(degree^k)** (`k_hop_neighbors` is explicitly "not a per-tick query") — propagation with early stop may actually be *cheaper* than BFS at equal recall; that's part of what Issue 655 measures (G2).

---

## PoC Addendum (Issue 655, 2026-08-16) — CLAIM CONFIRMED

[Bench 655](../.benchmarks/655_selection_propagation_poc.md) ran the
three-way head-to-head (defend-wrong §3.6). The operator shipped as
`katgpt-core/src/selection_propagation.rs` behind the opt-in
`selection_propagation` feature.

**Verdict: PASS — the claim holds decisively.**

| arm (h≥2 means, chain/tail recall@k) | chain | tail |
|---|---|---|
| single-hop | 0.293 | 0.046 |
| BFS-decay (shipped `fuse_graph_candidates` defaults) | 0.267 | 0.058 |
| **propagation (Mass blend)** | **0.789** | **0.730** |
| propagation (Mean blend — literal KEEP edge_avg) | 0.297 | 0.051 |

36/36 cells won at h≥2, zero losses. G1/G3/G4 PASS; G2 µs-scale PASS
(73 µs @ N=1024/k=32) with the "cheaper than BFS" sub-hypothesis honestly
NEGATED in the sparse regime (BFS visited 78-816 nodes at degree ≈ 6 — the
O(degree^k) blowup needs dense graphs; at equal recall the comparison is
vacuous since no k_hop reaches 0.73 tail recall under calibrated
 distractors).

**Findings beyond the claim (all in Bench 655):**

1. **The literal KEEP `edge_avg` is degenerate** — for a node supported by
   one selected node, `w·rel/w = rel`: the edge weight cancels. Measured
   catastrophic (tail 0.051 vs Mass 0.730). The PPR-style **Mass** blend
   (`next = (1-α)·seed + α·Σ w·rel`) is the correct operatorization; the
   shipped `PropagationBlend::Mean` stays as the falsified arm + a
   bit-exact unit test pinning the cancellation.
2. **The shipped BFS-decay fusion is actively harmful under calibrated
   distractors** — worse than single-hop in the h=2/d=24 cells (0.100 vs
   0.328 chain recall). Mechanism: proximity ≠ relevance; the +0.18
   distance-1 bonus rewards distractor neighbors of the head. The shipped
   G5 use case (transitive callers, zero lexical overlap, no distractor
   pressure) is unaffected — but any consumer adding distractor-dense
   corpora should not ship the default fusion.
3. **1-hop does not tie — propagation wins there too** (distractor pressure
   exists even at h=1; at d=0 the gap narrows to 0.431 vs 0.347 but never
   inverts).
4. **The membership fixpoint rarely fires in distractor-dense worlds**
   (0/32 queries stable within 16 iters at N=1024/k=32 — boundary churn
   among near-tied distractors). `max_iters` is the operative stop; results
   stay bit-deterministic. The early-stop is valuable on clean graphs;
   a damped/hysteresis stop is optional follow-up, not needed for the
   verdict.

**F1/F2 consumers routed** (the claim passed, so the consumer issues filed):
riir-ai Issue 703 (riir-rag graph fusion upgrade) + Issue 704 (engram chain
recall). The feature stays opt-in until the first consumer lands
(`grapem_rodrigues`/`mop_path_entropy` precedent — gain proven on synthetic
POC; default-on lands WITH the consumer, not before).
