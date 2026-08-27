# Research 511: Memory Layers at Scale — Synthesis vs Shipped PKM Substrate

> **Source:** Berges, Oğuz, Haziza, Yih, Zettlemoyer, Ghosh (Meta FAIR), "Memory Layers at Scale", [arXiv:2412.09764](https://arxiv.org/abs/2412.09764), Dec 2024. Code: github.com/facebookresearch/memory.
> **Date:** 2026-08-27
> **Status:** DISTILLED — pending owner decision (training arm filed as riir-train Plan 358; two low-cost bench checks identified below)
> **Classification:** Public (the primitive ships publicly in katgpt-core; this note is the lineage synthesis + moat record)
> **Related Research:** 387 (FwPKM — the *descendant* paper; this paper is its ancestor), 455 (Hebbian Kernel Memory — our value-construction track), 199/481 (memory caching / TFIDF slot ranking), 453 (Variable Rank Domain Expert — PEER-adjacent), 302+ndb (HOPE shard capacity metric)
> **Companion:** riir-train `.plans/358_memory_layers_kb_adapters_training.md` (the model-based arm); riir-ai `.research/351` (KBLaM — the companion paper distilled same session)

---

## TL;DR

"Memory Layers at Scale" is the middle paper of the PKM lineage: Lample et al. 2019 (product-key memory) → **this paper** (scale to 128B memory params, shared pool, swilu, beat MoE/PEER) → FwPKM 2601.00671 (Research 387, fast-weight updates). Our `katgpt-core/src/product_key_memory/` (Plan 408, DEFAULT-ON, GOAT'd: G1 1670× latency, G2 Jaccard 1.0 vs brute force) shipped the retrieval factorization independently of reading this paper; Research 387 scoped it from the descendant.

**The load-bearing finding is a moat statement, not a new mechanism:** a prior-art sweep (2026-08-27) confirms that **no published work does product-key retrieval over a frozen external KV table at inference** and **no published work constructs memory-layer VALUES via Hebbian/local rules instead of gradient descent** — and our stack already ships exactly that combination: `FrozenProductKeyMemory` (BLAKE3-committed snapshot, atomic swap) + `PkmEpisodicStore` (δ-rule write gate = one GD step at η=1, Plan 408 Phase 5) + `hebbian_kernel_memory`/`sleep_hebbian` (deterministic fact construction, Plan 455 / ndb Plan 322). What the field trains, we construct; what the field shards across H100s, we commit with BLAKE3. The paper's own findings delimit why this is safe: gains concentrate on **factual recall** (the regime Hebbian association solves exactly on clean cues), key sophistication is second-order (their A8: random/sink keys gave minor inconsistent gains), and gains are largest early in training (sparse memory is the *fast-knowledge* channel — the degenerate case is a one-step write, which is what δ-rule does).

**Actionable deltas that do NOT ship:** (1) key-dim = value-dim/2 allocation rule (our const generics make this a one-line bench sweep); (2) the swilu output-gate *shape* (`y ⊙ silu(xW₁)W₂` — signed, input-scaled; our episodic read gate is plain σ); (3) the `reverse_indices` backward strategy as a **determinism** result (sort-by-slot batched accumulation gives bit-identical float sums across thread counts — atomics cannot; directly relevant to our replay/quorum needs); (4) the facts-vs-reasoning scaling law as an architecture decision rule. The trained arm (Memory+ from scratch / retrofit) is honestly filed to riir-train Plan 358 with the paper's actual numbers.

---

## 1. Position in the lineage — what each paper added

| Paper | Contribution | Our coverage |
|---|---|---|
| Lample et al. 2019 (arXiv:1907.05242) | PKM factorization: two √N codebooks, Cartesian top-k, O(√N) retrieval | ✅ `ProductKeyMemory` (Plan 408, default-on) |
| **This paper (2412.09764)** | Scale to 128B params; **shared pool across layers**; **swilu gating**; key-dim rule; 3-layer centered stride-8 placement; beats dense-2×/MoE/PEER; facts ≫ reasoning; custom EmbeddingBag kernels (3TB/s fwd); backward = atomics/lock/reverse_indices | Partial — see §3 |
| PEER (2407.04153) | Rank-one experts via product keys (values are matrices, not vectors) | Adjacent (453 variable-rank experts is our cousin) |
| UltraMem / UltraMemV2 (2411.12364 / 2508.18756) | The competitor line; V2 redesigned the gating this paper introduced | Not consumed (their gating redesign contradicts swilu — contested territory) |
| FwPKM (2601.00671, Research 387) | Test-time gradient updates on the table (forbidden → δ-rule analog) | ✅ `PkmEpisodicStore` (408 Phase 5) |

## 2. Coverage table (paper mechanism → shipped status)

| # | Mechanism | Status in our stack | Evidence |
|---|---|---|---|
| A1 | Product-key top-k lookup | ✅ SHIPPED, default-on, GOAT'd | `katgpt-core/src/product_key_memory/`, `.benchmarks/408_pkm_goat.md` |
| A2 | Shared pool across query sites | ✅ free by construction | `Arc<FrozenProductKeyMemory>` — one committed pool, N readers (the paper's "shared across 3 layers" is the same organizational fact at model-layer grain; ours is at NPC/system grain) |
| A3 | swilu output gating | 🟡 gate concept ships; the *signed input-scaled* shape does not | `PkmEpisodicStore` read gate is σ-family; swilu = `x·σ(x)` variant — candidate extension, one bench |
| A4 | key-dim = value-dim/2 | 🟡 untested rule | `ProductKeyMemory<SQRT_N, D_K, D_V>` makes it a const-generic sweep; paper found a flat optimum at ½ (robust, not sharp) |
| A5 | Scaling law: facts ↑ monotone to 64M keys; facts ≫ reasoning | 📋 evidence → decision rule | Use: product-key memory for fact-shaped load (lore, tables, "who said what"); planner/decision substrate for reasoning-shaped load. Cf. ndb HOPE capacity metric (Research 302) — the provisioning law `N ≈ ρ·F` |
| A6 | Gains largest early (200B tok) | ✅ limit case shipped | Sparse memory is the fast-knowledge channel; δ-rule one-step writes are the degenerate "instantly learned fact" — no warmup by construction |
| A7 | Backward strategies (atomics/lock/reverse_indices) | 🟡 determinism angle unclaimed | `reverse_indices` (sort-by-slot, batched apply) = **fixed accumulation order → bit-identical sums across thread counts**. Directly relevant to concurrent Hebbian writes into a shared table under replay/quorum. No such scheduler ships in our episodic path |
| A8 | Random negative keys + sink anchor: minor/inconsistent | ✅ simplification license | Their negative result licenses BLAKE3-derived keys (construction quality second-order). The **sink anchor → closed-form abstention** (`abstain iff max score ≤ anchor + β`) is a free OOD detector — see riir-ai Research 351 §refusal, where it composes with KBLaM's threshold refusal |
| A9 | Trainable values (the memory content) | ✅ modelless analog shipped | `PkmEpisodicStore` δ-rule + `hebbian_kernel_memory` + ndb `sleep_hebbian` (deterministic, margin-verified). Trained arm → riir-train Plan 358 Phase A |
| A10 | Parallel sharded EmbeddingBag (multi-GPU) | — out of scope | Engineering for H100 clusters; our tables are 32–128MB scale, single-node |

## 3. The moat record (why this note exists)

The 2026-08-27 prior-art sweep (6 searches; Lample/PEER/UltraMem/Kim&Jung-2010/frozen-base-memory/Hebbian lines) found:

1. **Frozen-base, inference-time product-key retrieval over an external KV table: unclaimed.** Closest published: frozen-base + *trained MLP memory* (OpenReview 1SMdxRtLBp), gradient-free *accumulating* persistent memory (ResearchGate 403119823), MeMo (2605.15156, trained side-network). None is PKM top-k over an externally supplied frozen table.
2. **Hebbian/local-rule construction of memory-layer values at LLM scale: unclaimed.** Closest: HeLa-Mem (2604.16839, agent KGs), H-Mem (2507.21474, recurrent nets). Neither builds KV-table values inside a memory layer.
3. **Shared cross-layer pool: open** (PEER/UltraMem are per-layer).
4. swilu gating: **contested** (UltraMemV2 redesigned it — active design territory, not open ground).

Our shipped default-on stack is exactly (1)+(2): frozen BLAKE3 keys + δ-rule/Hebbian-constructed values + committed freeze/thaw + O(√N) retrieval. **Selling-point sentence: "Our NPCs' memory tables are constructed by local association rules and committed like blocks — no training run, no cluster, provably bit-identical across nodes — while the field trains theirs on H100s."** This is the paper-validated regime (facts, early-gains, keys-second-order) so the concession ledger stays honest: trained values buy error-corrected completions under the model's own query distribution and task-shaped denoising; our crosstalk is bounded by the closed-form load/√d predictor and gated by load factor, not eliminated.

## 4. Honest deltas (the "what remains" ledger)

- **swilu read gate**: candidate A/B on `PkmEpisodicStore` — measure retrieved-row crosstalk SNR with/without the signed gate on a loaded pool. Cheap (one bench); not scheduled unless the episodic path shows leakage.
- **key-dim sweep**: one-line const-generic bench reproducing the paper's flat-½ optimum. Candidate rider on any future 408-family bench.
- **reverse_indices write scheduler**: the determinism (not perf) argument — bit-identical concurrent writes. Relevant the day concurrent Hebbian writes into one shared table land (currently writes are single-writer per store). Tracked here, not filed.
- **Trained arm**: riir-train Plan 358 Phase A (drafter from-scratch, 3–6 GPU-h; Gemma-2 2B QLoRA retrofit, 15–25 GPU-h; artifact is NeuronShard-shaped and hot-swaps through the existing freeze machinery).

## 5. Verdict

**Gain (GOAT-tier synthesis; not Super-GOAT).** Q1 fails — PKM is Lample 2019 prior art and our primitive already ships (387 scoped it from the descendant paper). The value here is (a) the moat record (§3), (b) the honest delta ledger (§4), (c) the training-arm routing with real numbers (Plan 358). No parity claim is made against the trained Memory+ — the modelless/constructed path is a different trade (determinism + zero training vs error-corrected denoising), delimited by the paper's own facts-recall regime.

## 6. Cross-references

- `.plans/408_Product_Key_Memory_Primitive.md` (landed; phases 1–5) + `.benchmarks/408_pkm_goat.md`
- Research 387 (FwPKM) — descendant paper; its fusion table F1–F6 landed as 408 Phases 4–5 + this note's Hebbian row
- Research 455 (Hebbian Kernel Memory) + ndb `.research/303_Hebbian_Fact_Storing_Shard_SuperGOAT_Guide.md` — the value-construction track
- riir-ai `.research/351_kblam_kb_attention_dilution_guide.md` — companion paper (same session); the abstention composite (A8 anchor × KBLaM threshold)
- riir-train `.plans/358_memory_layers_kb_adapters_training.md` — Phase A
- Prior art: Lample 2019 (1907.05242), PEER (2407.04153), UltraMem (2411.12364), UltraMemV2 (2508.18756), Kim & Jung (2010.03881)

---

## TL;DR

Middle paper of the PKM lineage (Lample 2019 → this → FwPKM/387). Our Plan 408 already shipped the retrieval factorization default-on; this paper's *unclaimed-in-literature* territory — frozen-table PKM retrieval + Hebbian-constructed values — is precisely our shipped combination (FrozenProductKeyMemory + PkmEpisodicStore + hebbian_kernel_memory/sleep_hebbian), now recorded as a moat line with the paper's own regime findings (facts ≫ reasoning, early gains, keys second-order) as the safety delimitation. Deltas: swilu gate shape, key-dim=½ sweep, reverse_indices-as-determinism, facts-scaling decision rule; trained arm → riir-train Plan 358. Verdict: Gain, not Super-GOAT.
