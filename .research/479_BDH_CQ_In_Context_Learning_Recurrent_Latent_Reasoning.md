# Research 479: BDH-CQ — In-Context Learning with Recurrent Latent Reasoning

> **Source:** [BDH-CQ: In-Context Learning with Recurrent Latent Reasoning](https://arxiv.org/abs/2608.09888) — Engdahl, Kosowski, Chorowski, Stamirowska, Uznański, et al. (Pathway / Bielik AI / NYU), 10 Aug 2026
> **Date:** 2026-08-14
> **Status:** Done
> **Related Research:** 325 (Latent Reasoning Taxonomy survey), 028 (HLA), 024 (δ-Mem), 387 (Fast-Weight PKM), 344 (Implicit Fixed-Point Halting), 273 (ELT Elastic Looped), 048/049 (HRM/TRM cousins), 453 (Variable-Rank Domain Experts), 411 (CoT vs Latent Thought formal comparison)
> **Cross-ref (riir-ai):** Research 043 (RiM Latent Workspace — SUPER GOAT, the fixed-slot cousin), 126 (CGSP curiosity), 147 (Engram Conditional Memory), 149 (Per-NPC Reasoning Depth), 161 (Cognitive Branch); `swarm/deliberation.rs` L2 layer
> **Classification:** Public

---

## TL;DR

BDH-CQ combines **in-context learning** (demonstrations update a recurrent associative memory `St = Uθ(St−1, Dt)` at inference — no weight updates, no growing KV cache) with **iterative latent reasoning** (post-ingestion workspace refinement `Hr+1 = Fθ(Hr, SK)` that reads frozen memory and decodes only the answer). A 150M config hits 29.5% pass@2 on ARC-AGI-1 at a computed **$0.0007/task — a new cost–accuracy Pareto frontier point** (57× cheaper than GPT 5.6 Luna; HRM/TRM transductive solvers cost $1.48–1.76/task). Behavioral analysis with controlled post-freeze interventions separates *extrapolation* failures (fixable by ONE demonstration at target complexity) from *execution* failures (wrong output structure) — the paper's most transferable finding.

**Distilled for katgpt-rs (modelless, inference-time):** the two-state interface (accumulating demonstration memory vs per-query reasoning workspace over frozen memory) **already ships as substrate composition** — δ-mem (`katgpt-core/src/delta_mem/`, delta-rule + surprise-gated writes) is `St`; `katgpt-micro-belief` `RecurrenceFamily` (A=accumulating belief loop, B=deliberation ticks) and `riir-games/swarm/deliberation.rs` are `Hr`; `GainCostLoopHalter` is the continuous (strictly more flexible) form of the paper's discrete LOW/MED/HIGH effort tiers; `bisimulation/operator.rs::infer_operators` is the demonstration-conditioned operator schema. **Two pieces do NOT ship** and are filed as issues: (1) the rule-application **consistency-gap metric** (strict-task vs test-pair analog — zero shipped cousin; Issue 586), and (2) **coverage-driven exemplar targeting** (the supported-context effect as a curiosity signal; riir-ai Issue 672).

---

## 1. Paper Core Findings

### 1.1 Architecture & system interface

- **BDH provenance**: post-Transformer family ("Dragon Hatchling", arXiv:2509.26507) — high-dimensional **positive sparse activations**, **low-rank communication**, **recurrent associative state**. BDH layer = ReLU-low-rank transform + linear attention in a large feature space. Working memory at inference relies on synaptic-plasticity-style (Hebbian) association.
- **Two states, different roles, same fabric**:
  - `St = Uθ(St−1, Dt)` — contextual memory, accumulates across demonstrations. Linear attention is the special case `St = St−1 + Uθ(Dt)`. No growing KV cache.
  - `H0 = Eθ(x⋆, SK)`, `Hr+1 = Fθ(Hr, SK)`, `ŷ = Gθ(HR)` — per-query reasoning workspace, **iterates R times conditioned on frozen SK**, decodes only the answer (never verbalizes intermediate states).
- **Reasoning effort tiers**: trained with mixed latent-effort levels, selectable at inference — LOW/MED/HIGH → pass@2 21%→27%→29.5%, cost −22%/−11%/0%.
- Training on ARC-style mixture (RE-ARC, ConceptARC, ARC-Heavy, ARC-GEN100K + private curation); no eval task IDs or eval demonstration pairs in training; **no parameter updates at inference**.
- **Proprietary boundary**: "Dimensions, exact update rules, and implementation details remain proprietary." Replication of the trained system is not possible from the paper.

### 1.2 Behavioral findings (controlled, post-freeze interventions)

| Finding | Numbers | Signature |
|---|---|---|
| Propagation/copying extrapolate | 48/48 across distances 2–8, copies 1–4 | no ceiling in range |
| Ordering cliff | len 6: 29/36 → len 8: 1/24; only 3/24 correct dimensions | **execution failure** (output construction breaks) |
| Nesting cliff | depth 5: 29/36; ALL correct dimensions, >99.9% cell accuracy | **extrapolation failure** (single localized relational error) |
| **Supported context** | +1 demo at target complexity: nesting 19/24→**24/24**, ordering 0/24→13/24 | coverage, not capacity, is the binding constraint |
| Dense binding | 8 simultaneous color mappings: 96/96 | strong contextual binding capacity |
| Composition asymmetry | rotation∘relocation 72/72; reflection∘relocation 47/72; swap∘relocation **0/72** | composition validity is representation-dependent |
| Conditional rule selection | 56.7% vs 100% control | cue→rule selection is a distinct weakness |
| **Unseen parameter values** | 0% — even for values *interpolating between demonstrated ones* | binding is demonstrated-value lookup, NOT interpolation |
| Consistency gap | ConceptARC pair 77.9% vs strict-task 59.4%; 52/160 tasks partial | rules often not applied consistently across inputs |
| Opaque-ID replication | no aggregate change (96 vs 95/160) | rules out request-side cue confounds |

### 1.3 Failure-signature diagnostic (methodology transfer)

Ordering failures = wrong output dimensions (whole-construction failure). Nesting failures = correct dimensions + localized error (relational slip). **Distinguishing execution vs extrapolation failure by output-structure preservation** is directly transferable to our PoC/bench tooling for NPC skill failures.

## 2. Distillation

### 2.1 Vocabulary translation (paper → shipped operator names)

| Paper term | Shipped equivalent (verified this session) |
|---|---|
| recurrent contextual memory `St` | `katgpt-core/src/delta_mem/` delta-rule state (`S'=(1−β)S−β·pred⊗k+β·v⊗k`, surprise-gated writes); `gdn2/kernel.rs` `S += k⊗delta`; `katgpt-sense` `evolve_belief` (leaky-integrator belief vs `TripleEvidence`) |
| latent workspace `Hr` iteration | `katgpt-micro-belief` `RecurrenceFamily` (A vs B modes); `riir-games/swarm/deliberation.rs` L2 (forward-sim over frozen swarm state); RiM memory blocks (riir-ai Research 043) |
| effort tiers LOW/MED/HIGH | `GainCostLoopHalter` (`gain_cost_halt.rs`) — continuous marginal-gain halting, strictly more expressive than discrete tiers |
| demonstration-conditioned operator schema | `bisimulation/operator.rs` `OperatorSchema`/`infer_operators` (infers reusable op from quotient, BLAKE3-committed); engram conditional memory (`katgpt-core/src/engram/`) |
| strict-task vs test-pair consistency gap | **NOTHING SHIPPED** → Issue 586 |
| supported-context effect | **NOTHING SHIPPED** as a curiosity/write-policy signal → riir-ai Issue 672 |

### 2.2 The two-state interface — consume, don't rebuild

The paper's system-level contribution maps onto existing substrate nearly 1:1. The one unshipped aspect is the **closed loop** (demos → associative writes → operator inference → workspace application → consistency measurement → coverage-targeted re-acquisition) as a single composed runtime. Per substrate-first: the pieces are consumed, not duplicated; what's missing is (a) the consistency measurement and (b) the coverage-driven acquisition policy — both filed as issues, not as a new parallel "BDH-CQ runtime".

### 2.3 Fusion (paper × δ-mem × OperatorSchema × CGSP)

**Fusion — "Consistency-gated demonstration teaching" (demonstration-teachable NPCs):**
demonstrations (observed (x, y) pairs of a skilled actor) → surprise-gated δ-mem writes (`St`) → `infer_operators` over accumulated associations → deliberation ticks apply the bound operator (`Hr` reads frozen `St`) → **consistency-gap measured across applications** (Issue 586) → gap high + failures clustered at complexity c → **curiosity targets ONE exemplar at c** (supported-context effect, riir-ai Issue 672) → high-salience engram write. Selling point if the PoC holds: *"NPCs learn new skills from a handful of demonstrations during gameplay — no retraining, negligible per-NPC cost — and know when they haven't learned them consistently."* Game hooks: pet taming by demonstration (extends Plan 016 rank-based taming), foreman NPC teaching harvest techniques, quest-giver teaching crafting ops.

### 2.4 Design cautions (paper data vs our assumptions)

1. **Demonstrated-value binding ≠ interpolation.** Unseen parameter values score 0% *even when in-range*; the boundary is "was it demonstrated". Our dot+sigmoid latent ops interpolate **by construction**. Caution for learned parameter-like quantities: anchor to nearest demonstrated value; sigmoid-blend only within demonstrated support (extends the bridge-function discipline, does not change it).
2. **Composition is representation-dependent.** rotation∘relocation 72/72 vs swap∘relocation 0/72 — operator semantics alone don't predict composability. For latent_functor composition chains: treat per-pair composition validity as empirical, not implied.
3. **Effort-tier training detail is not actionable for us**: continuous gain-cost halting subsumes discrete tiers at inference; the paper's joint-training trick is model-based-only.
4. **Opaque-identifier replication** (rule out request-side cues) is a cheap rigor pattern worth adopting in our bench docs when a result claims "learning".

## 3. Verdict

**Tier: Gain** — useful, actionable, not Super-GOAT.

**Novelty gate (honest scoring):**
- **Q1 no prior art? NO.** Published conjunction is novel (verified: nothing combines weight-free demo-driven associative memory + latent workspace iteration + discrete effort tiers — nearest: Titans 2501.00663, PERK, e1 2510.27042, all diverge). But against OUR stack all four mechanisms ship as substrate (δ-mem, RecurrenceFamily/deliberation, GainCostLoopHalter, OperatorSchema). The composition is unshipped; the parts are not.
- **Q2 new behavior class? CONDITIONAL.** Demonstration-teachable NPCs with measured consistency would be a new class; unproven modelless — requires the Issue 586/672 PoCs (defend-wrong §3.6: quality claims need head-to-head PoC, architectural reasoning insufficient).
- **Q3 selling point? CONDITIONAL** on the same PoCs.
- **Q4 force multiplier? YES** — connects δ-mem, micro-belief/sense, deliberation, gain-cost halting, engram, OperatorSchema, CGSP (≥5 pillars).

2 solid / 2 conditional → Gain with filed follow-ups. No "candidate" hedging: the Super-GOAT claim is declined *because substrate exists*, and the two missing pieces are issues, not deferred glory.

**MOAT gate (katgpt-rs):** in scope — latent-reasoning taxonomy line (325/344/400/411) + base evaluation primitive (consistency gap is generic, no game semantics). Game-behavior fusion routed to riir-ai (Issue 672). Nothing for riir-chain/riir-neuron-db (no commitment/shard angle — the St/Hr split is local latent state, correctly outside the sync boundary).

**riir-train redirect (§3.5 Path 0.5), justified out-of-scope:** the model-based track cannot replicate this paper — architecture dims, exact update rules, and training recipe are proprietary; no ARC harness or ARC product surface exists in the stack (verified: zero `ArcTask` code); nearest live training pipelines (riir-gpu SFT+GRPO, riir-train LoRA-Muon/EGA) consume game/text domains, not ARC-style visual grids. Path 0 decomposition instead yielded the modelless composition above.

## 4. Action items filed

- **katgpt-rs Issue 586** — operator consistency-gap metric (PoC + gate design; the one mechanism with zero shipped cousin). **RESOLVED + removed 2026-08-14** (Bench 633): `bisimulation/consistency.rs` shipped behind opt-in `operator_consistency`; GOAT G1+G2+G4 ALL PASS (regime separation; 208ns @ N=64; 0 allocs). Design notes recorded in the bench doc — notably the lookup-binding divergence: the trained model partially transfers one exemplar to neighboring levels (0/24→13/24), the modelless analog repairs exactly the demonstrated level and must **re-target after re-measure** (Issue 672 implements that loop).
- **riir-ai Issue 672** — demonstration-coverage curiosity targeting (supported-context effect → exemplar-seeking curiosity + engram write policy). **RESOLVED + removed 2026-08-14** (riir-ai Bench 674): `riir-games/swarm/coverage_curiosity.rs` behind opt-in `demo_coverage_curiosity`; falsifiable A/B PASSed (targeted +0.344 closure vs demonstrated-range generic −0.001 at equal budget, 64/64 seeds; 2.6× the steelman-uniform budget-efficiency; policy path 125ns / 0 allocs). Honest divergence recorded: the paper's partial transfer (ordering 0/24→13/24 from one demo) does NOT reproduce modellessly — lookup binding repairs exactly the demonstrated level, so the policy re-targets after re-measure (3 writes for 3 missing levels). Stays opt-in pending a real game consumer (demonstration-teachable pets). **CONSUMER LANDED 2026-08-14** (riir-mmorpg-examples Issue 059 / Bench 013): the live-domain A/B ran on the real quest-combat systems — `DemonstratedSkill` over monster ranks, the pet's own swings as applications (`ApplicationRecorder` substrate bridge), hero kills within observation radius as the demonstrations, pet ATTENTION as the observation budget (the pet holds its own attack — real DPS cost — to watch the hero's fight at the plan-targeted rank). Falsifiable live A/B PASSed: targeted closure 1.000 vs generic (base-game) 0.667, 32/32 seeds; marginal cost ~37ns/tick; full-loop integration proven in the unscripted quest loop. The sell-side claim now has live-domain evidence behind it.

## 5. Relationship to existing research

- **riir-ai Research 043 (RiM)** — the fixed-slot latent workspace cousin (SUPER GOAT, shipped as slots). BDH-CQ is the recurrent-state generalization: workspace is iterated recurrence over associative state rather than reserved token positions. The two notes now bracket the design space (static slots ↔ recurrent state).
- **325 (taxonomy survey)** — BDH-CQ occupies "recurrent-depth + in-context memory" — a cell the survey flagged as sparsely populated; this paper fills it with a costed system.
- **024 (δ-Mem) / 387 (PKM)** — the `St` lineage. BDH-CQ's linear-attention special case is exactly the delta-rule accumulation we ship.
- **048/049 (HRM/TRM)** — the transductive competitors ($1.48–1.76/task); BDH-CQ's weight-free ingestion is the differentiator, and the same differentiator separates our self-adaptive track from per-task fine-tuning.
- **273 (ELT) / 344 (fixed-point halting)** — effort scaling; continuous halting subsumes the discrete tiers.
- **453 (Variable-Rank Domain Experts)** — BDH's "low-rank communication in large feature space" is the same widening-then-project shape.

## 6. Key insight

> The paper's durable result is not the ARC score — it is the demonstration that **memory (what was shown) and reasoning (what is being computed) can be two roles of one recurrent substrate, separated only by a freeze line**: `St` accumulates, then freezes; `Hr` iterates against the frozen copy. Our stack already has both halves and the freeze discipline; what we lack is the **measurement** of whether a bound rule is applied consistently (Issue 586) and the **acquisition policy** that one well-chosen exemplar at the failing complexity is the cheapest fix (Issue 672). Coverage, not capacity, was the binding constraint in the paper — assume the same here until the PoCs say otherwise.

## 7. Paper Metadata

- arXiv:2608.09888v1 [cs.NE], 10 Aug 2026. 17 pages.
- Eval: ARC-AGI-1 public 400-task split; ConceptARC; controlled post-freeze ladders (propagation/copying/ordering/nesting; motif composition; conditional selection; panel union; support chains).
- Independent black-box audit reproduced 29.5% pass@2 (Bielik + NYU co-authors).
- Code/data: github.com/pathwaycom/arc-task-gen (task generator only; model proprietary).
- Cost basis: 0.85 H200 GPU-sec/task @ $3/hr → $0.00070/task.
