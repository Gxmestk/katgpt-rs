# Research 450: Algorithmic Syntactic Causal Identification (Monoidal Signature Fixing)

> **Source:** Cakiqi & Little, *Algorithmic syntactic causal identification*, arXiv:2403.09580v2 [cs.AI], 30 Jan 2025. https://arxiv.org/pdf/2403.09580
> **Date:** 2026-07-18
> **Status:** Active
> **Related Research:** 398 (Canvas Engineering — closest cousin: declared causal topology + reachability), 219 (DEC operators / Stokes substrate), 296 (Stokes vocabulary crosswalk), 043 (Interventional SFT — do-calculus for token masking, training context), 312 (Viable Manifold Graph — CSR transitive closure), 278 (Engram — conditional pattern memory), 141 (riir-ai KG triple typology), 300/301 (riir-neuron-db Experience Graph DB foundation)
> **Related Plans:** 419 (Canvas schema compiler), 251 (DEC operators), 312 (Viable manifold graph), 319 (riir-neuron-db Experience Graph)
> **Classification:** Public

---

## TL;DR

The paper gives a **purely syntactic, algorithmic characterization of general causal identification by fixing** — the graph-rewriting algorithm behind Pearl's do-calculus / Richardson et al. 2012 Theorem 49 (the ID algorithm). The key move is to replace probability theory with **symmetric monoidal categories (SMCs)**: causal models become *signatures* `(Σ₀ objects, Σ₁ morphisms, dom, cod)`, and causal identification becomes a sequence of signature manipulations `Hide` (marginalize), `Control` (intervene), `Fix = Control ∘ Hide`, `Fixseq` (recursive fixing sequence), `Simplify` (delete identity/dead modules), and `Combine` (exterior signatures). The output is a *purely syntactic interventional signature* `Σ_{Y|do(A)}` — independent of any semantic interpretation (probabilistic, deterministic, min-plus, relational). The paper derives syntactic analogues of back-door and front-door adjustment and shows it scales to 4-node ADMGs with bidirected confounders.

**Distilled for katgpt-rs (modelless, inference-time):** a generic `AdmgSignature` data structure + the six signature-manipulation functions (`hide`, `control`, `fix`, `fixseq`, `simplify`, `combine_exteriors`) + the top-level `identify(Y, do(A))` driver from Theorem 1. Pure graph rewriting, zero gradient descent, zero probability theory. The output signature is a sub-ADMG with explicit domain/codomain per node that can be lowered to (a) an attention mask, (b) a KG-triple filter, (c) a validator rule, or (d) a Claim-Rubric evidence graph — depending on the consumer.

**Verdict: Gain.** The algorithm is genuinely novel to our stack (no prior art — verified by grep across all 7 repos in BOTH paper vocabulary and codebase vocabulary: `causal identification`, `do-calculus`, `ADMG`, `fixing`, `monoidal`, `district`, `fixable` all return zero codebase hits). It is **strictly more than Canvas Engineering 398 ships**: Canvas *declares* a topology and reads off reachability guarantees (exact marginal independence for binary masks); this paper *algorithmically derives* the interventional signature from an arbitrary ADMG with latent confounders. But: (a) the paper is purely theoretical with no empirical validation, (b) no concrete consumer in the 20Hz tick needs graph-rewriting causal identification today (NPCs decide via forward-pass + sigmoid gate; validators check chain consistency via commitment, not causal-graph rewriting), (c) the closest real use case is *offline* (consolidation, claim verification, KG curation, counterfactual reasoning-pack queries), and (d) Canvas already covers the declared-topology half. Per §1.55, this is **Gain, not Pass** — it is a genuinely novel modelless primitive that connects to ≥4 pillars (DEC graph substrate, KG triples, experience_graph, Canvas, validators, CLR) and could become a Super-GOAT fusion **if a concrete consumer materializes**. The fusion idea (Canvas × this × KG triples × experience_graph → offline counterfactual NPC reasoning) is tracked as TBD — needs Q1–Q4 re-check before any Super-GOAT re-evaluation.

---

## 1. Paper Core Findings

### 1.1 The move: probability theory → symmetric monoidal categories

Classical causal identification (Pearl 2009, Shpitser 2008, Richardson et al. 2012) is expressed in **classical probability theory** over causal Bayes nets (CBNs): random variables, joint distributions, conditioning, the do-operator. The Markov property binds the distribution to the graph.

The paper's claim: probability theory is **one** axiomatic foundation, not the only one. **Symmetric monoidal categories (SMCs)** are an alternative axiomatization where causal models are *signatures* and causal manipulations are *signature rewriting operations*. The Markov property is no longer required — the algorithm is **purely syntactic**, applicable to any SMC interpretation:

- Markov categories (probabilistic) — recovers Pearl's classical do-calculus.
- Sets-and-functions (deterministic) — recovers Pearl's front-door formula `f_{Y|do(X)}(x) = f_Y(f_{X'}(), f_Z(x))`.
- **Min-plus semifield** (Cakiqi & Little 2022) — gives `q(y|do(x)) = min_u [q(y|x,u) + q(u)]`, where `q` are *biases* / *clique potentials* widely encountered in ML.
- Relational databases, HDLs, distributed systems (Petri nets), ML pipelines.

This is the key unblock: causal identification **without** probability theory.

### 1.2 The signature (Definition 1)

A symmetric monoidal signature `Σ = (Σ₀, Σ₁, dom, cod)` where:
- `Σ₀` — set of object terms (the variables/nodes).
- `Σ₁` — set of morphism variables (the causal modules — one per node).
- `dom`, `cod : Σ₁ → Mon(Σ₀)` — give the domain/codomain of each morphism.
- `Mon(Σ₀) = (Σ₀, ⊗, 1)` — the free commutative monoid (object expressions like `A²B`).
- `v : 1 → A` — morphism with no input (source/exogenous).
- `u : A → 1` — morphism with no output (deletion/sink).

The **maximal model** is the expression where every causal module appears once, composed via `·` (sequential) and `⊗` (parallel), inserting identities/copies as needed for type matching. The **exterior signature** `Ext(Σ)` hides the internals behind a single composite morphism.

### 1.3 From ADMG to signature (§2.2)

For an ADMG `G = (V_G, E_G)`, the generated signature is:
- `Σ_G,0 = V_G` (objects = nodes).
- `Σ_G,1 = { Module(V') | V' ∈ V_G }` (one morphism per node).
- For each node `V` with causal module `v = Module(V)`:
  - `dom(Module(V)) = ⊗_{V' ∈ pa_G(V)} V'` (parents as monoidal product).
  - `cod(Module(V)) = V^{|ch_G(V)|+1}` (self + one copy per child).

**Chain-factored signature** `Σ_F` replaces `pa_G` / `ch_G` with `pre_G` / `succ_G` from a topological ordering — needed so domain/codomain are explicit. This is the working representation for the algorithm.

### 1.4 The six signature manipulations (§2.4 — the algorithm)

| Op | Definition | Intuition |
|----|-----------|-----------|
| `Hide_V(Σ)` | Sets `cod'(Module(V)) = V^{|succ_G(V)|}` (drops the +1 self-copy) | **Marginalization** — V's output is no longer observed downstream. |
| `Control_V(Σ)` | Sets `dom'(Module(V)) = V` (replaces module with copy-id); for all other `v'`: `cod'(v') = cod(v') \ dom(Module(V))` (multiset diff — deletes incoming V-wires) | **Intervention** — Pearl's `do(V)` cuts incoming edges. |
| `Fix_V` | `Control_V ∘ Hide_V` | **Fixing** = control then hide = the Richardson fixing operator. |
| `Fixseq_W` | Recursive: pick fixable `V ∈ W'`; if `ch_Σ(V) = ∅` use `Hide_V` else `Fix_V`; recurse. | **Valid fixing sequence** for a set W. |
| `DeleteId(Σ)` / `Simple(Σ)` | Drop modules where `dom = cod = V` (identities) or `cod = 1` (dead-ends); iterate to fixed point. | **Simplification** — prune no-op modules. |
| `Ext(Σ₁) ∪ Ext(Σ₂)` | Union objects + morphisms; `cod'(Module(V)) = V^{|ch_Σ(V)|+1}` over the combined graph. | **Combine** exterior signatures of separately-fixed districts. |

### 1.5 Theorem 1 — Syntactic ID algorithm

For ADMG `G`, cause `A ⊂ V_G`, effect `Y ⊂ V_G`, with `A ∩ Y = ∅`:
1. `Y⋆ = an_{G_{V_G\A}}(Y)` — ancestors of Y in the sub-ADMG after removing A.
2. `D⋆` = districts of subgraph `G_{Y⋆}` (connected components via bidirected edges).
3. **Identifiability condition**: for every district `D' ∈ D⋆`, the set `V_G \ D'` must be a valid fixing sequence.
4. **If identifiable**, the interventional signature is:

   ```
   Σ_{Y|do(A)} = Hide_{Y⋆\Y} ( ⋃_{D' ∈ D⋆} Simple(Fixseq_{V_G\D'}(Σ_F)) )
   ```

That's the entire algorithm. It produces a signature where the causal modules are explicitly typed — interpretable as probability distributions, deterministic functions, min-plus biases, or anything else.

### 1.6 Applications in the paper

- **Back-door adjustment** (§3.1): single-confounder ADMG → `Fix_X` produces the syntactic back-door formula.
- **Front-door adjustment** (§3.2): classic front-door ADMG with `X ↔ Y` → district decomposition `{Y}, {Z}` → produces the syntactic front-door formula.
- **Complex 4-node example** (§3.3, Richardson et al. 2012 Example 51): ADMG `X3 ← X1 → X2, X2 → X3, X3 → X4, X2 ↔ X4` → identify `Σ_{X4|do(X2)}` via three district fixings.

---

## 2. Distillation (modelless, inference-time)

### 2.1 Modelless split — the entire algorithm is modelless

Every operation in Theorem 1 is **graph rewriting on a signature**:
- `Hide`, `Control`, `Fix` are signature field updates (`dom`/`cod` rewrites).
- `Fixseq` is a recursive pick-fixable-then-apply loop.
- `Simplify` is a fixed-point iteration dropping identity/dead modules.
- `Combine` is a union + codomain recompute.

**Zero gradient descent. Zero probability theory. Zero training.** The output is a structural artifact (a signature), not a numerical distribution. This is as modelless as it gets — the entire paper is inference-time graph manipulation.

**§3.5 modelless-unblock check:** N/A — this paper has no training loop to defer. The "value" is the algorithm itself, not a training-target math. Path 0 decomposition returns "no training components" because the algorithm has no training components.

### 2.2 Vocabulary translation (paper ↔ codebase) — MANDATORY before novelty claim

| Paper term | Codebase equivalent | Verified shipped? |
|------------|---------------------|-------------------|
| "causal identification" / "ID algorithm" | (none — no graph-rewriting ID primitive) | **NO** |
| "do-calculus" / "intervention" | SFT token masking (R043 — training context only); `Control` analog (none at graph level) | **Partial — training only** |
| "ADMG" / "acyclic directed mixed graph" / "bidirected edges" / "confounding" | `KgTriple` (head→rel→tail, directed only); `ExperienceGraph` (parent/sibling edges, directed); `CanvasTopology` (directed connections) — **none have bidirected confounder edges** | **NO for bidirected; YES for DAG** |
| "district" / "fixable" / "fixing sequence" | (none) | **NO** |
| "monoidal signature" / "string diagram" / "symmetric monoidal category" | (none — no SMC primitive) | **NO** |
| "marginalization" / `Hide` | (none at signature level; `softmax`-style marginalization in attention is different) | **NO** |
| "exterior signature" / "maximal model" | (none — closest is `CommittedFieldBlend.pi` aggregation) | **NO** |
| "topological ordering" / `pre_G` / `succ_G` | Canvas `FlowGraph` CSR (Plan 419 — reachability BFS); Viable Manifold Graph (Plan 312 — transitive closure); `ZonePoset` (cubical_nerve/poset.rs — ancestors/transitive closure) | **YES (graph substrate)** |
| "back-door adjustment" / "front-door adjustment" | (none — no graph-level adjustment primitive) | **NO** |
| "Markov category" / "affine SMC" | (none — no category-theory primitive) | **NO** |
| "min-plus semifield interpretation" | Tropical (max,+) algebra (Plan 337, R321) — **the closest codebase cousin for the semantic interpretation, but it's a different layer** | **Partial — different layer** |

**Grep summary (both vocab sets, all 7 repos, notes + code):**
- `causal.identif|do.calculus|back.door|front.door|interventional|ADMG|acyclic.directed.mixed|monoidal|fixing.sequence|syntactic.causal|string.diagram` → ZERO codebase hits; the only `.md` hits are Research 043 (Interventional SFT, training-context do-calculus for token masking) and Canvas 398 (which mentions "causal topology" but means *declared* topology, not algorithmic identification).
- `causal.identif|do.calculus|back.door|front.door|ADMG|monoidal|fixable|district` on `**/*.rs` → ZERO meaningful hits (one stray `back-door` comment in a test about coordinate leakage; one `do-calculus` comment in `riir-gpu/dataloader.rs` referencing Research 043).

**Three-layer check (notes + code + vocab translation) confirms: Q1 NO prior art — the algorithm itself does not ship.** Canvas 398 ships declared topology + reachability; it does NOT ship algorithmic identification via fixing.

### 2.3 Prior-art surface — what already ships (verified grep + read)

1. **`katgpt-rs/crates/katgpt-core/src/canvas/`** (Plan 419, Research 398) — `CanvasSchema` + `CanvasLayout` + `CanvasTopology` + `compile_schema` + `reachability.rs` (`FlowGraph` CSR + `can_reach` BFS). **Closest cousin.** Ships declared topology → compiled mask + exact-marginal-independence guarantee for binary masks. **Does NOT ship algorithmic identification via fixing** — Canvas's reachability is *given* by the declared topology; this paper's identification is *derived* from an arbitrary ADMG with latent confounders. Different direction.
2. **`katgpt-rs/crates/katgpt-core/src/cubical_nerve/poset.rs`** — `ZonePoset` (distributive meet-semilattice) with `leq`, `meet`, transitive closure. The graph-poset substrate. **Not causal identification** — it's order theory for the cubical nerve functor.
3. **`katgpt-rs/crates/katgpt-core/src/viable_manifold_graph.rs`** (Plan 312) — CSR `SafeManifoldGraph` + transitive closure. **Reachability substrate, not identification.**
4. **`riir-neuron-db/src/experience_graph/`** (Plan 319) — `ExperienceGraph` papaya-backed node store + `as_of.rs` bi-temporal log + `graph.rs` latent-seeded NS traversal. **Graph database substrate**; no causal identification.
5. **`riir-ai/crates/riir-engine/src/kg/`** — `KgTriple` (head, relation, tail, confidence) + statistical extraction pipeline. **The ADMG source if we ever wire causal ID** — but currently directed-only, no bidirected confounder edges, no identification algorithm.
6. **`riir-ai/crates/riir-engine/src/kg_hyperedge/`** — hyperedge extension. Could express multi-entity confounders but doesn't do identification.
7. **`riir-ai/crates/riir-engine/src/causal_validation/`** — OV-circuit mech-interp harness (Plan 360). **Different "causal"** — head-importance via patched forwards, not graph-rewriting identification.
8. **`katgpt-rs/crates/katgpt-core/src/causal_head_importance/`** (Plan 358) — attention-mass / OV-norm cheap proxy. **Different layer entirely.**
9. **`katgpt-rs/.research/043_Interventional_SFT_Causal_Token_Masking.md`** + `riir-gpu/dataloader.rs` — Pearl do-calculus applied to SFT loss masking (`labels[i] = -100` for agent tokens). **Training context** — operates at token level, not graph level. Different layer.
10. **Tropical (max,+) algebra** (Plan 337, R321) — closest codebase cousin for the *min-plus semifield interpretation* the paper mentions in §3.1. Different layer (algebra vs causal identification).
11. **DEC operators** (`katgpt-rs/crates/katgpt-core/src/cubical_nerve/`, Plans 251–252) — `exterior_derivative` (d), `codifferential` (δ), `hodge_laplacian` (Δ). **Graph/topology substrate**; maps the "boundary operator" intuition but for cell complexes, not ADMGs.
12. **`riir-neuron-db/src/vibe.rs`** — `KgTripleTemplate { subject, predicate, object }` (BLAKE3 hashes). Used for vibe phase cycling, not causal identification.

**The gap (confirmed):** no `AdmgSignature`, no `fixing_sequence`, no `identify(Y, do(A))`, no district-decomposition, no back-door / front-door adjustment at the graph level. The pieces are scattered (Canvas topology, DEC graph substrate, KG triples, experience_graph, Tropical algebra) but the unified causal-identification algorithm does not ship.

### 2.4 Compute-unit translation (the R368 lesson — does NOT trigger here)

The paper has no "N LLM calls/step" structure. Its compute unit is "one signature manipulation" — pure graph rewrite. For us, the analog compute unit is "one `Fix_V` operation" = O(|Σ₁|) field updates (update `dom`/`cod` for the fixed module + delete V-wires from other modules' codomains). The full `identify(Y, do(A))` is `O(|D⋆| × |fixing_sequence| × |Σ₁|)` — for a k-node ADMG, this is `O(k³)` worst case, but typically `O(k²)` because fixing sequences are short.

**No false-PASS risk from conflating LLM-as-implementation with LLM-as-mechanism** — the paper is pure structural algorithm, not agent orchestration.

### 2.5 Fusion — what novel combination does this enable?

**Fusion idea (novelty TBD — needs concrete consumer + PoC before any Super-GOAT re-evaluation):**

> Algorithmic syntactic causal identification × Canvas declared topology × KG triples (`KgTriple` / `KgTripleTemplate`) × `experience_graph` (`ExperienceGraph` AS-OF + latent-seeded NS traversal) × Claim Rubric (L1/L2/L3 evidence ladder) → **An offline counterfactual reasoning primitive that, given a game-world KG with latent confounders (e.g., unobserved social tensions between factions, unobserved resource shortages affecting multiple NPCs), algorithmically derives the interventional signature `Σ_{Y|do(A)}` — "what happens to quest state Y if we intervene on NPC X's behavior, given the confounders we can't observe?"**

The NEW capability this fusion would produce: **syntactic counterfactual query planning over the game-world KG**, without requiring a probabilistic model of the world. None of the constituents alone does this:
- Canvas declares topology; it does not derive interventional signatures from an existing graph.
- KG triples are a static fact store; they don't answer counterfactuals.
- `experience_graph` does AS-OF temporal queries + latent ANN traversal; it doesn't do causal identification.
- Claim Rubric verifies claims against evidence; it doesn't derive what *would* happen under intervention.
- DEC operators model fields on cell complexes; they don't identify causal effects in confounded graphs.

**But:** the honest caveats are significant:
1. **No concrete consumer needs this today.** NPCs decide via forward-pass + sigmoid gate. Validators check commitment consistency. The KG is used for retrieval, not counterfactual reasoning. The Claim Rubric verifies, not predicts.
2. **The algorithm operates offline** (5–10ms+ for non-trivial ADMGs). It cannot run in the 20Hz tick. It would slot into consolidation / quest-authoring / claim-verification / GM tooling — not the hot loop.
3. **The paper provides no empirical validation** — it's a theoretical CS paper with worked examples but no benchmarks. The gain is unmeasured.
4. **The min-plus / Tropical interpretation is interesting but speculative** — the paper mentions it as one possible interpretation; we ship Tropical (Plan 337) but not wired to a causal layer.
5. **ADMGs with bidirected confounder edges don't naturally arise in our KG** — `KgTriple` is directed (head→relation→tail). We'd need to *add* a confounder layer (e.g., "this faction tension affects both NPC X and quest Y but is never directly observed") — which is itself a research question.

Per §1.5, "candidate" language is avoided: this is a **Gain** with a tracked fusion follow-up, NOT a deferred Super-GOAT commitment. Re-open the Super-GOAT gate IF/WHEN a concrete consumer materializes (e.g., a future "counterfactual NPC reasoning" plan, or a GM tool that needs to answer "what if" questions over the game-world KG).

### 2.6 Latent-space reframing (mandatory per workflow §1 step 3)

The paper is already substrate-independent (the whole point is "syntactic, not probabilistic"). The latent-space reframings:

- **(a) HLA per-NPC latent state**: not directly applicable. HLA is a continuous affect manifold; the algorithm operates on discrete graph signatures. Bridge: if each HLA axis were modeled as a graph node with causal edges to other axes, the algorithm could identify which axes causally affect which — but this is a speculative modeling choice, not a proven gain.
- **(b) `latent_functor/` operations**: partial. The functors are vector ops (zone_gating, reestimation, arithmetic, cross_game); they don't naturally form an ADMG. Bridge: a `CausalFunctor` that applies `identify(Y, do(A))` to a latent-state subgraph — but no concrete consumer.
- **(c) `cgsp_runtime/` curiosity signals**: not applicable. Curiosity is a scalar exploration drive; causal identification is a graph query.
- **(d) LatCal fixed-point commitment (riir-chain)**: **strongest latent-reframing angle.** LatCal is already a deterministic, committed, raw-numeric substrate. The signature manipulations (`Hide`, `Control`, `Fix`) are also deterministic graph rewrites. A LatCal-committed `AdmgSignature` could be quorum-verified: "all nodes agree that the interventional signature of `do(NPC_X.dies)` on `quest_Y.state` is this subgraph." This would be a **sync-boundary bridge** for causal claims. But again — no concrete consumer.
- **(e) NeuronShard / freeze envelope / consolidation / AnyRAG / vibe KG**: partial. The `vibe.rs` `KgTripleTemplate` is the natural input format; `MerkleFrozenEnvelope` is the natural commitment layer; Raven/δ-Mem consolidation is the natural offline processor. But no consumer wires them to a causal-identification algorithm.
- **(f) DEC Stokes operators (`exterior_derivative` d, `codifferential` δ, `hodge_decompose`)**: the paper's `Hide` is structurally similar to DEC's `δ` (codifferential = boundary-of-coboundary = "marginalize the dual field"); `Control` is similar to restricting to a subcomplex; `Fix = Control ∘ Hide` mirrors `δ ∘ d`. But this is a loose analogy — DEC operates on cell complexes (geometry), the paper operates on ADMGs (causality). **Not the same primitive.**

The latent reframing confirms: the algorithm is genuinely orthogonal to our latent-state machinery. It's a **graph-layer** primitive that would *consume* latent outputs (KG triples, shard commitments) and *produce* interventional signatures consumable by verifiers / claim rubrics / GM tooling. It does not replace any latent op.

---

## 3. Verdict — Gain

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|---------|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars) | Open primitive + private guide + plans |
| **GOAT** | Provable gain over existing approach, but not a new class of capability | Plan + implement, feature flag + benchmark |
| **Gain** | Incremental improvement, useful but not headline-worthy | Plan only, behind feature flag |
| **Pass** | Not relevant, OR training-only (→ riir-train note, stop) | One-line note |

**Verdict: Gain.**

**One-line reasoning:** The algorithmic syntactic causal identification primitive (Theorem 1's `identify(Y, do(A))` via `Fixseq` + `Simple` + district-combine) is genuinely novel to our stack (zero prior-art hits across all 7 repos in both paper and codebase vocabulary) and is purely modelless (graph rewriting, no training, no probability theory), but (a) the paper is purely theoretical with no empirical validation, (b) no concrete consumer in the 20Hz tick or even the offline consolidation loop needs graph-rewriting causal identification today, (c) Canvas Engineering 398 already ships the declared-topology + reachability half, (d) the new half (algorithmic identification via fixing) would require an ADMG-with-confounders input layer that doesn't exist in our KG, and (e) the strongest use case (offline counterfactual NPC reasoning, GM "what-if" tooling, claim verification under unobserved confounders) is speculative — no plan currently calls for it.

### 3.1 Novelty gate (Q1–Q4)

- **Q1 (No prior art?): YES.** Three-layer check (notes + code + vocabulary translation) confirms zero prior art for `AdmgSignature` / `fixing_sequence` / `identify(Y, do(A))` / district decomposition / back-door-front-door at the graph level. Canvas 398 ships declared topology + reachability; it does NOT derive interventional signatures algorithmically.
- **Q2 (New class of behavior?): PARTIAL → NO today.** "Algorithmic counterfactual query planning over a game-world KG with latent confounders" IS a new capability class — but no consumer needs it today, and the algorithm is offline-only (5–10ms+, not 20Hz-tick-compatible). The capability is new; the demand is unproven.
- **Q3 (Product selling point?): POTENTIAL but unproven.** "NPCs that reason counterfactually about the game-world KG, syntactically (no probability theory), even under unobserved confounders" is a real selling point IF a consumer materializes. Today it is a solution seeking a problem.
- **Q4 (Force multiplier?): YES.** Connects to Canvas (closest cousin), DEC graph substrate, KG triples (`KgTriple` / `KgTripleTemplate`), `experience_graph` (AS-OF + latent-seeded traversal), Claim Rubric, validators, Tropical (min-plus interpretation), LatCal commitment (sync-boundary bridge angle), freeze envelope. ≥7 pillars/systems.

**Q2 and Q3 fail today → not Super-GOAT.** Per §1.5, "candidate" language is avoided. This is a **Gain** with a tracked fusion follow-up, not a deferred Super-GOAT commitment.

### 3.2 MOAT gate per domain (§1.6)

- **katgpt-rs (public engine):** the `AdmgSignature` + six signature-manipulation functions is a **paper-derived fundamental primitive** (generic graph-rewriting math, no game/chain/shard semantics). Ships behind feature flag `causal_identification` if implemented. **In scope** as a public engine primitive — the algorithm is category-theoretic graph rewriting, broadly applicable.
- **riir-ai (private runtime):** the fusion (offline counterfactual NPC reasoning) is **pillar-level IF a consumer materializes**. Today deferred — no guide created (Gain verdict, not Super-GOAT).
- **riir-chain (private chain):** the LatCal-committed interventional signature angle is interesting (sync-boundary bridge for causal claims) but speculative. No reroute needed.
- **riir-neuron-db (private shards):** the `vibe.rs` KgTripleTemplate + `MerkleFrozenEnvelope` would be the natural input/output layers. Today no consumer.
- **riir-train:** N/A — no training component.

### 3.3 §3.6 defend-wrong PoC — NOT triggered now

The §3.6 PoC rule triggers for PASS verdicts that downgrade on "runtime analog already ships" OR for quality-parity claims. This verdict is **Gain** (not PASS) and makes **no quality-parity claim** — it explicitly states the gain is unmeasured (no empirical validation in the paper) and the consumer is unspecified. A PoC is the right *follow-up* for the fusion's potential Super-GOAT re-evaluation (concrete consumer + head-to-head against a simpler graph-reachability baseline), not a blocker for the Gain.

### 3.4 §1.55 PASS-vs-Gain check — Gain, not Pass

The paper produces a **novel modelless primitive** that does not ship. Per §1.55: "If the mechanism does not ship AND it's modelless → verdict is Gain or higher." Pass would require either (a) the mechanism ships (it doesn't) or (b) the mechanism is training-only (it isn't — it's pure graph rewriting). Therefore: **Gain.**

---

## 4. Distilled primitive — what would ship in katgpt-rs (NOT scheduled; tracked for future plan)

**This section documents the primitive shape IF a consumer materializes. It is NOT a plan — no implementation is scheduled.**

### 4.1 Core types (sketch)

```rust
// katgpt-rs/crates/katgpt-core/src/causal_id/types.rs (hypothetical)

/// Object term in a monoidal signature — a node in the ADMG.
/// Backed by a BLAKE3 hash so it can reference KG triples / shards / zones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SigObject(pub [u8; 32]);

/// Morphism variable — a causal module (one per node).
/// `dom` / `cod` are monoidal expressions over `SigObject`.
#[derive(Clone, Debug)]
pub struct SigMorphism {
    pub label: [u8; 32],          // BLAKE3 of the module name
    pub dom: MonoidalExpr,        // ⊗ of parent objects
    pub cod: MonoidalExpr,        // self + child copies
}

/// Monoidal expression: a sorted multiset of `SigObject` (the free commutative monoid).
/// Stored as a sorted `SmallVec<[SigObject; 4]>` for cache locality.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MonoidalExpr(SmallVec<[SigObject; 4]>);

/// An ADMG-derived monoidal signature.
#[derive(Clone, Debug)]
pub struct AdmgSignature {
    pub objects: Vec<SigObject>,
    pub morphisms: Vec<SigMorphism>,
    /// Bidirected-edge districts (sets of objects connected via ↔ edges).
    /// Empty for a plain DAG.
    pub districts: Vec<SmallVec<[SigObject; 4]>>,
}
```

### 4.2 The six signature manipulations (sketch)

```rust
// katgpt-rs/crates/katgpt-core/src/causal_id/manip.rs (hypothetical)

/// Marginalize V: drop the +1 self-copy from V's morphism codomain.
pub fn hide(sig: &mut AdmgSignature, v: SigObject);

/// Intervene on V (Pearl's do(V)): replace V's morphism with copy-id,
/// delete all V-wires from other morphisms' codomains.
pub fn control(sig: &mut AdmgSignature, v: SigObject);

/// Fix V = Control ∘ Hide (Richardson fixing).
pub fn fix(sig: &mut AdmgSignature, v: SigObject);

/// Recursive fixing sequence for a set W. Returns Err if no valid sequence.
pub fn fixseq(sig: &mut AdmgSignature, w: &[SigObject]) -> Result<(), FixingError>;

/// Drop identity modules (dom = cod = V) and dead-end modules (cod = 1).
/// Iterate to fixed point.
pub fn simplify(sig: &mut AdmgSignature);

/// Combine exterior signatures of separately-fixed districts.
pub fn combine_exteriors(districts: &[AdmgSignature]) -> AdmgSignature;
```

### 4.3 Top-level driver (Theorem 1)

```rust
// katgpt-rs/crates/katgpt-core/src/causal_id/identify.rs (hypothetical)

/// Identify Σ_{Y|do(A)} from an ADMG signature.
///
/// Returns `Err(NotIdentifiable)` if any district's fixing set is invalid
/// (per Theorem 1's identifiability condition).
pub fn identify(
    sig: &AdmgSignature,
    cause: &[SigObject],   // A
    effect: &[SigObject],  // Y
) -> Result<AdmgSignature, IdentificationError> {
    // 1. Y⋆ = an_{G_{V\A}}(Y)
    // 2. D⋆ = districts of G_{Y⋆}
    // 3. For each D' ∈ D⋆: check V\D' is a valid fixing sequence
    // 4. Apply Fixseq + Simple per district
    // 5. Combine exteriors
    // 6. Hide Y⋆\Y
}
```

### 4.4 Interpretations (the substrate-independent bonus)

The same `identify` output signature can be interpreted in multiple SMCs:

```rust
pub trait SmcInterpretation {
    /// Interpret a morphism as a conditional probability distribution.
    fn as_conditional_dist(&self, m: &SigMorphism) -> Option<ConditionalDist>;

    /// Interpret as a deterministic function (sets-and-functions SMC).
    fn as_deterministic_fn(&self, m: &SigMorphism) -> Option<DeterministicFn>;

    /// Interpret as a min-plus bias/potential (Tropical SMC).
    fn as_min_plus_potential(&self, m: &SigMorphism) -> Option<MinPlusPotential>;
}
```

This is where the fusion with Tropical (Plan 337, R321) would land — the min-plus interpretation `q(y|do(x)) = min_u [q(y|x,u) + q(u)]` is a natural codebase fit.

---

## 5. Risks and honest caveats

1. **No concrete consumer today.** This is the dominant risk. The algorithm is interesting but no plan calls for it. Implementing it without a consumer is gold-plating.
2. **The paper is purely theoretical.** Worked examples (back-door, front-door, 4-node) but no benchmarks, no complexity analysis beyond the recursive `Fixseq`, no comparison to alternatives.
3. **`O(k³)` worst case.** For a k-node ADMG, `Fixseq` over `V_G \ D'` can be expensive. Real game-world KGs may have thousands of nodes; identification may be too slow for any but the smallest subgraphs.
4. **ADMGs with bidirected confounder edges don't naturally arise.** Our `KgTriple` is directed. We'd need to *model* confounders (unobserved faction tensions, unobserved resource shortages) explicitly — itself a research question.
5. **Canvas 398 already covers half.** Declared topology + reachability may be enough for current needs. The marginal value of algorithmic identification (over declared topology) is unproven.
6. **The category-theory framing is heavy.** SMCs, monoidal signatures, string diagrams — this is not lightweight infrastructure. The implementation would need to be carefully abstracted to avoid leaking category theory into the rest of the codebase.
7. **The min-plus / Tropical interpretation is a footnote in the paper**, not a validated use case. Don't over-anchor on it.

---

## 6. Plan — NOT scheduled

No plan created. The fusion follow-up (§2.5) is tracked here, not in `.plans/`. Re-open a Super-GOAT gate IF:

- A concrete consumer materializes (e.g., a future "counterfactual NPC reasoning" plan, or a GM "what-if" tool that needs to answer counterfactual questions over the game-world KG).
- A PoC in `riir-ai/crates/riir-poc/` shows the algorithm produces non-trivial interventional signatures on a realistic game-world KG (≥10 nodes, ≥1 bidirected confounder).
- The PoC beats a simpler baseline (graph reachability via Canvas `FlowGraph`, or plain Bayesian-network inference if we ever ship one).

If all three pass, the Super-GOAT gate (Q1–Q4 + MOAT per domain) should be re-run before creating the private guide.

---

## 7. Cross-references

- **Closest cousin:** [Research 398](.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md) — Canvas schema compiler + reachability semantics. **The declared-topology + reachability half.** This paper adds the algorithmic-identification half.
- **DEC substrate:** [Research 219](.research/219_Topological_Neural_Operators_DEC_Inference.md) + [Research 296](.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md) + Plans 251/252 — graph/topology substrate.
- **Viable Manifold Graph:** [Research 294](.research/294_Viable_Manifold_Graph_Primitive.md) + Plan 312 — CSR reachability.
- **KG triples:** `riir-ai/crates/riir-engine/src/kg/mod.rs` + `riir-neuron-db/src/vibe.rs` — the natural ADMG source if causal ID is ever wired.
- **Experience Graph:** `riir-neuron-db/src/experience_graph/` (Plan 319, Research 300/301) — graph database substrate, potential consumer.
- **Interventional SFT:** [Research 043](.research/043_Interventional_SFT_Causal_Token_Masking.md) — Pearl do-calculus at the token level (training context). Different layer.
- **Tropical (max,+):** Plan 337, Research 321 — the min-plus interpretation the paper mentions in §3.1.
- **Claim Rubric:** Plan 307 — L1/L2/L3 evidence ladder, potential consumer for interventional signatures.
- **LatCal commitment:** `riir-chain/src/encoding/latcal*.rs` — sync-boundary bridge angle for committed causal claims.

---

## TL;DR

**Verdict: Gain.** Algorithmic syntactic causal identification (arXiv:2403.09580) is a **purely modelless, graph-rewriting algorithm** for deriving interventional signatures `Σ_{Y|do(A)}` from ADMGs with latent confounders, using symmetric monoidal categories instead of probability theory. It is **genuinely novel** to our stack (zero prior-art hits across all 7 repos, both paper and codebase vocabulary, three-layer check). It is **strictly more than Canvas Engineering 398 ships** (Canvas *declares* topology + reads off reachability; this paper *algorithmically derives* interventional signatures from arbitrary confounded ADMGs). But: (a) the paper is purely theoretical with no empirical validation, (b) **no concrete consumer needs it today** — NPCs decide via forward-pass + sigmoid gate, validators check commitment consistency, the KG is used for retrieval not counterfactual reasoning, (c) the algorithm is offline-only (5–10ms+, not 20Hz-tick-compatible), (d) ADMGs with bidirected confounder edges don't naturally arise in our directed `KgTriple` substrate. The fusion opportunity (Canvas × this × KG triples × experience_graph × Claim Rubric → offline counterfactual NPC reasoning + GM "what-if" tooling) is tracked as TBD — re-open the Super-GOAT gate IF a concrete consumer materializes and a PoC beats a simpler reachability baseline. **No plan, no guide, no implementation scheduled.** Research note only.
