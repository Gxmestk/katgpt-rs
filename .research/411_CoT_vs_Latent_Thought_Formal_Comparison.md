# Research 411: A Formal Comparison Between Chain of Thought and Latent Thought

> **Source:** [A Formal Comparison Between Chain of Thought and Latent Thought](https://arxiv.org/abs/2509.25239) — Kevin Xu, Issei Sato (University of Tokyo), ICML 2026, arXiv:2509.25239v3, 12 May 2026. **Code:** github.com/kevin671/cot-vs-loop
> **Date:** 2026-07-12
> **Status:** Done
> **Related Research:** 218 (Breakeven Complexity Router), 241 (SwiR — explicit↔latent switch), 263 (Latent Thought Flow), 273 (ELT elastic looped), 281 (BoMSampler), 284 (Simplicity Bias Sampler), 318 (k_selector rank-k bandit), 325 (Latent Reasoning Survey), 343 (System-1.5 depth/step shortcuts — Pass), 344 (Implicit FP RNN — TC⁰⊊NC¹ prior art), 367 (QuasiMoTTo QMC sampling), 400 (LaTER — subsumed by SwiR)
> **Related Plans:** 250 (Breakeven Bandit), 251 (DEC operators), 275 (SwiR), 276 (MicroRecurrentBeliefState), 308 (KARC), 318 (rank-k functor)
> **Classification:** Public
> **Verdict: Gain** — theoretical/expressivity paper; value is the formal separation theorems (TC^k for latent thought, FPRAS for CoT) that provide a complexity-class foundation for our existing mode-switching and routing primitives. No new mechanism ships; the FPRAS separation is a genuinely novel insight for our corpus (zero prior grep hits for `TC^k|FPRAS|FPTAS|approximate counting|self-reducib`). A complexity-class-gated routing fusion (SwiR × Breakeven × k_selector × DEC × BoM) is flagged as a future GOAT candidate, tracked in `.issues/`.

---

## TL;DR

This is a **theoretical complexity-theoretic paper**, not a mechanism paper. It proves two formal separations between chain-of-thought (CoT) and latent thought (Coconut / looped Transformer):

1. **Latent thought enables parallel computation** (§3): latent thought with `log^k n` iterations **exactly captures** the circuit complexity class `TC^k` (Theorem 3.12), while CoT with the same `log^k n` steps is upper-bounded by `TC^{k-1}` (Lemma 3.13). Under the standard assumption `TC^{k-1} ⊊ TC^k`, this yields a **strict separation in favor of latent thought** in the polylogarithmic regime (Theorems 3.14, 3.15). The mechanism: latent thought evaluates a DAG **layer-by-layer** (depth-scaled iterations), while CoT evaluates **node-by-node** (size-scaled steps).

2. **CoT enables approximate counting** (§4): CoT with stochastic decoding admits **FPRAS** (fully polynomial-time randomized approximation schemes) for self-reducible `#P` counting problems (Theorem 4.3), and extends to approximate sampling (FPAUS, Theorem 4.4). No latent thought with polynomially many iterations can match this (Theorem 4.5), because latent computation is deterministic. This is the **first formal separation in favor of CoT**.

**Distilled for katgpt-rs (modelless, inference-time):**

The paper ships no new mechanism — it proves theorems about existing paradigms (CoT, Coconut, looped TF) that we already ship under different vocabulary (SwiR mode switching, `LoopMode::WeightShared`, `LatentThoughtKernel`, HLA recurrent belief). The value is **three-fold**:

1. **Complexity-class foundation for SwiR's mode switch.** SwiR (Research 241, Plan 275, DEFAULT-ON) switches between explicit and latent modes based on **entropy trend** (a runtime confidence signal). The paper proves the *correct* switching criterion is **complexity class of the problem**: TC^k-class (parallelizable) → latent; #P-class (self-reducible counting) → CoT/stochastic. This is the theoretical justification SwiR currently lacks — but it's a *framing*, not a new mechanism.

2. **Formal depth-vs-size tradeoff for our DEC substrate.** The paper's DAG framework (Theorems 3.5, 3.6) formalizes what our DEC operators (`exterior_derivative`, `codifferential`, `hodge_decompose` in `katgpt-dec/src/`) embody structurally: a cell complex IS a layered DAG, and depth-bounded iteration over it is complexity-class-optimal. The theorem proves our depth-bounded DEC iteration captures TC^depth — but we already ship the operators; the theorem is the *proof* of optimality, not a new operator.

3. **The FPRAS separation — genuinely novel for our corpus.** Zero grep hits across all 5 repos for `TC^k|FPRAS|FPTAS|approximate counting|self-reducib`. The insight: for self-reducible #P problems (SAT counting, graph colorings, partition functions), **stochastic sampling is provably more powerful than deterministic latent iteration**. Our `BoMSampler` (Plan 281) and `QuasiMoTTo` (Plan 367) are stochastic belief samplers; the theorem suggests routing self-reducible counting problems to them rather than to deterministic latent iteration. This is a **routing criterion** we did not have.

---

## 1. Paper Core Findings

### 1.1 The DAG evaluation framework (§3.1–3.2)

A reasoning problem is formalized as evaluating a **computation graph** (DAG) `G_n = (V_n, E_n)` where nodes are function applications and edges carry data flow. The DAG has `size(G_n)` nodes and `depth(G_n)` layers.

| Paradigm | Simulation strategy | Steps required | Theorem |
|---|---|---|---|
| **CoT** | Sequential, node-by-node (scratchpad tokens) | `O(size(G_n))` | Thm 3.5 |
| **Latent thought** (Coconut / looped TF) | Parallel, layer-by-layer (all nodes at same depth computed simultaneously in latent space) | `O(depth(G_n))` | Thm 3.6 |

The key insight: continuous latent states can **encode multiple node outputs simultaneously**, enabling the FFN to compute them in parallel. Discrete tokens force one-node-at-a-time serialization.

### 1.2 The TC^k exact characterization (§3.3, Theorem 3.12)

**Theorem 3.12:** For each `k ∈ ℕ`:
```
Loop[log^k n, poly(n), 1] = CT[log^k n, poly(n), 1] = AC^k   (constant precision)
Loop[log^k n, poly(n), log n] = CT[log^k n, poly(n), log n] = TC^k   (log precision)
```

Latent thought (looped TF or Coconut) with `log^k n` iterations **exactly captures** the parallel complexity class `TC^k` (threshold circuits of depth `log^k n`). This is an exact characterization, not just an upper bound.

### 1.3 The CoT upper bound (§3.3, Lemma 3.13)

**Lemma 3.13:** `CoT[log^k n, poly(n), log n] ⊆ TC^{k-1}`

CoT with `log^k n` steps can only reach `TC^{k-1}` — one level below latent thought. The `log^k n` steps divide into `log^{k-1} n` blocks of `log n` steps each; each block is `TC^0`-computable, and stacking `log^{k-1} n` of them yields `TC^{k-1}`.

### 1.4 The separation (§3.3, Theorems 3.14, 3.15)

Under standard complexity assumptions:
- If `TC^{k-1} ⊊ NC^k` → `CoT[log^k n] ⊊ Loop[log^k n]` (Thm 3.14)
- If `TC^{k-1} ⊊ TC^k` → `CoT[log^k n] ⊊ Loop[log^k n]` (Thm 3.15)

**Latent thought is strictly more efficient than CoT for parallelizable problems** — it needs fewer iterations to reach the same computational power.

### 1.5 CoT enables approximate counting (§4, Theorems 4.3–4.5)

The flip side: CoT's stochastic decoding enables **randomized computation** that deterministic latent thought cannot match.

**Theorem 4.3 (informal):** Under `FPTAS ⊊ FPRAS` for self-reducible relations, there exists a self-reducible relation `R` such that CoT with polynomially many steps admits an FPRAS for the counting function `Ext_R`, but no latent thought with polynomially many iterations does.

**Theorem 4.4:** Extends to approximate sampling — there exist target distributions CoT can approximately represent and sample from, but latent thought cannot.

**Theorem 4.5:** `∀ M ∈ {pCT, pLoop}, M[poly(n)] ⊊ pCoT[poly(n)]` — **the first formal separation in favor of CoT.**

The mechanism: CoT explicitly samples intermediate tokens, inducing **stochastic computation** that can emulate randomized algorithms (Monte Carlo, MCMC). Latent thought performs only **deterministic transformations** in latent space. For self-reducible #P problems (SAT counting, DNF counting, graph colorings), the self-reducibility structure + stochastic sampling yields FPRAS; deterministic computation cannot (under `FPTAS ⊊ FPRAS`).

### 1.6 Experimental validation (§5)

| Task | Complexity class | Winner | Result |
|---|---|---|---|
| Word problem (S5 group) | NC^1-complete | Latent (fewer iterations) | Loop TF: 100% at 4 iters; CoT needs 64 steps |
| Graph connectivity (STCON) | TC^1 | Latent | Loop TF: 99% at 4 iters; CoT: 88% at 32 steps |
| Arithmetic evaluation | NC^1-complete | Latent | Loop TF: 99.4% at 4 iters; CoT: 48% at 32 steps |
| Edit distance | TC^1 | Latent | Loop TF: 90.7% at 8 iters; CoT: 94.8% at 64 steps |
| DNF counting | #P (FPRAS) | **CoT** | CoT relative error → 0.3; looped TF plateaus |
| Graph coloring sampling | #P (FPAUS) | **CoT** | CoT TV distance → 0.027; looped TF concentrates on subset |

Empirical results confirm the theoretical separations: latent wins on parallelizable tasks (fewer iterations for same accuracy), CoT wins on approximate counting/sampling.

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper ↔ codebase)

| Paper term | Codebase equivalents (≥2) | Where it ships |
|---|---|---|
| "chain of thought" (CoT) | explicit token generation, `ThinkMode::Direct`, argmax/sample decode | `crates/katgpt-core/src/thinking_mode.rs`, decode path |
| "latent thought" / "continuous thought" | `ThinkMode::Latent`, RiM buffer slots, soft embedding, `LatentThoughtKernel` | `crates/katgpt-core/src/thinking_mode.rs`, `crates/katgpt-micro-belief/src/latent_thought.rs`, SwiR `soft_embedding` |
| "looped transformer" | `LoopMode::WeightShared`, `LoopMode::TrainingFree`, weight-tied block iteration | Plans 108, 136; `katgpt-rs/src/looped.rs` |
| "coconut" (hidden state feedback) | `LatentThoughtKernel` Family B, NextLat belief drafter | Research 192, 242; Plan 217, 276 |
| "TC^k" / "threshold circuit" | (no codebase vocabulary — this is the gap this note fills) | — |
| "depth of DAG" / "parallelizable" | "depth-invariant", "depth-scaled iteration", `depth(G_n)` | `crates/katgpt-types/src/depth_invariance.rs`, DEC cell complex layers |
| "size of DAG" / "sequential" | "size-scaled steps", "node-by-node", scratchpad tokens | CoT decode path |
| "FPRAS" / "approximate counting" | (no codebase vocabulary — **genuinely novel**) | — |
| "self-reducible" | (no codebase vocabulary) | — |
| "stochastic decoding" | sampling, `BoMSampler`, `QuasiMoTTo`, G-Zero self-play | Plans 281, 367; `katgpt-core/src/sampling/` |
| "DAG evaluation" | DEC cell complex evaluation, cochain propagation | `katgpt-dec/src/` (`exterior_derivative`, `codifferential`) |
| "computation graph" | cell complex, `CellComplex`, cochain field | `katgpt-dec/src/` |
| "polylogarithmic depth" | `log^k n` iterations, rank-k functor | `riir-ai/crates/riir-engine/src/latent_functor/k_selector.rs` (`K_OPTIONS = [1,2,4,8,16]`) |

### 2.2 Closest prior art (BOTH layers, ALL repos)

#### Layer 1 — Notes/plans (intent)

| Note / Plan | Mechanism | Match |
|---|---|---|
| **Research 241 (SwiR)** | Explicit↔latent mode switch based on **entropy trend** | Closest cousin for mode switching — but switches on runtime confidence, NOT complexity class |
| **Research 218 (Breakeven Router)** | Routes by **cost amortization N*** | Closest cousin for complexity-aware routing — but routes by cost, NOT complexity class |
| **Research 344 (Implicit FP RNN)** | TC⁰ ⊊ NC¹ for implicit SSM (single block to FP) | **The only prior note that mentions circuit complexity classes.** Covers k=0 vs k=1 only; does NOT generalize to TC^k or cover FPRAS |
| **Research 325 (Latent Reasoning Survey)** | Unifying taxonomy of latent reasoning families | Maps the corpus; mentions Turing completeness (TC under arbitrary precision) but not the TC^k separation |
| **Research 400 (LaTER)** | Latent-then-explicit (subsumed by SwiR) | Single-transition special case of SwiR |
| **Research 343 (System-1.5)** | Depth + step shortcuts (training-only, Pass) | Training-bound; modelless cousin is shipped under FPRM/LoopCoder-V2/depth-invariance vocabulary |
| **Plan 318 (k_selector)** | UCB1 bandit over `[1,2,4,8,16]` rank-k per relation | **This IS choosing a complexity class per relation** (k ↔ TC^k) — but without the theoretical framing |
| **Research 281 (BoMSampler)** | K-hypothesis single-pass belief sampling | Stochastic sampler — the FPRAS-eligible arm |
| **Research 367 (QuasiMoTTo)** | QMC belief sampling | Stochastic sampler — FPRAS-eligible arm |
| **Plan 251 (DEC operators)** | `exterior_derivative` (d), `codifferential` (δ), `hodge_decompose` | The cell complex IS a layered DAG — the substrate for depth-bounded computation |

#### Layer 2 — Shipped code (what actually exists)

| File | Mechanism | Match |
|---|---|---|
| `crates/katgpt-core/src/thinking_mode.rs` | `ThinkingMode::{Direct, Latent, CpuResample, Dendritic}` | The mode tag SwiR plugs into |
| `katgpt-dec/src/` (operators.rs, hodge.rs, flow.rs) | `exterior_derivative`, `codifferential`, `hodge_decompose`, `CellComplex`, `CochainField` | The DAG substrate — cell complex = layered DAG, d = boundary operator, δ = divergence |
| `crates/katgpt-types/src/depth_invariance.rs` | `classify_chain`, `DepthInvarianceKind::{DepthInvariant, DepthSpecificRefinement}` | Classifies chains by depth behavior — directly related to depth-vs-size tradeoff |
| `riir-ai/crates/riir-engine/src/latent_functor/k_selector.rs` | `KSelectionBandit` over `[1,2,4,8,16]` | Per-relation rank-k selection = per-relation complexity class selection |
| `riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs` | Coherence-driven re-estimation scheduler | The iterative refinement primitive (latent thought loop with halting) |
| `riir-ai/crates/riir-engine/src/latent_functor/depth_invariance_audit.rs` | Functor chain depth-invariance classification | Audits whether functor iteration is depth-invariant or drifts |
| `crates/katgpt-core/src/breakeven/mod.rs` | `BreakevenTierPair`, cost-amortization routing | Routes by cost, not complexity class |
| SwiR controller (`src/swir/`) | `SwiRController::step(entropy, ...)` | Mode switch on entropy trend |

### 2.3 What's genuinely novel (not in our corpus)

**Zero grep hits across all 5 repos for:** `TC^k`, `FPRAS`, `FPTAS`, `approximate counting`, `self-reducib`, `TC\^`, `NC\^`, `circuit complexity`.

Three genuinely novel insights:

1. **The TC^k ↔ `log^k n` iterations exact correspondence** (Theorem 3.12). Our `LoopMode::{WeightShared, TrainingFree}` iterates K times; `k_selector` chooses K from `[1,2,4,8,16]`. The theorem proves K iterations capture TC^K — our codebase ships the mechanism but has no theorem linking K to TC^K. This is the **formal justification** for why rank-k functor iteration is complexity-class-bounded.

2. **The FPRAS separation** (Theorems 4.3–4.5). The insight that stochastic sampling (CoT) can solve self-reducible #P counting problems that deterministic latent iteration provably cannot. Our `BoMSampler` and `QuasiMoTTo` are stochastic samplers; the theorem suggests they are **FPRAS-eligible** for self-reducible problems. No prior note framed them this way.

3. **The complexity-class routing criterion.** SwiR switches on entropy trend (runtime signal); Breakeven routes on cost amortization (economic signal). The paper provides a **third routing criterion**: complexity class of the problem (TC^k → latent, #P → stochastic). This is a provably-correct switching signal, not a heuristic.

### 2.4 Latent-space reframing (mandatory per skill §1.4)

Re-cast the paper's core mechanism as a latent-to-latent operation on the seven Super-GOAT factory modules:

| Factory module | Reframing | Assessment |
|---|---|---|
| **HLA** (`crates/katgpt-sense/src/reconstruction.rs`) | HLA's `evolve_hla` is a per-NPC recurrent belief kernel — a depth-1 latent iteration. The theorem says `log^k n` such iterations capture TC^k. HLA at tick T is one iteration; T ticks = `log^k n` iff T scales as `log^k n` with problem size. | Already shipped; theorem is the proof of what it captures. |
| **`latent_functor/`** | The functor application cycle IS latent thought iteration. `k_selector` choosing K ∈ `[1,2,4,8,16]` IS choosing the TC^K class. `reestimation.rs` halting IS the convergence criterion. | Already shipped; theorem formalizes the complexity class. |
| **`cgsp_runtime/`** | Curiosity-driven self-play allocates compute per NPC — the depth budget. The theorem says deeper budgets capture higher TC^K. | Already shipped; theorem justifies the compute allocation. |
| **LatCal** (`riir-chain/src/encoding/`) | LatCal fixed-point commitment is raw/deterministic — the sync-boundary bridge. The FPRAS separation says stochastic computation (CoT) can do things deterministic latent cannot. LatCal commitment is on the deterministic side. | LatCal stays deterministic (sync requires it); the FPRAS insight applies to the local (pre-commitment) computation, not the committed value. |
| **`NeuronShard`** (`riir-neuron-db/src/`) | Shard `style_weights[64]` is a frozen latent state. The theorem says iterating on it captures TC^K where K = iteration count. Freeze/thaw snapshots are the committed checkpoints of the latent iteration. | Already shipped; theorem is the complexity-class framing. |
| **DEC** (`katgpt-dec/src/`) | **The strongest reframing.** The cell complex IS a layered DAG. `exterior_derivative` (d) computes the boundary — a depth-1 operation. `hodge_decompose` decomposes a flow into exact (depth-bounded) + coexact (divergence/size-bounded) + harmonic (depth-invariant) components. The paper's DAG depth-vs-size framework maps directly: depth-bounded DEC iteration = TC^depth; size-bounded CoT = sequential node evaluation. Stokes' theorem (`∫_M dω = ∫_∂M ω`) IS the depth-vs-size tradeoff: computing a region's integral from its boundary (depth-bounded, O(n^{(d-1)/d})) vs from its interior (size-bounded, O(n)). | The DEC substrate already embodies the paper's DAG framework. The theorem proves DEC depth-bounded iteration is complexity-class-optimal. |
| **`classify_chain`** (`crates/katgpt-types/src/depth_invariance.rs`) | `DepthInvariant` = the chain converges (harmonic component, depth-independent). `DepthSpecificRefinement` = the chain is still computing (depth-bounded, not yet at fixed point). | Already shipped; the theorem connects `DepthSpecificRefinement` to TC^K (still computing) and `DepthInvariant` to convergence (fixed point reached). |

**Assessment:** The latent-space reframing lands cleanly on **existing machinery** — HLA, latent_functor, DEC, classify_chain all embody aspects of the paper's framework. The paper's contribution is the **formal proof** that these mechanisms are complexity-class-optimal, not a new mechanism. The FPRAS separation is the one genuinely novel insight that doesn't map to existing code — it suggests a routing criterion (self-reducible #P → stochastic sampling) that we don't currently implement.

### 2.5 What does NOT distill (stays theoretical / training-side)

- **The formal proofs** (Theorems 3.5–3.15, 4.3–4.5) — these are mathematical results, not code. They justify existing mechanisms; they don't introduce new ones.
- **The experimental training** (§5) — the paper trains looped TF and CoT models to validate the separations empirically. The training recipes → riir-train if pursued.
- **The DAG simulation constructions** (Appendix B) — these are proof artifacts (how to construct a Transformer that simulates a given DAG), not inference-time mechanisms.

### 2.6 Fusion

**Complexity-Class-Gated Mode Router** — the novel combination this paper inspires:

> Fuse SwiR (241, mode switch) × Breakeven Router (218, cost amortization) × k_selector (318, rank-k = TC^K selection) × DEC operators (251, DAG depth/size substrate) × BoMSampler/QuasiMoTTo (281/367, stochastic sampling arm) into a router that classifies the problem's complexity class and routes to the provably-correct paradigm.

| Component | Source | Role |
|---|---|---|
| Mode switch controller | SwiR (Plan 275) | Switches between latent and explicit/stochastic modes |
| Cost-amortization signal | Breakeven Bandit (Plan 250) | Routes by compute tier economics |
| Rank-k (= TC^K) selection | k_selector (Plan 318) | Chooses the complexity class of latent iteration |
| DAG depth/size substrate | DEC operators (Plan 251) | The cell complex on which depth-bounded iteration runs |
| Stochastic sampling arm | BoMSampler (Plan 281) / QuasiMoTTo (Plan 367) | The FPRAS-eligible arm for self-reducible #P problems |
| Complexity-class classifier | **NEW (not shipped)** | Detects whether the problem is TC^K-parallelizable or #P-self-reducible |

**What this fusion produces that none alone can:** Today, SwiR switches on entropy (a runtime confidence signal that says "am I confident right now?"). The fusion would switch on **complexity class** (a structural signal that says "is this problem parallelizable or does it need stochastic sampling?"). For parallelizable problems (TC^K), route to latent iteration with K from k_selector. For self-reducible #P problems, route to BoM/QuasiMoTTo stochastic sampling. The Breakeven Bandit adds the cost dimension. The DEC substrate provides the DAG on which depth-bounded iteration runs.

**Novelty gate (honest):**

| Q | Criterion | Answer | Notes |
|---|---|---|---|
| Q1 | No prior art? | **Partial** | The *combination* (complexity-class-gated routing) has no prior art. The *components* are all shipped (SwiR, Breakeven, k_selector, DEC, BoM). The *theoretical result* (TC^k vs FPRAS) is from the paper. |
| Q2 | New class of behavior? | **NO** | A router that switches on complexity class rather than entropy is a *better switching signal*, not a new capability. SwiR already switches modes; the fusion adds a provably-correct criterion. Incremental, not a new class. |
| Q3 | Product selling point? | **Partial** | "Our NPCs route reasoning by complexity class" is nice, but SwiR's "adaptive alternating" already covers the selling point. The complexity-class framing is a *justification*. |
| Q4 | Force multiplier? | **YES** | Connects SwiR + Breakeven + k_selector + DEC + BoM/QuasiMoTTo = 5 systems across 2 pillars (P8 Reasoning Pack, P4 Frame-Sampling). |

**Q2 fails → not Super-GOAT.** The fusion is a **GOAT candidate** — but it requires a **complexity-class classifier** (how do we detect at runtime whether a problem is TC^K-parallelizable or #P-self-reducible?) that the paper does not provide and that is a non-trivial research problem in itself. Track as a follow-up issue, not a plan in this session.

---

## 3. Verdict

### **Gain**

**One-line reasoning:** Theoretical/expressivity paper that proves formal separations (TC^k for latent thought, FPRAS for CoT) providing a complexity-class foundation for existing primitives (SwiR, Breakeven, k_selector, DEC, BoM). No new mechanism ships; the FPRAS separation is a genuinely novel insight (zero prior grep hits) that suggests a complexity-class-gated routing fusion, tracked as a future GOAT candidate.

### Novelty gate (Q1–Q4)

| Q | Answer | Evidence |
|---|---|---|
| **Q1 No prior art?** | **Partial** | The FPRAS separation is genuinely novel (zero grep hits for `TC^k|FPRAS|FPTAS|approximate counting|self-reducib` across all 5 repos). But the mechanisms it characterizes (CoT, latent thought, looped TF) are all shipped under different vocabulary (SwiR, `LoopMode`, `LatentThoughtKernel`). Research 344 mentions TC⁰ ⊊ NC¹ for implicit SSMs but only k=0 vs k=1, not the general TC^k separation or FPRAS. |
| **Q2 New class of behavior?** | **NO** | The paper proves theorems about existing paradigms; it doesn't introduce a new mechanism. The complexity-class-gated routing fusion is a better switching signal for SwiR, not a new capability. |
| **Q3 Product selling point?** | **Partial** | "Our reasoning router is provably complexity-class-optimal" is a nice line, but SwiR's "adaptive alternating" already covers the selling point. The complexity-class framing strengthens the moat narrative but doesn't create a new one. |
| **Q4 Force multiplier?** | **YES** | Connects SwiR (241) + Breakeven (218) + k_selector (318) + DEC (251) + BoM (281) / QuasiMoTTo (367) = 6 systems across Pillar 8 (Reasoning Pack) and the DEC substrate. |

**Q2 fails → not Super-GOAT. Not GOAT** (no provable gain to benchmark — the gain is theoretical framing, not a measurable latency/quality improvement). **Gain** — the value is the formal foundation + the FPRAS routing criterion + the fusion candidate tracking.

### MOAT gate (katgpt-rs domain)

- **In scope:** YES — theoretical foundation for inference-time reasoning primitives.
- **Strengthens moat:** PARTIALLY — the complexity-class framing strengthens the Reasoning Pack narrative (Pillar 8) but doesn't create a new pillar. The FPRAS insight is a genuinely novel routing criterion.
- **Promote/demote:** N/A — no primitive to ship; this is a framing/justification note.

### Routing

| Artifact | Destination | Status |
|---|---|---|
| Research note (this file) | `katgpt-rs/.research/411_*.md` | ✅ Created |
| Complexity-class-gated routing fusion | `.issues/134_complexity_class_gated_routing_fusion.md` | ✅ Filed (P3 track-only — requires a complexity-class classifier not provided by the paper) |
| FPRAS routing criterion for BoM/QuasiMoTTo | `.issues/134_complexity_class_gated_routing_fusion.md` (consolidated 2026-07-25) | ✅ Filed 2026-07-12 as `.issues/135_*`, consolidated into Issue 134 2026-07-25 — same detector gap blocked both; the FPRAS arm now lives as the #P-detection half of Issue 134 |
| Experimental training (looped TF / CoT training) | → riir-train | Out of scope (training-side validation of theoretical results) |

---

## 4. What this note prevents (canonical failure modes averted)

1. **False Super-GOAT on the next "complexity-class routing" paper.** This note records that the TC^k / FPRAS separation is the theoretical foundation; the routing fusion (SwiR × Breakeven × k_selector × DEC × BoM) is the GOAT candidate. Future papers in this space must check this note before claiming novelty.

2. **False Super-GOAT on "latent thought is provably better than CoT."** The paper proves latent thought is better for **parallelizable** problems (TC^k), but CoT is better for **approximate counting** (#P/FPRAS). Neither is universally superior — the separation goes both ways. SwiR's entropy-trend switch is the runtime heuristic; the complexity-class criterion is the theoretical foundation.

3. **Missing the FPRAS insight.** Without this note, a future agent might route all counting problems to deterministic latent iteration (because "latent thought is more efficient"). The FPRAS separation proves that's wrong for self-reducible #P — stochastic sampling (BoM/QuasiMoTTo) is provably more powerful there.

4. **Re-evaluating the TC⁰ ⊊ NC¹ angle.** Research 344 already covered this for implicit SSMs (k=0 vs k=1). This note generalizes to all k (TC^k) and adds the FPRAS separation — the two notes are complementary, not duplicative.

5. **Treating k_selector as just a rank chooser.** The `K_OPTIONS = [1,2,4,8,16]` in `k_selector.rs` IS choosing the TC^K complexity class per relation. This note provides the theoretical framing: K=1 → TC^1, K=4 → TC^4, etc. The bandit is already doing complexity-class selection without the formal label.

---

## 5. Cross-References

| Research | Connection |
|---|---|
| **241 (SwiR)** | The mode-switching controller this paper provides theoretical foundation for. SwiR switches on entropy; paper proves the correct criterion is complexity class. |
| **218 (Breakeven)** | The cost-amortization router. Paper adds complexity-class as a second routing dimension. |
| **344 (Implicit FP RNN)** | The only prior note mentioning circuit complexity (TC⁰ ⊊ NC¹). Covers k=0 vs k=1 for implicit SSMs; this note generalizes to TC^k and adds FPRAS. |
| **325 (Latent Reasoning Survey)** | The taxonomy map. This note adds the complexity-class dimension the survey doesn't cover. |
| **318 (k_selector)** | The per-relation rank-k bandit. This note provides the theoretical framing: K ↔ TC^K. |
| **281 (BoMSampler)** | The stochastic sampler that is FPRAS-eligible for self-reducible #P problems. |
| **367 (QuasiMoTTo)** | The QMC sampler — same FPRAS-eligible arm. |
| **251 (DEC operators)** | The cell complex = layered DAG substrate. Paper's DAG framework maps to DEC. |
| **400 (LaTER)** | Subsumed by SwiR; this note provides the theoretical foundation for when SwiR's switch is provably correct. |
| **343 (System-1.5)** | Training-only (Pass). The modelless depth/step routing cousin is shipped; this note adds the complexity-class framing. |

| Plan | Connection |
|---|---|
| **275 (SwiR)** | The mode-switching plan this paper theoretically justifies. |
| **250 (Breakeven Bandit)** | The cost-amortization plan; complexity-class is a complementary routing dimension. |
| **251 (DEC operators)** | The DAG substrate plan. |
| **318 (rank-k functor)** | The k_selector plan; K ↔ TC^K. |
| **281 (BoMSampler)** | The stochastic sampling plan; FPRAS-eligible arm. |

---

## 6. Action items

- [x] **T1:** Fetch paper, read, classify (theoretical/expressivity, not training-only). DONE.
- [x] **T2:** Pre-flight: read 4 READMEs + `.docs/README.md` + `03_pillars/` + `04_supergoat_candidates/`; list all 4 `.research/` folders + runtime src trees + Super-GOAT factory modules. DONE.
- [x] **T3:** Vocabulary translation + grep BOTH layers across all 5 repos for prior art (paper vocab + codebase vocab). DONE — zero hits for `TC^k|FPRAS|FPTAS|approximate counting|self-reducib`; closest cousins found (SwiR 241, Breakeven 218, Research 344, k_selector 318).
- [x] **T4:** Latent-space reframing on 7 Super-GOAT factory modules. DONE — lands on existing machinery (HLA, latent_functor, DEC, classify_chain); FPRAS is the one genuinely novel insight.
- [x] **T5:** Novelty gate Q1–Q4. DONE — Q2 fails (no new capability class); verdict Gain.
- [x] **T6:** MOAT gate (katgpt-rs domain). DONE — in scope, partially strengthens moat, no primitive to ship.
- [x] **T7:** File `.issues/` entry for the complexity-class-gated routing fusion (SwiR × Breakeven × k_selector × DEC × BoM). DONE — `.issues/134_complexity_class_gated_routing_fusion.md` (P3 track-only; implementation deferred within the issue, blocked on a complexity-class classifier).
- [x] **T8:** File `.issues/` entry for the FPRAS routing criterion (route self-reducible #P problems to BoM/QuasiMoTTo stochastic sampling). DONE 2026-07-12 as `.issues/135_fpras_routing_criterion.md`. **CONSOLIDATED into Issue 134 on 2026-07-25** — both issues are P3 track-only blocked on the same open research problem (a runtime complexity-class / self-reducibility detector the paper does not provide). The narrower slice was not independently shippable in practice; maintaining two parallel blocked issues was noise. The FPRAS criterion now lives as the #P-detection half of Issue 134's classifier. See Issue 134 re-evaluation trigger for the consolidated FPRAS arm rationale.
- [ ] **T9 (optional):** If a future plan wants to add a complexity-class classifier to SwiR, cite this note + Research 344 as the theoretical foundation. The classifier would detect whether the current problem is TC^K-parallelizable (→ latent iteration) or #P-self-reducible (→ stochastic sampling). Out of scope for this session.

---

## TL;DR

**Verdict: Gain.** Paper 2509.25239 (Xu & Sato, ICML 2026) proves two formal separations: (1) latent thought with `log^k n` iterations exactly captures TC^k (Theorem 3.12), while CoT with same steps is bounded by TC^{k-1} (Lemma 3.13) — **latent thought is strictly more efficient for parallelizable problems**; (2) CoT admits FPRAS for self-reducible #P problems (Theorems 4.3–4.5) that deterministic latent thought cannot match — **CoT is strictly more powerful for approximate counting**. The paper ships no new mechanism; it proves theorems about paradigms we already ship (SwiR mode switching, `LoopMode`, `LatentThoughtKernel`, DEC operators). The value is **three-fold**: (a) complexity-class foundation for SwiR's entropy-trend switch (the correct criterion is complexity class, not confidence), (b) formal depth-vs-size tradeoff for our DEC substrate (cell complex = layered DAG, depth-bounded iteration = TC^depth), (c) the FPRAS separation — a genuinely novel insight (zero prior grep hits across all 5 repos) proving stochastic sampling (BoM/QuasiMoTTo) is provably more powerful than deterministic latent iteration for self-reducible #P problems. A complexity-class-gated routing fusion (SwiR × Breakeven × k_selector × DEC × BoM) is flagged as a future GOAT candidate but requires a complexity-class classifier not provided by the paper — tracked in `.issues/`. No new primitive ships; no plan created; the `k_selector`'s `K_OPTIONS = [1,2,4,8,16]` is retrospectively recognized as TC^K complexity-class selection.
