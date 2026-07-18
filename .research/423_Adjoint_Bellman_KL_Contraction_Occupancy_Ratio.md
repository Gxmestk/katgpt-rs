# Research 423: FORE — Fitted Occupancy-Ratio Evaluation via Adjoint Bellman KL Contraction

> **Source:** [Fitted Occupancy-Ratio Evaluation without Bellman Completeness](https://arxiv.org/pdf/2607.05375) — Lars van der Laan (Stanford) & Nathan Kallus (Netflix + Cornell), arXiv:2607.05375v1 [stat.ML], 6 Jul 2026
> **Date:** 2026-07-14
> **Status:** Done — verdict locked (GOAT)
> **Related Research (katgpt-rs):** 298 (Inverting Bellman — the *inverse* problem, `Q → P`; FORE is the *adjoint* problem, `d_ν → d^π`), 308 (Bisimulation operator — closest shipped cousin, state-equivalence compaction; FORE adds occupancy-ratio equivalence), 219/296 (DEC Stokes substrate — adjoint Bellman ≡ codifferential under Markov-kernel cochain), 271 (MIT 6S184 diffusion-flow vocabulary crosswalk — same `fitted iteration` family), 322 (Conformal floor — applies to FORE's downstream UQ applications)
> **Related Research (riir-ai):** 123 (Latent Functor Runtime — CLR/CGSP re-estimation scheduler, the primary fusion target for "occupancy-weighted FQE without Bellman completeness"), 158 (Committed Personality — freeze/thaw convergence via KL contraction), 165 (Per-NPC Conformal UQ — the "Report the Floor" gate)
> **Related Plans:** TBD on GOAT gate pass — katgpt-rs `occupancy/` primitive (open) + riir-ai CLR stabilization fusion (private, post-PoC)
> **Classification:** Public

---

## TL;DR

The paper proves that the **adjoint Bellman operator** `B^γ_π` on discounted occupancy ratios is a **strict KL contraction** by factor γ (Lemma 3.1), and uses this to build **FORE** — a fitted-iteration estimator of the discounted occupancy ratio `ω_π,γ = d^π,γ / d_ν` whose population convergence requires **only realizability of the target ratio**, not Bellman completeness of a value/critic class. Each iteration solves a single-level KL projection (a cross-entropy / log-partition objective) using one-step target-policy transitions; no minimax, no critic class, no trajectory products. Finite-sample bounds decompose into a geometrically-decaying initialization term, a quadratic log-ratio approximation term, and a local-Rademacher statistical term. Downstream: doubly-robust value estimation with product-form error, and **occupancy-weighted FQE that converges without Bellman completeness** (the headline application).

**Distilled for katgpt-rs (modelless, inference-time):**

Two transferable primitives, both genuinely novel — **zero prior art** across the 5-repo quintet (grep for `occupancy|adjoint|Bellman|FQE|FORE|relative.entropy.contraction|doubly.robust` returns only unrelated hits: Baird appears only in test names, "Bellman" only in Research 298/308 which solve different problems):

1. **The adjoint Bellman KL contraction** (Lemma 3.1) — a pure information-theoretic fact (joint convexity of KL + Markov-kernel data processing inequality) that gives a **γ-contraction in relative entropy** for any pushforward-mixture operator `B^γ_π ω = (1−γ)ω_0 + γ · d((ων)P_π)/dν`. This is substrate-independent: any "weighted pushforward through a stochastic kernel, mixed with an initial weight" inherits the contraction. Ships as a runtime stability diagnostic and (eventually) a Lean 4 theorem.
2. **The FORE algorithm** (Algorithm 1) — a deterministic fitted iteration: at each step, minimize `Λ_ν(h) − (1−γ)Ê_0{h(X)} − γ · (Σ_i ω̂^(k)(X_i) h(X_i^+)) / (Σ_i ω̂^(k)(X_i))` over a log-ratio class `H`, then set `ω̂^(k+1) = exp(ĥ) / Σ_j exp(ĥ(X_j))`. The objective is convex in `h` for linear classes; nonlinear classes use batched SGD on the variational form of the log-partition. **Modelless**: no gradient descent through base weights, no LLM calls — the supervised learner (GBTrees / NN / linear) is the paper's instantiation, but the *decision structure* (KL projection of adjoint Bellman image via cross-entropy) is substrate-independent.

**The whole RL training pipeline** (PQN+HER, DQN, neural network Q-function fitting, behavioral cloning) is `→ riir-train` and explicitly out of scope. What is in scope is the **post-hoc occupancy-ratio estimation primitive** that runs against offline transition data at cold/warm tier, plus the **adjoint-Bellman KL contraction** as a stability theorem.

---

## 1. Paper Core Findings

### 1.1 The problem: occupancy ratios in offline RL

Offline policy evaluation (OPE) must correct the mismatch between (a) the distribution `ν` of observed transitions and (b) the discounted occupancy distribution `d^π,γ = (1−γ) Σ_t γ^t d_0 P^t_π` induced by a target policy `π`. The **discounted occupancy ratio** `ω_π,γ = d^π,γ / d_ν` converts offline-distribution averages into target-occupancy averages:

```
V^π(r) = E_{d^π,γ}{r(X)} = E_ν{ω_π,γ(X) · r(X)}
```

A single fitted ratio evaluates **any** target-occupancy functional — rewards, costs, feature moments, visitation probabilities. Existing approaches split into two camps:

- **Value-based (FQE)**: regress Bellman targets onto a Q-function class. Convergence requires **Bellman completeness** (Bellman images stay in the class) or projected-operator stability — value-function realizability alone is **not** sufficient (Amortila et al. 2020, Foster et al. 2021, Wang et al. 2021).
- **Ratio-based (DualDICE, MWL, minimax)**: enforce occupancy-balance moments over a critic class. Convergence shifts the approximation burden to **coupled ratio+critic classes** — requires critic richness, dual realizability, or adjoint Bellman completeness (Uehara et al. 2021).

### 1.2 The adjoint Bellman operator and its KL contraction (Lemma 3.1 — the core insight)

The occupancy ratio satisfies the **adjoint Bellman equation** (taking Radon–Nikodym derivatives of the Bellman equation `d^π,γ = (1−γ)d_0 + γ d^π,γ P_π`):

```
ω_π,γ = B^γ_π ω_π,γ,    where   B^γ_π ω := (1−γ)ω_0 + γ · d((ων)P_π) / dν
```

The operator `B^γ_π` cannot be evaluated pointwise (we don't know `P_π`), but its **action against critic functions** can be estimated from one-step transitions:

```
E_ν{(B^γ_π ω)(X) · f(X)} = (1−γ) E_{d_0}{f(X)} + γ E_ν{ω(X) · f(X^+)}
```

**Lemma 3.1 (KL contraction).** For any `ω, ω̃ ∈ Δ_ν` (probability densities on ν) and `γ ∈ [0, 1)`:

```
D_ν(B^γ_π ω ∥ B^γ_π ω̃)  ≤  γ · D_ν(ω ∥ ω̃)
```

*Proof sketch* (3 lines, all standard information theory):
- Joint convexity of KL: `D_KL((1−γ)d_0 + γ(ων)P_π ∥ (1−γ)d_0 + γ(ω̃ν)P_π) ≤ γ · D_KL((ων)P_π ∥ (ω̃ν)P_π)`
- Data processing inequality for the Markov kernel `P_π`: `D_KL((ων)P_π ∥ (ω̃ν)P_π) ≤ D_KL(ων ∥ ω̃ν) = D_ν(ω ∥ ω̃)`
- Combine, divide by γ.

**This is a pure information-theoretic fact.** It depends only on `P_π` being a Markov kernel — nothing about the MDP's reward structure, nothing about value-function realizability. The undiscounted case (`γ=1`) requires a one-step KL strong data processing condition (Appendix E, Condition C7) — a Doeblin minorization `P_π(·|x) ≥ ελ(·)` gives `α = 1−ε`.

### 1.3 KL projection onto a normalized exponential class (Lemma 3.2)

FORE restricts to a hypothesis class `W = {ω_h : h ∈ H}` of normalized exponentials `ω_h(x) = exp(h(x) − Λ_ν(h))`, where `Λ_ν(h) = log E_ν e^{h(X)}` is the log-partition. KL projection of `B^γ_π ω` onto `W` reduces to a **single-level supervised learning objective**:

```
argmin_{h ∈ H}  D_ν(B^γ_π ω ∥ ω_h)   =   argmin_{h ∈ H}  {  Λ_ν(h) − (1−γ) E_{d_0}{h(X)} − γ E_ν{ω(X) h(X^+)}  }
```

The RHS depends only on (i) the log-partition `Λ_ν(h)` — a softmax-normalized expectation under `ν`, (ii) an initial-state moment `E_{d_0}{h(X)}`, and (iii) a one-step successor moment `E_ν{ω(X) h(X^+)}` with `X^+ ∼ P_π(·|X)`. All three have direct sample analogues.

### 1.4 The FORE algorithm (Algorithm 1)

```
Input:   offline transitions {(S_i, A_i, S'_i)}_{i=1..n}, initial moment estimator P̂_0,
         target policy π, discount γ, log-ratio class H, iteration count K

1: Draw A^+_i ∼ π(·|S'_i); set X^+_i = (S'_i, A^+_i)
2: Initialize ω̂^(0)(x) ≡ 1
3: for k = 0 .. K−1:
4:    ĥ_{k+1} ∈ argmin_{h ∈ H}  {  log((1/n) Σ_i e^{h(X_i)})
                                  − (1−γ) P̂_0 h
                                  − γ · [(Σ_i ω̂^(k)(X_i) h(X^+_i)) / (Σ_i ω̂^(k)(X_i))]  }
5:    ω̂^(k+1)(x) ← exp(ĥ_{k+1}(x)) / ((1/n) Σ_i exp(ĥ_{k+1}(X_i)))
6: end for
Return:  ω̂^(K)
```

The objective is **convex in `h` for linear classes**. For nonlinear classes, use the variational form `Λ_ν(h) = inf_{a ∈ ℝ}{a − 1 + (1/n) Σ_i e^{h(X_i) − a}}` and batched SGD in `(θ, a)`.

### 1.5 Convergence: only realizability, no Bellman completeness (Theorems 4.1, 4.2)

Let `ε_KL := inf_{v ∈ W} ‖log ω_π,γ − log v‖_{L²(ν)}` (the log-ratio approximation error; 0 under realizability).

**Theorem 4.1 (population recursion).** Under Conditions C1–C4 (coverage, closed convex log-ratio class, log-square-integrability, bounded log class):

```
D_ν(ω^(K) ∥ ω_π,γ)  ≤  γ^K · D_ν(ω^(0) ∥ ω_π,γ)  +  C_app · (1 − γ^K)/(1 − γ) · ε²_KL
```

The first term decays geometrically; the second is **quadratic in the log-ratio approximation error of the fixed point** — *not* an inherent adjoint Bellman error of the form `sup_ω inf_ω̃ D_ν(B^γ_π ω ∥ ω̃)`. This is the structural distinction from FQE.

**Theorem 4.2 (finite-sample).** With empirical normalization, generalized KL divergence obeys (w.p. ≥ 1−δ):

```
D^gen_ν(ω̂^(K) ∥ ω_π,γ)  ≤  C_fit · ((1+γ)/2)^K · D^gen_ν(ω̂^(0) ∥ ω_π,γ)
                         +  C_fit · ε²_KL / (1 − γ)
                         +  C_fit · (r²_n,fit + log(1/δ)/n) / (1 − γ)²
```

where `r_n,fit` is a local Rademacher critical radius. For `d`-dimensional linear classes, `r²_n,fit ≲ d log(n)/n`. For Hölder/Sobolev classes of smoothness `s > d/2`, `r²_n,fit ≲ n^{−2s/(2s+d)}`.

**Horizon dependence**: approximation term scales as `ε_KL / √(1−γ)` at the value level (favorable); statistical term scales as `1/(1−γ)` (the familiar FQE value-level horizon dependence).

### 1.6 Three policy-evaluation applications (Section 5)

1. **Direct reward reweighting**: `V̂^π = (1/n) Σ_i ω_fit(X_i) · r_i`.
2. **Doubly robust** (Theorem 5.2): combine `ω_fit` with a fitted `Q̂`. Error is `C_χ · E_FORE · ‖T^π Q − Q‖_⋆` — the product of ratio error and Bellman residual. Vanishes if either `ω = ω_π` or `Q = Q^π`.
3. **FORE-weighted FQE without Bellman completeness** (Theorem 5.3) — **the headline application**. Use `ω_fit` as a fixed projection weight in fitted Q-iteration. The projected Bellman operator becomes a `√γ`-contraction in the target-occupancy norm `‖·‖_⋆`, restoring stability **without Bellman completeness of the Q-class**:

```
‖Q^(K_Q) − Q^π‖_⋆  ≤  γ^{K_Q/2} ‖Q^(0) − Q_{Q,⋆}‖_⋆
                   +  (1 − γ^{K_Q/2})/(1 − √γ) · C_χ · ε_Bell · E_FORE
                   +  1/(1 − √γ) · inf_{q ∈ Q} ‖q − Q^π‖_⋆
```

Under Bellman completeness `ε_Bell = 0`; otherwise the ratio-estimation effect is **attenuated by** `ε_Bell`.

### 1.7 Numerical validation (Section 6)

Two carefully constructed counterexamples isolate the "realizable Q but not Bellman complete" regime:

- **Baird-style finite MRP** (6 upper + 1 lower state, scalar feature). Linear FQE iteration multiplier = **2.103** (expands errors). FORE multiplier = 0.1425 (contracts). FORE-reweighted FQE multiplier = 0.801 (contracts). Tabular FQE (Bellman complete) multiplier = 0.95.
- **Linear-Gaussian** (`X ∈ ℝ²`, Gaussian target policy, quadratic value class missing the `s², sa` directions). Linear FQE multiplier = **1.2206** (expands). FORE multiplier = 0.0855 (contracts strongly). FORE-reweighted FQE multiplier = 0.6825 (contracts).

FORE stabilizes exactly where linear FQE diverges, with no change to the value class — only the projection distribution changes.

### 1.8 The backward-regression variant (Appendix F) — requires adjoint Bellman completeness

A simpler density-ratio regression variant exists (Algorithm 2: regress `ω(X)` on `X^+`, then update `ω ← (1−γ)ω_0 + γ c_π · m(X)`). Its population error **does not vanish** without adjoint Bellman completeness of the regression class — same failure mode as standard FQE. **The main KL-projected FORE is strictly preferred.**

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalents (grep-verified) |
|---|---|
| discounted occupancy ratio `ω_π,γ` | (none shipped) — closest cousin is `dirichlet.rs` (Dirichlet weights), `committed_field_blend.rs::CommittedFieldBlend::pi` (per-archetype weight vector), `manifold_bandit.rs` (Thompson sampling weights). **No density-ratio-over-distributions primitive ships.** |
| adjoint Bellman operator `B^γ_π` | (none shipped under that name) — structurally isomorphic to **DEC `codifferential`** (adjoint of `exterior_derivative` under L² inner product); also `latent_functor/reestimation.rs::ReestimationScheduler` (the "coherence < tau_reest → re-derive" loop is a fitted iteration on coherence ratios) |
| Markov kernel pushforward `(ων)P_π` | `forward_kernel` / `pushforward` — **not shipped as a named primitive**; closest is `katgpt_hla::evolve_hla` (per-NPC belief pushforward through sense kernel) and `induced_cwm` (`advance()` on induced `GameState`) |
| KL projection onto normalized exponential class | `dirichlet.rs` (cousin), `softmax` normalization appears in `product_key_memory.rs` (with explicit "deviation from sigmoid rule" comment) |
| fitted Q-iteration | `mcts.rs` (tree-based, not fitted), `crates/katgpt-core/src/cgsp/dual_pool.rs` (online routing, not fitted) — **no fitted Bellman regression ships** |
| Bellman completeness | not in codebase vocabulary; closest concept is `subspace_phase_gate` (N≥d sufficient condition for freezing, a different sufficiency gate) |
| doubly robust estimation | not shipped |
| concentrability / coverage | not shipped as a named condition; closest is `neighbor_heal.rs` (k-HLA-neighbor coverage) and `diverse_retrieval` (wedge coverage) |
| one-step transition data | **ships** — `engram/` (conditional pattern memory), `delta_mem/` (online associative memory), `EventLog` entries; `bisimulation/transition_graph.rs::TransitionGraph` (observed (s, op, s') triples) |
| stationary distribution | closest is the "dormant subspace" / leaky-integrator decay gate; no explicit stationary-distribution computation |

**The grep returns ZERO prior art for the paper's core mechanism.** The closest three cousins each solve a *different* problem:

| Closest cousin | What it solves | How FORE differs |
|---|---|---|
| Research 298 / bisimulation (Inverting Bellman) | **Inverse problem**: given frozen `Q`, recover `P` (`P_∞ = M⁺Q + (I − M⁺M)P₀`). Closed-form extraction of the *transition kernel*. | **Adjoint problem**: given offline `ν` and target `π`, recover `ω = d^π,γ / d_ν`. Fitted iteration to estimate the *density ratio*. |
| Research 308 / bisimulation quotient | **Equivalence problem**: quotient states by bisimulation (same reward + same transition distribution over classes). Paige-Tarjan O((S+E) log S). | **Reweighting problem**: weight states by their occupancy ratio (no equivalence classes; soft weights in ℝ_+). |
| latent_functor/reestimation | **Coherence problem**: when coherence < tau_reest, re-derive the latent functor's K-basis. Fitted iteration on subspace coherence. | **Density problem**: when log-ratio approximation is poor, re-fit the log-partition objective. Fitted iteration on KL divergence. Same skeleton, different projection geometry. |

### 2.2 Latent-space reframing (mandatory before verdict)

Re-cast the core mechanism as a latent-to-latent operation on each Super-GOAT factory module:

| Substrate | FORE reframing | Fusion product |
|---|---|---|
| **HLA per-NPC state** (`riir-ai/crates/riir-engine/src/hla/kernel.rs`, 8-dim valence/arousal/desperation/calm/fear + 3) | `ω_π,γ` becomes the **personality-shift ratio**: how to reweight prior HLA snapshots to estimate the current-target HLA distribution. KL contraction gives per-cycle convergence `D_KL ≤ γ^K D_KL^(0)` under successive freeze/thaw cycles. | **Personality convergence guarantee**: each freeze/thaw cycle contracts KL toward the target personality by factor γ, with NO requirement that the HLA basis is closed under the personality-evolution operator (the HLA analogue of "no Bellman completeness"). |
| **latent_functor** (`reestimation.rs`, CLR/CGSP cycle) | FORE-weighted FQE (Section 5.2) directly maps: replace the reestimation scheduler's projection weight with a fitted FORE ratio over the NPC's engram history. | **CLR convergence without Bellman completeness**: the CLR re-estimation loop's stability currently depends on `coherence > tau_reest` (an implicit completeness gate). Adding the FORE ratio as projection weight restores contraction under the weaker assumption of CLR-ratio realizability. |
| **cgsp_runtime** (curiosity, exploration, collapse recovery) | The undiscounted variant (Appendix E, Condition C7) applies to NPC cognition cycles (γ=1). The Doeblin minorization `P_π(·|x) ≥ ελ(·)` is a **mixing condition** — NPC zone attention satisfies it when zone-routing probabilities are bounded below. | **Crowd-scale curiosity with provable convergence**: each NPC's curiosity signal contracts in KL by factor α = 1−ε per cycle, without requiring the curiosity basis to be closed under zone transitions. |
| **DEC Stokes operators** (`exterior_derivative`, `codifferential`, `hodge_decompose`) | The adjoint Bellman operator is structurally the **codifferential** in the Markov-kernel cochain complex: `B^γ_π` is the L²(ν)-adjoint of the forward transition operator, just as `δ` is the L²-adjoint of `d`. The KL contraction is the information-theoretic analogue of `δ∘d` being negative-semidefinite. | **DEC-native occupancy-ratio cohomology**: the harmonic component of `hodge_decompose` on a belief cochain identifies the "irreducible" occupancy ratio that no reweighting can compress — a modelless analogue of the Kolmogorov-Sinai entropy of the target policy. |
| **NeuronShard + freeze/thaw** (`freeze.rs::MerkleFrozenEnvelope`, `consolidation.rs::ConsolidationPipeline`) | The occupancy ratio `d^π,γ / d_ν` becomes the **consolidation-density ratio**: how much should this wake-event ensemble be reweighted to match the target consolidated style. KL contraction gives consolidation convergence under successive sleep cycles. | **Consolidation convergence guarantee**: each Raven/δ-Mem sleep cycle contracts KL toward the target style by factor γ, with NO requirement that the style_weights basis is closed under the consolidation operator (the `can_freeze` gate's current "N≥d" sufficiency becomes a special case). |
| **LatCal fixed-point commitment** (`riir-chain/src/encoding/latcal*.rs`) | The occupancy ratio is a real-valued quantity that crosses the sync boundary as a **scalar projection** (sigmoid-bounded): `σ(log ω_π,γ(x))` is the per-state commitment weight. LatCal commitment of the log-ratio parameters `{θ_1, …, θ_d}` of a linear log-ratio class gives quorum-verifiable OPE. | **Verifiable OPE receipts**: the log-ratio parameters are committed via LatCal determinant audit, the per-state scalar weights cross sync raw and deterministic; the full ratio `ω_π,γ` stays local (semantic domain). |

**The DEC codifferential isomorphism is the deepest structural connection.** The adjoint Bellman operator is the L²(ν)-adjoint of the forward Markov kernel `P_π`, exactly as the codifferential `δ` is the L²-adjoint of `d`. The KL contraction `D_ν(B^γ_π ω ∥ B^γ_π ω̃) ≤ γ D_ν(ω ∥ ω̃)` is the information-theoretic shadow of the operator identity `‖δ φ‖² ≤ ‖d δ φ‖² + ‖δ² φ‖²` (the Hodge-Laplacian being negative-semidefinite on coexact forms). This means **the adjoint Bellman KL contraction is a discrete, probabilistic instance of the DEC codifferential contraction that already ships in `katgpt-dec`** — but formulated in probability-space rather than cochain-space. The fusion product is a unified "pushforward contraction" primitive that subsumes both.

### 2.3 Fusion candidates (cross-pollination with existing corpus)

The three strongest fusion products, in priority order:

#### Fusion A — FORE × CLR re-estimation scheduler (riir-ai primary target)

**Inputs**: this paper (Section 5.2 occupancy-weighted FQE) × Research 123 (`latent_functor/reestimation.rs`) × Plan 317 (feeling-brain mux scatter consolidation).

**Product**: Replace the CLR re-estimation scheduler's `coherence > tau_reest` trigger with a fitted FORE ratio over the NPC's engram table. The resulting CLR cycle contracts in KL by factor γ per re-estimation round **without requiring the latent-functor K-basis to be closed under the re-estimation operator**. Concretely: at each re-estimation tick, fit `ω_fit^(k)` to the engram's wake-event distribution vs. the target personality's occupancy; use `ω_fit` as the projection weight in the K-basis re-derivation.

**Why this matters**: the current CLR re-estimation scheduler silently fails when the NPC's recent experience distribution drifts outside the K-basis's span — the "coherence < tau_reest" gate fires, triggering a costly re-derivation. The FORE ratio projects back into a stable norm where the re-derivation contracts. This is the **occupancy-weighted fitted-CLR** primitive.

**Validation**: head-to-head on the existing CLR benchmark — baseline (coherence gate) vs. FORE-weighted (ratio-projected), measuring (a) number of re-derivation triggers per 1000 ticks, (b) KL divergence from target personality at tick 1000, (c) wall-clock overhead. This is a **PoC-required** fusion per §3.6 (quality-parity claim needs empirical settlement in `riir-poc`).

#### Fusion B — Adjoint Bellman KL contraction × freeze/thaw runtime (riir-neuron-db target)

**Inputs**: Lemma 3.1 (KL contraction) × `riir-neuron-db/src/freeze.rs::MerkleFrozenEnvelope` × Research 158 (Committed Personality Runtime).

**Product**: A per-cycle convergence guarantee for freeze/thaw personality drift. Each freeze/thaw cycle contracts KL toward the target personality by factor γ, **provided** the personality-evolution operator is a Markov kernel (which it is, by construction of the archetype blend). The guarantee is: after K cycles, `D_KL(personality_K ∥ target) ≤ γ^K · D_KL(personality_0 ∥ target) + C_app · ε²_KL`, where `ε_KL` is the log-personality-ratio approximation error of the archetype blend class.

**Why this matters**: the current freeze/thaw runtime ships a Lean-proven *reader* invariant (readers never see torn snapshots, Issue 348 T2) but no *convergence* guarantee. Adding the KL-contraction theorem gives a per-cycle progress bound — a modelless, forever-verified, refactor-immune guarantee. This is a candidate for the next Lean 4 theorem in `RiirAiProof/` (alongside the existing HLA boundedness and freeze/thaw reader invariant theorems).

**Validation**: PoC not required for the theorem itself (it's a direct application of Lemma 3.1). PoC IS required for the runtime wiring (does the live freeze/thaw cycle actually achieve the γ-contraction empirically, given float precision and archetype-blend approximation error?).

#### Fusion C — FORE × bisimulation quotient (katgpt-rs primary target)

**Inputs**: FORE × Research 308 / `katgpt-core/src/bisimulation/`.

**Product**: An **occupancy-ratio-based state equivalence** that is strictly weaker than bisimulation but sufficient for off-policy evaluation. Two states are "OPE-equivalent" iff they have the same fitted FORE ratio `ω_fit(s, ·)` under the target policy. This quotient is coarser than bisimulation (which requires same reward + same transition distribution) but finer than reward-only equivalence.

**Why this matters**: bisimulation's Paige-Tarjan refinement is O((S+E) log S) — expensive for large state spaces. FORE-ratio equivalence can be computed incrementally as a byproduct of the fitted iteration, at no extra cost. For game maps with `|S|` in the 10⁵–10⁶ range, this is the difference between feasible and infeasible state-abstraction for OPE.

**Validation**: head-to-head on a toy MDP — bisimulation quotient size vs. FORE-ratio quotient size vs. ground-truth OPE error. This is the natural GOAT-gate benchmark.

### 2.4 What does NOT translate (honest negative results)

- **The full RL training pipeline** (PQN+HER, DQN, neural fitted Q-iteration with backprop) → `riir-train`. The modelless unblock protocol (§3.5) does not apply because FORE is already modelless — there is nothing to unblock.
- **Behavioral policy estimation** (`ν`-from-data). FORE assumes `ν` is given. For NPCs, `ν` is the empirical distribution of engram entries — a non-trivial estimation problem in its own right, but one we already solve via `delta_mem` and `engram`. No new primitive needed; just plumbing.
- **The backward-regression variant** (Appendix F). It requires adjoint Bellman completeness, defeating the whole point. Skip it.
- **Continuous high-dimensional state spaces** (the paper's limitation, §7). FORE's log-ratio class must be rich enough to approximate `ω_π,γ`, which is hard when `|S|` is large. For our use cases (NPC personalities in 8–64-dim HLA/shard space, not raw pixel space), this is feasible — but it is the binding constraint on practical adoption.

---

## 3. Verdict

### 3.1 Tier

**GOAT** — provable gain (convergence without Bellman completeness) over existing approaches, but not a new class of capability. The primitive is genuinely novel (zero prior art), modelless (no GD through base weights), and has three concrete fusion targets. The selling point is a **guarantee multiplier** on existing pillars (CLR re-estimation stability, freeze/thaw convergence, bisimulation-based abstraction), not a new pillar.

**One-line reasoning**: FORE's adjoint-Bellman KL contraction is a transferable information-theoretic primitive that strengthens convergence guarantees for three shipped systems (CLR, freeze/thaw, bisimulation) — but it does not by itself enable any capability those systems cannot already approximate, just slower or with weaker guarantees.

### 3.2 Novelty gate (§1.5 Q1–Q4)

| Q | Answer | Evidence |
|---|---|---|
| **Q1 No prior art?** | **YES** | Grep across all 5 repos for `occupancy\|adjoint\|Bellman\|FQE\|FORE\|relative.entropy.contraction\|doubly.robust\|importance.ratio` returns ZERO substantive hits. Closest cousins (Research 298 inverting Bellman, Research 308 bisimulation, latent_functor/reestimation) solve different problems — verified by reading their TL;DRs. |
| **Q2 New class of behavior?** | **NO (partial)** | The codebase has no OPE / off-policy correction / occupancy-reweighting primitive — but the *applications* (CLR stabilization, freeze/thaw convergence, state abstraction) are capability enhancements to systems that already ship, not new capabilities. |
| **Q3 Product selling point?** | **NO (partial)** | Cannot finish "Our NPCs do X no competitor can" with FORE alone — the closest is "NPC personality convergence has a provable KL contraction guarantee", which is a *guarantee* about an existing capability, not a *new* capability. |
| **Q4 Force multiplier?** | **YES** | Connects to ≥2 pillars: Reasoning Pack (CLR/CGSP re-estimation), Self-Learn NPCs (freeze/thaw convergence), Pillar 1 Fourier Spatial AI (DEC codifferential isomorphism), Pillar 2 riir-neuron-db (consolidation convergence). |

**Q2 + Q3 fail → not Super-GOAT.** Proceed to GOAT.

### 3.3 MOAT gate per domain (§1.6)

| Domain | Verdict | Reasoning |
|---|---|---|
| **katgpt-rs** (public engine) | **In scope — ship behind feature flag** | FORE is a generic modelless primitive (occupancy-ratio estimation). No game IP, no chain IP, no shard IP. The KL contraction lemma is a pure information-theoretic fact. Open primitive: `katgpt-rs/crates/katgpt-core/src/occupancy/`. |
| **riir-ai** (private runtime) | **In scope — fusion target** (Fusion A, Fusion B) | CLR re-estimation stabilization and freeze/thaw convergence guarantee are runtime IP. Cross-reference from the katgpt-rs note; full private guide deferred to post-PoC. |
| **riir-chain** (private chain) | **Optional** (Fusion C-extension) | LatCal commitment of log-ratio parameters for verifiable OPE receipts is a chain-side bridge, but not a headline moat amplifier. Defer. |
| **riir-neuron-db** (private shards) | **In scope — fusion target** (Fusion B) | Consolidation convergence guarantee via KL contraction is a shard-side moat amplifier. Cross-reference from the katgpt-rs note. |
| **riir-train** (private training) | **Out of scope** | The paper's training pipeline (PQN+HER etc.) is training-method research. The modelless FORE primitive itself stays here. |

### 3.4 Softmax-vs-sigmoid tension (honest caveat)

The AGENTS.md global rule says "Use sigmoid not softmax". FORE's math requires a **normalized exponential class** `ω_h(x) = exp(h(x) − Λ_ν(h))`, which is structurally a softmax over the offline sample. This is in tension with the sigmoid rule.

**Resolution**: the rule's intent is "don't use softmax where sigmoid-on-direction-vectors suffices for projections onto learned directions" (the semantic-domain rule). FORE's use of softmax is **not a projection onto learned directions** — it is a **density-ratio normalization** over a discrete sample, which is the correct mathematical operation (the log-partition `Λ_ν(h)` is the cumulant-generating function of the empirical distribution). The sigmoid rule does not apply to density-ratio normalization; it applies to direction-vector projections. This is the same carve-out that `product_key_memory.rs` already documents ("Deviation from the global sigmoid rule — these are convex-combination coefficients over the k²-restricted candidate set, not a probability/UQ claim").

**Action**: document this carve-out in the FORE primitive's module docs, explicitly citing the PKM precedent.

### 3.5 "Report the Floor" gate (UQ-bearing primitive check)

FORE itself produces a density ratio, not a probability interval or coverage guarantee — it is **borderline UQ-bearing**. The downstream applications (doubly robust value estimation, occupancy-weighted FQE) DO produce value estimates with error bounds, which are UQ-bearing.

**Decision**: the GOAT gate for the FORE primitive itself does NOT require the conformal-naive floor (FORE is a ratio estimator, not a forecaster). The GOAT gate for any downstream value-estimation application (Fusion A's CLR stabilization, if it produces a value estimate) DOES require the floor per the §1.6 rule. Tracked as a follow-up.

### 3.6 Defend-wrong PoC requirement (§3.6)

This verdict does **not** claim "already ships" or "parity" with the paper — it claims **novelty** (zero prior art). Therefore the §3.6 PoC requirement does not apply to the verdict itself.

However, the **fusion claims** (Fusion A: CLR stabilization; Fusion B: freeze/thaw convergence guarantee) DO make quality-parity claims ("FORE-weighted CLR converges as well as or better than coherence-gated CLR"). Per §3.6, these claims require a head-to-head PoC in `riir-poc` before they become feature flags. The PoC protocol:

1. Three competitors on a controlled toy MDP (Baird-style or Linear-Gaussian from the paper's §6):
   - (i) the paper's FORE algorithm (modelless port)
   - (ii) a frozen/no-adaptation baseline (use the prior personality's ratio)
   - (iii) the shipped runtime analog (CLR re-estimation with coherence gate, for Fusion A)
2. Print a verdict table: value RMSE, KL divergence from target, wall-clock, alloc count.
3. If the PoC refutes the quality claim, **do not silently revise the verdict** — record raw numbers in this note's §"PoC Addendum" and downgrade the fusion to a tracked follow-up issue.

---

## 4. Open primitive design (katgpt-rs target)

**Module**: `katgpt-rs/crates/katgpt-core/src/occupancy/` (new module, behind `occupancy_ratio` feature flag)

**Public API sketch** (subject to plan refinement):

```rust
pub struct OccupancyRatioEstimator<H: LogRatioClass> {
    log_ratio_class: H,
    gamma: f32,
    k_iterations: usize,
}

pub trait LogRatioClass {
    type Params;
    fn evaluate(&self, params: &Self::Params, x: &[f32]) -> f32;  // h(x)
    fn fit_kl_projection(
        &self,
        transitions: &TransitionBatch,   // (X_i, X^+_i) pairs
        initial_moments: &InitialMoments, // P̂_0 h estimator
        current_ratio: &[f32],            // ω̂^(k)(X_i)
        gamma: f32,
    ) -> Self::Params;
}

pub struct TransitionBatch<'a> {
    pub states: &'a [f32],         // [n * state_dim] flattened
    pub successors: &'a [f32],     // [n * state_dim] flattened (X^+)
    pub rewards: Option<&'a [f32]>,
    pub n: usize,
    pub state_dim: usize,
}

impl<H: LogRatioClass> OccupancyRatioEstimator<H> {
    pub fn fit(
        &self,
        transitions: &TransitionBatch,
        initial_moments: &InitialMoments,
    ) -> Vec<f32>;  // ω_fit(X_i) — the fitted occupancy ratio at each transition

    pub fn value_estimate(&self, ratio: &[f32], rewards: &[f32]) -> f32;  // V̂^π = mean(ω · r)
}

// ── The theorem-statement module (no impl, just the contract) ──
pub mod kl_contraction {
    //! Lemma 3.1 (KL contraction of the adjoint Bellman operator).
    //!
    //! For any Markov kernel P_π, discount γ ∈ [0, 1), and any two probability
    //! densities ω, ω̃ ∈ Δ_ν:
    //!
    //!     D_ν(B^γ_π ω ∥ B^γ_π ω̃)  ≤  γ · D_ν(ω ∥ ω̃)
    //!
    //! where B^γ_π ω = (1−γ)ω_0 + γ · d((ων)P_π)/dν.
    //!
    //! This is a pure information-theoretic fact (joint convexity of KL + DPI).
    //! Candidate for Lean 4 formalization in RiirAiProof/Runtime/.
}
```

**GOAT gate** (per AGENTS.md):

| Gate | Requirement |
|---|---|
| **G1 correctness** | Implement the Baird-style MRP from §6.1 exactly. FORE must converge to the known occupancy ratio `ω_π,γ(upper) = 0.2211, ω_π,γ(lower) = 15.7987` to within 1% relative error after K=20 iterations on n=10000 transitions. |
| **G2 perf** | FORE fit on n=10000, state_dim=8, K=20 must complete in < 100 ms on Apple Silicon (the cold-tier budget). Linear log-ratio class only for the perf gate; nonlinear classes are out of scope for promotion. |
| **G3 no-regression** | `cargo check --all-features` and `cargo test -p katgpt-core --lib` must pass unchanged. Feature is opt-in (`occupancy_ratio = []`). |
| **G4 alloc-free** | The inner KL-projection loop must be zero-allocation in steady state (pre-allocated scratch buffers via `Vec::with_capacity` + `clear()` reuse, per the optimization guidelines). The outer `fit()` may allocate the output `Vec<f32>`. |
| **G5 modelless-ness** | No gradient descent through any base weight. The `LogRatioClass::fit_kl_projection` trait method may use gradient descent on its own *parameters* (that's the supervised learner), but must not touch any `NeuronShard`, `LoRAWeightVersion`, or `SenseModule` weights. |
| **G6 floor** (UQ-bearing) | N/A for the ratio estimator itself. If a downstream value-estimation application is added, it MUST benchmark against `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` per the §1.6 rule. |

**Promotion path**: ship behind `occupancy_ratio` (opt-in). If G1–G5 pass, promote to default-on in `katgpt-core` only if a downstream consumer (Fusion A CLR stabilization) demonstrates the gain in `riir-poc`. Otherwise stays opt-in as an engine primitive that consumers can opt into.

---

## 5. Honest caveats — READ BEFORE PLANNING

1. **The paper is theoretical.** The gain is in convergence *guarantees* (no Bellman completeness needed), not in new *functionality*. The numerical experiments (§6) are carefully constructed counterexamples, not real-world benchmarks. The primitive's value depends entirely on whether the fusion targets (CLR, freeze/thaw, bisimulation) actually hit the "realizable but not Bellman complete" regime in practice. **This is unknown without a PoC.**

2. **The softmax-vs-sigmoid tension is real.** FORE's normalized exponential class is structurally softmax. The §3.4 carve-out (density-ratio normalization ≠ direction-vector projection) is principled but should be reviewed by the team before committing.

3. **Offline transition data is the binding input.** FORE requires one-step target-policy successor pairs `(X_i, X^+_i)` where `X^+_i ∼ P_π(·|X_i)`. For NPCs, this means the engram table must record not just (state, action, reward, next-state) but also (next-state, target-policy-action). This is **additional instrumentation** on the engram/delta_mem subsystems — not a free lunch.

4. **The DEC codifferential isomorphism (§2.2) is an architectural insight, not a proven theorem.** The claim "adjoint Bellman ≡ codifferential in the Markov-kernel cochain complex" is structurally compelling but has not been formalized. It may yield a Lean 4 theorem eventually, but treat it as a research hypothesis, not a fact, until proven.

5. **Continuous high-dimensional state spaces are the paper's acknowledged limitation** (§7). FORE's log-ratio class must approximate `ω_π,γ` in L²(ν). For 8-dim HLA or 64-dim style_weights, this is feasible. For raw pixel state or 1000+-d transformer activations, it is not. Do not over-promote.

6. **The Fusion A claim (CLR stabilization) is the strongest commercial angle but also the riskiest.** If the PoC shows the coherence gate is already good enough in practice (because NPC engram distributions don't drift outside the K-basis span in normal operation), the fusion is a no-op. The PoC must include a stress-test regime where the drift is deliberate.

7. **The Fusion B claim (freeze/thaw convergence guarantee) is the cleanest theoretical angle but depends on the personality-evolution operator being a Markov kernel.** This is true by construction for archetype blends (convex combinations of frozen archetype directions preserve the Markov property) but may not hold for arbitrary personality-evolution operators (e.g., LLM-steered personality updates). State the assumption explicitly in any Lean 4 theorem.

8. **No claim of "already ships" or "parity" is made.** The verdict is GOAT on the basis of **novelty** (zero prior art) + **three concrete fusion targets**, not on the basis of architectural coverage of an existing primitive. The §3.6 PoC requirement applies to the fusion claims, not to the verdict itself.

---

## 6. Cross-references

- **Research 298** (Inverting Bellman) — the inverse problem (`Q → P`). FORE is the adjoint problem (`d_ν → d^π,γ`). Both use the same `(I − γP)^{−1}` resolvent algebra; both could share a `bellman_resolvent` substrate module.
- **Research 308** (Bisimulation operator) — the closest shipped cousin in `katgpt-core/src/bisimulation/`. Fusion C target.
- **Research 219** (DEC Topological Neural Operators) — the parent note for the DEC substrate. The adjoint Bellman ≡ codifferential isomorphism (§2.2) extends the DEC vocabulary crosswalk.
- **Research 123** (riir-ai Latent Functor Runtime) — Fusion A primary target.
- **Research 158** (riir-ai Committed Personality) — Fusion B primary target.
- **Research 165** (riir-ai Per-NPC Conformal UQ) — the "Report the Floor" gate for any downstream UQ-bearing application.
- **katgpt-rs/.research/271** (MIT 6S184 Diffusion/Flow Vocabulary Crosswalk) — the "fitted iteration" family.
- **katgpt-rs/.research/322** (Conformal Seasonal Pools) — the conformal-naive floor specification.

## TL;DR

**Verdict: GOAT.** The adjoint Bellman KL contraction (Lemma 3.1) is a genuinely novel information-theoretic primitive with **zero prior art** across the 5-repo quintet. The FORE algorithm (Algorithm 1) is a modelless fitted-iteration estimator (no GD through base weights, no LLM calls — the supervised learner is the paper's instantiation, the KL projection is the substrate-independent technique). Three concrete fusion targets: (A) CLR re-estimation stabilization without Bellman completeness, (B) freeze/thaw personality convergence guarantee (candidate Lean 4 theorem), (C) FORE-ratio-based state equivalence for cheaper-than-bisimulation abstraction. **Not Super-GOAT**: the primitive is a guarantee multiplier on existing pillars, not a new capability class (Q2/Q3 fail the novelty gate). **Softmax-vs-sigmoid tension** is real but principled (density-ratio normalization ≠ direction-vector projection; same carve-out as PKM). **PoC required** for the fusion quality claims (Fusion A CLR stabilization, Fusion B freeze/thaw convergence) per §3.6 before any feature flag promotes. **Open primitive**: `katgpt-rs/crates/katgpt-core/src/occupancy/` behind `occupancy_ratio` feature flag. **No private guide yet** — deferred to post-PoC when the fusion selling point is empirically validated.
