# Research 444: Invariant Manifold of Inductive Reasoning (IMIR)

> **Source:** [Invariant Learning Dynamics of Transformers in Inductive Reasoning Tasks](https://arxiv.org/pdf/2607.11875) — Musat, Pimentel, Zucchet, Hofmann (ETH Zürich / Stanford), arXiv:2607.11875v2 [cs.LG], 14 Jul 2026
> **Date:** 2026-07-16
> **Status:** Done
> **Related Research:** 355 (LieFlow — group-orbit invariance probe, the closest shipped cousin), 314 (f-divergence group invariance — theoretical cousin, deferred), 397 (MAG — direction mining, the probe replacement), 409 (MANCE — local tangent concept erasure), 408 (TILR — alignment-gated subspace projection), 406 (Spectral Rewiring — weight-delta SVD projection), 271 (MIT 6S184 diffusion/flow textbook vocabulary crosswalk — M as DEC derivative)
> **Related Plans:** 301 (subspace_phase_gate — the SVD primitive), 418 (MAG), 425 (TILR), 423 (spectral_rewire), 426 (manifold_erasure)
> **Classification:** Public

---

## TL;DR

The paper proves that transformers trained on a unified class of "block-list" inductive tasks (in-context n-grams, k-hop induction, indirect object identification, conditional retrieval, permutation inversion — all subsumed by `[x₁¹…x₁ᵏ¹] … [xₙ¹…xₙᵏⁿ] → xₙᵏⁿ⁺¹`) have their key-query and output weights **confined to a low-dimensional Invariant Manifold of Inductive Reasoning (IMIR)**. The IMIR is spanned by interpretable basis matrices built from three ingredients: **selection matrices** (token `T = {I⁽ᵗ⁾, C}` for identity / association; position `P = {I⁽ᵖ⁾, M, M², …, Mᵏ⁻¹}` for identity / shifts), **writing bases** `V₁:ℓ` (recursive output-value composites through layers), and **action bases** (their inverses). Gradient descent never leaves the IMIR (Theorem 1). The paper uses this to (i) prove IWL starves ICL of gradient by a data-dependent factor (Theorem 2), (ii) prove burstiness amplifies the ICL gradient ∝ b (Theorem 3), (iii) auto-discover circuits via greedy backward elimination on IMIR coordinates (§5, Algorithm 1), and (iv) exhibit the (α, β, γ, δ) IMIR sub-manifold as an explicit lottery ticket.

**Why it matters here:** Most of the paper is **training-dynamics theory** (gradient confinement, circuit competition, lottery ticket) → that redirects to riir-train, full stop. The modelless residue is the §5 circuit-detection algorithm + the §3.2 basis construction: given trained weights and a data-symmetry group, **project onto the commutant basis and greedily ablate irrelevant directions**. That residue is a *refinement* of two shipped primitives — `group_invariance_probe` (Plan 355, sampling-and-scoring on a hypothesis group) and `subspace_phase_gate` (Plan 301, SVD-based subspace identification) — not a new capability class.

**Distilled for katgpt-rs (modelless, inference-time):**

The single transferable insight is the **commutant construction**: given a group `U ⊆ O(d)` acting on representations (token automorphisms, position shifts), the admissible weight operators are exactly the **commutant** `C(U) = {W : WU = UW ∀ U ∈ U}`. The paper's `T = {I⁽ᵗ⁾, C}` for binary associations and `P = {I⁽ᵖ⁾, M, …, Mᵏ⁻¹}` for k-step position shifts are concrete instances — `{I, C}` is the commutant of the association-preserving permutation group on token embeddings (after centering); `{I, M, …, Mᵏ⁻¹}` is the commutant of the offset-shift group on sinusoidal position embeddings. This is a more principled alternative to `group_invariance_probe`'s current sample-then-score loop: instead of probing random `g ∈ G`, compute the commutant basis directly and project onto it.

No training, no gradients. The basis is determined by the data symmetry, not learned.

---

## 1. Paper Core Findings

### 1.1 The block-list task class (§2)

A unifying framework: the next token `xₙᵏⁿ⁺¹` depends on a list of `n` blocks of ≤ `k` tokens each. Subsumes:
- **In-context associative recall:** `[ab] [cd] … [a] → b` (k=2)
- **k-hop induction:** chained bigram lookups
- **In-context n-grams:** `[x¹…xˡ]` blocks of length ℓ
- **Indirect object identification, conditional retrieval, permutation inversion, in-context language learning, reasoning with fragments of natural language** — all fit the structure

Two data symmetries pin the analysis:
- **Token symmetry:** association-preserving permutations π of common tokens (the automorphism group `Aut(V, R)` of the relation R); rare tokens are lexinvariant (re-sampled per sequence)
- **Position symmetry:** independent random block offsets ϕᵢ,ᵦ ⇒ shift-invariance within a block

### 1.2 Selection / writing / action bases (§3.2) — the modelless residue

Three nested bases built from the data symmetries:

- **Selection matrices** `T = {I⁽ᵗ⁾, C}`, `P = {I⁽ᵖ⁾, M, M², …, Mᵏ⁻¹}` where `I⁽ᵗ⁾`/`I⁽ᵖ⁾` are token/position identity projectors, `C` maps each token to its associated partner (`Ceₐ = eᵦ` for {a,b} ∈ R), and `M` shifts positions back one step (`Mpᵢ = pᵢ₋₁`; `Mʲpᵢ = pᵢ₋ⱼ`).
- **Writing basis** at layer ℓ: `V_ℓ = {I} ∪ {W_P^(ℓ,h) W_V^(ℓ,h) : h ∈ [H]}` — the residual identity plus each head's output-value map.
- **Action basis** for layers 1..ℓ: `V₁:ℓ = V_ℓ V_ℓ₋₁ … V₁`, with inverse `V_ℓ:₁ = {V⁺ : V ∈ V₁:ℓ}` (Moore-Penrose pseudo-inverse). Contains `(H+1)^ℓ` matrices.
- **Basis weights** at layer ℓ: `W_ℓ = V₁:ℓ₋₁ (T ∪ P) V_ℓ₋₁:₁`. Each basis weight composes three operations: read from a prior head's write space, apply a token/position action, write into another head's write space.
- **Basis output weights:** `W̄ = T V_L:₁`.

### 1.3 Theorem 1 (Gradient Confinement) + Corollary 1 (Invariant Manifold)

Under six assumptions (R-invariance of common tokens, independent positional offsets, lexinvariance of rare tokens, orthonormal centered embeddings, merged key-query parameterization, fixed orthogonal output-value maps), the population gradient of a transformer on a block-list task is **confined to `S = span(W_ℓ) × span(W̄)`**:

```
W ∈ S  ⟹  E[∇_W L(W)] ∈ S
```

The proof's engine: the two data symmetries (token permutation, position offset shift) lift to orthogonal rotations `E⁽ᵗ⁾, E⁽ᵖ⁾` on the embedding subspaces; attention scores are invariant to both; the population gradient must commute with both; the commutant of the permutation group on `U⁽ᵗ⁾` (after centering) is exactly `span{I⁽ᵗ⁾, C}`, and the commutant of the offset-shift group on `U⁽ᵖ⁾` is exactly `span{I⁽ᵖ⁾, M, …, Mᵏ⁻¹}`.

**This is the load-bearing structural insight for our purposes:** the IMIR's coordinates are not learned — they are the **commutant of the data-symmetry group**.

### 1.4 Theorems 2 & 3 — circuit competition (TRAINING THEORY → riir-train)

- **Theorem 2:** IWL (the `δ` direction, `W̄ ∝ C`) emerges first and suppresses the ICL gradient by a data-dependent factor `f_r + (1−f_r)·f_v` (rare-token frequency + within-class variability). ICL is starved exactly on the data IWL solves.
- **Theorem 3:** In the distractor-dominated regime, the ICL gradient is proportional to burstiness `b`.

These are statements about gradient descent dynamics — they describe what training does, not what inference does. → riir-train.

### 1.5 §5 Automated circuit detection (Algorithm 1) — INFERENCE-TIME

Given a trained model, project its weights onto the IMIR coordinates, then **greedy backward elimination**: repeatedly drop the single direction whose ablation increases loss the least, until the cheapest ablation would push loss above budget `τ`. Surviving directions = essential circuits.

Two auto-discovered 5-layer circuits on 2-hop induction: 4 canonical induction units (previous-token head + two item-match heads + readout) + 3-4 "aiding" units (sub-manifold helpers that lift accuracy 0.594 → 0.922). The aiding units would be invisible to attention-head-granularity circuit discovery.

### 1.6 The (α, β, γ, δ) lottery ticket

The 4-parameter IMIR sub-manifold for a 2-layer single-head model:
- α: layer-1 attention to previous position (`W^(1) ∝ M`)
- β: layer-2 token-identity match through V⁽¹⁾ write space
- γ: output readout through V⁽²⁾ write space
- δ: in-weights association (`W̄ ∝ C`)

Constrained training on these 4 coordinates closely matches unconstrained training — the IMIR sub-manifold IS a winning lottery ticket. → riir-train.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | ≥2 codebase equivalents |
|---|---|
| invariant manifold / low-dimensional subspace | **subspace**, **invariant subspace** (Plan 425 TILR `U_r`), **subspace_phase_gate** (Plan 301), **group_invariance_probe** (Plan 355, the closest cousin) |
| selection matrix (token `I⁽ᵗ⁾`, `C`) | **direction vector** (HLA `EmotionDirections`), **MAG-mined direction** (Plan 418), **ConstraintPruner** (identity / association projections) |
| position shift `M, M², …, Mᵏ⁻¹` | **circulant**, **shift operator**, **DEC exterior derivative** `d` on a 1D cell complex (the IMIR's `M` is exactly `d` on the position axis — see §2.4) |
| action basis `V₁:ℓ`, inverse `V_ℓ:₁` | **latent functor stage application** (`latent_functor/` — recursive composition through layers), **freeze/thaw delta** (`MerkleFrozenEnvelope` between two snapshots) |
| commutant `C(U) = {W : WU = UW}` | **invariant operator**, **equivariant operator** (R314 f-divergence group invariance — same commutant structure for transformation models), **gauge-invariant compose** (Plan 270, the R+-scaling instance) |
| greedy backward elimination (Algorithm 1) | **ConstraintPruner** (`is_valid` + `propagate`), **pipeline_pruner** (stage-wise pruning), **bandit arm elimination** (`BanditPruner`) |
| circuit detection / automated interpretability | **CNA** (Plan 087 — contrastive neuron attribution), **causal_head_importance** (Plan 362 — Hydra head importance), **Jacobian Lens** (R388 — SVD-based concept readout) |
| token automorphism group `Aut(V, R)` | **group action** (`GroupAction` trait in `group_invariance_probe.rs`), **hypothesis Lie group** (LieFlow R355) |
| association matrix `C` | **direction vector**, **linear probe** (the in-weights "memorized association" direction = an `EmotionDirections`-style frozen vector) |
| in-context learning (ICL) vs in-weights learning (IWL) | **runtime lookup** vs **frozen committed state** (the ICL/IWL split is the runtime-vs-freeze split — `KarcShard`/`ArchetypeBlendShard` committed state vs runtime `latent_functor` re-estimation) |
| block-list task class | **block-structured context**, **windowed retrieval** (closest analog: KV-cache windowing, `tiered_kv/`) |

### 2.2 Fusion grep results (both layers, all 5 repos, both vocabularies)

**Paper-vocabulary grep** (`invariant_manifold|induction_head|block_list|circuit_detect|greedy.*elim|backward.*elim|commutant|automorphism|action_basis|selection_matrix|writing_basis|ICL|IWL`) returned **ZERO** hits across `.research/` notes and **ZERO** hits in shipped code. The IMIR-specific vocabulary is genuinely absent — no note frames a transformer-training-dynamics result in this vocabulary, and no shipped primitive uses the commutant construction.

**Codebase-vocabulary grep** surfaced **5 close cousins**:

| Cousin | Mechanism | Overlap with IMIR residue | Difference |
|---|---|---|---|
| **`group_invariance_probe.rs`** (Plan 355, LieFlow R355) | Sample `g ∈ G`, score `σ(β·(1−d(q, g·q)))`, classify subgroup by score concentration | Same goal (find symmetry-invariant operators); same data-symmetry framing | **Sample-then-score** (Monte Carlo on the group); IMIR computes the **commutant basis directly** (closed form for permutation + shift groups). MC is O(n_samples); commutant is O(d³) one-shot. |
| **`subspace_phase_gate.rs`** (Plan 301) | SVD-based participation ratio, numerical rank, phase-transition gate | Same "is the data confined to a low-dim subspace?" question | Subspace of `ℝᵈ` (linear-algebraic); IMIR is subspace **defined by a group action** (the commutant, not the SVD). Orthogonal — a dataset can be group-invariant without being low-rank. |
| **`manifold_erasure.rs`** (Plan 426, MANCE R409) + **`tilr.rs`** (Plan 425, R408) + **`spectral_rewire.rs`** (Plan 423, R406) | Subspace projection + γ-gate + trust region / on-manifold decomposition | Same "project onto interpretable basis" pattern | All operate on **latent state** (TILR/MANCE) or **weight deltas** (spectral_rewire); IMIR operates on **full weight matrices** with a symmetry-defined basis. All use SVD-discovered subspaces; IMIR uses group-commutant subspaces. |
| **`causal_head_importance/`** (Plan 362) + **`crates/katgpt-pruners/src/cna.rs`** (Plan 087) | Head-importance scoring + contrastive neuron attribution | Same "which circuit matters?" interpretability goal | Head/neuron granularity; IMIR is sub-head granularity (weight-structure basis directions). Different abstraction level. |
| **`gauge_invariant_bridge.rs`** + Plan 270 (LoRA-Muon) | R+-scaling gauge invariance for LoRA composition | Same "factor out the gauge freedom" pattern (commutant of R+) | Narrower gauge (1-parameter R+); IMIR's commutant is for the full permutation + shift group. Plan 270 is the R+-instance of the general commutant construction. |

**Conclusion of the fusion grep:** every component of the IMIR residue (symmetry-defined basis + projection + greedy ablation) has a shipped cousin, but the specific integration — *automorphism-group-derived token/position selection bases + recursive action bases through layers + greedy backward elimination on the resulting interpretable coordinates* — is **not shipped**. The gap is the **commutant construction**: given a group action `U`, compute `C(U) = {W : WU = UW}` directly rather than sample-and-score.

### 2.3 Latent-space reframing (mandatory per workflow §1 step 3)

The paper operates on attention-only transformer weights (d ∈ [128, 1024]). Re-cast on the codebase's latent-state kernels:

**(a) HLA per-NPC latent state (8-dim, `riir-ai/crates/riir-engine/src/hla/`)**

The IMIR's token-selection basis `T = {I⁽ᵗ⁾, C}` is structurally identical to the HLA affect-direction ecosystem: `I⁽ᵗ⁾` is the identity (no projection); `C` is a learned association matrix (`Ceₐ = eᵦ` for associated pairs) — **exactly an `EmotionDirections`-style frozen direction vector** that encodes a "this emotion pairs with that emotion" association. The 8-dim HLA space has a natural automorphism group: the named-axis permutations that preserve the valence/arousal/desperation/calm/fear roles. The commutant of that group constrains which operators the runtime may apply — exactly the no-harm refinement that TILR (Plan 425) achieves via SVD, but derived from the *semantic* symmetry rather than from data.

**(b) `latent_functor/` operations (`riir-ai/crates/riir-engine/src/latent_functor/`)**

The IMIR's writing/action bases `V₁:ℓ` and `V_ℓ:₁` are recursive composites through layers — structurally identical to the latent functor's "functor application" pattern. A functor entry's `W^(ℓ)` constrained to `V₁:ℓ₋₁ (T ∪ P) V_ℓ₋₁:₁` is exactly the IMIR constraint specialized to the functor's stage-wise composition. **This is the bridge to §2.4** — the position-shift `M` on the functor's tick axis IS the DEC exterior derivative.

**(c) `cgsp_runtime/` curiosity signals**

ICL vs IWL competition (Theorems 2-3) is the runtime-vs-frozen split: ICL = runtime lookup (curiosity-driven exploration); IWL = committed shard state. The gradient-starvation theorem says IWL wins first and suppresses ICL — at runtime, this means a frozen `KarcShard`/`ArchetypeBlendShard` snapshot dominates until curiosity explicitly overrides. This is the `latent_functor/reestimation.rs` coherence-trigger pattern (already shipped), just named in transformer-training vocabulary. → riir-train for the training-side claim; the runtime side already ships.

**(d) `NeuronShard` style_weights / freeze envelope (`riir-neuron-db/src/`)**

A frozen shard's `style_weights[64]` is a vector, not a matrix; the IMIR's matrix-basis projection doesn't apply directly. BUT the **commutant insight** does: given the BLAKE3-committed basis of a shard, the admissible update operators are constrained to commute with the basis permutations. This is a **freeze-envelope invariant** — a frozen shard can only be modified by operators in the commutant of its committed basis. This is a **new constraint on the `MerkleFrozenEnvelope` update protocol** (currently no such constraint). Speculative; no plan this session.

**(e) LatCal fixed-point commitment (`riir-chain/src/encoding/`)**

The IMIR coordinates (α, β, γ, δ for the 2-layer model; more generally the commutant-basis coefficients) are deterministic scalars once the basis is fixed. Committing them via LatCal would be a sync-boundary bridge: latent weights → commutant projection → scalar coefficients → LatCal fixed-point → BLAKE3. The latent payload (the basis itself) stays local; only the scalar coefficients cross sync. Speculative; no plan this session.

**(f) DEC Stokes-calculus operators (`katgpt-rs/crates/katgpt-dec/`)**

The IMIR's position-selection basis `P = {I⁽ᵖ⁾, M, M², …, Mᵏ⁻¹}` is exactly the **polynomial algebra in the DEC exterior derivative `d` on a 1D cell complex** (the position axis). `M = d` (shift back by one cell); `Mʲ = dʲ` (but `d² = 0` for `j ≥ 2` on a 1D complex, so the basis truncates at `k=2` for the pure exterior derivative — the paper's `k-1` upper bound comes from the sinusoidal embedding's frequency structure, not from `d²=0`). The token-selection basis `T = {I⁽ᵗ⁾, C}` has no DEC analog (tokens aren't a cell complex). This is a **vocabulary bridge**, not an actionable fusion — it says "the IMIR's position-shift algebra and our DEC substrate share a common ancestor (shift-circulant operators)".

### 2.4 Fusion (the highest-value combinations — speculative, documented for the record)

| Fusion | What it would produce | Blocker |
|---|---|---|
| **Commutant basis × `group_invariance_probe`** | Replace sample-then-score with closed-form commutant construction for permutation + shift groups. Faster (O(d³) one-shot vs O(n_samples·d²)), more principled (exact basis vs MC estimate). | Small. Concrete: add `commutant_basis(group_action) -> Vec<Matrix>` to `group_invariance_probe.rs`. **This is the actionable Gain — see §3.** |
| IMIR position-basis × DEC `exterior_derivative` | Frame `M` as `d` on the position axis; reuse DEC infrastructure for position-shift projections. | Vocabulary bridge only — `d²=0` truncates the IMIR's polynomial at k=2; the paper's k>2 comes from sinusoidal structure, not exterior algebra. No code change. |
| IMIR circuit detection × `causal_head_importance` | Sub-head-granularity circuit discovery on committed shards. | Requires a transformer substrate we don't have at runtime. Our "circuits" are latent_functor stages + HLA affect axes, not attention heads. |
| ICL/IWL split × freeze/thaw | Theorem 2's "IWL starves ICL" as a runtime pattern: frozen committed state dominates until curiosity explicitly overrides. | Already ships as `latent_functor/reestimation.rs` coherence-trigger. No new code. |

---

## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars). | Open primitive + private guide + plans. |
| **GOAT** | Provable gain over existing approach, but not a new class of capability. | Plan + implement. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only OR small enhancement, behind feature flag. |
| **Pass** | Mechanism already ships OR training-only (→ riir-train). | No files. |

### Verdict: **Gain** (small, deferred)

**One-line reasoning:** The paper's training-dynamics theory (Theorems 1-3, lottery ticket) redirects to riir-train; the modelless residue (§5 circuit detection + §3.2 commutant-basis construction) is a refinement of the already-shipped `group_invariance_probe` (Plan 355) — replacing its sample-then-score loop with a closed-form commutant construction for permutation + shift groups.

**Novelty gate (Q1–Q4):**

1. **No prior art?** ⚠️ **PARTIAL.** The general symmetry-invariant-operator framing partially ships via `group_invariance_probe` (Plan 355 — group orbit invariance test, MC sampling), `subspace_phase_gate` (Plan 301 — SVD-based subspace ID), and the Plan 270 gauge-invariant compose (the R+-scaling instance of the commutant). BUT the specific **closed-form commutant construction for permutation + shift groups** (`{I⁽ᵗ⁾, C}` for binary associations; `{I⁽ᵖ⁾, M, …, Mᵏ⁻¹}` for k-step shifts) is **not shipped**. → Q1 = **PARTIAL YES** (the construction is novel; the framing is not).
2. **New class of behavior?** **NO.** The residue is (a) project weights/latents onto a symmetry-defined basis — already a shipped capability class (group_invariance_probe, subspace_phase_gate, TILR); (b) greedy ablation — already covered by `ConstraintPruner` / `pipeline_pruner`. The new-capability claim would be "automated circuit discovery for our runtime" — but our runtime is per-NPC HLA / latent_functor state, not a transformer; the analog is weak. The runtime ICL/IWL split already ships as `latent_functor/reestimation.rs` coherence-trigger.
3. **Product selling point?** **NO.** Cannot finish "our NPCs do X no competitor can". The IMIR's selling point is about transformer training dynamics — we don't train transformers at runtime. The residue's selling point ("project frozen shards onto interpretable bases") is a refinement of the existing direction-vector ecosystem (MAG / EmotionDirections / CommittedFieldBlend), not a new pillar.
4. **Force multiplier?** **MODERATE.** Connects to `group_invariance_probe`, `subspace_phase_gate`, `manifold_erasure`, `tilr`, `spectral_rewire`, MAG, HLA EmotionDirections, NeuronShard freeze, DEC `exterior_derivative`. But force-multiplier alone doesn't make Super-GOAT.

→ **Q2 + Q3 fail → not Super-GOAT.** No private architectural guide triggered; no mandatory Super-GOAT outputs.

**GOAT vs Gain:**

- The commutant construction is a **more principled** alternative to MC sampling, but it is not **provably better** on our substrate. Our latent states (HLA 8-dim, NeuronShard 64-dim) have either trivial symmetry (BLAKE3-pinned basis) or named-axis symmetry (HLA valence/arousal/etc.) — the commutant for the latter is small and mostly already captured by the named-axis structure.
- The greedy-backward-elimination circuit-detection algorithm is a **direct port** of existing primitives (`ConstraintPruner` + `pipeline_pruner`).
- §1.55 value-extraction scan: the paper exposes no failure mode in shipped code (our group_invariance_probe makes no transformer-training parity claim), contradicts no config, and only weakly unblocks Issue 011 (LieFlow fusion, closed conditional on Plan 354 Phases 1-3).

→ **Gain.** A small, tracked enhancement to `group_invariance_probe.rs`: add a `commutant_basis` helper for permutation + shift groups as a more principled alternative to MC sampling. Behind the existing `group_invariance_probe` feature flag. No full plan; tracked in `.issues/157`.

> **Update (2026-07-16):** Issue 157 is **CLOSED — all tasks done.** The helper shipped as `commutant_basis<U: GroupAction>` + `commutant_of_matrices` (core solver) + `commutant_binary_association` + `commutant_shift` (closed-form constructors) + 10 unit tests. See the module doc's "Commutant basis" section for usage guidance. Issue file removed per the noise-reduction rule.

### MOAT gate per domain (§1.6)

| Domain | In scope? | MOAT contribution |
|--------|-----------|-------------------|
| **katgpt-rs** (public engine) | ✅ YES — but small | The commutant construction is a generic linear-algebra primitive (research-grade refinement of `group_invariance_probe`). Correct home for the helper. **Not a base-foundation primitive** — it's a refinement of an existing one. |
| **riir-ai** (private runtime) | NO for the residue | The per-NPC ICL/IWL analog already ships as `latent_functor/reestimation.rs`. No new runtime IP. |
| **riir-chain** (private chain) | NO | No chain/LatCal/sync-boundary angle actionable today. The "commutant coefficients → LatCal" fusion (§2.4) is speculative. |
| **riir-neuron-db** (private shards) | NO for the residue | The "commutant constraint on `MerkleFrozenEnvelope` updates" fusion (§2.3d) is speculative — no current shard layout has a non-trivial automorphism group (BLAKE3 pins the basis). |
| **riir-train** | **YES — for §3-4 training theory** | Theorem 1 (gradient confinement), Theorem 2 (IWL starves ICL), Theorem 3 (burstiness amplifies ICL), and the lottery ticket result (§4.3) are training-dynamics theory. → note "→ riir-train" and stop; no files there this session. |

### Why not GOAT (the honest demotion)

The closest the codebase comes to the IMIR residue is **`group_invariance_probe.rs` (Plan 355)** + **`subspace_phase_gate.rs` (Plan 301)** + **`ConstraintPruner` / `pipeline_pruner`**. A developer who reads Plan 355, replaces the sample-then-score loop with a closed-form commutant computation for the permutation + shift group, and wires the result into a greedy-ablation pass has built the IMIR residue in ~100 lines on top of existing primitives. The *construction* (commutant of permutation group = `{I, C}` after centering; commutant of shift group = `{I, M, …, Mᵏ⁻¹}`) is one-paragraph linear algebra. The *validated integration* with our actual latent-state kernels (HLA 8-dim, NeuronShard 64-dim) has no obvious target — those spaces have either trivial or named-axis symmetry, where the commutant is small and already implicit in the axis structure. **Claiming GOAT here would be claiming a gain over a combination that is 80% assembled from shipped pieces on a substrate where the symmetry is mostly trivial** — the false-GOAT failure mode.

### §3.5 Modelless-unblock protocol check

Not applicable — the paper's training theory redirects to riir-train by classification (it IS training theory), not by deferral-after-blocked-gate. The modelless residue (§5 circuit detection + §3.2 commutant basis) was never deferred; it was extracted directly.

### §3.6 Defend-wrong PoC

Not applicable — the verdict makes no quality-parity claim ("matches the paper's numbers"). The verdict is architectural-only: "the commutant construction is a more principled alternative to MC sampling". No PoC needed for a Gain that adds a helper function.

---

## 4. Latent vs Raw Boundary

The modelless residue operates entirely in **weight space** (full matrices) and **latent state** (HLA vectors, shard style_weights). Nothing crosses the sync boundary.

If the speculative "commutant coefficients → LatCal" fusion (§2.4) ever materializes, the commutant-basis coefficients (α, β, γ, δ-style scalars) would be the raw deterministic scalars crossing sync; the basis matrices themselves stay local. Same shape as the existing sync-boundary bridge pattern (`chain_karc_shard`, `chain_archetype_blend_shard` — latent payload local, BLAKE3 receipt crosses).

---

## 5. What ships (the Gain scope)

A single opt-in helper added to `katgpt-rs/crates/katgpt-core/src/group_invariance_probe.rs` behind the existing `group_invariance_probe` feature flag:

```rust
/// Compute the commutant basis `C(U) = {W : WU = UW ∀ U ∈ U}` for a finite
/// group action `U` on `ℝᵈ`. Returns a basis `Vec<Matrix>` such that every
/// symmetry-invariant linear operator is a linear combination of the basis
/// elements.
///
/// For the paper's two concrete groups (IMIR, Musat et al. 2026):
///   - Binary-association permutation group on centered token embeddings
///     ⟹ commutant = span{I⁽ᵗ⁾, C}  (identity + association matrix)
///   - k-step shift group on sinusoidal position embeddings
///     ⟹ commutant = span{I⁽ᵖ⁾, M, M², …, Mᵏ⁻¹}
///
/// This is the closed-form alternative to [`discover_subgroup_into`]'s
/// sample-then-score loop: instead of probing random `g ∈ G`, compute the
/// invariant-operator basis directly.
pub fn commutant_basis<U: GroupAction>(group: &U, d: usize) -> Vec<Vec<Vec<f32>>> { ... }
```

**No benchmark, no GOAT gate.** The helper is a more-principled alternative to MC sampling; it does not claim a provable gain. It lands behind the existing feature flag and is documented as an alternative construction. ~~Tracked in `.issues/157`.~~ **Shipped 2026-07-16** (Issue 157 closed + removed).

**What does NOT ship (documented as speculative in §2.4):**
- IMIR position-basis × DEC `exterior_derivative` bridge (vocabulary only, no code change)
- IMIR circuit detection × `causal_head_importance` (requires transformer substrate we don't have)
- Commutant constraint on `MerkleFrozenEnvelope` updates (no shard layout has non-trivial automorphism today)
- Commutant coefficients → LatCal commitment (speculative sync-boundary bridge)

---

## 6. Constraints check

| Constraint | Status |
|------------|--------|
| Modelless / inference-time | ✅ The residue (commutant basis + greedy ablation) is pure linear algebra — no training, no gradients. The training theory (Theorems 1-3) redirects to riir-train by classification. |
| Latent-to-latent preferred | ✅ The commutant construction operates on weight/latent operators. Nothing decodes to tokens. |
| Use sigmoid not softmax | ✅ No probability distribution claimed. The commutant is a deterministic basis. |
| Freeze/thaw over fine-tuning | ✅ The IMIR's "winning ticket" is a 4-parameter sub-manifold — exactly a frozen-snapshot pattern. The residue applies to frozen weights, not runtime weight mutation. |
| 5-repo discipline | ✅ Helper lands in katgpt-core (generic linear algebra). Training theory → riir-train note. No game/chain/shard IP in the open primitive. |
| Raw scalars at sync boundary | ✅ Nothing crosses sync in the Gain scope. The speculative LatCal fusion (§2.4) would commit scalars, not vectors. |
| Zero-alloc hot path | ✅ The commutant basis is computed once (offline / consolidation tier), not per-tick. |

---

## 7. Open questions / risks

1. **Does the commutant construction actually beat MC sampling on our substrate?** Our latent states (HLA 8-dim, NeuronShard 64-dim) have either trivial symmetry (BLAKE3-pinned) or named-axis symmetry (HLA valence/arousal/etc.). For named-axis symmetry, the commutant is small (mostly diagonal operators) and already implicit in the axis structure. **Mitigation:** the helper ships as an alternative, not a replacement — `discover_subgroup_into` (MC) stays the default; `commutant_basis` is opt-in for callers who know their group is permutation or shift.
2. **Is the runtime ICL/IWL analog (§2.3c) real or just vocabulary resonance?** The paper's Theorem 2 is about gradient descent on transformers. Our runtime "IWL starves ICL" claim maps to `latent_functor/reestimation.rs` coherence-trigger — but that's a runtime heuristic, not a theorem. The mapping is **descriptive, not proven**. **Mitigation:** the note labels this as vocabulary resonance, not a proven runtime theorem. No code change claimed.
3. **DEC `exterior_derivative` × position-shift `M` bridge — is it actionable?** The IMIR's `P = {I⁽ᵖ⁾, M, …, Mᵏ⁻¹}` is a polynomial in the shift; DEC's `d` on a 1D complex has `d² = 0`, truncating the polynomial at k=2. The paper's k>2 comes from sinusoidal frequency structure, not exterior algebra. **Mitigation:** documented as a vocabulary bridge only; no code change.
4. **Compute-unit translation (R368 lesson).** The paper's compute unit is "a transformer trained on a block-list task". Our runtime's compute unit is "an NPC tick" or "a latent_functor stage application". The IMIR theorem says training confines weights to the commutant; at runtime, no such training happens. **The residue is a post-hoc projection onto the commutant basis — applicable to any frozen weight matrix or committed shard, not just trained transformers.** This is the modelless extraction.

---

## TL;DR

Musat et al. prove that transformers trained on inductive tasks have weights confined to a low-dimensional Invariant Manifold (IMIR) spanned by interpretable basis matrices — the **commutant of the data-symmetry group** (token automorphisms + position shifts). Most of the paper (Theorems 1-3, lottery ticket) is training-dynamics theory → riir-train. The modelless residue (§5 circuit detection + §3.2 commutant-basis construction) is a **refinement** of two already-shipped primitives: `group_invariance_probe` (Plan 355, sample-then-score) and `subspace_phase_gate` (Plan 301, SVD-based subspace ID). The actionable piece is small — add a `commutant_basis(group_action) -> Vec<Matrix>` helper for permutation + shift groups as a closed-form alternative to MC sampling. **Verdict: Gain (deferred, tracked in `.issues/157`).** The commutant vocabulary is genuinely absent from the corpus (zero grep hits); the closest cousin is LieFlow's `group_invariance_probe`; the runtime ICL/IWL analog already ships as `latent_functor/reestimation.rs` coherence-trigger. No Super-GOAT (Q2/Q3 fail — no new capability class, no product selling point on our substrate); no GOAT (the construction is one-paragraph linear algebra on top of existing primitives, with no obvious non-trivial-symmetry target in our latent states).

> **Update (2026-07-16):** Gain **shipped**. Issue 157 closed + removed. `commutant_basis` + `commutant_of_matrices` + `commutant_binary_association` + `commutant_shift` landed behind `group_invariance_probe` with 10 unit tests (all pass). `cargo clippy` clean.
