# Research 524: GFlowNet CombOpt — Constructive MDPs for Graph Combinatorial Optimization

**Status:** RECORD — modelless track covered (shipped prior art + measured negative); training track filed as riir-train Plan 368 (`../riir-train/.plans/368_gflownet_transition_db_quest_grammar.md`).

- **Paper:** Dinghuai Zhang, Hanjun Dai, Esmeralda S. Whitammer, Aaron Courville, Yoshua Bengio, Ling Pan — ["Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets"](https://arxiv.org/abs/2305.17010) (arXiv:2305.17010, NeurIPS 2023, 177+ citations). Code: zdhNarsil/GFlowNet-CombOpt.
- **Date distilled:** 2026-08-31
- **Verdict:** GAIN (per-track). Modelless: all inference-time components have shipped analogs (and the reward-modulation class was measured NEGATIVE in Bench 011). Training: the transition-buffer DB + forward-looking (FL) intermediate-energy recipe is directly applicable to the quest-grammar LoRA pipeline and the G-Zero DPO/GRPO path → Plan 368.

---

## TL;DR

Trains **conditional GFlowNets** to sample solutions to four NP-hard graph CO problems (max independent set, max clique, min dominating set, max cut) with probability ∝ exp(−E(x)/T). The paper's two real technical contributions beyond the GFlowNet family baseline: (1) **per-problem constructive MDPs** with three-valued states {0, 1, ⊘} whose transitions **proactively enforce constraints** (MIS: add a vertex → all its neighbors are auto-marked 0, so *every* intermediate state is completable to a valid solution and every terminal state is feasible by construction); (2) **transition-based FL training** — break trajectories into individual transitions, train a detailed-balance loss on a random transition buffer with a handcrafted intermediate energy Ẽ(s) (partial-solution quality) folded into the loss, which fixes long-range credit assignment on ~400-step trajectories that trajectory-level losses cannot even fit in GPU memory. Selling point vs RL: **diverse high-quality candidates** (solution-level, not trajectory-level, entropy) — avoids mode collapse; inference = sample 20, take best. Beats PPO and supervised baselines on MIS/MC/MDS/MCut; near-solver quality on several benchmarks.

**Pinned claim (before search, per the TTPO rule):** the *modelless* claim would have been "feasibility-preserving constructive MDP design + dense Ẽ(s) per-step signals as a search primitive for the DDTree/quest-grammar stack" — that claim is KILLED by shipped coverage (DDTree screened build prunes-before-expand during construction; DeltaBanditPruner already consumes Ẽ-delta rewards) **plus a measured negative** (Bench 011: GFlowNet flow regularization → no DDTree gain, "reward modulation, not selection"). The *training* claim — "transition-buffer FL-DB credit assignment as an upgrade to outcome-level LoRA SFT / end-of-trajectory GRPO, yielding a sampler over valid quest programs with a diversity axis the deterministic drafter structurally cannot reach" — survives (no GFlowNet training pipeline exists in any repo; web prior art shows no GFlowNet-based game-content generation).

---

## 1. Paper Core Findings (Path 0 decomposition)

### 1.1 Constructive MDPs with proactive constraint enforcement (§3.2, App. B)

State = binary vector with three values per vertex: 0 (excluded), 1 (in set), ⊘ (unspecified). Initial state all-⊘. Actions flip one ⊘ to 0 or 1; the **transition function then auto-resolves dependent variables**: MIS — set vertex to 1 ⇒ all neighbors flip to 0 (constraint never even becomes violable); max clique — vertices not connected to the whole current set flip to 0; MDS — remove a vertex only when every neighbor stays dominated (isolated vertices forced to 1); max cut — void vertices that would decrease the cut flip to 0. Termination = fully specified = **order-maximal feasible solution**. No stop action needed. This is constraint programming's forward-checking/MAC moved into the generative process: feasibility is structural, not filtered.

### 1.2 Transition-based training (§3.3)

Trajectory-level GFlowNet losses need n network passes per trajectory (n up to ~400 here) and store all intermediate activations — 250-batch × 400-step does not fit a 40 GB GPU. Fix (from Deleu et al. 2022): sample random **transitions** from completed rollouts into a buffer, minibatch 64, DB loss per transition. Memory ∝ batch, not trajectory length; converges faster (Fig. 4).

### 1.3 Forward-looking (FL) loss with intermediate energy Ẽ(s) (§3.3, Eq. 6)

Terminal-only reward starves credit assignment on long trajectories. FL adds a **continuation of the terminal energy to every intermediate state** — Ẽ(s), handcrafted per problem (MIS/clique: minus current set size; MDS: plus current set size; cut: cut edges so far) — into the DB loss:

```
ℓ_FL(s,s') = ( −Ẽ(s) + log F̃(s) + log P_F(s'|s) + Ẽ(s') − log F̃(s') − log P_B(s|s') )²
```

Dense per-transition supervision; semantically coherent with the terminal reward by construction. This is potential-based reward shaping (Ng et al. 1999) wearing a GFlowNet uniform.

### 1.4 Optimization-as-sampling + temperature (§3.1, Prop 1)

Sample x ∝ exp(−E(x)/T). Perfectly trained: T→∞ ⇒ uniform over **feasible**; T→0 ⇒ uniform over **optimal**. Inverse temperature 500 works across all four tasks; temperature annealing is optional (ablation: same performance without).

### 1.5 Conditional amortization (§2.1)

Every GFlowNet component (P_F, flow, P_B) is conditioned on the graph instance g (GIN encoder, 5×256; separate flow net — sharing one net was unstable; uniform backward policy is enough). Off-policy robust: mixing up to 75% uniform exploration noise barely hurts.

### 1.6 Inference protocol

Sample 20 candidates, take the best. The diversity is the point: mode-collapsed samplers waste the best-of-N budget on duplicates.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalents (≥2 each) |
|---|---|
| "constructive MDP, void states ⊘" | "DDTree partial draft", "quest program prefix", "draft tree node" |
| "proactive constraint transitions" | "ConstraintPruner during build", "prune-before-expand", "invalid action masking" (riir-ai Research 026 RTS) |
| "transition buffer + DB loss" | "DeltaBanditPruner observe delta", "replay_backward.rs BackwardSample", "bandit arm updates" |
| "FL intermediate energy Ẽ(s)" | "screening relevance per depth", "delta reward", "partial-state score in build_screened" |
| "reward-proportional diverse sampling" | "BoM single-pass diverse sampling (Research 248)", "population TTS (Plan 260 MaxProof)", "seed diversification (Research 098)", "SDE-noise DDTree variants" |
| "flow conservation on state DAG" | "codifferential δ (katgpt-dec)", "discrete divergence", "divergence-free cochain" (Research 296 crosswalk) |
| "conditional GFlowNet amortization" | "TernaryDraftModel amortized drafter", "drafter LoRA (grammar_training.rs)" |
| "temperature annealing" | "sampling temperature", "Boltzmann/softmax sampling" |

### 2.2 Closest prior art (BOTH layers, ALL repos)

**Layer 1 — notes/plans:**

| Note / Plan | Mechanism | Match |
|---|---|---|
| **Research 023 (GFlowNet Shortest Paths)** | Same GFlowNet family; distilled "min flow = shortest paths" theorem modellessly into Plan 052 | Closest family cousin — different paper (permutation puzzles, TB + flow regularization), NOT this CO-MDP paper |
| **Research 263 (Latent Thought Flow)** | Continuous GFlowNet (EW-SubTB) for latent reasoning; verdict GAIN, training redirected | Second family cousin; its §2.3 prior-art table already maps the family |
| **Plan 052** | Shipped the distillations: FlowPruner, balanced DDTree, flow-weighted bandit reward, ReplayBackwardWalker | The modelless GFlowNet distillation ALREADY RAN — with a measured result (below) |
| **riir-ai Research 026 (RTS invalid-action masking)** | Proactive action masking during play | C1 coverage (constraint enforcement during generation) |
| **riir-ai Research 344 (crowd distribution targeting) + IFD** | Reward-shaped crowd target diversity | C5 coverage on the game side |

**Layer 2 — shipped code:**

| File | Mechanism | Match |
|---|---|---|
| `katgpt-rs/src/speculative/dd_tree.rs` `build_screened` | Pruner applied **during** tree construction (invalid children rejected as generated) | C1 — feasibility enforced during construction, reject-style vs the paper's auto-resolve style |
| `katgpt-rs/crates/katgpt-speculative/src/flow_pruner.rs` (`FlowPruner`, `bandit` feature) | GFlowNet-inspired stop-probability regularization | Family already shipped (Plan 052 D1) |
| `katgpt-rs/crates/katgpt-pruners/src/g_zero/delta_bandit.rs` `lambda_length` | Ẽ-delta / trajectory-length bandit rewards | C3 — dense per-step signals already consumed as bandit rewards |
| `katgpt-rs/src/pruners/bomber/replay_backward.rs` | Backward policy extraction from replays | C2's modelless shadow (backward walk without gradients) |
| `riir-ai/crates/riir-games quest_grammar` (`grammar_training.rs`, `quest_training.rs`) | LoRA-trained quest drafter + ConstraintPruner | The training-track consumer — outcome-level SFT today |
| `.benchmarks/011_bt_rank_goat.md` | **Measured: GFlowNet flow regularization → "No DDTree gain — reward modulation, not selection"** | The decisive measured negative for the modelless track |

### 2.3 What does NOT transfer modellessly

- The DB/FL **gradient** objectives (need backprop) → riir-train.
- The GIN instance-conditioning amortizer (unnecessary modellessly: each instance is a fresh walk; amortization only pays when re-solving thousands of instances).
- Prop 1's T→0 optimality guarantee (needs the trained flow). Modelless substitute: Metropolis/best-of-N over the feasible set with a greedy floor (Caro–Wei: maximal independent set ≥ n/(Δ+1)) — a bound, not the guarantee.

---

## 3. Three-track adversarial panel (merged)

**No-GD advocate (modelless) — extracted 7 items, all disposed:**

| # | Extraction | Disposition (audited) |
|---|---|---|
| 1 | 3-valued MDP as modelless state machine; feasibility as pruner soundness property | **Covered** — DDTree `build_screened` enforces constraints during construction (reject-style); riir-ai RTS masking. Signal-diff: the paper auto-*resolves* dependent variables (add v ⇒ neighbors=0), DDTree *rejects* invalid children — for feasibility maintenance the two are equivalent; the paper's forced completions only matter for sampling uniformity, which the modelless track doesn't need (Bench 011 negative) |
| 2 | T→∞ half of Prop 1: uniform random walk on the MDP samples exactly uniform-feasible, free | True but low-value: uniform-feasible is the *worst* sampler for CO quality; the useful limit (T→0) needs the trained flow |
| 3 | Ẽ(s) potentials are closed-form per problem; consume directly as per-step guidance | **Covered + measured negative** — DeltaBanditPruner consumes Ẽ-deltas; Bench 011 measured this class at "no DDTree gain — reward modulation, not selection" |
| 4 | Metropolis temperature spectrum reconstructs the Boltzmann family modellessly | **Discard** — generic MCMC, not a paper-specific mechanism; stack temperature sampling ships; no new capability class |
| 5 | Transition decomposition legitimizes per-step bandit over actions | **Covered** — the bandit pruners ARE per-step bandits; Bench 011 again |
| 6 | Diversity measured as pairwise Hamming distance over best-of-N; restart diversity needs no training | **Covered** — BoM (Research 248), MaxProof (Plan 260), seed diversification (Research 098), SDE-noise DDTree variants |
| 7 | Greedy floor |x| ≥ n/(Δ+1) as a G8 correctness floor | **Recorded** — Caro–Wei bound adopted into Plan 368's G8 gate as the modelless-parity floor |
| DEC fusion idea: cut = d(indicator 0-cochain); δ(edge cochain) = per-vertex imbalance as dense violation signal | **Audited discard** — δ-of-edge-cochain computes exactly the local constraint-violation count the pruner already computes in O(deg) directly; no signal the pruner lacks. Recorded here as a crosswalk note (Research 296 vocabulary), not filed |

**Model-based advocate (trained weights) — recipe extracted, ACCEPTED → Plan 368:**
- Transition-buffer FL-DB is the single biggest credit-assignment upgrade available to the stack's two long-trajectory trainers: quest-grammar LoRA (currently outcome-level SFT — one gradient signal per program) and G-Zero DPO/GRPO (currently end-of-trajectory credit, n forward passes per rollout).
- The ConstraintPruner IS the transition mask: replay buffers never contain illegal transitions — zero new infrastructure to make training constraint-clean.
- The load-bearing axis is **G3 diversity**: a flow sampler emits a *distribution* over valid programs (paper: 20 samples, distinct modes); the deterministic modelless drafter structurally emits ≈1. Consumers: living-world quest variety, healer corpus expansion, DPO curriculum.
- Recipe + GPU-hours + GOAT gate → `../riir-train/.plans/368_gflownet_transition_db_quest_grammar.md` (2–6 GPU-hours per quest domain on the 4090; promotion only if G2 quality parity AND G3 ≥5× diversity both pass; otherwise honest-loser artifact per the demote-loser rule).

---

## 4. Verdict (per track — TTPO rule)

### Modelless track: covered — no file beyond this note

- Q1 (no prior art): **NO.** Plan 052 already ran the modelless GFlowNet distillation for the family; C1/C3/C5 have shipped analogs in DDTree/DeltaBanditPruner/BoM respectively.
- Q2 (new behavior class): **NO.** Feasibility-by-construction ≡ prune-during-build; diverse best-of-N ships four ways.
- Q3/Q4: moot.
- Additional kill: **Bench 011 measured the reward-modulation class negative on the exact surface (DDTree) this paper's modelless shadow would touch.** Filing a plan against a measured negative without a new mechanism would be re-litigating a closed verdict.

### Training track: GAIN → riir-train Plan 368

- Applicable per Path 0.5: the training loop IS the value (transition-buffer DB + FL dense credit), and the stack has two concrete consumers with a measured capability gap (outcome-level credit; zero solution diversity from the deterministic drafter).
- Serving-envelope note: the trained artifact runs in the hot path as *just another drafter* (same shape as the existing LoRA drafter), so envelope fit is fine — but the modelless drafter holds the default slot and the promotion bar is deliberately double (G2 + G3) per the demote-loser rule.
- Published prior art: the paper itself (177 citations) is the technique's home; follow-ups exist (GFlowVLM CVPR'25, LGGFN AAAI'26, adversarial GFlowNet vehicle routing '25) — no GFlowNet-for-game-content prior art found; the novelty is the stack application, which is what the plan claims, nothing more.

## 5. Fusion

**GFlowNet-family position in the stack (post this note):** Research 023 owns the *shortest-path* theorem distillation (FlowPruner et al.), Research 263 owns the *continuous latent-flow* family, this note owns the *discrete CO constructive-MDP* family — and routes its training half to Plan 368. If Plan 368's G3 diversity axis ever passes, the natural fusion with Research 263's recorded "cost-aware reward-proportional scorer" follow-up is a single **diverse-valid-program sampler** serving quest generation (riir-games) and healer corpus expansion (riir-clippy) from one checkpoint. That fusion stays unfiled until a G3 pass exists to build on.

## 6. Cross-References

- `../riir-train/.plans/368_gflownet_transition_db_quest_grammar.md` — the filed training plan (recipe, GPU-hours, GOAT gate).
- `.research/023_GFlowNet_Shortest_Paths.md` + `.plans/052_gflownet_modelless_distillation.md` — the family's modelless distillation (FlowPruner, balanced DDTree) + Bench 011's measured negative.
- `.research/263_Latent_Thought_Flow_Reward_Proportional_Latent_Reasoning.md` — continuous-GFlowNet cousin; its prior-art table §2.3 is the family map.
- `.benchmarks/011_bt_rank_goat.md` — "reward modulation, not selection" — the measured negative that closes the modelless track.
- `../riir-ai/.research/026_RTS_Intransitive_Balancing_Invalid_Action_Masking.md` — proactive action masking (C1 game-side coverage).
- `.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` — the δ/divergence vocabulary used in the DEC discard.
- riir-train `.research/414_code_llm_survey_rlvr_clippy_l4.md` — the RLVR/L4 context for dense-credit training on the healing side.

## 7. References

- Zhang, Dai, Whitammer, Courville, Bengio, Pan — arXiv:2305.17010, NeurIPS 2023.
- Bengio et al. 2021/2023 (GFlowNet foundations); Malkin et al. 2022 (TB); Deleu et al. 2022 (transition-based BF — the training trick this paper adopts); Pan et al. 2023a (forward-looking, the FL source); Madan et al. 2023 (SubTB partial episodes).
- Ng, Harada, Russell 1999 (potential-based reward shaping — what Ẽ(s) is, mathematically).
- Caro & Wei 1995 (the n/(Δ+1) maximal-independent-set floor used as the G8 parity bound).
