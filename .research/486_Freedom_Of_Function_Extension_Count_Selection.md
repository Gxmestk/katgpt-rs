# Research 486: Freedom of Function — Extension-Count Selection for Best-of-K

> **Source:** "Why the Third Axis Is Freedom" — Michael Timothy Bennett (ANU, Machine Intelligence and Normative Theory Lab), 16 Aug 2026. Zenodo [10.5281/zenodo.21965230](https://doi.org/10.5281/zenodo.21965230) / arXiv:2608.05423. Analyzes Explorative Modeling (XM), Gladstone, Ji & Du, arXiv:2607.27372.
> **Date:** 2026-08-16
> **Status:** DISTILLED — PoC PASS (Issue 665 T1–T4, 2026-08-17; see §PoC
> Addendum) — primitive shipped opt-in `freedom_selection`; promotion pending
> a production consumer A/B
> **Related Research:** 369 (Renoise-CE best-of-N), 323 (TEMP diversity fingerprints), 320 (best_belief), 240 (CGSP curiosity), 479→riir-games coverage curiosity, 484 (percepta skill entropy)
> **Related Issues:** [665](../.issues/665_freedom_guided_best_of_k_poc.md) — PoC: freedom-guided best-of-K selection mode
> **Classification:** Public

---

## TL;DR

Bennett proves that XM's best-of-K generative training works because it increases **freedom of function** — w(π) = Π_c (2^{a_c} − 1), the count of completions of a policy in a finite embodied vocabulary, where a_c = permitted-output count at context c — and that freedom provably orders generalization (future compatibility ∝ freedom in the unseen-context vocabulary). For this stack the value is **modelless**: freedom is a closed-form, deterministic *selection criterion* (not a loss), and the paper's winning intervention (choose among near-best candidates the one that opens an unvisited region, at fixed K) is a selection rule our best-of-K substrates lack. Zero workspace substrate exists for extension-count/least-commitment ranking (audited 2026-08-16).

**Distilled for katgpt-rs (modelless, inference-time):**
1. **Normalized log-freedom score** — Σ_c log(2^{a_c} − 1) over a declared finite output partition; deterministic, zero-alloc ranking of models/checkpoints/shards by *generalization plasticity* rather than loss/similarity.
2. **Freedom-guided best-of-K selection** — among candidates within a loss gate of the winner, select the one maximizing freedom gain (opening an unoccupied cell). A new selection mode for `best_of_n_stability` / `best_of_k_rollouts` / `BoMSampler`.
3. **Marginal-value formula** — gain of another candidate is q(1−q)^K, peaking at q = 1/(K+1): a closed-form exploration-budget priority.
4. **Theorem-7 allocation** — optimal mass under nonuniform targets is the threshold form [1−(λ/(K·p_j))^{1/(K−1)}]+, converging to uniform-over-support as K grows.

---

## 1. Paper Core Findings

**Context.** XM (Gladstone et al. 2026) trains generators by drawing K candidates per target and updating through the closest — claiming "exploration" as a third pretraining axis (after data and parameters) with 6.2× data efficiency, FID 1.43 unguided ImageNet. Bennett's rebuttal: an axis must *rank trained models*; K cannot (two K=25 models encode wildly different function). Freedom can.

**Freedom (Stack Theory, Bennett 2023–2026).** An embodied vocabulary v (finite set of distinctions a body can make) induces a language L_v; a policy π is a statement; its extension Ext(π) = compatible completions; **freedom w(π) = |Ext(π)|**. For a generator with permission profile F(c) (outputs permitted at context c), encoding as exclusions gives the closed form **w(π_F) = Π_c (2^{a_c} − 1)**, a_c = |F(c)|. Weakest *correct* policies are likeliest to remain correct under unknown future demands (prior optimality proofs, Bennett 2025a/2026a) — the anti-Occam position: prefer the weakest hypothesis, not the shortest.

**Key theorems (all modelless math):**
- **Coverage identity (Thm 1):** R_K(π) = ∫₀^∞ E[(1−q_π(c,Y,t))^K] dt — best-of-K applies the transform 1−(1−q)^K to acceptable mass. Small-q regime ≈ Kq: exploration multiplies chances that exist; q=0 stays unreachable at any K.
- **Marginal value (Cor 1):** R_K − R_{K+1} = ∫ E[q(1−q)^K] dt — max at q = 1/(K+1). Certain regions need nothing; zero-mass regions get nothing.
- **Uniform targets (Thm 4):** K=1 — every sound law optimal (collapse invisible). **K≥2 — unique optimum is uniform over full valid support** = the freedom-maximizing sound policy. Exploration starts discriminating freedom exactly at K=2.
- **Balanced support (Thm 5):** hit probability strictly increasing in permitted count a (hence in freedom 2^a−1) for all K≥2.
- **Complementarity (Thm 6):** gain of another candidate = (1/m)(1−1/a)^K, **strictly increasing in a** — freer policies gain more from exploration; exploration increasingly discriminates freedom. (The two axes amplify each other.)
- **Nonuniform optimum (Thm 7):** q*_j = [1−(λ/(K·p_j))^{1/(K−1)}]+ — finite K favors frequent targets via a threshold; as K→∞ converges to uniform over full support. Frequency-governed → coverage-governed.
- **Future compatibility (Cor 4):** P(remains compatible) = Π_{c∈C_u} (2^{a_c}−1)/(2^{|Y|}−1) — **generalization probability is proportional to freedom in the unseen-context vocabulary.**
- **Mode count ≠ freedom:** supports of equal cardinality can differ in extension structure (richer vocabularies distinguish them); Example 1 shows greater freedom with *lower* finite-K access when mass is degenerate (0.998/0.001/0.001) — mass mediates access, permission mediates generality.

**Experiments.** (1) 144 paired runs: larger K → measured freedom rises (uniform: +8.32 log-freedom at K=8; saturates at full support K=16); context-dependent law shows hit and freedom *separating* (hit 0.71@K=8 → 0.65@K=32 while freedom keeps rising) — the mass-mediation prediction. (2) Freedom-selected checkpoints beat child-validation selection on a long-tail→uniform shift: 0.317→0.389 parent hit, **29/30 worlds** (sign-flip p=1.9e-9), accepting −0.022 child hit for +2.69 active outputs. (3) Matched ImageNet (XJumpy Small, K=25 fixed, seed 33): freedom-guided near-best selection improved FID 66.76→64.28 (−3.7%), guided FID −6.8%, IS +4.6%, **recall +11.7%** (precision −4.0% — broader coverage, less concentration); advantage grew 150k→200k. The controller: 16 prototype regions/class over cached VAE latents, decayed occupancy table, primary term = Δ log extension count when a candidate activates a region below threshold.

**Honest limitations (paper's own §13):** freedom needs a *declared finite vocabulary + admissibility rule* (raw support saturates useless when softmax/dense generators put positive mass everywhere; discontinuous at zero-mass boundary); Exp 2 is transductive common-pool (selector read unlabelled parent contexts; not equal-compute); Exp 3 is one matched run, shared seed 33, no random-near-best control (cannot yet separate freedom guidance from mere relaxation of the min-loss choice); coupled (non-iid) candidates treated only in Remark 1.

## 2. Distillation

### 2.1 Path 0 decomposition (mandatory before any riir-train deferral)

| Paper component | Nature | Modelless analog | Verdict |
|---|---|---|---|
| Freedom score Π(2^{a_c}−1) / Σlog | Closed-form math | YES — deterministic scoring over support counts | katgpt-rs primitive |
| Coverage identity + marginal value q(1−q)^K | Closed-form math | YES — applies to shipped best-of-K inference | consume in selection modes |
| Thm-7 threshold allocation | Closed-form math | YES — budget/priority formula | candidate for bandit/curiosity budgets |
| Freedom-guided near-best selection (Exp 3) | Selection rule (no GD in selector) | YES — new mode for best-of-K substrates | **PoC issue 665** |
| Freedom as model/checkpoint selector (Exp 2) | Ranking criterion | YES — ranks shards/adapters/frozen snapshots | fusion lead (neuron-db/riir-ai) |
| XM winner-takes-gradient training (XM paper's, not this paper's) | Training loop (GD) | NO | **redirect, see §4** |

### 2.2 Vocabulary translation (paper → codebase; grep BOTH sets)

| Paper term | Codebase equivalents | Note |
|---|---|---|
| freedom / extension count / compatibility volume | support size, mode count, covered bitmask, active outputs | **zero direct hits** — concept absent |
| best-of-K / min-of-N / winner-takes-gradient | `best_of_n_stability`, `best_of_k_rollouts`, `BoMSampler`, CLR best-of-N baseline | inference-side covered, training-side absent |
| permission profile / permitted outputs | support of output distribution, `DemonstratedSkill.covered` bitmask | mass ≠ permission — the paper's key split |
| mode collapse / coverage | `EntropyCollapse`, δmg mass-gravity, hesitation tokens, effective rank, TEMP spread | 5+ distinct signals ship |
| exploration / curiosity | `DerivativeCuriosity`, `inject_exploration`, coverage_curiosity, QMC pass@k | covered |
| weakest correct policy / least commitment | *(none)* — closest: `SlackCorePartition` "least expressive 25%" consolidation | **the gap** |
| generative expressivity (mode-count proxy) | MoA/HLA expressivity hierarchies (architecture property, not selection) | different sense |

⚠️ **Disambiguation:** `speculative/qmc/mod.rs` "max coverage, **min freedom** (pairwise MI = −∞)" uses "freedom" as *sample independence* — Bennett's freedom axis is the opposite notion (compatibility volume). Keep the translation table honest; do not conflate.

### 2.3 Signal-diff vs closest shipped cousins (§3.6 defense)

| Cousin | Signal it consumes | Freedom-guidance signal | Diff |
|---|---|---|---|
| `coverage_curiosity` (riir-games, Issue 672) | has the *teacher* demonstrated level L? (external exemplar coverage) → seek ONE exemplar at uncovered level | has the *learner* recently produced region R? (self-output occupancy) → re-rank own candidates to open regions | external-exemplar targeting vs self-candidate re-ranking; complementary, fuse-able (Thm 6 gives the priority order) |
| `best_of_n_stability` (renoise_ce, R369) | perturbation-stability (drift under contraction) | Δ extension count within loss gate | stability vs coverage-opening |
| `best_of_k_rollouts` modes (dd_tree) | BestQ relevance / mode@K frequency / Top1 residual | unvisited-region opening | none of the three consume occupancy |
| `retrieve_diverse` (neuron-db) | Clifford-wedge span across shards (embedding-space independence) | completion count in declared output vocabulary | structural independence in latent space vs permission structure in output space; freedom has the provable generalization link |
| `best_belief` (Plan 336) | Beta(1+S,1+F) ε-quantile (history) | support profile (structure) | history vs structure; both non-loss selection |
| chiaroscuro `OpPromotion` (Plan 269) | utilization entropy → keep/demote ops | log-freedom product over contexts | mass-distribution entropy vs permission/completion count |

### 2.4 What does NOT transfer

- Raw support as freedom is useless for dense/softmax generators (positive mass everywhere → constant support) — any impl needs a declared finite partition + admissibility rule (the paper's own limitation). Our analog: occupancy thresholds on a codebook/prototype grid (exactly what Exp 3's controller did with 16 regions/class).
- Exp 2's transductive advantage included reading unlabelled test-distribution contexts — a leak-shaped advantage unless consumers genuinely have deployment-context access (our Warm-tier/shard retrieval does, SFT model selection doesn't).
- ImageNet evidence is single-seed without a random-near-best control — the *relaxation-vs-freedom* confound is unresolved upstream; our PoC must include a random-near-best arm.

## 3. Fusion (novel combinations; novelty TBD — each needs its own gate)

1. **FreedomGain selection mode** (katgpt-rs): fourth `WidthSelectionMode` / renoise-ce mode — near-best gate + Δ-log-freedom primary term. Falsifiable vs BestQ/mode@K/Top1Converged/random-near-best on a controlled toy → **issue 665**.
2. **Theorem-6 demonstration priority** (riir-ai/games): marginal value of a demonstration (1/m)(1−1/a)^K increasing in uncovered-set size a — a closed-form *ordering* for `plan_observation` targets in coverage_curiosity (currently cluster-triggered, priority-free). Fuse: teach the largest-uncertainty-band pet first.
3. **Freedom-tiered consolidation** (riir-neuron-db): `SlackCorePartition` consolidates "least expressive" shards by heuristic; freedom gives the principled cut — consolidate low-freedom shards (nearly determined; averaging loses little), preserve high-freedom shards. Also a freeze-gate signal: can_freeze ∧ low-freedom → freeze; high-freedom → keep consolidating.
4. **Per-NPC plasticity scalar** (riir-ai): normalized log-freedom over a behavior vocabulary as a *latent semantic* scalar (dot+sigmoid projection per house rules) — "how much is left unspecified" — driving which NPCs remain curious (Thm 6: high-freedom NPCs gain more per demonstration) vs which are effectively converged. Latent-domain (semantic), local-only, never synced — fits the sync boundary cleanly.
5. **Checkpoint selection under distribution shift** (riir-train/riir-ai): when picking among frozen snapshots after a domain shift, freedom (computed on unlabelled shift contexts) vs child-validation loss — the Exp-2 shape. Applies wherever we hold candidate snapshots (freeze/thaw envelope).

## 4. Verdict

**Tier: Gain → GOAT (PoC-gated).** One-line reasoning: the criterion is novel-in-workspace, modelless, and externally evidenced (29/30 + FID), but our own quality claim is unproven and the paper's headline evidence carries an unresolved relaxation confound — so distill now, promote to plan+feature-flag only if issue 665's PoC beats the random-near-best control. Not Super-GOAT: the game-context behavior class (§3.2/§3.4) is inferred, not demonstrated, and freedom is Bennett's prior theory (we consume, not invent).

**MOAT gate (katgpt-rs):** generic math over support profiles + inference-time selection modes — no game/chain/shard semantics → katgpt-rs is the correct home for the open primitive; fusion leads 2–5 route to riir-ai/riir-neuron-db as future guides/plans if their own gates pass.

**Training-side redirect (Path 0.5, explicit justification):** XM's winner-takes-gradient is *not* applicable to current riir-train pipelines — corpus→LoRA SFT is next-token cross-entropy (no generative-matching loss), civ trajectory Q-learning samples actions not generations, arena labels by outcome. Deferred to the systematic backstop (≥3 recipe gaps → batch plan); reopen trigger: any pipeline where one target admits many valid generations (e.g., future game-content or asset generation).

**Published prior art (novelty context, all external):**
- Bennett's own line (the theory being consumed): AGI-16 "weakest not shortest" (2023), "Is complexity an illusion?" (2024), PhD thesis Stack Theory (2025), "Are flat minima an illusion?" (2026a), "The wrong razor" (2026b).
- Classical ancestor: **Mitchell version spaces / least-commitment learning (1982)** — weakest-correct-hypothesis is the anti-Occam inversion of minimum-description-length; Bennett's addition is the embodied-language extension count + optimality proofs.
- Entropy relative: Jaynes maxent (1957) — Prop 3 shows entropy and freedom agree only under symmetric independence; extension structure is the differentiator. SAC max-entropy RL is the RL-side entropy analog.
- Best-of-K training lineage (cited by paper): MCL (2012), SMCL (2016), Fan et al. min-of-N (2017), IMLE (2018) — XM is the scaling framework around an older update; Bennett supplies the *criterion* explaining why it works.

**Follow-ups:** issue 665 (PoC) is the only actionable file; fusion leads 2–5 recorded here for future pickup — do not open plans until each has a consumer and a falsifiable gate.

---

## PoC Addendum (2026-08-17, Issue 665 T1–T4)

**Verdict: T4 gate PASS — the relaxation confound is separated; freedom
guidance is real on the controlled toy.** Primitive shipped opt-in
(`freedom_selection`) in katgpt-rs; PoC in riir-poc
`examples/freedom_best_of_k.rs`.

- **Toy:** 4 contexts × 8 cells, 40 steps, K=8 candidates/step, gate 0.5,
  64 seeds; child = long-tail within context (0.35/0.35/0.05×6), parent =
  uniform (Exp-2 shape). Candidate loss = per-cell quality (uniform, fixed
  per seed) + ≈N(0, 0.35) draw noise — INDEPENDENT of cell identity by
  design (if loss tracked −log child_p, tail cells could never enter the
  gate and all arms would degenerate to min-loss; the paper's controller
  gates on validation loss of genuinely-similar candidates).
- **Arms (matched pools — all replay identical candidates):** MinLoss /
  RandomNearBest (the confound control, SAME gate) / Stability (the REAL
  `best_of_n_stability` substrate) / FreedomGain (the REAL
  `best_of_n_freedom` substrate).

| arm | parent_hit | child_mass | mean_loss | active cells | log-freedom |
|---|---:|---:|---:|---:|---:|
| MinLoss | 0.4453 | 2.4750 | 0.0183 | 14 | 9.338 |
| RandomNearBest | 0.5156 | 2.9719 | 0.2141 | 16 | 11.084 |
| Stability | 0.4453 | 2.4750 | 0.0183 | 14 | 9.338 |
| FreedomGain | **0.7075** | **3.4242** | 0.0932 | **22** | **15.566** |

- **Per-seed wins: FreedomGain vs MinLoss 64/64, vs RandomNearBest 64/64**
  (paper Exp-2 shape 29/30, here a clean sweep). Decomposition: relaxation
  alone buys +0.070 parent-hit (RandomNearBest − MinLoss); freedom guidance
  buys +0.192 MORE on top (FreedomGain − RandomNearBest) — 73% of the total
  gain is the freedom signal, not the gate.
- **Honest findings:** (1) Stability ≡ MinLoss bit-identical on this toy —
  the shipped selection family (BestQ / mode@K / Top1Converged analogs) is
  occupancy-blind by construction; that is the gap FreedomGain fills, not a
  straw man. (2) FreedomGain ALSO wins child coverage (3.42 vs 2.48) — in a
  coverage toy opening cells strictly adds covered mass on BOTH
  distributions; the real-world trade appears on a loss/quality axis, where
  freedom pays +0.075 mean loss (within the 0.5 gate) for +8 active cells —
  the Exp-2 trade shape (paper: −0.022 child hit for +2.69 active outputs).
  (3) child_mass sums to ≤ 4.0 (per-context mass 1.0 × 4 contexts) — it is
  covered child MASS, not a probability; compare arms relatively.
- **API notes (deviations from the issue sketch, documented):**
  `freedom_gain` takes 3 args (per-cell occupancy + cell→context partition +
  candidate cell) — the 2-arg sketch cannot distinguish "occupied cell in
  context a" from "fresh cell in context a". Empty contexts (a=0) are
  excluded from the product (factor 1; the raw 2^0−1=0 zero-annihilates);
  first activation pinned to FIRST_ACTIVATION_GAIN=2.0 > ln 3 (raw increment
  is +∞ from the excluded state — ordering "open an unvisited context
  first" preserved with a finite constant).
- **G1:** deterministic (seeded fastrand, no wall-clock) — seed-0 double-run
  bit-identical, asserted in the example. **Cost:** the primitive is
  closed-form f32 arithmetic — no bench needed at PoC tier (selection-time,
  not hot-path; the substrate's one Vec alloc per call is documented).
- **Promotion status:** stays opt-in. The T4 pass opens (does not compel)
  the promotion path: a production consumer A/B (the switch_cost / Issue-663
  precedent) + a GOAT gate. T5 (Thm-7 allocation formula) remains unbundled
  and unstarted — separate gate if ever consumed.
