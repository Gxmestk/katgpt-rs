# Research 509: Rewarding Progress — Prover-Policy Advantages for Process Verifiers

> **Source:** [Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning](https://arxiv.org/abs/2410.08146) — Setlur, Nagpal, Fisch, Geng, Eisenstein, R. Agarwal, A. Agarwal, Berant, Kumar (Google DeepMind/Research), ICML 2025. 339 citations — the canonical PRM paper.
> **Date:** 2026-08-27
> **Status:** RECORD (Issue 692 closed 2026-08-27: T1–T3 landed `fcfb5f8c`/`983c6fb8`, T4 refuted-by-mechanism — Fusion D correction below, T5 GOAT PASS [Bench 684](../.benchmarks/684_prover_selection_goat.md) `1b65662f` — `prover_selection` DEFAULT-ON in katgpt-core; riir-train Plan 356 remains the open training arm)
> **Classification:** Public (generic math) + private consumers + riir-train training arm
> **Related:** 250 (self-advantage — single-policy cousin), 160/180 (SDPG centered_log_ratio — oracle/student cousin), 373 (ReMax expected-max — same Bernoulli-K form), 494 (conformal dual-threshold), 322-riir-ai (civ critic stop rule — adjacent, closed), 426-riir-train (TETHER blend — level-signal blend family)

---

## TL;DR

The paper's thesis: **process rewards should be advantages measured under a *different* policy**. A^μ(s,a) := Q^μ(s,a) − V^μ(s) — the change in a *prover* policy μ's success likelihood across the step — not Q-values (which conflate the previous state's promise with the action's quality), and not the base policy's own advantage (μ=π reproduces outcome-only gradients exactly). Good provers are **complementary**: high variance of A^μ across the base policy's actions (**distinguishability**) + non-negative inner product with base advantages (**alignment**, Theorem 3.1). Weak-but-complementary provers beat strong ones; Bo4 is the empirical sweet spot; K→∞ kills the signal (Q→1 ∀ steps ⇒ A→0). Trained PAV verifiers give +8% search accuracy at 1.5–5× less compute and 5–6× RL sample efficiency — but the paper's own Limitations concedes the verifier is amortized Monte-Carlo, **circumventable by running prover rollouts directly**.

**Distilled for us (modelless):**
1. **Prover-selection statistics** — D(μ)=E Var_{a~π}[A^μ], Al(μ)=E[A^μ·A^π] are offline-computable from logged outcomes; Theorem 3.1 is a predicted-gain pre-gate. We currently rank drafters/rules/critics by **strength** (mean acceptance, Elo) — the exact selector the paper refutes.
2. **Advantage centering in cross-state selection** — beam/MCTS/candidate-pool selection across *different* states must subtract the per-state baseline V(s); within-state argmax is centering-invariant (honest negative for two tempting consumers, below).
3. **K\* interior-optimum law (derived here, beyond the paper)** — for the BoK advantage A(K) = (1−V)^K − (1−Q)^K, the K maximizing distinguishability is **K\* = ln(ln(1−Q)/ln(1−V)) / ln((1−V)/(1−Q))** (verified: Q=.5,V=.3 → K\*≈1.98, peak at K=2). The paper sweeps K empirically; this closed form removes the sweep.
4. **Potential-difference immunity** — difference-based signals resist level-inflation farming (the paper's "REPHRASE THE PROBLEM" degenerate optimum); level-based signals do not. Our semantic domain already bets this way (emotion *kicks* on change) — the paper supplies the theorem-grade rule for every *new* scoring surface: **score the Δ, gate on the level**.
5. **First-pit attribution** — fault = first step where Q̂ drops ≈0 on an incorrect rollout from a high-value state; a thresholded changepoint detector on logged Bernoulli means, no learned parameters.

**Redirect → riir-train (Plan 356):** the trained PAV itself — a gemma-2-2b QLoRA edit-scorer attacking the clippy L4 fixer's measured 0/60 G1 (~30 GPU-h), plus the dense per-edit reward arm.

---

## 1. Paper core

### 1.1 Why advantages, not Q-values

Q^π(s,a) mixes the state's promise with the action's quality. Beam search retaining by absolute Q keeps a bad step from a good state (paper Fig 2a: a_{1,1} with −0.05 progress survives from high-Q s_1 while +0.20 a_{2,1} from low-Q s_2 is pruned). The advantage A(s,a) = Q(s,a) − V(s) decouples them: positive/negative values supervise progress and failure alike, including steps in *incorrect* traces — which is what diversifies exploration. Formally advantages are potential differences (Ng et al. 1999) ⇒ policy-invariant shaping, immune to the degenerate optimum where a trivial step harvests level-reward (App G: Q^μ-rewarded policies converge to emitting "REPHRASE THE PROBLEM" forever; A^μ assigns it ≈0 because the prover's success likelihood doesn't move).

### 1.2 Why the prover must differ from the base

- μ = π ⇒ the αA^π term in the effective reward reproduces exactly the outcome-only policy gradient (advantage already computed by PG).
- Too-strong μ ⇒ succeeds from any prefix ⇒ A^μ ≈ 0 everywhere (no distinguishability).
- Too-weak μ ⇒ fails from any prefix ⇒ A^μ ≈ 0.
- **Theorem 3.1 (informal):** E[V^{πt+1}−V^{πt}] ≳ γ·E_s Var_{a~πt}[A^μ(s,a)] + γ·E_s E_{a~πt}[A^μ·A^{πt}] — improvement grows with prover **distinguishability** and **alignment**. Weak provers can improve stronger bases (Prop F.1: complementarity η ⇒ Ω(η²) gain) — this is NOT distillation; the prover's ceiling does not bound the gain.
- BoK provers: Q^{BoK(π)}(s,a) = 1−(1−Q^π(s,a))^K; K=4 dominates across 2B/9B/27B bases; for 27B a *weaker* 9B prover beats the 27B itself.

### 1.3 Numbers

Search: beam with r_eff = Q^π + αA^μ (α∈[0.2,0.6]) is >8% more accurate and 1.5–5× more compute-efficient than ORM best-of-N. RL: dense r_eff (α up to 5.0, REINFORCE + token-value baseline, KL 0.001) gives 5–6× sample efficiency, +6–7% accuracy, 8× Pass@N. Data: ~300k (prefix, MC-Q) pairs per base, n_mc=20 partial rollouts/prefix, first-pit curation, Q-bucket class balance; n_cov>n_mc at low budget, n_mc>n_cov at high.

---

## 2. Path 0 decomposition (component table)

| # | Component | Coverage in stack | Signal-diff vs paper | Extraction |
|---|-----------|-------------------|----------------------|------------|
| 1 | A^μ = Δ success likelihood (cross-policy, outcome-grounded) | **partial** — `self_advantage_margin` (250, shipped `self_advantage_gate`) is the model's OWN pre/post-recursion logit ratio: single-policy, no outcome grounding; SDPG `centered_log_ratio` (Plan 180) is oracle-vs-student at bandit level — oracle is a *stronger teacher*, PAV's prover is deliberately *intermediate* | own-logit-change vs cross-policy outcome-delta; imitation direction vs complementarity | **yes** — MC estimate from logged outcomes = difference of two Bernoulli means; the trained PAV is only an amortization of it |
| 2 | Effective reward Q^π + αA^μ as test-time search score | **partial** — QGF tilt `logits + w·q_gradient` has the blend shape; `WidthSelectionMode::BestQ` (dd_tree) selects rollouts by Q | QGF per-state tilt is argmax-invariant under centering (V(s) is action-constant) — the fix applies ONLY to cross-state selection | **yes** — one FMA on two logged statistics; needs cross-state consumer |
| 3 | Distinguishability + alignment prover-selection stats (Thm 3.1) | **none** — we rank drafters by mean acceptance, rules by Elo, critics by strength; no variance+inner-product selector anywhere | strength-only ordering IS the paper's documented failure mode | **yes** — pure moments of logged per-(s,a) outcomes |
| 4 | BoK transform Q^{BoK}=1−(1−Q)^K | **covered** — `iid_at_least_one` (qmc_halter), `expected_max_over_m` (remax), `tamper_detection_probability` (rtdc): same closed form at 3 sites | direct Bernoulli-K success everywhere; **nobody differences it into an advantage or optimizes K** | K\* law derived here (§2.1) — removes the sweep |
| 5 | Effective-reward/blend coefficient α | **partial** — TETHER (riir-clippy 033, `selection_tether.rs`) fits a blend ρ from realized outcomes; QGF `DualLeoMixer` α-blends two heads | TETHER blends two *level* signals; PAV blends level + *difference* signal | **yes** — TETHER's least-squares fit is the natural α-estimator for r_eff too |
| 6 | First-pit attribution | **partial** — riir-ai 313 (step-level fault attribution guide, SkillAdaptor delta qualification) documents the concept; no generic runtime kernel ships | guide-level; game_sync kill-credit is claim-based, not temporal-changepoint | **yes** — thresholded first-crossing on logged Q̂ sequence |
| 7 | Dense per-step rewards for online RL | **n/a (track c)** — loss_grpo group-baseline notes "no external value model needed"; the paper's claim is a learned per-step baseline beats it 5–6× | — | **no** — needs the trained verifier → Plan 356 |
| 8 | Trained PAV (amortized MC) | **none** | — | **no** — but paper-conceded circumventable by direct rollouts (the modelless path we'd take first) |

### 2.1 Derived law: the interior K\* (novel, beyond the paper)

With a shared state baseline V and per-action Q, the BoK advantage is A_a(K) = (1−V)^K − (1−Q_a)^K. Treating the (V,Q) pair, dA/dK = 0 gives:

**K\* = ln( ln(1−Q) / ln(1−V) ) / ln( (1−V)/(1−Q) )**   (maximizes |A|; pairwise form — Var over actions inherits the same interior peak when V is shared)

Verified numerically: Q=.5, V=.3 → K\*≈1.98 (A(1)=.20, A(2)=.24, A(3)=.22 — peak at 2 ✓); Q=.2, V=.1 → K\*≈3.4. Limits behave: Q→V ⇒ degenerate (advantage vanishes, no interior peak); Q→1 ⇒ K\*→1. The paper's empirical "Bo4 dominates" is the population-aggregate of this per-context law. Use: pick best-of-K budgets (drafter retries, anytime-inference K, sampling best-of-K) from measured (Q,V) instead of sweeping.

> **T3 status (2026-08-27, Issue 692 T3, katgpt-rs `983c6fb8`):** the law is now gate-pinned in `katgpt_core::prover_selection` (`k_star` + `bok_advantage`; exhaustive both-halves (Q,V) grids assert argmax|A| ∈ {floor K\*, ceil K\*}). **Erratum to the anchors above**: anchor 1 confirmed exactly (K\*≈1.9747, peak 2), but anchor 2's "Q=.2, V=.1 → K\*≈3.4" is arithmetically off — the true value is **K\*≈6.371** with the empirical peak at K=6 (A(6)=.2693 > A(7)=.2686). The law itself holds everywhere on the grid; only this anchor line was wrong.

### 2.2 Honest negatives (consumers examined and rejected)

- **QGF / DualLeoOracle (civ):** the paper's centering fix does NOT apply — `tilt_logits` selects an action *within* a state, and subtracting the action-mean V(s) preserves argmax (the shipped T9/T10 correctness checks pin exactly this invariance for same-signal tilts). The fix only bites in *cross-state* selection, which QGF is not. And the civ critic axis is closed by riir-ai Research 322; the PAV prover theory does not resolve the Q-vs-forecast trap named there. Do not reopen.
- **riir-clippy within-pool candidate ranking:** candidates for one (rule, span) share the state ⇒ V(s) shared ⇒ centering is rank-invariant within the pool. Only the *cross-rule* fan-out pool (Issue 024's crowding regime) could benefit — and the trajectory store is thin-evidence-starved (98.5% of candidates unseen, Issue 039; starved pools make selection differences ≤0.04pp — the Issue 026 doctrine). A lift axis now would be noise, not signal. Revisit only after store densification (Issue 039's reopen trigger).

---

## 3. Fusion

**Fusion A — complementarity selector × rating/bandit (katgpt-core, open primitive):** a `prover_selection` module beside `rating`/`bandit`: given logged per-(context, action) Bernoulli outcomes for a base policy π and candidate provers {μ_i}, compute D(μ_i) + Al(μ_i) and rank provers by Theorem 3.1's bound. Consumers: (a) **drafter selection** — we rank drafters by mean acceptance (strength); the paper predicts ranking by D+Al instead, and that a *weaker but complementary* drafter/verifier pair can beat a stronger one; (b) **future QGF oracle consumers** (seal/quest — civ closed); (c) **riir-clippy rule ranking** — Elo is strength-only; D+Al is the complementary axis (post-densification). What none of the cousins alone gives: a *predicted-gain pre-gate* — skip wiring a prover when the bound says ≈0 gain, before any runtime cost.

**Fusion B — potential-difference scoring law × emotion kicks × self_advantage (design law, ships as convention):** the stack's change-based signals (emotion kicks on Δ, `hp_to_emotion_kick`, `decay_confidence`, self-advantage margin) are all potential differences — the paper proves this family is structurally immune to level-farming. Encode as the rule for every new scoring surface: **score the Δ, gate on the level** (level for admission/thresholds, difference for ranking/reward). The EVPI gate (riir-ai 738) is the binary extreme of the same family (fires iff the plausible set straddles the decision boundary = infinite-margin Δ); a continuous Δ-success variant is the natural interpolation if a consumer appears.

**Fusion C — first-pit kernel × fix-verify blame × kill-credit (katgpt-core + consumers):** generic `first_pit(q_seq, ε)` changepoint detector; consumer 1: when a multi-edit `--fix --verify` batch fails the compile gate, revert from the first pit forward instead of error-line matching alone; consumer 2: kill-credit temporal localization (which moment in a multi-step engagement turned the fight — a game-design call, opt-in). Complements `causal_id`'s counterfactual graphs with a temporal localizer.

**Fusion D — BestAdvantage rollout mode × dd_tree (smallest code delta):** `WidthSelectionMode::BestQ` exploits (re-picks the same high-Q tree); add `BestAdvantage` scoring rollouts by Q_i − mean_j Q_j (centering across the K rollouts = cross-state comparison within the fan-out). Gate: diversity+quality of selected paths vs BestQ at equal K.

> **Fusion D status (2026-08-27, Issue 692 T4): REFUTED BY MECHANISM — no code shipped.** The K rollouts of one `best_of_k_rollouts` call form a **single within-state pool** (they share the same marginals/prefix), and `mean_j Q_j` is rollout-independent, so `argmax_i (Q_i − mean_j Q_j) ≡ argmax_i Q_i` — selection identical to `BestQ` at every seed, by the same rank-invariance argument §2.2 used to reject the QGF and riir-clippy consumers. The per-depth-centered steelman (`Σ_d (q_{i,d} − mean_j q_{j,d})`) also collapses: all rollouts share the depth count, so the baseline sums to one constant shift. "Cross-state" bites only when candidates come from *different* states (the bench-684 beam-retention direction) — the dd_tree fan-out is not that. The enum arm is NOT added (no-op API surface). Record: [Bench 684](../.benchmarks/684_prover_selection_goat.md).

---

## 4. Verdict

### **GOAT** (Gain — files created)

**One-line:** the prover-advantage framework is the canonical prior art for process rewards, but its *modelless half* — complementarity selection statistics (Thm 3.1 as a pre-gate), cross-state advantage centering, the derived K\* law, first-pit attribution — ships nowhere in our stack, and one honest application (strength→complementarity selection inversion) touches three existing surfaces.

### Novelty gate (honest)

| Q | Criterion | Answer | Notes |
|---|-----------|--------|-------|
| Q1 | No prior art? | **NO** | The paper itself (339 cites; AgentPRM, PRIME, PRM survey follow-ups) is the mechanism's prior art. Our delta: the D+Al selector as a *deployed inference-component ranking* (drafters/rules, not RL provers) + the K\* closed form (derived, beyond the paper). |
| Q2 | New behavior class? | Partial | Complementarity-based selection inverts a ranking policy; measurable, not a new capability class. |
| Q3 | Product selling point? | Uncertain | "Our runtime picks its critics by complementarity, not strength" needs a measured consumer (drafter selection is the cheapest honest test). |
| Q4 | Force multiplier? | **YES** | Connects rating/bandit/sampling (katgpt-core), self_advantage (250), TETHER (033), dd_tree BestQ, clippy self_evolve/Elo, cgsp, fix-verify. |

Not all 4 YES → not Super-GOAT. Gain with issue + plan.

### Why not PASS (reverse-grep evidence)

- Issue 039 measured 98.5%-unseen candidates and *deferred selection work* on store densification — the lift/advantage axis is the documented follow-up shape once density lands.
- dd_tree ships `BestQ` — the paper's exact diagnosed exploit-only selector, one enum arm from the fix.
- riir-clippy L4 fixer G1 is a measured 0/60 (Bench 465/467) with 9/29 reachable misses (Bench 037) — the PAV edit-scorer (Plan 356) attacks a live documented gap.
- No-GD advocate's items 1/3/6 have zero shipped analogs (grep-verified).

### Routing

| Artifact | Destination | Status |
|----------|------------|--------|
| Research note (this file) | `katgpt-rs/.research/509_*.md` | ✅ |
| Open primitive issue (D+Al stats, K\* law, first-pit kernel, potential-difference law, BestAdvantage mode) | `katgpt-rs/.issues/692_*.md` | ✅ |
| Training plan (PAV edit-scorer → clippy L4; dense per-edit rewards; BoK relabel; token-value baseline in loss_grpo) | `riir-train/.plans/356_*.md` | ✅ |
| PASS-Redirects to cousins | n/a (Gain, not PASS) | — |

---

## 5. PoC obligation (defend-wrong, for whoever picks up 692)

Any quality claim ("complementarity selection beats strength selection for drafters") needs a head-to-head in `riir-poc`-style harness: paper-arm (D+Al-ranked prover), frozen baseline (strength-ranked), shipped analog — on a controlled toy (e.g., the Go-puzzle or doc-repro speculative harness), verdict table printed. Latency/architectural claims need only a bench. The K\* law needs only the exhaustive Bernoulli sweep gate (verify peak == closed form across a (Q,V) grid) — that one is pure math, no PoC.

> **T5 status (2026-08-27, Issue 692 T5, katgpt-rs `1b65662f`): DISCHARGED — GOAT PASS, T1 promoted to DEFAULT-ON.** [Bench 684](../.benchmarks/684_prover_selection_goat.md) runs the exact three-arm shape above on a controlled PAV harness (64×8 Bernoulli logs, cross-state beam retention, 16 seeds × α ∈ {0.2,0.4,0.6}): the D+Al pick (peer, strength 0.49) beats the strength pick (flat 0.95 solver) at every α (mean +0.005–0.007 retained-θ, 75–87% cell win-rate), beats the shipped no-prover baseline at α ≥ 0.4, and the anti-aligned prover's bound goes negative (pre-gate rejects). Honest α=0.2 noise-floor finding recorded in the bench. T4 (dd_tree BestAdvantage) refuted by mechanism — see the Fusion D correction above.

## 6. Cross-references

- `katgpt-rs/.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md` — self-advantage (single-policy cousin, shipped)
- `katgpt-rs/.plans/180_sdpg_bandit_modelless.md` — SDPG `centered_log_ratio` (oracle/student cousin)
- `katgpt-rs/.research/373_ReMax_Expected_Max_Retry_Aggregation.md` — same Bernoulli-K closed form
- `katgpt-core/src/speculative/qmc_halter.rs::iid_at_least_one` — the shipped BoK transform
- `riir-clippy/src/selection_tether.rs` — adaptive blend (level-signal family)
- `riir-ai/.research/322_civ_alternative_critic_post_stop_rule_verdict.md` — the closed critic axis (honest negative documented §2.2)
- `riir-ai/.research/341_nesy_rag_evpi_gated_active_perception.md` — decision-flip cousin (binary extreme)
- Setlur et al. 2024, arXiv:2410.08146; Ng et al. 1999 (potential-based shaping); Snell et al. 2024 (beam search with PRMs)
