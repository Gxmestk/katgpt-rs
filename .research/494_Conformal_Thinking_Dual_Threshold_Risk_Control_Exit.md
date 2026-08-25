# Research 494: Conformal Thinking — Dual-Threshold Risk-Controlled Exit

> **Source:** "Conformal Thinking: Risk Control for Reasoning on a Compute Budget" [arXiv:2602.03814](https://arxiv.org/abs/2602.03814) — Xi Wang, Anushri Suresh, Alvin Zhang, Rishi More, William Jurayj, Benjamin Van Durme, Mehrdad Farajtabar, Daniel Khashabi, Eric Nalisnick (JHU + Apple), ICML 2026.
> **Date:** 2026-08-20
> **Status:** Active (plan opened: [Plan 575](../.plans/575_risk_controlled_exit_primitive.md))
> **Related Research:** 322 (conformal pools — the floor rule this note extends to exits), 266 (FPRM halting), 282 (LoopCoder gain-cost halting), 344 (implicit fixed-point halting), 052 (SimpleTES evaluation-driven scaling), 316 (DSpark confidence-scheduled admission), 243 (Bebop entropy-bounded acceptance), 255 (CLR reliability)
> **Related Plans:** 340 (ConformalIntervalCalibrator — the calibration substrate), 304 (GainCostLoopHalter — the exit consumer), 575 (this paper's plan)
> **Cross-ref (riir-ai):** Research 339 (per-NPC risk-controlled think-budget guide)
> **Classification:** Public (open primitive note; game wiring lives in riir-ai Research 339)
> **PASS-Redirects (synthesis):** Valluri, Nguyen & Grover [arXiv:2608.20359 "Self-Speculation for Faster Reasoning Models"] — the complementary half of this note's partial-CoT convergence premise: where Conformal Thinking/Plan 575 EXITS on budget-controlled confidence trajectories (halt early, save tokens), SSR DRAFTS from them (reuse the partial-CoT answer distribution as a speculation source, verified against the full-budget distribution, same model, training-free). PASS for us: the exit half is what ships (`RiskControlledExit` plan + SWIR `</think>`/ForceAnswerPrefix budget control on qwen38 — we pre-fill the empty think block, non-thinking branch, so no live CoT phase has ever been decoded); the draft half additionally needs concurrent request streams to hide the model-drafter behind ongoing CoT (our engine is single-stream batch-1 CUDA-graph decode by design) and loses economically to the shipped ~free lookup drafter (15.15/16 acceptance, riir-ai Issue 742). Their suffix-only-beats-prefix ablation (1.318× vs 1.086×) is the interesting datapoint for any future span-recovery work. Reopen trigger: a live-thinking-mode chat consumer.

---

## TL;DR

The paper converts "when should a compute loop stop?" from a hand-tuned threshold into an **interpretable risk budget with a finite-sample guarantee**. Two exits: an **upper threshold** (stop when confident — risks false positives) and a novel **parametric lower threshold** — a squeezed sigmoid `λ−(t) = σ(c(ωt − sB), l, u)` acting as a *confidence schedule* that stops hopeless instances early (stop-when-not-progressing). Both thresholds are calibrated on a validation set via distribution-free risk control (UCB + Hoeffding correction: `Risk̂ + √(log(1/δ)/2n) ≤ ε`).

**Distilled for katgpt-rs (modelless, inference-time):** The entire load-bearing mechanism is training-free statistics + decision math. The stack today splits into two halves that have never been joined: **calibrated thresholds stop nothing** (Plan 340 conformal floor feeds detection/intervals only) and **stopping mechanisms aren't calibrated** (Plan 304's τ=1.0, FPRM τ=0.1, DEQ ε=0.05, Bebop entropy gate, MCTS fixed-512 — all hand-set). This paper is the join: `RiskControlledExit` — a generic dual-threshold exit policy over ANY confidence trajectory, with offline UCB calibration. The lower threshold is literally the house sigmoid primitive (`sigmoid`, never softmax) used as a scheduled exit bound — a shape no shipped mechanism has.

---

## 1. Paper Core Findings

1. **Reframe budget-setting as risk control.** Adaptive reasoning ("stop when confidence ≥ λ") doesn't remove the budget hyperparameter — it renames it to a threshold, which is *less* interpretable (signal-dependent, arbitrary range; their Fig. 1 left). User-facing quantity should be an **error tolerance ε**, mapped to thresholds automatically with finite-sample protection.
2. **Dual-threshold exit.** `τ = min{t ≥ 1 : s̃t ≥ λ+ ∨ s̃t ≤ λ−(t)}`, λ+ > λ−, exits mutually exclusive. Upper = confident success (saves post-convergence tokens). Lower = confident failure (saves tokens on unsolvable instances that would otherwise burn the full budget).
3. **Parametric dynamic lower threshold** (the novel mechanism): `λ−(t; c, s, l, u) = σ(c(ωt − sB), l, u)` where `σ(z,l,u) = (u−l)/(1+e^−z) + l`, B = total token budget, ωt = tokens used at step t. The model must raise confidence **on a schedule** to earn the right to keep reasoning. Shape family: linear (s=0.5, cB≪1), exponential (s>1), log (s<0), constant (c→0). Calibrated parameter: {c, s, l}.
4. **Four losses** (all ∈ [0,1]): correctness — FP loss `I[s̃t ≥ λ+]·I[ft ≠ y*]` (upper), farsighted FN loss `I[s̃t ≤ λ−]/(T−t+1) · Σk≥t I[fk = y*]` (lower — checks ALL future solutions, so stopping a would-be-solved instance costs more); efficiency — normalized regret `J+ = max(0, t−t′)/T` (tokens wasted after first-correct) and past-wrongness `J− = Σk≤t I[fk ≠ y*]/T`. Efficiency picks among *feasible* candidates only.
5. **UCB calibration** (Bates 2021 / Jazbec 2024 lineage): feasible iff `Risk̂(λ;V) + √(log(1/δ)/2n) ≤ ε`. Naive cross-validation frequently violates the target on resampled validation sets (their Fig. 4); UCB stays under y=x. Two-step decoupled selection: pick λ+ first at ε+, then c conditioned on λ+ at ε−.
6. **Empirics:** Qwen3-8B/30B-A3B, DeepSeek-R1-32B, Qwen3-VL-8B on AIME/DeepScaleR/GPQA/MathVision. (a) UCB controls risk where naive fails, worse for small validation sets; (b) ensembling signals (pick most efficient feasible signal per ε) beats any fixed signal; (c) **lower-threshold gains grow with unsolvable fraction** — at 3:1 solvable:unsolvable, upper-only captures most savings; at 1:1 and 1:3, upper-only clusters at the full-budget wall while dual-exit shifts the whole accuracy/token curve left (their Fig. 6). Solvable instances exit via upper; unsolvable via lower.
7. **Failure modes (their own honesty):** monotonicity of risk in (λ+, c) is *assumed*; two-step decoupling loses some rigor; lower threshold is length-shift sensitive (schedule shape depends on horizon B); under distribution shift where the lower threshold filters proportionally more correct than incorrect (p_i < p_c), the upper guarantee breaks (their App. C).

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent (grep set) |
|---|---|
| upper threshold / stop-when-confident | acceptance gate, commit-now, `confidence_threshold`, MCTS "confident line" (absent) |
| lower threshold / stop-when-hopeless | stall counter, oscillation halt (`cos θ < 0`), `EarlyStopGate` (lower-only), RefusedFloor |
| confidence schedule | adaptive γ (Bebop), survival-prob admission (DSpark), `adaptive_threshold` (ppot) |
| risk control / UCB / Hoeffding | conformal quantiles (Plan 340), union bound (`QmcHalter`), UCB1 bandits |
| normalized regret | gain/cost halting (Plan 304 `halt_decision(gain, cost, τ)`) |
| validation-set calibration | residual pools, `AcceptanceForecastH2` regression, EMA calibration |

### 2.2 Signal-diff vs shipped cousins (the containing expression, not the name)

| Cousin | Consumes | Emits | Missing vs paper |
|---|---|---|---|
| `ConformalIntervalCalibrator<F>` (Plan 340, `katgpt-core/src/conformal/`) | forecast residuals | coverage interval `[point+q_lo, point+q_hi]` | **No exit decision consumes it.** Interval-UQ ≠ decision-risk. Nearest substrate for the calibration half. |
| `GainCostLoopHalter` (Plan 304, `gain_cost_halt.rs`) | step-size/angular change + erank | halt when `gain < cost·τ` (τ=1.0 hand-set), oscillation halt | Two hopeless-side halts, no confident-stop, **no guarantee**. Fusion consumer #1: τ → risk-calibrated bounds. |
| `EarlyStopGate<P>` (Plan 083, `speculative/types.rs`) | screening relevance | zero relevance when `inner_rel < confidence_threshold` (default 0.0 = disabled) | Lower-only, uncalibrated, heuristic prune not exit. |
| `QmcHalter` (`speculative/qmc_halter.rs`) | hit count, k, event prob | union-bound coverage ceiling `min(1,kp) ≥ target` | Finite-sample bound but single-sided ceiling for sampling, not exit risk; no dual, no schedule. |
| `AcceptanceForecastH2` (Bebop, `ict/bebop_upgrade.rs`) | entropy → acceptance regression | EMA adaptive-γ draft length | Regression-calibrated (ECE-fit), not distribution-free; single direction; adaptive-γ = open Issue 023 (unproven) — **this paper's UCB makes it provable**. |
| `PrefixScheduler` (DSpark, Plan 339) | survival-prob trajectory `Πc_i` | non-anticipating admission + Θ-drop early stop | Confidence-scheduled admission, calibrated by fit; throughput-direction only. |
| `MctsSearchBudget` (`mcts.rs`) | — | fixed 512 advances / rollout cap | **Budget-exhaustion-only. No confidence stop of any kind.** |
| `SwarmDeliberationSystem` (riir-ai, Issue 054) | 40-tick stall counter | fixed 8-dir × horizon-10 search | Fixed trigger, fixed search; the game-side consumer documented in riir-ai Research 339. |
| riir-clippy `BetaPosterior` selection | per-candidate S/F counts | ε-quantile of Beta(1+S,1+F) | Different axis (candidate selection, thin-evidence regime) — but `EvolveRecorder` already records labeled outcomes = **ready-made validation set** for a heal-loop exit gate. |

**Nobody combines: confidence trajectory + risk-calibrated dual thresholds (UCB+Hoeffding) + parametric sigmoid progress schedule.** The gap is the composition, not any single primitive.

### 2.3 Fusion

- **× Plan 340 (conformal floor):** the floor rule ("Report the Floor") currently governs interval-UQ primitives. This paper extends the same philosophy to **exit decisions**: the floor for an exit primitive = fixed-budget exit (the paper's own token-based baseline, Fig. 6) + single-threshold hand-tuned exit. A risk-controlled exit must hold realized risk ≤ ε at *strictly better efficiency* than both floors, else G2 FAILs.
- **× Plan 304 (gain-cost halter):** `halt_decision(gain, cost, τ)` keeps its shape; τ stops being a constant and becomes the UCB-calibrated dual bound. The oscillation halt gains a confident-stop sibling.
- **× Bebop Issue 023 (adaptive-γ):** the entropy→acceptance schedule gets a distribution-free guarantee instead of an unproven regression — closes a standing open issue class.
- **× MCTS:** budget-exhaustion-only termination → dual exit (confident-line stop + hopeless-position stop), labels from self-play outcomes (free in-game).
- **× riir-clippy:** fixpoint-pass exit + draft-selection stop; `EvolveRecorder` outcomes are the calibration set; risk = reverted-edit rate ≤ ε.
- **Game-side (riir-ai Research 339):** per-NPC think budgets — upper exit = commit action; lower exit = abandon deliberation, fall back to the reactive L0 layer. Crowd framing: the paper's Fig. 6 solvable-ratio result maps to crowd composition — trivial decisions exit fast (upper), stuck NPCs free the 20Hz tick budget early (lower).

### 2.4 Why the game setting fits BETTER than the LLM setting

The paper's worst failure mode — lower-threshold length-shift sensitivity (their Fig. 8) — is structurally milder here: the game tick budget B is a **design constant** (fixed horizon), not a random reasoning length. Labels (win/lose, kill credit, quest completion, compile verdicts) are free at runtime. Distribution-shift caveats remain real (App. C p_i ≥ p_c monitor needed as a runtime tripwire).

## 3. Verdict

**Super-GOAT.** Novel composition in-stack (4/4): (1) zero prior refs + audits prove the calibrated-stop gap; (2) new capability class — *guaranteed* error-rate compute halting (all shipped halters are hand-set τ); (3) selling point: **"Our NPCs and engine loops stop thinking the moment further thought is provably unlikely to help — at a user-specified risk budget with finite-sample guarantees"**; (4) force multiplier across Plan 340 + Plan 304 + MCTS + Bebop/023 + DSpark + deliberation + riir-clippy.

**MOAT gate:** katgpt-rs — fundamental decision-math primitive (dual threshold + UCB calibration + sigmoid schedule), leaf-clean, no game semantics → **in scope**. Game/engine wiring → riir-ai Research 339 (private). Heal-loop consumer → riir-clippy (private).

**Outputs:** open primitive (Plan 575, katgpt-core `risk_control_exit`), private guide (riir-ai Research 339), this note.

### Honest caveats (defend-wrong posture)

- **Quality parity unproven here.** The paper's numbers are LLM reasoning; ours will be game/loop domains. Plan 575's GOAT gate carries the PoC: dual-exit vs fixed-budget floor vs single-threshold floor on (a) MCTS toy, (b) loop-halting bench, (c) heal fixpoint. Until then, only the architectural + statistical axes are claimed.
- Monotonicity of risk in (λ+, c) is assumed, not proven — must be verified per consumer (the calibration harness plots risk-vs-hyperparam curves and refuses non-monotone spans).
- Two-step decoupling (λ+ then c) trades rigor for practice — record as accepted.
- UCB union-bound over the grid: paper uses the simple correction; for wide grids use δ/|G| (multiple-comparison variant) — encode both.
- Probe signal (their 2-layer MLP) is the only trained component — out of scope for the open primitive; the modelless substitute is a sigmoid projection onto belief direction vectors (house style).

### Floor-rule extension (binding for Plan 575)

This is a UQ-bearing primitive (claims a risk bound). **Exit-floor rule:** benchmark against (a) fixed-budget exit and (b) single-threshold hand-tuned exit on BOTH axes — realized risk (must hold ≤ ε) and efficiency (must beat both floors at matched risk). Borrowing the paper's Fig. 6 protocol: sweep ε, plot accuracy-equivalent-vs-compute for all three policies.
