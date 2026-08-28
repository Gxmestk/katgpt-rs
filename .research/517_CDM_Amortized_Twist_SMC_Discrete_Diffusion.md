# Research 517: CDM — Amortized Twisted SMC for Reward-Tilted Discrete Diffusion

> **Source:** [Contrastive Distribution Matching for Amortized Sequential Monte Carlo in Discrete Diffusion](https://arxiv.org/abs/2605.23346) — Jaihoon Kim, Taehoon Yoon, Prin Phunyaphibarn, Seungjun Kim, Morteza Mardani, Minhyuk Sung (KAIST / UMich / NVIDIA), arXiv:2605.23346, 2026. [Project](https://cdm-smc.github.io) · [Code](https://github.com/KAIST-Visual-AI-Group/CDM) (MIT)
> **Date:** 2026-08-28
> **Status:** Active — verdict Gain; two plans opened
> **Related Research:** 505 (Mean-Field Distributional Steering — the closest cousin; ships the SMC/tilt substrate), 290 (Latent Field Steering — pointwise special case), 236 (QGF — per-action reward-gradient tilt), 248 (BoM single-pass diverse sampling — the no-per-step-access floor), 369 (Renoise-CE — re-noising as a scoring operator), 034 (D2F — our discrete-diffusion decode substrate)
> **Related Plans:** 577 (distributional_steering primitive — landed, G1 partial FAIL, opt-in), 581 (twist_smc — opaque-reward steering + modelless amortization, opened this session)
> **Cross-ref (riir-train):** Plan 361 (CDM contrastive twist head — the model-based arm, Path 0.5)
> **Classification:** Public

---

## TL;DR

Twisted SMC samples a discrete diffusion model from a reward-tilted target `p*_t ∝ p^base_t · ψ*_t` (ψ* = exponentiated soft value `E[R(x₀)|x_t]`), but discrete state spaces have no Tweedie shortcut, so prior work estimates the twist per (particle, step) by `M` Monte Carlo rollouts + reward queries — the cost that dominates inference (up to ~50×). CDM amortizes the twist into a tiny scalar head on the **frozen** denoiser's final features, trained once with a **contrastive** (positive = target samples, negative = model samples) forward-KL objective, using the **closed-form forward kernel** to re-noise a buffer of clean positives across many timesteps (many gradient updates per expensive reward query). Inference: one backbone pass yields denoising logits AND the twist, <5% overhead (down to 0.5%); proposal-agnostic (composes with RL/LoRA-fine-tuned proposals); preserves diversity where fine-tuning mode-collapses.

**Distilled for katgpt-rs (modelless, inference-time):** the paper's *correctness* lives entirely in closed forms — SMC weights, ESS, the forward kernel `q(x_t|x₀)` — while the trained head is a variance-reduction overlay (any positive ψ keeps the estimator consistent). The modelless slice: steer sampling toward an **opaque** (black-box) scorer — which the shipped `distributional_steering` Table-2 rewards (Linear/Moment/Mmd, all closed-form `δR/δμ`) do not cover — via (i) an **x̂₀ posterior-mean reward proxy** (the denoiser already emits `p(x₀|x_t)`; one reward query replaces `M` rollouts), (ii) **state-keyed value memoization** (resampled particles revisit prefixes), (iii) a **one-shot ridge / kernel-table twist** fit offline from cached (features, R) pairs. The trained-head arm is Plan 361 (riir-train), and the two arms share one GOAT gate at matched reward-query budget.

---

## 1. Paper Core Findings

1. **Twist bottleneck is real and structural in discrete diffusion.** Continuous diffusion gets a cheap plug-in twist from Tweedie's formula; discrete models need `M` rollouts per (particle, step). Inference cost grows ∝ M·K·T reward queries; CDM's scaling curves beat SMC/BoN/Soft-Value at matched wall-clock across 4 domains (toxic text / DNA enhancer / protein designability / dLLM alignment).
2. **Contrastive > regression for twist learning.** Gradient = `E_{p*_t}[∇log ψ] − E_{p^φ_t}[∇log ψ]`; the negative term improves convergence vs the regression-only Soft Value baseline (Li et al. 2025) at matched training budget.
3. **Forward-kernel buffer reuse decouples reward cost from updates.** `p*_t(x_t) = Σ_{x₀} p*_0(x₀) q(x_t|x₀)` with closed-form `q` — clean positives from buffer `B` re-noise across all `t`; many updates per reward query. (Authors: structurally unavailable to autoregressive LMs.)
4. **Head placement is the whole inference story.** Scalar head on final features → one pass gives logits + twist; 5%→0.5% overhead by head shape (MLP / MLP+PE / small Transformer).
5. **Diversity: twisting ≠ fine-tuning.** SMC over the frozen base keeps Self-BLEU/PPL/cluster-count healthy; d1/DRAKES arms mode-collapse while CDM wins reward AND diversity. Composes additively with fine-tuned proposals.
6. **Honest instability caveat (their own README):** the contrastive loss is unbounded, grad-clip is null, late-training diverges — the released DNA head is **epoch 7 of 500**, selected by downstream sampled reward, not loss.

## 2. Path-0 decomposition + coverage (signal-diff per §3.6)

| CDM component | Ships? | Where / delta |
|---|---|---|
| SMC shell: particles, incremental log-weights, ESS-guarded resampling | **YES** (arch.) | `distributional_steering.rs` (Plan 577/Bench 682): `FkStepper` FK weights, `systematic_resample_into` / `residual_resample_into` (sampling consumers only), ESS guard in the demo; `speculative/qmc` resampling strategies. Signal-diff: same mechanism class. R505 caveat 2 documents the resampling-vs-persistent-agents split. |
| Closed-form twist for measure-rewards | **YES** | `MeasureReward` rows Linear/Moment/Mmd (paper's Table 2 analog). **Signal-diff vs CDM:** Ψ = δR/δμ needs a *closed-form, measure-defined* reward; CDM's ψ_t = `E[R(x₀)|x_t]` serves an *opaque external scorer* (classifier/ESMFold) — a coverage gap, not a mechanism gap. |
| Opaque-reward per-(particle,step) MC twist (the 50× cost) | **NO** | Modelless extraction (No-GD advocate): x̂₀ posterior-mean proxy; state-keyed value memo (papaya + BLAKE3 key); one-shot ridge/Nadaraya-Watson table. All closed-form; any positive ψ stays consistent. |
| Amortized twist head (scalar head, frozen backbone, <5%) | **NO** (model-based) | riir-train Plan 361: contrastive `loss_twist` + forward-kernel buffer + head-shape sweep. |
| Contrastive pos/neg objective | **NO** (training) | Structurally similar pair-loss shape to `loss_grpo/loss_dpo`; self_evolve's accepted/reverted trajectories are the same pos/neg signal shape. |
| Forward-kernel re-noising reuse | **PARTIAL** (inference op) | `dllm_solver` Q-Sample (Plan 222) re-noises to *refine*; `renoise_ce` (R369/P406) re-noises to *score*. Re-noising to *label* a buffer (carry R(x₀) to many t) is the missing third use — trivial given `q` is closed-form. |
| Select-by-downstream-reward (epoch-7 lesson) | **PARTIAL** | House GOAT gates already select by downstream metric; edge_lora `arena_eval` is the harness precedent. Formalize for twist checkpoints. |
| Diversity preservation vs fine-tuning | **YES** (validated) | SMC-over-frozen-base + `entropic_tilt` KL budget bound concentration by construction; confirms the house sigmoid/KL-budget stance. Not new. |

**Adversarial panel (2 advocates, merged):** the No-GD advocate's rows 1–13 (SMC consistency theorem, ESS scheduling law, deterministic resampling kernels, re-noising-as-inference-op, memoization, non-parametric/ridge tables, x̂₀ proxy, β-family sweep, diagnostics-as-gates, stability ledger) and the model-based advocate's rows 1–9 (twist head in the dLLM/GDSD trainer, `loss_twist.rs`, trajectory-store buffer, head sweep, reward-based checkpoint selection, composition study, **code-repair validator-composite reward — an exact free oracle**, AR variant on quest_grammar, learned-reward-head-last) are folded into the two plans. One advocate finding discarded with reason: none — both tracks landed in plans.

## 3. Fusion

The novel combination this paper triggers in-stack: **R505's SMC/tilt substrate (population steering, closed-form rewards) × opaque black-box rewards × amortization** — where amortization comes in two competing arms under ONE gate:

- **Modelless arm (Plan 581, katgpt-rs):** `x̂₀` proxy (1 query/particle/step) + state-keyed value memo + one-shot ridge table + β/KL-budget selection. Zero gradients; deterministic; replayable.
- **Model-based arm (Plan 361, riir-train):** contrastive twist head on the frozen dLLM/GDSD backbone; buffer fed by the closed-form forward kernel + the self_evolve trajectory store (reward = validator verdict); deployed as reward-tilted SMC over **generated code fixes** — a domain where the reward oracle (parses ∧ validator-pass ∧ lint-delta) is deterministic and free, and where the healer's actual bottleneck (diverse candidate enumeration under a hard validator) lives.

Neither arm alone is the paper; the paper's own evidence says the trained head must **earn its keep against the table** (their artifact was an early checkpoint of an unstable loss — the stability ledger favors the modelless slice as the default, the head as the variance-reduction upgrade if it wins the gate).

## 4. Verdict

**Tier: Gain** — incremental over shipped substrate, useful, not headline-worthy for the stack.

- Q1 no prior art? **NO** — the paper + twisted-SMC line (Zhao et al. 2024) + Soft Value (Li et al. 2025); our R505/P577 ships reward-tilted SMC machinery (Bench 682).
- Q2 new behavior class? **NO** — reward-steered sampling exists in-tree (closed-form rewards); the delta is the reward *class* (opaque) and the cost *curve* (amortized), not a new capability.
- Q3 product selling point? **PARTIAL** — "steer any scorer at <5% overhead" needs the trained head; the modelless trio is infrastructure-grade.
- Q4 force multiplier? **YES** — R505 + BoM + qgf + dllm_solver/d2f + self_evolve store + healer candidate generation.

**MOAT gate (katgpt-rs):** fits — sampling-slot primitive via fusion; public distillation stays here (paper math is public); the trained-head recipe is private (riir-train Plan 361). Opt-in features both arms; promote-if-gain, demote-loser vs BoM/R505 arms applies.

**Routing:** Research note here (public) · Plan 581 (katgpt-rs, modelless `twist_smc`) · Plan 361 (riir-train, `twist_head`) · no `.issues/` (no standalone PoC — the GOAT gates inside the plans are the PoCs).

## 5. Honest caveats

- Web-search MCP was rate-limited this session; prior-art check ran via DDG fetch — it surfaced only the CDM paper + mirrors; the published landscape inside the paper's own references (twisted SMC, DRAKES, d1, Soft Value) is treated as the prior-art boundary. A deeper sweep (§4.5) not run.
- Their DNA finding is load-bearing for BOTH plans: unbounded contrastive loss + null grad-clip ⇒ late divergence; **select by downstream sampled reward, clamp ψ-logits, clip grads** — encoded as hard requirements in Plan 361 and as the β/KL-budget selection rule in Plan 581.
- UQ discipline: the weighted empirical measure is a distribution claim only if we *market* it as one; both plans keep it a ranking/steering signal until any UQ claim is conformal-floor-gated (Plan 340 floor rule).
