# Research 503: Flow-Matching Anomaly Detection — Scoring Function Choice Beats the Model

> **STATUS 2026-08-26 — riir-train Plan 350 CLOSED as a NEGATIVE RESULT.** Both arms built (TCCM 3x256 contraction field; Forest-Flow in-tree GBM) and measured on riir-chain Issue 112 T1 fixtures against its registered classical floor: best arm AUROC 0.9365 / AUPRC 0.2953 vs floor 0.9333 / 0.3564, gate 0.9633 / 0.3864, 0/3 seeds clearing. The learned fields reach PARITY on ranking and LOSE on AUPRC by 0.061 — the classical floor stays the detector. The paper's central claim (trajectory scores beat single-point Decision under contamination) did NOT reproduce on 8-lane sparse integer deltas: all three scores within ~0.01 AUROC, Decision the single best cell. See riir-train `.benchmarks/529_plan350_flow_anomaly_arms.md`.

> **Source:** "Unsupervised Anomaly Detection Using Flow Matching on Tabular Data" — [arXiv:2608.19801](https://arxiv.org/abs/2608.19801), Philip Konz, Tejaswini Medi, Margret Keuper (Univ. Mannheim / MPI-INF), 2026-08-20
> **Date:** 2026-08-24
> **Status:** DISTILLED — actionable deltas filed (riir-chain Issue 112, riir-train Plan 350)
> **Related Research:** 420 (VFD velocity-field disagreement — trajectory-aggregated flow score), 369 (renoise-CE perturb-re-resolve drift), 375 (kernelized stochastic-interpolant VelocityFieldEnsemble), 099 (eigenspace structural anomaly), 200 (quantization outlier collapse), 364 (TabFM tabular foundation — Pass), 322 (conformal floor / "Report the Floor")
> **Related Plans:** riir-train 350 (tabular flow-matching anomaly arms), katgpt-rs 340/432 (floor + VFD substrates)
> **Cross-ref:** riir-chain Issue 112 (anomaly-detector poisoning hardening)
> **Classification:** Public (general scoring methodology; no game/chain IP in the distilled principles)

---

## TL;DR

The paper's headline is an evaluation finding, not a new model: for flow-matching anomaly detectors, **the anomaly-scoring function matters more than the model**, and trajectory-aggregated scores (summed velocity error along a probe path; Euler-integrated endpoint error) are robust to training-set contamination where the original single-step score collapses. Distilled modellessly, the entire scoring geometry survives without any learned field — the TCCM target law (`v = −(x−μ)`) is closed-form, and Forest-Flow's linear transport has constant exact velocity (`x−z`) — leaving **aggregation extent** as the real robustness knob. The stack already ships the mechanism-level cousins (`VelocityFieldEnsemble` default-on, VFD, `renoise_ce`, deterministic replay), but **riir-chain's `AnomalyDetector::is_anomalous` is a Welford single-point scorer — exactly the paper's contamination-sensitive "Decision score" design class** — and the stack contains **zero robust statistics** (median/MAD) and has never registered recall of its anomaly detectors on planted anomalies.

**Distilled for katgpt-rs (modelless, inference-time):**
The scoring-function taxonomy transfers as a *design law*, not a new primitive: (1) aggregate a consistency signal over a trajectory/window rather than at a single point — coherent deviations scale with M while noise scales √M, and contamination bias is bounded by the corrupted fraction (detectability `d′ ≈ √M·s̄/σ_eff`; tolerance `κ_max = s̄/b̄`); (2) probe points must stay in-distribution (the paper's nt=5>nt=20 finding; our QGF one-step projection already applies this); (3) `‖r‖² = Σⱼrⱼ²` is an exact per-feature attribution identity — no Shapley needed; (4) robust references (componentwise median, breakdown 0.5) bound reference-poisoning influence in closed form, strictly stronger than the paper's empirical contamination invariance.

---

## 1. Paper Core Findings

Setting: fully-unsupervised tabular anomaly detection (financial transactions; anomalies present in training at ρ = 0.786%–11.27%). Two model families:

- **TCCM** (Li et al., arXiv:2510.18328, NeurIPS 2025): one MLP with sinusoidal time embedding, trained so `f_θ([x; Embed(t)]) ≈ −x` — a contraction field toward the origin (origin = stable fixed point). Deterministic, no Monte Carlo.
- **Forest-Flow** (Jolicoeur-Martineau et al., AISTATS 2024): nt XGBoost regressors, one per discrete time level on the linear interpolation `x_t = (1−t)z + t·x` with constant target velocity `v_true = x − z` (noise→data transport). K Monte Carlo noise draws at scoring time.

Three scoring functions (the actual contribution):

| Score | Definition (TCCM form) | Contamination behavior |
|---|---|---|
| **Decision** | `‖f_θ(x,1) + x‖₂` — single-step residual at one time point | Sensitive: any local bias from contaminated training directly corrupts the score (B2B: 0.697→0.488 AUROC zero→full contamination) |
| **Deviation** | `Σ_i ‖f_θ(x(t_i),t_i) + x(t_i)‖²` over the probe path `x(t_i)=(1−t_i)x`, early half only | Stable (0.679→0.476 still degrades; FF variant better) |
| **Reconstruction** | Euler-integrate the field to the endpoint; score terminal error (distance from origin) | **Most robust + most hyperparameter-stable** (TCCM on B2B: 0.869→0.846; FF-XGB best on Campaign/Waveform) |

Mechanism (paper's own explanation): contamination corrupts only *parts* of the learned vector field; aggregating prediction errors along the trajectory reduces the influence of local deviations and emphasizes overall consistency of the learned dynamics. Small local inaccuracies compound coherently for anomalies (systematic endpoint deviation grows ~linearly with horizon) but average out for normal samples.

Secondary findings: for TCCM Deviation, **fewer time steps is better** (nt=5 > nt=20 — probe points stay near the training distribution; off-distribution probes inject noise); Reconstruction is insensitive to Monte Carlo K (trajectory aggregation already stabilizes — deterministic aggregation suffices); per-feature residuals give exact explainability; residual-based scores are Lipschitz-continuous in the input (Decision provably; Dev/Rec inherited in practice).

## 2. Distillation — Path 0 decomposition (mandatory inventory)

The paper is training-based, but Path 0 (training-target decomposition) shows **most components are closed-form** — the learned models approximate dynamics we can write down:

| # | Component | Ships? (signal-diff checked) | Extraction (closed-form?) | Verdict |
|---|---|---|---|---|
| 1 | TCCM contraction field | No (closest: Hopfield LLG flow-speed halt `cp_hopfield/llg.rs`; AttractorKernel) | **YES** — `v(x) = −(x−μ)`, exact flow `φ_t(x) = μ+(x−μ)e^{−t}`, envelope `‖x(t)−μ‖ = ‖x(0)−μ‖·e^{−t}` | Modelless: contraction-envelope conformance gate against any decay/consensus/cooldown law |
| 2 | Forest-Flow linear transport | Partial (`VelocityFieldEnsemble` combines fields; not transport validation) | **YES** — constant velocity `x−z` on the linear path; every intermediate observation must satisfy `x_obs(t) ≈ z + t(x−z)` | Modelless: windowed path-conformance validation (= formalized netcode interpolation validation; `‖x−z‖/Δt ≤ v_max` is the integrated form) |
| 3 | Decision score (single-point residual) | **YES** — riir-chain `AnomalyDetector::is_anomalous` (Welford mean/σ on L2 norm, `pipeline.rs:141-173`); `AvatarAntiCheatValidator` per-tick velocity | trivially | Covered — and the paper says it is the **weak** design class |
| 4 | Deviation score (trajectory-aggregated residual) | Partial — VFD (`velocity_field_disagreement.rs`) aggregates **ensemble disagreement** along a trajectory; different signal (epistemic UQ, not sample-vs-normality) | **YES** — sum of squared innovations vs any known predictor | Gap: trajectory-aggregated residual vs **known reference dynamics** not shipped as a score family |
| 5 | Reconstruction score (Euler endpoint error) | Conceptually YES — deterministic replay (`reconcile_game_state`) IS endpoint divergence with known dynamics; `renoise_ce` is perturb-re-resolve | **YES** — replay divergence; coherent deviation grows ~T·δ̄ vs O(√T·σ) | Covered conceptually; not framed/composed as a scoring family |
| 6 | Trajectory-aggregation robustness law | **No** design rule anywhere | **YES** [derived]: `d′ ≈ √M·s̄/σ_eff`; contamination budget `κ_max = s̄/b̄` | Open primitive candidate (design law for choosing window M + deployment tolerance) |
| 7 | In-distribution probe scheduling | Partial — QGF one-step projection (in-distribution oracle evaluation); `contrastive_scope` ScopeModel declines OOD | **YES** [derived]: L-Lipschitz reference + support radius → closed-form valid probe window | Partially covered (as principle, not as flow-time-step law) |
| 8 | Per-feature residual attribution | **YES** — `per_position_residual` (manifold_residual.rs), `AttributionProbe`, spectral_pencil Hellmann–Feynman | **YES** — exact identity `‖r‖² = Σⱼrⱼ²` | Covered |
| 9 | χ² thresholds + PPV at prevalence | Partial — Chebyshev FPR pinned on is_anomalous; conformal floor (Issue 010) for forecast-UQ | **YES** — `D/σ² ~ χ²_{M·d}`; `PPV = sens·ρ/(sens·ρ+(1−spec)(1−ρ))` | Partially covered; the anomaly-detection floor instantiation is unregistered |
| 10 | Robust reference (median-μ) | **NOT FOUND** — no median/MAD/trimmed estimators anywhere in the stack | **YES** — classical; breakdown point 0.5 → provable contamination budget κ ≤ 0.5 | **Open gap — the most concrete actionable** (→ riir-chain Issue 112) |
| 11 | Trained contamination robustness | No | No (requires training) | riir-train Plan 350 (Path 0.5; gated on floor first) |
| 12 | B2B planted-anomaly fixture generator | **NOT FOUND** — no benchmark has ever registered recall of `is_anomalous` / anti-cheat validators on planted anomalies | YES (data generation, no learning) | Open — first-ever detector recall registration (→ riir-chain Issue 112 T1) |

**Adversarial panel** (2 advocates, neutral-merged): No-GD extracted 13 modelless candidates (items 1–10 above + segment-localization triage, scheduled-reference dynamics, score-tail drift monitoring — early/late-half breach distinguishes "observation anomalous" from "reference stale" with one extra summation). Model-based extracted 10 recipe items; survivors after filtering: the fixture generator + contamination-eval harness (0 GPU, unblocks everything), TCCM/FF-XGB trained arms gated on the modelless floor (MAD/Mahalanobis/IsolationForest + split-conformal calibration), karma-farming + NPC-population-pathology feature fixtures as P2 extensions. Discarded with reasons: token-space D2F scoring transfer (weakest inference — gated to die fast vs perplexity thresholding), weight-snapshot screen (adjacent to shipped `dist_guard` Issue 743), healer-trajectory screening (blocked on the measured P5 verdict-loop gap).

## 3. Fusion (paper × shipped substrates)

1. **Poisoning-robust conformance scoring** = paper's trajectory-aggregation law (§2.6) × robust median reference (§2.10) × riir-chain's `AnomalyDetector`. The chain detector estimates its baseline from the observed stream — contaminated *by design* (attackers send deltas). The paper's finding predicts exactly this failure: single-point scores inherit local baseline bias. Harden with MAD-based robust stats + window-aggregated scoring; benchmark with planted poisoning. (Filed: riir-chain Issue 112.)
2. **Windowed movement conformance** = Forest-Flow's constant-velocity linear transport (§2.2) × `AvatarAntiCheatValidator`. Today's per-tick velocity bound misses *within-bound* cheats (micro-teleports each under the cap; bot-like variance collapse). A multi-point path-conformance window (all intermediate observations on `z + t(x−z)` + inter-event variance regularity) is the modelless Deviation score on the linear law. Not filed separately — recorded here as the next candidate if Issue 112's benchmark shows the same single-point fragility on the game side.
3. **Anomaly-detection floor** = Issue 010 "Report the Floor" law × this paper's eval protocol (contamination curves 0→ρ, AUROC **and AUPRC** at extreme imbalance, 5 seeds, per-scoring-function). An anomaly threshold claiming FPR control is UQ-bearing → floor-gated from the initial gate. Floor = MAD/robust-z + Mahalanobis + IsolationForest + split-conformal calibration. (Filed: riir-train Plan 350 P0.)
4. **renoise-CE × contraction tension**: our own Bench 406 caveat says the renoise drift score "may be smaller or absent on contractive operators" — TCCM *is* a contraction. Any future contraction-shaped normality model in the stack should score via Deviation/endpoint (issue: the endpoint of a contraction is the reference itself → score = distance-from-reference after integration, which is item 1's envelope gate).

## 4. Verdict

**Gain.**

| Question | Answer |
|---|---|
| Prior art (Q1)? | **No novelty** at the mechanism level: TCCM (NeurIPS 2025), Forest-Flow (AISTATS 2024), diffusion-AD reconstruction scoring (Livernoche ICLR 2024), Flow Mismatching (arXiv:2605.23070), Deep-Flow (arXiv:2602.17586), contamination-robust deep AD (ROBOD NeurIPS 2022, RandNet, RDA, EntropyStop KDD 2024 — all cited in the paper). → Super-GOAT Q1 FAILS. |
| New behavior class (Q2)? | Partial — contamination robustness is a robustness property of an existing capability, not a new class. |
| Product selling point (Q3)? | Moderate: "poisoning-robust fraud/anomaly detection" (chain) is an improvement claim on a shipped detector. |
| Force multiplier (Q4)? | Yes — connects chain anomaly detection + game anti-cheat + flow-scoring substrates (VFD/renoise/VelocityFieldEnsemble) + the UQ-floor law. |

Not all 4 YES → **not Super-GOAT**. But not Pass either: the reverse-grep found real, undocumented gaps this paper fills — `is_anomalous` single-point design = the paper's contamination-sensitive class with a structurally poisoned baseline; zero robust statistics in the stack; zero planted-anomaly recall registration for any shipped detector. Actionable → **Gain** (files created: this note + riir-chain Issue 112 + riir-train Plan 350).

**MOAT gate:** the distilled *principles* (aggregation law, robust reference, per-feature identity) are classical/public — no moat in the math itself. The moat accrues in the private composition: poisoning-hardened chain detection + the first anomaly-floor registration on our own fixtures + (if the floor loses) frozen TCCM arms consumed via freeze/thaw. Primitive-level additions (median/MAD streaming estimator) go where their consumer lives (riir-chain), not katgpt-core — textbook statistics is an engineering gap fill, not a research primitive.

**Path 0.5 record (mandatory for the training half):** Path 0 decomposition above — components 1,2,4,5,6,7,8,9,10 extract closed-form; component 11 (learned normality fields) genuinely requires GD. Paths 1–3 (freeze/thaw correction / deterministic LoRA / latent-space correction) do not apply — nothing to correct in a model we don't have; the gap is a missing model class, not a biased one. Affordability: tiny (TCCM MLP 0.1–0.2 GPU-h on M3 Metal; FF-XGB CPU-minutes). Sequencing per the Issue 010 law: register the modelless floor on our fixtures FIRST (Plan 350 P0); trained arms (P1) carry an explicit kill condition — if the classical floor matches them on our domains, the plan closes as a negative result (which is itself the deliverable: the floor *is* the detector).

## 5. Honest scope notes

- The paper's numbers (0.82–0.87 AUROC) are on *its* financial datasets; nothing transfers to i64 money-delta streams or game telemetry without measurement — that is precisely what Issue 112 T1 / Plan 350 P0 measure.
- The poisoning exploit path for `is_anomalous` (inflate running mean/variance, then pass a large delta) is **hypothesized from structure**, not yet demonstrated — the benchmark task measures actual degradation under planted poisoning (defend-wrong: the PoC defends or refutes the hardening need).
- Per §3.6 signal-diff discipline: VFD ≠ paper's Deviation (ensemble-epistemic vs sample-vs-normality — one read of `velocity_field_disagreement.rs` confirms the consumed signal is pairwise inter-field disagreement, not observed-vs-reference residual); deterministic replay ≈ paper's Reconstruction in mechanism (endpoint divergence under known dynamics) but is binary-accept, not a calibrated score.
