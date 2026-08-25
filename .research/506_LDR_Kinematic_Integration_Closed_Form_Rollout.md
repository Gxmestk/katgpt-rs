# Research 506: LDR — Kinematic Integration as Closed-Form Rollout (Measured-Not-Learned Dynamics)

> **Source:** "Learning How the World Evolves: Extrapolative Video World Models via Latent Dynamics Reasoning" [arXiv:2608.09926](https://arxiv.org/abs/2608.09926) — Haodong Li, Shaoteng Liu, Tianyu Wang, Chongjian Ge, et al., Manmohan Chandraker (Adobe + UCSD), 10 Aug 2026. Code: [github.com/adobe-research/LDR](https://github.com/adobe-research/LDR)
> **Date:** 2026-08-25
> **Status:** Active
> **Related Research:** 192 (NextLat belief latent dynamics), 245 (latent spatial memory video world models), 288 (KARC delay-basis ridge forecaster), 420 (VFD velocity-field disagreement UQ), 426 (temporal straightening), 465 (ODEWorld continuous latent ODE)
> **Related Plans:** 578 (kinematic rollout primitive)
> **Cross-ref (riir-ai):** Research 345 (anticipatory think-brain guide), Issue 757 (anticipatory belief PoC)
> **Classification:** Public

---

## TL;DR

LDR predicts video frames by casting the latent transition as **explicit kinematic integration**: finite-difference init of velocity + acceleration from 3 conditioning latents, then a **fixed** semi-implicit Euler chain (orders 2→1→0) where only a tiny `tanh(MLP)` regresses the **3rd-and-higher-order residual**. Its headline (ID-OOD error gap 20×+ smaller than a DiT-S video diffusion baseline, 26× fewer params, 143× faster) is — by the paper's **own ablation table** — carried by the **fixed math, not the learned part**: removing dynamics reasoning widens the gap 13× (0.013→0.168) while keeping the reasoning but swapping the latent only degrades to 0.133 (still 2nd best vs baseline 0.506).

**Distilled for katgpt-rs (modelless, inference-time):** the unrolled integrator **is Newton's forward-difference extrapolation formula** — `ŝₙ = ŝ₀ + C(n,1)Δt·ṡ₀ + C(n,2)Δt²·s̈₀ + C(n,3)Δt³·j` — a closed form computable at any horizon with zero loops and zero weights. The whole benchmark family (uniform, parabola, collision, bouncing, looming) admits complete closed-form solutions: the modelless version is **exact** on it, giving ID-OOD gap ≡ 0 **by construction** — a provable (Lean-able) strengthening of the paper's empirical 20×. Ships as a `kinematics` operator family in katgpt-core: FD state estimation, O(1) closed-form rollout, time-to-contact (looming τ), kinematic regime predicates, residual-surprise events, and an extrapolation-horizon admission bound.

---

## 1. Paper Core Findings

1. **The decomposition.** Latent transition = fixed kinematic integration + learned ≥3rd-order residual. Init from 3 latents by forward differences (Eq. 1); per step: `s̈ ← s̈₋₁ + f_θ(ṡ, ŝ)` (f_θ = tanh 3-layer MLP w256, ~143K params of the 4.1M total), then `ṡ ← ṡ + s̈Δt`, `ŝ ← ŝ + ṡΔt`. Δt=1. The integration chain is fixed (Alg. 1); **only f_θ is learned**.
2. **Structured latent.** Per-channel soft-argmax over conv features → (centroid μ, extent σ). Appearance-free by construction — this is what transfers red-ball→blue-square. Decode by warping the conditioning frame.
3. **Extrapolation comes from the architecture, not the data** (cites Xu et al. 2021, *How Neural Networks Extrapolate*): OOD, a plain regressor reverts toward the training mean; a network with the reasoning built in follows the bias.
4. **Ablations attribute the win to the fixed math.** w/o dynamics reasoning: gap 0.168 (13× worse), and under joint 5-task training it **fails even in-distribution** (0.494 ID error — misapplies one task's dynamics to another: a regime-classification failure). w/o structured latent: 0.133 (2nd best). Full LDR: 0.013.
5. **Numbers.** 256²: single-task gap ratio 23.9×, joint 27.7×; 4.1M vs 106.1M params; 143× faster (single forward pass vs 50 DDIM steps). Higher resolution makes LDR better and the baseline worse.
6. **Efficiency claim is architectural.** No iterative sampling, no test-time optimization — the extrapolation is "not simply a matter of scale or compute".

## 2. Path 0 Component Table (coverage + extraction)

| LDR component | Ships? (coverage) | Modelless extraction |
|---|---|---|
| FD init of ṡ, s̈ from ≥3 obs | **Partial** — `remote_smoothing.rs` 1st-order 2-sample FD velocity (view-only); `TemporalDerivativeKernel` dual-EMA latent velocity | **YES** — closed form; central 3-point stencil is strictly better (O(Δt²) vs O(Δt)) on same data |
| Fixed integration-chain rollout | **Partial** — 1st-order bounded dead-reckon only; **zero 2nd-order/acceleration extrapolation anywhere in the workspace** | **YES** — Newton forward-difference closed form, O(1) per query via coefficient lattice |
| Learned ≥3rd-order residual | **Partial** — KARC ridge (latent affect, own-trajectory, never positions) | **YES** — ridge/RLS on (ṡ, ŝ) features; bounded-jerk / geometric-drag schedule family with closed forms + terminal limits |
| Structured latent (μ, σ) | **Partial** — BAKE (μ, λ) extent-without-dynamics; `GenericSpatialBelief` pos+confidence frozen | **YES** — soft-argmax statistic operator on any bounded vector |
| Looming / time-to-contact | **NO — zero hits workspace-wide** (`looming\|time_to_contact\|optic_flow\|approach_velocity`) | **YES** — τ = σ/σ̇ (Lee's optic-flow TTC), log-extent variant |
| Regime discrimination | **Partial** — KARC `regime_gate` (residual-variance mux vs seasonal-naive) | **YES** — closed-form predicates on measured derivatives (uniform/parabolic/impulse/looming/drag) + hysteresis FSM |
| Residual surprise → events | **Partial** — KARC surprise + delta-mem write gate (latent only) | **YES** — z-score/CUSUM on raw position residuals; impulse vs force discrimination (bounce/collision onset) |
| Horizon/trust bound | **NO** — `remote_smoothing` hardcodes `extrapolate_secs = 0.3` | **YES** — error-propagation bound B(k) = ε_p + kΔt·ε_v + C(k,2)Δt²·ε_a; admission horizon k* closed form. **UQ-bearing → conformal floor mandatory** |
| Exactness certificate | **NO** | **YES** — with exact init + f≡0 the chain is exact at ANY horizon on {deg ≤ 3 polynomials} ∪ {geometric-drag}; ID-OOD gap ≡ 0 on that family. **Lean-provable** |

**Funnel result:** value = the MATH (integration chain + decomposition), not the training loop → **MODELLESS-VALIDABLE** core. The honest GD remainder is the *behavioral* residual (see §8).

## 3. Adversarial Panel Merge (three-track, mandatory — paper trains f_θ)

- **No-GD advocate** (19-item inventory): the fixed math is the whole story on the kinematic axis — unrolled integrator = Newton's formula; all 5 benchmark tasks closed-form solvable; exactness certificate converts the empirical 20× into a provable ∞× in-family; α-β-γ tracker (Kalata closed-form gains) as the observation-merged sibling; RLS as the modelless f_θ; two-body closest-approach/intercept solver; τ-ordering law.
- **Model-based advocate:** trained core is tiny (143K params, **1–2 GPU-hrs on one 4090**; the paper's 8×A100 budget buys *pixels* — encoder/decoder/perceptual loss ~95% of compute, all skippable on structured state). Honest value axis = the **behavioral residual** (other agents' pursuit curves, aggro switches — genuinely unknowable under fog-of-war; ridge cannot represent state-conditional nonlinear switches). Graceful degradation: tanh saturation off-support → residual → 0 → **collapses to the modelless floor, not garbage**. Weakest point: value band may be empty at POC scale (the dual_leo G5 precedent) — the trained arm is a *subscription* (1–2 GPU-hrs per AI patch).
- **Coordinator merge:** modelless core is unconditionally adoptable (Path 0 ALL-analog). The trained arm is **deferred to riir-train with a named trigger** (Issue 757 PoC PASS ∧ ridge arm insufficient on behavioral residuals) — filing it now would front-load a plan whose precondition (non-empty value band) is unverified. Discarded advocate findings: none (both tracks' findings retained; the pixel-level replication P2 was discarded by the advocate itself — no consumer, violates single-consumer rule).

## 4. Distillation — the `kinematics` operator family (katgpt-core)

All operators: pure f32 math, zero heap, zero deps, `#[repr(C)]` POD state, per-channel independent (SIMD-able).

1. **`finite_difference_state`** — (ŝ, ṡ, s̈) estimator from an observation ring. Central 3-point stencil (O(Δt²)); observation-budget ladder n∈{1,2,3,4} → predictor order {0,1,2,3}. NaN/zero-Δt screens.
2. **`kinematic_extrapolate`** — O(1) k-step rollout: `dot([1, nΔt, C(n,2)Δt², C(n,3)Δt³], [ŝ, ṡ, s̈, j])`. Coefficient lattice `[f32; K]` precomputed. Bit-identical to the step-by-step chain (same op order). Exact on deg ≤ 3 motion.
3. **Schedule family** (deterministic f_θ stand-ins): constant-jerk; `j_max·tanh(λ·x)` clamped correction; geometric drag `s̈ₙ₊₁ = ρ·s̈ₙ` with closed forms + terminal velocity `ṡ_∞ = ṡ₀ + Δt·s̈₀/(1−ρ)`.
4. **`time_to_contact`** — τ = σ/σ̇ from an extent channel; log-extent variant `1/τ = d(ln σ)/dt`; σ̇≈0 → τ=∞. The looming task solved exactly.
5. **`regime_predicates`** — closed-form kinematic regime classifier (uniform / parabolic(g) / impulse / looming / drag) + sigmoid-gated hysteresis FSM. Solves the paper's joint-training failure mode (direct regression misapplies "downward pull" to uniform motion) with zero weights.
6. **`residual_event`** — prediction-residual surprise: per-channel z-score → sigmoid gate; CUSUM/Page-Hinkley sustained-drift variant; **impulse vs force discrimination** (`|Δṡ|/Δt ≫ running |s̈|`) → bounce/collision onset, wall inference, restitution estimate `e = |ṡ_after|/|ṡ_before|`.
7. **`extrapolation_horizon`** — admission bound B(k) and trust horizon k*; confidence `sigmoid((thr − B(k))/s)`. **UQ-bearing: must beat the conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340) on CRPS/coverage/Winkler, or ship rank-only (k* ordering) with no coverage claim — the rank-only fallback is itself a legitimate GOAT pass.**
8. **`soft_argmax_statistic`** — (μ, σ) geometric summary of any bounded vector over a coordinate lattice; σ ≈ 0 peaked / large diffuse (deterministic concentration proxy). The appearance-invariance operator.
9. **`closest_approach`** — two-body t* = −(p·v)/(v·v), miss distance, intercept quadratic, head-on elastic resolve; t*-ascending urgency ordering law.

### Fusion (paper × shipped substrate)

- **× KARC (katgpt-core/karc + riir-ai karc_bridge)**: KARC is the thesis already shipped for *latent affect* (delay embedding ≈ implicit FD; ridge = the learned residual). The kinematics family is its *positional* twin — and KARC's conformal sidecar + regime gate plug directly into `extrapolation_horizon` + `regime_predicates`. Fusion product: a KARC member whose basis is the kinematic feature map — one forecaster spanning analytic + learned-linear regimes.
- **× `GenericSpatialBelief` two-brain (riir-ai)**: belief currently holds a *frozen* `last_known_pos` + `decay_confidence`. Fusing `kinematic_extrapolate` yields a **predictive stale belief** — still subjective, still diverging from the info brain by design, but diverging *less uselessly*. Gated by `decay_confidence` × `extrapolation_horizon`. (Guide: riir-ai Research 345.)
- **× EVPI gate (riir-ai Plan 544)**: the horizon bound defines the plausible set's growth rate — the re-observation trigger becomes "bound crossed decision-relevance threshold", not a clock.
- **× fear/flee (riir-ai Issue 070)**: `flee_trigger_radius` is a static distance disc. `time_to_contact` gives an ecologically-sound trigger (small-fast threat vs big-slow threat same τ).
- **× `remote_smoothing` (riir-mmorpg-examples)**: its hardcoded 0.3 s extrapolation bound becomes a *derived* k*; the 2-sample FD velocity upgrades to the ladder (its own Plan 025 T7 deferred exactly this).
- **× VelocityFieldEnsemble (DEFAULT-ON)**: kinematic schedules become zero-cost frozen members; VFD disagreement over them = epistemic UQ on extrapolations.

## 5. Novelty Gate (all four)

1. **No prior art?** Published (25 searches): the specific combination — fixed universal kinematics as the integrated part + residual confined to ≥3rd order + order-drop 2→1→0 + structured geometric latent + OOD evaluation — **no published match**. Closest: ODE2VAE (learns the *entire* acceleration field), UDE/Rackauckas (domain-specific mechanistic part, not order-split), SEGNO (2nd-order structure, learned acceleration), KalmanNet, Huang 2023 (CV prior + residual, 1st order only). 2nd-order dead reckoning is textbook DIS — but with **no learned residual**, and nobody in that literature ships the hybrid (Walker 2021 does the *inverse*: NN replaces DR). **Agent-belief/NPC application: unpublished.** In-stack: looming/TTC zero hits; 2nd-order position extrapolation zero hits; predictive belief zero (frozen by design). ✓
2. **New behavior class?** Yes — NPCs that *anticipate*: extrapolated stale beliefs (lead decisions between fog-of-war sightings), looming-based flee *timing* (time-to-contact, not distance), prediction-residual events ("something non-kinematic happened" → curiosity/KG triples). None exist today: fear is a static disc; belief is frozen. ✓ *(detected via game-context reframe, step 4)*
3. **Product selling point?** "Our NPCs know where unseen entities *are now* and *when* a threat arrives — not just where they last saw them. Measured physics, not memorized pixels: the prediction works on enemies it has never seen." ✓
4. **Force multiplier?** ≥2 pillars: two-brain/four-tier memory × reasoning/curiosity (surprise bus) × fear/flee × EVPI × conformal UQ; plus netcode consumer. ✓

## 6. Verdict

**Super-GOAT.** One-line: the paper's own ablations prove the extrapolation lives in fixed arithmetic we can ship closed-form, exact, and provable — and its application to per-NPC anticipatory belief is a new capability class with no published or shipped prior art.

**MOAT gate:** katgpt-rs = fundamental/base primitive via fusion (pure numerical math, no game semantics — physics constants are config). riir-ai = pillar-level guide fusing ≥2 pillars. Both pass.

**Mandatory outputs (this session):** open primitive → this note + Plan 578; architectural guide → riir-ai Research 345; falsifiable PoC → riir-ai Issue 757 (§3.6 — behavioral quality claims are PoC-pending, not proven).

## 7. Prior Art (closest three + the classic)

| Work | What it does | Diff |
|---|---|---|
| **ODE2VAE** (Yıldız et al., NeurIPS 2019) | 2nd-order latent ODE over (position, momentum) for video; Bayesian-NN acceleration integrated by an ODE solver | Learns the *whole* acceleration field; no fixed kinematic part, no residual split, no OOD claim |
| **Universal Differential Equations** (Rackauckas et al. 2020) | Mechanistic ODE backbone + neural residual inside one ODE | Mechanistic part is *domain-specific*, split not order-based, goal is fit not OOD |
| **SEGNO** (Liu et al., ICLR 2024) | 2nd-order motion equations in an equivariant GNN (molecules) | Acceleration still learned; no fixed integrator, no residual-order decomposition |
| 2nd-order dead reckoning (DIS standard, 90s; Pantel & Wolf 2002) | `D(t)=P+vΔt+½aΔt²` — fixed integrator, textbook | **No learned residual**; netcode-only; nobody ships the hybrid |

Supporting: Xu et al. 2021 (extrapolation follows architecture); "When do neural networks learn world models?" (ICML 2025 — low-degree bias provably recovers world models: direct theoretical support for keeping dynamics low-order); PhyWorld (Kang et al., ICML 2025) — the benchmark, whose conclusion "scaling neither model nor data helps extrapolation" is the paper's and ours.

## 8. Honest Caveats

1. **The paper's quality claim is pixel-level; ours is not.** We make three claim types: architectural (ships — grep-proven), exactness (provable on the analytic family), and **behavioral (PoC-PENDING — Issue 757 must falsify anticipation-beats-frozen before any product claim)**.
2. **The anti-cousin is load-bearing.** `GenericSpatialBelief` is frozen *by documented design* (two-brain divergence contract, `bench_434` "sharing is NOT observation"). The distillation frames extrapolation as *predictive staleness* — belief stays subjective, fog-gated, never synced — but the burden of proof that anticipation improves decisions sits on the PoC, not the architecture.
3. **UQ obligation.** `extrapolation_horizon` is UQ-bearing → conformal floor (Plan 340) before any calibrated claim; rank-only fallback documented.
4. **Sync/anti-cheat boundary.** Extrapolated positions are think-brain only; they never cross `SyncBlock`, never back movement claims (prediction ≠ validation — the anti-cheat rule unchanged).
5. **Trained-arm risk (dual_leo precedent).** The behavioral-residual MLP may have an empty value band at POC scale. Deferred to riir-train **with trigger**: Issue 757 PoC PASS ∧ ridge arm insufficient. Recipe on file (§3): tanh-MLP w256 over (ṡ, ŝ) + staggered integration, horizon curriculum 4→full by 8K steps, AdamW 1e-4/wd 0.01/clip 1.0/batch 256/10K steps, z-scored channels, archetype conditioning, **1–2 GPU-hrs on the 4090**, 3-arm GOAT (zero-residual floor / KARC-ridge / trained f_θ) with the paper's own ID-OOD gap ratio as G5 metric, multi-seed (the paper reports a single run).
6. **Single-run source.** All paper numbers are one seed (42). Our gates are multi-seed by house rule.

## 9. Routing

| Piece | Home |
|---|---|
| `kinematics` operator family (this note, Plan 578) | `katgpt-core/src/kinematics/` — public, leaf-clean |
| Anticipatory think-brain guide | riir-ai Research 345 |
| Falsifiable anticipation PoC | riir-ai Issue 757 (riir-poc harness) |
| Behavioral-residual trainer (deferred, trigger-gated) | riir-train `.plans/` when triggered |
| Consumer wiring (belief extrapolation, looming flee, EVPI bound) | riir-ai plans after PoC |
