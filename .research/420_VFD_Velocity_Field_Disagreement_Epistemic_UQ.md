# Research 420: VFD — Velocity-Field Disagreement Epistemic UQ for Flow-Matching Ensembles

> **Source:** Römer, Seeliger, Liu, Sturgis, Bagatella, Marta, Krause, Schoellig — *Uncertainty Quantification for Flow-Based Vision-Language-Action Models*. [arxiv 2606.18043](https://arxiv.org/abs/2606.18043). TU Munich + ETH Zurich, June 2026.
> **Date:** 2026-07-13
> **Status:** Active
> **Related Research:** 375 (Velocity-Field Ensemble — the substrate this extends), 322 (Conformal Seasonal Pools — the UQ floor), 281 (BoMSampler — sibling sample-disagreement UQ), 288 (KARC), 268 (QGF — already ships a `QgfVarianceSignal` slot for "ensemble KL"), 296 (Stokes/DEC — mass-conservation sanity check)
> **Related Plans:** 376 (Velocity-Field Ensemble primitive — Phase 6 G7 UQ gate is deferred, THIS fills it), 340 (Conformal-naive floor), 281 (BoMSampler)
> **Cross-ref (riir-ai):** Research 170 (Per-NPC Velocity-Field Ensemble Composition Guide — the Super-GOAT this GOAT activates the UQ axis of)
> **Classification:** Public

---

## TL;DR

The paper derives a **modelless epistemic uncertainty estimator for flow-matching / continuous-time generative models**: given a small (M=2 is sufficient) ensemble of velocity fields `{v_s^i(x, y)}` trained on the same data via OT Gaussian paths, the **velocity-field disagreement (VFD)** score

```
u_e(y; V) = (1 / (M·(M−1)·N_s)) · E_{x_0 ~ p_0} [ Σ_{i≠j} Σ_ℓ κ_{s_ℓ} · ‖v_{s_ℓ}^i(x_{s_ℓ}^{(i)}, y) − v_{s_ℓ}^j(x_{s_ℓ}^{(i)}, y)‖²₂ ]
```

(approximating the average pairwise KL divergence between ensemble members along the generative ODE) is a well-calibrated, training-free epistemic-UQ scalar. Theorem 4.1 proves the bound `KL(p_θ1 ‖ p_θ2) = ∫₀¹ κ_s · E_{x_s ~ p_{s}^{θ_1}} [‖v_s^{θ_1} − v_s^{θ_2}‖²] ds` with `κ_s = s / (1 − s)` for OT Gaussian conditional paths.

**Distilled for katgpt-rs (modelless, inference-time):**

The codebase already ships `VelocityFieldEnsemble<P, D>` (Plan 376, default-on, `crates/katgpt-core/src/velocity_field_ensemble.rs`) and its `Schedule::optimal_diffusion(t)` — but ships it **as a non-UQ algebraic combiner**. Plan 376 Phase 6 explicitly defers the UQ gate (G7): "Primitive still ships as non-UQ; the gate is pre-validated for future UQ claims." VFD is the **specific UQ estimator that activates the deferred G7 gate** — it consumes the SAME P frozen velocity fields used for ridge-combination, but evaluates them **independently along an ODE integration** and measures pairwise disagreement weighted by the existing `Schedule`'s `κ_s` profile.

**The SAVE half of the paper (active fine-tuning via VFD-guided expert demonstration acquisition) → riir-train.** It is a training method (4,000 SGD steps per round × 15 rounds × 3 seeds = ~1,880 GPU-hours). Out of scope for this workflow.

---

## 1. Paper Core Findings

### 1.1 The primitive (Theorem 4.1 + VFD score, eqs. 5–7)

For two flow-matching distributions `p_θ1(x | y)`, `p_θ2(x | y)` induced by velocity fields `v_s^{θ_1}(x, y)`, `v_s^{θ_2}(x, y)` trained via OT Gaussian conditional paths `p_s(x | x_1) = N(x | s·x_1, (1−s)² I)`, the KL divergence is

```
KL(p_θ1(·|y) ‖ p_θ2(·|y)) = ∫₀¹ κ_s · E_{x_s ~ p_s^{θ_1}(·|y)} [ ‖v_s^{θ_1}(x_s, y) − v_s^{θ_2}(x_s, y)‖²₂ ] ds
```

with `κ_s = s / (1 − s)`. **The weighting matters:** velocity differences at higher flow-matching time `s` (less noise, more data-like) are more informative of epistemic uncertainty. This is the paper's central mathematical contribution — turning an intractable KL divergence between high-dim flow-matching posteriors into a 1-D integral of velocity differences along an ODE path.

**VFD score (eq. 7):** given M ensemble members and `N_s` ODE steps of size `δ_s = 1/N_s`,

```
u_e(y; V) = 1 / (M (M−1) N_s) · E_{x_0 ~ N(0,I)} [ Σ_{i≠j} Σ_{ℓ=0}^{N_s − 1} κ_{s_ℓ} · ‖v_{s_ℓ}^i(x_{s_ℓ}^{(i)}, y) − v_{s_ℓ}^j(x_{s_ℓ}^{(i)}, y)‖²₂ ]
```

where each member's trajectory `x_{s_{ℓ+1}}^{(i)} = x_{s_ℓ}^{(i)} + v_{s_ℓ}^i(x_{s_ℓ}^{(i)}, y) · δ_s` is integrated under ITS OWN velocity field. The expectation is approximated by a batch of B parallel integrations (B=5 in the paper).

### 1.2 Two-member ensemble is sufficient (§6.2)

Calibration (negative Spearman ρ between VFD and per-task success rate) is essentially flat for M ∈ {2, 3, 4}: M=2 gives ρ = 0.71 ± 0.03 — the same as M=3 and M=4 within noise. **M=2 is the production choice.** This is critical for our 10K-NPC regime: only 2 velocity fields per NPC are needed for VFD.

### 1.3 VFD beats six UQ baselines on flow-based VLAs (Table 1)

| Metric | Action-L2 | ACE | DECU | GU | Entropy | Perplexity | **VFD** |
|---|---|---|---|---|---|---|---|
| −Spearman ρ | 0.50 | 0.31 | 0.31 | 0.62 | 0.10 | −0.04 | **0.71** |
| −Pearson | 0.48 | 0.36 | 0.23 | 0.65 | 0.23 | 0.02 | **0.71** |

VFD is the best-calibrated epistemic-UQ estimator for flow-matching models in this benchmark. The closest competitor (GU = Generative Uncertainty via last-layer Laplace) requires ~5 hours per checkpoint to fit the Laplace posterior; VFD requires only forward ODE integration (~30 minutes for the same setup).

### 1.4 Failure detection at runtime (§6.4)

Calibrated per-task thresholds (one-sided conformal prediction band from 10 successful rollouts) on the VFD score achieve: **67% accuracy, 79% TPR, 0.54 timestep-wise accuracy** — beating ACE, STAC, RND-OE on the same LIBERO benchmark. This is the deployment-time use: at each inference timestep, compute VFD; if it exceeds the calibrated threshold, flag the rollout as likely-to-fail.

### 1.5 SAVE → riir-train (out of scope here)

The SAVE framework (§5, Algorithm 1) uses VFD to prioritize which expert demonstrations to collect, then fine-tunes the VLA ensemble (4,000 gradient steps × 15 rounds × replay ratio 0.5). This is **gradient descent through base weights** → out of scope for katgpt-rs and riir-ai. The paper's 22% sample-efficiency gain (Table 2) is a training-method result.

**Routing:** VFD estimator → katgpt-rs (this note). SAVE active-fine-tuning loop → riir-train (one-line redirect, no files in this session).

### 1.6 The proof structure (Appendix A.2)

The proof of Theorem 4.1 is the math worth keeping in mind when implementing:
1. Write `KL(p_1 ‖ p_2) = ∫₀¹ ∂_s KL(p_s^1 ‖ p_s^2) ds` (telescoping, since `p_0^1 = p_0^2` = base Gaussian).
2. Apply the continuity equation `∂_s p_s + div(p_s · u_s) = 0` and the divergence theorem to convert the KL derivative into a velocity-field inner product.
3. Apply **Lemma A.1**: `∇ log p_s(x) = (s · u_s(x) − x) / (1 − s)` (Tweedie's formula + flow-matching conditional structure) to convert the score-function term into a velocity-field term.
4. Combine → `∂_s KL = κ_s · E[‖u_1 − u_2‖²]`.

**Key insight:** the `κ_s = s/(1−s)` weighting arises from the **OT Gaussian conditional path structure** — the same structure Plan 376's `Schedule::Linear` (`α=1−t, β=t, γ=1`) and `Schedule::Trigonometric` (`α=cos(πt/2), β=sin(πt/2), γ=π/2`) implement. The codebase's `Schedule::optimal_diffusion(t) = α_t γ_t / β_t` is `1/κ_s`-shaped by construction. **VFD consumes the SAME schedule that the velocity-field ensemble already ships.**

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface — verify before any novelty claim)

| Paper mechanism | Shipped cousin | File / Plan |
|---|---|---|
| Closed-form velocity-field ensemble with P frozen members | **`VelocityFieldEnsemble<P, D>`** + `VelocityField` trait | Plan 376, `crates/katgpt-core/src/velocity_field_ensemble.rs` (DEFAULT-ON since 2026-07-04) |
| OT Gaussian schedule `α_t, β_t, γ_t` + optimal diffusion `D*_t = α_t γ_t / β_t` | **`Schedule::{Linear, Trigonometric}::optimal_diffusion(t)`** | Same file, lines 540–597. Linear gives `D*_t = (1−t)/t = 1/κ_s`. |
| Single stochastic-interpolant ODE step under optimal-diffusion SDE | **`stochastic_interpolant_step_into`** | Same file, lines 633–670. Decoupled from the ensemble — takes any precomputed drift. |
| Generic ensemble disagreement → confidence bridge | **`QgfVarianceSignal`** trait + `confidence_from_disagreement(disagreement: f32) -> f32` | `crates/katgpt-core/src/qgf/adaptive.rs:141-174`. The docstring explicitly mentions "ensemble KL" as a future implementor. **This is the integration slot.** |
| Sample-based ensemble disagreement UQ (sibling) | **BoMSampler** (K-hypothesis belief sampling) | Plan 281, Research 281. Disagreement on TOKEN SAMPLES, not on velocity fields along ODE. |
| Conformal UQ floor (the gate any UQ primitive must beat) | **`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`** | Plan 340. Plan 376 Phase 6 already benchmarked the velocity-field ensemble against this floor (BEATS on AR(1): CRPS 0.93×, Winkler 0.94×). |
| DEC mass-conservation sanity check on a vector field | **`belief_mass_divergence`, `codifferential`** | Plan 314, Research 296. Can validate `div(b̂) ≈ 0` on the combined drift. |
| Velocity-field ensemble Super-GOAT guide | **Research 170 (riir-ai)** + Plan 385 (deferred runtime) | The runtime integration Plan; VFD plugs in as the per-NPC failure-detection signal. |
| LatCal commitment of per-target solved η | **`EtaCommitBatch` sidecar** (DONE 2026-07-07) | riir-chain/.research/008. Commits the COMBINED weights; does NOT commit a UQ scalar. |

### 2.2 What the paper adds that none of the above does alone

1. **A modelless epistemic-UQ scalar for flow-matching models specifically.** BoMSampler measures disagreement on token samples; the QgfVarianceSignal trait accepts "ensemble KL" but no shipped implementor produces it from velocity fields. VFD computes pairwise velocity-field disagreement along an ODE path weighted by `κ_s = s/(1−s)`, deriving the score from the SAME OT Gaussian path structure the velocity-field ensemble already uses. **No shipped primitive does this.**

2. **A theoretical guarantee that VFD approximates the average pairwise KL between flow-matching posteriors** (Theorem 4.1). The bound is `≤ 1/(M(M−1)) Σ_{i≠j} KL(p_θi ‖ p_θj)` (eq. 4c via Jensen). Other disagreement scores (Action-L2 on terminal samples, ACE on entropy of grid-binned positions) lack this KL-approximation guarantee. VFD is **principled**, not heuristic.

3. **The 2-member sufficiency result.** Calibration is flat for M ∈ {2,3,4}. For our 10K-NPC regime, this means the VFD cost is one extra velocity-field evaluation per ODE step per NPC — affordable at plasma-tier (µs budget) when the velocity fields are KARC-class (sub-µs eval).

4. **Runtime failure detection via conformal-calibrated VFD thresholds.** The paper demonstrates 67% accuracy / 79% TPR for flagging imminent failures. This is the deployment-time use that connects to CLR (claim rubric L1/L2/L3) and to the sleep-time anticipator (Plan 334) — NPCs that "know when they don't know" and either abstain, delegate, or escalate.

5. **The proof's reuse of Tweedie's formula + continuity equation.** Lemma A.1 (`∇ log p_s = (s·u_s − x)/(1−s)`) is a reusable identity for ANY flow-matching-on-OT-Gaussian-paths model. It connects to the DEC substrate: the continuity equation `∂_s p_s + div(p_s u_s) = 0` is exactly `belief_mass_divergence` (Research 296). **The VFD proof uses the same machinery the DEC substrate encodes.**

### 2.3 Fusion (the GOAT move)

VFD is a **direct extension of the velocity-field ensemble Super-GOAT (R375 / P376)** — it activates the deferred G7 UQ gate. The fusion table below lists the systems it multiplies:

| Fusion partner | What it ships | What VFD adds | Fusion product |
|---|---|---|---|
| **R375 VelocityFieldEnsemble (P376)** | Algebraic ridge-optimal combination of P frozen velocity fields; ships NON-UQ | Pairwise velocity-field disagreement along ODE → calibrated epistemic UQ scalar | "The velocity-field ensemble becomes UQ-bearing: ridge-combined super-forecaster PLUS a calibrated epistemic-UQ readout from the SAME P members. Two uses, one substrate." |
| **QgfVarianceSignal (P268)** | Trait accepting normalized `[0,1]` disagreement; docstring names "ensemble KL" as future | A concrete `VfdVarianceSignal` impl producing normalized disagreement from VFD score | "QGF's adaptive guidance weight now has a flow-matching-native variance probe — closes the docstring's open item." |
| **BoMSampler (R281/P281)** | Sample-disagreement UQ on tokens | Velocity-disagreement UQ on flow-matching models — the continuous-time analog | "Two UQ probes for two model classes: BoM for autoregressive LLM drafters, VFD for flow-matching/velocity-field models. Composable." |
| **Conformal Seasonal Pools / Issue 010 floor (R322/P340)** | The mandatory UQ floor | VFD is itself UQ-bearing → must beat the floor | "VFD-vs-floor benchmark: does flow-matching-native epistemic UQ beat the conformal-naive floor on the AR(1) corpus (or a flow-matching-specific corpus)?" — mandatory G2 of the GOAT gate |
| **CLR Claim Rubric (P307)** | L1/L2/L3 evidence ladder | VFD score as a calibrated confidence input to the L1 evidence gate | "NPCs flag claims/actions as low-confidence when their velocity-field ensemble disagrees — CLR L1 gets a UQ-driven emit/silent gate." |
| **Sleep-Time Anticipator (P334)** | Offline query anticipation | VFD as the priority signal: high-VFD queries get anticipated first | "Sleep-time compute pre-resolves the high-uncertainty queries the runtime expects to face." |
| **Schedule (Plan 376)** | OT Gaussian α/β/γ + optimal_diffusion D*_t | VFD consumes `κ_s = s/(1−s)` from the SAME schedule (Linear: `κ_s = t/(1−t) = 1/D*_t`) | "Zero new schedule parameters — VFD's weighting is determined by the schedule already in Plan 376." |
| **DEC Stokes (R296/P314)** | `belief_mass_divergence`, `codifferential` | Sanity check: VFD measures member divergence; `div(b̂)` measures combined-drift mass conservation. **Independent failure modes.** | "Two UQ-relevant divergence signals: member-vs-member (VFD) and combined-vs-mass-conservation (DEC). Cross-validate." |
| **riir-ai Plan 385 (deferred runtime)** | Per-NPC velocity-field ensemble wiring | Per-NPC VFD as the "abstain when uncertain" signal | "Per-NPC failure detection: each NPC's ensemble members disagree → NPC delegates, escalates, or picks a safe fallback action." |
| **riir-chain EtaCommitBatch (R008)** | LatCal-committed solved η per target | Optional: commit per-target VFD threshold as one extra LatCal scalar | "Anti-cheat: the quorum-verifiable VFD threshold per target prevents a node from silently lowering its abstention bar." |

**Force multiplier count: ≥8 systems.** But — for the verdict — these multipliers are extensions of ONE existing Super-GOAT (velocity-field ensemble), not new pillar connections. This is what makes VFD a **GOAT that activates a deferred axis of an existing Super-GOAT**, not a new Super-GOAT.

### 2.4 Latent-space reframing (mandatory per fusion protocol §1.3)

Operating on each Super-GOAT factory module:

(a) **HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`): each NPC has 2 frozen HLA-evolution kernels (e.g., the global `evolve_hla` + one tuned variant from freeze/thaw divergence). VFD measures their disagreement on the next-HLA-direction prediction along the interpolant ODE → per-NPC epistemic UQ on HLA evolution. The 5 synced affect scalars (valence/arousal/desperation/calm/fear) become UQ-tagged — `fear` can carry a "this NPC is uncertain about its own fear prediction" sidecar.

(b) **latent_functor** (`riir-engine/src/latent_functor/`): two functor instances (different direction vectors) applied to the same source-state → VFD on their outputs along the functor-application ODE → "how divergent are the two relational stances this NPC could take?" This is a modelless analog of "personality uncertainty" — the NPC has multiple committed directions and measures their disagreement.

(c) **cgsp_runtime curiosity** (`riir-engine/src/cgsp_runtime/`): curiosity becomes `curiosity_t = α · ‖actual − b̂_forecast‖ + β · VFD_t` — combining forecast error (single-forecaster surprise) with ensemble disagreement (multi-forecaster uncertainty). Two complementary surprise signals.

(d) **LatCal fixed-point commitment** (`riir-chain/src/encoding/`): the per-target VFD threshold (one f32) can cross sync as one extra LatCal fixed-point scalar alongside the existing `EtaCommitBatch`. Anti-cheat: nodes cannot silently lower their abstention bar.

(e) **NeuronShard / freeze envelope** (`riir-neuron-db/src/`): the 2-member ensemble for VFD is **already** 2 `VelocityFieldEnsembleShard`s — VFD reads them, doesn't mutate them. No new shard type. The conformal-calibrated VFD threshold is a separate small artifact (one f32 per task) — could be a sidecar or a config field.

(f) **DEC Stokes-calculus** (`katgpt-core/src/dec/`): The VFD proof's continuity-equation step (eq. 17, `∂_s p_s + div(p_s u_s) = 0`) IS `belief_mass_divergence`. The VFD score is a discrete approximation of the velocity-field difference; `codifferential(b̂)` is the divergence of the COMBINED drift. **They are complementary, not redundant**: VFD measures member-vs-member divergence; `codifferential` measures combined-vs-mass-conservation divergence. Cross-checking the two is a sanity gate.

---

## 3. §3.5 Modelless Unblock Protocol (MANDATORY — passed)

Before any riir-train deferral (note: only the SAVE half of the paper goes to riir-train; VFD itself is fully modelless):

**Path 1 (freeze/thaw snapshot correction):** **PASS.** The M ensemble members are frozen snapshot artifacts (M `MerkleFrozenEnvelope`s). The conformal-calibrated VFD threshold is a frozen config. No weight mutation. VFD is a pure read of frozen velocity fields.

**Path 2 (raw/lora reader-writer hot-swap):** **N/A.** VFD does not modify weights. It READS velocity-field outputs at ODE states. No hot-swap involved.

**Path 3 (latent-space correction):** **PASS (trivially).** VFD IS a latent-space operation: it computes pairwise differences of velocity-field outputs (latent vectors) and integrates them weighted by `κ_s`. No gradient descent. No backprop. Closed-form scalar output.

**Decision protocol result:** All three paths pass (trivially) → **MODELLESS-VALIDABLE.** VFD ships in katgpt-rs without any riir-train dependency.

**Note on SAVE (the active-fine-tuning half):** SAVE itself is gradient descent (4,000 steps × 15 rounds × replay). Per §3.5 the modelless paths do NOT unblock it — SAVE is a genuine riir-train dependency. **But VFD (the UQ estimator) is separable from SAVE** — VFD is computed BEFORE any fine-tuning decision, and SAVE is just one consumer of VFD (along with runtime failure detection, CLR gating, sleep-time prioritization). The VFD primitive ships independently in katgpt-rs.

---

## 4. Verdict

### Tier: **GOAT** (modelless UQ extension that activates the deferred G7 gate of an existing Super-GOAT)

| Q | Answer | Evidence |
|---|--------|----------|
| **Q1: No prior art?** | **Partial.** No shipped primitive computes velocity-field pairwise disagreement along an ODE path weighted by `κ_s = s/(1−s)` (paper-vocabulary grep `VFD\|velocity.*disagreement\|epistemic_uncertainty.*flow` returns zero hits at the .rs layer). BUT — `QgfVarianceSignal` (P268) explicitly names "ensemble KL" as a future implementor, and BoMSampler (P281) does sample-disagreement UQ. The CATEGORY (ensemble-disagreement UQ) is partially covered; the SPECIFIC mechanism (velocity-field disagreement under OT Gaussian paths with κ_s weighting, derived as a KL upper bound) is novel. |
| **Q2: New capability class?** | **NO.** "Ensemble-disagreement epistemic UQ" is an existing category (BoM, the QGF slot). VFD is a new INSTANCE for the velocity-field ensemble primitive specifically — it activates a deferred axis (Plan 376 Phase 6 G7) of an existing Super-GOAT, but does not create a new pillar. |
| **Q3: Product selling point?** | **YES.** "Our NPCs know when they don't know — the per-NPC 2-member velocity-field ensemble measures its own disagreement along the interpolant ODE, weighted by κ_s, producing a calibrated epistemic-UQ scalar per inference. NPCs abstain, delegate, or escalate when uncertain. No training." Concrete, demoable, fills the runtime failure-detection gap of the velocity-field ensemble Super-GOAT. |
| **Q4: Force multiplier?** | **YES (≥8 systems)** but they are extensions of the velocity-field ensemble Super-GOAT, not new pillar connections. |

**One-line reasoning:** The paper's value is **not** SAVE (which is training → riir-train); it is **Theorem 4.1 + the VFD score (eq. 7)** — a principled, KL-approximating epistemic-UQ estimator for flow-matching models that consumes the SAME `Schedule` and `VelocityField` substrate Plan 376 already ships. VFD fills the deferred G7 UQ gate of the velocity-field ensemble Super-GOAT, multiplies ≥8 systems, and gives the runtime a modelless "abstain when uncertain" signal — but it is a GOAT (provable gain on an existing primitive's missing axis), not a Super-GOAT (new pillar).

**Why not Super-GOAT (honest demotion):** the Super-GOAT bar requires "new capability class". VFD's capability — ensemble-disagreement epistemic UQ — exists in the codebase in sibling form (BoMSampler, the QgfVarianceSignal slot). The contribution is a NEW MECHANISM (κ_s-weighted velocity-field pairwise disagreement with KL-bound guarantee) for an EXISTING CAPABILITY applied to a SPECIFIC SUBSTRATE (flow-matching / velocity-field ensemble). The moat was already created by R375/P376; VFD strengthens it on the UQ axis. Per the verdict tier definitions, this is GOAT, not Super-GOAT.

### MOAT gate per domain

| Domain | Verdict | Reason |
|---|---|---|
| **`katgpt-rs`** (public engine) | **In scope — GOAT** | Paper-derived fundamental UQ primitive for the velocity-field ensemble (already in katgpt-rs). Lands behind a feature flag; GOAT gate (G1–G5 below) decides promote-to-default. The primitive stays substrate-agnostic: it works on any flow-matching / continuous-time generative model that exposes velocity fields along an ODE. |
| `riir-ai` | **Deferred follow-up** | The runtime integration (per-NPC VFD, CLR L1 gating, sleep-time prioritization) is pillar-level (connects R170 Super-GOAT to P8 Reasoning Pack to P334 Sleep-Time). **Not in this session** — defer to a future riir-ai plan after the katgpt-rs GOAT gate passes. |
| `riir-chain` | **Optional follow-up** | Commit per-target VFD threshold as one LatCal scalar alongside `EtaCommitBatch`. Anti-cheat use. **Not in this session.** |
| `riir-neuron-db` | **Out of scope** | No new shard type — VFD reads existing `VelocityFieldEnsembleShard`s. |
| `riir-train` | **Out of scope for VFD** (VFD is modelless). **SAVE → riir-train** (one-line redirect: "SAVE active-fine-tuning loop with VFD-guided data acquisition is a training method → riir-train/.research"). No files in this session. |

---

## 5. Plan (see katgpt-rs/.plans/432)

**Phase 1 — Open primitive in katgpt-core** (behind `velocity_field_disagreement` feature):
- `VfdScore<M, D>` struct + `VfdScratch` for zero-alloc batched computation.
- `fn vfd_score_into(ensembles: &[&VelocityField; M], y: &State, scratch: &mut VfdScratch, schedule: Schedule, n_steps: usize, batch: usize, rng: &mut R) -> f32`
- Reuses `Schedule::optimal_diffusion(t)` to derive `κ_s = α_s γ_s / β_s` (the paper's weighting).
- Reuses `stochastic_interpolant_step_into` for ODE integration (each member along its own velocity field).
- Implements `QgfVarianceSignal` for `VfdScore` (closes the docstring's "ensemble KL" open item).

**Phase 2 — GOAT gate** (mandatory per AGENTS.md):
- **G1 (mechanics):** 2 synthetic linear velocity fields with KNOWN analytic KL on a 2D Gaussian toy — verify VFD approximates the analytic KL within tolerance.
- **G2 (UQ floor, mandatory per Issue 010):** benchmark VFD-calibrated intervals against `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` on the AR(1) corpus (same as Plan 376 Phase 6) — VFD must beat the floor on CRPS / coverage / Winkler.
- **G3 (no regression):** `--features velocity_field_disagreement` adds zero warnings; `--all-features` clean.
- **G4 (latency):** M=2, D=8, N_s=10, B=5 → VFD score computation ≤ 50µs (plasma-tier budget).
- **G5 (QGF integration):** `VfdVarianceSignal` impl + smoke test that QGF's adaptive weight responds to VFD.

**Phase 3 — Promotion decision:** if G1–G5 pass → promote to default-on (closes Plan 376 Phase 6 G7 deferred gate, activates the velocity-field ensemble's UQ axis).

---

## 6. Caveats and known risks

1. **The KL bound is approximate, not exact.** Theorem 4.1 assumes "marginal probability densities and velocity fields decay sufficiently fast at infinity" (the `p_s · u_s → 0` as `‖x‖ → ∞` boundary condition). For finite-width neural velocity fields on bounded latent spaces (HLA `d=8`, shards `d=64`), this holds empirically but is NOT formally proven. The VFD score is an **upper bound** on epistemic uncertainty (eq. 4c via Jensen), not the exact value.

2. **The 2-member sufficiency is empirically demonstrated for VLAs, not for game AI.** The paper's M=2 result is on LIBERO benchmark with SmolVLA backbone. Whether M=2 suffices for our per-NPC HLA-evolution kernels is open — the GOAT gate G1 measures this on the synthetic toy. **Mitigation:** the API accepts arbitrary M; M=2 is the documented default but not a hard constraint.

3. **VFD is computed by ODE integration under ONE member's velocity field, then evaluating the OTHER members at those states.** This is asymmetric: `Σ_{i≠j}` sums over both orderings (member i's trajectory, member j evaluated; then member j's trajectory, member i evaluated). Plan 432 must implement this correctly — the paper's eq. 7 is unambiguous but easy to mis-implement as "single shared trajectory, evaluate both members" (which would be Action-L2, not VFD).

4. **VFD requires a forward ODE integration per inference.** For real-time game AI (20Hz tick, 10K NPCs), the cost is M × N_s × (one velocity-field eval) per NPC per tick. With M=2, N_s=10, eval = 200ns → 4µs per NPC per tick → 40ms per tick for 10K NPCs. **This exceeds the 20Hz tick budget (50ms total) on the surface.** Mitigation: (a) VFD is computed OFF the hot path (sleep-time, per-NPC async); (b) VFD threshold is checked once per NPC per `T_vfd` ticks, not every tick; (c) the conformal threshold is pre-calibrated, so VFD is a single scalar comparison after the score is computed. The Plan 432 G4 latency gate enforces this.

5. **`κ_s = s/(1−s)` diverges at s=1.** The paper handles this by evaluating on a grid `s_ℓ = ℓ·δ_s` for `ℓ ∈ {0, …, N_s−1}`, so the largest weight is `κ_{1−δ_s} = (1−δ_s)/δ_s` (finite for `δ_s > 0`). Plan 432 must use the same grid convention — never evaluate at `s = 1` exactly.

6. **The conformal threshold calibration requires successful rollouts.** The paper calibrates per-task thresholds from 10 successful rollouts. For game AI, this maps to "10 successful episodes per task type" — a non-trivial data requirement for cold-start tasks. Mitigation: fall back to a global threshold until per-task calibration data accumulates; the sleep-time anticipator (P334) is the natural place to accumulate it.

7. **VFD does not subsume BoMSampler.** BoM operates on autoregressive token-sample disagreement; VFD operates on flow-matching velocity-field disagreement. They are **complementary** — different model classes. A future fusion (`BoM ⊕ VFD` joint UQ readout) is possible but out of scope for this note.

8. **The riir-train dependency (SAVE) is separable but the paper presents them together.** A reviewer might object "this whole paper is about active fine-tuning, which is training". **Response:** §4 (VFD) and §5 (SAVE) are separable — VFD is the UQ estimator computed BEFORE any fine-tuning decision; SAVE is one consumer of VFD among many (runtime failure detection is another, with no training). The VFD primitive ships in katgpt-rs without SAVE; SAVE's training-method research goes to riir-train.

9. **The QGF integration is a thin bridge but must respect the existing `QgfVarianceSignal` semantics.** VFD's raw score is unbounded (it's a sum of weighted L2 norms). The `normalized_disagreement(&self) -> f32` trait requires `[0, 1]`. Plan 432 must define a sigmoid normalization `σ(VFD / τ_vfd)` with a configurable `τ_vfd` — and document that this is a heuristic normalization, not a probability.

10. **The DEC cross-check (caveat from R296):** `codifferential(b̂) ≈ 0` is a sanity check on the COMBINED drift's mass conservation. VFD measures MEMBER-vs-MEMBER disagreement. They are independent — a combined drift can be mass-conserving while members disagree wildly (and vice versa). The fusion (§2.3 row 7) is a CROSS-VALIDATION gate, not a replacement.

---

## 7. Cross-References

- **Substrate primitive:** `katgpt-rs/crates/katgpt-core/src/velocity_field_ensemble.rs` (default-on, Plan 376).
- **Schedule consumed:** `Schedule::optimal_diffusion(t)` (same file, lines 585–597) — VFD's `κ_s = 1/D*_t` (Linear) or its trigonometric equivalent.
- **ODE integrator consumed:** `stochastic_interpolant_step_into` (same file, lines 633–670) — decoupled from the ensemble, reusable.
- **Integration slot:** `QgfVarianceSignal` trait + `confidence_from_disagreement` (`katgpt-rs/crates/katgpt-core/src/qgf/adaptive.rs:141-174`). The docstring explicitly names "ensemble KL" as a future implementor — VFD closes this open item.
- **UQ floor:** `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340). Plan 376 Phase 6 benchmarked the velocity-field ensemble against this floor on AR(1) — Plan 432 G2 must benchmark VFD-calibrated intervals against the same floor.
- **Super-GOAT this activates:** `riir-ai/.research/170_per_npc_velocity_field_ensemble_composition_guide.md` (the per-NPC velocity-field ensemble Super-GOAT guide; its G7 gate is exactly what VFD fills).
- **Sibling UQ primitive:** BoMSampler (Plan 281, Research 281) — sample-disagreement UQ for autoregressive models. VFD is the velocity-field-disagreement UQ for flow-matching models. Complementary.
- **DEC cross-check:** `belief_mass_divergence`, `codifferential` (Plan 314, Research 296). Independent sanity signal.
- **Training-method redirect:** SAVE (§5 of the paper) → `riir-train/.research/` (active fine-tuning via VFD-guided demonstration acquisition — gradient descent, out of scope for this workflow).

---

## TL;DR (one-line)

VFD (Velocity-Field Disagreement) is the modelless epistemic-UQ estimator that activates the deferred G7 UQ gate of the velocity-field ensemble Super-GOAT (R375/P376): given M=2 frozen velocity fields, integrate one member's ODE, evaluate the other member at those states, weight pairwise differences by `κ_s = s/(1−s)` (already shipped via `Schedule::optimal_diffusion`), produce a calibrated KL-approximating epistemic-UQ scalar per inference — Verdict **GOAT** (new mechanism for an existing capability on an existing substrate, not a new pillar; SAVE active-fine-tuning half → riir-train).
