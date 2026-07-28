# Research 460: CD-LAM — Latent Confounder Audit Diagnostics

> **Source:** [Causally Debiased Latent Action Model for Embodied Action Conditioned World Models](https://arxiv.org/abs/2607.09185) — Yufan Wei, Kun Zhou, Lingjun Mao, et al. (Aether AI / UCSD), arXiv:2607.09185v1, 10 Jul 2026
> **Code:** [yufanwei/CD-LAM](https://github.com/yufanwei/CD-LAM) (Apache-2.0) — PyTorch, 96×H100, three-stage fine-tuning pipeline
> **Date:** 2026-07-28
> **Status:** Done — verdict locked (**GAIN for katgpt-rs** [diagnostics]; training recipe → riir-train)
> **Classification:** Public (this note). Training recipe → riir-train.
> **Related Research:** 374 (OTF-LAM — factorized cousin; different solution to related problem), 360 (AdaJEPA — JEPA world-model PASS precedent), 053 (CNA — modelless contrastive cousin), 324 (Latent Trajectory Geometry — geometric diagnostic cousin, different axis), 457→causal_id (graph-level confounder identification cousin, different abstraction level), 276 (PersonalityWeightedComposition — direction vector consumer), 418 (MAG — runtime-mined directions, primary audit target), 425 (TILR — trajectory-invariant refinement, audit target), 309 (Latent Field Steering — runtime direction updates, audit target), 321 (Committed Personality Blend — archetype directions, audit target)
> **Related Plans:** TBD via Issue 194 (the `LatentConfounderAudit` primitive implementation)
> **Domain:** katgpt-rs (this note, public — the generic diagnostic primitive)

> **Correction note (2026-07-28):** this paper was initially verdicted PASS in commit `feb2273b` on the grounds that "OTF-LAM solves the same problem." That was a false-PASS — OTF-LAM factorizes transitions (different problem, different solution), and the diagnostic framework (§III-B + Appendix A) is a transferable modelless primitive that does not ship. The verdict was revised to GAIN for the diagnostic half after deeper grep + property-level vocabulary translation. See §7 for the honest correction record.

---

## TL;DR

CD-LAM identifies a real failure mode — **reconstruction-trained conditioning latents entangle action-relevant signal with action-irrelevant confounders** — and provides both a training cure (three fine-tuning losses) and a diagnostic framework (three forward-pass metrics) to measure it. The training cure is genuinely gradient-descent and routes to riir-train. **The diagnostic framework is the transferable modelless half**: three properties that audit any conditioning latent for confounder purity, computable in pure forward passes with zero gradient steps.

**Distilled for katgpt-rs (modelless, inference-time):** a `LatentConfounderAudit` primitive — three norm/cosine diagnostics over an encoder function `E: (Obs, Obs) → Latent`:

1. **Zero-transition response** `R₀ = ‖E(x, x)‖ / D` — a no-op input should produce a near-zero latent.
2. **Shift-invariance response** `R_shift = ‖E(x, T(x))‖ / D` — a nuisance-transformed input should produce a near-zero latent.
3. **Shortcut leakage** `L_shortcut = E[cos(zᵢ, zⱼ) | same-action, diff-context] − E[cos(zᵢ, zⱼ) | diff-action, same-context]` — action similarity should dominate context similarity in cosine structure.

Where `D = RMS(‖E(x, x′)‖)` over ordinary transitions (normalization denominator). All three are O(d) per check, zero-allocation, feature-gated. Tracked in Issue 194.

---

## 1. Paper Core Findings

### 1.1 The confounder problem (§III-A)

A Latent Action Model (LAM) encodes a frame transition `(oₜ, oₜ₊₁)` into a latent action `zₜ` used to condition a world model. Trained with reconstruction-only loss, nothing prevents `zₜ` from carrying **action-irrelevant confounders**:

```
zₜ ≈ f(Aₜ, Cₜ, Vₜ)
```

where `Aₜ` = embodiment dynamics (the signal we want), `Cₜ` = scene context, `Vₜ` = source-side visual factors (background, camera). The reconstruction objective only requires *predictive sufficiency* — any transition-predictive factor can enter the latent. Once `Cₜ`/`Vₜ` leak into `zₜ`, the world model's action condition is partly specifying video continuation rather than embodiment dynamics. Robot action adaptation then inherits this contaminated latent.

### 1.2 Three diagnostic metrics (§III-B, Appendix A) — the modelless half

CD-LAM defines three forward-pass diagnostics computed on the LAM encoder `μϕ(oₜ, oₜ₊₁)` alone, before any world-model rollout. A clean latent action space should pass all three:

| Diagnostic | Formula | What it tests | Clean value |
|---|---|---|---|
| **Zero-transition response** | `R₀ = ‖μϕ(oₜ, oₜ)‖₂ / D` | Duplicated-frame input → should produce near-zero latent | ≈ 0 |
| **Camera-shift response** | `R_shift = ‖μϕ(oₜ, T₃(oₜ))‖₂ / D` | Synthetic 3-pixel translation → should produce small latent | ≈ 0 |
| **Shortcut leakage** | `L = E[cos(zᵢ,zⱼ) \| same-ep, diff-prim] − E[cos(zᵢ,zⱼ) \| diff-ep, same-prim]` | Same-primitive/diff-context should be closer than diff-primitive/same-context | < 0 |

Where `D = RMS(‖μϕ(oₜ, oₜ₊₁)‖₂) + ε` over the audit split (normalization). `T₃` translates by 3 pixels. The shortcut leakage uses coarse primitive labels (12-way verb categories).

**Empirical evidence (Table I):** DreamDojo's reconstruction-trained LAM fails all three — zero-transition response 0.527 (median), camera-shift 0.555/0.545 (mean h/v), shortcut leakage 0.151. After CD-LAM debiasing: 0.043, 0.156/0.110, 0.014. The diagnostics correctly identify the confounded encoder.

### 1.3 Three debiasing objectives (§IV) — the training half (→ riir-train)

1. **Embodiment-centric reconstruction** `L_emb` — weighted MSE with SAM3 foreground mask: `W = α_fg·M + α_bg·(1−M)`, `α_fg > α_bg`. Emphasizes embodiment-dynamics regions.
2. **Action-centric contrastive learning** `L_ctr` — softplus loss on primitive-pair cosine: `softplus(−yᵢⱼ(τ·vᵢᵀvⱼ + b))`. Same-primitive pulled together, different pushed apart.
3. **Latent space calibration** `L_cal = L_KL-fb + L_zero` — free-bit KL (capacity control) + zero-transition anchoring (`‖z⁰ₜ‖ / sg(s_Δ) < m_zero`).

Three-stage pipeline: LAM debias (1k steps) → ACWM debias (2k steps) → robot action bridge MLP (3k–6k steps). Results: FDCE drops 30–42%, 12× fewer adaptation updates.

---

## 2. Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent | Where it applies |
|---|---|---|
| latent action `zₜ` | HLA state delta, functor displacement, shard projection, MAG-mined direction projection | `sense/`, `latent_functor/`, `NeuronShard::project`, MAG directions |
| encoder `μϕ(oₜ, oₜ₊₁)` | any `Fn(&Obs, &Obs) → Latent` | `evolve_hla`, `extract_functor`, shard project, dot-product projection |
| zero-transition `(oₜ, oₜ)` | identical observation pair, no-op transition | feed same obs twice through HLA/functor |
| camera-shift `T₃(oₜ)` | nuisance transform on observation (tick drift, coordinate offset, view transform) | apply irrelevant transform, check latent delta |
| shortcut leakage | context confound in cosine structure | same-decision/diff-context vs diff-decision/same-context |
| embodiment-centric (foreground) | decision-relevant subspace | the signal the latent *should* encode |
| action-irrelevant confounder | factors that shouldn't influence the latent but do | background context leaking into HLA affect |
| primitive labels (12-way verbs) | coarse behavior/action categories | game action types (move/attack/flee/harvest/...) |
| reconstruction objective | (N/A — we don't train via reconstruction) | our latents are constructed or runtime-mined, not reconstruction-trained |

**Key translation insight:** the reconstruction objective is the *root cause* of confounders in CD-LAM's setting. Our HLA direction vectors are partly hand-constructed (where the property holds by construction) but also **runtime-mined** (MAG Plan 418, TILR Plan 425, Latent Field Steering Plan 309) — mined directions CAN carry confounders because the mining signal (activation geometry, trajectory invariance) is not a purity constraint.

---

## 3. Distillation — what's modelless, what's training

### 3.1 Split verdict

| Component | Route | Reason |
|---|---|---|
| L_emb (foreground-weighted MSE) | → riir-train | Gradient-descent loss optimizing encoder/decoder weights |
| L_ctr (contrastive softplus) | → riir-train | Gradient-descent loss organizing latent space via primitive labels |
| L_cal (KL + zero-norm margin) | → riir-train | Gradient-descent capacity control + anchoring |
| Bridge MLP gη | → riir-train | Supervised regression needing labeled robot data |
| **Three diagnostics (R₀, R_shift, L_shortcut)** | **→ katgpt-rs (GAIN)** | **Pure forward-pass norm/cosine checks — zero gradient steps** |

### 3.2 §3.5 Path 0 — training-target decomposition (for the objectives)

- L_emb = weighted MSE → **TRAINING LOSS**. The weighting (foreground mask) is a training signal; the MSE optimizes weights. No modelless analog (it IS the gradient).
- L_ctr = contrastive softplus → **TRAINING LOSS**. Shapes latent space via backprop on labels. The modelless analog (CNA, Research 053) discovers contrastive structure post-hoc but cannot *reshape* a frozen encoder's latent space — different operation.
- L_cal = KL + zero-norm margin → **TRAINING LOSS**. Capacity control during training; anchoring requires gradient steps to push zero-transition latents toward zero.

**Path 0 conclusion:** all three objectives are genuinely gradient-descent losses. Paths 1–3 (freeze/thaw, deterministic LoRA, latent projection) cannot remove bias baked into trained weights without retraining. Training recipe → riir-train.

### 3.3 The diagnostics survive the modelless test

The three metrics are computed by:
1. Calling the encoder on specific input pairs (same-same, shifted, labeled pairs)
2. Computing norms and cosines on the outputs
3. Normalizing by the RMS of ordinary-transition norms

Zero gradient steps. Zero weight mutation. Pure forward-pass measurement. They are inference-time diagnostics — they audit a frozen encoder, they don't change it.

### 3.4 Fusion — LatentConfounderAudit × MAG × TILR × Latent Field Steering × HLA

The novel combination this paper enables:

```
         ┌──────────────────────────────────────────────────┐
         │  LatentConfounderAudit (3 forward-pass checks)   │
         │  R₀ + R_shift + L_shortcut on any encoder E      │
         └─────────────┬────────────────────────────────────┘
                       │ audits
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
  HLA evolve      extract_functor    MAG-mined direction
  (per-NPC        (displacement      (runtime activation
   affect)         vector)            geometry mining)
                       │
                       ▼
              "Is this mined direction carrying
               action-irrelevant confounders?"
                       │
              ┌────────┴────────┐
              ▼                 ▼
         PASS (deploy)     FAIL (reject + re-mine)
```

Today, MAG (Plan 418) mines direction vectors from activation geometry with no purity check. A mined direction could project onto confounding factors (background activation patterns that correlate with the target behavior but don't cause it). The audit would catch this before deployment — run the three checks on the mined direction, reject if confounders detected.

---

## 4. Mandatory latent-space reframing (per SKILL §1 step 3)

How the three diagnostics look when applied to the codebase's latent-state kernels:

**(a) HLA per-NPC latent state** (`evolve_hla: (state, obs) → state'`):
- R₀: feed identical obs twice → the state delta `‖state' − state‖` should be ≈ 0. If not, the direction vectors project onto obs features that don't change but still produce non-zero dot products (a confounder).
- R_shift: apply an irrelevant obs transform (e.g., shift all values by a constant) → the HLA delta should not change. If it does, direction vectors are sensitive to nuisance variation.
- L_shortcut: same-emotion/diff-zone NPC pairs should have closer HLA deltas than diff-emotion/same-zone pairs. If reversed, zone context is leaking into the affect representation.

**(b) latent_functor** (`extract_functor: (source, target) → displacement`):
- R₀: source == target → displacement should be ≈ 0.
- R_shift: shift both source and target by the same irrelevant offset → displacement should not change (translation invariance).
- L_shortcut: same-displacement/diff-scene pairs should be closer than diff-displacement/same-scene.

**(c) NeuronShard style_weights** (`shard.project(obs) → latent`):
- R₀: project same obs twice → identical result (determinism check, trivially passes).
- R_shift: irrelevant obs transform → projection should be invariant.
- L_shortcut: same-style/diff-content pairs closer than diff-style/same-content.

**(d) MAG-mined direction vectors** (`dot(d, activation) → projection`):
- R₀: same activation twice → same projection (trivially passes).
- R_shift: irrelevant activation transform → projection should be invariant.
- L_shortcut: same-behavior/diff-context activations project closer than diff-behavior/same-context. **This is the primary audit target** — MAG has no built-in purity check.

---

## 5. §3.5 Modelless unblock check (for the training objectives)

Already covered in §3.2 above. Path 0 fails for L_emb/L_ctr/L_cal (genuinely gradient-descent losses). Paths 1–3 cannot fix biased trained weights. **Training recipe → riir-train.**

The diagnostics are not subject to §3.5 — they are already modelless by construction (forward-pass measurement, not a gate to unblock).

---

## 6. Novelty gate (§1.5)

| Q | Criterion | Answer | Evidence |
|---|---|---|---|
| Q1 | No prior art? | **YES** | Grep for `confound.audit|zero.transition|shortcut.leak|latent.audit|bias.audit` across `**/*.rs` = zero hits in latent-vector code. `causal_id` (Plan 457) handles confounders at KG-triple graph level (ADMG bidirected edges), not latent-vector level. `latent_trajectory_geometry` (Plan 342) measures trajectory shape (length/curvature/cosine), not representational purity. `compaction/audit.rs` audits compaction decisions, not latent confounders. Different abstraction levels. |
| Q2 | New class of behavior? | **Partial** | A new *diagnostic class* (latent purity audit), but not a new *capability*. It catches bugs/misconfigurations; it doesn't unlock something NPCs couldn't do before. |
| Q3 | Product selling point? | **Weak** | "Our mined direction vectors are audited for confounders" is a quality/reliability argument, not a capability unlock. Cannot finish "our NPCs do X that no competitor can" with this alone. |
| Q4 | Force multiplier? | **YES** | Connects to MAG (418), TILR (425), Latent Field Steering (309), Committed Personality Blend (321), HLA, functor, shard projections — 7 systems across multiple pillars. |

**Q2 partial + Q3 weak → NOT Super-GOAT. GAIN candidate only.** Ship behind a feature flag, no private guide needed.

---

## 7. MOAT gate per domain (§1.6)

| Domain | Verdict | Reason |
|---|---|---|
| **katgpt-rs** | **GAIN — ship here** | Generic diagnostic math (norm ratios, cosine gaps) — the public engine's quality tool. Feature-flagged, GOAT-gated. |
| riir-ai | Consumer | Will call the primitive to audit runtime-mined directions (MAG, TILR, Latent Field Steering). No new code in riir-ai initially — consumes the public primitive. |
| riir-chain | N/A | Not a chain concern. |
| riir-neuron-db | Consumer | Could audit loaded shard projections before deployment. |
| riir-train | **Training recipe** | L_emb + L_ctr + L_cal + pipeline. Out of scope for this note. |

**katgpt-rs is the correct home.** The primitive is generic math over any encoder function; the katgpt-rs engine is where generic modelless inference primitives live.

---

## 8. Honest correction record — the false-PASS

**Initial verdict (commit `feb2273b`, 2026-07-28):** PASS (→ riir-train). Reasoning: "OTF-LAM solves the same problem; contrastive piece has CNA cousin; diagnostics are video-specific."

**Why it was wrong:**

1. **OTF-LAM ≠ CD-LAM.** OTF-LAM (Research 374) factorizes transitions into K codebook primitives via VQ + sigmoid relevance gate — a *structural* solution to agent ambiguity. CD-LAM *identifies three named confounder types and provides targeted debiasing objectives* — a *corrective* solution to representational impurity. Different problems, different mechanisms. Claiming "factorized_action solves the same problem" was architectural hand-waving without a PoC (§3.6 violation).

2. **The diagnostics are NOT video-specific.** The three properties (zero-transition response, shift-invariance, shortcut leakage) are general latent-space hygiene invariants. The *paper's instantiation* uses video-specific transforms (SAM3 masks, pixel shifts, point tracks), but the *properties themselves* translate to any conditioning latent. Dismissing them as "video-specific" skipped the property-level vocabulary translation the skill mandates.

3. **CNA is NOT the modelless analog of L_ctr.** CNA (Research 053) discovers which neurons matter for a behavior from forward-pass activation differences. L_ctr shapes the latent space during training via backprop on primitive labels. CNA is a *post-hoc discovery* tool; L_ctr is a *training-time organization* tool. The actual modelless analog of L_ctr's *effect* would be runtime latent-space reorganization, which is closer to TILR (425) or subspace steering (412) — not CNA.

**Lesson:** the #1 false-PASS failure mode (architectural coverage claim without PoC) struck exactly as the skill warns. The correction required:
- Property-level vocabulary translation (not just mechanism-name matching)
- Grep at the right abstraction level (latent-vector confounders, not graph-level or trajectory-shape)
- Honest distinction between "related problem" and "same problem"

---

## 9. Verdict

**GAIN for katgpt-rs** — the diagnostic framework is a modelless, novel, actionable primitive. Ship behind `latent_confounder_audit` feature flag. GOAT gate in Issue 194.

**Training recipe → riir-train** — L_emb + L_ctr + L_cal + three-stage pipeline. §3.5 Path 0 confirmed: genuinely gradient-descent losses.

**Not Super-GOAT** — Q2 (new capability class) and Q3 (product selling point) fail. This is a diagnostic/quality tool, not a capability unlock. No private guide needed.

---

## TL;DR

CD-LAM's training recipe (three debiasing losses + fine-tuning pipeline) is genuinely → riir-train. But the paper's **diagnostic framework** — three forward-pass metrics (zero-transition response, shift-invariance, shortcut leakage) that audit any conditioning latent for confounder purity — is a modelless, novel, actionable primitive for katgpt-rs. It doesn't ship. It applies to our runtime-mined direction vectors (MAG, TILR, Latent Field Steering) where confounders could leak in without detection. Verdict: GAIN. Tracked in Issue 194.
