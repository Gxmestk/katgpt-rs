# Research 451: Delta-Lattice Tunneling → Transfer-Matrix Band-Structure Analyzer

> **Sources (cross-domain literature):**
> - **Physics (the original):** Kronig & Penney, *Quantum Mechanics of Electrons in Crystal Lattices* (Proc. Roy. Soc. A **130**, 499–513, 1931) — the periodic delta-function (Dirac-comb) lattice + the band-structure result `|Tr(M/2)| ≤ 1` defines allowed bands.
> - **Textbook treatment (delta-function version):** D. J. Griffiths, *Introduction to Quantum Mechanics* §5.3; M. Born & E. Wolf, *Optics* §1.6 (multilayer-film transfer matrices — the same math in classical optics, no quantum required).
> - **ML — Deep Equilibrium Models (the headline ML anchor):** Bai, Koltun, Kolter, *Stabilizing Equilibrium Models by Jacobian Regularization*, **[arXiv:2106.14342](https://arxiv.org/abs/2106.14342)** (ICML 2021 Short Oral). The DEQ fixed-point Jacobian `J_*` IS the per-period transfer matrix in our reframing. Their `ρ(J_*) < 1` condition is the band-stability criterion. Their training-time scalar spectral-radius regularizer is the single-number version of what our primitive provides as a *runtime, per-mode, classified band structure*.
> - **ML — Weight-matrix spectral properties (model-quality signal):** Martin & Mahoney, *Implicit Self-Regularization in Deep Neural Networks*, **[arXiv:1810.01075](https://arxiv.org/abs/1810.01075)** (2018) + *Traditional and Heavy-Tailed Self Regularization*, **[arXiv:1901.08276](https://arxiv.org/abs/1901.08276)** (ICML 2019). The empirical spectral density (ESD) of a DNN weight matrix carries the model's training phase + generalization signature. The 5+1 phases of training correspond to increasing amounts of self-regularization, observable in the ESD. This is the same lens (matrix spectrum → model property) applied offline to trained weights — our primitive applies it at runtime to per-tick operator chains.
> - **ML — Loss-surface geometry:** Pennington, Schoenholz, Ganguli, *Geometry of Neural Network Loss Surfaces via Random Matrix Theory* (ICML 2017, [proceedings.mlr.press/v70/pennington17a](https://proceedings.mlr.press/v70/pennington17a.html)). Eigenvalue distribution of the Hessian at critical points classifies them by index.
> - **ML — Recurrent network stability (classical):** Arjovsky, Shah, Bengio, *Unitary Evolution RNNs* (ICML 2016) — constrain `|λ|=1` (the unit circle = allowed band edge) for stable long-term propagation; Vorontsov et al., *Orthogonal RNNs*; coRNN, nnRNN — all use the same eigenvalue-magnitude criterion this primitive formalizes as a band classifier.
> **User prompt (2026-07-18):** "Delta Lattice Tunneling sound cool, can we make use of that some how?"
> **Date:** 2026-07-18
> **Status:** Active
> **Related Research:** 311 (Analytic Lattice — closest cousin: cross-entity operator composition), 296 (Stokes/DEC vocabulary crosswalk), 279 (Diffusion curse of dimensionality / subspace clustering — closest cousin for `subspace_phase_gate`), 051 (Deep Manifold boundary conditions), 169 (Oscillatory State-Space Modelless Distillation — closest spectral-stability cousin), 205 (Deep Manifold neural-network math), 219 (Topological Neural Operators / DEC), 230 (Semiseparable State-Space Duality — linear-recurrence transfer-matrix angle)
> **Related Plans:** 330 (Analytic Lattice Encoder/Decoder — closest plan), 301 (Subspace Phase-Gate — closest plan for "phase transition"), 251 (DEC operators — shared substrate), 353 (`riir-ai` HLA Lean boundedness proofs — the per-tick analog this would extend to multi-tick), 458 (Transfer Matrix Band-Structure — the plan for this primitive)
> **Classification:** Public (katgpt-rs)

---

## TL;DR

The Kronig-Penney delta-lattice model is a 1931 quantum-mechanics toy model: a wave packet hits a 1D periodic array of delta-function barriers V(x) = Σ V₀·b·δ(x − na) and either **tunnels through** (allowed band, propagating) or **is reflected** (forbidden gap, evanescent). The math that survives when you strip the physics is the **transfer-matrix method**: propagate a state vector `[ψ, ψ']^T` through a sequence of media by composing per-segment 2×2 (or k×k) matrices M_n, then read off the **band structure** from the eigenvalues of the composite M.

**Distilled for katgpt-rs (modelless, inference-time):** a generic `TransferMatrixBandStructure` primitive that takes a sequence of square matrices `[M_1, M_2, …, M_N]` (or a single periodic `M` applied N times) and returns:

1. **Composite** `M = M_N · M_{N−1} · … · M_1` (already half-shipped as `analytic_lattice::compose_chain`).
2. **Eigenvalues** `{λ_i}` of `M` (or of the per-period `M`).
3. **Bloch propagation factor** `μ_i = λ_i^(1/N)` per eigenvalue — the per-site growth/decay factor.
4. **Band classification**: `|μ_i| ≈ 1` → **propagating** (allowed band); `|μ_i| < 1 − ε` → **evanescent / decaying** (forbidden gap, suppresses mode); `|μ_i| > 1 + ε` → **unstable / growing** (forbidden, runaway mode).
5. **Band edges**: tick/snapshot indices where a mode crosses `|μ| = 1` (resonant tunneling peaks).
6. **Transmission amplitude** `t = (M⁻¹)_00` (entry-to-exit coupling for a finite stack).
7. **Reflection amplitude** `r = −M₁₀·M₀₀⁻¹` (what bounces back, never propagates).

Pure linear algebra. Zero gradient descent, zero Schrödinger equation, zero physics constants. The QM vocabulary ("Bloch factor", "Dirac comb", "Brillouin zone") is just *naming* — the math is a stability/eigenvalue diagnostic over a stack of linear operators.

**Verdict: Gain.** The mechanism is genuinely novel to our stack — verified by grep across all 7 repos in BOTH vocabularies (`kronig|penney|dirac_comb|delta_lattice|bloch_theorem|brillouin|transfer_matrix|band_gap|tunneling|allowed_band|forbidden_band|spectral_gap|wave_function|schrodinger|tight_binding|periodic_barrier|scattering_matrix` → **zero hits**, except the unrelated `spectral_gap` in `crates/katgpt-spectral/src/spectral.rs` which is a participation-ratio for KV compression, completely different concept). It is **strictly more than `analytic_lattice::compose_chain` ships**: that primitive returns the composite operator only; this primitive analyzes its *spectral band structure* and classifies modes as propagating / decaying / growing. The closest conceptual cousin is `subspace_phase_gate.rs` (`participation_ratio`, `numerical_rank`, `phase_transition_gate`), which gives the discrete `N ≥ d` *sample-sufficiency* threshold — but does not characterize multi-step propagation.

But it is **not Super-GOAT** (Q2/Q3 fail):
- **Q2 (new class of behavior)** — NO. It is a *measurement* (band structure of an existing operator sequence), not a *capability*. HLA's per-tick boundedness is already proven (Plan 353 Lean theorems); this extends the *same idea* to multi-tick propagation. Diagnostics are GOAT/Gain-tier, not Super-GOAT.
- **Q3 (product selling point)** — NO clear sentence. "Our NPCs have band-gap-aware latent propagation" is interesting engineering, not a customer-facing feature. No "we do X that no competitor can" snaps into focus.
- **Q4 (force multiplier)** — YES but thin. Connects to: `analytic_lattice/compose_chain` (consumes its output), `subspace_phase_gate` (extends `phase_transition_gate` from sample-sufficiency to mode-propagation), `latent_functor/reestimation` (band-edge as a re-estimation trigger signal), HLA Lean proofs (the multi-tick extension), DEC `hodge_decompose` (band structure on cochains). But every consumer integration is "use as a diagnostic signal" — not "create new behavior".

Per §1.55, this is **Gain, not Pass** — genuinely novel modelless math, real fusion hooks, zero grep prior art. A plan should ship the primitive behind a feature flag with a GOAT gate (latency + correctness vs naive eigenvalue methods), and the **multi-tick HLA stability diagnostic** as the headline consumer integration. The Super-GOAT re-evaluation is tracked as TBD — needs a concrete consumer that converts the band-structure signal into a behavior (e.g., "auto-prune forbidden modes before they destabilize the HLA recurrence"), at which point Q2/Q3 may flip and the gate should be re-run.

---

## 1. Paper Core Findings

### 1.1 The Kronig-Penney model

The 1931 Kronig-Penney model is a 1D toy model for an electron in a crystal lattice. The potential is periodic:

```
V(x) = Σ_{n=−∞}^{∞} V₀ · b · δ(x − n·a)
```

where `a` is the lattice spacing, `V₀` is the barrier strength, and `b` is the barrier width (the delta limit takes `b → 0, V₀ → ∞` such that `V₀·b` stays finite). Solutions to the time-independent Schrödinger equation `-ℏ²/2m · ψ'' + V·ψ = E·ψ` obey **Bloch's theorem**: `ψ(x + a) = e^{ika} · ψ(x)`, where `k` is the crystal momentum (the "Bloch factor").

Matching wavefunction + derivative across each delta gives a 2×2 **transfer matrix** M_n per period:

```
[ ψ(x+a)  ]   [ 1   0 ] [ cos(κa) + (V₀/κ)·sin(κa)   (1/κ)·sin(κa) ] [ ψ(x)  ]
[ ψ'(x+a) ] = [ V₀  1 ] [ −κ·sin(κa) + V₀·cos(κa)     cos(κa)       ] [ ψ'(x) ]
                ──────────   ───────────────────────────────────────────
                delta jump             free propagation
```

For N periods: `M = M_n^N`. The eigenvalues of `M_n` determine everything:

- **λ = e^{±ika}** with real `k` → propagating Bloch wave → **allowed band**.
- **λ = e^{±κa}** with real `κ` → exponentially decaying → **forbidden gap**.
- The condition `|Tr(M_n)/2| ≤ 1` defines the allowed band; `|Tr(M_n)/2| > 1` defines the gap.

### 1.2 The transfer-matrix method (the transferable primitive)

Strip the physics — the math is substrate-independent:

**Given** a sequence of k×k matrices `M_1, M_2, …, M_N` (or a single periodic `M` applied N times), **define**:

| Quantity | Formula | Meaning |
|---|---|---|
| Composite | `M = M_N · M_{N−1} · … · M_1` | The full propagation operator |
| Eigenvalues | `λ_i ∈ spectrum(M)` (or `spectrum(M_n)` for periodic) | Mode-specific growth/decay factors |
| Bloch factor | `μ_i = λ_i^{1/N}` (or `λ_i` itself for the per-period matrix) | Per-site propagation factor |
| Band class | propagating if `|μ_i| ≈ 1`, decaying if `< 1 − ε`, growing if `> 1 + ε` | Allowed vs forbidden |
| Transmission | `t = (M⁻¹)_{00}` | Entry-to-exit amplitude |
| Reflection | `r = −M_{10} · M_{00}^{-1}` | Back-scattered amplitude |
| Resonance | local maxima of `|t|` as a function of "energy" (input frequency / scale) | Tunneling peaks |

The **resonance peaks** are the headline emergent behavior: for certain input frequencies, a finite stack becomes perfectly transparent (`|t| = 1`). In latent-space terms: certain input directions pass through N layers completely unchanged — a zero-distortion channel.

### 1.3 Why "Dirac comb" specifically

The **delta-function simplification** is what makes the math closed-form. For a general potential `V(x)`, M_n must be computed numerically (e.g. by solving the ODE). For delta barriers, M_n has an analytic closed form — making the whole pipeline a pure linear-algebra computation, no PDE solver needed. **This is what makes the model distillable into a katgpt-rs primitive.**

In our codebase vocabulary: a "delta barrier" = a per-stage *discrete jump operator* applied between two *free-propagation steps*. This already has a name in the codebase — it's literally what `TransportOperator` in `analytic_lattice/mod.rs` is. The missing piece is *eigenvalue analysis* of the composed chain.

### 1.4 ML literature anchors (the grounding this is NOT just a physics toy)

The transfer-matrix method is not just physics — it is the **stability theory of linear dynamical systems**, which is the foundation of multiple active ML research areas. Three lines of ML literature converge on the same math:

**1. Deep Equilibrium Models + Jacobian regularization (the headline anchor).** Bai, Koltun, Kolter ([arXiv:2106.14342](https://arxiv.org/abs/2106.14342), ICML 2021) train DEQ models by adding a Jacobian regularization loss `||J_*(z)||_F²` to push the spectral radius `ρ(J_*) < 1`. This is the **single-number, training-time, scalar-regularizer version of the band-stability criterion**. Their `J_*` IS the per-period transfer matrix in Kronig-Penney vocabulary. Their `ρ(J_*) < 1` IS the `|μ| < 1` "decaying band" classifier. What they do NOT do:

- **Mode-resolved classification** — they regularize the Frobenius norm (a scalar), not per-eigenvalue band labels.
- **Runtime diagnostic** — their loss is computed during training; the deployed model has no `J_*` introspection.
- **Actionable as a pruner** — they nudge weights via SGD; they do not project out unstable modes at inference time.

Our primitive extends Bai/Kolter along exactly those three axes: **per-mode** + **runtime** + **pruner-ready**. This is a modelless extension of their training-time regularizer — and the extension is non-trivial because (a) computing per-mode eigenvalues at runtime needs to be sub-µs, (b) classifying into bands needs a principled `ε` threshold (the band-edge), and (c) the projection step (if the pruner ships) needs to preserve the model's expressed behavior while suppressing the unstable direction.

**2. Implicit self-regularization + ESD phases (the weight-matrix angle).** Martin & Mahoney ([arXiv:1810.01075](https://arxiv.org/abs/1810.01075) + [arXiv:1901.08276](https://arxiv.org/abs/1901.08276), ICML 2019) read off a DNN's training quality from the empirical spectral density (ESD) of its weight matrices. Their "5+1 phases of training" correspond to increasing amounts of self-regularization, observable as a transition from bulk-Marchenko-Pastur ESD → heavy-tailed ESD with a few outlier eigenvalues. **This is the same lens (matrix spectrum → model property) applied offline to trained weights.** Our primitive applies it at runtime to per-tick operator chains — the analog for HLA / `latent_functor` is: "how many modes are in the allowed band, how many in the decaying band, how many in the growing band?". A clean ESD with no growing-band modes is the runtime analog of Martin/Mahoney's well-regularized phase.

**3. Orthogonal/unitary RNNs (the classical recurrence angle).** Arjovsky et al. (uRNN, ICML 2016), Vorontsov et al. (orthogonal RNNs), coRNN, nnRNN all constrain the recurrent weight matrix to have `|λ| = 1` (orthogonal) or `|λ| = 1` on a related manifold (unitary). This is the **band-edge constraint** — every mode sits exactly on `|μ| = 1`, the boundary between allowed and forbidden. Their motivation (long-term dependency learning) is exactly the band-stability problem: a mode with `|μ| < 1` decays and forgets; a mode with `|μ| > 1` explodes. Their solution is to hard-constrain the matrix to the band edge; our primitive is the diagnostic that tells you, for an *unconstrained* matrix, where each mode sits.

**Why this grounding matters for the verdict:** the math is well-established ML theory, not a stretch from physics. The Bai/Kolter paper alone is a direct precedent — the proposal here is a strict superset of their regularizer (per-mode + runtime + pruner-ready). This moves the verdict from "interesting physics-to-ML crossover" to "well-grounded ML primitive with a clear delta over a cited ICML 2021 paper". The Gain tier holds; the TBD path to Super-GOAT (via the band-gap pruner) becomes more credible because Bai/Kolter already proved the training-time regularizer works.

---

## 2. Distillation — fusion protocol

### 2.1 Vocabulary translation (paper ↔ codebase)

| Paper / physics term | Codebase-equivalent | Verified shipped? |
|---|---|---|
| "transfer matrix M_n per period" | `TransportOperator { k, data: Vec<f32> }` row-major k×k | YES (`crates/katgpt-core/src/analytic_lattice/mod.rs`) |
| "compose M = M_N · … · M_1" | `compose_chain(&[TransportOp]) -> TransportOp` / `compose_chain_into` | YES (`analytic_lattice/chain.rs`) |
| "batch prefix factoring" | `batch_compose_chain` / `batch_compose_chain_into` | YES (`crates/katgpt-core/src/analytic_lattice/batch_chain.rs`) |
| "Bloch factor μ = λ^{1/N}" | (NEW — needs eigenvalue + nth-root) | NO — gap |
| "allowed band / forbidden gap" classification | (NEW — band classifier) | NO — gap |
| "transmission amplitude t" | (NEW — matrix inverse + entry read) | NO — gap |
| "reflection amplitude r" | (NEW) | NO — gap |
| "resonant tunneling peak" | (NEW — local-max of |t| over input freq) | NO — gap |
| "Dirac comb / delta barrier" | `TransportOperator` with jump structure | partial — `TransportOperator` is general |
| "lattice spacing a" | tick / layer / round / snapshot index | n/a (consumer semantics) |
| "energy E" (input frequency / scale) | input direction in latent space | n/a (consumer semantics) |
| "Brillouin zone" | the closed set of propagating input directions | NO — gap (new diagnostic) |
| "spectral gap" (band-edge gap) | (NOT the same as `katgpt-spectral::spectral_gap` which is participation-ratio for KV compression) | n/a — different concept, same name |
| "phase transition N ≥ d" | `subspace_phase_gate::phase_transition_gate` | YES (closest cousin for "phase transition") but operates on samples, not on operator eigenvalues |

**Closest shipped spectral concepts** (verified by grep, all different mechanisms):
- `crates/katgpt-spectral/src/spectral.rs::spectral_gap(eigenvalues, d_eff) -> Option<f32>` — participation-ratio gap `λ_{d_eff}/λ_{d_eff+1}`, used for KV-cache compression bit allocation. **NOT** band-structure spectral gap.
- `subspace_phase_gate.rs::participation_ratio` / `numerical_rank` / `phase_transition_gate` — intrinsic-dimension estimation + the `N ≥ d` sample-sufficiency necessary condition. **NOT** multi-step propagation analysis.
- `crates/katgpt-dec/src/hodge.rs::betti_numbers` — counts zero eigenvalues of the Hodge Laplacian (topological holes in a cochain). **NOT** eigenvalue magnitude classification.
- `katgpt-dec/src/operators.rs::hodge_laplacian` — gives `δd + dδ`, whose spectrum *can* feed band-structure analysis (fusion hook, see §3.6).

### 2.2 Closest cousins (4 — 3 shipped + 1 cited ML paper)

1. **`analytic_lattice::compose_chain`** (Plan 330, ships in `crates/katgpt-core/src/analytic_lattice/`) — **closest by far**. Composes a sequence of k×k `TransportOperator`s via row-major matmul. The proposed primitive is literally `eigenvalues(compose_chain(ops)) + band_classify(λ) + transmission(M)`. The matmul chain already exists; only the spectral analysis is missing.

2. **`subspace_phase_gate::phase_transition_gate`** (Plan 301, ships in `crates/katgpt-core/src/subspace_phase_gate.rs`) — **closest "phase transition" cousin**. Implements the necessary condition `N ≥ d` from Wang et al. (arxiv 2409.02426) Theorem 4: a d-dimensional subspace cannot be recovered from fewer than d samples. The proposed primitive extends "phase transition" from *sample sufficiency* to *mode propagation* — same mathematical shape (`|Tr(M)/2| ≤ 1` is the `N ≥ d` analog for transfer matrices).

3. **`riir-ai` HLA boundedness Lean proofs** (Plan 353, ships in `riir-ai/.proofs/RiirAiProof/Hla/Bounded.lean`) — **closest "stability" cousin**. Proves that each per-tick sigmoid-derived belief scalar is in `(0, 1)` over ℝ. The proposed primitive extends this to *multi-tick* propagation: over N ticks, which linear combinations of the 8 HLA dimensions grow (`|μ|>1`), decay (`|μ|<1`), or stay bounded (`|μ|≈1`)? This is the empirical N-tick analog of the per-tick Lean theorem.

4. **Bai, Koltun, Kolter — DEQ Jacobian Regularization** ([arXiv:2106.14342](https://arxiv.org/abs/2106.14342), ICML 2021) — **closest ML literature precedent**. Their `ρ(J_*) < 1` condition is the scalar spectral-radius version of our band-stability criterion; their Jacobian regularization loss is the training-time version of what we propose as a runtime diagnostic + pruner. The delta: ours is **per-mode** (not scalar), **runtime** (not training-time), and **pruner-ready** (not a regularizer). See §1.4 for the full delta analysis.

### 2.3 Fusion — what novel combination does this enable?

**Headline fusion: `compose_chain` × `transfer_matrix_band_structure` × HLA recurrence → multi-tick HLA stability diagnostic.**

HLA's per-tick update is a 2nd-order linear recurrence on an 8-dim state. Treat one tick's update as `M_tick ∈ R^{8×8}`. Over N ticks, the propagation is `M_tick^N`. The band structure of `M_tick` tells you:

- Which of the 8 affect dimensions are **stable across ticks** (`|λ| ≈ 1`): these are the long-memory channels (good for committed personality).
- Which **decay** (`|λ| < 1`): these are short-memory channels (good for transient signal reactivity).
- Which **grow** (`|λ| > 1`): these are runaways — the recurrence will diverge unless projected out. The HLA Lean proofs already prevent per-tick unboundedness, but a multi-tick chain can still produce slow exponential growth that violates quorum determinism over a long session.

This is **strictly more information than the per-tick Lean proof** — and it's a runtime diagnostic (Lean is offline). The natural consumer is `latent_functor/reestimation.rs` ("coherence-driven re-estimation scheduler"): a band-edge crossing (`|μ| → 1` from below, i.e. a mode about to go unstable) is a *much earlier* signal than the current `coherence < tau_reest` trigger.

| Fusion | Source A | Source B | Novel combination? |
|---|---|---|---|
| **`compose_chain` × band-structure analyzer** | `analytic_lattice/compose_chain` (matmul chain, ships) | eigenvalue decomposition + band classifier (NEW) | **NEW (Gain-tier)** — turns the composed operator into a spectral diagnostic |
| **HLA multi-tick stability** | HLA per-tick update `M_tick` (ships) | transfer-matrix eigenvalue analysis (NEW) | **NEW** — multi-tick analog of Plan 353's per-tick Lean boundedness |
| **Band-edge as re-estimation trigger** | `latent_functor/reestimation.rs` `tau_reest` gate (ships) | `|μ_i| → 1` band-edge detector (NEW) | **NEW** — earlier + more principled re-estimation signal than coherence threshold |
| **Band-gap pruner** | `ConstraintPruner` trait (ships) | "forbidden mode" projection `v ← v − Σ ⟨v, u_forbidden⟩ u_forbidden` (NEW) | **NEW (TBD)** — prunes latent directions in the forbidden gap before they cause divergence |
| **DEC Hodge-Laplacian band structure** | `katgpt-dec/src/operators.rs::hodge_laplacian` (ships) | band classifier on `Δ`'s spectrum (NEW) | **NEW (speculative)** — topological band structure of belief cochains (analog of topological insulators) |

---

## 3. Latent-space reframing (mandatory per workflow §1.5 step 3)

For each Super-GOAT factory module, what does the transfer-matrix / band-structure lens look like when operating on it?

### 3.1 HLA (`katgpt-rs/crates/katgpt-core/src/sense/` + `riir-ai/.../hla/`)

**The headline target.** HLA is a per-tick recurrence on an 8-dim affect state. The recurrence kernel is approximately linear in the regime where sigmoid saturations are bounded away from 0 and 1 (verified empirically by the f32 spec-match test in `crates/riir-engine/tests/hla_bounds_spec_match.rs`). Treating the linearization `M_tick ∈ R^{8×8}` as a transfer matrix:

- **Allowed band** (`|μ| ≈ 1` over the relevant tick horizon) — these affect channels survive across many ticks. They are the long-term-memory channels (committed personality). A personality vector that lives in the allowed band is replay-deterministic and quorum-stable.
- **Decaying band** (`|μ| < 1 − ε`) — these channels forget quickly. Useful for transient signal reactivity (a sudden threat, a momentary opportunity) but NOT useful for committed state.
- **Growing band** (`|μ| > 1 + ε`) — these channels will diverge. They MUST be projected out before they corrupt quorum determinism. This is a new pruner signal.

**Concrete consumer integration:** `latent_functor/reestimation.rs` already has a "coherence-driven re-estimation scheduler when `coherence < tau_reest`". The band-edge detector (`|μ_i| → 1` from below) is an *earlier and more principled* trigger — it fires before coherence visibly degrades. Same integration shape, better signal.

### 3.2 `latent_functor/` (`zone_gating`, `reestimation`, `arithmetic`, `cross_game`, `k_selector`, `quality_gate`)

Each functor application `F: latent → latent` is a vector field; its Jacobian `J_F(x_*)` at a fixed point `x_*` is the transfer matrix for infinitesimal perturbations. The eigenvalues of `J_F(x_*)` determine whether `x_*` is attracting (`|λ| < 1`), repelling (`|λ| > 1`), or a center (`|λ| = 1`). This is textbook dynamical-systems stability analysis — it just hasn't been wrapped as a primitive in `riir-ai/crates/riir-engine/src/latent_functor/quality_gate.rs`. The proposed primitive ships the wrapper.

### 3.3 `cgsp_runtime/` (curiosity-guided self-play)

Curiosity-driven exploration picks actions where the world-model prediction error is high. The transfer-matrix lens reframes this as: *where is the Jacobian of the world-model transition `J_T` in a regime with large growing eigenvalues?* Large `|λ_max(J_T)|` = the model is *sensitive* to inputs in that direction = high-information region to explore. This fuses naturally with `pulse_bridge.rs` (Temporal Derivative Kernel, Plan 277) — the dual fast/slow surprise signal already approximates this; the band-structure lens gives it a closed-form underpinning.

### 3.4 LatCal (`riir-chain/src/encoding/latcal*.rs`)

LatCal is the deterministic raw↔latent bridge via 2×2 matrix arithmetic obfuscation + fixed-point commitment. The transfer-matrix lens applies *trivially* here: each LatCal operation IS a 2×2 transfer matrix. Composing N LatCal operations = N-period transfer matrix = N-layer band structure. The band structure of a LatCal chain is a **new forensic signal** — a tampered chain (modified determinants, replaced entries) will have a different band structure than the canonical chain. This is a `forensic/` signal (sync-boundary side) — fuses with `chain_asset_fingerprinting` (Plan 322) and `chain_divergence_detector` (Plan 014). **Speculative fusion; needs PoC before claiming gain.**

### 3.5 `NeuronShard` (`riir-neuron-db/src/`)

`style_weights[64]` is a 64-dim direction vector — not naturally a transfer matrix. BUT: a sequence of snapshots `V_0, V_1, V_2, …` (a freeze/thaw version chain) IS a stack of "media" through which the *committed personality* propagates. The per-snapshot transition matrix `M_n` can be approximated by `V_n · V_{n−1}^{+}` (least-squares transition). The band structure of the snapshot chain tells you: which personality modes survive across snapshots (committed core) vs which drift (transient tuning). This is a `mape_k.rs` self-healing signal — a frozen shard's personality should live in the allowed band; if it crosses into the growing band, the shard has drifted. **Speculative fusion; needs PoC.**

### 3.6 DEC Stokes operators (`katgpt-rs/crates/katgpt-dec/`)

The DEC substrate ships `exterior_derivative d`, `codifferential δ`, `hodge_laplacian Δ = δd + dδ`, `hodge_decompose`. The Hodge Laplacian's *spectrum* on a cell complex is the natural object for topological band-structure analysis:

- Zero eigenvalues (βₖ many) = harmonic modes = topologically protected (the "topological insulator" analog).
- Small eigenvalues = near-harmonic modes = slow-decaying cochains.
- Large eigenvalues = fast-decaying (high-frequency) cochains.

This is the substrate for a **topological band classifier** on belief cochains (`interest_cohain`, default-on per Plan 335 Phase 1). The fusion is: `betti_numbers` (ships) gives the count of zero modes; the band classifier gives the *full distribution* of mode stability. Speculative — needs a concrete consumer (e.g. "is this KG-triple region topologically protected against drift?") before claiming gain.

---

## 4. Latent vs raw boundary (per AGENTS.md)

| Artifact | Domain | Sync? |
|---|---|---|
| Per-tick HLA transfer matrix `M_tick` | **Latent** (semantic affect dynamics) | NO — local to NPC, not committed |
| Band classification `{propagating, decaying, growing}` per mode | **Latent** | NO — local diagnostic |
| Band-edge trigger fired (boolean) | **Latent → raw bridge**: a single bit, sync-safe | MAYBE — if it drives a committed `ReestimationTrigger` event, that event is raw + BLAKE3-committed; the underlying band classification stays local |
| Forbidden-mode pruned state vector | **Latent** | NO — local projection only; the *synced* artifacts are the resulting 5 committed scalars (valence/arousal/desperation/calm/fear), not the pruned latent vector |
| Snapshot-chain band structure (riir-neuron-db) | **Latent** (frozen snapshot metadata) | NO — local to the shard; the *receipt* (BLAKE3 hash of the snapshot, ships in `MerkleFrozenEnvelope`) crosses sync, but the band structure does not |
| LatCal chain band structure (riir-chain) | **Raw** (deterministic commitment) | YES — band structure of a LatCal chain is computed from committed entries, so it is bit-identical across quorum nodes; can be a forensic receipt |

**Sync-boundary rule (per AGENTS.md):** the band-structure analyzer is a *local latent diagnostic*. Its only sync-boundary crossings are (a) the boolean band-edge trigger bit (which feeds the existing `ReestimationScheduler` and becomes a committed event), and (b) the LatCal-chain forensic receipt (which is computed from already-committed entries). The full eigenvalue spectrum, eigenvector decomposition, and band classification stay local — never cross sync. This is the standard latent-to-raw bridge pattern (5 scalars cross, full vector stays).

---

## 5. Verdict — **Gain**

### 5.1 Tier rationale

| Q | Answer | Evidence |
|---|---|---|
| Q1 — No prior art? | **YES** | Grep across all 7 repos in BOTH vocabularies (`kronig`, `penney`, `dirac_comb`, `delta_lattice`, `bloch`, `brillouin`, `transfer_matrix`, `band_gap`, `tunneling`, `allowed_band`, `forbidden_band`, `wave_function`, `schrodinger`, `tight_binding`, `periodic_barrier`, `scattering_matrix`) → zero hits. The unrelated `spectral_gap` in `crates/katgpt-spectral/src/spectral.rs` is participation-ratio for KV compression — different mechanism, coincidental name overlap. The closest cousins (`analytic_lattice::compose_chain`, `subspace_phase_gate::phase_transition_gate`, HLA Lean boundedness Plan 353) cover adjacent but distinct territory. |
| Q2 — New class of behavior? | **NO** | It is a *measurement* (band structure of an existing operator sequence), not a *capability*. HLA per-tick boundedness is already proven (Lean Plan 353); this is the multi-tick empirical extension — same idea, more general regime. The "band-gap pruner" fusion (§2.3 row 4) is *potentially* a new behavior, but it's TBD — needs a concrete consumer + PoC before claiming behavior-class novelty. |
| Q3 — Product selling point? | **NO clear sentence** | "Our NPCs have band-gap-aware latent propagation" is engineering quality, not a customer-facing feature. Cannot complete "we do X that no competitor can" in a sentence that means anything to a buyer. |
| Q4 — Force multiplier? | **YES but thin** | Connects to: `analytic_lattice/compose_chain` (consumes output), `subspace_phase_gate` (extends from sample-sufficiency to mode-propagation), `latent_functor/reestimation` (band-edge as new trigger), HLA Lean boundedness (multi-tick extension), DEC `hodge_laplacian` (topological band structure). But every integration is "use as a diagnostic signal" — none creates a new behavior class on its own. |

**Not all 4 YES → not Super-GOAT.** Per §1.55, this is **Gain, not Pass** — there are actionable improvements:

1. **A genuinely novel primitive** that doesn't ship (verified by grep).
2. **A real latent-space reframing** (5 Super-GOAT factory modules have natural consumers).
3. **A concrete fusion** with `analytic_lattice/compose_chain` that turns its output into a spectral diagnostic.
4. **A modelless, deterministic, zero-physics implementation** (pure linear algebra over shipped `TransportOperator`).

### 5.2 MOAT gate per domain (§1.6)

- **`katgpt-rs` (public engine):** ✅ **In scope**. Pure inference substrate, generic numeric, no game/chain/shard semantics. Sits naturally next to `analytic_lattice/`, `subspace_phase_gate`, `katgpt-dec`. Ships behind a feature flag.
- **`riir-ai` (private runtime):** The headline consumer integration (HLA multi-tick stability → reestimation trigger) lives here as a thin bridge. NOT pillar-level — does not create a new sloppy-test winner. Neutral moat contribution.
- **`riir-chain` / `riir-neuron-db`:** Speculative fusion (LatCal chain forensic / snapshot chain drift detection). TBD; not actionable today without PoC.
- **`riir-train`:** NOT a dep — fully modelless.

### 5.3 Why not Super-GOAT

The honest verdict is **Gain, with a TBD path to Super-GOAT**. The path requires:

1. A concrete consumer that converts the band-structure signal into a *behavior* (not just a diagnostic). The strongest candidate is the **band-gap pruner**: project out forbidden latent modes before they destabilize the HLA recurrence. If this measurably improves long-session HLA stability (e.g., 10K-tick sessions without coherence collapse), Q2 flips to YES.
2. A product framing: "Our NPCs run unbounded-length sessions without latent-state collapse — band-gap-aware pruning keeps the affect recurrence stable forever." If that sentence becomes true and provable, Q3 flips to YES.
3. A force-multiplier story across ≥2 pillars: HLA (Pillar 1 extension) + DEC (topological band structure) + latent_functor (band-edge trigger). Already potentially true; needs the pruner PoC to make it concrete.

**Re-evaluation gate:** if (1) + (2) + (3) all hit, run the §1.5 novelty gate again. Until then, this stays Gain and ships behind a feature flag with a GOAT gate (latency + correctness vs naive eigendecomposition).

---

## 6. What stays open vs private

| Artifact | Repo | Status |
|---|---|---|
| `TransferMatrixBandStructure` primitive (composite + eigenvalues + Bloch factor + band classifier + transmission/reflection) | `katgpt-rs/crates/katgpt-core/src/analytic_lattice/band_structure.rs` (NEW) | **Public** — generic numeric, leaf-clean |
| HLA `M_tick` linearization + multi-tick stability bridge | `riir-ai/crates/riir-engine/src/hla/multi_tick_band.rs` (NEW, private) | **Private** — game-runtime IP |
| Band-edge → `tau_reest` reestimation trigger | `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` (extension) | **Private** — runtime trigger logic |
| Forbidden-mode pruner (TBD) | `katgpt-rs/crates/katgpt-core/src/pruners/band_gap_pruner.rs` (NEW, public) + `riir-ai` integration | **Public primitive** + **private integration** |
| LatCal chain forensic band structure (speculative) | `riir-chain/src/forensic/` (extension, gated) | **Private** — chain forensic IP |
| Snapshot-chain drift detection (speculative) | `riir-neuron-db/src/mape_k.rs` (extension) | **Private** — shard IP |

---

## 7. Validation protocol (GOAT gate, in katgpt-rs)

When planned, the GOAT gate (per AGENTS.md) for the `transfer_matrix_band_structure` feature flag:

- **G1 (correctness)** — unit tests on known 2×2 cases (resonant tunneling through N=5 identical delta barriers; closed-form Kronig-Penney allowed-band boundaries recovered to within f32 ε). Mirrors the canonical textbook examples in Griffiths Ch. 5.
- **G2 (perf)** — `criterion` bench at k=8 (HLA scale) and k=16 (analytic_lattice scale): composite + eigenvalue decomposition + band classification must be **sub-µs** at k=8 and **sub-10µs** at k=16 on commodity hardware. Baseline: naive Jacobi eigensolver.
- **G3 (no-regression)** — all existing `analytic_lattice` tests pass unchanged; the new module is additive only.
- **G4 (alloc-free)** — `CountingAllocator` audit: zero allocations on the hot path after the initial `Vec` for the composite. Reuses `compose_chain_into`'s scratch buffer pattern.
- **G5 (deterministic)** — bit-identical output across `aarch64-apple-darwin` (native) + `x86_64-apple-darwin` (Rosetta 2). Mirrors `chain_aoi` G1 (the cross-arch bit-identity gate, see `riir-chain/.benchmarks/017_promotion_session.md`). Required because HLA is replay-deterministic and quorum-committed.
- **G6 (UQ floor — N/A)** — this is not a UQ-bearing primitive (no probability distribution, no predictive interval). The "Report the Floor" rule (Research 322 / Plan 340) does not apply.

**Promotion criterion:** if G1–G5 pass AND the HLA multi-tick stability bridge shows measurable improvement on a 10K-tick session bench (coherence stays above `tau_reest` for longer with band-edge-triggered re-estimation than with the current coherence-threshold trigger), promote `transfer_matrix_band_structure` to default-on in katgpt-rs. The bridge in riir-ai stays behind its own feature flag (`hla_multi_tick_band`).

---

## 8. References

- **Kronig-Penney original:** R. de L. Kronig & W. G. Penney, *Quantum Mechanics of Electrons in Crystal Lattices*, Proc. Roy. Soc. A **130**, 499–513 (1931).
- **Modern treatment (delta-function version):** D. J. Griffiths, *Introduction to Quantum Mechanics* §5.3 (Cambridge, 2nd ed., 2017). Joachain & Bransden §4.5.
- **Transfer-matrix method:** J. H. Luscombe, *Thermodynamics and Statistical Mechanics* (the formal method in any QM textbook); M. Born & E. Wolf, *Optics* §1.6 (multilayer-film transfer matrices — the same math in classical optics).
- **ML-adjacent transfer matrices:** Linear-attention recurrences as discrete-time linear systems — see Research 230 (Semiseparable State-Space Duality), Research 169 (Oscillatory State-Space Modelless Distillation).
- **Closest shipped cousins:**
  - `crates/katgpt-core/src/analytic_lattice/mod.rs` — `TransportOperator`, `compose_chain` (the matmul chain this would extend with spectral analysis)
  - `crates/katgpt-core/src/subspace_phase_gate.rs` — `phase_transition_gate` (the closest "phase transition" primitive, sample-sufficiency regime)
  - `crates/katgpt-dec/src/operators.rs` — `hodge_laplacian` (the spectral substrate for a topological band classifier)
  - `crates/katgpt-dec/src/hodge.rs` — `betti_numbers`, `harmonic_projector` (zero-eigenvalue modes — the "topologically protected" band)
  - `riir-ai/.proofs/RiirAiProof/Hla/Bounded.lean` (Plan 353) — per-tick HLA boundedness Lean theorem (the offline proof this would extend empirically to multi-tick)

## TL;DR

**Verdict: Gain.** Kronig-Penney's transferable math is the **transfer-matrix band-structure method** — given a sequence of k×k matrices `[M_1, …, M_N]` (or a periodic `M` applied N times), compute the composite, its eigenvalues, the Bloch propagation factor `μ = λ^{1/N}` per mode, and classify each mode as propagating (`|μ|≈1`, allowed band), decaying (`|μ|<1`, evanescent), or growing (`|μ|>1`, unstable). Pure linear algebra, modelless, zero physics. Zero grep prior art across all 7 repos in both vocabularies. The matmul chain already ships as `analytic_lattice::compose_chain`; the missing piece is the *spectral analysis + band classifier*. **Not Super-GOAT** (Q2 fails — it's a measurement, not a capability; Q3 fails — no clear product selling point) but **not Pass** (genuinely novel math, real fusion hooks into HLA multi-tick stability / `latent_functor/reestimation` / DEC `hodge_laplacian` / subspace_phase_gate). The TBD path to Super-GOAT requires a concrete behavior-class consumer — strongest candidate is the **band-gap pruner** (project out forbidden latent modes before they destabilize the HLA recurrence across long sessions); if it works empirically, re-run the §1.5 gate. Plan should ship the public primitive in `katgpt-rs/crates/katgpt-core/src/analytic_lattice/band_structure.rs` behind a `transfer_matrix_band_structure` feature flag, with a GOAT gate (G1 correctness on known Kronig-Penney band edges, G2 sub-µs perf at k=8, G5 cross-arch bit-identity). The HLA multi-tick bridge in riir-ai is the headline consumer — extends Plan 353's per-tick Lean boundedness to N-tick empirical stability.
