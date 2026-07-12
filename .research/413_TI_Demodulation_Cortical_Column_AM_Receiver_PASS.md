# Research 413: TI Demodulation — Cortical Column as AM Radio Receiver — PASS

> **Source:** Ruffini, Mercadal, Just, de Palma Aristides, Castaldo, Canals, Mirasso — "The cortical column as a tuned receiver: a network mechanism for temporal-interference stimulation", TN0484 / WP0185, Zenodo DOI:10.5281/zenodo.20844275, 2026-07-10.
> **Date:** 2026-07-12
> **Status:** Done — PASS
> **Classification:** Public
> **Related Research:** 371 (DMFT Hopf regime classifier — covers the near-Hopf amplification half), 169 (Oscillatory SSM / LinOSS — covers oscillatory eigenvalue / near-critical resonance), 276 (MicroRecurrentBeliefState — per-NPC attractor/leaky substrate)

**Verdict:** → PASS, near-Hopf half subsumed by 371/169; sigmoid-curvature-as-demodulator half requires infrastructure we don't have.

---

## Paper core

Computational neuroscience paper about **temporal-interference (TI) brain stimulation** — how a cortical column demodulates an external amplitude-modulated (AM) electric field. Two ingredients:

1. **Sigmoid curvature σ'' as square-law demodulator** (Eq. 10): the firing-rate sigmoid's *second derivative* (not slope) acts as a diode — squaring the AM carrier folds spectral energy to baseband, producing a line at the envelope (beat) frequency Ω. `A_Ω ≈ ½ σ''(v*) ε² m`. Demodulation is zero at the sigmoid inflection (v* = v₀), peaks on the flanks, and reverses sign across it. This is the novel half.

2. **Near-Hopf resonance amplification** (Eq. 11): the recurrent synaptic network near a Hopf bifurcation acts as a tuned resonator with gain `G(Ω) ∝ 1/γ`, diverging as the Hopf is approached. The 2nd-order synapse low-passes the carrier away; the near-critical focus amplifies the recovered envelope at the network's natural frequency. **Timing-not-rate**: the resonance amplifies AC envelope (spike timing), not DC mean rate — matching in-vivo TI observations.

---

## Why PASS

### Near-Hopf amplification — already shipped (Research 371 + 169)

| Paper mechanism | Shipped cousin | File / Research |
|---|---|---|
| Hopf bifurcation detection (complex conjugate eigenvalues cross imaginary axis) | `HopfBoundary` — closed-form 2×2 Jacobian eigenvalue discriminant | Research 371, Plan 371 |
| Regime classification (Static / Noise-sustained oscillation / Irregular switching / Global limit cycle) | `RegimeClassifier` with `Regime` enum | Research 371, Plan 371 |
| Near-critical susceptibility (∝ 1/γ) | Oscillatory eigenvalues on imaginary axis (LinOSS) | Research 169, Plan 189 |
| Mean-field order parameters (κ, κ_a, Q) over NPC population | `MeanFieldOverlap` aggregates per-NPC HLA onto learned direction | Research 371, Plan 371 |
| Coupling-controlled gain (J → J* at Hopf) | `HopfBoundary` discriminant `τ_a·τ_m·β·G_eff > (τ_a + τ_m)²/4` | Research 371 |
| State-dependent amplification (β = arousal) | β mapped to HLA `arousal ∈ [0,1]` | Research 371 §2.4 |

The paper's second ingredient (near-Hopf resonance) is ~100% covered by the DMFT regime classifier shipped in Plan 371. The `HopfBoundary` discriminant is a one-line closed-form check; the `RegimeClassifier` produces the paper's four-regime taxonomy; β is mapped to HLA arousal. Research 371 already fuses this with DEC continuity (Fokker-Planck), Temporal Derivative Kernel (surprise→β), and Committed Personality (per-archetype β).

### Sigmoid curvature σ'' as demodulator — novel but non-transferable

The paper's first ingredient (sigmoid curvature as square-law demodulator) IS mathematically novel to our codebase:
- We use sigmoid **slope** (σ') for ranking preservation (Lean-proven in `KatgptProof`), gating, and HLA projections.
- We **never** use sigmoid **curvature** (σ'') as an operator. Grep for `sigmoid.*curvature|σ''|square.?law|demodulat` returns zero hits in `katgpt-rs/src/` or `riir-engine/src/`.
- The closest is `Plan 414` (HLA Committed-Belief Lipschitz Probe), which mentions sigmoid curvature only as a caveat on the first-order bound's tightness.

**However, the demodulation mechanism is non-transferable because it requires infrastructure our codebase doesn't have:**

1. **Carrier/envelope timescale separation.** The paper's mechanism requires `f_c ≫ Δf` (kHz carrier, Hz envelope). Our codebase operates at a single 20Hz tick with no kHz carrier. HLA projections are discrete-time, single-rate. There is no "fast signal" for the sigmoid curvature to demodulate.

2. **Recurrent sigmoid + synapse loop.** The demodulation requires the sigmoid to sit inside a recurrent loop with 2nd-order synaptic dynamics (the JR/LaNMM/NMM2 circuit). Our HLA projections are **feed-forward** — `σ(dot(h, d))` projects onto a direction vector, no recurrence. The recurrent dynamics exist only diagnostically in Research 371's `MeanFieldOverlap` (which *classifies* the regime, it doesn't *simulate* the ODE).

3. **Continuous-time ODE simulation.** The paper integrates JR/LaNMM/NMM2 ODEs with RK4 at `Δt = 0.05–0.2 ms`. Our codebase doesn't run continuous-time neural mass simulations — the closest is `MicroRecurrentBeliefState` (Plan 276), which is a discrete-time leaky/attractor kernel, not a 2nd-order ODE near a Hopf.

4. **Application domain mismatch.** The paper is about **transcranial brain stimulation** — how an external electric field's envelope is recovered by cortical tissue. Our codebase is about game AI inference and neuro-symbolic chain transport. There is no "external AM field" in a game AI tick.

### Fusion search — no productive combination

Attempted fusion with:
- **HLA per-NPC state × sigmoid curvature**: HLA's 8 dimensions could have different natural frequencies (valence=slow, arousal=fast), and their interaction through sigmoid curvature could produce cross-frequency coupling. But HLA projections are feed-forward `σ(dot(h, d))` — no recurrence, no carrier, no envelope. The curvature term would be `σ''(dot(h,d)) · (δh)²`, which is a second-order perturbation correction, not a demodulation operator.
- **Research 371 Hopf boundary × sigmoid curvature**: The Hopf boundary detects when the crowd is near criticality; the sigmoid curvature could be the "detector" that extracts the envelope of an external modulated signal. But there is no external modulated signal in a game AI tick — game events are discrete impulses, not AM carriers.
- **DEC operators × sigmoid curvature**: The paper's HAM framework [10] (referenced but not the paper's contribution) generalizes demodulation to nested constant-Q envelopes. This could relate to DEC multi-scale operations. But the HAM framework is a separate paper (bioRxiv 2025.11.03.686310), not this one, and is itself speculative.

No fusion produces a capability that existing primitives don't already cover. The near-Hopf amplification is shipped (371); the demodulation half has no substrate to operate on.

---

## TL;DR

Computational neuroscience paper about TI brain stimulation. The near-Hopf amplification half is fully covered by Research 371 (HopfBoundary + RegimeClassifier) and Research 169 (oscillatory eigenvalues). The sigmoid-curvature-as-demodulator half is mathematically novel (we use σ' but never σ'') but requires a carrier/envelope timescale separation, recurrent ODE dynamics, and continuous-time simulation that our discrete 20Hz tick, feed-forward HLA projections don't have. The paper's value is biophysical (how cortical columns demodulate external electric fields), not transferable to modelless inference or game AI. No files, no plan, no issue.
