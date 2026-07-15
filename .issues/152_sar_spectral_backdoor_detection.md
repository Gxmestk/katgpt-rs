# Issue 152: SAR × Statistically-Undetectable Backdoors — Spectral Backdoor Detection PoC

**Opened:** 2026-07-15
**Origin:** Research 406 §8.3 Fusion G (addendum)
**Related Research:** [406 (SAR)](../.research/406_Spectral_Rewiring_Weight_Delta_Purification.md), [422 (Backdoors)](../.research/422_Statistically_Undetectable_Backdoors_Model_Provenance.md)
**Status:** Open — needs PoC; highly speculative

## TL;DR

**Fusion hypothesis:** `spectral_rewire`'s rewiring matrix `M = UᵀΔWV` may
expose a **spectral signature of a planted backdoor** that the backdoor's
construction (R422) deliberately hides from uniform-norm tests (`‖Az‖∞`
cryptographically small).

R422 proves the backdoor is *statistically undetectable* in TV-distance
(`d_TV(F, bF) = o(1)` — no efficient distinguisher can tell backdoored from
honest). The open question: does SAR's *directional* decomposition in the
base's singular coordinates break that undetectability?

This is genuinely novel — R422's prior-art audit (§2.2) found zero modelless
backdoor detectors in the codebase. SAR adds a spectral lens that may or may
not see what uniform-norm tests miss.

## The mechanism

R422 constructs a backdoored first-layer matrix `A` (frozen compressing
Gaussian m×n) via a secret `z ∈ {±1}^n` so that:
- `‖Az‖∞ ≤ δ₀ = O(β√m · 2^(−n/m))` — cryptographically small (the backdoor is
  *invisible* to uniform-norm / TV-distance tests)
- For the holder of `z`: `x' = x + z` collides with any `x` to within `δ₀`

SAR decomposes a weight delta `ΔW = W_backdoored − W_honest` into:
```
ΔW = ΔW* + ΔW⊥
ΔW* = U M Vᵀ    where M = Uᵀ ΔW V  (rewiring matrix in base singular coords)
ΔW⊥ = off-manifold residual
```

**Hypothesis:** the backdoor's `z` is constructed to be small in `‖Az‖∞`, but
it has a *specific directional structure* (aligned with the planted lattice
directions). When projected onto the base SVD basis, this directional structure
should show up as **an anomalous spike in a specific row/column of `M`** —
concentrated off-manifold energy in the singular directions the backdoor uses,
even when the aggregate `on_manifold_fraction` looks normal.

## Why this is plausible (and why it might fail)

**Plausible:**
- `‖·‖∞` (uniform norm) and the SVD-coordinate decomposition measure different
  things. A vector can be small in `‖·‖∞` but have a large component along a
  specific singular direction.
- SAR's `M` is a full r×r matrix of directional interactions; a planted `z`
  aligned with specific low-σ directions should leave a structured trace.

**Might fail:**
- R422's undetectability is in TV-distance, which is stronger than `‖·‖∞`. If
  the backdoor construction makes the FULL weight distribution statistically
  close (not just small in one norm), then SVD-coordinate projections are also
  statistically close — SAR sees nothing.
- The honest-vs-backdoored delta `ΔW` is the difference of two weight matrices
  that are TV-close; the delta itself may have no special structure in the base
  SVD coordinates even if the individual matrices do.

**This is exactly why a PoC is mandatory** (skill §3.6) — architectural
plausibility is NOT quality proof.

## The PoC

### Setup

Use R422's Fashion-MNIST proof-of-concept (§1.5):
- Honest model: 784 → 256 (frozen Gaussian) → 512 → 1024 → 2048, LeakyReLU(0.01)
- Backdoored model: same architecture, first layer replaced with
  `BackdoorMatrix(z, n=784, m=256)`
- Both trained to comparable accuracy (~86.5% backdoored vs ~89% honest per R422)

### Test

1. Compute `ΔW = W_backdoored_first_layer − W_honest_first_layer` (784×256).
2. SAR-decompose against an honest base's SVD: `M = U_honestᵀ ΔW V_honest`.
3. Measure: does `M` have anomalous structure (e.g., a specific row/column
   with energy >> the median, or a low-rank spike) that a random delta of the
   same Frobenius norm does not?
4. Control: repeat with `ΔW_random` of matched `‖ΔW‖_F`. If `M_backdoor` and
   `M_random` are statistically indistinguishable → SAR does NOT detect the
   backdoor. If `M_backdoor` has a clear anomaly → it does.

### Pass criterion

`M_backdoor` shows a detectable anomaly (e.g., max row-energy > 3σ above the
random-delta distribution) that survives multiple `z` seeds.

### Fail criterion

`M_backdoor` ≈ `M_random` in distribution → SAR's spectral lens does not break
R422's TV-distance undetectability. Close this issue; the backdoor stays
undetectable to modelless spectral methods (R422's verdict stands).

## Routing if the PoC passes

This would be a **genuinely novel capability class** (first modelless
spectral backdoor detector) and a potential Super-GOAT:
1. **Open primitive** (already shipped): `spectral_rewire`'s
   `RewiringDiagnostics` (diagonal_energy, off_diagonal_energy,
   spectral_norm_estimate, rewiring_sparsity) — may need a new
   `backdoor_anomaly_score` diagnostic.
2. **Architectural guide:** `riir-chain/.research/` (chain security — model
   provenance is a chain concern; connects to forensic fingerprinting P7+).
3. **Plan:** `riir-chain/.plans/` to wire the detector into the chain's
   neuron-vessel attestation path (Plan 003 Phase 8 already does LatCal
   committed projection + verifier replay; spectral backdoor detection would
   be a new attestation mode).

## Routing if the PoC fails

Close this issue with the negative result. Document in Research 406 §8
addendum + Research 422 as a refuted fusion. R422's PASS verdict (backdoor
construction is training-side, modelless verification is a corollary) stands
unchallenged.

## Open questions

- [ ] R422's construction is for a FEEDFORWARD DNN with frozen Gaussian first
      layer. Our transformers don't have this architecture. Does the spectral
      signature generalize to transformer weight deltas? (Probably not directly
      — this PoC is specifically on R422's setup, not our models.)
- [ ] Is this even the right framing? R422 says the backdoor is undetectable
      *given the weights*. SAR operates on weight deltas between two models.
      We would need BOTH the honest and backdoored model to compute the delta —
      which means we already have the honest reference. The realistic threat
      model is "we have a suspect model, no honest reference" — and there SAR
      has no delta to decompose. **This may be a fatal scope problem.**

## Honest assessment

**This fusion is the weakest of the three new ones (E/F/G).** The scope problem
(need both models to compute the delta) may make it impractical even if the
spectral signature exists. It is filed as an issue per the skill's
"fusion idea — novelty TBD" rule, but the priority is LOW behind Issue 151
(SAR × QuasiMoTTo, which has a clear compound-gain story and no scope problem).
