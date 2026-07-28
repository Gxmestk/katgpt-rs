# Issue 199 — Heterogeneous-composition imbalance diagnostic PoC

**Type:** PoC / proof task (defend-wrong follow-up to the Twins paper PASS verdict).
**Severity:** Medium — unproven claim that our composed-latent runtimes are
immune to the imbalance class arXiv:2607.22531 documents for image diffusion.
**Triggered by:** arXiv:2607.22531 (Twins: Focal Loss for unified ViT+VAE
representation) — PASS verdict at commit `8abd07ef`. The verdict's "covered
by shipped primitives" claims were architectural-coverage-only (§3.6
violation); this issue tracks the unproven quality-parity gap.
**Cross-ref research notes:** 279 (Subspace Phase-Gate), 394 (Within-Class
Effective Rank), 409 (MANCE Two-NN) — all carry the Twins PASS-Redirects line.

## The hypothesis under test

arXiv:2607.22531 §2 proves that when you compose two heterogeneous latent
spaces (semantic ViT + pixel VAE) into one representation and train a
diffusion model on the union, the model silently underfits the
high-intrinsic-dimension / high-frequency / condition-independent
component because the low-ID / low-frequency / condition-aligned component
dominates the loss landscape. They call this "optimization imbalance" and
trace it to three sources:

1. **Spectral bias** — NN spectral bias favors low-frequency signals.
2. **Intrinsic dimensionality** — low-ID manifolds are easier to fit.
3. **Conditional alignment** — condition-aligned features collapse under
   conditioning; condition-independent features stay high-entropy.

**The unproven transfer claim:** our composed-latent runtimes combine
heterogeneous direction fields / branches via sigmoid-gated weighted sums.
We have no evidence either way that those blends suffer the same imbalance
class. The Twins paper is about training dynamics; our blends are
inference-time weighted sums. The math substrate (weighted sum of
heterogeneous fields) is the same shape, but the failure mode
(gradient-descent underfitting) does not directly apply to inference.

**So this is NOT "does the same bug exist" — it is "does an analogous
imbalance class exist for inference-time sigmoid-gated blends of
heterogeneous fields, and if so, do our existing diagnostics catch it?"**

## The composed-latent sites to probe

Three runtimes that combine heterogeneous latent sources via weighted sums:

### Site 1 — `CommittedFieldBlend<N, D>::apply_blended`

`katgpt-rs/crates/katgpt-core/src/committed_field_blend.rs:188-224`

Mechanism (verbatim):
```rust
dz_out[..D].fill(0.0);
for (k, field_k) in fields.iter().enumerate() {
    let dz_k = field_k.evolve(z, dz_scratch);
    let gate = sigmoid(pi[k] / tau);
    simd_fused_scale_acc(dz_out, dz_k, gate, D);
}
// => dz_out[j] = Σ_k sigmoid(pi_k/tau) · f_k(z)[j]
```

N archetype direction fields, each producing a D-dim displacement, summed
with sigmoid gates. If the `f_k` have heterogeneous spectral profiles /
intrinsic dims / magnitudes, the unweighted sum can be silently dominated
by whichever field has the largest-magnitude or lowest-frequency output —
the sigmoid gate rebalances *total contribution* but not *spectral / ID
coverage*.

### Site 2 — `PersonalityWeightedComposition<N, D>::compose_into`

`katgpt-rs/crates/katgpt-core/src/personality_composition/` (Plan 297,
default-on). Same shape: sigmoid-gated weighted sum of N direction vectors
per layer. README L2829-2840 documents the API. Same potential imbalance
class — the layers with the highest-magnitude recent direction can
dominate the composition.

### Site 3 — `BranchBank<E>` + `BranchRouter::route`

`katgpt-rs/crates/katgpt-core/src/branching/bank.rs` (Plan 329). Branches
are routed by anchor similarity; the routed branch's episodic + procedural
stores contribute to the NPC's cognition. Heterogeneous branches (combat
vs dialog vs social) likely have heterogeneous intrinsic dimensionality —
the router may systematically favor the lowest-ID branch (easiest to match
anchors against) and starve the high-ID branches.

## The diagnostic question (PoC scope)

For each of the three sites, construct a controlled adversarial case where
the heterogeneous components have deliberately mismatched:

- (a) **Intrinsic dimensionality** — e.g. field_1 produces outputs on a
  2-dim linear subspace, field_2 produces outputs on an 8-dim isotropic
  cloud. Does `apply_blended` produce an output whose intrinsic dim is
  closer to 2 (dominated by the easy field) or 8 (balanced)?
- (b) **Spectral profile** — e.g. field_1 produces smooth low-frequency
  displacements, field_2 produces high-frequency texture-like
  displacements. Does the blend retain high-frequency content or wash it
  out?
- (c) **Magnitude asymmetry** — e.g. field_1 has 10× the output magnitude
  of field_2. Does the sigmoid gate alone correct this, or does the
  high-magnitude field dominate the blend direction?

Then check whether our existing diagnostics catch the imbalance:

| Existing primitive | What it would need to detect |
|---|---|
| `within_class_effective_rank` (Plan 415 / R394) | Designed for class-labeled states, not heterogeneous channels within one token. Need to verify whether treating "field-of-origin" as the class label reproduces the Twins paper's PCA collapse signal. |
| `subspace_phase_gate::participation_ratio` / `numerical_rank` (Plan 301 / R279) | Estimates intrinsic dim of ONE latent space. Need to verify whether running it on the blended output vs on each field's output separately detects the dominance. |
| `effective_rank` (R113 / Plan 151) | Global entropy-rank. Same question — does the blended output's effective rank reveal which field dominates? |

## PoC plan

Live in `riir-ai/crates/riir-poc/benches/composition_imbalance_modelless.rs`
(the "defend-wrong" R&D crate per research skill §3.6). Use
`CARGO_TARGET_DIR=/tmp/issue199` per AGENTS.md, clean up when done.

### Tasks

- [ ] **T1** — Construct three synthetic `ArchetypeFieldSource<D>` impls
      with deliberately mismatched properties:
      - `LowIdField` — output on a fixed 2-dim linear subspace.
      - `HighIdField` — output on an 8-dim isotropic cloud (Gaussian).
      - `HighFreqField` — output with strong high-frequency content
        (sinusoidal at the Nyquist rate of the D-dim grid).
- [ ] **T2** — Construct matched controls: three fields with IDENTICAL
      intrinsic dim / spectral profile / magnitude (the balanced case).
- [ ] **T3** — Run `CommittedFieldBlend<3, D>::apply_blended` on both the
      mismatched and balanced cases with `pi = [0, 0, 0]` (uniform gates).
      Collect 1000 samples of `dz_out`.
- [ ] **T4** — For each sample, compute:
      - `effective_rank(samples, D)`
      - `participation_ratio` / `numerical_rank`
      - `within_class_effective_rank(samples, D, field_origin_labels)`
      - Radial power spectrum (FFT) — the Twins paper's Fig 4 diagnostic
- [ ] **T5** — Verdict table:
      | Case | Does blend suffer imbalance? | Which diagnostic catches it? |
      |---|---|---|
      | Mismatched ID | ? | ? |
      | Mismatched spectral | ? | ? |
      | Mismatched magnitude | ? | ? |
      | Balanced (control) | no (sanity) | n/a |
- [ ] **T6** — If T5 shows the existing diagnostics DO catch the
      imbalance → verdict stands (the architectural coverage is also
      quality parity). Close this issue with the PoC as the permanent
      regression check.
- [ ] **T7** — If T5 shows the existing diagnostics DO NOT catch the
      imbalance → this is a Gain, not a Pass. File a plan for a
      `composition_imbalance_diagnostic` primitive that wraps the
      three-source analysis (spectral / ID / conditional) into one
      probe over a blended output. Re-open the Twins verdict.
- [ ] **T8** — Repeat T3–T5 for `PersonalityWeightedComposition` and
      `BranchBank::route` (the other two composed-latent sites). Different
      inputs, same diagnostic battery.

## What this PoC does NOT test

- The Focal Loss training trick — genuinely training-only, out of scope.
- The Twins representation itself (channel-wise concat of ViT+VAE) —
  image-diffusion-specific, out of scope.
- Whether inference-time blends CAN suffer optimization imbalance —
  they cannot, by definition (no gradient descent). The hypothesis under
  test is the **analogous** imbalance class: output dominance by the
  low-ID / low-frequency / high-magnitude component in a weighted sum.

## Why this matters

The Twins paper's diagnostic framework (spectral / Two-NN / conditional
alignment) is a *vocabulary* for asking "is this blend balanced?" that we
do not currently apply to our composed-latent runtimes. If the PoC at T7
shows our existing primitives miss the imbalance, we have a real gap in
our per-NPC cognitive-stack quality assurance — silent dominance of one
archetype / branch / personality direction over the others would degrade
NPC behavioral diversity without any visible error signal.

The cost of the PoC is ~1 day. The cost of a missed composition-imbalance
bug in production NPC behavior is behavioral monoculture at crowd scale —
exactly the failure mode the Within-Class Effective Rank (Plan 415) was
shipped to catch, but never validated against the blend runtime itself.

## References

- arXiv:2607.22531 — Twins paper (the PASS verdict that triggered this issue).
- `katgpt-rs/.research/394_GNN_Survey_Within_Class_Effective_Rank_Fusion.md` — closest cousin (conditional-alignment axis).
- `katgpt-rs/.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md` — closest cousin (intrinsic-dim axis).
- `katgpt-rs/.research/409_MANCE_Manifold_Aware_Concept_Erasure.md` — closest cousin (Two-NN axis).
- `katgpt-rs/crates/katgpt-core/src/committed_field_blend.rs:188-224` — Site 1 (apply_blended_with_pi).
- `katgpt-rs/crates/katgpt-core/src/branching/bank.rs` — Site 3 (BranchBank).
- `riir-ai/crates/riir-poc/` — the defend-wrong R&D crate where the PoC lives.
- Research skill §3.6 — the "architectural coverage ≠ quality parity" rule this issue enforces.
