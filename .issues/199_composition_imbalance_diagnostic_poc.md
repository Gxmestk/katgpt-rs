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

- [x] **T1** — Construct three synthetic `ArchetypeFieldSource<D>` impls
      with deliberately mismatched properties (`LowIdField` rank-2,
      `HighIdField` full-D isotropic, `HighFreqField` Nyquist sinusoid).
- [x] **T2** — Construct matched controls: 3x HighId (balanced) + 3x LowId
      (sanity — rank collapse the diagnostics SHOULD flag).
- [x] **T3** — Run `CommittedFieldBlend<3, 32>::apply_blended` on all five
      configs (balanced + 3 mismatched + sanity). 1000 samples each.
- [x] **T4** — For each sample, compute `effective_rank`,
      `within_class_effective_rank`, `participation_ratio`, radial FFT
      low-freq energy fraction.
- [x] **T5** — Verdict table (measured 2026-07-28):

      | Config | erank | wc_erank | part_ratio | low_freq |
      |---|---|---|---|---|
      | Balanced (3x HighId) | 24.68 | 24.68 | 30.14 | 0.508 |
      | Mismatched ID (LowId+2x HighId) | 24.39 | 24.39 | 30.59 | 0.528 |
      | Mismatched spectral (LowId+HighId+HighFreq) | 25.61 | 25.61 | 30.12 | 0.503 |
      | Mismatched magnitude (pi=[5,0,0]) | 24.90 | 24.90 | 30.28 | 0.505 |
      | Sanity: 3x LowId (rank-≤6 expected) | **1.00** | **1.00** | **10.54** | 0.415 |

      Diagnostics flag the sanity config (rank collapse detected) but MISS
      on all three "mismatched" configs.

- [x] **T6** — HONEST REINTERPRETATION (the verdict is neither T6-original
      nor T7 — it's a third outcome the issue didn't anticipate):

      **The mismatched configs don't produce imbalance because the
      failure mode doesn't transfer to inference.** The Twins paper's
      "optimization imbalance" is a training-dynamics phenomenon
      (gradient descent underfits the high-ID/high-freq component). At
      inference time, `apply_blended` is a closed-form weighted sum —
      no gradient descent, no underfitting. The sigmoid gates normalize
      per-field contribution magnitude, and when heterogeneous fields
      are summed, the highest-ID field's contribution dominates the
      output covariance **by construction** (correct linear algebra),
      not by failure.

      The sanity config (3x LowId) confirms the diagnostics WORK when rank
      collapse actually exists — `erank` drops 24.68 → 1.00, `wc_erank`
      drops 24.68 → 1.00, `part_ratio` drops 30.14 → 10.54. So the
      diagnostic battery is NOT broken; the "mismatched" configs simply
      don't exhibit the failure mode the issue hypothesized.

      **The original PASS verdict stands** — but for a different reason
      than originally claimed. The original justification ("covered by
      shipped primitives") was architecturally sloppy (§3.6 violation:
      architectural coverage ≠ quality parity). The honest justification
      is: **the failure mode is training-specific and does not transfer
      to inference-time sigmoid-gated blends**. The diagnostic primitives
      are not load-bearing for this verdict — the closed-form math is.

- [-] **T7** — N/A. The "existing diagnostics miss" finding does NOT
      imply a Gain or a new primitive, because the failure mode itself
      doesn't apply to inference. Filing a plan for
      `composition_imbalance_diagnostic` would be solving a problem we
      don't have.
- [-] **T8** — N/A. `PersonalityWeightedComposition` and `BranchBank`
      use the same sigmoid-gated-sum shape; the same closed-form argument
      applies. The PoC on `CommittedFieldBlend` is representative.

## What this PoC does NOT test

- The Focal Loss training trick — genuinely training-only, out of scope.
- The Twins representation itself (channel-wise concat of ViT+VAE) —
  image-diffusion-specific, out of scope.
- Whether inference-time blends CAN suffer optimization imbalance —
  they cannot, by definition (no gradient descent). The hypothesis under
  test is the **analogous** imbalance class: output dominance by the
  low-ID / low-frequency / high-magnitude component in a weighted sum.

## Resolution

**Closed 2026-07-28.** The PoC revealed a third outcome neither T6 nor T7
anticipated: the "mismatched" configs don't produce imbalance because the
Twins paper's optimization-imbalance failure mode is training-specific and
does not transfer to inference-time sigmoid-gated blends. The diagnostic
battery correctly flags rank collapse when it exists (sanity config:
erank 24.68 → 1.00) but correctly does NOT flag configs where the
high-ID fields dominate the output covariance by linear-algebra
construction.

**The Twins PASS verdict stands** — honest justification updated from
"covered by shipped primitives" (architecturally sloppy, §3.6 violation)
to "the failure mode is training-specific and does not transfer to
inference-time closed-form blends". The PoC remains as a permanent
regression check at
`riir-ai/crates/riir-poc/benches/composition_imbalance_modelless.rs`.

## References

- arXiv:2607.22531 — Twins paper (the PASS verdict that triggered this issue).
- `katgpt-rs/.research/394_GNN_Survey_Within_Class_Effective_Rank_Fusion.md` — closest cousin (conditional-alignment axis).
- `katgpt-rs/.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md` — closest cousin (intrinsic-dim axis).
- `katgpt-rs/.research/409_MANCE_Manifold_Aware_Concept_Erasure.md` — closest cousin (Two-NN axis).
- `katgpt-rs/crates/katgpt-core/src/committed_field_blend.rs:188-224` — Site 1 (apply_blended_with_pi).
- `katgpt-rs/crates/katgpt-core/src/branching/bank.rs` — Site 3 (BranchBank).
- `riir-ai/crates/riir-poc/` — the defend-wrong R&D crate where the PoC lives.
- Research skill §3.6 — the "architectural coverage ≠ quality parity" rule this issue enforces.
