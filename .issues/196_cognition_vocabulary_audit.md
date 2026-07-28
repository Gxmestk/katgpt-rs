# Issue 196 — Cognition vocabulary audit (Issue 195 follow-up)

**Type:** Audit record / documentation.
**Severity:** Low (this issue mostly records a *negative* result — most
vocabulary is legitimate — plus one scope gap for Issue 195 and a handful
of deferred stylistic notes).
**Triggered by:** Issue 195 (HLA naming collision) raised the question
"do we have this problem elsewhere?" — this issue records what an audit
found so the next agent does not need to re-run it.

## What was audited

~30 neural / cognitive / math vocabulary terms across `katgpt-rs` and
`riir-ai` were grepped and the underlying math verified against the name:
`attention`, `functor`, `curiosity`, `imagination`, `reservoir`, `expert`,
`cognition`, `memory`, `hippocampal`, `belief`, `sample`, `manifold`,
`geometric`, `tropical`, `conformal`, `sleep`/`wake`, `bisimulation`,
`mean_field`, `BoM`, `KARC`, `delta_mem`, `product_key`, `soup`, etc.

## Finding 1 — the valuable negative result: most vocabulary is earned

The HLA collision (Issue 195) is an **outlier, not a pattern**. The audit
confirmed that the vast majority of neural/cognitive vocabulary in the
stack is paper-anchored and the math matches the name:

| Term | Verified math | Verdict |
|---|---|---|
| `curiosity` | `curiosity(x) = ‖x_off_manifold‖² + (1−decay)²·‖x_on_manifold‖²` — prediction-error curiosity (the legitimate RL/active-inference formulation); gates via `s_lp > tau_curiosity` | ✅ real |
| `set_attention`, `funcattn`, `gdn2_attention`, `katgpt-attn-match` | real attention kernels (Q/K/V mass, softmax) — each paper-cited | ✅ real |
| `KARC` (Kolmogorov-Arnold Reservoir Computing) | delay-embedding × Chebyshev/Fourier basis × closed-form ridge readout | ✅ real |
| `BoMSampler` (Bag of Multi-hypotheses) | K-hypothesis belief sampling — paper-cited | ✅ real |
| `manifold_*` (pruner / erasure / bandit) | real manifold geometry (tangent SVD, spectral weighting, σ^α projection) | ✅ real |
| `conformal_predictive_intervals` | real conformal prediction (split conformal, empirical quantiles, CRPS/Winkler) | ✅ real |
| `geometric_product`, `tropical_algebra`, DEC operators | real Clifford algebra / (max,+) semiring / exterior calculus | ✅ real |
| `mean_field_regime`, `bisimulation_operator_inference` | real mean-field theory / Paige-Tarjan partition refinement | ✅ real |
| `delta_mem` | real δ-rule associative memory (rank×rank matrix, sigmoid write gate) — paper-grounded | ✅ real |

**Lesson:** do not assume the HLA failure mode is systemic. Before opening
a "naming smells" issue for any of the above, re-verify the math — the
audit above already confirmed these earn their names.

## Finding 2 — Pattern A: mild vocabulary inflation (DEFERRED by verdict)

A small number of names borrow prestige from fancier fields than the math
warrants. The **docs are honest** about what is underneath (none pretend
to be a neural net), so the cost is stylistic, not dangerous. Deferred
per the verdict below — not worth the churn of a rename.

| Name | Actual mechanism | Borrowed from |
|---|---|---|
| `latent_functor` / `FunctorEntry` | displacement vector `f` + coherence + commitment (the module doc literally says "`f` (the displacement)") | category theory ("functor") |
| `HippocampalCache` | min-heap of `(score_bits, slot_idx)` + keys/vals arrays — a top-W KV cache | neuroscience ("hippocampal") |
| `cognition_sdk` / "think-brain" | stateful WASM module emitting actions + KG triples (game-AI sense of "cognition", not deep-learning sense) | cognitive science |
| `MemorySoupArtifact` | LoRA checkpoint bank (`n_checkpoints`, `gate_dim`, `lora_rank`) | colloquial ("soup") |

- [-] **T1** — `latent_functor`: rename considered, deferred. Pretentious but the module doc shows vector-displacement math immediately; a grep for "functor" won't produce a wrong verdict the way "HLA" did.
- [-] **T2** — `HippocampalCache`: deferred. Borrowed neuroscience prestige for a heap cache, but no collision and the struct fields make the mechanism obvious.
- [-] **T3** — `cognition_sdk` / "think-brain": deferred. "Cognition" is the standard game-AI term for stateful NPC decision logic; the Two-Brain Model framing in the SDK docs is architecturally honest.
- [-] **T4** — `MemorySoup`: deferred. Colloquial, no confusion risk.

**Why these are NOT the HLA failure mode:** none of them (a) collide with
a second distinct concept sharing the same name, (b) embed a category
error in the name itself (like "Attention" on a leaky integrator), or
(c) sit in a position where a grep would mislead an auditor. They are
taste, not bugs.

## Finding 3 — Pattern B: HLA leak in `questbench.rs` (scope gap for Issue 195)

The per-NPC "HLA" misnaming (Issue 195) reached one site the 195 task
list (T1–T10) does not currently cover:

`katgpt-rs/crates/katgpt-core/src/questbench.rs` L634-639:
```rust
pub enum MemoryTier {
    Hot,    // CPU SIMD — standard decode
    Warm,   // HLA KG — O(1) relation lookup   ← per-NPC HLA, not Transformer attention
    Cold,   // Turso — async episode retrieval
    Freeze, // external knowledge
}
```

That "HLA KG" comment refers to the per-NPC leaky-integrator-backed KG
triple lookup — the same misnamed concept Issue 195 targets. It is a
comment, not a symbol, so it is easy to miss in a symbol-rename pass.

- [ ] **T5** — Extend Issue 195's T5 (the comment-sweep task for
  `babel_codec/branching/bridge/cce`) to also cover
  `katgpt-core/src/questbench.rs::MemoryTier::Warm`. Update the comment
  to "belief KG — O(1) relation lookup" (or whatever the post-195-rename
  vocabulary settles on). This is folded into Issue 195's rename, NOT a
  separate rename here — filing separately would duplicate 195.

## Verdict

- **Systemic naming problem?** No. The HLA collision is an outlier.
  ~30 vocabulary terms audited; the large majority are paper-anchored
  and match their math (Finding 1).
- **Mild inflation worth fixing?** No — 4 deferred cases (Finding 2)
  are stylistic and honestly documented; renaming them is churn without
  the audit-cost reduction that justifies the 195 rename.
- **Actionable item:** one missed HLA site in `questbench.rs` (Finding 3)
  → fold into Issue 195 T5.

**Resolution path:** once T5 is folded into Issue 195 (or 195 lands and
its G4 residual-grep sweeps comments too), this issue can be removed per
the noise-reduction rule. The negative result (Finding 1) is the
permanent value — it is why this issue exists.

## References

- [Issue 195](.issues/195_hla_naming_collision.md) — the HLA naming
  collision that triggered this audit.
- `katgpt-rs/crates/katgpt-core/src/questbench.rs` L634-639 — the missed
  HLA site (Finding 3).
- `riir-ai/crates/riir-engine/src/latent_functor/mod.rs` — "functor" =
  displacement vector (Finding 2, T1).
- `katgpt-rs/crates/katgpt-core/src/hippocampal_cache.rs` — "hippocampal"
  heap cache (Finding 2, T2).
- `katgpt-rs/crates/katgpt-core/examples/sleep_time_02_curiosity_inversion.rs`
  — `curiosity(x) = ‖x_off_manifold‖² + ...` (Finding 1, the legitimate
  curiosity math).
