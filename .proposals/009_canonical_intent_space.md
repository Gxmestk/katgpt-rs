# Proposal 009: Canonical Intent Space — Plug-and-Play Any Base Model

**Date:** 2026-07-25
**Research:** [katgpt-rs/.research/459_canonical_intent_space_plug_and_play.md](../.research/459_canonical_intent_space_plug_and_play.md)
**Training-side counterpart:** [riir-train/.research/406_git_rebasin_universal_subspace.md](../../riir-train/.research/406_git_rebasin_universal_subspace.md)
**Source papers:**
- Git Re-Basin (arxiv 2209.04836) — permutation alignment
- Universal Weight Subspace Hypothesis (arxiv 2512.05117) — shared spectral subspaces
- Lottery Ticket Hypothesis (arxiv 1803.03635) — sparse winning subnetworks

**Status:** Proposal — **P2 RAN 2026-07-26, G5 FAILED (Bench 422) for square Procrustes; P1 RAN 2026-07-26, G5 PASSED (Bench 423) for joint-SVD SubspaceAdapter at k ∈ {2,4}; P3 RAN 2026-07-26, G6a FAILED (Bench 424) for modelless centroid construction; partial signal in d_diff construction (+0.46 cross-arch agreement, below 0.5 threshold).** Cross-arch path **not closed** (P1 G5 still holds — shared subspace preserves pairwise alignment) but the canonical direction is harder to construct than expected. Super-GOAT cross-arch claim is **on hold** pending either a better modelless canonical-direction construction (intermediate-layer probe is highest-value next experiment) or intermediate-layer evidence. Intra-arch claim unaffected. See [riir-train/.benchmarks/424_canon_p3_rust_discrimination.md](../../riir-train/.benchmarks/424_canon_p3_rust_discrimination.md). Plan remains unopened.
**Target repos:** `katgpt-rs` (primary, public) + `riir-train` (secondary, private)
**User constraint:** "this should be in katgpt-rs and riir-train as possible bc other riir-* is focus on game" — no files in riir-ai/chain/neuron-db for this work.

---

## Goal

Ship a **canonical intent space** primitive in `katgpt-rs` that lets any frozen base model (Gemma, MiniCPM5, Llama, Qwen) consume the same direction-vector overlays (Rust idioms, NPC personality, emotion vectors, style) without retraining. Each base model carries a deterministic `ModelAdapter` projecting canonical directions into its specific latent space.

This generalizes the existing use case ("we loaded Gemma in our latent space and play game") to "we loaded **any model** in our latent space."

The design fuses three foundational papers:
- **Git Re-Basin** — same-architecture model alignment via permutation symmetries
- **Universal Weight Subspace** — empirical evidence for cross-model shared spectral subspaces
- **Lottery Ticket** — sparse mask transfer across aligned models

---

## Architecture

```text
                    ┌─────────────────────────────────────────┐
                    │  Canonical Intent Space                 │
                    │  (architecture-neutral, owned by katgpt)│
                    │                                         │
                    │  d_Rust_idiom, d_curiosity, d_valence,  │
                    │  d_NPC_personality, d_style, ...        │
                    └────────────────┬────────────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
        ProcrustesAdapter    SubspaceAdapter       MaskAdapter
        (same-arch swap)     (cross-arch joint)    (lottery ticket)
        substrate:           extends:              substrate:
        procrustes.rs        spectral_rewire.rs    spectral_flatness.rs
                │                    │                    │
                └────────────────────┼────────────────────┘
                                     ▼
                          model_specific_latent
                                     │
                                     ▼
                       frozen_base_model.decode()
                                     │
                                     ▼
                              tokens / actions
```

**Functor semantics (user's framing):** `F: CanonicalIntent × ModelAdapter → ModelSpecificLatent`. Linear adapters (Procrustes, Subspace) preserve canonical-space operations (sum, scale, sigmoid-gate). The mask adapter is elementwise, also commuting through linear ops.

---

## Open primitive spec (katgpt-rs)

### Location

New module `crates/katgpt-core/src/canon/` (sibling of `sense/`, `dec/`, `closure/`). Not a new crate — follows existing substrate-module pattern. If the surface grows >1000 LOC, split to `crates/katgpt-canon/` (deferred until needed).

### Surface (~400 LOC P0)

```rust
// crates/katgpt-core/src/canon/mod.rs

/// Architecture-neutral intent direction.
/// Unit-norm f32 vector + BLAKE3 tag for sync/commit.
#[derive(Clone, Debug)]
pub struct CanonicalIntent {
    pub tag: [u8; 32],        // BLAKE3 of label
    pub direction: Vec<f32>,  // unit-norm in canonical space
}

impl CanonicalIntent {
    pub fn new(label: &str, direction: Vec<f32>) -> Self { /* normalize + blake3 */ }
    pub fn dim(&self) -> usize { self.direction.len() }
    pub fn dot(&self, other: &CanonicalIntent) -> f32 { /* cosine since unit */ }
}

/// Projects a canonical intent into a specific base model's latent space.
/// Modelless: zero training, deterministic given adapter state.
pub trait ModelAdapter: Send + Sync {
    /// Apply adapter; write into `out` (len = target_dim).
    /// Zero-alloc hot path: caller-owned buffer.
    fn project_into(&self, canonical: &CanonicalIntent, out: &mut [f32]);

    /// Inverse projection for diagnostics ("what intent is this latent expressing?").
    fn extract_from(&self, model_latent: &[f32]) -> Vec<f32>;

    fn target_dim(&self) -> usize;

    /// BLAKE3 of adapter state — for freeze/thaw attestation + cross-node verify.
    fn commitment(&self) -> [u8; 32];
}

// crates/katgpt-core/src/canon/procrustes_adapter.rs
pub struct ProcrustesAdapter {
    rotation: Vec<f32>,  // row-major d×d, from procrustes.rs
    target_dim: usize,
    commitment: [u8; 32],
}

// crates/katgpt-core/src/canon/subspace_adapter.rs
pub struct SubspaceAdapter {
    basis: Vec<f32>,     // row-major d_target × d_canonical, top-k SVD
    target_dim: usize,
    commitment: [u8; 32],
}

// crates/katgpt-core/src/canon/mask_adapter.rs
pub struct MaskAdapter {
    mask: Vec<u32>,      // bit-packed
    target_dim: usize,
    commitment: [u8; 32],
}
```

### Feature gates

- `canon` (P0, opt-in) — `CanonicalIntent` + `ModelAdapter` trait + `ProcrustesAdapter`
- `canon_subspace` (P1, opt-in) — `SubspaceAdapter` (implies `spectral_rewire`)
- `canon_mask` (P4, opt-in) — `MaskAdapter`

Promotion to default-on gated on G1–G6 (§GOAT gate below).

### Tests

- G1 correctness: `project_into` preserves ranking — canonical directions with higher dot-product produce model-latent directions with higher dot-product (Pearson > 0.95 on synthetic).
- G2 perf: `project_into` < 50µs at d=64 on Apple Silicon (criterion bench).
- G4 alloc-free: `project_into` does zero heap allocations after adapter construction (asserted via `assert_no_alloc` in debug builds).
- Determinism (G3 foundation): bit-identical output across x86_64/aarch64/wasm32 (mirrors `procrustes_determinism.rs`).

---

## Training-side counterpart (riir-train)

Lives in `riir-train/.research/406_git_rebasin_universal_subspace.md`. Three training-only pieces:

1. **STE permutation discovery** — Git Re-Basin's straight-through estimator. Only needed if activation/weight matching (which are modelless) underperform.
2. **Iterative Magnitude Pruning (IMP)** — lottery ticket mask discovery. Feeds `MaskAdapter`.
3. **Stitching layer fallback** — if P2 G5 fails (Procrustes loses too much cross-arch), train a small stitching layer per model pair. The adapter then becomes `TrainedStitchingAdapter` (new variant, riir-train hosts training).

Activation collection for activation matching also lives in riir-train (run N models on a shared prompt set, dump activation matrices).

---

## GOAT gate

Per Research 459 §6. The two star gates decide Super-GOAT vs GOAT:

| Gate | Floor | Target | Decision |
|---|---|---|---|
| G1 correctness | baseline + sysprompt | canonical steering ≥ floor on Rust eval | required to ship at all |
| G2 perf | baseline latency | < 50µs project_into | required |
| G3 no-regression | baseline on MMLU-lite | no capability loss | required |
| G4 alloc-free | hot path | 0 alloc after construction | required |
| **G5 cross-model preservation** | cosine sim floor 0.5 | **> 0.7 on held-out prompts across N≥2 same-arch models** | **GO: Super-GOAT path. NO-GO: fall back to trained stitching (riir-train).** **P2 RESULT (2026-07-26, Bench 422): G5 FAILED for square Procrustes + random projection.** Mean cos = -0.27 (proj_dim=16) and -0.08 (proj_dim=64) on Gemma2-2B ↔ MiniCPM5-1B held-out. Three structural blockers surfaced: (1) dim mismatch 2304 vs 1536, (2) O(d³) Newton-Schulz infeasible at d=1536, (3) underdetermined with n=40 ≪ 2·d. **P1 RESULT (2026-07-26, Bench 423): G5 PASSED for joint-SVD SubspaceAdapter.** Same models, same corpus, same n_train/n_test. Replaced random projection with joint SVD: top-k right singular vectors of M=[A\|B] define the shared subspace. Mean cos = +0.87 (k=2), +0.75 (k=4), +0.68 (k=8), +0.64 (k=16). GO at k ∈ {2, 4}; the cross-arch shared subspace is genuinely low-dimensional. The P2 negative cosine was an artifact of random projection, not a property of the models — refuted. Cross-arch path restored modellessly; Recipe C (trained stitching) no longer a blocker. |
| **G6 cross-architecture gain** | baseline Llama + Rust sysprompt | canonical steering transferred Gemma → Llama beats floor on Rust eval | **GO: cross-arch Super-GOAT. NO-GO: demote to intra-arch GOAT (still ships).** **G6a RESULT (2026-07-26, Bench 424): G6a FAILS for the modelless centroid construction.** Centroid agreement after Procrustes = −0.33 (per-model train centroids in shared subspace point in opposite directions even after rotation — Procrustes aligns shape, not location). Difference-of-means construction shows partial signal (+0.46 cross-arch agreement, below 0.5 threshold; per-model Rust-vs-Python margins +0.08 / +0.14 — both positive but asymmetric). JS discrimination negative on both models (−0.32 / −0.03) — the centroid captures a token-count confound, not Rust-idiom signal. Full G6b (generation with steering) deferred pending substrate work + a passing G6a config. |

Per "Report the Floor" rule (Research 322), G1/G6 floor is **good system prompt**, not "no system prompt". Most style gains evaporate against a well-crafted prompt — G6 is the honesty gate.

---

## Phases

### P0 — Skeleton + ProcrustesAdapter (katgpt-rs)
- [ ] T1.1 Create `crates/katgpt-core/src/canon/mod.rs` with `CanonicalIntent` + `ModelAdapter` trait
- [ ] T1.2 Implement `ProcrustesAdapter` wrapping `katgpt_spectral::procrustes::orthogonal_procrustes`
- [ ] T1.3 G1/G2/G4 tests on synthetic canonical directions
- [ ] T1.4 Feature flag `canon`, default-off
- [ ] T1.5 Determinism test across x86_64/aarch64/wasm32

### P1 — SubspaceAdapter (katgpt-rs)
- [ ] T2.1 Implement joint SVD across N models (extends `spectral_rewire.rs`)
- [ ] T2.2 G1/G2/G4 tests on single-model subspace
- [ ] T2.3 Feature flag `canon_subspace`

### P2 — Cross-model validation (the make-or-break)
- [ ] T3.1 Load Gemma-2-2B + MiniCPM5-1B (both at `riir-train/data/*.gguf`)
- [ ] T3.2 Collect hidden states on 50-prompt Rust code snippet set
- [ ] T3.3 Fit Procrustes R: gemma_hidden ↔ minicpm_hidden
- [ ] T3.4 **G5: measure cos(R · h_gemma, h_minicpm) on held-out prompts. Decision point.**
- [ ] T3.5 If G5 fails → open issue for stitching-layer fallback in riir-train, demote to GOAT

### P3 — Rust-style canonical direction + G6 (the real test)
- [x] T4.1 Construct `d_Rust_idiom` canonical direction (modelless centroid + difference-of-means, Bench 424)
- [ ] T4.2 G6a: measure cross-arch discrimination of canonical direction on Rust vs non-Rust (Bench 424 — **FAIL for centroid, PARTIAL for d_diff**)
- [ ] T4.2a Intermediate-layer probe (layers 6/12/18 of 24-26) — highest-value next experiment per Git Re-Basin
- [ ] T4.2b Length-normalized projections (address the JS token-count confound)
- [ ] T4.2c Larger contrastive corpus for d_diff (30-50 Python prompts vs current 10)
- [ ] T4.3 G6b: steer Gemma and MiniCPM via the same canonical direction; measure Rust eval delta vs sysprompt floor (REQUIRES `forward_llama_with_embedding` substrate in riir-engine — deferred until G6a passes)
- [ ] T4.4 If G6 fails → keep as intra-arch GOAT, narrow the selling point, demote verdict

### P4 — MaskAdapter (auxiliary, gated on P3)
- [ ] T5.1 IMP mask discovery in riir-train (lottery ticket)
- [ ] T5.2 `MaskAdapter` impl in katgpt-rs (elementwise apply, modelless)
- [ ] T5.3 Test mask transfer across Procrustes-aligned models

### P5 — Promotion / demotion
- [ ] T6.1 If G1–G6 all pass → promote `canon` to default-on; write benchmark note in `.benchmarks/`
- [ ] T6.2 If G5/G6 fail → keep opt-in, document scope limit, ship as intra-arch GOAT

---

## Scope discipline

**What this proposal does NOT do:**

- Does not modify riir-ai/chain/neuron-db (user explicit override — those repos are game/chain/shard-focused).
- Does not change the existing `latent_functor/procrustes_bridge.rs` in riir-ai (that's the game-side consumer; it can adopt the new `canon` module later if beneficial, but that's a separate plan).
- Does not implement STE permutation discovery in katgpt-rs (training → riir-train).
- Does not implement IMP mask discovery in katgpt-rs (training → riir-train).
- Does not train stitching layers unless P2 G5 fails (modelless-first mandate per AGENTS.md §"MANDATORY: exhaust modelless paths before deferring to riir-train").

**What stays private vs open:**

| Open (katgpt-rs, MIT) | Private (riir-train) |
|---|---|
| `CanonicalIntent` type | STE permutation discovery code |
| `ModelAdapter` trait | IMP mask discovery |
| `ProcrustesAdapter` impl | Trained stitching layers (fallback only) |
| `SubspaceAdapter` impl | Activation collection on training data |
| `MaskAdapter` apply (not discovery) | Per-model trained adapter weights |
| Joint SVD algorithm | |

---

## Why this is Super-GOAT candidate (and what would demote it)

**Super-GOAT case (Q1–Q4 all YES mechanically):**
- Novel: no prior art on Git Re-Basin permutation algorithms or canonical intent unification in the 7-repo stack
- New capability: plug-and-play any base model — currently impossible
- Selling point: swap Gemma → Llama without retraining overlays
- Force multiplier: ≥5 systems connect

**What would demote to GOAT:**
- G5 fails: Procrustes loses too much cross-architecture → fall back to trained stitching. Still ships, just needs a training step. Becomes "plug-and-play any same-arch model" instead of "any model".
- G6 fails: canonical steering transferred cross-arch doesn't beat a good system prompt → narrow to "intra-architecture snapshot swap". Still useful (Gemma-A ↔ Gemma-B), narrower selling point.

**What would kill it entirely:**
- G1 fails on the modelless path AND riir-train stitching also fails G1 → no plug-and-play at any tier. Document the negative result and move on. (Unlikely — Research 406 SAR already proved single-model SVD purification works modellessly.)

---

## References

- Research: [katgpt-rs/.research/459_canonical_intent_space_plug_and_play.md](../.research/459_canonical_intent_space_plug_and_play.md)
- riir-train counterpart: [riir-train/.research/406_git_rebasin_universal_subspace.md](../../riir-train/.research/406_git_rebasin_universal_subspace.md)
- Substrate shipped:
  - [Issue 001 / Plan 152 — orthogonal Procrustes](../crates/katgpt-spectral/src/procrustes.rs)
  - [Plan 423 / Research 406 — SAR spectral rewiring](../.research/406_Spectral_Rewiring_Weight_Delta_Purification.md)
- Cousin research:
  - [178 Rosetta cross-model](../.research/178_Rosetta_Neurons_Cross_Model_Alignment.md)
  - [238 LoRA-Muon gauge invariant](../.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md)
  - [227 GPart isometric](../.research/227_GPart_Isometric_Partition_Inference.md)
  - [231 SOPTV sparse off-principal](../.research/231_Sparse_Off_Principal_Task_Vector_OPD.md)
- Source papers:
  - [Git Re-Basin (arxiv 2209.04836)](https://arxiv.org/abs/2209.04836)
  - [Universal Weight Subspace (arxiv 2512.05117)](https://arxiv.org/abs/2512.05117)
  - [Lottery Ticket Hypothesis (arxiv 1803.03635)](https://arxiv.org/abs/1803.03635)
