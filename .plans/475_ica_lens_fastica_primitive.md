# Plan 475: ICA Lens — FastICA Non-Gaussian Direction Primitive + ERF Diagnostic

**Date:** 2026-08-11
**Status:** Open
**Research:** `.research/475_ICA_Lens_FastICA_Non_Gaussian_Directions.md`
**Paper:** [arXiv:2606.11722](https://arxiv.org/abs/2606.11722) (Liu & Han, Jun 2026)
**Related Plans:** 418 (MAG — verdict-supervised cousin, Super-GOAT), 415 (Within-Class Effective Rank), 412 (Subspace Steering Field), 301 (subspace_phase_gate — ships `jacobian_svd_at_into`), 203 (Kurtosis Gate — ships `excess_kurtosis()`), 151 (NITP — ships `effective_rank`), 287 (Sink-Aware — ships `stable_rank_update_into`)
**Feature gate:** `ica_lens` (opt-in; promote to default only if G2 + G3 PASS on a realistic substrate)
**Crate target:** `katgpt-rs/crates/katgpt-spectral/src/ica_lens.rs` (sibling to `hla_eigenbasis.rs` — both operate on activation windows; ICA consumes the EigenbasisTracker's Gram as its whitening step)

---

## Goal

Ship the FastICA fixed-point iteration (non-Gaussianity-maximizing rotation after whitening) as a modelless inference-time primitive, plus the ERF (Effective Receptive Field) diagnostic. The primitive fills the missing third corner of the direction-acquisition triangle: designer-authored (R290) / verdict-supervised (MAG R397) / **unsupervised-statistical (this plan)**.

**The load-bearing claim:** on non-Gaussian data, FastICA directions are strictly more non-Gaussian than PCA directions ranked by kurtosis post-hoc. The GOAT gate G2 measures this gap on both synthetic and realistic substrates.

---

## Phases

### Phase 1 — FastICA primitive (CORE)

- [ ] **T1.1** Add `ica_lens.rs` to `katgpt-spectral/src/`. Module docstring cites Research 475 + arXiv:2606.11722 + the relationship to `EigenbasisTracker` (ICA consumes its Gram as the whitening step).
- [ ] **T1.2** Implement `FastIcaConfig` struct:
  ```rust
  pub struct FastIcaConfig {
      /// Target component count (the paper's `m`). Halved on adaptive refit.
      pub n_components: usize,
      /// Max FastICA iterations per component.
      pub max_iters: u32,
      /// LIM convergence threshold (the paper's `τ`, default 1e-4).
      pub lim_threshold: f32,
      /// Row-normalize activations before whitening (paper recipe A1, default true).
      pub row_normalize: bool,
      /// Acceptance rule: Strict (max-LIM < τ) or P95 (p95-LIM < τ, paper recipe A2).
      pub acceptance: IcaAcceptance,
      /// Adaptive refit: halve n_components on failure down to this floor (paper recipe A3, default 16).
      pub min_components: usize,
      /// Contrast function: LogCosh (default), Exp, or Cubic.
      pub contrast: IcaContrast,
  }
  pub enum IcaAcceptance { Strict, P95 }
  pub enum IcaContrast { LogCosh, Exp, Cubic }
  ```
- [ ] **T1.3** Implement `FastIcaScratch` (pre-allocated workspace):
  ```rust
  pub struct FastIcaScratch {
      /// The rotation matrix W (m × m).
      w: Vec<f32>,
      /// Source scores S = Z W^T (n × m).
      s: Vec<f32>,
      /// Per-component LIM values (m).
      lim: Vec<f32>,
      /// Workspace for the fixed-point update (m).
      w_new: Vec<f32>,
      /// Workspace for tanh + 1-tanh² expectations (n).
      tanh_buf: Vec<f32>,
      /// Reuse EigenbasisScratch for the whitening step.
      eigenbasis_scratch: EigenbasisScratch,
  }
  ```
- [ ] **T1.4** Implement `fastica_into(window, t_dim, d_dim, config, scratch, out_reading_map, out_writing_map, out_source_scores) -> FastIcaResult`:
  - Step 1: row-normalize (if config.row_normalize).
  - Step 2: center + whiten via `EigenbasisScratch` (reuse the existing Gram + eigvec machinery from `hla_eigenbasis.rs`).
  - Step 3: FastICA fixed-point iteration with symmetric orthogonalization.
  - Step 4: adaptive refit loop (halve `n_components` until acceptance or `min_components`).
  - Step 5: compute reading map `R = WK`, writing map `D = R^†` (pseudoinverse via the existing eigvecs).
- [ ] **T1.5** Implement `IcaAcceptance` check (Strict max-LIM vs P95-LIM).
- [ ] **T1.6** Implement the three contrast functions (LogCosh default, Exp, Cubic).
- [ ] **T1.7** Add `FastIcaResult` struct: `(reading_map, writing_map, source_scores, n_accepted, n_unstable, status, component_kurtosis)`. The `component_kurtosis` field uses the existing `excess_kurtosis()` from Plan 203 to rank the recovered directions.
- [ ] **T1.8** Add feature gate `ica_lens` to `katgpt-spectral/Cargo.toml`. Default-off.
- [ ] **T1.9** Unit tests:
  - Synthetic non-Gaussian source (mixture of two Laplace + one Uniform) → FastICA recovers the source directions within cosine ≥ 0.95.
  - Gaussian source → FastICA returns an arbitrary rotation (any W is valid; kurtosis ≈ 0).
  - Row-norm vs raw on outlier-heavy input → row-norm version converges, raw fails.
  - P95 vs Strict acceptance → P95 accepts a layer with 1 unstable component that Strict rejects.
  - Adaptive refit → a layer that fails at m=d accepts at m=d/2.
- [ ] **T1.10** Property test: bit-identical results across two runs with same input + seed (G2 determinism warmup).

### Phase 2 — ERF diagnostic (the novel diagnostic)

- [ ] **T2.1** Implement `effective_receptive_field(component_scores_full, component_scores_suffix, k_max) -> usize`:
  - Input: the signed scores of a component at a target token under (a) full context and (b) suffixes of increasing length.
  - Output: the minimum suffix length `k` such that the component is in the top-15 by absolute score AND preserves its sign.
  - Returns `k_max` if no suffix recovers within the window.
- [ ] **T2.2** Implement `erf_batch(reading_map, activations, evidence_indices, k_max) -> Vec<f32>`:
  - For each component, average the per-evidence-example ERF.
  - Reuses the FastIca reading map to project suffix activations.
- [ ] **T2.3** Unit tests:
  - Token-local component (activates from suffix length 1) → ERF = 1.
  - Context-dependent component (needs suffix length ≥ 5) → ERF ≥ 5.
  - Unrecoverable component → ERF = k_max.
- [ ] **T2.4** Property test: ERF is monotone non-increasing in suffix length (longer suffix → more likely to recover).

### Phase 3 — GOAT gate

- [ ] **T3.1 (G1 — Latency)** Fitting time on (T=512, D=8, m=8) ≤ 100 µs (CPU, release mode). On (T=10K, D=64, m=64) ≤ 10 ms.
- [ ] **T3.2 (G2 — Quality, the load-bearing gate)** FastICA directions are MORE non-Gaussian than PCA + kurtosis-ranking on:
  - (a) Synthetic non-Gaussian source (mixture of Laplace + Uniform): FastICA mean kurtosis ≥ 2× PCA mean kurtosis.
  - (b) Realistic high-dim substrate (synthetic d=64 non-Gaussian mixture mimicking NeuronShard style_weights): FastICA mean kurtosis ≥ 1.5× PCA mean kurtosis.
  - (c) If a realistic low-dim HLA-like substrate (d=8) is available: FastICA mean kurtosis ≥ 1.2× PCA mean kurtosis. (Lower bar — at d=8 the gap may be small.)
  - **GATE FAILS if (a) OR (b) fail.** (c) is informational only — if it fails, the HLA application is scoped to "diagnostic only" (compute ICA offline, audit designer-authored axes, don't replace at runtime).
- [ ] **T3.3 (G3 — No regression)** All existing `katgpt-spectral` + `katgpt-core` tests pass under `--features ica_lens`. The `EigenbasisTracker` API is unchanged; FastICA is additive.
- [ ] **T3.4 (G4 — Alloc-free steady-state)** After the first call, `fastica_into` allocates 0 bytes for a given `(T, D, m)` triple (all scratch pre-allocated).
- [ ] **T3.5 (G5 — Determinism)** Bit-identical results across two runs on the same machine (no nondeterministic floating-point reduction order).
- [ ] **T3.6** Decision: if G2 (a) + (b) + G3 + G4 + G5 PASS → promote `ica_lens` to default-on. If G2 (c) FAILS but (a) + (b) PASS → keep opt-in, document the HLA scope-limit. If G2 (a) OR (b) FAILS → keep opt-in as a research artifact, do NOT promote.

### Phase 4 — Docs + integration hooks

- [ ] **T4.1** Add module docstring to `ica_lens.rs` citing the paper, the relationship to `EigenbasisTracker` + `excess_kurtosis`, and the three stability recipes.
- [ ] **T4.2** Add a README.md section under "Feature Showcase" describing ICA Lens, the GOAT verdict, and the fusion angles (F1–F4 from Research 475).
- [ ] **T4.3** Add an example under `katgpt-rs/examples/` showing FastICA on a synthetic non-Gaussian source + comparison to PCA + kurtosis-ranking.
- [ ] **T4.4** Cross-reference from `hla_eigenbasis.rs` docstring: "for the non-Gaussianity-maximizing variant, see `ica_lens.rs` (Plan 475)".

---

## Implementation notes

### Why `katgpt-spectral` (not `katgpt-core`)

FastICA consumes `EigenbasisTracker` (which lives in `katgpt-spectral/src/hla_eigenbasis.rs`) as its whitening step. Placing the primitive alongside its closest dependency keeps the import graph clean. `katgpt-core` is for substrate-level primitives with no heavy deps; `katgpt-spectral` already hosts the eigenbasis machinery.

### The FastICA update (parallel variant, paper §3)

```
For each component j = 1..m:
    w_j ← E[z · tanh(w_j^T z)] − E[1 − tanh²(w_j^T z)] · w_j

W ← symmetric_orthogonalize(W)    // W(W^T W)^{-1/2}
```

The symmetric orthogonalization uses the eigendecomposition of `W W^⊤` (small m × m matrix; reuse the Jacobi eigensolver from `data_probe/geometry.rs`).

### The reading map vs writing map

- **Reading map** `R = WK`: projects normalized centered activations into signed component scores. Used for non-Gaussianity analysis, top-example retrieval, sparse probing, ERF computation.
- **Writing map** `D = R^†`: maps component coordinates back to the activation space. Columns are the activation-space direction vectors. Used for steering-style edits + SAE-decoder comparison.

Both are computed once at fit time and frozen as `MerkleFrozenEnvelope` artifacts (same pattern as MAG).

### The ERF suffix schedule

The paper uses `k = 1, 2, ..., K_max = 11` (token window). For our substrate (NPC ticks), the schedule is `k = 1, 2, 4, 8, 16, 32, 64` (exponential — covers reactive to deliberative timescales at 20 Hz tick). The primitive is parameterized over the schedule; the default is exponential with `K_max = 64`.

---

## Non-goals

- **SAE training / dictionary learning.** The paper positions ICA as a complement to SAEs, not a replacement. We do not train SAEs (Research 143 rejected them as not modelless + retrieval-specific). ICA gives us the compact-direction benefit without the training cost.
- **GPU FastICA.** The paper's PyTorch implementation runs on GPU. Our primitive is CPU-first (no GPU dep) — fitting on our substrate sizes (T ≤ 10K, D ≤ 64) is sub-second on CPU. GPU FastICA is a future optimization if a consumer needs it.
- **Human annotation protocol.** The paper's Section 5 is research methodology. We do not ship an ICA Explorer web tool.
- **Per-layer corpus fitting (LLM-specific).** The paper fits ICA at each residual-stream layer of GPT-2 / Gemma / Qwen. Our substrate doesn't have "residual-stream layers" in the LLM sense; we fit per substrate (HLA window, NeuronShard corpus, latent_functor state).
- **Overcomplete ICA variants.** The paper's Limitations section mentions overcomplete ICA, JADE, Infomax, extended Infomax, heavy-tail-aware objectives. Out of scope for this plan — the basic FastICA + the three stability recipes is the load-bearing contribution.

---

## Failure modes (honest)

1. **G2 (c) fails at d=8.** The HLA application is scoped to "diagnostic only" — compute ICA directions offline, audit the designer-authored HLA axes for non-Gaussianity, but don't replace them at runtime. The primitive still ships (useful for NeuronShard d=64 and future high-dim substrates).
2. **The EigenbasisTracker Gram is not the right whitening input.** `EigenbasisTracker` maintains a rolling-window Gram for the HOT PATH (per-tick incremental update). FastICA needs the FULL Gram of the fitting corpus, not the rolling window. **Mitigation:** FastICA takes the activation window directly (not the tracker), builds its own Gram via `EigenbasisScratch`. The tracker is for online PCA; FastICA is offline.
3. **The p95-LIM acceptance rule lets unstable components through.** Downstream consumers (CommittedFieldBlend) might blend an unstable archetype direction. **Mitigation:** the `FastIcaResult.component_kurtosis` field exposes per-component stability; consumers filter by it.
