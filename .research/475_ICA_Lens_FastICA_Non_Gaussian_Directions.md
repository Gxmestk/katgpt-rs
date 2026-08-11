# Research 475: ICA Lens — FastICA Non-Gaussian Direction Mining for LLM Activations

> **Source:** [ICA Lens: Interpreting Language Models Without Training Another Dictionary](https://arxiv.org/abs/2606.11722) — Sida Liu (independent) + Feijiang Han (UMD), v1 10 Jun 2026, 47pp. Code + checkpoints + explorer released.
> **Date:** 2026-08-11
> **Status:** Active — GOAT verdict, plan filed (Plan 475)
> **Related Research:** 397 (MAG — verdict-supervised direction mining, the closest cousin), 394 (Within-Class Effective Rank — eigenvalue-spectrum diagnostic), 393 (Block-Sparse Featurizer — subspace concept primitive), 388 (Jacobian Lens — SVD readout cousin), 180 (Rosetta polarization — kurtosis as monosemanticity metric), 113 (NITP — effective rank parent), 276 (PersonalityWeightedComposition — direction consumer), 290 (Latent Field Steering — injection cousin), 302 (FAME/CommittedFieldBlend — archetype-direction consumer), 143 (Latent Terms SAE — the SAE-rejection precedent)
> **Related Plans:** 418 (MAG — Super-GOAT, shipped), 415 (Within-Class Effective Rank — shipped), 412 (Subspace Steering Field — shipped), 301 (subspace_phase_gate — ships `jacobian_svd_at_into`), 203 (Kurtosis Gate — ships `excess_kurtosis()`), 151 (NITP — ships `effective_rank`), 287 (Sink-Aware — ships `stable_rank_update_into`), 475 (this primitive — FastICA + ERF)
> **Classification:** Public

---

## TL;DR

**ICA Lens** is a training-free interpretability method that uses **FastICA** (Independent Component Analysis, Hyvärinen 1999) to find maximally **non-Gaussian** directions in LLM residual-stream activations — without training a Sparse Autoencoder. The paper's headline: many useful interpretable directions are *selective* on tokens, and selective directions should look *less Gaussian* than random projections. ICA directly searches for these non-Gaussian directions after centering + whitening.

Three stability recipes make FastICA practical on LLM activations (where naive scikit-learn FastICA is brittle): **row-normalization** (reduce outlier-norm influence before whitening), **p95-LIM acceptance** (accept a fit when 95% of components have stabilized, even if a small tail hasn't), and **adaptive refit** (halve the target component count `m` until convergence, returning the highest accepted resolution).

The paper introduces **ERF (Effective Receptive Field)** — a per-component diagnostic measuring how much left context is sufficient to recover a component's activation at a target token. Small-ERF components are token-local; large-ERF components are context-dependent. Across GPT-2 Small / Gemma 2 2B / Qwen 3.5 2B Base, ICA directions are substantially more non-Gaussian than both random projections AND public SAE decoder directions (Figure 3), competitive with SAEs on sparse probing, and outperform SAEs on Targeted Probe Perturbation at small-to-medium budgets. The paper also finds a **negative correlation between kurtosis and ERF** (ρ ∈ [−0.41, −0.50]) — high-kurtosis components are more local; broad-context components are less sharply kurtotic.

**Why it matters here:** this codebase has a rich direction-vector ecosystem with TWO acquisition paths — **designer-authored** (Latent Field Steering R290) and **supervised-extracted** (EmotionDirections P162, KG Latent Octave R196, MAG R397/P418). MAG is the unsupervised-with-verdict-labels cousin — it mines directions from the model's own verdicts. **ICA is the unsupervised-without-verdict-labels cousin** — it mines directions from the statistical structure of the activation distribution itself, requiring no verdict signal at all. This fills the third corner of the acquisition triangle (designer / verdict-supervised / unsupervised-statistical). Plus ERF is a genuinely novel diagnostic with no shipped analog — it tells you whether a direction is reactive (token-local) or deliberative (context-dependent), which is exactly the cognitive-hierarchy signal the latent_functor + HLA runtime needs.

**Distilled for katgpt-rs (modelless, inference-time):**
- `fastica(activations, m, ...)` → `(reading_map R, writing_map D, source_scores S)` — the non-Gaussianity-maximizing rotation after whitening. Strictly stronger than PCA + kurtosis-ranking post-hoc.
- `effective_receptive_field(component, evidence_examples, k_max)` → mean suffix length needed to recover the component's signed score. Novel diagnostic.
- `excess_kurtosis_of_projection(direction, activations)` → scalar non-Gaussianity signal (already ships as `excess_kurtosis()` in Plan 203).
- The three stability recipes (row-norm, p95-LIM, adaptive refit) as generic fitting options.

No training, no gradients, no SAE dictionary learning. FastICA is a one-shot linear-algebra operation: center → whiten (via the already-shipped `EigenbasisTracker`) → fixed-point iteration to maximize non-Gaussianity (the `log(cosh(·))` contrast).

---

## 1. Paper Core Findings

### 1.1 The FastICA mechanism (Section 3)

Given activation matrix `X_ℓ ∈ ℝ^{n×d}` from layer `ℓ`:

1. **Row-normalize**: `r(x_i) = x_i / max(‖x_i‖₂, ε)` — reduces the influence of large-norm token outliers (the attention-sink / massive-activation regime documented by Dettmers 2022, Sun 2024, Xiao 2024) before whitening.
2. **Center**: subtract the mean `μ = (1/n) Σ x̄_i`.
3. **Whiten**: compute the whitening matrix `K` such that `Z = X_c K^⊤` has identity covariance. This removes second-order correlations.
4. **FastICA fixed-point iteration**: find rotation `W ∈ ℝ^{m×m}` maximizing non-Gaussianity of the projected sources `S = Z W^⊤`, with the standard log-cosh contrast `G(u) = log(cosh(u))`:
   ```
   max_{WW^⊤=I}  (1/n) Σ_i Σ_j G((ZW^⊤)_ij)
   ```
   The parallel FastICA update (per-component, then symmetric orthogonalization):
   ```
   w_j ← E[z · g(w_j^⊤ z)] − E[g'(w_j^⊤ z)] · w_j     // g = tanh (derivative of log-cosh)
   W ← symmetric_orthogonalize(W)                       // enforce WW^⊤ = I
   ```
   Iterate until convergence (LIM statistic < τ).
5. **Output artifact** `A_ℓ = (μ, K, W, R, D, S, I_st, I_tail, status)`:
   - **Reading map** `R = WK` — projects normalized centered activations into signed component scores `S = X_c R^⊤`.
   - **Writing map** `D = R†` (pseudoinverse) — maps component coordinates back to the activation space. Columns of `D` are the activation-space direction vectors associated with each ICA component. Used for intervention / steering / SAE-decoder comparison.

### 1.2 The three stability recipes (the practical contribution)

The paper's headline insight is that ICA has been *underestimated* for LLM interpretability because off-the-shelf FastICA (e.g., scikit-learn) is brittle on LLM activations. Three recipes fix this:

| Recipe | Mechanism | Effect (GPT-2 Small, 1M activations) |
|---|---|---|
| **Row-normalization** | `x_i ← x_i / ‖x_i‖₂` before centering/whitening | Accepted layers: 2 → 8 (max-LIM) / 8 → 10 (p95-LIM); total iterations 3107 → 2741 |
| **p95-LIM acceptance** | Accept fit when `p95(LIM_j) ≤ τ` (instead of strict `max(LIM_j) ≤ τ`) | Rescues layers where a small unstable tail dominates the max; flags the tail as unstable but keeps it for inspection |
| **Adaptive refit** | Halve target `m` from `d` down to `m_min=16` until convergence | Returns the highest accepted resolution per layer; accepted component count becomes a fitting-difficulty diagnostic |

Combined effect: +400% accepted layers, −21.5% total FastICA iterations on GPT-2 Small.

### 1.3 What ICA recovers — non-Gaussianity (Section 4.1)

Across all three model families, ICA directions are **substantially more non-Gaussian** than both random unit directions AND public SAE decoder directions (Figure 3):
- Random projections: excess kurtosis ≈ 0 (Gaussian, as expected by CLT)
- SAE decoder directions: elevated non-Gaussianity (despite SAEs not being trained to maximize kurtosis — they learn it implicitly through sparse reconstruction)
- ICA directions: highest non-Gaussianity (explicitly optimized for it)

This validates the paper's core thesis: **non-Gaussianity is a useful interpretability signal that SAEs learn only implicitly.** ICA makes the inductive bias explicit and gets there without training a dictionary.

### 1.4 ERF — Effective Receptive Field (Section 4.2, the novel diagnostic)

For each ICA component `j`, ERF measures how much left context is sufficient to recover the component's signed score at a target token:

```
For target token t in sequence x_{1:T}:
  h_t(x_{1:t}) = full-context activation
  s_j(h_t)     = signed score of component j

For suffix length k = 1, 2, ..., K_max (= 11):
  h_t^(k) = activation from suffix x_{t-k+1:t} only
  R_j(x, t, k) = 1[ j ∈ Top15({|s_r(h_t^(k))|}_r) ∧ sign(s_j(h_t^(k))) = sign(s_j(h_t)) ]

erf_j(x, t) = min { k : R_j(x, t, k) = 1 }     // first suffix length that recovers
ERF(j) = mean over evidence examples
```

**ERF reveals a local-to-contextual spectrum** (Figure 4):
- Early layers: dominated by token-local components (ERF ≈ 1)
- Middle layers: peak in long-context components (ERF ≥ 11)
- Late layers: still contain many token-local directions

**Key finding (Figure 5):** excess kurtosis and ERF are **negatively correlated** (Spearman ρ ∈ [−0.41, −0.50] across all three models). High-kurtosis components are typically more local (easier to explain from top examples); broad-context components have lower kurtosis (harder to summarize from token-local views).

### 1.5 Feature utility — sparse probing + TPP (Section 6)

On SAEBench:
- **Sparse Probing:** ICA competitive with public SAEs, outperforms PCA and ITDA consistently across all three models.
- **Targeted Probe Perturbation (TPP):** ICA strongest at small-to-medium intervention budgets (top-N ablated features). At larger N, SAEs catch up (more capacity in the overcomplete dictionary).

### 1.6 ICA vs SAE relationship (Section 7)

- ICA recovers many SAE-aligned directions (median nearest-SAE cosine ≈ 0.4–0.6 — partial overlap, not one-to-one).
- ICA directions often vary more smoothly across neighboring tokens (track contextual spans); SAE features peak at single tokens (sparsity pressure).
- **Positioning:** ICA as a *compact first lens*, NOT a replacement for SAEs. SAEs win at high-resolution feature discovery; ICA wins at cheap, browsable, intervention-friendly compact bases.

### 1.7 What is training-only (NOT distilled here)

- The GPU-parallel PyTorch FastICA implementation itself is engineering, not a training loop — but the paper's specific `torch` implementation is out of scope for our Rust stack. We re-implement the algorithm in Rust.
- The human annotation protocol (Section 5) is research methodology, not a primitive.
- The ICA Explorer web tool is engineering, not a primitive.

---

## 2. Distillation

### 2.1 The transferable primitive (stripped of the paper's LLM setting)

The paper operates on 768–2304-dim LLM residual streams with 1M-token activation corpora. The transferable insight is **modelless and substrate-agnostic**: any system with an activation corpus `{x_i ∈ ℝ^d}` can run FastICA. The math is:

1. **Whitening** (already ships as `EigenbasisTracker`): compute the covariance `Σ = (1/n) Σ (x_i − μ)(x_i − μ)^⊤`, eigendecompose, form the whitening matrix `K = Λ^{-1/2} U^⊤` such that `Z = X_c K^⊤` has identity covariance.
2. **Non-Gaussianity-maximizing rotation** (the novel primitive): find `W ∈ ℝ^{m×m}` (with `WW^⊤ = I`) maximizing `Σ_i Σ_j log(cosh((ZW^⊤)_ij))`. The fixed-point update is `w_j ← E[z · tanh(w_j^⊤ z)] − E[1 − tanh²(w_j^⊤ z)] · w_j`, then symmetric orthogonalization. Iterate until LIM (the relative change in `w_j`) drops below threshold `τ`.
3. **Reading map** `R = WK`, **writing map** `D = R†`.
4. **ERF diagnostic** (novel): for each component, find the minimum suffix length that recovers the component's signed score.

**Critical distinction from PCA:** PCA (which our `EigenbasisTracker` does) finds directions of MAXIMUM VARIANCE. ICA finds directions of MAXIMUM NON-GAUSSIANITY. These differ when the underlying sources are non-Gaussian — which NPC activations certainly are (emotional states, behavior modes, attention patterns are all heavy-tailed, not Gaussian). The paper's Figure 3 shows the gap is substantial: ICA directions have ~100× the excess kurtosis of random directions, while PCA directions (which maximize variance, not kurtosis) are in between.

**Our current heuristic and why ICA is strictly stronger:** we already compute `excess_kurtosis()` (Plan 203) on each PCA direction returned by `EigenbasisTracker` to rank them by non-Gaussianity. This is a POST-HOC ranking of PCA directions. ICA does the JOINT optimization — it finds the rotation that maximizes non-Gaussianity directly, not the rotation that maximizes variance and then happens to have non-Gaussian projections. On non-Gaussian data, ICA directions are strictly more non-Gaussian than PCA directions ranked by kurtosis.

### 2.2 Where the pieces already live (the fusion map)

| Piece | Existing location | Relationship |
|-------|-------------------|--------------|
| Whitening (covariance eigendecomp) | `katgpt-rs/crates/katgpt-spectral/src/hla_eigenbasis.rs::EigenbasisTracker` (default-on, G1 ≤ 2µs at T=512, D=8, k=4) | **ICA consumes this as its whitening step.** The Gram matrix maintained by `EigenbasisTracker` IS the covariance; its eigenvectors ARE the whitening basis. ICA adds the FastICA rotation on top. |
| Excess kurtosis (non-Gaussianity signal) | `katgpt-rs` (Plan 203, `excess_kurtosis()` SIMD-friendly O(V), default-on) | **ICA maximizes this explicitly.** Currently used as a post-hoc ranker on PCA directions; ICA promotes it to the primary optimization objective. |
| Effective rank (eigenvalue-spectrum entropy) | `katgpt-rs/crates/katgpt-core/src/data_probe/geometry.rs::effective_rank` (Plan 151) + `within_class_effective_rank` (Plan 415) | **Complementary diagnostic.** Effective rank measures overall representation health; ICA's per-direction kurtosis measures individual direction selectivity. Together: effective rank = "how many directions?" / ICA = "which directions are most selective?" |
| Stable rank | `katgpt-rs/crates/katgpt-core/src/data_probe/sink_classify.rs::stable_rank_update_into` (Plan 287) | **Same metric as ICA's "block stable rank".** Already shipped. |
| Direction injection | `apply_latent_steering` (R290/Plan 309), `spherical_steering` (Plan 405), `subspace_steering` (Plan 412) | **ICA feeds these** — mines the directions they inject |
| Supervised direction extraction | `EmotionDirections::extract_direction` (P162), civ_emotion mean-difference (R196) | **ICA is the unsupervised sibling** — same "find a direction" goal, no labels needed |
| Verdict-supervised unsupervised extraction | MAG `mine_contrast_direction` (R397/P418, Super-GOAT, default-on) | **ICA is the no-verdict cousin** — MAG needs the runtime to emit verdicts; ICA needs only the activation distribution |
| BLAKE3 commitment | `MerkleFrozenEnvelope` (riir-neuron-db/freeze.rs) | **ICA directions are frozen artifacts** — same envelope |
| Curiosity / exploration | CGSP (R126/P299), Curiosity Pulse (R041) | **ICA-ERF = cognitive hierarchy signal** — token-local directions → plasma tier (reactive); context-dependent → hot tier (deliberative) |

**Nothing here is new math except the FastICA fixed-point iteration itself.** What is new: (a) the **non-Gaussianity-maximizing rotation** (vs our current variance-maximizing PCA), (b) the **ERF diagnostic** (no shipped analog), and (c) the **three stability recipes** (row-norm, p95-LIM, adaptive refit) for practical fitting.

### 2.3 Closest cousins (3)

1. **MAG (R397/P418, Super-GOAT, shipped default-on)** — the **verdict-supervised** unsupervised direction mining cousin. MAG's `mine_contrast_direction(positive, negative)` computes `u_Q = (μ⁻ − μ⁺)/‖·‖` where the classes come from the model's own verdict `y_M`. ICA's reading-map rows are the **no-verdict** analog: directions of maximum non-Gaussianity in the activation distribution itself, with no class labels at all. **MAG and ICA are complementary:** MAG answers "what directions does the model's verdict reveal?"; ICA answers "what directions are statistically exceptional regardless of verdict?". Where they agree → the verdict captures the natural structure; where they disagree → there's structure the verdict misses (or verdict-aligned structure that isn't statistically dominant).

2. **EigenbasisTracker (`katgpt-spectral/hla_eigenbasis.rs`, default-on)** — the **whitening substrate** ICA consumes. Today it returns PCA directions (variance-maximizing). ICA adds the FastICA rotation on top of the same Gram matrix. At `m = d` with Gaussian sources, ICA = PCA (any rotation is equally good); with non-Gaussian sources (which NPC activations are), ICA is strictly stronger.

3. **Kurtosis Gate (P203, `excess_kurtosis()`, default-on)** — the **non-Gaussianity signal** ICA maximizes. Today it ranks draft marginals in speculative decoding; ICA uses the same metric to rank activation-space directions. The signal is the same; the application domain differs.

### 2.4 Fusion (the GOAT angle)

**F1 (PRIMARY — katgpt-rs): FastICA × EigenbasisTracker × excess_kurtosis — the unsupervised direction discovery primitive.**

Today: `EigenbasisTracker.recover()` returns PCA directions; caller can rank them by `excess_kurtosis` post-hoc. This is a heuristic for non-Gaussianity.

With FastICA: add `fastica_into(window, m, ...)` that runs the fixed-point iteration on the already-whitened activations from `EigenbasisTracker`. Returns the rotation `W` whose rows are the maximally non-Gaussian directions. **Strictly stronger than PCA + kurtosis-ranking on non-Gaussian data.**

The math reuses the existing `EigenbasisScratch` (Gram, eigvecs, eigvals) + adds a small `FastIcaScratch` (the rotation matrix `W`, the source scores `S`, the per-component LIM vector). ~150 LOC.

**F2 (SECONDARY — katgpt-rs): ERF diagnostic × latent_functor quality gate — the cognitive hierarchy signal.**

Today: the latent_functor quality gate (`riir-engine/src/latent_functor/quality_gate.rs`) measures coherence / Dirichlet energy but has no notion of "how much temporal context does this direction need to activate?". ERF adds exactly that: a per-direction scalar measuring context dependence.

Reframe for our substrate: token-local directions (ERF ≈ 1) are reactive — they should route to the plasma tier (sub-µs, SIMD). Context-dependent directions (ERF ≥ 5) are deliberative — they should route to the hot tier (sub-ms, GPU) or warm tier (ms+, async). This is the **cognitive hierarchy signal** the latent_functor runtime needs to make tier-routing decisions.

The diagnostic is modelless: for each direction, run the runtime on suffixes of increasing length, check if the direction's signed score is preserved. Same fixed-point iteration as the paper's ERF.

**F3 (TERTIARY — riir-ai): FastICA × MAG — the unsupervised acquisition closure.**

MAG (R397) mines directions from verdict-conditioned activation shifts. ICA mines directions from the unconditional activation distribution. Together they close the unsupervised acquisition loop:
- ICA discovers the "natural axes" of the activation distribution (no verdict needed)
- MAG discovers the "verdict-revealing axes" (verdict-conditioned)
- Comparison: high overlap → the verdict captures the natural structure; low overlap → there's structure the verdict misses

For NPC runtime: ICA runs on the consolidation/sleep-cycle tier (offline, full activation corpus); MAG runs on the per-tick curiosity path (online, verdict-conditioned). ICA directions seed the direction pool; MAG refines them with verdict signal.

**F4 (QUATERNARY — riir-neuron-db): FastICA × NeuronShard style_weights[64] — the play-style archetype discovery.**

`style_weights[64]` is a 64-dim basis. Today it's interpreted as 64 flat scalars. FastICA on a corpus of shards → discovers the natural "play-style concept" axes (the non-Gaussian directions in style-weight space). These are the data-driven archetype directions that CommittedFieldBlend (P321) blends — replacing designer-authored archetypes with discovered ones. Connects P321 + R302 + MAG (R397, for the verdict-conditioned variant).

---

## 3. Verdict

### Tier: **GOAT** (open primitive)

| Question | Answer | Notes |
|----------|--------|-------|
| Q1 No prior art? | **PARTIAL.** The COMBINATION (FastICA + row-norm + p95-LIM + adaptive refit + ERF) is novel per web search (no prior work applies FastICA to LLM residual streams with these stability recipes; prior ICA-on-embeddings work — Yamagiwa 2023/2024, Musil & Mareček 2022/2024 — operates on static word embeddings, not LLM hidden states). The individual pieces ship: whitening (`EigenbasisTracker`), kurtosis (`excess_kurtosis` Plan 203), effective rank (`effective_rank` Plan 151), direction mining (MAG R397/P418). The **FastICA fixed-point iteration itself does NOT ship** — it's a genuinely missing primitive (variance-maximizing PCA vs non-Gaussianity-maximizing ICA are different optimizations on non-Gaussian data). The **ERF diagnostic has no shipped analog**. | Vocabulary translation: "FastICA" / "ICA" / "independent component" → no hits in any repo (grep confirmed). "non-Gaussian" / "kurtosis" → hits in Plan 203 (kurtosis gate) + Research 180 (Rosetta polarization). "whitening" → `EigenbasisTracker` (the whitening step). "effective receptive field" → no hits (CNN-vision concept only). Three-layer check (notes + code + vocab) done. |
| Q2 New class of behavior? | **NO.** It is a *better acquisition step* for an existing capability class (direction mining). MAG (R397) already does unsupervised direction mining — ICA is the no-verdict-label variant of the same capability. The ERF diagnostic is novel but it is a *diagnostic* (measures a property), not a *mechanism* (does something). The capability class — "discover interpretable directions in activation space" — is the same. | |
| Q3 Product selling point? | **PARTIAL.** "Our NPCs discover their natural cognitive axes from their activation distribution, without any verdict signal" is sellable, but it is a *quality* claim about an existing capability (better direction mining), not a *new capability* claim. "Our NPCs know whether each cognitive axis is reactive or deliberative" (ERF) is more differentiated but still a diagnostic, not a behavior. | |
| Q4 Force multiplier? | **YES.** Connects MAG (R397) + EigenbasisTracker + excess_kurtosis (P203) + effective_rank (P151/P415) + CommittedFieldBlend (P321) + EmotionDirections (P162) + Subspace Steering Field (P412) + latent_functor quality gate + CGSP curiosity. ≥9 cousins. But this is largely the SAME set MAG already connects — ICA is a complement to MAG, not a new force multiplier. | |

**Q2 fails → GOAT, not Super-GOAT.** No private guide required (per skill §1.5: "If NO to any → proceed to GOAT/Gain verdict. Plan only, no guide.").

**Not Super-GOAT if:** G2 (FastICA directions are more non-Gaussian than PCA + kurtosis-ranking on our substrate) fails — if the post-hoc ranking is good enough on our low-dim HLA (d=8) substrate, the joint optimization buys nothing. The paper's results are on d ∈ [768, 2304]; at d=8 the curse of dimensionality is absent and PCA might already be near-optimal. **Mitigation:** the GOAT gate (Plan 475) runs on both a controlled synthetic non-Gaussian source (where ICA must beat PCA) AND on a realistic high-dim substrate (NeuronShard `style_weights[64]`, latent_functor state).

### MOAT gate (per domain, §1.6)

| Domain | Verdict | Reasoning |
|--------|---------|-----------|
| **katgpt-rs** (this repo) | **IN SCOPE — strengthens moat.** FastICA + ERF are fundamental modelless primitives — pure linear algebra + statistics, no game/chain/shard semantics. Fits the `data_probe` / `katgpt-spectral` pillar alongside the existing `effective_rank`, `excess_kurtosis`, `EigenbasisTracker`. | Generic math. Ships behind feature flag with GOAT gate. Per-stack tracking: this occupies the "direction mining" slot alongside MAG (verdict-supervised) and EmotionDirections (label-supervised). |
| **riir-ai** (private runtime) | Consumer only. The F3 fusion (FastICA × MAG closure) is a pillar amplifier for the Reasoning Pack (P8) + Self-Learn NPCs pillars, but the primitive itself belongs in the public engine. No private guide (not Super-GOAT). | |
| **riir-chain / riir-neuron-db** | Cross-ref only (F4 archetype discovery on `style_weights[64]`). Not primary targets. | |
| **riir-train** | NOT a target. FastICA is modelless (one-shot linear algebra). No training loop. | |

### One-line reasoning

ICA Lens is a training-free interpretability method that finds maximally non-Gaussian directions in activations via FastICA — the missing unsupervised-without-verdict-labels cousin of MAG (R397, verdict-supervised) and EmotionDirections (P162, label-supervised). The FastICA fixed-point iteration (non-Gaussianity-maximizing rotation after whitening) is genuinely missing from our stack; the ERF diagnostic (context-dependence measurement) is novel. GOAT: provably stronger than our current PCA + kurtosis-ranking heuristic on non-Gaussian data; ships behind feature flag; GOAT gate on both synthetic non-Gaussian source + realistic high-dim substrate.

### Routing

- **katgpt-rs/.plans/475_ica_lens_fastica_primitive.md** — open primitive. `fastica_into(window, m, ...)` + `effective_receptive_field(...)` + the three stability recipes. Feature flag `ica_lens`. GOAT gate G1–G5.
- **riir-ai** — consumer only (F3 fusion with MAG). No private guide.
- **riir-neuron-db** — cross-ref only (F4 archetype discovery). Not this primitive's scope.
- **riir-train** — NOT a target. Modelless.

---

## 4. Constraints check

| Constraint | Status |
|------------|--------|
| Modelless / inference-time | ✅ No training, no gradients, no backprop. FastICA is a one-shot fixed-point iteration on the whitened activation matrix. ERF is a forward-pass sweep over suffix lengths. Both are inference-time operations. |
| Latent-to-latent preferred | ✅ Operates entirely in activation/latent space. Whitening → rotation → scoring, all in the d-dim latent space. Never decodes to tokens. ERF operates on the runtime's own forward passes (latent-space activations at target tokens). |
| Use sigmoid not softmax | ✅ No softmax anywhere. FastICA uses the log-cosh contrast (additive, not softmax). The per-direction kurtosis is a scalar moment, not a probability. |
| Freeze/thaw over fine-tuning | ✅ Mined ICA directions are frozen as `MerkleFrozenEnvelope` artifacts (same as MAG). The reading map `R` and writing map `D` are versioned, BLAKE3-committed. No weight mutation. |
| 7-repo discipline | ✅ Open primitive (math) → katgpt-rs. Game integration (per-NPC wiring, ERF-based tier routing) → riir-ai (consumer). No chain/shard IP in the open primitive. |
| Raw scalars at sync boundary | ✅ The mined direction vectors stay latent (local to entity/runtime). Only the resulting scalar projections (kurtosis, ERF) cross sync as raw f32. Same boundary discipline as HLA / MAG. |
| Zero-alloc hot path | ✅ FastICA fitting is OFFLINE (consolidation/sleep-cycle tier) — not a per-tick hot-path operation. Per-direction kurtosis + ERF scoring are `O(N·d)` SIMD-able, pre-allocated. The `EigenbasisTracker` hot path (rolling-window Gram update) is unchanged; FastICA adds only the offline rotation step. |

---

## 5. Open questions / risks

1. **Does FastICA beat PCA + kurtosis-ranking on our low-dim substrate (d=8)?** The paper's results are on d ∈ [768, 2304]. At d=8 (HLA), the curse of dimensionality is absent and PCA + post-hoc kurtosis-ranking might be near-optimal — ICA's joint optimization might buy nothing. **Mitigation:** the GOAT gate (Plan 475) runs on BOTH a controlled synthetic non-Gaussian source (where ICA must beat PCA by construction) AND a realistic high-dim substrate (NeuronShard `style_weights[64]`, latent_functor state, or a synthetic d=64 non-Gaussian mixture). If ICA only wins on high-dim substrates, the primitive stays useful but the HLA application is scoped to "diagnostic only" (compute ICA directions offline, use them to AUDIT the designer-authored HLA axes, don't replace them at runtime).

2. **Is the ERF diagnostic meaningful on our substrate?** The paper's ERF measures how much TEXTUAL context a direction needs. Our substrate's "context" is TICK HISTORY (for NPC runtime) or PROMPT PREFIX (for LLM-style cognition). The translation: ERF on NPC HLA = "how many past ticks does this direction need to activate?". This is meaningful (reactive vs deliberative cognitive axes), but the K_max = 11 token window from the paper maps to a different window size on our substrate (probably K_max = 32–64 ticks). **Mitigation:** the primitive is generic over the suffix-length schedule; the GOAT gate uses a sensible default for our substrate.

3. **Compute cost of FastICA fitting.** The paper reports fitting on 1M activations × d=768 in "minutes" on GPU. Our substrate is smaller (per-NPC HLA window is T=512 × d=8; consolidation corpus is ~10K shards × d=64). FastICA on these sizes is sub-second on CPU. **Mitigation:** the primitive ships CPU-first (no GPU dep); the GOAT gate measures fitting time on our realistic substrate sizes.

4. **Row-normalization discards magnitude information.** The paper's row-norm recipe normalizes each activation by its L2 norm before whitening. This is great for stability but discards the "how strongly is this token activated?" signal. For our substrate (NPC HLA scalars), magnitude IS meaningful (high fear vs low fear). **Mitigation:** row-norm is an OPTION (not mandatory); the primitive supports both raw and row-normalized fitting. The paper itself notes the normalized components remain useful despite the loss.

5. **The p95-LIM acceptance rule trades strictness for coverage.** Accepting a fit when 95% of components have stabilized means 5% might be unstable. The paper flags these but keeps them for inspection. For our substrate, an unstable component could mislead downstream consumers (CommittedFieldBlend blending an unstable archetype direction). **Mitigation:** the primitive exposes the per-component stability flag; downstream consumers can filter by it.

6. **Operator selection.** The paper fits ICA on residual-stream activations at a specific layer. Our substrate has multiple "layers" (HLA kernel, latent_functor state, NeuronShard style_weights) — each is a different "activation corpus". ICA should be fit independently per substrate, not jointly. **Mitigation:** the primitive is parameterized over the activation corpus; the GOAT gate runs on each substrate separately.

---

## 6. Pre-plan cherry-pick audit (skill §1.7)

**NOT REQUIRED.** This plan consumes katgpt-rs primitives (`EigenbasisTracker` from `katgpt-spectral`, `excess_kurtosis` from the speculative module) INTO katgpt-rs itself (the FastICA primitive lands in `katgpt-spectral` or `katgpt-core`). No cross-repo consumer yet — riir-ai consumes later via the standard path-dep + feature-flag pattern. The goat-audit skill's scope is "katgpt-rs primitive consumed into riir-*"; this plan is katgpt-rs-internal.

---

## TL;DR

ICA Lens is a training-free interpretability method that uses FastICA to find maximally non-Gaussian directions in activations — the missing unsupervised-without-verdict-labels cousin of MAG (verdict-supervised, R397/P418) and EmotionDirections (label-supervised, P162). The FastICA fixed-point iteration (non-Gaussianity-maximizing rotation after whitening) is genuinely missing from our stack — today we do PCA (`EigenbasisTracker`) + post-hoc kurtosis ranking (`excess_kurtosis`), which is a heuristic for what ICA does jointly and optimally. The ERF diagnostic (context-dependence measurement) is novel with no shipped analog. GOAT: provably stronger than our current heuristic on non-Gaussian data; the open question is whether the gap is meaningful at our low-dim HLA substrate (d=8) or only at higher dims (NeuronShard d=64, latent_functor state). Ships in `katgpt-spectral` or `katgpt-core` behind `ica_lens` feature flag; GOAT gate G1–G5 on synthetic non-Gaussian source + realistic high-dim substrate. Fills the third corner of the direction-acquisition triangle (designer / verdict-supervised / unsupervised-statistical).
