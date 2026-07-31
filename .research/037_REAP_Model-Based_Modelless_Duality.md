# Research: REAP Model-Based/Modelless Duality — Architecture Mapping (37)

> Source: [REAP the Experts: Why Pruning Prevails for One-Shot MoE compression](https://arxiv.org/abs/2510.13999) by Mike Lasby et al. (Cerebras), ICLR 2026
> Local: `.raw/reap/` (upstream Python), `.raw/reap-mlx/` (MLX port, TypeScript)
> Date: 2026-03 (paper), distilled 2026-07
> **Verdict: CONCEPTUAL ALIGNMENT — REAP's model-based/modelless spectrum is already captured by existing trait architecture. No new abstractions needed.**
> **PASS-Redirects (synthesis):** LatentMoE [arXiv:2601.18089 "LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"] (NVIDIA, 2026-01) — PASS: from-scratch MoE training architecture (latent projection + α-scaled expert pool). The post-training compression cousin is MoLAE [arXiv:2503.23100] (low-rank factorization of trained expert weights), which IS the modelless-compression territory REAP covers at the pruning axis. LatentMoE explicitly contrasts itself with MoLAE: LatentMoE compresses ALL experts via a shared projection and scales N to compensate (training-time); MoLAE compresses only FC2 with grouped projections (post-training). Our `ScreeningPruner`/`BanditPruner`/`ConstraintPruner` trait stack (the modelless/model-based duality REAP validates) covers the inference-time routing axis; expert-weight-level low-rank compression would be a separate note if we ever serve MoE LLMs (we don't).

## TL;DR

REAP's expert pruning methods form a spectrum from **modelless** (`frequency` — router counts only, zero inference) to **model-based** (`reap` — gate × activation norm, requires forward pass). This is the same duality our system uses for token-level routing decisions. Our `ScreeningPruner` / `BanditPruner<P>` / `ConstraintPruner` trait stack already absorbs both modes. REAP is an instance of our pattern applied at expert granularity instead of token granularity.

**No new code needed.** The idea is already distilled in the architecture.

---

## REAP's Model-Based/Modelless Spectrum

REAP supports multiple pruning saliency methods. They naturally separate into modelless (no expert execution needed) and model-based (requires forward pass through experts):

| Method | Type | Signal | Cost |
|--------|------|--------|------|
| `frequency` | **Modelless** | Router top-k count per expert | Zero inference |
| `weighted_frequency_sum` | **Modelless** | Sum of gate values (no expert output) | Zero inference |
| `ean_sum` / `ean_mean` | **Light model-based** | Expert activation norms | One forward pass |
| `reap` | **Full model-based** | `mean(g_j(x) × ‖f_j(x)‖)` | One forward pass |
| `reap_l2` | **Full model-based** | L2 variant of REAP saliency | One forward pass |
| `ean_ca` | **Full model-based** | Routed characteristic activation norm | One forward pass |

### The REAP Saliency Formula

```text
saliency_j = mean( g_j(x) * ||f_j(x)|| )
```

Where:
- `g_j(x)` = router softmax weight for expert j (cheap, no expert execution)
- `f_j(x)` = expert output for input x (requires forward pass through expert)
- Mean is taken over routed tokens for that expert

This is literally: **routing signal × activation magnitude** — a cheap signal combined with a model-computed signal.

---

## Our System's Existing Model-Based/Modelless Spectrum

| Component | Type | Signal | Cost |
|-----------|------|--------|------|
| `NoScreeningPruner` | **Modelless** | Returns 1.0 for all | Zero |
| `ConstraintPruner` | **Modelless** | Static rules (syntax validity) | Zero |
| `BanditPruner<P>` | **Modelless→model-based** | Q-values from online rewards | O(1) per step |
| `FlowPruner<P>` | **Modelless** | Flow bonus from GFlowNet theorem | Zero |
| `DeltaBanditPruner` | **Model-based bridge** | δ signal from model log-probs | Inference pass |
| `DeltaGatedAbsorbCompress` | **Model-based** | δ-gated heuristic promotion | Inference pass |
| G-Zero Phase 2 (DPO/GRPO) | **Full model-based** | Gradient updates to LoRA weights | Training pass |

---

## The Three-Layer Mapping

Both systems have the same three conceptual layers:

### Layer 1: ROUTING (modelless)

Cheap signal that doesn't require model execution.

| REAP | Our Stack |
|------|-----------|
| Router gate values `g_j(x)` | BanditPruner Q-values |
| Router top-k frequency counts | ConstraintPruner (static rules) |
| `weighted_frequency_sum` | KeywordRouter domain scores |

### Layer 2: ACTIVATION SCORING (model-based observation)

Signal that requires model forward pass.

| REAP | Our Stack |
|------|-----------|
| Expert output norms `‖f_j(x)‖` | DDTree + ScreeningPruner |
| `ean_sum`, `ean_mean` | ScreeningPruner::relevance() |
| `reap` (combined) | δ signal from log-probs |

### Layer 3: COMBINED SALIENCY (product of both)

The unified signal used for decision-making.

| REAP | Our Stack |
|------|-----------|
| `saliency_j = g_j(x) × ‖f_j(x)‖` | `blended = ln(P_draft) + ln(R_domain × R_bandit)` |
| Top-k pruning on saliency | δ = `(1/T) Σ [log πG(at|q,h,a<t) − log πG(at|q,a<t)]` |
| Super-expert preservation | AbsorbCompress heuristic promotion |

---

## Concrete Component Mapping

| Our Concept | REAP Equivalent | Reap-MLX Field |
|-------------|-----------------|----------------|
| `BanditPruner` Q-values | `frequency` / `weighted_frequency_sum` | `frequency`, `gateValueSum` |
| `ScreeningPruner::relevance()` | `ean_mean` / `ean_sum` | `eanMean`, `eanSum` |
| `δ` (HintDelta) | `reap` saliency | `reap` = gate × norm |
| `FlowPruner` trajectory bonus | — (not in REAP) | — |
| `AbsorbCompress` heuristic promotion | Super-expert preservation | `preserveSuperExperts` |
| `ConstraintPruner::is_valid()` | Min-experts-per-layer safety | `minExpertsPerLayer` |
| `BanditStrategy` selection | Top-k smallest saliency pruning | `selectPruning()` |

---

## Why No New Code Is Needed

The existing trait composition already handles both modes:

```rust
// Modelless: static rules, zero inference
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}

// Unified interface: can be modelless OR model-based depending on impl
pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

// Bridge: modelless Q-values that CAN incorporate model-based rewards
pub struct BanditPruner<P: ScreeningPruner> { /* ... */ }

// Fully model-based: forward pass
pub trait SpeculativeVerifier: Send + Sync {
    fn speculate(&mut self, draft_weights, draft_config, token, pos, rng) -> Vec<usize>;
}
```

`BanditPruner<P: ScreeningPruner>` is already model-based when `P` wraps something that accesses model outputs. The `relevance()` return value is the unified signal — it can be computed modellessly (rules, bandits, frequency) or model-based (δ, activation norms, REAP saliency).

The mathematical structure is identical in both systems:

```text
combined_signal = cheap_routing_signal × model_computed_quality_signal
```

REAP computes this at **expert granularity** (which expert to prune). We compute this at **token granularity** (which token to accept). Same math, different resolution.

---

## G-Zero Phase Alignment

The G-Zero two-phase design mirrors the REAP spectrum explicitly:

| Phase | Mechanism | REAP Analog | Updates |
|-------|-----------|-------------|---------|
| **Phase 1 (Modelless)** | δ → AbsorbCompress + BanditPruner | `frequency` pruning | Heuristics/rules only |
| **Phase 2 (Model-Based)** | δ → GRPO + DPO | `reap` pruning (gate × norm) | LoRA gradient updates |

Phase 1 is REAP's `frequency` — use routing signals without expert execution. Phase 2 is REAP's `reap` — use full model-based saliency for higher quality decisions.

---

## Potential Future Integration

If we ever want to implement REAP-style expert pruning in Rust for our inference pipeline, the trait architecture is ready:

```text
ExpertScreeningPruner (modelless: frequency-based expert selection)
  → impl ScreeningPruner where token_idx maps to expert_id
  → relevance() returns gate frequency as saliency

ExpertActivationScreener (model-based: REAP saliency)
  → wraps ExpertScreeningPruner + activation norms
  → relevance() returns g_j(x) × ‖f_j(x)‖
```

But this is a **feature**, not a conceptual gap. The idea is already distilled.

---

## References

- Paper: https://arxiv.org/abs/2510.13999
- Upstream repo: https://github.com/CerebrasResearch/reap
- Local copy: `.raw/reap/`
- MLX port: `.raw/reap-mlx/`
- Related research: `21_G-Zero_Self-Play_Open-Ended_Generation.md` (δ signal), `23_GFlowNet_Shortest_Paths.md` (flow pruning), `07_Screening_Absolute_Relevance.md` (ScreeningPruner design)