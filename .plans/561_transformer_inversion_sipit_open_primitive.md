# Plan 561: Transformer Inversion — SipIt Open Primitive

**Date:** 2026-07-26
**Research:** [katgpt-rs/.research/232_Task_Relevant_Identifiability_Specialist.md](../.research/232_Task_Relevant_Identifiability_Specialist.md) (Gain-Redirects line) + cross-refs in `.research/158` (MUX) + `.research/244` (FaithfulnessProbe)
**Source paper:** [arXiv:2510.15511](https://arxiv.org/abs/2510.15511) — Nikolaou, Mencattini, Crisostomi, Santilli, Panagakis, Rodolà, *Language Models are Injective and Hence Invertible*, ICLR 2026
**Reference impl:** <https://github.com/giorgosnikolaou/SIPIT>
**Target:** `katgpt-rs/crates/katgpt-core/src/inversion/` (new module) + Cargo feature `transformer_inversion`
**Status:** Active — Phase 1 (skeleton + random policy) pending consumer

---

## Goal

Ship a **generic, modelless, zero-alloc** SipIt-style transformer inversion primitive in `katgpt-core` that, given a frozen decoder-only transformer's forward function and the layer-ℓ per-position hidden state matrix `H̆^(ℓ) ∈ R^{T×d}`, recovers the discrete input token sequence `s = ⟨s₁, …, s_T⟩` exactly in `O(T·|V|)` verifier calls (worst case) via per-position vocabulary search with a gradient-guided policy on a continuous proxy embedding.

**No game IP, no chain IP, no neuron-shard IP.** This is a public-engine adoption hook for transparency / interpretability / audit tooling on standard decoder-only text transformers. Per the paper's threat model (§3): the natural setting is one where an adversary obtains the hidden-state matrix — through a leaked KV-cache, shared-inference pipeline, or API exposing intermediate representations.

**GOAT gate** (per AGENTS.md): feature flag `transformer_inversion`, default OFF. Opt-in research primitive — promote only if a concrete consumer (in riir-ai or external) demonstrates a quality/security/latency gain. No promotion until consumer exists.

---

## Scope — what this primitive IS and IS NOT

### IS (in scope)

- **A modelless inference-time algorithm.** The gradient is on a continuous proxy embedding `e^(j)` at position `t`, not on model parameters. No backprop through weights. Per AGENTS.md §3.5 Path 0: genuinely modelless.
- **A generic vocabulary-indexed search.** Given (a) a forward function `F(v; π, t) → R^d` that returns the layer-ℓ hidden state at position `t` when the prefix is `π` and the current token is `v`, (b) the observed per-position states `H̆`, (c) a tolerance `ε`, and (d) a policy `Π` that enumerates `V \ C` — recovers `s` exactly via local verifier tests.
- **A trait + a default implementation.** The `InversionPolicy` trait abstracts over random / gradient-guided / external policies. The default ships random-without-replacement (worst-case `T·|V|` tests); a gradient-guided policy ships behind a `grad_policy` sub-feature (uses caller-supplied autodiff — we do NOT add an autodiff dependency).

### IS NOT (out of scope — rejected fusions, documented to prevent re-litigation)

- **❌ NOT applicable to HLA.** HLA is a sigmoid-bounded per-NPC 8-dim belief kernel (`katgpt-core::hla::kernel`), not a decoder-only text transformer. Theorem 2.2 requires real-analytic activations (GELU/SiLU/SwiGLU/GeGLU) and a vocabulary-indexed embedding lookup. HLA has neither. Do not wire SipIt into NPC cognition.
- **❌ NOT a sync-boundary compression primitive.** The sync boundary (per AGENTS.md) commits **raw scalars** (valence/arousal/desperation/calm/fear = 5 f32 = 20 bytes) or **weight shards** (NeuronShard fixed-size Pod). Transmitting a layer-ℓ hidden state matrix `T×d×4` bytes (e.g. 50 tokens × 768 dim = 153 KB) would be a ~7000× bandwidth **increase** over the 20-byte scalar sync, not a decrease. The "transmit compact h, run SipIt on receiver" fusion is backwards.
- **❌ NOT a Cold-tier storage compression.** `riir-neuron-db` already stores weight shards (fixed-size Pods, BLAKE3-committed via `MerkleFrozenEnvelope`). For NPC cognition we don't store text prompt strings; we store direction vectors + scalars. SipIt has no consumer here.
- **❌ NOT a lossless activation-hashing scheme.** Theorem 2.2 is measure-zero over the **parameter space** (Pr[collision] = 0 under absolutely continuous initialization). It is **NOT** bit-exact over f32 representations — the paper itself uses `torch.allclose(rtol=1e-5, atol=1e-8)` for the empirical collision check (§E.1). BLAKE3 over f32 activations will collide for any two prompts whose activations fall within float precision; the theorem doesn't prevent this. (For bit-exact commitment we already BLAKE3-commit raw scalars + weight shards.)
- **❌ NOT a substitute for the sync-boundary rule.** "Never send full embedding over network when scalar projection suffices — sync the 5 scalars, not the 64-dim vector" (AGENTS.md) remains absolute. SipIt does not modify this rule.

---

## Architecture

```
crates/katgpt-core/src/inversion/
├── mod.rs              ← public API: invert_sequence, InversionConfig, InversionResult, InversionError
├── verifier.rs         ← local verifier: AcceptanceRegion, accept_observation
├── policy.rs           ← InversionPolicy trait + RandomPolicy + GradientGuidedPolicy
├── recovery.rs         ← invert_sequence driver: T-outer × |V|-inner loop, prefix-conditioned
└── tests.rs            ← toy-transformer unit tests (random-init tiny GPT, G1 exact recovery)
```

**Cargo feature:** `transformer_inversion = ["dep:fastrand"]` (always-on fastrand already in core). Sub-feature `grad_policy = []` (no extra deps; the caller supplies `∇F` via a closure — we don't take an autodiff dep).

### Public API sketch

```rust
/// Configuration for SipIt-style inversion.
#[derive(Clone, Debug)]
pub struct InversionConfig {
    /// Acceptance tolerance ε. Theory: ε < Δ_π,t / 2. Practice: small + backoff.
    pub tolerance: f32,
    /// Max vocabulary trials per position (default |V|).
    pub max_trials_per_position: usize,
    /// Policy for candidate enumeration.
    pub policy: InversionPolicy,
}

/// Observed per-position layer-ℓ hidden states `H̆^(ℓ) ∈ R^{T×d}`, row-major.
pub struct ObservedStates<'a> {
    pub states: &'a [f32],   // T * d
    pub t_len: usize,
    pub d_len: usize,
}

/// Forward signature: given prefix `π` and a candidate token `v` at position `t`,
/// return the layer-ℓ hidden state at position `t`.
///
/// Caller supplies this. It wraps their transformer's forward pass up to layer ℓ
/// (with prefix conditioning). No autodiff required for the random policy.
pub trait InversionForward {
    fn hidden_at(&self, prefix: &[u32], candidate: u32, position: usize) -> Result<Vec<f32>, InversionError>;
}

/// Gradient hook (only used by `GradientGuidedPolicy`). Caller supplies ∇_e F.
pub trait InversionGradient {
    fn grad_hidden_at(&self, prefix: &[u32], proxy: &[f32], position: usize) -> Result<Vec<f32>, InversionError>;
}

#[derive(Clone, Copy, Debug)]
pub enum InversionPolicy {
    /// Uniform-without-replacement. Worst case T·|V| tests.
    Random,
    /// Gradient-guided ranking (paper Alg 3). Caller must supply `InversionGradient`.
    GradientGuided { step_size: f32, grad_clip: f32 },
}

pub enum InversionResult {
    /// Exact recovery (within tolerance) at every position.
    Recovered(Vec<u32>),
    /// Could not verify any candidate at `failed_position` within `max_trials_per_position`.
    Failed { failed_position: usize, candidates_tried: usize },
}

/// Run SipIt. Outer loop over positions t=1..T; inner loop enumerates V \ C until
/// the observed state h̆_t falls in the acceptance region A_{π,t}(v; ε).
pub fn invert_sequence<F: InversionForward>(
    observed: &ObservedStates,
    vocab_size: u32,
    forward: &F,
    grad: Option<&dyn InversionGradient>,
    config: &InversionConfig,
) -> Result<InversionResult, InversionError>;
```

**Allocation discipline** (per AGENTS.md optimization rules):
- Prefix `π` reuses a single `Vec<u32>` that grows by 1 per outer iteration.
- Visited set `C` is a `Vec<bool>` of length `|V|`, reset per position.
- Gradient proxy `e^(j)` reuses a single `Vec<f32>` of length `d`.
- `hidden_at` output writes into a caller-supplied `&mut [f32]` scratch (`_into` variant) — no per-trial allocation.
- The `Vec<f32>` return in `InversionForward::hidden_at` above is for clarity; the production API uses `hidden_at_into(&mut self, ..., out: &mut [f32])`.

---

## Phases

### Phase 1 — Skeleton + Random Policy + G1 (CORE)

- [ ] **T1.1** Create `crates/katgpt-core/src/inversion/` module skeleton with `#[cfg(feature = "transformer_inversion")]`. Add `transformer_inversion = []` feature to `crates/katgpt-core/Cargo.toml`. Export from `crates/katgpt-core/src/lib.rs` behind feature gate.
- [ ] **T1.2** Implement `ObservedStates`, `InversionConfig`, `InversionPolicy::Random`, `InversionResult`, `InversionError` in `mod.rs` + `verifier.rs`.
- [ ] **T1.3** Implement `InversionForward` trait + `RandomPolicy` enumeration (uniform-without-replacement via fastrand permutation).
- [ ] **T1.4** Implement `invert_sequence` driver in `recovery.rs` — outer T-loop × inner |V|-loop with early break on acceptance. Zero-allocation hot path (verify with `dhat` bench in Phase 3).
- [ ] **T1.5** Add unit tests in `tests.rs`:
  - Toy 2-layer decoder-only transformer (GELU activation, d=16, |V|=32, T=8), random init.
  - `g1_exact_recovery_random_init` — generate random prompts, run forward to get `H̆^(ℓ)`, run `invert_sequence`, assert exact recovery.
  - `g1_recovers_when_two_prompts_differ_only_at_position_t` — direct test of the paper's causality argument (Lemma D.2).
  - `g1_no_false_positive_on_mismatched_observed` — wrong `H̆` does not produce the original prompt.
- [ ] **T1.6** `cargo clippy -p katgpt-core --features transformer_inversion --all-targets` clean. `cargo test -p katgpt-core --features transformer_inversion --lib inversion::` green.

### Phase 2 — Gradient-Guided Policy (paper Alg 3)

- [ ] **T2.1** Implement `InversionGradient` trait in `policy.rs`. Caller supplies `∇_e F` via a closure (no autodiff dep).
- [ ] **T2.2** Implement `GradientGuidedPolicy` with step size γ, gradient clipping at norm 1, periodic projection to nearest vocab embedding every K=50 proposals (paper §E.1).
- [ ] **T2.3** Test: same toy transformer, verify gradient-guided finds the token in < 0.5% · |V| trials on average (paper reports <0.25% for |V|=32K-128K; we should be in the same ballpark relative to |V|).
- [ ] **T2.4** `cargo clippy` + `cargo test` clean.

### Phase 3 — G2 Latency + G4 Alloc-Free Bench

- [ ] **T3.1** Add `benches/inversion_bench.rs` — criterion bench on the toy transformer, measuring median time per position for random vs gradient-guided policy. Compare to paper's "28s for 20-token GPT-2 Small" baseline (note: paper measures on A100 + 50257 vocab; we measure on CPU + tiny vocab, so direct comparison is not meaningful — instead establish the linear-in-|V| scaling claim).
- [ ] **T3.2** G2 gate: median time per position scales linearly with |V| for the random policy; gradient-guided is sub-linear (paper Fig 4 reports <0.25% of |V|).
- [ ] **T3.3** G4 alloc-free: `dhat` bench shows 0 bytes allocated steady-state (excluding the prefix `Vec` growth of 4 bytes/position, which is amortized via `Vec::push`).

### Phase 4 — Robustness (paper Thm 3.2)

- [ ] **T4.1** Implement `ObservedStates` with optional perturbation `ę_t = h_t + e_t`, `‖e_t‖ < ε < Δ_π,t / 2`.
- [ ] **T4.2** Test: inject noise at varying ε, verify recovery holds while `ε < Δ/2` and fails when `ε > Δ/2`. Direct empirical measurement of the margin `Δ_π,t` on the toy transformer.
- [ ] **T4.3** Document the relationship to quantization (paper Table 2: FP4/INT8 quantization preserves injectivity in practice, more than doubles the minimum distance). Add a note in the module doc.

### Phase 5 — Promotion / Demotion Decision

- [ ] **T5.1** If a concrete consumer materializes in riir-ai (e.g., a transparency/audit feature on a text-LLM path) — wire it, run the GOAT gate at the consumer level, promote `transformer_inversion` to opt-in-recommended in the docs.
- [ ] **T5.2** If no consumer materializes within ~3 months — keep as opt-in research infrastructure. Do NOT promote to default (no consumer = no GOAT gain to measure).
- [-] **T5.3** (deferred) If a future text-LLM consumer in katgpt-rs itself (e.g., a speculative-decode audit mode) wants this — wire it then. Speculative today uses the drafter's own hidden states; no inversion needed.

---

## What this plan does NOT do (rejected fusions — do NOT re-add without amending this plan)

| Proposal | Why rejected | Where to find the analysis |
|---|---|---|
| Apply SipIt to HLA per-NPC state | HLA is a sigmoid-bounded kernel, not a text transformer; theorem doesn't transfer | Research 232 Gain-Redirects |
| Activation-based sync compression | Sync already commits 32-byte hash; transmitting activations is 96× bandwidth increase | Research 232 Gain-Redirects |
| Lossless activation hashing | Theorem is measure-zero over parameters, not bit-exact over f32 | Research 232 Gain-Redirects |
| Cold-tier prompt re-hydration | SipIt needs model weights + per-position matrix; activations are 15-1000× larger than prompts | Research 232 Gain-Redirects |
| Transmit compact h for quorum audit | Violates AGENTS.md sync-boundary rule (sync scalars, not embeddings) | Research 232 Gain-Redirects |

---

## References

- **Source paper:** Nikolaou et al., *Language Models are Injective and Hence Invertible*, ICLR 2026. [arXiv:2510.15511](https://arxiv.org/abs/2510.15511).
- **Reference implementation:** <https://github.com/giorgosnikolaou/SIPIT>
- **Closest shipped cousins:** Research 158 (MUX — different lossless mechanism), Research 232 (Task-Relevant Identifiability — different identifiability sense), Research 244 (FaithfulnessProbe — same last-token substrate, different operation).
- **Commercial strategy:** public open primitive per `.research/003_Commercial_Open_Source_Strategy_Verdict.md`. No game / chain / shard IP. The open primitive is the adoption hook for transparency/audit tooling on standard text transformers; the private game runtime does not consume it (HLA is the cognitive substrate, not a text transformer).
