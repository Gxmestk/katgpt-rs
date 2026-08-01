//! Transformer-level FFN Mixture-of-Experts (MoE) — DeepSeek-V3 §3.3.
//!
//! Implements the auxiliary-loss-free load balancing from DeepSeek-V3
//! (`noaux_tc` top-K method) with the Kimi-K3 sigmoid router. See Research 328
//! for the full distillation and Proposal 032 for the architectural context.
//!
//! # The mechanism (single-token decode)
//!
//! ```text
//! 1. Router affinity (sigmoid — independent per expert, NEVER softmax):
//!    logits[e] = dot(e_e, h)         for each routed expert e ∈ [0, N_r)
//!    s[e]      = sigmoid(logits[e])  ∈ (0, 1)
//!
//! 2. noaux_tc bias + top-K selection (THE LOAD-BEARING DETAIL):
//!    biased[e] = s[e] + b_e           (b_e = e_score_correction_bias[e])
//!    topk_idx  = argtopK(biased, K_r) (the K experts with highest BIASED score)
//!    topk_s    = s[topk_idx]          ← RAW s values (NOT biased!)
//!
//! 3. Renormalization (uses RAW scores — bias does NOT participate):
//!    g[k]      = topk_s[k] / sum(topk_s)
//!
//! 4. Shared expert (always on, no gating):
//!    out       = swiglu_ffn(shared_weights, h)
//!
//! 5. Routed experts (weighted by renormalized g):
//!    for k in 0..K_r:
//!        out  += g[k] * swiglu_ffn(expert[topk_idx[k]], h)
//!
//! 6. Residual add is CALLER's responsibility (this fn writes the FFN output
//!    into `hidden_out`; the caller adds it to the residual stream).
//! ```
//!
//! # Why the bias is selection-only (Research 328 §3.3)
//!
//! DeepSeek-V3's design choice is deliberate: bias affects WHICH experts fire
//! (load balancing) but NOT how much each fires (signal strength). If the bias
//! leaked into renormalization, the model would conflate "picked for load
//! balancing" with "confident about this token" — degrading quality. The G1
//! test `g1_bias_does_not_leak_into_renormalization` enforces this bit-identically.

use katgpt_core::simd::simd_matmul_rows;
use katgpt_core::sigmoid;
use katgpt_core::types::math::swiglu_inplace;

// ─── Config ─────────────────────────────────────────────────────────────────

/// MoE configuration parameters.
///
/// Mirrors the Kimi-K3 / DeepSeek-V3 config fields. See Research 328 §4 for the
/// 0.40B-specific values.
#[derive(Clone, Debug)]
pub struct MoeConfig {
    /// Number of routed experts (`N_r`). Kimi-K3-0.40B: 8.
    pub num_experts: usize,
    /// Number of shared experts (`N_s`), always on. Kimi-K3-0.40B: 1.
    pub num_shared_experts: usize,
    /// Top-K routed experts per token (`K_r`). Kimi-K3-0.40B: 2.
    pub num_experts_per_token: usize,
    /// SwiGLU intermediate dim (`d_ffn`). Loaded from `moe_intermediate_size`
    /// in Phase 5; Phase 3 tests parametrize this.
    pub moe_intermediate_size: usize,
    /// Hidden dim (`d`). Kimi-K3-0.40B: 1024.
    pub hidden_size: usize,
    /// When true (Kimi-K3), the router uses sigmoid (per-expert independent);
    /// when false (DeepSeek-V3 paper default), softmax over experts.
    /// This impl assumes sigmoid — the AGENTS.md global rule forbids softmax.
    pub use_sigmoid_router: bool,
    /// Renormalize selected experts' raw scores to sum to 1. Kimi-K3: true.
    pub renormalize: bool,
}

impl MoeConfig {
    /// Kimi-K3-0.40B MoE configuration.
    ///
    /// `moe_intermediate_size` is set to 1024 as a placeholder (matches the
    /// hidden dim) — Phase 5 loader will overwrite with the actual value from
    /// the safetensors config.
    pub fn kimi_k3_0_40b() -> Self {
        Self {
            num_experts: 8,
            num_shared_experts: 1,
            num_experts_per_token: 2,
            moe_intermediate_size: 1024,
            hidden_size: 1024,
            use_sigmoid_router: true,
            renormalize: true,
        }
    }

    /// `N_r` — number of routed experts.
    #[inline]
    pub fn n_routed(&self) -> usize {
        self.num_experts
    }

    /// `K_r` — top-K experts per token.
    #[inline]
    pub fn k_routed(&self) -> usize {
        self.num_experts_per_token
    }

    /// `d_ffn` — SwiGLU intermediate dim.
    #[inline]
    pub fn d_ffn(&self) -> usize {
        self.moe_intermediate_size
    }

    /// `d` — hidden dim.
    #[inline]
    pub fn d(&self) -> usize {
        self.hidden_size
    }
}

// ─── Weights ────────────────────────────────────────────────────────────────

/// Per-expert SwiGLU FFN weights (gate + up + down).
///
/// Layouts (all row-major `Vec<f32>`):
/// - `gate_proj`: `[d_ffn, d]` — `out[o] = dot(row_o, hidden)`
/// - `up_proj`:   `[d_ffn, d]`
/// - `down_proj`: `[d, d_ffn]` — projects the SwiGLU output back to hidden dim
#[derive(Clone, Debug)]
pub struct SwiGluExpertWeights {
    pub gate_proj: Vec<f32>, // [d_ffn, d]
    pub up_proj: Vec<f32>,   // [d_ffn, d]
    pub down_proj: Vec<f32>, // [d, d_ffn]
}

/// MoE layer weight matrices.
///
/// Naming follows DeepSeek-V3 / Kimi-K3 conventions. See Research 328 §6 for
/// the safetensors tensor-name mapping (Phase 5 loader concern).
#[derive(Clone, Debug)]
pub struct MoeWeights {
    /// Router centroid matrix `[N_r, d]` — row `e` is expert `e`'s centroid
    /// `e_e`. The router logit for expert `e` is `dot(e_e, h)`.
    pub router_weight: Vec<f32>, // [N_r, d]
    /// noaux_tc per-expert bias `b_e` `[N_r]`. Added to the sigmoid score for
    /// top-K SELECTION ONLY — does NOT participate in renormalization.
    /// (Research 328 §3.3 — the load-bearing detail.)
    pub e_score_correction_bias: Vec<f32>, // [N_r]
    /// Routed expert FFN weights, length `N_r`.
    pub experts: Vec<SwiGluExpertWeights>, // [N_r]
    /// Shared expert FFN weights, length `N_s` (always on, no gating).
    pub shared_experts: Vec<SwiGluExpertWeights>, // [N_s]
}

impl MoeWeights {
    /// Test-only constructor: fills all weights with deterministic pseudo-
    /// random values from `seed`. Mirrors `MlaWeights::random` in katgpt-attn.
    ///
    /// The RNG is a simple xorshift — not cryptographic, just deterministic
    /// for reproducible G1 tests.
    pub fn random(config: &MoeConfig, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let d = config.d();
        let d_ffn = config.d_ffn();
        let n_r = config.n_routed();
        let n_s = config.num_shared_experts;

        let router_weight = (0..n_r * d).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
        // Bias range: small ±0.5 — large enough to flip top-K selection in tests.
        let e_score_correction_bias =
            (0..n_r).map(|_| rng.next_f32() * 1.0 - 0.5).collect();
        let experts = (0..n_r)
            .map(|_| SwiGluExpertWeights::random(&mut rng, d, d_ffn))
            .collect();
        let shared_experts = (0..n_s)
            .map(|_| SwiGluExpertWeights::random(&mut rng, d, d_ffn))
            .collect();

        Self {
            router_weight,
            e_score_correction_bias,
            experts,
            shared_experts,
        }
    }
}

impl SwiGluExpertWeights {
    /// Test-only random fill (same RNG as `MoeWeights::random`).
    pub fn random(rng: &mut Rng, d: usize, d_ffn: usize) -> Self {
        let gate_proj = (0..d_ffn * d).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
        let up_proj = (0..d_ffn * d).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
        let down_proj = (0..d * d_ffn).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

// ─── Scratch ────────────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for `moe_forward_token`.
///
/// Allocated once at startup; reused across tokens. All hot-path state lives
/// here — `moe_forward_token` itself performs zero allocations (G4 gate).
pub struct MoeForwardScratch {
    /// Router logits `[N_r]` — `dot(e_e, h)` per expert.
    pub router_logits: Vec<f32>,
    /// Sigmoid scores `[N_r]` — `s[e] = sigmoid(logits[e])`.
    pub sigmoid_scores: Vec<f32>,
    /// Biased scores `[N_r]` — `s[e] + b_e` for top-K selection.
    pub biased_scores: Vec<f32>,
    /// Top-K selected expert indices `[K_r]`.
    pub topk_indices: Vec<usize>,
    /// Renormalized gating weights `[K_r]` (sum to 1 when `renormalize=true`).
    pub topk_weights: Vec<f32>,
    /// SwiGLU intermediate buffer `[d_ffn]` — gate + up projection result.
    /// Reused across experts (one expert at a time).
    pub expert_intermediate: Vec<f32>,
    /// SwiGLU up-projection buffer `[d_ffn]` — `up_proj · h`.
    /// Separate from `expert_intermediate` because `swiglu_inplace` needs both
    /// the gate buffer (which becomes the SiLU(gate)⊙up output) + the up
    /// buffer as inputs. Reused across experts.
    pub expert_up: Vec<f32>,
    /// Expert output buffer `[d]` — single-expert SwiGLU output before axpy
    /// into `hidden_out`.
    pub expert_output: Vec<f32>,
}

impl MoeForwardScratch {
    /// Allocate scratch sized for `config`. Call once at startup.
    pub fn new(config: &MoeConfig) -> Self {
        let n_r = config.n_routed();
        let k_r = config.k_routed();
        let d = config.d();
        let d_ffn = config.d_ffn();
        Self {
            router_logits: vec![0.0; n_r],
            sigmoid_scores: vec![0.0; n_r],
            biased_scores: vec![0.0; n_r],
            topk_indices: vec![0; k_r],
            topk_weights: vec![0.0; k_r],
            expert_intermediate: vec![0.0; d_ffn],
            expert_up: vec![0.0; d_ffn],
            expert_output: vec![0.0; d],
        }
    }
}

// ─── Forward ────────────────────────────────────────────────────────────────

/// Single-token MoE decode forward.
///
/// Writes the FFN output (shared expert + weighted sum of top-K routed experts)
/// into `hidden_out`. The caller is responsible for the residual add
/// (`h' = h + hidden_out`) — this function does NOT add the residual.
///
/// # Arguments
/// * `weights` — MoE layer weights (router + experts + shared + bias)
/// * `config` — MoE configuration
/// * `hidden_in` — input hidden state `[d]`
/// * `hidden_out` — output buffer `[d]`, overwritten with the FFN output
/// * `scratch` — pre-allocated scratch (see `MoeForwardScratch`)
///
/// # Allocation discipline (G4)
///
/// Zero allocations in this function. All intermediate state lives in
/// `scratch`, which is allocated once at startup.
pub fn moe_forward_token(
    weights: &MoeWeights,
    config: &MoeConfig,
    hidden_in: &[f32],
    hidden_out: &mut [f32],
    scratch: &mut MoeForwardScratch,
) {
    let n_r = config.n_routed();
    let k_r = config.k_routed();
    let d = config.d();
    let d_ffn = config.d_ffn();

    debug_assert_eq!(hidden_in.len(), d);
    debug_assert_eq!(hidden_out.len(), d);
    debug_assert_eq!(weights.router_weight.len(), n_r * d);
    debug_assert_eq!(weights.e_score_correction_bias.len(), n_r);
    debug_assert_eq!(weights.experts.len(), n_r);
    debug_assert!(!weights.shared_experts.is_empty());
    debug_assert_eq!(scratch.router_logits.len(), n_r);
    debug_assert_eq!(scratch.sigmoid_scores.len(), n_r);
    debug_assert_eq!(scratch.biased_scores.len(), n_r);
    debug_assert_eq!(scratch.topk_indices.len(), k_r);
    debug_assert_eq!(scratch.topk_weights.len(), k_r);
    debug_assert_eq!(scratch.expert_intermediate.len(), d_ffn);
    debug_assert_eq!(scratch.expert_up.len(), d_ffn);
    debug_assert_eq!(scratch.expert_output.len(), d);

    // ── 1. Router logits: `logits[e] = dot(e_e, h)` per expert ───────────
    simd_matmul_rows(
        &mut scratch.router_logits[..n_r],
        &weights.router_weight,
        hidden_in,
        n_r,
        d,
    );

    // ── 2. Sigmoid scores (independent per expert, NEVER softmax) ─────────
    for e in 0..n_r {
        scratch.sigmoid_scores[e] = sigmoid(scratch.router_logits[e]);
    }

    // ── 3. noaux_tc biased scores: `biased[e] = s[e] + b_e` ──────────────
    for e in 0..n_r {
        scratch.biased_scores[e] = scratch.sigmoid_scores[e] + weights.e_score_correction_bias[e];
    }

    // ── 4. Top-K selection by BIASED score ───────────────────────────────
    select_topk_indices(
        &scratch.biased_scores[..n_r],
        k_r,
        &mut scratch.topk_indices[..k_r],
    );

    // ── 5. Renormalize selected experts' RAW scores (NOT biased!) ────────
    //    Per Research 328 §3.3: the bias is selection-only.
    let mut sum = 0.0f32;
    for k in 0..k_r {
        let idx = scratch.topk_indices[k];
        sum += scratch.sigmoid_scores[idx];
    }
    // Numerical guard (Research 328 §7.3): floor at f32::MIN_POSITIVE;
    // if absurdly small, fall back to uniform 1/K_r.
    if sum < 1.0e-20 {
        let uniform = 1.0 / (k_r as f32);
        for k in 0..k_r {
            scratch.topk_weights[k] = uniform;
        }
    } else if config.renormalize {
        let inv = 1.0 / sum;
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            scratch.topk_weights[k] = scratch.sigmoid_scores[idx] * inv;
        }
    } else {
        // moe_renormalize=false: use raw sigmoid scores as weights.
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            scratch.topk_weights[k] = scratch.sigmoid_scores[idx];
        }
    }

    // ── 6. Shared expert (always on) — write directly into hidden_out ────
    //    The shared expert's output becomes the base; routed experts axpy on top.
    let shared = &weights.shared_experts[0];
    swiglu_expert_forward(
        shared,
        hidden_in,
        &mut scratch.expert_intermediate,
        &mut scratch.expert_up,
        hidden_out,
        d,
        d_ffn,
    );
    // If N_s > 1, accumulate remaining shared experts (Kimi-K3-0.40B has N_s=1).
    for s in 1..weights.shared_experts.len() {
        let shared = &weights.shared_experts[s];
        swiglu_expert_forward(
            shared,
            hidden_in,
            &mut scratch.expert_intermediate,
            &mut scratch.expert_up,
            &mut scratch.expert_output,
            d,
            d_ffn,
        );
        // hidden_out += expert_output (axpy into the shared-expert accumulator)
        for (ho, eo) in hidden_out.iter_mut().zip(scratch.expert_output.iter()).take(d) {
            *ho += *eo;
        }
    }

    // ── 7. Routed experts (weighted by renormalized g) ───────────────────
    for k in 0..k_r {
        let idx = scratch.topk_indices[k];
        let w = scratch.topk_weights[k];
        let expert = &weights.experts[idx];
        swiglu_expert_forward(
            expert,
            hidden_in,
            &mut scratch.expert_intermediate,
            &mut scratch.expert_up,
            &mut scratch.expert_output,
            d,
            d_ffn,
        );
        // hidden_out += w * expert_output
        for (ho, eo) in hidden_out.iter_mut().zip(scratch.expert_output.iter()).take(d) {
            *ho += w * *eo;
        }
    }
}

/// Per-expert SwiGLU FFN forward.
///
/// Computes `out = down_proj · (SiLU(gate_proj · h) ⊙ up_proj · h)` where
/// `SiLU(x) = x · sigmoid(x)`. The gate projection lands in `intermediate`
/// (caller-allocated scratch `[d_ffn]`); the up projection lands in `up_buf`
/// (caller-allocated scratch `[d_ffn]`); the down projection lands in `out`
/// (caller-allocated `[d]`).
///
/// Uses the SIMD-optimized `swiglu_inplace` from katgpt-types. Allocation-free.
#[inline]
fn swiglu_expert_forward(
    expert: &SwiGluExpertWeights,
    hidden_in: &[f32],
    intermediate: &mut [f32],
    up_buf: &mut [f32],
    out: &mut [f32],
    d: usize,
    d_ffn: usize,
) {
    debug_assert_eq!(expert.gate_proj.len(), d_ffn * d);
    debug_assert_eq!(expert.up_proj.len(), d_ffn * d);
    debug_assert_eq!(expert.down_proj.len(), d * d_ffn);
    debug_assert_eq!(intermediate.len(), d_ffn);
    debug_assert_eq!(up_buf.len(), d_ffn);
    debug_assert_eq!(out.len(), d);

    // gate_proj · h → intermediate (becomes SiLU(gate·h) ⊙ up·h after swiglu)
    simd_matmul_rows(intermediate, &expert.gate_proj, hidden_in, d_ffn, d);

    // up_proj · h → up_buf
    simd_matmul_rows(up_buf, &expert.up_proj, hidden_in, d_ffn, d);

    // intermediate = SiLU(intermediate) ⊙ up_buf  (in-place SwiGLU)
    swiglu_inplace(intermediate, up_buf);

    // down_proj · intermediate → out
    simd_matmul_rows(out, &expert.down_proj, intermediate, d, d_ffn);
}

// ─── Top-K selection ────────────────────────────────────────────────────────

/// Partial-selection top-K: picks the indices of the K largest values in
/// `scores`, in DESCENDING order of score.
///
/// Writes the selected indices into `out_idx[..k]`. Alloc-free.
///
/// Implementation: insertion-sort the first K elements, then scan the rest
/// replacing the current minimum when a larger value is found. O(n·k) — for
/// Kimi-K3-0.40B (n=8, k=2) this is 16 comparisons, faster than a full sort.
/// For larger n the `katgpt-attn::dash_attn::block_topk::argtopk_with_scratch`
/// SIMD primitive would be preferred, but the dep isn't worth pulling for
/// this small-n case.
fn select_topk_indices(scores: &[f32], k: usize, out_idx: &mut [usize]) {
    debug_assert!(k <= scores.len());
    debug_assert_eq!(out_idx.len(), k);

    if k == 0 {
        return;
    }

    // Seed with the first k indices, sorted descending by score.
    out_idx[0] = 0;
    for i in 1..k {
        let idx = i;
        let val = scores[idx];
        // Insertion-sort idx into out_idx[0..i] (which is sorted desc).
        let mut j = i;
        while j > 0 && scores[out_idx[j - 1]] < val {
            out_idx[j] = out_idx[j - 1];
            j -= 1;
        }
        out_idx[j] = idx;
    }

    // Scan the rest; replace the current minimum (last slot) when larger.
    for i in k..scores.len() {
        let val = scores[i];
        // The minimum of the current top-K is at out_idx[k-1] (descending sort).
        if val > scores[out_idx[k - 1]] {
            // Replace the minimum + bubble it up to its sorted position.
            out_idx[k - 1] = i;
            let mut j = k - 1;
            while j > 0 && scores[out_idx[j - 1]] < scores[out_idx[j]] {
                out_idx.swap(j - 1, j);
                j -= 1;
            }
        }
    }
}

// ─── Test RNG (deterministic xorshift) ──────────────────────────────────────

/// Deterministic xorshift RNG for G1 tests. Mirrors `MlaWeights::random`.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the degenerate all-zero state.
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64 — deterministic, fast, good enough for G1 test inputs.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f32 in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits for the mantissa (f32 has 23 explicit bits + implicit 1).
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny config for fast tests: 4 experts, 1 shared, K=2, d=8, d_ffn=16.
    fn tiny_config() -> MoeConfig {
        MoeConfig {
            num_experts: 4,
            num_shared_experts: 1,
            num_experts_per_token: 2,
            moe_intermediate_size: 16,
            hidden_size: 8,
            use_sigmoid_router: true,
            renormalize: true,
        }
    }

    #[test]
    fn test_select_topk_descending_order() {
        let scores = [0.5, 0.9, 0.1, 0.7, 0.3];
        let mut idx = [0usize; 3];
        select_topk_indices(&scores, 3, &mut idx);
        // Top-3 by score: 0.9 (idx 1), 0.7 (idx 3), 0.5 (idx 0).
        assert_eq!(idx, [1, 3, 0]);
    }

    #[test]
    fn test_select_topk_handles_ties() {
        // Ties resolved by insertion order (stable).
        let scores = [0.5, 0.5, 0.5, 0.5];
        let mut idx = [0usize; 2];
        select_topk_indices(&scores, 2, &mut idx);
        // First two slots win on ties (>= comparison keeps earlier idx).
        assert_eq!(idx, [0, 1]);
    }

    #[test]
    fn test_moe_forward_runs_without_panic() {
        let config = tiny_config();
        let weights = MoeWeights::random(&config, 42);
        let mut scratch = MoeForwardScratch::new(&config);
        let hidden_in = vec![0.1; config.d()];
        let mut hidden_out = vec![0.0; config.d()];
        moe_forward_token(&weights, &config, &hidden_in, &mut hidden_out, &mut scratch);
        // Sanity: output is finite + non-zero (SwiGLU of random weights is non-zero).
        assert!(hidden_out.iter().all(|v| v.is_finite()));
        assert!(hidden_out.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn test_moe_forward_kimi_k3_dims() {
        // Smoke test: full Kimi-K3-0.40B dims run without panic.
        let config = MoeConfig::kimi_k3_0_40b();
        let weights = MoeWeights::random(&config, 123);
        let mut scratch = MoeForwardScratch::new(&config);
        let hidden_in = vec![0.05; config.d()];
        let mut hidden_out = vec![0.0; config.d()];
        moe_forward_token(&weights, &config, &hidden_in, &mut hidden_out, &mut scratch);
        assert!(hidden_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_renormalization_weights_sum_to_one() {
        let config = tiny_config();
        let mut weights = MoeWeights::random(&config, 7);
        // Zero out shared expert so it doesn't influence the weight check.
        for se in &mut weights.shared_experts {
            for v in &mut se.gate_proj {
                *v = 0.0;
            }
            for v in &mut se.up_proj {
                *v = 0.0;
            }
            for v in &mut se.down_proj {
                *v = 0.0;
            }
        }
        let mut scratch = MoeForwardScratch::new(&config);
        let hidden_in = vec![0.2; config.d()];
        let mut hidden_out = vec![0.0; config.d()];
        moe_forward_token(&weights, &config, &hidden_in, &mut hidden_out, &mut scratch);
        // After forward, scratch.topk_weights holds the renormalized g values.
        let sum: f32 = scratch.topk_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "renormalized weights must sum to 1, got {}",
            sum
        );
    }
}
