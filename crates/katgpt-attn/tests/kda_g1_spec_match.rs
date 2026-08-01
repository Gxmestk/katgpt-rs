//! G1 spec-match tests for KDA (Kimi Delta Attention).
//!
//! These tests verify the f32 `kda_forward_token` implementation against an
//! independent f64 reference written directly from the Kimi Linear equations
//! (Research 329 §2, arxiv 2510.26692 equations 1 + 10).
//!
//! The f64 reference is written WITHOUT reusing the f32 impl's code — it's a
//! from-scratch implementation of the same math. If both agree within f32
//! precision, we have confidence the f32 impl correctly encodes the equations.
//!
//! The load-bearing detail tested here is the **β-broadcast into the gdn2
//! kernel's `erase_b`** — the f32 impl calls `gdn2_recurrent_step` with
//! `b = β·ones` + `w_val = β` + `gate_config = Kda`, which Research 329 §4.2
//! proves is algebraically identical to KDA eq 1. The f64 reference computes
//! eq 1 directly via the expanded form
//! (`S_new = Diag(α)·S_old + β·k⊗(v − S_decayed^T·k)`).
//!
//! The TRUE correctness gate is Phase 6 (logits match real PyTorch weights).
//! This G1 gate catches transcription errors in our own code before we get to
//! Phase 6.

#![cfg(feature = "kda_linear")]

use katgpt_attn::gdn2::kda_forward::{
    KdaConfig, KdaForwardScratch, KdaLayerCache, KdaWeights, kda_forward_token,
};

// ─── f64 weight mirror ─────────────────────────────────────────────────────

/// f64 mirror of `KdaWeights`. Used to feed the f64 reference.
struct KdaWeightsF64 {
    qk_proj: Vec<f64>,
    v_proj: Vec<f64>,
    beta_proj: Vec<f64>,
    alpha_down: Vec<f64>,
    alpha_up: Vec<f64>,
    qk_conv_weight: Vec<f64>,
    v_conv_weight: Vec<f64>,
    gate_down: Vec<f64>,
    gate_up: Vec<f64>,
    head_norm_weight: Vec<f64>,
    o_proj: Vec<f64>,
}

fn weights_to_f64(w: &KdaWeights) -> KdaWeightsF64 {
    KdaWeightsF64 {
        qk_proj: w.qk_proj.iter().map(|&x| x as f64).collect(),
        v_proj: w.v_proj.iter().map(|&x| x as f64).collect(),
        beta_proj: w.beta_proj.iter().map(|&x| x as f64).collect(),
        alpha_down: w.alpha_down.iter().map(|&x| x as f64).collect(),
        alpha_up: w.alpha_up.iter().map(|&x| x as f64).collect(),
        qk_conv_weight: w.qk_conv_weight.iter().map(|&x| x as f64).collect(),
        v_conv_weight: w.v_conv_weight.iter().map(|&x| x as f64).collect(),
        gate_down: w.gate_down.iter().map(|&x| x as f64).collect(),
        gate_up: w.gate_up.iter().map(|&x| x as f64).collect(),
        head_norm_weight: w.head_norm_weight.iter().map(|&x| x as f64).collect(),
        o_proj: w.o_proj.iter().map(|&x| x as f64).collect(),
    }
}

// ─── f64 primitives ────────────────────────────────────────────────────────

fn matmul_f64(out: &mut [f64], w: &[f64], x: &[f64], rows: usize, cols: usize) {
    for r in 0..rows {
        let mut sum = 0.0;
        for c in 0..cols {
            sum += w[r * cols + c] * x[c];
        }
        out[r] = sum;
    }
}

fn dot_f64(a: &[f64], b: &[f64], len: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn swish_f64(x: f64) -> f64 {
    x * sigmoid_f64(x)
}

/// L2-normalize a vector in-place. Returns the original norm.
fn l2_normalize_f64(v: &mut [f64]) -> f64 {
    let mut sum_sq = 0.0;
    for &x in v.iter() {
        sum_sq += x * x;
    }
    let norm = sum_sq.sqrt();
    if norm > 1e-30 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    norm
}

/// RMSNorm: out = γ * x / sqrt(mean(x²) + eps). Matches the f32 impl's `rmsnorm_into`.
fn rmsnorm_f64(x: &[f64], gamma: &[f64], eps: f64, out: &mut [f64]) {
    let sum_sq = dot_f64(x, x, x.len());
    let mean_sq = sum_sq / x.len() as f64;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv_rms * gamma[i];
    }
}

// ─── f64 ShortConv1D ───────────────────────────────────────────────────────
// Independent reimplementation — does NOT share code with the f32 ShortConv1D.

struct ShortConvF64 {
    weight: Vec<f64>,
    buf: Vec<f64>,
    buf_idx: usize,
    n_channels: usize,
    kernel_size: usize,
}

impl ShortConvF64 {
    fn new(n_channels: usize, kernel_size: usize) -> Self {
        Self {
            weight: vec![0.0; n_channels * kernel_size],
            buf: vec![0.0; n_channels * kernel_size],
            buf_idx: 0,
            n_channels,
            kernel_size,
        }
    }

    fn forward(&mut self, x: &[f64], out: &mut [f64]) {
        let ks = self.kernel_size;
        let nc = self.n_channels;
        // Push input
        for (c, &xc) in x.iter().enumerate().take(nc) {
            self.buf[c * ks + self.buf_idx] = xc;
        }
        let new_buf_idx = (self.buf_idx + 1) % ks;
        let newest_slot = self.buf_idx;
        for (c, out_slot) in out.iter_mut().enumerate().take(nc) {
            let mut acc = 0.0;
            let w_off = c * ks;
            let b_off = c * ks;
            for k in 0..ks {
                let tap = self.weight[w_off + k];
                if tap != 0.0 {
                    let slot = (newest_slot + ks - k) % ks;
                    acc += tap * self.buf[b_off + slot];
                }
            }
            *out_slot = acc;
        }
        self.buf_idx = new_buf_idx;
    }
}

// ─── f64 KDA forward reference ─────────────────────────────────────────────

/// Independent f64 reference for the KDA forward pass.
///
/// Implements Kimi Linear eq 1 via the expanded form (Research 329 §2.1):
///   `S_decayed = Diag(α) · S_old`
///   `r = S_decayed^T · k`                (read with key k)
///   `S_new = S_decayed + β · k ⊗ (v − r)`  (delta-rule update)
///   `o = S_new^T · q`                     (readout)
///
/// This is the formula BEFORE the gdn2 kernel's β-broadcast transformation —
/// if the f32 impl's `b = β·ones` + `w_val = β` configuration is wrong, this
/// reference will diverge (catching the error).
fn kda_forward_f64_reference(
    config: &KdaConfig,
    weights: &KdaWeightsF64,
    tokens: &[Vec<f64>],
) -> Vec<f64> {
    let d = config.hidden_size;
    let dk = config.head_dim;
    let dv = config.head_dim;
    let n_h = config.n_heads;
    let qk_dim = dk * n_h;
    let r_a = config.alpha_rank;
    let r_g = config.gate_rank;
    let ks = config.conv_kernel_size;
    let alpha_eps = config.alpha_eps as f64;
    let rms_eps = config.rms_eps as f64;

    // Per-head state matrices S ∈ R^{dk × dv}, row-major.
    let mut state: Vec<Vec<f64>> = (0..n_h).map(|_| vec![0.0; dk * dv]).collect();
    // ShortConv ring buffers
    let mut qk_conv = ShortConvF64::new(qk_dim, ks);
    let mut v_conv = ShortConvF64::new(qk_dim, ks);
    qk_conv.weight.copy_from_slice(&weights.qk_conv_weight);
    v_conv.weight.copy_from_slice(&weights.v_conv_weight);

    let mut last_output = vec![0.0f64; d];

    for h in tokens {
        // ── Step 1: Projections ─────────────────────────────────────────────
        let mut z_qk = vec![0.0; qk_dim];
        let mut z_v = vec![0.0; qk_dim];
        matmul_f64(&mut z_qk, &weights.qk_proj, h, qk_dim, d);
        matmul_f64(&mut z_v, &weights.v_proj, h, qk_dim, d);

        let mut alpha_hidden = vec![0.0; r_a];
        let mut log_alpha_flat = vec![0.0; qk_dim];
        matmul_f64(&mut alpha_hidden, &weights.alpha_down, h, r_a, d);
        matmul_f64(&mut log_alpha_flat, &weights.alpha_up, &alpha_hidden, qk_dim, r_a);

        let mut beta_pre = vec![0.0; n_h];
        let mut beta = vec![0.0; n_h];
        matmul_f64(&mut beta_pre, &weights.beta_proj, h, n_h, d);
        for i in 0..n_h {
            beta[i] = sigmoid_f64(beta_pre[i]);
        }

        // ── Step 2: ShortConv + Swish ───────────────────────────────────────
        let mut z_qk_conv = vec![0.0; qk_dim];
        let mut z_v_conv = vec![0.0; qk_dim];
        qk_conv.forward(&z_qk, &mut z_qk_conv);
        v_conv.forward(&z_v, &mut z_v_conv);
        for i in 0..qk_dim {
            z_qk_conv[i] = swish_f64(z_qk_conv[i]);
            z_v_conv[i] = swish_f64(z_v_conv[i]);
        }

        // ── Step 3: Per-head recurrent step (KDA eq 1 via expanded form) ────
        let mut o_concat = vec![0.0; qk_dim];
        for head in 0..n_h {
            let off = head * dk;

            // q_h = k_h = L2Norm(z_qk_conv[h slice])
            let mut qk_h = vec![0.0; dk];
            qk_h.copy_from_slice(&z_qk_conv[off..off + dk]);
            l2_normalize_f64(&mut qk_h);

            // v_h = z_v_conv[h slice] (no norm)
            let v_h = &z_v_conv[off..off + dk];

            // α_h = exp(log_α).max(eps)
            let mut alpha_h = vec![0.0; dk];
            for i in 0..dk {
                let a = log_alpha_flat[off + i].exp();
                alpha_h[i] = if a < alpha_eps { alpha_eps } else { a };
            }

            let beta_h = beta[head];

            // ── KDA eq 1 expanded form ───────────────────────────────────────
            // S_decayed = Diag(α) · S_old  (scale each row i by α[i])
            // r = S_decayed^T · k          (matvec; r ∈ R^{dv})
            // S_new = S_decayed + β · k ⊗ (v − r)
            // o = S_new^T · q              (matvec; o ∈ R^{dv})
            let s = &mut state[head];

            // S_decayed + r computation fused.
            let mut s_decayed = vec![0.0; dk * dv];
            for i in 0..dk {
                let a = alpha_h[i];
                for j in 0..dv {
                    s_decayed[i * dv + j] = s[i * dv + j] * a;
                }
            }

            // r = S_decayed^T · k  (r[j] = Σ_i S_decayed[i,j] * k[i])
            let mut r = vec![0.0; dv];
            for j in 0..dv {
                let mut acc = 0.0;
                for i in 0..dk {
                    acc += s_decayed[i * dv + j] * qk_h[i];
                }
                r[j] = acc;
            }

            // delta = β · (v − r)
            let mut delta = vec![0.0; dv];
            for j in 0..dv {
                delta[j] = beta_h * (v_h[j] - r[j]);
            }

            // S_new = S_decayed + k ⊗ delta  (outer product accumulate)
            for i in 0..dk {
                let ki = qk_h[i];
                for j in 0..dv {
                    s[i * dv + j] = s_decayed[i * dv + j] + ki * delta[j];
                }
            }

            // o = S_new^T · q  (o[j] = Σ_i S_new[i,j] * q[i])
            let mut o_h = vec![0.0; dv];
            for j in 0..dv {
                let mut acc = 0.0;
                for i in 0..dk {
                    acc += s[i * dv + j] * qk_h[i];
                }
                o_h[j] = acc;
            }

            // Head-wise RMSNorm on o_h
            let mut o_h_norm = vec![0.0; dk];
            rmsnorm_f64(&o_h, &weights.head_norm_weight, rms_eps, &mut o_h_norm);

            // Copy into o_concat[head slice]
            for i in 0..dk {
                o_concat[head * dk + i] = o_h_norm[i];
            }
        }

        // ── Step 4: Output gate (low-rank) ──────────────────────────────────
        let mut gate_hidden = vec![0.0; r_g];
        let mut gate_pre = vec![0.0; qk_dim];
        let mut gate = vec![0.0; qk_dim];
        matmul_f64(&mut gate_hidden, &weights.gate_down, h, r_g, d);
        matmul_f64(&mut gate_pre, &weights.gate_up, &gate_hidden, qk_dim, r_g);
        for i in 0..qk_dim {
            gate[i] = sigmoid_f64(gate_pre[i]);
        }
        for i in 0..qk_dim {
            o_concat[i] *= gate[i];
        }

        // ── Step 5: Output projection ───────────────────────────────────────
        matmul_f64(&mut last_output, &weights.o_proj, &o_concat, d, qk_dim);
    }

    last_output
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn run_f32(config: &KdaConfig, weights: &KdaWeights, tokens: &[Vec<f32>]) -> Vec<f32> {
    let mut cache = KdaLayerCache::new(config);
    let mut scratch = KdaForwardScratch::new(config);
    let mut last_out = Vec::new();
    for h in tokens {
        let out = kda_forward_token(config, weights, &mut cache, &mut scratch, h);
        last_out = out.to_vec();
    }
    last_out
}

fn max_diff_f32_f64(a: &[f32], b: &[f64]) -> f32 {
    let mut max_d = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y as f32).abs();
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

// ─── Tests ─────────────────────────────────────────────────────────────────

fn small_config() -> KdaConfig {
    KdaConfig {
        head_dim: 8,
        n_heads: 2,
        hidden_size: 16,
        conv_kernel_size: 4,
        alpha_rank: 8,
        gate_rank: 8,
        alpha_eps: 1e-5,
        rms_eps: 1e-6,
    }
}

#[test]
fn g1_kda_matches_reference_small_dims() {
    let config = small_config();
    let weights = KdaWeights::random(&config, 42);
    let weights_f64 = weights_to_f64(&weights);

    let tokens_f32: Vec<Vec<f32>> = (0..3)
        .map(|t| {
            (0..config.hidden_size)
                .map(|i| ((i + t * 13) as f32).sin() * 0.1)
                .collect()
        })
        .collect();
    let tokens_f64: Vec<Vec<f64>> = tokens_f32
        .iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();

    let out_f32 = run_f32(&config, &weights, &tokens_f32);
    let out_f64 = kda_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_diff_f32_f64(&out_f32, &out_f64);
    eprintln!("g1_kda_small_dims: max_diff = {max_diff:.2e}");
    assert!(
        max_diff < 1e-4,
        "G1 FAIL small dims: max_diff = {max_diff:.2e} (tol 1e-4)"
    );
}

#[test]
fn g1_kda_matches_reference_kimi_k3_0_40b_dims() {
    // Full 0.40B KDA config. Slower but verifies real-scale math.
    let config = KdaConfig::kimi_k3_0_40b();
    let weights = KdaWeights::random(&config, 777);
    let weights_f64 = weights_to_f64(&weights);

    let tokens_f32: Vec<Vec<f32>> = (0..3)
        .map(|t| {
            (0..config.hidden_size)
                .map(|i| ((i + t * 13) as f32).sin() * 0.01)
                .collect()
        })
        .collect();
    let tokens_f64: Vec<Vec<f64>> = tokens_f32
        .iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();

    let out_f32 = run_f32(&config, &weights, &tokens_f32);
    let out_f64 = kda_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_diff_f32_f64(&out_f32, &out_f64);
    eprintln!("g1_kda_0.40B_dims: max_diff = {max_diff:.2e}");
    assert!(
        max_diff < 1e-3,
        "G1 FAIL 0.40B dims: max_diff = {max_diff:.2e} (tol 1e-3)"
    );
}

#[test]
fn g1_smoke_kimi_k3_0_40b_dims_finite() {
    // Quick finite-output smoke test at full 0.40B dims.
    let config = KdaConfig::kimi_k3_0_40b();
    let weights = KdaWeights::random(&config, 99);
    let tokens: Vec<Vec<f32>> = (0..3)
        .map(|t| vec![0.1f32 + t as f32 * 0.01; config.hidden_size])
        .collect();
    let out = run_f32(&config, &weights, &tokens);
    assert_eq!(out.len(), config.hidden_size);
    for &v in &out {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

/// The load-bearing test: prove that `erase_b = β·ones` + `w_val = β` in the
/// gdn2 kernel produces KDA eq 1 bit-identically to the expanded form.
///
/// This catches the canonical §4.2 error: if the f32 impl used `w_val = 1.0`
/// (not β) OR `erase_b = ones` (not β·ones), the math would diverge from eq 1.
/// The f64 reference uses the expanded form directly, so any misconfiguration
/// surfaces as a max_diff exceeding tolerance.
#[test]
fn g1_kda_beta_broadcast_correctness() {
    // Smaller-than-small config for tight tolerance + fast iteration.
    let config = KdaConfig {
        head_dim: 4,
        n_heads: 1,
        hidden_size: 8,
        conv_kernel_size: 2,
        alpha_rank: 4,
        gate_rank: 4,
        alpha_eps: 1e-5,
        rms_eps: 1e-6,
    };
    let weights = KdaWeights::random(&config, 123);
    let weights_f64 = weights_to_f64(&weights);

    let tokens_f32: Vec<Vec<f32>> = (0..5)
        .map(|t| {
            (0..config.hidden_size)
                .map(|i| ((i + t * 7) as f32).sin() * 0.1)
                .collect()
        })
        .collect();
    let tokens_f64: Vec<Vec<f64>> = tokens_f32
        .iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();

    let out_f32 = run_f32(&config, &weights, &tokens_f32);
    let out_f64 = kda_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_diff_f32_f64(&out_f32, &out_f64);
    eprintln!("g1_kda_beta_broadcast: max_diff = {max_diff:.2e}");
    // Tight tolerance at this small scale — the math should be near-bit-identical
    // modulo only f32 rounding in matvec accumulation order.
    assert!(
        max_diff < 1e-4,
        "G1 FAIL β-broadcast: max_diff = {max_diff:.2e} (tol 1e-4). \
         This indicates the gdn2_recurrent_step Kda gate config is NOT computing \
         eq 1 correctly — check erase_b = β·ones AND w_val = β."
    );
}

/// Causality test: token t's output depends only on tokens [0..=t].
///
/// Run forward on tokens [t0, t1, t2] → capture o2. Then run forward on
/// [t0, t1, t2, t3] → the first 3 outputs (state after t2) should be
/// bit-identical regardless of whether t3 is processed afterward. KDA's state
/// is causal — future tokens don't retroactively change past outputs.
#[test]
fn g1_kda_causality_state_does_not_depend_on_future() {
    let config = small_config();
    let weights = KdaWeights::random(&config, 7);

    let t0: Vec<f32> = (0..config.hidden_size).map(|i| (i as f32).sin() * 0.1).collect();
    let t1: Vec<f32> = (0..config.hidden_size)
        .map(|i| ((i + 13) as f32).sin() * 0.1)
        .collect();
    let t2: Vec<f32> = (0..config.hidden_size)
        .map(|i| ((i + 26) as f32).sin() * 0.1)
        .collect();
    let t3: Vec<f32> = (0..config.hidden_size)
        .map(|i| ((i + 39) as f32).sin() * 0.1)
        .collect();

    // Path A: process [t0, t1, t2] only — capture o2.
    let out_a = run_f32(&config, &weights, &[t0.clone(), t1.clone(), t2.clone()]);

    // Path B: process [t0, t1, t2, t3] — capture o2 (third output).
    let mut cache = KdaLayerCache::new(&config);
    let mut scratch = KdaForwardScratch::new(&config);
    let _o0 = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &t0).to_vec();
    let _o1 = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &t1).to_vec();
    let o2_b = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &t2).to_vec();
    let _o3 = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &t3).to_vec();

    let max_diff = max_diff_f32_f64(&out_a, &o2_b.iter().map(|&x| x as f64).collect::<Vec<_>>());
    eprintln!("g1_kda_causality: max_diff (o2 with vs without t3) = {max_diff:.2e}");
    // o2 in path A == o2 in path B (t3 has no retroactive effect).
    // Tolerance 0 because both paths use the same code path + same RNG-initialized weights.
    assert_eq!(
        out_a, o2_b,
        "KDA state must be causal: t3 must not change the output at t2"
    );
}
