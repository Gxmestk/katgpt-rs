//! KDA analytic backward pass (Issue 389 T4 implementation).
//!
//! Implements the analytic gradient of `kda_forward_token` w.r.t. all
//! trainable parameters (`KdaWeights`) and the input hidden state `h`.
//! Derived in `riir-train/.issues/389_kda_gpu_backward_ssm_research.md` T2
//! by manual differentiation of the forward.
//!
//! # Scope
//!
//! Single-token backward (the BPTT building block). Given:
//! - `dL/doutput` — upstream gradient `[hidden_size]`
//! - `dL/dS_t` — gradient flowing in from the *next* BPTT step (init 0 at t=L)
//!
//! Produces:
//! - `dL/dθ` for every weight θ in `KdaWeights` (accumulated into `KdaGradients`)
//! - `dL/dh` — gradient w.r.t. the input hidden state `[hidden_size]`
//! - `dL/dS_{t-1}` — gradient to propagate to the *previous* BPTT step
//!
//! The full BPTT-over-a-sequence loop is Plan 318 Phase C C5 work; this module
//! provides the per-token primitive that the loop composes.
//!
//! # Why this lives here
//!
//! katgpt-rs is modelless-by-mandate (no training at runtime), BUT this module
//! is the **CPU reference** for the GPU backward (C5). It belongs alongside the
//! forward for two reasons: (1) the CPU primitive stays public so riir-train can
//! consume it; (2) the finite-difference gradient check (T4) needs it in-tree.
//! Production inference never calls this — it's gated behind `kda_linear`
//! (same as the forward) and only consumed by riir-train-gpu.

use crate::gdn2::kda_forward::{KdaConfig, KdaForwardScratch, KdaLayerCache, KdaWeights};
use crate::gdn2::kda_forward::kda_forward_token;
use katgpt_core::simd::{simd_outer_product_acc, simd_sum_sq};

// ─── Saved activations ──────────────────────────────────────────────────────

/// Forward-pass saved activations needed by the backward.
///
/// Populated by [`kda_forward_token_with_saved`]. All tensors are owned
/// snapshots taken AFTER the forward completes (the forward mutates its
/// scratch in-place; we clone the values it leaves behind + reconstruct
/// per-head intermediates from the post-forward state).
#[derive(Clone)]
pub struct KdaSavedActivations {
    /// Input hidden state `h`. `[hidden_size]`.
    pub h: Vec<f32>,
    /// `z_q = W^q · h` (pre-conv). `[proj]`.
    pub z_q: Vec<f32>,
    /// `z_k = W^k · h` (pre-conv). `[proj]`.
    pub z_k: Vec<f32>,
    /// `z_v = W^v · h` (pre-conv). `[proj]`.
    pub z_v: Vec<f32>,
    /// `z_q_conv` post-SiLU (= the conv output after SiLU). `[proj]`.
    pub z_q_conv: Vec<f32>,
    /// `z_k_conv` post-SiLU. `[proj]`.
    pub z_k_conv: Vec<f32>,
    /// `z_v_conv` post-SiLU. `[proj]`.
    pub z_v_conv: Vec<f32>,
    /// `f_a_hidden = W^{f_a} · h`. `[dk]`.
    pub f_a_hidden: Vec<f32>,
    /// `g_raw = W^{f_b} · f_a_hidden`. `[proj]`.
    pub g_raw: Vec<f32>,
    /// `beta_pre = W^β · h`. `[n_heads]`.
    pub beta_pre: Vec<f32>,
    /// `g_out = W^g · h`. `[proj]`.
    pub g_out: Vec<f32>,
    /// Post-norm+gate output (= o_concat). `[proj]`.
    pub o_concat: Vec<f32>,
    /// Per-head activations (indexed `[head]`).
    pub heads: Vec<KdaHeadActivations>,
    /// ShortConv ring buffer snapshot for q (BEFORE the forward step). `[proj*ks]`.
    pub conv_buf_q: Vec<f32>,
    /// ShortConv ring buffer snapshot for k (BEFORE the forward step). `[proj*ks]`.
    pub conv_buf_k: Vec<f32>,
    /// ShortConv ring buffer snapshot for v (BEFORE the forward step). `[proj*ks]`.
    pub conv_buf_v: Vec<f32>,
    /// Ring buffer write index BEFORE the forward step.
    pub conv_buf_idx: usize,
}

/// Per-head forward activations.
#[derive(Clone)]
pub struct KdaHeadActivations {
    /// `S_{t-1}` — incoming state (before decay+update). `[dk*dk]`.
    pub s_prev: Vec<f32>,
    /// `S' = S_{t-1} · diag(a)` — post-decay. `[dk*dk]`.
    pub s_prime: Vec<f32>,
    /// `S''` — post-update (= outgoing state S_t). `[dk*dk]`.
    pub s_post: Vec<f32>,
    /// `q_h` — L2-normed + scaled query. `[dk]`.
    pub q_normed: Vec<f32>,
    /// `k_h` — L2-normed key. `[dk]`.
    pub k_normed: Vec<f32>,
    /// `v_h` — value (= z_v_conv head slice). `[dk]`.
    pub v: Vec<f32>,
    /// `decay[i] = max(exp(gk[i]), alpha_eps)`. `[dk]`.
    pub a_decay: Vec<f32>,
    /// `gk[i] = -alpha_head · softplus(g_plus[i])`. `[dk]`.
    pub gk: Vec<f32>,
    /// `g_plus[i] = g_raw[off+i] + dt_bias[off+i]`. `[dk]`.
    pub g_plus: Vec<f32>,
    /// `delta[j] = beta_h · v_h[j] − r[j]`. `[dk]`.
    pub delta: Vec<f32>,
    /// `out_raw[j]` — pre-RMSNorm readout output. `[dk]`.
    pub out_raw: Vec<f32>,
    /// `alpha_head = exp(A_log[h])`.
    pub alpha_head: f32,
    /// `beta_h = sigmoid(beta_pre[h])`.
    pub beta_h: f32,
    /// `1 / sqrt(mean(out_raw²) + rms_eps)`.
    pub inv_rms: f32,
}

// ─── Gradient accumulator ───────────────────────────────────────────────────

/// Gradient accumulator mirroring `KdaWeights`.
///
/// All fields are accumulated (`+=`) by `kda_backward_token`. Initialize with
/// [`KdaGradients::zeros_like`] before the backward loop.
#[derive(Clone)]
pub struct KdaGradients {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub q_conv_weight: Vec<f32>,
    pub k_conv_weight: Vec<f32>,
    pub v_conv_weight: Vec<f32>,
    pub a_log: Vec<f32>,
    pub f_a_proj: Vec<f32>,
    pub f_b_proj: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub beta_proj: Vec<f32>,
    pub g_proj: Vec<f32>,
    pub o_norm_weight: Vec<f32>,
    pub o_proj: Vec<f32>,
}

impl KdaGradients {
    /// Allocate zeroed gradients matching the given weights' shapes.
    pub fn zeros_like(w: &KdaWeights) -> Self {
        Self {
            q_proj: vec![0.0; w.q_proj.len()],
            k_proj: vec![0.0; w.k_proj.len()],
            v_proj: vec![0.0; w.v_proj.len()],
            q_conv_weight: vec![0.0; w.q_conv_weight.len()],
            k_conv_weight: vec![0.0; w.k_conv_weight.len()],
            v_conv_weight: vec![0.0; w.v_conv_weight.len()],
            a_log: vec![0.0; w.a_log.len()],
            f_a_proj: vec![0.0; w.f_a_proj.len()],
            f_b_proj: vec![0.0; w.f_b_proj.len()],
            dt_bias: vec![0.0; w.dt_bias.len()],
            beta_proj: vec![0.0; w.beta_proj.len()],
            g_proj: vec![0.0; w.g_proj.len()],
            o_norm_weight: vec![0.0; w.o_norm_weight.len()],
            o_proj: vec![0.0; w.o_proj.len()],
        }
    }
}

// ─── Forward with saved activations ─────────────────────────────────────────

/// Run the KDA forward + capture all activations needed for the backward.
///
/// Produces the same output as [`kda_forward_token`]; additionally snapshots
/// all intermediates the backward needs. Returns `(output, saved)` where
/// `output` is an owned `Vec<f32>` (length `hidden_size`).
///
/// The output is owned (not a borrow) because building `saved` requires
/// immutable reads of the forward's scratch after it has been mutably
/// borrowed — a lifetime conflict an owned return resolves.
pub fn kda_forward_token_with_saved(
    config: &KdaConfig,
    weights: &KdaWeights,
    cache: &mut KdaLayerCache,
    scratch: &mut KdaForwardScratch,
    h: &[f32],
) -> (Vec<f32>, KdaSavedActivations) {
    let d = config.hidden_size;
    let dk = config.head_dim;
    let n_h = config.n_heads;
    let proj = dk * n_h;
    let scale = config.q_scale();

    // Snapshot conv ring buffers + state BEFORE the forward mutates them.
    let conv_buf_q = cache.q_conv.buf.clone();
    let conv_buf_k = cache.k_conv.buf.clone();
    let conv_buf_v = cache.v_conv.buf.clone();
    let conv_buf_idx = cache.q_conv.buf_idx;
    let s_prev_per_head: Vec<Vec<f32>> =
        cache.heads.iter().map(|hd| hd.s.clone()).collect();

    // Run the stock forward (fills scratch + mutates cache in-place).
    let output_ref = kda_forward_token(config, weights, cache, scratch, h);
    let output: Vec<f32> = output_ref.to_vec();

    // Reconstruct per-head activations from the now-filled scratch + saved S_{t-1}.
    let mut head_acts = Vec::with_capacity(n_h);
    for head in 0..n_h {
        let off = head * dk;
        let s_prev = &s_prev_per_head[head];

        // q_normed: L2Norm(z_q_conv slice) then scale. Recompute from saved z_q_conv.
        let mut q_normed = vec![0.0f32; dk];
        q_normed.copy_from_slice(&scratch.z_q_conv[off..off + dk]);
        l2_normalize_eps_kda(&mut q_normed);
        for i in 0..dk {
            q_normed[i] *= scale;
        }

        // k_normed: L2Norm(z_k_conv slice), no scale.
        let mut k_normed = vec![0.0f32; dk];
        k_normed.copy_from_slice(&scratch.z_k_conv[off..off + dk]);
        l2_normalize_eps_kda(&mut k_normed);

        // v: identity (slice of z_v_conv).
        let v = scratch.z_v_conv[off..off + dk].to_vec();

        // Decay derivation.
        let alpha_head = weights.a_log[head].exp();
        let mut gk = vec![0.0f32; dk];
        let mut g_plus = vec![0.0f32; dk];
        let mut a_decay = vec![0.0f32; dk];
        for i in 0..dk {
            let gp = scratch.g_raw[off + i] + weights.dt_bias[off + i];
            g_plus[i] = gp;
            let g = -alpha_head * softplus(gp);
            gk[i] = g;
            let a = g.exp();
            a_decay[i] = if a < config.alpha_eps { config.alpha_eps } else { a };
        }

        // S' = decay(S_{t-1}).
        let mut s_prime = vec![0.0f32; dk * dk];
        for i in 0..dk {
            for j in 0..dk {
                s_prime[i * dk + j] = s_prev[i * dk + j] * a_decay[i];
            }
        }

        // S'' = post-forward state (= cache.heads[head].s after the forward).
        let s_post = cache.heads[head].s.clone();

        // beta_h.
        let beta_h = scratch.beta[head];

        // r, delta.
        let mut r = vec![0.0f32; dk];
        for j in 0..dk {
            for i in 0..dk {
                r[j] += s_prime[i * dk + j] * beta_h * k_normed[i];
            }
        }
        let delta: Vec<f32> = (0..dk).map(|j| beta_h * v[j] - r[j]).collect();

        // out_raw: readout from S'' (before norm). = Σ_i S''[i,j] · q_normed[i].
        let out_raw: Vec<f32> = (0..dk)
            .map(|j| {
                let mut acc = 0.0f32;
                for i in 0..dk {
                    acc += s_post[i * dk + j] * q_normed[i];
                }
                acc
            })
            .collect();

        // inv_rms.
        let sum_sq = simd_sum_sq(&out_raw, dk);
        let mean_sq = sum_sq / dk as f32;
        let inv_rms = 1.0 / (mean_sq + config.rms_eps).sqrt();

        head_acts.push(KdaHeadActivations {
            s_prev: s_prev.clone(),
            s_prime,
            s_post,
            q_normed,
            k_normed,
            v,
            a_decay,
            gk,
            g_plus,
            delta,
            out_raw,
            alpha_head,
            beta_h,
            inv_rms,
        });
    }

    let saved = KdaSavedActivations {
        h: h.to_vec(),
        z_q: scratch.z_q.clone(),
        z_k: scratch.z_k.clone(),
        z_v: scratch.z_v.clone(),
        z_q_conv: scratch.z_q_conv.clone(),
        z_k_conv: scratch.z_k_conv.clone(),
        z_v_conv: scratch.z_v_conv.clone(),
        f_a_hidden: scratch.f_a_hidden.clone(),
        g_raw: scratch.g_raw.clone(),
        beta_pre: scratch.beta_pre.clone(),
        g_out: scratch.g_out.clone(),
        o_concat: scratch.o_concat.clone(),
        heads: head_acts,
        conv_buf_q,
        conv_buf_k,
        conv_buf_v,
        conv_buf_idx,
    };

    let _ = (proj, d);
    (output, saved)
}

// ─── Backward ───────────────────────────────────────────────────────────────

/// Analytic backward of `kda_forward_token` for a single token.
///
/// Consumes the [`KdaSavedActivations`] (from [`kda_forward_token_with_saved`])
/// + upstream `d_output` (= dL/doutput, `[hidden_size]`) + incoming state grad
/// `ds_next` (= dL/dS_t from the future BPTT step, `[n_heads][dk*dk]`).
///
/// Produces:
/// - `grads` — weight gradients (accumulated `+=`).
/// - `dh_out` — dL/dh (`[hidden_size]`, **overwritten**).
/// - `ds_prev_out` — dL/dS_{t-1} (`[n_heads][dk*dk]`, **overwritten**).
///
/// `dh_scratch` and `ds_prev_scratch` are caller-allocated work buffers (sized
/// `[hidden_size]` and `[n_heads * dk * dk]` respectively). The caller threads
/// `ds_prev_out` from step t into `ds_next` of step t-1 for BPTT.
pub fn kda_backward_token(
    config: &KdaConfig,
    weights: &KdaWeights,
    saved: &KdaSavedActivations,
    d_output: &[f32],
    dh_out: &mut [f32],
    grads: &mut KdaGradients,
    ds_next: &[Vec<f32>],
    ds_prev_out: &mut [Vec<f32>],
) {
    let d = config.hidden_size;
    let dk = config.head_dim;
    let n_h = config.n_heads;
    let proj = dk * n_h;
    let scale = config.q_scale();

    debug_assert_eq!(d_output.len(), d);
    debug_assert_eq!(dh_out.len(), d);
    debug_assert_eq!(ds_next.len(), n_h);
    debug_assert_eq!(ds_prev_out.len(), n_h);

    dh_out.fill(0.0);

    // ── Step 5 backward: output projection ─────────────────────────────────
    // output = W^o · o_concat.
    // grads.o_proj += outer(d_output, o_concat)
    // do_concat = W^oᵀ · d_output
    simd_outer_product_acc(&mut grads.o_proj, d_output, &saved.o_concat, d, proj);
    let mut do_concat = vec![0.0f32; proj];
    transpose_matvec(&mut do_concat, &weights.o_proj, d_output, d, proj);

    // Per-head work buffers (reused across heads).
    let mut dout_raw = vec![0.0f32; dk];
    let mut ds_post = vec![0.0f32; dk * dk];
    let mut ds_prime = vec![0.0f32; dk * dk];
    let mut dq_normed = vec![0.0f32; dk];
    let mut dk_normed = vec![0.0f32; dk];
    let mut dv_h = vec![0.0f32; dk];
    let mut ddelta = vec![0.0f32; dk];
    let mut dr = vec![0.0f32; dk];
    let mut derase_b = vec![0.0f32; dk];
    let mut da_decay = vec![0.0f32; dk];
    let mut dgk = vec![0.0f32; dk];
    let mut dg_plus = vec![0.0f32; dk];
    let mut dg_out_h = vec![0.0f32; dk];
    let mut dz_q_conv = vec![0.0f32; proj];
    let mut dz_k_conv = vec![0.0f32; proj];
    let mut dz_v_conv = vec![0.0f32; proj];
    let mut dg_raw = vec![0.0f32; proj];
    let mut dbeta_pre = vec![0.0f32; n_h];

    for head in 0..n_h {
        let off = head * dk;
        let ha = &saved.heads[head];

        // ── Step 4 backward: FusedRMSNormGated ────────────────────────────
        let do_concat_h = &do_concat[off..off + dk];
        let inv_rms = ha.inv_rms;
        let gamma = &weights.o_norm_weight;
        let mut dy = vec![0.0f32; dk];
        for i in 0..dk {
            let sig_gout = sigmoid(saved.g_out[off + i]);
            let d_oc_i = do_concat_h[i];
            let y_i = ha.out_raw[i] * inv_rms;
            dy[i] = d_oc_i * gamma[i] * sig_gout;
            grads.o_norm_weight[i] += d_oc_i * y_i * sig_gout;
            dg_out_h[i] = d_oc_i * y_i * gamma[i] * sig_gout * (1.0 - sig_gout);
        }
        // RMSNorm backward: dL/dout_raw_j = dy_j·inv_rms − out_raw_j · Σ_i(dy_i·out_raw_i) / (dk · rms³)
        let rms = 1.0 / inv_rms;
        let rms_cubed = rms * rms * rms;
        let sum_dy_out: f32 = (0..dk).map(|i| dy[i] * ha.out_raw[i]).sum();
        let correction = sum_dy_out / (dk as f32 * rms_cubed);
        for j in 0..dk {
            dout_raw[j] = dy[j] * inv_rms - ha.out_raw[j] * correction;
        }

        // ── Step 3 backward: gdn2_recurrent_step ──────────────────────────
        // ds_post = ds_next (incoming from future) + readout contribution.
        for idx in 0..dk * dk {
            ds_post[idx] = ds_next[head][idx];
        }
        // Readout: out[j] = Σ_i S''[i,j] · q_h[i]
        for i in 0..dk {
            let qi = ha.q_normed[i];
            let mut dq_i = 0.0f32;
            for j in 0..dk {
                ds_post[i * dk + j] += dout_raw[j] * qi;
                dq_i += dout_raw[j] * ha.s_post[i * dk + j];
            }
            dq_normed[i] = dq_i;
        }
        // Update: S''[i,j] = S'[i,j] + k_h[i] · delta[j]
        for i in 0..dk {
            let mut dk_i = 0.0f32;
            for j in 0..dk {
                ds_prime[i * dk + j] = ds_post[i * dk + j]; // passthrough
                dk_i += ds_post[i * dk + j] * ha.delta[j];
            }
            dk_normed[i] = dk_i;
        }
        for j in 0..dk {
            let mut dd_j = 0.0f32;
            for i in 0..dk {
                dd_j += ds_post[i * dk + j] * ha.k_normed[i];
            }
            ddelta[j] = dd_j;
        }
        // delta: delta[j] = beta_h · v_h[j] − r[j]
        let mut dbeta_from_delta = 0.0f32;
        for j in 0..dk {
            dv_h[j] = ddelta[j] * ha.beta_h;
            dr[j] = -ddelta[j];
            dbeta_from_delta += ddelta[j] * ha.v[j];
        }
        // Read: r[j] = Σ_i S'[i,j] · beta_h · k_h[i]  (erase_b = beta_h broadcast)
        let mut dbeta_from_read = 0.0f32;
        for i in 0..dk {
            let mut dk_from_read = 0.0f32;
            for j in 0..dk {
                ds_prime[i * dk + j] += dr[j] * ha.beta_h * ha.k_normed[i];
                derase_b[i] += dr[j] * ha.s_prime[i * dk + j] * ha.k_normed[i]; // = dL/derase_b[i]
                dk_from_read += dr[j] * ha.s_prime[i * dk + j] * ha.beta_h;
                dbeta_from_read += dr[j] * ha.s_prime[i * dk + j] * ha.k_normed[i];
            }
            dk_normed[i] += dk_from_read;
        }
        // Decay: S'[i,j] = S_{t-1}[i,j] · a[i]
        for i in 0..dk {
            let mut da_i = 0.0f32;
            for j in 0..dk {
                ds_prev_out[head][i * dk + j] = ds_prime[i * dk + j] * ha.a_decay[i];
                da_i += ds_prime[i * dk + j] * ha.s_prev[i * dk + j];
            }
            da_decay[i] = da_i;
        }

        // ── Step 2 backward: gates + decay derivation ─────────────────────
        // dbeta_h total = dbeta_from_delta + Σ_i derase_b[i]
        //   (derase_b[i] above already accumulated dr[j]·S'[i,j]·k_h[i]; summing
        //    over i gives the erase_b broadcast reduction. But dbeta_from_read
        //    double-counts the same thing — let's reconcile.)
        // Actually dbeta_from_read = Σ_i derase_b[i] (since erase_b[i]=beta_h).
        // So dbeta_h_total = dbeta_from_delta + dbeta_from_read.
        let dbeta_h_total = dbeta_from_delta + dbeta_from_read;
        dbeta_pre[head] = dbeta_h_total * ha.beta_h * (1.0 - ha.beta_h);

        // a[i] = max(exp(gk[i]), alpha_eps) — relu-clamp
        let mut dalpha_head = 0.0f32;
        for i in 0..dk {
            let exp_gk = ha.gk[i].exp();
            let mask = if ha.a_decay[i] > config.alpha_eps { 1.0 } else { 0.0 };
            dgk[i] = da_decay[i] * exp_gk * mask;
            dalpha_head += dgk[i] * (-softplus(ha.g_plus[i]));
            dg_plus[i] = dgk[i] * (-ha.alpha_head * sigmoid(ha.g_plus[i]));
        }
        grads.a_log[head] += dalpha_head * ha.alpha_head;

        // g_plus → g_raw + dt_bias
        for i in 0..dk {
            dg_raw[off + i] = dg_plus[i];
            grads.dt_bias[off + i] += dg_plus[i];
        }

        // ── L2Norm backward for q and k ───────────────────────────────────
        // Forward (l2_normalize_eps_kda): inv = 1/(||x|| + 1e-6), x_norm = x · inv.
        // q also scales by `scale` after norm.
        // dL/dx_j = inv · (dL/dy_j − x_j · inv² · Σ_i dL/dy_i · x_i)
        // For q: dL/dy_j (unscaled) = dq_normed[j] · scale (backward through scale).

        // q path:
        let z_q_slice = &saved.z_q_conv[off..off + dk];
        let norm_q = l2_norm_value(z_q_slice);
        let inv_q = 1.0 / (norm_q + 1e-6);
        let inv_q_sq = inv_q * inv_q;
        let mut sum_dq_z = 0.0f32;
        for i in 0..dk {
            sum_dq_z += dq_normed[i] * scale * z_q_slice[i];
        }
        for i in 0..dk {
            let d_unscaled = dq_normed[i] * scale;
            dz_q_conv[off + i] = inv_q * (d_unscaled - z_q_slice[i] * sum_dq_z * inv_q_sq);
        }

        // k path (no scale):
        let z_k_slice = &saved.z_k_conv[off..off + dk];
        let norm_k = l2_norm_value(z_k_slice);
        let inv_k = 1.0 / (norm_k + 1e-6);
        let inv_k_sq = inv_k * inv_k;
        let mut sum_dk_z = 0.0f32;
        for i in 0..dk {
            sum_dk_z += dk_normed[i] * z_k_slice[i];
        }
        for i in 0..dk {
            dz_k_conv[off + i] = inv_k * (dk_normed[i] - z_k_slice[i] * sum_dk_z * inv_k_sq);
        }

        // v path (identity):
        for i in 0..dk {
            dz_v_conv[off + i] = dv_h[i];
        }

        // dg_out scatter (for W^g backward below)
        for _i in 0..dk {
            // dg_out_h already holds the head slice; we'll use it in the projection backward
        }
    }

    // ── Step 1 backward: ShortConv + SiLU ──────────────────────────────────
    // Forward: z_q_conv[c] = silu( Σ_k weight[c,k] · buf_q[c, slot(k)] )
    // The forward saved z_q_conv as POST-SiLU. We need pre-SiLU for silu'.
    // We recover pre-SiLU by inverting silu numerically (Newton, ~5 iters).
    // Then: dL/d(conv_out) = dL/dz_q_conv · silu'(conv_out)
    //       dL/dweight[c,k] += dL/d(conv_out[c]) · buf_q[c, slot(k)]
    //       dL/dz_q[c] = dL/d(conv_out[c]) · weight[c,0]  (current-token only)
    //
    // IMPORTANT: the saved conv_buf_* snapshots were taken BEFORE the forward
    // wrote the current input. The conv forward reads the buffer AFTER writing
    // the current input at buf_idx. We must patch the buffer: write the current
    // input (saved.z_q for q) at conv_buf_idx before using it for weight grads.
    let mut conv_buf_q_patched = saved.conv_buf_q.clone();
    let mut conv_buf_k_patched = saved.conv_buf_k.clone();
    let mut conv_buf_v_patched = saved.conv_buf_v.clone();
    let ks = config.conv_kernel_size;
    for c in 0..proj {
        conv_buf_q_patched[c * ks + saved.conv_buf_idx] = saved.z_q[c];
        conv_buf_k_patched[c * ks + saved.conv_buf_idx] = saved.z_k[c];
        conv_buf_v_patched[c * ks + saved.conv_buf_idx] = saved.z_v[c];
    }

    backward_conv_silu(
        &saved.z_q_conv,
        &mut dz_q_conv,
        &conv_buf_q_patched,
        weights.q_conv_weight.as_slice(),
        saved.conv_buf_idx,
        config.conv_kernel_size,
        proj,
        &mut grads.q_conv_weight,
    );
    backward_conv_silu(
        &saved.z_k_conv,
        &mut dz_k_conv,
        &conv_buf_k_patched,
        weights.k_conv_weight.as_slice(),
        saved.conv_buf_idx,
        config.conv_kernel_size,
        proj,
        &mut grads.k_conv_weight,
    );
    backward_conv_silu(
        &saved.z_v_conv,
        &mut dz_v_conv,
        &conv_buf_v_patched,
        weights.v_conv_weight.as_slice(),
        saved.conv_buf_idx,
        config.conv_kernel_size,
        proj,
        &mut grads.v_conv_weight,
    );

    // ── Step 0 backward: projections ──────────────────────────────────────
    // Each: y = W · h.  dL/dW += outer(dL/dy, h).  dL/dh += Wᵀ · dL/dy.
    proj_backward(&mut grads.q_proj, dh_out, &dz_q_conv, &saved.h, &weights.q_proj, proj, d);
    proj_backward(&mut grads.k_proj, dh_out, &dz_k_conv, &saved.h, &weights.k_proj, proj, d);
    proj_backward(&mut grads.v_proj, dh_out, &dz_v_conv, &saved.h, &weights.v_proj, proj, d);

    // g_out: dg_out assembled from per-head dg_out_h values.
    let mut dg_out_full = vec![0.0f32; proj];
    for head in 0..n_h {
        let off = head * dk;
        // Recompute dg_out_h for this head (we didn't save it across iterations).
        // Actually we need to redo the RMSNorm step 4 backward to get dg_out...
        // Simpler: the forward gate path is g_out = W^g · h (proj-wide), and
        // dg_out comes from the sigmoid'(g_out) · y · gamma in step 4 backward.
        // We computed dg_out_h inside the loop but didn't save it. Let's recompute
        // the per-head dg_out here by re-reading do_concat.
        let ha = &saved.heads[head];
        let inv_rms = ha.inv_rms;
        let gamma = &weights.o_norm_weight;
        for i in 0..dk {
            let sig_gout = sigmoid(saved.g_out[off + i]);
            let d_oc_i = do_concat[off + i];
            let y_i = ha.out_raw[i] * inv_rms;
            dg_out_full[off + i] = d_oc_i * y_i * gamma[i] * sig_gout * (1.0 - sig_gout);
        }
    }
    proj_backward(&mut grads.g_proj, dh_out, &dg_out_full, &saved.h, &weights.g_proj, proj, d);
    proj_backward(&mut grads.beta_proj, dh_out, &dbeta_pre, &saved.h, &weights.beta_proj, n_h, d);

    // Two-stage gate: f_a_hid = W^{f_a} · h, then g_raw = W^{f_b} · f_a_hid.
    let mut df_a_hidden = vec![0.0f32; dk];
    proj_backward(&mut grads.f_b_proj, &mut df_a_hidden, &dg_raw, &saved.f_a_hidden, &weights.f_b_proj, proj, dk);
    proj_backward(&mut grads.f_a_proj, dh_out, &df_a_hidden, &saved.h, &weights.f_a_proj, dk, d);
}

// ─── Backward helpers ───────────────────────────────────────────────────────

/** Backward through SiLU + ShortConv1D for one of q/k/v.
 *
 * Given `dz_conv` (grad w.r.t. post-SiLU output) + the saved post-SiLU values
 * `z_conv_saved`, recovers the pre-SiLU conv output (via Newton inversion of
 * silu), then:
 * - accumulates `dL/dconv_weight` into `grad_conv_weight`.
 * - writes `dL/dz_preconv` (current-token contribution only) into `dz_conv`
 *   (overwriting the input — the caller reuses it as the projection-backward
 *   input).
 */
#[allow(clippy::too_many_arguments)]
fn backward_conv_silu(
    z_conv_saved: &[f32],      // post-SiLU values [proj]
    dz_conv: &mut [f32],        // IN: grad w.r.t. post-SiLU; OUT: dL/dz_preconv [proj]
    conv_buf: &[f32],          // ring buffer snapshot (pre-forward) [proj*ks]
    conv_weight: &[f32],       // [proj*ks]
    conv_buf_idx: usize,       // ring index (pre-forward)
    kernel_size: usize,
    n_channels: usize,
    grad_conv_weight: &mut [f32], // accumulated
) {
    let ks = kernel_size;
    let newest_slot = conv_buf_idx; // the forward wrote the current input here

    // Phase 1: read dz_conv + compute dconv_out (SiLU backward).
    // We must finish ALL reads before any writes since dz_conv is &mut.
    let mut dconv_out = vec![0.0f32; n_channels];
    for c in 0..n_channels {
        let y = z_conv_saved[c]; // post-SiLU
        let x = silu_inverse(y); // pre-SiLU (= conv output)
        let sig = sigmoid(x);
        let silu_deriv = sig * (1.0 + x * (1.0 - sig));
        dconv_out[c] = dz_conv[c] * silu_deriv;
    }

    // Phase 2: ShortConv backward + write dL/dz_preconv into dz_conv.
    // Forward: conv_out[c] = Σ_k weight[c,k] · buf[c, slot(k)]
    //   where slot(k) = (newest_slot + ks − k) % ks.
    // dL/dweight[c,k] += dconv_out[c] · buf[c, slot(k)]
    // dL/d(current input)[c] = dconv_out[c] · weight[c, 0]   (k=0 = current token)
    let mut dz_pre = vec![0.0f32; n_channels];
    for c in 0..n_channels {
        let w_off = c * ks;
        let b_off = c * ks;
        let dc = dconv_out[c];
        for k in 0..ks {
            let slot = (newest_slot + ks - k) % ks;
            grad_conv_weight[w_off + k] += dc * conv_buf[b_off + slot];
        }
        dz_pre[c] = dc * conv_weight[w_off]; // k=0 tap (current token)
    }

    // Overwrite dz_conv with the pre-conv grad.
    dz_conv.copy_from_slice(&dz_pre);
}

/// Backward through a linear projection `y = W · h` (simd_matmul_rows convention).
/// `grad_W[r,c] += grad_y[r] · h[c]`. `grad_h[c] += Σ_r W[r,c] · grad_y[r]`.
fn proj_backward(
    grad_w: &mut [f32],
    grad_h: &mut [f32],
    grad_y: &[f32],
    h: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
) {
    // grad_W += outer(grad_y, h)
    for r in 0..rows {
        let gy = grad_y[r];
        let row_off = r * cols;
        for c in 0..cols {
            grad_w[row_off + c] += gy * h[c];
        }
    }
    // grad_h += Wᵀ · grad_y
    for c in 0..cols {
        let mut acc = 0.0f32;
        for r in 0..rows {
            acc += w[r * cols + c] * grad_y[r];
        }
        grad_h[c] += acc;
    }
}

/// `out = Wᵀ · v` for row-major `[rows, cols]` W. Overwrites `out`.
fn transpose_matvec(out: &mut [f32], w: &[f32], v: &[f32], rows: usize, cols: usize) {
    for c in 0..cols {
        let mut acc = 0.0f32;
        for r in 0..rows {
            acc += w[r * cols + c] * v[r];
        }
        out[c] = acc;
    }
}

/// L2-normalize with eps outside the sqrt (mirrors kda_forward.rs).
fn l2_normalize_eps_kda(x: &mut [f32]) {
    let norm_sq = simd_sum_sq(x, x.len());
    let inv_norm = 1.0 / (norm_sq.sqrt() + 1e-6);
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

/// Compute `sqrt(Σ x_i²)` (no eps).
fn l2_norm_value(x: &[f32]) -> f32 {
    simd_sum_sq(x, x.len()).sqrt()
}

// ─── Math helpers (mirror kda_forward.rs) ───────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    crate::gdn2::kernel::sigmoid(x)
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x >= 20.0 {
        x
    } else if x >= 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Inverse SiLU via Newton's method: solve `silu(x) = x · sigmoid(x) = y` for x.
/// Converges in ~5-8 iterations for |y| < ~10.
fn silu_inverse(y: f32) -> f32 {
    let mut x = y; // good initial guess for small |y|; silu(x) ≈ x for large x.
    for _ in 0..10 {
        let sig = sigmoid(x);
        let f = x * sig - y;
        let fp = sig * (1.0 + x * (1.0 - sig));
        if fp.abs() < 1e-12 {
            break;
        }
        let delta = f / fp;
        x -= delta;
        if delta.abs() < 1e-8 {
            break;
        }
        if !x.is_finite() {
            return y;
        }
    }
    x
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn2::kda_forward::{KdaForwardScratch, KdaLayerCache};

    fn small_config() -> KdaConfig {
        KdaConfig {
            head_dim: 8,
            n_heads: 2,
            hidden_size: 16,
            conv_kernel_size: 4,
            alpha_eps: 1e-5,
            rms_eps: 1e-5,
        }
    }

    /// Smoke test: forward-with-saved produces the same output as the stock forward.
    #[test]
    fn forward_with_saved_matches_stock() {
        let config = small_config();
        let weights = KdaWeights::random(&config, 42);
        let h: Vec<f32> = (0..config.hidden_size).map(|i| (i as f32) * 0.1 - 0.5).collect();

        let mut cache1 = KdaLayerCache::new(&config);
        let mut scratch1 = KdaForwardScratch::new(&config);
        let out1 = kda_forward_token(&config, &weights, &mut cache1, &mut scratch1, &h);

        let mut cache2 = KdaLayerCache::new(&config);
        let mut scratch2 = KdaForwardScratch::new(&config);
        let (out2, _saved) = kda_forward_token_with_saved(&config, &weights, &mut cache2, &mut scratch2, &h);

        for i in 0..config.hidden_size {
            assert!((out1[i] - out2[i]).abs() < 1e-5, "output mismatch at {}: {} vs {}", i, out1[i], out2[i]);
        }
    }

    /// Backward smoke test: runs without panicking, produces finite gradients.
    #[test]
    fn backward_smoke_finite() {
        let config = small_config();
        let weights = KdaWeights::random(&config, 42);
        let h: Vec<f32> = (0..config.hidden_size).map(|i| (i as f32) * 0.1 - 0.5).collect();

        let mut cache = KdaLayerCache::new(&config);
        let mut fwd_scratch = KdaForwardScratch::new(&config);
        let (_output, saved) = kda_forward_token_with_saved(&config, &weights, &mut cache, &mut fwd_scratch, &h);

        let d_output = vec![0.1f32; config.hidden_size];
        let mut dh = vec![0.0f32; config.hidden_size];
        let mut grads = KdaGradients::zeros_like(&weights);
        let ds_next: Vec<Vec<f32>> = (0..config.n_heads).map(|_| vec![0.0; config.head_dim * config.head_dim]).collect();
        let mut ds_prev: Vec<Vec<f32>> = (0..config.n_heads).map(|_| vec![0.0; config.head_dim * config.head_dim]).collect();

        kda_backward_token(&config, &weights, &saved, &d_output, &mut dh, &mut grads, &ds_next, &mut ds_prev);

        for &g in &dh {
            assert!(g.is_finite(), "non-finite dh: {}", g);
        }
        for &g in &grads.a_log {
            assert!(g.is_finite(), "non-finite a_log grad: {}", g);
        }
        for &g in grads.q_proj.iter().take(20) {
            assert!(g.is_finite(), "non-finite q_proj grad: {}", g);
        }
    }

    /// Verify S'' reconstruction: s_prime + k⊗delta should equal s_post.
    #[test]
    fn saved_activations_s_post_consistent() {
        let config = small_config();
        let weights = KdaWeights::random(&config, 42);
        let h: Vec<f32> = (0..config.hidden_size).map(|i| (i as f32) * 0.1 - 0.5).collect();

        let mut cache = KdaLayerCache::new(&config);
        let mut fwd_scratch = KdaForwardScratch::new(&config);
        let (_output, saved) = kda_forward_token_with_saved(&config, &weights, &mut cache, &mut fwd_scratch, &h);

        let dk = config.head_dim;
        for head in 0..config.n_heads {
            let ha = &saved.heads[head];
            for i in 0..dk {
                for j in 0..dk {
                    let reconstructed = ha.s_prime[i * dk + j] + ha.k_normed[i] * ha.delta[j];
                    let actual = ha.s_post[i * dk + j];
                    assert!(
                        (reconstructed - actual).abs() < 1e-4,
                        "head {} S'' mismatch at [{},{}]: reconstructed {} vs actual {}",
                        head, i, j, reconstructed, actual
                    );
                }
            }
        }
    }
}
