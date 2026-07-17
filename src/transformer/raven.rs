use crate::types::{self};
use super::*;

/// Convert token ids to readable characters (a-z, _ for BOS).
pub fn tokens_to_string(tokens: &[usize]) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    // Pre-allocate the exact capacity: one char per token, avoiding
    // the repeated growth+realloc that String::collect performs.
    let mut out = String::with_capacity(tokens.len());
    for &t in tokens {
        out.push(match t {
            0..=25 => CHARS[t] as char,
            _ => '_',
        });
    }
    out
}

/// Sparse router: computes Top-K routing vector from raw logits (zero-alloc variant).
///
/// Implements: `r_t = Normalize(TopK(Sigmoid(raw_logits)))`
/// Unselected slots get 0.0 → completely frozen during update.
///
/// Uses pre-allocated buffers to avoid heap allocations on the hot path.
#[inline]
pub fn raven_compute_router_into(
    raw_logits: &[f32],
    top_k: usize,
    scored: &mut Vec<(usize, f32)>,
    r_t: &mut Vec<f32>,
) {
    let num_slots = raw_logits.len();
    let top_k = top_k.min(num_slots);

    // Negate logits in-place into r_t scratch buffer.
    // Replace scalar `r_t[i] = -x` loop with copy + SIMD scale: two passes but
    // vectorized, wins for num_slots >= 16 (Raven typically uses 16-64 slots).
    // simd_scale_inplace(x, -1.0) compiles to a single SIMD negate per chunk.
    r_t.resize(num_slots, 0.0);
    r_t[..num_slots].copy_from_slice(&raw_logits[..num_slots]);
    katgpt_core::simd::simd_scale_inplace(&mut r_t[..num_slots], -1.0);
    katgpt_core::simd::simd_exp_inplace(&mut r_t[..num_slots]);
    // r_t now holds exp(-x). Compute sigmoid(x) = 1/(1+exp(-x)) via SIMD:
    //   add_scalar(+1) → reciprocal → done. Replaces scalar 1/(1+e) per slot.
    katgpt_core::simd::simd_add_scalar_inplace(&mut r_t[..num_slots], 1.0);
    katgpt_core::simd::simd_reciprocal_inplace(&mut r_t[..num_slots]);
    // Write (index, sigmoid) pairs directly into pre-sized scored buffer
    // (avoids push reallocation). Index writes are sequential and trivially
    // auto-vectorizable by LLVM.
    scored.resize(num_slots, (0, 0.0));
    for (i, &sig) in r_t[..num_slots].iter().enumerate() {
        scored[i] = (i, sig);
    }

    // Partial sort: find Top-K by descending score (O(n) average)
    if top_k < num_slots {
        // total_cmp: eliminates the per-element NaN branch from partial_cmp.
        // Sigmoid outputs are always finite (bounded (0,1)), so total_cmp
        // matches partial_cmp exactly without the predicted-branch stall.
        scored.select_nth_unstable_by(num_slots - top_k, |a, b| a.1.total_cmp(&b.1));
    }

    // Fill r_t with zeros for final output
    r_t[..num_slots].fill(0.0);
    let mut sum = 0.0f32;

    // Keep only Top-K (the last top_k elements after partial sort are the largest)
    for (idx, score) in scored.iter().rev().take(top_k) {
        r_t[*idx] = *score;
        sum += *score;
    }

    // Normalize so selected slots sum to 1.0.
    // SIMD scale is branch-free and vectorized; replaces scalar `*v *= inv_sum` loop.
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        katgpt_core::simd::simd_scale_inplace(&mut r_t[..num_slots], inv_sum);
    }
}

/// Backward-compatible wrapper that allocates fresh buffers.
pub fn raven_compute_router(raw_logits: &[f32], top_k: usize) -> Vec<f32> {
    let n = raw_logits.len();
    let mut scored = Vec::with_capacity(n);
    let mut r_t = Vec::with_capacity(n);
    raven_compute_router_into(raw_logits, top_k, &mut scored, &mut r_t);
    r_t
}

/// Gated memory update: Raven Equation 18.
///
/// For each slot:
///   `decay = exp(forget_rate × r_t[slot])`
///   `H_new = decay × H_old + (1 - decay) × new_content`
///
/// When `r_t[slot] == 0`: `decay = exp(0) = 1.0` → `H_new = H_old` (FROZEN)
/// When `r_t[slot] > 0`: `decay < 1.0` → old content decays, new writes in
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn raven_update(
    keys: &mut [f32],
    values: &mut [f32],
    new_key: &[f32],
    new_value: &[f32],
    r_t: &[f32],
    forget_rate: f32,
    num_slots: usize,
    kv_dim: usize,
) {
    for (slot, &route) in r_t.iter().enumerate().take(num_slots) {
        let decay = (forget_rate * route).exp();
        let write = 1.0 - decay;
        let offset = slot * kv_dim;

        katgpt_core::simd::simd_fused_decay_write(
            &mut keys[offset..offset + kv_dim],
            decay,
            &new_key[..kv_dim],
            write,
        );
        katgpt_core::simd::simd_fused_decay_write(
            &mut values[offset..offset + kv_dim],
            decay,
            &new_value[..kv_dim],
            write,
        );
    }
}

/// Readout: attention over fixed slot memory.
/// `O(num_slots × kv_dim)` — constant regardless of sequence length.
/// Zero-alloc readout: computes attention-weighted slot values into pre-allocated buffers.
///
/// Fused 2-pass optimization over `raven_readout` (3-pass):
/// - Pass 1: Q·K^T dot products + find max
/// - Pass 2: exp(scores - max) + weighted value accumulation + normalize
///
/// Returns `&mut output[..kv_dim]` (borrowed from the provided output buffer).
#[inline]
pub fn raven_readout_into<'a>(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    num_slots: usize,
    kv_dim: usize,
    scores: &'a mut [f32],
    output: &'a mut [f32],
) -> &'a mut [f32] {
    debug_assert!(scores.len() >= num_slots);
    debug_assert!(output.len() >= kv_dim);

    // Pass 1: Q·K^T + find max
    let mut max_score = f32::NEG_INFINITY;
    for s in 0..num_slots {
        let k_off = s * kv_dim;
        let dot = katgpt_core::simd::simd_dot_f32(query, &keys[k_off..k_off + kv_dim], kv_dim);
        unsafe {
            *scores.get_unchecked_mut(s) = dot;
        }
        // Branch-free max reduction (single SIMD instruction).
        max_score = max_score.max(dot);
    }

    // Pass 2: fused exp + accumulate + normalize (SIMD batch)
    output[..kv_dim].fill(0.0);
    katgpt_core::simd::simd_add_scalar_inplace(&mut scores[..num_slots], -max_score);
    katgpt_core::simd::simd_exp_inplace(&mut scores[..num_slots]);
    let sum_exp = katgpt_core::simd::simd_sum_f32(&scores[..num_slots]);

    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        for s in 0..num_slots {
            let weight = unsafe { *scores.get_unchecked(s) * inv_sum };
            let v_off = s * kv_dim;
            katgpt_core::simd::simd_fused_scale_acc(
                &mut output[..kv_dim],
                &values[v_off..v_off + kv_dim],
                weight,
                kv_dim,
            );
        }
    }

    &mut output[..kv_dim]
}

/// Allocating wrapper for backward compatibility (tests, benchmark).
pub fn raven_readout(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    num_slots: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; num_slots];
    let mut output = vec![0.0f32; kv_dim];
    raven_readout_into(
        query,
        keys,
        values,
        num_slots,
        kv_dim,
        &mut scores,
        &mut output,
    );
    output
}

/// Forward pass using `RavenKVCache` instead of `MultiLayerKVCache`.
///
/// Identical computation to `forward()` except attention:
/// - Generates router logits from K projection (dummy: use K directly)
/// - Calls `raven_update()` instead of writing to flat KV array
/// - Calls `raven_readout()` instead of scanning all past positions
/// - Everything else (RMSNorm, MLP, residual, LM head) stays identical
pub fn forward_raven<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut RavenKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // Loop-invariant value hoisted outside the layer loop
    let scale = ctx.attn_scale;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    katgpt_core::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // Adaptive Depth Tier: cap layer count at inference time (Plan 284 T10).
    let max_layer = ctx
        .depth_tier
        .map_or(config.n_layer, |t| t.max_layers(config.n_layer));

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate().take(max_layer) {
        // layer_idx used by delta_routing cfg blocks below
        #[cfg(not(feature = "delta_routing"))]
        let _ = layer_idx;
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Raven: generate router logits from K (dummy projection)
        // For PoC: use first num_slots elements of K repeated as logits.
        // In production, this would be a learned linear projection: W_route × x_t
        // Reuse pre-allocated query buffer for router logits (zero-alloc)
        // Buffer is pre-sized in ForwardContext::new() to max(kv_dim, 64, num_slots).
        let num_slots = cache.num_slots;
        // Fast path: when num_slots <= kvd, just copy first num_slots K elements.
        // Avoids per-iteration modulo (slow on most ISAs).
        if num_slots <= kvd {
            ctx.raven_query_buf[..num_slots].copy_from_slice(&ctx.k[..num_slots]);
        } else {
            for (i, slot) in ctx.raven_query_buf[..num_slots].iter_mut().enumerate() {
                *slot = ctx.k[i % kvd];
            }
        }

        // Raven: compute sparse routing vector (zero-alloc via pre-allocated buffers)
        raven_compute_router_into(
            &ctx.raven_query_buf,
            cache.top_k,
            &mut cache.router_scored,
            &mut cache.router_r_t,
        );

        // Stack-allocated copy to avoid self-borrow (cache.keys vs cache.router_r_t)
        // num_slots is typically 16-64 floats — fits on stack
        let mut r_t = [0.0f32; 64];
        let copy_len = cache.router_r_t.len().min(64);
        r_t[..copy_len].copy_from_slice(&cache.router_r_t[..copy_len]);

        // Raven: gated update (only selected slots are modified)
        raven_update(
            &mut cache.keys,
            &mut cache.values,
            &ctx.k,
            &ctx.v,
            &r_t,
            cache.forget_rate,
            cache.num_slots,
            kvd,
        );

        // Raven: readout via attention over fixed slots (O(num_slots) not O(pos))
        ctx.attn_out[..n].fill(0.0);

        ctx.raven_query_buf[..kvd].fill(0.0);
        for h in 0..config.n_head {
            let q_off = h * hd;
            // Each head reads from the slot memory using its query slice
            let head_query = &ctx.q[q_off..q_off + hd];
            // Pad/reshape query to kv_dim for slot attention (reuse pre-allocated buffer)
            let kv_group = ctx.kv_group_lut[h] as usize;
            for (d, &hq) in head_query.iter().enumerate() {
                ctx.raven_query_buf[kv_group * hd + d] = hq * scale;
            }

            let slot_values = raven_readout_into(
                &ctx.raven_query_buf,
                &cache.keys,
                &cache.values,
                cache.num_slots,
                kvd,
                &mut cache.readout_scores,
                &mut cache.readout_output,
            );

            // Extract this head's attention output (single memcpy vs hd unsafe writes).
            ctx.attn_out[q_off..q_off + hd]
                .copy_from_slice(&slot_values[kv_group * hd..kv_group * hd + hd]);
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        #[cfg(feature = "gated_mlp")]
        {
            // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
            types::matmul(&mut ctx.hidden, &layer_weights.mlp_w1, &ctx.x, config.mlp_hidden, n);
            types::matmul(&mut ctx.hidden2, &layer_weights.mlp_w_up, &ctx.x, config.mlp_hidden, n);
            types::swiglu_inplace(&mut ctx.hidden, &ctx.hidden2);
        }
        #[cfg(not(feature = "gated_mlp"))]
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2 (W_down): sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        {
            let block_size = 4; // Default B=4
            let block_idx = layer_idx / block_size;
            let pos_in_block = layer_idx % block_size;

            // Compute delta: current x minus pre-layer residual (xr was saved after first rmsnorm)
            if block_idx < ctx.block_deltas.len() {
                katgpt_core::simd::simd_fused_sub_acc(
                    &mut ctx.block_deltas[block_idx][..n],
                    &ctx.x[..n],
                    &ctx.xr[..n],
                    n,
                );
            }

            // At block boundary: route accumulated deltas from all completed blocks
            if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                ctx.depth_route_blocks(
                    block_idx,
                    layer_idx,
                    &weights.delta_routing_query[layer_idx],
                    &weights.delta_routing_norm[layer_idx],
                    n,
                    weights,
                );
            }
        }
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head (uses matmul_parallel for large vocab)
    standard_lm_head(
        &mut ctx.logits,
        &ctx.x,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}
