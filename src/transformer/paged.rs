use super::*;
use crate::types::{self};

/// Forward pass using `PagedKVCache` instead of `MultiLayerKVCache`.
///
/// Identical computation to `forward()` but stores KV in paged memory,
/// enabling copy-on-write fork for DDTree branch exploration.
/// Builds a temporary flat KV buffer per layer for attention computation.
#[inline(always)]
pub fn forward_paged<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    paged_cache: &mut PagedKVCache,
    seq_idx: usize,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = crate::types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // Ensure pages allocated for this sequence up to pos
    paged_cache.ensure_pages(seq_idx, pos);

    // Flat KV cache for attention computation (pre-allocated, reused from ForwardContext)
    // Note: no initial fill(0.0) needed — the inner loop below reads every position
    // from the paged cache and overwrites the flat buffer for each layer.
    let t_n = pos + 1;
    let flat_kv_len = t_n * kvd;

    // Loop-invariant values hoisted outside the layer loop
    let scale = ctx.attn_scale;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    katgpt_core::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // Wall Attention: reset prefix sums at sequence start (Plan 173).
    #[cfg(feature = "wall_attention")]
    if pos == 0 {
        ctx.wall_prefix.reset();
    }

    // Adaptive Depth Tier: cap layer count at inference time (Plan 284 T10).
    let max_layer = ctx
        .depth_tier
        .map_or(config.n_layer, |t| t.max_layers(config.n_layer));

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate().take(max_layer) {
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Wall Attention: gate projection + prefix sum update + Q/K rescale (Plan 173).
        #[cfg(feature = "wall_attention")]
        if let Some(ref wall_cfg) = config.wall_config {
            let n_kv = config.n_kv_head;
            let hd = config.head_dim;
            for kv_h in 0..n_kv {
                let k_off = kv_h * hd;
                let w_g = &layer_weights.attn_wg[k_off..k_off + hd];
                let k_slice = &ctx.k[k_off..k_off + hd];
                ctx.wall_prefix.compute_gate_and_update(
                    layer_idx,
                    kv_h,
                    k_slice,
                    w_g,
                    wall_cfg.gate_bias,
                    wall_cfg.gate_max,
                );
            }
            ctx.wall_prefix
                .rescale_query(layer_idx, &mut ctx.q, &ctx.kv_group_lut, config.n_head);
            ctx.wall_prefix.rescale_key(layer_idx, &mut ctx.k);
        }

        // Write K,V to paged cache
        paged_cache.write_kv(layer_idx, seq_idx, pos, &ctx.k, &ctx.v);

        // Build flat KV from paged cache for attention
        {
            let flat_key = &mut ctx.paged_flat_key[..flat_kv_len];
            let flat_value = &mut ctx.paged_flat_value[..flat_kv_len];
            for t in 0..t_n {
                let k_slice = &mut flat_key[t * kvd..(t + 1) * kvd];
                let v_slice = &mut flat_value[t * kvd..(t + 1) * kvd];
                paged_cache.read_kv(layer_idx, seq_idx, t, k_slice, v_slice);
            }

            // Multi-head attention with GQA (reuse existing attention_head)
            for h in 0..config.n_head {
                let kv_group = ctx.kv_group_lut[h] as usize;
                unsafe {
                    attention_head(
                        &ctx.q,
                        flat_key,
                        flat_value,
                        &mut ctx.attn_out,
                        &mut ctx.scores,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        t_n,
                        scale,
                    );
                }
            }
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
            types::matmul(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx.hidden2,
                &layer_weights.mlp_w_up,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
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
