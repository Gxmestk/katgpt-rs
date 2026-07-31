use super::*;

// ---------------------------------------------------------------------------

/// Bidirectional prefill: process prompt tokens with full mutual attention.
///
/// For each transformer layer:
///   Phase A: Compute K/V for all prompt positions → store in KV cache
///   Phase B: For each position, attend to ALL prompt K/V (bidirectional)
///
/// Returns logits for the last prompt position (used to sample first gen token).
/// KV cache is populated as a side effect, shared with subsequent decode calls.
///
/// Zero-copy: no allocations. Reuses ForwardContext buffers per-position,
/// PrefillContext::hidden for multi-layer inter-layer state.
///
/// For RiM buffer slots (Plan 172): use `rim_extend_tokens()` to append buffer
/// tokens before calling this function. The logit readout will naturally come
/// from the last buffer position.
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill<'a>(
    ctx: &'a mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    let prompt_len = tokens.len().min(prefill.max_prompt_len);
    if prompt_len > 0 {
        cache.advance_pos(prompt_len - 1);
    }
    let n = config.n_embd;
    let kvd = crate::types::kv_dim(config);
    let hd = config.head_dim;
    let _n_kv = config.n_kv_head;

    assert!(prompt_len > 0, "prefill requires at least one token");
    assert!(
        prompt_len <= config.block_size,
        "prompt_len {prompt_len} exceeds block_size {}",
        config.block_size
    );

    // Initialize hidden states for multi-layer (single-layer computes on-the-fly)
    if config.n_layer > 1 {
        for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
            let tok_off = token * n;
            let pos_off = p * n;
            katgpt_core::simd::simd_add_into(
                &mut prefill.hidden[p * n..(p + 1) * n],
                &weights.wte[tok_off..tok_off + n],
                &weights.wpe[pos_off..pos_off + n],
            );
        }
    }

    // Wall Attention: reset prefix sums at prefill start (Plan 173).
    #[cfg(feature = "wall_attention")]
    if config.wall_config.is_some() {
        ctx.wall_prefix.reset();
    }

    // Adaptive Depth Tier: cap layer count at inference time (Plan 284 T10).
    let max_layer = ctx
        .depth_tier
        .map_or(config.n_layer, |t| t.max_layers(config.n_layer));

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate().take(max_layer) {
        let layer_cache = &mut cache.layers[layer_idx];

        // ── Phase A: Compute K/V for ALL positions → store in cache ──
        for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
            // Load hidden state
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                katgpt_core::simd::simd_add_into(
                    &mut ctx.x[..n],
                    &weights.wte[tok_off..tok_off + n],
                    &weights.wpe[pos_off..pos_off + n],
                );
            }

            // Pre-attention norm (matches forward_base exactly: double rmsnorm)
            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            // K/V projections
            crate::types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut prefill.lora_buf);
            }
            crate::types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
            #[cfg(feature = "domain_latent")]
            if layer_idx == config.n_layer / 2
                && let Some(dl) = domain_latent
            {
                katgpt_core::simd::simd_add_inplace(&mut ctx.k[..kvd], &dl.embedding[..kvd]);
                katgpt_core::simd::simd_add_inplace(&mut ctx.v[..kvd], &dl.embedding[..kvd]);
            }

            // Wall Attention: gate projection + prefix sum update + Q/K rescale (Plan 173).
            // Prefill processes positions sequentially, accumulating prefix sums per-layer.
            // K is rescaled before cache storage; Q is rescaled before Phase B reuse.
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
                ctx.wall_prefix.rescale_key(layer_idx, &mut ctx.k);
            }

            // Store K/V in cache
            let pos_off = p * kvd;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.k.as_ptr(),
                    layer_cache.key.as_mut_ptr().add(pos_off),
                    kvd,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.v.as_ptr(),
                    layer_cache.value.as_mut_ptr().add(pos_off),
                    kvd,
                );
            }

            // Q projection (fused: avoids redundant hidden load + rmsnorm in Phase B)
            crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Wall Attention: rescale Q with accumulated prefix sum (Plan 173).
            #[cfg(feature = "wall_attention")]
            if config.wall_config.is_some() {
                ctx.wall_prefix.rescale_query(
                    layer_idx,
                    &mut ctx.q,
                    &ctx.kv_group_lut,
                    config.n_head,
                );
            }

            // Store Q and xr for Phase B reuse
            let q_off = p * n;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.q.as_ptr(),
                    prefill.queries.as_mut_ptr().add(q_off),
                    n,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.xr.as_ptr(),
                    prefill.residuals.as_mut_ptr().add(q_off),
                    n,
                );
            }
        }

        // ── Phase B: Bidirectional attention for ALL positions ──
        // Loads pre-computed Q and xr from fused Phase A, skipping redundant
        // hidden state load + double rmsnorm + Q matmul per position.

        // Tiled attention: batch-compute all positions for large prompts (Plan 115)
        // Avoids O(N²) score matrix materialization when prompt_len >= 128
        #[cfg(feature = "tiled_attention")]
        let use_tiled = prompt_len >= 128;

        // Hoist constant scale outside per-position loop (Pattern 3: avoid recomputing unchanged values)
        let attn_scale = ctx.attn_scale;

        #[cfg(feature = "tiled_attention")]
        if use_tiled {
            let tiled_size = config.n_head * prompt_len * hd;
            // Repack Q: (position, head) → (head, position) contiguous layout
            for h in 0..config.n_head {
                for p in 0..prompt_len {
                    let src_off = p * n + h * hd;
                    let dst_off = h * prompt_len * hd + p * hd;
                    ctx.tiled_q[dst_off..dst_off + hd]
                        .copy_from_slice(&prefill.queries[src_off..src_off + hd]);
                }
            }
            // Repack K/V with GQA expansion: (position, kv_group) → (head, position)
            for h in 0..config.n_head {
                let kv_group = ctx.kv_group_lut[h] as usize;
                for p in 0..prompt_len {
                    let kv_src = p * kvd + kv_group * hd;
                    let dst_off = h * prompt_len * hd + p * hd;
                    ctx.tiled_k[dst_off..dst_off + hd]
                        .copy_from_slice(&layer_cache.key[kv_src..kv_src + hd]);
                    ctx.tiled_v[dst_off..dst_off + hd]
                        .copy_from_slice(&layer_cache.value[kv_src..kv_src + hd]);
                }
            }
            katgpt_core::tiled_attention_batched(
                &ctx.tiled_q[..tiled_size],
                &ctx.tiled_k[..tiled_size],
                &ctx.tiled_v[..tiled_size],
                &mut ctx.tiled_out[..tiled_size],
                1,
                config.n_head,
                prompt_len,
                hd,
            );
        }

        for p in 0..prompt_len {
            let q_off = p * n;

            // Load residual (xr) for output projection
            unsafe {
                std::ptr::copy_nonoverlapping(
                    prefill.residuals.as_ptr().add(q_off),
                    ctx.xr.as_mut_ptr(),
                    n,
                );
            }

            // ── Attention computation (tiled or per-head) ──
            ctx.attn_out[..n].fill(0.0);

            #[cfg(feature = "tiled_attention")]
            if use_tiled {
                // Unpack tiled output: (head, position) → attn_out for this position
                for h in 0..config.n_head {
                    let src_off = h * prompt_len * hd + p * hd;
                    let dst_off = h * hd;
                    ctx.attn_out[dst_off..dst_off + hd]
                        .copy_from_slice(&ctx.tiled_out[src_off..src_off + hd]);
                }
            } else {
                // Per-head attention for small prompts (below threshold)
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prefill.queries.as_ptr().add(q_off),
                        ctx.q.as_mut_ptr(),
                        n,
                    );
                }
                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h] as usize;
                    unsafe {
                        attention_head(
                            &ctx.q,
                            &layer_cache.key,
                            &layer_cache.value,
                            &mut ctx.attn_out,
                            &mut ctx.scores,
                            h * hd,
                            kv_group * hd,
                            kvd,
                            hd,
                            prompt_len,
                            attn_scale,
                        );
                    }
                }
            }

            #[cfg(not(feature = "tiled_attention"))]
            {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prefill.queries.as_ptr().add(q_off),
                        ctx.q.as_mut_ptr(),
                        n,
                    );
                }
                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h] as usize;
                    unsafe {
                        attention_head(
                            &ctx.q,
                            &layer_cache.key,
                            &layer_cache.value,
                            &mut ctx.attn_out,
                            &mut ctx.scores,
                            h * hd,
                            kv_group * hd,
                            kvd,
                            hd,
                            prompt_len,
                            attn_scale,
                        );
                    }
                }
            }

            // Output projection + residual
            crate::types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut prefill.lora_buf);
            }
            katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

            // MLP: residual → RMSNorm → MLP → residual
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);
            #[cfg(feature = "gated_mlp")]
            {
                // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
                crate::types::matmul(
                    &mut ctx.hidden,
                    &layer_weights.mlp_w1,
                    &ctx.x,
                    config.mlp_hidden,
                    n,
                );
                crate::types::matmul(
                    &mut ctx.hidden2,
                    &layer_weights.mlp_w_up,
                    &ctx.x,
                    config.mlp_hidden,
                    n,
                );
                crate::types::swiglu_inplace(&mut ctx.hidden, &ctx.hidden2);
            }
            #[cfg(not(feature = "gated_mlp"))]
            crate::types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut prefill.lora_buf);
            }
            // MLP w2 (with sparse support)
            #[cfg(feature = "sparse_mlp")]
            {
                let alive = crate::types::sparse_matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                );
                if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                    crate::types::matmul(
                        &mut ctx.x,
                        &layer_weights.mlp_w2,
                        &ctx.hidden,
                        n,
                        config.mlp_hidden,
                    );
                }
            }
            #[cfg(not(feature = "sparse_mlp"))]
            crate::types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut prefill.lora_buf);
            }
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

            // Store hidden state for next layer (multi-layer only)
            if config.n_layer > 1 {
                prefill.hidden[p * n..(p + 1) * n].copy_from_slice(&ctx.x[..n]);
            }
        }
    }

    // Snapshot hidden state (last position)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head (parallel for large vocab, serial fallback for small)
    crate::types::matmul_parallel(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}
