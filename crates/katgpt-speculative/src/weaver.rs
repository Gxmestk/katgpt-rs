//! Weaver inference-only logit corrector (Issue 131).
//!
//! Mirrors riir-train-engine's Weaver adapter as a read-only inference module.
//! Loads trained weights from a safetensors checkpoint, runs the 7-step forward
//! pass (conditioning → causal attention → SwiGLU → top-K gather → residual add
//! → renormalize), and applies the residual correction to DFlash draft logits
//! over the top-K candidate set.
//!
//! ## Architecture
//!
//! ```text
//!  h_verifier ──► RMSNorm(norm_cond) ──► W_c ──┐
//!  h_dflash[D] ─► RMSNorm(norm_cond) ──► W_c ──┤ + pos_emb ──► u_seq[D+1]
//!                                                │
//!  u_seq ──► Wq/Wk/Wv ──► causal MHA ──► W_o + residual ──► RMSNorm(norm_attn)
//!                                                                       │
//!                    SwiGLU(W_gate, W_up) → W_down + residual ──► RMSNorm(norm_mlp)
//!                                                                       │
//!                  u_final[D+1] ──► top-K gather: residual_k = <h, emb[topk_k]>
//!                                                                       │
//!        corrected = dflash_logits + weaver_residual ──► softmax over K
//! ```
//!
//! ## No-harm contract
//!
//! Zero-init weights produce **zero residual** (all matmul outputs are zero,
//! all RMSNorm scales are zero → inv_rms is finite but the dot product with
//! zero embedding rows is still zero). This means the corrector is a safe
//! no-op when no trained checkpoint is available. The feature is opt-in
//! (`weaver_runtime`); when off, DFlash behavior is bit-identical.
//!
//! ## References
//!
//! - riir-train `crates/riir-train-engine/src/weaver.rs` — training-side model
//! - riir-train Plan 314 — training plan (synthetic GOAT gate DONE)
//! - katgpt-rs Issue 131 — this integration
//! - arXiv:2607.06763 §3.2 — "Trees from Marginals" (Oda et al.)

use core::f32;

// ── Config ───────────────────────────────────────────────────────────────

/// Weaver model hyperparameters. Recovered from safetensors metadata on load.
#[derive(Debug, Clone)]
pub struct WeaverConfig {
    /// Hidden dimension (d_model). Paper default 2048; the real-data precompute
    /// uses 2304 (Gemma2-2B).
    pub hidden_dim: usize,
    /// Number of attention heads. head_dim = hidden_dim / n_heads.
    pub n_heads: usize,
    /// Candidate token count K. Weaver only projects over these.
    pub k_candidates: usize,
    /// Transformer layers. Fixed at 1 per the paper.
    pub n_layer: usize,
    /// SwiGLU intermediate dimension.
    pub d_ff: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Maximum drafter depth D. Position embeddings allocated for [0, D).
    pub max_depth: usize,
}

impl Default for WeaverConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 2048,
            n_heads: 16,
            k_candidates: 512,
            n_layer: 1,
            d_ff: 5824,
            rms_eps: 1e-6,
            max_depth: 8,
        }
    }
}

impl WeaverConfig {
    /// Per-head dimension.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.n_heads
    }
}

// ── Weights ──────────────────────────────────────────────────────────────

/// Weaver learnable weights. Stored as flat `Vec<f32>` (row-major).
///
/// All matrices use `[in_dim, out_dim]` row-major layout:
/// `output[j] = Σ_i input[i] · weight[i · out_dim + j]`.
///
/// This is a read-only mirror of riir-train's `WeaverWeights` — no optimizer
/// state, no gradient buffers.
#[derive(Debug, Clone)]
pub struct WeaverWeights {
    /// Conditioning projection W_c `[hidden, hidden]`.
    pub w_c: Vec<f32>,
    /// Attention query projection `[hidden, hidden]`.
    pub w_q: Vec<f32>,
    /// Attention key projection `[hidden, hidden]`.
    pub w_k: Vec<f32>,
    /// Attention value projection `[hidden, hidden]`.
    pub w_v: Vec<f32>,
    /// Attention output projection `[hidden, hidden]`.
    pub w_o: Vec<f32>,
    /// SwiGLU gate projection `[hidden, d_ff]`.
    pub w_gate: Vec<f32>,
    /// SwiGLU up projection `[hidden, d_ff]`.
    pub w_up: Vec<f32>,
    /// SwiGLU down projection `[d_ff, hidden]`.
    pub w_down: Vec<f32>,
    /// RMSNorm scale (conditioning). `[hidden]`
    pub norm_cond: Vec<f32>,
    /// RMSNorm scale (post-attention). `[hidden]`
    pub norm_attn: Vec<f32>,
    /// RMSNorm scale (post-MLP). `[hidden]`
    pub norm_mlp: Vec<f32>,
    /// Learned position embeddings for drafter lookaheads. `[max_depth, hidden]`
    /// Position 0 (verifier) gets no position embedding.
    pub pos_emb: Vec<f32>,
    /// Config snapshot.
    pub config: WeaverConfig,
}

impl WeaverWeights {
    /// Create zero-initialized weights for the given config.
    ///
    /// Zero weights produce zero residuals — the corrector is a safe no-op
    /// before a trained checkpoint is loaded.
    pub fn zeros(config: WeaverConfig) -> Self {
        let h = config.hidden_dim;
        let ff = config.d_ff;
        let md = config.max_depth;
        Self {
            w_c: vec![0.0; h * h],
            w_q: vec![0.0; h * h],
            w_k: vec![0.0; h * h],
            w_v: vec![0.0; h * h],
            w_o: vec![0.0; h * h],
            w_gate: vec![0.0; h * ff],
            w_up: vec![0.0; h * ff],
            w_down: vec![0.0; ff * h],
            norm_cond: vec![0.0; h],
            norm_attn: vec![0.0; h],
            norm_mlp: vec![0.0; h],
            pos_emb: vec![0.0; md * h],
            config,
        }
    }

    /// Deserialize weights from safetensors bytes.
    ///
    /// The format mirrors riir-train's `weights_to_safetensors_bytes`:
    /// 12 tensor keys (`w_c`, `w_q`, …, `pos_emb`) stored as flat 1-D F32,
    /// plus metadata (`hidden_dim`, `n_heads`, `k_candidates`, `d_ff`,
    /// `max_depth`). `rms_eps` and `n_layer` are not stored — hardcoded
    /// to `1e-6` and `1` respectively.
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self, WeaverLoadError> {
        let st = safetensors::SafeTensors::deserialize(bytes)
            .map_err(WeaverLoadError::SafetensorsParse)?;

        // Read config from the safetensors JSON header metadata. The
        // safetensors 0.4 crate doesn't expose a public metadata() accessor,
        // so we parse the raw header bytes directly.
        let header = parse_safetensors_header(bytes)?;
        let hidden_dim = extract_meta(&header, "hidden_dim")?;
        let n_heads = extract_meta(&header, "n_heads")?;
        let k_candidates = extract_meta(&header, "k_candidates")?;
        let d_ff = extract_meta(&header, "d_ff")?;
        let max_depth = extract_meta(&header, "max_depth")?;
        let config = WeaverConfig {
            hidden_dim,
            n_heads,
            k_candidates,
            n_layer: 1,
            d_ff,
            rms_eps: 1e-6,
            max_depth,
        };

        // Read each tensor via TensorView::data().
        let read = |name: &str, expected: usize| -> Result<Vec<f32>, WeaverLoadError> {
            let tv = st
                .tensor(name)
                .map_err(|e| WeaverLoadError::TensorMissing {
                    name: name.to_string(),
                    source: e,
                })?;
            let raw = tv.data();
            let n = raw.len() / 4;
            if n != expected {
                return Err(WeaverLoadError::ShapeMismatch {
                    tensor: name.to_string(),
                    expected,
                    actual: n,
                });
            }
            // Safe little-endian f32 decode — no alignment assumption.
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };

        let h = hidden_dim;
        let ff = d_ff;
        let md = max_depth;
        Ok(Self {
            w_c: read("w_c", h * h)?,
            w_q: read("w_q", h * h)?,
            w_k: read("w_k", h * h)?,
            w_v: read("w_v", h * h)?,
            w_o: read("w_o", h * h)?,
            w_gate: read("w_gate", h * ff)?,
            w_up: read("w_up", h * ff)?,
            w_down: read("w_down", ff * h)?,
            norm_cond: read("norm_cond", h)?,
            norm_attn: read("norm_attn", h)?,
            norm_mlp: read("norm_mlp", h)?,
            pos_emb: read("pos_emb", md * h)?,
            config,
        })
    }
}

// ── Input / Output ───────────────────────────────────────────────────────

/// Borrowed input for the Weaver forward pass.
pub struct WeaverInput<'a> {
    /// Verifier hidden state at the prefix position. `[hidden_dim]`
    pub h_verifier: &'a [f32],
    /// Drafter lookahead hidden states. `D` slices, each `[hidden_dim]`.
    pub h_dflash: &'a [&'a [f32]],
    /// Top-K candidate token ids per draft depth. `D` slices, each `[K]`.
    pub topk_ids: &'a [&'a [u32]],
    /// DFlash draft logits over the K candidates per depth. `D` slices, each `[K]`.
    pub dflash_logits: &'a [&'a [f32]],
    /// Shared vocab embedding (verifier/drafter), row-major. `[V * hidden_dim]`
    pub embedding: &'a [f32],
    /// Vocabulary size V (for bounds checking on gathered ids).
    pub vocab_size: usize,
}

/// Output of the Weaver forward pass.
#[derive(Debug, Clone)]
pub struct WeaverOutput {
    /// Weaver residual logits per depth. `[D][K]`
    pub weaver_residual: Vec<Vec<f32>>,
    /// Corrected logits (dflash + weaver). `[D][K]`
    pub corrected_logits: Vec<Vec<f32>>,
    /// Corrected probabilities (softmax of corrected_logits over K). `[D][K]`
    pub corrected_probs: Vec<Vec<f32>>,
    /// Drafter depth D.
    pub depth: usize,
    /// Candidate count K.
    pub k: usize,
}

// ── High-level corrector ─────────────────────────────────────────────────

/// Convenience wrapper holding loaded weights.
///
/// Construct via [`WeaverCorrector::from_checkpoint`] (reads a safetensors file
/// from disk, optionally verifying a `.blake3` sidecar) or
/// [`WeaverCorrector::from_weights`] (from a pre-loaded `WeaverWeights`).
pub struct WeaverCorrector {
    weights: WeaverWeights,
}

impl WeaverCorrector {
    /// Wrap pre-loaded weights.
    pub fn from_weights(weights: WeaverWeights) -> Self {
        Self { weights }
    }

    /// Deserialize from safetensors bytes (e.g. from `include_bytes!` or mmap).
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self, WeaverLoadError> {
        Ok(Self::from_weights(WeaverWeights::from_safetensors_bytes(
            bytes,
        )?))
    }

    /// Load from a file path. If a `<path>.blake3` sidecar exists, the file's
    /// BLAKE3 hash is verified before deserialization.
    pub fn from_checkpoint(path: impl AsRef<std::path::Path>) -> Result<Self, WeaverLoadError> {
        use std::fs;
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(WeaverLoadError::Io)?;

        // Optional BLAKE3 sidecar verification.
        let sidecar = path.with_extension("safetensors.blake3");
        if sidecar.exists() {
            let expected_hex = fs::read_to_string(&sidecar)
                .map_err(WeaverLoadError::Io)?
                .trim()
                .to_string();
            let actual = blake3::hash(&bytes).to_hex().to_string();
            if actual != expected_hex {
                return Err(WeaverLoadError::Blake3Mismatch {
                    expected: expected_hex,
                    actual,
                });
            }
        }

        Self::from_safetensors_bytes(&bytes)
    }

    /// Run the Weaver forward pass and produce corrected probabilities.
    pub fn correct(&self, input: &WeaverInput) -> WeaverOutput {
        weaver_forward(&self.weights, input)
    }

    /// Borrow the underlying weights.
    pub fn weights(&self) -> &WeaverWeights {
        &self.weights
    }

    /// Apply the Weaver correction to full-vocabulary DFlash marginals (T3).
    ///
    /// This is the **marginal corrector** integration point (Issue 131 option 1).
    /// After `dflash_predict_with` produces full-vocab marginals, the caller
    /// invokes this method to apply the Weaver residual correction over the
    /// top-K candidates at each draft depth.
    ///
    /// ## What it does
    ///
    /// For each draft depth `d ∈ [0, depth)`:
    /// 1. Select the top-K=32 token ids from `marginals[d*vocab..(d+1)*vocab]`
    ///    (by probability mass).
    /// 2. Build a `WeaverInput` from the top-K ids + drafter logits + hidden
    ///    states + shared embedding.
    /// 3. Run `weaver_forward` → corrected top-K probabilities.
    /// 4. Write the corrected probabilities back to `marginals`, zeroing all
    ///    non-top-K positions and renormalizing.
    ///
    /// ## No-harm contract
    ///
    /// When the corrector holds zero-init weights, the Weaver residual is zero,
    /// so the corrected top-K probabilities equal the drafter's top-K
    /// probabilities. After zeroing non-top-K and renormalizing over K, the
    /// result is a **truncated** version of the original marginal — the top-K
    /// mass is preserved but redistributed. This is a minor change (the top-K
    /// typically captures >99% of the probability mass), and the tree builder
    /// only consumes the top-K anyway.
    ///
    /// ## Arguments
    ///
    /// - `marginals` — full-vocab marginals, shape `[depth * vocab_size]`,
    ///   row-major (depth-outer). Modified in-place.
    /// - `h_verifier` — verifier hidden state at the prefix position, `[hidden_dim]`.
    /// - `h_dflash` — drafter hidden states per depth, `[depth][hidden_dim]`.
    /// - `embedding` — shared token embedding, `[vocab_size * hidden_dim]`.
    /// - `vocab_size` — vocabulary size.
    ///
    /// ## Returns
    ///
    /// `Ok(())` on success, or an error if the shapes are inconsistent.
    pub fn correct_marginals_inplace(
        &self,
        marginals: &mut [f32],
        h_verifier: &[f32],
        h_dflash: &[&[f32]],
        embedding: &[f32],
        vocab_size: usize,
    ) -> Result<(), WeaverCorrectError> {
        let cfg = &self.weights.config;
        let k = cfg.k_candidates;
        let h = cfg.hidden_dim;
        let depth = h_dflash.len();

        if marginals.len() != depth * vocab_size {
            return Err(WeaverCorrectError::MarginalsShape {
                expected: depth * vocab_size,
                actual: marginals.len(),
            });
        }
        if h_verifier.len() != h {
            return Err(WeaverCorrectError::HiddenShape {
                context: "h_verifier",
                expected: h,
                actual: h_verifier.len(),
            });
        }
        if embedding.len() < vocab_size * h {
            return Err(WeaverCorrectError::EmbeddingShape {
                expected: vocab_size * h,
                actual: embedding.len(),
            });
        }
        for (di, h_d) in h_dflash.iter().enumerate() {
            if h_d.len() != h {
                return Err(WeaverCorrectError::HiddenShape {
                    context: "h_dflash[di]",
                    expected: h,
                    actual: h_d.len(),
                });
            }
            let _ = di; // index not needed for the check
        }
        if depth > cfg.max_depth {
            return Err(WeaverCorrectError::DepthExceedsConfig {
                depth,
                max_depth: cfg.max_depth,
            });
        }
        if k > vocab_size {
            // Can't select K candidates from fewer than K tokens. Correct
            // nothing — leave marginals unchanged (safe no-op).
            return Ok(());
        }

        // ── Per-depth: select top-K, correct, write back ──
        // Reuse scratch buffers across depths to avoid per-depth allocation.
        // We process one depth at a time (depth is typically 4-8), building a
        // fresh single-depth WeaverInput per iteration.
        let mut topk_ids: Vec<u32> = vec![0; k];
        let mut topk_logits: Vec<f32> = vec![0.0; k];

        for di in 0..depth {
            let marg_row = &marginals[di * vocab_size..(di + 1) * vocab_size];

            // Select top-K token ids by probability mass.
            // Use partial selection sort (same as precompute_weaver_real_data).
            let mut top: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
            for (vid, &p) in marg_row.iter().enumerate() {
                if !p.is_finite() {
                    continue;
                }
                let pos = top.partition_point(|&(_, v)| v > p);
                if pos < k {
                    top.insert(pos, (vid, p));
                    if top.len() > k {
                        top.pop();
                    }
                }
            }
            // Fill topk_ids + topk_logits (log-space for the Weaver residual).
            for (ki, &(vid, p)) in top.iter().enumerate().take(k) {
                topk_ids[ki] = vid as u32;
                // Recover approximate logits from probs (Weaver adds residual
                // to logits, then softmaxes). Use log(p) clamped to avoid -inf.
                topk_logits[ki] = if p > 1e-30 { p.ln() } else { -69.07 }; // ln(1e-30)
            }
            // If fewer than K valid tokens, pad with zeros (id=0, logit=-69).
            for ki in top.len()..k {
                topk_ids[ki] = 0;
                topk_logits[ki] = -69.07;
            }

            // Build WeaverInput for this single depth.
            let h_dflash_slice: &[&[f32]] = &h_dflash[di..di + 1];
            let topk_ids_slice: &[&[u32]] = &[&topk_ids[..]];
            let topk_logits_slice: &[&[f32]] = &[&topk_logits[..]];

            let input = WeaverInput {
                h_verifier,
                h_dflash: h_dflash_slice,
                topk_ids: topk_ids_slice,
                dflash_logits: topk_logits_slice,
                embedding,
                vocab_size,
            };

            let out = weaver_forward(&self.weights, &input);

            // Write back: zero the full-vocab row, then write corrected top-K.
            let out_row = &out.corrected_probs[0]; // depth 0 of the single-depth input
            let marg_out = &mut marginals[di * vocab_size..(di + 1) * vocab_size];
            for v in marg_out.iter_mut() {
                *v = 0.0;
            }
            for (ki, &(vid, _)) in top.iter().enumerate().take(k) {
                marg_out[vid] = out_row[ki];
            }
            // marg_out already sums to ~1.0 (Weaver softmaxes over K), but
            // renormalize for safety (floating-point drift).
            let sum: f32 = marg_out.iter().sum();
            if sum > 1e-30 {
                let inv = 1.0 / sum;
                for v in marg_out.iter_mut() {
                    *v *= inv;
                }
            }
        }

        Ok(())
    }
}

/// Errors from `correct_marginals_inplace`.
#[derive(Debug)]
pub enum WeaverCorrectError {
    /// `marginals.len()` != `depth * vocab_size`.
    MarginalsShape { expected: usize, actual: usize },
    /// A hidden-state slice had the wrong length.
    HiddenShape {
        context: &'static str,
        expected: usize,
        actual: usize,
    },
    /// `embedding.len()` < `vocab_size * hidden_dim`.
    EmbeddingShape { expected: usize, actual: usize },
    /// Draft depth exceeds the Weaver config's `max_depth`.
    DepthExceedsConfig { depth: usize, max_depth: usize },
}

impl std::fmt::Display for WeaverCorrectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MarginalsShape { expected, actual } => write!(
                f,
                "marginals shape mismatch: expected {expected} (= depth*vocab), got {actual}"
            ),
            Self::HiddenShape { context, expected, actual } => write!(
                f,
                "{context} shape mismatch: expected {expected}, got {actual}"
            ),
            Self::EmbeddingShape { expected, actual } => write!(
                f,
                "embedding shape mismatch: expected >={expected}, got {actual}"
            ),
            Self::DepthExceedsConfig { depth, max_depth } => write!(
                f,
                "draft depth {depth} exceeds Weaver max_depth {max_depth}"
            ),
        }
    }
}

impl std::error::Error for WeaverCorrectError {}

// ── Forward pass ─────────────────────────────────────────────────────────

/// The 7-step Weaver forward pass.
///
/// See the module-level doc for the architecture diagram. Returns corrected
/// logits + probabilities over the K candidates per draft depth.
pub fn weaver_forward(weights: &WeaverWeights, input: &WeaverInput) -> WeaverOutput {
    let cfg = &weights.config;
    let h = cfg.hidden_dim;
    let k = cfg.k_candidates;
    let n_heads = cfg.n_heads;
    let head_dim = cfg.head_dim();
    let d_ff = cfg.d_ff;
    let eps = cfg.rms_eps;
    let d_depth = input.h_dflash.len();
    let seq_len = d_depth + 1; // verifier + drafter lookaheads

    debug_assert_eq!(input.h_verifier.len(), h);
    debug_assert_eq!(input.topk_ids.len(), d_depth);
    debug_assert_eq!(input.dflash_logits.len(), d_depth);
    for di in 0..d_depth {
        debug_assert_eq!(input.h_dflash[di].len(), h);
        debug_assert_eq!(input.topk_ids[di].len(), k);
        debug_assert_eq!(input.dflash_logits[di].len(), k);
    }

    // ── Step 1: Conditioning sequence u[0..seq_len] ──
    // Pre-allocate scratch buffers reused across all positions — avoids
    // per-position Vec allocations (5 * seq_len allocs eliminated).
    let mut u_cond = vec![0.0f32; seq_len * h];
    let mut normed_buf = vec![0.0f32; h];
    let mut post_buf = vec![0.0f32; h];
    for pos in 0..seq_len {
        let raw = if pos == 0 {
            input.h_verifier
        } else {
            input.h_dflash[pos - 1]
        };
        rmsnorm_into(raw, &weights.norm_cond, eps, &mut normed_buf);
        let u_row = &mut u_cond[pos * h..(pos + 1) * h];
        matmul_vec(&normed_buf, &weights.w_c, h, h, u_row);
        if pos > 0 {
            let pe = &weights.pos_emb[(pos - 1) * h..pos * h];
            for j in 0..h {
                u_row[j] += pe[j];
            }
        }
    }

    // ── Step 2: QKV projections ──
    let mut q = vec![0.0f32; seq_len * h];
    let mut kk = vec![0.0f32; seq_len * h];
    let mut v = vec![0.0f32; seq_len * h];
    for pos in 0..seq_len {
        let u_row = &u_cond[pos * h..(pos + 1) * h];
        matmul_vec(u_row, &weights.w_q, h, h, &mut q[pos * h..(pos + 1) * h]);
        matmul_vec(u_row, &weights.w_k, h, h, &mut kk[pos * h..(pos + 1) * h]);
        matmul_vec(u_row, &weights.w_v, h, h, &mut v[pos * h..(pos + 1) * h]);
    }

    // ── Step 3: Causal multi-head attention ──
    let attn_scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * h];
    let mut scores = vec![0.0f32; seq_len];
    for head in 0..n_heads {
        let ho = head * head_dim;
        for qi in 0..seq_len {
            let q_row = &q[qi * h + ho..qi * h + ho + head_dim];
            // Causal: attend to kj ∈ [0, qi]
            let mut max_s = f32::NEG_INFINITY;
            for kj in 0..=qi {
                let k_row = &kk[kj * h + ho..kj * h + ho + head_dim];
                let s = dot(q_row, k_row) * attn_scale;
                scores[kj] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_e = 0.0;
            for s in scores[..=qi].iter_mut() {
                *s = (*s - max_s).exp();
                sum_e += *s;
            }
            let inv_sum = 1.0 / sum_e;
            let out_row = &mut attn_out[qi * h + ho..qi * h + ho + head_dim];
            for kj in 0..=qi {
                let w = scores[kj] * inv_sum;
                let v_row = &v[kj * h + ho..kj * h + ho + head_dim];
                katgpt_core::simd::simd_fused_scale_acc(out_row, v_row, w, head_dim);
            }
        }
    }

    // ── Step 4: Output projection + residual + post-attn RMSNorm ──
    let mut u_attn_normed = vec![0.0f32; seq_len * h];
    let mut tmp = vec![0.0f32; h];
    for pos in 0..seq_len {
        let o_row = &attn_out[pos * h..(pos + 1) * h];
        matmul_vec(o_row, &weights.w_o, h, h, &mut tmp);
        let u = &u_cond[pos * h..(pos + 1) * h];
        for j in 0..h {
            post_buf[j] = u[j] + tmp[j];
        }
        rmsnorm_into(&post_buf, &weights.norm_attn, eps, &mut u_attn_normed[pos * h..(pos + 1) * h]);
    }

    // ── Step 5: SwiGLU MLP + residual + post-MLP RMSNorm ──
    let mut u_final = vec![0.0f32; seq_len * h];
    let mut gate = vec![0.0f32; d_ff];
    let mut up = vec![0.0f32; d_ff];
    let mut act = vec![0.0f32; d_ff];
    let mut down = vec![0.0f32; h];
    for pos in 0..seq_len {
        let u_row = &u_attn_normed[pos * h..(pos + 1) * h];
        matmul_vec(u_row, &weights.w_gate, h, d_ff, &mut gate);
        matmul_vec(u_row, &weights.w_up, h, d_ff, &mut up);
        for j in 0..d_ff {
            act[j] = silu(gate[j]) * up[j];
        }
        matmul_vec(&act, &weights.w_down, d_ff, h, &mut down);
        for j in 0..h {
            post_buf[j] = u_row[j] + down[j];
        }
        rmsnorm_into(&post_buf, &weights.norm_mlp, eps, &mut u_final[pos * h..(pos + 1) * h]);
    }

    // ── Steps 6 + 7: Top-K gather + residual add + softmax over K ──
    let mut weaver_residual = vec![vec![0.0f32; k]; d_depth];
    let mut corrected_logits = vec![vec![0.0f32; k]; d_depth];
    let mut corrected_probs = vec![vec![0.0f32; k]; d_depth];
    let mut gathered = vec![0.0f32; k * h]; // scratch for gathered embedding rows

    for di in 0..d_depth {
        let pos = di + 1; // skip verifier position 0
        let h_weaver = &u_final[pos * h..(pos + 1) * h];
        let ids = input.topk_ids[di];
        let dfl = input.dflash_logits[di];

        // Gather K embedding rows.
        for (ki, &tid) in ids.iter().enumerate() {
            let tid = tid as usize;
            debug_assert!(tid < input.vocab_size, "topk id {} >= vocab {}", tid, input.vocab_size);
            let row = &input.embedding[tid * h..(tid + 1) * h];
            gathered[ki * h..(ki + 1) * h].copy_from_slice(row);
        }

        // Compute residual logits.
        for ki in 0..k {
            let grow = &gathered[ki * h..(ki + 1) * h];
            weaver_residual[di][ki] = dot(h_weaver, grow);
        }

        // Corrected = dflash + weaver_residual.
        for ki in 0..k {
            corrected_logits[di][ki] = dfl[ki] + weaver_residual[di][ki];
        }

        // Softmax over K candidates.
        let mut max_c = f32::NEG_INFINITY;
        for cl in corrected_logits[di][..k].iter() {
            if *cl > max_c {
                max_c = *cl;
            }
        }
        let mut sum_e = 0.0;
        for ki in 0..k {
            let e = (corrected_logits[di][ki] - max_c).exp();
            corrected_probs[di][ki] = e;
            sum_e += e;
        }
        let inv_sum = 1.0 / sum_e;
        for cp in corrected_probs[di][..k].iter_mut() {
            *cp *= inv_sum;
        }
    }

    WeaverOutput {
        weaver_residual,
        corrected_logits,
        corrected_probs,
        depth: d_depth,
        k,
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Matrix-vector multiply: `output[j] = Σ_i input[i] · weight[i · out_dim + j]`.
///
/// The weight matrix is `[in_dim, out_dim]` row-major. Uses AXPY iteration
/// (for each input element, scale-add the weight row into output) which is
/// cache-friendly for this layout. The inner AXPY delegates to
/// `simd_fused_scale_acc` for NEON/AVX2 dispatch.
#[inline]
fn matmul_vec(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize, output: &mut [f32]) {
    output[..out_dim].fill(0.0);
    for i in 0..in_dim {
        let xi = input[i];
        let row = &weight[i * out_dim..(i + 1) * out_dim];
        katgpt_core::simd::simd_fused_scale_acc(output, row, xi, out_dim);
    }
}

/// RMSNorm: `output = x / sqrt(mean(x²) + eps) · scale`.
///
/// Writes into `output` — zero allocation. The caller must ensure `output`
/// has the same length as `x` and `scale`.
#[inline]
fn rmsnorm_into(x: &[f32], scale: &[f32], eps: f32, output: &mut [f32]) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for (o, (&v, &s)) in output.iter_mut().zip(x.iter().zip(scale.iter())) {
        *o = v * inv_rms * s;
    }
}

/// SiLU / Swish activation: `x / (1 + e^{-x})`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Dot product — delegates to `simd_dot_f32` for NEON/AVX2 dispatch.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    katgpt_core::simd::simd_dot_f32(a, b, a.len().min(b.len()))
}

// ── Error type ───────────────────────────────────────────────────────────

/// Errors that can occur during Weaver checkpoint loading.
#[derive(Debug)]
pub enum WeaverLoadError {
    /// safetensors format parse failure.
    SafetensorsParse(safetensors::SafeTensorError),
    /// A metadata key is missing or unparseable.
    MetadataParse { key: String, value: String },
    /// A tensor is missing from the checkpoint.
    TensorMissing {
        name: String,
        source: safetensors::SafeTensorError,
    },
    /// A tensor has the wrong number of elements.
    ShapeMismatch {
        tensor: String,
        expected: usize,
        actual: usize,
    },
    /// BLAKE3 sidecar verification failed.
    Blake3Mismatch { expected: String, actual: String },
    /// Filesystem I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for WeaverLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SafetensorsParse(e) => write!(f, "safetensors parse error: {e}"),
            Self::MetadataParse { key, value } => {
                write!(f, "cannot parse metadata '{key}' from value '{value}'")
            }
            Self::TensorMissing { name, .. } => write!(f, "tensor '{name}' not found in checkpoint"),
            Self::ShapeMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "tensor '{tensor}' has {actual} elements, expected {expected}"
            ),
            Self::Blake3Mismatch { expected, actual } => {
                write!(f, "BLAKE3 mismatch: expected {expected}, got {actual}")
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for WeaverLoadError {}

/// Extract the UTF-8 JSON header from raw safetensors bytes.
fn parse_safetensors_header(bytes: &[u8]) -> Result<String, WeaverLoadError> {
    if bytes.len() < 8 {
        return Err(WeaverLoadError::SafetensorsParse(
            safetensors::SafeTensorError::InvalidHeader,
        ));
    }
    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) as usize;
    if bytes.len() < 8 + header_len {
        return Err(WeaverLoadError::SafetensorsParse(
            safetensors::SafeTensorError::InvalidHeader,
        ));
    }
    std::str::from_utf8(&bytes[8..8 + header_len])
        .map(String::from)
        .map_err(|_| WeaverLoadError::SafetensorsParse(
            safetensors::SafeTensorError::InvalidHeader,
        ))
}

/// Search the JSON header for a metadata value like `"hidden_dim":"2048"`.
/// This avoids needing a full JSON parser — the safetensors metadata keys
/// are simple string-to-string maps with numeric values.
fn extract_meta(header: &str, key: &str) -> Result<usize, WeaverLoadError> {
    // Look for the pattern: "key":"value"
    // The metadata section appears as: "__metadata__":{"hidden_dim":"2048",...}
    let needle = format!("\"{key}\":\"");
    let start = header
        .find(&needle)
        .ok_or_else(|| WeaverLoadError::MetadataParse {
            key: key.to_string(),
            value: "<missing>".to_string(),
        })?;
    let val_start = start + needle.len();
    let val_end = header[val_start..]
        .find('"')
        .ok_or_else(|| WeaverLoadError::MetadataParse {
            key: key.to_string(),
            value: "<unterminated>".to_string(),
        })?;
    let val_str = &header[val_start..val_start + val_end];
    val_str.parse::<usize>().map_err(|_| WeaverLoadError::MetadataParse {
        key: key.to_string(),
        value: val_str.to_string(),
    })
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Small config for fast tests.
    fn test_config() -> WeaverConfig {
        WeaverConfig {
            hidden_dim: 32,
            n_heads: 4,
            k_candidates: 8,
            n_layer: 1,
            d_ff: 64,
            rms_eps: 1e-6,
            max_depth: 3,
        }
    }

    /// Build a dummy WeaverInput for the test config.
    fn test_input(cfg: &WeaverConfig, vocab_size: usize) -> WeaverInput<'static> {
        // We can't easily build 'static slices; leak the data.
        let h = cfg.hidden_dim;
        let k = cfg.k_candidates;
        let d = cfg.max_depth;

        let h_verifier: &'static [f32] = Box::leak(vec![0.5f32; h].into_boxed_slice());
        let mut h_dflash: Vec<&'static [f32]> = Vec::with_capacity(d);
        let mut topk_ids: Vec<&'static [u32]> = Vec::with_capacity(d);
        let mut dflash_logits: Vec<&'static [f32]> = Vec::with_capacity(d);
        for di in 0..d {
            h_dflash.push(Box::leak(
                (0..h).map(|i| 0.3 + 0.01 * (di + i) as f32).collect::<Vec<f32>>().into_boxed_slice(),
            ));
            topk_ids.push(Box::leak(
                (0..k).map(|i| (i as u32) % vocab_size as u32).collect::<Vec<u32>>().into_boxed_slice(),
            ));
            dflash_logits.push(Box::leak(
                (0..k).map(|i| (i as f32) * 0.1).collect::<Vec<f32>>().into_boxed_slice(),
            ));
        }
        let emb: &'static [f32] =
            Box::leak(vec![0.1f32; vocab_size * h].into_boxed_slice());

        WeaverInput {
            h_verifier,
            h_dflash: Box::leak(h_dflash.into_boxed_slice()),
            topk_ids: Box::leak(topk_ids.into_boxed_slice()),
            dflash_logits: Box::leak(dflash_logits.into_boxed_slice()),
            embedding: emb,
            vocab_size,
        }
    }

    // ── G1: Correctness gates ──

    #[test]
    fn g1_zero_weights_produce_zero_residual() {
        let cfg = test_config();
        let weights = WeaverWeights::zeros(cfg.clone());
        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        assert_eq!(out.depth, cfg.max_depth);
        assert_eq!(out.k, cfg.k_candidates);

        // Zero weights → zero attention, zero MLP, but RMSNorm of non-zero input
        // is non-zero. However, all weight matrices are zero, so every matmul
        // output is zero. The final u_final is RMSNorm(post_mlp, norm_mlp=0)
        // which is zero (scale=0). So residuals are dot(0_vector, embedding) = 0.
        for di in 0..out.depth {
            for ki in 0..out.k {
                assert!(
                    out.weaver_residual[di][ki].abs() < 1e-6,
                    "non-zero residual at di={di} ki={ki}: {}",
                    out.weaver_residual[di][ki]
                );
            }
        }
    }

    #[test]
    fn g1_corrected_probs_sum_to_one() {
        let cfg = test_config();
        let weights = WeaverWeights::zeros(cfg.clone());
        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        for di in 0..out.depth {
            let sum: f32 = out.corrected_probs[di].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "probs at di={di} sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn g1_no_nan_or_inf_in_output() {
        let cfg = test_config();
        let weights = WeaverWeights::zeros(cfg.clone());
        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        for di in 0..out.depth {
            for ki in 0..out.k {
                assert!(out.corrected_probs[di][ki].is_finite(), "NaN/Inf in probs");
                assert!(out.corrected_logits[di][ki].is_finite(), "NaN/Inf in logits");
                assert!(out.weaver_residual[di][ki].is_finite(), "NaN/Inf in residual");
            }
        }
    }

    #[test]
    fn g1_zero_weights_corrected_equals_dflash() {
        // With zero residual, corrected_logits == dflash_logits.
        let cfg = test_config();
        let weights = WeaverWeights::zeros(cfg.clone());
        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        for di in 0..out.depth {
            for ki in 0..out.k {
                let diff = (out.corrected_logits[di][ki] - input.dflash_logits[di][ki]).abs();
                assert!(diff < 1e-6, "corrected != dflash at di={di} ki={ki}: diff={diff}");
            }
        }
    }

    // ── G3: No-regression gate (feature isolation) ──

    #[test]
    fn g3_nonzero_weights_change_logits() {
        // With non-zero weights, the corrected logits should differ from dflash.
        let cfg = test_config();
        let mut weights = WeaverWeights::zeros(cfg.clone());
        // Set non-zero RMSNorm scales so u_final is non-zero.
        for s in &mut weights.norm_cond {
            *s = 1.0;
        }
        for s in &mut weights.norm_attn {
            *s = 1.0;
        }
        for s in &mut weights.norm_mlp {
            *s = 1.0;
        }
        // Identity W_c (so conditioning preserves the hidden state direction).
        for i in 0..cfg.hidden_dim {
            weights.w_c[i * cfg.hidden_dim + i] = 1.0;
        }

        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        // With non-zero norm scales, u_final should be non-zero, so residuals
        // should be non-zero (dot of non-zero vector with non-zero embedding).
        let any_nonzero = (0..out.depth)
        .flat_map(|_di| 0..out.k)
        .any(|_| true);
        assert!(any_nonzero, "output should have entries");

        // At least some residuals should be non-zero with non-zero norms.
        let max_residual = (0..out.depth)
            .flat_map(|_di| 0..out.k)
            .map(|di_k| {
                let di = di_k / out.k;
                let ki = di_k % out.k;
                out.weaver_residual[di][ki].abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_residual > 1e-6,
            "expected non-zero residuals with non-zero norm scales, got max={max_residual}"
        );
    }

    // ── Safetensors round-trip ──

    #[test]
    fn safetensors_roundtrip() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let cfg = WeaverConfig {
            hidden_dim: 16,
            n_heads: 2,
            k_candidates: 4,
            n_layer: 1,
            d_ff: 32,
            rms_eps: 1e-6,
            max_depth: 2,
        };
        let original = WeaverWeights::zeros(cfg.clone());

        // Serialize to safetensors bytes.
        let h = cfg.hidden_dim;
        let ff = cfg.d_ff;
        let md = cfg.max_depth;

        let tensors: Vec<(String, safetensors::tensor::TensorView)> = vec![
            ("w_c".to_string(),      TensorView::new(Dtype::F32, vec![original.w_c.len()],      bytemuck::cast_slice(&original.w_c)).unwrap()),
            ("w_q".to_string(),      TensorView::new(Dtype::F32, vec![original.w_q.len()],      bytemuck::cast_slice(&original.w_q)).unwrap()),
            ("w_k".to_string(),      TensorView::new(Dtype::F32, vec![original.w_k.len()],      bytemuck::cast_slice(&original.w_k)).unwrap()),
            ("w_v".to_string(),      TensorView::new(Dtype::F32, vec![original.w_v.len()],      bytemuck::cast_slice(&original.w_v)).unwrap()),
            ("w_o".to_string(),      TensorView::new(Dtype::F32, vec![original.w_o.len()],      bytemuck::cast_slice(&original.w_o)).unwrap()),
            ("w_gate".to_string(),   TensorView::new(Dtype::F32, vec![original.w_gate.len()],   bytemuck::cast_slice(&original.w_gate)).unwrap()),
            ("w_up".to_string(),     TensorView::new(Dtype::F32, vec![original.w_up.len()],     bytemuck::cast_slice(&original.w_up)).unwrap()),
            ("w_down".to_string(),   TensorView::new(Dtype::F32, vec![original.w_down.len()],   bytemuck::cast_slice(&original.w_down)).unwrap()),
            ("norm_cond".to_string(), TensorView::new(Dtype::F32, vec![original.norm_cond.len()], bytemuck::cast_slice(&original.norm_cond)).unwrap()),
            ("norm_attn".to_string(), TensorView::new(Dtype::F32, vec![original.norm_attn.len()], bytemuck::cast_slice(&original.norm_attn)).unwrap()),
            ("norm_mlp".to_string(), TensorView::new(Dtype::F32, vec![original.norm_mlp.len()], bytemuck::cast_slice(&original.norm_mlp)).unwrap()),
            ("pos_emb".to_string(),  TensorView::new(Dtype::F32, vec![original.pos_emb.len()],  bytemuck::cast_slice(&original.pos_emb)).unwrap()),
        ];

        let metadata = Some(std::collections::HashMap::from([
            ("format".to_string(), "weaver_v1".to_string()),
            ("hidden_dim".to_string(), h.to_string()),
            ("n_heads".to_string(), cfg.n_heads.to_string()),
            ("k_candidates".to_string(), cfg.k_candidates.to_string()),
            ("d_ff".to_string(), ff.to_string()),
            ("max_depth".to_string(), md.to_string()),
        ]));

        let bytes = safetensors::serialize(tensors, &metadata).expect("serialize");

        // Deserialize.
        let loaded = WeaverWeights::from_safetensors_bytes(&bytes).expect("deserialize");

        // Verify config.
        assert_eq!(loaded.config.hidden_dim, h);
        assert_eq!(loaded.config.n_heads, cfg.n_heads);
        assert_eq!(loaded.config.k_candidates, cfg.k_candidates);
        assert_eq!(loaded.config.d_ff, ff);
        assert_eq!(loaded.config.max_depth, md);

        // Verify weights bit-identically.
        assert_eq!(loaded.w_c, original.w_c);
        assert_eq!(loaded.w_q, original.w_q);
        assert_eq!(loaded.w_k, original.w_k);
        assert_eq!(loaded.w_v, original.w_v);
        assert_eq!(loaded.w_o, original.w_o);
        assert_eq!(loaded.w_gate, original.w_gate);
        assert_eq!(loaded.w_up, original.w_up);
        assert_eq!(loaded.w_down, original.w_down);
        assert_eq!(loaded.norm_cond, original.norm_cond);
        assert_eq!(loaded.norm_attn, original.norm_attn);
        assert_eq!(loaded.norm_mlp, original.norm_mlp);
        assert_eq!(loaded.pos_emb, original.pos_emb);
    }

    // ── Helper function tests ──

    #[test]
    fn matmul_vec_identity() {
        // Identity matrix: output should equal input.
        let n = 4;
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; n];
        matmul_vec(&input, &identity, n, n, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn silu_zero_is_zero() {
        assert!(silu(0.0).abs() < 1e-10);
    }

    #[test]
    fn silu_large_positive_approximates_x() {
        assert!((silu(10.0) - 10.0).abs() < 0.1);
    }

    #[test]
    fn rmsnorm_unit_scale_preserves_direction() {
        let x = vec![3.0, 4.0]; // ||x|| = 5, mean(x²) = 12.5
        let scale = vec![1.0, 1.0];
        let mut out = vec![0.0, 0.0];
        rmsnorm_into(&x, &scale, 1e-6, &mut out);
        // RMS = sqrt(12.5 + eps) ≈ 3.536
        let rms = (12.5f32 + 1e-6).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-4);
        assert!((out[1] - 4.0 / rms).abs() < 1e-4);
    }

    // ── T3: correct_marginals_inplace (marginal corrector integration) ──

    /// Build a tiny WeaverConfig + WeaverCorrector for the marginal-corrector test.
    fn corrector_config() -> WeaverConfig {
        WeaverConfig {
            hidden_dim: 16,
            n_heads: 2,
            k_candidates: 4,
            n_layer: 1,
            d_ff: 32,
            rms_eps: 1e-6,
            max_depth: 2,
        }
    }

    #[test]
    fn t3_correct_marginals_zero_weights_preserves_topk() {
        // With zero-init weights, the Weaver residual is zero, so the corrected
        // top-K probabilities equal the drafter's top-K probabilities (after
        // renorm over K). Non-top-K positions are zeroed.
        let cfg = corrector_config();
        let corrector = WeaverCorrector::from_weights(WeaverWeights::zeros(cfg.clone()));

        let h = cfg.hidden_dim;
        let k = cfg.k_candidates;
        let depth = cfg.max_depth;
        let vocab = 32usize; // > k so top-K selection is meaningful

        // Build marginals: each depth has a peaked distribution with 5 peaks
        // (> k=4 so top-K selection is meaningful and pads correctly).
        let mut marginals = vec![0.0f32; depth * vocab];
        for di in 0..depth {
            marginals[di * vocab + di] = 0.5; // peak at token `di`
            marginals[di * vocab + (vocab - 1)] = 0.2; // second peak
            marginals[di * vocab + 10] = 0.15; // third peak
            marginals[di * vocab + 11] = 0.1; // fourth peak
            marginals[di * vocab + 12] = 0.05; // fifth peak (below K, gets zeroed)
            // rest is 0
        }

        // Save the original top-K for comparison.
        let mut orig_topk: Vec<(usize, f32)> = Vec::new();
        for di in 0..depth {
            let row = &marginals[di * vocab..(di + 1) * vocab];
            let mut sorted: Vec<(usize, f32)> =
                row.iter().cloned().enumerate().filter(|(_, p)| *p > 0.0).collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            orig_topk.extend(sorted.into_iter().take(k));
        }

        // Hidden states + embedding (arbitrary non-degenerate values).
        let h_verifier: Vec<f32> = vec![0.5; h];
        let h_dflash_owned: Vec<Vec<f32>> = (0..depth).map(|di| {
            (0..h).map(|j| 0.3 + 0.01 * (di * h + j) as f32).collect()
        }).collect();
        let h_dflash: Vec<&[f32]> = h_dflash_owned.iter().map(|v| v.as_slice()).collect();
        let embedding: Vec<f32> = vec![0.1; vocab * h];

        corrector
            .correct_marginals_inplace(&mut marginals, &h_verifier, &h_dflash, &embedding, vocab)
            .expect("correction should succeed");

        // After correction: exactly K non-zero entries per depth (the top-K).
        for di in 0..depth {
            let row = &marginals[di * vocab..(di + 1) * vocab];
            let nonzero_count = row.iter().filter(|&&p| p > 1e-10).count();
            assert_eq!(
                nonzero_count, k,
                "depth {di}: expected exactly {k} non-zero entries, got {nonzero_count}"
            );
            // Probabilities must sum to 1.0 (renormalized over K).
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "depth {di}: probs sum to {sum}");
            // No NaN/Inf.
            for &p in row {
                assert!(p.is_finite(), "depth {di}: NaN/Inf in probs");
            }
        }

        // The top-K token ids should be preserved (the same tokens that had
        // mass before still have mass after, with zero weights). The top-4 by
        // mass are: di (0.5), vocab-1 (0.2), 10 (0.15), 11 (0.1). Token 12
        // (0.05) is rank 5 — below K, gets zeroed.
        for di in 0..depth {
            let row = &marginals[di * vocab..(di + 1) * vocab];
            assert!(row[di] > 1e-6, "depth {di}: original peak at {di} lost mass");
            assert!(row[vocab - 1] > 1e-6, "depth {di}: original peak at {} lost mass", vocab - 1);
            assert!(row[10] > 1e-6, "depth {di}: original peak at 10 lost mass");
            assert!(row[11] > 1e-6, "depth {di}: original peak at 11 lost mass");
            // Token 12 was rank 5 — below K=4, should be zeroed.
            assert!(row[12] < 1e-10, "depth {di}: rank-5 token at 12 should be zeroed");
        }
    }

    #[test]
    fn t3_correct_marginals_rejects_bad_shapes() {
        let cfg = corrector_config();
        let corrector = WeaverCorrector::from_weights(WeaverWeights::zeros(cfg.clone()));

        let h = cfg.hidden_dim;
        let depth = cfg.max_depth;
        let vocab = 32usize;

        let h_verifier = vec![0.5f32; h];
        let h_dflash_owned: Vec<Vec<f32>> = (0..depth).map(|_| vec![0.3; h]).collect();
        let h_dflash: Vec<&[f32]> = h_dflash_owned.iter().map(|v| v.as_slice()).collect();
        let embedding = vec![0.1f32; vocab * h];

        // Wrong marginals length (depth * vocab + 1).
        let mut bad_marginals = vec![0.0f32; depth * vocab + 1];
        let err = corrector
            .correct_marginals_inplace(&mut bad_marginals, &h_verifier, &h_dflash, &embedding, vocab)
            .unwrap_err();
        assert!(matches!(err, super::WeaverCorrectError::MarginalsShape { .. }));

        // Wrong h_verifier length.
        let mut marginals = vec![0.0f32; depth * vocab];
        let bad_h = vec![0.5f32; h + 1];
        let err = corrector
            .correct_marginals_inplace(&mut marginals, &bad_h, &h_dflash, &embedding, vocab)
            .unwrap_err();
        assert!(matches!(err, super::WeaverCorrectError::HiddenShape { .. }));

        // Depth exceeds max_depth.
        let too_deep_owned: Vec<Vec<f32>> = (0..(depth + 1)).map(|_| vec![0.3; h]).collect();
        let too_deep: Vec<&[f32]> = too_deep_owned.iter().map(|v| v.as_slice()).collect();
        let mut deep_marginals = vec![0.0f32; (depth + 1) * vocab];
        let err = corrector
            .correct_marginals_inplace(&mut deep_marginals, &h_verifier, &too_deep, &embedding, vocab)
            .unwrap_err();
        assert!(matches!(err, super::WeaverCorrectError::DepthExceedsConfig { .. }));
    }
}
