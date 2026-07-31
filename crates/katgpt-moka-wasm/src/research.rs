//! Research hooks for the Issue 565 / Research 463 quant-error-LoRA PoC.
//!
//! Exposes Moka's internal weight layout + a corrected forward pass so the
//! `riir-poc` bench can test quantization-compensation strategies (weight-space
//! SVD, output-space data-aware SVD, top-K sparse bypass) against the real
//! Moka v1 weights WITHOUT reimplementing the forward pass.
//!
//! This module is gated behind the `research` feature (OFF by default — the
//! WASM browser build never enables it). It does not change any production
//! behavior; it only adds `pub` accessors + a `forward_corrected_with_scratch`
//! entry point.
//!
//! # The corrected forward pass
//!
//! [`forward_corrected_with_scratch`] runs the SAME arithmetic as
//! [`crate::moka::forward_with_scratch`], but after each conv/linear it
//! optionally applies a reader-LoRA correction: `y += correction(x)`. The
//! correction is supplied via [`LayerCorrection`] (dense low-rank A·B) or
//! [`SparseCorrection`] (COO scatter) — raw slices, not `QuantErrorLora`, so
//! this crate stays free of the katgpt-core dependency (the PoC builds the
//! correction factors from `katgpt_core::quant_error_lora` and passes the raw
//! A/B slices here).
//!
//! [`crate::moka::forward_with_scratch`]: ../moka/fn.forward_with_scratch.html

use crate::board::{AREA as BOARD_AREA, SIZE as BOARD_SIZE};
use crate::moka::{
    BOTTLENECK_CHANNELS, INPUT_PLANES, NUM_BLOCKS, POLICY_CHANNELS, POLICY_MOVES,
    TRUNK_CHANNELS, VALUE_CHANNELS, MokaScratch, MokaWeights,
    global_mean_max_into, relu_inplace,
};

// ─── Weight accessors ───────────────────────────────────────────────────────

/// A reference to a single conv/linear weight matrix + its bias + the kernel
/// size (1 for linear / 1×1 conv, 3 for 3×3 conv).
///
/// The weight is row-major `[out_dim × in_dim]` where `in_dim = k·k·in_ch`.
/// This is the standard im2col layout: `weight[oc * in_dim + ic]`.
pub struct LayerWeightRef<'a> {
    pub w: &'a [f32],
    pub b: &'a [f32],
    pub out_dim: usize,
    pub in_dim: usize,
    pub k: usize,
}

impl MokaWeights {
    /// Iterator over all conv/linear weight matrices in the order the forward
    /// pass evaluates them. Each entry is `(name, layer_ref)`. Useful for the
    /// PoC to build per-layer error matrices + corrections in one pass.
    pub fn iter_layers(&self) -> Vec<(&'static str, LayerWeightRef<'_>)> {
        let mut out = Vec::with_capacity(6 + NUM_BLOCKS * 5);
        let (stem_w, stem_b) = self.stem_w();
        out.push(("stem", LayerWeightRef {
            w: stem_w, b: stem_b,
            out_dim: TRUNK_CHANNELS, in_dim: 3 * 3 * INPUT_PLANES, k: 3,
        }));
        for block in self.blocks_ref() {
            let (rw, rb) = block.reduce_w();
            out.push(("residual.reduce", LayerWeightRef {
                w: rw, b: rb, out_dim: BOTTLENECK_CHANNELS, in_dim: TRUNK_CHANNELS, k: 1,
            }));
            let (fw, fb) = block.first_w();
            out.push(("residual.first", LayerWeightRef {
                w: fw, b: fb, out_dim: BOTTLENECK_CHANNELS, in_dim: 3 * 3 * BOTTLENECK_CHANNELS, k: 3,
            }));
            if let Some(g) = block.global_ref() {
                let (gh_w, gh_b) = g.hidden_w();
                let g_hidden_out = gh_b.len();
                out.push(("residual.global.hidden", LayerWeightRef {
                    w: gh_w, b: gh_b, out_dim: g_hidden_out, in_dim: BOTTLENECK_CHANNELS * 2, k: 1,
                }));
                let (go_w, go_b) = g.output_w();
                let g_out_out = go_b.len();
                out.push(("residual.global.output", LayerWeightRef {
                    w: go_w, b: go_b, out_dim: g_out_out, in_dim: g_hidden_out, k: 1,
                }));
            }
            let (sw, sb) = block.second_w();
            out.push(("residual.second", LayerWeightRef {
                w: sw, b: sb, out_dim: BOTTLENECK_CHANNELS, in_dim: 3 * 3 * BOTTLENECK_CHANNELS, k: 3,
            }));
            let (ew, eb) = block.expand_w();
            out.push(("residual.expand", LayerWeightRef {
                w: ew, b: eb, out_dim: TRUNK_CHANNELS, in_dim: BOTTLENECK_CHANNELS, k: 1,
            }));
        }
        let (pc_w, pc_b) = self.policy_conv_w();
        out.push(("policy.conv", LayerWeightRef {
            w: pc_w, b: pc_b, out_dim: POLICY_CHANNELS, in_dim: TRUNK_CHANNELS, k: 1,
        }));
        let (pl_w, pl_b) = self.policy_linear_w();
        let policy_lin_in = pl_w.len() / pl_b.len();
        out.push(("policy.linear", LayerWeightRef {
            w: pl_w, b: pl_b, out_dim: POLICY_MOVES, in_dim: policy_lin_in, k: 1,
        }));
        let (vc_w, vc_b) = self.value_conv_w();
        out.push(("value.conv", LayerWeightRef {
            w: vc_w, b: vc_b, out_dim: VALUE_CHANNELS, in_dim: TRUNK_CHANNELS, k: 1,
        }));
        let (vh_w, vh_b) = self.value_hidden_w();
        let value_hidden_out = vh_b.len();
        let value_hidden_in = vh_w.len() / value_hidden_out;
        out.push(("value.hidden", LayerWeightRef {
            w: vh_w, b: vh_b, out_dim: value_hidden_out, in_dim: value_hidden_in, k: 1,
        }));
        let (vo_w, vo_b) = self.value_output_w();
        out.push(("value.output", LayerWeightRef {
            w: vo_w, b: vo_b, out_dim: 1, in_dim: value_hidden_out, k: 1,
        }));
        out
    }

    /// Total parameter count (all weights + biases). For verifying against
    /// the known Moka v1 size (~105K).
    pub fn total_params(&self) -> usize {
        self.iter_layers()
            .iter()
            .map(|(_, l)| l.w.len() + l.b.len())
            .sum()
    }
}

// ─── Correction types ───────────────────────────────────────────────────────

/// A dense low-rank correction: `y += alpha * B · (A · x)`.
///
/// `a` is `[rank × in_dim]` row-major; `b` is `[out_dim × rank]` row-major.
/// Matches the layout of `katgpt_core::quant_error_lora::QuantErrorLora`.
pub struct LayerCorrection<'a> {
    pub a: &'a [f32], // [rank × in_dim]
    pub b: &'a [f32], // [out_dim × rank]
    pub rank: usize,
    pub alpha: f32,
}

/// A sparse COO correction: `y[row] += val * x[col]` for each stored element.
pub struct SparseCorrection<'a> {
    pub rows: &'a [u32],
    pub cols: &'a [u32],
    pub vals: &'a [f32],
}

/// Which correction to apply at a given layer (if any). The PoC builds a
/// `Vec<Correction>` aligned with `MokaWeights::iter_layers()` — same order,
/// same length, each slot `None` for layers the PoC chose not to correct.
pub enum Correction<'a> {
    None,
    Dense(LayerCorrection<'a>),
    Sparse(SparseCorrection<'a>),
    /// Full dense matvec correction: `y += M · x`. Used as the measurement
    /// vehicle for G1/G2 — the correction matrix M = (W_target - W_f32) is
    /// precomputed per strategy, and applied directly without LoRA encoding.
    /// The production representation (compact rank-r LoRA or sparse COO) is
    /// what the PoC measures the QUALITY of; this variant measures the exact
    /// effect of a given correction matrix on the forward output.
    Full { mat: &'a [f32], out_dim: usize, in_dim: usize },
}

impl<'a> Correction<'a> {
    /// Apply the correction to `y` (adds to the existing conv/linear output).
    /// `x` is the conv input patch (the im2col vector), `y` is the output
    /// channel slice for ONE spatial position (length out_dim).
    #[inline]
    fn apply_into(&self, x: &[f32], y: &mut [f32], scratch: &mut [f32]) {
        match self {
            Correction::None => {}
            Correction::Dense(d) => {
                let r = d.rank;
                debug_assert!(scratch.len() >= r);
                debug_assert_eq!(d.a.len(), r * x.len());
                debug_assert_eq!(d.b.len(), y.len() * r);
                // Intermediate: A · x → scratch[r].
                let in_dim = x.len();
                for k in 0..r {
                    let a_row = &d.a[k * in_dim..(k + 1) * in_dim];
                    let mut acc = 0.0f32;
                    for i in 0..in_dim {
                        acc += a_row[i] * x[i];
                    }
                    scratch[k] = acc;
                }
                // y += alpha * B · intermediate.
                let scale = d.alpha;
                for o in 0..y.len() {
                    let b_row = &d.b[o * r..(o + 1) * r];
                    let mut acc = 0.0f32;
                    for k in 0..r {
                        acc += b_row[k] * scratch[k];
                    }
                    y[o] += scale * acc;
                }
            }
            Correction::Sparse(s) => {
                for n in 0..s.vals.len() {
                    let o = s.rows[n] as usize;
                    let i = s.cols[n] as usize;
                    y[o] += s.vals[n] * x[i];
                }
            }
            Correction::Full { mat, out_dim, in_dim } => {
                debug_assert_eq!(mat.len(), out_dim * in_dim);
                debug_assert_eq!(y.len(), *out_dim);
                debug_assert_eq!(x.len(), *in_dim);
                for o in 0..*out_dim {
                    let row = &mat[o * in_dim..(o + 1) * in_dim];
                    let mut acc = 0.0f32;
                    for i in 0..*in_dim {
                        acc += row[i] * x[i];
                    }
                    y[o] += acc;
                }
            }
        }
    }
}

// ─── Conv2d + linear with optional correction ───────────────────────────────

/// Conv2d producing [h×w×out_ch] output, with an optional per-output-channel
/// correction applied at each spatial position.
///
/// Mirrors `moka::conv2d_into` but threads a `Correction` through.
#[allow(clippy::too_many_arguments)]
fn conv2d_corrected_into(
    input: &[f32],
    h: usize,
    w: usize,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    weight: &[f32],
    bias: &[f32],
    patch: &mut [f32],
    out: &mut [f32],
    correction: &Correction<'_>,
    lora_scratch: &mut [f32],
) {
    let patch_len = k * k * in_ch;
    if k == 1 {
        for pos in 0..h * w {
            let pslice = &input[pos * in_ch..pos * in_ch + in_ch];
            let obase = pos * out_ch;
            for oc in 0..out_ch {
                let wbase = oc * in_ch;
                out[obase + oc] = bias[oc];
                for ic in 0..in_ch {
                    out[obase + oc] += weight[wbase + ic] * pslice[ic];
                }
            }
            // Apply correction for this spatial position.
            correction.apply_into(pslice, &mut out[obase..obase + out_ch], lora_scratch);
        }
        return;
    }

    let pad = k / 2;
    for y in 0..h {
        for x in 0..w {
            patch[..patch_len].fill(0.0);
            for ky in 0..k {
                let iy = y + ky;
                if iy < pad || iy >= h + pad {
                    continue;
                }
                let iy = iy - pad;
                for kx in 0..k {
                    let ix = x + kx;
                    if ix < pad || ix >= w + pad {
                        continue;
                    }
                    let ix = ix - pad;
                    let src = (iy * w + ix) * in_ch;
                    let dst = (ky * k + kx) * in_ch;
                    patch[dst..dst + in_ch].copy_from_slice(&input[src..src + in_ch]);
                }
            }

            let obase = (y * w + x) * out_ch;
            let pslice = &patch[..patch_len];
            for oc in 0..out_ch {
                let wbase = oc * patch_len;
                let mut acc = bias[oc];
                for i in 0..patch_len {
                    acc += weight[wbase + i] * pslice[i];
                }
                out[obase + oc] = acc;
            }
            // Apply correction for this spatial position.
            correction.apply_into(pslice, &mut out[obase..obase + out_ch], lora_scratch);
        }
    }
}

/// Linear layer with optional correction.
fn linear_corrected_into(
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
    correction: &Correction<'_>,
    lora_scratch: &mut [f32],
) {
    for o in 0..out_dim {
        let base = o * in_dim;
        let mut acc = bias[o];
        for i in 0..in_dim {
            acc += weight[base + i] * input[i];
        }
        out[o] = acc;
    }
    correction.apply_into(input, out, lora_scratch);
}

// ─── Corrected full forward pass ────────────────────────────────────────────

/// The correction bundle applied during [`forward_corrected_with_scratch`].
///
/// The PoC builds this once per quantization strategy (rank, calibration set)
/// and reuses it across all forward passes. Fields mirror the layer structure
/// of `MokaWeights`; `None` means no correction at that layer.
pub struct ForwardCorrections<'a> {
    pub stem: Correction<'a>,
    /// Flat list aligned with `MokaWeights::iter_layers()`'s block entries
    /// (reduce, first, [global.hidden, global.output], second, expand) per
    /// block — in the SAME order the forward pass evaluates them. Use
    /// [`MokaBlockCorrections`] to build this per-block.
    pub block_layers: Vec<Correction<'a>>,
    pub policy_conv: Correction<'a>,
    pub policy_linear: Correction<'a>,
    pub value_conv: Correction<'a>,
    pub value_hidden: Correction<'a>,
    pub value_output: Correction<'a>,
}

/// Run the full Moka forward pass with quantized weights + per-layer
/// corrections. Same arithmetic as `forward_with_scratch` but each conv/linear
/// output gets `+= correction(patch)`.
///
/// `lora_scratch` must be sized for the largest rank any dense correction uses
/// (sparse corrections don't use it). A safe size: the max rank across all
/// dense corrections, or `in_dim` of the largest layer if unsure.
///
/// Returns `(policy_logits[POLICY_MOVES], value_tanh)`.
#[allow(clippy::too_many_arguments)]
pub fn forward_corrected_with_scratch(
    weights: &MokaWeights,
    features: &[f32],
    scratch: &mut MokaScratch,
    corrections: &ForwardCorrections<'_>,
    lora_scratch: &mut [f32],
) -> ([f32; POLICY_MOVES], f32) {
    let (
        trunk, expand, hidden_a, hidden_b, head4, head2,
        patch, pooled, gh, gbias, value_h, policy,
    ) = scratch.lend_all();

    // Stem conv 3×3.
    let (stem_w, stem_b) = weights.stem_w();
    conv2d_corrected_into(
        features, BOARD_SIZE, BOARD_SIZE, INPUT_PLANES, TRUNK_CHANNELS, 3,
        stem_w, stem_b, patch, trunk,
        &corrections.stem, lora_scratch,
    );
    relu_inplace(&mut trunk[..BOARD_AREA * TRUNK_CHANNELS]);

    // Walk the block_layers corrections in lockstep with the residual blocks.
    let mut ci = 0usize;
    for block in weights.blocks_ref() {
        let (rw, rb) = block.reduce_w();
        conv2d_corrected_into(
            trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, BOTTLENECK_CHANNELS, 1,
            rw, rb, patch, hidden_a,
            &corrections.block_layers[ci], lora_scratch,
        );
        ci += 1;
        relu_inplace(hidden_a);

        let (fw, fb) = block.first_w();
        conv2d_corrected_into(
            hidden_a, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, 3,
            fw, fb, patch, hidden_b,
            &corrections.block_layers[ci], lora_scratch,
        );
        ci += 1;
        relu_inplace(hidden_b);

        if let Some(g) = block.global_ref() {
            global_mean_max_into(hidden_b, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, pooled);
            let (ghw, ghb) = g.hidden_w();
            let g_hidden_out = ghb.len();
            linear_corrected_into(
                pooled, BOTTLENECK_CHANNELS * 2, g_hidden_out,
                ghw, ghb, gh,
                &corrections.block_layers[ci], lora_scratch,
            );
            ci += 1;
            relu_inplace(&mut gh[..g_hidden_out]);
            let (gow, gob) = g.output_w();
            let g_out_out = gob.len();
            linear_corrected_into(
                gh, g_hidden_out, g_out_out,
                gow, gob, gbias,
                &corrections.block_layers[ci], lora_scratch,
            );
            ci += 1;
            for pos in 0..BOARD_AREA {
                let row = &mut hidden_b[pos * BOTTLENECK_CHANNELS..(pos + 1) * BOTTLENECK_CHANNELS];
                for c in 0..BOTTLENECK_CHANNELS {
                    row[c] += gbias[c];
                }
            }
        }

        let (sw, sb) = block.second_w();
        conv2d_corrected_into(
            hidden_b, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, 3,
            sw, sb, patch, hidden_a,
            &corrections.block_layers[ci], lora_scratch,
        );
        ci += 1;
        relu_inplace(hidden_a);
        let (ew, eb) = block.expand_w();
        conv2d_corrected_into(
            hidden_a, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, TRUNK_CHANNELS, 1,
            ew, eb, patch, expand,
            &corrections.block_layers[ci], lora_scratch,
        );
        ci += 1;

        for i in 0..BOARD_AREA * TRUNK_CHANNELS {
            let v = trunk[i] + expand[i];
            trunk[i] = if v < 0.0 { 0.0 } else { v };
        }
    }

    // Policy head.
    let (pc_w, pc_b) = weights.policy_conv_w();
    conv2d_corrected_into(
        trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, POLICY_CHANNELS, 1,
        pc_w, pc_b, patch, head4,
        &corrections.policy_conv, lora_scratch,
    );
    relu_inplace(head4);
    let (pl_w, pl_b) = weights.policy_linear_w();
    let policy_lin_in = pl_w.len() / pl_b.len();
    linear_corrected_into(
        head4, policy_lin_in, POLICY_MOVES,
        pl_w, pl_b, policy,
        &corrections.policy_linear, lora_scratch,
    );

    // Value head.
    let (vc_w, vc_b) = weights.value_conv_w();
    conv2d_corrected_into(
        trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, VALUE_CHANNELS, 1,
        vc_w, vc_b, patch, head2,
        &corrections.value_conv, lora_scratch,
    );
    relu_inplace(head2);
    let (vh_w, vh_b) = weights.value_hidden_w();
    let value_hidden_dim = vh_b.len();
    let value_hidden_in = vh_w.len() / value_hidden_dim;
    linear_corrected_into(
        head2, value_hidden_in, value_hidden_dim,
        vh_w, vh_b, value_h,
        &corrections.value_hidden, lora_scratch,
    );
    relu_inplace(&mut value_h[..value_hidden_dim]);
    let (vo_w, vo_b) = weights.value_output_w();
    let mut value_out = [0f32; 1];
    linear_corrected_into(
        value_h, value_hidden_dim, 1,
        vo_w, vo_b, &mut value_out,
        &corrections.value_output, lora_scratch,
    );

    let mut logits = [0f32; POLICY_MOVES];
    logits.copy_from_slice(&policy[..POLICY_MOVES]);
    (logits, value_out[0].tanh())
}

// ─── Activation collection (Strategy B proper calibration) ──────────────────
//
// The Issue 565 PoC's Strategy B (output-space data-aware SVD) was initially
// calibrated with truncated board features — which is a POOR approximation of
// the actual layer inputs (especially for conv layers where the input is a 3×3
// patch, not the full board). `forward_collecting_activations` runs the same
// arithmetic as `forward_with_scratch` but appends each layer's ACTUAL input
// vectors into `layer_inputs`, so the PoC can build a proper calibration set.
//
// `layer_inputs` must be pre-sized to `iter_layers().len()`. For conv layers,
// each spatial position produces one `in_dim`-length vector (the im2col patch).
// For linear layers, each forward produces one `in_dim`-length vector.

/// Conv2d that also collects the input patch for each spatial position.
#[allow(clippy::too_many_arguments)]
fn conv2d_collecting_into(
    input: &[f32],
    h: usize,
    w: usize,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    weight: &[f32],
    bias: &[f32],
    patch: &mut [f32],
    out: &mut [f32],
    collect: &mut Vec<f32>,
) {
    let patch_len = k * k * in_ch;
    if k == 1 {
        for pos in 0..h * w {
            let pslice = &input[pos * in_ch..pos * in_ch + in_ch];
            let obase = pos * out_ch;
            for oc in 0..out_ch {
                let wbase = oc * in_ch;
                let mut acc = bias[oc];
                for ic in 0..in_ch {
                    acc += weight[wbase + ic] * pslice[ic];
                }
                out[obase + oc] = acc;
            }
            collect.extend_from_slice(pslice);
        }
        return;
    }

    let pad = k / 2;
    for y in 0..h {
        for x in 0..w {
            patch[..patch_len].fill(0.0);
            for ky in 0..k {
                let iy = y + ky;
                if iy < pad || iy >= h + pad {
                    continue;
                }
                let iy = iy - pad;
                for kx in 0..k {
                    let ix = x + kx;
                    if ix < pad || ix >= w + pad {
                        continue;
                    }
                    let ix = ix - pad;
                    let src = (iy * w + ix) * in_ch;
                    let dst = (ky * k + kx) * in_ch;
                    patch[dst..dst + in_ch].copy_from_slice(&input[src..src + in_ch]);
                }
            }

            let obase = (y * w + x) * out_ch;
            let pslice = &patch[..patch_len];
            for oc in 0..out_ch {
                let wbase = oc * patch_len;
                let mut acc = bias[oc];
                for i in 0..patch_len {
                    acc += weight[wbase + i] * pslice[i];
                }
                out[obase + oc] = acc;
            }
            collect.extend_from_slice(pslice);
        }
    }
}

/// Linear layer that also collects the input vector.
fn linear_collecting_into(
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
    collect: &mut Vec<f32>,
) {
    for o in 0..out_dim {
        let base = o * in_dim;
        let mut acc = bias[o];
        for i in 0..in_dim {
            acc += weight[base + i] * input[i];
        }
        out[o] = acc;
    }
    collect.extend_from_slice(&input[..in_dim]);
}

/// Run the full Moka forward pass, collecting each layer's input vectors.
///
/// Same arithmetic as `forward_with_scratch`. `layer_inputs[i]` accumulates
/// the input vectors for layer `i` (in `iter_layers()` order): for conv layers,
/// `BOARD_AREA` vectors of length `in_dim` per call; for linear layers, 1 vector.
/// Each slot grows by `n_positions × in_dim` f32 values per call.
///
/// The collected data forms a column-major `[in_dim × n_cal]` matrix suitable
/// for `QuantErrorLora::from_error_data_aware`.
pub fn forward_collecting_activations(
    weights: &MokaWeights,
    features: &[f32],
    scratch: &mut MokaScratch,
    layer_inputs: &mut [Vec<f32>],
) -> ([f32; POLICY_MOVES], f32) {
    let (
        trunk, expand, hidden_a, hidden_b, head4, head2,
        patch, pooled, gh, gbias, value_h, policy,
    ) = scratch.lend_all();

    let mut li = 0usize; // layer_inputs index, walks in iter_layers() order

    // Stem conv 3×3.
    let (stem_w, stem_b) = weights.stem_w();
    conv2d_collecting_into(
        features, BOARD_SIZE, BOARD_SIZE, INPUT_PLANES, TRUNK_CHANNELS, 3,
        stem_w, stem_b, patch, trunk, &mut layer_inputs[li],
    );
    li += 1;
    relu_inplace(&mut trunk[..BOARD_AREA * TRUNK_CHANNELS]);

    for block in weights.blocks_ref() {
        let (rw, rb) = block.reduce_w();
        conv2d_collecting_into(
            trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, BOTTLENECK_CHANNELS, 1,
            rw, rb, patch, hidden_a, &mut layer_inputs[li],
        );
        li += 1;
        relu_inplace(hidden_a);

        let (fw, fb) = block.first_w();
        conv2d_collecting_into(
            hidden_a, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, 3,
            fw, fb, patch, hidden_b, &mut layer_inputs[li],
        );
        li += 1;
        relu_inplace(hidden_b);

        if let Some(g) = block.global_ref() {
            global_mean_max_into(hidden_b, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, pooled);
            let (ghw, ghb) = g.hidden_w();
            let g_hidden_out = ghb.len();
            linear_collecting_into(
                pooled, BOTTLENECK_CHANNELS * 2, g_hidden_out,
                ghw, ghb, gh, &mut layer_inputs[li],
            );
            li += 1;
            relu_inplace(&mut gh[..g_hidden_out]);
            let (gow, gob) = g.output_w();
            let g_out_out = gob.len();
            linear_collecting_into(
                gh, g_hidden_out, g_out_out,
                gow, gob, gbias, &mut layer_inputs[li],
            );
            li += 1;
            for pos in 0..BOARD_AREA {
                let row = &mut hidden_b[pos * BOTTLENECK_CHANNELS..(pos + 1) * BOTTLENECK_CHANNELS];
                for c in 0..BOTTLENECK_CHANNELS {
                    row[c] += gbias[c];
                }
            }
        }

        let (sw, sb) = block.second_w();
        conv2d_collecting_into(
            hidden_b, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, 3,
            sw, sb, patch, hidden_a, &mut layer_inputs[li],
        );
        li += 1;
        relu_inplace(hidden_a);
        let (ew, eb) = block.expand_w();
        conv2d_collecting_into(
            hidden_a, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, TRUNK_CHANNELS, 1,
            ew, eb, patch, expand, &mut layer_inputs[li],
        );
        li += 1;

        for i in 0..BOARD_AREA * TRUNK_CHANNELS {
            let v = trunk[i] + expand[i];
            trunk[i] = if v < 0.0 { 0.0 } else { v };
        }
    }

    // Policy head.
    let (pc_w, pc_b) = weights.policy_conv_w();
    conv2d_collecting_into(
        trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, POLICY_CHANNELS, 1,
        pc_w, pc_b, patch, head4, &mut layer_inputs[li],
    );
    li += 1;
    relu_inplace(head4);
    let (pl_w, pl_b) = weights.policy_linear_w();
    let policy_lin_in = pl_w.len() / pl_b.len();
    linear_collecting_into(
        head4, policy_lin_in, POLICY_MOVES,
        pl_w, pl_b, policy, &mut layer_inputs[li],
    );
    li += 1;

    // Value head.
    let (vc_w, vc_b) = weights.value_conv_w();
    conv2d_collecting_into(
        trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, VALUE_CHANNELS, 1,
        vc_w, vc_b, patch, head2, &mut layer_inputs[li],
    );
    li += 1;
    relu_inplace(head2);
    let (vh_w, vh_b) = weights.value_hidden_w();
    let value_hidden_dim = vh_b.len();
    let value_hidden_in = vh_w.len() / value_hidden_dim;
    linear_collecting_into(
        head2, value_hidden_in, value_hidden_dim,
        vh_w, vh_b, value_h, &mut layer_inputs[li],
    );
    li += 1;
    relu_inplace(&mut value_h[..value_hidden_dim]);
    let (vo_w, vo_b) = weights.value_output_w();
    let mut value_out = [0f32; 1];
    linear_collecting_into(
        value_h, value_hidden_dim, 1,
        vo_w, vo_b, &mut value_out, &mut layer_inputs[li],
    );

    let mut logits = [0f32; POLICY_MOVES];
    logits.copy_from_slice(&policy[..POLICY_MOVES]);
    (logits, value_out[0].tanh())
}
