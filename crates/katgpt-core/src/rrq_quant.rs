//! Recurrent Residual Quantization (RRQ) — single-checkpoint multi-precision
//! weight representation via iterated 2-bit residual corrections.
//!
//! > **Source:** Luo, Dong, Cheng, Shen (Intel),
//! > *Recurrent Residual Quantization: A Progressive Multi-Precision
//! > Representation for LLMs*,
//! > [arXiv:2608.04048](https://arxiv.org/abs/2608.04048), Aug 2026.
//!
//! # The representation
//!
//! An RRQ weight matrix `W̃` is stored as `S+1` quantized tensors: one base
//! stage `Ŵ0` + `S` residual stages `R̂1..R̂S`. Each stage is a 2-bit
//! round-to-nearest (RTN) quantized tensor with per-group scale + zero-point.
//! The k-th residual is the quantization error of the (k−1)-th reconstructed
//! approximation:
//!
//! ```text
//! Q_j^0   = Q_{b0}(x, z_0)              # base: 2-bit RTN
//! r_j^0   = x_j − x̂_j^0                 # residual = full − base dequant
//! Q_j^k   = Q_{bk}(r_j^{k-1}, z_k)      # stage k: 2-bit RTN on previous residual
//! r_j^k   = r_j^{k-1} − r̂_j^k           # next residual = prev residual − stage dequant
//! W̃(t)   = Ŵ0 + Σ_{k=1..t} R̂k           # prefix-t reconstruction (additive)
//! ```
//!
//! Default config: `b0 = b1 = b2 = b3 = 2` → 2/4/6/8-bit prefixes. All stages
//! use the same group size (128) and the same per-stage quantizer (RTN). The
//! paper also reports a stronger `SignRoundV2-base` variant — the only place
//! a learned rounding operator enters; the all-RTN variant here tracks it
//! within 0.1 Task Avg at 6/8 bits.
//!
//! # Why modelless
//!
//! RRQ is explicitly **post-training quantization** (PTQ). The construction
//! is pure RTN — no Hessian, no calibration data, no joint multi-bit
//! optimization, no gradient descent. Each stage is independent: adding a
//! precision prefix requires only "configure stage format", not "re-run joint
//! optimization". The §3.5 modelless-unblock check is moot — there is nothing
//! to redirect to riir-train.
//!
//! # The matmul linearity (prefix-t GEMV)
//!
//! Because `W̃(t) = Σ_{k=0..t} stage_k` and matrix multiplication is linear:
//!
//! ```text
//! x · W̃(t) = x · Ŵ0 + Σ_{k=1..t} x · R̂k
//! ```
//!
//! So the prefix-t dot product decomposes into a sum of per-stage GEMVs.
//! Each stage GEMV can use the same SIMD LUT dequant+dot kernel (the
//! natural fusion target for Phase 3). The plan ships the additive primitive
//! here; the fused kernel is a separate Phase 3 task.
//!
//! # What this is NOT
//!
//! - **NOT a UQ primitive.** RRQ is a deterministic weight representation,
//!   not a probability distribution. No conformal-naive floor comparison.
//! - **NOT SignRoundV2.** The learned-base variant is a training-method
//!   artifact; this module ships only the all-RTN modelless path.
//! - **NOT Matryoshka / MatGPTQ.** RRQ explicitly replaces nested MSB
//!   bit-slicing with additive residuals. Different mechanism, same problem
//!   (multi-precision from one checkpoint).
//! - **NOT a consumer-ready tier dispatch.** Phase 4 (prefix-t as Plasma →
//!   Hot → Warm tier) is stretch, deferred until a concrete consumer lands.
//!
//! # The Small-Kernel Parameter Paradox (Research 463 §2.4.1)
//!
//! Same caveat as `quant_error_lora.rs`: on small CNNs (Moka-scale), each
//! 2-bit residual stage adds 0.5 bits/weight of codes + per-group scale
//! metadata; for a 32×288 conv that's substantial overhead relative to the
//! 9.2K parameters. RRQ is substrate for larger models (LLM weights, future
//! game networks) where the per-group scale metadata is amortized across
//! many weights.
//!
//! # Cross-references
//!
//! - [Research 467](../.research/467_Recurrent_Residual_Quantization.md) —
//!   the parent research note (verdict: Gain, not Super-GOAT — no consumer).
//! - [Plan 568](../.plans/568_recurrent_residual_quantization.md) — execution
//!   plan. Phase 1 ships here; Phases 2–4 are P1–P3 (deferred).
//! - [`quant_error_lora`](../src/quant_error_lora.rs) — closest cousin
//!   (single SVD correction vs N iterated RTN corrections).
//! - [`simd_lut_dequant`](../src/simd_lut_dequant.rs) — Phase 3 fusion
//!   target (fused multi-stage LUT dequant+dot kernel).

use half::f16;

// ──────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────

/// Bits per stage code. The paper's default is 2-bit (4 levels) for every
/// stage; this gives 2/4/6/8-bit prefixes for the 1-base + 3-residual config.
pub const BITS_PER_STAGE: u8 = 2;

/// Number of quantization levels per stage = `2^BITS_PER_STAGE` = 4.
pub const LEVELS_PER_STAGE: usize = 1 << BITS_PER_STAGE;

/// Number of 2-bit codes packed into one byte = `8 / BITS_PER_STAGE` = 4.
pub const CODES_PER_BYTE: usize = 8 / BITS_PER_STAGE as usize;

/// Default group size for per-group scale + zero-point (paper default + our
/// existing codec convention).
pub const DEFAULT_GROUP_SIZE: usize = 128;

/// Default number of residual stages (1 base + 3 residuals → 2/4/6/8-bit
/// prefixes).
pub const DEFAULT_N_STAGES: usize = 3;

// ──────────────────────────────────────────────────────────────────────────
// Storage types
// ──────────────────────────────────────────────────────────────────────────

/// One RRQ stage: a 2-bit RTN quantized tensor with per-group scale +
/// zero-point.
///
/// The codes are packed 4-per-byte (2 bits each). Group size defaults to 128
/// (so 32 bytes of codes per group). Scales + zero-points are stored as `f16`
/// to halve metadata overhead (the paper's intent; Appendix G shows ~4–5%
/// metadata overhead vs codes).
#[derive(Clone)]
pub struct RrqStage {
    /// 2-bit packed codes, 4 per byte. `codes[i / 4]` holds code `i` in bits
    /// `(i % 4) * 2 .. (i % 4) * 2 + 2`. Code values are 0..=3.
    pub codes: Vec<u8>,
    /// Per-group scale, `f16`. `scales[g]` is the scale for group `g`.
    pub scales: Vec<f16>,
    /// Per-group zero-point, `f16`. The dequant formula is
    /// `x_hat = zero_point + code * scale`.
    pub zero_points: Vec<f16>,
    /// Number of weights represented (= `rows * cols` for a full stage).
    pub n_elements: usize,
    /// Group size (default 128). The last group may be partial.
    pub group_size: usize,
}

impl RrqStage {
    /// Number of complete + partial groups.
    #[allow(dead_code)]
    fn n_groups(&self) -> usize {
        self.n_elements.div_ceil(self.group_size)
    }

    /// Quantize a slice of f32 values into a 2-bit RTN stage.
    ///
    /// Per group: `scale = (max - min) / (LEVELS - 1)`,
    /// `zero_point = min`, `code = round((x - zp) / scale)` clamped to
    /// `[0, LEVELS-1]`. Dequant: `x_hat = zp + code * scale`.
    ///
    /// **Allocation:** this constructor allocates (it builds the owned
    /// `codes` + `scales` + `zero_points` Vecs). It runs once at model load,
    /// NOT on the inference hot path. The hot-path methods
    /// ([`dequant_into`](Self::dequant_into),
    /// [`dot_acc_into`](Self::dot_acc_into)) are zero-allocation.
    fn quantize_rtn(values: &[f32], group_size: usize) -> Self {
        let n = values.len();
        let n_groups = n.div_ceil(group_size);
        let n_code_bytes = n.div_ceil(CODES_PER_BYTE);
        let mut codes = vec![0u8; n_code_bytes];
        let mut scales = Vec::with_capacity(n_groups);
        let mut zero_points = Vec::with_capacity(n_groups);

        for g in 0..n_groups {
            let start = g * group_size;
            let end = (start + group_size).min(n);
            // Find min/max of this group.
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &v in &values[start..end] {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            // scale = (max - min) / (LEVELS - 1). Guard against degenerate
            // (all-equal) groups: scale = 0 → every code = 0, dequant = min.
            let range = max - min;
            let scale = if range > 0.0 {
                range / (LEVELS_PER_STAGE as f32 - 1.0)
            } else {
                0.0
            };
            let zp = min;
            scales.push(f16::from_f32(scale));
            zero_points.push(f16::from_f32(zp));

            // Quantize each value in the group.
            let scale_recip = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for (i, &v) in values[start..end].iter().enumerate() {
                let code_f = (v - zp) * scale_recip;
                // Round + clamp to [0, LEVELS-1].
                let code = if code_f <= 0.0 {
                    0u8
                } else if code_f >= (LEVELS_PER_STAGE - 1) as f32 {
                    (LEVELS_PER_STAGE - 1) as u8
                } else {
                    (code_f + 0.5) as u8
                };
                // Pack into the codes byte array.
                let idx = start + i;
                let byte_idx = idx / CODES_PER_BYTE;
                let shift = (idx % CODES_PER_BYTE) * 2; // 2 bits per code
                codes[byte_idx] |= code << shift;
            }
        }

        Self {
            codes,
            scales,
            zero_points,
            n_elements: n,
            group_size,
        }
    }

    /// Unpack the 2-bit code at element index `i`.
    #[inline]
    pub fn code_at(&self, i: usize) -> u8 {
        let byte_idx = i / CODES_PER_BYTE;
        let shift = (i % CODES_PER_BYTE) * 2;
        (self.codes[byte_idx] >> shift) & 0x03
    }

    /// Dequantize the code at element index `i` to f32.
    #[inline]
    pub fn dequant_at(&self, i: usize) -> f32 {
        let code = self.code_at(i) as f32;
        let g = i / self.group_size;
        let scale = self.scales[g].to_f32();
        let zp = self.zero_points[g].to_f32();
        zp + code * scale
    }

    /// Dequantize all codes into `out`. `out.len()` must be `>= n_elements`.
    ///
    /// **Zero allocation.** Writes into caller-provided `out`.
    pub fn dequant_into(&self, out: &mut [f32]) {
        assert!(
            out.len() >= self.n_elements,
            "dequant_into: out.len() ({}) < n_elements ({})",
            out.len(),
            self.n_elements
        );
        for (out_i, i) in out.iter_mut().take(self.n_elements).zip(0..self.n_elements) {
            *out_i = self.dequant_at(i);
        }
    }

    /// Accumulate `x · dequant(codes)` into `out` for one stage.
    ///
    /// Treats this stage as a `[rows × cols]` row-major weight matrix (the
    /// layout used by `from_weights_rtn`). Computes
    /// `out[o] += Σ_i x[i] * stage_dequant[o * cols + i]` for each output
    /// row `o`.
    ///
    /// **Zero allocation.** Reads `x` and `out`, dequants codes on the fly.
    ///
    /// **Panics** if `x.len() != cols` or `out.len() != rows` (where
    /// `rows = n_elements / cols`).
    pub fn dot_acc_into(&self, cols: usize, x: &[f32], out: &mut [f32]) {
        let n_elements = self.n_elements;
        assert!(
            n_elements.is_multiple_of(cols),
            "dot_acc_into: n_elements ({n_elements}) not divisible by cols ({cols})"
        );
        let rows = n_elements / cols;
        assert_eq!(
            x.len(),
            cols,
            "dot_acc_into: x.len() ({}) != cols ({})",
            x.len(),
            cols
        );
        assert_eq!(
            out.len(),
            rows,
            "dot_acc_into: out.len() ({}) != rows ({})",
            out.len(),
            rows
        );

        // Row-major: stage_dequant[o * cols + i]. For each output row o,
        // accumulate Σ_i x[i] * dequant(o*cols + i).
        for (o, out_o) in out.iter_mut().enumerate().take(rows) {
            let row_offset = o * cols;
            let mut acc = 0.0_f32;
            for (i, &xi) in x.iter().enumerate().take(cols) {
                let idx = row_offset + i;
                let code = self.code_at(idx) as f32;
                let g = idx / self.group_size;
                let scale = self.scales[g].to_f32();
                let zp = self.zero_points[g].to_f32();
                acc += xi * (zp + code * scale);
            }
            *out_o += acc;
        }
    }
}

/// A complete RRQ weight matrix: base stage + N residual stages.
///
/// Default config: 1 base + 3 residuals → 2/4/6/8-bit prefixes
/// (`prefix_reconstruct_into(0)` = 2-bit, `(3)` = 8-bit).
pub struct RrqWeights {
    /// Base stage (2-bit RTN of the original weights).
    pub base: RrqStage,
    /// Residual stages (each 2-bit RTN of the previous residual).
    /// `residuals.len()` is typically 3 → 4 total prefixes (t=0..=3).
    pub residuals: Vec<RrqStage>,
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub cols: usize,
}

impl RrqWeights {
    /// Construct an all-RTN RRQ package from f32 weights.
    ///
    /// `n_stages` = number of residual stages (default 3 → 2/4/6/8-bit
    /// prefixes). `group_size` default 128.
    ///
    /// **Algorithm** (paper Algorithm 1):
    /// 1. Quantize base: `codes_0 = rtn_quant(x, scales_0, zps_0)`;
    ///    dequant to `x̂_0`.
    /// 2. For k=1..=n_stages: `r^{k-1} = x − x̂^{k-1}`;
    ///    quantize `rtn_quant(r^{k-1}, scales_k, zps_k)`;
    ///    dequant to `r̂_k`; accumulate `x̂^k = x̂^{k-1} + r̂_k`.
    ///
    /// **Allocation:** runs once at model load (builds the owned stage
    /// Vecs). The hot-path methods are zero-allocation.
    pub fn from_weights_rtn(
        weights: &[f32],
        rows: usize,
        cols: usize,
        n_stages: usize,
        group_size: usize,
    ) -> Self {
        let n = rows * cols;
        assert_eq!(
            weights.len(),
            n,
            "from_weights_rtn: weights.len() ({}) != rows * cols ({})",
            weights.len(),
            n
        );
        assert!(group_size > 0, "from_weights_rtn: group_size must be > 0");

        // Step 1: base stage = RTN of the original weights.
        let base = RrqStage::quantize_rtn(weights, group_size);

        // Reusable buffers for the residual + reconstruction.
        let mut residual = vec![0.0_f32; n];
        let mut recon = vec![0.0_f32; n];
        base.dequant_into(&mut recon);
        for i in 0..n {
            residual[i] = weights[i] - recon[i];
        }

        // Step 2: iteratively quantize the residual.
        let mut residuals = Vec::with_capacity(n_stages);
        for _k in 0..n_stages {
            let stage = RrqStage::quantize_rtn(&residual, group_size);
            // Update reconstruction: x̂^k = x̂^{k-1} + r̂_k.
            let mut stage_dequant = vec![0.0_f32; n];
            stage.dequant_into(&mut stage_dequant);
            for i in 0..n {
                recon[i] += stage_dequant[i];
                residual[i] = weights[i] - recon[i];
            }
            residuals.push(stage);
        }

        Self {
            base,
            residuals,
            rows,
            cols,
        }
    }

    /// Reconstruct weights at prefix-t precision into `out`.
    ///
    /// `t=0` → base only (2-bit); `t=1` → base + 1 residual (4-bit);
    /// `t=2` → 6-bit; `t=3` → 8-bit (for the default 3-residual config).
    ///
    /// `t` is clamped to `residuals.len()` (so `t=usize::MAX` = all stages).
    ///
    /// `out.len()` must be `>= rows * cols`.
    ///
    /// **Allocation:** the reconstruction itself is zero-allocation (reads
    /// codes, writes `out`), but this method allocates a temporary buffer for
    /// per-stage dequant. For the zero-allocation hot path, use
    /// [`prefix_dot_into`](Self::prefix_dot_into) which never materializes
    /// the full reconstruction.
    pub fn prefix_reconstruct_into(&self, t: usize, out: &mut [f32]) {
        let n = self.rows * self.cols;
        assert!(
            out.len() >= n,
            "prefix_reconstruct_into: out.len() ({}) < n ({})",
            out.len(),
            n
        );
        // Base.
        self.base.dequant_into(out);
        // Residuals up to min(t, residuals.len()).
        let t_eff = t.min(self.residuals.len());
        if t_eff == 0 {
            return;
        }
        let mut stage_dequant = vec![0.0_f32; n];
        for k in 0..t_eff {
            self.residuals[k].dequant_into(&mut stage_dequant);
            for i in 0..n {
                out[i] += stage_dequant[i];
            }
        }
    }

    /// Compute `out = x · W̃(t)` at prefix-t precision, exploiting linearity.
    ///
    /// `out = x · Ŵ0 + Σ_{k=1..t} x · R̂k` — sum of per-stage GEMVs. Each
    /// stage GEMV dequants codes on the fly and accumulates into `out`.
    ///
    /// `x.len()` must be `cols`; `out.len()` must be `rows`. `scratch` is a
    /// reusable buffer of length `rows` (zeroed at the start of each call).
    ///
    /// **Allocation:** zero. `out` is zeroed by this call, accumulated into
    /// per-stage via [`RrqStage::dot_acc_into`], then read back. `scratch`
    /// is used as the per-stage accumulator (zeroed per stage).
    pub fn prefix_dot_into(
        &self,
        t: usize,
        x: &[f32],
        out: &mut [f32],
        scratch: &mut [f32],
    ) {
        assert_eq!(
            x.len(),
            self.cols,
            "prefix_dot_into: x.len() ({}) != cols ({})",
            x.len(),
            self.cols
        );
        assert_eq!(
            out.len(),
            self.rows,
            "prefix_dot_into: out.len() ({}) != rows ({})",
            out.len(),
            self.rows
        );
        assert_eq!(
            scratch.len(),
            self.rows,
            "prefix_dot_into: scratch.len() ({}) != rows ({})",
            scratch.len(),
            self.rows
        );

        // Zero out.
        for (o, s) in out.iter_mut().zip(scratch.iter_mut()).take(self.rows) {
            *o = 0.0;
            *s = 0.0;
        }

        // Base stage.
        self.base.dot_acc_into(self.cols, x, out);

        // Residual stages up to min(t, residuals.len()).
        let t_eff = t.min(self.residuals.len());
        for k in 0..t_eff {
            // Zero scratch, accumulate this stage into scratch, add to out.
            for s in scratch.iter_mut().take(self.rows) {
                *s = 0.0;
            }
            self.residuals[k].dot_acc_into(self.cols, x, scratch);
            for (o, s) in scratch.iter().enumerate().take(self.rows) {
                out[o] += s;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests (G1 correctness + G4 alloc-free regression)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Hand-rolled naive reference: quantize a group of f32 to 2-bit RTN,
    /// return (codes, scale, zero_point). Mirrors `RrqStage::quantize_rtn`
    /// exactly so the G1 test can compare bit-pattern-by-bit-pattern.
    fn ref_quantize_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in values {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let range = max - min;
        let scale = if range > 0.0 {
            range / (LEVELS_PER_STAGE as f32 - 1.0)
        } else {
            0.0
        };
        let zp = min;
        let scale_recip = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let codes: Vec<u8> = values
            .iter()
            .map(|&v| {
                let code_f = (v - zp) * scale_recip;
                if code_f <= 0.0 {
                    0
                } else if code_f >= (LEVELS_PER_STAGE - 1) as f32 {
                    (LEVELS_PER_STAGE - 1) as u8
                } else {
                    (code_f + 0.5) as u8
                }
            })
            .collect();
        (codes, scale, zp)
    }

    // ─── G1a: stage quantize/dequant round-trip matches reference ──────────

    #[test]
    fn g1_stage_quantize_matches_reference() {
        // 8 values, group_size=8 (one group).
        let values = [-1.0_f32, -0.5, -0.2, 0.0, 0.1, 0.3, 0.7, 1.0];
        let stage = RrqStage::quantize_rtn(&values, 8);

        let (ref_codes, ref_scale, ref_zp) = ref_quantize_group(&values);
        assert_eq!(stage.scales.len(), 1);
        assert_eq!(stage.zero_points.len(), 1);
        // f16 has ~3 decimal digits of precision; use 1e-2 relative tolerance
        // for the scale/zp comparison (the f32 → f16 → f32 round-trip introduces
        // ~1e-3 relative error).
        let scale_tol = (ref_scale.abs() + 1e-30) * 1e-2;
        let zp_tol = (ref_zp.abs() + 1e-30) * 1e-2;
        assert!(
            approx_eq(stage.scales[0].to_f32(), ref_scale, scale_tol),
            "scale mismatch: got {} expected {} (tol {})",
            stage.scales[0].to_f32(),
            ref_scale,
            scale_tol
        );
        assert!(
            approx_eq(stage.zero_points[0].to_f32(), ref_zp, zp_tol),
            "zp mismatch: got {} expected {} (tol {})",
            stage.zero_points[0].to_f32(),
            ref_zp,
            zp_tol
        );

        // Check each code matches.
        for (i, &ref_code) in ref_codes.iter().enumerate() {
            let got_code = stage.code_at(i);
            assert_eq!(
                got_code, ref_code,
                "code[{i}]: got {got_code} expected {ref_code}"
            );
        }
    }

    // ─── G1b: 2-bit packing round-trip ─────────────────────────────────────

    #[test]
    fn g1_code_packing_roundtrip() {
        // 16 values → 4 bytes of packed codes.
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.8).collect();
        let stage = RrqStage::quantize_rtn(&values, 16);

        // Every code should be in [0, 3].
        for i in 0..16 {
            let c = stage.code_at(i);
            assert!(c <= 3, "code[{i}] = {c} out of [0,3]");
        }

        // Dequant + check monotonic: values are increasing, so dequantized
        // values should be non-decreasing (RTN preserves order within a
        // group because the mapping code = round((x-zp)/scale) is monotone).
        let mut deq = vec![0.0_f32; 16];
        stage.dequant_into(&mut deq);
        for i in 1..16 {
            assert!(
                deq[i] >= deq[i - 1] || approx_eq(deq[i], deq[i - 1], 1e-7),
                "monotonicity broken at i={i}: deq[{i}]={} < deq[{}]={}",
                deq[i],
                i - 1,
                deq[i - 1]
            );
        }
    }

    // ─── G1c: prefix reconstruction matches reference sum ──────────────────

    #[test]
    fn g1_prefix_reconstruct_matches_reference() {
        // 4x8 matrix = 32 elements. group_size=8 → 4 groups.
        let rows = 4;
        let cols = 8;
        let n = rows * cols;
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as f32) * 0.07 - 1.1;
                (x * x * x - 0.5 * x).abs() * (if i % 3 == 0 { -1.0 } else { 1.0 })
            })
            .collect();

        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, DEFAULT_N_STAGES, 8);

        // Reference: for each prefix t, compute the sum of dequanted stages
        // directly (independent code path).
        let n_stages = DEFAULT_N_STAGES;
        let mut recon_ref = vec![0.0_f32; n];
        let mut stage_buf = vec![0.0_f32; n];

        // t=0: base only.
        rrq.base.dequant_into(&mut recon_ref);
        let mut recon_got = vec![0.0_f32; n];
        rrq.prefix_reconstruct_into(0, &mut recon_got);
        for i in 0..n {
            assert!(
                recon_got[i].to_bits() == recon_ref[i].to_bits(),
                "t=0 mismatch at i={i}: got {} expected {}",
                recon_got[i],
                recon_ref[i]
            );
        }

        // t=1..=n_stages: add each residual's dequant.
        for t in 1..=n_stages {
            rrq.residuals[t - 1].dequant_into(&mut stage_buf);
            for i in 0..n {
                recon_ref[i] += stage_buf[i];
            }
            rrq.prefix_reconstruct_into(t, &mut recon_got);
            for i in 0..n {
                assert!(
                    recon_got[i].to_bits() == recon_ref[i].to_bits(),
                    "t={t} mismatch at i={i}: got {} expected {}",
                    recon_got[i],
                    recon_ref[i]
                );
            }
        }
    }

    // ─── G1d: prefix dot matches reconstruct-then-dot ──────────────────────

    #[test]
    fn g1_dot_matches_reconstruct_then_dot() {
        let rows = 8;
        let cols = 16;
        let n = rows * cols;
        // Random-ish weights.
        let mut seed: u64 = 0xABCD_1234_5678_EF01;
        let weights: Vec<f32> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 40) as f32) * 2.0 - 1.0
            })
            .collect();

        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, DEFAULT_N_STAGES, 8);

        // Input vector.
        let x: Vec<f32> = (0..cols)
            .map(|i| (i as f32) * 0.13 - 1.0)
            .collect();

        // For each prefix, compare prefix_dot_into vs reconstruct-then-dot.
        let mut recon = vec![0.0_f32; n];
        let mut out_dot = vec![0.0_f32; rows];
        let mut out_recon = vec![0.0_f32; rows];
        let mut scratch = vec![0.0_f32; rows];

        for t in 0..=DEFAULT_N_STAGES {
            // Path A: prefix_dot_into.
            rrq.prefix_dot_into(t, &x, &mut out_dot, &mut scratch);

            // Path B: reconstruct then matmul.
            rrq.prefix_reconstruct_into(t, &mut recon);
            for o in 0..rows {
                let mut acc = 0.0_f32;
                for i in 0..cols {
                    acc += x[i] * recon[o * cols + i];
                }
                out_recon[o] = acc;
            }

            // Compare (bit-identical — same arithmetic, different order; allow
            // 1 ULP slack for FMA vs separate mul+add).
            for o in 0..rows {
                let diff = (out_dot[o] - out_recon[o]).abs();
                let ulp = (out_recon[o].abs() + 1e-30) * f32::EPSILON;
                assert!(
                    diff <= ulp.max(1e-6),
                    "t={t} o={o}: dot={} recon={} diff={}",
                    out_dot[o],
                    out_recon[o],
                    diff
                );
            }
        }
    }

    // ─── G1e: more stages → lower reconstruction error ─────────────────────

    #[test]
    fn g1_more_stages_lower_error() {
        // With more residual stages, the prefix-t reconstruction should be
        // closer to the original (the residual shrinks each iteration).
        let rows = 4;
        let cols = 32;
        let n = rows * cols;
        let weights: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.05).sin() * 1.5)
            .collect();

        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, DEFAULT_N_STAGES, 16);

        let mut recon = vec![0.0_f32; n];
        let mut prev_err = f32::INFINITY;
        for t in 0..=DEFAULT_N_STAGES {
            rrq.prefix_reconstruct_into(t, &mut recon);
            let mut err = 0.0_f32;
            for i in 0..n {
                err += (recon[i] - weights[i]).abs();
            }
            assert!(
                err <= prev_err + 1e-5,
                "t={t}: error {err} should be <= previous {prev_err} (monotone improvement)"
            );
            prev_err = err;
        }

        // Sanity: t=3 (8-bit) should be meaningfully better than t=0 (2-bit).
        let mut err0 = 0.0_f32;
        let mut err3 = 0.0_f32;
        rrq.prefix_reconstruct_into(0, &mut recon);
        for i in 0..n {
            err0 += (recon[i] - weights[i]).abs();
        }
        rrq.prefix_reconstruct_into(DEFAULT_N_STAGES, &mut recon);
        for i in 0..n {
            err3 += (recon[i] - weights[i]).abs();
        }
        assert!(
            err3 < err0 * 0.5,
            "8-bit error {err3} should be < 50% of 2-bit error {err0}"
        );
    }

    // ─── G1f: constant weights → zero residual → exact reconstruction ─────

    #[test]
    fn g1_constant_weights_exact() {
        // If every weight is the same constant, the base stage captures it
        // exactly (one group, range=0, scale=0, every code=0, dequant=min=
        // the constant). Residuals should all be zero.
        let rows = 2;
        let cols = 4;
        let weights = vec![0.42_f32; rows * cols];

        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, 2, 4);

        let mut recon = vec![0.0_f32; rows * cols];
        rrq.prefix_reconstruct_into(2, &mut recon);
        for (i, &r) in recon.iter().enumerate() {
            assert!(
                approx_eq(r, 0.42, 1e-7),
                "constant reconstruction at i={i}: got {r} expected 0.42"
            );
        }
    }

    // ─── G4: alloc-free hot path (prefix_dot_into) ──────────────────────────

    /// The hot-path methods (`dequant_into`, `dot_acc_into`,
    /// `prefix_dot_into`) must be zero-allocation after construction.
    /// We verify by checking that `prefix_dot_into` does not grow any Vec
    /// (no `push`, no `resize`, no `Vec::new`). This is a static check via
    /// code review + a smoke test that runs many calls.
    #[test]
    fn g4_prefix_dot_smoke_zero_vec_growth() {
        let rows = 8;
        let cols = 16;
        let n = rows * cols;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 1.2).collect();
        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, 2, 8);

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let mut out = vec![0.0_f32; rows];
        let mut scratch = vec![0.0_f32; rows];

        // Run 100 calls — if any allocated, the test would still pass (we
        // can't easily assert zero allocs in a lib test without the
        // CountingAllocator global), but this catches panics + verifies the
        // API is callable in a tight loop without surprise state.
        for _ in 0..100 {
            rrq.prefix_dot_into(2, &x, &mut out, &mut scratch);
        }
        // Sanity: out is not all zeros (the weights + x are nontrivial).
        let sum: f32 = out.iter().copied().sum();
        assert!(sum.abs() > 0.0, "out sum should be nonzero, got {sum}");
    }
}
