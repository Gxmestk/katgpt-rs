//! Bit-plane packed ternary weights with group-wise f16 scale (Issue 578).
//!
//! The `Q2_0_g128` container. Ternary `{-1, 0, +1}` like [`crate::TernaryWeights`],
//! but the scale is per **128-weight group** (as in [`crate::BinaryWeights`])
//! instead of per row.
//!
//! ## Why a third tier
//!
//! | Tier | Alphabet | Scale | Holds `Q2_0_g128`? |
//! |---|---|---|---|
//! | `TernaryWeights` | `{-1, 0, +1}` | per row, f32 | no — wrong scale granularity |
//! | `BinaryWeights` | `{-1, +1}` | per 128, f16 | no — no zero state |
//! | **`TernaryGroupWeights`** | `{-1, 0, +1}` | per 128, f16 | **yes** |
//!
//! `BinaryWeights::from_ternary_no_zeros` cannot bridge the gap: it returns
//! `None` the moment any weight is zero, and ternary sparsity is the whole
//! point of the format.
//!
//! ## Footprint
//!
//! Two bit-planes = 2 bits/weight, plus 16 bits per 128 weights = **2.125
//! bits/weight**. At 27B params that is 7.17 GB — matching Ternary-Bonsai-27B's
//! ~7.2 GB deployed size, so loading a `Q2_0_g128` tensor into this container is
//! a *repack*, not an expansion.
//!
//! ## Group/block alignment
//!
//! [`GROUP_SIZE`] is 128 and blocks are 64 bits, so **every group spans exactly
//! two whole `u64` blocks**. Kernels rely on this: a group boundary is never in
//! the middle of a block, so the scale can be switched between blocks without
//! splitting a word.

use crate::GROUP_SIZE;
use half::f16;

/// Ternary `{-1, 0, +1}` bit-plane weights with a per-128-weight f16 scale.
///
/// 64 weights per block stored as two `u64` bitmasks:
/// - `pos_bits[block]` bit k set → `weight[row][k] = +1`
/// - `neg_bits[block]` bit k set → `weight[row][k] = -1`
/// - both clear → weight = 0 (implicit, no storage)
///
/// **Representation invariant:** `pos_bits[i] & neg_bits[i] == 0` for every `i`.
/// A weight cannot be both `+1` and `-1`. See [`Self::invariant_holds`].
///
/// `group_scale[r * groups_per_row + g]` rescales group `g` of row `r`.
#[cfg(feature = "ternary_group_scale")]
#[derive(Clone, Debug)]
pub struct TernaryGroupWeights {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,       // cols.div_ceil(64)
    pub groups_per_row: usize, // cols.div_ceil(GROUP_SIZE)
    pub pos_bits: Vec<u64>,    // [rows * blocks64]
    pub neg_bits: Vec<u64>,    // [rows * blocks64]
    pub group_scale: Vec<f16>, // [rows * groups_per_row]
}

/// Hook for GPU-accelerated ternary matvec (Issue 599 unblock path).
///
/// When provided to `forward_qwen_deltanet_ternary_with_hook`, replaces
/// `simd_ternary_group_matvec_parallel` for each projection. The GPU
/// implementation (riir-gpu `GemvTernaryCubeCL`) uploads weights once at
/// construction and dispatches CubeCL kernels per matvec call.
///
/// The contract is identical to `simd_ternary_group_matvec_parallel`:
/// `y = w @ x` where `x.len() == w.cols` and `y.len() == w.rows`.
///
/// Thread-safety: implementations must be `Send + Sync` because the forward
/// may be called from any thread. The GPU implementation holds pre-uploaded
/// handles keyed by the `TernaryGroupWeights` pointer identity (the weights
/// themselves are immutable after load).
#[cfg(feature = "ternary_group_scale")]
pub trait TernaryMatvecHook: Send + Sync {
    /// `y = w @ x` — same contract as `simd_ternary_group_matvec_parallel`.
    ///
    /// Implementations SHOULD pre-upload weights at construction time and
    /// cache the GPU handles by `(pos_bits.as_ptr(), neg_bits.as_ptr())`.
    /// The first call for a new weight matrix uploads; subsequent calls hit
    /// the cache.
    fn matvec(&self, w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]);
}

/// Hook for fused FFN dispatch (Issue 601).
///
/// When present, the forward path calls `ffn()` for the SwiGLU MLP portion
/// instead of 3 separate `matvec` calls. This enables the GPU implementation
/// to chain gate+up GEMVs + SwiGLU + down GEMV in a single command buffer,
/// eliminating per-matvec sync readback (2.574× speedup, Issue 600).
///
/// `out` MAY alias `x` (the forward path saves the residual before calling).
#[cfg(feature = "ternary_group_scale")]
pub trait TernaryFfnHook: Send + Sync {
    /// Fused FFN: `out = down @ swiglu(gate @ x, up @ x)`.
    ///
    /// - `gate_w`: [mlp_dim × n_embd]
    /// - `up_w`: [mlp_dim × n_embd]
    /// - `down_w`: [n_embd × mlp_dim]
    /// - `x`: [n_embd] — RMSNorm'd input
    /// - `out`: [n_embd] — FFN output (pre-residual-add)
    fn ffn(
        &self,
        gate_w: &TernaryGroupWeights,
        up_w: &TernaryGroupWeights,
        down_w: &TernaryGroupWeights,
        x: &[f32],
        out: &mut [f32],
    );
}

#[cfg(feature = "ternary_group_scale")]
impl TernaryGroupWeights {
    /// Create all-zero weights with unit group scale.
    pub fn new(rows: usize, cols: usize) -> Self {
        let blocks64 = cols.div_ceil(64);
        let groups_per_row = cols.div_ceil(GROUP_SIZE);
        Self {
            rows,
            cols,
            blocks64,
            groups_per_row,
            pos_bits: vec![0u64; rows * blocks64],
            neg_bits: vec![0u64; rows * blocks64],
            group_scale: vec![f16::ONE; rows * groups_per_row],
        }
    }

    /// Set the ternary value at `(row, col)`. Panics if out of bounds or if
    /// `value` is not in `{-1, 0, +1}`.
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        assert!(
            (-1..=1).contains(&value),
            "ternary value must be -1, 0, or +1"
        );
        let idx = row * self.blocks64 + (col >> 6);
        let mask = 1u64 << (col & 63);
        // Clear both planes first, then set at most one — keeps the
        // pos & neg == 0 invariant by construction.
        self.pos_bits[idx] &= !mask;
        self.neg_bits[idx] &= !mask;
        match value {
            1 => self.pos_bits[idx] |= mask,
            -1 => self.neg_bits[idx] |= mask,
            _ => {}
        }
    }

    /// Get the ternary value at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> i8 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let idx = row * self.blocks64 + (col >> 6);
        let mask = 1u64 << (col & 63);
        let pos = (self.pos_bits[idx] & mask) != 0;
        let neg = (self.neg_bits[idx] & mask) != 0;
        pos as i8 - neg as i8
    }

    /// Scale applied to group `g` of row `r`.
    #[inline]
    pub fn scale_at(&self, row: usize, group: usize) -> f32 {
        self.group_scale[row * self.groups_per_row + group].to_f32()
    }

    /// Set the scale for group `g` of row `r`.
    #[inline]
    pub fn set_scale(&mut self, row: usize, group: usize, scale: f32) {
        self.group_scale[row * self.groups_per_row + group] = f16::from_f32(scale);
    }

    /// Check the representation invariant: no weight is both `+1` and `-1`.
    ///
    /// G1 gate helper — a loader that mis-parses `Q2_0_g128` will typically
    /// violate this, so it is a cheap corruption check after loading.
    pub fn invariant_holds(&self) -> bool {
        self.pos_bits
            .iter()
            .zip(&self.neg_bits)
            .all(|(p, n)| p & n == 0)
    }

    /// Widen row-scale ternary weights into the group-scale tier.
    ///
    /// Lossless in the weights (bit-planes are copied verbatim) and lossless in
    /// the scale up to f32→f16 rounding: each row's single `row_scale` is
    /// broadcast across that row's groups.
    ///
    /// This is the direction [`crate::BinaryWeights::from_ternary_no_zeros`]
    /// cannot go — it drops the zero state and returns `None`. Widening always
    /// succeeds, so it returns `Self`, not `Option<Self>`.
    #[cfg(feature = "plasma_path")]
    pub fn from_ternary(tw: &crate::TernaryWeights) -> Self {
        let mut out = Self::new(tw.rows, tw.cols);
        out.pos_bits.copy_from_slice(&tw.pos_bits);
        out.neg_bits.copy_from_slice(&tw.neg_bits);
        for r in 0..tw.rows {
            let scale = f16::from_f32(tw.row_scale[r]);
            let base = r * out.groups_per_row;
            out.group_scale[base..base + out.groups_per_row].fill(scale);
        }
        out
    }

    /// Quantize f32 weights to ternary with **group-wise** error compensation.
    ///
    /// Mirrors [`crate::TernaryWeights::quantize_from_f32`], but the scale and
    /// the error carry reset per 128-weight group rather than per row — finer
    /// error compensation, which is the reason `Q2_0_g128` exists.
    ///
    /// For each group:
    /// ```text
    /// scale     = mean(|w|) over the group
    /// threshold = 0.5 * scale
    /// adjusted  = value + carry
    ///   adjusted >  threshold -> +1
    ///   adjusted < -threshold -> -1
    ///   otherwise             ->  0
    /// carry     = adjusted - q * scale
    /// ```
    pub fn quantize_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(weights.len(), rows * cols, "weights slice must be rows*cols");
        let mut out = Self::new(rows, cols);

        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let row_base = r * out.blocks64;
            let group_base = r * out.groups_per_row;

            for g in 0..out.groups_per_row {
                let g_start = g * GROUP_SIZE;
                let g_end = (g_start + GROUP_SIZE).min(cols);
                let group = &row[g_start..g_end];

                // scale = mean(|group|); guard the all-zero group so the
                // threshold stays finite and quantization yields all zeros.
                let abs_sum: f32 = group.iter().map(|v| v.abs()).sum();
                let scale = match abs_sum > 0.0 {
                    true => abs_sum / group.len() as f32,
                    false => 1.0,
                };
                out.group_scale[group_base + g] = f16::from_f32(scale);
                // Quantize against the f16-rounded scale the kernel will
                // actually apply, not the f32 ideal — otherwise the carry
                // compensates for an error the forward pass never makes.
                let scale = out.group_scale[group_base + g].to_f32();

                let threshold = 0.5 * scale;
                let mut carry = 0.0f32;
                for (i, &val) in group.iter().enumerate() {
                    let adjusted = val + carry;
                    let q = match adjusted {
                        a if a > threshold => 1i8,
                        a if a < -threshold => -1i8,
                        _ => 0i8,
                    };
                    let col = g_start + i;
                    let idx = row_base + (col >> 6);
                    let mask = 1u64 << (col & 63);
                    // Branch-free plane writes; at most one bit is set, so the
                    // pos & neg == 0 invariant holds by construction.
                    out.pos_bits[idx] |= ((q == 1) as u64) * mask;
                    out.neg_bits[idx] |= ((q == -1) as u64) * mask;
                    carry = adjusted - (q as f32 * scale);
                }
            }
        }

        out
    }

    /// Checksum over all values: `Σ_r Σ_g scale[r,g] * (popcount(pos) - popcount(neg))`
    /// within that group. Used for cross-implementation verification.
    pub fn checksum(&self) -> f32 {
        let mut total = 0.0f32;
        for r in 0..self.rows {
            let row_base = r * self.blocks64;
            let group_base = r * self.groups_per_row;
            for g in 0..self.groups_per_row {
                // GROUP_SIZE == 2 * 64, so a group is exactly blocks [2g, 2g+2)
                // — clamped for a trailing partial group.
                let b_start = g * (GROUP_SIZE / 64);
                let b_end = (b_start + GROUP_SIZE / 64).min(self.blocks64);
                let mut sum: i32 = 0;
                for b in b_start..b_end {
                    let idx = row_base + b;
                    sum += self.pos_bits[idx].count_ones() as i32;
                    sum -= self.neg_bits[idx].count_ones() as i32;
                }
                total += self.group_scale[group_base + g].to_f32() * sum as f32;
            }
        }
        total
    }

    /// Bytes of weight payload (bit-planes + scales), excluding `Vec` overhead.
    ///
    /// `2 bits/weight + 16 bits per GROUP_SIZE weights` = 2.125 bits/weight at
    /// `GROUP_SIZE = 128`.
    pub fn encoded_bytes(&self) -> usize {
        self.pos_bits.len() * 8 + self.neg_bits.len() * 8 + self.group_scale.len() * 2
    }
}
