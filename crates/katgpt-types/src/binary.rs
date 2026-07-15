//! Binary bit-plane packed weights — the Plasma tier (Issue 145, Bonsai 27B).
//!
//! Strict subset of ternary encoding: each weight is `{-1, +1}` (no zero state).
//! `sign_bits[block]` bit k set → `weight[row][k] = +1`; clear → `-1`.
//! Equivalent to ternary with `pos_bits = sign_bits` and `neg_bits = !sign_bits`.
//!
//! Group-wise FP16 scale (one `f16` per 128 weights) gives finer error
//! compensation than ternary's single `row_scale`, at 0.125 bits/weight
//! overhead vs ternary's 0.5 bits (row-wise f32).

use half::f16;

/// Binary `{-1, +1}` bit-plane packed weights.
///
/// 64 weights per block stored as one `u64` sign bitmask:
/// - bit set → weight = +1
/// - bit clear → weight = -1
///
/// `group_scale[r * groups_per_row + g]` (f16) rescales each 128-weight group.
/// Memory: 1 bit/weight + 16 bits per 128 weights = 1.125 bits/weight.
#[cfg(feature = "binary_plasma")]
#[derive(Clone, Debug)]
pub struct BinaryWeights {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,        // (cols + 63) / 64
    pub groups_per_row: usize,  // (cols + 127) / 128
    pub sign_bits: Vec<u64>,    // [rows * blocks64]
    pub group_scale: Vec<f16>,  // [rows * groups_per_row]
}

/// Group size for group-wise scaling (Bonsai default).
pub const GROUP_SIZE: usize = 128;

#[cfg(feature = "binary_plasma")]
impl BinaryWeights {
    /// Create zeroed (all +1) binary weights with unit scale.
    pub fn new(rows: usize, cols: usize) -> Self {
        let blocks64 = cols.div_ceil(64);
        let groups_per_row = cols.div_ceil(GROUP_SIZE);
        Self {
            rows,
            cols,
            blocks64,
            groups_per_row,
            // all-ones = every weight is +1 (bit set → +1)
            sign_bits: vec![u64::MAX; rows * blocks64],
            group_scale: vec![f16::ONE; rows * groups_per_row],
        }
    }

    /// Set a single binary value at (row, col). `value` must be -1 or +1.
    /// bit set → +1, bit clear → -1.
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        assert!(value == 1 || value == -1, "binary value must be -1 or +1");
        let block = col >> 6;
        let bit = col & 63;
        let mask = 1u64 << bit;
        let idx = row * self.blocks64 + block;
        if value == 1 {
            self.sign_bits[idx] |= mask;
        } else {
            self.sign_bits[idx] &= !mask;
        }
    }

    /// Get the binary value at (row, col): +1 if bit set, -1 if clear.
    pub fn get(&self, row: usize, col: usize) -> i8 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let block = col >> 6;
        let bit = col & 63;
        let mask = 1u64 << bit;
        let idx = row * self.blocks64 + block;
        if (self.sign_bits[idx] & mask) != 0 {
            1
        } else {
            -1
        }
    }

    /// Construct binary weights from an existing ternary weight matrix,
    /// assuming the ternary weights have NO zeros (strict binary subset).
    ///
    /// Every ternary zero is forced to the nearest sign (preserves the
    /// ternary `pos_bits` directly; `neg_bits` is ignored since it must
    /// equal `!pos_bits` for a valid binary subset). Group scale is set
    /// uniformly to the ternary `row_scale` — this makes the binary matvec
    /// bit-identical to the ternary matvec (used by the T0.1 property test).
    ///
    /// Returns `None` if the ternary weights contain zeros (not a binary subset).
    ///
    /// Gated behind `plasma_path` in addition to `binary_plasma` because it
    /// references `TernaryWeights` (the ternary substrate). A binary-only
    /// consumer (e.g. WASM edge with `binary_plasma` alone) has no ternary
    /// weights to convert from; this cross-format bridge is only meaningful
    /// when both formats are available.
    #[cfg(feature = "plasma_path")]
    pub fn from_ternary_no_zeros(
        ternary: &crate::TernaryWeights,
    ) -> Option<Self> {
        let n_words = ternary.rows * ternary.blocks64;
        for i in 0..n_words {
            // Binary subset invariant: pos XOR neg must be all-ones (no both-clear = zero).
            if (ternary.pos_bits[i] ^ ternary.neg_bits[i]) != u64::MAX {
                return None;
            }
        }
        let mut bw = Self::new(ternary.rows, ternary.cols);
        // sign_bits = pos_bits (bit set → +1, matching ternary pos convention)
        bw.sign_bits = ternary.pos_bits.clone();
        // Uniform group scale = ternary row_scale, broadcast across all groups in each row.
        for r in 0..ternary.rows {
            let scale_f16 = f16::from_f32(ternary.row_scale[r]);
            for g in 0..bw.groups_per_row {
                bw.group_scale[r * bw.groups_per_row + g] = scale_f16;
            }
        }
        Some(bw)
    }

    /// Quantize f32 weights to binary `{-1, +1}` with group-wise error
    /// compensation (Bonsai-style PTQ, group size 128).
    ///
    /// For each group g in each row r:
    ///   scale = mean(|weights in group|)
    ///   for each weight: adjusted = value + carry
    ///     q = sign(adjusted)  (no zero branch — binary has no zero state)
    ///     carry = adjusted - (q * scale)
    ///   group_scale[g] = scale
    ///
    /// This is modelless (deterministic, no training). The sign-threshold
    /// with error compensation mirrors Ciot's row-wise scheme but applied
    /// per 128-weight group for finer fidelity.
    pub fn quantize_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(weights.len(), rows * cols, "weights slice must be rows*cols");
        let mut bw = Self::new(rows, cols);
        let groups_per_row = bw.groups_per_row;

        for r in 0..rows {
            let row_base = r * cols;
            let sign_base = r * bw.blocks64;
            let group_base = r * groups_per_row;

            for g in 0..groups_per_row {
                let g_start = g * GROUP_SIZE;
                let g_end = (g_start + GROUP_SIZE).min(cols);
                let group = &weights[row_base + g_start..row_base + g_end];
                let group_len = group.len();

                // scale = mean(|group|)
                let abs_sum: f32 = group.iter().map(|w| w.abs()).sum();
                let scale = if abs_sum > 0.0 {
                    abs_sum / group_len as f32
                } else {
                    1.0
                };
                bw.group_scale[group_base + g] = f16::from_f32(scale);

                let mut carry = 0.0f32;
                for (i, &val) in group.iter().enumerate() {
                    let adjusted = val + carry;
                    // Binary: sign only, no zero state. sign(0) = +1 by convention.
                    let q: i8 = if adjusted >= 0.0 { 1 } else { -1 };
                    let col = g_start + i;
                    let block = col >> 6;
                    let bit = col & 63;
                    let mask = 1u64 << bit;
                    let idx = sign_base + block;
                    if q == 1 {
                        bw.sign_bits[idx] |= mask;
                    } else {
                        bw.sign_bits[idx] &= !mask;
                    }
                    carry = adjusted - (q as f32 * scale);
                }
            }
        }

        bw
    }

    /// Compute a checksum over all values (sum of group-scaled row sums).
    /// Used for cross-implementation verification.
    pub fn checksum(&self) -> f32 {
        let mut total = 0.0f32;
        for r in 0..self.rows {
            let mut row_sum = 0.0f32;
            let sign_base = r * self.blocks64;
            for g in 0..self.groups_per_row {
                let g_start = g * GROUP_SIZE;
                let g_end = (g_start + GROUP_SIZE).min(self.cols);
                let mut group_sign_sum: i32 = 0;
                for col in g_start..g_end {
                    let block = col >> 6;
                    let bit = col & 63;
                    let mask = 1u64 << bit;
                    let sign = if (self.sign_bits[sign_base + block] & mask) != 0 {
                        1
                    } else {
                        -1
                    };
                    group_sign_sum += sign;
                }
                let scale = self.group_scale[r * self.groups_per_row + g].to_f32();
                row_sum += scale * group_sign_sum as f32;
            }
            total += row_sum;
        }
        total
    }

    /// Byte size of the encoded representation (excluding the 20-byte file header).
    /// Used by the G2 storage GOAT gate.
    pub fn encoded_bytes(&self) -> usize {
        let scale_bytes = self.group_scale.len() * 2; // f16 = 2 bytes
        let sign_bytes = self.sign_bits.len() * 8; // u64 = 8 bytes
        scale_bytes + sign_bytes
    }
}

#[cfg(test)]
#[cfg(feature = "binary_plasma")]
mod tests {
    use super::*;

    #[test]
    fn test_new_all_positive() {
        let bw = BinaryWeights::new(2, 64);
        assert_eq!(bw.rows, 2);
        assert_eq!(bw.cols, 64);
        assert_eq!(bw.blocks64, 1);
        assert_eq!(bw.groups_per_row, 1);
        // All bits set → all +1
        assert_eq!(bw.get(0, 0), 1);
        assert_eq!(bw.get(0, 63), 1);
        assert_eq!(bw.get(1, 0), 1);
    }

    #[test]
    fn test_set_get() {
        let mut bw = BinaryWeights::new(1, 128);
        bw.set(0, 0, 1); // bit set → +1
        bw.set(0, 1, -1); // bit clear → -1
        bw.set(0, 64, -1);
        bw.set(0, 100, 1);
        assert_eq!(bw.get(0, 0), 1);
        assert_eq!(bw.get(0, 1), -1);
        assert_eq!(bw.get(0, 2), 1); // default +1
        assert_eq!(bw.get(0, 64), -1);
        assert_eq!(bw.get(0, 100), 1);
    }

    #[test]
    fn test_groups_per_row() {
        let bw = BinaryWeights::new(1, 128);
        assert_eq!(bw.groups_per_row, 1);
        let bw = BinaryWeights::new(1, 129);
        assert_eq!(bw.groups_per_row, 2);
        let bw = BinaryWeights::new(1, 256);
        assert_eq!(bw.groups_per_row, 2);
        let bw = BinaryWeights::new(1, 1);
        assert_eq!(bw.groups_per_row, 1);
    }

    #[test]
    fn test_quantize_basic() {
        // All positive → all +1, scale = mean
        let w = vec![2.0f32; 128];
        let bw = BinaryWeights::quantize_from_f32(&w, 1, 128);
        assert_eq!(bw.get(0, 0), 1);
        assert!((bw.group_scale[0].to_f32() - 2.0).abs() < 1e-3);

        // All negative → all -1
        let w = vec![-3.0f32; 128];
        let bw = BinaryWeights::quantize_from_f32(&w, 1, 128);
        assert_eq!(bw.get(0, 0), -1);
        assert!((bw.group_scale[0].to_f32() - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_encoded_bytes() {
        let bw = BinaryWeights::new(4, 128);
        // sign_bits: 4 rows × 2 blocks × 8 bytes = 64 bytes
        // group_scale: 4 rows × 1 group × 2 bytes = 8 bytes
        assert_eq!(bw.encoded_bytes(), 64 + 8);
    }

    #[test]
    #[cfg(feature = "plasma_path")]
    fn test_from_ternary_no_zeros_rejects_zeros() {
        // Construct a ternary weights with a zero
        let mut tw = crate::TernaryWeights::new(1, 64);
        tw.pos_bits[0] = 0; // all zero → invalid binary subset
        tw.neg_bits[0] = 0;
        assert!(BinaryWeights::from_ternary_no_zeros(&tw).is_none());
    }
}
