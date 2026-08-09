//! Ternary bit-plane matvec with group-wise f16 scale — the `Q2_0_g128`
//! kernel (`ternary_group_scale` feature, Issue 578). Scalar / NEON paths.
//!
//! Combines the two shipped tiers' halves: the two bit-planes of
//! [`crate::TernaryWeights`] (so the zero state survives) with the per-128
//! f16 `group_scale` of [`crate::BinaryWeights`].
//!
//! # Kernel design
//!
//! [`GROUP_SIZE`] is 128 and blocks are 64 bits, so **a group is exactly two
//! whole `u64` blocks** — a group boundary never splits a word. The NEON path
//! walks groups, and within each group its two blocks, switching only the
//! scale splat.
//!
//! The group scale is folded **into the sign vector** (`±scale` instead of
//! `±1`), the same trick the binary kernel uses (Issue 145 Gate A): the 4
//! accumulators then span the entire row with no per-group reset and no
//! per-group horizontal sum, keeping the out-of-order pipeline fed. Cost is one
//! extra `vmulq` per 4 lanes versus the row-scale ternary kernel, which applies
//! its single scale once at the very end.
//!
//! # Scalar/NEON agreement is close, not bit-identical
//!
//! Unlike [`super::simd_ternary_matvec`] — where both paths apply one row scale
//! at the end and therefore agree bit-for-bit — the two paths here associate
//! the scale differently:
//!
//! - scalar: `Σ_g scale_g · (Σ_{col∈g} sign·x)` — scale applied once per group
//! - NEON:   `Σ_g Σ_{col∈g} (scale_g·sign)·x` — scale folded per element
//!
//! These are equal in exact arithmetic but not in f32. Agreement is ~1e-6
//! relative (asserted in tests). The scalar form is the reference because it
//! matches how `Q2_0_g128` is defined and how llama.cpp dequantizes it.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

#[cfg(feature = "ternary_group_scale")]
use super::simd_level;
#[cfg(all(feature = "ternary_group_scale", target_arch = "aarch64"))]
use super::SimdLevel;
#[cfg(feature = "ternary_group_scale")]
use crate::{GROUP_SIZE, TernaryGroupWeights};

/// Blocks per group. `GROUP_SIZE / 64` — exactly 2 at the shipped group size.
#[cfg(feature = "ternary_group_scale")]
const BLOCKS_PER_GROUP: usize = GROUP_SIZE / 64;

/// Scalar reference: `y[r] = Σ_g group_scale[r,g] · Σ_{col∈g} sign(col) · x[col]`
///
/// `sign = +1` where the pos plane is set, `-1` where the neg plane is set,
/// `0` where neither is (the state `BinaryWeights` cannot represent).
#[cfg(feature = "ternary_group_scale")]
pub fn ternary_group_matvec_scalar(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    for r in 0..w.rows {
        let row_base = r * w.blocks64;
        let group_base = r * w.groups_per_row;
        let mut row_sum = 0.0f32;
        for g in 0..w.groups_per_row {
            let g_start = g * GROUP_SIZE;
            let g_end = (g_start + GROUP_SIZE).min(w.cols);
            let mut group_acc = 0.0f32;
            for col in g_start..g_end {
                let idx = row_base + (col >> 6);
                let mask = 1u64 << (col & 63);
                let pos = (w.pos_bits[idx] & mask) != 0;
                let neg = (w.neg_bits[idx] & mask) != 0;
                let sign = pos as i32 - neg as i32;
                group_acc += sign as f32 * unsafe { *x.get_unchecked(col) };
            }
            row_sum += w.group_scale[group_base + g].to_f32() * group_acc;
        }
        y[r] = row_sum;
    }
}

#[cfg(all(feature = "ternary_group_scale", target_arch = "aarch64"))]
unsafe fn neon_ternary_group_matvec(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    // Safety: caller guarantees x.len() == w.cols and y.len() == w.rows.
    unsafe {
        use core::arch::aarch64::{float32x4_t, uint32x4_t, *};
        assert_eq!(x.len(), w.cols);
        assert_eq!(y.len(), w.rows);

        // SWAR bit-position masks: AND a splatted byte with these to isolate
        // each bit of a nibble into its own lane (same construction as the
        // row-scale ternary kernel).
        let mask_lo_arr: [u32; 4] = [1, 2, 4, 8];
        let mask_hi_arr: [u32; 4] = [16, 32, 64, 128];
        let mask_lo: uint32x4_t = vld1q_u32(mask_lo_arr.as_ptr());
        let mask_hi: uint32x4_t = vld1q_u32(mask_hi_arr.as_ptr());
        let one_u: uint32x4_t = vdupq_n_u32(1);

        for r in 0..w.rows {
            let row_base = r * w.blocks64;
            let group_base = r * w.groups_per_row;

            // 4 accumulators span the ENTIRE row (no per-group reset) — the
            // scale rides inside the sign vector instead.
            let mut acc0: float32x4_t = vdupq_n_f32(0.0);
            let mut acc1: float32x4_t = vdupq_n_f32(0.0);
            let mut acc2: float32x4_t = vdupq_n_f32(0.0);
            let mut acc3: float32x4_t = vdupq_n_f32(0.0);

            for g in 0..w.groups_per_row {
                let scale = w.group_scale[group_base + g].to_f32();
                let scale_v: float32x4_t = vdupq_n_f32(scale);

                // A group is exactly BLOCKS_PER_GROUP whole blocks; the final
                // group of a ragged row may be short.
                let b_start = g * BLOCKS_PER_GROUP;
                let b_end = (b_start + BLOCKS_PER_GROUP).min(w.blocks64);

                for b in b_start..b_end {
                    let idx = row_base + b;
                    let pos_word = w.pos_bits[idx];
                    let neg_word = w.neg_bits[idx];

                    let base_col = b * 64;
                    let remaining = match base_col + 64 <= w.cols {
                        true => 64,
                        false => w.cols - base_col,
                    };

                    // 32-element unroll: 4 × 8-element chunks, one per accumulator.
                    let mut col = 0usize;
                    while col + 32 <= remaining {
                        fmla_scaled_nibble8(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                        fmla_scaled_nibble8(
                            &mut acc1, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                        fmla_scaled_nibble8(
                            &mut acc2, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                        fmla_scaled_nibble8(
                            &mut acc3, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                    }

                    // Remaining 8-element chunks.
                    while col + 8 <= remaining {
                        fmla_scaled_nibble8(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                    }

                    // Remaining 4-element chunk.
                    if col + 4 <= remaining {
                        let byte_off = col / 8;
                        let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as u32;
                        let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as u32;
                        let pos_splat = vdupq_n_u32(pos_byte);
                        let neg_splat = vdupq_n_u32(neg_byte);
                        let pos_nz =
                            vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_lo), one_u));
                        let neg_nz =
                            vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_lo), one_u));
                        // vcgeq gives all-ones (-1 as i32) where set, so
                        // neg - pos yields +1 for pos, -1 for neg, 0 for neither.
                        let sign_f = vcvtq_f32_s32(vsubq_s32(neg_nz, pos_nz));
                        let scaled = vmulq_f32(sign_f, scale_v);
                        let x_v = vld1q_f32(x.as_ptr().add(base_col + col));
                        acc0 = vfmaq_f32(acc0, scaled, x_v);
                        col += 4;
                    }

                    // Scalar tail (0-3 elements).
                    let mut scalar_acc = 0.0f32;
                    while col < remaining {
                        let bit_mask = 1u64 << col;
                        let pos = ((pos_word & bit_mask) != 0) as i32 as f32;
                        let neg = ((neg_word & bit_mask) != 0) as i32 as f32;
                        scalar_acc += (pos - neg) * scale * *x.get_unchecked(base_col + col);
                        col += 1;
                    }
                    if scalar_acc != 0.0 {
                        acc0 = vaddq_f32(acc0, vsetq_lane_f32(scalar_acc, vdupq_n_f32(0.0), 0));
                    }
                }
            }

            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            y[r] = vaddvq_f32(acc0);
        }
    }
}

/// NEON inner-loop helper: 8 elements (one byte of each bit-plane, both
/// nibbles) with SWAR mask construction, group scale folded into the sign.
///
/// Identical to the row-scale kernel's `fmla_nibble8` except for the
/// `vmulq_f32(sign_f, scale_v)` that turns `±1` into `±scale`.
#[cfg(all(feature = "ternary_group_scale", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn fmla_scaled_nibble8(
    acc: &mut core::arch::aarch64::float32x4_t,
    pos_word: u64,
    neg_word: u64,
    col: usize,
    base_col: usize,
    x: &[f32],
    mask_lo: core::arch::aarch64::uint32x4_t,
    mask_hi: core::arch::aarch64::uint32x4_t,
    one_u: core::arch::aarch64::uint32x4_t,
    scale_v: core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let byte_off = col / 8;
        let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as u32;
        let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as u32;

        let pos_splat = vdupq_n_u32(pos_byte);
        let neg_splat = vdupq_n_u32(neg_byte);

        let pos_lo = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_lo), one_u));
        let neg_lo = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_lo), one_u));
        let pos_hi = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_hi), one_u));
        let neg_hi = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_hi), one_u));

        // sign = neg - pos → +1 where pos, -1 where neg, 0 where neither.
        let sign_lo_f = vcvtq_f32_s32(vsubq_s32(neg_lo, pos_lo));
        let sign_hi_f = vcvtq_f32_s32(vsubq_s32(neg_hi, pos_hi));

        // Fold the group scale in: ±1 → ±scale (0 stays 0).
        let scaled_lo = vmulq_f32(sign_lo_f, scale_v);
        let scaled_hi = vmulq_f32(sign_hi_f, scale_v);

        let x_lo = vld1q_f32(x.as_ptr().add(base_col + col));
        let x_hi = vld1q_f32(x.as_ptr().add(base_col + col + 4));

        *acc = vfmaq_f32(*acc, scaled_lo, x_lo);
        *acc = vfmaq_f32(*acc, scaled_hi, x_hi);
    }
}

/// Group-scale ternary matvec: `y = W × x`.
///
/// Dispatches to NEON where available, scalar otherwise.
///
/// **x86_64/AVX2 is not yet specialized** — it falls back to the scalar path.
/// The AVX2 port is deliberately deferred: it cannot be validated on the
/// aarch64 development machine, and an unvalidated hand-written AVX2 kernel is
/// worse than a correct scalar one. Tracked in Issue 578.
#[cfg(feature = "ternary_group_scale")]
#[inline]
pub fn simd_ternary_group_matvec(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    match simd_level() {
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe { neon_ternary_group_matvec(w, x, y) },
        _ => ternary_group_matvec_scalar(w, x, y),
    }
}

/// Batched group-scale ternary matmul: for each `batch[i]`, `y[i] = W × x[i]`.
///
/// Mirrors `simd_ternary_matmul_batch` / `simd_binary_matmul_batch`.
#[cfg(feature = "ternary_group_scale")]
pub fn simd_ternary_group_matmul_batch(
    w: &TernaryGroupWeights,
    x: &[f32],
    batch: usize,
    y: &mut [f32],
) {
    /// Below this, rayon's per-task overhead (~1-5µs) outweighs the work.
    const PARALLEL_BATCH_MIN: usize = 4;

    if batch < PARALLEL_BATCH_MIN {
        for b in 0..batch {
            let x_off = b * w.cols;
            let y_off = b * w.rows;
            simd_ternary_group_matvec(w, &x[x_off..], &mut y[y_off..]);
        }
        return;
    }

    use rayon::prelude::*;
    y.par_chunks_mut(w.rows)
        .zip(x.par_chunks(w.cols))
        .enumerate()
        .for_each(|(b, (y_chunk, x_chunk))| {
            if b < batch {
                simd_ternary_group_matvec(w, x_chunk, y_chunk);
            }
        });
}

#[cfg(all(test, feature = "ternary_group_scale"))]
mod tests {
    use super::*;
    use crate::TernaryGroupWeights;

    /// Deterministic pseudo-random f32 in [-1, 1). No rand dep, reproducible.
    fn pseudo(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn filled(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
        let mut s = seed;
        let mut w = TernaryGroupWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let v = pseudo(&mut s);
                let q = match v {
                    v if v > 0.33 => 1i8,
                    v if v < -0.33 => -1i8,
                    _ => 0i8,
                };
                w.set(r, c, q);
            }
            for g in 0..w.groups_per_row {
                // Vary scale per group so a per-row-scale bug cannot pass.
                w.set_scale(r, g, 0.5 + 0.25 * (g % 4) as f32);
            }
        }
        w
    }

    #[test]
    fn set_get_roundtrips_all_three_states() {
        let mut w = TernaryGroupWeights::new(3, 200);
        for c in 0..200 {
            w.set(0, c, 1);
            w.set(1, c, -1);
            w.set(2, c, 0);
        }
        for c in 0..200 {
            assert_eq!(w.get(0, c), 1);
            assert_eq!(w.get(1, c), -1);
            assert_eq!(w.get(2, c), 0);
        }
        // Overwrites must clear the other plane, not OR into it.
        w.set(0, 5, -1);
        assert_eq!(w.get(0, 5), -1);
        w.set(0, 5, 0);
        assert_eq!(w.get(0, 5), 0);
        assert!(w.invariant_holds());
    }

    #[test]
    fn group_geometry_matches_group_size() {
        let w = TernaryGroupWeights::new(2, 300);
        assert_eq!(GROUP_SIZE, 128);
        assert_eq!(BLOCKS_PER_GROUP, 2);
        assert_eq!(w.blocks64, 300usize.div_ceil(64));
        assert_eq!(w.groups_per_row, 300usize.div_ceil(128));
        // 2.125 bits/weight: 2 planes * 8B/64w + 2B/128w.
        let w2 = TernaryGroupWeights::new(1, 128);
        assert_eq!(w2.encoded_bytes(), 8 * 2 + 8 * 2 + 2);
    }

    #[test]
    fn neon_matches_scalar_reference() {
        // Cover exact multiples of GROUP_SIZE and ragged tails (partial group,
        // partial block, and a sub-4 scalar tail).
        for &(rows, cols) in &[(4, 128), (3, 256), (2, 300), (5, 65), (1, 7), (2, 129)] {
            let w = filled(rows, cols, 0x51ED_u64.wrapping_add(cols as u64));
            let mut s = 0xABCD_u64;
            let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();

            let mut y_scalar = vec![0.0f32; rows];
            let mut y_simd = vec![0.0f32; rows];
            ternary_group_matvec_scalar(&w, &x, &mut y_scalar);
            simd_ternary_group_matvec(&w, &x, &mut y_simd);

            for r in 0..rows {
                let denom = y_scalar[r].abs().max(1.0);
                let rel = (y_scalar[r] - y_simd[r]).abs() / denom;
                assert!(
                    rel < 1e-6,
                    "rows={rows} cols={cols} r={r}: scalar={} simd={} rel={rel:e}",
                    y_scalar[r],
                    y_simd[r]
                );
            }
        }
    }

    #[test]
    fn zero_state_is_actually_skipped() {
        // All-zero weights must produce exactly 0 regardless of scale — the
        // state BinaryWeights cannot represent.
        let mut w = TernaryGroupWeights::new(2, 256);
        for r in 0..2 {
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 3.0);
            }
        }
        let x = vec![1.0f32; 256];
        let mut y = vec![9.9f32; 2];
        simd_ternary_group_matvec(&w, &x, &mut y);
        assert_eq!(y, vec![0.0, 0.0]);
    }

    #[test]
    fn per_group_scale_is_applied_not_per_row() {
        // Row of all +1 with distinct per-group scales against x = 1:
        // y = Σ_g scale_g * group_len. A row-scale implementation cannot match.
        let cols = 256; // exactly 2 groups
        let mut w = TernaryGroupWeights::new(1, cols);
        for c in 0..cols {
            w.set(0, c, 1);
        }
        w.set_scale(0, 0, 1.0);
        w.set_scale(0, 1, 4.0);
        let x = vec![1.0f32; cols];
        let mut y = vec![0.0f32; 1];
        simd_ternary_group_matvec(&w, &x, &mut y);
        assert!((y[0] - (128.0 + 512.0)).abs() < 1e-3, "got {}", y[0]);
    }

    #[test]
    fn quantize_from_f32_holds_invariant_and_tracks_signs() {
        let (rows, cols) = (3, 300);
        let mut s = 0x1234_u64;
        let src: Vec<f32> = (0..rows * cols).map(|_| pseudo(&mut s) * 2.0).collect();
        let w = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        assert!(w.invariant_holds());
        assert_eq!(w.groups_per_row, cols.div_ceil(GROUP_SIZE));
        // Large-magnitude inputs must not quantize to zero.
        let mut big = vec![0.0f32; cols];
        big[0] = 10.0;
        big[1] = -10.0;
        let wb = TernaryGroupWeights::quantize_from_f32(&big, 1, cols);
        assert_eq!(wb.get(0, 0), 1);
        assert_eq!(wb.get(0, 1), -1);
    }

    /// The reason `Q2_0_g128` exists: a scale per 128 weights tracks a signal
    /// whose magnitude varies along the row better than one scale per row.
    ///
    /// Asserted as a *comparison*, not an absolute bound. In an error-feedback
    /// quantizer the sum-reconstruction error equals the final carry
    /// (`Σ q·scale = Σ w - carry_final`), and the carry grows wherever `|w|`
    /// runs above the local scale — so the absolute error is signal-dependent
    /// and not worth pinning. What must hold is that finer scale granularity
    /// does not do *worse*.
    #[cfg(feature = "plasma_path")]
    #[test]
    fn group_scale_tracks_varying_magnitude_better_than_row_scale() {
        // Magnitude ramps 10x across the row: group 0 is small, group 1 large.
        // A single row scale must compromise between them; per-group cannot.
        let cols = 256;
        let src: Vec<f32> = (0..cols)
            .map(|i| {
                let amp = match i < 128 {
                    true => 0.1,
                    false => 1.0,
                };
                amp * (i as f32 / 8.0).sin()
            })
            .collect();

        let gw = TernaryGroupWeights::quantize_from_f32(&src, 1, cols);
        let rw = crate::TernaryWeights::quantize_from_f32(&src, 1, cols);
        assert!(gw.invariant_holds());

        // Reconstruct each weight and compare against the source elementwise.
        let err_group: f32 = (0..cols)
            .map(|c| {
                let scale = gw.scale_at(0, c / GROUP_SIZE);
                (gw.get(0, c) as f32 * scale - src[c]).abs()
            })
            .sum();
        let err_row: f32 = (0..cols)
            .map(|c| (rw.get(0, c) as f32 * rw.row_scale[0] - src[c]).abs())
            .sum();

        assert!(
            err_group < err_row,
            "group-scale L1 error {err_group} should beat row-scale {err_row}"
        );
    }

    #[test]
    fn batch_matches_per_row_calls() {
        let (rows, cols, batch) = (4, 256, 6);
        let w = filled(rows, cols, 0x77);
        let mut s = 0x99_u64;
        let x: Vec<f32> = (0..cols * batch).map(|_| pseudo(&mut s)).collect();

        let mut y_batch = vec![0.0f32; rows * batch];
        simd_ternary_group_matmul_batch(&w, &x, batch, &mut y_batch);

        for b in 0..batch {
            let mut y_one = vec![0.0f32; rows];
            simd_ternary_group_matvec(&w, &x[b * cols..(b + 1) * cols], &mut y_one);
            for r in 0..rows {
                assert_eq!(y_batch[b * rows + r], y_one[r], "batch {b} row {r}");
            }
        }
    }

    #[cfg(feature = "plasma_path")]
    #[test]
    fn widening_from_row_scale_ternary_preserves_result() {
        let (rows, cols) = (3, 256);
        let mut s = 0xFEED_u64;
        let src: Vec<f32> = (0..rows * cols).map(|_| pseudo(&mut s)).collect();
        let tw = crate::TernaryWeights::quantize_from_f32(&src, rows, cols);
        let gw = TernaryGroupWeights::from_ternary(&tw);

        // Bit-planes copied verbatim.
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(tw.get(r, c), gw.get(r, c), "r={r} c={c}");
            }
        }
        assert!(gw.invariant_holds());

        // Same matvec up to the f32→f16 scale rounding.
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
        let mut y_row = vec![0.0f32; rows];
        let mut y_grp = vec![0.0f32; rows];
        super::super::simd_ternary_matvec(&tw, &x, &mut y_row);
        simd_ternary_group_matvec(&gw, &x, &mut y_grp);
        for r in 0..rows {
            let denom = y_row[r].abs().max(1.0);
            assert!(
                (y_row[r] - y_grp[r]).abs() / denom < 1e-3,
                "r={r}: row={} group={}",
                y_row[r],
                y_grp[r]
            );
        }
    }
}
