//! SIMD LUT Dequantization — software analog of StreamDQ near-memory DQ.
//!
//! Distilled from StreamDQ (Jeong et al., SK Hynix, arXiv:2607.08993, 2026-07-09).
//! The paper ships DeQuantization Blocks (DQBs) in HBM base dies that perform
//! on-the-fly weight dequant via a shared FP32 ALU + format-specific LUT. This
//! module is the **software SIMD analog**: pre-compute a stack-allocated f32 LUT
//! per quant group, then replace the per-element integer-arithmetic INT→FP cast
//! with a single LUT lookup.
//!
//! # The technique (paper §3.6 → software)
//!
//! | Paper (hardware)                      | This module (software)                     |
//! |---------------------------------------|--------------------------------------------|
//! | Shared FP32 ALU + format type-cast    | `QuantLut` trait, generic over format      |
//! | LUT-based INT→FP conversion           | `Int4Lut` / `UInt4Lut` / `Int8Lut`         |
//! | S/Z co-located buffer                 | `build(scale, zero)` bakes affine into LUT |
//! | Sideband tag dispatch                 | (deferred — see Plan 431 Phase 6)          |
//!
//! The LUT for INT4 is `[f32; 16]` = 64 bytes = exactly one cache line. It fits
//! in L1 for the duration of a block decode, and the inner loop becomes pure
//! gather — no CVT instruction, no FP32 multiply per element.
//!
//! # Honest expectation (Research 418 §4.1)
//!
//! The paper's 7× speedup is **hardware-only** (eliminates CUDA-core overhead +
//! HBM write-back). In software SIMD, the win is bounded by gather latency vs
//! CVT latency and by L1 cache pressure. Realistic target: 1.0–1.5× microbench,
//! plausibly a regression on slow-gather platforms. **The GOAT gate (Phase 4)
//! settles it honestly.**
//!
//! # Allocation discipline (G4)
//!
//! Every LUT is a stack `[f32; N]`. `dequant_via_lut` takes `&mut [f32]` output
//! by reference. The hot path has zero `Vec`/`Box`/`String`/`format!`. No
//! allocations by construction.
//!
//! See: `katgpt-rs/.research/418_StreamDQ_SIMD_LUT_DeQuant.md`
//! See: `katgpt-rs/.plans/431_simd_lut_dequant.md`

// ──────────────────────────────────────────────────────────────────────────
// Phase 1: scalar reference. SIMD inner loops land in Phase 2.
// ──────────────────────────────────────────────────────────────────────────

/// Format-specific pre-computed dequantization lookup table.
///
/// The LUT bakes in the full affine transform `lut[code] = (signed(code) − z) · s`
/// so that the hot loop is a single indexed read — no per-element multiply, no
/// integer-to-float conversion. One LUT per quant group; the LUT lives on the
/// stack for the duration of a block decode.
///
/// # Conventions per implementation
///
/// | Type       | Domain      | `signed(code)`              |
/// |------------|-------------|------------------------------|
/// | `UInt4Lut` | `0..16`     | `code as f32` (no sign-ext)  |
/// | `Int4Lut`  | `−8..+8`    | sign-extend low 4 bits       |
/// | `Int8Lut`  | `−128..+128`| `code as i8 as f32`          |
///
/// The `(scale, zero)` pair follows the standard asymmetric-quantization form
/// `fp = (code − zero) · scale`, where `zero` is expressed in **code units**
/// (not FP units). A caller holding an FP-space offset `offset_fp` (e.g. Q4_K's
/// `dmin * m0`) passes `zero = offset_fp / scale`.
pub trait QuantLut: Sized {
    /// Number of entries. 16 for nibble formats, 256 for byte formats.
    const LUT_LEN: usize;

    /// Build the LUT for this format at the given affine `(scale, zero)`.
    ///
    /// Entry `i` is precomputed as `(signed(i) − zero) · scale`, so subsequent
    /// [`Self::lookup`] calls are pure indexed reads with no FP arithmetic.
    fn build(scale: f32, zero: f32) -> Self;

    /// Look up the pre-computed FP32 value for `code`.
    ///
    /// `code` is the already-extracted (shifted + masked) quant index. The LUT
    /// may mask internally as a safety net, so passing an un-masked nibble is
    /// still correct — only the low `log2(LUT_LEN)` bits are used.
    fn lookup(&self, code: u8) -> f32;
}

// ──────────────────────────────────────────────────────────────────────────
// Concrete LUT types
// ──────────────────────────────────────────────────────────────────────────

/// Unsigned 4-bit LUT (codes `0..16`). Use for Q4_K low/high nibbles.
///
/// 64 bytes = one cache line.
#[derive(Clone, Copy)]
pub struct UInt4Lut([f32; 16]);

/// Signed 4-bit LUT (codes `−8..+8` in two's complement of the low 4 bits).
///
/// 64 bytes = one cache line. Use for symmetric INT4 formats.
#[derive(Clone, Copy)]
pub struct Int4Lut([f32; 16]);

/// Signed 8-bit LUT (codes `−128..+128`). 1 KB.
#[derive(Clone, Copy)]
pub struct Int8Lut([f32; 256]);

impl QuantLut for UInt4Lut {
    const LUT_LEN: usize = 16;

    #[inline]
    fn build(scale: f32, zero: f32) -> Self {
        let mut lut = [0.0_f32; 16];
        let mut i = 0u8;
        while i < 16 {
            lut[i as usize] = (i as f32 - zero) * scale;
            i += 1;
        }
        UInt4Lut(lut)
    }

    #[inline(always)]
    fn lookup(&self, code: u8) -> f32 {
        // Safety net: mask to 4 bits. The caller already masks via the `mask`
        // argument to `dequant_via_lut`, but this keeps direct LUT use safe.
        self.0[(code as usize) & 0x0F]
    }
}

impl QuantLut for Int4Lut {
    const LUT_LEN: usize = 16;

    #[inline]
    fn build(scale: f32, zero: f32) -> Self {
        let mut lut = [0.0_f32; 16];
        let mut i = 0u8;
        while i < 16 {
            // Sign-extend the low 4 bits: 0..7 stay, 8..15 become −8..−1.
            let signed = ((i as i8) << 4) >> 4;
            lut[i as usize] = (signed as f32 - zero) * scale;
            i += 1;
        }
        Int4Lut(lut)
    }

    #[inline(always)]
    fn lookup(&self, code: u8) -> f32 {
        self.0[(code as usize) & 0x0F]
    }
}

impl QuantLut for Int8Lut {
    const LUT_LEN: usize = 256;

    #[inline]
    fn build(scale: f32, zero: f32) -> Self {
        let mut lut = [0.0_f32; 256];
        // `0u8..=255u8` is a valid Rust range — no wrap-around hazard because
        // the compiler knows the upper bound is the type's max.
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = (i as i8 as f32 - zero) * scale;
        }
        Int8Lut(lut)
    }

    #[inline(always)]
    fn lookup(&self, code: u8) -> f32 {
        self.0[code as usize]
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Generic shared-FP32-ALU dequantize primitive
// ──────────────────────────────────────────────────────────────────────────

/// Generic LUT-accelerated dequantize.
///
/// For each byte in `codes`, extracts the quant index via `(byte >> shift) & mask`
/// and writes `lut.lookup(index)` to the corresponding slot in `out`. The LUT
/// already contains the full affine `(code − zero) · scale`, so the inner loop
/// is a single lookup — no per-element multiply, no integer-to-float cast.
///
/// # Arguments
///
/// - `codes` — packed quant bytes. For nibble formats, each byte holds two codes
///   (call this function twice per block: `shift=0` for low nibbles, `shift=4`
///   for high nibbles). For byte formats, each byte is one code (`shift=0`).
/// - `lut` — a pre-built [`QuantLut`] for this group's `(scale, zero)`.
/// - `shift` — right-shift to apply before masking: `0` for low nibble / byte,
///   `4` for high nibble.
/// - `mask` — bit-mask after the shift: `0x0F` for nibbles, `0xFF` for bytes.
/// - `out` — destination slice. Must be at least `codes.len()` long. Only the
///   first `min(codes.len(), out.len())` elements are written.
///
/// # Allocation discipline
///
/// Zero allocations. The LUT is stack-allocated by the caller; `out` is a
/// caller-owned `&mut [f32]`.
///
/// # Example
///
/// ```
/// use katgpt_core::simd_lut_dequant::{UInt4Lut, QuantLut, dequant_via_lut};
///
/// // Q4_K-style: low nibbles, scale=0.5, zero=8.0 (asymmetric: fp = (code-8)*0.5)
/// let lut = UInt4Lut::build(0.5, 8.0);
/// let codes = [0x12, 0x34]; // low nibbles: 2, 4
/// let mut out = [0.0_f32; 2];
/// dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out);
/// assert_eq!(out, [(2.0 - 8.0) * 0.5, (4.0 - 8.0) * 0.5]); // [-3.0, -2.0]
/// ```
#[inline]
pub fn dequant_via_lut<L: QuantLut>(
    codes: &[u8],
    lut: &L,
    shift: u32,
    mask: u8,
    out: &mut [f32],
) {
    let n = codes.len().min(out.len());
    let mut i = 0;
    while i < n {
        let idx = (codes[i] >> shift) & mask;
        out[i] = lut.lookup(idx);
        i += 1;
    }
}

/// Scalar reference arithmetic dequantize — the "current path" comparator.
///
/// Computes `(signed(code) − zero) · scale` per element, matching the arithmetic
/// cast path in `q4k.rs` today. Used by the GOAT gate (Plan 431 Phase 4 G1) to
/// verify the LUT path is bit-exact. This is NOT the hot path — it exists to be
/// the correctness oracle.
///
/// The `is_signed_4bit` flag selects between unsigned-nibble (`0..16`) and
/// signed-nibble (`−8..+8`) interpretation. For byte-aligned formats, set
/// `shift = 0` and `mask = 0xFF`.
#[inline]
pub fn dequant_arithmetic_ref(
    codes: &[u8],
    scale: f32,
    zero: f32,
    shift: u32,
    mask: u8,
    is_signed_4bit: bool,
    out: &mut [f32],
) {
    let n = codes.len().min(out.len());
    let mut i = 0;
    while i < n {
        let idx = (codes[i] >> shift) & mask;
        let code_f = if is_signed_4bit {
            (((idx as i8) << 4) >> 4) as f32
        } else if mask == 0xFF {
            idx as i8 as f32
        } else {
            idx as f32
        };
        out[i] = (code_f - zero) * scale;
        i += 1;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LUT build correctness ────────────────────────────────────────────

    #[test]
    fn test_uint4_lut_build() {
        let lut = UInt4Lut::build(2.0, 3.0);
        for i in 0..16u8 {
            let expected = (i as f32 - 3.0) * 2.0;
            assert_eq!(
                lut.0[i as usize],
                expected,
                "UInt4Lut[{}] = {} != expected {}",
                i,
                lut.0[i as usize],
                expected
            );
        }
    }

    #[test]
    fn test_int4_lut_build_sign_extension() {
        let lut = Int4Lut::build(1.0, 0.0);
        // Codes 0..7 map to 0.0..7.0; codes 8..15 map to -8.0..-1.0.
        for i in 0..8u8 {
            assert_eq!(lut.0[i as usize], i as f32, "Int4Lut unsigned region {}", i);
        }
        for i in 8..16u8 {
            let signed = i as i8 - 16; // 8→-8, 9→-7, ..., 15→-1
            assert_eq!(
                lut.0[i as usize],
                signed as f32,
                "Int4Lut signed region {}",
                i
            );
        }
    }

    #[test]
    fn test_int8_lut_build_sign_extension() {
        let lut = Int8Lut::build(1.0, 0.0);
        for i in 0..128u8 {
            assert_eq!(lut.0[i as usize], i as f32, "Int8Lut unsigned region {}", i);
        }
        for i in 128..=255u8 {
            let signed = i as i8; // 128→-128, ..., 255→-1
            assert_eq!(
                lut.0[i as usize],
                signed as f32,
                "Int8Lut signed region {}",
                i
            );
        }
    }

    // ── dequant_via_lut correctness ──────────────────────────────────────

    #[test]
    fn test_dequant_uint4_low_nibble() {
        let lut = UInt4Lut::build(0.5, 8.0);
        let codes = [0x12_u8, 0x34, 0xAB, 0xCD];
        let mut out = [0.0_f32; 4];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out);
        // Low nibbles: 2, 4, 0xB=11, 0xD=13
        let expected = [(2.0 - 8.0) * 0.5, (4.0 - 8.0) * 0.5, (11.0 - 8.0) * 0.5, (13.0 - 8.0) * 0.5];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_dequant_uint4_high_nibble() {
        let lut = UInt4Lut::build(0.5, 8.0);
        let codes = [0x12_u8, 0x34, 0xAB, 0xCD];
        let mut out = [0.0_f32; 4];
        dequant_via_lut(&codes, &lut, 4, 0x0F, &mut out);
        // High nibbles: 1, 3, 0xA=10, 0xC=12
        let expected = [(1.0 - 8.0) * 0.5, (3.0 - 8.0) * 0.5, (10.0 - 8.0) * 0.5, (12.0 - 8.0) * 0.5];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_dequant_int4_signed() {
        let lut = Int4Lut::build(1.0, 0.0);
        let codes = [0x12_u8, 0x89]; // low nibbles: 2, 9
        let mut out = [0.0_f32; 2];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out);
        // code 2 → +2.0; code 9 → sign-extended -7.0
        assert_eq!(out, [2.0, -7.0]);
    }

    #[test]
    fn test_dequant_int8() {
        let lut = Int8Lut::build(1.0, 0.0);
        let codes = [0_u8, 127, 128, 255];
        let mut out = [0.0_f32; 4];
        dequant_via_lut(&codes, &lut, 0, 0xFF, &mut out);
        assert_eq!(out, [0.0, 127.0, -128.0, -1.0]);
    }

    // ── Bit-exact: LUT path == arithmetic path (Plan 431 Phase 4 G1) ─────

    #[test]
    fn test_lut_bit_exact_vs_arithmetic_uint4() {
        let scale = 0.3_f32;
        let zero = 5.5_f32;
        let lut = UInt4Lut::build(scale, zero);
        let codes: Vec<u8> = (0..=255).collect();
        let mut out_lut = vec![0.0_f32; 256];
        let mut out_ref = vec![0.0_f32; 256];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out_lut);
        dequant_arithmetic_ref(&codes, scale, zero, 0, 0x0F, false, &mut out_ref);
        let mut max_diff = 0.0_f32;
        for i in 0..256 {
            let diff = (out_lut[i] - out_ref[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert_eq!(max_diff, 0.0, "LUT path must be bit-exact vs arithmetic");
    }

    #[test]
    fn test_lut_bit_exact_vs_arithmetic_int4() {
        let scale = 0.7_f32;
        let zero = -2.0_f32;
        let lut = Int4Lut::build(scale, zero);
        let codes: Vec<u8> = (0..=255).collect();
        let mut out_lut = vec![0.0_f32; 256];
        let mut out_ref = vec![0.0_f32; 256];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out_lut);
        dequant_arithmetic_ref(&codes, scale, zero, 0, 0x0F, true, &mut out_ref);
        let mut max_diff = 0.0_f32;
        for i in 0..256 {
            let diff = (out_lut[i] - out_ref[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert_eq!(max_diff, 0.0, "Int4LUT path must be bit-exact vs arithmetic");
    }

    #[test]
    fn test_lut_bit_exact_vs_arithmetic_int8() {
        let scale = 0.1_f32;
        let zero = 50.0_f32;
        let lut = Int8Lut::build(scale, zero);
        let codes: Vec<u8> = (0..=255).collect();
        let mut out_lut = vec![0.0_f32; 256];
        let mut out_ref = vec![0.0_f32; 256];
        dequant_via_lut(&codes, &lut, 0, 0xFF, &mut out_lut);
        dequant_arithmetic_ref(&codes, scale, zero, 0, 0xFF, false, &mut out_ref);
        let mut max_diff = 0.0_f32;
        for i in 0..256 {
            let diff = (out_lut[i] - out_ref[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert_eq!(max_diff, 0.0, "Int8LUT path must be bit-exact vs arithmetic");
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_dequant_empty() {
        let lut = UInt4Lut::build(1.0, 0.0);
        let codes: [u8; 0] = [];
        let mut out: [f32; 0] = [];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out);
        // No panic, no write.
    }

    #[test]
    fn test_dequant_out_shorter_than_codes() {
        let lut = UInt4Lut::build(1.0, 0.0);
        let codes = [0x01_u8, 0x02, 0x03, 0x04];
        let mut out = [0.0_f32; 2];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out);
        // Only first 2 elements written: low nibbles 1, 2.
        assert_eq!(out, [1.0, 2.0]);
    }

    #[test]
    fn test_lookup_safety_mask() {
        // Direct lookup with out-of-range code should still be safe (masked).
        let lut = UInt4Lut::build(1.0, 0.0);
        assert_eq!(lut.lookup(0xFF), lut.lookup(0x0F)); // 0xFF & 0x0F == 0x0F
        assert_eq!(lut.lookup(0x10), lut.lookup(0x00)); // 0x10 & 0x0F == 0x00
    }

    // ── Q4_K-style integration smoke test (the target consumer) ──────────

    #[test]
    fn test_q4k_style_block_decode() {
        // Mimics the Q4_K hot path shape (Research 418 §2.1):
        //   d_sc0 * (qs[i] & 0x0F) as f32 - m0_val
        // Here: d_sc0 = 0.25, m0_val = 1.5, so zero = m0_val / d_sc0 = 6.0.
        let d_sc0 = 0.25_f32;
        let m0_val = 1.5_f32;
        let zero = m0_val / d_sc0;
        let lut = UInt4Lut::build(d_sc0, zero);

        // 32 packed bytes = 64 nibbles (one Q4_K super-block's qs section)
        let qs: [u8; 32] = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x00, 0xFF, 0x01, 0x23, 0x45, 0x67,
            0x89, 0xAB, 0xCD, 0xEF,
        ];
        let mut dst_low = [0.0_f32; 32];
        let mut dst_high = [0.0_f32; 32];
        dequant_via_lut(&qs, &lut, 0, 0x0F, &mut dst_low);
        dequant_via_lut(&qs, &lut, 4, 0x0F, &mut dst_high);

        // Verify against the arithmetic form the Q4_K code uses today.
        for i in 0..32 {
            let low = d_sc0 * (qs[i] & 0x0F) as f32 - m0_val;
            let high = d_sc0 * (qs[i] >> 4) as f32 - m0_val;
            assert_eq!(dst_low[i], low, "low nibble {} mismatch", i);
            assert_eq!(dst_high[i], high, "high nibble {} mismatch", i);
        }
    }
}
