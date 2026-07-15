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
//! | Sideband tag dispatch                 | (deferred — see Plan 452 Phase 6)          |
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

    /// Raw f32 slice of the LUT, for SIMD gather paths (Phase 2). Length == `LUT_LEN`.
    ///
    /// The SIMD backends (NEON scalar-gather, AVX2 `_mm_i32gather_ps`) need the
    /// raw f32 pointer to do indexed reads. This method exposes it without
    /// breaking the trait abstraction.
    fn as_f32_slice(&self) -> &[f32];
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

    #[inline(always)]
    fn as_f32_slice(&self) -> &[f32] {
        &self.0
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

    #[inline(always)]
    fn as_f32_slice(&self) -> &[f32] {
        &self.0
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

    #[inline(always)]
    fn as_f32_slice(&self) -> &[f32] {
        &self.0
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
/// Dispatches to NEON (aarch64), AVX2 (x86_64 + compile-time AVX2 feature), or
/// scalar fallback based on the target architecture. On NEON the gather is
/// scalar (NEON has no native gather — the plan acknowledges this); on AVX2 the
/// hardware `_mm256_i32gather_ps` instruction does the gather natively. On WASM,
/// the scalar fallback is used — WASM SIMD128 has no gather instruction, and
/// the scalar path is already efficient for the LUT lookup pattern.
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
#[allow(clippy::needless_return)] // return is needed for cfg-gated arch dispatch
pub fn dequant_via_lut<L: QuantLut>(
    codes: &[u8],
    lut: &L,
    shift: u32,
    mask: u8,
    out: &mut [f32],
) {
    let lut_slice = lut.as_f32_slice();
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dequant_via_lut_neon(codes, lut_slice, shift, mask, out) }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { dequant_via_lut_avx2(codes, lut_slice, shift, mask, out) }
        return;
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        dequant_via_lut_scalar(codes, lut_slice, shift, mask, out);
    }
}

/// Scalar LUT dequantize — the portable fallback and Phase 1 reference.
///
/// Also used as the correctness oracle by the Phase 4 GOAT gate G1 test.
/// Exposed publicly so consumers on platforms without SIMD can call it directly,
/// and so tests can verify the SIMD paths are bit-exact against it.
#[inline]
pub fn dequant_via_lut_scalar(
    codes: &[u8],
    lut_slice: &[f32],
    shift: u32,
    mask: u8,
    out: &mut [f32],
) {
    let n = codes.len().min(out.len());
    let mut i = 0;
    while i < n {
        let idx = ((codes[i] >> shift) & mask) as usize;
        out[i] = lut_slice[idx & (lut_slice.len() - 1)];
        i += 1;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// NEON backend (aarch64)
// ──────────────────────────────────────────────────────────────────────────
// NEON has no native gather instruction. The shift+mask is vectorized (8 bytes
// at a time), but the LUT gather is scalar extraction. The store is vectorized
// (2× float32x4_t per 8 elements). This is the honest implementation — the win
// on NEON comes from the pre-computed LUT (no per-element multiply), not from
// vectorized gather (which doesn't exist). The GOAT gate (Phase 4) settles
// whether this is enough.

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dequant_via_lut_neon(
    codes: &[u8],
    lut_slice: &[f32],
    shift: u32,
    mask: u8,
    out: &mut [f32],
) {
    use core::arch::aarch64::{
        vand_u8, vdup_n_s8, vdup_n_u8, vget_lane_u8, vld1q_f32, vld1_u8, vshl_u8, vst1q_f32,
    };

    unsafe {
        let n = codes.len().min(out.len());
        // NEON right-shift = vshl by negative amount.
        let neg_shift = (shift as i8).wrapping_neg();
        let shift_vec = vdup_n_s8(neg_shift);
        let mask_vec = vdup_n_u8(mask);
        let mask_bits = lut_slice.len().trailing_zeros() as u8; // log2(LUT_LEN): 4 for INT4, 8 for INT8
        // For INT8 (256 entries, mask_bits=8), all u8 values are valid indices.
        // For INT4 (16 entries, mask_bits=4), mask to prevent OOB from stray bits.
        // `1u8 << 8` overflows, so cap at 8 and use 0xFF as the index mask.
        let idx_mask: u8 = if mask_bits >= 8 { 0xFF } else { (1u8 << mask_bits) - 1 };

        // Process 8 bytes at a time: load 8 codes, shift+mask, scalar gather, NEON store.
        let mut i = 0;
        while i + 8 <= n {
            // Load 8 packed code bytes.
            let code_vec = vld1_u8(codes.as_ptr().add(i));
            // Shift right by `shift`, then mask.
            let shifted = vshl_u8(code_vec, shift_vec);
            let masked = vand_u8(shifted, mask_vec);
            // Scalar gather: extract 8 lanes and index into the LUT.
            // vget_lane_u8 requires a compile-time constant lane index, so we
            // unroll all 8 extractions explicitly.
            let gathered = [
                *lut_slice.get_unchecked((vget_lane_u8(masked, 0) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 1) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 2) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 3) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 4) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 5) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 6) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 7) & idx_mask) as usize),
            ];
            // Store 8 f32 via 2× float32x4_t.
            let lo = vld1q_f32(gathered.as_ptr());
            let hi = vld1q_f32(gathered.as_ptr().add(4));
            vst1q_f32(out.as_mut_ptr().add(i), lo);
            vst1q_f32(out.as_mut_ptr().add(i + 4), hi);
            i += 8;
        }
        // Tail (1–7 remaining elements): scalar fallback.
        while i < n {
            let idx = ((codes[i] >> shift) & mask) as usize;
            out[i] = *lut_slice.get_unchecked(idx & (lut_slice.len() - 1));
            i += 1;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// AVX2 backend (x86_64 + target_feature = "avx2")
// ──────────────────────────────────────────────────────────────────────────
// AVX2 has native gather: `_mm_i32gather_ps`. This is the software analog of
// the paper's hardware gather in the DQB's shared FP32 ALU. The gather takes
// 8 i32 indices and reads 8 f32 values from the LUT base in a single
// instruction. This is where the real win is expected (if any).
//
// Approach: load 8 bytes → zero-extend to 8× i32 → shift+mask on i32 lanes →
// gather 8× f32 from LUT → store. The shift+mask on i32 (not byte) avoids the
// byte-level shift complication (AVX2 `_mm_srli_epi32` operates on 32-bit lanes).

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dequant_via_lut_avx2(
    codes: &[u8],
    lut_slice: &[f32],
    shift: u32,
    mask: u8,
    out: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm256_and_si256, _mm256_castsi128_si256, _mm256_i32gather_ps, _mm256_set1_epi32,
        _mm256_srli_epi32, _mm256_storeu_ps, _mm_cvtepu8_epi32, _mm_loadl_epi64,
    };

    unsafe {
        let n = codes.len().min(out.len());
        let lut_ptr = lut_slice.as_ptr();
        let mask_vec = _mm256_set1_epi32(mask as i32);

        // Process 8 bytes at a time using AVX2 gather.
        let mut i = 0;
        while i + 8 <= n {
            // Load 8 bytes (low 64 bits of __m128i), zero-extend to 8× i32.
            let raw_bytes = _mm_cvtepu8_epi32(_mm_loadl_epi64(
                codes.as_ptr().add(i) as *const _,
            ));
            // Widen to 256-bit, shift right + mask on i32 lanes.
            let idx256 = _mm256_castsi128_si256(raw_bytes);
            let shifted = _mm256_srli_epi32(idx256, shift as i32);
            let masked = _mm256_and_si256(shifted, mask_vec);
            // Gather 8× f32 from LUT base using the masked indices.
            // Scale = 4 (sizeof(f32)); indices are byte offsets = idx * 4.
            let gathered = _mm256_i32gather_ps(lut_ptr, masked, 4);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), gathered);
            i += 8;
        }
        // Tail (1–7 remaining elements): scalar fallback.
        while i < n {
            let idx = ((codes[i] >> shift) & mask) as usize;
            out[i] = *lut_slice.get_unchecked(idx & (lut_slice.len() - 1));
            i += 1;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Fused DeQuant + Dot (Phase 3 — the strongest fusion candidate)
// ──────────────────────────────────────────────────────────────────────────
// The fused kernel avoids spilling dequantized values to memory. Instead of
// `dequant_via_lut → out[]` then `simd_dot_f32(out, x)`, we gather the LUT value
// into a register and immediately FMA it with `x[i]` into the accumulator.
// This is the software analog of the paper's fused DQ-GEMM kernel.

/// Generic fused LUT dequantize + dot product.
///
/// Computes `Σ lut[(codes[i] >> shift) & mask] × x[i]` without spilling the
/// dequantized values to memory. The LUT already contains the affine
/// `(code − zero) · scale`, so this is `Σ ((code_i − zero) · scale) × x_i` — the
// fused dequantize-dot kernel.
///
/// Returns `0.0` for empty inputs. Processes `min(codes.len(), x.len())` elements.
///
/// # Allocation discipline
///
/// Zero allocations. All intermediate values stay in registers (SIMD paths) or
/// a single stack accumulator (scalar path).
///
/// # Example
///
/// ```
/// use katgpt_core::simd_lut_dequant::{UInt4Lut, QuantLut, dequant_dot_via_lut};
///
/// let lut = UInt4Lut::build(1.0, 0.0); // lut[i] = i
/// let codes = [0x02_u8, 0x03]; // low nibbles: 2, 3
/// let x = [10.0_f32, 20.0];
/// let dot = dequant_dot_via_lut(&codes, &lut, &x, 0, 0x0F);
/// assert_eq!(dot, 2.0 * 10.0 + 3.0 * 20.0); // 80.0
/// ```
#[inline]
#[allow(clippy::needless_return)] // return is needed for cfg-gated arch dispatch
pub fn dequant_dot_via_lut<L: QuantLut>(
    codes: &[u8],
    lut: &L,
    x: &[f32],
    shift: u32,
    mask: u8,
) -> f32 {
    let lut_slice = lut.as_f32_slice();
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dequant_dot_via_lut_neon(codes, lut_slice, x, shift, mask) };
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { dequant_dot_via_lut_avx2(codes, lut_slice, x, shift, mask) };
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        dequant_dot_via_lut_scalar(codes, lut_slice, x, shift, mask)
    }
}

/// Scalar fused dequant+dot — portable reference and correctness oracle.
///
/// Uses 4 independent accumulators with `mul_add` to match the SIMD path's
/// FMA semantics (single-rounding), same pattern as `simd::scalar_dot_f32`.
#[inline]
pub fn dequant_dot_via_lut_scalar(
    codes: &[u8],
    lut_slice: &[f32],
    x: &[f32],
    shift: u32,
    mask: u8,
) -> f32 {
    let n = codes.len().min(x.len());
    let mask_len = lut_slice.len();
    let mut acc = [0.0_f32; 4];
    let chunks = n / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            let idx0 = ((codes[i] >> shift) & mask) as usize & (mask_len - 1);
            let idx1 = ((codes[i + 1] >> shift) & mask) as usize & (mask_len - 1);
            let idx2 = ((codes[i + 2] >> shift) & mask) as usize & (mask_len - 1);
            let idx3 = ((codes[i + 3] >> shift) & mask) as usize & (mask_len - 1);
            acc[0] = lut_slice.get_unchecked(idx0).mul_add(*x.get_unchecked(i), acc[0]);
            acc[1] = lut_slice.get_unchecked(idx1).mul_add(*x.get_unchecked(i + 1), acc[1]);
            acc[2] = lut_slice.get_unchecked(idx2).mul_add(*x.get_unchecked(i + 2), acc[2]);
            acc[3] = lut_slice.get_unchecked(idx3).mul_add(*x.get_unchecked(i + 3), acc[3]);
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < n {
        let idx = ((codes[i] >> shift) & mask) as usize & (mask_len - 1);
        unsafe {
            sum = lut_slice.get_unchecked(idx).mul_add(*x.get_unchecked(i), sum);
        }
        i += 1;
    }
    sum
}

// ── NEON fused dequant+dot ─────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dequant_dot_via_lut_neon(
    codes: &[u8],
    lut_slice: &[f32],
    x: &[f32],
    shift: u32,
    mask: u8,
) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vdup_n_s8, vdup_n_u8, vdupq_n_f32, vfmaq_f32, vget_lane_u8, vld1q_f32,
        vld1_u8, vand_u8, vshl_u8,
    };

    unsafe {
        let n = codes.len().min(x.len());
        let neg_shift = (shift as i8).wrapping_neg();
        let shift_vec = vdup_n_s8(neg_shift);
        let mask_vec = vdup_n_u8(mask);
        let mask_bits = lut_slice.len().trailing_zeros() as u8;
        let idx_mask: u8 = if mask_bits >= 8 { 0xFF } else { (1u8 << mask_bits) - 1 };

        // 4 independent accumulators (float32x4_t each) to hide FMA latency.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0;
        // Process 16 elements per iteration (4× float32x4_t FMA).
        while i + 16 <= n {
            for &off in &[0, 4, 8, 12] {
                let code_vec = vld1_u8(codes.as_ptr().add(i + off));
                let shifted = vshl_u8(code_vec, shift_vec);
                let masked = vand_u8(shifted, mask_vec);
                // Scalar gather into [f32;4] — NEON has no gather.
                let gathered = [
                    *lut_slice.get_unchecked((vget_lane_u8(masked, 0) & idx_mask) as usize),
                    *lut_slice.get_unchecked((vget_lane_u8(masked, 1) & idx_mask) as usize),
                    *lut_slice.get_unchecked((vget_lane_u8(masked, 2) & idx_mask) as usize),
                    *lut_slice.get_unchecked((vget_lane_u8(masked, 3) & idx_mask) as usize),
                ];
                let dequant_vec = vld1q_f32(gathered.as_ptr());
                let x_vec = vld1q_f32(x.as_ptr().add(i + off));
                match off {
                    0 => acc0 = vfmaq_f32(acc0, dequant_vec, x_vec),
                    4 => acc1 = vfmaq_f32(acc1, dequant_vec, x_vec),
                    8 => acc2 = vfmaq_f32(acc2, dequant_vec, x_vec),
                    12 => acc3 = vfmaq_f32(acc3, dequant_vec, x_vec),
                    _ => unreachable!(),
                }
            }
            i += 16;
        }

        // Reduce 4 accumulators to 1.
        let mut acc = vaddq_f32(acc0, acc1);
        acc = vaddq_f32(acc, acc2);
        acc = vaddq_f32(acc, acc3);

        // Process remaining 4-element chunks.
        while i + 4 <= n {
            let code_vec = vld1_u8(codes.as_ptr().add(i));
            let shifted = vshl_u8(code_vec, shift_vec);
            let masked = vand_u8(shifted, mask_vec);
            let gathered = [
                *lut_slice.get_unchecked((vget_lane_u8(masked, 0) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 1) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 2) & idx_mask) as usize),
                *lut_slice.get_unchecked((vget_lane_u8(masked, 3) & idx_mask) as usize),
            ];
            let dequant_vec = vld1q_f32(gathered.as_ptr());
            let x_vec = vld1q_f32(x.as_ptr().add(i));
            acc = vfmaq_f32(acc, dequant_vec, x_vec);
            i += 4;
        }

        // Horizontal sum of float32x4_t.
        let mut sum = core::arch::aarch64::vaddvq_f32(acc);

        // Tail (1–3 remaining elements): scalar.
        while i < n {
            let idx = ((codes[i] >> shift) & mask) as usize & (lut_slice.len() - 1);
            sum = lut_slice
                .get_unchecked(idx)
                .mul_add(*x.get_unchecked(i), sum);
            i += 1;
        }
        sum
    }
}

// ── AVX2 fused dequant+dot ─────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dequant_dot_via_lut_avx2(
    codes: &[u8],
    lut_slice: &[f32],
    x: &[f32],
    shift: u32,
    mask: u8,
) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_and_si256, _mm256_castsi128_si256, _mm256_fmadd_ps,
        _mm256_i32gather_ps, _mm256_loadu_ps, _mm256_set1_epi32, _mm256_setzero_ps,
        _mm256_srli_epi32, _mm_cvtepu8_epi32, _mm_loadl_epi64,
    };

    unsafe {
        let n = codes.len().min(x.len());
        let lut_ptr = lut_slice.as_ptr();
        let mask_vec = _mm256_set1_epi32(mask as i32);

        // 2 independent accumulators (float32x8_t each) to hide FMA latency.
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let mut i = 0;
        // Process 16 elements per iteration (2× float32x8_t FMA).
        while i + 16 <= n {
            for (slot, &off) in [0, 8].iter().enumerate() {
                let raw = _mm_cvtepu8_epi32(_mm_loadl_epi64(
                    codes.as_ptr().add(i + off) as *const _,
                ));
                let idx256 = _mm256_castsi128_si256(raw);
                let shifted = _mm256_srli_epi32(idx256, shift as i32);
                let masked = _mm256_and_si256(shifted, mask_vec);
                let gathered = _mm256_i32gather_ps(lut_ptr, masked, 4);
                let x_vec = _mm256_loadu_ps(x.as_ptr().add(i + off));
                match slot {
                    0 => acc0 = _mm256_fmadd_ps(gathered, x_vec, acc0),
                    1 => acc1 = _mm256_fmadd_ps(gathered, x_vec, acc1),
                    _ => unreachable!(),
                }
            }
            i += 16;
        }

        // Process remaining 8-element chunk.
        let mut acc = _mm256_add_ps(acc0, acc1);
        if i + 8 <= n {
            let raw = _mm_cvtepu8_epi32(_mm_loadl_epi64(
                codes.as_ptr().add(i) as *const _,
            ));
            let idx256 = _mm256_castsi128_si256(raw);
            let shifted = _mm256_srli_epi32(idx256, shift as i32);
            let masked = _mm256_and_si256(shifted, mask_vec);
            let gathered = _mm256_i32gather_ps(lut_ptr, masked, 4);
            let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
            acc = _mm256_fmadd_ps(gathered, x_vec, acc);
            i += 8;
        }

        // Horizontal sum of __m256.
        let mut sum = {
            let lo = core::arch::x86_64::_mm256_castps256_ps128(acc);
            let hi = core::arch::x86_64::_mm256_extractf128_ps(acc, 1);
            let sum128 = core::arch::x86_64::_mm_add_ps(lo, hi);
            let shuf = core::arch::x86_64::_mm_movehdup_ps(sum128);
            let sums = core::arch::x86_64::_mm_add_ps(sum128, shuf);
            let shuf2 = core::arch::x86_64::_mm_movehl_ps(sums, sums);
            let total = core::arch::x86_64::_mm_add_ss(sums, shuf2);
            core::arch::x86_64::_mm_cvtss_f32(total)
        };

        // Tail (1–7 remaining elements): scalar.
        while i < n {
            let idx = ((codes[i] >> shift) & mask) as usize & (lut_slice.len() - 1);
            sum = lut_slice
                .get_unchecked(idx)
                .mul_add(*x.get_unchecked(i), sum);
            i += 1;
        }
        sum
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Scalar reference arithmetic dequantize — the "current path" comparator.
///
/// Computes `(signed(code) − zero) · scale` per element, matching the arithmetic
/// cast path in `q4k.rs` today. Used by the GOAT gate (Plan 452 Phase 4 G1) to
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

    // ── Bit-exact: LUT path == arithmetic path (Plan 452 Phase 4 G1) ─────

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

    // ── SIMD vs scalar bit-exact (Phase 2 T2.1–T2.3) ──────────────────────

    /// Verify the SIMD path (NEON/AVX2 dispatch) is bit-exact against the
    /// scalar reference path on a large input that exercises the unrolled
    /// loop (multiples of 8) AND the tail (non-multiple of 8).
    #[test]
    fn test_simd_vs_scalar_bit_exact_uint4_large() {
        let scale = 0.123_f32;
        let zero = 3.7_f32;
        let lut = UInt4Lut::build(scale, zero);
        let lut_slice = lut.as_f32_slice();
        // 100 elements: 12×8=96 in the unrolled loop + 4 tail.
        let codes: Vec<u8> = (0..100u32).map(|i| (i * 7) as u8).collect();
        let mut out_simd = vec![0.0_f32; 100];
        let mut out_scalar = vec![0.0_f32; 100];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out_simd);
        dequant_via_lut_scalar(&codes, lut_slice, 0, 0x0F, &mut out_scalar);
        assert_eq!(out_simd, out_scalar, "SIMD path must be bit-exact vs scalar");
    }

    #[test]
    fn test_simd_vs_scalar_bit_exact_int8_large() {
        let scale = 0.05_f32;
        let zero = 100.0_f32;
        let lut = Int8Lut::build(scale, zero);
        let lut_slice = lut.as_f32_slice();
        // 300 elements: 37×8=296 in the unrolled loop + 4 tail.
        let codes: Vec<u8> = (0..300u32).map(|i| (i * 3) as u8).collect();
        let mut out_simd = vec![0.0_f32; 300];
        let mut out_scalar = vec![0.0_f32; 300];
        dequant_via_lut(&codes, &lut, 0, 0xFF, &mut out_simd);
        dequant_via_lut_scalar(&codes, lut_slice, 0, 0xFF, &mut out_scalar);
        assert_eq!(out_simd, out_scalar, "SIMD path must be bit-exact vs scalar");
    }

    #[test]
    fn test_simd_vs_scalar_bit_exact_high_nibble() {
        let lut = UInt4Lut::build(0.9, 2.0);
        let lut_slice = lut.as_f32_slice();
        let codes: Vec<u8> = (0..200u32).map(|i| (i * 13) as u8).collect();
        let mut out_simd = vec![0.0_f32; 200];
        let mut out_scalar = vec![0.0_f32; 200];
        dequant_via_lut(&codes, &lut, 4, 0x0F, &mut out_simd);
        dequant_via_lut_scalar(&codes, lut_slice, 4, 0x0F, &mut out_scalar);
        assert_eq!(out_simd, out_scalar);
    }

    #[test]
    fn test_simd_vs_scalar_exact_alignment_boundary() {
        // Test at exact 8-element boundaries: 8, 16, 24 elements (no tail).
        let lut = UInt4Lut::build(1.0, 0.0);
        let lut_slice = lut.as_f32_slice();
        for &n in &[8usize, 16, 24, 7, 9, 15] {
            let codes: Vec<u8> = (0..n as u32).map(|i| (i | 0x80) as u8).collect();
            let mut out_simd = vec![0.0_f32; n];
            let mut out_scalar = vec![0.0_f32; n];
            dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out_simd);
            dequant_via_lut_scalar(&codes, lut_slice, 0, 0x0F, &mut out_scalar);
            assert_eq!(out_simd, out_scalar, "mismatch at n={}", n);
        }
    }

    // ── Fused DeQuant + Dot (Phase 3 T3.1–T3.3) ───────────────────────────

    #[test]
    fn test_dequant_dot_basic() {
        let lut = UInt4Lut::build(1.0, 0.0); // lut[i] = i
        let codes = [0x02_u8, 0x03];
        let x = [10.0_f32, 20.0];
        let dot = dequant_dot_via_lut(&codes, &lut, &x, 0, 0x0F);
        assert_eq!(dot, 2.0 * 10.0 + 3.0 * 20.0);
    }

    #[test]
    fn test_dequant_dot_empty() {
        let lut = UInt4Lut::build(1.0, 0.0);
        let codes: [u8; 0] = [];
        let x: [f32; 0] = [];
        assert_eq!(dequant_dot_via_lut(&codes, &lut, &x, 0, 0x0F), 0.0);
    }

    #[test]
    fn test_dequant_dot_with_scale_zero() {
        // lut[i] = (i - 5) * 2 = {−10,−8,−6,...,4}
        let lut = UInt4Lut::build(2.0, 5.0);
        let codes = [0x00_u8, 0x01, 0x02, 0x03];
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let dot = dequant_dot_via_lut(&codes, &lut, &x, 0, 0x0F);
        // Manual: (−10)(1) + (−8)(2) + (−6)(3) + (−4)(4) = −10 − 16 − 18 − 16 = −60
        assert_eq!(dot, -60.0);
    }

    /// Fused dequant+dot must produce the same result as the two-step path:
    /// (1) dequant codes into buffer, (2) dot product of buffer with x.
    /// Due to FMA reordering the result may not be bit-exact, but should be
    /// within f32 precision (~1e-5 relative).
    #[test]
    fn test_fused_vs_two_step_close() {
        let lut = UInt4Lut::build(0.03, 4.2);
        let lut_slice = lut.as_f32_slice();
        let codes: Vec<u8> = (0..128u32).map(|i| (i * 17) as u8).collect();
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1 - 6.4).sin()).collect();

        // Two-step: dequant then scalar dot.
        let mut buf = vec![0.0_f32; 128];
        dequant_via_lut_scalar(&codes, lut_slice, 0, 0x0F, &mut buf);
        let two_step: f32 = buf.iter().zip(&x).map(|(b, xi)| b * xi).sum();

        // Fused.
        let fused = dequant_dot_via_lut(&codes, &lut, &x, 0, 0x0F);

        // Should be very close (FMA can differ by ~1 ULP per accumulation).
        let rel_diff = (fused - two_step).abs() / two_step.abs().max(1e-10);
        assert!(
            rel_diff < 1e-5,
            "fused={} two_step={} rel_diff={}",
            fused,
            two_step,
            rel_diff
        );
    }

    /// Fused SIMD path must match the fused scalar path within f32 precision.
    #[test]
    fn test_fused_simd_vs_scalar_close() {
        let lut = Int8Lut::build(0.01, 100.0);
        let lut_slice = lut.as_f32_slice();
        // 300 elements: exercises 16-element loop + 8-element chunk + tail.
        let codes: Vec<u8> = (0..300u32).map(|i| (i * 7) as u8).collect();
        let x: Vec<f32> = (0..300).map(|i| (i as f32 * 0.03).cos()).collect();

        let scalar = dequant_dot_via_lut_scalar(&codes, lut_slice, &x, 0, 0xFF);
        let simd = dequant_dot_via_lut(&codes, &lut, &x, 0, 0xFF);

        let rel_diff = (simd - scalar).abs() / scalar.abs().max(1e-10);
        assert!(
            rel_diff < 1e-5,
            "simd={} scalar={} rel_diff={}",
            simd,
            scalar,
            rel_diff
        );
    }

    /// Fused path at exact SIMD-width boundaries (16, 8, 4, tail).
    #[test]
    fn test_fused_alignment_boundaries() {
        let lut = UInt4Lut::build(0.5, 3.0);
        let lut_slice = lut.as_f32_slice();
        for &n in &[1usize, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33] {
            let codes: Vec<u8> = (0..n as u32).map(|i| (i * 11) as u8).collect();
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
            let scalar = dequant_dot_via_lut_scalar(&codes, lut_slice, &x, 0, 0x0F);
            let simd = dequant_dot_via_lut(&codes, &lut, &x, 0, 0x0F);
            let rel_diff = (simd - scalar).abs() / scalar.abs().max(1e-10);
            assert!(
                rel_diff < 1e-5 || simd.abs() < 1e-10,
                "n={}: simd={} scalar={} rel_diff={}",
                n,
                simd,
                scalar,
                rel_diff
            );
        }
    }
}
