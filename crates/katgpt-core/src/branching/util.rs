//! Shared hot-path numeric helpers for the `branching` module.
//!
//! Previously `dot` was duplicated verbatim in `bank.rs`, `router.rs`, and
//! `projection.rs` (as `dot_fixed`). Centralising here removes the DRY
//! violation while keeping the same codegen contract: zero-allocation,
//! auto-vectorizable inner loop, `#[inline]` so each call site still
//! specialises to its actual slice length.

/// Dot product of two f32 slices.
///
/// Uses the shorter length when dimensions mismatch — callers are expected to
/// pre-normalise dimensions.
///
/// Session 32 (Bench 636 Phase 4): switched from `sum += a[i] * b[i]`
/// (double-rounding: mul then add) to `a[i].mul_add(b[i], sum)` (single-
/// rounding FMA). This matches the SIMD paths in `katgpt_types::simd::dot`
/// which already use NEON `vfmaq` / AVX2 `_mm256_fmadd_ps` — both are FMA.
/// The routing decision uses threshold comparisons (`>= tau_snap`); the
/// last-ULP difference between FMA and non-FMA does not flip any threshold.
/// At 8-dim HLA (the production case), this gives the compiler the FMA
/// contraction hint it needs on targets without hardware FMA (or when
/// `-C contraction=off`). On aarch64 (which has hardware FMA), the scalar
/// loop was already contracting to `fmadd` — this change is a no-op on
/// aarch64 but helps x86_64 / WASM / debug builds.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum = a[i].mul_add(b[i], sum);
    }
    sum
}

/// Dot product of a fixed-size `D`-array with a (possibly shorter) slice.
///
/// Const-generic specialisation: when `b.len() >= D`, LLVM can fully unroll
/// the `D`-length inner loop. Falls back to `D.min(b.len())` if the caller
/// passes a shorter vector (defensive — callers should pre-normalise).
///
/// Session 32 (Bench 636 Phase 4): same FMA switch as `dot` above.
#[inline]
pub(crate) fn dot_fixed<const D: usize>(a: &[f32; D], b: &[f32]) -> f32 {
    let n = D.min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum = a[i].mul_add(b[i], sum);
    }
    sum
}
