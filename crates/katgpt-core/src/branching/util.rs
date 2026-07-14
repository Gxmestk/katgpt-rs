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
/// pre-normalise dimensions. The `for i in 0..n` form (rather than
/// `a.iter().zip(b)`) is preserved because LLVM already auto-vectorises it
/// and the indexed form keeps the door open for an `unsafe get_unchecked`
/// fast path if profiling ever justifies it.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Dot product of a fixed-size `D`-array with a (possibly shorter) slice.
///
/// Const-generic specialisation: when `b.len() >= D`, LLVM can fully unroll
/// the `D`-length inner loop. Falls back to `D.min(b.len())` if the caller
/// passes a shorter vector (defensive — callers should pre-normalise).
#[inline]
pub(crate) fn dot_fixed<const D: usize>(a: &[f32; D], b: &[f32]) -> f32 {
    let n = D.min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}
