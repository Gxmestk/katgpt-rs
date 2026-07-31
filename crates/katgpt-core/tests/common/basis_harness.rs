//! Shared structured-basis construction harness for the Plan 332 basis-probe
//! tests.
//!
//! Consolidates the bit-identical helpers that were previously copy-pasted
//! across:
//! - `tests/apollonian_basis_probe.rs` (Plan 332 T5.1 rotation-invariance probe)
//! - `tests/funcattn_structured_basis_g1.rs` (Plan 332 G1 gate)
//! - `tests/funcattn_structured_basis_k_sweep.rs` (Plan 332 k-sweep)
//!
//! # Why shared (DRY case)
//!
//! All five helpers were byte-identical across at least two of the three
//! consumers before extraction (3 share all of `l2_normalize` /
//! `gram_schmidt_rows` / `cosine`; 2 of the 3 also share `lcg_next` +
//! `random_orthonormal_w`). The PRNG constants and the L2 normalization
//! epsilon (`1e-12`) are tuned together; changing one in isolation would
//! silently break the G1 / G2 cosine-similarity gates in every consumer.
//!
//! # What stays inline
//!
//! - `apollonian_basis_probe.rs`'s `random_orthonormal_w_rect`: uses an
//!   inline closure RNG (different lineage from `lcg_next`), so it is NOT
//!   consolidated.
//! - The Plan 332-specific signal builders (`make_multiscale_x`,
//!   `make_broadband_signal`): per-test variants, not shared.
//!
//! # Usage from a test file
//!
//! ```ignore
//! #[path = "common/basis_harness.rs"]
//! mod basis_harness;
//! use basis_harness::{cosine, gram_schmidt_rows, l2_normalize};
//! // or, for files that also need the LCG + orthonormal-w helpers:
//! // use basis_harness::{cosine, gram_schmidt_rows, l2_normalize, lcg_next,
//! //     random_orthonormal_w};
//! ```
//!
//! From a bench file, the path is `#[path = "../tests/common/basis_harness.rs"]`.

/// PCG-style LCG step. Returns values centered around 0 in approximately
/// `[-0.5, 0.5)`. Matches the canonical PCG `state · mul + add` form with the
/// published Numeric Recipes constants (`6364136223846793005` /
/// `1442695040888963407`).
///
/// `#[allow(dead_code)]`: only 2 of the 3 consumers use this;
/// `apollonian_basis_probe.rs` uses an inline closure RNG instead.
#[allow(dead_code)]
pub fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (1u64 << 31) as f32 - 0.5
}

/// In-place L2 normalization. The `1e-12` floor avoids division-by-zero on
/// a zero vector; matches the epsilon every consumer used.
pub fn l2_normalize(v: &mut [f32]) {
    let mut s = 0.0f32;
    for &x in v.iter() {
        s += x * x;
    }
    let norm = s.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// Modified Gram-Schmidt over row-major `(k, d)` matrix `w` (rows =
/// vectors to orthonormalize). Operates in place; produces row-orthonormal
/// output (rows are orthonormal).
pub fn gram_schmidt_rows(w: &mut [f32], k: usize, d: usize) {
    for i in 0..k {
        for j in 0..i {
            let mut dot = 0.0f32;
            for l in 0..d {
                dot += w[i * d + l] * w[j * d + l];
            }
            for l in 0..d {
                w[i * d + l] -= dot * w[j * d + l];
            }
        }
        l2_normalize(&mut w[i * d..(i + 1) * d]);
    }
}

/// Cosine similarity between two equal-length slices.
///
/// Uses `.max(1e-12)` on the denominator to avoid division-by-zero —
/// the canonical Plan 332 lineage epsilon. (Other tests in the workspace
/// use different zero-handling conventions; do not blindly consolidate
/// across lineages.)
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

/// Build a `(k, d)` matrix with orthonormal rows from a seed. Used to
/// construct basis matrices `w_basis` / `w_q` / `w_k` / `w_v`.
///
/// `#[allow(dead_code)]`: only 2 of the 3 consumers use this;
/// `apollonian_basis_probe.rs` uses its own `random_orthonormal_w_rect`
/// with an inline closure RNG (different lineage).
#[allow(dead_code)]
pub fn random_orthonormal_w(seed: u64, k: usize, d: usize) -> Vec<f32> {
    let mut s = seed;
    let mut w = vec![0.0f32; k * d];
    for v in w.iter_mut() {
        *v = lcg_next(&mut s);
    }
    gram_schmidt_rows(&mut w, k, d);
    w
}
