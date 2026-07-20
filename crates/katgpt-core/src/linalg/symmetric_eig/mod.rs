//! Symmetric eigendecomposition via Householder tridiagonalization + implicit-shift QL.
//!
//! Drop-in alternative to [`crate::karc::jacobi_eigen`] for large symmetric
//! matrices. Same storage convention (`eigvecs[i*n + j] = V[i, j]` row-major,
//! column k of V paired with `eigvals[k]`), but ~5-10× faster at `n ≥ 256`
//! because:
//! - Householder reduction is `O(n³/3)` with cache-friendly blocked access
//!   (one pass over the lower triangle per reflection).
//! - Tridiagonal QL sweeps cost `O(n²)` per sweep, not `O(n³)` like full Jacobi
//!   sweeps. Two-three sweeps typically suffice per eigenvalue.
//!
//! Total cost: `~3·n³` FLOPs vs Jacobi's `~50·n³` (10 sweeps × `~5n³` per sweep).
//!
//! # Algorithm (Golub-van Loan §8.3 + Numerical Recipes §11.3)
//!
//! **Phase 1 — Householder tridiagonalization (Golub-van Loan 8.3.1).**
//! Reduces `A` to symmetric tridiagonal `T = Qᵀ · A · Q` via `n-2` Householder
//! reflections. Q is accumulated into the `eigvecs` output (starting from I).
//!
//! **Phase 2 — Implicit-shift QL with Wilkinson shift (Golub-van Loan 8.3.3,
//! Numerical Recipes `tqli`).** Reduces `T` to diagonal form via Givens
//! rotations, accumulating into Q. The implicit shift avoids the explicit
//! `O(n²)` per-iteration shift computation and the corresponding accuracy loss.
//!
//! # Determinism
//!
//! Pure Rust, deterministic sweep order, no platform-dependent dispatch. Two
//! calls with identical inputs produce bit-identical outputs on the same host.
//! This is what backs the `karc_householder_eig` feature gate's G4 bit-
//! reproducibility contract (Issue 186 T4).
//!
//! # References
//!
//! - Golub-van Loan, *Matrix Computations* 4th ed., §8.3 (Symmetric QR algorithm).
//! - Press-Teukolsky-Vetterling-Flannery, *Numerical Recipes* 3rd ed., §11.3
//!   (`tqli` — Tridiagonal QL Implicit).
//! - Issue 186 — Path B deliberation + acceptance criteria.
//! - Issue 185 — T1+T2 implementation of the consumer
//!   (`karc::large_dh::low_rank_fit_jacobi_bstep`).

#[cfg(test)]
mod tests;

/// Scratch buffers for [`symmetric_eig`]. Owns ~`n²` f64 for the working copy
/// of A plus `O(n)`-sized temps. Lazily sized via [`ensure_capacity`].
///
/// [`ensure_capacity`]: SymmetricEigScratch::ensure_capacity
#[derive(Default)]
pub struct SymmetricEigScratch {
    /// Working copy of `A` (`n×n` f64). Reduced to tridiagonal form during
    /// [`tridiagonalize`]; the diagonal lands in `a_work[i*n+i]` and the
    /// subdiagonal in `a_work[(i+1)*n+i]`.
    a_work: Vec<f64>,
    /// Householder vector `v` (length `n`; only the first `n-k-1` entries are
    /// meaningful at tridiagonalization step `k`).
    v_buf: Vec<f64>,
    /// `p = β · A_sub · v` temp (length `n`).
    p_buf: Vec<f64>,
    /// `w = p - K·v` temp (length `n`).
    w_buf: Vec<f64>,
    /// Tridiagonal subdiagonal `e[i] = T[i, i+1]` (length `n`; `e[n-1]` is a
    /// sentinel set to 0 by the QL phase).
    subdiag: Vec<f64>,
}

impl SymmetricEigScratch {
    /// Allocate empty. Use [`ensure_capacity`] before the first call to
    /// [`symmetric_eig`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Grow buffers to handle an `n×n` eigendecomposition. Idempotent — only
    /// allocates if the current capacity is too small.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.a_work.len() < n * n {
            self.a_work.resize(n * n, 0.0);
        }
        if self.v_buf.len() < n {
            self.v_buf.resize(n, 0.0);
        }
        if self.p_buf.len() < n {
            self.p_buf.resize(n, 0.0);
        }
        if self.w_buf.len() < n {
            self.w_buf.resize(n, 0.0);
        }
        if self.subdiag.len() < n {
            self.subdiag.resize(n, 0.0);
        }
    }
}

/// Symmetric eigendecomposition via Householder tridiagonalization + implicit-shift QL.
///
/// Computes `A = V · diag(λ) · Vᵀ` for a symmetric `n×n` row-major f64 matrix
/// `a_in`, writing eigenvalues into `eigvals` (length `n`, unsorted) and
/// eigenvectors into `eigvecs` (length `n*n`, row-major: `eigvecs[i*n + j] = V[i, j]`,
/// so column `k` of `V` — the eigenvector paired with `eigvals[k]` — is the
/// length-`n` stride-`n` slice starting at `eigvecs[k]`).
///
/// Storage convention matches [`crate::karc::jacobi_eigen`] exactly. This
/// function is a drop-in replacement for the G-path (large `d_h`) call in
/// `karc::large_dh::low_rank_fit_jacobi_bstep`; the A-path (`r×r`) stays on
/// Jacobi where its constant-factor is smaller.
///
/// # When to prefer this over Jacobi
///
/// - `n ≤ 16`: use Jacobi — lower constant, equally accurate.
/// - `n ≥ 256`: use this — 5-10× faster due to the `O(n³)` vs `O(n⁴)`-ish
///   sweep-cost gap (Jacobi does ~10 sweeps × `O(n³)` each; Householder+QL
///   does one `O(n³/3)` reduction + `O(n³)` worth of QL rotations).
/// - `n ≥ 18_720` (Issue 186 target): Jacobi is infeasible (~16 hours
///   projected); Householder+QL is the only modelless path to a feasible
///   one-time G eigendecomp.
///
/// # Panics
///
/// - Input slice length mismatches (debug_assert only — release builds skip).
/// - QL fails to converge after `max_iters_per_eigval` iterations per
///   eigenvalue (Numerical Recipes uses 30; we follow). This indicates either
///   a pathologically close-to-defective matrix or a bug in the implementation.
///   Caller should regularize (e.g. `A + ε·I`) or investigate.
///
/// # Determinism
///
/// Pure Rust, fixed sweep order. Bit-identical output for identical input.
pub fn symmetric_eig(
    eigvals: &mut [f64],
    eigvecs: &mut [f64],
    a_in: &[f64],
    scratch: &mut SymmetricEigScratch,
    n: usize,
    max_iters_per_eigval: usize,
) {
    debug_assert_eq!(a_in.len(), n * n, "a_in must be n*n");
    debug_assert_eq!(eigvals.len(), n, "eigvals must be n");
    debug_assert_eq!(eigvecs.len(), n * n, "eigvecs must be n*n");

    scratch.ensure_capacity(n);

    // Initialize eigvecs = I (will accumulate Q from Householder + Q' from QL).
    for i in 0..n {
        for j in 0..n {
            eigvecs[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    // Copy A into the working buffer.
    scratch.a_work[..n * n].copy_from_slice(&a_in[..n * n]);

    // Phase 1: reduce A to tridiagonal T = Qᵀ A Q, accumulating Q into eigvecs.
    tridiagonalize(
        &mut scratch.a_work[..n * n],
        n,
        eigvecs,
        &mut scratch.v_buf[..n],
        &mut scratch.p_buf[..n],
        &mut scratch.w_buf[..n],
    );

    // Extract diagonal into eigvals and subdiagonal into scratch.subdiag.
    // After tridiagonalize, a_work is tridiagonal:
    //   a_work[i*n + i]     = T[i, i]
    //   a_work[(i+1)*n + i] = T[i+1, i] = T[i, i+1]  (symmetric)
    //   everything else is 0.
    for (i, e) in eigvals.iter_mut().enumerate().take(n) {
        *e = scratch.a_work[i * n + i];
    }
    for i in 0..(n - 1) {
        scratch.subdiag[i] = scratch.a_work[(i + 1) * n + i];
    }
    if n > 0 {
        scratch.subdiag[n - 1] = 0.0; // Sentinel required by tqli.
    }

    // Phase 2: implicit-shift QL on the tridiagonal, accumulating into eigvecs.
    tqli_implicit_shift(
        eigvals,
        &mut scratch.subdiag[..n],
        n,
        eigvecs,
        max_iters_per_eigval,
    );
}

/// Householder tridiagonalization with explicit Q accumulation.
///
/// Reduces `a` (n×n symmetric row-major) to tridiagonal form in place, and
/// accumulates the orthogonal factor into `q` (which must start as `I`).
/// Post-condition: `a` is tridiagonal (the lower triangle below the subdiagonal
/// is zeroed by the reflections; the upper triangle is symmetric).
///
/// Uses scratch buffers `v`, `p`, `w` (each length `n`).
///
/// Algorithm: Golub-van Loan §8.3.1. For each `k = 0..n-2`, compute the
/// Householder vector that zeros `a[k+2..n, k]`, apply `H · A · H` to the
/// bottom-right block, and accumulate `Q · H` into `q`.
fn tridiagonalize(
    a: &mut [f64],
    n: usize,
    q: &mut [f64],
    v: &mut [f64],
    p: &mut [f64],
    w: &mut [f64],
) {
    if n <= 2 {
        // 1x1 and 2x2 symmetric matrices are already tridiagonal.
        return;
    }
    for k in 0..(n - 2) {
        let block_start = k + 1;
        let block_size = n - block_start;

        // ||x||² where x = a[k+1..n, k] is the part of column k below the diagonal.
        let mut sigma = 0.0_f64;
        for i in block_start..n {
            let aik = a[i * n + k];
            sigma += aik * aik;
        }
        if sigma < f64::MIN_POSITIVE {
            // Column already zero below the diagonal — skip (no reflection needed).
            continue;
        }
        let norm_x = sigma.sqrt();

        // alpha = -sign(a[k+1, k]) * ||x||  (sign chosen to avoid cancellation in v[0]).
        let a_k1_k = a[block_start * n + k];
        let alpha = if a_k1_k >= 0.0 { -norm_x } else { norm_x };

        // v[0..block_size] = x - alpha * e_1
        //   v[0] = a[k+1, k] - alpha
        //   v[i>0] = a[k+1+i, k]
        v[0] = a_k1_k - alpha;
        for i in 1..block_size {
            v[i] = a[(block_start + i) * n + k];
        }

        let mut v_norm_sq = 0.0_f64;
        for &vi in v.iter().take(block_size) {
            v_norm_sq += vi * vi;
        }
        if v_norm_sq < f64::MIN_POSITIVE {
            continue;
        }
        let beta = 2.0 / v_norm_sq;

        // Apply A_sub ← H A_sub H where A_sub = a[block_start..n, block_start..n].
        // For symmetric A_sub, this factors as:
        //   p = beta * A_sub * v
        //   K = beta * (v·p) / 2
        //   w = p - K * v
        //   A_sub ← A_sub - v wᵀ - w vᵀ   (symmetric rank-2 update)
        for (i, p_i) in p.iter_mut().enumerate().take(block_size) {
            let row_offset = (block_start + i) * n + block_start;
            let mut s = 0.0_f64;
            for (j, &vj) in v.iter().enumerate().take(block_size) {
                s += a[row_offset + j] * vj;
            }
            *p_i = beta * s;
        }
        let mut vp = 0.0_f64;
        for i in 0..block_size {
            vp += v[i] * p[i];
        }
        let k_coef = beta * vp * 0.5;
        for i in 0..block_size {
            w[i] = p[i] - k_coef * v[i];
        }
        // Symmetric rank-2 update: A_sub[i,j] -= v[i]*w[j] + w[i]*v[j].
        // Cache-friendly: one pass per row of A_sub; v, w fit in L1/L2.
        for i in 0..block_size {
            let vi = v[i];
            let wi = w[i];
            let row_offset = (block_start + i) * n + block_start;
            for j in 0..block_size {
                a[row_offset + j] -= vi * w[j] + wi * v[j];
            }
        }

        // Set the new subdiagonal value at (k+1, k) and zero the rest of column
        // k below the subdiagonal. Symmetric: also update row k.
        a[block_start * n + k] = alpha;
        a[k * n + block_start] = alpha;
        for i in (block_start + 1)..n {
            a[i * n + k] = 0.0;
            a[k * n + i] = 0.0;
        }

        // Accumulate Q ← Q · H where H acts on rows/cols block_start..n.
        // For each row i of Q: s = beta · (Q[i, block_start..n] · v);
        // then Q[i, block_start..n] -= s · v.
        for i in 0..n {
            let row_offset = i * n + block_start;
            let mut s = 0.0_f64;
            for j in 0..block_size {
                s += q[row_offset + j] * v[j];
            }
            s *= beta;
            for j in 0..block_size {
                q[row_offset + j] -= s * v[j];
            }
        }
    }
}

/// Implicit-shift QL iteration on a symmetric tridiagonal matrix.
///
/// `d[0..n]` is the diagonal; `e[0..n]` is the subdiagonal with `e[i] = T[i, i+1]`
/// for `i = 0..n-1` and `e[n-1]` a sentinel (set to 0). Both are modified in
/// place: on return, `d` holds the eigenvalues (unsorted) and `e` is destroyed.
///
/// `z[n*n]` accumulates the eigenvectors as columns: column `k` of `z` (row-major
/// layout: indices `z[k], z[n+k], ..., z[(n-1)*n + k]`) is the eigenvector for
/// `d[k]` on return. `z` must start as the orthogonal factor `Q` from
/// [`tridiagonalize`] (or identity if the input was already tridiagonal).
///
/// Numerical Recipes `tqli` (3rd ed. §11.3) port. The implicit-shift technique
/// avoids the `O(n)` per-iteration shift application that would otherwise be
/// required for explicit-shift QR/QL; the bulge-chase applies `O(n)` Givens
/// rotations per iteration, each costing `O(n)` for eigenvector accumulation,
/// giving `O(n²)` per deflation and `O(n³)` overall.
fn tqli_implicit_shift(
    d: &mut [f64],
    e: &mut [f64],
    n: usize,
    z: &mut [f64],
    max_iters_per_eigval: usize,
) {
    if n <= 1 {
        return;
    }
    e[n - 1] = 0.0; // Sentinel: the QL bulge-chase never reads e[n-1] legitimately.

    for l in 0..n {
        let mut iter = 0;
        loop {
            // Find smallest m in [l, n-1) with |e[m]| negligible relative to
            // (|d[m]| + |d[m+1]|). This is the bottom of the active block [l, m]:
            // the matrix splits there, so we can deflate [l, m] independently.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                // NR's exact check: (|e[m]| + dd) == dd, i.e. |e[m]| is below
                // the rounding threshold of dd.
                if e[m].abs() + dd == dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                // e[l] is negligible → eigenvalue at position l is deflated.
                break;
            }

            iter += 1;
            if iter > max_iters_per_eigval {
                panic!(
                    "symmetric_eig: QL failed to converge at l={} after {} iterations \
                     (d[l..l+2]={:?}, e[l..l+2]={:?})",
                    l,
                    max_iters_per_eigval,
                    &d[l..(l + 2).min(n)],
                    &e[l..(l + 2).min(n)]
                );
            }

            // Wilkinson shift from the leading 2×2 block at (l, l+1).
            // This is the eigenvalue of [[d[l], e[l]], [e[l], d[l+1]]] closer
            // to d[l+1] (QL pushes mass toward the lower-left).
            let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = (g * g + 1.0).sqrt();
            // The shift: μ = d[m] - (d[l] - μ') where μ' is the closer-of-two
            // eigenvalue of the 2×2 block. Numerical Recipes condenses this
            // to a single expression; see NR §11.3 for the algebra.
            let r_signed = if g >= 0.0 { r } else { -r };
            let mut g = d[m] - d[l] + e[l] / (g + r_signed);

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;
            let mut broke_with_r_zero = false;

            // Bulge chase: from i = m-1 down to l. Each Givens rotation moves
            // the "bulge" one position toward the top-left; the final rotation
            // at i = l zeros out e[l], deflating d[l].
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = (f * f + g * g).sqrt();
                e[i + 1] = r;
                if r == 0.0 {
                    // Underflow recovery: if r is exactly zero, f and g are
                    // both zero, meaning e[i] = 0 (so s·e[i] = 0) AND g (the
                    // bulge from the previous step) is 0. Recover by deflating
                    // at i+1 and restarting.
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    broke_with_r_zero = true;
                    break;
                }
                s = f / r;
                c = g / r;
                let g_new = d[i + 1] - p;
                let r2 = (d[i] - g_new) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g_new + p;
                g = c * r2 - b;

                // Eigenvector accumulation: rotate columns i and i+1 of z by
                // [[c, s], [-s, c]] (the inverse transpose of the active rotation).
                // Applies to ALL rows of z, not just the active block, because the
                // QL step is a similarity transform on the full matrix.
                for k in 0..n {
                    let row_offset = k * n;
                    let f = z[row_offset + i + 1];
                    z[row_offset + i + 1] = s * z[row_offset + i] + c * f;
                    z[row_offset + i] = c * z[row_offset + i] - s * f;
                }
            }

            if broke_with_r_zero {
                // Restart the outer loop (recompute m).
                continue;
            }

            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}
