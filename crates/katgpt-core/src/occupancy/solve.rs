//! Self-contained Cholesky decomposition + triangular solves for the
//! KL-projection Newton step.
//!
//! Mirrors the proven pattern in `crate::funcattn` (Plan 029) but kept private
//! to this module — no cross-module dependency, no external linear-algebra
//! crate. All operations are in-place over caller-provided buffers (G4
//! alloc-free inner loop).

/// Cholesky-decompose a symmetric positive-definite (SPD) matrix `a` in place
/// into lower-triangular `L` (so `a = L · Lᵀ`). Row-major layout, `dim × dim`.
///
/// Returns `true` on success, `false` if `a` is not positive definite (a
/// diagonal entry became non-positive during factorization). The caller should
/// add jitter and retry on `false`.
///
/// Standard right-looking in-place Cholesky. Cost `O(dim³/3)`. For `dim ≤ 8`
/// (the FORE use case: scalar Baird feature or ≤ 8-dim HLA) this is < 200 FLOPs.
#[inline]
pub(crate) fn cholesky_inplace(a: &mut [f32], dim: usize) -> bool {
    debug_assert_eq!(a.len(), dim * dim);
    for j in 0..dim {
        let mut diag = a[j * dim + j];
        if j > 0 {
            let l_row = &a[j * dim..j * dim + j];
            for &l in l_row {
                diag -= l * l;
            }
        }
        if diag <= 0.0 {
            return false;
        }
        let sqrt_diag = diag.sqrt();
        a[j * dim + j] = sqrt_diag;
        let inv_diag = 1.0 / sqrt_diag;
        // Update the lower-triangular part below the diagonal in column j.
        for i in (j + 1)..dim {
            let mut s = a[i * dim + j];
            if j > 0 {
                let l_row_j = &a[j * dim..j * dim + j];
                let l_row_i = &a[i * dim..i * dim + j];
                for k in 0..j {
                    s -= l_row_i[k] * l_row_j[k];
                }
            }
            a[i * dim + j] = s * inv_diag;
        }
    }
    true
}

/// Solve `L · Lᵀ · x = b` for `x`, given the Cholesky factor `L` (lower
/// triangular, row-major `dim × dim`). Writes `x` into the provided buffer;
/// uses `y_buf` (length `dim`) as scratch for the intermediate forward solve.
///
/// Forward substitution: `L · y = b`. Back substitution: `Lᵀ · x = y`.
#[inline]
pub(crate) fn cholesky_solve_into(
    l: &[f32],
    b: &[f32],
    dim: usize,
    y_buf: &mut [f32],
    x: &mut [f32],
) {
    debug_assert_eq!(l.len(), dim * dim);
    debug_assert_eq!(b.len(), dim);
    debug_assert_eq!(y_buf.len(), dim);
    debug_assert_eq!(x.len(), dim);

    // Forward: y[i] = (b[i] − Σ_{j<i} L[i,j]·y[j]) / L[i,i]
    for i in 0..dim {
        let mut s = b[i];
        if i > 0 {
            let l_row = &l[i * dim..i * dim + i];
            for j in 0..i {
                s -= l_row[j] * y_buf[j];
            }
        }
        y_buf[i] = s / l[i * dim + i];
    }
    // Back: x[i] = (y[i] − Σ_{j>i} L[j,i]·x[j]) / L[i,i]   (Lᵀ row i = L col i)
    for i in (0..dim).rev() {
        let mut s = y_buf[i];
        if i + 1 < dim {
            for j in (i + 1)..dim {
                s -= l[j * dim + i] * x[j];
            }
        }
        x[i] = s / l[i * dim + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `cholesky_inplace` + `cholesky_solve_into` reconstructs the
    /// solution to a known SPD system.
    #[test]
    fn cholesky_solves_known_spd_system() {
        // A = [[4, 2], [2, 3]]  (SPD, eigenvalues 1.382, 5.618)
        // Row-major: [a00, a01, a10, a11] = [4, 2, 2, 3].
        // Cholesky only reads the lower triangle; upper entries are ignored.
        let mut a = [4.0_f32, 2.0, 2.0, 3.0];
        // Solve A x = b with b = [6, 7].
        // Analytical: x = [0.5, 2.0]  (verify: A·x = [4·0.5+2·2, 2·0.5+3·2] = [6, 7] ✓)
        let b = [6.0_f32, 7.0];
        let mut y = [0.0_f32; 2];
        let mut x = [0.0_f32; 2];
        assert!(cholesky_inplace(&mut a, 2));
        cholesky_solve_into(&a, &b, 2, &mut y, &mut x);
        assert!((x[0] - 0.5).abs() < 1e-5, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-5, "x[1] = {}", x[1]);
    }

    /// A non-PD matrix should fail Cholesky.
    #[test]
    fn cholesky_rejects_indefinite_matrix() {
        // A = [[1, 2], [2, 1]]  (eigenvalues -1, 3 → indefinite)
        let mut a = [1.0_f32, 0.0, 2.0, 1.0];
        assert!(!cholesky_inplace(&mut a, 2));
    }
}
