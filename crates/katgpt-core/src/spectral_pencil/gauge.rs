//! `gauge` — the canonical gauge: orthogonal-invariance canonicalization
//! (Issue 676 T8; paper §4.3).
//!
//! The pencil's function is invariant under simultaneous conjugation
//! `Aᵢ ↦ QᵀAᵢQ` — the parametrization carries a free orthogonal gauge,
//! which makes raw coefficient bytes unstable (the same function has
//! infinitely many matrix representations). Commitment (BLAKE3 over
//! canonical bytes, the per-NPC genome use case) needs ONE canonical
//! representative.
//!
//! ## The canonicalization
//!
//! Diagonalize `A₀` (pinned Jacobi): `A₀ = V·Λ·Vᵀ` with eigenvalues
//! sorted ascending and **each eigenvector sign-fixed** (the entry of
//! largest absolute value is made positive; ties broken by lowest
//! index). The canonical pencil is
//!
//! ```text
//! { Λ,  Vᵀ·A₁·V,  …,  Vᵀ·Aₙ·V }
//! ```
//!
//! — same function (`f`-invariance is orthogonal invariance), one
//! representative. **Idempotent**: re-canonicalizing the canonical form
//! is a no-op (`Λ` is diagonal-sorted; its Jacobi pass exits on the
//! first residual check with `V = I`; identity columns pass the sign
//! fix untouched).
//!
//! ## Honest boundary: repeated eigenvalues in A₀
//!
//! Within a repeated-λ block the eigenvector basis is a free rotation —
//! the pinned Jacobi schedule picks ONE basis deterministically per
//! binary, so canonical bytes are **stable for the same input on the
//! same binary** (what commitment needs), but two different
//! representations of the same function can still canonicalize
//! differently across blocks. Consumers requiring cross-representation
//! equality should sort/round their eigenvalues first (lattice
//! quantization is the committed-float policy anyway).

use crate::spectral_pencil::dense::{DenseScratch, jacobi_eigen};
use crate::spectral_pencil::sym::SymPacked;

/// Sign-fix an eigenvector column in place: the entry of largest
/// absolute value becomes positive (ties → lowest index wins).
fn sign_fix_column<const D: usize>(v: &mut [f32; D]) {
    let mut best_i = 0_usize;
    let mut best_abs = 0.0_f32;
    for (i, &x) in v.iter().enumerate() {
        let a = x.abs();
        if a > best_abs {
            best_abs = a;
            best_i = i;
        }
    }
    if v[best_i] < 0.0 {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
}

/// Canonicalize a pencil (module doc). Returns the canonical
/// `{Λ, VᵀA₁V, …}` pencil; `scratch` is caller-owned.
#[must_use]
pub fn canonicalize<const D: usize, const N: usize>(
    a0: &SymPacked<D>,
    a: &[SymPacked<D>; N],
    scratch: &mut DenseScratch<D>,
) -> (SymPacked<D>, [SymPacked<D>; N]) {
    // Eigendecomposition of A0 (values ascending, columns = vectors).
    let full0 = a0.to_full();
    jacobi_eigen(&full0, true, scratch);

    // V (columns sign-fixed) and Λ.
    let mut v = [[0.0_f32; D]; D];
    for c in 0..D {
        let mut col = [0.0_f32; D];
        for (r, o) in col.iter_mut().enumerate() {
            *o = scratch.v[r][c];
        }
        sign_fix_column(&mut col);
        for (r, &cv) in v.iter_mut().zip(col.iter()) {
            r[c] = cv;
        }
    }
    let mut lam = SymPacked::<D>::zeroed();
    for (i, &val) in scratch.values.iter().enumerate() {
        lam.data[i][i] = val;
    }

    // Aᵢ' = Vᵀ Aᵢ V — three fixed loops per feature, packed on output
    // (offs must round-trip the √2 convention — build full, then pack).
    let mut out = [SymPacked::<D>::zeroed(); N];
    for (m, o) in a.iter().zip(out.iter_mut()) {
        let fm = m.to_full();
        // T = Aᵢ·V
        let mut t = [[0.0_f32; D]; D];
        for (r, trow) in t.iter_mut().enumerate() {
            for (c, tv) in trow.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (l, &fl) in fm[r].iter().enumerate() {
                    acc += f64::from(fl) * f64::from(v[l][c]);
                }
                *tv = acc as f32;
            }
        }
        // full = Vᵀ·T, then pack
        let mut fout = [[0.0_f32; D]; D];
        for (r, orow) in fout.iter_mut().enumerate() {
            for (c, ov) in orow.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (l, &vl) in v.iter().enumerate() {
                    acc += f64::from(vl[r]) * f64::from(t[l][c]);
                }
                *ov = acc as f32;
            }
        }
        *o = SymPacked::pack_from_full(&fout);
    }
    (lam, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_pencil::init::seeded_dense;

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / 2.0_f32.powi(31)) * 2.0 - 1.0
        }
    }

    /// Random orthogonal Q via pinned Householder QR of a gaussian.
    fn random_orthogonal<const D: usize>(rng: &mut Lcg) -> [[f32; D]; D] {
        // Reuse the init module's machinery by seeding through
        // seeded_dense's QR? Simpler: Gram-Schmidt of a random matrix —
        // deterministic, pinned, adequate for the invariance test.
        let mut cols = [[0.0_f32; D]; D];
        for (c, col) in cols.iter_mut().enumerate() {
            for (r, e) in col.iter_mut().enumerate() {
                let mut acc = 0.0_f32;
                for _ in 0..3 {
                    acc += rng.next_f32() + (r * 7 + c * 13) as f32 / 31.0;
                }
                *e = acc - 1.5;
            }
        }
        // Gram-Schmidt over columns.
        for c in 0..D {
            for p in 0..c {
                let mut dot = 0.0_f64;
                for row in &cols {
                    dot += f64::from(row[c]) * f64::from(row[p]);
                }
                let d = dot as f32;
                for row in &mut cols {
                    row[c] -= d * row[p];
                }
            }
            let mut n = 0.0_f64;
            for row in &cols {
                n += f64::from(row[c]) * f64::from(row[c]);
            }
            let n = (n as f32).sqrt().max(f32::MIN_POSITIVE);
            for row in &mut cols {
                row[c] /= n;
            }
        }
        cols
    }

    /// T8 G1a: f-invariance under random conjugations — conjugate the
    /// SUM `A(x)` (conjugating only A₀ while leaving Aᵢ raw is NOT the
    /// same matrix).
    #[test]
    fn f_is_invariant_under_random_conjugation() {
        const D: usize = 6;
        const N: usize = 3;
        let init = seeded_dense::<D, N>(b"gauge-invariance", 3);
        let mut scratch = DenseScratch::<D>::new();
        let mut rng = Lcg(31);
        for trial in 0..64 {
            let q = random_orthogonal::<D>(&mut rng);
            let mut x = [0.0_f32; N];
            for v in x.iter_mut() {
                *v = rng.next_f32() * 3.0;
            }
            // ax = A(x); axc = Qᵀ·ax·Q — same matrix, conjugated.
            let mut ax = init.a0;
            for (m, &xi) in init.a.iter().zip(x.iter()) {
                ax.add_scaled_into(m, xi);
            }
            let fa = ax.to_full();
            let mut t = [[0.0_f32; D]; D];
            for (r, trow) in t.iter_mut().enumerate() {
                for (c, tv) in trow.iter_mut().enumerate() {
                    let mut acc = 0.0_f64;
                    for (l, &fl) in fa[r].iter().enumerate() {
                        acc += f64::from(fl) * f64::from(q[l][c]);
                    }
                    *tv = acc as f32;
                }
            }
            let mut fc = [[0.0_f32; D]; D];
            for (r, orow) in fc.iter_mut().enumerate() {
                for (c, ov) in orow.iter_mut().enumerate() {
                    let mut acc = 0.0_f64;
                    for l in 0..D {
                        acc += f64::from(q[l][r]) * f64::from(t[l][c]);
                    }
                    *ov = acc as f32;
                }
            }
            jacobi_eigen(&fa, false, &mut scratch);
            let v1 = scratch.values;
            jacobi_eigen(&fc, false, &mut scratch);
            let v2 = scratch.values;
            for k in 0..D {
                assert!(
                    (v1[k] - v2[k]).abs() < 1e-4,
                    "trial {trial} k {k}: {} vs {}",
                    v1[k],
                    v2[k]
                );
            }
        }
    }

    /// T8 G1b: canonicalization preserves f, and is idempotent
    /// (canonical bytes stable — re-canonicalize → identical).
    #[test]
    fn canonical_form_preserves_f_and_is_idempotent() {
        const D: usize = 5;
        const N: usize = 3;
        let init = seeded_dense::<D, N>(b"gauge-canonical", 2);
        let mut scratch = DenseScratch::<D>::new();

        let (c0, ca) = canonicalize(&init.a0, &init.a, &mut scratch);

        // f-preservation on a sample of xs.
        let mut rng = Lcg(77);
        for _ in 0..64 {
            let mut x = [0.0_f32; N];
            for v in x.iter_mut() {
                *v = rng.next_f32() * 3.0;
            }
            let mut ax = init.a0;
            for (m, &xi) in init.a.iter().zip(x.iter()) {
                ax.add_scaled_into(m, xi);
            }
            let mut cx = c0;
            for (m, &xi) in ca.iter().zip(x.iter()) {
                cx.add_scaled_into(m, xi);
            }
            jacobi_eigen(&ax.to_full(), false, &mut scratch);
            let v1 = scratch.values;
            jacobi_eigen(&cx.to_full(), false, &mut scratch);
            for (k, (a, b)) in v1.iter().zip(scratch.values.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "canonical f drifted at k {k}: {a} vs {b}"
                );
            }
        }

        // Idempotency: re-canonicalize the canonical form.
        let (c0b, cab) = canonicalize(&c0, &ca, &mut scratch);
        assert_eq!(c0.data, c0b.data, "Λ not idempotent");
        for (a, b) in ca.iter().zip(cab.iter()) {
            for r in 0..D {
                for c in 0..D {
                    assert!(
                        (a.data[r][c] - b.data[r][c]).abs() < 1e-5,
                        "canonical Aᵢ not idempotent at ({r},{c})"
                    );
                }
            }
        }
    }

    /// T8 G1c: canonicalization is deterministic — same input twice →
    /// identical bytes (the commitment contract).
    #[test]
    fn canonicalization_is_bit_deterministic() {
        const D: usize = 5;
        const N: usize = 2;
        let init = seeded_dense::<D, N>(b"gauge-determinism", 3);
        let mut s1 = DenseScratch::<D>::new();
        let mut s2 = DenseScratch::<D>::new();
        let (a, b) = canonicalize(&init.a0, &init.a, &mut s1);
        let (a2, b2) = canonicalize(&init.a0, &init.a, &mut s2);
        assert_eq!(a.data, a2.data);
        for (x, y) in b.iter().zip(b2.iter()) {
            assert_eq!(x.data, y.data);
        }
    }
}
