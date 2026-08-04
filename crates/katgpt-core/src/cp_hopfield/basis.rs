//! SU(d) generalized Gell-Mann basis + structure constants.
//!
//! # Why this is computed, not tabulated
//!
//! Plan 567 T1.3 called for hardcoded `f_abc` lookup tables per `d`. That is
//! tractable at `d = 2` (Levi-Civita `ε_abc`) and `d = 3` (the 9 canonical
//! Gell-Mann values) but not beyond: at `d = 8` the Bloch dimension is
//! `D2 = 63`, so a dense table is `63³ = 250_047` entries — ~1 MB of mostly
//! zeros, transcribed by hand. Instead the generators are built in **closed
//! form** and the structure constants are contracted out of them:
//!
//! ```text
//! t_abc = Tr(λ_a λ_b λ_c)      f_abc = Im(t_abc)/2      d_abc = Re(t_abc)/2
//! ```
//!
//! This is still `O(1)`-per-recall (construction happens once, at
//! [`GellMannBasis::new`]) and still fully deterministic and modelless — closed-form
//! matrix construction plus exact contraction, no fitting. It additionally
//! *derives* the canonical Pauli and Gell-Mann values rather than trusting a
//! transcription, which the unit tests check against the literature.
//!
//! The stored constants are **sparse**: `f_abc` is totally antisymmetric and
//! `d_abc` totally symmetric, so both are vanishingly sparse for `d ≥ 4`. Sparse
//! storage also makes the Lie bracket `O(nnz)` instead of `O(D2²)`.
//!
//! # Generator convention
//!
//! `Tr(λ_a λ_b) = 2 δ_ab`. Ordering is grouped by column `k = 1..d-1`: for each
//! `k`, the symmetric and antisymmetric off-diagonals `(j, k)` for `j < k`, then
//! the `k`-th diagonal generator. This reproduces the canonical orderings exactly
//! — `(σ_x, σ_y, σ_z)` at `d = 2` and `λ_1..λ_8` at `d = 3` — so literature
//! values are directly comparable.

use super::complex::C32;

/// Magnitude below which a computed structure constant is treated as exactly zero.
///
/// The generators have entries in `{0, ±1, ±i}` and `√(2/(l(l+1)))`, so genuine
/// non-zero constants are `≥ ~1/d` in magnitude; anything near `f32` round-off is
/// a cancellation artifact. `1e-5` sits comfortably between the two.
const SPARSITY_TOL: f32 = 1e-5;

/// One non-zero entry of one SU(d) generator.
#[derive(Clone, Copy, Debug)]
struct BasisEntry {
    row: u16,
    col: u16,
    val: C32,
}

/// One non-zero structure constant, stored as `(i, j, k) -> val`.
///
/// Both `f_abc` and `d_abc` are contracted as `out[i] += val · x[j] · y[k]`, which
/// works for either tensor because `f` is cyclic-invariant (totally antisymmetric)
/// and `d` is totally symmetric, so the index that comes out front is
/// interchangeable with the ones being summed over.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Triple {
    pub(crate) i: u16,
    pub(crate) j: u16,
    pub(crate) k: u16,
    pub(crate) val: f32,
}

/// The `d²−1` generators of SU(d) in the generalized Gell-Mann basis.
///
/// Stored sparsely: each generator has at most `d` non-zero entries, so the
/// Bloch projection `s_a = ⟨ξ|λ_a|ξ⟩` costs `O(d²)` total rather than `O(d⁴)`.
#[derive(Clone, Debug)]
pub struct GellMannBasis<const D: usize> {
    entries: Vec<BasisEntry>,
    /// `entries[offsets[a] .. offsets[a+1]]` are generator `a`'s non-zeros.
    offsets: Vec<u32>,
}

impl<const D: usize> GellMannBasis<D> {
    /// Real Bloch dimension `d² − 1`.
    pub const BLOCH_DIM: usize = D * D - 1;

    /// Build the basis in closed form.
    ///
    /// # Panics
    /// Panics if `D < 2` (SU(1) is trivial — there is no manifold to recall on).
    pub fn new() -> Self {
        assert!(D >= 2, "cp_hopfield: CP^(d-1) requires d >= 2, got d = {D}");

        let mut entries: Vec<BasisEntry> = Vec::with_capacity(Self::BLOCH_DIM * D);
        let mut offsets: Vec<u32> = Vec::with_capacity(Self::BLOCH_DIM + 1);
        offsets.push(0);

        let push = |es: &mut Vec<BasisEntry>, offs: &mut Vec<u32>, new: &[BasisEntry]| {
            es.extend_from_slice(new);
            offs.push(es.len() as u32);
        };

        for k in 1..D {
            for j in 0..k {
                // Symmetric: E_jk + E_kj.
                push(
                    &mut entries,
                    &mut offsets,
                    &[
                        BasisEntry {
                            row: j as u16,
                            col: k as u16,
                            val: C32::ONE,
                        },
                        BasisEntry {
                            row: k as u16,
                            col: j as u16,
                            val: C32::ONE,
                        },
                    ],
                );
                // Antisymmetric: −i(E_jk − E_kj).
                push(
                    &mut entries,
                    &mut offsets,
                    &[
                        BasisEntry {
                            row: j as u16,
                            col: k as u16,
                            val: C32::new(0.0, -1.0),
                        },
                        BasisEntry {
                            row: k as u16,
                            col: j as u16,
                            val: C32::new(0.0, 1.0),
                        },
                    ],
                );
            }
            // Diagonal: √(2/(k(k+1))) · (Σ_{j<k} E_jj − k·E_kk).
            let scale = (2.0 / (k as f32 * (k as f32 + 1.0))).sqrt();
            let mut diag: Vec<BasisEntry> = (0..k)
                .map(|j| BasisEntry {
                    row: j as u16,
                    col: j as u16,
                    val: C32::real(scale),
                })
                .collect();
            diag.push(BasisEntry {
                row: k as u16,
                col: k as u16,
                val: C32::real(-scale * k as f32),
            });
            push(&mut entries, &mut offsets, &diag);
        }

        debug_assert_eq!(offsets.len(), Self::BLOCH_DIM + 1);
        Self { entries, offsets }
    }

    /// Materialize generator `a` as a dense `d×d` Hermitian matrix.
    pub fn generator_dense(&self, a: usize) -> [[C32; D]; D] {
        let mut m = [[C32::ZERO; D]; D];
        for e in self.entries_of(a) {
            m[e.row as usize][e.col as usize] = e.val;
        }
        m
    }

    #[inline]
    fn entries_of(&self, a: usize) -> &[BasisEntry] {
        let lo = self.offsets[a] as usize;
        let hi = self.offsets[a + 1] as usize;
        &self.entries[lo..hi]
    }

    /// Project a qudit onto its generalized Bloch vector: `s_a = ⟨ξ|λ_a|ξ⟩`.
    ///
    /// Phase-invariant by construction (`|ξ⟩` and `e^{iθ}|ξ⟩` give the same `s`),
    /// which is exactly what makes `s` a coordinate on `CP^(d-1)` rather than on
    /// the sphere `S^(2d-1)`.
    ///
    /// The result satisfies `|s|² = 2(1 − 1/d)` when `|ξ⟩` is normalized.
    ///
    /// # Panics
    /// Panics if `out.len() != d² − 1`.
    pub fn bloch_projection_into(&self, state: &[C32; D], out: &mut [f32]) {
        assert_eq!(
            out.len(),
            Self::BLOCH_DIM,
            "cp_hopfield: bloch output must be d²−1 = {}",
            Self::BLOCH_DIM
        );
        for (a, o) in out.iter_mut().enumerate() {
            let mut acc = C32::ZERO;
            for e in self.entries_of(a) {
                let braket = state[e.row as usize].conj().mul(state[e.col as usize]);
                acc = acc.mul_add(braket, e.val);
            }
            // λ_a is Hermitian, so the expectation value is real up to round-off.
            *o = acc.re;
        }
    }

    /// Reconstruct the density matrix `ρ = (1/d)I + (1/2) Σ_a s_a λ_a`.
    ///
    /// For an on-manifold `s` this `ρ` is a rank-1 projector `|ξ⟩⟨ξ|`. For an
    /// off-manifold `s` it is Hermitian but not idempotent, which is what
    /// [`super::recaller::CpHopfieldRecaller::project_to_manifold`] exploits: the
    /// closest pure state is `ρ`'s top eigenvector.
    ///
    /// # Panics
    /// Panics if `bloch.len() != d² − 1`.
    pub fn density_from_bloch(&self, bloch: &[f32]) -> [[C32; D]; D] {
        assert_eq!(
            bloch.len(),
            Self::BLOCH_DIM,
            "cp_hopfield: bloch input must be d²−1 = {}",
            Self::BLOCH_DIM
        );
        let mut rho = [[C32::ZERO; D]; D];
        let inv_d = 1.0 / D as f32;
        for i in 0..D {
            rho[i][i] = C32::real(inv_d);
        }
        for (a, &s_a) in bloch.iter().enumerate() {
            if s_a == 0.0 {
                continue;
            }
            let w = 0.5 * s_a;
            for e in self.entries_of(a) {
                let r = e.row as usize;
                let c = e.col as usize;
                rho[r][c] = rho[r][c].add(e.val.scale(w));
            }
        }
        rho
    }
}

impl<const D: usize> Default for GellMannBasis<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Sparse SU(d) structure constants.
///
/// - `f_abc` from `[λ_a, λ_b] = 2i f_abc λ_c` — totally antisymmetric. Drives the
///   Lie-bracket product `[s ×_f B]_c = f_cab s_a B_b` in the LLG flow.
/// - `d_abc` from `{λ_a, λ_b} = (4/d) δ_ab I + 2 d_abc λ_c` — totally symmetric.
///   Drives the non-linear `CP^(d-1)` constraint `d_abc s_a s_b = (2 − 4/d) s_c`.
#[derive(Clone, Debug)]
pub struct StructureConstants {
    d: usize,
    f: Vec<Triple>,
    d_sym: Vec<Triple>,
}

impl StructureConstants {
    /// Contract the structure constants out of a basis.
    ///
    /// Cost is `O(D2² · d³)` — about 4M `f32` operations at `d = 8`, a few
    /// milliseconds, paid once. Not on any hot path.
    pub fn new<const D: usize>(basis: &GellMannBasis<D>) -> Self {
        let n = GellMannBasis::<D>::BLOCH_DIM;
        let mut f = Vec::new();
        let mut d_sym = Vec::new();

        let dense: Vec<[[C32; D]; D]> = (0..n).map(|a| basis.generator_dense(a)).collect();

        for a in 0..n {
            for b in 0..n {
                // M = λ_a λ_b
                let mut m = [[C32::ZERO; D]; D];
                for r in 0..D {
                    for t in 0..D {
                        let art = dense[a][r][t];
                        if art == C32::ZERO {
                            continue;
                        }
                        for c in 0..D {
                            m[r][c] = m[r][c].mul_add(art, dense[b][t][c]);
                        }
                    }
                }
                for c in 0..n {
                    // t_abc = Tr(M λ_c). Since λ's are Hermitian,
                    // Tr([λ_a,λ_b]λ_c) = 2i·Im(t) and Tr({λ_a,λ_b}λ_c) = 2·Re(t),
                    // giving f = Im(t)/2 and d = Re(t)/2.
                    let mut t = C32::ZERO;
                    for e in basis.entries_of(c) {
                        t = t.mul_add(m[e.col as usize][e.row as usize], e.val);
                    }
                    let f_val = 0.5 * t.im;
                    let d_val = 0.5 * t.re;
                    if f_val.abs() > SPARSITY_TOL {
                        f.push(Triple {
                            i: a as u16,
                            j: b as u16,
                            k: c as u16,
                            val: f_val,
                        });
                    }
                    if d_val.abs() > SPARSITY_TOL {
                        d_sym.push(Triple {
                            i: a as u16,
                            j: b as u16,
                            k: c as u16,
                            val: d_val,
                        });
                    }
                }
            }
        }

        Self { d: D, f, d_sym }
    }

    /// Complex dimension `d`.
    #[inline]
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Number of stored non-zero `f_abc` entries (all index permutations).
    #[inline]
    pub fn f_nnz(&self) -> usize {
        self.f.len()
    }

    /// Number of stored non-zero `d_abc` entries (all index permutations).
    #[inline]
    pub fn d_nnz(&self) -> usize {
        self.d_sym.len()
    }

    /// Look up a single `f_abc`. Linear scan — diagnostics and tests only.
    pub fn f(&self, a: usize, b: usize, c: usize) -> f32 {
        Self::lookup(&self.f, a, b, c)
    }

    /// Look up a single `d_abc`. Linear scan — diagnostics and tests only.
    pub fn d_sym(&self, a: usize, b: usize, c: usize) -> f32 {
        Self::lookup(&self.d_sym, a, b, c)
    }

    fn lookup(ts: &[Triple], a: usize, b: usize, c: usize) -> f32 {
        ts.iter()
            .find(|t| t.i as usize == a && t.j as usize == b && t.k as usize == c)
            .map(|t| t.val)
            .unwrap_or(0.0)
    }

    #[inline]
    pub(crate) fn f_triples(&self) -> &[Triple] {
        &self.f
    }

    #[inline]
    pub(crate) fn d_triples(&self) -> &[Triple] {
        &self.d_sym
    }
}
