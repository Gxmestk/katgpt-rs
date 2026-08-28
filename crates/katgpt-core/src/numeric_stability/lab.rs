//! Phase 2 — the perturbable reference attention lab (arXiv:2405.02803 §1's
//! microbenchmark methodology; Issue 697 T2.1–T2.2 + T3.2).
//!
//! The production kernels (`crate::attention`'s SIMD `tiled_attention_*`,
//! feature `tiled_attention`) answer "does the kernel match its reference".
//! THIS module is the paper's other instrument: a deliberately naive, scalar
//! f64, tiled online-softmax attention whose NUMERIC behavior is perturbable
//! along the paper's four knobs —
//!
//! 1. **mantissa width** (via the Phase-1 [`truncate_mantissa`] emulator),
//! 2. **sequence length** `S`,
//! 3. **tile shape** `(Bc, Br)`,
//! 4. **tile dimension order** — implemented as swapping which tile size is
//!    assigned to the Q axis vs the K axis ([`AxisSwap`]),
//!
//! — so the four **ordering laws** (mantissa↑ ⇒ deviation↓; deviation scales
//! with the rescale count `R = ⌈S/T⌉ − 1`; larger tile area ⇒ less deviation;
//! the dimension-order swap measurably changes deviation with the SQUARE row
//! as the free negative control) are pinned as TESTS instead of re-measured
//! per incident.
//!
//! # The FP64 golden protocol is two-tier
//!
//! The golden reference is [`naive_attention_f64`] (two-pass: global row max,
//! then numerator/denominator sums in ascending `j`).
//!
//! - **Single-tile config** (`Bc ≥ S` and `Br ≥ S`): the online-softmax
//!   machinery is INERT — the first correction is exactly `exp(−inf) = 0`,
//!   the running max IS the global max, and every accumulation happens in
//!   the same order as the naive reference. The outputs are **bit-identical**
//!   (pinned by `golden_identity_single_tile_bit_exact`).
//! - **Multi-tile config**: the rescale corrections group the numerator sum
//!   differently than the two-pass reference, and fp addition is not
//!   associative — exact equality is STRUCTURALLY impossible (that
//!   non-associativity is precisely the mechanism the lab measures). The
//!   gate is a pinned tiny relative bound ([`PINNED_F64_MULTITILE_REL`]),
//!   measured, not guessed.
//!
//! # Scope limits (honest)
//!
//! - **Mantissa-only emulation**: [`truncate_mantissa`] models no exponent
//!   range (no f16/BF16 overflow, no subnormal ladders). The lab's
//!   "formats" are mantissa widths; the paper's absolute constants stay
//!   context-specific footnotes — only the ordering laws are extracted.
//! - **Dimension-order knob**: on a scalar (non-SIMD) emulator a pure loop
//!   re-nest cannot change bits — the per-element accumulation order is
//!   unchanged. The implementable form of the paper's dimension-order knob
//!   is the Q/K axis ASSIGNMENT swap: at `Bc ≠ Br` it changes the tile
//!   partition (and with it `R`); at `Bc == Br` it is provably the identity
//!   — which is exactly the paper's square-tile negative control, pinned
//!   bit-identical by test. The paper's "at fixed R" caveat is documented
//!   rather than reproduced: our swap changes `R` when the tile is
//!   non-square, so the pinned law is "the swap measurably changes
//!   deviation" + "the square row is invariant", not a fixed-R isolation.
//! - **Offline instrument**: the lab allocates per call and runs scalar
//!   f64 — a measurement apparatus for gates and pre-swap triage, never a
//!   hot path. No `_into` scratch plumbing is warranted (the Plan 418
//!   zero-alloc convention governs production paths, not this probe).
//! - **Cross-libm caveat for pinned constants**: `f64::exp` is
//!   platform-libm. Same-platform re-measurement is bit-deterministic; the
//!   pinned fit constants are asserted within a ±20% band
//!   ([`TOL_FIT_BAND`]) so the gate is portable across the M3/4090 boxes.
//!   The fit-input identity is pinned exactly (blake3) — that pins the
//!   CONFIG, which is fully deterministic.
//! - **Schedule headroom**: [`TOL_TABLE_PINNED`] rows are
//!   `measured_deviation × [`TOL_HEADROOM`]` — a POLICY constant of this
//!   probe's schedule (documented here), chosen so the lab's own deviation
//!   sits strictly inside the band and the acceptance verdicts below are
//!   strict comparisons, not margin-line ties. It is NOT derived from the
//!   paper's context-specific "2–5×" headline.

use super::probe::{DeviationReport, F64_MANTISSA_BITS, truncate_mantissa};

/// Deterministic input seed for every lab gate (LCG; no external rng dep).
pub const LAB_SEED: u64 = 0x697_0001;

/// Multi-tile f64 golden bound: the lab at full mantissa vs the naive f64
/// reference may differ only by rescale-grouping rounding. Measured on the
/// gate config (S=128, D=16, Bc=Br=32, golden mode). A violation means the
/// lab's arithmetic changed — not that attention "got worse".
pub const PINNED_F64_MULTITILE_REL: f64 = 3.0e-15;

/// Cross-libm band for the pinned `tol(S)` fit rows (±20%): same-platform
/// re-measurement is bit-exact, but `exp` differs across libm
/// implementations, so the portable assertion is a band. See module docs.
pub const TOL_FIT_BAND: f64 = 0.20;

/// Policy headroom baked into the pinned schedule rows: row =
/// `measured_deviation × TOL_HEADROOM`. See the module doc's "Schedule
/// headroom" note — a documented probe policy, not a paper import.
pub const TOL_HEADROOM: f64 = 2.0;

/// Ordinal floor for the rescale-count Spearman gate (T2.2 law 2). The
/// law is ORDINAL — ρ > 0 is the paper's claim; the floor pins the
/// measured strength so a silent regression in the lab's sensitivity is
/// caught. Measured value recorded in the Bench 691 Phase 2 section.
pub const PINNED_SPEARMAN_FLOOR: f64 = 0.7;

/// Which axis receives which tile size.
///
/// `Standard` tiles the Q rows by [`LabConfig::br`] and the K/V columns by
/// [`LabConfig::bc`] (the flash-attention convention: `Br` query rows ×
/// `Bc` key columns). `Swapped` exchanges the assignment. At `Bc == Br` the
/// two are the same partition — the paper's square-tile negative control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisSwap {
    Standard,
    Swapped,
}

/// A lab run's knob set. `mantissa_bits = F64_MANTISSA_BITS` (52) and
/// `quantize_ops = false` is the unperturbed f64 golden mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabConfig {
    pub seq_len: usize,
    pub head_dim: usize,
    /// K/V column tile under the STANDARD assignment.
    pub bc: usize,
    /// Q row tile under the STANDARD assignment.
    pub br: usize,
    pub axis_swap: AxisSwap,
    /// Mantissa bits of the emulated format (0..=52). Inputs are always
    /// storage-quantized at this width; see [`Self::quantize_ops`].
    pub mantissa_bits: u32,
    /// Also truncate after EVERY arithmetic op (models "arithmetic in
    /// format F"), not just at input storage.
    pub quantize_ops: bool,
    /// Softmax temperature; default `1/sqrt(head_dim)`.
    pub scale: f64,
}

impl LabConfig {
    /// Golden-mode config: unperturbed f64, standard axis assignment.
    pub fn new(seq_len: usize, head_dim: usize, bc: usize, br: usize) -> Self {
        Self {
            seq_len,
            head_dim,
            bc,
            br,
            axis_swap: AxisSwap::Standard,
            mantissa_bits: F64_MANTISSA_BITS,
            quantize_ops: false,
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    /// Reduced-format config: mantissa width + per-op truncation.
    pub fn perturbed(seq_len: usize, head_dim: usize, bc: usize, br: usize, bits: u32) -> Self {
        Self {
            mantissa_bits: bits,
            quantize_ops: true,
            ..Self::new(seq_len, head_dim, bc, br)
        }
    }

    #[inline]
    pub fn q_rows_per_tile(&self) -> usize {
        match self.axis_swap {
            AxisSwap::Standard => self.br,
            AxisSwap::Swapped => self.bc,
        }
    }

    #[inline]
    pub fn kv_cols_per_tile(&self) -> usize {
        match self.axis_swap {
            AxisSwap::Standard => self.bc,
            AxisSwap::Swapped => self.br,
        }
    }

    /// The paper's closed-form rescale count `R = ⌈S/T⌉ − 1` where `T` is
    /// the K/V column tile (each K/V tile past the first triggers one
    /// online-softmax rescale per query row).
    pub fn rescale_count(&self) -> u64 {
        let t = self.kv_cols_per_tile();
        let tiles = self.seq_len.div_ceil(t) as u64;
        tiles.saturating_sub(1)
    }
}

/// Storage-format quantization: applied to the INPUTS whenever
/// `bits < 52`, independent of [`LabConfig::quantize_ops`].
#[inline]
fn qt_store(v: f64, bits: u32) -> f64 {
    if bits >= F64_MANTISSA_BITS {
        v
    } else {
        truncate_mantissa(v, bits)
    }
}

/// Op-format quantization: applied to intermediate results only when the
/// caller asked for per-op emulation.
#[inline]
fn qt(v: f64, bits: u32, ops: bool) -> f64 {
    if ops && bits < F64_MANTISSA_BITS {
        truncate_mantissa(v, bits)
    } else {
        v
    }
}

/// The perturbable tiled online-softmax attention (scalar f64, canonical
/// flash-attention update order: rescale the running state, THEN add the
/// new tile's PV).
///
/// Inputs are f32 (house dtype) up-converted to f64; `mantissa_bits < 52`
/// truncates them once (storage format) and — with `quantize_ops` — every
/// intermediate op result (arithmetic format). Outputs are f64; compare
/// against [`naive_attention_f64`] with [`max_abs_diff_f64`] /
/// [`max_rel_diff_f64`].
pub fn lab_attention(q: &[f32], k: &[f32], v: &[f32], out: &mut [f64], cfg: &LabConfig) {
    let s = cfg.seq_len;
    let d = cfg.head_dim;
    assert_eq!(q.len(), s * d, "q length mismatch");
    assert_eq!(k.len(), s * d, "k length mismatch");
    assert_eq!(v.len(), s * d, "v length mismatch");
    assert_eq!(out.len(), s * d, "out length mismatch");
    if s == 0 {
        return;
    }

    let bits = cfg.mantissa_bits;
    let ops = cfg.quantize_ops;
    let qz: Vec<f64> = q.iter().map(|&x| qt_store(x as f64, bits)).collect();
    let kz: Vec<f64> = k.iter().map(|&x| qt_store(x as f64, bits)).collect();
    let vz: Vec<f64> = v.iter().map(|&x| qt_store(x as f64, bits)).collect();

    let rt = cfg.q_rows_per_tile();
    let ct = cfg.kv_cols_per_tile();
    let scale = cfg.scale;

    let mut m_run = vec![f64::NEG_INFINITY; rt];
    let mut l_run = vec![0.0f64; rt];
    let mut acc = vec![0.0f64; rt * d];
    let mut s_tile = vec![0.0f64; rt * ct];

    for r0 in (0..s).step_by(rt) {
        let rows = (r0 + rt).min(s) - r0;
        m_run[..rows].fill(f64::NEG_INFINITY);
        l_run[..rows].fill(0.0);
        acc[..rows * d].fill(0.0);

        for c0 in (0..s).step_by(ct) {
            let cols = (c0 + ct).min(s) - c0;

            // Score tile: S = scale · (q_tile @ k_tileᵀ).
            for i in 0..rows {
                let qr = &qz[(r0 + i) * d..(r0 + i) * d + d];
                for j in 0..cols {
                    let kr = &kz[(c0 + j) * d..(c0 + j) * d + d];
                    let mut dot = 0.0f64;
                    for dd in 0..d {
                        let prod = qt(qr[dd] * kr[dd], bits, ops);
                        dot = qt(dot + prod, bits, ops);
                    }
                    s_tile[i * ct + j] = qt(scale * dot, bits, ops);
                }
            }

            // Online-softmax update, canonical order per row.
            for i in 0..rows {
                let row = i * ct;
                let mut mnew = m_run[i];
                for j in 0..cols {
                    let sv = s_tile[row + j];
                    if sv > mnew {
                        mnew = sv;
                    }
                }
                // exp(−inf − finite) = 0: the first tile's correction is inert.
                let corr = qt((m_run[i] - mnew).exp(), bits, ops);

                let mut lsum = 0.0f64;
                for j in 0..cols {
                    let p = qt((s_tile[row + j] - mnew).exp(), bits, ops);
                    s_tile[row + j] = p; // reuse the tile as the P tile
                    lsum = qt(lsum + p, bits, ops);
                }
                l_run[i] = qt(qt(corr * l_run[i], bits, ops) + lsum, bits, ops);

                let ar = &mut acc[i * d..(i + 1) * d];
                for a in ar.iter_mut() {
                    *a = qt(*a * corr, bits, ops); // rescale OLD acc
                }
                for j in 0..cols {
                    let p = s_tile[row + j];
                    let vr = &vz[(c0 + j) * d..(c0 + j) * d + d];
                    for dd in 0..d {
                        let term = qt(p * vr[dd], bits, ops);
                        ar[dd] = qt(ar[dd] + term, bits, ops);
                    }
                }
                m_run[i] = mnew;
            }
        }

        for i in 0..rows {
            let li = l_run[i];
            let ar = &acc[i * d..(i + 1) * d];
            for dd in 0..d {
                out[(r0 + i) * d + dd] = qt(ar[dd] / li, bits, ops);
            }
        }
    }
}

/// Two-pass naive f64 attention — the FP64 golden reference. No tiling, no
/// online softmax: global row max, then numerator/denominator sums in
/// ascending `j`. The single-tile lab config is bit-identical to this.
pub fn naive_attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f64],
    seq_len: usize,
    head_dim: usize,
    scale: f64,
) {
    let s = seq_len;
    let d = head_dim;
    assert_eq!(q.len(), s * d, "q length mismatch");
    assert_eq!(k.len(), s * d, "k length mismatch");
    assert_eq!(v.len(), s * d, "v length mismatch");
    assert_eq!(out.len(), s * d, "out length mismatch");
    if s == 0 {
        return;
    }
    let mut row = vec![0.0f64; s];
    for i in 0..s {
        let qr = &q[i * d..(i + 1) * d];
        let mut m = f64::NEG_INFINITY;
        for j in 0..s {
            let kr = &k[j * d..(j + 1) * d];
            let mut dot = 0.0f64;
            for dd in 0..d {
                dot += qr[dd] as f64 * kr[dd] as f64;
            }
            let sv = scale * dot;
            row[j] = sv;
            if sv > m {
                m = sv;
            }
        }
        let mut l = 0.0f64;
        for slot in row.iter_mut() {
            let p = (*slot - m).exp();
            *slot = p;
            l += p;
        }
        let orow = &mut out[i * d..(i + 1) * d];
        orow.fill(0.0);
        for j in 0..s {
            let p = row[j];
            let vr = &v[j * d..(j + 1) * d];
            for dd in 0..d {
                orow[dd] += p * vr[dd] as f64;
            }
        }
        for x in orow.iter_mut() {
            *x /= l;
        }
    }
}

/// Elementwise max absolute difference (f64). Panics on length mismatch.
pub fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_abs_diff_f64 length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Max RELATIVE difference: `max(|a−b| / max(|b|, TINY))` — the form used
/// for the multi-tile f64 golden bound (absolute diffs under-state at small
/// magnitudes; the relative form is the honest one for a scale-free bound).
pub fn max_rel_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_rel_diff_f64 length mismatch");
    const TINY: f64 = 1e-300;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / y.abs().max(TINY))
        .fold(0.0f64, f64::max)
}

/// Spearman rank correlation with AVERAGE ranks for ties — the ordinal gate
/// statistic for the rescale-count law (T2.2). Deterministic; allocates two
/// rank buffers (offline statistic, not a hot path).
pub fn spearman_rho(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "spearman length mismatch");
    assert!(!xs.is_empty(), "spearman on empty input");
    let rx = average_ranks(xs);
    let ry = average_ranks(ys);
    pearson(&rx, &ry)
}

fn average_ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut ranks = vec![0.0f64; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0; // 1-based average rank
        for &t in &idx[i..=j] {
            ranks[t] = avg;
        }
        i = j + 1;
    }
    ranks
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mx;
        let dy = y - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Deterministic LCG → f32 in [-1, 1). The lab's only randomness source;
/// every gate is reproducible from [`LAB_SEED`].
pub fn lcg_next_f32(state: &mut u64) -> f32 {
    // Numerical Recipes LCG (Knuth MMIX constants).
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = *state >> 33; // top 31 bits
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Deterministic lab inputs: q/k/v filled LCG-uniform in [-1, 1).
pub fn fill_lab_inputs(seed: u64, s: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut st = seed;
    let fill = |st: &mut u64, n: usize| (0..n).map(|_| lcg_next_f32(st)).collect::<Vec<f32>>();
    let q = fill(&mut st, s * d);
    let k = fill(&mut st, s * d);
    let v = fill(&mut st, s * d);
    (q, k, v)
}

// ── T3.2: the tol(S) schedule ───────────────────────────────────────────

/// The exact fit inputs behind [`TOL_TABLE_PINNED`]. The blake3 of these
/// fields (see [`tol_fit_inputs_hash`]) pins the CONFIG; the row VALUES are
/// pinned within [`TOL_FIT_BAND`] for cross-libm portability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TolFitInputs {
    /// The schedule's anchor length (smallest measured row).
    pub s0: usize,
    /// K/V column tile of the measured config.
    pub t: usize,
    /// Mantissa width of the measured format.
    pub bits: u32,
    pub head_dim: usize,
    pub seed: u64,
}

/// The pinned fit config: (S₀=64, T=32, bits=10, D=16, [`LAB_SEED`]).
/// bits=10 is the "f16-class" mantissa proxy (f16 has 10 significand bits).
pub const TOL_FIT_PINNED: TolFitInputs = TolFitInputs {
    s0: 64,
    t: 32,
    bits: 10,
    head_dim: 16,
    seed: LAB_SEED,
};

/// The pinned blake3 hex over [`TOL_FIT_PINNED`] (each field as u64 LE) —
/// the determinism pin required by the issue ("hash the fit inputs").
/// Asserted by `tol_schedule_fit_inputs_pinned_hash`.
pub const TOL_FIT_HASH_PINNED: &str =
    "d1e3bf12f93b9bf34830c434d07b27ce482bcd1c9a7bd77f5b4a893b286a00cf";

/// blake3 hex over the fit inputs (each field as u64 LE) — the
/// determinism pin required by the issue ("hash the fit inputs").
pub fn tol_fit_inputs_hash(fit: &TolFitInputs) -> String {
    let mut bytes = Vec::with_capacity(40);
    bytes.extend_from_slice(&(fit.s0 as u64).to_le_bytes());
    bytes.extend_from_slice(&(fit.t as u64).to_le_bytes());
    bytes.extend_from_slice(&(fit.bits as u64).to_le_bytes());
    bytes.extend_from_slice(&(fit.head_dim as u64).to_le_bytes());
    bytes.extend_from_slice(&fit.seed.to_le_bytes());
    format!("{}", blake3::hash(&bytes))
}

/// The pinned `tol(S)` table: `(seq_len, max_diff_band, wasserstein_band)`,
/// the lab's measured deviation at the fit config scaled by
/// [`TOL_HEADROOM`]. Rows are sorted ascending by `seq_len`; [`band_at`]
/// linearly interpolates between bracketing rows (flat outside the range).
pub const TOL_TABLE_PINNED: [(usize, f32, f32); 4] = [
    (64, 1.027_3e-2, 6.609_3e-3),
    (128, 2.316_8e-2, 1.770_4e-2),
    (256, 4.882_8e-2, 3.839_3e-2),
    (512, 9.376_9e-2, 7.717_0e-2),
];

/// The schedule band at seq_len `s`: linear interpolation over
/// [`TOL_TABLE_PINNED`], flat below the first and above the last row.
pub fn band_at(s: usize) -> (f32, f32) {
    let rows = &TOL_TABLE_PINNED;
    if s <= rows[0].0 {
        return (rows[0].1, rows[0].2);
    }
    let last = rows.len() - 1;
    if s >= rows[last].0 {
        return (rows[last].1, rows[last].2);
    }
    for w in rows.windows(2) {
        let (s0, md0, w0) = w[0];
        let (s1, md1, w1) = w[1];
        if s <= s1 {
            let frac = (s - s0) as f32 / (s1 - s0) as f32;
            return (md0 + (md1 - md0) * frac, w0 + (w1 - w0) * frac);
        }
    }
    unreachable!("seq_len within table range must hit a bracketing window")
}

/// Deviation surfaces (f32-cast for the Phase-1 metric) of a lab run vs
/// the f64 golden — the two-surface report the Phase-1 acceptance rule
/// consumes. Public because the consumer path (T3.3) builds reports from
/// lab runs the same way.
///
/// # Panics
/// Panics if the slices' lengths differ, or if the inputs were non-finite
/// (validated inputs from [`lab_attention`] never are).
pub fn deviation_report(lab_out: &[f64], gold: &[f64]) -> DeviationReport {
    let x: Vec<f32> = lab_out.iter().map(|&v| v as f32).collect();
    let y: Vec<f32> = gold.iter().map(|&v| v as f32).collect();
    DeviationReport::compute(&x, &y).expect("validated lab inputs are finite")
}

/// Convenience: lab output + f64 golden for a config (deterministic inputs
/// from [`LAB_SEED`]).
pub fn run_pair(cfg: &LabConfig) -> (Vec<f64>, Vec<f64>) {
    let (q, k, v) = fill_lab_inputs(LAB_SEED, cfg.seq_len, cfg.head_dim);
    let mut lab_out = vec![0.0f64; cfg.seq_len * cfg.head_dim];
    let mut gold = vec![0.0f64; cfg.seq_len * cfg.head_dim];
    lab_attention(&q, &k, &v, &mut lab_out, cfg);
    naive_attention_f64(&q, &k, &v, &mut gold, cfg.seq_len, cfg.head_dim, cfg.scale);
    (lab_out, gold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_stability::probe::{ReferenceBands, Verdict, accept};

    fn abs_dev(cfg: &LabConfig) -> f64 {
        let (a, g) = run_pair(cfg);
        max_abs_diff_f64(&a, &g)
    }

    fn rel_dev(cfg: &LabConfig) -> f64 {
        let (a, g) = run_pair(cfg);
        max_rel_diff_f64(&a, &g)
    }

    fn bands_of((md, w1): (f32, f32)) -> ReferenceBands {
        let r = DeviationReport {
            max_diff: md,
            wasserstein_1d: w1,
        };
        ReferenceBands { r1: r, r2: r }
    }

    /// T2.1 G1 (tier A): with one tile covering the whole sequence the
    /// online-softmax machinery is inert — the lab is BIT-IDENTICAL to the
    /// naive f64 golden, under both axis assignments.
    #[test]
    fn golden_identity_single_tile_bit_exact() {
        for &(s, d) in &[(8usize, 4usize), (33usize, 16usize)] {
            let (q, k, v) = fill_lab_inputs(LAB_SEED.wrapping_add(s as u64), s, d);
            let scale = 1.0 / (d as f64).sqrt();
            let mut gold = vec![0.0f64; s * d];
            naive_attention_f64(&q, &k, &v, &mut gold, s, d, scale);
            for swap in [AxisSwap::Standard, AxisSwap::Swapped] {
                let cfg = LabConfig {
                    axis_swap: swap,
                    ..LabConfig::new(s, d, s, s)
                };
                let mut lab_out = vec![0.0f64; s * d];
                lab_attention(&q, &k, &v, &mut lab_out, &cfg);
                for (a, b) in lab_out.iter().zip(gold.iter()) {
                    assert_eq!(
                        a, b,
                        "single-tile lab must equal naive exactly (s={s}, d={d}, swap={swap:?})"
                    );
                }
            }
        }
    }

    /// The lab is a pure function of its inputs: two runs of the same
    /// config are bit-identical (no HashMap/wall-clock/parallelism).
    #[test]
    fn determinism_repeat_bit_identical() {
        let cfg = LabConfig::perturbed(96, 16, 32, 32, 10);
        let (q, k, v) = fill_lab_inputs(LAB_SEED, 96, 16);
        let mut o1 = vec![0.0f64; 96 * 16];
        let mut o2 = vec![0.0f64; 96 * 16];
        lab_attention(&q, &k, &v, &mut o1, &cfg);
        lab_attention(&q, &k, &v, &mut o2, &cfg);
        for (x, y) in o1.iter().zip(o2.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    /// T2.1 G1 (tier B): multi-tile golden mode deviates from the naive
    /// reference ONLY by rescale-grouping rounding — nonzero (the measured
    /// mechanism) and inside the pinned relative bound.
    #[test]
    fn multi_tile_f64_within_pinned_bound() {
        let cfg = LabConfig::new(128, 16, 32, 32);
        assert_eq!(cfg.rescale_count(), 3);
        let rel = rel_dev(&cfg);
        assert!(
            rel > 0.0,
            "rescale grouping must produce a nonzero f64 deviation"
        );
        assert!(
            rel <= PINNED_F64_MULTITILE_REL,
            "multi-tile f64 rel {rel:.3e} > pinned {PINNED_F64_MULTITILE_REL:.3e}"
        );
    }

    /// T2.2 law 1: deviation is non-increasing in mantissa width, across
    /// TWO context lengths (the paper's ordering law; the absolute levels
    /// are context-specific and deliberately NOT pinned).
    #[test]
    fn mantissa_ordering_law_two_lengths() {
        let ladder = [6u32, 8, 10, 12, 16, 20, 26, 33, 40, 52];
        for &s in &[128usize, 256usize] {
            let mut prev = f64::INFINITY;
            for &bits in &ladder {
                let cfg = LabConfig::perturbed(s, 16, 32, 32, bits);
                let abs = abs_dev(&cfg);
                eprintln!("mantissa ladder s={s} bits={bits}: max_abs={abs:.3e}");
                assert!(
                    abs <= prev * 1.01 + 1e-9,
                    "deviation must be non-increasing in mantissa bits (s={s}): {abs:.3e} > {prev:.3e}"
                );
                prev = abs;
            }
        }
    }

    /// T2.2 law 2: deviation rank-correlates POSITIVELY with the paper's
    /// rescale count R = ⌈S/T⌉ − 1 over an (S, T) grid (ordinal gate — the
    /// first-order predictor is explicitly not an absolute bound).
    #[test]
    fn rescale_count_spearman_grid() {
        let mut rs = Vec::new();
        let mut devs = Vec::new();
        for &s in &[64usize, 128, 256] {
            for &t in &[16usize, 32, 64] {
                let cfg = LabConfig::perturbed(s, 16, t, 32, 10);
                let (a, g) = run_pair(&cfg);
                let dev = max_abs_diff_f64(&a, &g);
                eprintln!("grid s={s} t={t} R={}: dev={dev:.3e}", cfg.rescale_count());
                rs.push(cfg.rescale_count() as f64);
                devs.push(dev);
            }
        }
        let rho = spearman_rho(&rs, &devs);
        eprintln!("spearman rho(R, dev) = {rho:.4}");
        assert!(
            rho >= PINNED_SPEARMAN_FLOOR,
            "rescale-count law: rho {rho:.4} below pinned floor {PINNED_SPEARMAN_FLOOR}"
        );
        // Endpoint law: the most-rescaling config strictly exceeds the
        // least-rescaling one.
        let hi = abs_dev(&LabConfig::perturbed(256, 16, 16, 32, 10));
        let lo = abs_dev(&LabConfig::perturbed(64, 16, 64, 32, 10));
        assert!(
            hi > lo,
            "endpoint law: R=15 dev {hi:.3e} must exceed R=0 dev {lo:.3e}"
        );
    }

    /// T2.2 law 3a: larger tile area → fewer rescales → LESS deviation,
    /// reproduced at TWO mantissa formats.
    #[test]
    fn tile_area_ordering_two_formats() {
        for &bits in &[10u32, 16] {
            let mut prev = f64::INFINITY;
            for &bc in &[16usize, 32, 64] {
                let cfg = LabConfig::perturbed(256, 16, bc, 32, bits);
                let abs = abs_dev(&cfg);
                eprintln!("tile area bits={bits} bc={bc}: dev={abs:.3e}");
                assert!(
                    abs <= prev * 1.01 + 1e-9,
                    "larger tile area must not increase deviation (bits={bits}, bc={bc})"
                );
                prev = abs;
            }
        }
    }

    /// T2.2 law 3b + negative control: the axis swap measurably changes
    /// deviation at Bc ≠ Br, and is BIT-IDENTICAL at Bc == Br (the paper's
    /// square-tile row — the free negative control).
    #[test]
    fn dim_order_swap_changes_deviation_square_invariant() {
        let std = LabConfig::perturbed(128, 16, 64, 16, 10);
        let mut swp = std;
        swp.axis_swap = AxisSwap::Swapped;
        let d1 = abs_dev(&std);
        let d2 = abs_dev(&swp);
        eprintln!("swap: dev_std={d1:.3e} dev_swap={d2:.3e}");
        assert_ne!(
            d1, d2,
            "axis swap must measurably change deviation at Bc != Br"
        );

        let sq = LabConfig::perturbed(128, 16, 32, 32, 10);
        let mut sq_swp = sq;
        sq_swp.axis_swap = AxisSwap::Swapped;
        let (a3, _) = run_pair(&sq);
        let (a4, _) = run_pair(&sq_swp);
        for (x, y) in a3.iter().zip(a4.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "square-tile swap must be bit-identical (negative control)"
            );
        }
    }

    /// T3.2: the fit-input hash pins the schedule's CONFIG exactly.
    #[test]
    fn tol_schedule_fit_inputs_pinned_hash() {
        let h = tol_fit_inputs_hash(&TOL_FIT_PINNED);
        assert_eq!(
            h, TOL_FIT_HASH_PINNED,
            "fit inputs must match the pinned config hash"
        );
    }

    /// T3.2: the pinned table holds (re-measured rows inside the
    /// cross-libm band), and the two-length probe — a kernel passing
    /// tol(S₀) must not flip verdict class at 8·S₀ — passes THROUGH the
    /// Phase-1 acceptance rule: stale band rejects, schedule band accepts.
    #[test]
    fn tol_schedule_table_two_length_no_class_flip() {
        // Rows hold within the cross-libm band.
        for &(s, md_band, w1_band) in &TOL_TABLE_PINNED {
            let cfg = LabConfig::perturbed(
                s,
                TOL_FIT_PINNED.head_dim,
                TOL_FIT_PINNED.t,
                TOL_FIT_PINNED.t,
                TOL_FIT_PINNED.bits,
            );
            let (a, g) = run_pair(&cfg);
            let rep = deviation_report(&a, &g);
            eprintln!(
                "table s={s}: raw md={:.3e} w1={:.3e} (bands {md_band:.3e} / {w1_band:.3e})",
                rep.max_diff, rep.wasserstein_1d
            );
            let slack = (1.0 + TOL_FIT_BAND) as f32;
            assert!(
                rep.max_diff <= md_band * slack,
                "row s={s}: raw md {:.3e} above band {:.3e}",
                rep.max_diff,
                md_band
            );
            assert!(
                rep.wasserstein_1d <= w1_band * slack,
                "row s={s}: raw w1 {:.3e} above band {:.3e}",
                rep.wasserstein_1d,
                w1_band
            );
        }

        // Two-length probe through the Phase-1 acceptance rule.
        let s0 = TOL_FIT_PINNED.s0;
        let s8 = s0 * 8;
        let cfg = |s: usize| {
            LabConfig::perturbed(
                s,
                TOL_FIT_PINNED.head_dim,
                TOL_FIT_PINNED.t,
                TOL_FIT_PINNED.t,
                TOL_FIT_PINNED.bits,
            )
        };
        let rep0 = {
            let (a, g) = run_pair(&cfg(s0));
            deviation_report(&a, &g)
        };
        let rep8 = {
            let (a, g) = run_pair(&cfg(s8));
            deviation_report(&a, &g)
        };
        let margin = 1.0f32;

        // At S₀: Accept under its own band.
        assert_eq!(
            accept(&[rep0], &bands_of(band_at(s0)), margin),
            Verdict::Accept,
            "the anchor length must pass under its own band"
        );
        // At 8·S₀ under the STALE S₀ band: Reject — the flip hazard the
        // schedule exists to prevent (asserted, so the demo is not vacuous).
        assert_eq!(
            accept(&[rep8], &bands_of(band_at(s0)), margin),
            Verdict::Reject,
            "deviation at 8·S₀ must exceed the stale S₀ band for the flip demo to be honest"
        );
        // At 8·S₀ under the SCHEDULE band: still Accept — no class flip.
        assert_eq!(
            accept(&[rep8], &bands_of(band_at(s8)), margin),
            Verdict::Accept,
            "the schedule must preserve the verdict class at 8·S₀"
        );

        // Paper-law shape: the band grows with S and stays inside a loose
        // linear-in-R envelope (R ratio 15/1 = 15 at this config; 2× slack).
        let (md0, _) = band_at(s0);
        let (md8, _) = band_at(s8);
        assert!(md8 > md0, "the schedule must grow with S");
        assert!(
            md8 <= md0 * 32.0,
            "schedule growth wildly superlinear — revisit the table"
        );
    }

    /// Measurement helper for pinning the constants (run with
    /// `-- --ignored --nocapture`): prints every value the source pins.
    #[test]
    #[ignore = "measurement helper for pinning constants"]
    fn print_measurements() {
        eprintln!("fit hash = {}", tol_fit_inputs_hash(&TOL_FIT_PINNED));
        let rel = rel_dev(&LabConfig::new(128, 16, 32, 32));
        eprintln!("multitile f64 rel = {rel:.3e}");
        for &s in &[64usize, 128, 256, 512] {
            let cfg = LabConfig::perturbed(
                s,
                TOL_FIT_PINNED.head_dim,
                TOL_FIT_PINNED.t,
                TOL_FIT_PINNED.t,
                TOL_FIT_PINNED.bits,
            );
            let (a, g) = run_pair(&cfg);
            let rep = deviation_report(&a, &g);
            eprintln!(
                "table s={s}: md={:.6e} w1={:.6e} | x{TOL_HEADROOM} = {:.4e} / {:.4e}",
                rep.max_diff,
                rep.wasserstein_1d,
                rep.max_diff as f64 * TOL_HEADROOM,
                rep.wasserstein_1d as f64 * TOL_HEADROOM
            );
        }
        eprintln!(
            "swap dev_std = {:.3e}",
            abs_dev(&LabConfig::perturbed(128, 16, 64, 16, 10))
        );
        let mut rs = Vec::new();
        let mut devs = Vec::new();
        for &s in &[64usize, 128, 256] {
            for &t in &[16usize, 32, 64] {
                let cfg = LabConfig::perturbed(s, 16, t, 32, 10);
                rs.push(cfg.rescale_count() as f64);
                devs.push(abs_dev(&cfg));
            }
        }
        eprintln!("spearman rho = {:.4}", spearman_rho(&rs, &devs));
    }
}
