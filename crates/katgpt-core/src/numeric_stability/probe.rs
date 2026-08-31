//! Phase 1 probe (Issue 697 T1.1–T1.5 + the T3.1 falsifiability gate):
//! mantissa-truncation format emulator, two-surface deviation reports,
//! reference bands, and the contextualization acceptance rule.
//!
//! Substrate reuse: the 1-D Wasserstein on scalar slices delegates to
//! `crate::mag::transfer` — the shared quantile-grid core
//! (`wasserstein1d_sorted_core`) and its `quantile_interp` helper live
//! there; this module re-implements neither the sort nor the interpolation.

use core::fmt;

// ── T1.1: format emulator ──────────────────────────────────────────

/// Stored mantissa bits of an IEEE-754 binary64 (`f64`) significand field.
pub const F64_MANTISSA_BITS: u32 = 52;

/// Emulate a storage format with `bits` significand (mantissa) bits by
/// CLEARING the low mantissa bits of the `f64` bit pattern (round toward
/// zero in mantissa space — the paper's precision knob, arXiv:2405.02803
/// §1's number-format axis).
///
/// Properties + edges, documented honestly:
///
/// - **Idempotent**: truncation only clears bits, so
///   `truncate_mantissa(truncate_mantissa(x, b), b) ==
///   truncate_mantissa(x, b)` bit-exactly (pinned by
///   `truncate_mantissa_idempotent_ladder`).
/// - **Sign-preserving**: bit 63 (sign) is never touched; −3.0 truncates
///   exactly like +3.0 mirrored. ±0 is preserved bit-exactly; a tiny
///   subnormal may truncate to signed zero but never flips sign.
/// - **No exponent modeling**: this is a MANTISSA-WIDTH-ONLY emulator. It
///   does NOT reproduce a real narrow format's exponent range — f16's 65504
///   overflow, BF16's widened range, or format-specific subnormal ladders
///   are out of scope. Subnormals simply lose their low bits (and may
///   truncate to signed zero).
/// - **NaN payloads may collapse to ±Inf**: NaN has an all-ones exponent
///   and a nonzero mantissa; if ALL of a NaN payload's set bits lie below
///   the cleared region the result becomes infinity (pinned as documented
///   behavior in `truncate_mantissa_documented_nan_and_subnormal_edges`).
///   Inputs to this module's measurement paths are validated finite before
///   they can reach a report, so the collapse is unreachable there.
/// - **Truncation bias**: for `x > 0` the result is `<= x`; for `x < 0` the
///   result is `>= x` (magnitude shrinks toward the bin's low edge) — the
///   same toward-zero bias a real RTZ quantizer has.
/// - For an f32-representable NORMAL value the `f64` image has 29 zero low
///   mantissa bits, so `bits >= 23` is the identity on such inputs; f32
///   SUBNORMALS pack their significand into the low `f64` mantissa and can
///   lose bits at ANY width.
///
/// # Examples
///
/// ```
/// use katgpt_core::numeric_stability::truncate_mantissa;
/// let x = 1.5f64; // = 1.1₂ — one mantissa bit, at the top of the field
/// assert_eq!(truncate_mantissa(x, 52), x);
/// assert_eq!(truncate_mantissa(x, 23), x); // f32-representable normal
/// assert_eq!(truncate_mantissa(x, 0), 1.0); // mantissa cleared → 1.0·2⁰
/// // Truncation is a projection: applying it again changes nothing.
/// let t = truncate_mantissa(1.0f64 + f64::EPSILON, 10);
/// assert_eq!(truncate_mantissa(t, 10), t);
/// ```
#[inline]
pub fn truncate_mantissa(value: f64, bits: u32) -> f64 {
    if bits >= F64_MANTISSA_BITS {
        return value;
    }
    let drop = F64_MANTISSA_BITS - bits; // <= 52, so the shift cannot overflow
    let mask: u64 = (1u64 << drop) - 1;
    f64::from_bits(value.to_bits() & !mask)
}

// ── T1.2: deviation report ─────────────────────────────────────────

/// Errors returned by the numeric-stability probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NumericStabilityError {
    /// The two tensor slices have different lengths.
    LengthMismatch,
    /// A tensor slice is empty (nothing to measure).
    Empty,
    /// A tensor slice contains a non-finite value (`NaN` or ±inf). The NaN
    /// policy is reject-at-the-boundary: garbage never becomes a report.
    NonFinite,
}

impl fmt::Display for NumericStabilityError {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch => write!(f, "numeric_stability: tensor length mismatch"),
            Self::Empty => write!(f, "numeric_stability: empty tensor"),
            Self::NonFinite => write!(f, "numeric_stability: non-finite tensor input"),
        }
    }
}

impl std::error::Error for NumericStabilityError {}

/// Two-surface numeric deviation between a variant and its baseline
/// (arXiv:2405.02803 §1's measurement pair).
///
/// - `max_diff` — elementwise bound `max_i |x_i − y_i|`. Cheap, local, and
///   blind to where mass moved: it cannot distinguish one outlier from a
///   systematic shift.
/// - `wasserstein_1d` — distribution-aware surface (the substrate's
///   quantile-grid W1 definition, delegated to `crate::mag::transfer` — see
///   the module docs). It sees mass movement that `max_diff` under-counts;
///   in exchange it has no hard elementwise bound. The paper's protocol
///   carries BOTH for exactly this complementarity.
///
/// Construct via [`DeviationReport::compute`] (cold, allocating) or
/// [`DeviationReport::compute_into`] (hot, caller scratch). Both are pure
/// functions of their inputs: deterministic, symmetric
/// (`compute(x, y)` ≡ `compute(y, x)` bit-exactly), permutation-invariant,
/// and free of HashMap/wall-clock/parallelism.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviationReport {
    /// Elementwise max absolute difference. Finite for validated inputs
    /// EXCEPT the documented saturation case (opposite-signed near
    /// `f32::MAX` operands overflow the difference to `+inf` — an honest
    /// "unboundedly different" answer; [`accept`] fails such reports closed
    /// to `Reject`).
    pub max_diff: f32,
    /// 1-D Wasserstein distance over the shared quantile grid
    /// (`crate::mag::transfer` definition).
    pub wasserstein_1d: f32,
}

impl DeviationReport {
    /// Bit-identity tuple for determinism pinning (the house FNV-anchor
    /// convention, applied to a two-float report).
    #[inline]
    pub fn bit_identity(&self) -> (u32, u32) {
        (self.max_diff.to_bits(), self.wasserstein_1d.to_bits())
    }

    /// Compute the deviation report between two 1-D tensor slices.
    /// Allocating cold-path form (wraps [`DeviationReport::compute_into`]).
    ///
    /// Errors: [`NumericStabilityError::LengthMismatch`],
    /// [`NumericStabilityError::Empty`],
    /// [`NumericStabilityError::NonFinite`] (NaN policy: rejected at the
    /// boundary, so the substrate sort below only ever sees a total order).
    pub fn compute(x: &[f32], y: &[f32]) -> Result<Self, NumericStabilityError> {
        let mut scratch_x = Vec::new();
        let mut scratch_y = Vec::new();
        Self::compute_into(x, y, &mut scratch_x, &mut scratch_y)
    }

    /// Zero-alloc hot-path form of [`DeviationReport::compute`].
    ///
    /// `scratch_x` / `scratch_y` are grow-only sort buffers (the Plan 418
    /// `*_into` pattern): they are `clear()`ed and refilled per call, so
    /// after one call at the largest tensor size every smaller call is
    /// allocation-free (steady state pinned by
    /// `g4_compute_into_steady_state_zero_alloc`). Scratch contents are
    /// unspecified after the call.
    pub fn compute_into(
        x: &[f32],
        y: &[f32],
        scratch_x: &mut Vec<f32>,
        scratch_y: &mut Vec<f32>,
    ) -> Result<Self, NumericStabilityError> {
        if x.len() != y.len() {
            return Err(NumericStabilityError::LengthMismatch);
        }
        if x.is_empty() {
            return Err(NumericStabilityError::Empty);
        }
        // NaN policy boundary: reject before any sort/compare so every
        // downstream comparison is a total order.
        if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
            return Err(NumericStabilityError::NonFinite);
        }
        let mut max_diff = 0.0_f32;
        for (a, b) in x.iter().zip(y.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        // Delegate to the mag substrate (Plan 418) — the shared
        // quantile-grid core; no metric math is duplicated here.
        let wasserstein_1d =
            crate::mag::transfer::wasserstein1d_scalar_into(x, y, scratch_x, scratch_y);
        Ok(Self {
            max_diff,
            wasserstein_1d,
        })
    }
}

// ── T1.3: the acceptance rule ──────────────────────────────────────

/// Verdict of the contextualization acceptance rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Verdict {
    /// Every report sits strictly inside every (binding) reference band
    /// scaled by the caller's margin.
    Accept,
    /// At least one report crosses the margin line on at least one surface:
    /// the deviation exceeds what the references contextualize as tolerated.
    Reject,
    /// Nothing rejects, but something sits EXACTLY on a margin line (or a
    /// guard rail fired: empty evidence, a nonsense margin, invalid bands).
    /// The honest answer is "measure more / fix the inputs", not a pass.
    Inconclusive,
}

/// The reference bands a variant's deviation is contextualized against.
///
/// `r1` is the two-draw init-divergence band (`reference_r1_two_draws`);
/// `r2` is the precision-change band (`reference_r2_roundtrip` /
/// `reference_r2_custom`), which is a **single-step lower bound** — see
/// those builders and `R2_LABEL`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceBands {
    /// R1: divergence between two draws of the same init distribution.
    pub r1: DeviationReport,
    /// R2: single-step lower bound on precision-change divergence
    /// (quantize→dequant round-trip proxy).
    pub r2: DeviationReport,
}

impl ReferenceBands {
    /// The conservative (smaller) max-diff band — the binding one under the
    /// dominance rule (a deviation must clear the STRICTER reference).
    #[inline]
    pub fn binding_max_diff(&self) -> f32 {
        self.r1.max_diff.min(self.r2.max_diff)
    }

    /// The conservative (smaller) Wasserstein band.
    #[inline]
    pub fn binding_wasserstein(&self) -> f32 {
        self.r1.wasserstein_1d.min(self.r2.wasserstein_1d)
    }

    /// A band set is valid iff all four band values are finite and
    /// non-negative. Invalid bands make `accept` return `Inconclusive`
    /// (fail-safe: no verdict is derived from nonsense references).
    #[inline]
    pub fn is_valid(&self) -> bool {
        [
            self.r1.max_diff,
            self.r2.max_diff,
            self.r1.wasserstein_1d,
            self.r2.wasserstein_1d,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0)
    }
}

/// The contextualization acceptance rule (arXiv:2405.02803 §2's dominance
/// idea, made total): `report` passes surface `m` iff
/// `report.m < band_m * margin`, where `band_m` is the BINDING (smaller) of
/// the two reference bands on that surface. Quantifier: EVERY report must
/// pass EVERY surface — a single failure rejects the batch.
///
/// Margin-line boundary, defined precisely: with `line_m = band_m * margin`,
///
/// - `report.m > line_m` → reject signal,
/// - `report.m == line_m` (exact `f32` equality) → margin-line signal,
/// - `report.m < line_m` → pass.
///
/// Any reject signal → [`Verdict::Reject`]; else any margin-line signal →
/// [`Verdict::Inconclusive`]; else [`Verdict::Accept`].
///
/// **`margin` has no default margin value and this API ships none** — the
/// caller must justify it from its OWN references. There is deliberately no
/// constant derived from the paper's context-specific "2–5×" headline: that
/// number measured ONE model at ONE seq-len on ONE machine, and importing it
/// here would be exactly the hand-picked-band practice this rule replaces.
/// Sensible callers pin `margin` per gate site and record it next to the
/// bands (e.g. `margin = 1.0` is the strict "dominated by the references,
/// full stop" reading; larger values buy slack and must be argued).
///
/// Guard rails (fail-safe / fail-closed):
/// - empty `reports` → `Inconclusive` (no evidence is not acceptance);
/// - non-finite or negative `margin` → `Inconclusive`;
/// - invalid bands ([`ReferenceBands::is_valid`] false) → `Inconclusive`;
/// - a report with a non-finite or negative value → `Reject` (garbage
///   measurement must not pass; includes the documented `max_diff`
///   overflow-to-inf saturation case).
///
/// A zero band (a degenerate reference, e.g. two identical draws) is legal:
/// with `margin = 0` only exact-identity passes, and any zero-deviation
/// report sits exactly ON the zero line → `Inconclusive` — the rule honestly
/// reports that such references carry no contextualizing information.
pub fn accept(reports: &[DeviationReport], refs: &ReferenceBands, margin: f32) -> Verdict {
    if reports.is_empty() {
        return Verdict::Inconclusive;
    }
    if !refs.is_valid() || !margin.is_finite() || margin < 0.0 {
        return Verdict::Inconclusive;
    }
    let line_md = refs.binding_max_diff() * margin;
    let line_w = refs.binding_wasserstein() * margin;
    let mut any_margin_line = false;
    for r in reports {
        // Fail closed on garbage reports (NaN compares false everywhere and
        // would otherwise silently "pass").
        let valid = r.max_diff.is_finite()
            && r.max_diff >= 0.0
            && r.wasserstein_1d.is_finite()
            && r.wasserstein_1d >= 0.0;
        if !valid {
            return Verdict::Reject;
        }
        if r.max_diff > line_md || r.wasserstein_1d > line_w {
            return Verdict::Reject;
        }
        if r.max_diff == line_md || r.wasserstein_1d == line_w {
            any_margin_line = true;
        }
    }
    if any_margin_line {
        Verdict::Inconclusive
    } else {
        Verdict::Accept
    }
}

// ── T1.4: reference builders ────────────────────────────────────────

/// R1 reference band: divergence between two draws from the caller's init
/// distribution (arXiv:2405.02803 §2's "different-random-init" reference).
///
/// The caller draws the SAME shape twice under two seeds (its own RNG —
/// this crate adds no dependency) and hands both slices here; the returned
/// report is the band the system demonstrably tolerates, because re-seeding
/// alone produces that much deviation.
///
/// Errors: as [`DeviationReport::compute`].
pub fn reference_r1_two_draws(
    draw_a: &[f32],
    draw_b: &[f32],
) -> Result<DeviationReport, NumericStabilityError> {
    DeviationReport::compute(draw_a, draw_b)
}

/// Zero-alloc quantize→dequant round-trip driver (the crate-supplied R2
/// quantizer, built on [`truncate_mantissa`]): for each element, promote
/// `f32 → f64` (exact), truncate the mantissa to `bits`, and cast back.
///
/// The cast back to `f32` is EXACT by construction — truncation preserves
/// `f32`-representability (normals keep their exponent and a truncated
/// mantissa that still fits 23 bits; f32 subnormals remain multiples of
/// `2^-149` or truncate to zero) — so there is no double rounding.
/// `out.len() >= values.len()` is required; `out[..values.len()]` receives
/// the round-tripped values.
pub fn roundtrip_truncate_mantissa_into(
    values: &[f32],
    bits: u32,
    out: &mut [f32],
) -> Result<(), NumericStabilityError> {
    if out.len() < values.len() {
        return Err(NumericStabilityError::LengthMismatch);
    }
    for (dst, &v) in out.iter_mut().zip(values.iter()) {
        *dst = truncate_mantissa(v as f64, bits) as f32;
    }
    Ok(())
}

/// Allocating convenience wrapper over
/// [`roundtrip_truncate_mantissa_into`] (cold-path band construction).
pub fn roundtrip_truncate_mantissa(values: &[f32], bits: u32) -> Vec<f32> {
    let mut out = vec![0.0_f32; values.len()];
    // Cannot fail: out is sized to values.
    let _ = roundtrip_truncate_mantissa_into(values, bits, &mut out);
    out
}

/// R2 reference band — single-step lower bound on precision-change
/// divergence (arXiv:2405.02803 §2's "FP16-vs-FP32 training" reference,
/// proxied modellessly).
///
/// **Label: single-step lower bound.** The band is the deviation of EXACTLY
/// ONE quantize→dequant round-trip over `baseline` (the crate's
/// mantissa-truncation driver). A faithful trained R2 — precision-change
/// divergence accumulated across layers and optimizer steps — needs
/// training runs and is out of scope for a modelless crate; real
/// precision-change divergence is a MULTI-event process (every weight is
/// re-quantized at every layer/step, and intermediates amplify residuals),
/// so this proxy LOWER-bounds it. `r2_lower_bound_label_tripwire` pins the
/// label AND the single-round-trip behavior, and demonstrates a composed
/// pipeline exceeding the band.
///
/// Errors: as [`DeviationReport::compute`] (non-finite baselines are
/// rejected at the report boundary).
pub fn reference_r2_roundtrip(
    baseline: &[f32],
    bits: u32,
) -> Result<DeviationReport, NumericStabilityError> {
    let roundtripped = roundtrip_truncate_mantissa(baseline, bits);
    DeviationReport::compute(baseline, &roundtripped)
}

/// R2 reference band with a CALLER-SUPPLIED quantize fn (e.g. a real RTN
/// round-to-nearest, a per-channel scale, a format cast). **Label: single-step
/// lower bound** — the same caveat as [`reference_r2_roundtrip`]: one
/// quantization event, a lower bound on real precision-change divergence.
///
/// The quantizer must map finite inputs to finite outputs; a non-finite
/// output is rejected by the report boundary
/// ([`NumericStabilityError::NonFinite`]).
pub fn reference_r2_custom(
    baseline: &[f32],
    quantize: impl Fn(f32) -> f32,
) -> Result<DeviationReport, NumericStabilityError> {
    let mut quantized = vec![0.0_f32; baseline.len()];
    for (dst, &v) in quantized.iter_mut().zip(baseline.iter()) {
        *dst = quantize(v);
    }
    DeviationReport::compute(baseline, &quantized)
}

/// The doc-truth label carried by every R2 surface (module docs, builder
/// docs, the `ReferenceBands::r2` field). `r2_lower_bound_label_tripwire`
/// pins both the label's presence and the single-round-trip behavior it
/// promises.
pub const R2_LABEL: &str = "single-step lower bound";

// ── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* tensor in `[0, 1)` — no rand dep, no
    /// HashMap, no wall-clock (the house LCG fixture convention).
    fn lcg_f32s(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0xA076_1D64_78BD_642F);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32) / ((1u32 << 24) as f32)
            })
            .collect()
    }

    /// Deterministic finite f64 stream (bits-reinterpreted xorshift,
    /// regenerating on non-finite — a fixed, seed-determined subsequence).
    fn lcg_f64_finite(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed ^ 0xDEAD_BEEF_CAFE_F00D;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = f64::from_bits(s);
            if v.is_finite() {
                out.push(v);
            }
        }
        out.shrink_to_fit();
        out
    }

    // ── T1.1: truncate_mantissa ─────────────────────────────────────

    #[test]
    fn truncate_mantissa_known_values() {
        // Full width is the identity; 1.0's mantissa is already all-zero.
        assert_eq!(
            truncate_mantissa(1.0, F64_MANTISSA_BITS).to_bits(),
            1.0f64.to_bits()
        );
        assert_eq!(truncate_mantissa(1.0, 0).to_bits(), 1.0f64.to_bits());
        // Bits beyond f64 are clamped to identity.
        assert_eq!(
            truncate_mantissa(1.0 + f64::EPSILON, 53),
            1.0 + f64::EPSILON
        );
        assert_eq!(
            truncate_mantissa(1.0 + f64::EPSILON, 100),
            1.0 + f64::EPSILON
        );
        // An f32-representable normal has 29 zero low mantissa bits in f64.
        assert_eq!(truncate_mantissa(1.5, 23).to_bits(), 1.5f64.to_bits());
        // Clearing everything leaves the exponent: 1.5 → 1.0.
        assert_eq!(truncate_mantissa(1.5, 0), 1.0);
        // 3.0 = 1.1₂·2¹ keeps its top mantissa bit (the 0.5 contribution)
        // at width 1 → still 3.0; only width 0 (mantissa fully cleared)
        // collapses it to 2.0. At width 51 its only set bit is above the
        // cut → unchanged.
        assert_eq!(truncate_mantissa(3.0, 1), 3.0);
        assert_eq!(truncate_mantissa(3.0, 0), 2.0);
        assert_eq!(truncate_mantissa(3.0, 51), 3.0);
        // The lowest f64 mantissa bit is exactly f64::EPSILON above 1.0.
        assert_eq!(truncate_mantissa(1.0 + f64::EPSILON, 51), 1.0);
        // f64::MAX with a zeroed mantissa collapses to 2^1023.
        assert_eq!(truncate_mantissa(f64::MAX, 0), 2f64.powi(1023));
    }

    #[test]
    fn truncate_mantissa_idempotent_ladder() {
        let values = lcg_f64_finite(0x1D1, 512);
        for &b in &[0u32, 1, 7, 10, 22, 23, 24, 40, 51, 52] {
            for &v in &values {
                let t = truncate_mantissa(v, b);
                let t2 = truncate_mantissa(t, b);
                assert_eq!(
                    t.to_bits(),
                    t2.to_bits(),
                    "truncation must be idempotent at width {b}"
                );
            }
        }
    }

    #[test]
    fn truncate_mantissa_sign_and_zero_edges() {
        // ±0 preserved bit-exactly (sign bit untouched by the mask).
        assert_eq!(truncate_mantissa(0.0, 0).to_bits(), 0.0f64.to_bits());
        assert_eq!(truncate_mantissa(-0.0, 0).to_bits(), (-0.0f64).to_bits());
        // Negatives truncate like positives mirrored (magnitude shrinks
        // toward the bin's low edge; sign bit never touched).
        assert_eq!(truncate_mantissa(-3.0, 1), -3.0);
        assert_eq!(truncate_mantissa(-3.0, 0), -2.0);
        assert_eq!(truncate_mantissa(-1.5, 0), -1.0);
        // Smallest subnormal truncates to signed zero (sign preserved).
        assert_eq!(
            truncate_mantissa(f64::from_bits(1), 0).to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(
            truncate_mantissa(f64::from_bits(0x8000_0000_0000_0001), 0).to_bits(),
            (-0.0f64).to_bits()
        );
        // Truncation bias: toward zero on both signs (documented).
        let v = 1.0 + 255.0 * f64::EPSILON;
        let t = truncate_mantissa(v, 44);
        assert!(t < v && t > 0.0);
        assert!(truncate_mantissa(-v, 44) > -v);
    }

    #[test]
    fn truncate_mantissa_documented_nan_and_subnormal_edges() {
        // A NaN whose ONLY payload bit is below the cut collapses to +Inf —
        // the documented hazard (measurement paths validate finite first).
        let low_payload_nan = f64::from_bits(0x7FF0_0000_0000_0001);
        assert_eq!(
            truncate_mantissa(low_payload_nan, 0).to_bits(),
            f64::INFINITY.to_bits()
        );
        // A quiet NaN keeps its (top) payload bit at width 51 but loses it
        // at width 0 → also +Inf.
        let quiet_nan = f64::from_bits(0x7FF8_0000_0000_0000);
        assert!(truncate_mantissa(quiet_nan, 51).is_nan());
        assert_eq!(
            truncate_mantissa(quiet_nan, 0).to_bits(),
            f64::INFINITY.to_bits()
        );
        // A mid-size subnormal loses exactly the cleared bits (honest
        // subnormal-truncation behavior, pinned so a "fix" is deliberate).
        let sub = f64::from_bits(0x000F_FFFF_FFFF_FFFF);
        let cut = truncate_mantissa(sub, 23);
        assert!(cut.is_subnormal() && cut < sub);
        assert_eq!(cut.to_bits(), 0x000F_FFFF_E000_0000);
    }

    // ── T1.2: DeviationReport ───────────────────────────────────────

    #[test]
    fn deviation_report_golden_quantile_grid() {
        // Hand-computed on the substrate's quantile-grid W1:
        // x = [0, 1], y = [0.5, 1]; t = 2; grid f = {0.25, 0.75};
        // q_x = {0.5, 1.0}, q_y = {0.75, 1.0} → W1 = (0.25 + 0)/2 = 0.125.
        // All values are exact binary fractions → exact f32 equality.
        let report = DeviationReport::compute(&[0.0, 1.0], &[0.5, 1.0]).unwrap();
        assert_eq!(report.max_diff, 0.5);
        assert_eq!(report.wasserstein_1d, 0.125);
    }

    #[test]
    fn deviation_report_rejects_bad_input() {
        assert_eq!(
            DeviationReport::compute(&[1.0], &[1.0, 2.0]),
            Err(NumericStabilityError::LengthMismatch)
        );
        assert_eq!(
            DeviationReport::compute(&[], &[]),
            Err(NumericStabilityError::Empty)
        );
        assert_eq!(
            DeviationReport::compute(&[f32::NAN], &[0.0]),
            Err(NumericStabilityError::NonFinite)
        );
        assert_eq!(
            DeviationReport::compute(&[0.0], &[f32::INFINITY]),
            Err(NumericStabilityError::NonFinite)
        );
    }

    #[test]
    fn deviation_report_bit_identical_symmetric_permutation_invariant() {
        let n = 512;
        let x = lcg_f32s(3, n);
        let y = lcg_f32s(4, n);
        let a = DeviationReport::compute(&x, &y).unwrap();
        // Determinism: identical inputs → bit-identical report.
        let b = DeviationReport::compute(&x, &y).unwrap();
        assert_eq!(a.bit_identity(), b.bit_identity());
        // Symmetry: swapping the sides is bit-exact.
        let swapped = DeviationReport::compute(&y, &x).unwrap();
        assert_eq!(a.bit_identity(), swapped.bit_identity());
        // Permutation invariance (same permutation on both sides) — no
        // iteration-order leakage anywhere in the pipeline.
        let mut idx: Vec<usize> = (0..n).collect();
        let mut s = 0x5EED_5EED_5EED_u64;
        for i in (1..n).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            idx.swap(i, (s >> 33) as usize % (i + 1));
        }
        let px: Vec<f32> = idx.iter().map(|&i| x[i]).collect();
        let py: Vec<f32> = idx.iter().map(|&i| y[i]).collect();
        let permuted = DeviationReport::compute(&px, &py).unwrap();
        assert_eq!(a.bit_identity(), permuted.bit_identity());
    }

    #[test]
    fn wasserstein_matches_independent_reference() {
        // Drift pin: an in-test transcription of the substrate's
        // quantile-grid W1 (sorted quantile functions on the max(m, n)
        // grid, linear interpolation with the 0.9999 clamp). If the
        // substrate grid changes shape, this reds and the deviation
        // numbers must be re-baselined deliberately.
        let x = lcg_f32s(9, 257); // odd length exercises the grid
        let y = lcg_f32s(10, 257);
        let report = DeviationReport::compute(&x, &y).unwrap();

        let mut sx = x.clone();
        let mut sy = y.clone();
        sx.sort_by(f32::total_cmp);
        sy.sort_by(f32::total_cmp);
        let q = |s: &[f32], f: f32| -> f32 {
            let pos = f.clamp(0.0, 0.9999) * s.len() as f32;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(s.len() - 1);
            let frac = pos - lo as f32;
            s[lo] * (1.0 - frac) + s[hi] * frac
        };
        let t = x.len().max(y.len());
        let mut dist = 0.0_f32;
        for k in 0..t {
            let f = (k as f32 + 0.5) / t as f32;
            dist += (q(&sx, f) - q(&sy, f)).abs();
        }
        assert_eq!(report.wasserstein_1d.to_bits(), (dist / t as f32).to_bits());
    }

    // ── T1.3: accept() ──────────────────────────────────────────────

    #[test]
    fn accept_rule_polarity_inside_line_outside() {
        // Hand-built bands: deviation of [d, 0, 0, 0] vs zeros has
        // max_diff = d and wasserstein = d/4 (three grid points at 0).
        let zeros = [0.0_f32; 4];
        let band = DeviationReport::compute(&zeros, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(band.max_diff, 1.0);
        // Grid quantiles of [0,0,0,1] at t=4 are {0, 0, 0.5, 1} → W1 = 1.5/4.
        assert_eq!(band.wasserstein_1d, 0.375);
        let refs = ReferenceBands { r1: band, r2: band };

        let report = |d: f32| DeviationReport::compute(&zeros, &[d, 0.0, 0.0, 0.0]).unwrap();
        // margin 1.0: line = band. Strictly inside → Accept.
        assert_eq!(accept(&[report(0.5)], &refs, 1.0), Verdict::Accept);
        // Exactly on the line → Inconclusive.
        assert_eq!(accept(&[report(1.0)], &refs, 1.0), Verdict::Inconclusive);
        // Beyond → Reject.
        assert_eq!(accept(&[report(2.0)], &refs, 1.0), Verdict::Reject);
        // Quantifier: every report must pass — one failure rejects the batch.
        assert_eq!(
            accept(&[report(0.5), report(0.25)], &refs, 1.0),
            Verdict::Accept
        );
        assert_eq!(
            accept(&[report(0.5), report(4.0)], &refs, 1.0),
            Verdict::Reject
        );
        // The W1 surface decides on its own too (max-diff flat, W1 over).
        // [0, d] vs [0, 2d]: max_diff = d = band, but W1 doubles.
        let flat = DeviationReport::compute(&[0.0, 1.0], &[0.0, 2.0]).unwrap();
        assert_eq!(flat.max_diff, 1.0);
        assert!(flat.wasserstein_1d > band.wasserstein_1d);
        assert_eq!(accept(&[flat], &refs, 1.0), Verdict::Reject);
    }

    #[test]
    fn accept_guard_rails_fail_closed() {
        let zeros = [0.0_f32; 4];
        let band = DeviationReport::compute(&zeros, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let refs = ReferenceBands { r1: band, r2: band };
        let inside = DeviationReport::compute(&zeros, &[0.5, 0.0, 0.0, 0.0]).unwrap();

        // No evidence is not acceptance.
        assert_eq!(accept(&[], &refs, 1.0), Verdict::Inconclusive);
        // Nonsense margins → Inconclusive (never a silent pass: NaN compares
        // false, which would otherwise accept everything).
        assert_eq!(accept(&[inside], &refs, f32::NAN), Verdict::Inconclusive);
        assert_eq!(
            accept(&[inside], &refs, f32::INFINITY),
            Verdict::Inconclusive
        );
        assert_eq!(accept(&[inside], &refs, -1.0), Verdict::Inconclusive);
        // Invalid bands → Inconclusive.
        let nan_refs = ReferenceBands {
            r1: DeviationReport {
                max_diff: f32::NAN,
                wasserstein_1d: 0.0,
            },
            r2: band,
        };
        assert!(!nan_refs.is_valid());
        assert_eq!(accept(&[inside], &nan_refs, 1.0), Verdict::Inconclusive);
        let neg_refs = ReferenceBands {
            r1: band,
            r2: DeviationReport {
                max_diff: 1.0,
                wasserstein_1d: -1e-6,
            },
        };
        assert_eq!(accept(&[inside], &neg_refs, 1.0), Verdict::Inconclusive);
        // Garbage reports fail CLOSED to Reject (NaN/negative/inf must not
        // pass; report fields are public so hand-built reports are possible).
        let nan_report = DeviationReport {
            max_diff: f32::NAN,
            wasserstein_1d: 0.0,
        };
        assert_eq!(accept(&[nan_report], &refs, 1.0), Verdict::Reject);
        let inf_report = DeviationReport {
            max_diff: f32::INFINITY,
            wasserstein_1d: 0.0,
        };
        assert_eq!(accept(&[inf_report], &refs, 1.0), Verdict::Reject);
        let neg_report = DeviationReport {
            max_diff: 0.1,
            wasserstein_1d: -0.5,
        };
        assert_eq!(accept(&[neg_report], &refs, 1.0), Verdict::Reject);
    }

    #[test]
    fn accept_margin_moves_the_line() {
        let zeros = [0.0_f32; 4];
        let band = DeviationReport::compute(&zeros, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let refs = ReferenceBands { r1: band, r2: band };
        let report = |d: f32| DeviationReport::compute(&zeros, &[d, 0.0, 0.0, 0.0]).unwrap();
        // margin 2.0 → line at 2×band: 1.5× passes, exactly 2× is the line,
        // 2.5× rejects.
        assert_eq!(accept(&[report(1.5)], &refs, 2.0), Verdict::Accept);
        assert_eq!(accept(&[report(2.0)], &refs, 2.0), Verdict::Inconclusive);
        assert_eq!(accept(&[report(2.5)], &refs, 2.0), Verdict::Reject);
        // Tightening the margin to 0.5 moves the line to 0.5 (below the
        // band): exactly-on-line is Inconclusive, inside is Accept, and a
        // 0.75 plant now REJECTS even though it is inside the raw band.
        assert_eq!(accept(&[report(0.5)], &refs, 0.5), Verdict::Inconclusive);
        assert_eq!(accept(&[report(0.25)], &refs, 0.5), Verdict::Accept);
        assert_eq!(accept(&[report(0.75)], &refs, 0.5), Verdict::Reject);
    }

    // ── T1.4: reference builders ────────────────────────────────────

    #[test]
    fn reference_builders_r1_and_r2() {
        // R1: two draws → a finite, mostly-nonzero band; identical draws →
        // the exact zero report.
        let a = lcg_f32s(1, 1024);
        let b = lcg_f32s(2, 1024);
        let r1 = reference_r1_two_draws(&a, &b).unwrap();
        assert!(r1.max_diff > 0.0 && r1.wasserstein_1d > 0.0);
        assert_eq!(
            reference_r1_two_draws(&a, &a).unwrap().bit_identity(),
            (0, 0)
        );
        // R2 at full width is the identity → the zero band.
        assert_eq!(
            reference_r2_roundtrip(&a, 52).unwrap().bit_identity(),
            (0, 0)
        );
        // R2 at 13 bits is strictly positive (truncation shifts values).
        let r2 = reference_r2_roundtrip(&a, 13).unwrap();
        assert!(r2.max_diff > 0.0);
        // The crate driver is the exact round-trip (cast-back exactness).
        let mut out = vec![0.0_f32; a.len()];
        roundtrip_truncate_mantissa_into(&a, 13, &mut out).unwrap();
        for (dst, &v) in out.iter().zip(a.iter()) {
            assert_eq!(*dst, truncate_mantissa(v as f64, 13) as f32);
        }
        // bits >= 23 on f32 normals is the identity round-trip.
        roundtrip_truncate_mantissa_into(&a, 23, &mut out).unwrap();
        assert_eq!(out, a);
        // Caller-supplied quantizer is honored exactly.
        let custom = reference_r2_custom(&a, |v| v.abs()).unwrap();
        let abs: Vec<f32> = a.iter().map(|v| v.abs()).collect();
        let expected = DeviationReport::compute(&a, &abs).unwrap();
        assert_eq!(custom.bit_identity(), expected.bit_identity());
        // Builder errors propagate from the report boundary.
        assert_eq!(
            reference_r2_roundtrip(&[], 13),
            Err(NumericStabilityError::Empty)
        );
        assert_eq!(
            reference_r2_custom(&a, |_| f32::NAN),
            Err(NumericStabilityError::NonFinite)
        );
        let mut small = vec![0.0_f32; 3];
        assert_eq!(
            roundtrip_truncate_mantissa_into(&a, 13, &mut small),
            Err(NumericStabilityError::LengthMismatch)
        );
    }

    // ── T3.1: the planted-deviation falsifiability gate ─────────────

    #[test]
    fn t31_planted_deviation_gate_accept_margin_line_reject() {
        const N: usize = 4096;
        let mut baseline = lcg_f32s(0x697, N);
        baseline[0] = 0.0; // exact plant site: |y[0] − 0| == y[0] bit-exactly
        let draw_b = lcg_f32s(0x698, N);
        let r1 = reference_r1_two_draws(&baseline, &draw_b).unwrap();
        let r2 = reference_r2_roundtrip(&baseline, 13).unwrap();
        let refs = ReferenceBands { r1, r2 };
        assert!(refs.is_valid(), "fixture bands must be valid");
        let binding_md = refs.binding_max_diff();
        assert!(binding_md > 0.0, "degenerate fixture: zero binding band");

        // Deviations at 0.1x / 1.0x / 10x of the reference band must land
        // Accept / margin-line / Reject at margin = 1.0. A gate that cannot
        // fail proves nothing — this is the can-it-fail proof.
        for (scale, expected) in [
            (0.1_f32, Verdict::Accept),
            (1.0, Verdict::Inconclusive),
            (10.0, Verdict::Reject),
        ] {
            let mut y = baseline.clone();
            y[0] = scale * binding_md;
            let report = DeviationReport::compute(&baseline, &y).unwrap();
            // The plant is exact: max_diff equals the planted delta.
            assert_eq!(report.max_diff, scale * binding_md);
            // The W1 surface must stay strictly inside the band for a
            // single-element plant (the distribution barely moves) — this
            // is what makes max_diff the deciding surface in this fixture;
            // if a fixture change breaks it, the gate design must be
            // revisited, not the assertion relaxed.
            assert!(
                report.wasserstein_1d < refs.binding_wasserstein(),
                "scale={scale}: plant moved W1 outside the band; fixture broken"
            );
            assert_eq!(
                accept(&[report], &refs, 1.0),
                expected,
                "scale={scale} report={report:?}"
            );
        }
    }

    // ── G2: perf (release-only, the house g2_ convention) ───────────

    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn g2_deviation_report_into_under_budget() {
        let n = 4096;
        let x = lcg_f32s(11, n);
        let y = lcg_f32s(12, n);
        let mut scratch_x = Vec::with_capacity(n);
        let mut scratch_y = Vec::with_capacity(n);
        let _ = DeviationReport::compute_into(&x, &y, &mut scratch_x, &mut scratch_y).unwrap();
        let iters = 200;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let r = DeviationReport::compute_into(&x, &y, &mut scratch_x, &mut scratch_y).unwrap();
            std::hint::black_box(r.bit_identity());
        }
        let ns_per = t0.elapsed().as_nanos() as f64 / iters as f64;
        eprintln!("g2: deviation_report_into = {ns_per:.1} ns/report (n={n})");
        // Measured 151-163 us/report across 4 release runs on this box —
        // the two 4096-element comparator sorts dominate (~37 ns/element
        // including the quantile grid). Bound is 1.5x the observed ceiling.
        assert!(
            ns_per < 250_000.0,
            "deviation_report_into too slow: {ns_per:.1} ns/report (bound 250000, n={n})"
        );
    }

    // ── G4: steady-state zero-alloc (debug-only: TrackingAllocator) ─

    #[test]
    #[cfg(all(test, debug_assertions))]
    fn g4_compute_into_steady_state_zero_alloc() {
        // n=1024, 20 iterations: an allocation-free hot path shows count==0
        // after ONE steady-state call, so the sweep exists to catch
        // state-dependent growth, not to burn debug-mode CPU (this suite
        // runs in parallel with debug timing gates — keep the footprint
        // small; the heavy perf measurement is the release-only g2 test).
        let n = 1024;
        let x = lcg_f32s(7, n);
        let y = lcg_f32s(8, n);
        let mut scratch_x: Vec<f32> = Vec::new();
        let mut scratch_y: Vec<f32> = Vec::new();
        // Warm-up grows the scratch to capacity.
        let _ = DeviationReport::compute_into(&x, &y, &mut scratch_x, &mut scratch_y).unwrap();
        crate::alloc::reset_alloc_stats();
        for _ in 0..20 {
            let r = DeviationReport::compute_into(&x, &y, &mut scratch_x, &mut scratch_y).unwrap();
            assert!(r.max_diff >= 0.0);
        }
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(
            count, 0,
            "compute_into must be zero-alloc in steady state (count={count})"
        );
    }

    // ── doc-truth tripwires (the EMPTY_HASH-preimage pattern) ───────

    /// Collapse whitespace and strip the `///` / `//!` doc markers per
    /// line, so doc-truth greps match phrases regardless of line wrapping.
    fn norm_doc(s: &str) -> String {
        let stripped: Vec<&str> = s
            .lines()
            .map(|line| {
                let t = line.trim_start();
                if let Some(rest) = t.strip_prefix("///") {
                    rest
                } else if let Some(rest) = t.strip_prefix("//!") {
                    rest
                } else {
                    line
                }
            })
            .collect();
        stripped
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The API/docs region of this file: everything before the test module.
    /// The doc-truth greps MUST scope here — `include_str!` reads the whole
    /// file, and the test module's own needle literals would otherwise
    /// self-trip the negative assertions.
    fn api_doc_region() -> String {
        const PROBE: &str = include_str!("probe.rs");
        let api = PROBE.split("#[cfg(test)]").next().unwrap_or(PROBE);
        norm_doc(api)
    }

    #[test]
    fn r2_lower_bound_label_tripwire() {
        const MOD: &str = include_str!("mod.rs");
        let probe = api_doc_region();
        let r#mod = norm_doc(MOD);
        // (a) The label is carried everywhere R2 appears: the const line,
        //     both builder docs, and the ReferenceBands::r2 field doc here;
        //     the module docs carry it too.
        assert_eq!(R2_LABEL, "single-step lower bound");
        assert!(
            probe.matches(R2_LABEL).count() >= 4,
            "R2 lost its single-step-lower-bound labeling"
        );
        assert!(
            r#mod.contains(R2_LABEL),
            "module docs lost the R2 lower-bound label"
        );
        // (b) Behavior matches the label: the R2 band is EXACTLY ONE
        //     quantize→dequant round-trip. If this reds because someone
        //     composed more steps into the builder, relabel the API — do
        //     not delete the assertion.
        let baseline = lcg_f32s(0xBEEF, 256);
        let band = reference_r2_roundtrip(&baseline, 13).unwrap();
        let manual: Vec<f32> = baseline
            .iter()
            .map(|&v| truncate_mantissa(v as f64, 13) as f32)
            .collect();
        let expected = DeviationReport::compute(&baseline, &manual).unwrap();
        assert_eq!(
            band.bit_identity(),
            expected.bit_identity(),
            "R2 must be a SINGLE round-trip"
        );
        // (c) The label's warning is real: a composed pipeline (amplify
        //     between quantization events — two "layers") exceeds the
        //     single-step band. Deterministic demonstration ladder.
        let single_md = band.max_diff;
        let mut exceeded = 0;
        for step in 1..=8u32 {
            let c = 1.0 + f64::from(step) * 0.05;
            let composed: Vec<f32> = baseline
                .iter()
                .map(|&v| truncate_mantissa(truncate_mantissa(v as f64, 13) * c, 13) as f32)
                .collect();
            let rep = DeviationReport::compute(&baseline, &composed).unwrap();
            if rep.max_diff > single_md {
                exceeded += 1;
            }
        }
        assert!(
            exceeded >= 1,
            "expected a composed pipeline to exceed the single-step band \
             (the reason the lower-bound label exists)"
        );
    }

    #[test]
    fn scope_limit_tripwire() {
        const MOD: &str = include_str!("mod.rs");
        let probe = api_doc_region();
        let r#mod = norm_doc(MOD);
        // The scope-limit sentence must survive verbatim (T1.5): the
        // protocol bounds divergence similarity, NOT training stability.
        assert!(
            r#mod.contains(
                "this protocol bounds DIVERGENCE SIMILARITY only — it is NOT a \
                 training-stability proof"
            ),
            "the scope-limit sentence was weakened or removed"
        );
        assert!(
            r#mod.contains("2510.04212"),
            "the stability-mechanism owner (arXiv:2510.04212) must stay cited"
        );
        // And the docs must never CLAIM stability (positive claims would
        // contradict the paper's explicit anti-claim).
        assert!(!r#mod.contains("guarantees training stability"));
        assert!(!r#mod.contains("ensures stable training"));
        assert!(!r#mod.contains("is a training-stability proof"));
        assert!(
            !probe.contains("stability proof"),
            "probe docs must not claim a stability proof"
        );
    }

    #[test]
    fn margin_has_no_default_doc_truth() {
        const MOD: &str = include_str!("mod.rs");
        let probe = api_doc_region();
        let r#mod = norm_doc(MOD);
        // The rule (T1.3 doc-truth): the margin is an explicit parameter
        // with no default, and the paper's context-specific headline is
        // recorded as a footnote, never wired in.
        assert!(
            r#mod.contains("no default margin"),
            "module docs must state the no-default-margin rule"
        );
        assert!(
            probe.contains("no default margin"),
            "accept() docs must state the no-default-margin rule"
        );
        assert!(
            r#mod.contains("2–5"),
            "module docs must mark the paper's 2–5x headline as context-specific"
        );
        assert!(
            !probe.contains("DEFAULT_MARGIN"),
            "a default-margin constant would contradict the no-default rule"
        );
        assert!(!r#mod.contains("DEFAULT_MARGIN"));
    }
}
