//! Issue 696 T1 — the DT2 **anti-common-mode scalar gate** (Research 433 /
//! riir-train, arXiv:2608.23664 "RVM: Scaling RL for Diffusion Models via
//! Velocity Matching", Choi et al. 2026, §C.3 DT/DT2 reward construction).
//!
//! A scalar signal that resists capture by the **population-dominant
//! degenerate mode** (the paper's reward hack: a "clean but static" video
//! whose frame-wide background motion dominates the mean). Four composable
//! closed-form steps, all modelless:
//!
//! ```text
//! (a) peak-quantile statistic  p  = mean of the fastest 5% of the population
//!                                 (select_nth_unstable_by partition at the
//!                                  P95 boundary, total_cmp ordering)
//! (b) context-scaled threshold τ  = TAU_GAIN·context   (paper: 6·min(H,W)/256
//!                                  — the caller folds min(H,W)/256 in)
//! (c) median subtraction       m' = p − median(population)
//!                                  — cancels the common mode EXACTLY (same
//!                                  bits), robust to the outliers being the
//!                                  thing detected
//! (d) band window              s  = clip((m'−τlo)/(τmid−τlo))
//!                                  · clip((τhi−m')/(τhi−τmid))
//!                                  — both extremes → 0, only the genuine
//!                                  middle band earns signal
//! ```
//!
//! The named consumer is the CLR crowd-panic re-enable (T3): per-NPC threat
//! exposure is the population, the zone median is the common mode (one
//! monster currently panics the entire swarm — riir-mmorpg-examples Plan
//! 019 demotion), and the band window keeps the genuine minority-exposure
//! middle from over-driving.
//!
//! # Honest deviations / parameterizations (doc-truth)
//!
//! - **`&mut [f32]` contract, not `&[f32]`.** Exact selection needs
//!   `select_nth_unstable_by`, which permutes the buffer in place. The
//!   permutation preserves the multiset (every step is a partition, never a
//!   value mutation); a caller-owned scratch copy (`*_into` variant) would
//!   double the buffer for no benefit. **Callers must treat `values` as
//!   reordered-but-equal after the call.**
//! - **Peak estimator confirmed against Research 433:** "mean magnitude of
//!   the *fastest 5%* of pixels" — implemented as the mean of the top
//!   `k = max(1, ceil(PEAK_FRAC·N))` values (the P95 boundary element
//!   itself is included; the pivot of the `select_nth_unstable_by(n−k)`
//!   partition is the k-th largest value).
//! - **Band edges are OUR parameterization.** The paper gives the band
//!   form but not its edges; DT saturates at `m = τ` (`min(m/τ, 1)`), so
//!   `τmid = τ` by construction; `τlo = τ/2` and `τhi = 2τ` are chosen
//!   constants ([`BAND_LO_FRAC`]/[`BAND_HI_FRAC`]) — tune per consumer,
//!   they are configuration, not paper-derived.
//! - **NaN / non-finite contract.** Partitioning uses `f32::total_cmp`
//!   (IEEE 754 totalOrder — deterministic, NaN sorts above +∞), so a
//!   NaN-poisoned population cannot panic or reorder nondeterministically;
//!   any non-finite OUTPUT statistic is refused as `0.0` (no admissible
//!   signal). A poisoned population is a caller bug — garbage in, zero out,
//!   loudly documented here.
//! - **Float cancellation is bit-exact only in narrow regimes.** Two
//!   measured limits (riir-ai Bench 794 §Findings, the CLR consumer PoC):
//!   (1) a constant population cancels to `m' = 0.0` with identical bits
//!   only when the top-5% mean accumulates EXACTLY (dyadic-friendly
//!   constants like 95); for a constant like `0.7f32` the accumulation
//!   drifts ~1 ulp, so peak − median ≈ −8e-8 ≠ 0.0 bit-wise — the gate is
//!   unaffected (the degenerate dynamic range closes the band
//!   independently of `m'`), but read "statistically common-mode-free",
//!   not "bit-free", for non-dyadic constants. (2) Shifting
//!   `[10×95, 14×5]` by `+1000` keeps `m' = 4.0` bit-exact (Sterbenz —
//!   exactly-representable shifts), but an arbitrary real-world shift
//!   carries rounding.
//! - **Composition note for per-element consumers** (Bench 794 §Findings 2):
//!   under the context-scaled threshold `τ = dynamic range`, the per-element
//!   band window's falling edge is structurally unreachable —
//!   `m'_i ≤ max − median ≤ range = τmid`, so the per-element window is a
//!   pure median-relative rising ramp and the band's refusal edge lives in
//!   the population statistic ([`score`]). A per-element consumer that wants
//!   a genuine hi-edge must derive τ from a scale LARGER than the observed
//!   range (e.g. the theoretical max at current density), at the cost of a
//!   config constant.
//!
//! # Cost
//!
//! Two O(N) selection passes (peak partition + median partition) + O(N)
//! scans; zero allocation end-to-end. Measured 6.0 µs at N=1000 in release
//! (6023 ns/call, M3 Max — see the G2 gate + `.benchmarks/688`) — the
//! issue's "sub-µs at N=1000" estimate assumed one partition; exact median
//! + exact top-5% need two, so sub-µs holds at N ≲ 160 (~6 ns/element).
//!
//! # Domain classification
//!
//! Latent, local, never synced: a scalar gate over caller-owned population
//! state. No sync dependency, no replay coupling, no chain surface.
//!
//! Feature: `anti_common_mode` (opt-in POC). **Promotion ONLY via the T3
//! consumer PoC** (CLR crowd-panic re-enable: border-band occupancy at
//! baseline AND Bench 010's distributed-threat detection retained) — the
//! primitive exists to kill a measured failure, and its value claim IS
//! that PoC. Until T3 passes this is an unproven extraction, not a GOAT.

/// The peak-quantile fraction: summarize by the fastest [`PEAK_FRAC`] of the
/// population (paper: the fastest 5% of pixels).
pub const PEAK_FRAC: f32 = 0.05;

/// The context-threshold gain: `τ = TAU_GAIN · context_scale` (paper:
/// `τ = 6·min(H,W)/256` — pass `context_scale = min(H,W)/256`).
pub const TAU_GAIN: f32 = 6.0;

/// Band rising-edge start: `τlo = BAND_LO_FRAC · τ` (ours — see the module
/// doc's parameterization note).
pub const BAND_LO_FRAC: f32 = 0.5;
/// Band peak: `τmid = BAND_MID_FRAC · τ` — the paper's DT saturation point
/// (`min(m/τ, 1)` saturates at `m = τ`), so this one IS paper-derived.
pub const BAND_MID_FRAC: f32 = 1.0;
/// Band falling-edge end: `τhi = BAND_HI_FRAC · τ` (ours — see the module
/// doc's parameterization note).
pub const BAND_HI_FRAC: f32 = 2.0;

/// Peak-quantile statistic: the mean of the top
/// `k = max(1, ceil(PEAK_FRAC·N))` values of the population.
///
/// Permutes `values` in place (multiset preserved — see the module doc's
/// `&mut` contract). Empty input → `0.0`. Non-finite output (only possible
/// from a non-finite population) → `0.0`.
#[must_use]
pub fn peak_quantile(values: &mut [f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    // ceil: never drop below one element, never exceed N (a NaN-poisoned
    // ceil result is clamped to N by the `.min(n as f32)` guard below).
    let k_raw = (n as f32 * PEAK_FRAC).ceil();
    let k = if k_raw.is_finite() {
        (k_raw.max(1.0).min(n as f32)) as usize
    } else {
        n
    };
    // Ascending partition at the (n−k)-th order statistic: the tail
    // `values[n−k..]` is exactly the k largest values (pivot included),
    // unsorted — averaging needs no order.
    values.select_nth_unstable_by(n - k, |a, b| a.total_cmp(b));
    let sum: f32 = values[n - k..].iter().sum();
    finish_finite(sum / k as f32)
}

/// Median of the population (exact for both parities: the average of the
/// two middle order statistics when N is even).
///
/// Permutes `values` in place (multiset preserved). Empty input → `0.0`.
#[must_use]
pub fn median(values: &mut [f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        let (.., mid, _) = values.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
        finish_finite(*mid)
    } else {
        let (.., lo, _) = values.select_nth_unstable_by(n / 2 - 1, |a, b| a.total_cmp(b));
        let lo = *lo;
        // The tail is partitioned ≥ lo but unsorted — scan for the upper
        // middle element (N/2 more comparisons, still O(N)).
        let hi = values[n / 2..]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        finish_finite((lo + hi) * 0.5)
    }
}

/// Context-scaled threshold: `τ = TAU_GAIN · context_scale` (paper:
/// `6·min(H,W)/256`; pass `context_scale = min(H,W)/256` to reproduce it).
/// Negative / non-finite context → `0.0` (a degenerate threshold yields a
/// degenerate band downstream, by design).
#[must_use]
pub fn context_threshold(context_scale: f32) -> f32 {
    if !context_scale.is_finite() || context_scale <= 0.0 {
        return 0.0;
    }
    TAU_GAIN * context_scale
}

/// Band-window thresholds derived from a context scale, with the module's
/// documented edge fractions. `from_context(0.0)` (or negative / non-finite)
/// produces a degenerate band that admits nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandThresholds {
    /// Rising edge start (`BAND_LO_FRAC·τ`).
    pub tau_lo: f32,
    /// Band peak (`BAND_MID_FRAC·τ`).
    pub tau_mid: f32,
    /// Falling edge end (`BAND_HI_FRAC·τ`).
    pub tau_hi: f32,
}

impl BandThresholds {
    /// Paper-shaped thresholds from a caller context scale.
    #[must_use]
    pub fn from_context(context_scale: f32) -> Self {
        let tau = context_threshold(context_scale);
        Self {
            tau_lo: BAND_LO_FRAC * tau,
            tau_mid: BAND_MID_FRAC * tau,
            tau_hi: BAND_HI_FRAC * tau,
        }
    }

    /// Thresholds from an explicit τ (context_threshold applied already).
    #[must_use]
    pub fn from_tau(tau: f32) -> Self {
        Self {
            tau_lo: BAND_LO_FRAC * tau,
            tau_mid: BAND_MID_FRAC * tau,
            tau_hi: BAND_HI_FRAC * tau,
        }
    }
}

/// Band window (step d): `clip((m−τlo)/(τmid−τlo)) · clip((τhi−m)/(τhi−τmid))`.
///
/// Both extremes score `0.0`; only the genuine middle band earns signal
/// (peak `1.0` at `m = τmid`). A degenerate band (`τlo < τmid < τhi`
/// violated — including the all-zero band from a zero context) admits
/// nothing: returns `0.0`. Non-finite `m` → `0.0`.
#[must_use]
pub fn band_window(m: f32, tau_lo: f32, tau_mid: f32, tau_hi: f32) -> f32 {
    if !(tau_lo < tau_mid && tau_mid < tau_hi && m.is_finite()) {
        return 0.0;
    }
    let rise = ((m - tau_lo) / (tau_mid - tau_lo)).clamp(0.0, 1.0);
    let fall = ((tau_hi - m) / (tau_hi - tau_mid)).clamp(0.0, 1.0);
    rise * fall
}

/// The composite DT2-shaped gate over a population (steps a+c+d, with the
/// band derived from step b):
///
/// `score = band_window(peak_quantile(pop) − median(pop), from_context(ctx))`
///
/// - `values` is permuted in place (multiset preserved — the `&mut`
///   contract documented at the module head).
/// - `context_scale` is the observable's dynamic-range scale (paper:
///   `min(H,W)/256`); `τ = 6·context_scale`.
/// - Constant / empty / single-element populations cancel to `m' = 0` and
///   score `0.0` for every context — the anti-common-mode property.
/// - Zero allocation end-to-end.
#[must_use]
pub fn score(values: &mut [f32], context_scale: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let peak = peak_quantile(values);
    let med = median(values);
    let band = BandThresholds::from_context(context_scale);
    band_window(peak - med, band.tau_lo, band.tau_mid, band.tau_hi)
}

/// Refuse non-finite statistics as `0.0` (the module's NaN contract).
#[inline]
fn finish_finite(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    /// 95 quiet members at `quiet` + 5 active outliers at `active` — the
    /// minority-active distribution the paper's reward hack lives in.
    fn population(quiet: f32, active: f32) -> Vec<f32> {
        let mut v = vec![quiet; 95];
        v.extend_from_slice(&[active; 5]);
        v
    }

    #[test]
    fn g1_peak_quantile_is_top_five_percent_mean() {
        // N=100: k = ceil(5.0) = 5 → the top-5 mean, exactly.
        let mut v = population(10.0, 14.0);
        let p = peak_quantile(&mut v);
        assert_eq!(p, 14.0, "top-5% mean of 14.0×5 over 10.0×95 is 14.0");
        // Multiset preserved through the permutation.
        v.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(&v[..95], &[10.0; 95]);
        assert_eq!(&v[95..], &[14.0; 5]);
        // N=1000 mixed magnitudes: the top 50 dominate the statistic.
        let mut big: Vec<f32> = (0..950).map(|i| 1.0 + i as f32 * 1e-3).collect();
        big.extend((0..50).map(|i| 100.0 + i as f32));
        let p2 = peak_quantile(&mut big);
        assert!((p2 - 124.5).abs() < 1e-3, "top-50 mean ≈ 124.5, got {p2}");
    }

    #[test]
    fn g1_peak_quantile_differs_from_mean_on_minority_active() {
        // The load-bearing estimator contrast: the minority (5%) owns the
        // peak statistic while the mean is dominated by the quiet majority.
        let mut v = population(0.0, 100.0);
        let p = peak_quantile(&mut v);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        assert_eq!(p, 100.0);
        assert!((mean - 5.0).abs() < 1e-4);
        assert!(
            (p - mean).abs() > 90.0,
            "peak-quantile ({p}) must not collapse toward the mean ({mean})"
        );
    }

    #[test]
    fn g1_median_cancels_population_dominant_mode_exactly() {
        // (1) Constant population: peak == median bit-for-bit, m' == +0.0
        //     with identical bits, score 0 — for ANY common mode value c.
        for &c in &[0.0f32, 10.0, 1e6, -3.5] {
            let mut v = vec![c; 100];
            let p = peak_quantile(&mut v);
            let m = median(&mut v);
            assert_eq!(p.to_bits(), m.to_bits(), "constant {c}: peak==median");
            assert_eq!((p - m).to_bits(), 0.0f32.to_bits(), "m' == +0.0");
            assert_eq!(score(&mut vec![c; 100], 1.0), 0.0);
        }
        // (2) Exactly-representable shift (Sterbenz case): m' is invariant
        //     under a +1000 common-mode shift, bit-exact.
        let mut v0 = population(10.0, 14.0);
        let m0 = peak_quantile(&mut v0) - median(&mut v0);
        let shifted: Vec<f32> = population(10.0, 14.0)
            .into_iter()
            .map(|x| x + 1000.0)
            .collect();
        let mut v1 = shifted;
        let m1 = peak_quantile(&mut v1) - median(&mut v1);
        assert_eq!(m0, 4.0, "peak 14 − median 10");
        assert_eq!(
            m1.to_bits(),
            m0.to_bits(),
            "+1000 common mode cancels bit-exact"
        );
        // (3) Even-N median is the exact two-middle average; odd-N is the
        //     middle order statistic.
        let mut even = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(median(&mut even), 2.5);
        let mut odd = vec![1.0f32, 2.0, 10.0];
        assert_eq!(median(&mut odd), 2.0);
    }

    #[test]
    fn g1_band_tails_score_zero() {
        let t = BandThresholds::from_tau(4.0); // lo 2 / mid 4 / hi 8
        // Below the rising edge and above the falling edge: exactly 0.
        assert_eq!(
            band_window(t.tau_lo - 1.0, t.tau_lo, t.tau_mid, t.tau_hi),
            0.0
        );
        assert_eq!(
            band_window(t.tau_hi + 1.0, t.tau_lo, t.tau_mid, t.tau_hi),
            0.0
        );
        assert_eq!(band_window(-100.0, t.tau_lo, t.tau_mid, t.tau_hi), 0.0);
        assert_eq!(band_window(100.0, t.tau_lo, t.tau_mid, t.tau_hi), 0.0);
        // Both tails through the full score: populations whose m' sits in
        // either tail score 0 while a mid-band population earns signal.
        // τ = 6·1 = 6 → band 3..12, peak at 6.
        let mut tail_low = vec![20.0f32; 90];
        tail_low.extend_from_slice(&[20.5; 10]); // m' = 0.5 < τlo = 3 → 0
        assert_eq!(score(&mut tail_low, 1.0), 0.0);
        let mut tail_high = vec![0.0f32; 90];
        tail_high.extend_from_slice(&[500.0; 10]); // m' = 500 > τhi = 12 → 0
        assert_eq!(score(&mut tail_high, 1.0), 0.0);
        let mut mid = vec![10.0f32; 90];
        mid.extend_from_slice(&[16.0; 10]); // m' = 6 = τmid → 1
        assert_eq!(score(&mut mid, 1.0), 1.0);
    }

    #[test]
    fn g1_band_window_shape_rises_then_falls() {
        let (lo, mid, hi) = (2.0f32, 4.0, 8.0);
        assert_eq!(band_window(lo, lo, mid, hi), 0.0);
        assert!((band_window(3.0, lo, mid, hi) - 0.5).abs() < 1e-6);
        assert!((band_window(mid, lo, mid, hi) - 1.0).abs() < 1e-6);
        assert!((band_window(6.0, lo, mid, hi) - 0.5).abs() < 1e-6);
        assert_eq!(band_window(hi, lo, mid, hi), 0.0);
        // Degenerate bands admit nothing (including the zero-τ band).
        assert_eq!(band_window(1.0, 0.0, 0.0, 0.0), 0.0);
        assert_eq!(band_window(1.0, 4.0, 2.0, 8.0), 0.0);
        assert_eq!(band_window(f32::NAN, lo, mid, hi), 0.0);
    }

    #[test]
    fn g1_context_threshold_paper_parameterization() {
        // Paper: τ = 6·min(H,W)/256 — fold min(H,W)/256 into context_scale.
        assert!((context_threshold(640.0 / 256.0) - 15.0).abs() < 1e-4);
        assert_eq!(context_threshold(0.0), 0.0);
        assert_eq!(context_threshold(-1.0), 0.0);
        assert_eq!(context_threshold(f32::NAN), 0.0);
        // Zero context → degenerate band → the gate refuses everything.
        assert_eq!(score(&mut population(0.0, 100.0), 0.0), 0.0);
    }

    #[test]
    fn g1_empty_and_degenerate_refusals() {
        assert_eq!(peak_quantile(&mut []), 0.0);
        assert_eq!(median(&mut []), 0.0);
        assert_eq!(score(&mut [], 1.0), 0.0);
        // Single element: no population contrast → m' = 0 → 0.
        assert_eq!(score(&mut [42.0f32], 1.0), 0.0);
        // Two elements: median = mean, peak = max → m' = |a−b|/2.
        let mut two = vec![2.0f32, 6.0];
        assert_eq!(median(&mut two), 4.0);
        assert!((peak_quantile(&mut two) - 6.0).abs() < 1e-6);
        // NaN-poisoned population: deterministic (total_cmp), refused to 0.
        let mut poisoned = vec![f32::NAN; 50];
        poisoned.extend_from_slice(&[1.0f32; 50]);
        let s = score(&mut poisoned, 1.0);
        assert!(s.is_finite(), "NaN-poisoned population must not leak NaN");
        assert_eq!(s, 0.0, "poisoned peak inflates m' past the band → tail → 0");
    }

    #[test]
    fn g1_statistic_is_permutation_invariant_and_multiset_preserving() {
        // The &mut contract: same multiset in, same gate out — the
        // permutation must not change what the gate measures.
        let v0 = population(3.0, 9.0);
        let mut a = v0.clone();
        let s1 = score(&mut a, 1.0);
        let s2 = score(&mut a, 1.0); // re-scoring an already-permuted buffer
        assert_eq!(s1, s2, "permutation-invariant statistic");
        // The buffer is reordered-but-equal: sorting it recovers the
        // original multiset exactly.
        let mut expect = v0;
        expect.sort_by(|x, y| x.total_cmp(y));
        a.sort_by(|x, y| x.total_cmp(y));
        assert_eq!(a, expect, "multiset preserved through the permutation");
    }

    #[cfg_attr(debug_assertions, ignore = "timing gate — release-only")]
    #[test]
    fn g2_score_under_budget_at_n1000() {
        // The issue asked sub-µs at N=1000 assuming one O(N) pass; the exact
        // estimator needs TWO select_nth_unstable passes (top-5% + median).
        // The pinned budget below is the regression floor at ~2× the
        // measured number (see .benchmarks/688 §G2 for the honest record vs
        // the issue's sub-µs ask) — NOT the sub-µs ask itself.
        const RUNS: u32 = 2_000;

        let mut v: Vec<f32> = (0..950).map(|i| (i % 97) as f32).collect();
        v.extend((0..50).map(|i| 40.0 + i as f32 * 0.25));
        let mut warm = v.clone();
        // Warm up caches / branch predictors.
        for _ in 0..100 {
            let s = score(&mut warm, 1.0);
            std::hint::black_box(s);
            warm.copy_from_slice(&v);
        }
        let t0 = std::time::Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..RUNS {
            let s = score(&mut warm, 1.0);
            acc += s;
            warm.copy_from_slice(&v);
        }
        let dt = t0.elapsed();
        let per = dt.as_nanos() as f64 / RUNS as f64;
        std::eprintln!("g2 score N=1000: {per:.0} ns/call (acc {acc})");
        assert!(acc.is_finite());
        assert!(
            per <= 12_000.0,
            "{per:.0} ns/call > 12 µs regression floor @ N=1000"
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    fn g4_alloc_free_scored_path() {
        // The lib test binary installs TrackingAllocator via
        // TEST_GLOBAL_ALLOC (crate::alloc) — the scored path allocates
        // nothing end-to-end (partitions + sums only). Construction (the
        // Vec itself) sits outside the measured region.
        let mut v: Vec<f32> = (0..1000).map(|i| (i % 13) as f32).collect();
        crate::alloc::reset_alloc_stats();
        let s = score(&mut v, 1.0);
        let p = peak_quantile(&mut v);
        let m = median(&mut v);
        let b = band_window(3.0, 1.0, 3.0, 9.0);
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(
            count, 0,
            "scored path must be alloc-free (s={s} p={p} m={m} b={b})"
        );
    }
}
