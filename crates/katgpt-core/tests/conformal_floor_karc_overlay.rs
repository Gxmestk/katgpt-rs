//! Issue 010 T7 — "Report the Floor" comparison for the KARC + conformal overlay
//! (Plan 308 + Plan 340 Phase 2).
//!
//! The KARC + conformal overlay is the composition
//! `ConformalIntervalCalibrator<KarcChannelForecaster<..>>`. It produces
//! coverage-guaranteed predictive intervals by feeding KARC's point forecast
//! into the conformal residual-pool machinery. This is the canonical
//! "UQ-bearing primitive" pattern documented in
//! `examples/conformal_karc_overlay.rs` + `conformal::karc_adapter`.
//!
//! ## Why this test exists
//!
//! Per AGENTS.md "Feature Flag Discipline" / Issue 010's "Report the Floor"
//! rule, every UQ-bearing primitive must benchmark against the
//! conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`
//! with `m=1`) on CRPS / coverage / Winkler. The other three grandfathered
//! UQ primitives — BoMSampler (T3), Sleep-Time Anticipator (T4), Best-Belief
//! Beta Selector (T5) — each shipped a floor test in the Issue 010 cycle.
//! KARC+overlay was the remaining grandfathered primitive without one. This
//! test closes that gap.
//!
//! ## Comparison angle
//!
//! Unlike BoM (where σ controls width, not data → EXCLUDED as UQ), KARC+overlay
//! is structurally similar to the floor: both wrap a point forecaster in the
//! SAME conformal math. The only difference is the wrapped forecaster
//! (KARC's reservoir vs. seasonal-naive's last-observation anchor). So the
//! question reduces to: **does KARC's point forecast produce tighter
//! conformal intervals than seasonal-naive's, at the same coverage?**
//!
//! ## Honest verdict (recorded from the canonical run)
//!
//! | Corpus | CRPS ratio | Winkler ratio | Coverage (nom 0.95) | Verdict |
//! |---|---|---|---|---|
//! | stationary_seasonal m=12, σ=0.5 | 5.74 | 21.73 | 0.916 vs 0.939 | **LosesToFloor** |
//! | Lorenz-x dt=0.02 (chaotic) | 0.0047 | 0.0059 | 0.932 vs 0.943 | **BeatsFloor** |
//!
//! **KARC+overlay is a chaotic-regime specialist, not a universal UQ
//! improvement.** On stationary seasonal data (period 12) with K=4 delay
//! embedding, KARC's reservoir cannot capture the period-12 cycle and
//! produces WORSE point forecasts than the trivial last-value anchor.
//! The conformal overlay faithfully calibrates around the worse point →
//! ~5.7× wider intervals at slightly lower coverage. On Lorenz-63
//! (KARC's design target), KARC's delay embedding captures the chaos and
//! produces ~210× tighter intervals than the floor at near-identical
//! coverage.
//!
//! The take-away for the "Report the Floor" policy: KARC+overlay is
//! **IN-SCOPE** as a UQ primitive (it produces calibrated intervals, the
//! coverage doesn't break) but **SCOPE-LIMITED** to regimes where KARC's
//! delay embedding can capture the signal's structure. The Plan 308 GOAT
//! gate (chaotic Lorenz double-scroll) remains the right canonical
//! validation target; stationary-seasonal is explicitly out-of-scope.
//!
//! ## Method
//!
//! The adapter `KarcOverlayAdapter` holds ONE pre-fitted KARC forecaster
//! (D=1, M=8, K=4) wrapped in a `ConformalIntervalCalibrator<
//! KarcChannelForecaster<..>>`. KARC is fit once on the warmup corpus slice
//! (the same data the floor's residual pool sees during warmup) — the
//! comparison is honest: both sides get the same warmup.
//!
//! Per tick:
//! 1. Build the delay state from the adapter's recent-observation window.
//! 2. KARC forecasts the next point.
//! 3. The calibrator reads the conformal interval around that point
//!    (`interval_from_point_into` — the documented KARC pattern).
//! 4. After the ground truth is revealed, `update_residual` pushes the
//!    residual and `step()` advances the calibrator's tick.
//!
//! ## Run
//!
//! ```bash
//! cargo test -p katgpt-core --test conformal_floor_karc_overlay \
//!   --features conformal_predictive_intervals,karc_forecaster -- --nocapture
//! ```

#![cfg(all(feature = "conformal_predictive_intervals", feature = "karc_forecaster"))]
#![allow(clippy::needless_range_loop)]

use katgpt_core::{
    ChebyshevBasis, ConformalIntervalCalibrator, DecayUnit, FloorComparisonReport, KarcChannelForecaster,
    KarcForecaster, OverallVerdict, PointForecaster, PredictiveInterval, PredictiveOutput, ResidualMode,
    TrajectoryCorpus, UqPrimitiveUnderTest, run_floor_comparison,
};

// ── KARC shape ────────────────────────────────────────────────────────────
// D=1 (scalar trajectory — matches the floor's 1-channel layout exactly),
// M=8 (Chebyshev features per coordinate — enough expressivity to capture
// Lorenz-x's smooth-but-chaotic dynamics), K=4 (delay-embedding depth).
// d_h = K·D·M = 32 features. Small enough to fit fast in-test; deep enough
// to expose KARC's reservoir advantage over the last-value anchor.
const D: usize = 1;
const M: usize = 8;
const K: usize = 4;

/// Capacity of the conformal residual ring buffer. Matches the canonical
/// floor config (FLOOR_CAPACITY = 256).
const POOL_CAPACITY: usize = 256;

/// α for the conformal interval (95% coverage).
const ALPHA: f32 = 0.05;

// ===== Lorenz-63 (deterministic f64 trajectory generator) ==================
//
// Used to construct a chaotic 1D corpus the floor cannot easily forecast
// (its anchor = last value misses the curvature). KARC's delay embedding
// should capture the structure → smaller residuals → tighter intervals.

const LORENZ_SIGMA: f64 = 10.0;
const LORENZ_RHO: f64 = 28.0;
const LORENZ_BETA: f64 = 8.0 / 3.0;

#[inline]
fn lorenz_rhs(state: &[f64; 3], out: &mut [f64; 3]) {
    let (x, y, z) = (state[0], state[1], state[2]);
    out[0] = LORENZ_SIGMA * (y - x);
    out[1] = x * (LORENZ_RHO - z) - y;
    out[2] = x * y - LORENZ_BETA * z;
}

fn rk4_step(state: &mut [f64; 3], dt: f64) {
    let mut k1 = [0.0; 3];
    let mut k2 = [0.0; 3];
    let mut k3 = [0.0; 3];
    let mut k4 = [0.0; 3];
    let mut tmp = [0.0; 3];

    lorenz_rhs(state, &mut k1);
    for j in 0..3 {
        tmp[j] = state[j] + 0.5 * dt * k1[j];
    }
    lorenz_rhs(&tmp, &mut k2);
    for j in 0..3 {
        tmp[j] = state[j] + 0.5 * dt * k2[j];
    }
    lorenz_rhs(&tmp, &mut k3);
    for j in 0..3 {
        tmp[j] = state[j] + dt * k3[j];
    }
    lorenz_rhs(&tmp, &mut k4);
    for j in 0..3 {
        state[j] += dt / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
    }
}

/// Generate `n` samples of the Lorenz-63 x-coordinate after discarding
/// `n_transient` transient samples. The sampling interval dt=0.02 is coarse
/// enough to keep consecutive samples decorrelated (well-conditioned Gram).
fn generate_lorenz_x(n_transient: usize, n: usize, dt: f64) -> Vec<f32> {
    let mut state = [0.1_f64, 0.0, 0.0];
    for _ in 0..n_transient {
        rk4_step(&mut state, dt);
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        rk4_step(&mut state, dt);
        out.push(state[0] as f32);
    }
    out
}

// ===== KarcOverlayAdapter ==================================================

/// Adapter wrapping a pre-fitted KARC + conformal overlay as a
/// `UqPrimitiveUnderTest`. Mirrors the production integration pattern
/// documented in `conformal::karc_adapter`: `interval_from_point_into`
/// (point-supplied read path), `update_residual` + `step` (write path).
pub struct KarcOverlayAdapter {
    /// KARC wrapped in the conformal calibrator. The calibrator's wrapped
    /// forecaster is `KarcChannelForecaster<..>` (channel 0 of D=1).
    calibrator: ConformalIntervalCalibrator<KarcChannelForecaster<ChebyshevBasis<M>, D, M, K>>,
    /// Recent-observation window: `window[0]` = y_{t−1} (most recent),
    /// `window[K-1]` = y_{t−K}. Used to build the K·D delay state.
    window: [f32; K],
    /// Pre-allocated delay-state buffer (length K·D). Reused per forecast —
    /// zero allocations on the hot path.
    delay_state: Vec<f32>,
    /// Last point forecast KARC produced (used for residual update).
    last_point: f32,
    /// α for the conformal interval.
    alpha: f32,
    /// Whether the adapter has seen enough observations to fill the delay
    /// window. Before this, the interval is undefined → adapter emits a
    /// wide placeholder (harness skips warmup ticks anyway).
    warmed_up: bool,
    /// Number of observations seen.
    n_seen: usize,
}

impl KarcOverlayAdapter {
    /// Construct an adapter with a PRE-FITTED KARC forecaster.
    ///
    /// The KARC is fit on the `warmup_corpus` slice (the same observations
    /// the harness will feed to `observe()` during warmup). This is the
    /// honest comparison: both KARC and the floor see the same warmup data.
    /// The alternative (fit incrementally as observations arrive) is also
    /// valid but more complex and not what production callers do — they fit
    /// once on a training slice and use the fitted model in production.
    pub fn new_fitted(warmup_corpus: &[f32], alpha: f32) -> Self {
        let basis = ChebyshevBasis::<M>::new();
        let mut karc = KarcForecaster::<ChebyshevBasis<M>, D, M, K>::with_capacity(basis, warmup_corpus.len());

        // Build training pairs: for each t ∈ [K-1, n-1], delay_state =
        // [y_t, y_{t-1}, ..., y_{t-K+1}] flattened (K·D values, here D=1 so
        // just K values), target = y_{t+1}.
        if warmup_corpus.len() > K {
            for t in (K - 1)..(warmup_corpus.len() - 1) {
                let mut ds = [0.0_f32; K * D];
                for k in 0..K {
                    ds[k * D..(k + 1) * D].copy_from_slice(&warmup_corpus[t - k..t - k + 1]);
                }
                let target = [warmup_corpus[t + 1]];
                karc.accumulate_pair(&ds, &target);
            }
        }
        // λ=1e-3 matches the example; the Lorenz-x Gram is well-conditioned
        // at dt=0.02 with this regularization. Fit failure → the adapter
        // still constructs but forecasts return 0.0 (KARC's "not fitted"
        // fallback). The test would then fail loudly on the first assertion.
        let _ = karc.fit_ridge(1e-3);

        let adapter = KarcChannelForecaster::new(karc, 0);
        let calibrator = ConformalIntervalCalibrator::new(
            adapter,
            1,                // n_channels (matches D=1)
            1,                // max_h (KARC is h=1)
            1,                // m=1 (non-seasonal — matches the floor)
            POOL_CAPACITY,
            0.0,              // exp_lambda=0 — equal weight, matches floor
            DecayUnit::Step,
            ResidualMode::HStep,
            false,            // orientation
        );

        Self {
            calibrator,
            window: [0.0; K],
            delay_state: vec![0.0_f32; K * D],
            last_point: 0.0,
            alpha,
            warmed_up: false,
            n_seen: 0,
        }
    }

    /// Shift the window left and append `y` at position 0.
    #[inline]
    fn push_observation(&mut self, y: f32) {
        for i in (1..K).rev() {
            self.window[i] = self.window[i - 1];
        }
        self.window[0] = y;
        self.n_seen += 1;
        if self.n_seen >= K {
            self.warmed_up = true;
        }
    }

    /// Build the delay state from the current window into `self.delay_state`.
    /// Layout: `[window[0], window[1], ..., window[K-1]]` (K values, since D=1).
    #[inline]
    fn build_delay_state(&mut self) {
        for k in 0..K {
            self.delay_state[k * D..(k + 1) * D].copy_from_slice(&self.window[k..k + 1]);
        }
    }
}

impl UqPrimitiveUnderTest for KarcOverlayAdapter {
    fn name(&self) -> &str {
        "KARC+overlay (Chebyshev M=8, K=4, D=1; pre-fitted on warmup)"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        // During warmup (not enough observations to fill the delay window),
        // emit a wide interval so the harness's scoring doesn't see garbage.
        // The harness skips warmup ticks anyway, but predict_next is called
        // once before the first scored observe — defensive.
        if !self.warmed_up {
            let wide = PredictiveInterval::new(-1e6, 0.0, 1e6, self.alpha);
            return PredictiveOutput::from_interval(wide);
        }

        // Build delay state, run KARC forecast.
        self.build_delay_state();
        let mut point = 0.0_f32;
        // KARC's PointForecaster impl via the adapter: takes &mut self,
        // writes one f32 to `point`. The wrapped KarcChannelForecaster
        // projects channel 0 of the D=1 output.
        self.calibrator
            .forecaster
            .forecast_into(&self.delay_state, 1, &mut point);
        self.last_point = point;

        // Read the conformal interval around that point. The calibrator's
        // residual pool is per-channel × per-horizon-bucket; for KARC we use
        // channel=0, h=1 (the only configured channel/horizon).
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, self.alpha);
        self.calibrator
            .interval_from_point_into(point, 0, 1, self.alpha, &mut iv);
        PredictiveOutput::from_interval(iv)
    }

    fn observe(&mut self, y: f32) {
        // Update the residual pool with the realized observation, then push
        // the observation into the delay window and advance the calibrator's
        // tick. Order matters: the residual uses `last_point` (the forecast
        // produced in the preceding predict_next), so we must update the
        // residual BEFORE pushing y into the window (which would change what
        // the NEXT forecast sees, not the residual we're scoring now).
        if self.warmed_up {
            self.calibrator
                .update_residual(y, self.last_point, 0, 1);
            self.calibrator.step();
        }
        self.push_observation(y);
    }
}

// ===== Tests ===============================================================

/// Sanity: adapter constructs + forecasts without panic after enough warmup.
#[test]
fn adapter_predicts_after_warmup() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.1, 200, 0x1234);
    let warmup = corpus.recommended_warmup;
    let mut adapter = KarcOverlayAdapter::new_fitted(&corpus.values[..warmup], ALPHA);

    // Feed the warmup observations through observe() so the window fills.
    for &y in &corpus.values[..warmup] {
        let _ = adapter.predict_next();
        adapter.observe(y);
    }

    // After warmup, predict_next should produce a finite interval.
    let out = adapter.predict_next();
    let iv = out.interval.expect("interval after warmup");
    assert!(iv.lower.is_finite(), "lower finite");
    assert!(iv.upper.is_finite(), "upper finite");
    assert!(iv.upper >= iv.lower, "upper >= lower");
}

/// Sanity: the adapter produces bit-identical intervals when run twice on the
/// same corpus. The floor comparison requires determinism.
#[test]
fn adapter_is_deterministic() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.1, 300, 0xABCD);
    let warmup = corpus.recommended_warmup;

    let run_once = || -> Vec<PredictiveInterval> {
        let mut adapter = KarcOverlayAdapter::new_fitted(&corpus.values[..warmup], ALPHA);
        for &y in &corpus.values[..warmup] {
            let _ = adapter.predict_next();
            adapter.observe(y);
        }
        let mut out = Vec::new();
        for &y in &corpus.values[warmup..] {
            let p = adapter.predict_next();
            out.push(p.interval.expect("interval"));
            adapter.observe(y);
        }
        out
    };

    let a = run_once();
    let b = run_once();
    assert_eq!(a.len(), b.len());
    for (i, (xa, xb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            (xa.lower.to_bits(), xa.upper.to_bits()),
            (xb.lower.to_bits(), xb.upper.to_bits()),
            "interval {i} differs across runs"
        );
    }
}

/// Comparison 1: stationary seasonal corpus.
///
/// The floor's seasonal-naive anchor (forecast = last value) is near-optimal
/// for a stationary seasonal signal with period m=12. KARC's delay embedding
/// (K=4) is too shallow to capture a period-12 cycle, so KARC's point
/// forecast can systematically overshoot/undershoot at curvature changes.
///
/// **Honest verdict: KARC+overlay LOSES on this corpus.** This is a real
/// finding, not a bug — KARC+overlay is a chaotic-regime specialist, not a
/// universal UQ improvement. The conformal overlay faithfully calibrates
/// around whatever point forecast KARC produces; when the point forecast is
/// worse than the floor's anchor, the overlay's intervals are wider (the
/// overlay doesn't lie about uncertainty).
///
/// Gate: we ACCEPT LosesToFloor here as a documented scope limitation.
/// The test asserts the weaker but still load-bearing properties:
/// - Coverage stays within 0.10 of the floor (no calibration break)
/// - The verdict is one of the scored outcomes (not NotApplicable)
#[test]
fn floor_comparison_stationary_seasonal() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 800, 0xCAFE_BABE);
    let warmup = corpus.recommended_warmup;

    let mut adapter = KarcOverlayAdapter::new_fitted(&corpus.values[..warmup], ALPHA);
    let report: FloorComparisonReport = run_floor_comparison(
        &mut adapter,
        &corpus.values,
        ALPHA,
        warmup,
        &corpus.name,
    );

    println!("── KARC+overlay vs floor on {} ──", corpus.name);
    println!("{:.?}", report);

    // Sanity: at least 100 scored steps.
    assert!(report.n_scored > 100, "n_scored = {}", report.n_scored);

    // Hard gate: KARC must NOT lose to the floor on coverage. The floor is
    // calibrated by construction; KARC's residual pool uses the same math.
    // A coverage regression > 0.10 means a bug in the adapter wiring (likely
    // the residual-push ordering or delay-state construction). KARC losing
    // on CRPS/Winkler is acceptable (scope limitation); losing on coverage
    // is not (would indicate broken calibration).
    let floor_cov = report.floor.coverage;
    let prim_cov = report.primitive.coverage;
    let cov_delta = (prim_cov - floor_cov).abs();
    assert!(
        cov_delta < 0.10,
        "coverage delta too large: floor={floor_cov:.4}, prim={prim_cov:.4}, |Δ|={cov_delta:.4}"
    );

    // Verdict: any of these is acceptable on seasonal. LosesToFloor is the
    // expected honest verdict (KARC's K=4 delay embedding can't capture
    // period-12). BeatsFloor would mean KARC found structure we didn't
    // expect; that's also fine. NotApplicable shouldn't happen (adapter
    // always emits intervals post-warmup).
    assert!(
        matches!(
            report.overall,
            OverallVerdict::BeatsFloor
                | OverallVerdict::TiesFloor
                | OverallVerdict::Mixed
                | OverallVerdict::LosesToFloor
        ),
        "KARC+overlay verdict on seasonal: {:?} — expected one of the scored outcomes",
        report.overall
    );
}

/// Comparison 2: Lorenz-63 x-coordinate (chaotic).
///
/// The floor's anchor (last value) cannot capture the chaotic curvature.
/// KARC's reservoir delay-embedding SHOULD produce smaller residuals →
/// tighter intervals at the same coverage. This is KARC's home turf.
///
/// Gate: KARC should BEAT the floor on CRPS and Winkler. A TIE is
/// acceptable (KARC may not have enough capacity at M=8, K=4 to fully
/// capture Lorenz). A LOSES verdict would be a significant negative
/// finding worth recording.
#[test]
fn floor_comparison_lorenz_x() {
    // Generate the chaotic trajectory + wrap as a corpus.
    let n_total = 2_000;
    let n_transient = 1_000;
    let traj = generate_lorenz_x(n_transient, n_total, 0.02);
    // Warmup: needs to be ≥ POOL_CAPACITY (256) so the residual pool fills,
    // AND large enough to fit KARC (the constructor uses corpus[..warmup]).
    let warmup = 600;
    let corpus = TrajectoryCorpus::from_slice("lorenz_x_dt0.02", &traj, warmup);

    let mut adapter = KarcOverlayAdapter::new_fitted(&corpus.values[..warmup], ALPHA);
    let report: FloorComparisonReport = run_floor_comparison(
        &mut adapter,
        &corpus.values,
        ALPHA,
        warmup,
        &corpus.name,
    );

    println!("── KARC+overlay vs floor on {} ──", corpus.name);
    println!("{:.?}", report);

    assert!(report.n_scored > 500, "n_scored = {}", report.n_scored);

    // Same coverage invariant: KARC's overlay uses the same conformal math
    // as the floor, so coverage should match within noise. A coverage gap
    // > 0.10 indicates a wiring bug.
    let floor_cov = report.floor.coverage;
    let prim_cov = report.primitive.coverage;
    let cov_delta = (prim_cov - floor_cov).abs();
    assert!(
        cov_delta < 0.10,
        "coverage delta too large: floor={floor_cov:.4}, prim={prim_cov:.4}, |Δ|={cov_delta:.4}"
    );

    // Verdict: we EXPECT BeatsFloor or at worst TiesFloor here. LOSES would
    // be a real negative finding (KARC's reservoir failing on its showcase
    // chaotic system). Mixed is acceptable if CRPS wins but Winkler ties.
    assert!(
        matches!(
            report.overall,
            OverallVerdict::BeatsFloor | OverallVerdict::TiesFloor | OverallVerdict::Mixed
        ),
        "KARC+overlay verdict on Lorenz-x: {:?} — expected Beats/Ties/Mixed (KARC's home turf)",
        report.overall
    );

    // Soft check: KARC's mean CRPS should be ≤ floor's mean CRPS + 10%.
    // (Roughly: KARC's reservoir should at least not catastrophically
    // under-perform on its canonical chaotic target.)
    let prim_crps = report.primitive.mean_crps_interval;
    let floor_crps = report.floor.mean_crps_interval;
    assert!(
        prim_crps <= floor_crps * 1.10,
        "KARC CRPS {prim_crps:.6} > floor CRPS {floor_crps:.6} × 1.10 — significant regression on Lorenz"
    );
}
