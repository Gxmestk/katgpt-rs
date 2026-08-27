//! Issue 695 — realization-gap triage primitives (Research 432 /
//! riir-train, arXiv:2608.24646 "DiffusionOPSD", Zhou et al. 2026).
//!
//! The decoupled measurement half of OPSD: what the target promises vs
//! what the fitting actually realizes, as a closed-form rate model +
//! a three-way triage. This is the instrument that diagnoses the Plan 336
//! clippy-L4 failure class (0/60 EM while loss improved — *which* half
//! was starved?).
//!
//! # The primitive (all closed-form, modelless)
//!
//! ```text
//! v_k = t + (1−η)^k·(v₀−t)                    multiplicative-step fixpoint
//! k_needed: (1−η)^k·gap ≤ tol  (closed form)  threshold budget
//! k_needed > 3/η  ⇒  RE-ANCHOR, don't iterate   (1−η)^(3/η) → e⁻³ ≈ .05
//! ρ̂(k,η,ε) = (1−(1−η)^k)·(1−cε²)             predicted realized fraction
//! triage(ρ): FittingStarved | TargetStarved | OnModel
//! ```
//!
//! # Design notes (honest scope)
//!
//! - **`c` is landscape-dependent, not universal.** The (1−cε²) factor
//!   absorbs the second-order loss a bound ε costs on a curved objective;
//!   its value depends on the curvature at the operating point in units of
//!   the promise. [`DEFAULT_C`] is a prior, NOT a calibration — consumers
//!   calibrate on frozen fixtures (the `calibration` gate demonstrates the
//!   protocol: on its fixture c* = 5.0 reproduces the observed ratio to
//!   1e-4, while DEFAULT_C does not — that gap is the protocol working).
//! - **Triage order is fit-first.** `FittingStarved` compares the
//!   observation against its own rate model (`observed < predicted·(1−tol)`)
//!   and therefore fires before the prediction-level `TargetStarved`
//!   check: an observation materially under its own model is the
//!   actionable diagnosis regardless of how low the ceiling is.
//! - **Clamps.** η ∈ [0,1] (η≥1 = jump to target at k≥1); observed ratio
//!   clamped to [0,1] (a regressing run realizes nothing; an over-delivering
//!   one saturates — the triage axes do not model overshoot).
//!
//! # UQ note (the Report-the-Floor rule)
//!
//! As a *point* diagnostic this module is not UQ-bearing. Promoting ρ̂ to
//! an interval predictor / coverage claim triggers the conformal-naive
//! floor rule (`.benchmarks/010`): gate against
//! `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` at that
//! promotion, not before.
//!
//! # Domain classification
//!
//! Latent, local, never synced: scalars over per-caller run state. No sync
//! dependency, no replay coupling.
//!
//! Feature: `bounded_target` (same gate as `bounded_target` — the issue's
//! "same feature or rider" clause; the pair ships together).

/// Default curvature-loss coefficient for [`rho_hat`] — a PRIOR, not a
/// calibration. Calibrate per use-site on frozen fixtures.
pub const DEFAULT_C: f32 = 0.5;

/// `observed < predicted·(1 − TRIAGE_FIT_TOL)` ⇒ [`Triage::FittingStarved`].
pub const TRIAGE_FIT_TOL: f32 = 0.10;
/// `predicted < TRIAGE_ACCEPT_MIN` ⇒ [`Triage::TargetStarved`] (once the
/// fit check passes).
pub const TRIAGE_ACCEPT_MIN: f32 = 0.50;

/// Closed-form position after `k` multiplicative steps of rate η toward
/// target `t`: `v_k = t + (1−η)^k·(v₀−t)`. O(1); `k = 0 ⇒ v₀` exactly,
/// `η = 1, k ≥ 1 ⇒ t` exactly.
pub fn fixpoint_position(v0: f32, t: f32, eta: f32, k: u32) -> f32 {
    let eta = eta.clamp(0.0, 1.0);
    t + (1.0 - eta).powi(k as i32) * (v0 - t)
}

/// Smallest `k` with `|v_k − t| ≤ tol`, in closed form.
/// `None` when unreachable (η = 0 and the gap exceeds tol).
pub fn iterations_to_threshold(v0: f32, t: f32, eta: f32, tol: f32) -> Option<u32> {
    let eta = eta.clamp(0.0, 1.0);
    let gap = (v0 - t).abs();
    if gap <= tol {
        return Some(0);
    }
    if eta <= 0.0 {
        return None;
    }
    if eta >= 1.0 {
        return Some(1);
    }
    // (1−η)^k·gap ≤ tol  ⇔  k ≥ ln(tol/gap)/ln(1−η)  (both logs negative).
    let k = ((tol / gap).ln() / (1.0 - eta).ln()).ceil();
    Some(k.max(1.0) as u32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// Inside the productive band — keep iterating.
    Iterate,
    /// `k_needed > 3/η`: ≥95% of the reachable movement has already
    /// happened ((1−η)^(3/η) ≈ e⁻³); the per-step gain is now < 5% of the
    /// original gap. The paper's budget law: the anchor is stale —
    /// RE-COMPUTE the target instead of grinding remaining iterations.
    ReAnchor,
}

/// The budget law: iterate or re-anchor?
pub fn budget_law(eta: f32, k_needed: u32) -> Advice {
    let eta = eta.clamp(f32::EPSILON, 1.0);
    if (k_needed as f32) > 3.0 / eta {
        Advice::ReAnchor
    } else {
        Advice::Iterate
    }
}

/// Predicted realized fraction: `ρ̂(k,η,ε) = (1−(1−η)^k)·(1−cε²)`, clamped
/// to [0,1]. The first factor is the absorption rate (how much of the
/// target movement k fitting steps at rate η absorb); the second is the
/// bound-loss factor (what fraction of the promised improvement survives
/// a bound of ε on the curved objective).
pub fn rho_hat(k: u32, eta: f32, eps: f32, c: f32) -> f32 {
    let eta = eta.clamp(0.0, 1.0);
    let absorb = 1.0 - (1.0 - eta).powi(k as i32);
    let bound_loss = (1.0 - c * eps * eps).max(0.0);
    (absorb * bound_loss).clamp(0.0, 1.0)
}

/// Observed vs predicted realization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rho {
    /// `realized / promised`, clamped to [0,1] (regression ⇒ 0).
    pub observed: f32,
    /// [`rho_hat`] at (k, η, ε, c).
    pub predicted: f32,
}

impl Rho {
    /// The triage verdict for this pair.
    pub fn triage(&self) -> Triage {
        triage(self)
    }
}

/// Compute the realization pair with [`DEFAULT_C`] (calibrate per
/// use-site; see module docs). A non-finite/non-positive promise yields
/// `observed = 0`.
pub fn realization_ratio(promised: f32, realized: f32, k: u32, eta: f32, eps: f32) -> Rho {
    realization_ratio_with_c(promised, realized, k, eta, eps, DEFAULT_C)
}

/// σ… c-aware variant: the calibrated curvature coefficient rides along.
pub fn realization_ratio_with_c(
    promised: f32,
    realized: f32,
    k: u32,
    eta: f32,
    eps: f32,
    c: f32,
) -> Rho {
    let observed = if promised.is_finite() && promised > 0.0 && realized.is_finite() {
        (realized / promised).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Rho {
        observed,
        predicted: rho_hat(k, eta, eps, c),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Triage {
    /// Observed within tolerance of the rate model's prediction.
    OnModel,
    /// `observed < predicted·(1−tol)` — the fitting loop realizes
    /// materially less than the rate model allows. Actionable: fit steps /
    /// LR / fitting-loop health, NOT the target.
    FittingStarved,
    /// Fit is on-model but `predicted < 0.5` — even perfect fitting
    /// realizes under half the promise. Actionable: the bound ε or the
    /// promise itself, NOT the fitting.
    TargetStarved,
}

/// The three-way triage (fit-first ordering — see module docs).
pub fn triage(rho: &Rho) -> Triage {
    if rho.observed < rho.predicted * (1.0 - TRIAGE_FIT_TOL) {
        Triage::FittingStarved
    } else if rho.predicted < TRIAGE_ACCEPT_MIN {
        Triage::TargetStarved
    } else {
        Triage::OnModel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g1_fixpoint_matches_iterative_loop() {
        // The closed form (one powi) vs k serial multiplies — same math,
        // different rounding; agreement to 1e-5 relative across a grid.
        let mut checked = 0;
        for &eta in &[0.05f32, 0.1, 0.3, 0.5, 0.9] {
            for &(v0, t) in &[(1.0f32, 0.0f32), (-2.0, 3.0), (0.5, 0.25)] {
                for &k in &[0u32, 1, 2, 5, 17, 60] {
                    let closed = fixpoint_position(v0, t, eta, k);
                    let mut v = v0;
                    for _ in 0..k {
                        v = t + (1.0 - eta) * (v - t);
                    }
                    let scale = (v0 - t).abs().max(1.0);
                    assert!(
                        (closed - v).abs() <= 1e-5 * scale,
                        "eta={eta} k={k}: closed {closed} vs loop {v}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 50);
    }

    #[test]
    fn g1_fixpoint_boundaries_exact() {
        assert_eq!(fixpoint_position(0.7, 0.2, 0.1, 0), 0.7, "k=0 ⇒ v₀");
        assert_eq!(fixpoint_position(0.7, 0.2, 1.0, 1), 0.2, "η=1 jumps");
        assert_eq!(fixpoint_position(0.7, 0.2, 1.0, 7), 0.2);
        // η clamped: η=2 behaves as the jump; η=−1 as frozen.
        assert_eq!(fixpoint_position(0.7, 0.2, 2.0, 1), 0.2);
        assert_eq!(fixpoint_position(0.7, 0.2, -1.0, 5), 0.7);
        // Long-run limit converges to t.
        let v = fixpoint_position(1.0, 0.0, 0.1, 200);
        assert!(v.abs() < 1e-4, "k=200 at η=0.1 leaves {v}");
    }

    #[test]
    fn g1_iterations_to_threshold_is_minimal() {
        // Returned k reaches the threshold; k−1 does not (when gap > tol).
        let (v0, t, eta, tol) = (1.0f32, 0.0f32, 0.1f32, 0.01f32);
        let k = iterations_to_threshold(v0, t, eta, tol).unwrap();
        assert!(k >= 1);
        let at_k = (fixpoint_position(v0, t, eta, k) - t).abs();
        assert!(at_k <= tol, "k={k} leaves {at_k} > {tol}");
        let before = iterations_to_threshold(v0, t, eta, tol).unwrap() - 1;
        if before > 0 {
            let at_before = (fixpoint_position(v0, t, eta, before) - t).abs();
            assert!(
                at_before > tol,
                "k={before} already satisfies {at_before} ≤ {tol} — not minimal"
            );
        }
        // Already within tolerance ⇒ 0; frozen η ⇒ unreachable.
        assert_eq!(iterations_to_threshold(0.001, 0.0, 0.1, 0.01), Some(0));
        assert_eq!(iterations_to_threshold(1.0, 0.0, 0.0, 0.01), None);
        assert_eq!(iterations_to_threshold(1.0, 0.0, 1.0, 0.01), Some(1));
    }

    #[test]
    fn g1_budget_law_boundary() {
        // η=0.1 ⇒ 3/η = 30: at and below → Iterate, above → ReAnchor.
        assert_eq!(budget_law(0.1, 29), Advice::Iterate);
        assert_eq!(budget_law(0.1, 30), Advice::Iterate);
        assert_eq!(budget_law(0.1, 31), Advice::ReAnchor);
        assert_eq!(budget_law(1.0, 4), Advice::ReAnchor);
    }

    #[test]
    fn g1_rho_hat_shape() {
        // Monotone in k; k→∞ at ε=0 saturates at 1; the ε term caps it.
        let a = rho_hat(1, 0.1, 0.0, DEFAULT_C);
        let b = rho_hat(10, 0.1, 0.0, DEFAULT_C);
        let c = rho_hat(100, 0.1, 0.0, DEFAULT_C);
        assert!(a < b && b < c && c < 1.0, "{a} {b} {c}");
        assert!((rho_hat(10_000, 0.1, 0.0, DEFAULT_C) - 1.0).abs() < 1e-4);
        let capped = rho_hat(10_000, 0.1, 0.5, DEFAULT_C);
        assert!(
            (capped - (1.0 - DEFAULT_C * 0.25)).abs() < 1e-4,
            "ε cap {capped}"
        );
        assert_eq!(rho_hat(0, 0.1, 0.0, DEFAULT_C), 0.0);
        // ε > 1/√c saturates the bound loss at 0 — nothing realizable.
        assert_eq!(rho_hat(100, 0.1, 10.0, DEFAULT_C), 0.0);
    }

    #[test]
    fn g1_triage_three_way_fixtures() {
        // OnModel: observation on the model, ceiling healthy.
        let r = Rho {
            observed: 0.80,
            predicted: 0.85,
        };
        assert_eq!(r.triage(), Triage::OnModel);
        // FittingStarved: observation materially under its own model.
        let r = Rho {
            observed: 0.10,
            predicted: 0.90,
        };
        assert_eq!(r.triage(), Triage::FittingStarved);
        // TargetStarved: fit on-model but the ceiling itself is low.
        let r = Rho {
            observed: 0.44,
            predicted: 0.45,
        };
        assert_eq!(r.triage(), Triage::TargetStarved);
        // Fit-first ordering: BOTH conditions fire ⇒ FittingStarved wins
        // (the observation is under its own model — that diagnosis is
        // actionable regardless of the ceiling).
        let r = Rho {
            observed: 0.05,
            predicted: 0.30,
        };
        assert_eq!(r.triage(), Triage::FittingStarved);
        // Guards: non-positive promise ⇒ observed 0 ⇒ starved verdicts.
        let r = realization_ratio(0.0, 1.0, 10, 0.1, 0.0);
        assert_eq!(r.observed, 0.0);
        assert_eq!(r.triage(), Triage::FittingStarved);
        // Regression clamps to 0 realized.
        let r = realization_ratio(1.0, -5.0, 10, 0.1, 0.0);
        assert_eq!(r.observed, 0.0);
    }

    #[test]
    fn g1_determinism_pure_functions() {
        let a = realization_ratio(2.0, 1.0, 12, 0.15, 0.05);
        let b = realization_ratio(2.0, 1.0, 12, 0.15, 0.05);
        assert_eq!(a, b);
        let _ = std::hint::black_box((
            fixpoint_position(0.3, 0.1, 0.2, 9),
            budget_law(0.07, 44),
            rho_hat(33, 0.11, 0.02, DEFAULT_C),
        ));
    }

    #[test]
    fn g1_calibration_protocol_on_frozen_fixture() {
        // The calibration gate: on a FROZEN fixture, the offline-calibrated
        // c reproduces the observed ratio; DEFAULT_C does not (that gap is
        // the protocol working — c is landscape-dependent, see module docs).
        //
        // Fixture: 1-D quadratic Q(v) = v², v₀ = 1, bounded step ε toward
        // the minimizer. Promised (first-order) improvement: 2·v₀·ε.
        // Realized: v₀² − (v₀−ε)² = 2v₀ε − ε². Observed ratio = 1 − ε/(2v₀).
        let (v0, eps) = (1.0f32, 0.1f32);
        let promised = 2.0 * v0 * eps;
        let realized = v0 * v0 - (v0 - eps) * (v0 - eps);
        let observed = realized / promised; // = 1 − ε/(2v₀) = 0.95
        assert!((observed - 0.95).abs() < 1e-6, "fixture math {observed}");
        // Offline calibration: c* = (1 − observed)/ε².
        let c_star = (1.0 - observed) / (eps * eps); // = 5.0
        assert!((c_star - 5.0).abs() < 1e-4, "c* {c_star}");
        // Calibrated prediction matches the observation (η=1, k=1: full
        // absorption — the fixture isolates the ε factor).
        let r = realization_ratio_with_c(promised, realized, 1, 1.0, eps, c_star);
        assert!(
            (r.predicted - observed).abs() <= 1e-4,
            "calibrated ρ̂ {r:?} vs observed {observed}"
        );
        // DEFAULT_C is NOT calibrated for this fixture — demonstrably.
        let r_default = realization_ratio(promised, realized, 1, 1.0, eps);
        assert!(
            (r_default.predicted - observed).abs() > 1e-3,
            "DEFAULT_C unexpectedly fits this fixture — update the docs"
        );
    }

    #[cfg_attr(debug_assertions, ignore = "timing gate — release-only")]
    #[test]
    fn g2_triage_path_o1_under_budget() {
        // The full ratio+triage path is closed-form (two powi + branches).
        // Measured 5.0 ns/call on M3 Max.
        const BUDGET_NS: f64 = 100.0;
        const N: u32 = 100_000;
        let mut acc = 0u32;
        for i in 0..64u32 {
            let r = realization_ratio(2.0, 1.0, 5 + i % 40, 0.05 + (i % 10) as f32 / 100.0, 0.05);
            acc += r.triage() as u32;
        }
        assert!(acc > 0, "warmup sanity");
        let t0 = std::time::Instant::now();
        acc = 0;
        for i in 0..N {
            let r = realization_ratio(2.0, 1.0, 5 + i % 40, 0.05 + (i % 10) as f32 / 100.0, 0.05);
            acc += r.triage() as u32 + std::hint::black_box(r.observed) as u32;
        }
        let dt = t0.elapsed();
        let per = dt.as_nanos() as f64 / N as f64;
        std::eprintln!("g2 realization_ratio+triage: {per:.1} ns/call");
        assert!(acc > 0);
        assert!(per <= BUDGET_NS, "{per:.1} ns > {BUDGET_NS} ns budget");
    }
}
