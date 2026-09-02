//! Two-channel imbalance collapse monitor — Issue 708 P2.
//!
//! The GenFirst transfer law (Research 437 — arXiv:2608.29335): collapse is
//! an IMBALANCE between concentration pressure and spread, not "entropy
//! below τ" — `D_KL(q‖p₀) = E_q[−log p₀(z)] − H(q)` degenerates when the two
//! channels diverge, and the paper's own Table 2 shows the absolute-entropy
//! reading alone mis-classifies. Transferred to modelless runtime surfaces
//! (per-NPC belief/emotion populations, span embeddings, fix trajectories —
//! the P1 consumer list) as two observable channels:
//!
//! - **Channel A (spread)** — [`crate::data_probe::entropy::
//!   knn_differential_entropy`] over the point population, in nats.
//! - **Channel B (concentration)** — mean |cosine| over deterministic
//!   strided point pairs. Directional/mode concentration raises it;
//!   isotropic rescaling leaves it invariant (scale-free by construction).
//!
//! The alarm is the CONJUNCTION `a_z ≤ −k_a AND b_z ≥ +k_b` — channel A
//! falling while channel B rises relative to a healthy baseline frozen after
//! a warm-up window. This is a *relative* detector: it fires on deviation
//! from the baseline, which crosses early in a degradation, while an
//! absolute `h < τ_low` detector only fires once the level has traversed
//! from healthy to collapsed. The lead-time GOAT gate
//! (`bench_708_imbalance_goat`) measures exactly that on the bench_681
//! fixture populations, and pins the mechanism boundary: on ISOTROPIC
//! contraction (every dim shrinking together) channel B stays flat, the
//! conjunction never fires early, and the absolute/derivative channels are
//! the right detectors — the imbalance monitor's early warning covers
//! DIRECTIONAL/mode collapse, which is the collapse shape the transfer law
//! describes (posterior collapse is directional by definition).
//!
//! # Contract
//!
//! - **Event-triggered reporting only** (Research 437 §transfer boundary):
//!   the monitor READS and REPORTS; it never mutates weights, budgets, or
//!   any closed-loop controller. Consumers decide what an alarm means.
//! - **Frozen baseline**: statistics are Welford-accumulated over the first
//!   `config.warmup_cycles` observations, then frozen. Warm-up populations
//!   must be healthy (feeding collapsed data during warm-up poisons the
//!   baseline — documented misuse, not defended against). Audit-cadence
//!   assumption: re-baseline per audit window; long-horizon drift beyond
//!   the window is out of scope by design.
//! - **Deterministic**: fixed inputs ⇒ bit-identical readings (no RNG; the
//!   pair set is the fixed stride pairing `i ↔ i + n/2`).
//! - **Zero steady-state allocation** (G4 by construction): channel A's
//!   bounded heap lives in [`KnnEntropyScratch`] (owned by the monitor,
//!   sized once at `new`); channel B is pure arithmetic over the flat
//!   slice. The [`ImbalanceReading`] return is a fixed-size `Copy` struct.
//! - **Duplicates ⇒ −∞** propagate honestly from P1: a fully collapsed
//!   population drives `a_z` to −∞ (and the alarm, with B rising) rather
//!   than being clamped.
//!
//! Opt-in `imbalance_monitor` (implies `knn_entropy`) until a consumer
//! promotes — the thin advisory latch on `S2FCollapseDetector`
//! (`katgpt-pruners`, feature `population_collapse`) is the first consumer
//! surface (no-default-consumer rule: opt-in until wired in a default
//! build).

use super::entropy::{KnnEntropyScratch, knn_differential_entropy};

// ──────────────────────────────────────────────────────────────────────────
// Config
// ──────────────────────────────────────────────────────────────────────────

/// Monitor configuration. Defaults are the calibrated operating point from
/// the GOAT gate (`bench_708_imbalance_goat`): k_a = k_b = 3σ conjunctive
/// margins, an 8-cycle healthy warm-up, KL estimator order k = 4.
#[derive(Clone, Copy, Debug)]
pub struct ImbalanceConfig {
    /// Channel A alarm margin in baseline σ: alarm requires
    /// `a_z ≤ −k_a` (entropy falling). Default 3.0.
    pub k_a: f64,
    /// Channel B alarm margin in baseline σ: alarm requires
    /// `b_z ≥ +k_b` (concentration rising). Default 3.0.
    pub k_b: f64,
    /// Healthy observations accumulated before the baseline freezes.
    /// Must be ≥ 2 (variance needs two samples). Default 8.
    pub warmup_cycles: usize,
    /// k-NN order for the channel-A estimator (P1's calibration band
    /// 1..=8). Default 4.
    pub entropy_k: usize,
}

impl Default for ImbalanceConfig {
    fn default() -> Self {
        Self {
            k_a: 3.0,
            k_b: 3.0,
            warmup_cycles: 8,
            entropy_k: 4,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Reading
// ──────────────────────────────────────────────────────────────────────────

/// One observation's report. Fixed-size `Copy` — the monitor allocates
/// nothing per observe (G4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImbalanceReading {
    /// Channel A level — kNN differential entropy, nats (−∞ on a
    /// duplicated/collapsed population).
    pub entropy: f64,
    /// Channel B level — mean |cosine| over the strided pair set, [0, 1].
    pub concentration: f64,
    /// Channel A deviation from the frozen baseline, in σ (negative =
    /// falling). 0.0 before the baseline freezes.
    pub a_z: f64,
    /// Channel B deviation from the frozen baseline, in σ (positive =
    /// rising). 0.0 before the baseline freezes.
    pub b_z: f64,
    /// Conjunctive imbalance alarm (`a_z ≤ −k_a && b_z ≥ +k_b`). Always
    /// `false` during warm-up.
    pub alarmed: bool,
    /// `true` once the baseline is frozen (warm-up complete) and z-scores
    /// are meaningful.
    pub warmed_up: bool,
}

// ──────────────────────────────────────────────────────────────────────────
// Monitor
// ──────────────────────────────────────────────────────────────────────────

/// Two-channel imbalance collapse monitor over point populations
/// (row-major flat `[n × d]` per observation). See the module docs for the
/// two-channel law, the frozen-baseline contract, and the isotropic scope
/// boundary.
pub struct ImbalanceMonitor {
    config: ImbalanceConfig,
    scratch: KnnEntropyScratch,
    // Welford accumulators over the warm-up window (channels A and B).
    warm_count: u64,
    mean_a: f64,
    m2_a: f64,
    mean_b: f64,
    m2_b: f64,
    // Frozen baseline (valid once `frozen`).
    frozen: bool,
    base_a: f64,
    base_b: f64,
    sigma_a: f64,
    sigma_b: f64,
}

impl ImbalanceMonitor {
    /// Construct a monitor. Panics if `config.warmup_cycles < 2` (the
    /// baseline variance needs two samples) or `config.entropy_k == 0`.
    pub fn new(config: ImbalanceConfig) -> Self {
        assert!(config.warmup_cycles >= 2, "warmup_cycles must be >= 2");
        assert!(config.entropy_k >= 1, "entropy_k must be >= 1");
        Self {
            scratch: KnnEntropyScratch::new(config.entropy_k),
            config,
            warm_count: 0,
            mean_a: 0.0,
            m2_a: 0.0,
            mean_b: 0.0,
            m2_b: 0.0,
            frozen: false,
            base_a: 0.0,
            base_b: 0.0,
            sigma_a: 0.0,
            sigma_b: 0.0,
        }
    }

    /// Observe one population (row-major `[n × d]`). Panics under the same
    /// shape contract as [`knn_differential_entropy`] (`n > k`, `d > 0`,
    /// `points.len() == n × d`).
    ///
    /// During warm-up the reading reports levels with `alarmed == false`,
    /// `warmed_up == false`; the observation-th warm-up sample freezes the
    /// baseline and from then on every reading carries z-scores and the
    /// conjunctive alarm.
    pub fn observe(&mut self, points: &[f32], n: usize, d: usize) -> ImbalanceReading {
        let entropy =
            knn_differential_entropy(points, n, d, self.config.entropy_k, &mut self.scratch);
        let concentration = strided_mean_abs_cosine(points, n, d);

        if !self.frozen {
            // Welford update (numerically stable single-pass moments).
            self.warm_count += 1;
            let count = self.warm_count as f64;
            let delta_a = entropy - self.mean_a;
            self.mean_a += delta_a / count;
            self.m2_a += delta_a * (entropy - self.mean_a);
            let delta_b = concentration - self.mean_b;
            self.mean_b += delta_b / count;
            self.m2_b += delta_b * (concentration - self.mean_b);

            if self.warm_count as usize >= self.config.warmup_cycles {
                // Freeze. σ floor guards a zero-variance warm-up (identical
                // populations) against a divide-by-tiny — with the floor the
                // z-scores saturate honestly instead of returning NaN.
                self.base_a = self.mean_a;
                self.base_b = self.mean_b;
                let var_a = self.m2_a / (self.warm_count - 1) as f64;
                let var_b = self.m2_b / (self.warm_count - 1) as f64;
                self.sigma_a = var_a.sqrt().max(1e-9);
                self.sigma_b = var_b.sqrt().max(1e-9);
                self.frozen = true;
            }
            if !self.frozen {
                return ImbalanceReading {
                    entropy,
                    concentration,
                    a_z: 0.0,
                    b_z: 0.0,
                    alarmed: false,
                    warmed_up: false,
                };
            }
            // Fell through: this observation froze the baseline — it is
            // scored against it below (warmed_up: true).
        }

        let a_z = (entropy - self.base_a) / self.sigma_a;
        let b_z = (concentration - self.base_b) / self.sigma_b;
        let alarmed = a_z <= -self.config.k_a && b_z >= self.config.k_b;
        ImbalanceReading {
            entropy,
            concentration,
            a_z,
            b_z,
            alarmed,
            warmed_up: true,
        }
    }

    /// Whether the baseline has frozen (warm-up complete).
    #[inline]
    pub fn warmed_up(&self) -> bool {
        self.frozen
    }

    /// The frozen baseline configuration.
    #[inline]
    pub fn config(&self) -> &ImbalanceConfig {
        &self.config
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Channel B — concentration
// ──────────────────────────────────────────────────────────────────────────

/// Mean |cosine| over the deterministic strided pair set `i ↔ i + n/2`
/// (`i < n/2`) — O(n·d), zero allocation, no RNG, bit-deterministic.
///
/// Deliberately NOT [`super::geometry::avg_cosine_similarity`]: that is the
/// exhaustive O(n²) offline metric over `&[Vec<f32>]` rows with a
/// normalization buffer; this is the flat-slice, subsampled-pair,
/// per-observe form the monitor's steady-state contract needs. The pair
/// set is fixed (stride = n/2), so the reading is a deterministic function
/// of the population — subsampling halves the resolution versus exhaustive
/// pairing, which the frozen-baseline σ absorbs (the z-score is relative to
/// the same estimator's own noise).
fn strided_mean_abs_cosine(points: &[f32], n: usize, d: usize) -> f64 {
    debug_assert_eq!(points.len(), n * d, "points.len() must be n × d");
    let half = n / 2;
    if half == 0 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    let mut count = 0u64;
    for i in 0..half {
        let j = i + half;
        let (dot, ni, nj) = dot_and_norms(points, i, j, d);
        let denom = ni * nj;
        if denom > 1e-12 {
            acc += (dot / denom).abs();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { acc / count as f64 }
}

/// Dot product and squared norms of rows `i` and `j` (f64 accumulation —
/// the P1 f64 discipline; the sqrt of each norm happens once per pair).
#[inline]
fn dot_and_norms(points: &[f32], i: usize, j: usize, d: usize) -> (f64, f64, f64) {
    let row_i = &points[i * d..(i + 1) * d];
    let row_j = &points[j * d..(j + 1) * d];
    let mut dot = 0.0f64;
    let mut ni = 0.0f64;
    let mut nj = 0.0f64;
    for t in 0..d {
        let a = row_i[t] as f64;
        let b = row_j[t] as f64;
        dot += a * b;
        ni += a * a;
        nj += b * b;
    }
    (dot, ni.sqrt(), nj.sqrt())
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 256;
    const D: usize = 8;

    /// Minimal deterministic LCG + Box–Muller (the entropy.rs test-module
    /// pattern — self-contained, no cross-test fixture dep).
    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            // Numerical Recipes LCG constants.
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f64) / (u32::MAX as f64)
        }
        fn gauss(&mut self) -> f64 {
            // Box–Muller; guard u1 against log(0).
            let u1 = (self.unit() + 1e-12).min(1.0);
            let u2 = self.unit();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    fn gaussian_population(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed);
        let mut out = vec![0.0f32; n * d];
        for v in out.iter_mut() {
            *v = rng.gauss() as f32;
        }
        out
    }

    /// Directional rank-1 collapse: preserve dim 0, scale dims 1..d by
    /// `(1 − λ)` (the bench_708 degradation shape).
    fn degrade_directional(base: &[f32], n: usize, d: usize, lam: f64) -> Vec<f32> {
        let keep = 1.0 - lam;
        let mut out = base.to_vec();
        for i in 0..n {
            for j in 1..d {
                out[i * d + j] = (base[i * d + j] as f64 * keep) as f32;
            }
        }
        out
    }

    /// Isotropic contraction: scale EVERY dim by `(1 − λ)` — the scope
    /// boundary control (channel B must stay invariant).
    fn degrade_isotropic(base: &[f32], lam: f64) -> Vec<f32> {
        let keep = (1.0 - lam) as f32;
        base.iter().map(|v| v * keep).collect()
    }

    fn monitor() -> ImbalanceMonitor {
        ImbalanceMonitor::new(ImbalanceConfig {
            k_a: 3.0,
            k_b: 3.0,
            warmup_cycles: 8,
            entropy_k: 4,
        })
    }

    #[test]
    fn config_validation_panics() {
        assert!(
            std::panic::catch_unwind(|| {
                let _ = ImbalanceMonitor::new(ImbalanceConfig {
                    warmup_cycles: 1,
                    ..ImbalanceConfig::default()
                });
            })
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| {
                let _ = ImbalanceMonitor::new(ImbalanceConfig {
                    entropy_k: 0,
                    ..ImbalanceConfig::default()
                });
            })
            .is_err()
        );
    }

    #[test]
    fn warmup_suppresses_alarms_then_freezes() {
        let mut mon = monitor();
        for c in 0..7 {
            let pop = gaussian_population(N, D, 1000 + c);
            let r = mon.observe(&pop, N, D);
            assert!(!r.alarmed, "warm-up cycle {c} must never alarm");
            assert!(!r.warmed_up);
            assert_eq!(r.a_z, 0.0);
            assert_eq!(r.b_z, 0.0);
        }
        // 8th warm-up observation freezes the baseline.
        let pop = gaussian_population(N, D, 1007);
        let r = mon.observe(&pop, N, D);
        assert!(r.warmed_up, "the warmup_cycles-th observe freezes");
        assert!(!r.alarmed, "freeze observation itself reports no alarm");
        assert!(mon.warmed_up());
    }

    #[test]
    fn healthy_population_never_alarms() {
        let mut mon = monitor();
        for c in 0..48 {
            let pop = gaussian_population(N, D, 2000 + c);
            let r = mon.observe(&pop, N, D);
            assert!(
                !r.alarmed,
                "healthy cycle {c} false-alarmed (a_z {:.2}, b_z {:.2})",
                r.a_z, r.b_z
            );
        }
    }

    #[test]
    fn directional_collapse_alarms_with_both_channels_moving() {
        let mut mon = monitor();
        for c in 0..8 {
            let pop = gaussian_population(N, D, 3000 + c);
            let _ = mon.observe(&pop, N, D);
        }
        // Ramp λ = 0.1, 0.2, … — the alarm must fire within a few cycles,
        // with BOTH channels moving in their collapse directions.
        let mut fired_at = None;
        for c in 1..=10u32 {
            let lam = 0.1 * c as f64;
            let base = gaussian_population(N, D, 3000 + 8 + c as u64);
            let pop = degrade_directional(&base, N, D, lam);
            let r = mon.observe(&pop, N, D);
            assert!(r.warmed_up);
            if r.alarmed {
                fired_at = Some(c);
                assert!(
                    r.a_z < 0.0,
                    "alarm requires channel A falling (a_z {:.2})",
                    r.a_z
                );
                assert!(
                    r.b_z > 0.0,
                    "alarm requires channel B rising (b_z {:.2})",
                    r.b_z
                );
                break;
            }
        }
        let fired_at = fired_at.expect("directional collapse must trip the alarm within 10 cycles");
        assert!(
            fired_at <= 5,
            "alarm fired at cycle {fired_at} — later than the calibrated band"
        );
    }

    /// The mechanism boundary: isotropic contraction drops entropy hard
    /// (channel A deeply negative) but channel B is scale-invariant and
    /// stays flat — the conjunction must NOT fire. This is the honest
    /// scope-limit pinned as a test: the imbalance early warning covers
    /// DIRECTIONAL collapse; isotropic drift belongs to the absolute /
    /// derivative channels.
    #[test]
    fn isotropic_shrink_does_not_alarm() {
        let mut mon = monitor();
        for c in 0..8 {
            let pop = gaussian_population(N, D, 4000 + c);
            let _ = mon.observe(&pop, N, D);
        }
        let base = gaussian_population(N, D, 4008);
        for c in 1..12u32 {
            let lam = 0.08 * c as f64;
            let pop = degrade_isotropic(&base, lam);
            let r = mon.observe(&pop, N, D);
            assert!(
                r.a_z < -3.0,
                "control validity: isotropic shrink must drop channel A hard (cycle {c} a_z {:.2})",
                r.a_z
            );
            assert!(
                r.b_z.abs() < 3.0,
                "channel B must stay flat under isotropic rescaling (cycle {c} b_z {:.2})",
                r.b_z
            );
            assert!(
                !r.alarmed,
                "isotropic shrink must NOT trip the imbalance alarm (cycle {c})"
            );
        }
    }

    #[test]
    fn determinism_x3_bit_identical() {
        let run = || {
            let mut mon = monitor();
            let mut trace = Vec::with_capacity(18);
            for c in 0..8 {
                let pop = gaussian_population(N, D, 5000 + c);
                trace.push(mon.observe(&pop, N, D));
            }
            for c in 1..10u32 {
                let base = gaussian_population(N, D, 5000 + 8 + c as u64);
                let pop = degrade_directional(&base, N, D, 0.1 * c as f64);
                trace.push(mon.observe(&pop, N, D));
            }
            trace
                .iter()
                .map(|r| {
                    format!(
                        "{:?}|{:e}|{:e}|{:e}",
                        r.entropy, r.concentration, r.a_z, r.b_z
                    )
                })
                .collect::<Vec<_>>()
                .join(";")
        };
        let r1 = run();
        let r2 = run();
        let r3 = run();
        assert_eq!(r1, r2, "run 2 diverged");
        assert_eq!(r2, r3, "run 3 diverged");
    }

    /// G4: zero allocations in steady state (the gaussianity.rs /
    /// latent_confounder_audit pattern — the lib test binary installs
    /// `alloc::TrackingAllocator` under cfg(test, debug_assertions); skip
    /// with a message if absent. The accessors are debug_assertions-gated
    /// in alloc.rs — gate the test to match.)
    #[test]
    #[cfg(debug_assertions)]
    fn g4_zero_alloc_steady_state() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        let mut mon = monitor();
        let warm: Vec<Vec<f32>> = (0..8)
            .map(|c| gaussian_population(N, D, 6000 + c))
            .collect();
        for pop in &warm {
            let _ = mon.observe(pop, N, D);
        }
        let steady = gaussian_population(N, D, 6008);

        // Sentinel: confirm the allocator is installed.
        reset_alloc_stats();
        let sentinel: Vec<u8> = vec![0u8; 256];
        let (sent_count, _) = get_alloc_stats();
        if sent_count == 0 {
            eprintln!("g4_zero_alloc_steady_state: TrackingAllocator not installed — SKIPPED");
            return;
        }
        drop(sentinel);

        reset_alloc_stats();
        for _ in 0..10 {
            let _ = mon.observe(&steady, N, D);
        }
        let (count, bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "ImbalanceMonitor::observe must be alloc-free in steady state \
             (count {count}, {bytes} bytes) — channel A's heap is owned scratch, \
             channel B is pure arithmetic"
        );
    }

    #[test]
    fn shape_mismatch_panics() {
        let mut mon = monitor();
        let warm = gaussian_population(N, D, 7000);
        // Wrong length during warm-up already routes into the entropy
        // estimator's assert — pin the panic on the monitor surface.
        let wrong = vec![0.0f32; N * D - 1];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = mon.observe(&wrong, N, D);
        }));
        assert!(r.is_err(), "shape mismatch must panic");
        drop(warm);
    }
}
