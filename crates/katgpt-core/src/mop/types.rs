//! MOP types — config + fixed-point solution (Plan 573 / Research 478).
//!
//! Leaf-clean: plain arrays + scalars. No game/chain/shard/emotion vocabulary.

use core::fmt;

/// Solver configuration for the MOP value-iteration operator (paper Eq. 7).
///
/// | Field | Paper symbol | Constraint | Semantics |
/// |---|---|---|---|
/// | `alpha` | α | `> 0` | action-entropy weight (temperature of the π\* softmax) |
/// | `beta` | β | `≥ 0` | state-transition-entropy weight — the risk knob (β=0: pure own-action exploration; β>0: attraction to stochastic regions, the fog-of-war analog) |
/// | `gamma` | γ | `(0, 1)` | discount factor |
/// | `tol` | — | `> 0` | sup-norm convergence tolerance on `ln z` deltas |
/// | `max_iter` | — | `≥ 1` | iteration cap |
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MopConfig {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub tol: f32,
    pub max_iter: u32,
}

impl MopConfig {
    /// Paper-default-style config: α=1, β=1, γ=0.95, tol=1e-9, 10k iters.
    pub fn paper_default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            gamma: 0.95,
            tol: 1e-9,
            max_iter: 10_000,
        }
    }

    /// Validate the constructor contract. Returns `Err` with the violated
    /// invariant's name.
    ///
    /// NOTE: the `!(x > bound)` / `x < bound` forms are deliberate NaN
    /// discipline — every comparison must reject NaN (the `partial_cmp`
    /// rewrite clippy suggests would silently admit NaN through the
    /// `PartialOrd` incomparable arm).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn validate(&self) -> Result<(), MopConfigError> {
        if !(self.alpha > 0.0) || self.alpha.is_nan() {
            return Err(MopConfigError::AlphaMustBePositive);
        }
        if self.beta < 0.0 || self.beta.is_nan() {
            return Err(MopConfigError::BetaMustBeNonNegative);
        }
        if !(self.gamma > 0.0 && self.gamma < 1.0) || self.gamma.is_nan() {
            return Err(MopConfigError::GammaMustBeOpenUnitInterval);
        }
        if !(self.tol > 0.0) || self.tol.is_nan() {
            return Err(MopConfigError::TolMustBePositive);
        }
        if self.max_iter == 0 {
            return Err(MopConfigError::MaxIterMustBeAtLeastOne);
        }
        Ok(())
    }
}

/// Config-contract violation (see [`MopConfig::validate`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MopConfigError {
    AlphaMustBePositive,
    BetaMustBeNonNegative,
    GammaMustBeOpenUnitInterval,
    TolMustBePositive,
    MaxIterMustBeAtLeastOne,
}

impl fmt::Display for MopConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            MopConfigError::AlphaMustBePositive => "alpha must be > 0",
            MopConfigError::BetaMustBeNonNegative => "beta must be >= 0",
            MopConfigError::GammaMustBeOpenUnitInterval => "gamma must be in (0, 1)",
            MopConfigError::TolMustBePositive => "tol must be > 0",
            MopConfigError::MaxIterMustBeAtLeastOne => "max_iter must be >= 1",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for MopConfigError {}

/// Fixed-point solution of the Eq. 7 map.
///
/// - `v_star[i] = (α/γ) · ln z*_i` — the optimal path-occupancy value.
/// - `ln_z[i] = ln z*_j` — the raw fixed point (solver's native space).
/// - `lse_args[i][k] = H̄[i,k] + Σ_j p[i,k,j]·ln z*_j` for available actions
///   (the paper's Eq. 5 softmax argument, materialized once at convergence so
///   [`pi_star`](crate::mop::MopSolver::pi_star) is a pure O(A) read).
///   Unavailable actions hold `f32::NEG_INFINITY` (exp → 0).
/// - `iterations` / `sup_delta` — convergence audit (the sup-norm of the last
///   `ln z` update; `sup_delta < tol` on a converged solve).
///
/// Note: `lse_args` is an addition to Research 478 §3.3's field list — it is
/// what makes the plan's `pi_star(&solution, s, out)` signature stateless.
#[derive(Clone, Debug)]
pub struct MopSolution<const N: usize, const A: usize> {
    pub v_star: [f32; N],
    pub ln_z: [f32; N],
    pub lse_args: [[f32; A]; N],
    pub iterations: u32,
    pub sup_delta: f32,
}

impl<const N: usize, const A: usize> Default for MopSolution<N, A> {
    fn default() -> Self {
        Self {
            v_star: [0.0; N],
            ln_z: [0.0; N],
            lse_args: [[f32::NEG_INFINITY; A]; N],
            iterations: 0,
            sup_delta: f32::INFINITY,
        }
    }
}
