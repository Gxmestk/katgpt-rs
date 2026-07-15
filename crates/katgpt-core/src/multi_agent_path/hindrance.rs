//! Hindrance estimator — one-step blocking count (Plan 440 T1.7).
//!
//! Distilled from Okumura & Nagai 2025. For agent `i` considering a move to
//! cell `u` at tick `t+1`, `hindrance(i → u)` counts how many sibling agents
//! `j ≠ i` have `u` in their `t+1` reachable neighborhood (i.e. how many agents
//! `i` would block by taking `u`). This is used as the 3rd PIBT tiebreak
//! term `⟨..., hindrance, ε⟩`.
//!
//! O(neighbors²) per agent — near-zero cost in practice (grid degree ≤ 5).
//!
//! # Pluggable seam
//!
//! The [`HindranceEstimator`] trait is the extension point. The default impl
//! [`BlockingCount`] is paper-faithful (uniform blocking weight). A private
//! consumer (riir-ai/318) plugs in affect-aware hindrance: blocks from
//! fearful NPCs count more (the social-cost extension).

use super::config::AgentId;
use super::position::Position;

/// Trait for hindrance estimation — pluggable seam #4.
///
/// See module docs.
pub trait HindranceEstimator<P: Position> {
    /// Estimate the hindrance cost of agent `agent` moving to `target` at `t+1`,
    /// given the current joint configuration `config`.
    ///
    /// Returns a non-negative scalar; lower is better (less blocking).
    /// Paper default: raw count of blocked siblings.
    fn hindrance(
        &mut self,
        agent: AgentId,
        target: &P,
        config: &super::config::JointConfig<P>,
    ) -> f32;
}

/// Paper-default hindrance: count siblings whose `t+1` neighborhood includes
/// `target`.
///
/// Each sibling `j ≠ i` contributes 1 if `target` is in `neighbors(config[j])`.
/// This is the raw collision-count proxy — a discrete approximation to the DEC
/// codifferential of the joint flow field (see Research 424 §2.3).
pub struct BlockingCount;

impl BlockingCount {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BlockingCount {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Position> HindranceEstimator<P> for BlockingCount {
    fn hindrance(
        &mut self,
        agent: AgentId,
        target: &P,
        config: &super::config::JointConfig<P>,
    ) -> f32 {
        let me = usize::from(agent);
        let mut count = 0u32;
        for (j, pos) in config.positions.iter().enumerate() {
            if j == me {
                continue;
            }
            // Does sibling j's neighborhood include target?
            for n in pos.neighbors() {
                if &n == target {
                    count += 1;
                    break; // count each sibling once
                }
            }
        }
        count as f32
    }
}

/// A hindrance estimator that weights blocks by a per-agent scalar.
///
/// This is the documented extension point for affect-aware hindrance
/// (riir-ai/318 Extension D): blocks from high-weight agents (e.g. fearful
/// NPCs) count more. The weight source is supplied by the caller via a
/// `weight(agent) -> f32` closure.
pub struct WeightedBlockingCount<F: Fn(AgentId) -> f32> {
    weight: F,
}

impl<F: Fn(AgentId) -> f32> WeightedBlockingCount<F> {
    pub fn new(weight: F) -> Self {
        Self { weight }
    }
}

impl<P: Position, F: Fn(AgentId) -> f32> HindranceEstimator<P> for WeightedBlockingCount<F> {
    fn hindrance(
        &mut self,
        agent: AgentId,
        target: &P,
        config: &super::config::JointConfig<P>,
    ) -> f32 {
        let me = usize::from(agent);
        let mut cost = 0.0f32;
        for (j, pos) in config.positions.iter().enumerate() {
            if j == me {
                continue;
            }
            for n in pos.neighbors() {
                if &n == target {
                    let sibling = AgentId(j as u32);
                    cost += (self.weight)(sibling);
                    break;
                }
            }
        }
        cost
    }
}
