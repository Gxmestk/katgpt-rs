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
use super::position::{GridPos, Position};

/// Trait for hindrance estimation — pluggable seam #4.
///
/// See module docs. The default impl is [`BlockingCount`] (raw sibling
/// count). For affect-aware hindrance (fearful NPCs count more), use
/// [`WeightedBlockingCount`] with a per-agent weight closure.
///
/// # Example: affect-aware hindrance (HLA fusion)
///
/// A fearful NPC blocking your path is a bigger cost than a calm NPC doing
/// the same — the social-cost extension from riir-ai/318 Extension D. The
/// weight is a closed-form sigmoid projection of the NPC's fear scalar,
/// not a learned value.
///
/// ```no_run
/// use katgpt_core::multi_agent_path::*;
/// use katgpt_core::multi_agent_path::position::*;
///
/// /// Per-agent fear weight table. In the real runtime this is populated
/// /// from each NPC's `HlaCacheProxy` fear scalar (the 5th HLA component):
/// /// `weight_i = 1 + γ · fear_i`.
/// ///
/// /// Fearful blockers cost more to displace, so agents route around them.
/// /// The base weight is 1.0 (paper-faithful when all `fear_i = 0`).
/// struct FearTable {
///     fear: Vec<f32>,
///     gamma: f32,
/// }
///
/// impl FearTable {
///     fn hindrance_estimator(&self) -> WeightedBlockingCount<impl Fn(AgentId) -> f32 + '_> {
///         let fear = &self.fear;
///         let gamma = self.gamma;
///         WeightedBlockingCount::new(move |agent: AgentId| {
///             let i = usize::from(agent);
///             // Base 1.0 (paper-faithful) + fear-scaled social cost.
///             1.0 + gamma * fear.get(i).copied().unwrap_or(0.0)
///         })
///     }
/// }
/// ```
///
/// # Custom estimator shape
///
/// A consumer that wants a fully custom estimator (e.g. one that reads the
/// DEC codifferential of the crowd flow at `target`) implements the trait
/// directly. The only constraint is that the return value is non-negative
/// and deterministic — the same `(agent, target, config)` triple must always
/// produce the same cost, to preserve deterministic replay.
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

    /// Pre-compute per-tick data structures for O(1) hindrance lookups
    /// (Issue 516 T1g).
    ///
    /// Called once at the start of each PIBT pass, before any `hindrance`
    /// calls. The default impl is a no-op (backward compatible — estimators
    /// that don't need precomputation are unaffected).
    ///
    /// `BlockingCount` overrides this to build a `reach_count` map:
    /// position → number of agents whose neighborhood includes it. This
    /// transforms `hindrance()` from O(n) per call to O(1), eliminating the
    /// O(n²) PIBT bottleneck at high agent density (n=1000).
    fn prepare(&mut self, _config: &super::config::JointConfig<P>) {}
}

/// Paper-default hindrance: count siblings whose `t+1` neighborhood includes
/// `target`.
///
/// Each sibling `j ≠ i` contributes 1 if `target` is in `neighbors(config[j])`.
/// This is the raw collision-count proxy — a discrete approximation to the DEC
/// codifferential of the joint flow field (see Research 424 §2.3).
///
/// Issue 516 T1g: uses a precomputed `reach_count` map (built in `prepare`)
/// for O(1) lookups instead of the original O(n) scan. The `prepare` method
/// is called once per PIBT pass by `greedy_pibt_pass`, eliminating the O(n²)
/// scaling that dominated latency at n ≥ 500.
pub struct BlockingCount {
    /// Cached reach counts: position → number of agents whose neighborhood
    /// includes it. Built by `prepare()`. Empty = not prepared (fallback to
    /// O(n) scan for correctness).
    reach_count: std::collections::HashMap<GridPos, u32>,
}

impl BlockingCount {
    pub fn new() -> Self {
        Self {
            reach_count: std::collections::HashMap::new(),
        }
    }
}

impl Default for BlockingCount {
    fn default() -> Self {
        Self::new()
    }
}

impl HindranceEstimator<GridPos> for BlockingCount {
    fn prepare(&mut self, config: &super::config::JointConfig<GridPos>) {
        self.reach_count.clear();
        self.reach_count.reserve(config.n_agents() * 5);
        for pos in &config.positions {
            for n in pos.neighbors() {
                *self.reach_count.entry(n).or_insert(0) += 1;
            }
        }
    }

    fn hindrance(
        &mut self,
        agent: AgentId,
        target: &GridPos,
        config: &super::config::JointConfig<GridPos>,
    ) -> f32 {
        // Fast path: precomputed reach counts from `prepare()`.
        if !self.reach_count.is_empty() {
            let total = self.reach_count.get(target).copied().unwrap_or(0);
            // Subtract 1 if the agent itself can reach `target` (we skip j==me).
            let my_pos = &config.positions[usize::from(agent)];
            let my_reaches = my_pos.neighbors().iter().any(|n| n == target) as u32;
            return (total - my_reaches) as f32;
        }

        // Fallback: O(n) scan (pre-T1g path, used if `prepare` wasn't called).
        let me = usize::from(agent);
        let mut count = 0u32;
        for (j, pos) in config.positions.iter().enumerate() {
            if j == me {
                continue;
            }
            for n in pos.neighbors() {
                if &n == target {
                    count += 1;
                    break;
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

/// Counter-flow hindrance estimator (Issue 147 — Guided-PIBT variant).
///
/// Wraps a base estimator (default [`BlockingCount`]) and adds a penalty for
/// moves where nearby agents are heading in the **opposite direction** — the
/// hallmark of head-on corridor deadlock. This is the lightest-weight
/// Guided-PIBT mechanism: it doesn't require explicit corridor detection or
/// global flow computation, just a per-pair direction-alignment check.
///
/// # Mechanism
///
/// For agent `i` considering move to `target`, each nearby sibling `j` (within
/// `scan_radius` cells) contributes a counter-flow penalty if `j`'s goal
/// direction is roughly anti-aligned with `i`'s goal direction. Two agents are
/// counter-flowing when their goal vectors point in opposite quadrants — they
/// will meet head-on if they continue on their current paths.
///
/// The penalty is scaled by `gamma` (default 2.0). At `gamma = 0` this
/// estimator degenerates to the base estimator (paper-faithful).
///
/// # Modelless
///
/// Entirely heuristic — direction is computed from Manhattan distance gradient
/// (no BFS, no training). The penalty is a closed-form scalar.
///
/// # Why this is "Guided-PIBT"
///
/// The paper says global guidance (Guided-PIBT) wins on corridor-heavy maps
/// because local finite-window guidance can't see oncoming traffic. This
/// estimator adds that oncoming-traffic awareness to the PIBT tiebreak without
/// requiring a global flow field — it's a local approximation to the global
/// signal. On maps with genuine long corridors (1-wide), a full flow-field
/// implementation would be stronger, but this captures the essential mechanism.
///
/// # Grid-specific
///
/// This estimator is concrete to [`GridPos`] (the substrate's default position
/// type) because the counter-flow check needs integer `(x, y)` coordinates to
/// compute direction vectors. For non-grid positions, implement the trait
/// directly with a position-appropriate direction metric.
pub struct CounterFlowHindrance {
    /// Counter-flow penalty multiplier.
    gamma: f32,
    /// How many cells away to scan for counter-flowing siblings.
    scan_radius: i32,
    /// Per-agent goal positions, updated each tick via [`set_goals`](Self::set_goals).
    goals: Vec<GridPos>,
}

impl CounterFlowHindrance {
    /// Create with default parameters (gamma=2.0, scan_radius=3).
    pub fn new() -> Self {
        Self {
            gamma: 2.0,
            scan_radius: 3,
            goals: Vec::new(),
        }
    }

    /// Set the counter-flow penalty multiplier (default 2.0).
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the scan radius in cells (default 3).
    pub fn with_scan_radius(mut self, radius: usize) -> Self {
        self.scan_radius = radius as i32;
        self
    }

    /// Update the goal cache. Call this each tick before planning.
    ///
    /// The goals are used to compute each agent's direction vector for the
    /// counter-flow check. Without this call, the estimator returns zero
    /// (no penalty), which means it adds nothing beyond the base estimator.
    pub fn set_goals(&mut self, goals: &[GridPos]) {
        self.goals.clear();
        self.goals.extend_from_slice(goals);
    }
}

impl Default for CounterFlowHindrance {
    fn default() -> Self {
        Self::new()
    }
}

impl HindranceEstimator<GridPos> for CounterFlowHindrance {
    fn hindrance(
        &mut self,
        agent: AgentId,
        target: &GridPos,
        config: &super::config::JointConfig<GridPos>,
    ) -> f32 {
        // Base blocking count (paper-faithful).
        let me = usize::from(agent);
        let mut base_cost = 0u32;
        for (j, pos) in config.positions.iter().enumerate() {
            if j == me {
                continue;
            }
            for n in pos.neighbors() {
                if &n == target {
                    base_cost += 1;
                    break;
                }
            }
        }

        // Counter-flow penalty requires goals.
        if self.goals.is_empty() || self.gamma == 0.0 || me >= self.goals.len() {
            return base_cost as f32;
        }

        let my_pos = &config.positions[me];
        let my_goal = &self.goals[me];

        // My direction vector (sign of goal displacement).
        let my_dx = (my_goal.x as i32 - my_pos.x as i32).signum();
        let my_dy = (my_goal.y as i32 - my_pos.y as i32).signum();

        let r = self.scan_radius;
        let mut counter_flow = 0u32;

        for (j, pos) in config.positions.iter().enumerate() {
            if j == me || j >= self.goals.len() {
                continue;
            }
            let dx = pos.x as i32 - my_pos.x as i32;
            let dy = pos.y as i32 - my_pos.y as i32;
            if dx.abs() > r || dy.abs() > r {
                continue;
            }
            let j_goal = &self.goals[j];
            let j_dx = (j_goal.x as i32 - pos.x as i32).signum();
            let j_dy = (j_goal.y as i32 - pos.y as i32).signum();

            // Counter-flow: anti-aligned direction (dot product < 0).
            let dot = my_dx * j_dx + my_dy * j_dy;
            if dot < 0 {
                counter_flow += 1;
            }
        }

        base_cost as f32 + self.gamma * counter_flow as f32
    }
}
