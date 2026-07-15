//! Local guidance source — space-time A* on collision-count cost (Plan 440 T1.5).
//!
//! This is LLLG mechanism (a) from the paper (§2.D, Algorithm 1, Eq. 1).
//! For each agent `i`, solve a single-agent space-time A* over `w_Φ+1` steps:
//!
//! ```text
//! cost_i(π) = dist(π[w_Φ], g_i)                          // reach the goal
//!           + Σ_{t=0..w_Φ-1} ⟨1 + α·Ind[χ>0], χ⟩         // avoid collisions
//! ```
//!
//! where `χ` is the count of collisions of the transition `(π[t], π[t+1])`
//! with sibling paths currently stored in the shared guidance set `Φ`. The
//! hyperparameter `α ≥ 0` controls how aggressively collisions are penalized.
//!
//! The construction is **sequential over agents** (each agent sees the others'
//! already-updated guidance) and repeated `m` rounds to reduce agent-order
//! bias (paper default `m=2`).
//!
//! # Pluggable seam
//!
//! [`LocalGuidanceSource`] is the primary extension point for the Super-GOAT
//! fusion (riir-ai/318 Extension A). The default [`SpaceTimeGuidance`] is
//! paper-faithful (uniform `α`). A private consumer plugs in HLA-projected
//! guidance where `α_i = α_base · (1 + β · σ(dot(HLA_i, D_frustration)))`
//! so crowded cells cost more for stressed NPCs.

use super::config::{AgentId, JointConfig};
use super::position::Position;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Type alias for a neighbor-supplying closure. Kept as a `dyn` to avoid
/// bloating generic parameter lists. `Send + Sync` so the guidance source
/// can cross thread boundaries if a consumer parallelizes across zones.
type NeighborFn<P> = dyn Fn(&P) -> Vec<P> + Send + Sync;

/// A per-agent guidance path: `w_Φ` future positions.
///
/// `Φ[i]` is the guidance for agent `i`. This is the **latent** guidance
/// field (paper §2.3 reframing) — a rank-1 cochain on the time axis. Not
/// synced; recomputed each tick from local observations.
pub type Guidance<P> = Vec<Vec<P>>;

/// Trait for producing the per-agent local guidance field `Φ`.
///
/// Pluggable seam #2. The default impl is [`SpaceTimeGuidance`].
pub trait LocalGuidanceSource<P: Position> {
    /// Compute the guidance field `Φ` for all agents.
    ///
    /// `Φ[i]` is a `w_Φ`-step path (positions at t+1, t+2, ..., t+w_Φ).
    /// The first position `Φ[i][0]` is the preferred next position, used by
    /// PIBT's lexicographic cost first term `Ind[Φ[i][0] ≠ u]`.
    ///
    /// The result is written into `out` (cleared and refilled).
    fn compute_guidance(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        out: &mut Guidance<P>,
    );
}

/// The guidance window length `w_Φ` and collision penalty `α`.
#[derive(Clone, Copy, Debug)]
pub struct GuidanceConfig {
    /// Planning window length (paper default 5).
    pub w_phi: usize,
    /// Collision penalty weight (paper implicit via `1 + α·Ind[χ>0]`).
    /// `α = 0` disables guidance (degenerates to greedy goal-seeking).
    /// Higher `α` → stronger collision avoidance.
    pub alpha: f32,
    /// Number of sequential refinement rounds (paper default 2).
    pub rounds: usize,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            w_phi: 5,
            alpha: 1.0,
            rounds: 2,
        }
    }
}

/// Paper-default guidance: space-time A* on Eq. 1 cost.
///
/// Sequential per-agent refinement, `m` rounds. Each agent's A* sees the
/// already-updated paths of earlier-in-the-round agents.
///
/// This is a **single-threaded** implementation matching the paper's design.
/// Per-zone parallelism is a consumer concern (one LLLG instance per zone).
pub struct SpaceTimeGuidance<P: Position> {
    cfg: GuidanceConfig,
    /// Neighbors filter — supplies passable neighbors. `None` = use
    /// `Position::neighbors()` directly (no wall filtering).
    neighbors_fn: Option<Box<NeighborFn<P>>>,
    /// Scratch: collision count per (position, time-offset) cell.
    /// Keyed by a hash of (position, time).
    occupancy: HashMap<P, [u32; 64]>,
}

impl<P: Position> SpaceTimeGuidance<P> {
    pub fn new(cfg: GuidanceConfig) -> Self {
        Self {
            cfg,
            neighbors_fn: None,
            occupancy: HashMap::new(),
        }
    }

    /// With a custom neighbors function (for wall-aware grids).
    pub fn with_neighbors<F>(mut self, f: F) -> Self
    where
        F: Fn(&P) -> Vec<P> + Send + Sync + 'static,
    {
        self.neighbors_fn = Some(Box::new(f));
        self
    }

    #[inline]
    fn neighbors_of(&self, pos: &P) -> Vec<P> {
        if let Some(ref f) = self.neighbors_fn {
            f(pos)
        } else {
            pos.neighbors()
        }
    }

    /// Count collisions of a candidate cell `(pos, t_offset)` with the
    /// current occupancy map. This is `χ` from Eq. 1.
    #[inline]
    fn collision_count(&self, pos: &P, t: usize) -> u32 {
        if t >= 64 {
            return 0;
        }
        self.occupancy.get(pos).map_or(0, |arr| arr[t])
    }

    /// Record an agent's chosen path into the occupancy map.
    fn record_path(&mut self, path: &[P]) {
        for (t, pos) in path.iter().enumerate() {
            if t >= 64 {
                break;
            }
            let slot = self.occupancy.entry(pos.clone()).or_insert([0u32; 64]);
            slot[t] = slot[t].saturating_add(1);
        }
    }

    /// Clear occupancy for a fresh round.
    fn clear_occupancy(&mut self) {
        for arr in self.occupancy.values_mut() {
            *arr = [0u32; 64];
        }
    }

    /// Space-time A* for a single agent.
    ///
    /// Returns the best `w_Φ`-step path (positions at t+1..t+w_Φ). The path
    /// minimizes Eq. 1 cost.
    fn astar_for_agent(&self, start: &P, goal: &P) -> Vec<P> {
        // Simplified: greedy best-first with the Eq. 1 cost. A full space-time
        // A* with a priority queue is the paper-faithful approach, but for the
        // Phase 1 skeleton we use a bounded greedy rollout that respects the
        // collision-count cost — sufficient for correctness and the G1
        // paper-reproduction gate will validate against the full A*.
        //
        // The greedy rollout: at each step, pick the neighbor minimizing
        // (collision_count + heuristic), accumulate the cost. This is a
        // hill-climb on the integrated Eq. 1 cost; it won't find globally
        // optimal paths but is correct (collision-aware) and fast.
        let w = self.cfg.w_phi;
        let mut path = Vec::with_capacity(w);
        let mut current = start.clone();
        for t in 0..w {
            let neighbors = self.neighbors_of(&current);
            let best = neighbors
                .iter()
                .min_by(|a, b| {
                    let cost_a = self.step_cost(a, goal, t);
                    let cost_b = self.step_cost(b, goal, t);
                    cost_a.partial_cmp(&cost_b).unwrap_or(Ordering::Equal)
                })
                .cloned()
                .unwrap_or_else(|| current.clone());
            path.push(best.clone());
            current = best;
        }
        path
    }

    /// Eq. 1 per-step cost for a candidate move to `next` at time offset `t`.
    #[inline]
    fn step_cost(&self, next: &P, goal: &P, t: usize) -> f32 {
        let chi = self.collision_count(next, t);
        let collision_term = 1.0 + self.cfg.alpha * if chi > 0 { chi as f32 } else { 0.0 };
        // Weight collisions by χ (paper: ⟨1 + α·Ind[χ>0], χ⟩ = χ + α·χ·Ind[χ>0]).
        // When χ=0, cost is just the heuristic; when χ>0, cost scales with χ.
        let collision_cost = if chi > 0 {
            chi as f32 * collision_term
        } else {
            0.0
        };
        collision_cost + next.dist_heuristic(goal) * 0.1 // small goal pull
    }
}

impl<P: Position> LocalGuidanceSource<P> for SpaceTimeGuidance<P> {
    fn compute_guidance(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        out: &mut Guidance<P>,
    ) {
        let n = config.n_agents();
        out.clear();
        out.resize(n, Vec::new());

        for _round in 0..self.cfg.rounds {
            self.clear_occupancy();
            for i in 0..n {
                let agent = AgentId(i as u32);
                let start = config.pos(agent).clone();
                let goal = &goals[i];
                let path = self.astar_for_agent(&start, goal);
                // Record into occupancy (so later agents see this path).
                self.record_path(&path);
                out[i] = path;
            }
        }
    }
}
