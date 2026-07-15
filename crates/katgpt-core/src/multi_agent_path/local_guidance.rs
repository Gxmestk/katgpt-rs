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
use std::collections::{HashMap, VecDeque};

/// Maximum number of cached BFS distance fields before the cache is cleared.
/// Bounds memory to ~MAX × map_size × sizeof(GridPos + f32). For 800 agents
/// on a 4096-cell map, this is ~4000 × 4096 × 20 bytes ≈ 320MB worst case
/// (rarely reached — goals cluster as agents converge).
const MAX_BFS_CACHE_ENTRIES: usize = 4000;

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
    /// BFS distance cache: goal → {position → true BFS distance}.
    ///
    /// This is the critical upgrade that fixes throughput on obstacle-heavy
    /// maps. The original Phase 1 greedy used Manhattan distance (the
    /// `Position::dist_heuristic` trait method) which ignores walls and leads
    /// agents into dead ends. BFS distance gives the true shortest path
    /// length around obstacles, so the greedy rollout follows the correct
    /// gradient.
    ///
    /// Cleared at the start of each `compute_guidance` call (within-tick
    /// reuse across agents that share a goal; across-tick recomputation).
    bfs_cache: HashMap<P, HashMap<P, f32>>,
}

impl<P: Position> SpaceTimeGuidance<P> {
    pub fn new(cfg: GuidanceConfig) -> Self {
        Self {
            cfg,
            neighbors_fn: None,
            occupancy: HashMap::new(),
            bfs_cache: HashMap::new(),
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
    #[allow(dead_code)]
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
    ///
    /// Phase 2 upgrade: uses BFS distance fields for the goal heuristic
    /// (fixes the Phase 1 bug where Manhattan distance led agents into dead
    /// ends on obstacle-heavy maps). The greedy rollout follows the BFS
    /// gradient toward the goal while avoiding collisions via the occupancy
    /// map.
    fn astar_for_agent(&mut self, start: &P, goal: &P) -> Vec<P> {
        // Compute BFS field first, then clone it out of self to avoid borrow conflict
        // with the occupancy lookups in the rollout below.
        let bfs = self.bfs_distance_field(goal).clone();
        let w = self.cfg.w_phi;
        let alpha = self.cfg.alpha;
        let mut path = Vec::with_capacity(w);
        let mut current = start.clone();
        let mut visited: HashMap<P, u32> = HashMap::new();
        for t in 0..w {
            let neighbors = self.neighbors_of(&current);
            // Capture only the values we need from self to avoid borrow conflicts.
            let occupancy = &self.occupancy;
            let best = neighbors
                .iter()
                .min_by(|a, b| {
                    let cost_a = step_cost_bfs(*a, &bfs, occupancy, alpha, t)
                        + cycle_penalty(*a, &visited, w);
                    let cost_b = step_cost_bfs(*b, &bfs, occupancy, alpha, t)
                        + cycle_penalty(*b, &visited, w);
                    cost_a.partial_cmp(&cost_b).unwrap_or(Ordering::Equal)
                })
                .cloned()
                .unwrap_or_else(|| current.clone());
            *visited.entry(best.clone()).or_insert(0) += 1;
            path.push(best.clone());
            current = best;
        }
        path
    }

    /// Compute or retrieve the BFS distance field from `goal`.
    ///
    /// BFS flood-fills from the goal outward through wall-aware neighbors,
    /// giving the true shortest-path distance from every reachable cell to
    /// the goal. This replaces Manhattan distance as the navigation heuristic
    /// and is the key fix for obstacle-heavy maps.
    ///
    /// Cached within a single `compute_guidance` call (multiple agents sharing
    /// the same goal reuse the same field).
    fn bfs_distance_field(&mut self, goal: &P) -> &HashMap<P, f32> {
        if !self.bfs_cache.contains_key(goal) {
            let field = self.compute_bfs(goal);
            self.bfs_cache.insert(goal.clone(), field);
        }
        &self.bfs_cache[goal]
    }

    /// BFS flood-fill from `goal` to all reachable cells.
    fn compute_bfs(&self, goal: &P) -> HashMap<P, f32> {
        let mut dist: HashMap<P, f32> = HashMap::new();
        let mut queue: VecDeque<P> = VecDeque::new();
        dist.insert(goal.clone(), 0.0);
        queue.push_back(goal.clone());
        while let Some(current) = queue.pop_front() {
            let d = dist[&current];
            for neighbor in self.neighbors_of(&current) {
                if !dist.contains_key(&neighbor) {
                    dist.insert(neighbor.clone(), d + 1.0);
                    queue.push_back(neighbor);
                }
            }
        }
        dist
    }
}

/// Eq. 1 per-step cost using BFS distance (free function to avoid borrow conflicts).
///
/// Replaces the Manhattan-distance heuristic with true BFS distance to
/// the goal. This is the difference that fixes obstacle-heavy maps.
#[inline]
fn step_cost_bfs<P: Position>(
    next: &P,
    bfs: &HashMap<P, f32>,
    occupancy: &HashMap<P, [u32; 64]>,
    alpha: f32,
    t: usize,
) -> f32 {
    let chi = if t >= 64 {
        0
    } else {
        occupancy.get(next).map_or(0, |arr| arr[t])
    };
    let collision_cost = if chi > 0 {
        chi as f32 * (1.0 + alpha)
    } else {
        0.0
    };
    let goal_dist = bfs.get(next).copied().unwrap_or(f32::MAX);
    collision_cost + goal_dist
}

/// Anti-cycling penalty (free function to avoid borrow conflicts).
#[inline]
fn cycle_penalty<P: Position>(pos: &P, visited: &HashMap<P, u32>, w: usize) -> f32 {
    visited.get(pos).map_or(0.0, |&n| n as f32 * w as f32)
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

        // BFS cache: persistent across ticks for amortization. Cleared only
        // when it exceeds MAX_BFS_CACHE_ENTRIES (to bound memory). This gives
        // ~10× amortization since most agents' goals persist for many ticks.
        if self.bfs_cache.len() > MAX_BFS_CACHE_ENTRIES {
            self.bfs_cache.clear();
        }

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
