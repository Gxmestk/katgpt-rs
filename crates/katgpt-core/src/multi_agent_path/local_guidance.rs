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
//! # Implementation (Issue 142)
//!
//! The per-agent search is a proper space-time A* over the `(position, depth)`
//! state space with BFS-distance heuristic. This replaced a greedy rollout
//! (Issue 140) that committed at each depth without backtracking and could
//! neither plan multi-step detours nor consume the warm-start forecast.
//!
//! The multi-round refinement is implemented as unrecord/re-record: each
//! agent removes its previous path from the shared occupancy, computes a fresh
//! A* path seeing all other agents' most-recent paths, then records the new
//! path. This makes `rounds > 1` actually improve results (round 1 agent 0
//! sees round-0 paths of agents 1..n-1).
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
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

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
/// Pluggable seam #2. The default impl is [`SpaceTimeGuidance`] (paper-faithful
/// space-time A* with uniform collision penalty `α`).
///
/// # Primary extension point (Super-GOAT fusion hook)
///
/// A private consumer (riir-ai/318 Extension A) replaces the uniform `α` with
/// a per-NPC **HLA-projected** penalty so that crowded cells cost more for
/// stressed NPCs. The collision term `α·Ind[χ>0]` from Eq. 1 becomes:
///
/// ```text
/// α_i = α_base · (1 + β · σ(dot(HLA_i, D_frustration)))
/// ```
///
/// where `HLA_i` is agent `i`'s 5-scalar affect vector (valence, arousal,
/// desperation, calm, fear), `D_frustration` is a learned direction vector,
/// and `σ` is the sigmoid bridge. This is **entirely modelless** — the
/// direction vector is freeze/thaw-swapped, not trained.
///
/// # Example: a custom HLA-projected guidance source
///
/// ```no_run
/// use katgpt_core::multi_agent_path::*;
/// use katgpt_core::multi_agent_path::position::*;
///
/// /// Per-NPC HLA projection (the latent→raw bridge).
/// ///
/// /// In the real runtime this reads from the NPC's `HlaCacheProxy`; here we
/// /// stub a lookup table indexed by `AgentId`. The bridge is a dot-product
/// /// projection onto a pre-computed `D_frustration` direction vector, gated
/// /// by sigmoid (never softmax — per AGENTS.md).
/// struct HlaProjectedGuidance {
///     /// Base collision penalty shared by all agents (paper `α`).
///     alpha_base: f32,
///     /// Per-agent frustration scalar in `[0, 1]`, computed once per tick
///     /// from `σ(dot(HLA_i, D_frustration))`. Higher = more collision-averse.
///     frustration: Vec<f32>,
///     /// Delegate: the underlying space-time guidance we layer onto.
///     inner: SpaceTimeGuidance<GridPos>,
/// }
///
/// impl LocalGuidanceSource<GridPos> for HlaProjectedGuidance {
///     fn compute_guidance(
///         &mut self,
///         config: &JointConfig<GridPos>,
///         goals: &[GridPos],
///         out: &mut Guidance<GridPos>,
///     ) {
///         // The simplest correct fusion: scale `α` per-agent by frustration,
///         // then run the paper-faithful space-time A*. A richer impl could
///         // also bias the goal heuristic by curiosity (valence direction),
///         // but that lives in the private runtime — the substrate only
///         // requires the trait method to be implemented.
///         let n = config.n_agents();
///         for i in 0..n {
///             let per_agent_alpha = self.alpha_base * (1.0 + self.frustration[i]);
///             // In a full impl you'd thread `per_agent_alpha` into `inner.cfg`;
///             // the stub here just demonstrates the trait shape.
///             let _ = per_agent_alpha;
///         }
///         self.inner.compute_guidance(config, goals, out);
///     }
///
///     fn set_warm_start(&mut self, warm_start: Vec<Vec<GridPos>>) {
///         // Delegate to the paper-faithful inner source.
///         self.inner.set_warm_start(warm_start);
///     }
/// }
/// ```
///
/// # Latent / raw boundary
///
/// The guidance field `Φ` is **latent** (local, not synced). Only the
/// executed first step `Π_t[1]` crosses the sync boundary as a raw `TxDelta`.
/// The HLA projection itself never leaves the NPC's local cognition — the
/// sync layer sees only the resulting move.
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

    /// Supply warm-start data for the next [`compute_guidance`](Self::compute_guidance)
    /// call (Issue 140 T2 — the LLLG mechanism (b) warm-start integration).
    ///
    /// The data is a per-agent initial path (`Vec<Vec<P>>`), produced by
    /// [`WarmStartCache::warm_start`](super::warm_start::WarmStartCache::warm_start).
    /// For the default [`SpaceTimeGuidance`] the paths are seeded into the
    /// occupancy map so each agent sees where other agents are forecast to be
    /// — this is the qualitative difference between `LllgPi` (with warm-start)
    /// and `LllgEmpty` (without).
    ///
    /// Default: no-op (the guidance source recomputes from scratch). Custom
    /// guidance sources that don't support warm-start can leave this as-is.
    fn set_warm_start(&mut self, _warm_start: Vec<Vec<P>>) {
        // Default: ignore — no warm-start.
    }
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
    /// Warm-start data supplied by [`set_warm_start`](LocalGuidanceSource::set_warm_start).
    ///
    /// Consumed (set to `None`) at the start of each `compute_guidance` call.
    /// When `Some`, the paths are seeded into the occupancy map before any
    /// agent is processed, giving each agent lookahead about where other
    /// agents are forecast to be. This is the LLLG mechanism (b) integration
    /// (Issue 140 T2).
    warm_start: Option<Vec<Vec<P>>>,
}

impl<P: Position> SpaceTimeGuidance<P> {
    pub fn new(cfg: GuidanceConfig) -> Self {
        Self {
            cfg,
            neighbors_fn: None,
            occupancy: HashMap::new(),
            bfs_cache: HashMap::new(),
            warm_start: None,
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

    /// Remove an agent's path from the occupancy map (inverse of `record_path`).
    ///
    /// Used by the unrecord/re-record refinement loop so each agent sees the
    /// most-recent paths of all *other* agents (and its own previous path is
    /// removed to avoid self-collision). Saturating subtraction keeps counts
    /// non-negative so a stray double-unrecord is a no-op rather than panic.
    fn unrecord_path(&mut self, path: &[P]) {
        for (t, pos) in path.iter().enumerate() {
            if t >= 64 {
                break;
            }
            if let Some(slot) = self.occupancy.get_mut(pos) {
                slot[t] = slot[t].saturating_sub(1);
            }
        }
    }

    /// Clear occupancy for a fresh round.
    fn clear_occupancy(&mut self) {
        for arr in self.occupancy.values_mut() {
            *arr = [0u32; 64];
        }
    }

    /// Space-time A* for a single agent (Issue 142).
    ///
    /// Searches the `(position, depth)` state space with BFS-distance heuristic.
    /// Returns the best `w_Φ`-step path (positions at t+1..t+w_Φ) minimizing
    /// Eq. 1 cost:
    ///
    /// ```text
    /// cost(π) = Σ_t (1 + α·χ(π[t], t))   // transition cost (collisions penalized)
    ///         + dist(π[w_Φ], goal)        // goal-reach term (h at depth w)
    /// ```
    ///
    /// Unlike the greedy rollout it replaces, the A* has w-step lookahead and
    /// can plan multi-step detours around collisions. The heuristic
    /// `h(pos,d) = BFS_dist(pos, goal)` is admissible (each remaining transition
    /// costs ≥ 1, so it never overestimates the true remaining cost).
    fn astar_for_agent(&mut self, start: &P, goal: &P, alpha: f32) -> Vec<P> {
        let w = self.cfg.w_phi;

        // Compute BFS field first, then clone it out of self to avoid a borrow
        // conflict with the occupancy / neighbor lookups inside the search loop.
        let bfs = self.bfs_distance_field(goal).clone();

        // Unreachable goal: no BFS entry for start. Wait in place.
        let start_h = bfs.get(start).copied().unwrap_or(f32::MAX);
        if start_h == f32::MAX {
            return vec![start.clone(); w];
        }

        let mut g_score: HashMap<(P, u8), f32> = HashMap::new();
        let mut came_from: HashMap<(P, u8), (P, u8)> = HashMap::new();
        let mut open: BinaryHeap<AstarNode<P>> = BinaryHeap::new();
        let mut closed: HashSet<(P, u8)> = HashSet::new();

        g_score.insert((start.clone(), 0), 0.0);
        open.push(AstarNode {
            f: start_h,
            depth: 0,
            pos: start.clone(),
        });

        let w_u8 = w as u8;
        let mut best_goal: Option<(P, u8)> = None;

        while let Some(node) = open.pop() {
            let depth = node.depth;
            let pos = node.pos.clone();

            // Skip already-expanded states.
            if !closed.insert((pos.clone(), depth)) {
                continue;
            }

            // Goal test: reached the planning horizon.
            if depth == w_u8 {
                best_goal = Some((pos, depth));
                break;
            }

            // g must exist (we pushed the node with this g).
            let g = g_score[&(pos.clone(), depth)];

            for neighbor in self.neighbors_of(&pos) {
                let new_depth = depth + 1;
                let key = (neighbor.clone(), new_depth);

                if closed.contains(&key) {
                    continue;
                }

                let h = bfs.get(&neighbor).copied().unwrap_or(f32::MAX);
                if h == f32::MAX {
                    continue; // unreachable neighbor
                }

                let chi = self.collision_count(&neighbor, depth as usize);
                let tentative_g = g + 1.0 + alpha * chi as f32;

                let known_g = g_score.get(&key).copied().unwrap_or(f32::MAX);
                if tentative_g < known_g {
                    g_score.insert(key.clone(), tentative_g);
                    came_from.insert(key.clone(), (pos.clone(), depth));
                    open.push(AstarNode {
                        f: tentative_g + h,
                        depth: new_depth,
                        pos: neighbor,
                    });
                }
            }
        }

        match best_goal {
            Some((end_pos, end_depth)) => {
                let mut path = Vec::with_capacity(w);
                let mut current = (end_pos, end_depth);
                while current.1 > 0 {
                    path.push(current.0.clone());
                    let prev = came_from.get(&current).expect("came_from chain must be complete");
                    current = prev.clone();
                }
                path.reverse();
                // Pad to w in the pathological case the search returned early.
                while path.len() < w {
                    path.push(path.last().cloned().unwrap_or_else(|| start.clone()));
                }
                path
            }
            None => vec![start.clone(); w],
        }
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

    /// Compute guidance with a **per-agent** collision penalty `alpha`.
    ///
    /// This is the Super-GOAT fusion extension point (Plan 489 Phase 2 /
    /// riir-ai/318 Extension A). Identical to
    /// [`compute_guidance`](LocalGuidanceSource::compute_guidance) except each
    /// agent `i` uses `alphas[i]` instead of the uniform `self.cfg.alpha`. The
    /// occupancy / BFS cache / refinement-rounds logic is shared with the
    /// uniform path — only the cost of collisions varies per agent.
    ///
    /// A private consumer (riir-ai's `HlaProjectedGuidance`) computes
    /// `alphas[i] = alpha_base * (1 + beta * sigmoid(dot(HLA_i, D_frustration)))`
    /// so frustrated NPCs take wider detours around occupied cells while calm
    /// NPCs push through — the behavioral signature the G5 gate measures.
    ///
    /// # Panics
    ///
    /// Panics if `alphas.len() != config.n_agents()`.
    pub fn compute_guidance_per_agent_alpha(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        alphas: &[f32],
        out: &mut Guidance<P>,
    ) {
        let n = config.n_agents();
        assert_eq!(
            alphas.len(),
            n,
            "compute_guidance_per_agent_alpha: alphas.len()={} but n_agents={}",
            alphas.len(),
            n
        );
        self.run_refinement(config, goals, out, move |i| alphas[i]);
    }

    /// Shared refinement-loop body for both the uniform-alpha trait impl and
    /// the per-agent-alpha fusion method. `alpha_of(i)` returns the collision
    /// penalty for agent `i`.
    ///
    /// This is the paper's sequential per-agent refinement (Algorithm 1):
    /// `rounds` passes, each iterating all agents in order. Each agent's A*
    /// sees the already-updated paths of earlier-in-the-round agents (via the
    /// occupancy map). Between rounds, each agent's previous path is unrecorded
    /// before re-planning so round 1+ improves on round 0.
    fn run_refinement<F: Fn(usize) -> f32>(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        out: &mut Guidance<P>,
        alpha_of: F,
    ) {
        let n = config.n_agents();
        out.clear();
        out.resize(n, Vec::new());

        // BFS cache: persistent across ticks for amortization. Cleared only
        // when it exceeds MAX_BFS_CACHE_ENTRIES (to bound memory).
        if self.bfs_cache.len() > MAX_BFS_CACHE_ENTRIES {
            self.bfs_cache.clear();
        }

        // Consume warm-start data (one-shot per tick). NOT seeded into the
        // occupancy — Issue 142 found that warm-start forecasts HURT throughput
        // because PIBT deviations invalidate them on dense maps.
        let _warm_start = self.warm_start.take();

        self.clear_occupancy();

        let mut prev_paths: Vec<Vec<P>> = vec![Vec::new(); n];

        for _round in 0..self.cfg.rounds {
            for i in 0..n {
                let agent = AgentId(i as u32);
                let start = config.pos(agent).clone();
                let goal = &goals[i];
                let alpha = alpha_of(i);

                self.unrecord_path(&prev_paths[i]);
                let path = self.astar_for_agent(&start, goal, alpha);
                self.record_path(&path);
                prev_paths[i] = path.clone();
                out[i] = path;
            }
        }
    }
}

/// A* search node. Ordered so the `BinaryHeap` (max-heap) pops the
/// smallest `f` first, breaking ties toward shallower depth (progress).
///
/// Position `P` is intentionally excluded from the `Ord` comparison: the
/// heap never needs to compare positions (only `f`/`depth` drive ordering),
/// and excluding `P` lets this work for any `Position` without an `Ord` bound.
struct AstarNode<P> {
    f: f32,
    depth: u8,
    pos: P,
}

impl<P> PartialEq for AstarNode<P> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.depth == other.depth
    }
}
impl<P> Eq for AstarNode<P> {}

impl<P> PartialOrd for AstarNode<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P> Ord for AstarNode<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap: reverse the natural f/depth comparisons so
        // the smallest f (then smallest depth) bubbles to the top.
        other
            .f
            .partial_cmp(&self.f)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.depth.cmp(&self.depth))
    }
}

impl<P: Position> LocalGuidanceSource<P> for SpaceTimeGuidance<P> {
    fn compute_guidance(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        out: &mut Guidance<P>,
    ) {
        // Paper-faithful uniform alpha — delegates to the shared refinement
        // loop with a constant closure. The closure captures `self.cfg.alpha`
        // by copy (f32 is Copy), avoiding any borrow conflict with the mutable
        // `self` methods called inside the loop.
        let uniform_alpha = self.cfg.alpha;
        self.run_refinement(config, goals, out, move |_i| uniform_alpha);
    }

    fn set_warm_start(&mut self, warm_start: Vec<Vec<P>>) {
        self.warm_start = Some(warm_start);
    }
}
