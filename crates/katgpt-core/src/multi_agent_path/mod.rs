//! Lifelong LaCAM with Local Guidance (LLLG) — modelless multi-agent
//! pathfinding substrate (Plan 440, Research 424, arXiv:2605.16855).
//!
//! Distilled from Arita & Okumura (AIST, AAAI 2026). A purely heuristic,
//! training-free, receding-horizon multi-agent pathfinder that scales to
//! **10,000 agents at <1s/step** with higher throughput than RHCR in dense
//! settings.
//!
//! # The four pluggable seams (Super-GOAT hooks)
//!
//! This substrate is generic over four mechanisms so a private consumer
//! (riir-ai/318) can fuse it with HLA, Crowd MCGS, and the warm-path stack
//! without forking:
//!
//! | Seam | Trait/enum | Default (paper) | Pluggable alternative |
//! |---|---|---|---|
//! | Cost function | [`CostFn<P>`] | Uniform (1/move) | Heightfield slope, threat cochain, faction zone penalty |
//! | Guidance source | [`LocalGuidanceSource<P>`] | Space-time A* on collision count | HLA-projected guidance (per-NPC emotional congestion avoidance) |
//! | Warm-start scheme | [`WarmStartScheme`] | `LllgPi` (prev solution suffix) | Personality-weighted blend |
//! | Hindrance estimator | [`HindranceEstimator<P>`] | Raw blocking count | Affect-aware blocking (fearful NPCs count more) |
//!
//! A consumer that uses all four defaults gets the paper's LLLG verbatim.
//!
//! # Modelless mandate
//!
//! Entirely heuristic — no training, no backprop, no gradient descent. The
//! only weight mutations are freeze/thaw (swapping frozen snapshots), which
//! is not used here at all. Promotion to default-on is allowed once G1–G4
//! pass (the substrate is modelless).
//!
//! # Latent vs raw boundary
//!
//! Per AGENTS.md sync-boundary rule:
//! - **Raw (synced):** joint configuration `Q_t`, executed joint action `Π_t[1]`.
//! - **Latent (local):** guidance field `Φ`, hindrance scalars, warm-start cache.
//! - **Bridge:** `Φ → Π_t[1]` — latent guidance selects the raw action.
//!
//! See Research 424 §2.5 for the full table.
//!
//! # Example
//!
//! ```no_run
//! use katgpt_core::multi_agent_path::*;
//! use katgpt_core::multi_agent_path::position::*;
//!
//! // 10×10 grid, 2 agents.
//! let map = GridMap::empty(10, 10);
//! let starts = vec![GridPos::new(0, 0), GridPos::new(9, 9)];
//! let config = JointConfig::new(starts);
//! let goals = vec![GridPos::new(9, 9), GridPos::new(0, 0)];
//!
//! let guidance_cfg = GuidanceConfig::default();
//! let mut guidance = SpaceTimeGuidance::new(guidance_cfg)
//!     .with_neighbors(|p| map.passable_neighbors(p));
//! let mut hindrance = BlockingCount::new();
//! let mut warm_start = WarmStartCache::new(WarmStartScheme::default(), guidance_cfg.w_phi);
//! let mut rng = fastrand::Rng::with_seed(42);
//!
//! let mut lacam = LifelongLaCam::new(warm_start);
//! let action = lacam.tick(&config, &goals, &mut guidance, &mut hindrance, &mut rng);
//! ```

#![allow(clippy::too_many_arguments)]

pub mod config;
pub mod hindrance;
pub mod local_guidance;
pub mod pibt;
pub mod position;
pub mod warm_start;

#[cfg(test)]
mod tests;

pub use config::{AgentId, GoalAssignment, JointAction, JointConfig, UniformGoals};
pub use hindrance::{BlockingCount, HindranceEstimator, WeightedBlockingCount};
pub use local_guidance::{
    Guidance, GuidanceConfig, LocalGuidanceSource, SpaceTimeGuidance,
};
pub use pibt::{pibt_step, Deadlock};
pub use position::{soft_cost, GridMap, GridPos, Position};
pub use warm_start::{WarmStartCache, WarmStartScheme};

// ─────────────────────────────────────────────────────────────────────
// CostFn trait — pluggable seam #1 (Plan 440 T1.2)
// ─────────────────────────────────────────────────────────────────────

/// Transition cost function — pluggable seam #1.
///
/// Returns the raw cost of transitioning from `from` to `to` in one step.
/// The default [`UniformCost`] returns 1.0 for any move (paper default).
///
/// # Extension points (private consumer, riir-ai/318)
///
/// A consumer's impl may incorporate:
/// - **Heightfield slope** — `cost = 1 + sigmoid(slope · β)` so uphill moves
///   cost more (the raw→latent bridge via [`soft_cost`]).
/// - **Threat cochain** — read the DEC codifferential of the threat field at
///   `to` and add it to the base cost.
/// - **Faction zone penalty** — higher cost in enemy territory.
/// - **Economy toll** — path through a toll gate costs gold.
///
/// All of these are modelless (closed-form, no training).
pub trait CostFn<P: Position> {
    /// Cost of moving from `from` to `to`. Must be ≥ 0.
    fn cost(&self, from: &P, to: &P) -> f32;
}

/// Paper-default cost: 1.0 per move (uniform).
pub struct UniformCost;

impl Default for UniformCost {
    fn default() -> Self {
        Self
    }
}

impl<P: Position> CostFn<P> for UniformCost {
    #[inline]
    fn cost(&self, _from: &P, _to: &P) -> f32 {
        1.0
    }
}

// ─────────────────────────────────────────────────────────────────────
// Orchestrator (Plan 440 T1.2)
// ─────────────────────────────────────────────────────────────────────

/// Type alias for the neighbor-supplying closure stored in the orchestrator.
type NeighborClosure<P> = Box<dyn Fn(&P) -> Vec<P> + Send + Sync>;

/// The LLLG orchestrator: one tick of receding-horizon windowed planning.
///
/// Generic over the position type `P`. Holds the warm-start cache and the
/// guidance config. The guidance source, hindrance estimator, and RNG are
/// passed by `&mut` to [`tick`](Self::tick) so they can be reused across
/// ticks without cloning.
///
/// # Lifecycle
///
/// 1. Construct once per zone (or per crowd) with the desired config.
/// 2. Call [`tick`](Self::tick) each game tick with the current config + goals.
/// 3. The returned [`JointAction`] is the collision-free first step.
pub struct LifelongLaCam<P: Position> {
    warm_start: WarmStartCache<P>,
    /// Scratch: per-agent guidance field `Φ`.
    guidance_scratch: Guidance<P>,
    /// Scratch: priority weights (uniform by default).
    priorities: Vec<f32>,
    /// Wall-aware neighbor function. When `None`, PIBT uses `Position::neighbors()`
    /// directly (no wall/bounds checking). Consumers with walls or bounded maps
    /// MUST set this via [`with_neighbors`](Self::with_neighbors).
    neighbors_fn: Option<NeighborClosure<P>>,
}

impl<P: Position> LifelongLaCam<P> {
    /// Construct with the given warm-start cache.
    ///
    /// The guidance config is owned by the [`LocalGuidanceSource`] you pass to
    /// [`tick`](Self::tick); the orchestrator does not hold a separate copy.
    /// Use [`with_neighbors`](Self::with_neighbors) to set a wall-aware neighbor
    /// function if your map has walls or bounds.
    pub fn new(warm_start: WarmStartCache<P>) -> Self {
        Self {
            warm_start,
            guidance_scratch: Vec::new(),
            priorities: Vec::new(),
            neighbors_fn: None,
        }
    }

    /// Set a wall-aware neighbor function.
    ///
    /// When set, PIBT uses this instead of `Position::neighbors()` to generate
    /// candidate moves, ensuring agents never move through walls or out of
    /// bounds. The guidance source should be configured independently with the
    /// same wall-aware function.
    pub fn with_neighbors<F>(mut self, f: F) -> Self
    where
        F: Fn(&P) -> Vec<P> + Send + Sync + 'static,
    {
        self.neighbors_fn = Some(Box::new(f));
        self
    }

    /// Set per-agent priorities (higher = processed first by PIBT).
    ///
    /// Empty = uniform (index order). Length must match `config.n_agents()`.
    pub fn set_priorities(&mut self, priorities: Vec<f32>) {
        self.priorities = priorities;
    }

    /// One tick of LLLG planning.
    ///
    /// The full pipeline per tick:
    /// 1. Compute the warm-start initialization from the previous tick's data.
    /// 2. Pass it to the guidance source via [`set_warm_start`](LocalGuidanceSource::set_warm_start).
    /// 3. Compute the guidance field `Φ` via the guidance source (pluggable).
    /// 4. Run PIBT to produce the collision-free joint action `Π_t[1]`.
    /// 5. Record the executed action + `Φ` into the warm-start cache for the
    ///    next tick.
    ///
    /// # Arguments
    ///
    /// - `config`: current joint configuration `Q_t` (raw, synced).
    /// - `goals`: per-agent goals `g_i` (raw).
    /// - `guidance`: the guidance source (pluggable seam #2).
    /// - `hindrance`: the hindrance estimator (pluggable seam #4).
    /// - `rng`: deterministic RNG for the PIBT `ε` tiebreak.
    ///
    /// # Returns
    ///
    /// The collision-free [`JointAction`] `Π_t[1]`.
    pub fn tick<G, H>(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        guidance: &mut G,
        hindrance: &mut H,
        rng: &mut fastrand::Rng,
    ) -> JointAction<P>
    where
        G: LocalGuidanceSource<P>,
        H: HindranceEstimator<P>,
    {
        // 1. Produce warm-start initialization for this tick.
        let warm = self.warm_start.warm_start();

        // 2. Pass it to the guidance source. The source consumes it inside
        //    `compute_guidance` (default sources ignore it via the trait's
        //    no-op `set_warm_start`).
        if !warm.is_empty() {
            guidance.set_warm_start(warm);
        }

        // 3. Compute guidance field Φ.
        guidance.compute_guidance(config, goals, &mut self.guidance_scratch);

        // 4. Run PIBT with wall-aware neighbors (if set).
        let action = pibt_step(
            config,
            &self.guidance_scratch,
            goals,
            &self.priorities,
            hindrance,
            self.neighbors_fn.as_deref(),
            rng,
        )
        .unwrap_or_else(|deadlock| {
            // Lifelong MAPF tolerates stalls: stuck agents wait.
            log::debug!(
                "LLLG deadlock: {} agents stuck, falling back to wait",
                deadlock.stuck_agents.len()
            );
            JointAction::from_wait(config)
        });

        // 5. Record the executed action + Φ for the next tick's warm-start.
        //    The "solution" prepends the executed PIBT action so the suffix
        //    extraction in `WarmStartCache::prev_solution_suffix` correctly
        //    skips the executed step (not the guidance's preferred first
        //    step, which may differ from the executed action — Issue 140 T2.6).
        let n = config.n_agents();
        let mut solution: Vec<Vec<P>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut path = Vec::with_capacity(self.guidance_scratch[i].len() + 1);
            path.push(action.moves[i].clone());
            path.extend(self.guidance_scratch[i].iter().cloned());
            solution.push(path);
        }
        self.warm_start
            .record(solution, self.guidance_scratch.clone());

        action
    }

    /// Access the warm-start cache (for scheme changes or inspection).
    pub fn warm_start_mut(&mut self) -> &mut WarmStartCache<P> {
        &mut self.warm_start
    }
}

impl<P: Position> Default for LifelongLaCam<P> {
    fn default() -> Self {
        let w_phi = GuidanceConfig::default().w_phi;
        Self::new(WarmStartCache::new(WarmStartScheme::default(), w_phi))
    }
}
