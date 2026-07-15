//! Core types — joint configuration, agent ID, joint action (Plan 440 T1.3).

use super::position::Position;

/// Agent identifier (newtype wrapper over `u32`).
///
/// Indices into the joint configuration and per-agent vectors. Cheap to copy,
/// compare, hash. `u32` supports up to ~4 billion agents — far beyond the
/// paper's 10K ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct AgentId(pub u32);

impl From<u32> for AgentId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<AgentId> for usize {
    #[inline]
    fn from(a: AgentId) -> Self {
        a.0 as usize
    }
}

/// A joint configuration: one position per agent at a single time step.
///
/// This is the raw, synced ground truth (`Q_t` in the paper). Index `i`
/// corresponds to `AgentId(i)`.
///
/// # Latent vs raw
///
/// This is **raw** — bit-identical across deterministic-replay nodes per the
/// sync-boundary rule. Never encoded as a latent embedding.
#[derive(Clone, Debug)]
pub struct JointConfig<P: Position> {
    pub positions: Vec<P>,
}

impl<P: Position> JointConfig<P> {
    pub fn new(positions: Vec<P>) -> Self {
        Self { positions }
    }

    pub fn n_agents(&self) -> usize {
        self.positions.len()
    }

    #[inline]
    pub fn pos(&self, agent: AgentId) -> &P {
        &self.positions[usize::from(agent)]
    }

    #[inline]
    pub fn pos_mut(&mut self, agent: AgentId) -> &mut P {
        &mut self.positions[usize::from(agent)]
    }
}

impl<P: Position> std::ops::Index<AgentId> for JointConfig<P> {
    type Output = P;
    #[inline]
    fn index(&self, agent: AgentId) -> &P {
        &self.positions[usize::from(agent)]
    }
}

impl<P: Position> std::ops::IndexMut<AgentId> for JointConfig<P> {
    #[inline]
    fn index_mut(&mut self, agent: AgentId) -> &mut P {
        &mut self.positions[usize::from(agent)]
    }
}

/// The executed joint action for one tick: one next-position per agent.
///
/// This is `Π_t[1]` in the paper — the first step of the windowed plan, which
/// is the only step actually committed. It is **raw** and synced (the move
/// committed to the chain as a `TxDelta`).
///
/// Collision profile (see `pibt.rs` module docs for the full analysis):
///
/// - **Edge collisions (swaps) are prevented by construction.**
/// - **Vertex collisions are prevented on uncongested maps**, but CAN occur on
///   congested maps: when an agent is "stuck" (no collision-free move AND its
///   current cell is committed by a higher-priority agent), it is forced to
///   wait in place, producing a vertex collision. This is a deliberate
///   throughput tradeoff — the alternatives (all-wait fallback, recursive
///   priority inheritance) both collapse throughput (Issues 140, 143; see
///   `pibt.rs` module docs §"Why not recursive priority inheritance?"). The
///   real fix is LaCAM-level search escalation (Phase 5).
///
/// Consumers that require guaranteed collision-freedom should inspect the
/// returned action for vertex collisions (the bench harness at
/// `riir-ai/crates/riir-poc/benches/lllg_crowd_mcgs_goal_bridge_goat.rs`
/// shows the detection pattern) or use an occupied-set baseline.
#[derive(Clone, Debug)]
pub struct JointAction<P: Position> {
    pub moves: Vec<P>,
}

impl<P: Position> JointAction<P> {
    pub fn new(moves: Vec<P>) -> Self {
        Self { moves }
    }

    pub fn from_wait(config: &JointConfig<P>) -> Self {
        Self {
            moves: config.positions.clone(),
        }
    }

    /// Apply this action to a config, producing the next-tick config.
    ///
    /// Edge-collision-free by PIBT's swap check; vertex collisions may be
    /// present on congested maps (see the [`JointAction`] struct doc and the
    /// `pibt.rs` module docs for the tradeoff analysis). This fn does not
    /// re-check.
    pub fn apply_to(&self, config: &JointConfig<P>) -> JointConfig<P> {
        debug_assert_eq!(self.moves.len(), config.n_agents());
        JointConfig::new(self.moves.clone())
    }
}

/// A per-agent goal assignment.
///
/// In lifelong MAPF, when an agent reaches its goal it is immediately
/// reassigned a new one. The [`GoalAssignment`] trait abstracts the
/// reassignment policy (paper: uniform random over free cells).
pub trait GoalAssignment<P: Position> {
    /// Return the current goal for `agent`, assigning a new one if needed.
    ///
    /// Called once per tick per agent. If the agent is at its goal, a new goal
    /// is assigned (the assignment is stored internally).
    fn goal_for(&mut self, agent: AgentId, current_pos: &P) -> P;
}

/// Uniform-random goal assignment over a fixed set of candidate goals.
///
/// Paper-default behavior: when an agent reaches its goal, pick a new goal
/// uniformly at random from `candidates`. The RNG is deterministic (seeded),
/// preserving replay.
pub struct UniformGoals<P: Position> {
    candidates: Vec<P>,
    /// Current goal per agent.
    current: Vec<P>,
    rng: fastrand::Rng,
}

impl<P: Position> UniformGoals<P> {
    pub fn new(candidates: Vec<P>, n_agents: usize, seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut current = Vec::with_capacity(n_agents);
        for _ in 0..n_agents {
            let idx = rng.usize(0..candidates.len());
            current.push(candidates[idx].clone());
        }
        Self {
            candidates,
            current,
            rng,
        }
    }

    fn random_goal(&mut self) -> P {
        let idx = self.rng.usize(0..self.candidates.len());
        self.candidates[idx].clone()
    }

    /// The current goal for `agent` (without triggering reassignment).
    ///
    /// Test/inspection accessor — does not mutate state.
    pub fn current_goal(&self, agent: AgentId) -> Option<&P> {
        self.current.get(usize::from(agent))
    }
}

impl<P: Position> GoalAssignment<P> for UniformGoals<P> {
    fn goal_for(&mut self, agent: AgentId, current_pos: &P) -> P {
        let idx = usize::from(agent);
        if idx >= self.current.len() {
            let g = self.random_goal();
            // grow lazily
            while self.current.len() <= idx {
                self.current.push(g.clone());
            }
            return self.current[idx].clone();
        }
        // If at goal, reassign.
        if &self.current[idx] == current_pos {
            let g = self.random_goal();
            self.current[idx] = g.clone();
            g
        } else {
            self.current[idx].clone()
        }
    }
}
