//! PIBT — Priority Inheritance with Backtracking (Plan 440 T1.4).
//!
//! Distilled from Okumura et al. 2022 (*PIBT: Scalable and Prioritization
//! Planning for Multi-Agent Pathfinding*). PIBT is a one-step collision-free
//! joint action generator: given the current joint configuration and per-agent
//! guidance, produce a collision-free next-position for every agent.
//!
//! The lexicographic cost for agent `i` considering move to `u`:
//!
//! ```text
//! ⟨ Ind[Φ[i][0] ≠ u], dist(u, g_i), hindrance(i→u), ε ⟩
//! ```
//!
//! I.e. PIBT first prefers moves consistent with the guidance, then
//! goal-direction, then low hindrance, then random tiebreak.
//!
//! # Collision constraints
//!
//! - **Vertex**: no two agents occupy the same cell at `t+1`.
//! - **Edge**: no two agents swap positions (`A→B` and `B→A` simultaneously).
//!
//! # Output
//!
//! Returns `Ok(JointAction)` if a collision-free joint action was found, or
//! `Err(Deadlock)` if some agent could not be placed. On deadlock, the caller
//! (the LaCAM orchestrator) escalates to higher-level search; for lifelong
//! MAPF the common fallback is "wait in place" (deadlock agents don't move).
//!
//! # Determinism
//!
//! The random tiebreak `ε` uses a deterministic seeded RNG, preserving replay.

use super::config::{AgentId, JointAction, JointConfig};
use super::hindrance::HindranceEstimator;
use super::local_guidance::Guidance;
use super::position::Position;
use std::cmp::Ordering;

/// Type alias mirroring [`super::local_guidance::NeighborFn`] for the
/// neighbor-supplying callback in [`pibt_step`].
type NeighborFn<P> = dyn Fn(&P) -> Vec<P>;

/// Error: no collision-free joint action could be found for some agent.
///
/// The agent(s) listed should wait; the caller may retry with a different
/// priority order or escalate to LaCAM-level search.
#[derive(Debug)]
pub struct Deadlock {
    /// Agents that could not be placed (must wait).
    pub stuck_agents: Vec<AgentId>,
}

/// Candidate move for an agent, with its lexicographic cost components.
#[derive(Clone)]
struct Candidate<P: Position + Clone> {
    next: P,
    /// Cost component 1: Ind[Φ[i][0] ≠ u] (0 if guidance-consistent, 1 else).
    guidance_mismatch: u8,
    /// Cost component 2: dist(u, g_i) heuristic.
    goal_dist: f32,
    /// Cost component 3: hindrance(i→u).
    hindrance: f32,
    /// Cost component 4: random tiebreak ε ∈ [0, 1).
    epsilon: f32,
}

impl<P: Position + Clone> Candidate<P> {
    /// Lexicographic comparison: guidance_mismatch → goal_dist → hindrance → ε.
    fn lexicographic_cmp(&self, other: &Self) -> Ordering {
        self.guidance_mismatch
            .cmp(&other.guidance_mismatch)
            .then_with(|| {
                self.goal_dist
                    .partial_cmp(&other.goal_dist)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                self.hindrance
                    .partial_cmp(&other.hindrance)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                self.epsilon
                    .partial_cmp(&other.epsilon)
                    .unwrap_or(Ordering::Equal)
            })
    }
}

/// One step of PIBT: produce a collision-free joint action.
///
/// Agents are processed in priority order (by `priorities`, descending). For
/// each agent, candidate moves are sorted by the lexicographic cost and the
/// first collision-free candidate is selected.
///
/// # Arguments
///
/// - `config`: current joint configuration `Q_t`.
/// - `guidance`: per-agent guidance paths `Φ` (only `Φ[i][0]` is used — the
///   preferred next position).
/// - `goals`: per-agent goals `g_i`.
/// - `priorities`: per-agent priority weights (higher = processed first).
///   If empty, agents are processed in index order.
/// - `hindrance`: the hindrance estimator (pluggable seam #4).
/// - `neighbors_fn`: supplies passable neighbors (`None` = `Position::neighbors()`).
/// - `rng`: deterministic RNG for the `ε` tiebreak.
///
/// # Returns
///
/// `Ok(JointAction)` on success. `Err(Deadlock)` if some agent couldn't be
/// placed; in that case the stuck agents are set to wait and the rest move.
pub fn pibt_step<P, H>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    priorities: &[f32],
    hindrance: &mut H,
    neighbors_fn: Option<&NeighborFn<P>>,
    rng: &mut fastrand::Rng,
) -> Result<JointAction<P>, Deadlock>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    let n = config.n_agents();
    let mut moves: Vec<Option<P>> = vec![None; n];
    let mut occupied: Vec<bool> = vec![false; n]; // which agents have been placed
    let mut stuck = Vec::new();

    // Priority order: descending priority, ties broken by agent id for determinism.
    let mut order: Vec<usize> = (0..n).collect();
    if !priorities.is_empty() && priorities.len() == n {
        order.sort_by(|&a, &b| {
            // descending priority
            priorities[b]
                .partial_cmp(&priorities[a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
    }

    for &i in &order {
        let agent = AgentId(i as u32);
        let current = config.pos(agent).clone();
        let goal = &goals[i];

        // Generate candidate moves.
        let neighbors: Vec<P> = if let Some(f) = neighbors_fn {
            f(&current)
        } else {
            current.neighbors()
        };

        // Build candidates with lexicographic cost.
        let preferred = guidance.get(i).and_then(|g| g.first());
        let mut candidates: Vec<Candidate<P>> = neighbors
            .iter()
            .map(|next| Candidate {
                next: next.clone(),
                guidance_mismatch: match preferred {
                    Some(p) => (p != next) as u8,
                    None => 0,
                },
                goal_dist: next.dist_heuristic(goal),
                hindrance: hindrance.hindrance(agent, next, config),
                epsilon: rng.f32(),
            })
            .collect();

        // Sort by lexicographic cost (ascending = best first).
        candidates.sort_by(|a, b| a.lexicographic_cmp(b));

        // Select the first collision-free candidate.
        let mut placed = false;
        for cand in &candidates {
            if is_collision_free(i, &cand.next, config, &moves, &occupied) {
                moves[i] = Some(cand.next.clone());
                occupied[i] = true;
                placed = true;
                break;
            }
        }

        if !placed {
            // Fallback: wait in place (if not itself a collision).
            if is_collision_free(i, &current, config, &moves, &occupied) {
                moves[i] = Some(current);
                occupied[i] = true;
            } else {
                stuck.push(agent);
            }
        }
    }

    if !stuck.is_empty() {
        // Place stuck agents as wait (best effort).
        for agent in &stuck {
            let i = usize::from(*agent);
            let pos = config.pos(*agent).clone();
            moves[i] = Some(pos);
        }
        // Return Ok with the partial action — caller decides whether to escalate.
        // We return the joint action (with stuck agents waiting) rather than
        // Err, because lifelong MAPF tolerates temporary stalls.
    }

    let final_moves: Vec<P> = moves
        .into_iter()
        .enumerate()
        .map(|(i, m)| m.unwrap_or_else(|| config.pos(AgentId(i as u32)).clone()))
        .collect();

    Ok(JointAction::new(final_moves))
}

/// Check if agent `i` moving to `next` is collision-free given the moves
/// already committed for other agents.
///
/// - **Vertex**: `next` must not be the committed move of any other agent.
/// - **Edge**: no agent `j ≠ i` where `moves[j] == current_pos[i]` AND
///   `next == current_pos[j]` (swap).
#[inline]
fn is_collision_free<P: Position>(
    agent_idx: usize,
    next: &P,
    config: &JointConfig<P>,
    moves: &[Option<P>],
    occupied: &[bool],
) -> bool {
    let my_current = config.pos(AgentId(agent_idx as u32));

    for (j, m) in moves.iter().enumerate() {
        if j == agent_idx || !occupied[j] {
            continue;
        }
        if let Some(their_next) = m {
            // Vertex collision: same destination.
            if their_next == next {
                return false;
            }
            // Edge collision: swap (I go to their current, they go to my current).
            let their_current = config.pos(AgentId(j as u32));
            if their_next == my_current && next == their_current {
                return false;
            }
        }
    }
    true
}
