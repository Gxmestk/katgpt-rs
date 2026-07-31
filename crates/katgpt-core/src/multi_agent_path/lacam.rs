//! Bounded one-step LaCAM escalation (Plan 453, Research 441).
//!
//! Replaces the fake "LaCAM escalation" (shuffled-priority retries in
//! `pibt.rs`) with the real LaCAM mechanism: a **constraint tree** +
//! **recursive PIBT with priority inheritance**. The critical insight from
//! reading the reference implementation (`Kei18/lacam/src/planner.cpp`):
//! LaCAM DOES use recursive PIBT (the piece Issues 140/143 tried and
//! reverted), but it works because the constraint tree bounds the recursion
//! and provides systematic backtracking.
//!
//! **Scope:** one-step LaCAM — find a collision-free joint action for the
//! current tick, bounded by a node/time budget, with greedy-PIBT fallback.
//! This is NOT multi-step LaCAM* (which degrades lifelong throughput per
//! Research 424 §1.5); it's the one-step collision-freedom mechanism.
//!
//! # Algorithm
//!
//! 1. Run greedy PIBT (fast path). If no stuck agents, return immediately.
//! 2. If stuck agents exist, explore the **constraint tree**: systematically
//!    try forcing different agents to different cells, re-running recursive
//!    PIBT for each constraint. The first collision-free config found wins.
//! 3. If the budget is exhausted before finding a collision-free config,
//!    fall back to the greedy PIBT result (current behavior — collisions on
//!    congested maps, but throughput preserved).
//!
//! # The constraint tree (LaCAM's low-level search)
//!
//! Each constraint is a chain `(agent_0 → cell_0, agent_1 → cell_1, ...)`.
//! The root constraint is empty. Each child extends the parent by one
//! `(agent, cell)` assignment. The agent at each depth is determined by the
//! priority order. When a constraint is popped, `get_new_config` applies the
//! forced assignments then runs recursive PIBT for the remaining agents.
//!
//! # Why this doesn't collapse throughput (the Issue 140/143 lesson)
//!
//! Issues 140/143 implemented recursive PIBT WITHOUT the constraint tree.
//! Without backtracking, a single priority-inheritance push can cascade
//! (A pushes B, B pushes C, ...) and stall the entire system. The constraint
//! tree bounds the cascade: when recursive PIBT fails, the constraint tree
//! tries a different root assignment. See Research 441 §5 for the full
//! prior-art comparison table.

use super::config::{AgentId, JointAction, JointConfig};
use super::flow::FlowField;
use super::hindrance::HindranceEstimator;
use super::local_guidance::Guidance;
use super::pibt::{compute_priority_order, greedy_pibt_pass};
use super::position::Position;
use std::collections::{HashMap, HashSet, VecDeque};

/// Default node budget for the constraint-tree search.
///
/// At roughly 1μs per node (constraint application + recursive PIBT), this is
/// ~1ms of overhead — acceptable for a congested-map tick. On open maps the
/// constraint tree is never entered (greedy PIBT fast path), so zero overhead.
pub const DEFAULT_MAX_NODES: usize = 1000;

/// Default wall-clock budget for the constraint-tree search (microseconds).
///
/// Checked periodically (every 64 nodes) to bound latency. When exhausted,
/// falls back to greedy PIBT.
pub const DEFAULT_TIME_BUDGET_US: u64 = 5000;

/// Default maximum constraint-tree depth (Issue 546 multi-step extension).
///
/// Caps how many agents the constraint tree can force-assign. With
/// `target_stuck_agents = true`, this is the maximum number of stuck agents
/// the tree will try to break free. The ht_chantry diagnostic (commit
/// `2a8c378d`) measured P95 max-cluster-size = 8, so depth 8 is the minimum
/// useful bound on the hardest map. Higher values cover more of the tail but
/// cost latency (combinatorial expansion).
pub const DEFAULT_MAX_DEPTH: usize = 8;

/// Minimum number of stuck agents before LaCAM escalation triggers.
///
/// Re-exports the `pibt.rs` constant semantics: on open maps, a few agents
/// may get stuck each tick due to random collisions — they resolve naturally
/// next tick. The escalation is only worth its overhead on genuinely congested
/// maps (systemic stuck agents).
///
/// Exposed as `pub(super)` so the `pibt_step_with_budget` wrapper in
/// `pibt.rs` can use the same threshold when deciding whether to enter the
/// LaCAM constraint-tree search.
pub(super) const MIN_STUCK_FOR_LACAM: usize = 1;

/// Budget for the constraint-tree search.
///
/// Bounds the LaCAM escalation to maintain real-time perf. When exhausted,
/// falls back to the greedy PIBT result.
#[derive(Clone, Copy, Debug)]
pub struct EscalationBudget {
    /// Maximum number of constraint-tree nodes to explore.
    pub max_nodes: usize,
    /// Wall-clock budget in microseconds. Checked every 64 nodes.
    pub time_budget_us: u64,
    /// Maximum constraint-tree depth (Issue 546 multi-step extension).
    ///
    /// Caps how many agents the tree can force-assign. Default 8 covers the
    /// P95 cluster size on ht_chantry. Ignored when `target_stuck_agents`
    /// is false (legacy behavior expands to depth `n`).
    pub max_depth: usize,
    /// Target stuck agents in the constraint tree (Issue 546 multi-step).
    ///
    /// When `true`, the constraint tree iterates over stuck agents (computed
    /// by the initial greedy PIBT pass) instead of all agents in priority
    /// order. This makes depth-K constraints target the K stuck agents
    /// directly, dramatically reducing the search space on maps where stuck
    /// agents are deep in the priority order (ht_chantry-style maze maps).
    ///
    /// Default `false` preserves Plan 453 behavior (paper-faithful BFS over
    /// priority order). Default-on for the multi-step extension via
    /// [`EscalationBudget::multistep_default`].
    pub target_stuck_agents: bool,
}

impl Default for EscalationBudget {
    fn default() -> Self {
        Self {
            max_nodes: DEFAULT_MAX_NODES,
            time_budget_us: DEFAULT_TIME_BUDGET_US,
            max_depth: DEFAULT_MAX_DEPTH,
            target_stuck_agents: false,
        }
    }
}

impl EscalationBudget {
    /// Multi-step LaCAM defaults (Issue 546 reopened plan).
    ///
    /// Stuck-agent targeting + depth 8 + larger node/time budget for
    /// maze-class maps. Use this on ht_chantry-class maps where the
    /// paper-faithful BFS-over-priority-order cannot reach stuck agents
    /// within the default budget.
    ///
    /// Latency budget: 100ms (vs 5ms default). At ~1µs/node that's 100K
    /// nodes — well within the Issue 546 acceptance criteria (≤ 500ms on
    /// the hard map).
    pub fn multistep_default() -> Self {
        Self {
            max_nodes: 100_000,
            time_budget_us: 100_000,
            max_depth: DEFAULT_MAX_DEPTH,
            target_stuck_agents: true,
        }
    }
}

/// One constraint in the LaCAM constraint tree.
///
/// A chain of `(agent_index, forced_cell)` pairs. The root constraint is
/// empty (depth 0). Each child extends the parent by one assignment.
#[derive(Clone)]
struct Constraint<P: Position + Clone> {
    /// Agent indices, in the order they were constrained.
    who: Vec<usize>,
    /// Forced cells, parallel to `who`.
    where_cells: Vec<P>,
}

impl<P: Position + Clone> Constraint<P> {
    fn empty() -> Self {
        Self {
            who: Vec::new(),
            where_cells: Vec::new(),
        }
    }

    fn depth(&self) -> usize {
        self.who.len()
    }

    /// Create a child constraint by appending one `(agent, cell)` assignment.
    fn child(&self, agent: usize, cell: P) -> Self {
        let mut c = Constraint {
            who: Vec::with_capacity(self.who.len() + 1),
            where_cells: Vec::with_capacity(self.where_cells.len() + 1),
        };
        c.who.extend_from_slice(&self.who);
        c.where_cells.extend_from_slice(&self.where_cells);
        c.who.push(agent);
        c.where_cells.push(cell);
        c
    }
}

/// FIFO queue of constraints (BFS-style exploration).
///
/// LaCAM uses FIFO to explore shallow constraints first (fewer forced
/// assignments), which are more likely to succeed (less constraining).
struct ConstraintQueue<P: Position + Clone> {
    queue: VecDeque<Constraint<P>>,
}

impl<P: Position + Clone> ConstraintQueue<P> {
    fn with_capacity(cap: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(cap),
        }
    }

    fn push(&mut self, c: Constraint<P>) {
        self.queue.push_back(c);
    }

    fn pop(&mut self) -> Option<Constraint<P>> {
        self.queue.pop_front()
    }
}

/// Error from `get_new_config`: the constraint was rejected (collision or
/// PIBT failure). The constraint tree tries the next constraint.
#[derive(Debug)]
struct ConstraintRejected;

/// Bounded one-step LaCAM escalation.
///
/// Replaces the shuffled-priority retry loop in `pibt_step` when the
/// `lacam_escalation` feature is enabled. Runs greedy PIBT first (fast path);
/// if stuck agents exist, explores the constraint tree to find a
/// collision-free config. Falls back to greedy PIBT if the budget is
/// exhausted.
///
/// Returns `Ok(JointAction)` always — same API contract as `pibt_step`.
///
/// Marked `pub` so benchmark harnesses (e.g. Plan 453 T3.3 latency sweep)
/// can call it directly with a custom [`EscalationBudget`]. The orchestrator
/// [`LifelongLaCam::tick`](super::LifelongLaCam::tick) calls this via
/// [`pibt_step`](super::pibt::pibt_step) with `EscalationBudget::default()`.
#[allow(clippy::too_many_arguments)]
pub fn lacam_escalation_step<P, H>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    priorities: &[f32],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&super::pibt::NeighborFn<P>>,
    rng: &mut fastrand::Rng,
    budget: EscalationBudget,
) -> JointAction<P>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    let n = config.n_agents();
    let order = compute_priority_order(n, priorities);

    // Empty backer set (same as pibt_step — swap technique is infrastructure-only).
    let no_backers = vec![false; n];

    // Phase A: greedy PIBT (fast path).
    let (greedy_moves, stuck) = greedy_pibt_pass(
        config,
        guidance,
        goals,
        hindrance,
        flow_field,
        neighbors_fn,
        rng,
        &order,
        &no_backers,
    );

    // Fast path: no stuck agents (or too few). Return greedy result.
    if stuck.len() < MIN_STUCK_FOR_LACAM {
        return JointAction::new(greedy_moves);
    }

    // Phase B: constraint-tree search.
    //
    // Issue 546 multi-step extension: when `target_stuck_agents` is set, the
    // constraint tree iterates over stuck agents (computed above by greedy
    // PIBT) instead of all agents in priority order. This dramatically
    // reduces the search space on maze maps where stuck agents are deep in
    // the priority order.
    //
    // Build the expansion order: either the full priority order (legacy,
    // paper-faithful) or just the stuck agents (Issue 546 multi-step).
    let stuck_indices: Vec<usize> = if budget.target_stuck_agents {
        stuck.iter().map(|a| a.0 as usize).collect()
    } else {
        Vec::new()
    };
    let expansion_order: Vec<usize> = if budget.target_stuck_agents {
        stuck_indices.clone()
    } else {
        order.clone()
    };
    let expansion_depth_cap: usize = if budget.target_stuck_agents {
        // Cap at min(max_depth, stuck.len()) — can't constrain more agents
        // than are stuck, and don't exceed the configured depth bound.
        budget.max_depth.min(expansion_order.len())
    } else {
        n
    };

    let mut queue = ConstraintQueue::<P>::with_capacity(budget.max_nodes);
    queue.push(Constraint::empty());

    let mut nodes_explored: usize = 0;
    let start = std::time::Instant::now();
    let time_budget = std::time::Duration::from_micros(budget.time_budget_us);

    // Pre-build the current_to_agent map (shared across all constraint attempts).
    let mut current_to_agent: HashMap<P, usize> = HashMap::with_capacity(n);
    for (i, pos) in config.positions.iter().enumerate() {
        current_to_agent.entry(pos.clone()).or_insert(i);
    }

    while let Some(constraint) = queue.pop() {
        nodes_explored += 1;

        // Budget check: node cap.
        if nodes_explored > budget.max_nodes {
            break;
        }
        // Budget check: time cap (every 64 nodes to reduce branch overhead).
        if (nodes_explored & 63) == 0 && start.elapsed() > time_budget {
            break;
        }

        // Expand: push children for the next agent in the expansion order.
        //
        // Legacy (Plan 453): expansion_order = priority order, depth cap = n.
        // Issue 546 multi-step: expansion_order = stuck agents, depth cap =
        // min(max_depth, stuck.len()). The constraint at depth K forces the
        // K-th agent in `expansion_order` to one of its neighbor cells.
        let depth = constraint.depth();
        if depth < expansion_depth_cap {
            let i = expansion_order[depth];
            let current = config.pos(AgentId(i as u32));
            let neighbors: Vec<P> = if let Some(f) = neighbors_fn {
                f(current)
            } else {
                current.neighbors()
            };
            // Shuffle for diversity (deterministic via seeded rng).
            let mut shuffled: Vec<P> = neighbors;
            // Fisher-Yates shuffle with seeded rng.
            for k in (1..shuffled.len()).rev() {
                let j = rng.usize(0..=k);
                shuffled.swap(k, j);
            }
            for cell in shuffled {
                queue.push(constraint.child(i, cell));
            }
        }

        // Try to build a collision-free config with this constraint.
        match get_new_config(
            config,
            &constraint,
            guidance,
            goals,
            hindrance,
            flow_field,
            neighbors_fn,
            rng,
            &order,
            &current_to_agent,
        ) {
            Ok(moves) => {
                // Verify collision-free (vertex + edge).
                if is_collision_free(&moves, config) {
                    return JointAction::new(moves);
                }
            }
            Err(ConstraintRejected) => continue,
        }
    }

    // Phase C: budget exhausted — fall back to greedy PIBT result.
    let _ = stuck_indices; // suppress unused warning when not targeting
    JointAction::new(greedy_moves)
}

/// Build a collision-free next configuration by applying the constraint's
/// forced assignments, then running recursive PIBT for the remaining agents.
///
/// Adapted from `Kei18/lacam/src/planner.cpp:get_new_config`. Returns
/// `Ok(moves)` if a collision-free config was built, `Err(ConstraintRejected)`
/// if any forced assignment collides or PIBT fails for an unconstrained agent.
#[allow(clippy::too_many_arguments)]
fn get_new_config<P, H>(
    config: &JointConfig<P>,
    constraint: &Constraint<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&super::pibt::NeighborFn<P>>,
    rng: &mut fastrand::Rng,
    order: &[usize],
    current_to_agent: &HashMap<P, usize>,
) -> Result<Vec<P>, ConstraintRejected>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    let n = config.n_agents();
    let mut moves: Vec<Option<P>> = vec![None; n];
    let mut occupied_next: HashSet<P> = HashSet::with_capacity(n);
    let mut constrained_agents: HashSet<usize> = HashSet::with_capacity(constraint.depth());

    // 1. Apply constraints: force specific agents to specific cells.
    for (k, agent_i) in constraint.who.iter().enumerate() {
        let cell = &constraint.where_cells[k];
        // Vertex collision check.
        if occupied_next.contains(cell) {
            return Err(ConstraintRejected);
        }
        // Swap collision check: the agent currently at `cell` (if any) is
        // committed to moving to agent_i's current position.
        if let Some(&j) = current_to_agent.get(cell)
            && j != *agent_i
            && let Some(their_next) = &moves[j]
            && their_next == config.pos(AgentId(*agent_i as u32))
        {
            return Err(ConstraintRejected); // swap
        }
        occupied_next.insert(cell.clone());
        moves[*agent_i] = Some(cell.clone());
        constrained_agents.insert(*agent_i);
    }

    // 2. Run recursive PIBT for unconstrained agents (in priority order).
    hindrance.prepare(config);
    for &i in order {
        if constrained_agents.contains(&i) {
            continue;
        }
        if moves[i].is_some() {
            continue;
        }
        let mut pibt_state = PibtState {
            moves: &mut moves,
            occupied_next: &mut occupied_next,
            config,
            guidance,
            goals,
            hindrance,
            flow_field,
            neighbors_fn,
            rng,
            current_to_agent,
        };
        if !pibt_state.func_pibt_recursive(i) {
            return Err(ConstraintRejected);
        }
    }

    // 3. Finalize: fill any remaining None with wait-in-place (shouldn't happen
    //    if PIBT succeeded for all unconstrained agents, but defensive).
    let final_moves: Vec<P> = moves
        .into_iter()
        .enumerate()
        .map(|(i, m)| m.unwrap_or_else(|| config.pos(AgentId(i as u32)).clone()))
        .collect();

    Ok(final_moves)
}

/// Mutable state bundle for recursive PIBT.
///
/// Groups the shared mutable state (`moves`, `occupied_next`) and read-only
/// context into a single struct so the recursion signature stays manageable.
/// Mirrors the implicit `this`-bundled state in `Kei18/lacam:funcPIBT`.
struct PibtState<'a, P, H>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    moves: &'a mut Vec<Option<P>>,
    occupied_next: &'a mut HashSet<P>,
    config: &'a JointConfig<P>,
    guidance: &'a Guidance<P>,
    goals: &'a [P],
    hindrance: &'a mut H,
    flow_field: &'a dyn FlowField<P>,
    neighbors_fn: Option<&'a super::pibt::NeighborFn<P>>,
    rng: &'a mut fastrand::Rng,
    current_to_agent: &'a HashMap<P, usize>,
}

impl<'a, P, H> PibtState<'a, P, H>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    /// Recursive PIBT with priority inheritance (LaCAM's `funcPIBT`).
    ///
    /// Returns `true` if agent `i` was successfully placed, `false` if it
    /// could not find a collision-free cell (including via PI pushes). The
    /// caller (`get_new_config`) rejects the constraint on `false`.
    ///
    /// The recursion is bounded by the constraint tree: when this returns
    /// false, the constraint tree tries a different root assignment. This is
    /// why recursive PI is safe here but collapsed throughput in Issues
    /// 140/143 (which had no constraint tree).
    fn func_pibt_recursive(&mut self, i: usize) -> bool {
        let agent = AgentId(i as u32);
        let current = self.config.pos(agent).clone();

        // Generate candidates: neighbors + wait, sorted by lexicographic cost.
        let neighbors: Vec<P> = if let Some(f) = self.neighbors_fn {
            f(&current)
        } else {
            current.neighbors()
        };

        let goal = &self.goals[i];
        let preferred = self.guidance.get(i).and_then(|g| g.first());

        // Build candidates with the same lexicographic cost as greedy PIBT.
        let mut candidates: Vec<super::pibt::Candidate<P>> = neighbors
            .iter()
            .map(|next| super::pibt::Candidate {
                next: next.clone(),
                guidance_mismatch: match &preferred {
                    Some(p) => (**p != *next) as u8,
                    None => 0,
                },
                flow_mismatch: self.flow_field.mismatch(&current, next),
                goal_dist: next.dist_heuristic(goal),
                hindrance: self.hindrance.hindrance(agent, next, self.config),
                epsilon: self.rng.f32(),
            })
            .collect();
        candidates.sort_by(|a, b| a.lexicographic_cmp(b));

        for cand in &candidates {
            let next = &cand.next;
            // Vertex collision check.
            if self.occupied_next.contains(next) {
                continue;
            }
            // Edge collision (swap) check.
            if let Some(&j) = self.current_to_agent.get(next)
                && j != i
                && let Some(their_next) = &self.moves[j]
                && their_next == &current
            {
                continue; // swap collision
            }

            // Reserve the cell.
            self.occupied_next.insert(next.clone());
            self.moves[i] = Some(next.clone());

            // Check the current occupant of `next`.
            let occupant = self.current_to_agent.get(next).copied();
            // Empty cell or staying → success.
            if occupant.is_none() || next == &current {
                return true;
            }

            // Priority inheritance: push the occupant.
            let k = occupant.unwrap();
            if k != i && self.moves[k].is_none() {
                if self.func_pibt_recursive(k) {
                    return true; // occupant moved, we get the cell
                }
                // Occupant couldn't move — undo our reservation and try next.
                self.occupied_next.remove(next);
                self.moves[i] = None;
                continue;
            }

            return true;
        }

        // Failed to find a collision-free cell. Stay in place if possible.
        let can_wait = !self.occupied_next.contains(&current);
        if can_wait {
            self.occupied_next.insert(current.clone());
            self.moves[i] = Some(current);
            true
        } else {
            false
        }
    }
}

/// Check if a joint action is collision-free (no vertex + no edge collisions).
///
/// Used to verify the output of `get_new_config` before accepting it. The
/// constraint application + recursive PIBT should always produce a
/// collision-free config when they succeed, but this defensive check guards
/// against bugs in the constraint/PIBT logic.
fn is_collision_free<P: Position>(moves: &[P], config: &JointConfig<P>) -> bool {
    let n = moves.len();
    // Vertex: all next positions distinct.
    let mut seen: HashSet<&P> = HashSet::with_capacity(n);
    for p in moves {
        if !seen.insert(p) {
            return false;
        }
    }
    // Edge: no swaps.
    for i in 0..n {
        for j in (i + 1)..n {
            let i_cur = &config.positions[i];
            let j_cur = &config.positions[j];
            if &moves[i] == j_cur && &moves[j] == i_cur {
                return false; // swap
            }
        }
    }
    true
}
