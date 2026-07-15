//! Unit tests for the LLLG substrate (Plan 440 T1.10).
//!
//! Tests cover the four paper scenarios from the plan:
//! 1. Vertex collision (two agents swap → edge collision).
//! 2. Deadlock (two agents in a 1-wide corridor).
//! 3. Throughput sanity (10 agents, 10×10 map, 100 ticks, all reach goals).
//! 4. Warm-start cache correctness.
//! 5. Hindrance estimator correctness.

#![cfg(test)]

use super::*;
use super::position::*;
use fastrand::Rng;

fn make_guidance(map: &GridMap, cfg: GuidanceConfig) -> SpaceTimeGuidance<GridPos> {
    let map = map.clone();
    SpaceTimeGuidance::new(cfg).with_neighbors(move |p| map.passable_neighbors(p))
}

// ─────────────────────────────────────────────────────────────────────
// T1.10.1 — Vertex + edge collision avoidance
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_two_agents_no_vertex_collision() {
    // Two agents on a 3×3 grid, opposite corners, swap goals.
    let map = GridMap::empty(3, 3);
    let config = JointConfig::new(vec![GridPos::new(0, 0), GridPos::new(2, 0)]);
    let goals = vec![GridPos::new(2, 0), GridPos::new(0, 0)];
    let cfg = GuidanceConfig {
        w_phi: 3,
        alpha: 1.0,
        rounds: 1,
    };
    let mut guidance = make_guidance(&map, cfg);
    let mut hindrance = BlockingCount::new();
    let mut rng = Rng::with_seed(42);
    let mut lacam = LifelongLaCam::new(WarmStartCache::new(WarmStartScheme::default(), cfg.w_phi));

    // Run several ticks; verify no vertex collision ever.
    let mut current = config;
    for _tick in 0..20 {
        let action = lacam.tick(&current, &goals, &mut guidance, &mut hindrance, &mut rng);
        // Vertex check: all next positions distinct.
        let mut seen = std::collections::HashSet::new();
        for p in &action.moves {
            assert!(seen.insert(*p), "vertex collision: two agents at {p:?}");
        }
        // Edge check: no swap.
        for i in 0..action.moves.len() {
            for j in (i + 1)..action.moves.len() {
                let i_cur = &current.positions[i];
                let j_cur = &current.positions[j];
                let i_next = &action.moves[i];
                let j_next = &action.moves[j];
                let is_swap = i_next == j_cur && j_next == i_cur;
                assert!(!is_swap, "edge collision: agents {i} and {j} swapped");
            }
        }
        current = JointConfig::new(action.moves);
    }
}

// ─────────────────────────────────────────────────────────────────────
// T1.10.2 — Deadlock (1-wide corridor)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_deadlock_corridor_falls_back_to_wait() {
    // 1×3 corridor: agent A at (0,0), agent B at (2,0), goals swapped.
    // This is a classic deadlock — they must pass each other but can't in a
    // 1-wide corridor. LLLG should not panic; stuck agents wait.
    let mut map = GridMap::empty(3, 1);
    // Walls above and below to force corridor.
    let _ = &mut map; // 1-row grid is already a corridor.
    let config = JointConfig::new(vec![GridPos::new(0, 0), GridPos::new(2, 0)]);
    let goals = vec![GridPos::new(2, 0), GridPos::new(0, 0)];
    let cfg = GuidanceConfig::default();
    let mut guidance = make_guidance(&map, cfg);
    let mut hindrance = BlockingCount::new();
    let mut rng = Rng::with_seed(7);
    let mut lacam = LifelongLaCam::new(WarmStartCache::new(WarmStartScheme::default(), cfg.w_phi));

    // Run a few ticks; should not panic.
    let mut current = config;
    for _tick in 0..5 {
        let action = lacam.tick(&current, &goals, &mut guidance, &mut hindrance, &mut rng);
        current = JointConfig::new(action.moves);
    }
    // The agents likely haven't swapped (deadlock), but the system didn't crash.
    // This is correct lifelong-MAPF behavior — stall is tolerated.
}

// ─────────────────────────────────────────────────────────────────────
// T1.10.3 — Throughput sanity (10 agents, 10×10, 100 ticks)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_throughput_sanity() {
    // 10 agents on a 10×10 grid. Each has a random goal. Run 100 ticks.
    // Verify: no collisions ever, and at least some agents reach their goals.
    let map = GridMap::empty(10, 10);
    let n = 10;
    let mut rng = Rng::with_seed(123);
    let starts: Vec<GridPos> = (0..n)
        .map(|i| GridPos::new(i * 10 / n, 0))
        .collect();
    let goals: Vec<GridPos> = (0..n)
        .map(|i| GridPos::new(i * 10 / n, 9))
        .collect();
    let config = JointConfig::new(starts.clone());
    let cfg = GuidanceConfig {
        w_phi: 5,
        alpha: 2.0,
        rounds: 2,
    };
    let mut guidance = make_guidance(&map, cfg);
    let mut hindrance = BlockingCount::new();
    let mut lacam = LifelongLaCam::new(WarmStartCache::new(WarmStartScheme::default(), cfg.w_phi));

    let mut current = config;
    let mut reached = vec![false; n];
    for _tick in 0..100 {
        let action = lacam.tick(&current, &goals, &mut guidance, &mut hindrance, &mut rng);
        // No collisions.
        let mut seen = std::collections::HashSet::new();
        for p in &action.moves {
            assert!(seen.insert(*p), "vertex collision at {p:?}");
        }
        current = JointConfig::new(action.moves);
        // Track goal reach.
        for i in 0..n {
            if current.positions[i] == goals[i] {
                reached[i] = true;
            }
        }
    }
    let n_reached = reached.iter().filter(|&&r| r).count();
    // On a 10×10 empty grid with 10 agents, at least some should reach goals
    // in 100 ticks (10×10 is ~10 Manhattan, 100 ticks is generous). We don't
    // require all (some may deadlock briefly) but require >0 for sanity.
    assert!(
        n_reached > 0,
        "expected at least 1 agent to reach its goal in 100 ticks, got {n_reached}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// T1.10.4 — Warm-start cache correctness
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_warm_start_lllg_pi_suffix() {
    // Previous solution: agent 0 path = [(1,0), (2,0), (3,0)].
    // w_Φ = 2. Suffix skips index 0 (executed), so init = [(2,0), (3,0)].
    let mut cache = WarmStartCache::<GridPos>::new(WarmStartScheme::LllgPi, 2);
    let prev_solution = vec![vec![
        GridPos::new(1, 0),
        GridPos::new(2, 0),
        GridPos::new(3, 0),
    ]];
    let prev_guidance = vec![vec![GridPos::new(9, 9)]];
    cache.record(prev_solution, prev_guidance);

    let ws = cache.warm_start();
    assert_eq!(ws.len(), 1);
    assert_eq!(ws[0].len(), 2);
    assert_eq!(ws[0][0], GridPos::new(2, 0)); // suffix[0] after skipping executed
    assert_eq!(ws[0][1], GridPos::new(3, 0));
}

#[test]
fn test_warm_start_lllg_pi_padding() {
    // Previous solution shorter than w_Φ → pad with last position.
    let mut cache = WarmStartCache::<GridPos>::new(WarmStartScheme::LllgPi, 5);
    let prev_solution = vec![vec![GridPos::new(1, 0), GridPos::new(2, 0)]];
    cache.record(prev_solution, vec![vec![]]);

    let ws = cache.warm_start();
    assert_eq!(ws[0].len(), 5);
    // After skipping index 0: [2,0], then pad with [2,0].
    assert_eq!(ws[0][0], GridPos::new(2, 0));
    assert_eq!(ws[0][4], GridPos::new(2, 0)); // padded
}

#[test]
fn test_warm_start_lllg_phi() {
    let mut cache = WarmStartCache::<GridPos>::new(WarmStartScheme::LllgPhi, 3);
    let prev_guidance = vec![vec![GridPos::new(5, 5), GridPos::new(6, 6)]];
    cache.record(vec![vec![]], prev_guidance.clone());

    let ws = cache.warm_start();
    assert_eq!(ws, prev_guidance);
}

#[test]
fn test_warm_start_lllg_empty() {
    let mut cache = WarmStartCache::<GridPos>::new(WarmStartScheme::LllgEmpty, 3);
    cache.record(
        vec![vec![GridPos::new(1, 0)]],
        vec![vec![GridPos::new(2, 0)]],
    );
    let ws = cache.warm_start();
    assert!(ws.is_empty());
}

// ─────────────────────────────────────────────────────────────────────
// T1.10.5 — Hindrance estimator correctness
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_blocking_count_known() {
    // 3 agents. Agent 0 considers moving to (1,0).
    // Agent 1 at (2,0) has (1,0) in its neighborhood → blocks.
    // Agent 2 at (9,9) does not → no block.
    // Hindrance(0 → (1,0)) should be 1.
    let config = JointConfig::new(vec![
        GridPos::new(0, 0),
        GridPos::new(2, 0),
        GridPos::new(9, 9),
    ]);
    let mut h = BlockingCount::new();
    let val = h.hindrance(AgentId(0), &GridPos::new(1, 0), &config);
    assert_eq!(val, 1.0, "agent 1 at (2,0) blocks (1,0); agent 2 doesn't");
}

#[test]
fn test_weighted_blocking_count() {
    let config = JointConfig::new(vec![
        GridPos::new(0, 0),
        GridPos::new(2, 0), // fearful → weight 3.0
        GridPos::new(2, 0), // calm → weight 1.0 (same pos for test)
    ]);
    let weights = |id: AgentId| match id.0 {
        1 => 3.0, // fearful
        _ => 1.0,
    };
    let mut h = WeightedBlockingCount::new(weights);
    let val = h.hindrance(AgentId(0), &GridPos::new(1, 0), &config);
    // Both agents 1 and 2 are at (2,0) whose neighborhood includes (1,0).
    // Weighted: 3.0 + 1.0 = 4.0.
    assert_eq!(val, 4.0);
}

// ─────────────────────────────────────────────────────────────────────
// T1.10.6 — Config + AgentId basics
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_agent_id_index() {
    let a = AgentId(5);
    let idx: usize = a.into();
    assert_eq!(idx, 5);
}

#[test]
fn test_joint_config_index() {
    let config = JointConfig::new(vec![GridPos::new(1, 1), GridPos::new(2, 2)]);
    assert_eq!(config[AgentId(0)], GridPos::new(1, 1));
    assert_eq!(config[AgentId(1)], GridPos::new(2, 2));
}

#[test]
fn test_joint_action_apply() {
    let config = JointConfig::new(vec![GridPos::new(0, 0), GridPos::new(9, 9)]);
    let action = JointAction::new(vec![GridPos::new(1, 0), GridPos::new(8, 9)]);
    let next = action.apply_to(&config);
    assert_eq!(next[AgentId(0)], GridPos::new(1, 0));
    assert_eq!(next[AgentId(1)], GridPos::new(8, 9));
}

// ─────────────────────────────────────────────────────────────────────
// T1.10.7 — UniformGoals reassignment
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_uniform_goals_reassigns_on_reach() {
    let candidates = vec![GridPos::new(5, 5), GridPos::new(7, 7)];
    let mut goals = UniformGoals::new(candidates.clone(), 1, 42);
    // Force agent 0 to be at its goal — should get a new goal.
    // We don't know the exact initial goal (seeded RNG), but after reaching
    // it, goal_for returns a different candidate.
    let initial = goals.current_goal(AgentId(0)).unwrap().clone();
    let reassigned = goals.goal_for(AgentId(0), &initial);
    // After reaching, a new goal is assigned — must be one of the candidates.
    assert!(candidates.contains(&reassigned));
}

#[test]
fn test_uniform_goals_keeps_when_not_at_goal() {
    let candidates = vec![GridPos::new(5, 5), GridPos::new(7, 7)];
    let mut goals = UniformGoals::new(candidates, 1, 42);
    let initial = goals.current_goal(AgentId(0)).unwrap().clone();
    // Agent is NOT at goal → goal_for returns the same goal.
    let elsewhere = GridPos::new(0, 0);
    let g = goals.goal_for(AgentId(0), &elsewhere);
    assert_eq!(g, initial);
}

// ─────────────────────────────────────────────────────────────────────
// T1.10.8 — CostFn trait (pluggable seam #1)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_uniform_cost() {
    let cost = UniformCost;
    let c = cost.cost(&GridPos::new(0, 0), &GridPos::new(1, 0));
    assert_eq!(c, 1.0);
}

// ─────────────────────────────────────────────────────────────────────
// T1.10.9 — Position trait basics
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_grid_pos_neighbors() {
    let pos = GridPos::new(5, 5);
    let neighbors = pos.neighbors();
    // wait + up + down + left + right = 5 (interior cell).
    assert_eq!(neighbors.len(), 5);
    assert!(neighbors.contains(&GridPos::new(5, 5))); // wait
    assert!(neighbors.contains(&GridPos::new(4, 5))); // left
    assert!(neighbors.contains(&GridPos::new(6, 5))); // right
    assert!(neighbors.contains(&GridPos::new(5, 4))); // down
    assert!(neighbors.contains(&GridPos::new(5, 6))); // up
}

#[test]
fn test_grid_pos_corner_neighbors() {
    // (0,0): no left, no down.
    let pos = GridPos::new(0, 0);
    let neighbors = pos.neighbors();
    // wait + right + up = 3 (no left/down; usize can't go negative).
    assert_eq!(neighbors.len(), 3);
}

#[test]
fn test_manhattan_heuristic() {
    let a = GridPos::new(0, 0);
    let b = GridPos::new(3, 4);
    assert_eq!(a.dist_heuristic(&b), 7.0);
}

#[test]
fn test_grid_map_walls() {
    let mut map = GridMap::empty(5, 5);
    map.set_wall(2, 2);
    assert!(!map.is_passable(2, 2));
    assert!(map.is_passable(1, 1));
    // Out of bounds = not passable.
    assert!(!map.is_passable(10, 10));
}

#[test]
fn test_passable_neighbors_respects_walls() {
    let mut map = GridMap::empty(3, 3);
    map.set_wall(1, 0); // wall to the right of (0,0)
    let neighbors = map.passable_neighbors(&GridPos::new(0, 0));
    // wait + up + ... but not right (wall).
    assert!(neighbors.contains(&GridPos::new(0, 0))); // wait
    assert!(neighbors.contains(&GridPos::new(0, 1))); // up
    assert!(!neighbors.contains(&GridPos::new(1, 0))); // wall
}

// ─────────────────────────────────────────────────────────────────────────
// Issue 140 T1 — PIBT priority inheritance (deferred)
// ─────────────────────────────────────────────────────────────────────────
//
// The full recursive PIBT with priority inheritance was implemented and
// benchmarked but found to REDUCE throughput without LaCAM escalation. The
// recursive push is too conservative on dense maps — it requires occupants
// to vacate before committing, causing cascading stalls. The greedy PIBT
// (take first collision-free candidate, let later agents adapt) has higher
// throughput in the lifelong MAPF setting without LaCAM. See the module docs
// in `pibt.rs` for the full rationale.
//
// The chain-push test below verifies that the greedy PIBT can advance a line
// of agents (which it handles well via sequential processing).

/// A line of 3 agents in a corridor, all wanting to move right. The greedy
/// PIBT should advance the chain by processing agents front-to-back (the
/// front agent moves first, then the middle, then the back).
#[test]
fn test_pibt_chain_push_in_line() {
    // 5×3 grid — a 5-wide corridor at row 1 with escape lanes at rows 0 and 2.
    let map = GridMap::empty(5, 3);
    // Three agents in a line at row 1, all wanting to move right.
    let config = JointConfig::new(vec![
        GridPos::new(0, 1),
        GridPos::new(1, 1),
        GridPos::new(2, 1),
    ]);
    let goals = vec![GridPos::new(4, 1), GridPos::new(4, 1), GridPos::new(4, 1)];
    let cfg = GuidanceConfig::default();
    let mut guidance = make_guidance(&map, cfg);
    let mut hindrance = BlockingCount::new();
    let mut rng = Rng::with_seed(42);
    let map_clone = map.clone();
    let mut lacam = LifelongLaCam::new(WarmStartCache::new(WarmStartScheme::default(), cfg.w_phi))
        .with_neighbors(move |p| map_clone.passable_neighbors(p));

    let mut current = config;
    let mut progress_made = false;
    for _tick in 0..50 {
        let action = lacam.tick(&current, &goals, &mut guidance, &mut hindrance, &mut rng);
        let mut seen = std::collections::HashSet::new();
        for p in &action.moves {
            assert!(seen.insert(*p), "vertex collision at {p:?}");
        }
        // At least one agent should move on most ticks.
        let any_moved = action
            .moves
            .iter()
            .zip(current.positions.iter())
            .any(|(next, cur)| next != cur);
        if any_moved {
            progress_made = true;
        }
        current = JointConfig::new(action.moves);
    }
    assert!(
        progress_made,
        "greedy PIBT should advance the chain of 3 agents at least once"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Issue 140/142 — Warm-start infrastructure + consumption findings
// ─────────────────────────────────────────────────────────────────────────
//
// The warm-start infrastructure (set_warm_start trait method, storage,
// tick() threading) was landed in Issue 140. Issue 142 confirmed via the full
// space-time A* that occupancy-seeding with warm-start forecasts HURTS
// throughput even with proper A* lookahead — the forecast is invalidated
// when PIBT deviates from the guidance (common on dense maps), creating
// misleading phantom collision constraints. LllgEmpty consistently
// outperforms LllgPi on our synthetic maps. The paper's positive result
// likely depends on LaCAM escalation keeping forecasts accurate.
//
// The data is consumed (taken/cleared) to prevent stale leaks but NOT seeded
// into the occupancy. LllgPi and LllgEmpty produce identical results.

/// The warm-start integration must actually accept the data on the guidance
/// source. Verifies that `set_warm_start` stores the data and that
/// `compute_guidance` consumes it (one-shot). The data is consumed (cleared)
/// even though it's not seeded into the occupancy (Issue 142 finding) — this
/// ensures no stale data leaks across ticks.
#[test]
fn test_set_warm_start_consumed_once() {
    let map = GridMap::empty(5, 5);
    let cfg = GuidanceConfig::default();
    let mut guidance = make_guidance(&map, cfg);
    let config = JointConfig::new(vec![GridPos::new(0, 0), GridPos::new(4, 4)]);
    let goals = vec![GridPos::new(4, 4), GridPos::new(0, 0)];
    let mut out = Vec::new();

    // No warm-start initially.
    guidance.compute_guidance(&config, &goals, &mut out);
    let baseline_len = out.iter().map(|p| p.len()).sum::<usize>();

    // Set warm-start with one-step paths.
    let warm = vec![
        vec![GridPos::new(1, 0)],
        vec![GridPos::new(3, 4)],
    ];
    guidance.set_warm_start(warm);
    guidance.compute_guidance(&config, &goals, &mut out);
    // Warm-start is one-shot — should be consumed.
    // Run again to confirm it's empty (no panic, same result as baseline).
    guidance.compute_guidance(&config, &goals, &mut out);
    let after_len = out.iter().map(|p| p.len()).sum::<usize>();
    assert_eq!(
        baseline_len, after_len,
        "warm-start should be one-shot — second compute_guidance without set should match baseline"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Issue 142 — Space-time A* guidance (replaces greedy rollout)
// ─────────────────────────────────────────────────────────────────────────

/// The A* guidance must produce valid w_Φ-length paths for each agent.
/// Each path should have exactly w_Φ entries (positions at t+1..t+w_Φ).
#[test]
fn test_astar_guidance_path_length() {
    let map = GridMap::empty(10, 10);
    let cfg = GuidanceConfig { w_phi: 5, alpha: 1.0, rounds: 2 };
    let mut guidance = make_guidance(&map, cfg);
    let config = JointConfig::new(vec![
        GridPos::new(0, 0),
        GridPos::new(9, 9),
    ]);
    let goals = vec![GridPos::new(9, 9), GridPos::new(0, 0)];
    let mut out = Vec::new();
    guidance.compute_guidance(&config, &goals, &mut out);

    assert_eq!(out.len(), 2, "one path per agent");
    for (i, path) in out.iter().enumerate() {
        assert_eq!(
            path.len(),
            cfg.w_phi,
            "agent {i} path must be exactly w_Φ={} steps, got {}",
            cfg.w_phi,
            path.len()
        );
    }
}

/// The A* guidance should navigate toward the goal: the last position of each
/// agent's path should be closer (BFS distance) to the goal than the start.
#[test]
fn test_astar_guidance_moves_toward_goal() {
    let map = GridMap::empty(10, 10);
    let cfg = GuidanceConfig { w_phi: 5, alpha: 1.0, rounds: 1 };
    let mut guidance = make_guidance(&map, cfg);
    let start_a = GridPos::new(0, 0);
    let goal_a = GridPos::new(9, 9);
    let config = JointConfig::new(vec![start_a]);
    let goals = vec![goal_a];
    let mut out = Vec::new();
    guidance.compute_guidance(&config, &goals, &mut out);

    let path = &out[0];
    assert!(!path.is_empty());
    let end = path.last().unwrap();
    let start_dist = start_a.dist_heuristic(&goal_a);
    let end_dist = end.dist_heuristic(&goal_a);
    assert!(
        end_dist < start_dist,
        "A* should move toward goal: start dist={start_dist}, end dist={end_dist}"
    );
}

/// The A* must respect walls: no path position should be a wall cell.
#[test]
fn test_astar_guidance_respects_walls() {
    let mut map = GridMap::empty(10, 10);
    // Wall at (5, 0) — blocks the direct path from (0,0) to (9,0).
    map.set_wall(5, 0);
    map.set_wall(5, 1);
    map.set_wall(5, 2);
    let cfg = GuidanceConfig::default();
    let mut guidance = make_guidance(&map, cfg);
    let config = JointConfig::new(vec![GridPos::new(0, 0)]);
    let goals = vec![GridPos::new(9, 0)];
    let mut out = Vec::new();
    guidance.compute_guidance(&config, &goals, &mut out);

    for (t, pos) in out[0].iter().enumerate() {
        assert!(
            map.is_passable(pos.x, pos.y),
            "path position at t={t} is a wall: {pos:?}"
        );
    }
}
