//! Plan 453 Phase 3 — Bounded One-Step LaCAM Escalation GOAT Gate Benchmark.
//!
//! Three measurement sections:
//!
//! - **G6c (collision-freedom delta)** — 60 agents, 20×20 grid with a 6-cell
//!   bottleneck gap, 200 ticks. Measures vertex + edge collision rates. The
//!   G6c delta = (collision-free rate with LaCAM) − (collision-free rate with
//!   naive no-planner baseline). Ported from the riir-ai G6c scenario
//!   (`.benchmarks/516_g6c_collision_freedom_delta.md`).
//! - **Latency sweep** — `max_nodes ∈ {100, 500, 1000, 5000}` on a congested
//!   scenario. Measures median + max per-tick latency and collision-free rate
//!   at each budget. Finds the knee — where latency exceeds 500ms or
//!   collision-freedom stops improving.
//!
//! The G1 throughput gate (T3.2) is measured by re-running `bench_440_lllg_paper_repro`
//! compiled with `--features lacam_escalation`. This bench focuses on the two
//! metrics that need the LaCAM-specific harness: collision-freedom and budget sweep.
//!
//! # Run
//!
//! ```bash
//! # G6c + latency sweep (this bench)
//! CARGO_TARGET_DIR=/tmp/453_phase3 cargo run --release -p katgpt-core \
//!     --features lacam_escalation --bench bench_453_lacam_escalation_goat -- --nocapture
//!
//! # G1 throughput (bench_440, compiled with lacam_escalation ON)
//! CARGO_TARGET_DIR=/tmp/453_phase3 cargo run --release -p katgpt-core \
//!     --features lacam_escalation --bench bench_440_lllg_paper_repro -- --nocapture
//! ```

#![cfg(feature = "lacam_escalation")]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use katgpt_core::multi_agent_path::*;
use std::time::Instant;

// ─── Collision metrics ──────────────────────────────────────────────────────

/// Per-tick collision breakdown for one simulation run.
#[derive(Clone, Debug)]
struct CollisionMetrics {
    /// Total ticks simulated.
    ticks: usize,
    /// Ticks with zero vertex AND zero edge collisions.
    collision_free_ticks: usize,
    /// Ticks with ≥1 vertex collision (two agents share a cell).
    vertex_collision_ticks: usize,
    /// Ticks with ≥1 edge collision (two agents swap positions).
    edge_collision_ticks: usize,
    /// Tick at which the first collision occurred.
    first_collision_tick: Option<usize>,
}

impl CollisionMetrics {
    fn collision_free_rate(&self) -> f64 {
        self.collision_free_ticks as f64 / self.ticks as f64
    }
}

/// Check a joint action for vertex and edge collisions against the current config.
fn classify_tick(
    moves: &[GridPos],
    current: &JointConfig<GridPos>,
) -> (bool /*vertex*/, bool /*edge*/) {
    use std::collections::HashSet;
    let mut seen: HashSet<GridPos> = HashSet::with_capacity(moves.len());
    let mut vertex = false;
    for p in moves {
        if !seen.insert(*p) {
            vertex = true;
        }
    }
    // Edge (swap): agents i,j where moves[i] == current[j] and moves[j] == current[i].
    let mut edge = false;
    let n = moves.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if moves[i] == current.positions[j] && moves[j] == current.positions[i] {
                edge = true;
                break;
            }
        }
        if edge {
            break;
        }
    }
    (vertex, edge)
}

// ─── G6c bottleneck scenario ────────────────────────────────────────────────

/// Build the G6c bottleneck map: 20×20 grid with a vertical wall at x=10
/// (all of column 10 blocked) except for a 6-cell gap in the middle
/// (rows 7..=12 open). This creates a classic bottleneck where agents must
/// funnel through 6 cells.
///
/// ```text
///   0         1         2
///   0123456789012345678901
///   ....................   ← top (open)
///  ....................
///  ....................
///  ....................
///  ....................
///  ....................
///  ....................
///  ..........#.........   ← row 6: wall at x=10
///  ...........(gap)....   ← row 7: gap start
///  ...........(gap)....
///  ...........(gap)....
///  ...........(gap)....
///  ...........(gap)....
///  ...........(gap)....   ← row 12: gap end
///  ..........#.........   ← row 13: wall resumes
///  ....................
///  ...etc
/// ```
fn g6c_bottleneck_map() -> GridMap {
    let w = 20;
    let h = 20;
    let mut map = GridMap::empty(w, h);
    let wall_x = 10;
    // Gap rows: 7..=12 (6 cells open).
    let gap_start = 7;
    let gap_end = 12; // inclusive
    for y in 0..h {
        if y < gap_start || y > gap_end {
            map.set_wall(wall_x, y);
        }
    }
    map
}

/// Naive no-planner baseline: each agent takes the first neighbor that reduces
/// Manhattan distance to its goal, ignoring other agents. This is the MAPF
/// field-standard "no motion planner" baseline (Stern et al. 2019).
fn naive_step(
    config: &JointConfig<GridPos>,
    goals: &[GridPos],
    neighbors_fn: &dyn Fn(&GridPos) -> Vec<GridPos>,
    rng: &mut fastrand::Rng,
) -> JointAction<GridPos> {
    let n = config.n_agents();
    let mut moves: Vec<GridPos> = Vec::with_capacity(n);
    for i in 0..n {
        let current = &config.positions[i];
        let goal = &goals[i];
        let neighbors = neighbors_fn(current);
        // Among neighbors + wait, pick the one closest to goal (with ε tiebreak).
        let mut candidates: Vec<(GridPos, usize, f32)> = neighbors
            .iter()
            .cloned()
            .map(|p| {
                let dist = p.dist_heuristic(goal) as usize;
                (p, dist, rng.f32())
            })
            .collect();
        // Include wait.
        let wait_dist = current.dist_heuristic(goal) as usize;
        candidates.push((*current, wait_dist, rng.f32()));
        candidates.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)));
        moves.push(candidates[0].0);
    }
    JointAction::new(moves)
}

/// Run the G6c scenario with a given planner, return collision metrics.
fn run_g6c(
    map: &GridMap,
    starts: &[GridPos],
    goals: &[GridPos],
    ticks: usize,
    seed: u64,
    use_lacam: bool,
) -> CollisionMetrics {
    let n = starts.len();
    let config = JointConfig::new(starts.to_vec());
    let cfg = GuidanceConfig {
        w_phi: 5,
        alpha: 1.0,
        rounds: 2,
        max_expansions: 0,
    };
    let map_clone = map.clone();
    let neighbors_fn = move |p: &GridPos| map_clone.passable_neighbors(p);

    let mut guidance = SpaceTimeGuidance::new(cfg).with_neighbors({
        let m = map.clone();
        move |p| m.passable_neighbors(p)
    });
    let mut hindrance = BlockingCount::new();
    let warm = WarmStartCache::new(WarmStartScheme::default(), cfg.w_phi);
    let flow = GridFlowField::from_map(map);
    let map_clone2 = map.clone();
    let mut lacam = LifelongLaCam::new(warm)
        .with_neighbors(move |p| map_clone2.passable_neighbors(p))
        .with_flow_field(flow);

    let mut current = config;
    let mut rng = fastrand::Rng::with_seed(seed);

    let mut collision_free = 0usize;
    let mut vertex_coll = 0usize;
    let mut edge_coll = 0usize;
    let mut first_coll: Option<usize> = None;

    for tick in 0..ticks {
        let action = if use_lacam {
            lacam.tick(&current, goals, &mut guidance, &mut hindrance, &mut rng)
        } else {
            naive_step(&current, goals, &neighbors_fn, &mut rng)
        };
        let (vertex, edge) = classify_tick(&action.moves, &current);
        if !vertex && !edge {
            collision_free += 1;
        } else {
            if first_coll.is_none() {
                first_coll = Some(tick);
            }
            if vertex {
                vertex_coll += 1;
            }
            if edge {
                edge_coll += 1;
            }
        }
        current = JointConfig::new(action.moves);
    }

    let _ = n;
    CollisionMetrics {
        ticks,
        collision_free_ticks: collision_free,
        vertex_collision_ticks: vertex_coll,
        edge_collision_ticks: edge_coll,
        first_collision_tick: first_coll,
    }
}

/// Place 60 agents on the left side of the bottleneck, goals on the right side.
/// Starts: rows 0..59 mod 10 (first 6 columns), goals: mirrored to the right
/// side (columns 14..19). All starts are on distinct grid cells.
fn g6c_starts_goals(map: &GridMap) -> (Vec<GridPos>, Vec<GridPos>) {
    let n = 60;
    // Left side passable cells (x < 10), ordered top-to-bottom, left-to-right.
    let left: Vec<GridPos> = (0..map.height)
        .flat_map(|y| (0..10).map(move |x| GridPos::new(x, y)))
        .filter(|p| map.is_passable(p.x, p.y))
        .collect();
    // Right side passable cells (x > 10).
    let right: Vec<GridPos> = (0..map.height)
        .flat_map(|y| (11..map.width).map(move |x| GridPos::new(x, y)))
        .filter(|p| map.is_passable(p.x, p.y))
        .collect();

    let starts: Vec<GridPos> = left.iter().take(n).cloned().collect();
    // Goals: right-side cells, paired by index (mirror order).
    let goals: Vec<GridPos> = (0..n).map(|i| right[i % right.len()]).collect();
    (starts, goals)
}

// ─── Latency sweep ──────────────────────────────────────────────────────────

/// Run N ticks calling `lacam_escalation_step` directly with a given budget.
/// Returns (median_per_tick_us, max_per_tick_us, collision_free_rate).
fn run_latency_sweep(
    map: &GridMap,
    starts: &[GridPos],
    goals: &[GridPos],
    ticks: usize,
    seed: u64,
    budget: EscalationBudget,
) -> (f64, f64, f64) {
    let n = starts.len();
    let config = JointConfig::new(starts.to_vec());
    let cfg = GuidanceConfig {
        w_phi: 5,
        alpha: 1.0,
        rounds: 2,
        max_expansions: 0,
    };
    let mut guidance = SpaceTimeGuidance::new(cfg).with_neighbors({
        let m = map.clone();
        move |p| m.passable_neighbors(p)
    });
    let mut hindrance = BlockingCount::new();

    let mut guidance_scratch: Guidance<GridPos> = Vec::new();
    let priorities = vec![1.0f32; n];

    let map_arc = std::sync::Arc::new(map.clone());
    let map_for_fn = map_arc.clone();
    let neighbor_fn = move |p: &GridPos| map_for_fn.passable_neighbors(p);

    let no_flow = NoFlow;
    let mut current = config;
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut tick_us: Vec<f64> = Vec::with_capacity(ticks);
    let mut collision_free = 0usize;

    for _ in 0..ticks {
        guidance.compute_guidance(&current, goals, &mut guidance_scratch);

        let neighbor_fn_ref: &NeighborFn<GridPos> = &neighbor_fn;
        let start = Instant::now();
        let action = lacam_escalation_step(
            &current,
            &guidance_scratch,
            goals,
            &priorities,
            &mut hindrance,
            &no_flow,
            Some(neighbor_fn_ref),
            &mut rng,
            budget,
        );
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;
        tick_us.push(elapsed_us);

        let (vertex, edge) = classify_tick(&action.moves, &current);
        if !vertex && !edge {
            collision_free += 1;
        }
        current = JointConfig::new(action.moves);
    }

    tick_us.sort_by(|a, b| a.total_cmp(b));
    let median = tick_us[tick_us.len() / 2];
    let max = tick_us.last().copied().unwrap_or(0.0);
    let rate = collision_free as f64 / ticks as f64;
    (median, max, rate)
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 453 Phase 3 — LaCAM Escalation GOAT Gate Benchmark ===\n");

    // ─── G6c: Collision-freedom delta ──────────────────────────────────────
    println!("─── G6c: Collision-freedom delta — 60 agents, 20×20, 6-cell gap, 200 ticks ───\n");

    let map = g6c_bottleneck_map();
    let (starts, goals) = g6c_starts_goals(&map);
    println!(
        "  Map: {}×{}, wall at x=10, gap rows 7..=12 (6 cells)",
        map.width, map.height
    );
    println!(
        "  Agents: {} starts on left (x<10), goals on right (x>10)",
        starts.len()
    );
    println!();

    // LaCAM path (orchestrator with lacam_escalation feature ON).
    let lacam_metrics = run_g6c(&map, &starts, &goals, 200, 42, true);
    println!("  LaCAM (constraint tree + recursive PIBT):");
    println!(
        "    collision-free: {}/{} ({:.1}%)",
        lacam_metrics.collision_free_ticks,
        lacam_metrics.ticks,
        lacam_metrics.collision_free_rate() * 100.0
    );
    println!(
        "    vertex collisions: {}/{} ({:.1}%)",
        lacam_metrics.vertex_collision_ticks,
        lacam_metrics.ticks,
        lacam_metrics.vertex_collision_ticks as f64 / lacam_metrics.ticks as f64 * 100.0
    );
    println!(
        "    edge collisions:   {}/{} ({:.1}%)",
        lacam_metrics.edge_collision_ticks,
        lacam_metrics.ticks,
        lacam_metrics.edge_collision_ticks as f64 / lacam_metrics.ticks as f64 * 100.0
    );
    println!(
        "    first collision at tick: {:?}",
        lacam_metrics.first_collision_tick
    );
    println!();

    // Naive baseline (no planner).
    let naive_metrics = run_g6c(&map, &starts, &goals, 200, 42, false);
    println!("  Naive (no planner — Stern et al. 2019 baseline):");
    println!(
        "    collision-free: {}/{} ({:.1}%)",
        naive_metrics.collision_free_ticks,
        naive_metrics.ticks,
        naive_metrics.collision_free_rate() * 100.0
    );
    println!(
        "    vertex collisions: {}/{} ({:.1}%)",
        naive_metrics.vertex_collision_ticks,
        naive_metrics.ticks,
        naive_metrics.vertex_collision_ticks as f64 / naive_metrics.ticks as f64 * 100.0
    );
    println!(
        "    edge collisions:   {}/{} ({:.1}%)",
        naive_metrics.edge_collision_ticks,
        naive_metrics.ticks,
        naive_metrics.edge_collision_ticks as f64 / naive_metrics.ticks as f64 * 100.0
    );
    println!();

    let g6c_delta = lacam_metrics.collision_free_rate() - naive_metrics.collision_free_rate();
    println!(
        "  G6c delta = {:.3} − {:.3} = {:.3}",
        lacam_metrics.collision_free_rate(),
        naive_metrics.collision_free_rate(),
        g6c_delta
    );
    let g6c_verdict = if g6c_delta >= 0.50 {
        "PASS (≥ 0.50)"
    } else {
        "FAIL (< 0.50)"
    };
    println!("  G6c gate: {g6c_verdict} (threshold ≥ 0.50)");
    println!();

    // ─── Latency sweep ─────────────────────────────────────────────────────
    println!(
        "─── Latency sweep — max_nodes ∈ {{100, 500, 1000, 5000}}, 60 agents, 200 ticks ───\n"
    );
    println!(
        "  {:>10}  {:>14}  {:>14}  {:>18}",
        "max_nodes", "median (µs)", "max (µs)", "collision-free %"
    );
    println!("  {}", "─".repeat(62));

    for &max_nodes in &[100usize, 500, 1000, 5000] {
        let budget = EscalationBudget {
            max_nodes,
            time_budget_us: 5_000_000, // 5s — generous, let max_nodes be the binding constraint
            max_depth: 8, // Issue 546 multi-step: same default as EscalationBudget::default
            target_stuck_agents: false, // Legacy Plan 453 behavior for this bench
        };
        let (median_us, max_us, rate) = run_latency_sweep(&map, &starts, &goals, 200, 42, budget);
        println!(
            "  {:>10}  {:>14.1}  {:>14.1}  {:>17.1}%",
            max_nodes,
            median_us,
            max_us,
            rate * 100.0
        );
    }
    println!();

    // ─── Summary ───────────────────────────────────────────────────────────
    println!("─── Summary ───\n");
    println!("  G6c delta: {g6c_delta:.3} ({g6c_verdict})");
    println!(
        "  G-col gate: vertex collision rate {:.1}% (target ≤ 10%)",
        lacam_metrics.vertex_collision_ticks as f64 / lacam_metrics.ticks as f64 * 100.0
    );
    println!();
    println!("  Note: G1 throughput is measured by re-running bench_440_lllg_paper_repro");
    println!(
        "  compiled with --features lacam_escalation. See .benchmarks/453_lacam_escalation_goat.md"
    );
    println!("  for the full results table.");
}
