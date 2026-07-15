//! Config sweep for ht_chantry (Issue 147).
//!
//! After fixing map connectivity (37 → 1 component, throughput 0.15 → 1.47),
//! the remaining gap is genuine bottleneck congestion. This sweep tests
//! whether global-ish guidance (larger w_Φ or lower α) helps before we
//! commit to a full Guided-PIBT flow-direction implementation.
//!
//! Run with:
//!   cargo run --manifest-path crates/katgpt-core/Cargo.toml \
//!     --example ht_chantry_config_sweep --features multi_agent_path --release

#![allow(clippy::needless_range_loop)]

use katgpt_core::multi_agent_path::*;
use std::collections::VecDeque;
use std::time::Instant;

/// Reproduce the bench's ht_chantry_approx + ensure_connected.
fn ht_chantry_connected(w: usize, h: usize) -> GridMap {
    let mut map = GridMap::empty(w, h);
    let mut rng = fastrand::Rng::with_seed(99);
    for y in (4..h).step_by(8) {
        let gap1 = 3 + rng.usize(0..6);
        let gap2 = w / 2 + rng.usize(0..6);
        for x in 0..w {
            if x != gap1 && x != gap1 + 1 && x != gap2 && x != gap2 + 1 {
                map.set_wall(x, y);
            }
        }
    }
    for x in (6..w).step_by(10) {
        let gap = 3 + rng.usize(0..6);
        for y in 0..h {
            if y != gap && y != gap + 1 {
                map.set_wall(x, y);
            }
        }
    }
    ensure_connected(&mut map);
    map
}

fn ensure_connected(map: &mut GridMap) {
    let w = map.width;
    let h = map.height;
    let mut changed = true;
    while changed {
        changed = false;
        let mut comp_id: Vec<i32> = vec![-1; w * h];
        let mut sizes: Vec<usize> = Vec::new();
        let mut next_id = 0i32;
        for sy in 0..h {
            for sx in 0..w {
                if !map.is_passable(sx, sy) || comp_id[sy * w + sx] != -1 {
                    continue;
                }
                let id = next_id;
                next_id += 1;
                let mut size = 0usize;
                let mut q: VecDeque<(usize, usize)> = VecDeque::new();
                q.push_back((sx, sy));
                comp_id[sy * w + sx] = id;
                while let Some((x, y)) = q.pop_front() {
                    size += 1;
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let nx = nx as usize;
                        let ny = ny as usize;
                        if map.is_passable(nx, ny) && comp_id[ny * w + nx] == -1 {
                            comp_id[ny * w + nx] = id;
                            q.push_back((nx, ny));
                        }
                    }
                }
                sizes.push(size);
            }
        }
        if next_id <= 1 {
            break;
        }
        let main_id = sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| *s)
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        'outer: for wy in 0..h {
            for wx in 0..w {
                if map.is_passable(wx, wy) {
                    continue;
                }
                let mut tm = false;
                let mut ts = -1i32;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = wx as i32 + dx;
                    let ny = wy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if !map.is_passable(nx, ny) {
                        continue;
                    }
                    let cid = comp_id[ny * w + nx];
                    if cid == main_id {
                        tm = true;
                    } else if cid != -1 {
                        ts = cid;
                    }
                }
                if tm && ts != -1 {
                    map.walls[wy][wx] = false;
                    changed = true;
                    break 'outer;
                }
            }
        }
    }
}

fn count_passable(map: &GridMap) -> usize {
    (0..map.height)
        .flat_map(|y| (0..map.width).map(move |x| (x, y)))
        .filter(|(x, y)| map.is_passable(*x, *y))
        .count()
}

fn run_sim(
    map: &GridMap,
    n_agents: usize,
    steps: usize,
    seed: u64,
    cfg: GuidanceConfig,
) -> (f64, u32) {
    run_sim_inner(map, n_agents, steps, seed, cfg, false, 2.0)
}

fn run_sim_counter_flow(
    map: &GridMap,
    n_agents: usize,
    steps: usize,
    seed: u64,
    cfg: GuidanceConfig,
    gamma: f32,
) -> (f64, u32) {
    run_sim_inner(map, n_agents, steps, seed, cfg, true, gamma)
}

fn run_sim_inner(
    map: &GridMap,
    n_agents: usize,
    steps: usize,
    seed: u64,
    cfg: GuidanceConfig,
    counter_flow: bool,
    gamma: f32,
) -> (f64, u32) {
    let passable: Vec<GridPos> = (0..map.height)
        .flat_map(|y| (0..map.width).map(move |x| GridPos::new(x, y)))
        .filter(|p| map.is_passable(p.x, p.y))
        .collect();

    let mut rng = fastrand::Rng::with_seed(seed);
    let mut indices: Vec<usize> = (0..passable.len()).collect();
    for i in (1..indices.len()).rev() {
        let j = rng.usize(0..=i);
        indices.swap(i, j);
    }
    let starts: Vec<GridPos> = indices[..n_agents].iter().map(|&i| passable[i]).collect();
    let config = JointConfig::new(starts);

    let goals: Vec<GridPos> = (0..n_agents)
        .map(|_| {
            let idx = rng.usize(0..passable.len());
            passable[idx]
        })
        .collect();

    let map_clone = map.clone();
    let mut guidance = SpaceTimeGuidance::new(cfg)
        .with_neighbors(move |p| map_clone.passable_neighbors(p));
    let mut hindrance_blocking = BlockingCount::new();
    let mut hindrance_counter = CounterFlowHindrance::new().with_gamma(gamma);
    let warm = WarmStartCache::new(WarmStartScheme::default(), cfg.w_phi);
    let map_clone2 = map.clone();
    let mut lacam = LifelongLaCam::new(warm)
        .with_neighbors(move |p| map_clone2.passable_neighbors(p));

    let mut current = config;
    let mut goals = goals;
    let mut completions = 0usize;
    let mut max_stops = 0u32;
    let mut stop_counts = vec![vec![0u32; map.width]; map.height];

    for _step in 0..steps {
        for i in 0..n_agents {
            if current.positions[i] == goals[i] {
                completions += 1;
                let idx = rng.usize(0..passable.len());
                goals[i] = passable[idx];
            }
        }
        if counter_flow {
            hindrance_counter.set_goals(&goals);
            let action = lacam.tick(&current, &goals, &mut guidance, &mut hindrance_counter, &mut rng);
            for i in 0..n_agents {
                if action.moves[i] == current.positions[i] {
                    let x = current.positions[i].x;
                    let y = current.positions[i].y;
                    if x < map.width && y < map.height {
                        stop_counts[y][x] += 1;
                        if stop_counts[y][x] > max_stops {
                            max_stops = stop_counts[y][x];
                        }
                    }
                }
            }
            current = JointConfig::new(action.moves);
        } else {
            let action = lacam.tick(&current, &goals, &mut guidance, &mut hindrance_blocking, &mut rng);
            for i in 0..n_agents {
                if action.moves[i] == current.positions[i] {
                    let x = current.positions[i].x;
                    let y = current.positions[i].y;
                    if x < map.width && y < map.height {
                        stop_counts[y][x] += 1;
                        if stop_counts[y][x] > max_stops {
                            max_stops = stop_counts[y][x];
                        }
                    }
                }
            }
            current = JointConfig::new(action.moves);
        }
    }

    (completions as f64 / steps as f64, max_stops)
}

fn main() {
    let map = ht_chantry_connected(71, 53);
    let n_passable = count_passable(&map);
    println!("=== ht_chantry Config Sweep (Issue 147) ===");
    println!("Map: 71x53, {n_passable} passable cells\n");

    // Quick sweep: 200 agents, 100 steps (fast iteration).
    let n_agents = 200;
    let steps = 100;

    println!("Testing (w_phi, alpha, rounds) combos — {n_agents} agents, {steps} steps:\n");

    let configs: &[(usize, f32, usize, &str)] = &[
        (5, 1.0, 2, "paper default"),
        (5, 0.0, 1, "alpha=0 (pure BFS, no coll-avoid)"),
        (5, 0.0, 2, "alpha=0, 2 rounds"),
        (10, 1.0, 2, "w_phi=10 (longer window)"),
        (10, 0.5, 2, "w_phi=10, alpha=0.5"),
        (10, 0.0, 1, "w_phi=10, alpha=0"),
        (15, 1.0, 2, "w_phi=15 (very long window)"),
        (15, 0.0, 1, "w_phi=15, alpha=0"),
        (20, 0.0, 1, "w_phi=20, alpha=0 (global BFS)"),
    ];

    for (w_phi, alpha, rounds, label) in configs {
        let cfg = GuidanceConfig {
            w_phi: *w_phi,
            alpha: *alpha,
            rounds: *rounds,
            max_expansions: 0,
        };
        let start = Instant::now();
        let (throughput, max_stops) = run_sim(&map, n_agents, steps, 42, cfg);
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "  w_phi={:>2}, alpha={:.1}, rounds={}: throughput={:>6.2}, max_stops={:>4}, time={:.1}s  [{label}]",
            w_phi, alpha, rounds, throughput, max_stops, elapsed
        );
    }

    // Also test at higher density for the best configs.
    println!("
Density scaling (w_phi=5, alpha=1.0, rounds=2 — paper default):");
    println!("Throughput vs agent count reveals corridor saturation:\n");
    for &n_agents in &[50, 100, 200, 400, 600] {
        let cfg = GuidanceConfig::default();
        let steps = 100;
        let start = Instant::now();
        let (throughput, max_stops) = run_sim(&map, n_agents, steps, 42, cfg);
        let elapsed = start.elapsed().as_secs_f64();
        let per_agent = throughput / n_agents as f64;
        println!(
            "  {n_agents:>4} agents: throughput={:>6.2}, per-agent={:.5}, max_stops={:>4}, time={:.1}s",
            throughput, per_agent, max_stops, elapsed
        );
    }

    // ── Counter-flow Guided-PIBT test (Issue 147) ──
    println!("\n── Counter-flow hindrance (Guided-PIBT variant) ──");
    println!("200 agents, 100 steps, w_phi=5, alpha=1.0:\n");
    let n_agents = 200;
    let steps = 100;
    for &gamma in &[0.0, 1.0, 2.0, 5.0, 10.0] {
        let cfg = GuidanceConfig::default();
        let start = Instant::now();
        let (throughput, max_stops) = if gamma == 0.0 {
            run_sim(&map, n_agents, steps, 42, cfg)
        } else {
            run_sim_counter_flow(&map, n_agents, steps, 42, cfg, gamma)
        };
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "  gamma={:>4.1}: throughput={:>6.2}, max_stops={:>4}, time={:.1}s {}",
            gamma, throughput, max_stops, elapsed,
            if gamma == 0.0 { "[baseline]" } else { "[counter-flow]" }
        );
    }

    // Higher density counter-flow test.
    println!("\nCounter-flow at 400 agents, 100 steps:\n");
    let n_agents = 400;
    for &gamma in &[0.0, 2.0, 5.0] {
        let cfg = GuidanceConfig::default();
        let start = Instant::now();
        let (throughput, max_stops) = if gamma == 0.0 {
            run_sim(&map, n_agents, steps, 42, cfg)
        } else {
            run_sim_counter_flow(&map, n_agents, steps, 42, cfg, gamma)
        };
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "  gamma={:>4.1}: throughput={:>6.2}, max_stops={:>4}, time={:.1}s {}",
            gamma, throughput, max_stops, elapsed,
            if gamma == 0.0 { "[baseline]" } else { "[counter-flow]" }
        );
    }
}
