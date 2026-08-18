//! Diagnostic for Issue 147: verify the ht_chantry failure mode.
//!
//! Checks two hypotheses:
//! - A: Disconnected regions (map-gen bug) — flood-fill connectivity check.
//! - B: Severe bottleneck congestion — BFS reachability of random goals.
//!
//! Run with:
//!   cargo run --manifest-path crates/katgpt-core/Cargo.toml \
//!     --example ht_chantry_diagnostic --features multi_agent_path

use katgpt_core::multi_agent_path::position::{GridMap, GridPos};
use std::collections::VecDeque;

/// Reproduce the bench's ht_chantry_approx generator (it's private to the bench binary).
fn ht_chantry_approx(w: usize, h: usize) -> GridMap {
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
    map
}

/// Flood-fill connectivity check: count reachable cells from a seed.
fn count_reachable(map: &GridMap, seed: GridPos) -> std::collections::HashSet<GridPos> {
    use std::collections::HashSet;
    let mut visited: HashSet<GridPos> = HashSet::new();
    let mut queue: VecDeque<GridPos> = VecDeque::new();
    if map.is_passable(seed.x, seed.y) {
        visited.insert(seed);
        queue.push_back(seed);
    }
    while let Some(p) = queue.pop_front() {
        for n in map.passable_neighbors(&p) {
            if n == p {
                continue;
            }
            if visited.insert(n) {
                queue.push_back(n);
            }
        }
    }
    visited
}

fn count_passable(map: &GridMap) -> usize {
    (0..map.height)
        .flat_map(|y| (0..map.width).map(move |x| (x, y)))
        .filter(|(x, y)| map.is_passable(*x, *y))
        .count()
}

/// Count connected components.
fn count_components(map: &GridMap) -> usize {
    use std::collections::HashSet;
    let mut visited: HashSet<GridPos> = HashSet::new();
    let mut components = 0usize;
    for y in 0..map.height {
        for x in 0..map.width {
            if !map.is_passable(x, y) {
                continue;
            }
            let p = GridPos::new(x, y);
            if visited.contains(&p) {
                continue;
            }
            components += 1;
            let reachable = count_reachable(map, p);
            visited.extend(reachable);
        }
    }
    components
}

/// Post-process a map to guarantee it is a single connected component.
/// (Reproduced from the bench file's ensure_connected.)
fn ensure_connected(map: &mut GridMap) {
    let w = map.width;
    let h = map.height;
    let mut component_id: Vec<i32> = vec![-1; w * h];
    let mut component_sizes: Vec<usize> = Vec::new();
    let mut next_id = 0i32;
    for sy in 0..h {
        for sx in 0..w {
            if !map.is_passable(sx, sy) || component_id[sy * w + sx] != -1 {
                continue;
            }
            let id = next_id;
            next_id += 1;
            let mut size = 0usize;
            let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
            queue.push_back((sx, sy));
            component_id[sy * w + sx] = id;
            while let Some((x, y)) = queue.pop_front() {
                size += 1;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if map.is_passable(nx, ny) && component_id[ny * w + nx] == -1 {
                        component_id[ny * w + nx] = id;
                        queue.push_back((nx, ny));
                    }
                }
            }
            component_sizes.push(size);
        }
    }
    if next_id <= 1 {
        return;
    }
    let mut changed = true;
    while changed {
        changed = false;
        component_id.fill(-1);
        component_sizes.clear();
        next_id = 0;
        for sy in 0..h {
            for sx in 0..w {
                if !map.is_passable(sx, sy) || component_id[sy * w + sx] != -1 {
                    continue;
                }
                let id = next_id;
                next_id += 1;
                let mut size = 0usize;
                let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
                queue.push_back((sx, sy));
                component_id[sy * w + sx] = id;
                while let Some((x, y)) = queue.pop_front() {
                    size += 1;
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let nx = nx as usize;
                        let ny = ny as usize;
                        if map.is_passable(nx, ny) && component_id[ny * w + nx] == -1 {
                            component_id[ny * w + nx] = id;
                            queue.push_back((nx, ny));
                        }
                    }
                }
                component_sizes.push(size);
            }
        }
        if next_id <= 1 {
            break;
        }
        let main_id = component_sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| *s)
            .map_or(0, |(i, _)| i as i32);
        'outer: for wy in 0..h {
            for wx in 0..w {
                if map.is_passable(wx, wy) {
                    continue;
                }
                let mut touches_main = false;
                let mut touches_small = -1i32;
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
                    let cid = component_id[ny * w + nx];
                    if cid == main_id {
                        touches_main = true;
                    } else if cid != -1 {
                        touches_small = cid;
                    }
                }
                if touches_main && touches_small != -1 {
                    map.walls[wy][wx] = false;
                    changed = true;
                    break 'outer;
                }
            }
        }
    }
}

/// Degree histogram: how many passable neighbors each cell has.
fn degree_histogram(map: &GridMap) -> [usize; 6] {
    let mut hist = [0usize; 6];
    for y in 0..map.height {
        for x in 0..map.width {
            if !map.is_passable(x, y) {
                continue;
            }
            let p = GridPos::new(x, y);
            let deg = map
                .passable_neighbors(&p)
                .iter()
                .filter(|n| **n != p)
                .count();
            let idx = deg.min(5);
            hist[idx] += 1;
        }
    }
    hist
}

fn main() {
    // ── BEFORE fix: raw maze (Issues 140–144 benchmarked this) ──
    let raw_map = ht_chantry_approx(71, 53);
    let raw_passable = count_passable(&raw_map);
    println!("=== ht_chantry Diagnostic (Issue 147) ===\n");
    println!("── BEFORE connectivity fix (raw maze) ──");
    println!(
        "Map: 71x53 = {} cells, {} passable ({:.1}% open)",
        71 * 53,
        raw_passable,
        raw_passable as f64 / (71 * 53) as f64 * 100.0
    );
    let raw_components = count_components(&raw_map);
    println!("Connected components: {raw_components}\n");

    // ── AFTER fix: maze with ensure_connected ──
    let mut fixed_map = ht_chantry_approx(71, 53);
    ensure_connected(&mut fixed_map);
    let fixed_passable = count_passable(&fixed_map);
    let fixed_components = count_components(&fixed_map);
    let holes_punched = fixed_passable - raw_passable;

    println!("── AFTER connectivity fix (ensure_connected) ──");
    println!(
        "Map: 71x53 = {} cells, {} passable ({:.1}% open)",
        71 * 53,
        fixed_passable,
        fixed_passable as f64 / (71 * 53) as f64 * 100.0
    );
    println!("Connected components: {fixed_components}");
    println!("Holes punched to connect: {holes_punched}\n");

    // ── Degree histogram on the fixed map ──
    let hist = degree_histogram(&fixed_map);
    println!("Passable-neighbor degree histogram (fixed map):");
    for (deg, &count) in hist.iter().enumerate() {
        if count > 0 {
            println!("  degree {deg}: {count} cells");
        }
    }
    let dead_ends = hist[1];
    let corridors = hist[2];
    println!(
        "\nDead-end cells (degree 1): {dead_ends} ({:.1}%)",
        dead_ends as f64 / fixed_passable as f64 * 100.0
    );
    println!(
        "Corridor cells (degree 2): {corridors} ({:.1}%)",
        corridors as f64 / fixed_passable as f64 * 100.0
    );

    // ── Sample BFS distances from a corner ──
    let seed_cell = if fixed_map.is_passable(0, 0) {
        GridPos::new(0, 0)
    } else {
        (0..fixed_map.height)
            .flat_map(|y| (0..fixed_map.width).map(move |x| GridPos::new(x, y)))
            .find(|p| fixed_map.is_passable(p.x, p.y))
            .unwrap()
    };
    let reachable_from_seed = count_reachable(&fixed_map, seed_cell);
    println!(
        "\nReachable from {:?}: {} / {} passable ({:.1}%)",
        seed_cell,
        reachable_from_seed.len(),
        fixed_passable,
        reachable_from_seed.len() as f64 / fixed_passable as f64 * 100.0
    );

    println!("\n=== Verdict ===");
    if fixed_components == 1 {
        println!(
            "FIXED: map is now a single connected component (was {raw_components}). \
             Re-run the benchmark to measure TRUE algorithmic throughput."
        );
    } else {
        println!("STILL DISCONNECTED: {fixed_components} components remain.");
    }
}
