//! Issue 546 de-risking diagnostic: deadlock-chain-length distribution on
//! ht_chantry.
//!
//! Plan 453 shipped bounded **one-step** LaCAM escalation (ht_chantry
//! throughput 0.01 → 0.28). The remaining G1 gap (0.28 < 0.30) requires
//! **multi-step** LaCAM. Before committing to a 2-3 day plan (~600-900 LOC
//! of new high-level config-search machinery), this diagnostic answers the
//! load-bearing question: **what depth does the constraint tree actually
//! need to reach?**
//!
//! ## Method
//!
//! For each tick of the bench_440 ht_chantry scenario (real MovingAI map,
//! 800 agents, 500 steps, `lacam_escalation` ON), we:
//!
//! 1. Identify **stuck agents** — those whose chosen move equals their
//!    current cell (`action.moves[i] == current.positions[i]`). These are
//!    the agents PIBT could not progress.
//! 2. Build a **blocking graph** over stuck agents: A→B if B's current
//!    cell is a passable neighbor of A's current cell (A is adjacent to B
//!    and would push into B). This captures the priority-inheritance
//!    chain shape that recursive PIBT inside `get_new_config` would have
//!    to unwind.
//! 3. Find weakly-connected components in this graph. **Component size is
//!    the chain length** a depth-K LaCAM constraint tree would need to
//!    resolve that cluster.
//! 4. Aggregate the distribution across all ticks and print a histogram.
//!
//! ## Interpretation
//!
//! - If ≥95% of clusters are size ≤ K, then depth-K LaCAM is sufficient.
//! - The tail (size > K) determines the fallback budget needed.
//! - If the tail is heavy (e.g., 10+ agents per cluster common), the maze
//!   has systemic structural deadlocks that no bounded-depth LaCAM can
//!   resolve — the fix must come from the flow field / guidance layer
//!   instead, and the multi-step plan would not close the gap.
//!
//! ## Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/546_diagnostic cargo run --release -p katgpt-core \
//!     --example ht_chantry_deadlock_chain_diagnostic \
//!     --features lacam_escalation -- --nocapture
//! ```

#![cfg(feature = "lacam_escalation")]
#![allow(clippy::too_many_arguments)]

use katgpt_core::multi_agent_path::*;
use std::collections::{HashMap, HashSet};

// ─── Map loading (mirror bench_440's ht_chantry-real path) ──────────────────

/// Load the real MovingAI ht_chantry map embedded at compile time.
/// Falls back to the synthetic approx if the real map is malformed (never
/// happens in practice — the file is checked in).
fn load_ht_chantry_real() -> GridMap {
    let embedded = include_str!("../benches/data/ht_chantry.map");
    GridMap::from_movingai(embedded).expect("ht_chantry.map must parse")
}

// ─── Connected-component analysis on the blocking graph ─────────────────────

/// Find weakly-connected component sizes in the blocking graph.
///
/// `stuck_indices`: indices into `positions` of stuck agents.
/// `positions`: current cell of every agent (stuck and unstuck).
/// `neighbors_fn`: passable-neighbor function for the map.
///
/// Returns the sizes of each connected component (unordered). Isolated
/// stuck agents (no stuck neighbor adjacent) contribute size 1.
fn blocking_components(
    stuck_indices: &[usize],
    positions: &[GridPos],
    neighbors_fn: &dyn Fn(&GridPos) -> Vec<GridPos>,
) -> Vec<usize> {
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        // Path compression.
        let mut cur = x;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    if stuck_indices.is_empty() {
        return Vec::new();
    }

    // Index only stuck agents for the union-find.
    let stuck_set: HashSet<usize> = stuck_indices.iter().copied().collect();
    // Position → stuck-agent-index (only stuck agents have entries).
    let mut pos_to_stuck: HashMap<GridPos, usize> = HashMap::with_capacity(stuck_indices.len());
    for (k, &i) in stuck_indices.iter().enumerate() {
        pos_to_stuck.insert(positions[i], k);
    }

    // Union-Find over stuck agents (indices 0..stuck_indices.len()).
    let n = stuck_indices.len();
    let mut parent: Vec<usize> = (0..n).collect();
    // `union` returns true if two distinct sets were merged.
    #[allow(clippy::needless_pass_by_value)]
    fn union(parent: &mut [usize], a: usize, b: usize) -> bool {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return false;
        }
        parent[ra] = rb;
        true
    }

    // For each stuck agent A, walk its passable neighbors. Any neighbor
    // occupied by another stuck agent B forms an A-B blocking edge.
    // (Position-adjacency captures the priority-inheritance chain shape:
    //  if A is next to B and both are stuck, A's preferred move is blocked
    //  by B's presence, which is exactly the push chain recursive PIBT
    //  unwinds.)
    for (k, &i) in stuck_indices.iter().enumerate() {
        let cell = &positions[i];
        for n_cell in neighbors_fn(cell) {
            if let Some(&k2) = pos_to_stuck.get(&n_cell)
                && k2 != k
            {
                union(&mut parent, k, k2);
            }
        }
    }

    // Tally component sizes.
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for k in 0..n {
        let root = find(&mut parent, k);
        *sizes.entry(root).or_insert(0) += 1;
    }
    let _ = stuck_set; // retained for clarity
    sizes.into_values().collect()
}

// ─── Main diagnostic ────────────────────────────────────────────────────────

fn main() {
    let map = load_ht_chantry_real();
    let n_passable: usize = (0..map.height)
        .flat_map(|y| (0..map.width).map(move |x| (x, y)))
        .filter(|(x, y)| map.is_passable(*x, *y))
        .count();
    println!("═══ Issue 546 Deadlock-Chain-Length Diagnostic ═══");
    println!(
        "Map: ht_chantry-real ({}×{}, {n_passable} passable)",
        map.width, map.height
    );

    // Same parameters as bench_440's G1 gate (the failing scenario).
    let n_agents: usize = 800;
    let steps: usize = 500;
    let seed: u64 = 42;
    println!("Agents: {n_agents}, Steps: {steps}, Seed: {seed}");
    println!("Planner: LifelongLaCam (lacam_escalation ON, default budget)");
    println!();

    assert!(
        n_agents < n_passable,
        "too many agents ({n_agents}) for {n_passable} passable cells"
    );

    // Place agents at random distinct passable positions (mirrors bench_440).
    let mut rng = fastrand::Rng::with_seed(seed);
    let passable: Vec<GridPos> = (0..map.height)
        .flat_map(|y| (0..map.width).map(move |x| GridPos::new(x, y)))
        .filter(|p| map.is_passable(p.x, p.y))
        .collect();
    let mut indices: Vec<usize> = (0..passable.len()).collect();
    // Fisher-Yates shuffle (deterministic).
    for k in (1..indices.len()).rev() {
        let j = rng.usize(0..=k);
        indices.swap(k, j);
    }
    let starts: Vec<GridPos> = indices[..n_agents].iter().map(|&i| passable[i]).collect();
    let config = JointConfig::new(starts.clone());

    // Random goals (may overlap with other agents' starts — mirrors bench_440).
    let mut goals: Vec<GridPos> = Vec::with_capacity(n_agents);
    for _ in 0..n_agents {
        let idx = rng.usize(0..passable.len());
        goals.push(passable[idx]);
    }

    // LLLG orchestrator (mirrors bench_440).
    let cfg = GuidanceConfig {
        w_phi: 5,
        alpha: 1.0,
        rounds: 2,
        max_expansions: 0,
    };
    let map_clone = map.clone();
    let mut guidance =
        SpaceTimeGuidance::new(cfg).with_neighbors(move |p| map_clone.passable_neighbors(p));
    let mut hindrance = BlockingCount::new();
    let warm = WarmStartCache::new(WarmStartScheme::default(), cfg.w_phi);
    let flow = GridFlowField::from_map(&map);
    let map_clone2 = map.clone();
    let mut lacam = LifelongLaCam::new(warm)
        .with_neighbors(move |p| map_clone2.passable_neighbors(p))
        .with_flow_field(flow);

    // Neighbor function for the blocking-graph analysis.
    let map_for_blocks = map.clone();
    let neighbors_fn = move |p: &GridPos| map_for_blocks.passable_neighbors(p);

    // ─── Run simulation, collect per-tick stuck-cluster distribution ──────
    let mut current = config;
    let mut completions: usize = 0;

    // Histogram: cluster_size → tick_count_hitting_that_size.
    // (Accumulates across all ticks; a single tick can contribute to
    // multiple buckets if it has clusters of different sizes.)
    let mut size_histogram: HashMap<usize, usize> = HashMap::new();
    // Per-tick max cluster size → distribution of "worst-case depth needed
    // this tick" (the load-bearing number for depth-K budgeting).
    let mut per_tick_max: HashMap<usize, usize> = HashMap::new();
    // Total clusters observed (for %-of-mass computation).
    let mut total_clusters: usize = 0;
    // Total stuck-agent observations (sum of cluster sizes).
    let mut total_stuck_observations: usize = 0;
    // Ticks with zero stuck agents (greedy fast path — no LaCAM needed).
    let mut fast_path_ticks: usize = 0;

    for _step in 0..steps {
        // Goal-completion + reassignment (mirrors bench_440).
        for (i, goal_i) in goals.iter_mut().enumerate().take(n_agents) {
            if current.positions[i] == *goal_i {
                completions += 1;
                let idx = rng.usize(0..passable.len());
                *goal_i = passable[idx];
            }
        }

        // Plan one tick.
        let action = lacam.tick(&current, &goals, &mut guidance, &mut hindrance, &mut rng);

        // Identify stuck agents.
        let stuck_indices: Vec<usize> = (0..n_agents)
            .filter(|&i| action.moves[i] == current.positions[i])
            .collect();

        if stuck_indices.is_empty() {
            fast_path_ticks += 1;
        } else {
            // Build blocking graph + find components.
            let components = blocking_components(&stuck_indices, &current.positions, &neighbors_fn);
            let tick_max = components.iter().copied().max().unwrap_or(0);
            *per_tick_max.entry(tick_max).or_insert(0) += 1;
            for &size in &components {
                *size_histogram.entry(size).or_insert(0) += 1;
                total_clusters += 1;
                total_stuck_observations += size;
            }
        }

        current = JointConfig::new(action.moves);
    }

    let throughput = completions as f64 / steps as f64;

    // ─── Report ───────────────────────────────────────────────────────────
    println!("─── Result ───");
    println!("Throughput: {throughput:.3} completions/step");
    println!(
        "Fast-path ticks (zero stuck): {fast_path_ticks}/{steps} ({:.1}%)",
        100.0 * fast_path_ticks as f64 / steps as f64
    );
    println!("Total stuck-clusters observed: {total_clusters}");
    println!("Total stuck-agent observations: {total_stuck_observations}");
    if total_clusters > 0 {
        println!(
            "Mean cluster size: {:.3} agents",
            total_stuck_observations as f64 / total_clusters as f64
        );
    }
    println!();

    println!("─── Cluster-size histogram (component size → cluster count) ───");
    println!("size | clusters  | %-mass(stuck) | %-mass(all-ticks)");
    println!("-----|-----------|---------------|----------------");
    let mut sizes: Vec<usize> = size_histogram.keys().copied().collect();
    sizes.sort_unstable();
    let max_size = *sizes.last().unwrap_or(&0);
    for size in &sizes {
        let count = size_histogram[size];
        let pct_mass = 100.0 * (count * size) as f64 / total_stuck_observations as f64;
        let pct_all = 100.0 * count as f64 / total_clusters as f64;
        let bar = "#".repeat((pct_all / 2.0).round() as usize);
        println!(" {size:>3} | {count:>7}   |   {pct_mass:>6.2}%      |  {pct_all:>6.2}%  {bar}");
    }
    println!();

    println!("─── Per-tick max-cluster-size distribution (depth-K sufficiency) ───");
    println!("This is the load-bearing table for depth-K LaCAM budgeting.");
    println!(
        "A tick with max-cluster-size K needs depth ≥ K to resolve all stuck clusters that tick."
    );
    println!();
    println!("max-size | ticks   | %-ticks | cumulative %-ticks (≤ this size)");
    println!("---------|---------|---------|--------------------------------");
    let mut max_sizes: Vec<usize> = per_tick_max.keys().copied().collect();
    max_sizes.sort_unstable();
    let mut cumulative = 0usize;
    let total_stuck_ticks: usize = per_tick_max.values().sum();
    for size in &max_sizes {
        let count = per_tick_max[size];
        cumulative += count;
        let pct = 100.0 * count as f64 / total_stuck_ticks as f64;
        let pct_cum = 100.0 * cumulative as f64 / total_stuck_ticks as f64;
        println!("   {size:>4}  | {count:>5}   | {pct:>5.2}% |  {pct_cum:>6.2}%");
    }
    println!();

    // ─── Verdict ──────────────────────────────────────────────────────────
    println!("─── Verdict (Issue 546 de-risking) ───");
    let p95_target = 95.0;
    // Find the smallest K such that cumulative %-ticks ≥ 95%.
    let mut cumulative2 = 0usize;
    let mut p95_k: Option<usize> = None;
    for size in &max_sizes {
        cumulative2 += per_tick_max[size];
        let pct_cum = 100.0 * cumulative2 as f64 / total_stuck_ticks as f64;
        if pct_cum >= p95_target && p95_k.is_none() {
            p95_k = Some(*size);
        }
    }
    let p99_target = 99.0;
    let mut cumulative3 = 0usize;
    let mut p99_k: Option<usize> = None;
    for size in &max_sizes {
        cumulative3 += per_tick_max[size];
        let pct_cum = 100.0 * cumulative3 as f64 / total_stuck_ticks as f64;
        if pct_cum >= p99_target && p99_k.is_none() {
            p99_k = Some(*size);
        }
    }

    match p95_k {
        Some(k) => {
            println!("P95 max-cluster-size = {k} (depth ≥ {k} resolves ≥95% of stuck ticks)");
        }
        None => println!("P95 max-cluster-size = N/A (insufficient data)"),
    }
    match p99_k {
        Some(k) => {
            println!("P99 max-cluster-size = {k} (depth ≥ {k} resolves ≥99% of stuck ticks)");
        }
        None => println!("P99 max-cluster-size = N/A (insufficient data)"),
    }
    println!("Max cluster size observed: {max_size}");
    println!();
    if let Some(k95) = p95_k {
        if k95 <= 2 {
            println!("→ DEPTH-2 LACAM IS SUFFICIENT to close the ht_chantry G1 gap");
            println!("  (P95 ≤ 2 means ≥95% of stuck ticks resolve with depth-2).");
            println!("  Recommend: commit to the multi-step plan with depth bound = 2.");
        } else if k95 <= 3 {
            println!("→ DEPTH-3 LACAM IS SUFFICIENT to resolve ≥95% of stuck ticks.");
            println!("  Depth-2 alone would leave >5% of ticks with unresolved clusters.");
            println!("  Recommend: multi-step plan with depth bound = 3, fallback to 2.");
        } else {
            println!("→ NO BOUNDED-DEPTH LACAM (≤{k95}) closes ≥95% of stuck ticks.");
            println!("  Tail is heavy — structural maze deadlocks dominate.");
            println!("  Recommend: multi-step LaCAM will NOT close the G1 gap alone;");
            println!("  pair with flow-field / guidance-layer work, or accept the");
            println!("  marginal G1 fail (0.28 vs 0.30) and move on.");
        }
    }
    println!();
    println!("═══ End diagnostic ═══");
}
