//! Issue 546 multi-step LaCAM A/B comparison on ht_chantry.
//!
//! Runs the ht_chantry-real map with two configurations:
//!   (A) Default budget (Plan 453 one-step LaCAM)
//!   (B) Multistep budget (Issue 546: stuck-agent targeting + depth 8)
//!
//! Reports the throughput delta. Stretch goal: ht_chantry ≥ 0.30 (close G1).
//! Minimum bar: no regression + measurable improvement.
//!
//! Run:
//!   CARGO_TARGET_DIR=/tmp/issue546_ab cargo run -p katgpt-core \
//!       --example ht_chantry_multistep_ab --features lacam_escalation \
//!       --release -- --nocapture

use katgpt_core::multi_agent_path::{
    BlockingCount, EscalationBudget, GridFlowField, GridMap, GridPos, GuidanceConfig,
    JointConfig, LifelongLaCam, SpaceTimeGuidance, WarmStartCache, WarmStartScheme,
};

const AGENTS: usize = 800;
const STEPS: usize = 200;
const SEED: u64 = 42;

fn run_with_budget(
    map: &GridMap,
    budget_name: &str,
    multistep: bool,
) -> (f64, f64, usize) {
    let mut rng = fastrand::Rng::with_seed(SEED);
    let passable: Vec<GridPos> = (0..map.height)
        .flat_map(|y| (0..map.width).map(move |x| GridPos::new(x, y)))
        .filter(|p| map.is_passable(p.x, p.y))
        .collect();
    assert!(passable.len() >= AGENTS, "map too small");

    let mut indices: Vec<usize> = (0..passable.len()).collect();
    // Fisher-Yates shuffle (deterministic via seeded rng).
    for k in (1..indices.len()).rev() {
        let j = rng.usize(0..=k);
        indices.swap(k, j);
    }
    let starts: Vec<GridPos> = indices[..AGENTS].iter().map(|&i| passable[i]).collect();
    let config = JointConfig::new(starts.clone());

    let mut goals: Vec<GridPos> = Vec::with_capacity(AGENTS);
    for _ in 0..AGENTS {
        let idx = rng.usize(0..passable.len());
        goals.push(passable[idx]);
    }

    let cfg = GuidanceConfig {
        w_phi: 5,
        alpha: 1.0,
        rounds: 2,
        max_expansions: 0,
    };
    let map_clone = map.clone();
    let mut guidance = SpaceTimeGuidance::new(cfg)
        .with_neighbors(move |p| map_clone.passable_neighbors(p));
    let mut hindrance = BlockingCount::new();
    let warm = WarmStartCache::new(WarmStartScheme::default(), cfg.w_phi);
    let flow = GridFlowField::from_map(map);
    let map_clone2 = map.clone();
    let lacam = LifelongLaCam::new(warm)
        .with_neighbors(move |p| map_clone2.passable_neighbors(p))
        .with_flow_field(flow);
    let mut lacam = if multistep {
        lacam.with_escalation_budget(EscalationBudget::multistep_default())
    } else {
        lacam
    };

    let mut current = config;
    let mut goals = goals;
    let mut completions = 0usize;
    let mut tick_times_ms: Vec<f64> = Vec::with_capacity(STEPS);

    for _ in 0..STEPS {
        // Reassign goals for agents that reached theirs (lifelong).
        for i in 0..AGENTS {
            if current.positions[i] == goals[i] {
                completions += 1;
                let idx = rng.usize(0..passable.len());
                goals[i] = passable[idx];
            }
        }

        let start_time = std::time::Instant::now();
        let action = lacam.tick(&current, &goals, &mut guidance, &mut hindrance, &mut rng);
        tick_times_ms.push(start_time.elapsed().as_secs_f64() * 1000.0);

        current = JointConfig::new(action.moves);
    }

    let throughput = completions as f64 / STEPS as f64;
    tick_times_ms.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    let median_ms = tick_times_ms[tick_times_ms.len() / 2];

    println!(
        "  {budget_name:<25}: throughput={throughput:>7.2} ({completions} completions / {STEPS} steps), median tick={median_ms:.2}ms"
    );

    // Touch the mut to silence warnings if unused.
    let _ = &mut lacam;

    (throughput, median_ms, completions)
}

fn main() {
    println!("=== Issue 546 multi-step LaCAM A/B on ht_chantry ===\n");
    println!("Config: {AGENTS} agents, {STEPS} steps, seed={SEED}\n");

    // Load the real MovingAI ht_chantry map (Issue 148). Falls back to
    // the synthetic approximation if the file is missing.
    let map = katgpt_core::multi_agent_path::GridMap::from_movingai(
        include_str!("../benches/data/ht_chantry.map"),
    )
    .unwrap_or_else(|| {
        eprintln!("WARNING: failed to parse data/ht_chantry.map, falling back to empty map");
        GridMap::empty(71, 53)
    });

    let npassable: usize = (0..map.height)
        .flat_map(|y| (0..map.width).map(move |x| (x, y)))
        .filter(|(x, y)| map.is_passable(*x, *y))
        .count();
    println!("Map: ht_chantry-real ({}×{} = {} cells, {} passable = {:.1}% open)\n",
        map.width, map.height, map.width * map.height, npassable,
        npassable as f64 / (map.width * map.height) as f64 * 100.0);

    let (t_default, m_default, c_default) = run_with_budget(&map, "default (Plan 453)", false);
    let (t_multistep, m_multistep, c_multistep) = run_with_budget(&map, "multistep (Issue 546)", true);

    println!();
    println!("─── Delta ───");
    let delta_t = t_multistep - t_default;
    let delta_pct = if t_default > 0.0 { delta_t / t_default * 100.0 } else { 0.0 };
    println!("  Throughput:   {delta_t:+.3} ({delta_pct:+.1}%)");
    let delta_latency = m_multistep - m_default;
    println!("  Median tick:  {delta_latency:+.2}ms (multistep is {}expensive)",
        if delta_latency > 0.0 { "more " } else { "less " });
    let delta_completions = c_multistep as i64 - c_default as i64;
    println!("  Completions:  {delta_completions:+}");

    println!();
    println!("─── Verdict ───");
    // Paper target ~17 (ratio ≥ 0.30 → throughput ≥ 5.1).
    let target_ratio = 0.30;
    let target_throughput = 17.0 * target_ratio; // ≈ 5.1
    println!("  Paper target: ~17 throughput units × {target_ratio} ratio = {target_throughput:.2}");
    let default_passes = t_default >= target_throughput;
    let multistep_passes = t_multistep >= target_throughput;
    println!("  Default   {t_default:.2} {} G1 ({})",
        if default_passes { "PASSES" } else { "FAILS" },
        if default_passes { "✓" } else { "✗" });
    println!("  Multistep {t_multistep:.2} {} G1 ({})",
        if multistep_passes { "PASSES" } else { "FAILS" },
        if multistep_passes { "✓" } else { "✗" });

    if multistep_passes && !default_passes {
        println!("\n  ★ Multistep CLOSES the ht_chantry G1 gap (Issue 546) ★");
    } else if t_multistep > t_default {
        println!("\n  ✓ Multistep improves throughput but doesn't fully close G1.");
        println!("    Pair with Proposal 006 (flow-field redesign) for full close.");
    } else if t_multistep < t_default {
        println!("\n  ✗ Multistep REGRESSES throughput — the stuck-agent targeting");
        println!("    hypothesis is wrong. Revert.");
    } else {
        println!("\n  · No change. The multi-step budget has zero effect on this map.");
    }
}
