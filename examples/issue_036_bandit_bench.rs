//! Issue 036 re-evaluation trigger check.
//!
//! Measures Bandit update() throughput 5 times to check whether the
//! re-evaluation trigger ("consistently below 420M across 5+ runs on
//! non-thermal-throttled hardware") is met.
//!
//! Run: cargo run --release --example issue_036_bandit_bench --features bandit

fn main() {
    use katgpt_rs::pruners::bandit::{BanditPruner, BanditStrategy};
    use katgpt_rs::speculative::types::NoScreeningPruner;
    use katgpt_rs::types::Rng;
    use std::time::Instant;

    let warmup = 100;
    let iters = 5_000;
    let num_arms = 100;
    let runs = 5;

    println!("=== Issue 036 Re-evaluation Trigger Check ===");
    println!(
        "BanditPruner update() — {iters} iters, {warmup} warmup, {num_arms} arms, {runs} runs"
    );
    println!("Re-evaluation trigger: consistently below 420M across 5+ runs");
    println!();

    let mut throughputs = Vec::with_capacity(runs);

    for run in 1..=runs {
        let mut rng = Rng::new(42 + run as u64);
        let mut bandit: BanditPruner<NoScreeningPruner> =
            BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);

        for _ in 0..warmup {
            let arm = (rng.next() as usize) % num_arms;
            bandit.update(arm, 0.5);
        }

        let start = Instant::now();
        for _ in 0..iters {
            let arm = (rng.next() as usize) % num_arms;
            bandit.update(arm, 0.5);
        }
        let elapsed = start.elapsed();
        let throughput = iters as f64 / elapsed.as_secs_f64();
        let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

        println!("  Run {run}: {throughput:.0} ops/s ({us:.3} µs/op)");
        throughputs.push(throughput);
    }

    let min = throughputs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = throughputs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean = throughputs.iter().sum::<f64>() / runs as f64;

    println!();
    println!("  Min:  {min:.0} ops/s");
    println!("  Max:  {max:.0} ops/s");
    println!("  Mean: {mean:.0} ops/s");
    println!();

    let below_threshold = throughputs.iter().filter(|&&t| t < 420_000_000.0).count();
    println!("  Runs below 420M: {below_threshold}/{runs}");

    if below_threshold == runs {
        println!("  ⚠️  RE-EVALUATION TRIGGER MET: all runs below 420M");
        println!("  → The Box<Extensions> refactor may be justified.");
    } else {
        println!("  ✅ Re-evaluation trigger NOT met: some runs ≥ 420M");
        println!("  → Issue 036 stays correctly deferred.");
    }

    // Note: this machine may thermal-throttle, which the trigger excludes.
    println!();
    println!("  NOTE: This machine may thermal-throttle. The re-evaluation trigger");
    println!("  requires 'non-thermal-throttled hardware'. Treat results as");
    println!("  provisional — a dedicated benchmark machine is needed for a");
    println!("  definitive verdict.");
}
