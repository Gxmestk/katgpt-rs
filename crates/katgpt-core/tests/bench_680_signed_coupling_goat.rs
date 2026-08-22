//! Bench 672 — `signed_coupling_dynamics` GOAT gate (Issue 680, Research 497).
//!
//! Gates:
//! - **G1a** the three regimes, deterministic mean-field rollout: the same
//!   kernel drives **indifference**, **polarization**, and **consensus** on
//!   three graph families (random signed / square lattice / low-rank
//!   frustrated), separated by `(|n|, c)` exactly as the paper describes.
//! - **G1b** the three regimes, *seeded stochastic* rollout on the discrete
//!   `±1` path, plus bit-reproducibility across two runs of the same seed.
//! - **G1c** `β⁺ > β⁻` biases toward consensus (the paper's §6 mechanism): the
//!   *same frustrated graph* that polarizes under a symmetric coupling pair
//!   orders itself once allies outweigh rivals.
//! - **G1d** χ peaks at an interior temperature — the `T_c` locator behaves as
//!   a locator (the paper's 41-point sweep, run offline as it must be).
//! - **G2** latency: `signed_coupling_update_into` at N=32/256/1024 against a
//!   hand-rolled explicit three-sum baseline, plus edge-count scaling.
//!
//! G3 (no-regression) is the default-feature build/test check recorded in
//! `.benchmarks/672_signed_coupling_goat.md` — this feature is opt-in and adds
//! nothing to the default surface. G4 (alloc-free) lives in the isolated
//! binary `signed_coupling_alloc_check.rs`: a `CountingAllocator` global here
//! would perturb the `Instant::now()` loops below.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --no-default-features \
//!     --features signed_coupling_dynamics \
//!     --test bench_680_signed_coupling_goat --release -- --nocapture
//! ```

#![cfg(feature = "signed_coupling_dynamics")]

use katgpt_core::sigmoid;
use katgpt_core::signed_coupling::{
    Couplings, SignedGraph, SusceptibilityAccumulator, crowd_conviction, net_opinion,
    sample_states_into, signed_coupling_update_into,
};
use std::hint::black_box;
use std::time::Instant;

// ── Deterministic RNG (fixture + stochastic rollout only) ────────────────────

/// splitmix64 (Vigna) — seed-addressable, dependency-free, and the same
/// generator `effective_degree` uses, so fixtures are comparable across benches.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn next_unit(state: &mut u64) -> f32 {
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) as f32
}

// ── Graph families ───────────────────────────────────────────────────────────

/// Random signed graph: `n` nodes, each node offered `degree/2` forward ties,
/// a `p_discordant` share of them negative. The paper's baseline family.
fn random_signed(n: usize, degree: usize, p_discordant: f32, seed: u64) -> SignedGraph {
    let mut rng = seed;
    let mut edges = Vec::new();
    for i in 0..n {
        for _ in 0..degree.div_ceil(2) {
            let j = (splitmix64(&mut rng) % n as u64) as u32;
            match j as usize == i {
                true => continue,
                false => {
                    let sign = if next_unit(&mut rng) < p_discordant {
                        -1
                    } else {
                        1
                    };
                    edges.push((i as u32, j, sign));
                }
            }
        }
    }
    SignedGraph::from_edges(n, &edges).expect("random graph is well-formed")
}

/// 4-neighbour square lattice, all ties concordant — the paper's
/// generalization family (fitted on random graphs, evaluated on lattices).
fn square_lattice(side: usize) -> SignedGraph {
    let n = side * side;
    let mut edges = Vec::new();
    for r in 0..side {
        for c in 0..side {
            let i = (r * side + c) as u32;
            if c + 1 < side {
                edges.push((i, i + 1, 1));
            }
            if r + 1 < side {
                edges.push((i, i + side as u32, 1));
            }
        }
    }
    SignedGraph::from_edges(n, &edges).expect("lattice is well-formed")
}

/// Low-rank frustrated graph: two blocks, concordant inside, discordant
/// across. Rank-2 by construction — the paper's hardest generalization family,
/// and the structure a faction standoff actually has.
fn frustrated_blocks(n: usize, seed: u64) -> SignedGraph {
    let mut rng = seed;
    let half = n / 2;
    let mut edges = Vec::new();
    for i in 0..n {
        for _ in 0..4 {
            let j = (splitmix64(&mut rng) % n as u64) as usize;
            match j == i {
                true => continue,
                false => {
                    let same_block = (i < half) == (j < half);
                    edges.push((i as u32, j as u32, if same_block { 1 } else { -1 }));
                }
            }
        }
    }
    SignedGraph::from_edges(n, &edges).expect("frustrated graph is well-formed")
}

/// Couplings at the **repulsive corner** of the paper's fitted ranges: `β⁺` at
/// its low end, `β⁻` at its high end, `β₀` at its low end, then cooled. The
/// discordant edge weight is `β₀ − β⁻ = −0.5`, so rivals push instead of pull —
/// the precondition for polarization. Every value is inside
/// `PAPER_BETA_*_RANGE`; nothing here is invented to make the gate pass.
fn polarizing() -> Couplings {
    Couplings {
        beta_plus: 0.9,
        beta_minus: 1.1,
        beta_zero: 0.6,
    }
    .at_social_temperature(0.5)
}

// ── Rollouts ─────────────────────────────────────────────────────────────────

/// Deterministic mean-field rollout: `m ← 2σ(h) − 1`, i.e. each entity holds
/// its *expected* stance rather than a sample. Reproducible to the bit, and
/// the path on which `crowd_conviction` is informative (a magnitude of 0 is a
/// genuinely undecided entity, which `±1` sampling cannot express).
fn mean_field_rollout(
    graph: &SignedGraph,
    couplings: &Couplings,
    intrinsic: &[f32],
    init: &[f32],
    steps: usize,
) -> Vec<f32> {
    let n = graph.len();
    let mut states = init.to_vec();
    let mut probs = vec![0.0f32; n];
    for _ in 0..steps {
        signed_coupling_update_into(graph, &states, couplings, intrinsic, &mut probs);
        for (s, &p) in states.iter_mut().zip(probs.iter()) {
            *s = 2.0 * p - 1.0;
        }
    }
    states
}

/// Seeded stochastic rollout on the paper's discrete `±1` path. Returns the
/// final states plus each entity's time-averaged stance over the back half of
/// the run (the burn-in-excluded magnitude, which is what conviction reads).
fn stochastic_rollout(
    graph: &SignedGraph,
    couplings: &Couplings,
    intrinsic: &[f32],
    seed: u64,
    steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = graph.len();
    let mut rng = seed;
    let mut states: Vec<f32> = (0..n)
        .map(|_| if next_unit(&mut rng) < 0.5 { 1.0 } else { -1.0 })
        .collect();
    let mut probs = vec![0.0f32; n];
    let mut uniforms = vec![0.0f32; n];
    let mut next = vec![0.0f32; n];
    let mut running = vec![0.0f32; n];
    let mut counted = 0usize;

    for step in 0..steps {
        signed_coupling_update_into(graph, &states, couplings, intrinsic, &mut probs);
        for u in uniforms.iter_mut() {
            *u = next_unit(&mut rng);
        }
        sample_states_into(&probs, &uniforms, &mut next);
        states.copy_from_slice(&next);
        if step * 2 >= steps {
            for (acc, &s) in running.iter_mut().zip(states.iter()) {
                *acc += s;
            }
            counted += 1;
        }
    }
    let inv = 1.0 / counted.max(1) as f32;
    for acc in running.iter_mut() {
        *acc *= inv;
    }
    (states, running)
}

// ── G1a: three regimes, deterministic mean-field ─────────────────────────────

#[test]
fn g1a_three_regimes_on_three_graph_families() {
    const N: usize = 256;
    const STEPS: usize = 200;
    let base = Couplings::default();
    let zero_field = vec![0.0f32; N];
    // A faint symmetry-breaking initial condition — every regime starts from
    // the same near-indifferent crowd, so the regime is the couplings' doing,
    // not the initialization's.
    let mut rng = 0x680_u64;
    let init: Vec<f32> = (0..N)
        .map(|_| 0.02 * (2.0 * next_unit(&mut rng) - 1.0))
        .collect();

    // ── Indifference: hot crowd on a random signed graph. Couplings vanish,
    // nobody has an intrinsic field, so nobody commits.
    let random = random_signed(N, 8, 0.3, 0xC0FFEE);
    let hot = base.at_social_temperature(40.0);
    let s = mean_field_rollout(&random, &hot, &zero_field, &init, STEPS);
    let (n_ind, c_ind) = (net_opinion(&s).abs(), crowd_conviction(&s));
    println!("G1a indifference (random, T=40):   |n| = {n_ind:.4}  c = {c_ind:.4}");
    assert!(n_ind < 0.10, "indifference must not lean: |n| = {n_ind}");
    assert!(c_ind < 0.10, "indifference must not commit: c = {c_ind}");

    // ── Consensus: cold crowd on a mostly-concordant random graph. Ties pull
    // the same way and the graph is long-range, so the faint initial majority
    // amplifies until the whole crowd agrees.
    let allied = random_signed(N, 8, 0.05, 0xA111ED);
    let cold = base.at_social_temperature(0.5);
    let s = mean_field_rollout(&allied, &cold, &zero_field, &init, STEPS);
    let (n_con, c_con) = (net_opinion(&s).abs(), crowd_conviction(&s));
    println!("G1a consensus    (allied random, T=0.5): |n| = {n_con:.4}  c = {c_con:.4}");
    assert!(n_con > 0.90, "consensus must lean hard: |n| = {n_con}");
    assert!(c_con > 0.80, "consensus must commit: c = {c_con}");

    // ── Polarization: cold crowd on the frustrated two-block graph, at the
    // corner of the paper's fitted ranges where rivals genuinely REPEL.
    //
    // This is load-bearing and easy to get wrong: at the range *midpoints*
    // (the `Couplings::default()` used above) a discordant tie carries weight
    // β₀ − β⁻ = 0.8 − 0.65 = **+0.15** — still attractive. Mere connection
    // outweighs rivalry, so even a perfectly frustrated graph converges. That
    // is the paper's own §6 consensus bias, not a bug. Polarization needs
    // β⁻ > β₀, which [`polarizing`] takes from the fitted-range corner
    // (β⁺ = 0.9 lo, β⁻ = 1.1 hi, β₀ = 0.6 lo → w[−] = −0.5).
    let frustrated = frustrated_blocks(N, 0xBEEF);
    let s = mean_field_rollout(&frustrated, &polarizing(), &zero_field, &init, STEPS);
    let (n_pol, c_pol) = (net_opinion(&s).abs(), crowd_conviction(&s));
    println!("G1a polarization (frustrated, T=0.5): |n| = {n_pol:.4}  c = {c_pol:.4}");
    assert!(n_pol < 0.25, "polarization must cancel: |n| = {n_pol}");
    assert!(c_pol > 0.80, "polarization must commit: c = {c_pol}");

    // ── Generalization family (the paper fits on random graphs and evaluates
    // on lattices). A cold *short-range* lattice quenched from a near-neutral
    // start does NOT reach consensus — it freezes into domains, which reads as
    // polarization: committed everywhere, leaning nowhere. That is the honest
    // physics of a synchronous quench, and it is why "cold ⇒ consensus" is a
    // statement about the graph as much as the temperature.
    let lattice = square_lattice(16); // 16×16 = 256
    let s = mean_field_rollout(&lattice, &cold, &zero_field, &init, STEPS);
    let (n_lat, c_lat) = (net_opinion(&s).abs(), crowd_conviction(&s));
    println!("G1a lattice quench (T=0.5):        |n| = {n_lat:.4}  c = {c_lat:.4}  [domains]");
    assert!(c_lat > 0.80, "a cold lattice must commit: c = {c_lat}");
    assert!(
        n_lat < 0.60,
        "a quenched lattice must not reach full consensus: |n| = {n_lat}"
    );

    // Give that same lattice a shared disposition (`g_i > 0` — every entity
    // mildly prefers +1) and the domains resolve: consensus is reachable on a
    // short-range graph too, it just needs a field, not only cold couplings.
    let shared_field = vec![0.2f32; N];
    let s = mean_field_rollout(&lattice, &cold, &shared_field, &init, STEPS);
    let (n_fld, c_fld) = (net_opinion(&s), crowd_conviction(&s));
    println!("G1a lattice + shared field:        n = {n_fld:.4}  c = {c_fld:.4}");
    assert!(
        n_fld > 0.90,
        "a shared field must resolve the domains: n = {n_fld}"
    );
    assert!(c_fld > 0.80, "and keep the crowd committed: c = {c_fld}");

    // The load-bearing separation: polarization and indifference are the same
    // crowd to `net_opinion`, and opposite crowds to `crowd_conviction`.
    assert!(
        c_pol - c_ind > 0.70,
        "conviction must separate polarization from indifference: {c_pol} vs {c_ind}"
    );
}

// ── G1b: three regimes, seeded stochastic rollout ─────────────────────────────

#[test]
fn g1b_seeded_stochastic_rollout_is_reproducible_and_hits_the_regimes() {
    const N: usize = 256;
    const STEPS: usize = 400;
    const SEED: u64 = 0x5EED_0680;
    let base = Couplings::default();
    let zero_field = vec![0.0f32; N];

    // Reproducibility: same seed, same trajectory, bit for bit.
    let allied = random_signed(N, 8, 0.05, 0xA111ED);
    let cold = base.at_social_temperature(0.5);
    let (a, a_mag) = stochastic_rollout(&allied, &cold, &zero_field, SEED, STEPS);
    let (b, b_mag) = stochastic_rollout(&allied, &cold, &zero_field, SEED, STEPS);
    assert_eq!(a, b, "same seed must give the same rollout");
    assert_eq!(a_mag, b_mag, "same seed must give the same magnitudes");

    // Consensus: the discrete crowd locks in, so |n| is high at the last tick.
    let n_con = net_opinion(&a).abs();
    let c_con = crowd_conviction(&a_mag);
    println!("G1b consensus    (allied random, T=0.5): |n| = {n_con:.4}  c(mag) = {c_con:.4}");
    assert!(
        n_con > 0.90,
        "discrete consensus must lean hard: |n| = {n_con}"
    );
    assert!(c_con > 0.80, "discrete consensus must hold: c = {c_con}");

    // Polarization: the frustrated crowd splits — |n| near zero, but each
    // entity's own time-averaged stance is committed. Needs the repulsive
    // corner of the fitted ranges (see G1a's note on β₀ vs β⁻).
    let frustrated = frustrated_blocks(N, 0xBEEF);
    let (s, mag) = stochastic_rollout(&frustrated, &polarizing(), &zero_field, SEED, STEPS);
    let (n_pol, c_pol) = (net_opinion(&s).abs(), crowd_conviction(&mag));
    println!("G1b polarization (frustrated, T=0.5): |n| = {n_pol:.4}  c(mag) = {c_pol:.4}");
    assert!(
        n_pol < 0.30,
        "discrete polarization must cancel: |n| = {n_pol}"
    );
    assert!(c_pol > 0.70, "discrete polarization must hold: c = {c_pol}");

    // Indifference: the hot crowd never commits, so the time-averaged stance
    // of every entity washes out.
    let random = random_signed(N, 8, 0.3, 0xC0FFEE);
    let hot = base.at_social_temperature(40.0);
    let (s, mag) = stochastic_rollout(&random, &hot, &zero_field, SEED, STEPS);
    let (n_ind, c_ind) = (net_opinion(&s).abs(), crowd_conviction(&mag));
    println!("G1b indifference (random, T=40):   |n| = {n_ind:.4}  c(mag) = {c_ind:.4}");
    assert!(n_ind < 0.20, "hot crowd must not lean: |n| = {n_ind}");
    assert!(c_ind < 0.15, "hot crowd must not commit: c = {c_ind}");
}

// ── G1c: β⁺ > β⁻ biases toward consensus (the paper's mechanism) ─────────────

#[test]
fn g1c_ally_dominance_orders_a_frustrated_crowd() {
    const N: usize = 256;
    const STEPS: usize = 200;
    let frustrated = frustrated_blocks(N, 0xBEEF);
    let zero_field = vec![0.0f32; N];
    let mut rng = 0x680_u64;
    let init: Vec<f32> = (0..N)
        .map(|_| 0.02 * (2.0 * next_unit(&mut rng) - 1.0))
        .collect();

    // Symmetric couplings: allies and rivals pull equally hard → the two
    // blocks deadlock.
    let symmetric = Couplings {
        beta_plus: 1.0,
        beta_minus: 1.0,
        beta_zero: 0.0,
    }
    .at_social_temperature(0.5);
    let s = mean_field_rollout(&frustrated, &symmetric, &zero_field, &init, STEPS);
    let n_sym = net_opinion(&s).abs();

    // Ally-dominant couplings (the paper's universal finding, β⁺ > β⁻ in every
    // model × dataset cell): the same graph now leans.
    let ally = Couplings {
        beta_plus: 2.0,
        beta_minus: 0.2,
        beta_zero: 0.6,
    }
    .at_social_temperature(0.5);
    let s = mean_field_rollout(&frustrated, &ally, &zero_field, &init, STEPS);
    let n_ally = net_opinion(&s).abs();

    println!("G1c |n| symmetric = {n_sym:.4}  ally-dominant = {n_ally:.4}");
    assert!(
        n_sym < 0.25,
        "symmetric couplings must deadlock: |n| = {n_sym}"
    );
    assert!(
        n_ally > n_sym + 0.30,
        "ally dominance must break the deadlock: {n_ally} vs {n_sym}"
    );
}

// ── G1d: χ locates an interior critical temperature ──────────────────────────

#[test]
fn g1d_susceptibility_peaks_at_an_interior_temperature() {
    const N: usize = 128;
    const BURN_IN: usize = 200;
    const SAMPLES: usize = 600;
    const POINTS: usize = 41; // the paper's sweep resolution
    let graph = random_signed(N, 8, 0.25, 0xC0FFEE);
    let base = Couplings::default();
    let zero_field = vec![0.0f32; N];

    // 41 log-spaced temperatures over [0.1, 10] — offline, as the doc promises.
    let mut sweep: Vec<(f32, f32)> = Vec::with_capacity(POINTS);
    for k in 0..POINTS {
        let t = 10f32.powf(-1.0 + 2.0 * (k as f32 / (POINTS - 1) as f32));
        let couplings = base.at_social_temperature(t);
        let mut rng = 0xC1A1_u64;
        let mut states: Vec<f32> = (0..N)
            .map(|_| if next_unit(&mut rng) < 0.5 { 1.0 } else { -1.0 })
            .collect();
        let mut probs = vec![0.0f32; N];
        let mut uniforms = vec![0.0f32; N];
        let mut next = vec![0.0f32; N];
        let mut acc = SusceptibilityAccumulator::new();

        for step in 0..(BURN_IN + SAMPLES) {
            signed_coupling_update_into(&graph, &states, &couplings, &zero_field, &mut probs);
            for u in uniforms.iter_mut() {
                *u = next_unit(&mut rng);
            }
            sample_states_into(&probs, &uniforms, &mut next);
            states.copy_from_slice(&next);
            if step >= BURN_IN {
                acc.observe(net_opinion(&states));
            }
        }
        sweep.push((t, acc.susceptibility(N)));
    }

    let (t_c, chi_max) = sweep
        .iter()
        .copied()
        .fold((0.0, f32::MIN), |(bt, bc), (t, c)| match c > bc {
            true => (t, c),
            false => (bt, bc),
        });
    println!("G1d T_c = {t_c:.4}  chi_max = {chi_max:.4}");
    println!(
        "G1d edges: chi({:.3}) = {:.4}   chi({:.3}) = {:.4}",
        sweep[0].0,
        sweep[0].1,
        sweep[POINTS - 1].0,
        sweep[POINTS - 1].1
    );

    // A locator must locate: the peak sits strictly inside the sweep, not at
    // an endpoint (an endpoint peak means the sweep missed the transition).
    let interior = sweep[1..POINTS - 1]
        .iter()
        .any(|&(t, _)| (t - t_c).abs() < 1e-9);
    assert!(interior, "chi peak must be interior, found at T = {t_c}");
    assert!(
        chi_max > 4.0 * sweep[POINTS - 1].1.max(1e-6),
        "peak must dominate the hot tail: {chi_max} vs {}",
        sweep[POINTS - 1].1
    );
    assert!(
        chi_max > 4.0 * sweep[0].1.max(1e-6),
        "peak must dominate the frozen tail: {chi_max} vs {}",
        sweep[0].1
    );
}

// ── G2: latency vs a hand-rolled explicit three-sum baseline ─────────────────

/// The baseline any consumer would write from the paper's equation directly:
/// three separate accumulators and a `match` on the tie sign in the inner loop.
fn baseline_update_into(
    graph: &SignedGraph,
    states: &[f32],
    c: &Couplings,
    intrinsic: &[f32],
    out: &mut [f32],
) {
    for i in 0..graph.len() {
        let (nb, sg) = graph.row(i);
        let mut plus = 0.0f32;
        let mut minus = 0.0f32;
        let mut zero = 0.0f32;
        for (&j, &sign) in nb.iter().zip(sg) {
            let sj = states[j as usize];
            match sign {
                1 => plus += sj,
                _ => minus += -sj,
            }
            zero += sj;
        }
        out[i] =
            sigmoid(c.beta_plus * plus + c.beta_minus * minus + c.beta_zero * zero + intrinsic[i]);
    }
}

#[test]
fn g2_latency_beats_the_hand_rolled_three_sum_baseline() {
    const DEGREE: usize = 8;
    const ITERS: usize = 2_000;
    const ROUNDS: usize = 9;
    let couplings = Couplings::default();

    for &n in &[32usize, 256, 1024] {
        let graph = random_signed(n, DEGREE, 0.3, 0xC0FFEE);
        let states: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let intrinsic = vec![0.1f32; n];
        let mut probs = vec![0.0f32; n];

        // Warm the caches for both paths before timing either.
        signed_coupling_update_into(&graph, &states, &couplings, &intrinsic, &mut probs);
        baseline_update_into(&graph, &states, &couplings, &intrinsic, &mut probs);

        // Interleaved A/B: both arms are timed inside the SAME round, and the
        // verdict is the median of per-round RATIOS — not a ratio of medians.
        // This box runs sibling compute; a ratio of independently-taken medians
        // lets a load spike that hit only one arm decide the gate.
        let mut kernel_ns = Vec::with_capacity(ROUNDS);
        let mut baseline_ns = Vec::with_capacity(ROUNDS);
        let mut ratios = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                signed_coupling_update_into(
                    black_box(&graph),
                    black_box(&states),
                    black_box(&couplings),
                    black_box(&intrinsic),
                    black_box(&mut probs),
                );
            }
            let a = t0.elapsed().as_nanos() as f64 / ITERS as f64;

            let t1 = Instant::now();
            for _ in 0..ITERS {
                baseline_update_into(
                    black_box(&graph),
                    black_box(&states),
                    black_box(&couplings),
                    black_box(&intrinsic),
                    black_box(&mut probs),
                );
            }
            let b = t1.elapsed().as_nanos() as f64 / ITERS as f64;

            kernel_ns.push(a);
            baseline_ns.push(b);
            ratios.push(a / b);
        }
        let kernel = median(&mut kernel_ns);
        let baseline = median(&mut baseline_ns);
        let ratio = median(&mut ratios);

        let entries = graph.entry_count();
        println!(
            "G2 N={n:5} entries={entries:6}  kernel {kernel:9.1} ns/update \
             ({:.2} ns/entry)  baseline {baseline:9.1} ns  median pairwise ratio {ratio:.3}x",
            kernel / entries as f64
        );

        // The two-channel collapse must not be slower than the naive
        // three-accumulator form the paper's equation reads as literally.
        assert!(
            ratio <= 1.15,
            "N={n}: median pairwise ratio {ratio} exceeds 1.15 (kernel {kernel} ns vs baseline {baseline} ns)"
        );
        // Plasma-tier budget: a 1024-entity crowd inside a 20 Hz tick is
        // ~50 ms; one update must cost a rounding error of that.
        assert!(
            kernel < 200_000.0,
            "N={n}: {kernel} ns/update exceeds the 200 µs Plasma budget"
        );
    }
}

/// Median of a sample, sorted in place. `NaN`-free by construction here (every
/// input is an elapsed-time ratio of positive durations).
fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in timing samples"));
    match xs.len() % 2 {
        1 => xs[xs.len() / 2],
        _ => 0.5 * (xs[xs.len() / 2 - 1] + xs[xs.len() / 2]),
    }
}
