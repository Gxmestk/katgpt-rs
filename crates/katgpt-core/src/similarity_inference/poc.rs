//! G2 Emergent Cooperation PoC (Plan 526 Phase 2 — the load-bearing quality gate).
//!
//! Synthetic crowd: N=64 entities, half shared-shard pairs (same deterministic
//! policy), half random-shard pairs (independent random policies). Each pair
//! plays T=50 random 2×2 matrix games (info-gathering, perfect monitoring),
//! then a terminal Prisoner's Dilemma. The cooperation rate per pair type is
//! the G2 signal:
//!
//! - **PASS**: shared-shard pairs cooperate at >80%; random-shard pairs at <20%.
//! - **FAIL**: cooperation does not emerge, OR emerges for random-shard pairs too.
//!
//! Per the skill §3.6 defend-wrong protocol: if G2 FAILS, the numbers are
//! recorded honestly in `.benchmarks/526_similarity_inference_goat.md` and the
//! verdict is downgraded (architectural coverage stands — the math is correct
//! per G1; quality claim is "unproven on this domain").

use crate::similarity_inference::{SimilarityPosterior, canonical_pd, embedded_best_response};

/// A synthetic agent in the PoC crowd. Each agent has one `SimilarityPosterior`
/// per partner (here: exactly one partner, paired at construction).
struct PoCAgent {
    /// This agent's deterministic policy: maps a situation seed to an action
    /// (0 or 1). For shared-shard pairs, both agents share the SAME function —
    /// they always play identically. For random-shard pairs, each has its own
    /// independently-seeded RNG.
    policy_seed: u64,
    /// Posterior belief that the assigned partner shares this agent's policy.
    posterior: SimilarityPosterior,
    /// Number of actions in the game (always 2 for the 2×2 PoC).
    n_actions: usize,
}

impl PoCAgent {
    /// Pick an action for the given situation seed under this agent's policy.
    /// Deterministic given (policy_seed, situation_seed) — shared-shard pairs
    /// always agree.
    fn act(&self, situation_seed: u64) -> u8 {
        // Deterministic hash → action. xorshift-style mix.
        let mixed = mix(self.policy_seed, situation_seed);
        (mixed % self.n_actions as u64) as u8
    }

    /// Observe the partner's action and update the posterior.
    fn observe(&mut self, self_action: u8, partner_action: u8) {
        if partner_action == self_action {
            self.posterior.observe_match(self.n_actions);
        } else {
            self.posterior.observe_mismatch(self.n_actions);
        }
    }

    /// Decide cooperate (0) or defect (1) at the terminal PD, using the
    /// embedded best response with uniform partner marginal.
    fn terminal_action(&self) -> u8 {
        let payoff = canonical_pd();
        let marginal = [0.5_f32, 0.5];
        embedded_best_response(self.posterior.omega(), &payoff, &marginal).unwrap()
    }

    /// Current ω — for diagnostics.
    fn omega(&self) -> f32 {
        self.posterior.omega()
    }
}

/// Deterministic mixing function (xorshift-style). Same inputs → same output.
fn mix(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(b);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// A pair of agents. `PairKind::Shared` means both agents share the same
/// policy_seed; `PairKind::Random` means independent seeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairKind {
    Shared,
    Random,
}

/// Run one PoC trial: build the crowd, run T info-gathering rounds, then the
/// terminal PD. Returns the cooperation rates per pair kind.
struct PoCTrial {
    /// Prior α on identity. Plan default: 0.1.
    prior_alpha: f32,
    /// Number of info-gathering rounds. Plan default: 50.
    info_rounds: u32,
    /// Number of actions per game. Plan default: 2.
    n_actions: usize,
}

impl Default for PoCTrial {
    fn default() -> Self {
        Self {
            prior_alpha: 0.1,
            info_rounds: 50,
            n_actions: 2,
        }
    }
}

impl PoCTrial {
    /// Run the trial with a given RNG seed. Returns (shared_coop_rate,
    /// random_coop_rate) — fraction of pairs where BOTH agents cooperated at
    /// the terminal PD.
    fn run(&self, pair_seed: u64, kind: PairKind) -> (f32, f32, f32, f32) {
        // Returns (shared_coop_rate, random_coop_rate,
        //          shared_omega_mean, random_omega_mean).
        let n_pairs_per_kind = 32_usize; // → 64 entities per kind, 128 total
        let mut shared_coops = 0_usize;
        let mut random_coops = 0_usize;
        let mut shared_omega_sum = 0.0_f32;
        let mut random_omega_sum = 0.0_f32;

        for pair_idx in 0..n_pairs_per_kind {
            let (a_coop, a_omega) = self.run_one_agent(pair_seed, kind, pair_idx, /*is_focal=*/ true);
            let (b_coop, b_omega) = self.run_one_agent(pair_seed, kind, pair_idx, /*is_focal=*/ false);
            let both_cooperated = (a_coop == 0 && b_coop == 0) as usize;
            match kind {
                PairKind::Shared => {
                    shared_coops += both_cooperated;
                    shared_omega_sum += 0.5 * (a_omega + b_omega);
                }
                PairKind::Random => {
                    random_coops += both_cooperated;
                    random_omega_sum += 0.5 * (a_omega + b_omega);
                }
            }
        }

        let n = n_pairs_per_kind as f32;
        (
            shared_coops as f32 / n,
            random_coops as f32 / n,
            shared_omega_sum / n,
            random_omega_sum / n,
        )
    }

    /// Run one agent through T info-gathering rounds, then the terminal PD.
    /// Returns (terminal_action, omega_after_info_gathering).
    fn run_one_agent(
        &self,
        pair_seed: u64,
        kind: PairKind,
        pair_idx: usize,
        is_focal: bool,
    ) -> (u8, f32) {
        // The focal's policy_seed: a function of (pair_seed, pair_idx, "focal").
        let focal_seed = mix(pair_seed, (pair_idx as u64).wrapping_mul(2));
        // The partner's policy_seed: same as focal for Shared, different for Random.
        let partner_seed = match kind {
            PairKind::Shared => focal_seed, // identical policy
            PairKind::Random => mix(pair_seed, (pair_idx as u64).wrapping_mul(2).wrapping_add(1)),
        };

        let my_seed = if is_focal { focal_seed } else { partner_seed };
        let their_seed = if is_focal { partner_seed } else { focal_seed };

        let mut agent = PoCAgent {
            policy_seed: my_seed,
            posterior: SimilarityPosterior::new(self.prior_alpha).unwrap(),
            n_actions: self.n_actions,
        };

        // Info-gathering: T rounds, each with a random situation_seed.
        for round in 0..self.info_rounds {
            let situation_seed = mix(pair_seed, pair_idx as u64 * 1000 + round as u64);
            let my_action = agent.act(situation_seed);
            let their_action = act_with_seed(their_seed, situation_seed, self.n_actions);
            agent.observe(my_action, their_action);
        }

        let omega = agent.omega();
        let terminal = agent.terminal_action();
        (terminal, omega)
    }
}

fn act_with_seed(policy_seed: u64, situation_seed: u64, n_actions: usize) -> u8 {
    let mixed = mix(policy_seed, situation_seed);
    (mixed % n_actions as u64) as u8
}

#[test]
fn g2_emergent_cooperation_poc() {
    // G2 assertion: shared-shard pairs cooperate at >80%; random-shard pairs
    // at <20%. Run multiple trials with different seeds and report the mean.
    let trial = PoCTrial::default();
    let n_seeds = 10_u64;
    let mut shared_coop_sum = 0.0_f32;
    let mut random_coop_sum = 0.0_f32;
    let mut shared_omega_sum = 0.0_f32;
    let mut random_omega_sum = 0.0_f32;

    for seed in 0..n_seeds {
        let (scoop, _rcoop, somega, _romega) = trial.run(seed, PairKind::Shared);
        let (_scoop, rcoop, _somega, romega) = trial.run(seed, PairKind::Random);
        shared_coop_sum += scoop;
        random_coop_sum += rcoop;
        shared_omega_sum += somega;
        random_omega_sum += romega;
    }

    let n = n_seeds as f32;
    let shared_coop_rate = shared_coop_sum / n;
    let random_coop_rate = random_coop_sum / n;
    let shared_omega_mean = shared_omega_sum / n;
    let random_omega_mean = random_omega_sum / n;

    // Report (visible in test output with --nocapture).
    eprintln!("G2 PoC results (mean over {n_seeds} seeds, 32 pairs/seed):");
    eprintln!("  Shared-shard coop rate: {shared_coop_rate:.3} (target >0.80)");
    eprintln!("  Random-shard coop rate: {random_coop_rate:.3} (target <0.20)");
    eprintln!("  Shared-shard mean ω:    {shared_omega_mean:.4}");
    eprintln!("  Random-shard mean ω:    {random_omega_mean:.4}");

    // G2 PASS criteria.
    assert!(
        shared_coop_rate > 0.80,
        "G2 FAIL: shared-shard coop rate {shared_coop_rate:.3} ≤ 0.80"
    );
    assert!(
        random_coop_rate < 0.20,
        "G2 FAIL: random-shard coop rate {random_coop_rate:.3} ≥ 0.20"
    );
}

#[test]
fn g2_shared_pairs_never_mismatch() {
    // Sanity check: under the Shared kind, both agents in a pair share the
    // same policy_seed, so they must always agree on actions. This is the
    // mechanism that drives their ω → 1.
    let focal_seed = 42_u64;
    let partner_seed = focal_seed; // Shared kind
    let n_actions = 2;
    for situation in 0..1000_u64 {
        let a = act_with_seed(focal_seed, situation, n_actions);
        let b = act_with_seed(partner_seed, situation, n_actions);
        assert_eq!(a, b, "shared-shard pair disagreed at situation {situation}");
    }
}

#[test]
fn g2_random_pairs_mismatch_frequently() {
    // Sanity check: random-shard pairs should disagree ~50% of the time
    // (for n_actions=2, uniform actions, P(agree) = 0.5).
    let focal_seed = 42_u64;
    let partner_seed = 43_u64; // different → Random kind
    let n_actions = 2;
    let mut agreements = 0_u32;
    let n = 1000_u32;
    for situation in 0..n as u64 {
        let a = act_with_seed(focal_seed, situation, n_actions);
        let b = act_with_seed(partner_seed, situation, n_actions);
        if a == b {
            agreements += 1;
        }
    }
    let agree_rate = agreements as f32 / n as f32;
    eprintln!("random-shard agreement rate: {agree_rate:.3} (expected ~0.5)");
    // Should be close to 0.5; allow generous bounds for the deterministic mix.
    assert!(agree_rate > 0.4 && agree_rate < 0.6, "agreement rate {agree_rate} outside [0.4, 0.6]");
}
