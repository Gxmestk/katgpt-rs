//! Bandit session orchestration (extracted from mod.rs by Issue 177).
//!
//! BanditEvent + BanditResult + BanditSession — multi-armed bandit episode
//! runner with strategy dispatch, trial-log persistence, and review metrics.

use std::sync::Arc;

use katgpt_types::Rng;

use super::environment::BanditEnv;
use super::{BanditStats, BanditStrategy, make_stats};
use crate::review_metrics::ReviewMetrics;
#[cfg(feature = "safe_bandit")]
use crate::safe_phased::SafePhasedState;
use crate::trial_log::{TrialLog, TrialRecord};

// ── Bandit Event ────────────────────────────────────────────────

/// Events emitted during bandit session execution.
#[derive(Clone, Debug)]
pub enum BanditEvent {
    /// An arm was pulled and reward observed.
    Pull {
        episode: usize,
        arm: usize,
        reward: f32,
        q_value: f32,
    },
    /// Episode completed with cumulative stats.
    EpisodeComplete {
        episode: usize,
        arm: usize,
        reward: f32,
        cumulative_reward: f32,
        cumulative_regret: f32,
    },
    /// Session completed with final stats.
    SessionComplete {
        total_episodes: usize,
        total_reward: f32,
        total_regret: f32,
        best_arm: usize,
        optimal_arm: usize,
    },
}

// ── Bandit Result ───────────────────────────────────────────────

/// Final result of a bandit session.
#[derive(Clone, Debug)]
pub struct BanditResult {
    /// Total episodes run.
    pub total_episodes: usize,
    /// Sum of all observed rewards.
    pub total_reward: f32,
    /// Sum of per-episode regret: Σ(optimal_reward - arm_expected_reward).
    pub total_regret: f32,
    /// Arm with highest Q-value at session end.
    pub best_arm: usize,
    /// True optimal arm from the environment.
    pub optimal_arm: usize,
    /// Final Q-value estimates.
    pub q_values: Vec<f32>,
    /// Final visit counts.
    pub visits: Vec<u32>,
}

impl BanditResult {
    /// Whether the bandit found the true optimal arm.
    pub fn found_optimal(&self) -> bool {
        self.best_arm == self.optimal_arm
    }

    /// Average reward per episode.
    pub fn avg_reward(&self) -> f32 {
        if self.total_episodes == 0 {
            0.0
        } else {
            self.total_reward / self.total_episodes as f32
        }
    }

    /// Average regret per episode.
    pub fn avg_regret(&self) -> f32 {
        if self.total_episodes == 0 {
            0.0
        } else {
            self.total_regret / self.total_episodes as f32
        }
    }
}

// ── Bandit Session ──────────────────────────────────────────────

/// Orchestrates multi-armed bandit episodes.
///
/// Runs N episodes of arm selection → reward observation → Q-value update.
/// Tracks cumulative reward and pseudo-regret. Emits events for logging.
///
/// # Example
///
/// ```rust,ignore
/// let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
/// let session = BanditSession::new(env, BanditStrategy::ThompsonSampling);
/// let (events, result) = session.run(500, &mut Rng::new(42));
/// assert!(result.found_optimal());
/// ```
pub struct BanditSession<E: BanditEnv> {
    env: E,
    strategy: BanditStrategy,
    stats: BanditStats,
    cumulative_reward: f32,
    cumulative_regret: f32,
    /// Optional review metrics for inference-time feedback tracking (Plan 036).
    review_metrics: Option<Arc<ReviewMetrics>>,
    /// Optional safe-phased state for PrudentBanker (Plan 137).
    #[cfg(feature = "safe_bandit")]
    safe_phased_state: Option<SafePhasedState>,
}

impl<E: BanditEnv> BanditSession<E> {
    /// Create a new bandit session with the given environment and strategy.
    pub fn new(env: E, strategy: BanditStrategy) -> Self {
        let num_arms = env.num_arms();
        #[cfg(feature = "safe_bandit")]
        let safe_phased_state = match &strategy {
            BanditStrategy::SafePhased {
                baseline_arm,
                delta,
                estimated_delay,
            } => Some(SafePhasedState::new(
                *baseline_arm,
                *delta,
                *estimated_delay,
                num_arms,
            )),
            _ => None,
        };
        let stats = make_stats(num_arms, &strategy);
        Self {
            env,
            strategy,
            stats,
            cumulative_reward: 0.0,
            cumulative_regret: 0.0,
            review_metrics: None,
            #[cfg(feature = "safe_bandit")]
            safe_phased_state,
        }
    }

    /// Enable review metrics tracking (Plan 036, builder pattern).
    ///
    /// After each episode, records whether the bandit's pick was the
    /// optimal arm vs whether a simulated random pick would have been.
    /// The same `Arc<ReviewMetrics>` can be shared across components.
    pub fn with_metrics(mut self, metrics: Arc<ReviewMetrics>) -> Self {
        self.review_metrics = Some(metrics);
        self
    }

    /// Select an arm based on the current strategy and stats.
    fn select_arm(&self, rng: &mut Rng) -> usize {
        let num_arms = self.env.num_arms();

        // Cold start: play each arm once
        for i in 0..num_arms {
            if self.stats.visit_count(i) == 0 {
                return i;
            }
        }

        match &self.strategy {
            BanditStrategy::Ucb1 => self.select_ucb1(),
            BanditStrategy::EpsilonGreedy { epsilon, .. } => {
                self.select_epsilon_greedy(*epsilon, rng)
            }
            BanditStrategy::ThompsonSampling => self.select_thompson(rng),
            BanditStrategy::VarianceEpsilon { .. } => self.select_variance_epsilon(rng),
            #[cfg(feature = "tes_loop")]
            BanditStrategy::Rpucg { .. } => self.select_ucb1(), // Flat bandit fallback; graph propagation in TesLoop
            BanditStrategy::RandOptAdaptive {
                density_threshold, ..
            } => {
                // Density-aware fallback: use threshold as epsilon until full implementation
                self.select_epsilon_greedy(*density_threshold, rng)
            }
            #[cfg(feature = "safe_bandit")]
            BanditStrategy::SafePhased { .. } => self.select_safe_phased(rng),
            BanditStrategy::CurvatureInfluence { .. } => self.select_ucb1(), // UCB1 base with CIAB scoring override
        }
    }

    fn select_ucb1(&self) -> usize {
        let n = self.env.num_arms();
        if n == 0 {
            return 0;
        }
        // Inline UCB1 scoring with ln(total) hoisted out of the per-arm loop.
        // Each `stats.ucb1_score(i)` call recomputes `total.ln()`; for N arms
        // that is N transcendental calls per arm-selection. We pull `ln_total`
        // and the q/visits slices out once and keep the loop branch-free.
        let total_pulls = self.stats.total_pulls();
        if total_pulls == 0 {
            return 0;
        }
        let ln_total = 2.0_f32 * (total_pulls as f32).ln();
        let q_values = self.stats.q_values();
        let visits = self.stats.visits();
        let mut best_idx = 0;
        let mut best_score = if visits[0] == 0 {
            f32::MAX
        } else {
            q_values[0] + (ln_total / visits[0] as f32).sqrt()
        };
        for i in 1..n {
            let s = if visits[i] == 0 {
                f32::MAX
            } else {
                q_values[i] + (ln_total / visits[i] as f32).sqrt()
            };
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        best_idx
    }

    fn select_epsilon_greedy(&self, epsilon: f32, rng: &mut Rng) -> usize {
        let num_arms = self.env.num_arms();
        if rng.uniform() < epsilon {
            // Explore: random arm
            (rng.uniform() * num_arms as f32) as usize % num_arms
        } else {
            // Exploit: best Q-value
            self.stats.best_arm()
        }
    }

    fn select_thompson(&self, rng: &mut Rng) -> usize {
        let n = self.env.num_arms();
        if n == 0 {
            return 0;
        }
        // Manual indexed loop mirroring select_ucb1: avoids iterator state-machine
        // overhead, tuple construction, and the partial_cmp().unwrap_or() branch
        // per element. `>=` preserves max_by's "last maximum wins on ties" semantics.
        let mut best_idx = 0;
        let mut best_score = self.stats.thompson_sample(0, rng);
        for i in 1..n {
            let s = self.stats.thompson_sample(i, rng);
            if s >= best_score {
                best_score = s;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Variance-minimized epsilon selection (RePlaid-inspired).
    ///
    /// Adapts exploration rate based on mean reward variance across arms.
    /// High variance → more exploration; low variance → more exploitation.
    fn select_variance_epsilon(&self, rng: &mut Rng) -> usize {
        let mean_var = self.stats.mean_reward_variance();
        let adapted_eps = match &self.strategy {
            BanditStrategy::VarianceEpsilon { epsilon, lr, .. } => {
                let factor = 1.0 + lr * mean_var.sqrt();
                (epsilon * factor).clamp(0.01, 1.0)
            }
            _ => 0.1,
        };
        let num_arms = self.env.num_arms();
        if rng.uniform() < adapted_eps {
            (rng.uniform() * num_arms as f32) as usize % num_arms
        } else {
            self.stats.best_arm()
        }
    }

    /// Decay epsilon (EpsilonGreedy only).
    fn decay_epsilon(&mut self) {
        if let BanditStrategy::EpsilonGreedy { epsilon, decay } = &mut self.strategy {
            *epsilon *= *decay;
        }
    }

    /// Select arm using safe-phased mixture (Plan 137).
    ///
    /// Uses UCB1 as the active arm selector, then applies safe mixture
    /// with the baseline arm based on current αₖ.
    #[cfg(feature = "safe_bandit")]
    fn select_safe_phased(&self, rng: &mut Rng) -> usize {
        let active_arm = self.select_ucb1();
        if let Some(ref state) = self.safe_phased_state {
            state.select_with_safe_mixture(active_arm, rng)
        } else {
            active_arm
        }
    }

    /// Update safe-phased state after observing reward (Plan 137).
    ///
    /// Uses the **active** arm's expected reward for gap tracking,
    /// not the selected arm. This ensures the gap accurately reflects
    /// how the exploratory active arm compares to the safe baseline,
    /// regardless of whether the mixture selected the baseline.
    #[cfg(feature = "safe_bandit")]
    fn update_safe_phased(&mut self, selected_arm: usize, reward: f32) {
        if let Some(ref mut state) = self.safe_phased_state {
            state.record_round();
            let baseline_arm = state.baseline_arm();
            // If baseline was selected, use expected reward for gap tracking
            // (to avoid always seeing 0 gap when baseline dominates)
            if selected_arm == baseline_arm {
                // Baseline selected: no gap contribution (we got what we expected)
                // But still track the active arm's hypothetical performance
                // by not accumulating any gap (baseline performed as expected)
            } else {
                // Active arm selected: compare its reward against baseline
                let baseline_expected = self.env.expected_reward(baseline_arm);
                state.update_phase_gap(baseline_expected, reward);
            }
            if state.should_soft_restart() {
                state.soft_restart();
            }
        }
    }

    /// Run the bandit session for `episodes` episodes.
    ///
    /// Returns `(events, result)`. Events include per-episode stats.
    /// Pseudo-regret = Σ(optimal_expected - chosen_arm_expected).
    pub fn run(mut self, episodes: usize, rng: &mut Rng) -> (Vec<BanditEvent>, BanditResult) {
        let mut events = Vec::with_capacity(episodes + 1);
        let optimal_arm = self.env.optimal_arm();
        let optimal_reward = self.env.optimal_reward();

        for episode in 0..episodes {
            let arm = self.select_arm(rng);
            let reward = self.env.pull(arm, rng);
            let q_before = self.stats.q_value(arm);

            events.push(BanditEvent::Pull {
                episode,
                arm,
                reward,
                q_value: q_before,
            });

            self.stats.update(arm, reward);
            self.cumulative_reward += reward;
            self.cumulative_regret += optimal_reward - self.env.expected_reward(arm);

            self.decay_epsilon();

            // Update safe-phased state (Plan 137)
            #[cfg(feature = "safe_bandit")]
            self.update_safe_phased(arm, reward);

            // Record review metrics (Plan 036)
            if let Some(ref metrics) = self.review_metrics {
                let reviewed_correct = arm == optimal_arm;
                // Simulate base (random) correctness deterministically
                let base_correct = episode % self.env.num_arms() == optimal_arm;
                metrics.record(base_correct, reviewed_correct);
            }

            events.push(BanditEvent::EpisodeComplete {
                episode,
                arm,
                reward,
                cumulative_reward: self.cumulative_reward,
                cumulative_regret: self.cumulative_regret,
            });
        }

        let best_arm = self.stats.best_arm();
        let result = BanditResult {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
            q_values: self.stats.q_values.to_vec(),
            visits: self.stats.visits.to_vec(),
        };

        events.push(BanditEvent::SessionComplete {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
        });

        (events, result)
    }

    /// Run the bandit session with trial log persistence.
    ///
    /// Same as [`run`](Self::run) but appends each episode's record to `trial_log`.
    /// The `config` string is attached to every record for later analysis.
    pub fn run_with_trial_log(
        mut self,
        episodes: usize,
        rng: &mut Rng,
        trial_log: &mut TrialLog,
        config: &str,
    ) -> (Vec<BanditEvent>, BanditResult) {
        let mut events = Vec::with_capacity(episodes + 1);
        let optimal_arm = self.env.optimal_arm();
        let optimal_reward = self.env.optimal_reward();
        let config_owned = config.to_string();

        for episode in 0..episodes {
            let arm = self.select_arm(rng);
            let reward = self.env.pull(arm, rng);
            let q_before = self.stats.q_value(arm);

            events.push(BanditEvent::Pull {
                episode,
                arm,
                reward,
                q_value: q_before,
            });

            self.stats.update(arm, reward);
            self.cumulative_reward += reward;
            self.cumulative_regret += optimal_reward - self.env.expected_reward(arm);

            self.decay_epsilon();

            // Update safe-phased state (Plan 137)
            #[cfg(feature = "safe_bandit")]
            self.update_safe_phased(arm, reward);

            // Record review metrics (Plan 036)
            if let Some(ref metrics) = self.review_metrics {
                let reviewed_correct = arm == optimal_arm;
                // Simulate base (random) correctness deterministically
                let base_correct = episode % self.env.num_arms() == optimal_arm;
                metrics.record(base_correct, reviewed_correct);
            }

            // Persist to trial log
            let record = TrialRecord {
                episode,
                player_id: 0,
                arm,
                reward,
                q_value: self.stats.q_value(arm),
                cumulative_reward: self.cumulative_reward,
                cumulative_regret: self.cumulative_regret,
                config: config_owned.clone(),
                note: String::new(),
                base_correct: None,
                reviewed_correct: None,
                anchors: None,
            };
            if let Err(e) = trial_log.append(&record) {
                eprintln!("trial_log write error at episode {episode}: {e}");
            }

            events.push(BanditEvent::EpisodeComplete {
                episode,
                arm,
                reward,
                cumulative_reward: self.cumulative_reward,
                cumulative_regret: self.cumulative_regret,
            });
        }

        let best_arm = self.stats.best_arm();
        let result = BanditResult {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
            q_values: self.stats.q_values.to_vec(),
            visits: self.stats.visits.to_vec(),
        };

        events.push(BanditEvent::SessionComplete {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
        });

        if let Err(e) = trial_log.flush() {
            eprintln!("trial_log flush error: {e}");
        }
        (events, result)
    }
}
