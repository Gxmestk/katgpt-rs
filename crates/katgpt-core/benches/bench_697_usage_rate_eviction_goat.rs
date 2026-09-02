//! Plan 585 Phase 3 — Usage-Rate (Mass/Age) KV Eviction GOAT bench
//! (Research 523, arXiv:2608.19920 "Learning how to Forget", Seeger et al.,
//! AWS 2026; gate instrument for `katgpt_core::kv_eviction`).
//!
//! # Modelless construction — read this first
//!
//! The "model" under test is a **constructed induction-pair KV cache**, the
//! deterministic abstraction of a trained induction head's cache: row j
//! carries `key = token t_j` and `value = payload t_{j+1}`, so answering a
//! query token X means "attend to the row keyed X and read its payload".
//! NO training, NO learned weights, NO gradients — the POLICY axis is what
//! is under test (which rows a fixed eviction policy keeps under budget),
//! not the model.
//!
//! # Workload (and its honest regime boundary)
//!
//! A Zipf token stream with **popularity drift**: the hot set of the last
//! 40% of the stream is the mirror of the first 40%. Queries are sampled
//! from the drifted distribution restricted to the tokens CURRENTLY LIVE in
//! the cache (attention queries match context content), so every live row
//! carries positive mass and eviction decisions are mass/rate-sensitive —
//! no zero-mass tie domination.
//!
//! Twelve needles are planted through the stream: a needle is a token that
//! is ABSENT from the Zipf vocabulary but recurs at a steady MID frequency
//! (queried every `NEEDLE_QUERY_EVERY` ticks after planting) and carries a
//! unique payload — the recurring-entity/pair analog of long-context
//! serving. The paper's mechanism is then exercised exactly: a recurring
//! token's mass grows ∝ its (constant) query rate × age, so its RATE is
//! constant and competitive (mass/age keeps it), while its cumulative mass
//! is linearly outgrown by the hot tokens' rows — raw-H2O's age bias evicts
//! it once enough higher-volume rows age past it. Final tally reads which
//! needles survived; no admission happens after the last query window.
//!
//! This is the regime where recency-weighted usage (mass/age) is the right
//! statistic and lifetime mass (raw-H2O) provably mis-ranks. On a
//! STATIONARY workload the two scores rank identically by construction —
//! the bench does not claim G8 there, and records that boundary here
//! rather than hiding it.
//!
//! # Gates (T3.4)
//!
//! - **G8 (headline):** mass/age recall ≥ raw-H2O recall at EVERY cap in
//!   the sweep. FAIL here = keep the negative artifact + demote (plan rule).
//! - **G1 determinism:** the full policy × cap × seed matrix run twice —
//!   accuracy tables bit-identical.
//! - **G2 update latency:** `observe` + `score` < 10 ns/row (release).
//! - **Canary demo (T2.2 non-vacuity at the bench level):** an intentionally
//!   over-evicted arm (ring at a crushing cap) must FAIL `runaway_gate`
//!   while the healthy mass/age arm PASSES on identical thresholds.
//!
//! # Sections
//!
//! - T3.1 planted age-bias fixture (must-fire): raw-H2O evicts the hot row,
//!   mass/age retains it — tie arm + strict arm.
//! - T3.2 recall at matched budget: 6 policies × 3 caps × 32 seeds —
//!   accuracy, eviction counts, generation stats.
//! - T3.3 Kendall-τ: per-head vs batch-summed eviction rankings.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features usage_rate_eviction --bench bench_697_usage_rate_eviction_goat --release -- --nocapture
//! ```

#![cfg(feature = "usage_rate_eviction")]

use katgpt_core::kv_eviction::{
    observe, runaway_gate, score, select_evict, select_evict_into, RunawayStats, UsageRow,
    UsageScoreTable,
};
use std::time::Instant;

// ── deterministic RNG (house pattern, bench_313 SimpleRng) ──────────────

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f32 {
        (self.below(10_000) as f32) / 10_000.0
    }
}

// ── shared constants ─────────────────────────────────────────────────────

const N_SEEDS: u64 = 32;
const STREAM_LEN: u64 = 192;
const N_DISTRACT: usize = 16;
const QUERIES_PER_TICK: u64 = 2; // the mass flow that pressures the budget
const NEEDLE_QUERY_EVERY: u64 = 8; // needles recur mid-frequency (see below)
const N_NEEDLES: usize = 12;
const NEEDLE_BASE: usize = 24; // needle token ids [24, 36); distractors [0, 24)
const PAYLOAD_BASE: usize = 36; // payload ids [36, 48)
const GEN_TARGET: usize = 8;
const GEN_CAP: usize = 64;

/// Drifted Zipf over the distractor vocab. `phase` 0.0 = early ranking,
/// 1.0 = fully mirrored ranking; the middle of the stream blends.
fn token_probability(tok: usize, phase: f32) -> f32 {
    let zipf = 1.0 / (tok as f32 + 1.0);
    let zipf_drift = 1.0 / ((N_DISTRACT - 1 - tok) as f32 + 1.0);
    zipf * (1.0 - phase) + zipf_drift * phase
}

fn phase_of(t: u64) -> f32 {
    let a = STREAM_LEN * 4 / 10;
    let b = STREAM_LEN * 6 / 10;
    if t < a {
        0.0
    } else if t > b {
        1.0
    } else {
        (t - a) as f32 / (b - a) as f32
    }
}

/// Sample a distractor token by inverse-CDF over the drifted Zipf.
fn sample_token(rng: &mut SimpleRng, phase: f32) -> usize {
    let pick = rng.unit();
    let total: f32 = (0..N_DISTRACT).map(|j| token_probability(j, phase)).sum();
    let mut cum = 0.0f32;
    for j in 0..N_DISTRACT {
        cum += token_probability(j, phase) / total;
        if pick <= cum {
            return j;
        }
    }
    N_DISTRACT - 1
}

// ── T3.1 planted age-bias fixture ────────────────────────────────────────

/// Returns ((tie_arm_ok, tie_arm_raw_indifferent), strict_arm_ok).
///
/// Tie arm: equal cumulative mass (1.0 each). raw-H2O is INDIFFERENT (the
/// ascending-index tie-break decides — documented, not asserted as "evicts
/// hot"); mass/age STRICTLY evicts the old-cold row. The policy difference
/// the paper names is mass/age's strictness where raw-H2O cannot decide.
///
/// Strict arm: the old row holds MORE total mass (1.1 > 1.0 — the paper's
/// age bias). raw-H2O now STRICTLY evicts the young-hot row while mass/age
/// still evicts the old-cold row: the two policies' keep-sets fully invert.
fn age_bias_fixture() -> ((bool, bool), bool) {
    let tick = 1_002u64;
    // Tie arm.
    let mut old = UsageRow { cum_mass: 0.0, admission_tick: 0 };
    let mut hot = UsageRow { cum_mass: 0.0, admission_tick: 1_000 };
    for step in 0..1000u64 {
        observe(&mut old, 0.001, step);
    }
    for step in 1000..1002u64 {
        observe(&mut hot, 0.5, step);
    }
    debug_assert_eq!(old.cum_mass, 1.0);
    debug_assert_eq!(hot.cum_mass, 1.0);
    let raw_tie = select_evict(&[old.cum_mass, hot.cum_mass], 1, &[]);
    let rate_tie = select_evict(&[score(&old, tick), score(&hot, tick)], 1, &[]);
    // raw tie-indifferent: the tie-break (ascending index) decides — it
    // evicts index 0 here; that is NOT a policy decision.
    let tie_raw_indifferent = raw_tie == vec![0];
    let tie_ok = rate_tie == vec![0]; // mass/age strictly evicts old-cold

    // Strict arm.
    let mut old2 = UsageRow { cum_mass: 0.0, admission_tick: 0 };
    let mut hot2 = UsageRow { cum_mass: 0.0, admission_tick: 1_000 };
    for step in 0..1000u64 {
        observe(&mut old2, 0.0011, step);
    }
    for step in 1000..1002u64 {
        observe(&mut hot2, 0.5, step);
    }
    let raw_strict = select_evict(&[old2.cum_mass, hot2.cum_mass], 1, &[]);
    let rate_strict = select_evict(&[score(&old2, tick), score(&hot2, tick)], 1, &[]);
    let strict_ok = raw_strict == vec![1] // raw evicts the hot row (lower mass)
        && rate_strict == vec![0]; // mass/age evicts the old-cold row

    ((tie_ok, tie_raw_indifferent), strict_ok)
}

// ── T3.2 streaming cache + policies ─────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Policy {
    Ring,
    RawH2o,
    MassAge,
    MassAgeSink,
    EgaEnergy,
    EgaUsage,
}

impl Policy {
    const ALL: [Policy; 6] = [
        Policy::Ring,
        Policy::RawH2o,
        Policy::MassAge,
        Policy::MassAgeSink,
        Policy::EgaEnergy,
        Policy::EgaUsage,
    ];
    fn name(&self) -> &'static str {
        match self {
            Policy::Ring => "ring",
            Policy::RawH2o => "raw_h2o",
            Policy::MassAge => "mass_age",
            Policy::MassAgeSink => "mass_age_sink",
            Policy::EgaEnergy => "ega_energy",
            Policy::EgaUsage => "ega_x_usage",
        }
    }
}

/// One live cache row (the induction pair: key token -> payload token).
#[derive(Clone, Copy)]
struct Row {
    token: usize,
    payload: usize,
    admitted: u64,
    alive: bool,
}

struct SimState {
    policy: Policy,
    rows: Vec<Row>,
    table: UsageScoreTable,
    tick: u64,
    cap: usize,
    evictions: u64,
}

impl SimState {
    fn new(policy: Policy, cap: usize) -> Self {
        Self {
            policy,
            rows: Vec::with_capacity(STREAM_LEN as usize),
            table: UsageScoreTable::with_capacity(STREAM_LEN as usize),
            tick: 0,
            cap,
            evictions: 0,
        }
    }

    /// Admit the just-pushed row (caller pushes, then calls this). Does
    /// NOT evict — trimming happens AFTER the tick's attention (real
    /// serving decides evictions on post-attention state; trimming on a
    /// zero-mass newborn makes the newborn the guaranteed victim of every
    /// mass-based policy, which is the degeneracy this structure refuses).
    fn admit_no_evict(&mut self) {
        let idx = self.rows.len() - 1;
        self.rows[idx].admitted = self.tick;
        self.rows[idx].alive = true;
        self.table.reset_row(idx, self.tick);
    }

    /// Trim down to cap under the policy. Call AFTER the tick's queries.
    fn evict_if_over(&mut self) {
        let live = self.rows.iter().filter(|r| r.alive).count();
        if live > self.cap {
            self.evict();
        }
    }

    fn policy_scores(&self) -> Vec<f32> {
        let live_idx: Vec<usize> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.alive)
            .map(|(i, _)| i)
            .collect();
        let mut scores = Vec::with_capacity(live_idx.len());
        match self.policy {
            // evict oldest = lowest admission tick: score = -tick so the
            // lowest score is the oldest row (select_evict returns lowest-k).
            Policy::Ring => {
                for &i in &live_idx {
                    scores.push(-(self.rows[i].admitted as f32));
                }
            }
            Policy::RawH2o => {
                for &i in &live_idx {
                    scores.push(self.table.row(i).cum_mass);
                }
            }
            Policy::MassAge | Policy::MassAgeSink | Policy::EgaUsage => {
                // per LIVE row (the table's live prefix includes dead slots)
                for &i in &live_idx {
                    scores.push(score(self.table.row(i), self.tick));
                }
            }
            Policy::EgaEnergy => {
                for &i in &live_idx {
                    scores.push(ega_energy(self.rows[i].token));
                }
            }
        }
        if self.policy == Policy::EgaUsage {
            // fusion: admission prior (static z-scored energy) x online
            // correction (mass/age rate), both z-scored, equal weight —
            // the Research 523 fusion shape.
            let rates = scores.clone();
            let energies: Vec<f32> = live_idx
                .iter()
                .map(|&i| ega_energy(self.rows[i].token))
                .collect();
            let zr = z_score(&rates);
            let ze = z_score(&energies);
            scores.clear();
            for k in 0..zr.len() {
                scores.push(0.5 * zr[k] + 0.5 * ze[k]);
            }
        }
        scores
    }

    fn evict(&mut self) {
        let live_idx: Vec<usize> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.alive)
            .map(|(i, _)| i)
            .collect();
        let over = live_idx.len().saturating_sub(self.cap);
        if over == 0 {
            return;
        }
        let scores = self.policy_scores();
        // sink pin: slot 0 (the first-admitted row — the sink position
        // convention) is pinned for the sink arm; other arms pin nothing.
        let pin_mask: Vec<bool> = match self.policy {
            Policy::MassAgeSink => live_idx.iter().map(|&i| i == 0).collect(),
            _ => vec![false; live_idx.len()],
        };
        let mut victims = Vec::new();
        select_evict_into(&scores, over, &pin_mask, &mut victims);
        for &loc in &victims {
            self.rows[live_idx[loc]].alive = false;
        }
        self.evictions += victims.len() as u64;
    }

    /// Sample a query token: drifted-Zipf-weighted over the tokens
    /// CURRENTLY LIVE in the cache (attention matches context content;
    /// every live token is reachable, so no zero-mass tie domination).
    fn sample_live_query(&self, rng: &mut SimpleRng, phase: f32) -> Option<usize> {
        let mut live_tokens: Vec<(usize, f32)> = Vec::new();
        let mut total = 0.0f32;
        for r in self.rows.iter().filter(|r| r.alive) {
            if r.token < NEEDLE_BASE
                && !live_tokens.iter().any(|(t, _)| *t == r.token)
            {
                let w = token_probability(r.token, phase);
                live_tokens.push((r.token, w));
                total += w;
            }
        }
        if live_tokens.is_empty() || total <= 0.0 {
            return None;
        }
        let pick = rng.unit() * total;
        let mut cum = 0.0f32;
        for (t, w) in &live_tokens {
            cum += w;
            if pick <= cum {
                return Some(*t);
            }
        }
        live_tokens.last().map(|(t, _)| *t)
    }

    /// Query a token: attend ALL live rows keyed by it, splitting the mass
    /// evenly (the softmax-with-equal-scores abstraction). Returns hits.
    fn query(&mut self, token: usize) -> Vec<usize> {
        let hits: Vec<usize> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.alive && r.token == token)
            .map(|(i, _)| i)
            .collect();
        if !hits.is_empty() {
            let w = 1.0 / hits.len() as f32;
            for &h in &hits {
                observe(self.table.row_mut(h), w, self.tick);
            }
        }
        hits
    }
}

/// EGA static key energy — the deterministic construction of riir-engine's
/// `dot(key, w_proj)` with a FIXED pseudo-random projection: energy is a
/// fixed permutation of token ids, so it is UNCORRELATED with needle-ness
/// (in riir-engine the projection is learned; a monotone-in-id construction
/// would make EGA a needle detector — an artifact this bench refuses).
/// Query-agnostic by construction: it cannot see usage at all.
fn ega_energy(token: usize) -> f32 {
    0.1 * (((token * 7 + 5) % 36) + 1) as f32
}

fn z_score(xs: &[f32]) -> Vec<f32> {
    let n = xs.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = xs.iter().sum::<f32>() / n as f32;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let std = var.sqrt().max(1e-9);
    xs.iter().map(|x| (x - mean) / std).collect()
}

struct TrialOutcome {
    recalled: usize,
    evictions: u64,
    output_len: usize,
    target_len: usize,
}

/// One recall trial.
///
/// Timeline per tick t in 0..STREAM_LEN:
/// 1. needle plant at its planned tick (admission),
/// 2. needle recurrence queries every NEEDLE_QUERY_EVERY ticks after the
///    plant (the mid-frequency mass signal),
/// 3. a distractor admission, its self-attention, and sampled live-token
///    queries from the drifted Zipf (the policy-differentiating mass),
///    then the trim.
///
/// After the stream: the final tally reads which needle rows survived — NO
/// mass is granted by the tally.
fn run_trial(policy: Policy, cap: usize, seed: u64) -> TrialOutcome {
    let mut rng = SimpleRng::new(seed);
    let mut st = SimState::new(policy, cap);

    // needles: token 24+k planted at (8 + 7k)% of the stream.
    let plant_tick: Vec<u64> = (0..N_NEEDLES)
        .map(|k| STREAM_LEN * (8 + 7 * k) as u64 / 100)
        .collect();
    let payloads: Vec<usize> = (0..N_NEEDLES)
        .map(|k| PAYLOAD_BASE + ((seed + k as u64) % 12) as usize)
        .collect();

    for t in 0..STREAM_LEN {
        st.tick = t;
        // 1. needle plant? (a plant IS an appearance: the token occurred
        // and was attended — grant its occurrence query immediately.)
        if let Some(k) = plant_tick.iter().position(|&pt| pt == t) {
            st.rows.push(Row {
                token: NEEDLE_BASE + k,
                payload: payloads[k],
                admitted: t,
                alive: true,
            });
            st.admit_no_evict();
            st.query(NEEDLE_BASE + k);
            st.evict_if_over();
            continue;
        }
        // 2. needle recurrence: every NEEDLE_QUERY_EVERY ticks after its
        // plant, a needle token is queried (the mid-frequency signal).
        for (k, &pt) in plant_tick.iter().enumerate() {
            if t > pt && (t - pt) % NEEDLE_QUERY_EVERY == 0 {
                st.query(NEEDLE_BASE + k);
            }
        }
        // 3. distractor admission + attention + trim. Order is load-
        // bearing: the newborn attends (self + sampled) BEFORE the policy
        // trims, so eviction decisions never see a zero-mass newborn.
        let phase = phase_of(t);
        let tok = sample_token(&mut rng, phase);
        st.rows.push(Row {
            token: tok,
            payload: PAYLOAD_BASE + rng.below(12) as usize,
            admitted: t,
            alive: true,
        });
        st.admit_no_evict();
        // self-attention: the incoming token attends the context NOW (the
        // causal-attention abstraction).
        st.query(tok);
        for _ in 0..QUERIES_PER_TICK {
            if let Some(qtok) = st.sample_live_query(&mut rng, phase) {
                st.query(qtok);
            }
        }
        st.evict_if_over();
    }

    // Final tally: which needles survived to be answerable?
    st.tick = STREAM_LEN + 1;
    let mut recalled = 0usize;
    for k in 0..N_NEEDLES {
        let hits: Vec<usize> = st
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.alive && r.token == NEEDLE_BASE + k)
            .map(|(i, _)| i)
            .collect();
        let ok = hits
            .iter()
            .any(|&h| st.rows[h].payload == payloads[k] && st.rows[h].admitted == plant_tick[k]);
        if ok {
            recalled += 1;
        }
    }

    // Generation phase (the canary demo instrument): continue the chain
    // from the LAST needle. A policy that lost the chain repeats in place
    // and runs to GEN_CAP; an intact chain stops at GEN_TARGET.
    let mut out_len = 0usize;
    let mut cur_tok = NEEDLE_BASE + N_NEEDLES - 1;
    for g in 0..GEN_CAP {
        st.tick = STREAM_LEN + 100 + g as u64;
        let hits = st.query(cur_tok);
        let next = if hits.is_empty() {
            cur_tok // lost the chain: the runaway signature (repeat in place)
        } else {
            st.rows[hits[0]].payload
        };
        out_len += 1;
        if g + 1 >= GEN_TARGET && next != cur_tok {
            break; // chain intact through the target: natural stop
        }
        cur_tok = next.min(N_DISTRACT - 1);
    }

    TrialOutcome {
        recalled,
        evictions: st.evictions,
        output_len: out_len,
        target_len: GEN_TARGET,
    }
}

// ── T3.3 Kendall tau ────────────────────────────────────────────────────

fn kendall_tau(a: &[usize], b: &[usize]) -> f32 {
    let rank = |v: &[usize]| -> std::collections::HashMap<usize, usize> {
        v.iter().enumerate().map(|(i, &x)| (x, i)).collect()
    };
    let ra = rank(a);
    let rb = rank(b);
    let common: Vec<usize> = ra.keys().copied().filter(|x| rb.contains_key(x)).collect();
    let n = common.len();
    if n < 2 {
        return f32::NAN;
    }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let (dai, dbi) = (ra[&common[i]], rb[&common[i]]);
            let (daj, dbj) = (ra[&common[j]], rb[&common[j]]);
            match (dai.cmp(&daj), dbi.cmp(&dbj)) {
                (std::cmp::Ordering::Equal, _) | (_, std::cmp::Ordering::Equal) => {}
                (o1, o2) if o1 == o2 => concordant += 1,
                _ => discordant += 1,
            }
        }
    }
    (concordant - discordant) as f32 / (concordant + discordant) as f32
}

// ── main ────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 585 / Bench 697: usage-rate (mass/age) KV eviction GOAT ===");
    println!(
        "modelless constructed induction-pair KV; drifted-Zipf workload (see header doc)\n"
    );

    // ── T3.1 age-bias fixture ──
    let ((tie_ok, tie_raw_indifferent), strict_ok) = age_bias_fixture();
    println!("T3.1 age-bias fixture:");
    println!(
        "  tie arm    (mass 1.0/1.0): mass/age strictly evicts old-cold={} ; raw-H2O tie-indifferent (index tie-break)={}",
        tie_ok, tie_raw_indifferent
    );
    println!(
        "  strict arm (old mass 1.1 > 1.0): raw evicts hot + mass/age evicts old-cold={}",
        strict_ok
    );
    let t31 = tie_ok && tie_raw_indifferent && strict_ok;
    println!("  T3.1 GATE: {}", pass_fail(t31));

    // ── T3.2 policy matrix (run TWICE for G1) ──
    let caps = [16usize, 32, 48, 64];
    let mut run_tables: Vec<Vec<Vec<u32>>> = Vec::new(); // [run][policy][cap]
    let mut run_evictions: Vec<Vec<Vec<u64>>> = Vec::new();
    let mut run_outlens: Vec<Vec<Vec<usize>>> = Vec::new();
    for _run in 0..2 {
        let mut acc = vec![vec![0u32; caps.len()]; Policy::ALL.len()];
        let mut ev = vec![vec![0u64; caps.len()]; Policy::ALL.len()];
        let mut ol = vec![vec![0usize; caps.len()]; Policy::ALL.len()];
        for (pi, &policy) in Policy::ALL.iter().enumerate() {
            for (ci, &cap) in caps.iter().enumerate() {
                let mut rec = 0u32;
                let mut evc = 0u64;
                let mut outsum = 0usize;
                for seed in 0..N_SEEDS {
                    let o = run_trial(policy, cap, 1_000 + seed);
                    rec += o.recalled as u32;
                    evc += o.evictions;
                    outsum += o.output_len;
                }
                acc[pi][ci] = rec;
                ev[pi][ci] = evc;
                ol[pi][ci] = outsum;
            }
        }
        run_tables.push(acc);
        run_evictions.push(ev);
        run_outlens.push(ol);
    }

    let total_per_cell = N_SEEDS * N_NEEDLES as u64;
    println!(
        "\nT3.2 recall at matched budget ({} seeds x {} needles, stream={}):",
        N_SEEDS, N_NEEDLES, STREAM_LEN
    );
    println!(
        "{:<16} {:>18} {:>18} {:>18} {:>18} {:>10}",
        "policy", "cap=16", "cap=32", "cap=48", "cap=64", "out@16"
    );
    for (pi, &policy) in Policy::ALL.iter().enumerate() {
        let row = &run_tables[0][pi];
        let cells: Vec<String> = row
            .iter()
            .map(|&r| {
                let pct = 100.0 * r as f64 / total_per_cell as f64;
                format!("{r:>4}/{tpc} {pct:4.1}%", tpc = total_per_cell)
            })
            .collect();
        let mean_out = run_outlens[0][pi][0] as f64 / N_SEEDS as f64;
        println!(
            "{:<16} {:>18} {:>18} {:>18} {:>18} {:>10.2}",
            policy.name(),
            cells[0],
            cells[1],
            cells[2],
            cells[3],
            mean_out
        );
    }
    // G8: the mass/age family must be >= raw_h2o at every cap. A per-cap
    // miss is recorded honestly — the regime boundary (WHERE the score
    // wins) is a finding, not noise to be tuned away.
    let mut g8_all = true;
    let mut g8_misses: Vec<String> = Vec::new();
    for ci in 0..caps.len() {
        let raw = run_tables[0][1][ci];
        for &pi in &[2usize, 3, 5] {
            if run_tables[0][pi][ci] < raw {
                g8_all = false;
                g8_misses.push(format!(
                    "{} ({}) < raw_h2o ({}) at cap {}",
                    Policy::ALL[pi].name(),
                    run_tables[0][pi][ci],
                    raw,
                    caps[ci]
                ));
            }
        }
    }
    for m in &g8_misses {
        println!("  G8 MISS: {m}");
    }
    println!(
        "  G8 GATE (mass/age family >= raw_h2o at every cap): {}",
        pass_fail(g8_all)
    );

    println!("\n  evictions@cap=16 per policy (victim CHOICE differs, count is pressure-set):");
    for (pi, &policy) in Policy::ALL.iter().enumerate() {
        println!("    {:<16} {}", policy.name(), run_evictions[0][pi][0]);
    }

    // ── G1 determinism ──
    let g1 = run_tables[0] == run_tables[1]
        && run_evictions[0] == run_evictions[1]
        && run_outlens[0] == run_outlens[1];
    println!("\n  G1 GATE (matrix double-run bit-identical): {}", pass_fail(g1));

    // ── canary demo (T2.2 non-vacuity at bench level) ──
    println!("\nCanary demo (RunawayStats + runaway_gate, r_max=1.5, p_cap_max=0.05):");
    let mut canary_ok = false;
    for &policy in Policy::ALL.iter() {
        // crushing cap for the over-eviction arm; cap=32 for the rest
        let cap = if policy == Policy::Ring { 8 } else { 32 };
        let mut outs = Vec::new();
        let mut tgts = Vec::new();
        for seed in 0..N_SEEDS {
            let o = run_trial(policy, cap, 7_000 + seed);
            outs.push(o.output_len);
            tgts.push(o.target_len);
        }
        let stats = RunawayStats::from_generations(&outs, &tgts, GEN_CAP);
        let gate = runaway_gate(&stats, 1.5, 0.05);
        println!(
            "  {:<14} cap={:<3} R_median={:<7.3} p_cap={:<5.2} n={} gate={}",
            policy.name(),
            cap,
            stats.r_median,
            stats.p_cap,
            stats.n,
            if gate { "PASS" } else { "FAIL" }
        );
        if policy == Policy::Ring {
            canary_ok = !gate; // over-eviction arm must FAIL
        }
        if policy == Policy::MassAge {
            canary_ok = canary_ok && gate; // healthy arm must PASS
        }
    }
    println!(
        "  canary non-vacuity (ring@8 FAILS, mass_age@32 PASSES): {}",
        pass_fail(canary_ok)
    );

    // ── T3.3 Kendall tau: per-head vs batch-summed ──
    println!("\nT3.3 Kendall-tau (per-head vs batch-summed keep-rankings, cap=32):");
    run_tau_section();

    // ── G2 update latency ──
    let ns_per_row = g2_update_latency();
    println!(
        "\n  G2 update latency: {:.2} ns/row (budget 10 ns): {}",
        ns_per_row,
        pass_fail(ns_per_row < 10.0)
    );

    // ── verdict ──
    let all = t31 && g8_all && g1 && canary_ok && ns_per_row < 10.0;
    if all {
        println!("\n=== VERDICT: ALL GATES PASS ===");
    } else {
        println!("\n=== VERDICT: MIXED — {} regime miss(es); boundary recorded honestly (negative artifact + opt-in per plan rule) ===", g8_misses.len());
    }
}

fn pass_fail(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}

fn g2_update_latency() -> f64 {
    let n = 10_000usize;
    let ticks = 1_000u64;
    let mut t = UsageScoreTable::with_capacity(n);
    for i in 0..n {
        t.reset_row(i, 0);
    }
    // warm
    for step in 0..100u64 {
        for i in 0..n {
            observe(t.row_mut(i), 1.0, step);
        }
    }
    let mut acc = 0.0f32;
    let start = Instant::now();
    for step in 0..ticks {
        for i in 0..n {
            observe(t.row_mut(i), 1.0, step);
        }
        let mut s = Vec::with_capacity(n);
        t.scores(step, &mut s);
        acc += s[0];
    }
    let elapsed = start.elapsed().as_secs_f64();
    let _ = acc;
    elapsed * 1e9 / (n as f64 * ticks as f64)
}

fn run_tau_section() {
    // 4 heads, distinct seeded streams; keep-ranking by per-head cum_mass
    // vs the batch-summed ranking (the coarsening H2O performs — each head
    // here shares the row-index space by construction; documented
    // simplification).
    let build = |h: u64| -> (Vec<usize>, Vec<f32>) {
        let mut st = SimState::new(Policy::RawH2o, usize::MAX); // no eviction
        let mut rng = SimpleRng::new(50 + h);
        for t in 0..120u64 {
            st.tick = t;
            let tok = rng.below(N_DISTRACT as u64) as usize;
            st.rows.push(Row {
                token: tok,
                payload: PAYLOAD_BASE + rng.below(12) as usize,
                admitted: t,
                alive: true,
            });
            st.admit_no_evict();
            st.query(tok);
            let qtok = rng.below(N_DISTRACT as u64) as usize;
            st.query(qtok);
            st.evict_if_over();
        }
        let mut ranked: Vec<(usize, f32)> = st
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.alive)
            .map(|(i, _)| (i, st.table.row(i).cum_mass))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        (ranked.iter().map(|(i, _)| *i).collect(), ranked.iter().map(|(_, m)| *m).collect())
    };
    let mut head_rankings: Vec<Vec<usize>> = Vec::new();
    let mut summed: Vec<f32> = Vec::new();
    for h in 0..4u64 {
        let (idx, masses) = build(h);
        head_rankings.push(idx);
        if summed.is_empty() {
            summed = masses;
        } else {
            for (s, m) in summed.iter_mut().zip(masses.iter()) {
                *s += m;
            }
        }
    }
    let mut summed_ranking: Vec<(usize, f32)> = summed.iter().copied().enumerate().collect();
    summed_ranking.sort_by(|a, b| b.1.total_cmp(&a.1));
    let summed_idx: Vec<usize> = summed_ranking.iter().map(|(i, _)| *i).collect();

    for (h, hr) in head_rankings.iter().enumerate() {
        let tau = kendall_tau(hr, &summed_idx);
        println!("  head{} vs summed: tau = {:.3}", h, tau);
    }
}
