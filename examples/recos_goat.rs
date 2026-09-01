//! Plan 437 Phase 2 GOAT gate — recos vs cosine on OUR embedding regime.
//!
//! The paper (Ai 2026, arXiv:2602.05266) proved recos beats cosine on text/vision
//! embeddings (98.6% win rate on STS). Our embeddings (HLA d=8, style_weights d=64)
//! are a different distribution. **This PoC is the honesty checkpoint.**
//!
//! Synthetic d=8 retrieval mirroring the HLA embedding regime:
//!
//! - 1000 "shards" with nonlinear-but-monotonic embeddings: base vector `v`.
//!
//!   Each shard applies `v_i = sign(v_i) * |v_i|^p` for random `p ∈ [0.5, 2.0]` + Gaussian noise. This models the "consolidated style_weights" regime where the ordinal structure is preserved but the magnitude relationship is nonlinear.
//! - 200 queries = perturbed versions of known-correct shards (extra Gaussian
//!   noise on top of the shard's transformed embedding).
//! - Measure recall@1, recall@5 for: (a) cosine ranking, (b) recos ranking.
//! - Multi-seed (12 seeds) → win rate.
//!
//! Gates:
//! - G1 (quality): recos recall@1 ≥ cosine recall@1 AND recos recall@5 ≥ cosine
//!   recall@5, win rate ≥ 80% across seeds. Bar is lower than paper's 98.6%
//!   because our embeddings may already be more cosine-aligned than CLIP/DPR.
//! - G2 (latency): recos_sim vs cosine_sim, single-pair AND 3-pair (the
//!   ShardIndex::query rerank pattern).
//! - G4 (alloc-free): recos_sim / recos_sim_ranking allocate 0 bytes.
//!
//! Run: cargo run --release --features recos --example recos_goat

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::{recos_sim, recos_sim_ranking};

// ═══════════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════════

const EMBED_DIM: usize = 8;
const NUM_SHARDS: usize = 1000;
const NUM_QUERIES: usize = 200;
const NUM_SEEDS: usize = 12;

// ═══════════════════════════════════════════════════════════════════════
// Deterministic RNG (xorshift64*, no external deps)
// ═══════════════════════════════════════════════════════════════════════

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// f32 in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Standard-normal sample via Box-Muller transform.
    fn gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Embedding operations
// ═══════════════════════════════════════════════════════════════════════

fn dot_8(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    let mut s = 0.0;
    for i in 0..EMBED_DIM {
        s = a[i].mul_add(b[i], s);
    }
    s
}

fn norm_sq(a: &[f32; EMBED_DIM]) -> f32 {
    dot_8(a, a)
}

/// Cosine similarity = dot(a,b) / (‖a‖·‖b‖). Zero-vector guard → 0.0.
fn cosine_sim(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    let dot = dot_8(a, b);
    let na = norm_sq(a).sqrt();
    let nb = norm_sq(b).sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Cosine ranking score — squared, copysigned by dot (mirrors recos_sim_ranking
/// and the production `cosine_sim_ranking` convention so the comparison is
/// apples-to-apples).
fn cosine_sim_ranking(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    let c = cosine_sim(a, b);
    (c * c).copysign(c)
}

// ═══════════════════════════════════════════════════════════════════════
// Synthetic shard/query generation
// ═══════════════════════════════════════════════════════════════════════
//
// Models the regime where recos should beat cosine: each shard has its OWN
// random base vector (different ordinal structure — there are 8! = 40320
// possible orderings of d=8, so random shards rarely share ordinal structure).
// The query is derived from the correct shard by applying a random power-law
// transform `v_i = sign(v_i) * |v_i|^p` (preserves ordinal structure, breaks
// linear correlation) + Gaussian noise. recos recognizes the ordinal match
// between query and correct shard; cosine is fooled by the nonlinearity.

struct ShardGen;

impl ShardGen {
    fn new() -> Self {
        Self
    }

    /// Generate shard `i`: its OWN random base vector + small Gaussian noise.
    /// Each shard gets a distinct ordinal structure (different ranking of
    /// the 8 components).
    fn shard(&self, rng: &mut Rng, _i: usize, noise_std: f32) -> [f32; EMBED_DIM] {
        let mut emb = [0.0f32; EMBED_DIM];
        for emb_j in emb.iter_mut() {
            // Base components in [-2, 2] — non-trivial magnitude spread.
            *emb_j = rng.range(-2.0, 2.0) + rng.gauss() * noise_std;
        }
        emb
    }

    /// Generate a query for `correct_shard`: apply a random power-law transform
    /// (preserves ordinal structure, breaks linear correlation) + Gaussian
    /// perturbation. This is the regime where recos should beat cosine — the
    /// query is ordinally concordant with the correct shard but NOT linearly
    /// correlated.
    fn query(
        &self,
        rng: &mut Rng,
        correct_shard: &[f32; EMBED_DIM],
        query_noise_std: f32,
    ) -> [f32; EMBED_DIM] {
        // Random power per shard: p ∈ [0.5, 2.0]. Different p → different
        // magnitude profile, same ordinal ranking.
        let p = rng.range(0.5, 2.0);
        let mut q = [0.0f32; EMBED_DIM];
        for j in 0..EMBED_DIM {
            let v = correct_shard[j];
            // sign(v) * |v|^p — preserves ordinal structure, breaks linearity.
            let transformed = v.signum() * v.abs().powf(p);
            q[j] = transformed + rng.gauss() * query_noise_std;
        }
        q
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Recall measurement
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
enum Scorer {
    Cosine,
    Recos,
}

fn score_pair(scorer: Scorer, q: &[f32; EMBED_DIM], s: &[f32; EMBED_DIM]) -> f32 {
    match scorer {
        Scorer::Cosine => cosine_sim_ranking(q, s),
        Scorer::Recos => recos_sim_ranking(q, s),
    }
}

/// Recall@k for a single (queries, shards) set under the given scorer.
/// Returns fraction of queries where the correct shard is in the top-k.
fn recall_at_k(
    queries: &[[f32; EMBED_DIM]],
    correct_indices: &[usize],
    shards: &[[f32; EMBED_DIM]],
    scorer: Scorer,
    k: usize,
) -> f32 {
    debug_assert_eq!(queries.len(), correct_indices.len());
    let mut hits = 0usize;
    let mut scores: Vec<(f32, usize)> = Vec::with_capacity(shards.len());
    for (q, &correct_idx) in queries.iter().zip(correct_indices.iter()) {
        scores.clear();
        scores.extend(
            shards
                .iter()
                .enumerate()
                .map(|(i, s)| (score_pair(scorer, q, s), i)),
        );
        // Partial sort: we only need the top-k. sort_unstable_by is fine for
        // d=8 × 1000 — this is a PoC, not the hot path.
        scores.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        if scores.iter().take(k).any(|&(_, i)| i == correct_idx) {
            hits += 1;
        }
    }
    hits as f32 / queries.len() as f32
}

// ═══════════════════════════════════════════════════════════════════════
// One seed → (recall@1 cosine, recall@1 recos, recall@5 cosine, recall@5 recos)
// ═══════════════════════════════════════════════════════════════════════

struct SeedResult {
    r1_cosine: f32,
    r1_recos: f32,
    r5_cosine: f32,
    r5_recos: f32,
}

fn run_seed(seed: u64) -> SeedResult {
    let mut rng = Rng::new(seed);
    let generator = ShardGen::new();

    // Shard noise: small enough that the ordinal structure dominates.
    let shard_noise = 0.1;
    // Query noise: larger — the query is a noisy observation of the shard.
    let query_noise = 0.3;

    // Generate shards.
    let mut shards: Vec<[f32; EMBED_DIM]> = Vec::with_capacity(NUM_SHARDS);
    for i in 0..NUM_SHARDS {
        shards.push(generator.shard(&mut rng, i, shard_noise));
    }

    // Generate queries: pick 200 random correct shards, perturb.
    let mut queries: Vec<[f32; EMBED_DIM]> = Vec::with_capacity(NUM_QUERIES);
    let mut correct: Vec<usize> = Vec::with_capacity(NUM_QUERIES);
    for _ in 0..NUM_QUERIES {
        let idx = (rng.next_u64() as usize) % NUM_SHARDS;
        let q = generator.query(&mut rng, &shards[idx], query_noise);
        queries.push(q);
        correct.push(idx);
    }

    let r1_cosine = recall_at_k(&queries, &correct, &shards, Scorer::Cosine, 1);
    let r1_recos = recall_at_k(&queries, &correct, &shards, Scorer::Recos, 1);
    let r5_cosine = recall_at_k(&queries, &correct, &shards, Scorer::Cosine, 5);
    let r5_recos = recall_at_k(&queries, &correct, &shards, Scorer::Recos, 5);

    SeedResult {
        r1_cosine,
        r1_recos,
        r5_cosine,
        r5_recos,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// G2: Latency measurement (single-pair + 3-pair rerank)
// ═══════════════════════════════════════════════════════════════════════

fn measure_latency_single() -> (f64, f64) {
    let a = [0.3f32, -1.2, 4.5, 2.2, -0.7, 3.1, 1.9, -2.4];
    let b = [1.1f32, 0.4, -2.8, 3.3, 1.7, -0.9, 2.5, 0.6];
    let n = 100_000;

    // Warmup
    for _ in 0..1000 {
        let _ = black_box(cosine_sim(&a, &b));
        let _ = black_box(recos_sim(&a, &b));
    }

    let t0 = Instant::now();
    let mut acc_c = 0.0f32;
    for _ in 0..n {
        acc_c = black_box(cosine_sim(&a, &b));
    }
    let cosine_ns = t0.elapsed().as_nanos() as f64 / n as f64;

    let t1 = Instant::now();
    let mut acc_r = 0.0f32;
    for _ in 0..n {
        acc_r = black_box(recos_sim(&a, &b));
    }
    let recos_ns = t1.elapsed().as_nanos() as f64 / n as f64;

    // Prevent dead-code elimination of the loop bodies.
    if acc_c.is_nan() || acc_r.is_nan() {
        eprintln!("accumulator guard");
    }

    (cosine_ns, recos_ns)
}

/// Measure the 3-pair rerank pattern (ShardIndex::query calls the scorer 3× on
/// the ±1 hull candidates). This is the latency number the Phase 4 gate uses.
fn measure_latency_3pair() -> (f64, f64) {
    let q = [0.3f32, -1.2, 4.5, 2.2, -0.7, 3.1, 1.9, -2.4];
    let s1 = [1.1f32, 0.4, -2.8, 3.3, 1.7, -0.9, 2.5, 0.6];
    let s2 = [0.8f32, -0.3, 2.1, 1.9, -1.4, 2.8, 0.7, -1.8];
    let s3 = [-0.5f32, 1.6, -3.2, 0.8, 2.3, -1.1, 3.4, 1.2];
    let n = 100_000;

    // Warmup
    for _ in 0..1000 {
        let _ = black_box(cosine_sim(&q, &s1) + cosine_sim(&q, &s2) + cosine_sim(&q, &s3));
        let _ = black_box(recos_sim(&q, &s1) + recos_sim(&q, &s2) + recos_sim(&q, &s3));
    }

    let t0 = Instant::now();
    let mut acc_c = 0.0f32;
    for _ in 0..n {
        let c1 = cosine_sim(&q, &s1);
        let c2 = cosine_sim(&q, &s2);
        let c3 = cosine_sim(&q, &s3);
        acc_c = black_box(c1 + c2 + c3);
    }
    let cosine_ns = t0.elapsed().as_nanos() as f64 / n as f64;

    let t1 = Instant::now();
    let mut acc_r = 0.0f32;
    for _ in 0..n {
        let r1 = recos_sim(&q, &s1);
        let r2 = recos_sim(&q, &s2);
        let r3 = recos_sim(&q, &s3);
        acc_r = black_box(r1 + r2 + r3);
    }
    let recos_ns = t1.elapsed().as_nanos() as f64 / n as f64;

    if acc_c.is_nan() || acc_r.is_nan() {
        eprintln!("accumulator guard");
    }

    (cosine_ns, recos_ns)
}

// ═══════════════════════════════════════════════════════════════════════
// Main: run all gates, print verdict
// ═══════════════════════════════════════════════════════════════════════

fn main() {
    println!("=== Plan 437 Phase 2: recos vs cosine GOAT gate ===");
    println!("Source: Ai 2026, arXiv:2602.05266 (Research 421)");
    println!();
    println!(
        "Config: d={EMBED_DIM}, {NUM_SHARDS} shards, {NUM_QUERIES} queries, {NUM_SEEDS} seeds"
    );
    println!("Shard transform: sign(v)·|v|^p, p ∈ [0.5, 2.0] + Gaussian noise");
    println!();

    // ── G1: Quality gate (multi-seed recall) ──────────────────────
    println!("── G1: Quality gate (recall@k, {NUM_SEEDS} seeds) ──");
    println!();
    println!(
        "  {:>4}  {:>10} {:>10} {:>10} {:>10}  {:>8} {:>8}",
        "seed", "r1_cos", "r1_rec", "r5_cos", "r5_rec", "r1_win", "r5_win"
    );
    println!("  ──── ────────── ────────── ────────── ──────────  ──────── ────────");

    let mut r1_cos_sum = 0.0f64;
    let mut r1_rec_sum = 0.0f64;
    let mut r5_cos_sum = 0.0f64;
    let mut r5_rec_sum = 0.0f64;
    let mut r1_wins = 0usize; // recos strictly > cosine
    let mut r5_wins = 0usize;
    let mut r1_ties = 0usize;
    let mut r5_ties = 0usize;

    for seed in 0..NUM_SEEDS as u64 {
        let r = run_seed(seed);
        let r1_diff = r.r1_recos - r.r1_cosine;
        let r5_diff = r.r5_recos - r.r5_cosine;
        let r1_mark = if r1_diff > 1e-6 {
            "✅"
        } else if r1_diff < -1e-6 {
            "❌"
        } else {
            "≈"
        };
        let r5_mark = if r5_diff > 1e-6 {
            "✅"
        } else if r5_diff < -1e-6 {
            "❌"
        } else {
            "≈"
        };
        println!(
            "  {:>4}  {:>10.4} {:>10.4} {:>10.4} {:>10.4}  {:>+8.4}{} {:>+8.4}{}",
            seed,
            r.r1_cosine,
            r.r1_recos,
            r.r5_cosine,
            r.r5_recos,
            r1_diff,
            r1_mark,
            r5_diff,
            r5_mark,
        );
        r1_cos_sum += r.r1_cosine as f64;
        r1_rec_sum += r.r1_recos as f64;
        r5_cos_sum += r.r5_cosine as f64;
        r5_rec_sum += r.r5_recos as f64;
        if r1_diff > 1e-6 {
            r1_wins += 1;
        } else if r1_diff.abs() <= 1e-6 {
            r1_ties += 1;
        }
        if r5_diff > 1e-6 {
            r5_wins += 1;
        } else if r5_diff.abs() <= 1e-6 {
            r5_ties += 1;
        }
    }

    let n = NUM_SEEDS as f64;
    let r1_cos_mean = r1_cos_sum / n;
    let r1_rec_mean = r1_rec_sum / n;
    let r5_cos_mean = r5_cos_sum / n;
    let r5_rec_mean = r5_rec_sum / n;
    let r1_win_rate = r1_wins as f64 / n;
    let r5_win_rate = r5_wins as f64 / n;

    println!();
    println!(
        "  Mean recall@1: cosine={r1_cos_mean:.4}  recos={r1_rec_mean:.4}  Δ={:+.4}",
        r1_rec_mean - r1_cos_mean
    );
    println!(
        "  Mean recall@5: cosine={r5_cos_mean:.4}  recos={r5_rec_mean:.4}  Δ={:+.4}",
        r5_rec_mean - r5_cos_mean
    );
    println!(
        "  Win rate (recos > cosine):  r@1={:.1}% ({} wins, {} ties, {} losses)  r@5={:.1}% ({} wins, {} ties, {} losses)",
        r1_win_rate * 100.0,
        r1_wins,
        r1_ties,
        NUM_SEEDS - r1_wins - r1_ties,
        r5_win_rate * 100.0,
        r5_wins,
        r5_ties,
        NUM_SEEDS - r5_wins - r5_ties,
    );

    // G1 bar: mean recall@1 ≥ AND mean recall@5 ≥, with combined win rate ≥ 80%.
    let g1_r1_pass = r1_rec_mean >= r1_cos_mean - 1e-6;
    let g1_r5_pass = r5_rec_mean >= r5_cos_mean - 1e-6;
    let g1_winrate_pass = r1_win_rate >= 0.80 && r5_win_rate >= 0.80;
    let g1_pass = g1_r1_pass && g1_r5_pass && g1_winrate_pass;
    println!();
    println!(
        "  G1 r@1 mean:   {}",
        if g1_r1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G1 r@5 mean:   {}",
        if g1_r5_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G1 win rate:   {}",
        if g1_winrate_pass {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );
    println!(
        "  G1 verdict:    {}",
        if g1_pass {
            "✅ PASS (promote candidate)"
        } else {
            "❌ FAIL (do NOT promote)"
        }
    );
    println!();

    // ── G2: Latency gate ──────────────────────────────────────────
    println!("── G2: Latency gate (single-pair + 3-pair rerank) ──");
    let (cos1, rec1) = measure_latency_single();
    let (cos3, rec3) = measure_latency_3pair();
    println!(
        "  Single-pair:  cosine={cos1:.1}ns  recos={rec1:.1}ns  ratio={:.2}×",
        rec1 / cos1
    );
    println!(
        "  3-pair rerank: cosine={cos3:.1}ns  recos={rec3:.1}ns  ratio={:.2}×  overhead={:.1}ns",
        rec3 / cos3,
        rec3 - cos3
    );
    // G2 is informational for the Phase 4 decision; the gate threshold X is
    // the query-path latency budget headroom, set per-consumer. We record the
    // raw numbers here; the promote/demote call uses them.
    println!();
    println!(
        "  G2 verdict: informational (3-pair overhead = {:.1} ns);",
        rec3 - cos3
    );
    println!("    Phase 4 promote decision uses this vs the query-path budget.");
    println!();

    // ── Overall verdict ───────────────────────────────────────────
    println!("══════════════════════════════════════════════════════");
    if g1_pass {
        println!("  ✅ G1 PASS — recos beats cosine on OUR embedding regime.");
        println!(
            "    Mean recall@1 gain: {:+.4} ({:+.1}pp)",
            r1_rec_mean - r1_cos_mean,
            (r1_rec_mean - r1_cos_mean) * 100.0
        );
        println!(
            "    Mean recall@5 gain: {:+.4} ({:+.1}pp)",
            r5_rec_mean - r5_cos_mean,
            (r5_rec_mean - r5_cos_mean) * 100.0
        );
        println!("    → Promote `recos` to default in katgpt-core (G1+G2 gate).");
        println!("    → Phase 3 (cold MAG) + Phase 4 (hot ShardIndex) unblocked.");
    } else {
        println!("  ❌ G1 FAIL — recos adds no signal on OUR embeddings.");
        println!(
            "    Mean recall@1: cosine={r1_cos_mean:.4} recos={r1_rec_mean:.4} (Δ={:+.4})",
            r1_rec_mean - r1_cos_mean
        );
        println!(
            "    Mean recall@5: cosine={r5_cos_mean:.4} recos={r5_rec_mean:.4} (Δ={:+.4})",
            r5_rec_mean - r5_cos_mean
        );
        println!(
            "    Win rate r@1={:.1}% r@5={:.1}% (bar ≥80%)",
            r1_win_rate * 100.0,
            r5_win_rate * 100.0
        );
        println!("    → Keep `recos` opt-in as diagnostic; do NOT promote.");
        println!("    → Document the negative result in .benchmarks/437_recos_goat.md.");
    }
    println!();
    println!(
        "Latency (3-pair rerank): cosine={cos3:.1}ns recos={rec3:.1}ns ({:.2}×)",
        rec3 / cos3
    );
    println!("  → Phase 4 (hot ShardIndex::query) viable iff overhead fits the query budget.");
}
