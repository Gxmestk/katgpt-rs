//! Issue 041 PoC: smooth-min vs plain cosine on synthetic multi-token retrieval.
//!
//! Research 385 §4 says the GOAT gate PoC should use "synthetic multi-token
//! retrieval task." Issue 041 claims the PoC is "blocked on consumer
//! prerequisites." This PoC resolves that contradiction by using synthetic
//! per-token embeddings — no real consumer needed.
//!
//! The smooth-min function and edit_penalty are defined inline (not shipped
//! in katgpt-core) — this is a PoC, not a feature-gated primitive.
//!
//! Gates:
//! - G1 (quality): smooth-min recall@5 > plain-cosine recall@5 on queries
//!   with ≥2 token mismatches (all 4 tokens mismatched by construction)
//! - G2 (latency): smooth-min overhead < 100 ns per query (beyond plain mean)
//! - G3 (β sensitivity): β=10⁴ is the best operating point (paper's setting)
//!
//! Run: cargo run --release --example issue_041_smooth_min_poc

use std::hint::black_box;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════
// The primitive under test (inline — not shipped in katgpt-core yet)
// ═══════════════════════════════════════════════════════════════════════

/// Smooth-minimum similarity for variable-length soft pattern matching.
///
/// `cosines` = per-position cosine similarities c₁..cₘ (each in [-1, 1]).
/// `beta` = sharpness (paper uses 1e4; β→∞ = plain min, β≈1 = plain sum).
/// Returns similarity in [0, 1].
///
/// Formula: 1 - log_β(Σ(β^(1-c_i) - 1) + 1)
fn smooth_min_similarity(cosines: &[f32], beta: f32) -> f32 {
    let log_beta = beta.ln();
    let sum = cosines
        .iter()
        .map(|&c| ((1.0 - c) * log_beta).exp() - 1.0)
        .sum::<f32>()
        + 1.0;
    1.0 - sum.ln() / log_beta
}

/// Insertion/deletion penalty using Zipfian-whitened norm.
/// `norm_sq` = squared norm of the edited token's embedding.
/// `gamma` = penalty scale (paper: γ = m·γ').
#[allow(dead_code)] // included for reference; this PoC uses fixed-length patterns
fn edit_penalty(norm_sq: f32, gamma: f32) -> f32 {
    (-norm_sq / gamma).exp()
}

// ═══════════════════════════════════════════════════════════════════════
// Simple deterministic RNG (no external dependencies)
// ═══════════════════════════════════════════════════════════════════════

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Synthetic embeddings
// ═══════════════════════════════════════════════════════════════════════

const EMBED_DIM: usize = 16;
const NUM_CLUSTERS: usize = 10;
const WORDS_PER_CLUSTER: usize = 5;
const NUM_WORDS: usize = NUM_CLUSTERS * WORDS_PER_CLUSTER; // 50
const TOKENS_PER_ITEM: usize = 4;
const CATALOG_SIZE: usize = 200;
const NUM_QUERIES: usize = 200;

fn normalize(v: &mut [f32; EMBED_DIM]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn dot(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    let mut s = 0.0;
    for i in 0..EMBED_DIM {
        s += a[i] * b[i];
    }
    s
}

struct Embeddings {
    /// [word_id] → unit-norm embedding
    words: Vec<[f32; EMBED_DIM]>,
    /// [word_id] → cluster_id
    word_cluster: Vec<usize>,
}

fn make_embeddings(rng: &mut Rng) -> Embeddings {
    // Cluster centroids: random unit vectors.
    // Cross-cluster cosine ≈ 0.0-0.2 (random high-dim unit vectors).
    let mut centroids: Vec<[f32; EMBED_DIM]> = Vec::with_capacity(NUM_CLUSTERS);
    for _ in 0..NUM_CLUSTERS {
        let mut c = [0.0f32; EMBED_DIM];
        for c_i in c.iter_mut() {
            *c_i = rng.range(-1.0, 1.0);
        }
        normalize(&mut c);
        centroids.push(c);
    }

    // Words: centroid + 0.5*noise, normalized.
    // Within-cluster cosine ≈ 0.5-0.8 (shared centroid).
    // Cross-cluster cosine ≈ 0.0-0.2.
    let mut words = Vec::with_capacity(NUM_WORDS);
    let mut word_cluster = Vec::with_capacity(NUM_WORDS);
    for (cluster_id, centroid) in centroids.iter().enumerate() {
        for _ in 0..WORDS_PER_CLUSTER {
            let mut w = *centroid;
            for w_i in w.iter_mut() {
                *w_i += rng.range(-0.5, 0.5);
            }
            normalize(&mut w);
            words.push(w);
            word_cluster.push(cluster_id);
        }
    }

    Embeddings { words, word_cluster }
}

// ═══════════════════════════════════════════════════════════════════════
// Catalog and queries
// ═══════════════════════════════════════════════════════════════════════

struct Item {
    token_ids: [usize; TOKENS_PER_ITEM],
}

struct Query {
    token_ids: [usize; TOKENS_PER_ITEM],
    correct_item_idx: usize,
}

fn make_catalog_and_queries(rng: &mut Rng, emb: &Embeddings) -> (Vec<Item>, Vec<Query>) {
    // Catalog: 200 items, each with 4 tokens from random clusters.
    let mut catalog = Vec::with_capacity(CATALOG_SIZE);
    for _ in 0..CATALOG_SIZE {
        let mut token_ids = [0usize; TOKENS_PER_ITEM];
        for token_id in token_ids.iter_mut() {
            let cluster = rng.below(NUM_CLUSTERS);
            let word_in_cluster = rng.below(WORDS_PER_CLUSTER);
            *token_id = cluster * WORDS_PER_CLUSTER + word_in_cluster;
        }
        catalog.push(Item { token_ids });
    }

    // Queries: 200 queries with ALL 4 tokens mismatched (≥2 by construction).
    // For each query, the correct item's tokens are replaced with DIFFERENT
    // words from the SAME cluster. This gives moderate cosine (~0.5-0.8) at
    // all 4 positions for the correct item.
    //
    // Distractors: other catalog items. Some will accidentally share clusters
    // at 1-2 positions (high cosine ~0.8-1.0) but differ at the rest (low
    // cosine ~0.0-0.2). This is the scenario where smooth-min should win:
    // it penalizes the low-cosine positions, while plain cosine averages
    // them with the high-cosine ones.
    let mut queries = Vec::with_capacity(NUM_QUERIES);
    for q in 0..NUM_QUERIES {
        let correct_idx = q * 2 % CATALOG_SIZE;
        let correct_item = &catalog[correct_idx];
        let mut query_tokens = [0usize; TOKENS_PER_ITEM];
        for (p, qt) in query_tokens.iter_mut().enumerate() {
            let correct_word = correct_item.token_ids[p];
            let cluster = emb.word_cluster[correct_word];
            let idx_in_cluster = correct_word % WORDS_PER_CLUSTER;
            // Pick a DIFFERENT word from the same cluster
            let offset = 1 + rng.below(WORDS_PER_CLUSTER - 1);
            let new_idx = (idx_in_cluster + offset) % WORDS_PER_CLUSTER;
            *qt = cluster * WORDS_PER_CLUSTER + new_idx;
        }
        // Verify all 4 tokens are different from the correct item
        let mismatches = query_tokens
            .iter()
            .zip(correct_item.token_ids.iter())
            .filter(|(q, c)| q != c)
            .count();
        assert!(
            mismatches >= 2,
            "Query {q} has only {mismatches} mismatches (need ≥2)"
        );
        queries.push(Query {
            token_ids: query_tokens,
            correct_item_idx: correct_idx,
        });
    }

    (catalog, queries)
}

// ═══════════════════════════════════════════════════════════════════════
// Scoring
// ═══════════════════════════════════════════════════════════════════════

fn per_position_cosines(query: &Query, item: &Item, emb: &Embeddings) -> [f32; TOKENS_PER_ITEM] {
    let mut cosines = [0.0f32; TOKENS_PER_ITEM];
    for (cos, (&q_id, &i_id)) in cosines
        .iter_mut()
        .zip(query.token_ids.iter().zip(item.token_ids.iter()))
    {
        *cos = dot(&emb.words[q_id], &emb.words[i_id]);
    }
    cosines
}

fn plain_cosine_score(cosines: &[f32]) -> f32 {
    cosines.iter().sum::<f32>() / cosines.len() as f32
}

// ═══════════════════════════════════════════════════════════════════════
// G1: Quality gate — recall@k
// ═══════════════════════════════════════════════════════════════════════

fn recall_at_k(
    catalog: &[Item],
    queries: &[Query],
    emb: &Embeddings,
    k: usize,
    use_smooth_min: bool,
    beta: f32,
) -> f32 {
    let mut hits = 0;
    for query in queries {
        // Score all items
        let mut scores: Vec<(f32, usize)> = catalog
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let cosines = per_position_cosines(query, item, emb);
                let score = if use_smooth_min {
                    smooth_min_similarity(&cosines, beta)
                } else {
                    plain_cosine_score(&cosines)
                };
                (score, i)
            })
            .collect();

        // Sort descending by score
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Check if correct item is in top k
        let correct_idx = query.correct_item_idx;
        if scores.iter().take(k).any(|&(_, i)| i == correct_idx) {
            hits += 1;
        }
    }
    hits as f32 / queries.len() as f32
}

/// Also measure recall at multiple k values for a richer picture.
fn recall_curve(
    catalog: &[Item],
    queries: &[Query],
    emb: &Embeddings,
    use_smooth_min: bool,
    beta: f32,
) -> Vec<(usize, f32)> {
    let ks = [1, 3, 5, 10, 20];
    ks.iter()
        .map(|&k| (k, recall_at_k(catalog, queries, emb, k, use_smooth_min, beta)))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
// G2: Latency gate
// ═══════════════════════════════════════════════════════════════════════

fn measure_latency(beta: f32) -> (f64, f64) {
    // Measure the AGGREGATION step only (cosine computation is shared).
    // 4 cosines → 1 score. This isolates the smooth-min vs plain-mean cost.
    let cosines: [f32; TOKENS_PER_ITEM] = [0.72, 0.65, 0.58, 0.81];
    let n = 1_000_000;

    // Warmup
    for _ in 0..10_000 {
        black_box(plain_cosine_score(&cosines));
        black_box(smooth_min_similarity(&cosines, beta));
    }

    // Measure plain cosine (mean)
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..n {
        sink += plain_cosine_score(&cosines);
    }
    black_box(sink);
    let plain_ns = start.elapsed().as_secs_f64() * 1e9 / n as f64;

    // Measure smooth-min
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..n {
        sink += smooth_min_similarity(&cosines, beta);
    }
    black_box(sink);
    let smooth_ns = start.elapsed().as_secs_f64() * 1e9 / n as f64;

    (plain_ns, smooth_ns)
}

// ═══════════════════════════════════════════════════════════════════════
// G3: β sensitivity
// ═══════════════════════════════════════════════════════════════════════

fn beta_sensitivity(catalog: &[Item], queries: &[Query], emb: &Embeddings) {
    let betas: [f32; 6] = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6];
    println!("  β          recall@5");
    println!("  ────────── ─────────");
    for &beta in &betas {
        let recall = recall_at_k(catalog, queries, emb, 5, true, beta);
        let marker = if (beta - 1e4).abs() < 0.5 { " ← paper" } else { "" };
        println!("  {:>10.0e} {:.4}{}", beta, recall, marker);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════

fn main() {
    println!("=== Issue 041 PoC: smooth-min vs plain cosine ===");
    println!("Synthetic multi-token retrieval (per Research 385 §4)");
    println!();
    println!(
        "Config: {NUM_WORDS} words, {NUM_CLUSTERS} clusters × {WORDS_PER_CLUSTER} words/cluster"
    );
    println!(
        "        {CATALOG_SIZE} items, {NUM_QUERIES} queries (all ≥2 mismatches), {TOKENS_PER_ITEM} tokens/item"
    );
    println!();

    let mut rng = Rng::new(42);
    let emb = make_embeddings(&mut rng);
    let (catalog, queries) = make_catalog_and_queries(&mut rng, &emb);

    // Verify embedding structure
    let same_cluster_cosine = dot(&emb.words[0], &emb.words[1]); // cluster 0, different words
    let diff_cluster_cosine = dot(&emb.words[0], &emb.words[WORDS_PER_CLUSTER]); // cluster 0 vs cluster 1
    println!("── Embedding structure check ──");
    println!(
        "  Same cluster, different word:  {:.4}  (expect 0.5-0.8)",
        same_cluster_cosine
    );
    println!(
        "  Different cluster:             {:.4}  (expect 0.0-0.2)",
        diff_cluster_cosine
    );
    println!();

    // Show a sample query's cosines against the correct item vs a distractor
    let sample_q = &queries[0];
    let correct_item = &catalog[sample_q.correct_item_idx];
    let correct_cosines = per_position_cosines(sample_q, correct_item, &emb);
    let correct_plain = plain_cosine_score(&correct_cosines);
    let correct_smooth = smooth_min_similarity(&correct_cosines, 1e4);

    // Find a distractor with high plain-cosine score (the kind smooth-min should reject)
    let mut best_distractor: Option<(usize, [f32; TOKENS_PER_ITEM], f32, f32)> = None;
    for (i, item) in catalog.iter().enumerate() {
        if i == sample_q.correct_item_idx {
            continue;
        }
        let cosines = per_position_cosines(sample_q, item, &emb);
        let plain = plain_cosine_score(&cosines);
        let smooth = smooth_min_similarity(&cosines, 1e4);
        if best_distractor.is_none_or(|(_, _, bp, _)| plain > bp) {
            best_distractor = Some((i, cosines, plain, smooth));
        }
    }

    println!("── Sample query (q=0) ──");
    println!(
        "  Correct item [{}]: cosines = {:?}",
        sample_q.correct_item_idx,
        correct_cosines
    );
    println!(
        "    plain={:.4}  smooth_min={:.4}",
        correct_plain, correct_smooth
    );
    if let Some((idx, cosines, plain, smooth)) = best_distractor {
        println!(
            "  Top distractor [{}]: cosines = {:?}",
            idx, cosines
        );
        println!("    plain={:.4}  smooth_min={:.4}", plain, smooth);
        println!(
            "  → Plain margin:    {:.4} (correct - distractor)",
            correct_plain - plain
        );
        println!(
            "  → Smooth-min margin: {:.4} (correct - distractor)",
            correct_smooth - smooth
        );
    }
    println!();

    // G1: Quality gate
    println!("── G1: Quality gate (recall@k) ──");
    let plain_curve = recall_curve(&catalog, &queries, &emb, false, 0.0);
    let smooth_curve = recall_curve(&catalog, &queries, &emb, true, 1e4);

    println!("  k     plain-cosine  smooth-min    gain");
    println!("  ───── ───────────── ───────────── ──────");
    for (plain, smooth) in plain_curve.iter().zip(smooth_curve.iter()) {
        let gain = smooth.1 - plain.1;
        let marker = if gain > 0.0 { " ✅" } else if gain < 0.0 { " ❌" } else { "" };
        println!(
            "  {:<5} {:.4}        {:.4}        {:+.4}{}",
            plain.0, plain.1, smooth.1, gain, marker
        );
    }

    let plain_recall5 = plain_curve[2].1; // k=5
    let smooth_recall5 = smooth_curve[2].1;
    let gain5 = smooth_recall5 - plain_recall5;
    println!();
    println!("  recall@5 gain: {:+.4} ({:+.1}pp)", gain5, gain5 * 100.0);
    if smooth_recall5 > plain_recall5 {
        println!("  ✅ G1 PASS: smooth-min beats plain cosine at recall@5");
    } else {
        println!("  ❌ G1 FAIL: smooth-min does not beat plain cosine at recall@5");
    }
    println!();

    // G2: Latency gate
    println!("── G2: Latency gate (<100ns overhead) ──");
    let (plain_ns, smooth_ns) = measure_latency(1e4);
    let overhead = smooth_ns - plain_ns;
    println!("  Plain cosine (mean):  {:.1} ns/call", plain_ns);
    println!("  Smooth-min (β=10⁴):   {:.1} ns/call", smooth_ns);
    println!("  Overhead:             {:.1} ns/call", overhead);
    if overhead < 100.0 {
        println!("  ✅ G2 PASS: overhead < 100 ns");
    } else {
        println!("  ❌ G2 FAIL: overhead ≥ 100 ns");
    }
    println!();

    // G3: β sensitivity
    println!("── G3: β sensitivity (paper: β=10⁴) ──");
    beta_sensitivity(&catalog, &queries, &emb);
    println!();

    // Overall verdict
    println!("══════════════════════════════════════════════════════");
    let g1_pass = smooth_recall5 > plain_recall5;
    let g2_pass = overhead < 100.0;
    if g1_pass && g2_pass {
        println!("  ✅ GOAT gate PASS — smooth-min is validated.");
        println!("  → Recommend implementing in katgpt-core behind");
        println!("    feature flag `smooth_min_similarity` (opt-in).");
        println!("  → Consumer wiring (ItemEmbedIndex, AnyRAG, soft");
        println!("    Engram) is the next step — separate from the");
        println!("    primitive itself.");
    } else {
        println!("  ❌ GOAT gate FAIL — smooth-min does not meet the gate.");
        if !g1_pass {
            println!("    G1 (quality): smooth-min recall@5 ≤ plain cosine");
        }
        if !g2_pass {
            println!("    G2 (latency): overhead ≥ 100 ns");
        }
        println!("  → Issue 041 stays blocked.");
    }
}
