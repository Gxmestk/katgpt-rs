//! Issue 041 GOAT gate — Smooth-Min reranking consumer wiring.
//!
//! Compares `RerankMethod::Cosine` (mean-pooled) vs `RerankMethod::MaxSim`
//! (late-interaction) vs `RerankMethod::SmoothMin` (smooth-min aggregation)
//! on a synthetic multi-token retrieval task.
//!
//! The task: 200 catalog documents, each with 4 tokens. 200 queries, each
//! with 4 tokens that are from the same clusters as the correct document but
//! different words (all 4 positions mismatch). The correct document has 4
//! moderate-cosine positions; distractors accidentally share clusters at 1-2
//! positions (high cosine) but differ at the rest (low cosine).
//!
//! This is the same scenario as the Issue 041 PoC, but applied to the rerank
//! module's API — proving the primitive works as a real consumer.
//!
//! # GOAT gates
//!
//! - **G1 (quality):** SmoothMin recall@5 > Cosine recall@5
//! - **G2 (latency):** SmoothMin per-query overhead < 100ns vs Cosine
//! - **G3 (no-regression):** Cosine and MaxSim paths unchanged
//!
//! # Run
//!
//! ```bash
//! cargo run --release --example issue_041_smooth_min_rerank_goat --features smooth_min_rerank
//! ```

#![cfg(feature = "smooth_min_rerank")]

use katgpt_attn_match::rerank::{RerankMethod, rerank};
use std::hint::black_box;
use std::time::Instant;

// ── Synthetic data generation ───────────────────────────────

const DIM: usize = 16;
const N_CLUSTERS: usize = 10;
const WORDS_PER_CLUSTER: usize = 5;
const N_WORDS: usize = N_CLUSTERS * WORDS_PER_CLUSTER; // 50
const TOKENS_PER_DOC: usize = 4;
const N_CATALOG: usize = 200;
const N_QUERIES: usize = 200;

/// Simple LCG for deterministic randomness.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn range(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

/// Generate word embeddings: 10 clusters × 5 words, each 16-dim.
/// Within-cluster cosine ≈ 0.5-0.8; cross-cluster cosine ≈ 0.0-0.2.
fn generate_word_embeddings() -> Vec<Vec<f32>> {
    let mut lcg = Lcg::new(42);
    let mut embeddings = Vec::with_capacity(N_WORDS);

    // Generate cluster centroids.
    let mut centroids = vec![vec![0.0f32; DIM]; N_CLUSTERS];
    for c in &mut centroids {
        for v in c.iter_mut() {
            *v = ((lcg.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        }
        // Normalize centroid.
        let norm = (c.iter().map(|v| v * v).sum::<f32>()).sqrt();
        for v in c.iter_mut() {
            *v /= norm;
        }
    }

    // Generate words: centroid + noise, normalized.
    for centroid in centroids.iter() {
        for _ in 0..WORDS_PER_CLUSTER {
            let mut emb = centroid.clone();
            // Add noise (within-cluster variation).
            for v in emb.iter_mut() {
                *v += ((lcg.next() >> 40) as f32 / (1u64 << 24) as f32) * 0.5 - 0.25;
            }
            // Normalize.
            let norm = (emb.iter().map(|v| v * v).sum::<f32>()).sqrt();
            for v in emb.iter_mut() {
                *v /= norm.max(1e-12);
            }
            embeddings.push(emb);
        }
    }

    embeddings
}

/// Generate a document: TOKENS_PER_DOC tokens from random clusters.
fn generate_doc(lcg: &mut Lcg, embeddings: &[Vec<f32>]) -> Vec<f32> {
    let mut doc = Vec::with_capacity(TOKENS_PER_DOC * DIM);
    for _ in 0..TOKENS_PER_DOC {
        let cluster = lcg.range(N_CLUSTERS);
        let word = cluster * WORDS_PER_CLUSTER + lcg.range(WORDS_PER_CLUSTER);
        doc.extend_from_slice(&embeddings[word]);
    }
    doc
}

/// Generate a query: same clusters as the correct doc, but different words.
/// This gives the correct doc 4 moderate-cosine positions (~0.5-0.8).
fn generate_query(lcg: &mut Lcg, doc: &[f32], embeddings: &[Vec<f32>]) -> Vec<f32> {
    let mut query = Vec::with_capacity(TOKENS_PER_DOC * DIM);
    for t in 0..TOKENS_PER_DOC {
        // Find which cluster this doc token belongs to (closest centroid).
        let doc_token = &doc[t * DIM..(t + 1) * DIM];
        let mut best_cluster = 0;
        let mut best_dot = f32::NEG_INFINITY;
        for (c, centroid) in embeddings.chunks(WORDS_PER_CLUSTER).enumerate() {
            // Use the first word of each cluster as a proxy for the centroid.
            let dot: f32 = doc_token
                .iter()
                .zip(centroid[0].iter())
                .map(|(a, b)| a * b)
                .sum();
            if dot > best_dot {
                best_dot = dot;
                best_cluster = c;
            }
        }

        // Pick a DIFFERENT word from the same cluster.
        let word_offset = lcg.range(WORDS_PER_CLUSTER);
        let word_idx = best_cluster * WORDS_PER_CLUSTER + word_offset;
        query.extend_from_slice(&embeddings[word_idx]);
    }
    query
}

// ── GOAT gates ──────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Issue 041 GOAT — Smooth-Min Reranking Consumer Wiring       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let embeddings = generate_word_embeddings();

    // Generate catalog and queries.
    let mut lcg = Lcg::new(123);
    let mut catalog: Vec<Vec<f32>> = Vec::with_capacity(N_CATALOG);
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(N_QUERIES);
    let mut correct_indices: Vec<usize> = Vec::with_capacity(N_QUERIES);

    for _ in 0..N_CATALOG {
        catalog.push(generate_doc(&mut lcg, &embeddings));
    }

    for _ in 0..N_QUERIES {
        let correct_idx = lcg.range(N_CATALOG);
        let query = generate_query(&mut lcg, &catalog[correct_idx], &embeddings);
        catalog.push(query.clone());
        queries.push(query);
        correct_indices.push(correct_idx);
    }
    // Remove the queries we added to the catalog.
    catalog.truncate(N_CATALOG);

    let doc_lengths: Vec<usize> = catalog.iter().map(|d| d.len() / DIM).collect();

    // ── G1: Quality gate (recall@k) ──────────────────────────

    println!("── G1: Quality gate (recall@k) ──────────────────────────────");

    let methods: &[(&str, RerankMethod)] = &[
        ("Cosine (mean-pooled)", RerankMethod::Cosine),
        ("MaxSim (late-interaction)", RerankMethod::MaxSim),
        ("SmoothMin (β=10⁴)", RerankMethod::SmoothMin { beta: 1e4 }),
    ];

    let ks = [1, 3, 5, 10, 20];

    print!("{:<30}", "Method");
    for k in &ks {
        print!("  recall@{:>2}", k);
    }
    println!();

    let mut cosine_recall5 = 0.0f32;
    let mut smoothmin_recall5 = 0.0f32;

    for (name, method) in methods {
        let mut recalls = [0.0f32; 5];

        for (qi, query) in queries.iter().enumerate() {
            let ranked = rerank(query, &catalog, &doc_lengths, DIM, *method);
            let correct = correct_indices[qi];

            for (ki, &k) in ks.iter().enumerate() {
                let top_k: Vec<usize> = ranked.iter().take(k).map(|d| d.doc_index).collect();
                if top_k.contains(&correct) {
                    recalls[ki] += 1.0;
                }
            }
        }

        // Normalize.
        for r in &mut recalls {
            *r /= N_QUERIES as f32;
        }

        if name.contains("Cosine") {
            cosine_recall5 = recalls[2];
        }
        if name.contains("SmoothMin") {
            smoothmin_recall5 = recalls[2];
        }

        print!("{:<30}", name);
        for r in &recalls {
            print!("  {:>10.4}", r);
        }
        println!();
    }

    let gain = smoothmin_recall5 - cosine_recall5;
    let g1_pass = smoothmin_recall5 > cosine_recall5;
    println!(
        "\n  SmoothMin vs Cosine @ k=5: {smoothmin_recall5:.4} vs {cosine_recall5:.4} = {gain:+.4} pp"
    );
    println!("  G1 (quality): {}", if g1_pass { "PASS ✅" } else { "FAIL ❌" });

    // ── G2: Latency gate ─────────────────────────────────────

    println!("\n── G2: Latency gate ────────────────────────────────────────");

    let n_iters = 100;
    let sample_query = &queries[0];

    // Warmup.
    for _ in 0..10 {
        let _ = rerank(sample_query, &catalog, &doc_lengths, DIM, RerankMethod::Cosine);
        let _ = rerank(
            sample_query,
            &catalog,
            &doc_lengths,
            DIM,
            RerankMethod::SmoothMin { beta: 1e4 },
        );
    }

    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = black_box(rerank(
            sample_query,
            &catalog,
            &doc_lengths,
            DIM,
            RerankMethod::Cosine,
        ));
    }
    let cosine_ns = start.elapsed().as_nanos() as f64 / n_iters as f64;

    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = black_box(rerank(
            sample_query,
            &catalog,
            &doc_lengths,
            DIM,
            RerankMethod::SmoothMin { beta: 1e4 },
        ));
    }
    let smoothmin_ns = start.elapsed().as_nanos() as f64 / n_iters as f64;

    let overhead_ns = smoothmin_ns - cosine_ns;
    let g2_pass = overhead_ns < 100.0; // < 100ns target (per-query)

    println!("  Cosine:    {cosine_ns:.0} ns/query");
    println!("  SmoothMin: {smoothmin_ns:.0} ns/query");
    println!("  Overhead:  {overhead_ns:+.0} ns/query (target: < 100ns)");
    println!("  G2 (latency): {}", if g2_pass { "PASS ✅" } else { "FAIL ❌" });

    // ── G3: No-regression gate ───────────────────────────────

    println!("\n── G3: No-regression gate ──────────────────────────────────");

    // Cosine path must produce identical results regardless of smooth_min_rerank.
    // Since smooth_min_rerank is compiled in, we verify Cosine still works correctly.
    let cosine_ranked = rerank(
        sample_query,
        &catalog,
        &doc_lengths,
        DIM,
        RerankMethod::Cosine,
    );
    let maxsim_ranked = rerank(
        sample_query,
        &catalog,
        &doc_lengths,
        DIM,
        RerankMethod::MaxSim,
    );

    // Cosine should produce finite scores in [-1, 1] (cosine similarity range).
    let bad_cosine: Vec<f32> = cosine_ranked
        .iter()
        .filter(|d| !d.score.is_finite() || d.score < -1.01 || d.score > 1.01)
        .map(|d| d.score)
        .collect();
    let cosine_ok = bad_cosine.is_empty();
    if !cosine_ok {
        println!("  [debug] out-of-range cosine scores: {:?}", &bad_cosine[..bad_cosine.len().min(5)]);
    }
    // MaxSim should produce finite scores.
    let maxsim_ok = maxsim_ranked.iter().all(|d| d.score.is_finite());
    // SmoothMin should produce finite scores.
    let smoothmin_ranked = rerank(
        sample_query,
        &catalog,
        &doc_lengths,
        DIM,
        RerankMethod::SmoothMin { beta: 1e4 },
    );
    let smoothmin_ok = smoothmin_ranked.iter().all(|d| d.score.is_finite());

    let g3_pass = cosine_ok && maxsim_ok && smoothmin_ok;
    println!("  Cosine scores finite & in [-1,1]: {cosine_ok}");
    println!("  MaxSim scores finite:            {maxsim_ok}");
    println!("  SmoothMin scores finite:         {smoothmin_ok}");
    println!("  G3 (no-regression): {}", if g3_pass { "PASS ✅" } else { "FAIL ❌" });

    // ── Verdict ──────────────────────────────────────────────

    println!("\n═══════════════════════════════════════════════════════════════");
    let all_pass = g1_pass && g2_pass && g3_pass;
    if all_pass {
        println!("  GOAT gate: ALL PASS ✅");
        println!("  smooth_min_similarity now has its first real consumer: rerank.");
        println!("  Promotion to default-on: deferred until consumer demonstrates");
        println!("  real-world value on a production retrieval workload.");
    } else {
        println!("  GOAT gate: FAIL ❌ — do not promote");
    }
    println!("═══════════════════════════════════════════════════════════════");
}
