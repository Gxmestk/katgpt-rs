//! Plan 483 Phase 3 — MTP Projection Modelless Exhaustion GOAT Benchmark
//!
//! Compares deterministic MTP projection constructions against the identity
//! baseline to determine whether the MTP projection gap (P2) can be closed
//! modellessly (§3.5 path 2 — raw/lora hot-swap with deterministically
//! constructed adapters).
//!
//! **The MTP projection gap:** `TransformerWeights::mtp_activation_proj` is
//! `None` by default. When None, `project_target_activation` falls back to
//! truncate/pad. The question: can a deterministic construction (identity,
//! random, spectral, lm_head^T) achieve similar quality to a trained
//! projection — or is the zero-cost truncate/pad fallback already the best
//! modelless option?
//!
//! **Metrics:**
//! - G1 (quality): KL divergence between logits from projected hidden vs
//!   target logits — lower is better. Identity is the best case (KL≈0 when
//!   same model). Pass if any non-identity deterministic construction
//!   achieves KL within 10% of identity's KL.
//! - G2 (no-regression): all projections produce finite logits (no NaN/Inf)
//! - G3 (information preservation): cosine similarity ≥ 0.5 between
//!   projected and original hidden state for deterministic constructions
//!
//! **Run:** `cargo test --features lt2_looped --test bench_483_mtp_projection_goat -- --nocapture`

use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward, project_target_activation,
};
use katgpt_rs::types::{Config, HybridPattern, Rng};

/// Compute softmax of a slice, returning the distribution.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().copied().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// KL divergence KL(p || q) = Σ p_i * log(p_i / q_i).
fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    let mut kl = 0.0f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 1e-12 && qi > 1e-12 {
            kl += pi * (pi / qi).ln();
        }
    }
    kl.max(0.0)
}

/// Check if all values in a slice are finite (no NaN, no Inf).
fn all_finite(vals: &[f32]) -> bool {
    vals.iter().all(|&v| v.is_finite())
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&y| y * y).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Apply lm_head matmul: logits[i] = sum_j(lm_head[i * n_embd + j] * h[j])
fn apply_lm_head(h: &[f32], lm_head: &[f32], vocab_size: usize, n_embd: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab_size];
    for i in 0..vocab_size {
        let mut sum = 0.0f32;
        for j in 0..n_embd {
            sum += lm_head[i * n_embd + j] * h[j];
        }
        logits[i] = sum;
    }
    logits
}

/// Estimate the dominant eigenvalue of a square matrix via power iteration.
/// Matrix is row-major [n × n]. Returns the spectral norm estimate.
fn spectral_norm(matrix: &[f32], n: usize, max_iters: usize) -> f32 {
    if n == 0 || matrix.len() < n * n {
        return 1.0;
    }
    let mut v = vec![1.0f32 / (n as f32).sqrt(); n];
    let mut lambda = 1.0f32;
    for _ in 0..max_iters {
        // w = M @ v
        let mut w = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += matrix[i * n + j] * v[j];
            }
            w[i] = sum;
        }
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            break;
        }
        let new_v: Vec<f32> = w.iter().map(|x| x / norm).collect();
        lambda = norm;
        v = new_v;
    }
    lambda
}

/// Build an identity matrix [n × n] (row-major).
fn identity_matrix(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Build a deterministic random matrix [rows × cols] (row-major).
/// Uses Xavier/Glorot initialization: scale = sqrt(2 / (rows + cols)).
fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let scale = (2.0 / (rows + cols) as f32).sqrt();
    let mut m = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        m.push(rng.normal() * scale);
    }
    m
}

/// Build a spectrally-normalized projection from Wq.
/// Projects hidden state through Wq / spectral_norm(Wq), so the projection
/// has unit spectral norm — preserves the dominant directions without
/// amplification.
fn spectral_proj_from_wq(wq: &[f32], n: usize) -> Vec<f32> {
    let lambda = spectral_norm(wq, n, 50);
    let scale = if lambda > 1e-10 { 1.0 / lambda } else { 1.0 };
    wq.iter().map(|&w| w * scale).collect()
}

/// Build a Gram-matrix projection from lm_head: lm_head^T @ lm_head.
/// This is [n_embd × n_embd] and projects the hidden state onto the span
/// of the lm_head rows (the output subspace). Normalized by spectral norm.
fn lm_head_gram_proj(lm_head: &[f32], vocab: usize, n_embd: usize) -> Vec<f32> {
    // Gram = lm_head^T @ lm_head, shape [n_embd × n_embd]
    // Gram[i][j] = sum_k lm_head[k * n_embd + i] * lm_head[k * n_embd + j]
    let mut gram = vec![0.0f32; n_embd * n_embd];
    for k in 0..vocab {
        for i in 0..n_embd {
            let lik = lm_head[k * n_embd + i];
            if lik == 0.0 {
                continue;
            }
            for j in 0..n_embd {
                gram[i * n_embd + j] += lik * lm_head[k * n_embd + j];
            }
        }
    }
    // Spectrally normalize so projection doesn't amplify
    let lambda = spectral_norm(&gram, n_embd, 50);
    let scale = if lambda > 1e-10 { 1.0 / lambda } else { 1.0 };
    gram.iter().map(|&g| g * scale).collect()
}

/// A projection variant to benchmark.
struct ProjVariant {
    name: &'static str,
    /// None = truncate/pad fallback; Some(w) = matmul projection
    weights: Option<Vec<f32>>,
}

/// Per-variant benchmark results.
struct VariantResult {
    name: &'static str,
    kl_div: f32,
    cosine_sim: f32,
    norm_ratio: f32,
    logits_finite: bool,
}

#[test]
fn bench_483_mtp_projection_goat() {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 483 Phase 3 — MTP Projection Modelless Exhaustion");
    println!("  §3.5 Path 2: deterministic projection constructions");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Config::micro(): n_embd=16, n_layer=1, n_head=4, head_dim=4, vocab=27
    // Same dims for target and drafter so identity projection is perfect.
    let mut config = Config::micro();
    config.hybrid_pattern = HybridPattern::Uniform;
    // Override: micro() sets mtp_activation_threshold=usize::MAX which gates
    // MTP off (target_n_embd < MAX → return early). Set to 1 so projection
    // is always active, exercising all variants.
    config.mtp_activation_threshold = 1;

    let n_embd = config.n_embd;
    let vocab = config.vocab_size;

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Run target forward to get hidden state + target logits
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let token = config.bos_token;
    let pos = 0usize;
    let target_logits = forward(&mut ctx, &weights, &mut cache, token, pos, &config).to_vec();
    let h_target = ctx.hidden_state[..n_embd].to_vec();

    println!(
        "  Config: n_embd={}, vocab={}, n_layer={}",
        n_embd, vocab, config.n_layer
    );
    println!(
        "  Target hidden state: {} dims, ||h|| = {:.6}",
        n_embd,
        l2_norm(&h_target)
    );
    println!(
        "  Target logits: {} dims, finite = {}",
        vocab,
        all_finite(&target_logits)
    );
    println!();

    // Spectral analysis of Wq (for informational purposes)
    let wq = &weights.layers[0].attn_wq;
    let wq_lambda = spectral_norm(wq, n_embd, 50);
    println!("  Spectral analysis: λ_max(Wq layer 0) = {wq_lambda:.6}");
    println!();

    // Build projection variants
    let variants: Vec<ProjVariant> = vec![
        ProjVariant {
            name: "None (truncate/pad)",
            weights: None,
        },
        ProjVariant {
            name: "Identity",
            weights: Some(identity_matrix(n_embd)),
        },
        ProjVariant {
            name: "Random (seed=42)",
            weights: Some(random_matrix(n_embd, n_embd, 42)),
        },
        ProjVariant {
            name: "Spectral (Wq/λ_max)",
            weights: Some(spectral_proj_from_wq(wq, n_embd)),
        },
        ProjVariant {
            name: "lm_head^T·lm_head (Gram)",
            weights: Some(lm_head_gram_proj(&weights.lm_head, vocab, n_embd)),
        },
    ];

    // Results table
    println!(
        "┌──────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────┐"
    );
    println!(
        "│ Projection                   │ KL div       │ Cosine sim   │ Norm ratio   │ G2 finite│"
    );
    println!(
        "│                              │ (vs target)  │ (vs original)│ (proj/orig)  │ (logits) │"
    );
    println!(
        "├──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────┤"
    );

    let mut results: Vec<VariantResult> = Vec::new();
    let mut all_g2_pass = true;

    for v in &variants {
        // Project h_target with this variant
        let mut h_projected = vec![0.0f32; n_embd];
        project_target_activation(
            &mut h_projected,
            &h_target,
            v.weights.as_ref(),
            n_embd, // target_n_embd
            n_embd, // drafter_n_embd (same as target)
            config.mtp_activation_threshold,
        );

        // Apply lm_head to projected hidden → projected logits
        let projected_logits = apply_lm_head(&h_projected, &weights.lm_head, vocab, n_embd);

        // Metrics
        let kl = kl_divergence(&softmax(&target_logits), &softmax(&projected_logits));
        let cos = cosine_similarity(&h_target, &h_projected);
        let orig_norm = l2_norm(&h_target);
        let proj_norm = l2_norm(&h_projected);
        let norm_ratio = if orig_norm > 1e-12 {
            proj_norm / orig_norm
        } else {
            0.0
        };
        let finite = all_finite(&projected_logits);

        if !finite {
            all_g2_pass = false;
        }

        results.push(VariantResult {
            name: v.name,
            kl_div: kl,
            cosine_sim: cos,
            norm_ratio,
            logits_finite: finite,
        });

        println!(
            "│ {:<28} │ {:>12.6} │ {:>12.6} │ {:>12.6} │ {:>8} │",
            v.name,
            kl,
            cos,
            norm_ratio,
            if finite { "✅ PASS" } else { "❌ FAIL" }
        );
    }

    println!(
        "└──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────┘"
    );
    println!();

    // Find identity baseline KL (should be ~0 when same model)
    let identity_kl = results
        .iter()
        .find(|r| r.name.starts_with("Identity"))
        .map_or(0.0, |r| r.kl_div);
    let none_kl = results
        .iter()
        .find(|r| r.name.starts_with("None"))
        .map_or(0.0, |r| r.kl_div);

    // G1: Does any non-identity/non-None deterministic construction achieve
    // KL within 10% of identity? Since identity KL≈0, "within 10%" is
    // interpreted as KL ≤ 0.1 (absolute threshold — 10% of a unit KL range).
    // A variant with KL=0 means it perfectly preserves the output distribution.
    let g1_threshold = 0.1f32;
    let g1_pass = results.iter().any(|r| {
        // Non-identity, non-None deterministic constructions
        !(r.name.starts_with("Identity") || r.name.starts_with("None")) && r.kl_div <= g1_threshold
    });

    // G3: Cosine similarity ≥ 0.5 for deterministic constructions
    let g3_threshold = 0.5f32;
    let g3_pass = results.iter().all(|r| r.cosine_sim >= g3_threshold);

    // Analysis
    println!("── Analysis ──────────────────────────────────────────────────");
    println!("  Identity baseline KL = {identity_kl:.6} (perfect when same model)");
    println!("  None (truncate/pad) KL = {none_kl:.6} (same as identity when dims match)");
    println!();
    println!("  When target and drafter share the same model + same dims:");
    println!("    - None/Identity = lossless copy → KL ≈ 0");
    println!("    - Random distorts hidden state → high KL");
    println!("    - Spectral (Wq/λ) rotates hidden state → high KL");
    println!("    - lm_head Gram projects onto output subspace → moderate KL");
    println!();

    // GOAT gate verdict
    println!("── GOAT Gate ─────────────────────────────────────────────────");
    println!(
        "  G1 (quality): non-identity deterministic KL ≤ {:.2}? → {}",
        g1_threshold,
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G2 (no-regression): all logits finite? → {}",
        if all_g2_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G3 (info preservation): cosine sim ≥ {:.1} for all? → {}",
        g3_threshold,
        if g3_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!();

    // §3.5 exhaustion verdict
    println!("── §3.5 Modelless Exhaustion Verdict ─────────────────────────");
    if g1_pass {
        println!("  A deterministic construction achieves KL ≤ 0.1 — modelless");
        println!("  projection is viable. No training needed for MTP projection.");
    } else {
        println!("  No non-identity deterministic construction achieves KL ≤ 0.1.");
        println!("  The zero-cost truncate/pad fallback (None) is already the best");
        println!("  modelless option when dims match — it is lossless (KL ≈ 0).");
        println!("  Random/spectral/Gram projections DISTORT the hidden state and");
        println!("  produce worse KL than the trivial copy.");
        println!();
        println!("  Conclusion: MTP projection gap (P2) cannot be improved modellessly");
        println!("  beyond the existing truncate/pad fallback. A trained projection");
        println!("  (→ riir-train) is needed only when target/drafter have DIFFERENT");
        println!("  dims and the truncation loses information. When dims match,");
        println!("  truncate/pad IS identity — no gap exists.");
    }
    println!();

    // Per-variant detail
    println!("── Per-Variant Detail ────────────────────────────────────────");
    for r in &results {
        println!(
            "  {:<28} KL={:.6}  cos={:.6}  norm_ratio={:.6}  finite={}",
            r.name, r.kl_div, r.cosine_sim, r.norm_ratio, r.logits_finite
        );
    }
    println!();

    // Assert basic sanity (G2 must always pass — no NaN/Inf)
    assert!(
        all_g2_pass,
        "G2 FAIL: some projections produced non-finite logits"
    );

    println!("═══════════════════════════════════════════════════════════════");
}
