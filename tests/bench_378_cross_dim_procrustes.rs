//! Issue 378 — Cross-Dimension MTP Projection: Procrustes Alignment Benchmark
//!
//! Tests whether Procrustes alignment (closed-form SVD) can close the cross-dim
//! MTP projection gap modellessly. The same-dim case is already modelless-closed
//! (truncate/pad = identity, KL=0). This benchmark tests the cross-dim case
//! (target_n_embd ≠ drafter_n_embd) where truncate/pad is lossy.
//!
//! **Methodology:**
//! 1. Create a target model (n_embd=16) and a draft model (n_embd=8).
//! 2. Run both on the same tokens to collect paired (h_target, h_draft) samples.
//! 3. Compute Procrustes P [8, 16] from paired samples via SVD of cross-covariance.
//! 4. Compare projections: truncate/pad vs Procrustes vs random.
//! 5. Quality metric: KL(target_logits, draft_lm_head @ projected_h_target).
//!
//! **Key insight:** Cross-dim projection ALWAYS loses information (d < D).
//! KL=0 is impossible. The question is whether Procrustes beats truncate/pad
//! by finding a better linear combination of the target's dimensions.
//!
//! **GOAT gate:** Procrustes KL < truncate/pad KL (improvement) AND
//! Procrustes KL ≤ 0.1 (quality threshold for modelless-closed).
//!
//! **Run:** `cargo test --release --test bench_378_cross_dim_procrustes -- --nocapture`

use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward, project_target_activation,
};
use katgpt_rs::types::{Config, HybridPattern, Rng};

// ── Metrics helpers (mirrors bench_483 for consistency) ─────────────────────

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().copied().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    let mut kl = 0.0f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 1e-12 && qi > 1e-12 {
            kl += pi * (pi / qi).ln();
        }
    }
    kl.max(0.0)
}

fn all_finite(vals: &[f32]) -> bool {
    vals.iter().all(|&v| v.is_finite())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&y| y * y).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

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

// ── Linear algebra helpers (self-contained, no nalgebra dep) ────────────────

/// Jacobi eigendecomposition of a symmetric matrix (f64 for numerical stability).
///
/// Decomposes `a_in` [r × r] (row-major, symmetric) into eigenvalues `eigvals` [r]
/// and eigenvectors `eigvecs` [r × r] (column j is the eigenvector for eigvals[j]).
///
/// Uses the cyclic Jacobi sweep algorithm. Adapted from katgpt-core/src/karc.rs.
fn jacobi_eigen_symmetric(
    eigvals: &mut [f64],
    eigvecs: &mut [f64],
    a_in: &[f64],
    r: usize,
    tol: f64,
    max_sweeps: usize,
) {
    let mut scratch = vec![0.0f64; r * r];
    scratch[..r * r].copy_from_slice(a_in);
    for i in 0..r {
        for j in 0..r {
            eigvecs[i * r + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for _ in 0..max_sweeps {
        let mut off_sq = 0.0f64;
        for p in 0..r {
            for q in (p + 1)..r {
                off_sq += scratch[p * r + q] * scratch[p * r + q];
            }
        }
        if off_sq < tol {
            break;
        }
        for p in 0..r {
            for q in (p + 1)..r {
                let apq = scratch[p * r + q];
                if apq.abs() < f64::MIN_POSITIVE {
                    continue;
                }
                let app = scratch[p * r + p];
                let aqq = scratch[q * r + q];
                let theta = if (app - aqq).abs() < f64::MIN_POSITIVE {
                    core::f64::consts::FRAC_PI_4
                } else {
                    0.5 * (2.0 * apq / (app - aqq)).atan()
                };
                let c = theta.cos();
                let s = theta.sin();
                for i in 0..r {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = scratch[i * r + p];
                    let aiq = scratch[i * r + q];
                    scratch[i * r + p] = c * aip - s * aiq;
                    scratch[p * r + i] = scratch[i * r + p];
                    scratch[i * r + q] = s * aip + c * aiq;
                    scratch[q * r + i] = scratch[i * r + q];
                }
                scratch[p * r + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                scratch[q * r + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                scratch[p * r + q] = 0.0;
                scratch[q * r + p] = 0.0;
                for i in 0..r {
                    let vip = eigvecs[i * r + p];
                    let viq = eigvecs[i * r + q];
                    eigvecs[i * r + p] = c * vip - s * viq;
                    eigvecs[i * r + q] = s * vip + c * viq;
                }
            }
        }
    }
    for i in 0..r {
        eigvals[i] = scratch[i * r + i];
    }
}

/// Compute the orthogonal Procrustes projection P [d × D] from paired samples.
///
/// Given cross-covariance C [d × D] = Σ_i h_draft_i ⊗ h_target_i,
/// computes P = U V^T where C = U Σ V^T is the reduced SVD.
///
/// Returns P in row-major [d × D] layout, matching `project_target_activation`'s
/// expected `mtp_proj` format: `out[i] = Σ_j P[i * D + j] * target_hidden[j]`.
fn compute_procrustes(
    cross_cov: &[f32], // [d × D], row-major
    d: usize,
    dd: usize,
) -> Vec<f32> {
    // M = C C^T [d × d] (symmetric positive semi-definite)
    let mut m = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = 0.0f64;
            for k in 0..dd {
                sum += cross_cov[i * dd + k] as f64 * cross_cov[j * dd + k] as f64;
            }
            m[i * d + j] = sum;
        }
    }

    let mut eigvals = vec![0.0f64; d];
    let mut eigvecs = vec![0.0f64; d * d];
    jacobi_eigen_symmetric(&mut eigvals, &mut eigvecs, &m, d, 1e-20, 100);

    // V^T = Σ^{-1} U^T C [d × D]
    let mut vt = vec![0.0f64; d * dd];
    for i in 0..d {
        let sigma = eigvals[i].sqrt();
        let inv_sigma = if sigma > 1e-10 { 1.0 / sigma } else { 0.0 };
        for k in 0..dd {
            let mut sum = 0.0f64;
            for j in 0..d {
                sum += eigvecs[j * d + i] * cross_cov[j * dd + k] as f64;
            }
            vt[i * dd + k] = inv_sigma * sum;
        }
    }

    // P = U V^T [d × D]
    let mut p = vec![0.0f32; d * dd];
    for i in 0..d {
        for k in 0..dd {
            let mut sum = 0.0f64;
            for j in 0..d {
                sum += eigvecs[i * d + j] * vt[j * dd + k];
            }
            p[i * dd + k] = sum as f32;
        }
    }

    p
}

/// Build a deterministic random matrix [rows × cols] (row-major, Xavier init).
fn random_matrix_f32(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let scale = (2.0 / (rows + cols) as f32).sqrt();
    let mut m = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        m.push(rng.normal() * scale);
    }
    m
}

/// Train a linear projection P [d × D] (row-major) via SGD on KL divergence.
///
/// Minimizes Σ_i KL(softmax(target_logits_i), softmax(draft_lm_head @ P @ h_target_i))
/// with `draft_lm_head` fixed (the draft model's output projection is not trained).
///
/// Returns the trained P [d × D].
///
/// Gradient derivation (see Issue 378 acceptance criterion 2):
///   projected_logits[j] = Σ_a Σ_b lm_head[j*d+a] * P[a*D+b] * h_target[b]
///   dKL/dlogit_j = y_p_j - y_t_j   (standard softmax-KL result)
///   dKL/dP[a,b]  = h_target[b] * Σ_j (y_p_j - y_t_j) * lm_head[j*d+a]
/// i.e. an outer product of h_target with a draft-dim gradient vector.
struct SgdConfig<'a> {
    h_targets_train: &'a [Vec<f32>],
    target_logits_train: &'a [Vec<f32>],
    draft_lm_head: &'a [f32], // [vocab × d], row-major
    vocab: usize,
    d: usize,  // draft_n_embd (output dim of P)
    dd: usize, // target_n_embd (input dim of P)
    epochs: usize,
    lr: f32,
    init_p: Option<Vec<f32>>, // optional warm-start (e.g. from Procrustes)
    seed: u64,
}

// Index-based loops are required here because we write to indexed elements of
// mutable scratch buffers (h_proj, logits, grad_logits, grad_h_proj, P) —
// `.iter().enumerate()` does not support in-place mutation patterns cleanly.
#[allow(clippy::needless_range_loop)]
fn train_sgd_projection(cfg: SgdConfig<'_>) -> Vec<f32> {
    // Initialize P (warm-start from Procrustes if provided, else Xavier random)
    let mut p = cfg.init_p.unwrap_or_else(|| random_matrix_f32(cfg.d, cfg.dd, cfg.seed));

    let n = cfg.h_targets_train.len();
    let mut epoch_kl = 0.0f32;

    for epoch in 0..cfg.epochs {
        let mut total_kl = 0.0f32;

        for i in 0..n {
            let h_t = &cfg.h_targets_train[i];
            let target_logits = &cfg.target_logits_train[i];
            let y_t = softmax(target_logits);

            // Forward: projected_logits[j] = Σ_a Σ_b lm_head[j*d+a] * P[a*D+b] * h_t[b]
            // Step 1: h_proj[a] = Σ_b P[a*D+b] * h_t[b]
            let mut h_proj = vec![0.0f32; cfg.d];
            for a in 0..cfg.d {
                let mut sum = 0.0f32;
                for b in 0..cfg.dd {
                    sum += p[a * cfg.dd + b] * h_t[b];
                }
                h_proj[a] = sum;
            }

            // Step 2: logits[j] = Σ_a lm_head[j*d+a] * h_proj[a]
            let mut logits = vec![0.0f32; cfg.vocab];
            for j in 0..cfg.vocab {
                let mut sum = 0.0f32;
                for a in 0..cfg.d {
                    sum += cfg.draft_lm_head[j * cfg.d + a] * h_proj[a];
                }
                logits[j] = sum;
            }

            let y_p = softmax(&logits);

            // KL for monitoring
            let kl = kl_divergence(&y_t, &y_p);
            total_kl += kl;

            // Backward: grad_logits[j] = y_p[j] - y_t[j]
            let mut grad_logits = vec![0.0f32; cfg.vocab];
            for j in 0..cfg.vocab {
                grad_logits[j] = y_p[j] - y_t[j];
            }

            // grad_h_proj[a] = Σ_j grad_logits[j] * lm_head[j*d+a]
            let mut grad_h_proj = vec![0.0f32; cfg.d];
            for a in 0..cfg.d {
                let mut sum = 0.0f32;
                for j in 0..cfg.vocab {
                    sum += grad_logits[j] * cfg.draft_lm_head[j * cfg.d + a];
                }
                grad_h_proj[a] = sum;
            }

            // grad_P[a,b] = grad_h_proj[a] * h_t[b]  (outer product)
            // Update: P -= lr * grad_P
            for a in 0..cfg.d {
                for b in 0..cfg.dd {
                    p[a * cfg.dd + b] -= cfg.lr * grad_h_proj[a] * h_t[b];
                }
            }
        }

        epoch_kl = total_kl / n as f32;
        if epoch < 5 || epoch % 50 == 0 || epoch == cfg.epochs - 1 {
            println!("    [SGD] epoch {:>4}/{}: train KL = {:.6}", epoch + 1, cfg.epochs, epoch_kl);
        }
    }

    println!("    [SGD] final train KL = {:.6} (lr={}, epochs={})", epoch_kl, cfg.lr, cfg.epochs);
    p
}

// ── Benchmark variants ──────────────────────────────────────────────────────

struct ProjVariant {
    name: &'static str,
    weights: Option<Vec<f32>>,
}

struct VariantResult {
    name: &'static str,
    kl_div: f32,
    cosine_sim: f32,
    #[allow(dead_code)]
    norm_ratio: f32,
    #[allow(dead_code)]
    logits_finite: bool,
}

// ── Main benchmark ──────────────────────────────────────────────────────────

#[test]
fn bench_378_cross_dim_procrustes() {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Issue 378 — Cross-Dim MTP Projection: Procrustes Alignment");
    println!("  §3.5 Path 2: closed-form SVD projection (modelless)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Target: Config::micro() as-is (n_embd=16, n_head=4, head_dim=4)
    let mut target_config = Config::micro();
    target_config.hybrid_pattern = HybridPattern::Uniform;
    target_config.mtp_activation_threshold = 1;

    // Draft: half the embedding dim (n_embd=8, n_head=2, head_dim=4)
    let mut draft_config = Config::micro();
    draft_config.n_embd = 8;
    draft_config.n_head = 2;
    draft_config.n_kv_head = 2;
    draft_config.mlp_hidden = 32; // 4 × n_embd
    draft_config.hybrid_pattern = HybridPattern::Uniform;
    draft_config.mtp_activation_threshold = 1;

    let target_n_embd = target_config.n_embd; // 16
    let draft_n_embd = draft_config.n_embd; // 8
    let vocab = target_config.vocab_size; // 27

    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let mut rng2 = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut rng2);

    // ── Step 1: Collect paired samples from both models ─────────────────
    let n_train = 20usize;
    let n_test = 10usize;
    let n_total = n_train + n_test;

    let mut h_targets: Vec<Vec<f32>> = Vec::with_capacity(n_total);
    let mut h_drafts: Vec<Vec<f32>> = Vec::with_capacity(n_total);
    let mut target_logits_all: Vec<Vec<f32>> = Vec::with_capacity(n_total);

    let mut t_ctx = ForwardContext::new(&target_config);
    let mut d_ctx = ForwardContext::new(&draft_config);

    for i in 0..n_total {
        let token = i % (vocab - 1);
        let pos = 0usize;

        let mut t_cache = MultiLayerKVCache::new(&target_config);
        let t_logits = forward(&mut t_ctx, &target_weights, &mut t_cache, token, pos, &target_config).to_vec();
        let h_t = t_ctx.hidden_state[..target_n_embd].to_vec();

        let mut d_cache = MultiLayerKVCache::new(&draft_config);
        let _d_logits = forward(&mut d_ctx, &draft_weights, &mut d_cache, token, pos, &draft_config);
        let h_d = d_ctx.hidden_state[..draft_n_embd].to_vec();

        target_logits_all.push(t_logits);
        h_targets.push(h_t);
        h_drafts.push(h_d);
    }

    println!("  Target: n_embd={}, Draft: n_embd={}, vocab={}", target_n_embd, draft_n_embd, vocab);
    println!("  Samples: {} train + {} test = {}", n_train, n_test, n_total);
    println!();

    // ── Step 2: Compute Procrustes from training samples ────────────────
    // Cross-covariance C [d × D] = Σ_i h_draft_i ⊗ h_target_i
    let mut cross_cov = vec![0.0f32; draft_n_embd * target_n_embd];
    for i in 0..n_train {
        let h_t = &h_targets[i];
        let h_d = &h_drafts[i];
        for j in 0..draft_n_embd {
            for k in 0..target_n_embd {
                cross_cov[j * target_n_embd + k] += h_d[j] * h_t[k];
            }
        }
    }

    let procrustes_p = compute_procrustes(&cross_cov, draft_n_embd, target_n_embd);

    // ── Step 2b: Train SGD projection on KL divergence (Issue 378 task 2) ──
    println!("── SGD Projection Training (Issue 378 acceptance criterion 2) ──");
    println!("  Training on {} paired samples, {} epochs, lr=0.1", n_train, 500);
    println!();
    let draft_lm_head_ref = &draft_weights.lm_head;
    let sgd_p = train_sgd_projection(SgdConfig {
        h_targets_train: &h_targets[..n_train],
        target_logits_train: &target_logits_all[..n_train],
        draft_lm_head: draft_lm_head_ref,
        vocab,
        d: draft_n_embd,
        dd: target_n_embd,
        epochs: 500,
        lr: 0.1,
        init_p: None, // cold start (random init)
        seed: 12345,
    });
    println!();

    // ── Step 2c: SGD warm-started from Procrustes (refinement pass) ─────
    println!("── SGD Warm-Start from Procrustes (refinement) ──");
    println!("  {} epochs, lr=0.05 (smaller for refinement)", 300);
    println!();
    let sgd_warm_p = train_sgd_projection(SgdConfig {
        h_targets_train: &h_targets[..n_train],
        target_logits_train: &target_logits_all[..n_train],
        draft_lm_head: draft_lm_head_ref,
        vocab,
        d: draft_n_embd,
        dd: target_n_embd,
        epochs: 300,
        lr: 0.05, // smaller lr for refinement
        init_p: Some(procrustes_p.clone()),
        seed: 12345,
    });
    println!();

    // ── Step 3: Build projection variants ───────────────────────────────
    let variants: Vec<ProjVariant> = vec![
        ProjVariant {
            name: "None (truncate/pad)",
            weights: None,
        },
        ProjVariant {
            name: "Procrustes (SVD)",
            weights: Some(procrustes_p.clone()),
        },
        ProjVariant {
            name: "SGD (cold start)",
            weights: Some(sgd_p.clone()),
        },
        ProjVariant {
            name: "SGD (warm Procrustes)",
            weights: Some(sgd_warm_p.clone()),
        },
        ProjVariant {
            name: "Random (control)",
            weights: Some(random_matrix_f32(draft_n_embd, target_n_embd, 999)),
        },
    ];

    // ── Step 4: Evaluate on held-out test samples ───────────────────────
    // For each variant: project h_target → h_projected, apply draft lm_head,
    // compute KL(target_logits, projected_logits).
    let draft_lm_head = &draft_weights.lm_head;

    println!("┌──────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────┐");
    println!("│ Projection                   │ KL div       │ Cosine sim   │ Norm ratio   │ G2 finite│");
    println!("│                              │ (vs target)  │ (vs h_draft) │ (proj/draft) │ (logits) │");
    println!("├──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────┤");

    let mut results: Vec<VariantResult> = Vec::new();
    let mut all_g2_pass = true;

    for v in &variants {
        let mut total_kl = 0.0f32;
        let mut total_cos = 0.0f32;
        let mut total_norm_ratio = 0.0f32;
        let mut all_finite_flag = true;

        for i in n_train..n_total {
            let h_t = &h_targets[i];
            let h_d = &h_drafts[i];
            let target_logits = &target_logits_all[i];

            let mut h_projected = vec![0.0f32; draft_n_embd];
            project_target_activation(
                &mut h_projected,
                h_t,
                v.weights.as_ref(),
                target_n_embd,
                draft_n_embd,
                target_config.mtp_activation_threshold,
            );

            let projected_logits =
                apply_lm_head(&h_projected, draft_lm_head, vocab, draft_n_embd);

            let kl = kl_divergence(&softmax(target_logits), &softmax(&projected_logits));
            let cos = cosine_similarity(h_d, &h_projected);
            let draft_norm = l2_norm(h_d);
            let proj_norm = l2_norm(&h_projected);
            let norm_ratio = if draft_norm > 1e-12 { proj_norm / draft_norm } else { 0.0 };
            let finite = all_finite(&projected_logits);

            total_kl += kl;
            total_cos += cos;
            total_norm_ratio += norm_ratio;
            if !finite {
                all_finite_flag = false;
            }
        }

        let n = n_test as f32;
        let avg_kl = total_kl / n;
        let avg_cos = total_cos / n;
        let avg_norm_ratio = total_norm_ratio / n;

        if !all_finite_flag {
            all_g2_pass = false;
        }

        results.push(VariantResult {
            name: v.name,
            kl_div: avg_kl,
            cosine_sim: avg_cos,
            norm_ratio: avg_norm_ratio,
            logits_finite: all_finite_flag,
        });

        println!(
            "│ {:<28} │ {:>12.6} │ {:>12.6} │ {:>12.6} │ {:>8} │",
            v.name,
            avg_kl,
            avg_cos,
            avg_norm_ratio,
            if all_finite_flag { "✅ PASS" } else { "❌ FAIL" }
        );
    }

    println!("└──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────┘");
    println!();

    // ── Step 5: Also measure draft's own KL (baseline, no projection) ───
    let mut draft_own_kl = 0.0f32;
    for i in n_train..n_total {
        let h_d = &h_drafts[i];
        let target_logits = &target_logits_all[i];
        let draft_logits = apply_lm_head(h_d, draft_lm_head, vocab, draft_n_embd);
        let kl = kl_divergence(&softmax(target_logits), &softmax(&draft_logits));
        draft_own_kl += kl;
    }
    draft_own_kl /= n_test as f32;
    println!("  Draft's own KL (no projection, draft's own hidden): {:.6}", draft_own_kl);
    println!();

    // ── Step 6: GOAT gate ───────────────────────────────────────────────
    let g1_threshold = 0.1f32;

    let procrustes_result = results.iter().find(|r| r.name.starts_with("Procrustes")).unwrap();
    let truncate_result = results.iter().find(|r| r.name.starts_with("None")).unwrap();
    let sgd_cold_result = results.iter().find(|r| r.name.starts_with("SGD (cold")).unwrap();
    let sgd_warm_result = results.iter().find(|r| r.name.starts_with("SGD (warm")).unwrap();
    let random_result = results.iter().find(|r| r.name.starts_with("Random")).unwrap();

    // Pick the best of all trained variants for the GOAT gate
    let best_trained = [procrustes_result, sgd_cold_result, sgd_warm_result]
        .iter()
        .copied()
        .min_by(|a, b| a.kl_div.partial_cmp(&b.kl_div).unwrap())
        .unwrap();

    let g1_pass = best_trained.kl_div <= g1_threshold;
    let g1_beats_truncate = best_trained.kl_div < truncate_result.kl_div;
    let g1_beats_random = best_trained.kl_div < random_result.kl_div;
    let g3_pass = best_trained.cosine_sim >= 0.5;

    println!("── GOAT Gate (best trained variant: {}) ─────────", best_trained.name);
    println!(
        "  G1 (quality): best trained KL ≤ {:.2}? → {:.6} {}",
        g1_threshold,
        best_trained.kl_div,
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G1b (vs truncate): best trained KL < truncate/pad KL? → {:.6} < {:.6} {}",
        best_trained.kl_div,
        truncate_result.kl_div,
        if g1_beats_truncate { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G1c (vs random): best trained KL < random KL? → {:.6} < {:.6} {}",
        best_trained.kl_div,
        random_result.kl_div,
        if g1_beats_random { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G2 (no-regression): all logits finite? → {}",
        if all_g2_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G3 (info preservation): cosine sim ≥ 0.5? → {:.6} {}",
        best_trained.cosine_sim,
        if g3_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!();
    println!("  Per-variant KL comparison:");
    println!("    Procrustes (SVD):   {:.6}", procrustes_result.kl_div);
    println!("    SGD (cold start):   {:.6}", sgd_cold_result.kl_div);
    println!("    SGD (warm Procr.):  {:.6}", sgd_warm_result.kl_div);
    println!("    truncate/pad:       {:.6}", truncate_result.kl_div);
    println!("    random control:     {:.6}", random_result.kl_div);
    println!();

    // ── Step 7: Verdict ─────────────────────────────────────────────────
    println!("── Verdict ───────────────────────────────────────────────────");
    if g1_pass && g1_beats_truncate {
        println!("  A trained/Procrustes projection PASSES the GOAT gate.");
        println!("  Cross-dim MTP projection is closed (KL ≤ {:.2}).", g1_threshold);
        println!();
        println!("  → Issue 378 acceptance criterion 3 (GOAT gate): MET by {}", best_trained.name);
    } else if g1_beats_truncate {
        println!("  Best trained projection ({}) BEATS truncate/pad but doesn't meet KL ≤ {:.2}.",
            best_trained.name, g1_threshold);
        println!("  Best trained KL = {:.6} vs truncate/pad KL = {:.6}",
            best_trained.kl_div, truncate_result.kl_div);
        println!("  vs random control KL = {:.6}", random_result.kl_div);
        println!();
        println!("  Root cause (rank deficiency): draft_lm_head [{v}×{d}] @ P [{d}×{D}]",
            v = vocab, d = draft_n_embd, D = target_n_embd);
        println!("  has rank ≤ {d}, but target_lm_head [{v}×{D}] has rank up to {D}.",
            d = draft_n_embd, v = vocab, D = target_n_embd);
        println!("  KL=0 is mathematically impossible when target rank > draft rank.");
        println!("  This is fundamental for random-init models with dim mismatch.");
        println!("  Same-family trained models (e.g. Gemma-2-2B + pruned Gemma)");
        println!("  may have correlated hidden spaces that close the gap further.");
        println!();
        println!("  → Issue 378 acceptance criterion 2 (SGD training): IMPLEMENTED");
        println!("  → Issue 378 acceptance criterion 3 (GOAT gate): ❌ FAIL on random models");
        println!("  → Needs real trained model pairs to evaluate properly");
    } else {
        println!("  Best trained projection ({}) does NOT beat truncate/pad.",
            best_trained.name);
        println!();
        println!("  → Issue 378 acceptance criterion 3 (GOAT gate): ❌ FAIL");
    }
    println!();

    // Assert basic sanity (G2 must always pass — no NaN/Inf)
    assert!(all_g2_pass, "G2 FAIL: some projections produced non-finite logits");

    println!("═══════════════════════════════════════════════════════════════");
}
