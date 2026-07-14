#![allow(clippy::needless_range_loop)]
//! Diagnostic — Plan 230 EMBED_DIM sweep (Issue 139 probe).
//!
//! Not a gate. Measures NN preservation across EMBED_DIM ∈ {8, 16, 24, 32, 48, 64}
//! to find the minimum m that satisfies the documented G1 threshold (≥ 90%).
//!
//! Run: cargo test --features shard_embedding --test diag_230_embed_dim_sweep -- --nocapture

#[cfg(feature = "shard_embedding")]
mod diag {
    use katgpt_core::shard_embedding::STYLE_DIM;

    fn make_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut rng = fastrand::Rng::with_seed(seed);
        move || rng.f32() * 2.0 - 1.0
    }

    /// Gram-Schmidt orthonormalization of `m` rows in R^n. Returns row-major m×n.
    fn gram_schmidt(m: usize, n: usize, rng: &mut impl FnMut() -> f32) -> Vec<Vec<f32>> {
        let mut rows: Vec<Vec<f32>> = (0..m).map(|_| (0..n).map(|_| rng()).collect()).collect();
        for i in 0..m {
            for k in 0..i {
                let dot: f32 = rows[i].iter().zip(rows[k].iter()).map(|(a, b)| a * b).sum();
                for j in 0..n {
                    rows[i][j] -= dot * rows[k][j];
                }
            }
            let norm_sq: f32 = rows[i].iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt();
            if norm > 1e-8 {
                let inv = 1.0 / norm;
                for x in rows[i].iter_mut() {
                    *x *= inv;
                }
            }
        }
        rows
    }

    /// Project a vector x ∈ R^n by an m×n orthonormal matrix → R^m.
    fn project(rows: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
        rows.iter().map(|r| r.iter().zip(x.iter()).map(|(a, b)| a * b).sum()).collect()
    }

    /// Top-k NN preservation rate: for each vector, compare true top-k NN set
    /// (by Euclidean distance in R^n) to projected top-k NN set (in R^m).
    /// Returns (top1_rate, top5_rate, top10_rate).
    fn nn_preservation(vectors: &[Vec<f32>], m: usize, _k_max: usize, rng_seed: u64) -> (f32, f32, f32) {
        let n = vectors.len();
        let mut rng = make_rng(rng_seed);
        let rows = gram_schmidt(m, STYLE_DIM, &mut rng);

        // Project all vectors once
        let projected: Vec<Vec<f32>> = vectors.iter().map(|v| project(&rows, v)).collect();

        let mut top1 = 0usize;
        let mut top5 = 0usize;
        let mut top10 = 0usize;
        for i in 0..n {
            // True NN distances
            let mut true_dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = vectors[i].iter().zip(vectors[j].iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (j, d)
                })
                .collect();
            true_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Projected NN distances
            let mut proj_dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = projected[i].iter().zip(projected[j].iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (j, d)
                })
                .collect();
            proj_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Compare top-k sets
            let true_top1: usize = true_dists[0].0;
            let true_top5: Vec<usize> = true_dists.iter().take(5).map(|(j, _)| *j).collect();
            let true_top10: Vec<usize> = true_dists.iter().take(10).map(|(j, _)| *j).collect();

            let proj_top1: usize = proj_dists[0].0;
            let proj_top5: Vec<usize> = proj_dists.iter().take(5).map(|(j, _)| *j).collect();
            let proj_top10: Vec<usize> = proj_dists.iter().take(10).map(|(j, _)| *j).collect();

            if proj_top1 == true_top1 { top1 += 1; }
            let overlap5 = proj_top5.iter().filter(|j| true_top5.contains(j)).count();
            let overlap10 = proj_top10.iter().filter(|j| true_top10.contains(j)).count();
            // top-k rate = overlap / k (so 5/5 = 100%)
            top5 += overlap5;
            top10 += overlap10;
        }

        let n_f = n as f32;
        (
            top1 as f32 / n_f,
            top5 as f32 / (n_f * 5.0),
            top10 as f32 / (n_f * 10.0),
        )
    }

    /// Cosine-similarity-based NN preservation (matches Plan 230's intended use case).
    /// Returns (top1_rate, top5_rate) using cosine similarity ranking.
    fn nn_preservation_cosine(vectors: &[Vec<f32>], m: usize, rng_seed: u64) -> (f32, f32) {
        let n = vectors.len();
        let mut rng = make_rng(rng_seed);
        let rows = gram_schmidt(m, STYLE_DIM, &mut rng);

        let projected: Vec<Vec<f32>> = vectors.iter().map(|v| project(&rows, v)).collect();

        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let ma: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if ma < 1e-8 || mb < 1e-8 { 0.0 } else { dot / (ma * mb) }
        };

        let mut top1 = 0usize;
        let mut top5 = 0usize;
        for i in 0..n {
            // True NN by cosine (highest cosine = nearest)
            let mut true_cos: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cos(&vectors[i], &vectors[j])))
                .collect();
            true_cos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut proj_cos: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cos(&projected[i], &projected[j])))
                .collect();
            proj_cos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let true_top1 = true_cos[0].0;
            let true_top5: Vec<usize> = true_cos.iter().take(5).map(|(j, _)| *j).collect();
            let proj_top1 = proj_cos[0].0;
            let proj_top5: Vec<usize> = proj_cos.iter().take(5).map(|(j, _)| *j).collect();

            if proj_top1 == true_top1 { top1 += 1; }
            top5 += proj_top5.iter().filter(|j| true_top5.contains(j)).count();
        }

        (top1 as f32 / n as f32, top5 as f32 / (n as f32 * 5.0))
    }

    #[test]
    fn diag_embed_dim_sweep_euclidean() {
        let n = 100;
        let mut rng = make_rng(123);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..STYLE_DIM).map(|_| rng()).collect())
            .collect();

        eprintln!("\n=== Plan 230 EMBED_DIM sweep (Euclidean NN, n={}, d_in={}) ===", n, STYLE_DIM);
        eprintln!("{:>6} | {:>10} | {:>10} | {:>10}", "m", "top1", "top5", "top10");
        eprintln!("{}", "-".repeat(46));
        // Average over 3 seeds for stability
        for &m in &[8usize, 16, 24, 32, 40, 48, 56, 64] {
            let mut t1 = 0.0f32;
            let mut t5 = 0.0f32;
            let mut t10 = 0.0f32;
            let seeds = 3usize;
            for s in 0..seeds {
                let (a, b, c) = nn_preservation(&vectors, m, 10, 42 + s as u64);
                t1 += a; t5 += b; t10 += c;
            }
            eprintln!(
                "{:>6} | {:>8.1}%  | {:>8.1}%  | {:>8.1}%",
                m,
                (t1 / seeds as f32) * 100.0,
                (t5 / seeds as f32) * 100.0,
                (t10 / seeds as f32) * 100.0,
            );
        }
        eprintln!();
    }

    #[test]
    fn diag_embed_dim_sweep_cosine() {
        let n = 100;
        let mut rng = make_rng(123);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..STYLE_DIM).map(|_| rng()).collect())
            .collect();

        eprintln!("\n=== Plan 230 EMBED_DIM sweep (Cosine NN, n={}, d_in={}) ===", n, STYLE_DIM);
        eprintln!("{:>6} | {:>10} | {:>10}", "m", "top1", "top5");
        eprintln!("{}", "-".repeat(34));
        for &m in &[8usize, 16, 24, 32, 40, 48, 56, 64] {
            let mut t1 = 0.0f32;
            let mut t5 = 0.0f32;
            let seeds = 3usize;
            for s in 0..seeds {
                let (a, b) = nn_preservation_cosine(&vectors, m, 42 + s as u64);
                t1 += a; t5 += b;
            }
            eprintln!(
                "{:>6} | {:>8.1}%  | {:>8.1}%",
                m,
                (t1 / seeds as f32) * 100.0,
                (t5 / seeds as f32) * 100.0,
            );
        }
        eprintln!();
    }

    /// Larger-N sweep — what happens at n=500 (more realistic shard count)?
    #[test]
    fn diag_embed_dim_sweep_n500() {
        let n = 500;
        let mut rng = make_rng(456);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..STYLE_DIM).map(|_| rng()).collect())
            .collect();

        eprintln!("\n=== Plan 230 EMBED_DIM sweep (Euclidean NN, n={}, d_in={}) ===", n, STYLE_DIM);
        eprintln!("{:>6} | {:>10} | {:>10}", "m", "top1", "top5");
        eprintln!("{}", "-".repeat(34));
        for &m in &[8usize, 16, 24, 32, 48, 64] {
            let (t1, t5, _) = nn_preservation(&vectors, m, 5, 42);
            eprintln!(
                "{:>6} | {:>8.1}%  | {:>8.1}%",
                m,
                t1 * 100.0,
                t5 * 100.0,
            );
        }
        eprintln!();
    }

    // ------------------------------------------------------------------
    // PCA probe (Option B from Issue 139).
    //
    // Tests whether PCA on *structured* (low-rank + noise) data can satisfy G1
    // at m=8 where random JL cannot. This is a synthetic upper bound on what
    // PCA could achieve on real style_weights — if it fails on ideally-low-
    // rank synthetic data, it will certainly fail on real data.
    // ------------------------------------------------------------------

    /// Build a PCA-style projection: top-m right singular vectors of the data
    /// matrix. Naive power iteration on the m×n covariance — correct enough
    /// for a diagnostic, not production code.
    fn pca_projection(data: &[Vec<f32>], m: usize) -> Vec<Vec<f32>> {
        let n = data.len();
        let d = data[0].len();
        // Covariance: d×d. Center first.
        let mean: Vec<f32> = (0..d)
            .map(|j| data.iter().map(|v| v[j]).sum::<f32>() / n as f32)
            .collect();
        let mut cov = vec![vec![0.0f32; d]; d];
        for v in data {
            for i in 0..d {
                for j in 0..d {
                    cov[i][j] += (v[i] - mean[i]) * (v[j] - mean[j]);
                }
            }
        }
        for i in 0..d {
            for j in 0..d {
                cov[i][j] /= n as f32;
            }
        }
        // Power iteration to extract top-m eigenvectors.
        // Each eigenvector initialized to random then iterated.
        let mut eigenvectors: Vec<Vec<f32>> = Vec::with_capacity(m);
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..m {
            let mut v: Vec<f32> = (0..d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            // Normalize
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() { *x /= norm; }
            // 30 power iterations
            for _ in 0..30 {
                let mut next = vec![0.0f32; d];
                for i in 0..d {
                    for j in 0..d {
                        next[i] += cov[i][j] * v[j];
                    }
                }
                // Deflate against already-found eigenvectors
                for ev in &eigenvectors {
                    let dot: f32 = next.iter().zip(ev.iter()).map(|(a, b)| a * b).sum();
                    for i in 0..d {
                        next[i] -= dot * ev[i];
                    }
                }
                let norm: f32 = next.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 { break; }
                for x in next.iter_mut() { *x /= norm; }
                v = next;
            }
            eigenvectors.push(v);
        }
        eigenvectors
    }

    /// Generate synthetic low-rank style_weights: k-dim subspace + Gaussian
    /// noise. This is the best-case for PCA — if it fails here, PCA is
    /// hopeless for Plan 230.
    fn synth_low_rank(n: usize, d: usize, k: usize, noise_std: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = fastrand::Rng::with_seed(seed);
        // Random k-dim subspace basis: d×k matrix
        let basis: Vec<Vec<f32>> = (0..d)
            .map(|_| (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect())
            .collect();
        // Orthonormalize basis columns via Gram-Schmidt (on transposed)
        let mut basis_t: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; d]).collect();
        for j in 0..k {
            for i in 0..d {
                basis_t[j][i] = basis[i][j];
            }
        }
        for j in 0..k {
            for jp in 0..j {
                let dot: f32 = basis_t[j].iter().zip(basis_t[jp].iter()).map(|(a, b)| a * b).sum();
                for i in 0..d {
                    basis_t[j][i] -= dot * basis_t[jp][i];
                }
            }
            let norm: f32 = basis_t[j].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in basis_t[j].iter_mut() { *x /= norm; }
            }
        }
        // Generate n samples: random coefficients in subspace + noise
        (0..n)
            .map(|_| {
                let coefs: Vec<f32> = (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect();
                let mut v = vec![0.0f32; d];
                for j in 0..k {
                    for i in 0..d {
                        v[i] += coefs[j] * basis_t[j][i];
                    }
                }
                // Add noise
                for x in v.iter_mut() {
                    *x += (rng.f32() * 2.0 - 1.0) * noise_std;
                }
                v
            })
            .collect()
    }

    fn nn_preservation_pca(data: &[Vec<f32>], m: usize) -> (f32, f32) {
        let rows = pca_projection(data, m);
        let n = data.len();
        let projected: Vec<Vec<f32>> = data.iter().map(|v| project(&rows, v)).collect();

        let mut top1 = 0usize;
        let mut top5 = 0usize;
        for i in 0..n {
            let mut true_dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = data[i].iter().zip(data[j].iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (j, d)
                })
                .collect();
            true_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut proj_dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = projected[i].iter().zip(projected[j].iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (j, d)
                })
                .collect();
            proj_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let true_top1 = true_dists[0].0;
            let true_top5: Vec<usize> = true_dists.iter().take(5).map(|(j, _)| *j).collect();
            let proj_top1 = proj_dists[0].0;
            let proj_top5: Vec<usize> = proj_dists.iter().take(5).map(|(j, _)| *j).collect();

            if proj_top1 == true_top1 { top1 += 1; }
            top5 += proj_top5.iter().filter(|j| true_top5.contains(j)).count();
        }
        (top1 as f32 / n as f32, top5 as f32 / (n as f32 * 5.0))
    }

    #[test]
    fn diag_pca_probe_low_rank() {
        let n = 100;
        eprintln!("\n=== PCA probe (n={}, d_in={}, Option B upper bound) ===", n, STYLE_DIM);
        eprintln!("{:>10} | {:>8} | {:>10} | {:>10}", "k_true", "noise", "top1", "top5");
        eprintln!("{}", "-".repeat(46));
        // Try several true ranks + noise levels
        for &(k, noise) in &[(4, 0.0), (4, 0.1), (8, 0.0), (8, 0.1), (8, 0.3), (16, 0.0), (16, 0.1), (32, 0.0)] {
            let data = synth_low_rank(n, STYLE_DIM, k, noise, 999);
            for &m in &[8usize, 16] {
                let (t1, t5) = nn_preservation_pca(&data, m);
                eprintln!(
                    "{:>10} | {:>8.2} | m={:>2} {:>6.1}% | {:>8.1}%",
                    k, noise, m, t1 * 100.0, t5 * 100.0,
                );
            }
        }
        eprintln!();
    }
}
