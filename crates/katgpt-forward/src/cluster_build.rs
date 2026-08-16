//! Load-time construction of the clustered-LM-head artifacts (Plan 574).
//!
//! [`clustered_lm_head`](crate::forward::clustered_lm_head) has shipped wired
//! into both forward call sites since Plan 117, but never ran in production
//! because nothing ever produced its two inputs — `mtp_cluster_classifier` and
//! `mtp_cluster_map` were always `None`. This module is the producer.
//!
//! # Why the centroid is the right classifier
//!
//! `logit[t] = dot(hidden, lm_head[t])`. Defining cluster `c`'s classifier row
//! as the centroid of its members' LM-head rows gives
//!
//! ```text
//! dot(hidden, centroid_c) = mean(logit[t] for t in c)
//! ```
//!
//! so the stage-1 score *is* the cluster's mean logit — a principled proxy for
//! "does the argmax live in here". K-means over the LM-head rows minimises
//! within-cluster variance, which is exactly what tightens that proxy.
//!
//! Both steps are deterministic functions of weights the model already ships:
//! **no gradients, no training**. This is a modelless primitive.
//!
//! # Why `lm_head`, not `wte`
//!
//! The stub this replaces took `wte`. The matmul being pruned is the LM head,
//! so `lm_head` is the matrix whose geometry matters. With tied embeddings the
//! two coincide; with untied embeddings `wte` is simply the wrong matrix.
//!
//! # Why the rows are projected first
//!
//! Naive Lloyd assignment costs `O(vocab × k × n_embd)` per iteration — at Qwen
//! scale (`vocab=152k, k=2048, n_embd=5120`) that is ~1.6e12 FLOPs *per
//! iteration*. A deterministic Johnson–Lindenstrauss projection to
//! [`PROJ_DIM`] preserves relative distances well enough for clustering while
//! cutting assignment to ~2e10. Centroids for the classifier are then computed
//! in the **full** space, so the emitted artifact is exact.

use katgpt_core::simd::simd_dot_f32;

/// Dimension the LM-head rows are projected into before k-means.
///
/// 64 keeps Johnson–Lindenstrauss distortion low for the cluster counts used
/// here while making assignment ~80× cheaper than clustering in full `n_embd`.
const PROJ_DIM: usize = 64;

/// Lloyd iterations. Cluster assignment converges fast in projected space;
/// beyond ~10 the map stops changing while cost keeps growing linearly.
const KMEANS_ITERS: usize = 10;

/// Fixed seed for the projection. Pinned so the emitted cluster map is
/// reproducible across runs and machines — a rebuilt map that differed run to
/// run would make the argmax-recall gate unfalsifiable.
const PROJ_SEED: u64 = 0x5EED_C105_7E12_0001;

/// splitmix64 finalizer — deterministic, well-distributed bit mixing.
#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build the `[proj_dim, n_embd]` ±1 Johnson–Lindenstrauss projection.
///
/// Materialised once (≈1.3 MB at Qwen scale) rather than hashing per element,
/// so the projection itself runs as SIMD dot products.
fn build_projection(n_embd: usize, proj_dim: usize) -> Vec<f32> {
    let mut proj = vec![0.0f32; proj_dim * n_embd];
    for (idx, slot) in proj.iter_mut().enumerate() {
        let bits = splitmix64(PROJ_SEED ^ (idx as u64));
        *slot = match bits & 1 {
            0 => -1.0,
            _ => 1.0,
        };
    }
    proj
}

/// Project every row of `rows` into `proj_dim` dimensions.
fn project_rows(rows: &[f32], count: usize, n_embd: usize, proj_dim: usize) -> Vec<f32> {
    let proj = build_projection(n_embd, proj_dim);
    let mut out = vec![0.0f32; count * proj_dim];
    for t in 0..count {
        let row = &rows[t * n_embd..(t + 1) * n_embd];
        for j in 0..proj_dim {
            let basis = &proj[j * n_embd..(j + 1) * n_embd];
            out[t * proj_dim + j] = simd_dot_f32(row, basis, n_embd);
        }
    }
    out
}

/// Squared L2 distance between two equal-length slices.
#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
}

/// Create a round-robin cluster assignment for tokens.
///
/// Token `i` is assigned to cluster `i / cluster_size`. Deterministic, no
/// training needed — the simple baseline that
/// [`cluster_map_from_embeddings`] must beat on argmax recall to be worth
/// shipping (Plan 574 gate G2).
pub fn cluster_map_round_robin(vocab_size: usize, cluster_size: usize) -> Vec<Vec<usize>> {
    if cluster_size == 0 {
        return Vec::new();
    }
    let num_clusters = vocab_size.div_ceil(cluster_size);
    let mut map: Vec<Vec<usize>> = (0..num_clusters)
        .map(|_| Vec::with_capacity(cluster_size))
        .collect();
    for token_id in 0..vocab_size {
        map[token_id / cluster_size].push(token_id);
    }
    map
}

/// Group tokens by LM-head-row similarity (k-means in projected space).
///
/// `lm_head` is `[vocab_size, n_embd]` row-major. Returns one `Vec<usize>` of
/// token IDs per non-empty cluster; every token in `0..vocab_size` appears in
/// exactly one cluster. Deterministic for a given input — see [`PROJ_SEED`].
///
/// Cluster *count* matches [`cluster_map_round_robin`]'s convention
/// (`vocab_size.div_ceil(cluster_size)`) so the two are directly comparable at
/// equal `topk`. Cluster *sizes* will not be uniform — that is the point.
///
/// Empty clusters are dropped, so the result may contain fewer clusters than
/// requested. Build the classifier from the returned map (not from the
/// requested count) so the two stay consistent —
/// [`cluster_classifier_from_map`] does this by construction.
pub fn cluster_map_from_embeddings(
    lm_head: &[f32],
    vocab_size: usize,
    n_embd: usize,
    cluster_size: usize,
) -> Vec<Vec<usize>> {
    // Degenerate shapes fall back to the baseline rather than erroring: the
    // caller gets a usable map, and round-robin is exactly what k-means would
    // reduce to when there is no geometry to exploit.
    if cluster_size == 0 || vocab_size == 0 || n_embd == 0 {
        return cluster_map_round_robin(vocab_size, cluster_size);
    }
    if lm_head.len() < vocab_size * n_embd {
        return cluster_map_round_robin(vocab_size, cluster_size);
    }

    let k = vocab_size.div_ceil(cluster_size);
    // One cluster per token (or fewer tokens than clusters) leaves nothing to
    // group; round-robin already yields the optimal partition.
    if k <= 1 || k >= vocab_size {
        return cluster_map_round_robin(vocab_size, cluster_size);
    }

    let proj_dim = PROJ_DIM.min(n_embd);
    let data = project_rows(lm_head, vocab_size, n_embd, proj_dim);

    // Strided init: evenly spaced rows spread the initial centers across the
    // vocabulary. Taking the *first* k rows (as a naive init does) clusters
    // them all in one corner of a sorted vocabulary and converges poorly.
    let stride = vocab_size / k;
    let mut centers = vec![0.0f32; k * proj_dim];
    for c in 0..k {
        let src = (c * stride).min(vocab_size - 1);
        centers[c * proj_dim..(c + 1) * proj_dim]
            .copy_from_slice(&data[src * proj_dim..(src + 1) * proj_dim]);
    }

    let mut labels = vec![0usize; vocab_size];
    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0f32; k * proj_dim];

    for _ in 0..KMEANS_ITERS {
        // Assignment
        for t in 0..vocab_size {
            let point = &data[t * proj_dim..(t + 1) * proj_dim];
            let mut best = 0usize;
            let mut best_dist = f32::MAX;
            for c in 0..k {
                let dist = sq_dist(point, &centers[c * proj_dim..(c + 1) * proj_dim]);
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            labels[t] = best;
        }

        // Update
        sums.fill(0.0);
        counts.fill(0);
        for t in 0..vocab_size {
            let c = labels[t];
            counts[c] += 1;
            let point = &data[t * proj_dim..(t + 1) * proj_dim];
            let sum_row = &mut sums[c * proj_dim..(c + 1) * proj_dim];
            for (s, p) in sum_row.iter_mut().zip(point) {
                *s += p;
            }
        }
        for c in 0..k {
            // An empty cluster keeps its previous center; re-seeding it would
            // break determinism guarantees for no measurable recall gain.
            match counts[c] {
                0 => {}
                n => {
                    let inv = 1.0 / n as f32;
                    let center_row = &mut centers[c * proj_dim..(c + 1) * proj_dim];
                    let sum_row = &sums[c * proj_dim..(c + 1) * proj_dim];
                    for (slot, s) in center_row.iter_mut().zip(sum_row) {
                        *slot = s * inv;
                    }
                }
            }
        }
    }

    let mut map: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (token_id, &c) in labels.iter().enumerate() {
        map[c].push(token_id);
    }
    map.retain(|cluster| !cluster.is_empty());
    map
}

/// Build the stage-1 classifier as per-cluster centroids of the LM-head rows.
///
/// Returns `[cluster_map.len(), n_embd]` row-major, matching the layout
/// [`clustered_lm_head`](crate::forward::clustered_lm_head) indexes as
/// `c * n_embd`. Centroids are computed in the **full** embedding space (the
/// projection is a clustering device only), so the score
/// `dot(hidden, centroid_c)` is exactly the cluster's mean logit.
#[allow(clippy::needless_range_loop)] // index-parallel over lm_head rows reads clearer here
pub fn cluster_classifier_from_map(
    lm_head: &[f32],
    cluster_map: &[Vec<usize>],
    n_embd: usize,
) -> Vec<f32> {
    let mut classifier = vec![0.0f32; cluster_map.len() * n_embd];
    for (c, tokens) in cluster_map.iter().enumerate() {
        if tokens.is_empty() {
            continue;
        }
        let row = &mut classifier[c * n_embd..(c + 1) * n_embd];
        for &t in tokens {
            let off = t * n_embd;
            if off + n_embd > lm_head.len() {
                continue;
            }
            for (slot, w) in row.iter_mut().zip(&lm_head[off..off + n_embd]) {
                *slot += w;
            }
        }
        let inv = 1.0 / tokens.len() as f32;
        for slot in row.iter_mut() {
            *slot *= inv;
        }
    }
    classifier
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::{clustered_lm_head, standard_lm_head};

    /// Two well-separated groups: even token IDs at `+1`, odd at `-1`.
    ///
    /// Round-robin partitions by *index*, so every one of its clusters is a
    /// 50/50 mix of the two groups. K-means must recover the actual geometry.
    fn two_group_lm_head(vocab: usize, n_embd: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; vocab * n_embd];
        for t in 0..vocab {
            let sign = match t % 2 {
                0 => 1.0,
                _ => -1.0,
            };
            for j in 0..n_embd {
                w[t * n_embd + j] = sign * (1.0 + (j as f32) * 0.01);
            }
        }
        w
    }

    #[test]
    fn every_token_lands_in_exactly_one_cluster() {
        let (vocab, n_embd) = (128, 16);
        let w = two_group_lm_head(vocab, n_embd);
        let map = cluster_map_from_embeddings(&w, vocab, n_embd, 16);

        let mut seen = vec![0usize; vocab];
        for cluster in &map {
            for &t in cluster {
                seen[t] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "each token must appear exactly once; got {seen:?}"
        );
        assert!(
            map.iter().all(|c| !c.is_empty()),
            "empty clusters must be dropped"
        );
    }

    #[test]
    fn map_is_deterministic_across_builds() {
        let (vocab, n_embd) = (96, 16);
        let w = two_group_lm_head(vocab, n_embd);
        let a = cluster_map_from_embeddings(&w, vocab, n_embd, 12);
        let b = cluster_map_from_embeddings(&w, vocab, n_embd, 12);
        assert_eq!(a, b, "PROJ_SEED is pinned — rebuilds must be identical");
    }

    /// The quality claim: k-means separates the two groups, round-robin cannot.
    #[test]
    fn kmeans_separates_groups_that_round_robin_mixes() {
        let (vocab, n_embd) = (128, 16);
        let w = two_group_lm_head(vocab, n_embd);

        let purity = |map: &[Vec<usize>]| -> f64 {
            let mut pure = 0usize;
            for cluster in map {
                let evens = cluster.iter().filter(|t| *t % 2 == 0).count();
                pure += evens.max(cluster.len() - evens);
            }
            pure as f64 / vocab as f64
        };

        let km = purity(&cluster_map_from_embeddings(&w, vocab, n_embd, 16));
        let rr = purity(&cluster_map_round_robin(vocab, 16));
        assert!(
            km > rr,
            "k-means purity {km} must beat round-robin {rr} on grouped data"
        );
        assert!(km > 0.99, "k-means should near-perfectly separate: {km}");
    }

    #[test]
    fn classifier_row_is_cluster_centroid() {
        let (vocab, n_embd) = (8, 4);
        let mut w = vec![0.0f32; vocab * n_embd];
        for t in 0..vocab {
            for j in 0..n_embd {
                w[t * n_embd + j] = t as f32;
            }
        }
        // One cluster holding tokens 0..4 — centroid is mean(0,1,2,3) = 1.5.
        let map = vec![vec![0usize, 1, 2, 3]];
        let classifier = cluster_classifier_from_map(&w, &map, n_embd);
        for j in 0..n_embd {
            assert!(
                (classifier[j] - 1.5).abs() < 1e-6,
                "centroid[{j}] = {}, want 1.5",
                classifier[j]
            );
        }
    }

    /// **Gate G1**: with `topk >= num_clusters` nothing is pruned, so the
    /// clustered head must reproduce the standard head bit-for-bit.
    #[test]
    fn g1_bit_identical_to_standard_head_when_nothing_is_pruned() {
        let (vocab, n_embd) = (64, 16);
        let w = two_group_lm_head(vocab, n_embd);
        let hidden: Vec<f32> = (0..n_embd).map(|j| 0.1 * (j as f32) - 0.5).collect();

        let map = cluster_map_from_embeddings(&w, vocab, n_embd, 8);
        let classifier = cluster_classifier_from_map(&w, &map, n_embd);

        let mut want = vec![0.0f32; vocab];
        standard_lm_head(&mut want, &hidden, &w, vocab, n_embd);

        let mut got = vec![0.0f32; vocab];
        let mut scores = vec![0.0f32; map.len()];
        let (mut idx_buf, mut out_buf) = (Vec::new(), Vec::new());
        clustered_lm_head(
            &mut got,
            &hidden,
            &w,
            &classifier,
            &map,
            vocab,
            n_embd,
            map.len(), // topk == num_clusters => no pruning
            &mut scores,
            &mut idx_buf,
            &mut out_buf,
        );

        assert_eq!(got, want, "no-pruning path must be bit-identical");
    }

    #[test]
    fn degenerate_shapes_fall_back_to_round_robin() {
        assert_eq!(cluster_map_from_embeddings(&[], 0, 16, 4), Vec::<Vec<usize>>::new());
        assert!(cluster_map_from_embeddings(&[], 10, 16, 0).is_empty());
        // Short weight buffer must not panic — falls back.
        let short = vec![0.0f32; 4];
        assert_eq!(
            cluster_map_from_embeddings(&short, 100, 16, 25),
            cluster_map_round_robin(100, 25)
        );
    }
}
