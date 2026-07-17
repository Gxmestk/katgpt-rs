//! RandOpt diagnostics (extracted from mod.rs by Issue 177).
//!
//! Plan 121 — modelless diagnostics for bandit arm-selection:
//! solution_density, spectral_discordance, select_arms_top_p.

// ── RandOpt Diagnostics (Plan 121) ──────────────────────────────

/// Solution density: fraction of scores ≥ base_score + margin.
/// From RandOpt (Neural Thickets) — measures how many perturbations improve over baseline.
pub fn solution_density(scores: &[f32], base_score: f32, margin: f32) -> f32 {
    match scores.is_empty() {
        true => 0.0,
        false => {
            let threshold = base_score + margin;
            let above = scores.iter().filter(|&&s| s >= threshold).count();
            above as f32 / scores.len() as f32
        }
    }
}

/// Spectral discordance: measures specialist vs generalist distribution.
/// D ∈ [0, 1], D→1 means specialists, D→0 means generalists.
/// Input: N arms × M tasks percentile-rank matrix.
pub fn spectral_discordance(performance_matrix: &[Vec<f32>]) -> f32 {
    if performance_matrix.is_empty() {
        return 0.0;
    }
    let n = performance_matrix.len();
    let m = performance_matrix.first().map_or(0, |r| r.len());
    if m <= 1 || n == 0 {
        return 0.0;
    }
    // For each arm, compute variance across tasks and accumulate average in one
    // pass — avoids materializing a `variances: Vec<f32>` allocation.
    // Normalize: max variance = 0.25 (for binary 0/1 with p=0.5)
    let inv_max_var = 4.0_f32; // 1 / 0.25
    let mut sum_normalized_var = 0.0f32;
    for row in performance_matrix {
        let var = if row.len() <= 1 {
            0.0
        } else {
            let n = row.len() as f32;
            let mean = row.iter().sum::<f32>() / n;
            row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n
        };
        sum_normalized_var += var * inv_max_var;
    }
    (sum_normalized_var / n as f32).min(1.0)
}

// ---------------------------------------------------------------------------
// Adaptive Top-p Arm Selection (dMoE distillation, Research 161, Plan 181)
// ---------------------------------------------------------------------------

/// Adaptive top-p arm selection for BanditPruner.
///
/// Replaces fixed top-k with dynamic arm budget based on score concentration.
/// When scores are concentrated (clear winner) → selects fewer arms → faster.
/// When scores are dispersed (uncertain) → selects more arms → better exploration.
///
/// # Arguments
/// * `q_values` - Bandit Q-values for each arm
/// * `ucb_bonus` - UCB exploration bonus for each arm
/// * `p` - Cumulative probability threshold (default: 0.85)
///
/// # Returns
/// Indices of selected arms, sorted by score descending.
#[cfg(feature = "bandit_top_p")]
pub fn select_arms_top_p(q_values: &[f32], ucb_bonus: &[f32], p: f32) -> Vec<usize> {
    let scores: Vec<f32> = q_values
        .iter()
        .zip(ucb_bonus.iter())
        .map(|(&q, &u)| q + u)
        .collect();
    let n = scores.len();

    if n == 0 {
        return vec![];
    }

    // Sort by score descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));

    let total: f32 = scores.iter().map(|s| s.max(0.0)).sum();
    if total <= 0.0 {
        return indices;
    }

    let mut cumsum = 0.0f32;
    let mut selected = Vec::with_capacity(n);
    for &idx in &indices {
        cumsum += scores[idx].max(0.0) / total;
        selected.push(idx);
        if cumsum >= p {
            break;
        }
    }
    selected
}

