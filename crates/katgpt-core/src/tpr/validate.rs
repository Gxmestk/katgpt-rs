//! Validation harness + diagnostics for the TPR binding algebra
//! (Issue 707 T6/T7) — the instruments the GOAT gate reads, kept out of the
//! runtime path.
//!
//! Three parts, matching Research 527 §6:
//!
//! - **(a) fit band** — [`validate_bindings`]: holdout residual band + unbind
//!   cosine + surgery additivity, all measured on states the fit never saw.
//! - **(c) systematicity** — [`withheld_pair_top1`] against
//!   [`AtomicNull`]: a per-pair lookup table CANNOT answer a withheld
//!   `(role, filler)` combination, so TPR beating it is the systematicity
//!   certificate. A null that also fails **in-distribution** is VACUOUS and
//!   certifies nothing — [`AtomicNull::coverage`] reports exactly that, and
//!   callers must check it before quoting an OOD win (the measured healer-corpus
//!   failure, riir-clippy `.benchmarks/062_withheld_pair_ood.md`).
//! - **diagnostics** — [`bow_router`] (does this state family carry binding
//!   structure at all?) and [`bic_select`] (which role scheme?).

use super::als::{als_fit, param_count};
use super::types::{AlsConfig, AlsInput, TprArtifact, TprBindings, TprError, TprScheme};
use super::{TprScratch, encode_into, project_into, surgery_delta_into, unbind_into};
use std::collections::BTreeMap;

/// Holdout validation report (T6 part a + b).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BindingReport {
    pub n_states: usize,
    /// Holdout `‖e − ê‖` band, `ê` the structural projection.
    pub residual_p50: f32,
    pub residual_max: f32,
    /// Cosine between the unbound filler and the artifact's own filler row,
    /// over every (state, binding) pair.
    pub unbind_cos_min: f32,
    pub unbind_cos_mean: f32,
    /// `max |e_surgery − e_reencoded|` — 0 iff surgery is exactly additive.
    pub surgery_max_abs_err: f32,
    /// Worst-case crosstalk envelope over the holdout binding counts.
    pub unbind_bound_max: f32,
}

/// **T6 (a)+(b)** — validate a fitted artifact on holdout states.
///
/// `unbind_cos_*` is measured from the **projected core** (i.e. through the
/// real `state → core → filler` path a consumer would use), not from the
/// planted core, so it prices the projection error too.
///
/// Surgery additivity compares the in-place edit against a full re-encode of
/// the edited binding set: the two agree exactly when the untouched bindings
/// are bit-preserved.
pub fn validate_bindings(
    art: &TprArtifact,
    states: &[f32],
    bindings: &[TprBindings],
    scratch: &mut TprScratch,
) -> Result<BindingReport, TprError> {
    let dim = art.dim;
    let d = art.d;
    let n = states.len() / dim.max(1);
    if bindings.len() != n {
        return Err(TprError::DimMismatch {
            what: "bindings",
            expected: n,
            got: bindings.len(),
        });
    }
    let mut rep = BindingReport {
        n_states: n,
        unbind_cos_min: f32::INFINITY,
        ..Default::default()
    };
    let mut residuals = Vec::with_capacity(n);
    let mut cos_sum = 0.0f64;
    let mut cos_count = 0usize;
    let mut recon = vec![0.0f32; dim];
    let mut got = vec![0.0f32; d];
    let mut edited = vec![0.0f32; dim];
    let mut reencoded = vec![0.0f32; dim];

    for (s, b) in bindings.iter().enumerate() {
        let state = &states[s * dim..(s + 1) * dim];
        let r = project_into(art, state, scratch, &mut recon)?;
        residuals.push(r);
        rep.unbind_bound_max = rep
            .unbind_bound_max
            .max(super::unbind_error_bound(art, b.len()));

        // Unbind every binding through the projected core (the real
        // state → core → filler path, so projection error is priced in).
        super::state_to_core_into(art, state, scratch)?;
        let core = std::mem::take(&mut scratch.x);
        for (&p, &v) in b.roles.iter().zip(b.fillers.iter()) {
            unbind_into(art, &core, p, &mut got)?;
            let truth = &art.fillers[v as usize * d..(v as usize + 1) * d];
            let c = cosine(&got, truth);
            rep.unbind_cos_min = rep.unbind_cos_min.min(c);
            cos_sum += c as f64;
            cos_count += 1;
        }
        scratch.x = core;

        // Surgery additivity: swap binding 0's filler for the next id.
        if let (Some(&p0), Some(&v0)) = (b.roles.first(), b.fillers.first()) {
            let v1 = ((v0 as usize + 1) % art.n_fillers.max(1)) as u16;
            let f_old = art.fillers[v0 as usize * d..(v0 as usize + 1) * d].to_vec();
            let f_new = art.fillers[v1 as usize * d..(v1 as usize + 1) * d].to_vec();
            encode_into(art, b, scratch, &mut edited)?;
            surgery_delta_into(art, &mut edited, p0, &f_old, &f_new, scratch)?;
            let mut swapped = b.clone();
            swapped.fillers[0] = v1;
            encode_into(art, &swapped, scratch, &mut reencoded)?;
            for (a, e) in edited.iter().zip(reencoded.iter()) {
                rep.surgery_max_abs_err = rep.surgery_max_abs_err.max((a - e).abs());
            }
        }
    }

    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    rep.residual_p50 = match residuals.is_empty() {
        true => 0.0,
        false => residuals[(residuals.len() - 1) / 2],
    };
    rep.residual_max = residuals.last().copied().unwrap_or(0.0);
    rep.unbind_cos_mean = match cos_count {
        0 => 0.0,
        c => (cos_sum / c as f64) as f32,
    };
    if !rep.unbind_cos_min.is_finite() {
        rep.unbind_cos_min = 0.0;
    }
    Ok(rep)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot = x.mul_add(*y, dot);
        na = x.mul_add(*x, na);
        nb = y.mul_add(*y, nb);
    }
    let den = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / den
}

/// Binding-set key: the sorted `(role, filler)` pair list. `BTreeMap` (not a
/// hash map) so iteration order — and therefore every reported number — is
/// deterministic across processes.
type BindKey = Vec<(u16, u16)>;

fn bind_key(b: &TprBindings) -> BindKey {
    let mut k: BindKey = b
        .roles
        .iter()
        .copied()
        .zip(b.fillers.iter().copied())
        .collect();
    k.sort_unstable();
    k
}

/// **T6 (c) null** — the atomic-dictionary memorizer: a per-binding-set mean
/// state. It answers in-distribution keys perfectly and CANNOT answer a
/// withheld `(role, filler)` combination, because that key was never observed.
///
/// This is the arm TPR must beat for the systematicity claim. Check
/// [`AtomicNull::coverage`] on the **in-distribution** arm first: a null at
/// 0% ID is vacuous and its OOD 0% certifies nothing.
#[derive(Debug, Clone, Default)]
pub struct AtomicNull {
    dim: usize,
    table: BTreeMap<BindKey, (Vec<f32>, u32)>,
}

impl AtomicNull {
    /// Memorize the mean state of every binding set in the training corpus.
    pub fn fit(dim: usize, states: &[f32], bindings: &[TprBindings]) -> Self {
        let mut table: BTreeMap<BindKey, (Vec<f32>, u32)> = BTreeMap::new();
        for (s, b) in bindings.iter().enumerate() {
            let e = &states[s * dim..(s + 1) * dim];
            let slot = table.entry(bind_key(b)).or_insert_with(|| (vec![0.0; dim], 0));
            for (acc, &v) in slot.0.iter_mut().zip(e.iter()) {
                *acc += v;
            }
            slot.1 += 1;
        }
        for (sum, count) in table.values_mut() {
            let c = *count as f32;
            for v in sum.iter_mut() {
                *v /= c;
            }
        }
        Self { dim, table }
    }

    /// Fraction of `candidates` this dictionary has an entry for — the
    /// vacuity check.
    pub fn coverage(&self, candidates: &[TprBindings]) -> f32 {
        match candidates.is_empty() {
            true => 0.0,
            false => {
                let hit = candidates
                    .iter()
                    .filter(|c| self.table.contains_key(&bind_key(c)))
                    .count();
                hit as f32 / candidates.len() as f32
            }
        }
    }

    /// Top-1 accuracy over `candidates`, scoring each by distance to its
    /// memorized mean. An unseen key scores `+∞` — the by-construction OOD
    /// failure. Ties and all-unseen candidate sets count as a MISS (never a
    /// coin flip in the null's favour).
    pub fn top1(&self, states: &[f32], truth: &[TprBindings], candidates: &[TprBindings]) -> f32 {
        let n = truth.len();
        let mut hits = 0usize;
        for (s, t) in truth.iter().enumerate() {
            let e = &states[s * self.dim..(s + 1) * self.dim];
            let mut best = f32::INFINITY;
            let mut best_i = usize::MAX;
            let mut tied = false;
            for (i, c) in candidates.iter().enumerate() {
                let score = match self.table.get(&bind_key(c)) {
                    None => f32::INFINITY,
                    Some((mean, _)) => l2(e, mean),
                };
                if score < best {
                    best = score;
                    best_i = i;
                    tied = false;
                } else if score == best && best_i != usize::MAX {
                    tied = true;
                }
            }
            let correct = best.is_finite()
                && !tied
                && best_i != usize::MAX
                && bind_key(&candidates[best_i]) == bind_key(t);
            if correct {
                hits += 1;
            }
        }
        match n {
            0 => 0.0,
            _ => hits as f32 / n as f32,
        }
    }
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut ss = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let dv = x - y;
        ss = dv.mul_add(dv, ss);
    }
    ss.sqrt()
}

/// **T6 (c) TPR arm** — top-1 accuracy of reconstruct-and-match: score every
/// candidate binding set by `‖e − (W·c(h) + b)‖` and take the argmin.
///
/// Composition is what makes this answerable OOD: an unseen `(role, filler)`
/// pair still has a fitted filler row and a fitted role, so its core — and
/// therefore its predicted state — exists.
///
/// `candidates` is ONE shared pool scored against every state (the retrieval
/// setting). It must contain each state's true binding set, or that state is
/// unanswerable by construction and the score is a property of the pool, not
/// of the primitive — check the pool before reading the number.
pub fn withheld_pair_top1(
    art: &TprArtifact,
    states: &[f32],
    truth: &[TprBindings],
    candidates: &[TprBindings],
    scratch: &mut TprScratch,
) -> Result<f32, TprError> {
    let dim = art.dim;
    let mut recon = vec![0.0f32; dim];
    let mut hits = 0usize;
    for (s, t) in truth.iter().enumerate() {
        let e = &states[s * dim..(s + 1) * dim];
        let mut best = f32::INFINITY;
        let mut best_i = usize::MAX;
        for (i, c) in candidates.iter().enumerate() {
            encode_into(art, c, scratch, &mut recon)?;
            let score = l2(e, &recon);
            if score < best {
                best = score;
                best_i = i;
            }
        }
        if best_i != usize::MAX && bind_key(&candidates[best_i]) == bind_key(t) {
            hits += 1;
        }
    }
    Ok(match truth.len() {
        0 => 0.0,
        n => hits as f32 / n as f32,
    })
}

/// **T7** BoW structure router report.
#[derive(Debug, Clone, PartialEq)]
pub struct BowRouterReport {
    /// Residual energy fraction of the `m = 1` shared-role (bag-of-fillers)
    /// fit.
    pub r_bow: f32,
    /// Residual energy fraction of the full structured fit.
    pub r_full: f32,
    /// `r_bow / r_full` — how much the structure actually buys.
    pub ratio: f32,
    /// `ratio > 1 + eps`: the family carries binding structure worth the
    /// structured machinery.
    pub structured: bool,
}

/// **T7** — is this state family structured at all?
///
/// Fits the `m = 1` shared-role null (every binding collapses onto one block:
/// a bag of fillers, order-free) and compares its residual energy against the
/// full fit. `ratio ≈ 1` ⇒ the roles carry nothing ⇒ skip the structured
/// machinery. This is the cheap gate that stops a structured primitive from
/// being adopted where a sum of embeddings would do.
pub fn bow_router(
    input: AlsInput<'_>,
    cfg: &AlsConfig,
    eps: f32,
) -> Result<BowRouterReport, TprError> {
    let (_full, full_rep) = als_fit(input, cfg)?;
    let flat: Vec<TprBindings> = input
        .bindings
        .iter()
        .map(|b| TprBindings {
            roles: vec![0; b.roles.len()],
            fillers: b.fillers.clone(),
        })
        .collect();
    let bow_input = AlsInput {
        dim: input.dim,
        n_fillers: input.n_fillers,
        states: input.states,
        bindings: &flat,
    };
    let mut bow_cfg = cfg.clone();
    bow_cfg.scheme = TprScheme::Orthogonal { arity: 1 };
    let (_bow, bow_rep) = als_fit(bow_input, &bow_cfg)?;

    let r_bow = bow_rep.residual_energy_fraction;
    let r_full = full_rep.residual_energy_fraction;
    let ratio = match r_full > 1e-9 {
        true => r_bow / r_full,
        // A perfect structured fit against a non-zero BoW residual is the
        // strongest possible structure signal; report it saturated rather
        // than as a division blow-up.
        false => match r_bow > 1e-9 {
            true => f32::MAX,
            false => 1.0,
        },
    };
    Ok(BowRouterReport {
        r_bow,
        r_full,
        ratio,
        structured: ratio > 1.0 + eps,
    })
}

/// Collapse role ids onto `n_slots` blocks (`p mod n_slots`) — the corpus a
/// lower-arity structure hypothesis actually sees.
fn fold_roles(bindings: &[TprBindings], n_slots: usize) -> Vec<TprBindings> {
    let n = n_slots.max(1) as u16;
    bindings
        .iter()
        .map(|b| TprBindings {
            roles: b.roles.iter().map(|&p| p % n).collect(),
            fillers: b.fillers.clone(),
        })
        .collect()
}

/// **T6 (c) control** — role-shuffle report.
#[derive(Debug, Clone, PartialEq)]
pub struct ShuffledRoleReport {
    /// Residual energy fraction of the true fit.
    pub r_true: f32,
    /// Residual energy fraction after the role labels are permuted.
    pub r_shuffled: f32,
    /// `r_shuffled / r_true`.
    pub ratio: f32,
    /// `ratio > 1 + eps`: destroying the role assignment destroyed real
    /// structure, so the fit was reading roles rather than memorizing states.
    pub degraded: bool,
}

/// **T6 (c) control** — permute each state's role labels and refit.
///
/// The dual of [`bow_router`]: the router asks whether roles buy anything over
/// a bag of fillers, this asks whether the SPECIFIC role assignment is
/// load-bearing. A fit that scores the same on shuffled roles was never using
/// them, whatever its residual says. The permutation is per-state and
/// deterministic in `seed` (Fisher-Yates over SplitMix64), so the control is
/// reproducible.
pub fn shuffled_role_control(
    input: AlsInput<'_>,
    cfg: &AlsConfig,
    seed: u64,
    eps: f32,
) -> Result<ShuffledRoleReport, TprError> {
    let (_, true_rep) = als_fit(input, cfg)?;
    let mut state = seed | 1;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let shuffled: Vec<TprBindings> = input
        .bindings
        .iter()
        .map(|b| {
            let mut roles = b.roles.clone();
            let n = roles.len();
            for i in (1..n).rev() {
                let j = (next() % (i as u64 + 1)) as usize;
                roles.swap(i, j);
            }
            TprBindings {
                roles,
                fillers: b.fillers.clone(),
            }
        })
        .collect();
    let shuf_input = AlsInput {
        dim: input.dim,
        n_fillers: input.n_fillers,
        states: input.states,
        bindings: &shuffled,
    };
    let (_, shuf_rep) = als_fit(shuf_input, cfg)?;
    let r_true = true_rep.residual_energy_fraction;
    let r_shuffled = shuf_rep.residual_energy_fraction;
    let ratio = match r_true > 1e-9 {
        true => r_shuffled / r_true,
        false => match r_shuffled > 1e-9 {
            true => f32::MAX,
            false => 1.0,
        },
    };
    Ok(ShuffledRoleReport {
        r_true,
        r_shuffled,
        ratio,
        degraded: ratio > 1.0 + eps,
    })
}

/// **T7** BIC scheme-selection result.
#[derive(Debug, Clone, PartialEq)]
pub struct BicSelection {
    /// Index into the candidate config list.
    pub best: usize,
    /// The winning artifact's frozen structure label.
    pub label: String,
    /// `N·ln(RSS/N) + p·ln(N)` per candidate, `N` = scalar observation count.
    pub scores: Vec<f64>,
}

/// **T7** — pick the role scheme by BIC over candidate configs.
///
/// `score(S) = N·ln(RSS_S / N) + p_S·ln N` with `N = n_states · dim` (the
/// scalar observation count, not the state count — the residual is measured
/// per coordinate) and `p_S` the fitted parameter count. Argmin wins and its
/// label becomes the frozen structure label.
pub fn bic_select(input: AlsInput<'_>, cfgs: &[AlsConfig]) -> Result<BicSelection, TprError> {
    if cfgs.is_empty() {
        return Err(TprError::DimMismatch {
            what: "bic candidates",
            expected: 1,
            got: 0,
        });
    }
    let n_obs = (input.n_states() * input.dim) as f64;
    let mut scores = Vec::with_capacity(cfgs.len());
    let mut best = 0usize;
    let mut best_score = f64::INFINITY;
    let mut best_label = String::new();
    for (i, cfg) in cfgs.iter().enumerate() {
        // A candidate with FEWER bind slots than the corpus uses is scored on
        // the folded corpus (`role → role mod n_slots`) rather than rejected:
        // that fold is what a coarser structure hypothesis MEANS, and at
        // `n_slots = 1` it is exactly the bag-of-fillers null. Rejecting it
        // would silently drop the most important candidate from the sweep.
        let folded = fold_roles(input.bindings, cfg.scheme.n_bind_slots());
        let cand_input = AlsInput {
            dim: input.dim,
            n_fillers: input.n_fillers,
            states: input.states,
            bindings: &folded,
        };
        let (art, rep) = als_fit(cand_input, cfg)?;
        let rss = rep.final_ssr.max(1e-30);
        let p = param_count(&art) as f64;
        let score = n_obs * (rss / n_obs).ln() + p * n_obs.ln();
        scores.push(score);
        if score < best_score {
            best_score = score;
            best = i;
            best_label = art.bic_label.clone();
        }
    }
    Ok(BicSelection {
        best,
        label: best_label,
        scores,
    })
}
