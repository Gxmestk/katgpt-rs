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
            let slot = table
                .entry(bind_key(b))
                .or_insert_with(|| (vec![0.0; dim], 0));
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
/// Fraction of `truth` whose binding set is present in the shared `candidates`
/// pool — the answerability check for [`withheld_pair_top1`].
///
/// A state whose true binding is absent from the pool **cannot** be scored
/// correctly, so a top-1 number computed over such a pool is a property of the
/// pool, not of the primitive. [`withheld_pair_top1`]'s doc has always said to
/// check this; before Issue 710 T4 there was nothing to check it with, which
/// is the [`AtomicNull::coverage`] discipline missing one function over.
#[must_use]
pub fn candidate_pool_coverage(truth: &[TprBindings], candidates: &[TprBindings]) -> f32 {
    match truth.is_empty() {
        true => 0.0,
        false => {
            let pool: std::collections::HashSet<_> = candidates.iter().map(bind_key).collect();
            let hit = truth.iter().filter(|t| pool.contains(&bind_key(t))).count();
            hit as f32 / truth.len() as f32
        }
    }
}

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
    ///
    /// **Read `vacuous` first** — same discipline as
    /// [`ShuffledRoleReport::degraded`] (Issue 710 T4).
    pub structured: bool,
    /// The null this router fits is **the same corpus and the same scheme**
    /// as the full fit, so `ratio == 1.0` is arithmetic, not evidence. Happens
    /// when the caller's scheme is already `arity == 1` and every role label is
    /// already `0` — i.e. the bag-of-fillers hypothesis was the input.
    pub vacuous: bool,
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
    let vacuous = cfg.scheme.arity() == 1
        && input
            .bindings
            .iter()
            .all(|b| b.roles.iter().all(|&p| p == 0));
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
        vacuous,
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

/// **T6 (c) control** — which permutation the role control draws.
///
/// The two arms answer the same question ("is the role assignment
/// load-bearing?") on corpora of different shape, and each is **vacuous** on
/// the other's shape — see [`role_shuffle_is_vacuous`]. Pick with
/// [`role_shuffle_mode_for`] (what [`shuffled_role_control`] does) or pin one
/// explicitly with [`shuffled_role_control_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleShuffleMode {
    /// Permute each state's role list **within that state**. The original
    /// arm — meaningful only when some state carries two bindings with
    /// distinct roles, since a 1-element list has no non-identity permutation.
    WithinState,
    /// Permute the role assignment **across** states, fillers untouched. The
    /// arm a retrieval corpus needs: one span carries one role and one filler,
    /// so the pairing can only be broken globally (Issue 710).
    CrossState,
}

/// Can this permutation arm produce a binding set different from its input?
///
/// A control whose permutation is a provable identity reports
/// `r_shuffled == r_true`, `ratio == 1.0`, `degraded == false` — which is
/// **indistinguishable from the real negative result** ("the specific role
/// assignment is not load-bearing"). This is the [`AtomicNull::coverage`]
/// discipline one function down: a control's report must carry whether the
/// control *could have* failed.
///
/// - [`RoleShuffleMode::WithinState`] is vacuous when no state carries two
///   **distinct** role labels — which includes every single-binding corpus
///   (Issue 710: `for i in (1..1).rev()` is an empty loop).
/// - [`RoleShuffleMode::CrossState`] is vacuous when the whole corpus uses
///   fewer than two distinct role labels; swapping equal labels is an
///   identity whatever the draw.
#[must_use]
pub fn role_shuffle_is_vacuous(bindings: &[TprBindings], mode: RoleShuffleMode) -> bool {
    match mode {
        RoleShuffleMode::WithinState => bindings
            .iter()
            .all(|b| b.roles.windows(2).all(|w| w[0] == w[1])),
        RoleShuffleMode::CrossState => {
            let mut roles = bindings.iter().flat_map(|b| b.roles.iter().copied());
            match roles.next() {
                None => true,
                Some(first) => roles.all(|r| r == first),
            }
        }
    }
}

/// The arm that is not vacuous on this corpus, preferring the within-state one.
///
/// Cross-state vacuity implies within-state vacuity, so when this returns
/// [`RoleShuffleMode::CrossState`] and *that* is vacuous too, the corpus has
/// no role structure to break at all and no arm can help — which the report's
/// `vacuous` flag then says.
#[must_use]
pub fn role_shuffle_mode_for(bindings: &[TprBindings]) -> RoleShuffleMode {
    match role_shuffle_is_vacuous(bindings, RoleShuffleMode::WithinState) {
        true => RoleShuffleMode::CrossState,
        false => RoleShuffleMode::WithinState,
    }
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
    ///
    /// **Read `vacuous` first.** A `false` here from a vacuous control is a
    /// no-op, not a negative result.
    pub degraded: bool,
    /// Which arm actually ran (resolved, never a request).
    pub mode: RoleShuffleMode,
    /// The control **could not have failed** on this corpus — its permutation
    /// is a provable identity, so `ratio == 1.0` and `degraded == false` carry
    /// no information. See [`role_shuffle_is_vacuous`] (Issue 710).
    pub vacuous: bool,
    /// How many role slots the drawn permutation actually changed. `0` on a
    /// non-vacuous corpus means every one of [`MAX_SHUFFLE_DRAWS`] draws came
    /// back the identity — vanishingly unlikely, but reported rather than
    /// assumed, because an identity draw would masquerade as a no-structure
    /// verdict exactly like a vacuous arm does.
    pub moved: usize,
}

/// Redraws allowed before a non-vacuous arm gives up on moving a role.
pub const MAX_SHUFFLE_DRAWS: u32 = 8;

/// SplitMix64 — the generator both arms draw from, so a permutation is
/// reproducible from `(seed, mode, corpus)` alone.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// In-place Fisher-Yates over one contiguous run of role slots.
fn shuffle_run(run: &mut [u16], state: &mut u64) {
    for i in (1..run.len()).rev() {
        let j = usize::try_from(splitmix64(state) % (i as u64 + 1)).expect("index fits usize");
        run.swap(i, j);
    }
}

/// Draw one permutation of the role assignment; fillers are never touched.
///
/// Both arms share the flatten → shuffle-runs → rebuild path; `mode` chooses
/// only the run boundaries (one run per state, or one run for the corpus).
/// Returns the permuted bindings and how many slots moved.
fn permute_roles(
    bindings: &[TprBindings],
    mode: RoleShuffleMode,
    seed: u64,
) -> (Vec<TprBindings>, usize) {
    let mut roles: Vec<u16> = bindings
        .iter()
        .flat_map(|b| b.roles.iter().copied())
        .collect();
    let original = roles.clone();
    let mut state = seed | 1;
    match mode {
        RoleShuffleMode::CrossState => shuffle_run(&mut roles, &mut state),
        RoleShuffleMode::WithinState => {
            let mut cursor = 0usize;
            for b in bindings {
                let n = b.len();
                shuffle_run(&mut roles[cursor..cursor + n], &mut state);
                cursor += n;
            }
        }
    }
    let moved = original
        .iter()
        .zip(roles.iter())
        .filter(|(a, b)| a != b)
        .count();

    let mut cursor = 0usize;
    let permuted = bindings
        .iter()
        .map(|b| {
            let n = b.len();
            let slice = roles[cursor..cursor + n].to_vec();
            cursor += n;
            TprBindings {
                roles: slice,
                fillers: b.fillers.clone(),
            }
        })
        .collect();
    (permuted, moved)
}

/// **T6 (c) control** — permute the role labels and refit, on whichever arm is
/// not vacuous for this corpus ([`role_shuffle_mode_for`]).
///
/// The dual of [`bow_router`]: the router asks whether roles buy anything over
/// a bag of fillers, this asks whether the SPECIFIC role assignment is
/// load-bearing. A fit that scores the same on shuffled roles was never using
/// them, whatever its residual says — **provided the shuffle could have moved
/// something**, which `ShuffledRoleReport::vacuous` reports and which the
/// pre-Issue-710 version of this function could not (it drew the within-state
/// arm unconditionally, an empty loop on the single-binding corpora every
/// retrieval consumer produces).
///
/// Multi-binding corpora resolve to [`RoleShuffleMode::WithinState`] and are
/// bit-identical to that earlier behaviour.
pub fn shuffled_role_control(
    input: AlsInput<'_>,
    cfg: &AlsConfig,
    seed: u64,
    eps: f32,
) -> Result<ShuffledRoleReport, TprError> {
    let mode = role_shuffle_mode_for(input.bindings);
    shuffled_role_control_with(input, cfg, seed, eps, mode)
}

/// [`shuffled_role_control`] with the permutation arm pinned by the caller.
///
/// Use this to assert a specific control rather than accept the resolved one
/// (a benchmark comparing both arms on the same artifact, say). The report
/// still carries `vacuous`, so pinning an arm cannot silently buy a no-op.
pub fn shuffled_role_control_with(
    input: AlsInput<'_>,
    cfg: &AlsConfig,
    seed: u64,
    eps: f32,
    mode: RoleShuffleMode,
) -> Result<ShuffledRoleReport, TprError> {
    let (_, true_rep) = als_fit(input, cfg)?;
    let vacuous = role_shuffle_is_vacuous(input.bindings, mode);

    // Draw 0 uses `seed` unchanged, so a non-degenerate corpus reproduces the
    // pre-710 permutation exactly; later draws exist only so an identity draw
    // cannot be reported as "roles carry nothing".
    let mut drawn = permute_roles(input.bindings, mode, seed);
    let mut attempt = 1u32;
    while !vacuous && drawn.1 == 0 && attempt < MAX_SHUFFLE_DRAWS {
        let s = seed.wrapping_add(u64::from(attempt).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        drawn = permute_roles(input.bindings, mode, s);
        attempt += 1;
    }
    let (shuffled, moved) = drawn;

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
        mode,
        vacuous,
        moved,
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
