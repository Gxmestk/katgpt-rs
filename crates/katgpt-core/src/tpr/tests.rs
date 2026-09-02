//! Issue 707 Phase 1–2 correctness tests: planted-TPR recovery, the
//! determinism gate, surgery additivity, the crosstalk envelope, and the
//! three-part validation harness (T6) + diagnostics (T7).

use super::*;
use crate::tpr::als::param_count;

/// Local deterministic RNG for the fixtures (the fit has its own).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn sym(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        u.mul_add(2.0, -1.0)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

struct Planted {
    dim: usize,
    d: usize,
    m: usize,
    n_fillers: usize,
    states: Vec<f32>,
    bindings: Vec<TprBindings>,
}

/// Generate a corpus that IS a TPR by construction: `e = W·(Σ r_p ⊗ f_v) + b`
/// with one binding per role block (the positive control of T6 part b).
fn planted(dim: usize, d: usize, m: usize, n_fillers: usize, n: usize, seed: u64) -> Planted {
    let k = m * d;
    let mut rng = Rng::new(seed);
    let w: Vec<f32> = (0..dim * k).map(|_| rng.sym()).collect();
    let bias: Vec<f32> = (0..dim).map(|_| 0.25 * rng.sym()).collect();
    let mut fillers = vec![0.0f32; n_fillers * d];
    for v in 0..n_fillers {
        let row = &mut fillers[v * d..(v + 1) * d];
        for x in row.iter_mut() {
            *x = rng.sym();
        }
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row.iter_mut() {
            *x /= nrm.max(1e-6);
        }
    }
    let mut states = vec![0.0f32; n * dim];
    let mut bindings = Vec::with_capacity(n);
    let mut core = vec![0.0f32; k];
    for s in 0..n {
        core.fill(0.0);
        let mut b = TprBindings::default();
        for p in 0..m {
            let v = rng.below(n_fillers);
            b.roles.push(p as u16);
            b.fillers.push(v as u16);
            for j in 0..d {
                core[p * d + j] += fillers[v * d + j];
            }
        }
        for i in 0..dim {
            let mut acc = bias[i];
            for j in 0..k {
                acc = w[i * k + j].mul_add(core[j], acc);
            }
            states[s * dim + i] = acc;
        }
        bindings.push(b);
    }
    Planted {
        dim,
        d,
        m,
        n_fillers,
        states,
        bindings,
    }
}

fn input(p: &Planted) -> AlsInput<'_> {
    AlsInput {
        dim: p.dim,
        n_fillers: p.n_fillers,
        states: &p.states,
        bindings: &p.bindings,
    }
}

fn fit_orthogonal(p: &Planted) -> (TprArtifact, AlsReport) {
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    als_fit(input(p), &cfg).expect("planted fit must converge")
}

// ---------------------------------------------------------------------------
// T-G1 — planted recovery + determinism
// ---------------------------------------------------------------------------

#[test]
fn planted_tpr_is_recovered() {
    let p = planted(24, 4, 3, 6, 160, 0xA11CE);
    let (art, rep) = fit_orthogonal(&p);
    assert_eq!(rep.monotone_violations, 0, "ALS objective must not increase");
    assert!(
        rep.residual_energy_fraction < 1e-3,
        "planted corpus must be explained: energy fraction {} (ssr {})",
        rep.residual_energy_fraction,
        rep.final_ssr
    );
    assert!(art.verify(), "artifact must carry a valid commitment");
    assert_eq!(art.n_fit_states, 160);
}

#[test]
fn double_fit_is_bit_identical() {
    let p = planted(16, 3, 3, 5, 96, 0xBEEF);
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    let (a, _) = als_fit(input(&p), &cfg).unwrap();
    let (b, _) = als_fit(input(&p), &cfg).unwrap();
    assert_eq!(a.commitment, b.commitment, "double fit must be bit-identical");
    assert_eq!(a.to_bytes(), b.to_bytes());
}

#[test]
fn holdout_unbind_and_surgery_hold() {
    let p = planted(24, 4, 3, 6, 160, 0xC0FFEE);
    let (art, _) = fit_orthogonal(&p);
    let mut scratch = TprScratch::new(&art);
    let hold = planted_holdout(&art, &p, 32, 0x5EED);
    let rep = validate_bindings(&art, &hold.0, &hold.1, &mut scratch).unwrap();
    assert!(
        rep.unbind_cos_min > 0.999,
        "unbind cosine floor {} (mean {})",
        rep.unbind_cos_min,
        rep.unbind_cos_mean
    );
    let scale: f32 = hold.0.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        rep.surgery_max_abs_err <= 1e-4 * scale.max(1.0),
        "surgery must be additive: err {} scale {}",
        rep.surgery_max_abs_err,
        scale
    );
}

/// Holdout states generated THROUGH the fitted artifact (so the truth is the
/// artifact's own filler table — the self-consistency invariant the runtime
/// path actually relies on).
fn planted_holdout(
    art: &TprArtifact,
    p: &Planted,
    n: usize,
    seed: u64,
) -> (Vec<f32>, Vec<TprBindings>) {
    let mut rng = Rng::new(seed);
    let mut scratch = TprScratch::new(art);
    let mut states = vec![0.0f32; n * art.dim];
    let mut bindings = Vec::with_capacity(n);
    for s in 0..n {
        let mut b = TprBindings::default();
        for r in 0..p.m {
            b.roles.push(r as u16);
            b.fillers.push(rng.below(p.n_fillers) as u16);
        }
        let mut out = vec![0.0f32; art.dim];
        encode_into(art, &b, &mut scratch, &mut out).unwrap();
        states[s * art.dim..(s + 1) * art.dim].copy_from_slice(&out);
        bindings.push(b);
    }
    (states, bindings)
}

// ---------------------------------------------------------------------------
// T2 — crosstalk envelope on a role-vector scheme
// ---------------------------------------------------------------------------

#[test]
fn role_vector_unbind_respects_the_bound() {
    let p = planted(20, 4, 3, 5, 120, 0xD1CE);
    let roles = vec![1.0, 0.15, 0.0, 0.0, 1.0, 0.2, 0.1, 0.0, 1.0];
    let cfg = AlsConfig::new(p.d, TprScheme::RoleVectors { arity: 3, roles });
    let (art, _) = als_fit(input(&p), &cfg).unwrap();
    assert!(art.unbind_basis.is_some(), "role-vector fit needs a basis");
    assert!(
        (0.0..=1.0).contains(&art.crosstalk_mu),
        "mu must be a coherence in [0,1], got {}",
        art.crosstalk_mu
    );
    let bound = unbind_error_bound(&art, 3);
    assert!(bound >= 0.0 && bound.is_finite());
    // The sigmoid gate must be monotone-decreasing in the binding count.
    let c1 = unbind_confidence(&art, 1);
    let c4 = unbind_confidence(&art, 4);
    assert!(c1 >= c4, "confidence must fall as crosstalk grows");
    assert!((0.0..=1.0).contains(&c1) && (0.0..=1.0).contains(&c4));
}

// ---------------------------------------------------------------------------
// T4 — structural projection
// ---------------------------------------------------------------------------

#[test]
fn projection_is_idempotent_on_the_manifold() {
    let p = planted(24, 4, 3, 6, 160, 0x1234);
    let (art, _) = fit_orthogonal(&p);
    let mut scratch = TprScratch::new(&art);
    let (states, bindings) = planted_holdout(&art, &p, 8, 0x9999);
    let mut out = vec![0.0f32; art.dim];
    let mut out2 = vec![0.0f32; art.dim];
    let _ = &bindings;
    let e = &states[..art.dim];
    let r1 = project_into(&art, e, &mut scratch, &mut out).unwrap();
    let r2 = project_into(&art, &out.clone(), &mut scratch, &mut out2).unwrap();
    // The projection carries the fit's ridge shrinkage (λ = 1e-3), so an
    // on-manifold state returns to itself up to that bias — measured against
    // the state NORM, which is what "the residual is negligible" means for an
    // L2 residual over `dim` coordinates.
    let norm: f32 = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    assert!(
        r1 / norm < 1e-2,
        "an on-manifold state must project to itself: residual {r1} / norm {norm}"
    );
    assert!(
        r2 <= r1 + 1e-3 * norm,
        "projection must be idempotent: {r1} -> {r2}"
    );
}

#[test]
fn projection_denoises_off_manifold_noise() {
    let p = planted(32, 4, 3, 6, 200, 0x77);
    let (art, _) = fit_orthogonal(&p);
    let mut scratch = TprScratch::new(&art);
    let (states, _) = planted_holdout(&art, &p, 16, 0x88);
    let mut rng = Rng::new(0x5A5A);
    let mut improved = 0usize;
    let mut out = vec![0.0f32; art.dim];
    for s in 0..16 {
        let clean = &states[s * art.dim..(s + 1) * art.dim];
        let noisy: Vec<f32> = clean.iter().map(|&v| v + 0.2 * rng.sym()).collect();
        project_into(&art, &noisy, &mut scratch, &mut out).unwrap();
        let before: f32 = clean
            .iter()
            .zip(noisy.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let after: f32 = clean
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if after < before {
            improved += 1;
        }
    }
    assert!(
        improved >= 15,
        "projection must denoise toward the manifold ({improved}/16)"
    );
}

// ---------------------------------------------------------------------------
// T6 (c) — systematicity vs the atomic-dictionary null
// ---------------------------------------------------------------------------

#[test]
fn tpr_generalizes_to_withheld_pairs_where_the_null_cannot() {
    // Corpus over (role, filler) pairs with one combination withheld.
    let p = planted(28, 4, 3, 6, 240, 0xABCD);
    let withheld: Vec<usize> = (0..p.bindings.len())
        .filter(|&s| p.bindings[s].fillers[0] == 0 && p.bindings[s].fillers[1] == 1)
        .collect();
    assert!(!withheld.is_empty(), "fixture must contain the held pair");
    let train_idx: Vec<usize> = (0..p.bindings.len())
        .filter(|s| !withheld.contains(s))
        .collect();

    let (tr_states, tr_bindings) = subset(&p, &train_idx);
    let (te_states, te_bindings) = subset(&p, &withheld);

    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    let tr_input = AlsInput {
        dim: p.dim,
        n_fillers: p.n_fillers,
        states: &tr_states,
        bindings: &tr_bindings,
    };
    let (art, _) = als_fit(tr_input, &cfg).unwrap();
    let mut scratch = TprScratch::new(&art);

    // Shared candidate pool: for EVERY test state, its truth plus the decoys
    // that differ in the withheld role's filler. A pool built from one state's
    // bindings would leave the others unanswerable and measure the fixture
    // rather than the primitive.
    let mut candidates: Vec<TprBindings> = Vec::new();
    for t in &te_bindings {
        for v in 0..p.n_fillers {
            let mut c = t.clone();
            c.fillers[0] = v as u16;
            if !candidates.iter().any(|x| x == &c) {
                candidates.push(c);
            }
        }
    }

    let null = AtomicNull::fit(p.dim, &tr_states, &tr_bindings);
    let id_coverage = null.coverage(&tr_bindings);
    assert!(
        id_coverage > 0.99,
        "the null must be informative in-distribution (coverage {id_coverage})"
    );
    let null_ood = null.top1(&te_states, &te_bindings, &candidates);
    let tpr_ood =
        withheld_pair_top1(&art, &te_states, &te_bindings, &candidates, &mut scratch).unwrap();
    assert_eq!(null_ood, 0.0, "the memorizer cannot answer a withheld pair");
    assert!(
        tpr_ood > 0.9,
        "composition must answer it: tpr {tpr_ood} vs null {null_ood}"
    );
}

fn subset(p: &Planted, idx: &[usize]) -> (Vec<f32>, Vec<TprBindings>) {
    let mut states = Vec::with_capacity(idx.len() * p.dim);
    let mut bindings = Vec::with_capacity(idx.len());
    for &s in idx {
        states.extend_from_slice(&p.states[s * p.dim..(s + 1) * p.dim]);
        bindings.push(p.bindings[s].clone());
    }
    (states, bindings)
}

// ---------------------------------------------------------------------------
// T7 — diagnostics
// ---------------------------------------------------------------------------

#[test]
fn bow_router_separates_structured_from_bag_of_fillers() {
    let structured = planted(24, 4, 3, 6, 200, 0x11);
    let cfg = AlsConfig::new(structured.d, TprScheme::Orthogonal { arity: structured.m });
    let s_rep = bow_router(input(&structured), &cfg, 0.05).unwrap();
    assert!(
        s_rep.structured,
        "a planted TPR must route to the structured path (ratio {})",
        s_rep.ratio
    );

    // Bag-of-fillers control: the state depends on the filler MULTISET only,
    // so the roles carry nothing and the m=1 null loses nothing.
    let mut bag = planted(24, 4, 3, 6, 200, 0x22);
    rebuild_bag_of_fillers(&mut bag, 0x33);
    let b_rep = bow_router(input(&bag), &cfg, 0.05).unwrap();
    assert!(
        !b_rep.structured,
        "an order-free family must NOT route structured (ratio {})",
        b_rep.ratio
    );
}

/// Rewrite the corpus so the state is a function of the filler multiset only.
fn rebuild_bag_of_fillers(p: &mut Planted, seed: u64) {
    let mut rng = Rng::new(seed);
    let d = p.d;
    let dim = p.dim;
    let w: Vec<f32> = (0..dim * d).map(|_| rng.sym()).collect();
    let mut fillers = vec![0.0f32; p.n_fillers * d];
    for x in fillers.iter_mut() {
        *x = rng.sym();
    }
    for (s, b) in p.bindings.iter().enumerate() {
        let mut sum = vec![0.0f32; d];
        for &v in &b.fillers {
            for j in 0..d {
                sum[j] += fillers[v as usize * d + j];
            }
        }
        for i in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc = w[i * d + j].mul_add(sum[j], acc);
            }
            p.states[s * dim + i] = acc;
        }
    }
}

#[test]
fn shuffled_roles_destroy_a_structured_fit_and_not_a_bag() {
    let p = planted(24, 4, 3, 6, 200, 0x99);
    let cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    let rep = shuffled_role_control(input(&p), &cfg, 0xFEED, 0.05).unwrap();
    assert!(
        rep.degraded,
        "permuting roles must cost a structured fit (ratio {}, {:e} -> {:e})",
        rep.ratio, rep.r_true, rep.r_shuffled
    );

    let mut bag = planted(24, 4, 3, 6, 200, 0xAA);
    rebuild_bag_of_fillers(&mut bag, 0xBB);
    let brep = shuffled_role_control(input(&bag), &cfg, 0xFEED, 0.05).unwrap();
    assert!(
        !brep.degraded,
        "an order-free family cannot be damaged by a role shuffle (ratio {})",
        brep.ratio
    );
}

#[test]
fn bic_prefers_the_true_arity() {
    let p = planted(24, 4, 3, 6, 200, 0x44);
    let cands = vec![
        AlsConfig::new(p.d, TprScheme::Orthogonal { arity: 1 }),
        AlsConfig::new(p.d, TprScheme::Orthogonal { arity: 3 }),
    ];
    let sel = bic_select(input(&p), &cands).unwrap();
    assert_eq!(sel.best, 1, "BIC must pick m=3: scores {:?}", sel.scores);
    assert!(sel.label.contains("m3"));
}

#[test]
fn param_count_matches_the_artifact_shape() {
    let p = planted(16, 3, 2, 4, 64, 0x55);
    let (art, _) = fit_orthogonal(&p);
    assert_eq!(
        param_count(&art),
        art.dim * art.core_len() + art.dim + art.n_fillers * art.d
    );
}

// ---------------------------------------------------------------------------
// T5 — L2,1
// ---------------------------------------------------------------------------

#[test]
fn l21_prune_refit_kills_dead_dimensions() {
    // A corpus whose fillers only ever vary in 2 of 4 coordinates leaves the
    // remaining dimensions unsupported; the prune must find them.
    let p = planted(24, 4, 3, 6, 200, 0x66);
    let mut cfg = AlsConfig::new(p.d, TprScheme::Orthogonal { arity: p.m });
    cfg.l21 = L21::PruneRefit { tau_frac: 0.05 };
    let (art, rep) = als_fit(input(&p), &cfg).unwrap();
    assert_eq!(art.pruned_dims, rep.pruned_dims);
    assert!(
        rep.prune_ssr_increase >= -1e-6,
        "prune+refit cannot IMPROVE the unpruned optimum"
    );
}

// ---------------------------------------------------------------------------
// G3 — kill switch
// ---------------------------------------------------------------------------

#[test]
fn kill_switch_parses_only_the_documented_value() {
    assert!(super::parse_kill(Some("0")));
    assert!(!super::parse_kill(Some("1")));
    assert!(!super::parse_kill(Some("")));
    assert!(!super::parse_kill(None));
}

// ---------------------------------------------------------------------------
// Error surface
// ---------------------------------------------------------------------------

#[test]
fn bad_ids_and_lengths_are_refused() {
    let p = planted(16, 3, 2, 4, 64, 0x77);
    let (art, _) = fit_orthogonal(&p);
    let mut out = vec![0.0f32; art.dim];
    let f = vec![0.0f32; art.d];
    assert!(matches!(
        bind_into(&art, 9, &f, &mut out),
        Err(TprError::BadId { .. })
    ));
    let short = vec![0.0f32; art.d - 1];
    assert!(matches!(
        bind_into(&art, 0, &short, &mut out),
        Err(TprError::DimMismatch { .. })
    ));
    let mut core = vec![0.0f32; art.core_len()];
    let bad = TprBindings::from_pairs(&[(0, 99)]);
    assert!(matches!(
        core_encode_into(&art, &bad, &mut core),
        Err(TprError::BadId { .. })
    ));
}

