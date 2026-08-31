//! Issue 687 — Orthogonal Factorization Primitives GOAT gate.
//!
//! G1   determinism ×3 bit-identical (GS output bits, defect bits, hinge
//!      mean/per-coord bits, Parseval residual bits) + the Hadamard dyadic
//!      exactness anchors: Parseval residual EXACTLY 0.0, recompose
//!      bit-exact == z, basis defect EXACTLY 0.0 (integer-core arithmetic —
//!      the cross-platform bit-identity witness).
//! G2   GS < 5µs @ d=64/K=14 (the 14 planner drive dir_vecs shape); hinge
//!      observe amortized ns/sample over a 1000-sample window at both the
//!      K=8×r=8 (Kr=64 latent) and K=14×r=1 (drive) shapes; head
//!      conditioning timing informational (construction-time by design).
//! G8   falsifiable negative controls: planted near-parallel pair ⇒ input
//!      defect fires (≥0.1, ≥100× healthy) + GS decorrelates (|cos| < 1e-6);
//!      planted dead channel ⇒ hinge fires EXACTLY on that coordinate with
//!      value == γ.
//!
//! G4 (zero steady-state alloc) lives in the module's own lib test — the
//! lib test binary installs the TrackingAllocator (bench_681 convention).
//!
//! std::time::Instant + harness=false (repo bench convention).
//!
//! Run: cargo bench -p katgpt-core --bench bench_687_orthogonal_factorization_goat
//!      (or `cargo test --release --features orthogonal_factorization
//!       --bench bench_687_orthogonal_factorization_goat -- --nocapture`)

use katgpt_core::orthogonal_factorization::head_conditioning;
use katgpt_core::orthogonal_factorization::{
    FactorActivityScratch, GAMMA_FAC_MIN, GAMMA_SCHED_C, factor_activity_hinge, gamma_schedule,
    hadamard_factorize, orthogonality_defect, orthonormalize_into, parseval_energy_check,
    recompose_into,
};
use katgpt_core::spectral_pencil::DenseScratch;
use katgpt_core::types::Rng;

const D: usize = 64;
const K14: usize = 14;

fn rng_dirs<const DD: usize>(k: usize, seed: u64) -> Vec<[f32; DD]> {
    let mut rng = Rng::new(seed);
    (0..k)
        .map(|_| {
            let mut v = [0.0_f32; DD];
            for s in v.iter_mut() {
                *s = rng.normal();
            }
            v
        })
        .collect()
}

fn max_abs_pair_cos<const DD: usize>(basis: &[[f32; DD]]) -> f32 {
    let mut m = 0.0_f32;
    for i in 0..basis.len() {
        let ni: f64 = basis[i].iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
        if ni == 0.0 {
            continue;
        }
        for j in (i + 1)..basis.len() {
            let nj: f64 = basis[j].iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
            if nj == 0.0 {
                continue;
            }
            let d: f64 = basis[i]
                .iter()
                .zip(basis[j].iter())
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            m = m.max((d / (ni * nj).sqrt()).abs() as f32);
        }
    }
    m
}

/// Dyadic z (multiples of 0.25, |z| ≤ 4) — every Hadamard intermediate is a
/// dyadic rational with ≤ 22 significand bits: exact in f32/f64.
fn dyadic_z<const DD: usize>() -> [f32; DD] {
    let mut z = [0.0_f32; DD];
    for (j, s) in z.iter_mut().enumerate() {
        *s = (((j * 7) % 33) as f32 - 16.0) * 0.25;
    }
    z
}

/// The planted fixture: an orthonormal 14-set with row 13 replaced by a
/// near-copy of row 0 (returns the healthy control too).
fn planted_and_healthy() -> ([[f32; D]; K14], [[f32; D]; K14]) {
    let base = rng_dirs::<D>(K14, 42);
    let mut healthy = [[0.0_f32; D]; K14];
    let mut d0 = 0.0_f32;
    orthonormalize_into(&base, &mut healthy, &mut d0);
    let mut rng = Rng::new(43);
    let mut w = healthy[0];
    for s in w.iter_mut() {
        *s += 0.01 * rng.normal();
    }
    let n: f64 = w
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    for s in w.iter_mut() {
        *s = (f64::from(*s) / n) as f32;
    }
    let mut planted = healthy;
    planted[13] = w;
    (planted, healthy)
}

fn bits_of<const DD: usize>(rows: &[[f32; DD]]) -> Vec<u32> {
    rows.iter()
        .flat_map(|r| r.iter().map(|x| x.to_bits()))
        .collect()
}

fn main() {
    let mut failures = 0_usize;

    // ── G1: determinism ×3 + Hadamard exactness anchors ─────────────────
    println!("═══ G1 — determinism ×3 + dyadic exactness anchors ═══");
    {
        let (planted, _) = planted_and_healthy();
        let mut basis = [[0.0_f32; D]; D];
        hadamard_factorize(&mut basis);
        let z = dyadic_z::<D>();

        let mut run_bits: Vec<Vec<u32>> = Vec::new();
        let mut run_defects: Vec<u32> = Vec::new();
        let mut run_hinge_means: Vec<u32> = Vec::new();
        let mut run_hinge_per: Vec<Vec<u32>> = Vec::new();
        let mut run_residuals: Vec<u32> = Vec::new();
        for _run in 0..3 {
            let mut out = [[0.0_f32; D]; K14];
            let mut defect = 0.0_f32;
            orthonormalize_into(&planted, &mut out, &mut defect);
            run_bits.push(bits_of(&out));
            run_defects.push(defect.to_bits());

            // Dead-channel population (deterministic seed 7) — exercises
            // nonzero hinge values in the determinism check.
            let mut rng = Rng::new(7);
            let mut activity = FactorActivityScratch::new(8, 8);
            for _ in 0..256 {
                let mut s = [0.0_f32; 64];
                for (idx, v) in s.iter_mut().enumerate() {
                    *v = if idx == 3 * 8 + 5 { 2.0 } else { rng.normal() };
                }
                activity.observe_sample(&s);
            }
            let gamma = gamma_schedule(GAMMA_FAC_MIN, GAMMA_SCHED_C, activity.count());
            let mut per = [0.0_f32; 64];
            let rep = factor_activity_hinge(&activity, gamma, &mut per);
            run_hinge_means.push(rep.mean_hinge.to_bits());
            run_hinge_per.push(per.iter().map(|h| h.to_bits()).collect());

            let mut coeffs = [0.0_f32; D];
            let prep = parseval_energy_check(&z, &basis, &mut coeffs);
            run_residuals.push(prep.residual_rel.to_bits());
        }
        let det_ok = run_bits[0] == run_bits[1]
            && run_bits[1] == run_bits[2]
            && run_defects[0] == run_defects[2]
            && run_hinge_means[0] == run_hinge_means[2]
            && run_hinge_per[0] == run_hinge_per[2]
            && run_residuals[0] == run_residuals[2];
        println!(
            "determinism ×3 bit-identical: {} (defect={:e}, hinge_mean={:.6}, parseval_res={:e})",
            if det_ok { "PASS" } else { "FAIL" },
            f32::from_bits(run_defects[0]),
            f32::from_bits(run_hinge_means[0]),
            f32::from_bits(run_residuals[0])
        );
        if !det_ok {
            failures += 1;
        }

        // Exactness anchors — the dyadic witness.
        let mut coeffs = [0.0_f32; D];
        let prep = parseval_energy_check(&z, &basis, &mut coeffs);
        let mut rec = [0.0_f32; D];
        recompose_into(&basis, &coeffs, &mut rec);
        let recompose_exact = rec
            .iter()
            .zip(z.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let defect_exact = orthogonality_defect(&basis) == 0.0;
        let anchor_ok = prep.residual_rel == 0.0 && recompose_exact && defect_exact;
        println!(
            "Hadamard d=64 anchors: parseval residual == 0.0 exactly: {} | recompose bit-exact: {} | basis defect == 0.0: {}",
            prep.residual_rel == 0.0,
            recompose_exact,
            defect_exact
        );
        if !anchor_ok {
            failures += 1;
        }
    }

    // ── G2: latency ─────────────────────────────────────────────────────
    println!("═══ G2 — latency (release) ═══");
    {
        // GS @ d=64/K=14 — the drive-direction shape. Target < 5µs.
        const REPS: usize = 1000;

        let (planted, _) = planted_and_healthy();
        let mut out = [[0.0_f32; D]; K14];
        let mut defect = 0.0_f32;
        for _ in 0..100 {
            orthonormalize_into(&planted, &mut out, &mut defect);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            orthonormalize_into(&planted, &mut out, &mut defect);
        }
        let gs_ns = t0.elapsed().as_secs_f64() / REPS as f64 * 1e9;
        println!(
            "GS d=64/K=14: {gs_ns:.0} ns/call (target < 5000 ns) — {}",
            if gs_ns < 5000.0 { "PASS" } else { "FAIL" }
        );
        if gs_ns >= 5000.0 {
            failures += 1;
        }

        // Hinge observe amortized over a 1000-sample window.
        // Shape A: K=8×r=8 (Kr=64 — the factorized-latent shape).
        let mut rng = Rng::new(1234);
        let mut samples = vec![0.0_f32; 1000 * 64];
        for s in samples.chunks_mut(64) {
            for v in s.iter_mut() {
                *v = rng.normal();
            }
        }
        let mut activity = FactorActivityScratch::new(8, 8);
        let t1 = std::time::Instant::now();
        for s in samples.chunks(64) {
            activity.observe_sample(s);
        }
        let obs_a_ns = t1.elapsed().as_secs_f64() / 1000.0 * 1e9;
        let gamma = gamma_schedule(GAMMA_FAC_MIN, GAMMA_SCHED_C, activity.count());
        let mut per = [0.0_f32; 64];
        let _ = factor_activity_hinge(&activity, gamma, &mut per);
        println!(
            "hinge observe K=8×r=8:  {obs_a_ns:.0} ns/sample (1000 samples) — {}",
            if obs_a_ns < 5000.0 { "PASS" } else { "FAIL" }
        );
        if obs_a_ns >= 5000.0 {
            failures += 1;
        }

        // Shape B: K=14×r=1 (the drive-direction shape) — informational.
        let mut samples_b = vec![0.0_f32; 1000 * 14];
        for s in samples_b.chunks_mut(14) {
            for v in s.iter_mut() {
                *v = rng.normal();
            }
        }
        let mut activity_b = FactorActivityScratch::new(14, 1);
        let t2 = std::time::Instant::now();
        for s in samples_b.chunks(14) {
            activity_b.observe_sample(s);
        }
        let obs_b_ns = t2.elapsed().as_secs_f64() / 1000.0 * 1e9;
        println!("hinge observe K=14×r=1: {obs_b_ns:.0} ns/sample (informational)");

        // Head conditioning — construction-time by design; informational.
        {
            let heads: Vec<Vec<[f32; D]>> =
                (0..8).map(|k| rng_dirs::<D>(8, 500 + k as u64)).collect();
            let head_slices: Vec<&[[f32; D]]> = heads.iter().map(|h| h.as_slice()).collect();
            let mut norms = [0.0_f32; 8];
            let mut scratch = DenseScratch::new();
            let t3 = std::time::Instant::now();
            let cert = head_conditioning(head_slices, &mut norms, &mut scratch);
            let cond_us = t3.elapsed().as_secs_f64() * 1e6;
            println!(
                "head_conditioning 8 heads d=64 (construction-time, informational): \
                 {cond_us:.0} µs total, σ_max = {:.4} (head {})",
                cert.sigma_max, cert.worst_head
            );
        }
    }

    // ── G8: falsifiable negative controls ───────────────────────────────
    println!("═══ G8 — negative controls ═══");
    {
        // (a) planted near-parallel pair.
        let (planted, healthy) = planted_and_healthy();
        let d_healthy = orthogonality_defect(&healthy);
        let d_planted = orthogonality_defect(&planted);
        println!(
            "(a) near-parallel pair: healthy defect = {:.2e}, planted = {:.4} \
             (ratio {:.0}×; gates: > 0.1 and > 100× healthy)",
            d_healthy,
            d_planted,
            d_planted / d_healthy.max(1e-30)
        );
        let defect_ok = d_planted > 0.1 && d_planted > 100.0 * d_healthy;
        if !defect_ok {
            failures += 1;
            println!("  FAIL: planted pair must fire the input defect");
        }
        let mut out = [[0.0_f32; D]; K14];
        let mut dg = 0.0_f32;
        orthonormalize_into(&planted, &mut out, &mut dg);
        let cos = max_abs_pair_cos(&out);
        let survivor: f64 = out[13].iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
        println!(
            "    GS decorrelation: max |cos| = {cos:.2e} (gate < 1e-6), survivor ‖b‖² = {survivor:.6}"
        );
        if cos >= 1e-6 || (survivor - 1.0).abs() > 1e-6 {
            failures += 1;
            println!("  FAIL: GS must decorrelate and keep the survivor unit-norm");
        }

        // (b) planted dead channel — hinge fires EXACTLY on that coordinate.
        let mut rng = Rng::new(7);
        let mut activity = FactorActivityScratch::new(8, 8);
        for _ in 0..256 {
            let mut s = [0.0_f32; 64];
            for (idx, v) in s.iter_mut().enumerate() {
                *v = if idx == 3 * 8 + 5 { 2.0 } else { rng.normal() };
            }
            activity.observe_sample(&s);
        }
        let gamma = gamma_schedule(GAMMA_FAC_MIN, GAMMA_SCHED_C, activity.count());
        let mut per = [0.0_f32; 64];
        let rep = factor_activity_hinge(&activity, gamma, &mut per);
        let dead_ok = per[3 * 8 + 5].to_bits() == gamma.to_bits()
            && rep.worst_flat == 3 * 8 + 5
            && per.iter().enumerate().all(|(idx, &h)| {
                if idx == 3 * 8 + 5 {
                    h.to_bits() == gamma.to_bits()
                } else {
                    h == 0.0
                }
            });
        println!(
            "(b) dead channel (k=3, j=5): hinge = {} (== γ exactly: {}), worst_flat = {} \
             (gate 29), mean = {:.6}",
            per[3 * 8 + 5],
            per[3 * 8 + 5].to_bits() == gamma.to_bits(),
            rep.worst_flat,
            rep.mean_hinge
        );
        if !dead_ok {
            failures += 1;
            println!("  FAIL: hinge must fire exactly on the planted dead coordinate");
        }
    }

    println!();
    if failures == 0 {
        println!("Issue 687 GOAT: ALL GATES PASS (opt-in; promotion deferred to a consumer)");
    } else {
        println!("Issue 687 GOAT: {failures} FAILURES");
        std::process::exit(1);
    }
}
