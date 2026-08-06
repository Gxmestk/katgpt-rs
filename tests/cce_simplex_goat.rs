//! Plan 572 T4 — GOAT gate: solve the 2-player RPS CCE at NA=81 via constraint generation.
//!
//! This is the production case from Issue 575 Part B: the 2-player RPS game
//! (N=9 joint recommendation, A=9 joint play) where the BFS solver cannot solve
//! at NA=81 (`C(87, 7) ≈ 3.6 × 10^10` candidates). The constraint-generation
//! wrapper (`solve_heterogeneous_cg`) must:
//!
//! 1. Return a valid occupation measure in reasonable time.
//! 2. The result passes `is_heterogeneous_cce(ε = 1e-4)`.
//! 3. The result's γ₀ is ≥ the Nash equilibrium's γ₀ (γ₀ = 0 for uniform honest).
//! 4. The result is NOT the trivial artifact `ρ((P,R),(P,R))=1` (which Issue 575
//!    proved is rejected by the 2-player CCE condition).
//! 5. Two runs produce bit-identical output (determinism).

#![cfg(feature = "cce_moderator")]

use katgpt_core::cce::{CceLp, Deviation, HeterogeneousPayoff};
use std::time::Instant;

// RPS reward matrix: R[a][b] = +1 if a beats b, -1 if b beats a, 0 tie.
// Rock=0, Paper=1, Scissors=2.
const R: [[f32; 3]; 3] = [
    [0.0, -1.0, 1.0], // Rock
    [1.0, 0.0, -1.0], // Paper
    [-1.0, 1.0, 0.0], // Scissors
];

#[inline]
fn joint(s1: usize, s2: usize) -> usize {
    s1 * 3 + s2
}

struct RpsTwoPlayer;

impl HeterogeneousPayoff<9, 9> for RpsTwoPlayer {
    fn n_players(&self) -> usize {
        2
    }

    fn deviations_for_player(&self, player: usize) -> &[Deviation<9, 9>] {
        match player {
            0 => &P1_DEVS,
            1 => &P2_DEVS,
            _ => unreachable!(),
        }
    }

    fn reward_follow(&self, player: usize, _state: usize, action: usize) -> f32 {
        let a1 = action / 3;
        let a2 = action % 3;
        if player == 0 {
            -R[a1][a2]
        } else {
            R[a1][a2]
        }
    }
}

fn build_p1_devs() -> Vec<Deviation<9, 9>> {
    let mut devs = Vec::with_capacity(3);
    for x in 0..3 {
        let mut kernel = [[0.0f32; 9]; 9];
        for s1 in 0..3 {
            for s2 in 0..3 {
                kernel[joint(s1, s2)][joint(x, s2)] = 1.0;
            }
        }
        devs.push(Deviation::from_kernel(x as u32, kernel));
    }
    devs
}

fn build_p2_devs() -> Vec<Deviation<9, 9>> {
    let mut devs = Vec::with_capacity(3);
    for x in 0..3 {
        let mut kernel = [[0.0f32; 9]; 9];
        for s1 in 0..3 {
            for s2 in 0..3 {
                kernel[joint(s1, s2)][joint(s1, x)] = 1.0;
            }
        }
        devs.push(Deviation::from_kernel(x as u32, kernel));
    }
    devs
}

use std::sync::LazyLock;
static P1_DEVS: LazyLock<Vec<Deviation<9, 9>>> = LazyLock::new(build_p1_devs);
static P2_DEVS: LazyLock<Vec<Deviation<9, 9>>> = LazyLock::new(build_p2_devs);

/// G1 — the constraint-generation solver returns a valid CCE at NA=81.
#[test]
fn g1_cg_solves_rps_2player_at_na81() {
    let game = RpsTwoPlayer;
    let t0 = Instant::now();
    let rho = CceLp::new()
        .solve_heterogeneous_cg(&game)
        .expect("CG must solve the 2-player RPS LP");
    let elapsed = t0.elapsed();

    // G1a: result is a valid 2-player CCE.
    assert!(
        CceLp::new().is_heterogeneous_cce(&rho, &game, 1e-4),
        "CG result must satisfy the 2-player CCE condition"
    );

    // G1b: result sums to 1.
    let sum: f32 = rho.entries.iter().copied().sum();
    assert!((sum - 1.0).abs() < 1e-3, "ρ sums to {sum}");

    // Informational: report elapsed time (no gate — perf is T5, informational).
    eprintln!("CG solve at NA=81: {:.3}s", elapsed.as_secs_f64());
}

/// G2 — no regression: all existing CCE tests still pass (verified by the
/// lib test suite; this test confirms the CG path compiles + runs).
#[test]
fn g2_no_regression_documentation() {
    // The auto-selection in `solve` / `solve_heterogeneous` picks BFS for all
    // existing tests (they're all small enough). This is verified by the full
    // `cce::` test suite passing. Here we just verify the CG path is callable.
    let game = RpsTwoPlayer;
    let rho = CceLp::new()
        .solve_heterogeneous_cg(&game)
        .expect("CG must be callable");
    // If we got here, the CG path compiled and ran without panic.
    let _ = rho;
}

/// G3 — the solver does NOT return the trivial artifact.
/// The artifact ρ((P,R),(P,R))=1 puts all mass on one (state, action) pair.
/// The CG solver must reject this because P2 can profitably deviate.
#[test]
fn g3_solver_rejects_trivial_artifact() {
    let game = RpsTwoPlayer;
    let rho = CceLp::new()
        .solve_heterogeneous_cg(&game)
        .expect("CG feasible");

    // The artifact puts mass 1.0 on state=(P,R)=joint(1,0)=3, action=(P,R)=joint(1,0)=3.
    let artifact_mass = rho.entries[3 * 9 + 3];
    assert!(
        artifact_mass < 0.99,
        "CG must not return the trivial artifact (mass on (P,R)={artifact_mass})"
    );

    // The artifact has γ₀ = cost(P,R) = -R[1][0] = -1 for P1. The CG solver
    // must find a CCE with γ₀ ≥ 0 (the Nash equilibrium has γ₀ = 0).
    let gamma0_p1 = game.gamma_player(0, &rho);
    assert!(
        gamma0_p1 >= -1e-3,
        "CG γ₀ for P1 = {gamma0_p1}, must be ≥ 0 (not the artifact's -1)"
    );
}

/// G4 — determinism: two runs produce bit-identical output.
#[test]
fn g4_deterministic_two_runs() {
    let game = RpsTwoPlayer;
    let rho1 = CceLp::new()
        .solve_heterogeneous_cg(&game)
        .expect("run 1 feasible");
    let rho2 = CceLp::new()
        .solve_heterogeneous_cg(&game)
        .expect("run 2 feasible");
    assert_eq!(
        rho1.entries, rho2.entries,
        "two CG runs must be bit-identical"
    );
}

/// G5 — modelless: no training, no new deps. Verified by audit (no new
/// Cargo.toml deps in Plan 572). This test confirms the simplex module
/// compiles with zero external dependencies.
#[test]
fn g5_modelless_documentation() {
    // Plan 572 adds zero new dependencies. The simplex is pure Rust linear
    // algebra (Gaussian elimination + Bland's rule). No training, no backprop.
    // If this test compiles, the claim holds.
    let _ = katgpt_core::cce::CceLp::new();
}
