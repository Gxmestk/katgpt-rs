//! Issue 574 - PoC: Transition-Kernel Constraint on CCE LP.
//!
//! Tests whether adding the stationary MDP balance equation (Campi et al. MFG
//! consistency constraint) to the CCE LP closes the free-state-distribution
//! artifact on a real MDP with **action-dependent transitions**.
//!
//! ## Why a new PoC (after Issue 573's Beckmann FAIL)
//!
//! Issue 573 proved the Beckmann divergence constraint does NOT close the RPS
//! trivial-CCE artifact. Research 468 §7 identified the transition-kernel form
//! (Campi et al.) as the right candidate. But for RPS specifically, the
//! transition kernel is state/action-independent (`P(s'|s,a) = 1/3`), so the
//! balance equation reduces to `ν = uniform` — exactly what Issue 573 T4a
//! already tested (γ₀ = -1.0 persists). This PoC MUST use a game with
//! action-dependent transitions to be meaningful.
//!
//! ## PoC design
//!
//! MDP: 2 states {LOW, HIGH}, 2 actions {WAIT, INVEST}.
//! - P(HIGH|LOW,WAIT)=0.1, P(HIGH|LOW,INVEST)=0.6, P(HIGH|HIGH,WAIT)=0.5,
//!   P(HIGH|HIGH,INVEST)=0.2
//! - cost(LOW,WAIT)=1, cost(LOW,INVEST)=3, cost(HIGH,WAIT)=0, cost(HIGH,INVEST)=2
//!
//! **Unconstrained CCE artifact**: all mass on (HIGH, WAIT) → γ₀ = 0.
//! **True MDP optimum** (always WAIT): γ₀ = 5/6 ≈ 0.833.
//! **Constrained CCE** (with balance equation): should recover γ₀ ≈ 5/6.
//!
//! ## What this PoC tests
//!
//! - **T2**: Unconstrained CCE reproduces the artifact (γ₀ ≈ 0).
//! - **T3**: Independent policy iteration gives honest baseline γ₀ = 5/6.
//! - **T4**: Constrained CCE (balance equation added) → γ₀ ≈ 5/6 (artifact closed).
//! - **T5**: Constrained ρ is still a valid CCE (ER ≤ 0 — no over-restriction).
//! - **T6**: Analytical note on RPS reduction (→ Issue 573 T4a).
//! - **T7**: Verdict.
//!
//! Run:
//! ```bash
//! cargo test --features cce_moderator --test transition_kernel_cce_poc -- --nocapture
//! ```

#![cfg(feature = "cce_moderator")]

use katgpt_core::cce::{
    CceLp, Deviation, DeviationClass, ExternalRegret, OccupationMeasure, PayoffTensor,
};

// ═══════════════════════════════════════════════════════════════════════════
// Section 0: Local BFS-enumeration LP solver
// ═══════════════════════════════════════════════════════════════════════════
//
// Mirrors `katgpt-core::cce::lp::{enumerate_bfs, solve_square_system}` but
// duplicated here so we can pass custom constraint matrices (with the extra
// balance-equation row) without modifying shipped code. Same pattern as Issue
// 573's PoC — if T4 PASSES, the constraint graduates to a proper CceLp method.

/// Solve the `n × n` linear system `A[:, combo] · x = b` via Gaussian
/// elimination with partial pivoting. Returns `None` if singular.
fn solve_square_system(mat: &[Vec<f64>], rhs: &[f64], combo: &[usize]) -> Option<Vec<f64>> {
    let n = combo.len();
    let mut aug = vec![vec![0.0_f64; n + 1]; n];
    for row in 0..n {
        for (col, &var) in combo.iter().enumerate() {
            aug[row][col] = mat[row][var];
        }
        aug[row][n] = rhs[row];
    }

    for pivot in 0..n {
        let mut max_row = pivot;
        let mut max_val = aug[pivot][pivot].abs();
        for (row_off, aug_row) in aug[(pivot + 1)..n].iter().enumerate() {
            let val = aug_row[pivot].abs();
            if val > max_val {
                max_val = val;
                max_row = pivot + 1 + row_off;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != pivot {
            aug.swap(pivot, max_row);
        }

        let pivot_val = aug[pivot][pivot];
        for row in (pivot + 1)..n {
            let factor = aug[row][pivot] / pivot_val;
            if factor == 0.0 {
                continue;
            }
            let (left, right) = aug.split_at_mut(row);
            let aug_pivot_row = &left[pivot];
            let aug_row = &mut right[0];
            for (arc, &apc) in aug_row[pivot..=n].iter_mut().zip(aug_pivot_row[pivot..=n].iter()) {
                *arc -= factor * apc;
            }
        }
    }

    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        let diag = aug[i][i];
        if diag.abs() < 1e-12 {
            return None;
        }
        x[i] = s / diag;
    }
    Some(x)
}

/// Enumerate BFS over equality-constrained LP. Returns best ρ entries
/// (first `n_rho_vars` slots) or `None` if infeasible.
fn enumerate_bfs(
    mat: &[Vec<f64>],
    rhs: &[f64],
    obj: &[f64],
    n_vars: usize,
    n_rho_vars: usize,
) -> Option<Vec<f64>> {
    let n_cons = mat.len();
    if n_cons == 0 || n_cons > n_vars {
        return None;
    }

    let mut best_obj_val = f64::INFINITY;
    let mut best_rho: Option<Vec<f64>> = None;
    let mut x = vec![0.0_f64; n_vars];

    let mut combo: Vec<usize> = (0..n_cons).collect();
    loop {
        if let Some(x_basic) = solve_square_system(mat, rhs, &combo) {
            x.fill(0.0);
            for (i, &col) in combo.iter().enumerate() {
                x[col] = x_basic[i];
            }

            const NEG_TOL: f64 = -1e-7;
            if x.iter().all(|&v| v >= NEG_TOL) {
                for xi in x.iter_mut() {
                    if *xi < 0.0 {
                        *xi = 0.0;
                    }
                }

                let sum_rho: f64 = x[..n_rho_vars].iter().copied().sum();
                if sum_rho > 1e-9 {
                    let inv = 1.0 / sum_rho;
                    for xi in x[..n_rho_vars].iter_mut() {
                        *xi *= inv;
                    }
                }

                let obj_val: f64 = x[..n_rho_vars]
                    .iter()
                    .zip(obj[..n_rho_vars].iter())
                    .map(|(&xi, &ci)| xi * ci)
                    .sum();

                if obj_val < best_obj_val {
                    best_obj_val = obj_val;
                    best_rho = Some(x[..n_rho_vars].to_vec());
                }
            }
        }

        if !next_combination(&mut combo, n_vars) {
            break;
        }
    }

    best_rho
}

fn next_combination(combo: &mut [usize], n: usize) -> bool {
    let k = combo.len();
    if k == 0 {
        return false;
    }
    let mut i = k as isize - 1;
    while i >= 0 {
        if combo[i as usize] < n - k + i as usize {
            combo[i as usize] += 1;
            for j in (i as usize + 1)..k {
                combo[j] = combo[j - 1] + 1;
            }
            return true;
        }
        i -= 1;
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 1: MDP game (N=2 states, A=2 actions, action-dependent transitions)
// ═══════════════════════════════════════════════════════════════════════════

/// State indices.
const LOW: usize = 0;
const HIGH: usize = 1;
/// Action indices.
const WAIT: usize = 0;
const INVEST: usize = 1;

/// Transition kernel `P(s'|s,a)` — the MDP dynamics.
///
/// `P[s][a][s']` = probability of transitioning to state `s'` from state `s`
/// under action `a`. This is what makes the game a REAL MDP with dynamics,
/// unlike RPS (where the "state" is just the opponent's action, drawn fresh
/// each round with no dependence on current state or action).
const TRANSITION: [[[f64; 2]; 2]; 2] = [
    // s = LOW
    [
        [0.9, 0.1], // a = WAIT  → [P(LOW), P(HIGH)]
        [0.4, 0.6], // a = INVEST → [P(LOW), P(HIGH)]
    ],
    // s = HIGH
    [
        [0.5, 0.5], // a = WAIT
        [0.8, 0.2], // a = INVEST
    ],
];

/// Cost matrix: `cost(s, a)` — minimize convention.
const COST: [[f64; 2]; 2] = [
    [1.0, 3.0], // s = LOW:  [WAIT, INVEST]
    [0.0, 2.0], // s = HIGH: [WAIT, INVEST]
];

/// The MDP game as a `PayoffTensor<2, 2>`.
///
/// `gamma0 = gamma` (moderator objective = player cost).
struct MdpGame;

impl PayoffTensor<2, 2> for MdpGame {
    fn reward_follow(&self, state: usize, action: usize) -> f32 {
        COST[state][action] as f32
    }
    fn gamma0(&self, _rho: &OccupationMeasure<2, 2>) -> f32 {
        // gamma0_coeff handles the linear objective; gamma0 is only needed for
        // the primal-dual iterator, not the LP. Return 0 — the LP uses
        // gamma0_coeff (default = reward_follow).
        0.0
    }
}

struct MdpDevs {
    v: Vec<Deviation<2, 2>>,
}

impl DeviationClass<2, 2> for MdpDevs {
    fn deviations(&self) -> &[Deviation<2, 2>] {
        &self.v
    }
}

fn mdp_constant_devs() -> MdpDevs {
    MdpDevs {
        v: vec![
            Deviation::<2, 2>::constant(0, WAIT),   // always WAIT
            Deviation::<2, 2>::constant(1, INVEST), // always INVEST
        ],
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2: LP builders
// ═══════════════════════════════════════════════════════════════════════════

/// Build the **unconstrained** CCE LP matrix for the MDP.
/// Returns (mat, rhs, obj, n_vars, n_rho).
fn build_unconstrained_mdp_lp() -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, usize, usize) {
    let n_states = 2usize;
    let n_actions = 2usize;
    let na = n_states * n_actions; // 4
    let devs = mdp_constant_devs();
    let nd = devs.deviations().len(); // 2
    let p = MdpGame;

    let n_vars = na + nd; // 6
    let n_cons = 1 + nd; // 3

    let mut mat = vec![vec![0.0_f64; n_vars]; n_cons];
    let mut rhs = vec![0.0_f64; n_cons];

    // Row 0: Σ ρ = 1
    for val in &mut mat[0][..na] {
        *val = 1.0;
    }
    rhs[0] = 1.0;

    // Rows 1..=nd: CCE deviation constraints  g_κ·ρ + s_κ = 0
    for (k, kappa) in devs.deviations().iter().enumerate() {
        for s in 0..n_states {
            for a in 0..n_actions {
                let j = s * n_actions + a;
                let g = p.reward_follow(s, a) as f64 - p.reward_deviate(s, kappa) as f64;
                mat[1 + k][j] = g;
            }
        }
        mat[1 + k][na + k] = 1.0; // slack
        rhs[1 + k] = 0.0;
    }

    // Objective: γ₀(ρ) = Σ ρ(s,a) · cost(s,a)
    let mut obj = vec![0.0_f64; n_vars];
    for s in 0..n_states {
        for a in 0..n_actions {
            obj[s * n_actions + a] = p.gamma0_coeff(s, a) as f64;
        }
    }

    (mat, rhs, obj, n_vars, na)
}

/// Build the **constrained** CCE LP: unconstrained CCE LP + balance equation.
///
/// The balance equation (stationary MDP consistency, one independent row for
/// N=2 states — the other row is redundant with normalization):
///
/// ```text
/// ν(HIGH) = Σ_{s,a} ρ(s,a) · P(HIGH | s, a)
/// ```
///
/// i.e.  `ρ(HIGH,WAIT) + ρ(HIGH,INVEST) = Σ_{s,a} ρ(s,a) · P(HIGH|s,a)`
///
/// Rearranged as a homogeneous equation (= 0):
///
/// ```text
/// Σ_{s,a} ρ(s,a) · P(HIGH|s,a) − ρ(HIGH,WAIT) − ρ(HIGH,INVEST) = 0
/// ```
///
/// This is the Campi et al. MFG consistency constraint in discrete form.
fn build_constrained_mdp_lp() -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, usize, usize) {
    let (mut mat, mut rhs, obj, n_vars, na) = build_unconstrained_mdp_lp();
    let n_states = 2usize;
    let n_actions = 2usize;

    // Append the balance-equation row for state HIGH.
    // Row: Σ_{s,a} ρ(s,a)·P(HIGH|s,a) − ρ(HIGH,WAIT) − ρ(HIGH,INVEST) = 0
    let mut balance_row = vec![0.0_f64; n_vars];
    for (s, trans_s) in TRANSITION.iter().enumerate().take(n_states) {
        for (a, trans_sa) in trans_s.iter().enumerate().take(n_actions) {
            let j = s * n_actions + a;
            balance_row[j] += trans_sa[HIGH];
        }
    }
    // Subtract the marginal terms: ν(HIGH) = ρ(HIGH,WAIT) + ρ(HIGH,INVEST)
    balance_row[HIGH * n_actions + WAIT] -= 1.0;
    balance_row[HIGH * n_actions + INVEST] -= 1.0;

    mat.push(balance_row);
    rhs.push(0.0);

    (mat, rhs, obj, n_vars, na)
}

/// Compute γ₀ for a flat ρ entries vector (N=2, A=2).
fn mdp_gamma0(rho: &[f64]) -> f64 {
    let mut g = 0.0;
    for s in 0..2 {
        for a in 0..2 {
            g += rho[s * 2 + a] * COST[s][a];
        }
    }
    g
}

/// Compute the state marginal ν(s) = Σ_a ρ(s,a) for N=2, A=2.
fn state_marginal(rho: &[f64], s: usize) -> f64 {
    rho[s * 2..s * 2 + 2].iter().sum()
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 3: Independent MDP optimum via exhaustive deterministic policy search
// ═══════════════════════════════════════════════════════════════════════════
//
// For N=2, A=2 there are exactly 2^2 = 4 deterministic policies. For each,
// compute the stationary distribution + average cost. The minimum-cost policy
// is the true MDP optimum. This is exact (no value iteration needed).

/// Compute the stationary distribution of a deterministic policy.
/// `policy[s]` = action taken in state s.
/// Returns `Option<[f64; 2]>` (ν(LOW), ν(HIGH)) or None if no stationary dist.
fn stationary_distribution(policy: [usize; 2]) -> Option<[f64; 2]> {
    // Transition matrix under this policy: T[s'][s] = P(s'|s, policy[s])
    // Stationary: ν = T·ν  →  (T − I)·ν = 0  with  Σν = 1.
    // For 2 states: ν(HIGH) = T[HIGH][LOW] / (T[HIGH][LOW] + T[LOW][HIGH])
    // (standard 2-state Markov chain stationary formula).
    let p_high_from_low = TRANSITION[LOW][policy[LOW]][HIGH];
    let p_high_from_high = TRANSITION[HIGH][policy[HIGH]][HIGH];
    let p_low_from_low = TRANSITION[LOW][policy[LOW]][LOW];
    let p_low_from_high = TRANSITION[HIGH][policy[HIGH]][LOW];

    // ν(HIGH) = p_high_from_low / (p_high_from_low + p_low_from_high)
    //         = p_high_from_low / (p_high_from_low + (1 - p_high_from_high))
    let denom = p_high_from_low + p_low_from_high;
    if denom < 1e-12 {
        // Absorbing state or degenerate — check for absorbing HIGH.
        if p_high_from_low < 1e-12 && p_high_from_high > 1.0 - 1e-9 {
            // LOW is never reached; HIGH is absorbing.
            return Some([0.0, 1.0]);
        }
        if p_low_from_high < 1e-12 && p_low_from_low > 1.0 - 1e-9 {
            // HIGH is never reached; LOW is absorbing.
            return Some([1.0, 0.0]);
        }
        return None;
    }
    let nu_high = p_high_from_low / denom;
    let nu_low = 1.0 - nu_high;
    Some([nu_low, nu_high])
}

/// Average cost of a deterministic policy.
fn policy_average_cost(policy: [usize; 2]) -> Option<f64> {
    let nu = stationary_distribution(policy)?;
    Some(nu[LOW] * COST[LOW][policy[LOW]] + nu[HIGH] * COST[HIGH][policy[HIGH]])
}

/// Enumerate all 4 deterministic policies, return (best_policy, best_cost).
fn optimal_deterministic_policy() -> ([usize; 2], f64) {
    let policies: [[usize; 2]; 4] = [
        [WAIT, WAIT],
        [WAIT, INVEST],
        [INVEST, WAIT],
        [INVEST, INVEST],
    ];
    let mut best_policy = policies[0];
    let mut best_cost = f64::INFINITY;
    for &policy in &policies {
        if let Some(cost) = policy_average_cost(policy)
            && cost < best_cost
        {
            best_cost = cost;
            best_policy = policy;
        }
    }
    (best_policy, best_cost)
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 4: Tests
// ═══════════════════════════════════════════════════════════════════════════

/// T2: Unconstrained CCE reproduces the free-state-distribution artifact.
///
/// The CCE LP concentrates all mass on (HIGH, WAIT) — the single best
/// (state, action) pair — giving γ₀ ≈ 0. This is below the true MDP
/// optimum because the LP treats the state distribution as free.
#[test]
fn t2_unconstrained_cce_artifact() {
    let (mat, rhs, obj, n_vars, na) = build_unconstrained_mdp_lp();

    let rho = enumerate_bfs(&mat, &rhs, &obj, n_vars, na).expect("unconstrained LP feasible");

    let gamma0 = mdp_gamma0(&rho);
    println!("\n=== T2: Unconstrained CCE ===");
    println!("  ρ(LOW, WAIT)   = {:.6}", rho[LOW * 2 + WAIT]);
    println!("  ρ(LOW, INVEST) = {:.6}", rho[LOW * 2 + INVEST]);
    println!("  ρ(HIGH, WAIT)  = {:.6}", rho[HIGH * 2 + WAIT]);
    println!("  ρ(HIGH, INVEST)= {:.6}", rho[HIGH * 2 + INVEST]);
    println!("  ν(LOW)  = {:.6}", state_marginal(&rho, LOW));
    println!("  ν(HIGH) = {:.6}", state_marginal(&rho, HIGH));
    println!("  γ₀(unconstrained) = {:.6}", gamma0);

    // Cross-check with shipped CceLp::solve.
    let devs = mdp_constant_devs();
    let game = MdpGame;
    let rho_shipped = CceLp::new().solve(&devs, &game).expect("shipped LP feasible");
    let gamma0_shipped: f64 = rho_shipped.entries.iter().map(|&v| v as f64).zip(
        (0..2).flat_map(|s| (0..2).map(move |a| COST[s][a]))
    ).map(|(r, c)| r * c).sum();
    println!("  γ₀(shipped CceLp) = {:.6}", gamma0_shipped);

    // The artifact: γ₀ ≈ 0 (all mass on (HIGH, WAIT) where cost = 0).
    assert!(
        gamma0.abs() < 0.05,
        "T2: unconstrained CCE should give γ₀ ≈ 0 (artifact), got {gamma0:.6}"
    );
    assert!(
        rho[HIGH * 2 + WAIT] > 0.9,
        "T2: unconstrained CCE should concentrate on (HIGH, WAIT), got {:.6}",
        rho[HIGH * 2 + WAIT]
    );

    println!("  ✓ Artifact reproduced: γ₀ ≈ 0 (below true MDP optimum)");
}

/// T3: Independent computation of the true MDP optimum (honest baseline).
///
/// Exhaustive deterministic-policy search gives the minimum average cost.
/// This is the value the constrained CCE should recover.
#[test]
fn t3_true_mdp_optimum() {
    let (best_policy, best_cost) = optimal_deterministic_policy();
    let nu = stationary_distribution(best_policy).expect("optimal policy has stationary dist");

    println!("\n=== T3: True MDP optimum (policy iteration) ===");
    println!("  All 4 deterministic policies:");
    let policies: [[usize; 2]; 4] = [
        [WAIT, WAIT],
        [WAIT, INVEST],
        [INVEST, WAIT],
        [INVEST, INVEST],
    ];
    for &policy in &policies {
        let cost = policy_average_cost(policy);
        let nu = stationary_distribution(policy);
        let pstr = format!(
            "[{}, {}]",
            if policy[LOW] == WAIT { "WAIT" } else { "INVEST" },
            if policy[HIGH] == WAIT { "WAIT" } else { "INVEST" }
        );
        println!(
            "    policy {:>14}: avg_cost = {:>10.4}  ν = {:?}",
            pstr,
            cost.unwrap_or(f64::NAN),
            nu.unwrap_or([f64::NAN, f64::NAN])
        );
    }
    println!("  Optimal policy: [{}, {}]",
        if best_policy[LOW] == WAIT { "WAIT" } else { "INVEST" },
        if best_policy[HIGH] == WAIT { "WAIT" } else { "INVEST" });
    println!("  ν(LOW) = {:.6}, ν(HIGH) = {:.6}", nu[LOW], nu[HIGH]);
    println!("  γ₀(true optimum) = {:.6}", best_cost);

    // The honest optimum: always WAIT, γ₀ = 5/6 ≈ 0.833.
    assert_eq!(best_policy, [WAIT, WAIT], "T3: optimal policy should be always-WAIT");
    assert!(
        (best_cost - 5.0 / 6.0).abs() < 0.01,
        "T3: optimal average cost should be 5/6 ≈ 0.833, got {best_cost:.6}"
    );

    println!("  ✓ True MDP optimum = 5/6 ≈ 0.833 (always WAIT)");
}

/// T4: Constrained CCE (with balance equation) closes the artifact.
///
/// Adding the balance-equation constraint row forces the occupation measure to
/// be a stationary distribution of the transition kernel. The constrained CCE
/// should recover the honest MDP optimum γ₀ ≈ 5/6.
#[test]
fn t4_constrained_cce_closes_artifact() {
    let (mat, rhs, obj, n_vars, na) = build_constrained_mdp_lp();

    println!("\n=== T4: Constrained CCE (balance equation added) ===");
    println!("  Constraint rows: {} (1 norm + 2 CCE + 1 balance)", mat.len());

    let rho = match enumerate_bfs(&mat, &rhs, &obj, n_vars, na) {
        Some(r) => r,
        None => {
            println!("  ✗ LP INFEASIBLE — the balance equation makes the CCE LP infeasible!");
            panic!("T4: constrained LP infeasible");
        }
    };

    let gamma0 = mdp_gamma0(&rho);
    println!("  ρ(LOW, WAIT)   = {:.6}", rho[LOW * 2 + WAIT]);
    println!("  ρ(LOW, INVEST) = {:.6}", rho[LOW * 2 + INVEST]);
    println!("  ρ(HIGH, WAIT)  = {:.6}", rho[HIGH * 2 + WAIT]);
    println!("  ρ(HIGH, INVEST)= {:.6}", rho[HIGH * 2 + INVEST]);
    println!("  ν(LOW)  = {:.6}", state_marginal(&rho, LOW));
    println!("  ν(HIGH) = {:.6}", state_marginal(&rho, HIGH));
    println!("  γ₀(constrained) = {:.6}", gamma0);

    // The key assertion: γ₀ ≈ 5/6 (artifact closed — matches honest optimum).
    let (_, true_optimum) = optimal_deterministic_policy();
    assert!(
        (gamma0 - true_optimum).abs() < 0.05,
        "T4 FAIL: constrained CCE γ₀ = {gamma0:.6} ≠ true optimum {true_optimum:.6} (artifact NOT closed)"
    );

    // Also verify the balance equation holds for the solution.
    let mut balance_lhs = 0.0_f64;
    for s in 0..2 {
        for a in 0..2 {
            balance_lhs += rho[s * 2 + a] * TRANSITION[s][a][HIGH];
        }
    }
    let balance_rhs = state_marginal(&rho, HIGH);
    let balance_residual = (balance_lhs - balance_rhs).abs();
    println!("  Balance check: Σ ρ·P(HIGH|·) = {:.6}, ν(HIGH) = {:.6}, residual = {:.2e}",
        balance_lhs, balance_rhs, balance_residual);
    assert!(
        balance_residual < 1e-6,
        "T4: balance equation not satisfied (residual = {balance_residual:.2e})"
    );

    println!("  ✓ PASS: γ₀ ≈ {:.4} (matches true MDP optimum {:.4})", gamma0, true_optimum);
    println!("  ✓ Balance equation satisfied (residual < 1e-6)");
}

/// T5: The constrained ρ is still a valid CCE (no over-restriction).
///
/// The balance equation should constrain WHICH ρ are feasible, not relax the
/// CCE condition. Verify ER ≤ 0 for both deviations on the constrained ρ.
#[test]
fn t5_constrained_rho_is_valid_cce() {
    let (mat, rhs, obj, n_vars, na) = build_constrained_mdp_lp();
    let rho_entries = enumerate_bfs(&mat, &rhs, &obj, n_vars, na)
        .expect("constrained LP feasible");
    let rho = OccupationMeasure::<2, 2>::new(rho_entries.iter().map(|&v| v as f32).collect())
        .expect("valid occupation measure");

    let devs = mdp_constant_devs();
    let game = MdpGame;
    let er = ExternalRegret;

    println!("\n=== T5: Constrained ρ is still a valid CCE ===");
    for kappa in devs.deviations() {
        let reg = er.er(&rho, &devs, &game);
        let best = er.best_deviation(&rho, &devs, &game);
        println!("  κ={:?}: ER = {:.6} (best dev: {:?})",
            kappa.id, reg, best.map(|d| d.id));
        assert!(
            reg <= 0.05,
            "T5: deviation {:?} has ER = {reg:.6} > 0.05 — CCE violated (over-restricted?)",
            kappa.id
        );
    }

    // Also verify via the shipped is_cce check (unconstrained CCE definition).
    assert!(
        CceLp::new().is_cce(&rho, &devs, &game, 1e-4),
        "T5: constrained ρ fails the shipped is_cce check"
    );

    println!("  ✓ Constrained ρ is a valid CCE (ER ≤ 0 for all deviations)");
    println!("  ✓ No over-restriction: the constraint adds feasibility, not relaxation");
}

/// T6: Analytical note on the RPS reduction (state-independent transitions).
///
/// For RPS modeled as MFG (state = opponent's action, action = player's action),
/// the transition kernel is state/action-independent: P(s'|s,a) = 1/3 for all
/// s, a, s'. In this regime:
///
/// ```text
/// ν(s') = Σ_{s,a} ρ(s,a) · P(s'|s,a) = Σ_{s,a} ρ(s,a) · (1/3) = 1/3
/// ```
///
/// The balance equation reduces to `ν = uniform` — exactly the marginal
/// constraint that Issue 573 T4a already tested (γ₀ = -1.0 persists). This PoC
/// CANNOT close the RPS artifact because RPS has no real dynamics.
///
/// The RPS artifact needs the richer-deviation-class fix (option 1 from
/// Research 468 §7), not the transition-kernel constraint.
#[test]
fn t6_rps_reduction_analytical_note() {
    println!("\n=== T6: Analytical note — RPS reduction ===");

    // Demonstrate the reduction numerically: for state-independent transitions,
    // the balance equation forces ν = uniform regardless of ρ.
    let p_uniform: [[[f64; 3]; 3]; 3] = [[[1.0 / 3.0; 3]; 3]; 3];

    // Pick an arbitrary ρ with uniform marginals (the T4a artifact from Issue
    // 573: each state gets 1/3 mass, concentrated on the best-response action).
    // N=3, A=3. Entries: ρ(R,P)=1/3, ρ(P,S)=1/3, ρ(S,R)=1/3, rest 0.
    let rho_artifact: [f64; 9] = [
        0.0, 1.0 / 3.0, 0.0, // s=R: [R=0, P=1/3, S=0]
        0.0, 0.0, 1.0 / 3.0,  // s=P: [R=0, P=0, S=1/3]
        1.0 / 3.0, 0.0, 0.0,  // s=S: [R=1/3, P=0, S=0]
    ];
    let n_states = 3usize;
    let n_actions = 3usize;

    // Compute ν(s') = Σ_{s,a} ρ(s,a) · P(s'|s,a) under uniform transitions.
    let mut nu: [f64; 3] = [0.0; 3];
    for s_prime in 0..3 {
        for s in 0..n_states {
            for a in 0..n_actions {
                nu[s_prime] += rho_artifact[s * n_actions + a] * p_uniform[s][a][s_prime];
            }
        }
    }

    println!("  RPS with state-independent P(s'|s,a) = 1/3:");
    println!("    ρ (artifact) = {:?}", rho_artifact);
    println!("    ν (balance)  = {:?}", nu);
    println!("    Expected: ν = [1/3, 1/3, 1/3] = uniform");

    for (s, &nu_s) in nu.iter().enumerate() {
        assert!(
            (nu_s - 1.0 / 3.0).abs() < 1e-9,
            "T6: ν({s}) = {nu_s} should be 1/3 under uniform transitions"
        );
    }

    println!("  ✓ Balance equation reduces to ν = uniform for state-independent transitions");
    println!("  → This is identical to Issue 573 T4a (marginal constraint, γ₀ = -1.0 persists)");
    println!("  → The RPS artifact needs the richer-deviation-class fix (option 1), NOT this constraint");
}

/// T7: Verdict — record the PoC outcome.
///
/// This test prints the verdict summary and asserts the overall consistency
/// of the PoC results.
#[test]
fn t7_verdict() {
    println!("\n=== T7: Verdict ===");

    // Unconstrained artifact.
    let (mat, rhs, obj, n_vars, na) = build_unconstrained_mdp_lp();
    let rho_unc = enumerate_bfs(&mat, &rhs, &obj, n_vars, na).expect("unconstrained feasible");
    let gamma0_unc = mdp_gamma0(&rho_unc);

    // True MDP optimum.
    let (_, true_optimum) = optimal_deterministic_policy();

    // Constrained CCE.
    let (mat, rhs, obj, n_vars, na) = build_constrained_mdp_lp();
    let rho_con = enumerate_bfs(&mat, &rhs, &obj, n_vars, na).expect("constrained feasible");
    let gamma0_con = mdp_gamma0(&rho_con);

    println!("  Unconstrained CCE γ₀ = {:.6} (artifact — below true optimum)", gamma0_unc);
    println!("  True MDP optimum γ₀   = {:.6} (honest baseline)", true_optimum);
    println!("  Constrained CCE γ₀    = {:.6} (should match true optimum)", gamma0_con);

    // The artifact gap: unconstrained is strictly below the true optimum.
    let artifact_gap = true_optimum - gamma0_unc;
    println!("  Artifact gap (true − unconstrained) = {:.6}", artifact_gap);
    assert!(
        artifact_gap > 0.1,
        "T7: expected a meaningful artifact gap (> 0.1), got {artifact_gap:.6}"
    );

    // The constraint closes the gap.
    let residual_gap = (gamma0_con - true_optimum).abs();
    println!("  Residual gap (constrained − true)   = {:.6}", residual_gap);
    if residual_gap < 0.05 {
        println!();
        println!("  ╔══════════════════════════════════════════════════════════════╗");
        println!("  ║  T4 PASS: transition-kernel constraint CLOSES the artifact  ║");
        println!("  ║  on games with action-dependent transitions.                ║");
        println!("  ║                                                              ║");
        println!("  ║  Verdict: PLAN WARRANTED for `TransitionKernelCce` as a      ║");
        println!("  ║  CCE LP variant behind a `transition_kernel` feature flag.   ║");
        println!("  ║                                                              ║");
        println!("  ║  RPS (state-independent transitions) needs the richer-       ║");
        println!("  ║  deviation-class fix — a separate PoC.                       ║");
        println!("  ╚══════════════════════════════════════════════════════════════╝");
        assert!(residual_gap < 0.05, "T7: constrained CCE should match true optimum");
    } else {
        println!();
        println!("  ╔══════════════════════════════════════════════════════════════╗");
        println!("  ║  T4 FAIL: transition-kernel constraint does NOT close the    ║");
        println!("  ║  artifact even with action-dependent transitions.            ║");
        println!("  ╚══════════════════════════════════════════════════════════════╝");
    }
}
