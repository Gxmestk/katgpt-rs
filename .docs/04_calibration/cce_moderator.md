# CCE Moderator — API Reference & Worked Examples

**Plan:** [295](../../.plans/295_lp_cce_moderator_primitive.md)
**Research:** [274](../../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md)
**Paper:** [arxiv 2606.20062](https://arxiv.org/pdf/2606.20062) — Campi, Cannerozzi, Tzouanas 2026
**Feature gate:** `cce_moderator` (**default-on**, Plan 295 + Plan 300)
**Crate:** `katgpt-rs/src/cce/`

---

## TL;DR

Generic, game-agnostic Coarse Correlated Equilibrium (CCE) primitives for
finite state-action games. Three public types:

- **`ExternalRegret`** — closed-form external-regret functional `ER(ρ) = max_κ (γ(ρ) − γ_dev(ρ, κ))`, plus uniqueness check (Assumption 6.2) and linear derivative (Lemma 6.5).
- **`CceLp`** — LP solver over occupation measures `ρ ∈ P(S × A)`. Finds the optimal CCE `ρ⋆ = argmin_{ρ ∈ CCE} γ₀(ρ)` via basic-feasible-solution enumeration.
- **`CcePrimalDual`** — Bregman primal-dual iterator with `O(N⁻¹ᐟ²)` averaged-iterate convergence (Euclidean potential = projected gradient descent).

All three are **modelless** (no backprop, no training), **generic** over
`<const N, const A>`, and contain **no game semantics** — the latent-space
reframing (state = HLA bucket, action = CGSP arm) lives in riir-ai Plan 325.

---

## Quick Start

```rust
use katgpt_rs::cce::{
    CceLp, CcePrimalDual, Deviation, DeviationClass, ExternalRegret,
    OccupationMeasure, PayoffTensor,
};

// 1. Define your game: impl PayoffTensor<N, A>.
struct MyGame;
impl PayoffTensor<4, 2> for MyGame {
    fn reward_follow(&self, state: usize, action: usize) -> f32 {
        // cost(s, a) — MINIMIZE convention.
        COST_MATRIX[state][action]
    }
    fn gamma0(&self, rho: &OccupationMeasure<4, 2>) -> f32 {
        self.gamma(rho) // default: moderator objective = player cost.
    }
}

// 2. Define the deviation class.
struct MyDevs { v: Vec<Deviation<4, 2>> }
impl DeviationClass<4, 2> for MyDevs {
    fn deviations(&self) -> &[Deviation<4, 2>] { &self.v }
}
let devs = MyDevs {
    v: vec![
        Deviation::<4, 2>::constant(0, 0),
        Deviation::<4, 2>::constant(1, 1),
    ],
};

// 3a. Solve for the optimal CCE via LP.
let rho_star = CceLp::new().solve(&devs, &MyGame).expect("LP feasible");
assert!(CceLp::new().is_cce(&rho_star, &devs, &MyGame, 1e-4));

// 3b. Or learn it online via primal-dual.
let report = CcePrimalDual::new::<4, 2>()
    .with_eta(0.05)
    .run(&devs, &MyGame, 10_000);
assert!((report.gamma0_avg - MyGame.gamma0(&rho_star)).abs() < 0.05);
```

---

## API Reference

### Core Types (`types.rs`)

#### `OccupationMeasure<const N, const A>`

A probability distribution over `S × A` (length `N·A`, row-major, sums to 1).

| Method | Description |
|---|---|
| `new(entries: Vec<f32>) -> Result<Self, OccupationMeasureError>` | Validate + construct. |
| `uniform() -> Self` | Uniform distribution `1/(N·A)` per entry. |
| `dirac(state, action) -> Self` | Point mass on one `(s, a)`. |
| `at(state, action) -> f32` | `ρ(s, a)`. |
| `marginal_state(state) -> f32` | `μ(s) = Σ_a ρ(s, a)`. |
| `flat_index(state, action) -> usize` | `(s, a) → s·A + a`. |

#### `Deviation<const N, const A>`

A fixed alternative policy `κ : S → P(A)`. Stored as `kernel: [[f32; A]; N]`.

| Constructor | Description |
|---|---|
| `constant(id, action)` | Always play `action` regardless of state. |
| `identity(id)` | Play the recommended action (requires `N == A`). |
| `from_kernel(id, kernel)` | Custom kernel (caller validates). |

#### `trait DeviationClass<N, A>`

A finite set of deviations `D = {κ₁, …, κ_K}`.

| Method | Description |
|---|---|
| `deviations(&self) -> &[Deviation<N, A>]` | Slice of all deviations. |
| `apply(κ, ρ) -> OccupationMeasure` (default) | Deviated measure `ρ'(s, a') = μ(s)·κ(s)[a']`. |

#### `trait PayoffTensor<N, A>`

The cost tensor. **Cost convention**: minimize.

| Method | Description |
|---|---|
| `reward_follow(s, a) -> f32` | Per-index cost `cost(s, a)`. **Required.** |
| `reward_deviate(s, κ) -> f32` (default) | `Σ_{a'} κ(s)[a']·cost(s, a')`. |
| `gamma(ρ) -> f32` (default) | `Γ(ρ) = Σ ρ·cost`. Cost of following. |
| `gamma_dev(ρ, κ) -> f32` (default) | `Γ_dev(ρ, κ) = Σ_s μ(s)·reward_deviate(s, κ)`. |
| `gamma0(ρ) -> f32` | Moderator objective `Γ₀`. **Required.** |
| `gamma0_coeff(s, a) -> f32` (default) | Per-index coefficient of `Γ₀` (default: `= reward_follow`). |

#### `trait TransitionKernel<N, A>` (Plan 569)

MDP transition dynamics for the transition-kernel-constrained CCE. Provides
`P(s'|s,a)` — the transition kernel. Used by `CceLp::solve_with_dynamics` to
add balance-equation rows enforcing stationary MDP consistency.

| Method | Description |
|---|---|
| `transition(s, a, s') -> f32` | `P(s'\|s,a)` — MUST sum to 1 over `s'` for each `(s, a)`. **Required.** |

### `ExternalRegret` (`external_regret.rs`)

Stateless regret evaluator. All methods take `&D` and `&P` per call.

```text
ER(ρ) = max_{κ ∈ D} (γ(ρ) − γ_dev(ρ, κ))
```

| Method | Returns | Notes |
|---|---|---|
| `er(ρ, d, p)` | `f32` | External regret. CCE condition: `ER ≤ 0`. |
| `best_deviation(ρ, d, p)` | `Option<&Deviation>` | Argmax κ. `None` if `D` empty. |
| `is_unique_maximizer(ρ, d, p, ε)` | `bool` | Assumption 6.2: top-2 gap > ε. |
| `linear_derivative(ρ, m_flat, d, p)` | `f32` | `∂ER/∂ρ[m]` per Lemma 6.5. |

**Convention**: `ER = 0` at Nash. `ER < 0` at strict CCE. `ER > 0` is NOT a CCE.

### `CceLp` (`lp.rs`)

LP-CCE solver via BFS enumeration + two-phase primal simplex + constraint generation.

| Method | Returns | Notes |
|---|---|---|
| `solve(d, p)` | `Result<OccupationMeasure, CceLpError>` | Optimal `ρ⋆ = argmin γ₀`. Auto-selects BFS (small LPs) or simplex (large). |
| `solve_with_dynamics(d, p, k)` | `Result<OccupationMeasure, CceLpError>` | Transition-kernel-constrained CCE (Plan 569). Adds `N-1` balance rows. |
| `solve_heterogeneous(game)` | `Result<OccupationMeasure, CceLpError>` | Full heterogeneous LP. Auto-selects BFS or simplex. |
| `solve_heterogeneous_cg(game)` | `Result<OccupationMeasure, CceLpError>` | **Constraint-generation solver** (Plan 572). Starts with no deviation constraints, iteratively adds the most-violated. Production path for large heterogeneous LPs (e.g., 2-player RPS at NA=81). |
| `solve_heterogeneous_cg_with_tolerance(game, ε)` | `Result<OccupationMeasure, CceLpError>` | CG with explicit convergence tolerance. |
| `is_cce(ρ, d, p, ε)` | `bool` | Verify `ER(ρ) ≤ ε`. |
| `is_heterogeneous_cce(ρ, game, ε)` | `bool` | Verify 2+ player CCE condition. |

**Auto-selection**: `solve` and `solve_heterogeneous` compute `C(n_vars, n_cons)`
and use BFS enumeration if `≤ 50_000` candidates (exact, fast), otherwise the
two-phase primal simplex (Plan 572). This is transparent — same signature,
same output contract.

**Constraint generation** (Plan 572): `solve_heterogeneous_cg` is the production
path for large heterogeneous LPs. It starts with no deviation constraints,
solves the relaxed LP, finds the most-violated deviation via the external-regret
separation oracle, adds it, and re-solves. Converges in `O(|active set|)`
iterations — typically far fewer than `Σ_i |D_i|` because only the binding
deviations end up active. At NA=81 (2-player RPS), converges in 4 iterations
(~2 ms).

**Complexity**: BFS is `O(C(N·A + |D|, 1 + |D|) · m³)` where `m = 1 + |D|`.
Exact for `N·A + |D| ≤ ~25`. The simplex is worst-case exponential (Bland's
rule) but never observed in practice on CCE-sized LPs. Both are deterministic.

**`solve_with_dynamics`** (Plan 569) adds `N-1` balance-equation rows enforcing
stationary MDP consistency: `ν(s') = Σ_{s,a} ρ(s,a)·P(s'|s,a)`. This closes the
free-state-distribution artifact (Issue 574 T4 PASS): on a 2-state MDP with
action-dependent transitions, the constrained CCE recovers the exact true MDP
optimum. Requires a `TransitionKernel<N, A>` impl providing `P(s'|s,a)`.

### `CcePrimalDual` (`primal_dual.rs`)

Bregman primal-dual iterator (Algorithm 1).

```text
ρ⁰ = uniform, λ⁰ = 0
for n = 1, 2, …:
    grad[m] = gamma0_coeff(m) + λⁿ⁻¹ · linear_derivative(m)
    ρⁿ = project_simplex(ρⁿ⁻¹ − η · grad)
    λⁿ = max(0, λⁿ⁻¹ + (1/√n) · ER(ρⁿ))
    ρ̄ⁿ = ((n−1)/n)·ρ̄ⁿ⁻¹ + (1/n)·ρⁿ
```

| Method | Returns |
|---|---|
| `new::<N, A>()` | Self (uniform init, λ=0, η=0.1). |
| `with_eta(η)` | Builder: override step size. |
| `with_initial_rho(ρ)` | Builder: override ρ⁰. |
| `step(d, p)` | `StepReport` (one iteration). |
| `run(d, p, n_steps)` | `ConvergenceReportRaw<N, A>` (averaged iterate + history). |

**Convergence**: averaged iterate `ρ̄ᴺ` satisfies `|γ₀(ρ̄ᴺ) − γ₀(ρ⋆)| = O(N⁻¹ᐟ²)`
and `ER(ρ̄ᴺ) ≤ O(N⁻¹ᐟ²)` (Theorem 6.1).

---

## Worked Example: Chicken Game

**Setup**: 2-player chicken, modeled as player-1-only CCE. State = (s₁, s₂)
joint recommendation (N=4). Action = a₁ ∈ {S, T} (A=2).

Reward matrix `R[a₁][s₂]`:

```text
        s₂=S  s₂=T
a₁=S     3     1
a₁=T     4     0
```

Cost = -reward (minimize convention).

### LP Solution

With `γ₀ = γ` (player 1 cost):

```text
ρ⋆ = δ_{(state (T,S), action T)}    (player 1 plays T against opponent S)
γ₀(ρ⋆) = -4    (reward 4)
```

With `γ₀ = -welfare` (welfare maximization):

```text
ρ⋆ = 0.5·δ_{(state (S,S), action S)} + 0.5·δ_{(state (S,T), action S)}
γ₀(ρ⋆) = -5.5    (welfare 5.5)
```

### Primal-Dual Convergence (Emission-Abatement, N=4, A=4)

| n | γ₀(ρ̄ⁿ) | gap to LP |
|---|---|---|
| 100 | 1.0799 | 0.0799 |
| 1000 | 1.0080 | 0.0080 |
| 10000 | 1.0008 | 0.0008 |

Empirical rate: `O(N⁻¹)` (slope -1.0), steeper than the paper's `O(N⁻¹ᐟ²)` worst-case bound.

---

## GOAT Gate Status

| Gate | Target | Status |
|---|---|---|
| G1 — CCE ≥ Nash | Welfare gain ≥ 5% on chicken + BoS | **PASS** (chicken +37.5%, BoS +108%) |
| G2 — Primal-dual convergence | gap < 0.05, ER ≤ 0.05, slope ≤ -0.3 | **PASS** (gap=0.0008, ER=0.00003, slope=-1.0) |
| G3 — Designer steering | Two Γ₀ → two different CCEs | **PASS** (selfish welfare 5.0 vs welfare-max 5.5) |
| G4 — Crowd-scale latency | < 50µs per NPC update | Pending (riir-ai Plan 325) |
| G5 — LatCal commitment | Bit-identical | Pending (riir-ai Plan 325) |

See `.benchmarks/029_cce_convergence.md` for G2 details, `tests/cce_vs_nash.rs` for G1, `examples/cce_demo.rs` for G3.

---

## Limitations

1. **Player-1-only CCE.** The deviation class `D` models only one player's deviations. Multi-player CCE (both players' constraints) requires extending `D` — deferred to riir-ai Plan 325. **For zero-sum games** (e.g., RPS), the single-player `solve` produces a trivial artifact that exploits the free state distribution (Issue 573/574/575). The fix is the 2-player CCE path: `is_heterogeneous_cce` with N=9, A=9 (joint recommendation state, joint play action) correctly rejects the artifact because player 2 can profitably deviate (Issue 575 Part B, T5 PASS). The `solve_heterogeneous_cg` constraint-generation solver (Plan 572) now SOLVES the 2-player CCE LP at NA=81 (~2 ms, 4 iterations), closing the verify-only gap. **General-sum 2-player CCE closure (Issue 577):** `solve_heterogeneous_cg` is verified on Chicken + PD (general-sum games with asymmetric per-player cost tensors) via CG-vs-BFS parity + CCE validity + no-profitable-deviation checks. Production consumers should route zero-sum AND general-sum 2-player games to `solve_heterogeneous_cg` + `is_heterogeneous_cce`.
2. **~~No dynamics~~ Dynamics available via `solve_with_dynamics` (Plan 569).** The base `solve` treats the state distribution as free. The `solve_with_dynamics` variant adds the stationary MDP balance equation (`ν(s') = Σ ρ(s,a)·P(s'|s,a)`), closing the free-state-distribution artifact on games with action-dependent transitions. For games with state-independent transitions (e.g., RPS), the balance equation reduces to a marginal constraint (Issue 573 T4a) and a richer deviation class also fails (Issue 575 Part A — the artifact is a fixed point of best-response). The correct fix for zero-sum games is the 2-player CCE path via `is_heterogeneous_cce` (Issue 575 Part B — player 2's profitable deviation rejects the artifact).
3. **~~BFS enumeration LP~~ Simplex + constraint generation shipped (Plan 572).** BFS enumeration remains the fast path for `C(n_vars, n_cons) ≤ 50_000`. For larger LPs, `solve` / `solve_heterogeneous` auto-select the two-phase primal simplex (Bland's rule, deterministic). `solve_heterogeneous_cg` adds constraint generation for the production 2-player case. No external LP solver dependency.
4. **Euclidean Bregman only.** KL potential (entropic mirror descent) is implemented in `bregman.rs` but not wired into `CcePrimalDual`.

---

## Cross-References

- **Plan**: [`katgpt-rs/.plans/295_lp_cce_moderator_primitive.md`](../../.plans/295_lp_cce_moderator_primitive.md)
- **Research**: [`katgpt-rs/.research/274_Optimal_CCE_Moderator_LP_No_Regret.md`](../../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md)
- **Private selling-point guide**: `riir-ai/.research/143_Latent_CCE_Moderator_Crowd_Emergent_Coordination.md`
- **Private runtime plan**: `riir-ai/.plans/325_latent_cce_moderator_runtime.md`
- **Benchmarks**: [`.benchmarks/029_cce_convergence.md`](../../.benchmarks/029_cce_convergence.md) (G2)
- **Tests**: `tests/cce_convergence.rs` (G2), `tests/cce_vs_nash.rs` (G1)
- **Example**: `examples/cce_demo.rs` (G3 designer steering)
