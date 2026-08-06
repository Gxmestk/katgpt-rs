# Research 468: Beckmann Transport Models — Divergence Constraint for CCE MFG Dynamics

> **Source:** Lee, Coeurdoux, Potaptchik, Du, Albergo, Vanden-Eijnden — *Beckmann Transport Models: From Autonomous Flows to One-Step Maps* [arXiv:2608.01692](https://arxiv.org/abs/2608.01692), May 2026
> **Date:** 2026-08-06
> **Status:** Active — **Gain** (not Super-GOAT; PoC-gated for GOAT promotion)
> **Related Research:** 296 (Stokes/DEC — codifferential IS the divergence operator), 271 (MIT 6.S184 flow-matching crosswalk), 219 (TNO → DEC parent), 274 (Optimal CCE in MFGs via LP), 371 (Mean-Field Regime Classifier — crowd-scale order parameters)
> **Related Plans:** 295 (LP-CCE Moderator — the shipped primitive with the documented MFG gap), 314 (Stokes Calculus Wrappers — `belief_mass_divergence`), 325-riir-ai (Latent CCE Runtime — shipped COMPLETE without closing the MFG gap)
> **Classification:** Public (katgpt-rs)

---

## TL;DR

BTM is a generative-modeling theory paper (autonomous flows, one-step maps, ImageNet FID). **Verdict on the paper's primary contribution: Pass** — we don't ship generative image models, learning the drift `b` / map `T` needs gradient descent (→ riir-train), and the EqM loss correction (FID 1.90→1.87) is irrelevant to game AI.

**But BTM's core equation — the Beckmann OT divergence constraint `∇·(νb) = μ₀ − μ₁` — is the modelless primitive our shipped CCE Moderator (Plan 295) explicitly documents as missing.** The CCE moderator's `Limitations` section (`.docs/04_calibration/cce_moderator.md` §Limitations #2) states verbatim: *"No dynamics. The LP treats the state distribution as free. MFG dynamics (occupation-measure flow constraints) are a Plan 325 follow-up."* Plan 325 (riir-ai) shipped COMPLETE (2026-06-22) without closing this gap. BTM's divergence equation IS the occupation-measure flow constraint in Beckmann OT form — and the DEC `codifferential` (`crates/katgpt-dec/src/`) already ships the operator.

**Verdict: Gain.** The actionable improvement: add a divergence-equation feasibility constraint to the CCE LP, restricting the occupation measure `ρ` to distributions reachable by valid transport from the initial distribution `μ₀`. This closes the "free state distribution" gap that causes the RPS trivial-CCE artifact (Bench 029: the LP exploits free state distribution to find a "CCE" that beats zero-sum baseline — shouldn't happen with honest dynamics).

**Distilled for katgpt-rs (modelless, inference-time):**
The Beckmann divergence constraint `∇·(νb) = μ₀ − ρ` (for some transport field `b` and occupation measure `ν > 0`) is a feasibility check on `ρ`. In the shipped DEC substrate, this is: `codifferential(νb) = μ₀ − ρ` — checkable via `belief_mass_divergence` (Plan 314). The CCE LP adds this as a constraint row: "the candidate `ρ` must be transport-feasible from `μ₀`". This is modelless (no gradient descent; it's a linear feasibility constraint on the LP).

---

## 1. Paper Core Findings

### 1.1 The BTM framework (§2)

BTM constructs generative transport via a **time-independent** drift `b(x)` satisfying the divergence equation:

```
∇·(νb) = μ₀ − μ₁                                    ... (9)
```

where `ν > 0` is a design knob (the FM case: `ν = ∫ρ_t dt`, the time-averaged interpolant density; the Poisson Flow case: `ν ≡ 1`). The autonomous flow `Ẋ_t = b(X_t)` transports `μ₀ → μ₁` when `μ₁` is singular (supported on a lower-dimensional manifold `M₁`).

The transport map `T(x₀) = X_τ(x₀)` (limit at first hitting of `M₁`) satisfies the **conservation equation**:

```
b(x) · ∇T(x) = 0  for x ∉ M₁,  T(x) = x for x ∈ M₁   ... (11)
```

This characterizes `T` as invariant along flow lines.

### 1.2 Beckmann's transportation problem (§2.3)

The divergence equation (9) IS the flux constraint of **Beckmann's transportation problem** (Beckmann 1952, Santambrogio 2015 §4.4):

```
minimize  ∫|b|²ν dx   subject to   ∇·(νb) = μ₀ − μ₁
```

BTM gives Beckmann's static OT formulation a **dynamical realization** (autonomous flow). This is the analog of Benamou-Brenier giving Monge-Kantorovich OT a dynamical realization.

### 1.3 Cost-chain inequality (§2.3, Eq 10, Appendix D)

```
W₂²(μ₀,μ₁)  ≤  ∫|b|²ν dx  ≤  ∫₀¹ ∫|b_t|² μ_t dt dx
```

The Beckmann action `∫|b|²ν` upper-bounds `W₂²` and is bounded by the FM Benamou-Brenier action.

### 1.4 One-step map learning (§2.6, Eq 15)

The map `T` can be learned directly via the Eulerian residual loss:

```
L(T) = E_{t,x₀,x₁} [ ‖T(I_t) − sg(T(I_t) + İ_t·∇T(I_t))‖² ]  +  λ E_{x₁}[‖T(x₁) − x₁‖²]
```

This is **training-based** (gradient descent on `T_θ`) — → riir-train. The structural insight (stop-gradient, predict-next-iterate) mirrors consistency / self-distillation patterns.

---

## 2. Distillation

### 2.1 What the paper's primary contribution means for us

BTM is a generative-modeling paper. We don't ship generative image/video models. The paper's contributions in that domain — autonomous flow matching, EqM loss correction (FID 1.90→1.87), one-step map learning on ImageNet 256×256 (17.58 FID without CFG) — are out of scope for katgpt-rs modelless inference. **This axis alone → Pass.**

### 2.2 The actionable connection: CCE Moderator's documented MFG gap

Our shipped **LP-CCE Moderator** (Plan 295, sourced from Campi/Cannerozzi/Tzouanas "Optimal CCEs in MFGs via LP + No-Regret Learning", arXiv:2606.20062) has a documented limitation:

> **`.docs/04_calibration/cce_moderator.md` §Limitations #2:**
> "No dynamics. The LP treats the state distribution as free. **MFG dynamics (occupation-measure flow constraints) are a Plan 325 follow-up.**"

> **`tests/cce_vs_nash.rs` L92-96:**
> "This is NOT a Nash comparison — it's a known artifact of the 1-shot model without dynamics. The honest-mediator constraint (or **MFG dynamics in riir-ai Plan 325**) would force the uniform state distribution and recover γ₀ = 0."

> **`.benchmarks/029_cce_moderator_goat.md` §G1 RPS:**
> "The LP exploits the free state distribution (concentrates on the most favorable (s,a) pair). Without dynamics or honest-mediator constraints, the 1-shot LP trivially finds a 'CCE' that beats the zero-sum baseline. This is a documented limitation — the fair comparison requires **MFG dynamics** (riir-ai Plan 325)."

**riir-ai Plan 325 shipped COMPLETE (2026-06-22)** — all 9 phases (HLA state adapter, zone mood signal, deviation class, faction moderator, designer steering, LatCal commitment, crowd-scale latency). **But it never added the MFG dynamics constraint.** The gap persists in shipped code: `ρ` is unconstrained (free state distribution).

### 2.3 BTM's divergence equation IS the missing constraint

The MFG forward equation (continuity equation) for a population distribution `μ_t` under velocity field `v_t`:

```
∂_t μ_t + ∇·(μ_t v_t) = 0
```

Integrated over the full time horizon, this yields the **Beckmann OT flux constraint**:

```
∇·(νb) = μ₀ − μ₁     where ν = ∫₀^∞ μ_t dt (occupation measure), b = v/‖v‖ (normalized velocity)
```

**This is exactly BTM's equation (9).** For our CCE: `μ₀` = initial state distribution, `μ₁ = ρ` = candidate CCE occupation measure. The constraint says: `ρ` must be reachable from `μ₀` by a valid transport — i.e., there must exist a transport field `b` and occupation measure `ν > 0` such that `∇·(νb) = μ₀ − ρ`.

In the shipped DEC substrate (Plan 251, Research 219/296), this is:

```
codifferential(ν · b) = μ₀ − ρ      ... checkable via belief_mass_divergence (Plan 314)
```

### 2.4 Why this closes the RPS trivial-CCE artifact

On RPS (zero-sum), the CCE LP currently exploits the free state distribution: it concentrates `ρ` on the single most-favorable `(s, a)` pair, producing a "CCE" that beats the zero-sum baseline (γ₀ < 0 for player 1's cost). This shouldn't happen — in a real game, the state distribution is constrained by the transition dynamics.

Adding the Beckmann divergence constraint: `ρ` must be transport-feasible from `μ₀`. A degenerate `ρ` concentrated on one point is NOT transport-feasible from a uniform `μ₀` unless the transport field `b` can move all mass to that point — which requires `b` to have a specific divergence structure that the constraint enforces. This should eliminate the trivial-CCE artifact.

### 2.5 Fusion — Beckmann Divergence × CCE × DEC Codifferential × Mean-Field Regime

| Component | Ships where | Role |
|---|---|---|
| `CceLp::solve` (LP over occupation measures) | katgpt-rs Plan 295 | The solver with the free-state-distribution gap |
| `codifferential` / `belief_mass_divergence` | katgpt-rs DEC (Plan 251/314) | The divergence operator (= `∇·`) |
| `MeanFieldOverlap` (κ, κ_a, Q order parameters) | katgpt-rs Plan 371 | Crowd-scale population state for `ν` |
| `ZoneMoodSignal` + `ζ` commitment | riir-ai Plan 325 | The broadcast medium the CCE operates over |

**Fusion:** `BeckmannFeasibleCce` — a CCE LP variant that adds a divergence-equation constraint row: `codifferential(ν · b_candidate) = μ₀ − ρ` for some `b_candidate`. This restricts `ρ` to transport-feasible distributions, closing the MFG dynamics gap.

### 2.6 Honest uncertainty

BTM's Beckmann formulation is ONE way to add dynamics; Campi et al. (our CCE source paper) use a **transition kernel** approach (`ρ(s') = Σ_s ρ(s)·P(s'|s,π(s))`). These are related but not identical:

| Formulation | Constraint | Continuous / discrete | Our substrate |
|---|---|---|---|
| Beckmann OT (BTM) | `∇·(νb) = μ₀ − ρ` | Continuous (cochain field) | DEC `codifferential` ✅ |
| Transition kernel (Campi) | `ρ(s') = Σ_s ρ(s)·P(s'\|s,π)` | Discrete (state-action) | NOT shipped as a CCE constraint |
| MFG continuity equation | `∂_t μ_t + ∇·(μ_t v_t) = 0` | Continuous, instantaneous | DEC `codifferential` (instantaneous form) ✅ |

The PoC (§3 below) must validate: does the Beckmann formulation actually close the RPS trivial-CCE artifact, or does it need the discrete transition-kernel form? If Beckmann suffices → Gain confirmed, plan for wiring. If not → the transition-kernel constraint is the right fix, and BTM's contribution is the theoretical lens (Stokes/DEC vocabulary for MFG dynamics) rather than the specific formulation.

### 2.7 Compute-unit translation

The paper frames its value as "learning `b` and `T` via gradient descent." Our compute unit for the CCE is different:

| Paper's compute unit | Our compute unit |
|---|---|
| Train `b_θ` via FM regression loss (Eq 5) | N/A — we don't train generative drifts |
| Train `T_θ` via Eulerian residual loss (Eq 15) | N/A — → riir-train if ever needed |
| ODE integration of `Ẋ_t = b(X_t)` at inference | `CceLp::solve` (LP over occupation measures) — a single linear solve, not ODE integration |
| Check `∇·(νb) = μ₀ − μ₁` | `codifferential` + `belief_mass_divergence` — a single DEC operator call |

The modelless analog of BTM's framework for our stack is: **use the divergence equation as a feasibility constraint on the CCE LP**, not as a training target or an ODE to integrate.

---

## 3. Verdict

**Gain.** Per §1.55: the mechanism (divergence equation) ships (DEC `codifferential`), and there is an actionable improvement (add the Beckmann divergence constraint to the CCE LP to close the documented MFG dynamics gap).

**Not GOAT yet** — needs a PoC (§3.6 defend-wrong rule). The PoC must prove that:
1. The Beckmann divergence constraint actually closes the RPS trivial-CCE artifact (the free-state-distribution exploitation).
2. The constraint doesn't over-restrict the feasible set (making the LP infeasible or degenerate).
3. The DEC `codifferential` on a discretized state space produces a valid linear constraint for the LP.

**Not Super-GOAT** — the divergence equation ships (DEC substrate); BTM's contribution is the Beckmann OT framing + the CCE connection, which is a fusion of existing primitives into a new constraint, not a new capability class. The selling point ("CCE with honest dynamics") strengthens the existing CCE moderator pillar rather than creating a new one.

### Routing

- **PoC** → `katgpt-rs/.issues/` (investigate Beckmann divergence constraint on the CCE LP)
- **If PoC passes** → `katgpt-rs/.plans/` (ship `BeckmannFeasibleCce` as a CCE LP variant, behind `beckmann_feasibility` feature flag)
- **Runtime wiring** → riir-ai (extend the `cce_runtime` from Plan 325 to use the Beckmann-feasible CCE)
- **Training aspects** (learning `b`/`T` for generative modeling) → riir-train (out of scope for this note)

### MOAT gate (§1.6)

- **katgpt-rs domain:** in-scope. Beckmann feasibility is a generic math primitive (divergence-constrained LP), no game/chain/shard semantics. Strengthens the CCE moderator (a katgpt-rs primitive) with a modelless dynamics constraint.
- **Force multiplier:** connects CCE moderator (Plan 295) + DEC substrate (Plan 251) + Mean-Field Regime (Plan 371) — ≥3 pillars.

---

## 4. Connection to the DEC / Stokes substrate

BTM's divergence equation `∇·(νb) = μ₀ − μ₁` is a specific instance of the general DEC identity. On a 2D grid `CellComplex`:

- `μ₀`, `μ₁` are rank-0 cochains (vertex densities — scalar fields on vertices)
- `νb` is a rank-1 cochain (edge flow — the transport current)
- `∇·(νb)` = `codifferential(νb)` = δ(νb) — a rank-0 cochain (vertex divergence)

The Beckmann constraint `δ(νb) = μ₀ − μ₁` says: the divergence of the transport current equals the source-sink difference. This is exactly what `belief_mass_divergence` (Plan 314) checks — except Plan 314 checks `δ(flow) ≈ 0` (the steady-state mass-conservation special case where `μ₀ = μ₁`). BTM generalizes to `μ₀ ≠ μ₁` (the transport case).

**The CCE constraint in DEC terms:** given `μ₀` (initial vertex density) and candidate `ρ` (CCE vertex density), check feasibility: does there exist an edge flow `j = νb` such that `δ(j) = μ₀ − ρ`? This is a linear feasibility problem on the edge-flow variables `j`, checkable in one `codifferential` solve.

---

## 5. What stays modelless vs what goes to riir-train

| Aspect | Track | Rationale |
|---|---|---|
| Beckmann divergence constraint on CCE LP | **Modelless** (katgpt-rs) | Linear feasibility check via DEC `codifferential`; no gradient descent |
| Transport-field `b` construction (given feasible `ρ`) | **Modelless** (katgpt-rs) | Solve `δ(j) = μ₀ − ρ` for `j` — a Poisson solve on the cochain |
| One-step map `T` learning (Eq 15) | **riir-train** | Gradient descent on `T_θ` |
| FM regression loss for drift `b` (Eq 5) | **riir-train** | Gradient descent on `b_θ` |
| EqM loss correction | **riir-train** | Image generation (FID), out of scope |

---

## 6. References

- **BTM paper:** [arXiv:2608.01692](https://arxiv.org/abs/2608.01692) — Lee et al., May 2026
- **CCE source paper:** [arXiv:2606.20062](https://arxiv.org/abs/2606.20062) — Campi, Cannerozzi, Tzouanas — Optimal CCEs in MFGs via LP + No-Regret Learning
- **DEC substrate:** Research 219 (TNO → DEC), Research 296 (Stokes vocabulary crosswalk), Plan 251 (DEC operators), Plan 314 (Stokes wrappers)
- **CCE moderator:** Plan 295 (LP-CCE primitive), Plan 300 (Subjective-CCE), riir-ai Plan 325 (runtime — shipped COMPLETE without MFG dynamics)
- **Mean-field:** Research 371 / Plan 371 (Mean-Field Regime Classifier)
- **Beckmann OT:** Beckmann, M. (1952) *A continuous model of transportation*, Econometrica 20(4):643–660; Santambrogio, F. (2015) *Optimal Transport for Applied Mathematicians* §4.4

---

## 7. PoC Addendum (Issue 573, 2026-08-06)

**Verdict: T4 FAIL — the Beckmann divergence feasibility constraint does NOT
close the RPS trivial-CCE artifact.**

### PoC setup

- RPS as MFG: N=3 states (opponent's action), A=3 actions (player 1's action).
- Unconstrained CCE LP reproduces the artifact: γ₀ = −1.0, ρ concentrated on
  the single most-favorable (state, action) pair.
- Two Beckmann constraint variants tested:
  - **T4a (marginal, isolated vertices):** forces ν = μ₀ = uniform. γ₀ = −1.0 —
    artifact persists. ρ spreads across all states (ν=1/3 each) but concentrates
    action mass on the best-response per state (R→P, P→S, S→R).
  - **T4b (edge-flow, connected path graph):** adds j⁺/j⁻ edge-flow variables +
    divergence constraint rows. γ₀ = −1.0 — artifact persists, ρ identical to
    unconstrained (the constraint is vacuous: any ν is transport-reachable on a
    connected graph).
- **T4c:** DEC `codifferential` on `CellComplex::grid_2d(3,1)` verified correct —
  δ(j=(1,0)) = [−1, 1, 0], sums to 0 (mass conservation). The operator is right;
  the formulation (feasibility vs cost) is the issue.
- **T5 (chicken):** marginal constraint does NOT over-restrict — welfare
  unchanged (5.5 → 5.5). The constraint simply has no effect on chicken's CCE.

### Root cause analysis

The Beckmann divergence constraint `δ(j) = μ₀ − ν` operates on the **state
marginal** `ν(s) = Σ_a ρ(s,a)`, not on the full occupation measure `ρ(s,a)`. It
restricts which state distributions are transport-reachable from `μ₀`, but does
NOT restrict the action distribution within each state. The CCE LP independently
optimizes each state's action distribution, concentrating on the best-response
action — which is the actual exploitation mechanism that no state-marginal
transport constraint can prevent.

The original hypothesis (§2.4) was wrong: it claimed "a degenerate ρ
concentrated on one point is NOT transport-feasible from a uniform μ₀." In fact,
on any connected graph, ALL probability distributions are mutually
transport-reachable — the edge flow just routes mass to the target.

### What WOULD close the artifact

1. **Richer deviation class** — state-dependent deviations that play
   best-response: "when state = Scissors, play Rock." The CCE constraint
   would then catch the best-response exploitation.
2. **Transition-kernel constraint** (Campi et al.) — `ρ(s',a') =
   Σ_s ρ(s,a)·P(s'|s,a)`. This couples state and action, preventing independent
   per-state optimization.
3. **Honest-mediator constraint** — both players' deviation classes, not just
   player 1's.

### Verdict per T6 branch

**T4 FAIL → Beckmann formulation doesn't close the artifact; the gap needs the
discrete transition-kernel form.** BTM's value for our stack is the **theoretical
lens** (DEC/Stokes vocabulary for MFG dynamics — `codifferential` IS the
divergence operator, T4c-verified), NOT the specific Beckmann OT feasibility
formulation. No plan for `BeckmannFeasibleCce` — the constraint is vacuous or
insufficient depending on topology.

### PoC artifact

[`tests/beckmann_cce_poc.rs`](../tests/beckmann_cce_poc.rs) — 6 tests, gated on
`cce_moderator + dec_operators`. Kept as a reproducible negative-result artifact
(mirrors Plan 410's Go-arena failure pattern: a falsified hypothesis is still a
valuable research outcome).
