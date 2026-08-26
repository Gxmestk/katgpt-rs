# Bench 682 — distributional_steering GOAT (Plan 577)

**Plan:** [577](../.plans/577_distributional_steering_primitive.md) ·
**Research:** [505](../.research/505_Mean_Field_Distributional_Steering.md) ·
**Paper:** [arXiv:2608.08770](https://arxiv.org/abs/2608.08770) (Howard & Nüsken, SPIGM @ ICML 2026)
**Date:** 2026-08-26 · **Feature:** `distributional_steering` (opt-in — see verdict)
**Module:** `crates/katgpt-core/src/distributional_steering.rs`

> **Renumber note:** the plan draft says `bench_577` / `.benchmarks/577_*`;
> 577 was already allocated (`577_emotion_direction_rank`, monotonic
> never-reuse rule; highwater 680). This bench is **682**. The plan file
> records the same note.

## Verdict table

| Gate | Criterion | Measured | Verdict |
|---|---|---|---|
| **G1 targeting** (promotion gate) | FK arm's optimality gap `Ĵ(μ̂)−Ĵ(μ*_ref)` minimized at λ=λ\*, both λ\*∈{5,10}, ≥2 noise schedules; gradient-only minimum elsewhere | λ\*=5: **2/2 schedules min at λ=5 ✓**; λ\*=10: 1/2 (sched=1 ✓ at 10; sched=0 lands at 5, flat curve Δ=0.013 ≈ noise floor); separation claim **NOT reproduced** (gradient-only ≈ FK, gaps differ in the 3rd decimal) | **FAIL (partial, 3/4 FK conditions)** → stays opt-in |
| **G2 perf** | FK+Picard ≤ 1 µs / particle / step @ N=1000 (release, M3) | d=1: **9045 ns**/particle/step (9.05 ms/step); d=8: 15420 ns; gradient-only baseline 2314 ns → FK/grad **3.91×**; breakdown: exact O(N²) kernel build (10⁶ fast_exp) + K_FP=3 matvecs (3×10⁶ fma) + gradient pass | **FAIL** at the literal gate — see §G2 (the threshold is structurally infeasible for exact MMD at N=1000) |
| **G3 no-regression** | default lib count unchanged; feature-on lib green | default **1951 passed / 7 ignored** (module compiles out — opt-in ✓); feature-on **1969 passed / 0 failed** (+18 module tests) | **PASS** |
| **G4 alloc-free** | 0 steady-state allocs over 1000 steps @ N=1000 | **0 allocations** (release, 13.9 s; construction outside the measured region; debug-mode run exceeds 60 s at which point libtest's own slow-warning allocates +2 on the shared counter — harness noise, measured at step 481 — so the gate is debug-ignored + release-recorded) | **PASS** |
| **T3.6 determinism** | two-run bit-identity, fixed iteration order, no HashMap | states + weights **bit-identical** (pinned in-test) | **PASS** |
| T2.5 Picard≡Alg-3 | Ψ̇ Picard vs implicit linear system `(I−MW+(Mw)wᵀ)Ψ̇=(MW−(Mw)wᵀ)c`, M=−2λk | max_diff/scale < 5e-2 @ δt=1e-3, K_FP=200 | PASS |
| T1.5 variations | Ψ vs numerical functional differentiation (probe differences) | MMD + Moment rows within 5e-2 | PASS |
| μ\* machinery | λ=0 ≡ p₁; λ=10 shifts mass toward the 3:1 target | window ratio 0.467 → 1.372 ✓ | PASS |

**GOAT verdict: G1 FAIL ⇒ `distributional_steering` stays opt-in.** No Cargo.toml default flip requested.

## G1 — what reproduced, what didn't (honest record)

The paper's 1-D experiment (base GMM −1/+1 unit-var 1:3; target 3:1; MMD²
reward RBF bandwidth 5.0; `J = λ*·MMD² + KL` with leave-one-out KL at RBF
0.2; N=500, T=30, δt=0.05, 8 seeds, CRN across arms and λ; analytic μ\* by
damped grid fixed point, represented by 4096 stratified particles evaluated
through the same estimators — same-footing, so estimator bias cancels in
the λ-argmin).

**Reproduced:**

- λ\*=5, both schedules: FK gap minimized exactly at λ=5
  (gaps ×10⁴ sched=0: `[3077, 953, 208, 176, 432, 584, 650, 673]` — a clean
  V at λ=5; sched=1: `[3000, 1299, 435, −6, 50, 151, 226, 274]`).
- λ\*=10, σ=1.0: min at λ=10
  (`[7497, 3917, 1860, 332, 23, −12, 2, 22]` — min at index 5 = λ 10).
- The J trade-off structure is exactly the theory's shape: MMD²̂ falls
  monotonically with λ (0.028 → 0.0015) while KL̂ grows (0.19 → 0.36), and
  the sum's minimum sits where the objective coefficient says it should.

**Not reproduced:**

- λ\*=10 at σ=0.5: min at λ=5 (gap 0.0153 vs 0.0187 at λ=10 — the curve is
  flat to within the 8-seed noise floor ≈±0.01; the population saturates its
  achievable closeness to ν past λ≈5 in this regime).
- **The separation claim** (gradient-only minimum "elsewhere"): gradient-only
  ≈ FK everywhere (gaps agree to the 3rd decimal). In this 1-D broad-kernel
  Langevin regime the position steering does nearly all the work and the FK
  weights are a small correction; the paper's separation lives in their
  diffusion-sampler regime where the reverse process (not the steering)
  moves particles and the reweighting carries the distribution.

**Harness bugs found and fixed on the way to this record** (all documented
because they change how the numbers should be read):

1. **λ² steering** — `steering()` is already λ-scaled; the harness multiplied
   by λ again. Positions exploded to ±100 (exactly λ²-scale). Fixed.
2. **Grid μ\* solver oscillation** — geometric damping α=0.5 oscillates at
   λ=10 (the tilt feedback gain scales with λ); the A/B ratio landed at 0.35
   (the DOWN phase). α=0.1 converges (0.467 → 1.372 at λ=10). Every pre-fix
   G1 number used a wrong reference.
3. **Research 505 Table-2 sign slip** — the note transcribes the MMD row as
   `Ψ = 2∫k(x,y)(μ−ν)(dy)`; for `R = −MMD²` (higher better) the calculus
   gives `Ψ = 2[emb_ν − emb_μ]` (the tilt must be HIGHER near the target —
   pinned by the finite-difference test + the BoM adapter test, which
   caught it empirically: the target hypothesis was DOWN-weighted to
   6.7e-5 before the flip). Module implements the corrected sign with an
   in-source note; Research 505 stands corrected by this record.
4. **Ψ̇ position-transport contamination** — evaluating the second Ψ term at
   the OLD positions imports a `∇Ψ·ΔX ~ λ²|∇Ψ|²δt` term that explodes with
   λ (max|Ψ̇| measured up to 230, weights collapse to ESS 1). Paper Alg 4
   evaluates both terms at the same advanced positions (Ψ̇ = pure measure
   drift); the transport belongs in the `b·∇Ψ` FK term with b = FULL
   simulated drift (the Girsanov overshoot correction). Fixed in the module.
5. **Picard stability bound** — the iteration Jacobian norm is
   ≈ `2λ·E_w|k−emb|` ≈ 0.2λ in this bandwidth-5 regime: **divergent for
   λ≳5 at damping 1.0 regardless of K_FP**. The harness uses adaptive
   damping `α = min(1, 2/λ)` + K_FP=8 for λ>2 (the paper's own "damping for
   strong tilts" guidance, tuned to this regime). With it, max|Ψ̇| ≤ 7.8 at
   λ=15 and ESS stays ≥ 440/500 with ≤ 5 ESS-guard resamples. **Consumer
   guidance: damping must scale as O(1/λ) for broad kernels** — encoded in
   the stepper docs.
6. **Weights-only degeneracy** (a real property, not a bug): without
   resampling, FK weights degenerate to ESS→1 by λ≈7.5 (30 steps ×
   clip 1.0). The G1 harness uses ESS-triggered systematic resampling (the
   paper's own sampling-consumer protocol); **weights-only remains the
   correct persistent-agent mode** (Research 505 caveat 2) with the
   documented caveat that strong-λ long-horizon weights-only runs
   degenerate — bounded by the clip, harmless for the crowd use-case where
   weights feed salience, not ancestry.

## G2 — the honest breakdown

Per step at N=1000 (release, M3 Max): 9.05 ms (d=1) / 15.4 ms (d=8) =
**9.0 / 15.4 µs per particle per step**. The plan's sub-µs gate is
structurally infeasible for exact MMD steering at N=1000: the kernel build
alone is 10⁶ `fast_exp` evaluations (~2–5 ns each even vectorized through
`simd_exp_inplace`), and Picard adds K_FP matvecs of the cached matrix.
1 µs/particle = 1 ms/step = 1 ns per (kernel eval + 3 matvec fma + gradient
fma) — below the cost of a single transcendental. The paper's
"Picard = 0.036–0.24% of runtime" is relative to network evaluations; a
modelless stack has no network to amortize against — **the kernel build IS
the cost here.**

What the gate CAN honestly say:

- FK/gradient-only ratio **3.91×** — the FK+Picard machinery (Picard
  matvecs + weight path + embedding) costs a bounded multiple (< 10×,
  asserted) of the gradient-only baseline that shares the same kernel
  build. The two arms' per-step difference is the true marginal FK cost:
  ~6.7 ms/step at N=1000.
- Scaling is O(N²)/step exactly as the math says; N=300 runs ≈ 0.9 µs
  /particle (sub-µs at N≲300).
- Approximate kernels (random features / Nyström) would change the scaling
  class — out of scope, recorded as the reopen path.

## Module-level findings (unit-test-pinned)

- **K_FP=1 bias** (paper's own observation) requires a **nonzero drift
  driver**: with `b = 0` the candidate weights equal the old weights, Ψ̇ ≡ 0
  identically (correct math), and every K_FP is vacuous. With a driver,
  K_FP=1 vs 3 differ measurably in BOTH warm- and cold-start regimes
  (λ=30, damping 0.3: warm −0.0612 vs 0.0456).
- **Damping rescues strong λ**: residual at damping 0.5 ≤ damping 1.0 on
  the λ=40 strong-tilt fixture (pinned).
- **Hot-path gradient ≡ cold-path** (cached-kernel vs naive): agreement
  < 1e-4 across N=20 random populations (pinned — the sign-corrected
  analytic form validated twice).
- **BoM composition** (T4.1): real `LeakyIntegrator::sample_k_states` → 8
  hypotheses → FK tilt weights against an MMD reward centered on hypothesis
  0: w₀ > uniform, weights normalize (pinned; runs under
  `--all-features` since `bom_sampling` is default-on).
- **Tilt residual** (T2.4): weak-tilt settled residual < 0.05 (pinned) —
  the cheap convergence certificate for consumers.

## Demo (T4.2)

`cargo run -p katgpt-core --features distributional_steering --example
distributional_steering_demo --release`:

```text
base population   : N=600, 2-D GMM 1:3 (A=25% mass, B=75%)
target dial       : 3:1 (A=75%) — MMD² reward, γ=0.25, λ=6
BEFORE: MMD²=0.33114  cluster shares A/B = 0.240/0.760
AFTER : MMD²=0.01125  (3.4% of before)
         weighted cluster shares A/B = 0.669/0.331  (target 0.75/0.25)
         resampling events: 1 (ESS-guard at N/2)
         ESS = 522.0 / 600
         top-10 weights: 0.0040 0.0039 ... (uniform 0.0017)
```

The 2-D dial demonstrably works: 29× MMD² reduction, cluster mass shares
0.24/0.76 → 0.67/0.33 toward the 0.75/0.25 target, gentle weight
concentration (2.3× uniform) — the "who carries the distribution" read-out.

## Disposition

- **Stays opt-in** (`distributional_steering = []`, no default change). The
  quality axis (G1) is partially proven — λ\*=5 fully, λ\*=10 one of two
  schedules — and the perf axis (G2) misses the plan's threshold by ~9×
  with a structural explanation. Neither is a promotion case.
- **Reopen paths**: (a) a diffusion-sampler-shaped harness (positions
  driven by the reverse process, not free Langevin) to reproduce the
  separation claim; (b) approximate kernel features for the G2 threshold;
  (c) the riir-ai crowd-targeting plan (Guide 344) should wait on (a).
- **T4.4 signal to riir-ai**: REPORT-ONLY. The crowd-affect-targeting
  consumer (Research 505 fusion F1 / riir-ai Guide 344) should NOT file its
  plan on this record: G1 is partial and G2 exceeds a 20 Hz tick budget at
  crowd scale (9 µs/particle × 1000 NPCs = 9 ms/tick — inside 50 ms, but
  18 ms at d=8 with everything else paused). The primitive is real,
  deterministic, alloc-free, and its 2-D dial demonstrably steers; the
  exact-target claim needs the (a) harness first.

## Reproduction

```sh
CARGO_TARGET_DIR=/tmp/plan577-tgt cargo test -p katgpt-core --features distributional_steering --lib
#   → 1969 passed / 0 failed (18 module tests)
CARGO_TARGET_DIR=/tmp/plan577-tgt cargo test -p katgpt-core --features distributional_steering \
    --test bench_682_distributional_steering_goat --release -- --nocapture
#   → 4 passed; prints the G1 table + G2 numbers above
CARGO_TARGET_DIR=/tmp/plan577-tgt cargo test -p katgpt-core --features distributional_steering \
    --test bench_682_distributional_steering_alloc_check --release -- --nocapture
#   → 1 passed (0 allocs / 1000 steps @ N=1000)
CARGO_TARGET_DIR=/tmp/plan577-tgt cargo run -p katgpt-core --features distributional_steering \
    --example distributional_steering_demo --release
```
