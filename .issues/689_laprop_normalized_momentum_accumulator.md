# Issue 689 (katgpt-rs): `NormalizedMomentumAccumulator` — clamp-free EMA with the LaProp Prop-1 bound

> **Source:** [riir-train Research 428](../../riir-train/.research/428_LaProp_Decoupled_Momentum_Adaptive_Optimizer.md) — LaProp (arXiv:2002.04839) Path-0 extraction, component C2/C3 (the modelless half; the optimizer itself lives in riir-train Plan 354).
> **One-line:** ship the normalize-before-accumulate ordering as a katgpt-core primitive — EMA over RMS-normalized intake with the closed-form accumulator bound `|m| ≤ 1/√(1−ν)` — so downstream accumulators delete their clamps and get a theorem instead.

## Evidence (verified production demand — the accumulate-raw-then-clamp anti-pattern)

1. **riir-clippy `src/evolve.rs::ema_step` (L90–102)** — the self_evolve direction-learning core (feeds `LatentSkillEvolution::record_outcome` AND the durable `TrajStore` learn rows): accumulates `rate · (fix_dir[i] − dir[i])` (RAW delta, unbounded intake) into `dir`, then `.clamp(-1.0, 1.0)`. The clamp is treating the symptom (unbounded raw intake); LaProp's ordering removes the cause. **This is a heavy-tailed-intake site** — one pathological fix-trajectory delta can jerk the direction; the clamp then hides it asymmetrically.
2. **katgpt-core `committed_field_blend.rs::commit`** — `pi_k = clamp(dot(summary, dir_k), ±pi_max)` (weaker demand: `pi_max=10` is partly intentional near-binary selection semantics; the bound would still make the safety half principled).
3. **SAFE by the ordering-law classification (do NOT touch):** `ReconstructionState::evolve_belief` / `LeakyIntegrator` (sigmoid-bounded activations + per-step `max_delta` — bounded input, per-step not accumulator bound), Elo (`katgpt_core::rating` — zero-sum bounded by K by construction).

## Proposed primitive (`crates/katgpt-core/src/laprop.rs`, feature `laprop`, opt-in)

```rust
pub struct NormalizedMomentumAccumulator<const D: usize> {
    m: [f32; D],      // momentum over NORMALIZED intake
    n: [f32; D],      // running second moment of RAW intake
    t: u32,
    mu: f32, nu: f32, eps: f32,
}
impl<const D: usize> NormalizedMomentumAccumulator<D> {
    pub fn new(mu: f32, nu: f32) -> Self;
    pub fn push(&mut self, x: &[f32; D]);            // m ← μ·m + (1−μ)·x/√(n̂+ε); n ← ν·n + (1−ν)·x²
    pub fn momentum(&self) -> &[f32; D];             // bias-corrected m̂
    pub const fn bound(&self) -> f32;                // 1/√(1−ν) — the closed-form Prop-1 bound
    pub const fn influence(&self) -> f32;            // (1−μ)/√(1−ν) — single-observation bound
    pub const fn coupling_cost(mu: f32, nu: f32) -> f32; // C8 doc note: Adam's 1/(1−μ/√ν) vs ours
}
// + scalar twin `NormalizedMomentumScalar` for 1-D signals (reward EMAs, priorities).
```

Zero crate deps, pure f32, caller-owned arrays (G4 alloc-free by construction). Module doc carries the ordering law (C3): *accumulate-then-normalize makes historical evidence's weight depend on the stale-to-fresh magnitude ratio — unbounded under heavy tails (LaProp §3.1, Var = ∞); normalize-then-accumulate makes it a function of μ alone.*

## GOAT gates

- **G1 (falsifiable both directions):** planted outlier — one `1e6` delta into unit-scale stream: (a) every `momentum()` component ≤ `bound()·(1+1ulp)` with NO clamp anywhere; (b) after k clean steps the outlier's residual influence is exactly `μ^k` (bit-identical to the formula); (c) the current clamped raw-EMA (`ema_step` shape) FAILS the same no-clamp assertion — the A/B proving the gain is real.
- **G2:** per-push cost ≤ clamped EMA + ~15 ns at D=8 (one extra mul + FMA per component).
- **G3:** default features unchanged (feature-gated, opt-in).
- **G4:** zero allocs per push (tracking-allocator pinned).
- **G5 (ν-dial, the fusion extension):** at ν=0 the normalized intake degenerates to sign(x) — assert the ternary-sign limit; monotone interpolation ν ∈ (0,1).
- Precision honesty: Prop 1 is per-component (L∞); L2 form is `√D/√(1−ν)` — pin both.

## Non-goals / scope notes

- NOT an optimizer — no gradient/loss semantics here (that's riir-train Plan 354).
- NOT UQ-bearing (bound, not interval) — no conformal-floor extension applies.
- World-novelty: none claimed — normalize-before-accumulate is the normalized-LMS family's classic pattern; our claim is no-analog-in-our-repos + the closed-form bound replacing clamps.
- Consumer migration (riir-clippy `ema_step` → this primitive) is a follow-up in THAT repo once the primitive lands — keep the behavior-change (clamp semantics → bound semantics) behind its own A/B there; `ema_step`'s returned delta-norm contract must be preserved.

## Tasks

- [ ] T1 primitive + unit tests (bound, influence, bias correction, ν=0 sign limit, planted-outlier G1 both directions)
- [ ] T2 scalar twin + `coupling_cost` doc note
- [ ] T3 bench + GOAT note + promote/demote decision (promote to default only if G1–G4 pass AND a consumer adopts)
