/-
! Spec for `learnable_band_gate` — the ideal ℝ contract (Plan 576).

The Rust band-pass difficulty gate
(`crates/katgpt-core/src/hint_regret/gate.rs::learnable_band_gate`):

```rust
pub fn learnable_band_gate(w: f32, w_lo: f32, w_hi: f32, kappa: f32) -> f32 {
    sigmoid(kappa * (w - w_lo)) * sigmoid(kappa * (w_hi - w))
}
```

We model the gate over ℝ with Mathlib's `Real.sigmoid`, the exact
mathematical object the Rust `fast_sigmoid` approximates on its
non-saturating domain (`|x| ≤ 40`).

**The Rust-vs-ℝ split (same shape the Bridge module documents for its
ranking theorem):** over ℝ the gate is STRICTLY inside `(0, 1)` for every
input — no saturation exists in infinite precision. The f32 implementation
reaches exactly `0.0` far outside the band (the `fast_sigmoid` ±40
early-exit) and exactly `1.0f32` deep inside the band at large `κ` (plain
f32 rounding: once an argument exceeds ~17, `1 − σ(x) < 2⁻²⁵` rounds σ(x)
to `1.0`). Those are documented rounding artifacts of factors that are
strictly inside `(0, 1)` over ℝ — the theorem below pins the ideal side of
the split: `KatgptProof.HintRegret.bandGate_mem_Ioo`.

The theorem carries NO side conditions — the product of two sigmoids lies
in `(0, 1)` for every real `w`, `w_lo`, `w_hi`, `κ` (even a negative `κ` or
an inverted band: each factor is a sigmoid, and sigmoids live in `(0, 1)`
regardless of their argument's sign).
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid

namespace KatgptProof.HintRegret

open Real

/-! ## Band gate spec

`bandGate w wLo wHi κ = σ(κ(w − wLo)) · σ(κ(wHi − w))` — rises through
`wLo`, falls through `wHi`, peaks at the band center. The Rust consumer is
the learnable-band difficulty gate (Guide 340 §2); `w` is the learner's
per-composition Beta posterior mean (win rate against this composition).
-/

/-- The sigmoid band-pass gate over ℝ. Mirrors
    `hint_regret::gate::learnable_band_gate` — same factor order, same
    argument shapes — so a transcription error here is caught by the
    concrete-instance `SpecTests.lean`. (`noncomputable`: `Real.sigmoid`
    rides on `Real.exp`, which has no computable realization.) -/
noncomputable def bandGate (w wLo wHi κ : ℝ) : ℝ :=
  sigmoid (κ * (w - wLo)) * sigmoid (κ * (wHi - w))

/-! ## The ideal (0, 1) contract

Each factor is a sigmoid, and `Real.sigmoid_pos` / `Real.sigmoid_lt_one`
hold for every real argument — no sign constraint on `κ`, no ordering
constraint on the band. A product of two positive-below-one numbers is
positive-below-one.
-/

/-- **Band-gate openness.** The ideal ℝ band gate is STRICTLY inside the
    open unit interval for every `(w, wLo, wHi, κ)` — never 0, never 1, no
    matter how steep the walls or how far outside the band. This is the ∀-
    form of the Rust property test (`band_gate_is_strictly_inside_open_
    unit_interval_where_not_saturated` samples it); the f32 gate's exact-0 /
    exact-1 saturation points are documented Rust-vs-ℝ divergences, not
    counterexamples. -/
theorem bandGate_mem_Ioo (w wLo wHi κ : ℝ) :
    bandGate w wLo wHi κ ∈ Set.Ioo 0 1 := by
  have hp1 := Real.sigmoid_pos (κ * (w - wLo))
  have hp2 := Real.sigmoid_pos (κ * (wHi - w))
  have h1 := Real.sigmoid_lt_one (κ * (w - wLo))
  have h2 := Real.sigmoid_le_one (κ * (wHi - w))
  constructor
  · exact mul_pos hp1 hp2
  · -- s1 * s2 ≤ s1 * 1 = s1 < 1 (0 < s1, s2 ≤ 1)
    have hs : bandGate w wLo wHi κ ≤ sigmoid (κ * (w - wLo)) * 1 := by
      unfold bandGate; exact mul_le_mul_of_nonneg_left h2 hp1.le
    rw [mul_one] at hs
    exact lt_of_le_of_lt hs h1

end KatgptProof.HintRegret
