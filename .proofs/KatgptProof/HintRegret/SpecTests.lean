/-
! Spec self-tests for the band gate — concrete instances with independently
known answers (the Bridge SpecTests pattern; Plans 441/425).

`bandGate_mem_Ioo` proves the spec against itself; the Rust property test
samples the f32 gate. **Neither catches a transcription error in the Lean
`bandGate` definition itself** (a wrong constant, a dropped factor, a
malformed argument shape) — only concrete instances with hand-computable
values close that gap.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import KatgptProof.HintRegret.Basic

namespace KatgptProof.HintRegret

open Real

/-! ## κ = 0 — the flat gate

At `κ = 0` both factors are `σ(0) = 1/2`, so the gate is EXACTLY `1/4`
regardless of `w` and the band — the Rust doc's "κ = 0 flattens to 0.25
everywhere" (`gate.rs`). This is the strongest single instance: a wrong
constant, dropped factor, or malformed argument shape moves the answer off
`1/4`.
-/

/-- κ = 0 flattens the gate to σ(0)·σ(0) = 1/4 — for ANY `w` and band. -/
example (w wLo wHi : ℝ) : bandGate w wLo wHi 0 = (1:ℝ) / 4 := by
  simp [bandGate, sigmoid_zero]; norm_num

/-! ## At the walls

At `w = wLo` the rising factor is `σ(0) = 1/2`; the gate is
`2⁻¹ · σ(κ(wHi − wLo))`, strictly below `1/2` because a sigmoid is strictly
below 1 — unconditionally (any κ, any band). Symmetric at `wHi`. These pin
the "rises THROUGH wLo, falls THROUGH wHi" shape: at either wall the gate
is exactly half the other factor's value, never more.
-/

/-- At the lower wall the gate is strictly below 1/2. -/
example (wLo wHi κ : ℝ) : bandGate wLo wLo wHi κ < 2⁻¹ := by
  rw [bandGate, sub_self, mul_zero, sigmoid_zero]
  have hlt : sigmoid (κ * (wHi - wLo)) < 1 := Real.sigmoid_lt_one _
  have hpos : (0:ℝ) < 2⁻¹ := by norm_num
  simpa using mul_lt_mul_of_pos_left hlt hpos

/-- At the upper wall the gate is strictly below 1/2 (symmetric). -/
example (wLo wHi κ : ℝ) : bandGate wHi wLo wHi κ < 2⁻¹ := by
  rw [bandGate, sub_self, mul_zero, sigmoid_zero]
  have hlt : sigmoid (κ * (wHi - wLo)) < 1 := Real.sigmoid_lt_one _
  have hpos : (0:ℝ) < 2⁻¹ := by norm_num
  simpa using mul_lt_mul_of_pos_left hlt hpos

end KatgptProof.HintRegret
