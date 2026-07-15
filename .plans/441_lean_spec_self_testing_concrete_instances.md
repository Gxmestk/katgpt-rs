# Plan 441: Lean Spec Self-Testing on Concrete Instances

**Date:** 2026-07-15
**Research:** [katgpt-rs/.research/425_symcrypt_verifiedcrypto_aeneas_methodology.md](../.research/425_symcrypt_verifiedcrypto_aeneas_methodology.md)
**Source:** [SymCrypt `feature/verifiedcrypto` README](https://github.com/microsoft/SymCrypt/blob/feature/verifiedcrypto/README-VERIFIEDCRYPTO.md) §4 "Running the Lean spec on test vectors"
**Target:** `katgpt-rs/.proofs/KatgptProof/Bridge/SpecTests.lean` + `katgpt-rs/.proofs/KatgptProof/Ssmax/SpecTests.lean`
**Status:** Phase 3 COMPLETE — GOAT gate G1-G4 ALL PASS

---

## Goal

Distill the cheapest gain from SymCrypt's verified-crypto methodology: **execute the Lean spec itself on concrete instances** (SymCrypt calls this "running the spec on test vectors"). This closes a real gap in our C1–C6 FV pattern (Research 351):

- Currently, we prove the spec (`Basic.lean`) against itself (`*.lean` theorems).
- Currently, we test Rust against the spec (`spec_match.rs` — finite samples).
- We do **NOT** test the spec against known-good values from the source paper.

A spec transcription error (e.g., `N^c` instead of `N^(-c)` in `alphaGold`, or `+` instead of `-` in `dot`) would make the proof prove the wrong property AND the spec-match test pass (Rust has the same typo, written by the same person). Only concrete-instance tests catch this — the paper's published values are the independent authority.

**GOAT gate:** G1 (spec matches known values) + G2 (negative test: injected typo fails) + G3 (existing proofs still build) + G4 (zero Rust-side cost).

## The gap (honest)

The `alphaGold` formula in `Ssmax/Basic.lean` is transcribed from arXiv:2607.01538 §2:

```lean
noncomputable def alphaGold (N c : ℝ) : ℝ := 1 / (1 + (N - 1) * N^(-c))
```

If this has a sign typo (`N^c` instead of `N^(-c)`), the theorems in `DilutionBound.lean` still type-check (they prove monotonicity of the transcribed formula, whatever it is). The Rust spec-match test (`ssmax_spec_match.rs`) samples Rust against the Lean spec — if the Rust `ssmax.rs` doc comment has the same typo, both sides agree on the wrong formula.

The paper provides concrete dilution curves. A spec test like `example : alphaGold 2 0 = 1/2` catches the transcription error because `2^0 = 1` regardless of sign, but `alphaGold 2 1 = 2/3` requires `2^(-1) = 1/2` (correct sign). With the wrong sign (`N^c`), `alphaGold 2 1 = 1 / (1 + 1 · 2^1) = 1/3 ≠ 2/3` — the test fails.

## Phase 1 — Bridge spec tests (concrete dot + sigmoid instances)

### Tasks

- [x] **T1.1** Create `katgpt-rs/.proofs/KatgptProof/Bridge/SpecTests.lean` with concrete-instance `example` proofs (3 dot + 3 sigmoid = 6 examples)
  - `dot ![1, 0] ![1, 0] = 1` (self-dot-product)
  - `dot ![1, 1] ![1, -1] = 0` (orthogonal vectors)
  - `dot ![2, 3] ![4, 5] = 23` (non-trivial)
  - `Real.sigmoid 0 = 1/2` (origin)
  - `Real.sigmoid 1 > 1/2` (positive half)
  - `Real.sigmoid (-1) < 1/2` (negative half)
- [x] **T1.2** Add `import KatgptProof.Bridge.SpecTests` to `KatgptProof.lean`.

## Phase 2 — Ssmax spec tests (concrete alphaGold instances from the paper)

### Tasks

- [x] **T2.1** Create `katgpt-rs/.proofs/KatgptProof/Ssmax/SpecTests.lean` with concrete-instance `example` proofs derived from the paper's dilution curve (5 alphaGold examples + helper lemma `rpow_neg_one_eq_inv`)
  - `alphaGold 2 0 = 1/2` (N=2, no sharpening — trivial bound)
  - `alphaGold 2 1 = 2/3` (N=2, c=1 — requires correct `N^(-c)` sign)
  - `alphaGold 10 0 = 1/10` (N=10, no sharpening)
  - `alphaGold 10 1 = 10/19` (N=10, c=1 — the paper's motivating regime)
  - `alphaGold 100 0 = 1/100` (N=100, no sharpening — dilution collapse)

## Phase 3 — GOAT gate

### Tasks

- [x] **T3.1** **G1 — Spec correctness:** `cd katgpt-rs/.proofs && lake build` passes (2283 jobs). All 11 examples compile: 3 dot + 3 sigmoid + 5 alphaGold. The spec produces paper-aligned values on concrete instances.
- [x] **T3.2** **G2 — Negative test (error-catching power):** Temporarily flipped `N^(-c)` → `N^c` in `alphaGold`. ALL 5 Ssmax spec test examples FAIL with "Tactic rewrite failed" — the `rpow_neg_one_eq_inv` helper no longer applies and the `alphaGold 2 1 = 2/3` instance would yield `1/3`. Error-catching power confirmed.
- [x] **T3.3** **G3 — No regression:** `action_bridge_ranking_preserved` still depends only on `{propext, Classical.choice, Quot.sound}`. All existing theorems build unchanged (2283 jobs vs 2281 baseline).
- [x] **T3.4** **G4 — Zero Rust-side cost:** No Rust files modified. Spec tests are Lean-only, gated by `lake build`.
- [x] **T3.5** Update `katgpt-rs/.proofs/README.md` to document the spec-test convention (addition to C6: each spec ships paired `SpecTests.lean` with concrete instances from the source paper).
- [x] **T3.6** Update Research 425 verdict from Gain → GOAT (proven gain via G1-G4).
