# Bench 576 — Hint-Regret VoI Primitive GOAT (katgpt-core, feature `hint_regret`)

> **Plan:** [576](../.plans/576_hint_regret_voi_primitive.md) · **Feature:** `hint_regret = ["best_belief"]` (opt-in) · **Date:** 2026-08-24 · **Host:** M3 Max (macOS, aarch64), debug-profile test binaries unless noted

The SPADE-distilled (arXiv:2608.19197 / Research 496) hint-regret primitive: paired CRN estimator (Phase 1), sigmoid band gate + three-regime triage + Wilson CI (Phase 2), Beta-LCB frontier ordering + regret-scored memory (Phase 3). All numbers below are from deterministic seeded runs — the same seeds reproduce them bit-identically.

## Gates

| Gate | Verdict | Measured |
|---|---|---|
| G1 — oracle calibration | ✅ PASS | reveal-the-arm bandit: `E[r]≈0.0595` (10⁶-pair MC), `h(K)=0.1000`, `max_err=0.0347` < 2×h(K), **coverage 1.000 ≥ 0.95 nominal** at K=738 across 10³ seeds; hinted shortest path at β=∞: **bit-exact** vs demo total |
| G2 — CRN variance ratio | ✅ PASS | **ratio 2.76×** (Var_crn=3.35e-4 vs Var_indep=9.24e-4, 1000 reps × n=64 pairs) vs the ≥2× gate; per-pair cost **24.7 ns** (record_pair + amortized estimate, 10⁴ pairs) vs the sub-µs gate |
| G3 — default untouched | ✅ PASS | default lib **1904 passed / 7 ignored — count-identical to clean HEAD** (worktree-verified); the module compiles out (`hint_regret` off → zero tests, zero code) |
| G4 — alloc-free | ✅ PASS | counting-allocator: **zero steady-state allocs** over 10⁴ pairs (estimator + memory; setup outside the measured region) |
| G-Floor — Report the Floor | ✅ PASS | paired-arm triage accuracy **0.997** vs single-arm banding floor **0.693** (2000 classifications, K=40/arm, matched 2K budget) — the UQ-bearing primitive beats its naive floor at 3.6× margin |
| G8 — learnable-share (simulated) | ✅ PASS | regret-gated selection learnable-share **1.000** vs uniform **0.273** on the synthetic curriculum (8 seeds, T=400, per-seed all-1.000 vs all-0.276); the gated policy never leaves the learnable band, uniform burns 73% of budget off-band |

All 6 GOAT tests + 1 alloc-check test green: `cargo test -p katgpt-core --features hint_regret --test bench_576_hint_regret_goat --test bench_576_hint_regret_alloc_check`.

## Lean (the ideal ℝ contract)

- `KatgptProof.HintRegret.Basic`: `bandGate_mem_Ioo` — the band gate `σ(κ(w−wLo))·σ(κ(wHi−w))` is **strictly inside (0,1) for every real input**, no side conditions (product of two sigmoids; `Real.sigmoid_pos`/`sigmoid_lt_one`). This is the ∀-form of the Rust property test.
- `KatgptProof.HintRegret.SpecTests`: concrete instances — κ=0 flattens to **exactly 1/4** (any w, band); at either wall strictly **< 1/2** (unconditionally). Catches transcription errors the theorem can't see.
- Axiom inventory: 39 theorems audited, all within `{propext, Classical.choice, Quot.sound}` (`proof_gate.sh` PASS).
- Negative tests: 8/8 perturbations caught incl. the two new HintRegret ones (P7 factor-drop — the theorem survives a dropped factor, only the SpecTests catch it; P8 flat-constant typo).

## Honest findings (fixture calibration, recorded for future sessions)

Three fixture findings from the validation run — all in the TEST layer, none in the primitive:

1. **House-sigmoid saturation is two-sided in f32.** The gate doc originally claimed "strictly inside (0,1) in and around the band". False at large κ: once a sigmoid argument exceeds ~17, `1−σ(x) < 2⁻²⁵` rounds σ(x) to exactly `1.0f32` (plain rounding, no early-exit) — the band center at κ=64 (args 19.2) measures exactly 1.0. The ±40 early-exit covers the 0-side. Doc + property test now state the honest two-surface contract; Lean pins the strict ideal over ℝ.
2. **The G2 CRN fixture needed variance-budget calibration + a noise-stream fix.** First run measured 1.61×. Root causes: (a) the adaptive ε-greedy policy's value estimates feed on the same noise — Var_policy grows with σ, capping the ratio (the noise→learning feedback loop); (b) the indep mode's extra draw shifted the shared stream, diverging policy paths the doc claimed were identical. Fix: ε=1.0 (pure uniform policy, feedback severed — `update` never influences picks) + a dedicated salted noise stream (picks now identical across modes by construction, `Var_indep − Var_crn ≈ 2σ²/n` exactly). Result 2.76×, matching the arm-spread/noise budget derivation. The estimator itself was never at fault — the cancellation delta tracked theory at every fixture setting.
3. **f32 vs f64 tolerances in fixtures** (salience ordering, bandit oracles) — fixed at authoring time during the first run.

## Promotion verdict

**Stays opt-in.** The plan's promotion criterion ("default-on only if G-Floor and G8 pass modellessly" — a necessary condition) is **met**: both gates pass modellessly (G-Floor 0.997 vs 0.693; G8 1.000 vs 0.273). Promotion remains deferred per the no-default-consumer rule: the landed consumer core (riir-mmorpg-examples `2c17f08`, behind its own opt-in `demo_coverage_curiosity`) predates this module and carries a local triage — the consumer pilot that promotes this primitive is the Phase 5 migration onto `katgpt_core::hint_regret` (which also removes the duplicate-substrate smell), owner-gated as the plan marks it.

## Files

- `crates/katgpt-core/src/hint_regret/{mod,gate,memory,tests}.rs` — Phases 1–3
- `crates/katgpt-core/tests/bench_576_hint_regret_goat.rs` — G1/G2/G-Floor/G8
- `crates/katgpt-core/tests/bench_576_hint_regret_alloc_check.rs` — G4
- `.proofs/KatgptProof/HintRegret/{Basic,SpecTests}.lean` — the ℝ contract + teeth
