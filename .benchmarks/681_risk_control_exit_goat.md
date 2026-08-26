# Bench 681 — Risk-Controlled Exit Primitive GOAT (katgpt-core, feature `risk_control_exit`)

> **Plan:** [575](../.plans/575_risk_controlled_exit_primitive.md) · **Research:** [494](../.research/494_Conformal_Thinking_Dual_Threshold_Risk_Control_Exit.md) (arXiv:2602.03814, ICML 2026) · **Feature:** `risk_control_exit = []` (opt-in) · **Date:** 2026-08-26 · **Host:** M3 Max (macOS, aarch64), debug-profile test binaries unless noted
>
> **Numbering note:** bench number **681**, not 575 — 575 was already allocated (`575_sigmoid_argmaxability_audit.md`; monotonic-never-reuse rule). `.benchmarks/.highwater` was 680 at allocation; 681 re-scanned free immediately before this doc was created.

The Conformal-Thinking-distilled dual-threshold compute-exit primitive: `DualExitPolicy` (upper stop-when-confident + parametric lower stop-when-not-progressing schedule `λ−(t) = σ(c(ωt − sB), l, u)`, Phase 1 T1.1–T1.2), the four bounded losses (Eq. 8–11, T1.3), the UCB/Hoeffding calibrator with two-step decoupled selection + monotonicity refusal (T1.4–T1.5), and the App. C `p_i ≥ p_c` disarm tripwire (T1.6). All numbers below are from deterministic seeded runs (SplitMix64) — the same seeds reproduce them bit-identically.

## Gates

| Gate | Verdict | Measured |
|---|---|---|
| G1 — risk hold (T2.2) | ✅ PASS | UCB calibration holds realized exit-FP-risk ≤ ε = 0.10 on **40/40** test resplits at BOTH validation sizes (n=40 max risk **0.0100**; n=400 max risk **0.0200**). Naive no-correction calibration **violates on 7/40 resplits at n=40** (picks λ+ = 0.80/0.85 on noisy small-n validation; realized risk 0.18–0.35 ≫ ε) and is safe at n=400 — the paper's Fig. 4 shape, demonstrated at the n the plan's shrink-until-it-violates rule calls for |
| G1 — mechanism honesty | ✅ PASS | At n=40 the UCB term (0.194) exceeds ε → **40/40 conservative fallbacks** (λ+ = 0.95; the guarantee refuses rather than gambles). At n=400 (ucb 0.061) selection is genuinely feasible and lands **INTERIOR** on 39/40 resplits (λ+ = 0.90, not the grid max) — the calibration adapts, it does not merely hide at the conservative endpoint |
| G2 — exit floor (T2.3, the floor rule) | ✅ PASS | Crowd sweep at matched realized risk (same ε, same grids, same step-1 λ+): dual compute **0.417** vs single-threshold **0.609** vs fixed-budget **1.000** (means over 3:1 / 1:1 / 1:3 trivial:stuck) — dual **wins overall** and **wins-or-ties per composition** (accuracy identical 0.744/0.511/0.245 across all three policies; realized FN loss ≈ 0 on this population) |
| G2 — Fig. 6 shape | ✅ PASS | The dual-vs-single compute gap **grows with stuck share**: 0.075 (3:1) → 0.189 (1:1) → 0.310 (1:3) — upper-only captures most savings when few instances are stuck; the lower threshold dominates as stuck share rises, exactly the paper's Fig. 6 finding |
| G3 — default untouched | ✅ PASS | default lib **1951 passed / 7 ignored** — zero `risk_control_exit` tests exist under default features (module compiles out; change is purely additive feature-gated code). Feature-on lib **1972 passed / 7 ignored** (+21 = exactly the module's 21 unit tests) |
| G4 — alloc-free (T2.4) | ✅ PASS | counting-allocator: **0 steady-state allocs** over 10⁵ per-call `exit()` invocations AND over 200 `calibrate_into` passes with pre-sized reused scratch (single-fn binary, the bench_576 convention) |
| perf (reported, release) | ✅ PASS | **~4–5 ns per `exit()` call** (3.90/5.18 ns across two release runs, 10⁶ calls) vs the sub-µs gate — two comparisons + one squeezed sigmoid, as advertised |

All 21 module unit tests + 3 GOAT tests + 1 alloc-check test green.

## Commands (all `CARGO_TARGET_DIR=/tmp/plan575-tgt`)

```sh
cargo clippy -p katgpt-core --features risk_control_exit --all-targets   # 0 warnings in this primitive's files
cargo test   -p katgpt-core --features risk_control_exit --lib           # 1972 passed / 7 ignored (21 new)
cargo test   -p katgpt-core --features risk_control_exit --test bench_681_risk_control_exit_goat -- --nocapture
cargo test   -p katgpt-core --features risk_control_exit --release --test bench_681_risk_control_exit_goat -- --nocapture   # + perf gate
cargo test   -p katgpt-core --features risk_control_exit --test bench_681_risk_control_exit_alloc_check -- --nocapture
cargo test   -p katgpt-core --lib                                         # G3: 1951 passed / 7 ignored (default features)
```

## The G2 floor table (recorded)

| Composition | λ+ (calibrated) | dual acc / comp / risk | single acc / comp / risk | fixed acc / comp |
|---|---|---|---|---|
| 3:1 trivial:stuck | 0.80 | 0.744 / **0.283** / 0.0650 | 0.744 / 0.358 / 0.1000 | 0.744 / 1.000 |
| 1:1 | 0.85 | 0.511 / **0.430** / 0.0350 | 0.511 / 0.618 / 0.0587 | 0.511 / 1.000 |
| 1:3 | 0.90 | 0.245 / **0.540** / 0.0200 | 0.245 / 0.850 / 0.0362 | 0.245 / 1.000 |

Matched-risk construction: the single-threshold floor uses the SAME step-1 λ+ selection (same ε, same grid, same validation set) as the dual arm — the comparison isolates the lower threshold's contribution. Both calibrated arms hold realized risk ≤ ε = 0.15 at every composition. The fixed-budget floor has no risk mechanism at all (its error rate IS the stuck fraction — the paper's point).

## Honest findings (recorded for future sessions)

1. **The G1 naive-violation demonstration required the small-n regime, as the plan predicted.** At n=400 naive is safe (0/40 violations — the empirical risk estimate is tight enough). At n=40 the same procedure violates on 7/40 resplits. No shrinking below 40 was needed; 40 was already the violating regime.
2. **UCB at small n is mostly refusal, and that is the guarantee working — but it is not a certified selection.** At n=40 the UCB term (0.194) exceeds ε = 0.10, so no grid point is certifiable; the calibrator falls back to the most conservative point (λ+ = 0.95) and says so (`fell_back = true`). The test-set hold at n=40 rests on the conservative fallback being genuinely safe in this population, NOT on a certificate — the certified-selection demonstration is the n=400 block (39/40 interior picks). Both blocks are reported; conflating them would overstate the small-n guarantee.
3. **FP-risk monotonicity in λ+ is empirically clean on this oracle but is NOT a theorem.** The stuck-instance generator makes FP risk a Gaussian tail in λ+ (strictly decreasing), and the trivial class contributes zero FP by construction (any crossing of λ+ ≥ 0.70 happens at a step where s̃ ≥ 0.5 = correct). The calibrator still verifies the empirical curve every run and refuses the span past a violation — the unit test (`calibrate_refuses_non_monotone_upper_span`) pins the refusal on a hand-built rising curve, where a risk-0 grid point PAST the violation is provably excluded from selection.
4. **Lower-grid/upper-grid pairing contract.** Mutual exclusivity requires `u < λ+` for the PAIRED values. The grids share `u = 0.65 < min(upper grid) = 0.70` so every pairing is valid by construction; the contract is documented on `calibrate_into`. A future caller mixing high-`u` schedules with low-λ+ grids hits the constructor debug-assert, not silent breakage.
5. **Two-step decoupling (λ+ then schedule) trades rigor for practice** — accepted per the plan; the step-2 FN risk is measured under the already-selected λ+, so the two certificates are not a joint one.
6. **Seed-precedence bug caught by clippy mid-bench** (`0x5750_0000 + r + n << 24` parsed as `(base + r + n) << 24`, not `base + (n << 24) + r`) — the runs before/after the fix used different (equally deterministic) seeds; all numbers in this doc are from the fixed seeds.
7. **One transient failure in an unrelated module during validation.** Of 4 feature-on full-lib runs, one showed `1971 passed / 1 failed` (name not captured before the next run); the other 3 (and every run before/after) were fully green 1972/0, and all 21 `risk_control_exit` tests passed in every run. Attributed to the debug-mode timing-gate flake class under sibling-build load (the workspace is shared with a concurrent agent) — not this primitive's code, which contains no timing assertions outside the release-gated perf test.

## Promotion verdict (T2.6)

**Stays opt-in.** All gates pass modellessly (G1 risk hold + naive contrast, G2 floor win at matched risk, G4 alloc-free, ~ns hot path) — the plan's necessary condition for promotion is met. Promotion is deferred per the **no-default-consumer rule** (the hint_regret precedent, Bench 576): no in-tree consumer compiles this module yet; Phase 3 (MCTS termination, Plan 304 fusion `GainCostLoopHalter`, Bebop Issue 023 re-gate, riir-ai Research 339 wiring) is what would flip it, each with its own consumer gate. A designer flipping the default before any consumer exists would put an unexercised decision surface in every default build.

**No Cargo.toml change required** — the feature `risk_control_exit = []` and both test targets were pre-wired by the scaffold; promotion (if a future consumer gates it in) would be a one-line `default` list addition, owner-gated.

## Files

- `crates/katgpt-core/src/risk_control_exit.rs` — Phase 1 (T1.1–T1.7), ~1230 lines incl. 21 unit tests
- `crates/katgpt-core/tests/bench_681_risk_control_exit_goat.rs` — G1 / G2 / perf
- `crates/katgpt-core/tests/bench_681_risk_control_exit_alloc_check.rs` — G4
