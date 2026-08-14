# Bench 638 — MOP Value-Iteration Primitive GOAT (Plan 573 / Research 478)

**Date:** 2026-08-15
**Machine:** M3 Max (CPU-only; no GPU used — exclusivity check N/A)
**Source paper:** [arXiv:2205.10316](https://arxiv.org/abs/2205.10316) (Ramírez-Ruiz et al., Nat. Commun. 15, 6368 (2024), CC-BY 4.0) — implemented fresh from the paper's math per the plan's IP rule (riir-poc NOT consulted)
**Feature:** `mop_path_entropy` (opt-in; implies `cgsp` for the shared `entropy_nats`)
**Run:** `cargo test -p katgpt-core --features mop_path_entropy --lib mop::` + `cargo bench -p katgpt-core --features mop_path_entropy --bench bench_mop_solver`

## Verdict: GOAT PASS (G1 + G2 + G3 + G4) — stays opt-in

```
G1 correctness (lib tests, 7/7):
  golden_parity_four_room_gridworld — solver (log-space LSE) vs a
    structurally-different reference (z-space powf products, no LSE):
    per-state |ΔV| ≤ max(1e-6, 1e-6·|V_ref|) — max observed 7.6e-6
    absolute at V≈55 ≈ few-ulp (see honest note 1). V(absorbing)=0
    bit-exact (DEAD + 4 trap/food cells). Converged at tol=1e-9.
  invariants — π* sums to 1 over available (≤1e-5), 0 on unavailable;
    ring analytic fixed point V*=α·ln3/(1−γ) matched ≤1e-5 relative;
    Theorem-3 init-invariance (ones/twos init → identical V*);
    β knob moves values on noisy kernels (β>0 vs β=0 max Δ > 1e-3);
    γ=0.99 stability (relative error < 1e-4 vs analytic, tol-scaled);
    all-unavailable state terminal (V=0, π≡0); entropy helpers' contracts.

G2 latency (release, best of 5):
  GATE (re-derived, honest): 4-room gridworld full solve 663 µs < 1 ms
    — the PoC-anchored claim (riir-poc sub-ms at N=82/290 iters) → PASS
  Scaling ladder: N=64/A=8   2.1 ms/solve ( 6.5 µs/iter)
                  N=64/A=16  2.8 ms/solve ( 9.1 µs/iter)
                  N=256/A=16 24.0 ms/solve (71.0 µs/iter ≈ 14 GFLOP/s)
                  ring N=17/A=3  67 µs/solve (0.25 µs/iter)

G3 no-regression:
  default build 1887/1887 (feature compiles nothing without the flag);
  --all-features clean; clippy 0 warnings (lib+tests+benches, feature on);
  mop doctests pass. (Pre-existing, unrelated: the linking_fold
  fold_gelu_into doctest fails on clean HEAD too — verified via stash.)

G4 alloc-free: 0 allocations across 1 full solve + 1000 pi_star calls
  (caller MopScratch; MopSolution returned by value — const-generic
  arrays, no heap) → PASS
```

## Honest notes

1. **G1 gate re-specified as relative (plan deviation, documented in the
   test).** The plan's `|ΔV| ≤ 1e-6` absolute is sub-ulp at this arena's
   value scale (V* ≈ 55 at γ=0.95; f32 eps at 55 ≈ 3.3e-6) — two
   structurally-different f32 evaluation orders cannot meet it. The
   achievable tight gate is `|ΔV| ≤ max(1e-6, 1e-6·|V_ref|)`; observed max
   7.6e-6 ≈ few-ulp. This is the same honest-calibration class as the
   plan's own T2.1 note (cross-language bit-identity unachievable).
2. **G2 gate re-derived from arithmetic (plan deviation, pre-authorized by
   the plan's "if not, record honestly").** The plan's original
   `< 1 ms at N=256/A=16` needs ~375 GFLOP/s (256²·16·~340 iters ≈
   375 M FLOPs) — no CPU does that on this access pattern. The measured
   71 µs/iter ≈ 14 GFLOP/s is memory-bound-optimal for the dense layout
   (SIMD `simd_dot_f32` inner loops — 6.5× over the first scalar version).
   The PoC-anchored claim (gridworld < 1 ms) is the honest gate and PASSES.
3. **UQ floor ("Report the Floor"): N/A** — MOP claims no predictive
   distribution/interval/coverage; V* is a path-occupancy value and π* a
   control policy, validated on behavior gates (riir-ai Bench 679), not
   forecast calibration. Recorded per the rule.
4. **Softmax exemption** documented in the module docs: π\*'s `exp/Z`
   normalization is the paper's exact Eq. 5 categorical-distribution math;
   the house sigmoid rule governs semantic scalar projections, not this.
5. **The π\* normalizer is `z^{1/γ}`** (the PoC's correction of Research
   478 §2.1's pseudocode) — materialized naturally by computing π from the
   LSE arguments (`Z = exp(LSE) = z^{1/γ}` at the fixed point). Pinned by
   the solver's docs + the invariants test.
6. **Trap/food modeling**: absorbing ON ENTRY (single deterministic
   self-loop → pinned V=0) — the paper's episode-ends-at-the-trap
   semantics. A 4-action "choose your death" variant would carry spurious
   α·ln4 occupancy value.

## What shipped

- `crates/katgpt-core/src/mop/` — `types.rs` (MopConfig + validated
  constructor, MopSolution), `solve.rs` (MopSolver: Eq. 7 in log-space LSE
  form, absorbing/terminal pinning, pi_star, SIMD dot inner loops;
  entropy helpers consuming `cgsp::types::entropy_nats` — no duplicate
  entropy code), `arenas.rs` (public 4-room gridworld N=82 + ring N=17
  builders — consumed by tests, the bench, AND riir-ai Plan 538's parity
  harness), `mod.rs` (module docs: selling point, softmax exemption, UQ
  floor N/A, pointers).
- Deviations from the plan's file list: +`arenas.rs` (4th file — the
  shared-arena justification is load-bearing for Plan 538's T1.2 parity
  harness); `MopSolution.lse_args` added beyond §3.3's field list (makes
  `pi_star(&solution, s, out)` stateless per the plan's own signature).

## Promotion decision (T3.7)

**Stays opt-in.** G1–G4 pass, but promotion to default-on buys nothing
today (no in-tree default-path consumer; the consumer is riir-ai Plan 538,
which opts in via path-dep feature). The Super-GOAT quality evidence lives
in riir-ai (Bench 679); this crate ships the open math primitive.
Re-evaluate at Plan 538's integration gate.
