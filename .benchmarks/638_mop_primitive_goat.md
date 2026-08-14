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

## Issue 654 addendum (2026-08-15): one-hot sparse-row fast path

**Change:** `solve`'s two dot sites + the prepare phase now detect one-hot
kernel rows (exactly 1 nonzero of N — the zone-KG deterministic-abstraction
shape) and compute the row dot as the single product `p[j\*]·ζ[j\*]` instead
of the dense `simd_dot_f32`. Rows with ≠ 1 nonzeros keep the dense SIMD dot.
Detection is a prepare-phase scan with **early exit on the 2nd nonzero** —
dense rows (the Bench 638 random fixtures) cost O(2)/row, so the dense path
is regression-free (re-benched: 2.21/2.80/24.76 ms vs the recorded
2.1/2.8/24.0 — unchanged within noise).

**Bit-identity (arch-independent, why this is not a re-derivation):** for a
one-hot row the dense dot reduces to the single term — every zero entry
contributes `±0`, an exact no-op against finite accumulators (all start at
`+0`), and the surviving term is correctly rounded in both paths (an FMA
into a zero accumulator equals the plain product). The only divergence —
the sign of a ±0 dot — is absorbed by the caller's `h_bar + dot` (h\u0304 is
never `−0`: β, α, H ≥ 0). Rows with ≥ 2 nonzeros stay dense: replicating the
per-arch SIMD lane accumulation order (NEON 4×4-lane + ADDV tree vs AVX2
vs wasm) would be fragile for marginal gain. Pinned by a dedicated unit
test (`onehot_fast_path_bit_identical_to_dense_dot` — both `h_bar` regimes,
planted ±0/subnormal/huge ζ edge values) plus the mixed-sparsity golden test.

**Re-gate (Bench 638 protocol):**

- **G1:** lib suite 10/10 (7 original + 3 new); the golden arena coverage is
  pinned (`fast_rows ≥ 250` assert — the 4-room fixture is all one-hot, so
  the golden gate exercises the fast path end-to-end). Second oracle: the
  riir-ai parity harness 6/6 — **V\* max|Δ| = 0.0 bit-identical, identical
  iteration counts** (86/276/229/290/284), unchanged from the pre-change
  record.
- **G2 (direct A/B, same fixtures, same 262 iters, this repo's LTO bench
  profile):**

  | Fixture | dense | sparse | speedup |
  |---|---|---|---|
  | one-hot N=64/A=4 | 682.0 µs | 391.3 µs | 1.74× |
  | one-hot N=64/A=16 | 1984.8 µs | 1209.2 µs | 1.64× |
  | one-hot N=256/A=16 | 20218.8 µs | 6044.0 µs | **3.35×** |
  | 4-room gridworld N=82/A=4 | 663.4 µs | 335.1 µs | 1.98× |

  N=256 gains most: the 4 MiB kernel exceeds L2, so the dense dot was
  memory-bound (streaming the full kernel every iteration); skipping 63 of
  64 entries per row avoids most of that traffic. N=64 kernels fit in cache
  → the win is the eliminated multiplies. The pre-change gridworld number
  (663.4 µs) reproduces the original Bench 638 record (663 µs) exactly.
  G2 gate (gridworld < 1 ms): **335 µs PASS**. Production-profile caveat:
  downstream workspaces without LTO (riir-ai compiles this crate at cgu=16,
  no LTO) see ~2× slower loop codegen — the consumer-side numbers live in
  riir-ai Bench 680's Issue 654 follow-up.
- **G3/G4:** clippy 0 warnings (`--all-targets`, feature on); default build
  compiles nothing new; **0 allocations** across solve + 1000 `pi_star`
  calls (the one-hot table is a new private `MopScratch` field — stack,
  no public-API change).

**Honest scope notes:** (1) `state_conditional_entropy` keeps its dense
walk — it runs once per row per solve (~0.3% of solve cost) and delegates
to the shared `cgsp::types::entropy_nats` (the no-duplicate-entropy rule;
re-implementing a sparse variant would duplicate substrate). (2) The G2
ceiling-shaped win is bounded by the LSE `exp()` floor, not the dot — see
riir-ai Bench 680's follow-up for the consumer-side decomposition.
