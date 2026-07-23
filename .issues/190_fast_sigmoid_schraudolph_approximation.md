# Issue 190 — Schraudolph Fast-Sigmoid Approximation for CGSP Hot Path

> **Spawned from:** rust-optimize session 2026-07-24 (post-structural_complexity fusion)
> **Date:** 2026-07-24
> **Type:** optimization (transcendental-function approximation)
> **Severity:** MEDIUM — attacks the transcendental floor (30-40% of CGSP cycle time)
> **Status:** OPEN (needs GOAT gate)

## Context

The CGSP cycle's transcendental-function floor was identified across prior
rust-optimize sessions: ~30-40% of `CgspLoop::cycle` time is `exp`/`ln` calls
via `cgsp::types::sigmoid` + `simd::fast_sigmoid`. The accessible code-level
optimizations (loop fusion, structural_complexity 3→1 pass fusion, DRY
dedup) are exhausted. The remaining lever is **approximating the
transcendental itself**.

### Where sigmoid is called in the CGSP hot path

| Call site | Frequency | Current impl |
|---|---|---|
| `HlaProjectionGuide::score` | 2× per candidate × k candidates/cycle | `cgsp::types::sigmoid` (two-branch stable, libm `exp`) |
| `DualPoolBandit::exploitation_probability` | 1× per cycle | `cgsp::types::sigmoid` |
| `entropy_nats` | 1× per cycle | `p.ln()` (libm `ln`) |

At default k=4: **9 transcendental calls per cycle** (8 sigmoid + 1 ln).

### Why SIMD batching is NOT the answer here

`simd_sigmoid_inplace` exists (`katgpt-types::simd::activations`) and wins
at ≥8 elements. But DEFAULT_K=4 → only 4-8 sigmoid calls per cycle, below
the SIMD win threshold. The scalar `fast_sigmoid` (libm `expf`, ~5ns/call)
is already the GOAT-validated path for this batch size.

## The Schraudolph Trick

**Reference:** Schraudolph, N. N. (1999). "A Fast, Compact Approximation
of the Exponential Function." *Neural Computation* 11(4):853-862.

The trick exploits the IEEE-754 float representation to approximate `exp(x)`
in ~2ns using integer arithmetic + a single multiply:

```text
exp(x) ≈ reinterpret_as_f32((int)(1512775 * x + 1072693248))
```

For sigmoid: `σ(x) = 1 / (1 + exp(-x))` → one Schraudolph exp + one
reciprocal. Total: ~3ns vs ~5ns for libm `expf` + reciprocal.

**Precision:** ~1% relative error (the original paper's bound; tighter
variants exist — e.g. with a 2nd-order correction term, <0.1% error).

### Does 1% error matter for CGSP?

The sigmoid in CGSP serves two roles:
1. **Guide scoring** — `sigmoid(λ · dot)` maps dot-product to [0,1] relevance.
   A 1% error shifts the score slightly but preserves monotonicity (the
   Schraudolph approximation is monotonic). The priority-table update is
   robust to small score perturbations (it's a bandit, not an argmax).
2. **Pool routing** — `sigmoid(w_E − w_X)` determines E-vs-X pool selection
   probability. A 1% error shifts the routing probability slightly. The
   reachability clamp `[ε, 1−ε]` already absorbs saturation; small errors
   in the middle of the range should be benign.

**This needs to be proven by the GOAT gate, not assumed.** The CGSP
regret-bound proofs (Research 249) may depend on exact sigmoid properties.

## Proposed approach

1. Implement `schraudolph_sigmoid(x: f32) -> f32` in
   `katgpt-types::simd::activations.rs` behind a new `schraudolph_sigmoid`
   feature flag. Include the 2nd-order correction variant.
2. Add a CGSP-specific gate: `cgsp::types::sigmoid` delegates to
   `schraudolph_sigmoid` when the feature is on, else the current stable
   sigmoid.
3. Run the GOAT gate:
   - **G1**: All 44 `cgsp::` tests pass (qualitative behavior preserved).
     Additionally, verify the CGSP regret bound holds (run the
     `g2_log_regret_synthetic` test with the approximation — the regret
     trajectory should stay within the theoretical bound).
   - **G2**: Benchmark `CgspLoop::cycle` with vs without the approximation.
     Target: ≥20% cycle-time reduction (the transcendental floor is
     30-40%; Schraudolph recovers ~40% of that).
   - **G3**: All katgpt-core tests pass (`cargo test --features cgsp`).
   - **G4**: Zero allocation (the approximation is pure arithmetic).

## Don't do this yet

This issue is a **captured-for-the-record** optimization candidate. Do NOT
implement until:
- A concrete production workload shows CGSP cycle time is a bottleneck (not
  just a micro-benchmark), OR
- The structural_complexity fusion (commit `51d6bfc8`) + prior session's
  loop fusion (`404ffd63`) are measured in situ and found insufficient.

The transcendental floor is real but may not matter at the system level
(the swarm tick has 1363× headroom; CGSP is one of many cognition steps).

## Cross-references

- `katgpt-types::simd::activations::fast_sigmoid` — the canonical scalar sigmoid
- `katgpt-types::simd::activations::simd_sigmoid_inplace` — SIMD sigmoid (≥8 elements)
- `cgsp::types::sigmoid` — the two-branch stable sigmoid used in the hot path
- `cgsp::guide::HlaProjectionGuide::score` — primary hot-path caller
- `cgsp::dual_pool::DualPoolBandit::exploitation_probability` — secondary caller
- Prior session commit `404ffd63` — guide-score + difficulty-admit loop fusion
- This session commit `51d6bfc8` — structural_complexity 3-pass → 1-pass fusion
