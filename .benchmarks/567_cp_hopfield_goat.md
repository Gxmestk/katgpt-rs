# Benchmark 567: CP^(d-1) Symmetric-Space Hopfield — GOAT Gate

**Date:** 2026-08-04
**Plan:** [.plans/567](../.plans/567_cp_hopfield_top_eigenvector_recall.md)
**Research:** [.research/466](../.research/466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md)
**Paper:** Galitski, *High-Capacity Generalized Hopfield Networks* — [alphaXiv 2607.hopfield-networks](https://www.alphaxiv.org/abs/2607.hopfield-networks) (JQI/UMD, 2026-07-31)
**Feature:** `cp_hopfield` — **STAYS OPT-IN**

Benches:
- `katgpt-core/benches/bench_567_cp_hopfield_goat.rs` (G2, G4, G7)
- `riir-ai/crates/riir-poc/benches/cp_hopfield_plan276_unblock.rs` (G5)

---

## Verdict summary

| Gate | Criterion | Result |
|---|---|---|
| G1 correctness | recall recovers corrupted memories below `α_c` | **PASS** — 27 unit tests |
| G2 capacity | measured `α_c` at our real `N`; capacity grows with `d` | **PASS** |
| G3 no-regression | opt-in, default-off; `--all-features` clean | **PASS** |
| G4 perf | `O(d³)` paths sub-µs at `d ≤ 4`, alloc-free | **PASS** (after fixing a plan cost-model error) |
| G5 Plan 276 unblock | flips ≤ 10× leaky **and** tracking ≥ leaky − 0.05 | **PASS, narrowly** — see §G5 |
| G6 Fusion B (KG capacity) | ≥ 3× cosine-ANN triple capacity | **NOT MEASURED** — Phase 6 deferred |
| G7 BBP gap | relative gap > 0.1 at finite `N` | **PASS** — strongest result |

**Promotion decision: keep `cp_hopfield` opt-in.** Per the Plan 567 T7.2 decision
table, default-on requires G5 **and** G6 **and** G7. G6 is unmeasured (Phase 6
deferred to riir-neuron-db), and G5 passes only in the narrow sense, so the
promotion precondition is not met.

---

## G2 — capacity (T4.2, T4.3, T4.4)

Haar-random memories, 40 % corruption, `α_c` = load where mean overlap `m̄` crosses
0.5. Sweep `α ∈ [0.02 … 10.0]`, 3–4 realizations per cell.

| d | N | `α_c` measured | `α_c` paper (asymptotic) | `m̄` at lowest load |
|---|---|---|---|---|
| 2 | 8 | 0.594 | 0.05 | 1.000 |
| 2 | 64 | 0.174 | 0.05 | 1.000 |
| 2 | 256 | 0.099 | 0.05 | 0.979 |
| 3 | 8 | **1.696** | 0.62 | 1.000 |
| 3 | 64 | 1.295 | 0.62 | 1.000 |
| 3 | 256 | 0.909 | 0.62 | 0.996 |
| 4 | 8 | 3.752 | 2.41 | 1.000 |
| 4 | 64 | 3.666 | 2.41 | 1.000 |

**The paper's `d`-scaling holds at every `N` measured.** At `N = 64`,
`α_c(d=3) = 1.295 > α_c(d=2) = 0.174` — a 7.4× gain from moving off the sphere,
which is the falsifiable core of the capacity claim.

### Finding 1 — the plan's finite-`N` risk is refuted

Plan 567's Risk Register lists *"finite-`N` `α_c` much lower than asymptotic"* and
flags `α_c(N=8, d=3) ≪ 0.62` as a threat to Fusion A. **The opposite is true.**
Measured `α_c` is consistently *higher* at small `N` and decreases monotonically
toward the paper's asymptote as `N` grows (d=3: 1.696 → 1.295 → 0.909 for
N = 8 → 64 → 256). Small `N` is favorable, not hostile, so Fusion A was never at
risk from finite-size effects.

Caveat on the `N = 8` row: `P = round(α·N)` is an integer, so realized load is
quantized to steps of 1/8 and the lowest testable load is `P/N = 0.125`. The
`α_c` interpolation runs against realized `P/N`, not requested `α`, precisely so
the reported crossing is a load that was actually exercised.

### Finding 2 — correlated memories break the `α_c` metric, not the mechanism

| Spread (rad) | `α_c` | `m̄` at α = 0.62 |
|---|---|---|
| 0.2 | above sweep (> 10) | 0.990 |
| 0.5 | above sweep (> 10) | 0.935 |
| 1.0 | above sweep (> 10) | 0.745 |

Correlated memories never cross the recall threshold within a sweep reaching
`α = 10`. Plan 567 T4.4 expected correlation to *reduce* `α_c`; measured, it
recalls the cued memory *better* (0.97 vs Haar's 0.45 at `α = 1.0`) because
near-parallel memories reinforce rather than interfere.

`α_c` is simply the wrong metric here. What correlation destroys is
**discriminability**, i.e. the paper's §1.6 shadow phenomenon: recall from one cue
drags un-cued correlated memories along. Covered by the unit test
`correlated_memories_show_shadow_phenomenon`, which measures overlap with a
*never-cued* memory — near-zero for Haar (winner-takes-all), high for correlated.
Per Research 466 §1.6 this is desirable for KG retrieval (related context) and
undesirable for personality recall (bleed).

---

## G4 — perf (T2.4, T3.7)

`d = 3`, `N = 64`, `P = 16`. Release build, `std::time::Instant`, `CountingAllocator`.

| Kernel | Before | After | Target |
|---|---|---|---|
| `recall_step` | 1946 ns | **331 ns** | < 1 µs |
| `project_to_manifold` | 511 ns | **239 ns** | < 1 µs |
| `llg_step_neuron` | 2082 ns | **589 ns** | < 1 µs |

Allocations: 0 per 100 calls on both `project_to_manifold` and `recall_step`.

### G4 initially FAILED — the plan's cost model was wrong

Research 466 bills recall as *"`O(d³)` per neuron per recall step — trivially cheap
(d=3 → 27 flops)"*. The `d³` eigendecomposition **is** trivial; the actual cost is
dominated by the `O(P·N·D2)` Mattis overlaps, which that framing omits entirely.
At `P=16, N=64, D2=8` that is ~8 K multiply-adds per `recall_step` versus the 27
flops the note advertises — two orders of magnitude apart. Two fixes:

1. **Cache the global Mattis sum.** `O_μ^(i) = (T_μ − s_i·s_i^μ)/N` where
   `T_μ = Σ_j s_j·s_j^μ` is independent of `i`, so no per-neuron loop is needed.
   Maintained incrementally on every state write (an exact algebraic identity, not
   an approximation); `build_memory_kernel` drops from `O(P·N·D2)` to
   `O(P·(D2+d²))`. All state writes funnel through one private `write_state` so
   the cache cannot drift, and `overlap_cache_matches_full_recompute` asserts
   agreement with a full recomputation after many sweeps.
2. **Shift the power iteration minimally.** The obvious shift — the Gershgorin
   *radius* — inflates every eigenvalue by roughly the spectral norm, pushing
   `λ₂/λ₁` toward 1 and doubling the iteration count for bit-identical output.
   Using the Gershgorin *lower bound* instead (and zero when the matrix is already
   provably non-negative) halves the work: on `ρ = |ξ⟩⟨ξ|` with `ξ = (1,1,1)/√3`
   the radius gives spectrum `(2,1,1)` (ratio 1/2) where the minimal shift gives
   `(4/3,1/3,1/3)` (ratio 1/4).

Side effect: the unit suite dropped from 1.11 s to 0.20 s.

---

## G7 — BBP gap (T5.4) — the strongest result

Relative gap `(λ_max − λ₂)/λ_max` of the memory kernel `K_i`, averaged over all
neurons, at `d = 3` using the *measured* `α_c = 1.295` rather than the paper's.

| N | α | α/α_c | Relative gap |
|---|---|---|---|
| 8 | 0.324 | 0.25 | 0.944 |
| 8 | 0.647 | 0.50 | 0.858 |
| 8 | 0.971 | 0.75 | 0.948 |
| 64 | 0.324 | 0.25 | 0.832 |
| 64 | 0.647 | 0.50 | 0.748 |
| 64 | 0.971 | 0.75 | 0.732 |

**Target was > 0.1; measured 0.73–0.95 everywhere.** BBP protection demonstrably
survives at the `N` this codebase actually uses, not merely asymptotically. Since
the entire capacity claim is a statement about this gap, G7 is the gate that most
directly validates the mechanism — and it passes with 7× margin at worst.

---

## G5 — Plan 276 unblock (LOAD-BEARING)

Full analysis in [Research 466 §3.6](../.research/466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md).
Plan 276 G2.1 protocol, 1000 steps, dim = 8, seed `0xC0FFEE`.

| Kernel | Flips | Tracking | Verdict |
|---|---|---|---|
| `LeakyIntegrator` | 1 | 1.000 | reference |
| `AttractorKernel` (random init) | 347 | 0.000 | fails both axes |
| **`CpHopfield` (task-aligned memories)** | **3** | **1.000** | **PASS** |
| `CpHopfield` (Haar-random memories) | 0 | 0.000 | degenerate FAIL |

**Flip count alone is not a sufficient criterion.** A kernel that ignores its input
is perfectly stable and scores 0 flips — better than `LeakyIntegrator`, whose
single flip is the *correct* phase transition. The gate therefore also requires
**tracking** (argmax correct in the settled tails of both driven phases).
Hysteresis means resisting noise, not resisting evidence.

What passes: CP² recall with task-aligned memories beats the demoted
`AttractorKernel` on *both* axes — flips 347 → 3 and tracking 0.000 → 1.000 — with
no gradient descent.

What does not: the Haar-random control fails at tracking 0.000 across all 20
(seed, snap) cells. So the BBP gap does **not** confer hysteresis from arbitrary
memories; the memory set must align with the beliefs to be recalled. Plan 276's
blocker allowed *"trained **or hand-set**"* weights, so what this refutes is the
*training* requirement, not the *alignment* requirement — which is exactly
freeze/thaw Path 1, and exactly what T5.1 intended (load
`NeuronShard::style_weights` as memories).

**Robustness caveat.** Flips are non-monotone in snap strength:
`[48, 1, 3, 21, 9]` across snap 0.00–1.00. Notably snap = 0 (manifold projection,
no memory snap) scores 48 — *worse* than leaky — so the projection alone costs
stability and the memory term must pay it back. The result depends on a
hyperparameter with no principled setting and should not be read as a clean margin.

---

## Reproduce

```bash
# G2 / G4 / G7
CARGO_TARGET_DIR=/tmp/cp_hopfield_goat cargo bench -p katgpt-core \
    --features cp_hopfield --bench bench_567_cp_hopfield_goat -- --nocapture

# G1
cargo test -p katgpt-core --features cp_hopfield --lib cp_hopfield

# G5 (riir-ai)
CARGO_TARGET_DIR=/tmp/plan567 cargo bench -p riir-poc \
    --bench cp_hopfield_plan276_unblock
```

---

## Follow-ups

1. **G6 / Phase 6** — Fusion B KG-triple capacity in riir-neuron-db. The blocker
   for promotion to default-on.
2. **Snap sensitivity** — the non-monotone sweep needs either a principled snap
   setting or a reformulation that removes the hyperparameter (the driven LLG flow,
   `llg_step_driven`, is the natural candidate: it couples input into the field
   rather than blending two state estimates post hoc).
3. **N=2 degeneracy** — re-run G5 at a non-degenerate `N` once a consumer supplies
   a real frozen memory set.
4. **Closed-form eigensolvers** (T1.6, unimplemented) — only worth doing if a
   consumer needs better than the current 331 ns.
