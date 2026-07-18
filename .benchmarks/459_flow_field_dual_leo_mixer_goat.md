# Plan 459 — FlowField × DualLeoMixer Fusion GOAT Gate Results

**Date:** 2026-07-18
**Repo:** `katgpt-rs`
**Features:** `flow_field_nav` + `dual_leo`
**Bench:** `crates/katgpt-core/benches/dual_flow_field_bench.rs`
**Run:** `cargo bench -p katgpt-core --bench dual_flow_field_bench --features flow_field_nav,dual_leo`

## TL;DR

- ✅ **G1 PASS** — `get_or_compute_dual(LeoOnly)` is bit-identical to `get_or_compute` with the same head. The refactor is correct.
- ✅ **G2 PASS** — Dual-mix path adds **1.11–1.12×** cache-miss overhead (well under the 1.5× gate). The mix is `O(cells × actions)`, negligible vs FFT.
- ❌ **G5 FAIL** — On a synthetic 64×64 landscape with broad-LEO + sharp-UVFA mock heads, **no α in {0.1..0.9} reduces stuck-NPC count by ≥30%** vs the LEO-only baseline. Best α (0.1) achieves 25.9% reduction; the paper's default α=0.3 achieves only 3.7%.

**Verdict (honest demotion):** The API is correct + cheap, so it stays landed as opt-in. It is **NOT promoted** as a recommended path — the documented quality gain does not hold on synthetic data. Real gain requires (a) a real `CivLeoNet` + trained UVFA pair, AND/OR (b) a pipeline change to mix **post-max potentials** instead of raw Q-slices (see "Root cause").

## Setup

- Grid: 64×64
- Goal: (32, 32) — center
- NPCs: 200 random starts
- Mock LEO teacher: **broad, multimodal** — inverse-distance peak at the goal PLUS a decoy peak at (16, 16), low sharpness (slowly decaying). Simulates "knows all goals".
- Mock UVFA student: **sharp, unimodal** — single inverse-distance peak at the commanded goal, high sharpness. No decoy. Simulates "precise on this goal".
- Quality metric: gradient-following simulator (bilinear-sampled `FlowField::lookup`, step size 0.5, 1024-step budget, 32-step stall detection). Counts Reached / Stuck / OutOfBounds over 200 random starts.

## G1 — Correctness (bit-identity)

`get_or_compute_dual` with `ActingMode::LeoOnly` produces a `FlowField` that is **bit-identical** to `get_or_compute` with the same head, same state, same goal.

| Run | Result |
|---|---|
| 1 | ✅ PASS |

This confirms the refactor (T1) preserved behavior: the helper `compute_from_q_slice` is pure, and the LeoOnly path through `combine_into` is a slice copy.

## G2 — Perf overhead

30 cold-cache computes on a 64×64 grid, `std::time::Instant` timing (no Criterion dep, per the crate convention).

| Path | Time | Ratio vs LeoOnly-single |
|---|---|---|
| `get_or_compute` (single, LeoOnly baseline) | 2.087 ms | 1.00× |
| `get_or_compute_dual` (UvfaOnly) | 1.949 ms | 0.95× |
| `get_or_compute_dual` (Lc α=0.3) | 2.318 ms | **1.11×** |
| `get_or_compute_dual` (Max α=0.3) | 1.983 ms | 0.95× |

**Gate (≤1.5×): PASS.** Worst case 1.11× — the α-mix is essentially free vs the FFT cost.

## G5 — Quality gain (the gate that failed)

200 NPC gradient-follow simulation. "Stuck" = either zero-flow local minimum or no progress for 32 consecutive steps.

| Config | Reached | Stuck | OOB | Avg steps (reached) |
|---|---|---|---|---|
| **LeoOnly (LEO baseline)** | 62.0% | **26.5%** | 11.5% | 43.7 |
| UvfaOnly (UVFA student) | 88.0% | 0.0% | 12.0% | 46.1 |
| Lc α=0.3 (paper default) | 64.5% | 25.5% | 10.0% | 43.8 |
| Max α=0.3 (optimistic) | 62.0% | 26.5% | 11.5% | 43.7 |

**Stuck-NPC reduction: 3.7% (Lc) / 0.0% (Max). Gate (≥30%): FAIL.**

### α-sweep (Lc mode, looking for any α that meets the gate)

| α | Reached | Stuck | Reduction vs LeoOnly |
|---|---|---|---|
| 0.10 | 69.5% | 19.5% | **25.9%** ← best, still under gate |
| 0.20 | 64.0% | 24.5% | 7.4% |
| 0.30 | 64.5% | 25.5% | 3.7% (paper default) |
| 0.40 | 64.0% | 25.5% | 3.7% |
| 0.50 | 62.5% | 27.0% | 0.0% |
| 0.60–0.90 | ≈62% | ≈26.5% | 0.0% |

Monotonic: more UVFA weight → fewer stuck, but even α=0.1 (mostly UVFA) only reaches 25.9% reduction — short of the 30% gate.

## Root cause (why G5 failed)

The downstream pipeline has two nonlinear operations:

1. **`LeoPotentialGrid::from_q_values`** applies `max-over-actions` per cell — `potential[x,y] = max_a Q[x,y,a]`. Max is not linear: `max_a (α·Q_leo[a] + (1-α)·Q_uvfa[a]) ≠ α·max_a Q_leo[a] + (1-α)·max_a Q_uvfa[a]` in general.
2. **FFT low-pass smoothing** is linear, but it operates on the post-max potential, so the nonlinearity above propagates.

Result: mixing raw Q-values at α=0.3 and *then* taking max-over-actions gives a different (and apparently less useful) field than the α=0.3 weighting would suggest. The LEO decoy peak survives the max-pool even when α is low, because the mix is done per-action before the max.

**The fix (out of scope for this plan)** is to mix at a different pipeline stage:
- Option A: Mix **post-max potentials** (`α·potential_leo[x,y] + (1-α)·potential_uvfa[x,y]`), not raw Q-slices. This is linear in the field that the FFT sees.
- Option B: Skip FFT on the UVFA path and blend the post-FFT LEO field with the raw UVFA gradient.

Either would require extending the `FlowFieldCache` API to accept two pre-built `LeoPotentialGrid`s instead of two heads. Worth a follow-up plan if real-network evidence suggests the gain is there.

## What this DOES prove

- The `get_or_compute_dual` API works as specified.
- All 5 `ActingMode`s are honored (Lc / LeoOnly / UvfaOnly / Max / Min).
- LeoOnly is bit-identical to the single-head path — no regression for callers that opt into the dual API but choose LeoOnly.
- UvfaOnly recovers the UVFA-only field — the dual API subsumes both single-head baselines.
- Perf overhead is negligible (≤1.12×).
- The α knob **does** have a monotonic effect — just not large enough on this landscape to meet the gate.

## What this DOES NOT prove

- That the fusion improves navigation quality on any real game scenario.
- That the paper's α=0.3 default is right for flow-field navigation (it isn't, on this synthetic landscape — α=0.1 dominates).
- That mixing raw Q-values is the right fusion point (evidence suggests it isn't — see Root cause).

## Promotion decision

**Stay opt-in, document honestly.** The `get_or_compute_dual` method is landed behind `feature = "dual_leo"` (already default-on, but the method itself is opt-in via API choice). The `flow/cache.rs` doc-comment must call out:

1. The default α=0.3 from the LEO paper does NOT necessarily translate to flow-field navigation.
2. The nonlinear max-over-actions + FFT pipeline can wash out the α-mix.
3. Real quality gain requires a real CivLeoNet + UVFA pair, OR a future revision that mixes post-max potentials.

## Follow-up

- **Real-network evidence**: wire `get_or_compute_dual` into `riir-games-civ` with `CivLeoNet` + a UVFA wrapper. Open a plan in `riir-ai` if civ navigation quality becomes a priority.
- **Pipeline-stage fusion**: open a plan in `katgpt-rs` to add `get_or_compute_dual_postmax` that mixes post-max potentials. The API would take two `LeoPotentialGrid`s instead of two heads. Pre-condition: evidence that post-max mixing is the right fusion point (could be the same plan as above).

## Connection to existing GOAT-proved work

| Plan / Issue | Status | Connection |
|---|---|---|
| Plan 155 (LEO All-Goals) | ✅ DEFAULT-ON SUPER GOAT | Source of `LeoHead` + `DualLeoMixer` traits. Plan 459 adds a 2nd consumer of `DualLeoMixer` (was 1: QuestLeoScorer; now 2). |
| Plan 242 (Fourier Flow Fields) | ✅ DEFAULT-ON | Source of `FlowFieldCache::get_or_compute`. Plan 459 adds the dual sibling `get_or_compute_dual`. |
| Plan 268 (QGF) | ✅ (opt-in) | `LeoHeadOracle` consumes `LeoHead`. A future `DualLeoOracle` would be a natural sibling — out of scope here. |
