# Plan 459: FlowField × DualLeoMixer Fusion (modelless navigation quality gain)

**Status:** DONE — honest demotion (G1+G2 PASS, G5 FAIL). See [`.benchmarks/459_flow_field_dual_leo_mixer_goat.md`](../.benchmarks/459_flow_field_dual_leo_mixer_goat.md).
**Branch:** `develop`
**Repo:** `katgpt-rs`
**Started:** 2026-07-18
**Completed:** 2026-07-18

## Context (the gap)

Plan 155 shipped two SUPER GOAT default-on traits:

- `LeoHead` — all-goals Q-head (`leo_all_goals` feature)
- `DualLeoMixer` — teacher (LEO) / student (UVFA) α-mixing (`dual_leo` feature), with 5 acting modes (`Lc | LeoOnly | UvfaOnly | Max | Min`) and `AlphaSchedule`

Audit on 2026-07-18 found **only ONE concrete `DualLeoMixer` consumer exists** (`QuestLeoScorer` in `riir-ai/crates/riir-games-quest/src/quest_grammar/leo_scoring.rs`). The base `LeoHead` trait is consumed in 5 places (Bomber/Go/TFT/Civ LEO nets + FlowFieldCache + QGF LeoHeadOracle), but the *dual* half is essentially unused.

`FlowFieldCache::get_or_compute<H: LeoHead>` (Plan 242) consumes the LEO teacher only — the cache builds a smoothed navigation field from raw `Q_LEO[:,:,g]` slices. This is functionally equivalent to running in `ActingMode::LeoOnly`, which the JAX source explicitly labels an **ablation**, not the default. The sweep winner was `ActingMode::Lc` at α=0.3.

**This plan fuses them.** Goal: add a `get_or_compute_dual` path that mixes LEO + UVFA Q-slices via `DualLeoMixer::combine_into` before building the `LeoPotentialGrid`, then prove the fusion produces a measurably better navigation field (shorter avg path length / fewer local minima) than the LEO-only baseline.

## Hypothesis (what we expect to prove)

On a 2D navigation grid, the LEO teacher produces a *broad* potential field (knows all goals → smooth but possibly multimodal), and a UVFA student produces a *sharp* field (precise on the commanded goal → unimodal but possibly noisy if under-trained). The paper's default `Lc` mix at α=0.3 should:

1. **H1 (quality gain):** Reduce average gradient-following path length to goal by ≥10% vs `LeoOnly` baseline, OR reduce local-minima cells by ≥30%.
2. **H2 (perf overhead):** Dual-mix path costs ≤1.5× the LeoOnly path on cache-miss (the α-mix is O(cells×actions), negligible vs FFT).
3. **H3 (cache parity):** Cache-hit latency is identical (the cached field is the same shape).
4. **H4 (modelless):** No training, no gradient descent. α is a deterministic scalar. SATISFIED BY CONSTRUCTION.

## Constraint Check

- **Modelless mandate (katgpt-rs/AGENTS.md):** α-mixing is a deterministic, modelless operation. ✅
- **Sync-boundary rule:** FlowField is a local latent structure, never synced. ✅
- **Manifold geometry / Stokes:** Flow field is 2D (d=2). Boundary-vs-volume perf rule says boundary-only is a win only for d≤3. The FFT-smooth path is a volume operation on the 2D potential grid — fine. ✅
- **Alloc-free hot loop (G4):** `DualLeoMixer::combine_into` writes into a pre-allocated buffer. ✅
- **Feature gate discipline:** New API ships behind existing `dual_leo` + `flow_field_nav` features. No new feature flag needed (composition of two default-on primitives).

## Tasks

- [x] T1: Refactor — extract `compute_from_q_slice` helper from `get_or_compute` (no behavior change).
- [x] T2: Add `FlowFieldCache::get_or_compute_dual<H1: LeoHead, H2: LeoHead, M: DualLeoMixer>`.
- [x] T3: Unit test — `get_or_compute_dual` produces a valid `FlowField` and respects `ActingMode::LeoOnly` (α=1.0 ≈ LEO baseline) and `ActingMode::UvfaOnly` (α=0.0 ≈ UVFA baseline).
- [x] T4: Write `crates/katgpt-core/benches/dual_flow_field_bench.rs` — quality metric (avg gradient-path length to goal) + perf metric (cache-miss compute time).
- [x] T5: Run GOAT gate. Capture results in `.benchmarks/459_flow_field_dual_leo_mixer_goat.md`.
- [x] T6: Document outcome + promote decision.
- [x] T7 (added during run): α-sweep {0.1..0.9} to characterize the actual quality curve after the default α=0.3 missed the gate.

## GOAT Gate Definition

| Gate | Criterion | Target | Method |
|------|-----------|--------|--------|
| **G1** | Correctness | `get_or_compute_dual` with `ActingMode::LeoOnly` produces bit-identical `FlowField` to `get_or_compute` with same LEO head, same state, same goal | Unit test bit-compare |
| **G2** | Perf overhead | Cache-miss latency dual/LeoOnly ratio | ≤ 1.5× on 64×64 grid |
| **G3** | No-regression | All existing `flow_field_nav` tests + bench pass | `cargo test -p katgpt-core --features flow_field_nav --lib`; existing bench unchanged |
| **G4** | Alloc-free hot path | `combine_into` writes into pre-allocated buffer; no `Vec::new` in compute path beyond the unavoidable `q_mixed` (which could itself be pre-allocated in a future revision) | Code inspection + criterion alloc counter (informational) |
| **G5** | **Quality gain** (the new gate) | Avg gradient-following path length to goal: `Lc(α=0.3)` < `LeoOnly` (LEO-broad field baseline) by ≥10%, OR local-minima count reduced ≥30% | New bench, 100 NPC starts, 64×64 grid |

**Promotion rule:** If G1-G4 pass AND G5 shows ≥10% path-length reduction OR ≥30% local-minima reduction → promote `get_or_compute_dual` as a recommended path (documented in `flow/cache.rs` doc-comment). Stays **opt-in** (caller chooses dual vs single) — does NOT replace `get_or_compute`, which remains the lowest-latency path for callers without a UVFA student.

**Demotion rule:** If G5 shows NO quality gain (within ±2% on both metrics), the fusion is still landed (it's correct + cheap) but the doc must say so honestly: "no measured quality gain on synthetic 2D grid; revisit with real CivLeoNet + a UVFA network when available."

## Outcome (filled in after T5 ran)

| Gate | Target | Result |
|------|--------|--------|
| G1 bit-identity | LeoOnly dual ≡ single-head | ✅ PASS |
| G2 perf overhead | ≤1.5× | ✅ PASS (1.11×) |
| G3 no-regression | All existing flow tests pass | ✅ PASS (46 existing + 5 new) |
| G4 alloc-free hot path | `combine_into` pre-allocated | ✅ PASS (code inspection) |
| G5 quality @α=0.3 | ≥30% stuck reduction | ❌ FAIL (3.7%) |
| G5' best-α sweep {0.1..0.9} | ≥30% stuck reduction at any α | ❌ FAIL (best 25.9% at α=0.1) |

**Decision:** API stays landed (correct + cheap, G1+G2 PASS). NOT promoted as a recommended path. Doc-comment in `flow/cache.rs` calls out the α=0.3 caveat and the post-max nonlinearity root cause. See [`.benchmarks/459_flow_field_dual_leo_mixer_goat.md`](../.benchmarks/459_flow_field_dual_leo_mixer_goat.md) for the full honest report including the α-sweep.

## Honest caveats up front

1. **Bench heads are synthetic.** The proof uses two mock heads (`BenchLeoTeacher` broad + `BenchUvfaStudent` sharp), not trained networks. The quality gain therefore demonstrates the *mechanism* works, not that it helps a specific game. Real CivleoNet + UVFA evidence requires riir-ai wiring (out of scope here — see "Follow-up").
2. **No Lean proof.** This is a perf/quality gate, not a correctness invariant. The Lean 4 instances in `.proofs/` are unaffected.
3. **`dual_leo` is already default-on.** This plan does not change default features; it adds a *consumer* of an existing default-on primitive.

## Follow-up (out of scope)

- Wire `get_or_compute_dual` into `riir-games-civ` (CivLeoNet + a UVFA wrapper) — separate plan in riir-ai.
- Wire into QGF `LeoHeadOracle` → `DualLeoOracle` — separate plan in katgpt-rs.
- Pre-allocate `q_mixed` inside `FlowFieldCache` scratch space (currently allocates per cache-miss; not a hot path so acceptable for v1).
