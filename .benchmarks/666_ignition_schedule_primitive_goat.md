# Benchmark 666: `IgnitionSchedule` modelless primitive GOAT (Issue 459 T5)

> **Primitive:** `katgpt-core/src/ignition.rs` behind opt-in feature `ignition_schedule`.
> **Provenance:** arXiv:2608.13335 "Neural Quadratic Forms" Thms 5–8 — riir-train Research 422 §3.5, Issue 459 T5 (the Path-0 modelless half).
> **Landed:** 2026-08-19 (4090 box).
> **Status:** opt-in. Promotion to default requires the consumer pilot win (below); decision stays with the owner.

## Surface

| Item | Form |
|---|---|
| `IgnitionSchedule::new(z0, k, zeta)` | constructor, contract-asserted (`0 < z0 < k`, `zeta > 0`) |
| `IgnitionSchedule::at(t)` | closed form `z(t) = K / (1 + ((K−z₀)/z₀)·e^{−ζt})` ≡ `K·σ(ζt − ln((K−z₀)/z₀))` — one `exp`, no iteration |
| `IgnitionSchedule::time_to_reach(target)` | per-curve inverse `t = ln((K−z₀)·target/((K−target)·z₀))/ζ` |
| `ignition_time(zeta, eps)` | the patience law `t* = ln(1/ε)/ζ` (Thm 8, capacity-free) |
| `order_by_ignition_into(zetas, &mut [usize])` | ζ-descending ignition order, index-ascending tie-break, caller-owned buffer |

Design anchors formalized: (1) sigmoid-in-time is the adoption shape GD itself
produces — the second grounding for sigmoid-not-softmax (after R315);
(2) patience ∝ 1/ζ — pre-ignition signal is ε-small, keying on raw rates
amplifies noise (the measured riir-clippy Issue 026 starved-pool negative is
the anchor this predicts).

## GOAT gates

| Gate | Evidence | Verdict |
|---|---|---|
| **G1** monotone ranking preservation | higher ζ ignites strictly earlier at every ε (t\* strictly decreasing over ζ ∈ [0.1, 4.0] at ε ∈ {1e-2, 1e-3, 1e-4}); curve ordering matches ζ ordering strictly pre-saturation (f32-exact K at saturation, by design); `order_by_ignition_into` output == observed threshold-crossing order; deterministic index-asc tie-break; ln(1/ε) amplification ratio exact to 1e-3 | **PASS** (4 tests) |
| **G2** ns-latency | closed form, no iteration: **3.88 ns/call** release / **13.15 ns** debug (n=100k; bounds 50/500 — 12.9× release headroom) | **PASS** |
| **G3** no-regression | default-features lib suite **1897 passed / 0 failed / 6 ignored** (exact pre-existing baseline — module cfg'd away); feature-on **1911/0/6** (+14); `cargo clippy -p katgpt-core --lib` 0 warnings in BOTH feature states | **PASS** |
| **G4** alloc-free | `TrackingAllocator` (per-thread, debug build): **0 allocations** across 1000× `at()` + 1000× `ignition_time()` + the ordering helper | **PASS** |

Correctness anchors beyond the gates (also in-module):
- **ODE RK4 anchor** — the closed form is the solution of the GLV dynamics
  `ż = ζ·z·(1 − z/K)`: RK4 (dt=0.01, 1000 steps) matches `at()` to < 5e-4
  relative at 10 checkpoints. This pins the formula to the theorem's dynamics,
  not just to a logistic shape.
- **Sigmoid-form identity** — `at(t)` ≡ `K·σ(ζt − ln((K−z₀)/z₀))` with the
  exact (non-approx) sigmoid to < 1e-5 relative (deliberately NOT
  `simd::fast_sigmoid` — the approximation would pollute the ranking gate).
- **Inverse roundtrip** — `at(time_to_reach(z*))` recovers `z*` to < 1e-5.

## Run

```bash
cargo test -p katgpt-core --features ignition_schedule --lib ignition::          # debug (14 tests)
cargo test --release -p katgpt-core --features ignition_schedule --lib ignition:: # release (13 — G4 is debug-only by design)
cargo test --release -p katgpt-core --features ignition_schedule --lib g2_latency -- --nocapture  # prints ns/call
```

## Consumer pilot (promotion gate — OPEN)

riir-clippy selection patience scaled by `ignition_time(ζ̂, ε)` vs fixed
patience on the heal-loop fixture; the Issue 026 starved-pool negative result
is the measured anchor (starved pools are all-ζ≈0 = plateaus beyond any
budget; amplifying pre-ignition evidence amplifies noise). Promotion to
default only on a measured win; stays opt-in otherwise.
