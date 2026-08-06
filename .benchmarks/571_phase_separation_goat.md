# Bench 571 — Phase Separation Probe GOAT Gate (G1–G4 ALL PASS)

**Date:** 2026-08-07
**Plan:** [Plan 571](../.plans/571_phase_separation_probe.md) — Phase Separation Probe
**Primitive:** `phase_separation` feature in `katgpt-core` (per-entity minimum circular distance on a phase circle, distilled from the Lonely Runner Conjecture)
**Verdict:** **ALL GATES PASS — promotion candidate.**

---

## Gate results

| Gate | Target | Result | Status |
|---|---|---|---|
| **G1** (determinism + LRC bound) | 8 lib unit tests | 8/8 PASS | ✅ PASS |
| **G2** (perf @ N=1000) | < 10µs/call | 7947 ns/call (7.9µs) | ✅ PASS |
| **G2** (O(N log N) scaling) | N=10000/N=1000 < 20× | 12.49× (target ~13×) | ✅ PASS |
| **G3** (no-regression) | 0 regressions, feature on vs off | 1845 → 1853 (+8 new), 0 regressions | ✅ PASS |
| **G4** (alloc-free steady-state) | 0 allocs / 1000 calls @ N=1000 | 0 allocs | ✅ PASS |
| **Smoke** (LRC N=7 range) | all separations ∈ [0, 0.5] | max = 0.0405, all valid | ✅ PASS |

## G2 perf breakdown

```
N=10    :       94.8 ns/call
N=100   :      633.8 ns/call
N=1000  :     7947.0 ns/call   ← under 10µs target
N=10000 :    99280.2 ns/call   ← 12.49× N=1000 (O(N log N) predicts ~13×)
```

The production path (`phase_separation_sorted`) uses a sort + adjacent-neighbor
scan. The scaling confirms O(N log N): the N=10000/N=1000 ratio is 12.49×, well
under the 20× gate target and close to the theoretical ~13× (N·log₂N scaling).

## G1 evidence (lib tests)

```
test phase_separation::tests::from_latent_projection_basic ... ok
test phase_separation::tests::from_speeds_and_tick_basic ... ok
test phase_separation::tests::g1_circle_wraparound ... ok
test phase_separation::tests::g1_edge_cases ... ok
test phase_separation::tests::g1_tie_handling_tick_zero ... ok
test phase_separation::tests::g1_integer_phases_bit_identical ... ok
test phase_separation::tests::g1_lrc_bound_n7 ... ok
test phase_separation::tests::g1_sorted_matches_naive_random ... ok
test result: ok. 8 passed; 0 failed; 0 ignored
```

The `g1_lrc_bound_n7` test confirms the Lonely Runner Conjecture bound: with 7
entities at integer speeds {1..7}, every entity cycles through
`phase_separation ≥ 1/7` (the LRC guarantee for N≤7, proven).

## G4 evidence

```
[✓ PASS] G4.alloc_free_steady_state        0 allocs / 1000 calls at N=1000
```

CountingAllocator (katgpt-core test harness) tracks heap allocations. 1000
steady-state calls at N=1000 with a pre-allocated scratch buffer → 0 allocations
after warmup. The sorted path reuses the caller-owned buffer.

## Promotion assessment

**This primitive is a promotion candidate.** All GOAT gates (G1–G4) PASS
modellessly (closed-form modular arithmetic + sigmoid + dot-product, no
training). The perf headroom is large: 7.9µs at N=1000 vs the 50ms tick budget
(6300× headroom). The alloc discipline is clean (0 allocs steady-state).

**Selling point (per the Cargo.toml comment):** the LRC guarantees every entity
cycles through `phase_separation ≥ 1/N` — a coverage guarantee no existing
primitive provides. This is a Think-brain primitive (latent, local-only) suitable
for zone-attention routing or curiosity/coverage scoring where "has this entity
been far enough from the crowd recently?" matters.

**Consumer gap:** no concrete consumer exists yet. Per GOAT discipline, promotion
to default-on is justified by the gate PASS even without a consumer (the
substrate is proven correct + fast + alloc-free; a consumer can opt in or the
primitive can default-on and wait for discovery). The Cargo.toml comment says
"Opt-in until GOAT gate G1–G4 PASS" — the gate has now PASSed.

**Recommendation:** promote to default-on in katgpt-core. Cherry-pick tracking
for riir-ai opens at promotion (the 7-day clock starts then).

## Reproduction

```bash
# G2 + G4 (bench)
CARGO_TARGET_DIR=/tmp/katgpt-plan-571 cargo build --release -p katgpt-core \
    --features phase_separation --bench bench_571_phase_separation_goat
/tmp/katgpt-plan-571/release/deps/bench_571_phase_separation_goat-* --nocapture

# G1 (lib tests)
CARGO_TARGET_DIR=/tmp/katgpt-plan-571 cargo test -p katgpt-core \
    --features phase_separation --lib phase_separation

# G3 (no-regression — feature off vs on)
CARGO_TARGET_DIR=/tmp/katgpt-plan-571 cargo test -p katgpt-core --lib
CARGO_TARGET_DIR=/tmp/katgpt-plan-571 cargo test -p katgpt-core --lib --features phase_separation
```

Apple Silicon (M3), release mode. Isolated `CARGO_TARGET_DIR=/tmp/katgpt-plan-571`.

## Environment

- **Hardware:** Apple M3 (8-core CPU)
- **Toolchain:** Rust stable (katgpt-core MSRV)
- **Date:** 2026-08-07
- **katgpt-rs commit:** `a036def3` (develop)
