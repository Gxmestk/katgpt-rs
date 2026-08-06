# Plan 571: Phase Separation Probe — Open Primitive (Lonely Runner Conjecture)

**Date:** 2026-08-06
**Research:** [katgpt-rs/.research/470_Lonely_Runner_Phase_Separation_Probe.md](../.research/470_Lonely_Runner_Phase_Separation_Probe.md)
**Source paper:** [arXiv:0710.4495](https://arxiv.org/abs/0710.4495) — Barajas & Serra, *The Lonely Runner with Seven Runners* (2007)
**Private guide:** [riir-ai/.research/334_phase_separation_game_runtime_guide.md](../../riir-ai/.research/334_phase_separation_game_runtime_guide.md)
**Target:** `katgpt-rs/crates/katgpt-core/src/phase_separation.rs` (new module) + Cargo feature `phase_separation`
**Status:** Active — Phase 1 (skeleton)

---

## Goal

Ship a generic, modelless `phase_separation_probe` that computes per-entity minimum circular distance on a phase circle. The primitive is the open (public) layer of the Super-GOAT fusion described in Research 470; the private game-runtime fusion (Salience Tri-Gate × Sleep-Time × KARC × feeling brain) lives in riir-ai per the private guide 334.

The primitive computes, for N entities each with a phase `φ_i ∈ [0, 1)`:

```
phase_separation(i) = min_{j ≠ i} ‖φ_i − φ_j‖ mod 1     ∈ [0, 0.5]
```

where `‖x‖ mod 1` is the distance to the nearest integer (circular distance on the unit torus). O(N log N) via sort + adjacent-neighbor scan.

**Theorem backing** (Lonely Runner Conjecture, proven for N ≤ 7 by Barajas & Serra 2007): for N entities with integer cycle speeds {s_1, ..., s_N} (gcd = 1), every entity i has some tick t where `phase_separation(i) ≥ 1/N`. The primitive computes the per-tick scalar; the theorem justifies using it as a behavior driver (guaranteed-peak property).

**GOAT gate**: G1 (determinism on integer phases — bit-identical), G2 (sub-µs at N=1000), G3 (no-regression, feature-flagged), G4 (alloc-free steady-state). No UQ floor (this is a deterministic distance metric, not a probability distribution).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Add `phase_separation` feature flag to `katgpt-rs/crates/katgpt-core/Cargo.toml` (opt-in, default-off).
- [ ] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/phase_separation.rs` module with:
  - `phase_separation(phases: &[f32], i: usize) -> f32` — O(N) naive (for correctness testing + small N).
  - `phase_separation_all(phases: &[f32], out: &mut [f32])` — O(N²) all-pairs (for correctness testing).
  - `phase_separation_sorted(phases_sorted: &[f32], scratch: &mut [f32], out: &mut [f32])` — O(N log N) via sort + adjacent-neighbor scan (production path).
  - All three write into caller-provided `&mut [f32]` — zero allocation.
- [ ] **T1.3** Implement the sorted-scan algorithm:
  1. Copy `phases` into `scratch`, sort ascending.
  2. For each entity i, its separation = min(distance to left neighbor, distance to right neighbor) on the circle. The circle wrap-around is handled by also checking the first/last elements (distance = 1.0 - last + first).
  3. Write into `out[i]`.
  - Edge cases: N=0 → return 0.0; N=1 → return 0.5 (single entity is maximally alone); N=2 → return `‖φ_0 − φ_1‖ mod 1`.
- [ ] **T1.4** Unit tests (G1 determinism):
  - `g1_integer_phases_bit_identical`: phases from integer speeds `{1, 2, 3, 4, 5, 6, 7}` at tick `t=42` → verify `phase_separation_all` produces the same f32 bits as a reference Python/reference Rust implementation.
  - `g1_circle_wraparound`: phases `{0.0, 0.49, 0.51}` → entity 0's separation should be `min(0.49, 0.49) = 0.49` (wrap-around: 1.0 - 0.51 + 0.0 = 0.49), NOT `min(0.49, 0.51) = 0.49` (same answer here, but test the wrap path explicitly).
  - `g1_edge_cases`: N=0 → 0.0; N=1 → 0.5; N=2 with phases `{0.0, 0.5}` → 0.5 each.
  - `g1_lrc_bound_n7`: with N=7 entities, integer speeds `{1,2,3,4,5,6,7}` (gcd=1), scan ticks t=0..1000 and verify every entity hits `phase_separation ≥ 1/7 ≈ 0.1428` at least once. This is the **theorem confirmation test** — the LRC says it must happen.
- [ ] **T1.5** Re-export at crate root: `katgpt-rs/crates/katgpt-core/src/lib.rs` → `#[cfg(feature = "phase_separation")] pub mod phase_separation;`

### G2 — Perf gate

- [ ] **T1.6** Add criterion bench `katgpt-rs/crates/katgpt-core/benches/bench_571_phase_separation_goat.rs`:
  - N = {10, 100, 1000, 10000} entities.
  - Measure `phase_separation_sorted` wall time.
  - Target: < 10µs at N=1000 (sub-µs expected).
  - Report O(N log N) scaling (N=10000 should be ~13× N=1000, not 100×).

### G3 — No-regression gate

- [ ] **T1.7** `cargo test -p katgpt-core --lib` passes with `phase_separation` off (default) AND on (`--features phase_separation`).
- [ ] **T1.8** `cargo clippy -p katgpt-core --all-targets --features phase_separation` zero warnings.

### G4 — Alloc-free gate

- [ ] **T1.9** `CountingAllocator` test: call `phase_separation_sorted` 1000 times on a pre-allocated scratch buffer at N=1000. Assert zero allocations after warmup (the sort writes into the caller-provided scratch; the scan is in-place).

---

## Phase 2 — API ergonomics (after Phase 1 GOAT passes)

### Tasks

- [ ] **T2.1** Add `PhaseSeparationProbe` struct (zero-sized, `Copy`) wrapping the sorted-scan with a cached scratch buffer:
  ```rust
  pub struct PhaseSeparationProbe {
      scratch: Vec<f32>,  // pre-allocated, reused across calls
  }
  impl PhaseSeparationProbe {
      pub fn new(capacity: usize) -> Self { ... }
      pub fn compute(&mut self, phases: &[f32], out: &mut [f32]) { ... }
  }
  ```
  This lets callers pre-allocate once and reuse across ticks (the MMORPG 20Hz tick pattern).
- [ ] **T2.2** Add `from_speeds_and_tick(speeds: &[u32], tick: u64, out_phases: &mut [f32])` helper that computes `(s_i * tick) mod P` into `out_phases` before the separation scan. `P` = the period (caller-specified, or LCM of speeds). This is the raw time-phase path (sync-safe).
- [ ] **T2.3** Add `from_latent_projection(latent_states: &[f32], direction: &[f32], out_phases: &mut [f32])` helper that computes `sigmoid(dot(d, latent_state_i))` into `out_phases`. This is the latent-phase path (local-only, not synced). Documents the bridge pattern explicitly.

---

## Phase 3 — Documentation (after Phase 2)

### Tasks

- [ ] **T3.1** Module-level rustdoc with:
  - The LRC citation + scope caveat (N≤7 proven, N>7 conjectured).
  - The raw-vs-latent boundary (per AGENTS.md).
  - The bridge pattern (raw time-phase → latent-projected phase → separation scalar).
  - Cross-ref to Research 470 + private guide 334.
- [ ] **T3.2** Example in `katgpt-rs/crates/katgpt-core/examples/phase_separation_demo.rs`:
  - 7 entities with integer speeds `{1,2,3,4,5,6,7}`.
  - Scan 1000 ticks, print the tick where each entity hits max separation.
  - Confirm the LRC bound (every entity hits ≥ 1/7).

---

## Non-goals

- **NOT implementing the fusion.** The Salience Tri-Gate / Sleep-Time / KARC / feeling-brain fusions are riir-ai tasks tracked in the private guide (334). This plan ships ONLY the generic primitive.
- **NOT formalizing the LRC in Lean 4.** The theorem is published; formalizing the 20-page case analysis is a separate research project. The primitive's invariant (min over a metric → non-negative, ≤ 0.5) is trivially true by construction.
- **NOT handling N > 7 with a proven bound.** The LRC is conjectured for N > 7. The primitive computes the scalar correctly for any N; only the *peak guarantee* is conjectural at scale. Honest framing in the docs.
- **NOT a UQ primitive.** `phase_separation` is a deterministic distance metric, not a probability distribution. No conformal-naive floor comparison needed (per the "Report the Floor" rule).

---

## See also

- [Research 470](../.research/470_Lonely_Runner_Phase_Separation_Probe.md) — public distillation + Super-GOAT verdict
- [riir-ai/.research/334](../../riir-ai/.research/334_phase_separation_game_runtime_guide.md) — private game-runtime guide + fusion map
- [Research 056](../.research/056_OpenAI_Unit_Distance_Disproof.md) — same combinatorial family (chromatic number bounds on distance graphs)
- [Plan 303](303_per_tick_salience_tri_gate.md) — Salience Tri-Gate (primary fusion target, riir-ai follow-up)
