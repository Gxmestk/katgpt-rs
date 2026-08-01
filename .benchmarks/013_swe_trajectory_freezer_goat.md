# Bench 013 — T5.5 SweTrajectoryFreezer GOAT Gate (substrate-level)

**Date:** 2026-08-02
**Proposal:** P011 Phase 5 T5.5
**Feature gate:** `swe_trajectory_freeze` (implies `latent_trajectory_geometry` + `committed_field_blend`)
**Bench:** `benches/bench_013_swe_trajectory_freezer_goat.rs`

## Context

T5.5 ships the `SweTrajectoryFreezer` primitive — composes
`latent_trajectory_geometry::from_states` + `committed_field_blend::CommittedFieldBlend`
(FAME) + a local BLAKE3 envelope into a two-stage pipeline:

1. **Fit** (offline) — `derive_directions` from cluster centroids of labeled
   training summaries (the T5.3b data-derived-directions fix).
2. **Freeze** (online) — `SweTrajectoryFreezer::freeze_attempt`: encode
   trajectory geometry into summary → project onto pre-fit directions via
   FAME sigmoid → commit via BLAKE3 envelope.

This bench is the **substrate-level** GOAT gate. It does NOT run the T5.6
G5 gate (cross-snapshot/model discrimination) — that requires real-model
trajectories, and T5.4 PARTIAL documented that depth trajectories alone
are insufficient (G3 FAIL at 29%).

## Results (synthetic regime, DIM=8, N_STEPS=100, N=3, D=32)

### G1: directions non-degenerate — ✅ PASS

Data-derived directions from cluster centroids:
- All 3 unit-norm (within 1e-4).
- Pairwise cosine < 0.99 (distinct modes produce distinct directions).

This is the T5.3b fix encoded as a gate: random directions (T5.3's failure
mode) would have produced near-orthogonal directions with pairwise cosine
≈ 0, but the gates would have been near 0.5 (uniform blend → 17% accuracy).
Data-derived directions produce the 100% accuracy below.

### G3: cross-mode discrimination — ✅ PASS

| mode               | argmax_k | matching_gate | correct |
|--------------------|----------|---------------|---------|
| oscillation        | 0        | 0.9814        | true    |
| oscillation        | 0        | 0.9750        | true    |
| committed_wrong    | 1        | 0.7107        | true    |
| committed_wrong    | 1        | 0.7117        | true    |
| converged_correct  | 2        | 0.7235        | true    |
| converged_correct  | 2        | 0.7167        | true    |

**Accuracy: 1.00 (6/6 correct)** vs target ≥ 0.80.

The matching gates are well-separated:
- oscillation → 0.98 (very high — the π-curvature signature is unmistakable)
- committed_wrong → 0.71 (moderate — straight-line drift)
- converged_correct → 0.72 (moderate — damped convergence)

The committed_wrong vs converged_correct separation is tighter (0.71 vs
0.72) but still discriminates correctly because the non-matching gates are
correspondingly lower.

### G2: freeze_attempt latency — ✅ PASS

**per_call: 4582 ns** (target < 5000 ns).

Tight but under budget. The hot path is:
1. `from_states` (the substrate's documented ~1.4µs at 100×32)
2. `encode_into` (negligible — 8 blocks of 4 f32 writes)
3. FAME `commit` (simd_dot_f32 × 3 + sigmoid × 3 + BLAKE3)
4. Envelope construction (2× BLAKE3 over 154-byte payload + 48-byte header)

### G4: alloc-free steady state — ✅ PASS (with honest caveat)

**per_call: 2 allocs** (target ≤ 2 — substrate-inherited budget).

The 2 allocs are **entirely substrate-side**, inherited from
`latent_trajectory_geometry::from_states`, which allocates two `Vec<f32>`
displacement buffers per call (documented in its source: "Allocated ONCE
up front" means once per CALL, not once per session).

The freeze pipeline itself is zero-alloc:
- `encode_into`: stack array writes
- FAME `commit`: stack-fixed `[[u8; 32]; N]` + stack Hasher
- Envelope: stack-fixed `[u8; 512]` payload buffer + stack Hasher

**Follow-up:** add a `from_states_into` variant to the substrate that takes
pre-allocated scratch buffers (matches the `simd_dot_f32` vs
`simd_dot_f32_into` pattern). Does not block T5.5 completion.

## GOAT gates summary

| Gate | Claim | Verdict |
|------|-------|---------|
| G1 | directions non-degenerate (unit-norm + distinct) | ✅ PASS |
| G2 | freeze_attempt latency < 5µs/call | ✅ PASS (4582 ns) |
| G3 | cross-mode discrimination ≥ 80% accuracy | ✅ PASS (100%) |
| G4 | alloc-free steady state (≤2 substrate-inherited) | ✅ PASS (2 allocs) |

## What this validates + what it does NOT

**Validates:** the `SweTrajectoryFreezer` primitive works end-to-end on
the synthetic regime T5.3b already proved discriminative. The two-stage
pipeline (fit + freeze) composes cleanly, the data-derived directions
produce non-degenerate blends, the BLAKE3 envelope is tamper-evident, and
the latency is within budget.

**Does NOT validate:** T5.6 G5 (cross-snapshot/model discrimination on
real-model trajectories). T5.4 PARTIAL documented that depth trajectories
are insufficient — the discriminative signal likely lives in iterative
refinement trajectories, not depth trajectories of a single forward pass.
T5.6 requires either:
1. Porting `tf_loop` to Kimi-K3 (Option 1 from T5.4 — largest scope), OR
2. Extracting cross-token generation trajectories (Option 2 variant), OR
3. Accepting the depth trajectory result + relying on data-derived
   directions to amplify the 29% distinct rate into useful discrimination.

## Promotion

`swe_trajectory_freeze` stays **opt-in**. This is a research-validation
primitive (Proposal 011 Phase 5); promotion to default requires the T5.6
G5 gate to pass on real-model trajectories, which is currently open.

## Files

- `crates/katgpt-core/src/swe_trajectory_freeze.rs` — the primitive (714
  lines incl. tests).
- `crates/katgpt-core/src/lib.rs` — module declaration + re-exports.
- `crates/katgpt-core/Cargo.toml` — feature gate
  `swe_trajectory_freeze = ["latent_trajectory_geometry", "committed_field_blend"]`.
- `Cargo.toml` — root forwarder `swe_trajectory_freeze = ["katgpt-core/swe_trajectory_freeze"]`.
- `benches/bench_013_swe_trajectory_freezer_goat.rs` — this GOAT gate.
- `crates/katgpt-core/src/committed_field_blend.rs` — added `#[derive(Clone, Debug)]`
  to `CommittedFieldBlend` (needed for `FrozenAttempt`'s derives; the struct
  is all-POD so the derives are sound).
