# Issue 194 — LatentConfounderAudit Primitive

**Filed:** 2026-07-28
**Priority:** P2 (modelless diagnostic primitive, no blocker)
**Related:** `.research/460_CD_LAM_Latent_Confounder_Audit_Diagnostics.md`, Research 374 (OTF-LAM), Research 418 (MAG — primary audit target), Research 425 (TILR), Research 309 (Latent Field Steering), Research 321 (Committed Personality Blend), Plan 342 (latent_trajectory_geometry — geometric diagnostic cousin), Plan 457 (causal_id — graph-level confounder cousin)
**Source:** [arXiv:2607.09185](https://arxiv.org/abs/2607.09185) — CD-LAM §III-B + Appendix A (diagnostic metrics)

## Context

CD-LAM (Research 460) defines three forward-pass diagnostics that audit a conditioning
latent for action-irrelevant confounders. The training recipe → riir-train, but the
**diagnostic framework** is modelless and ships in katgpt-rs as a generic latent-space
audit tool.

The three diagnostics catch a real failure mode in our stack: **runtime-mined direction
vectors** (MAG Plan 418, TILR Plan 425, Latent Field Steering Plan 309) can carry
confounders because the mining signal (activation geometry, trajectory invariance) is not
a purity constraint. A mined direction could project onto confounding activation patterns
that correlate with the target behavior but don't cause it. Today there is no check for
this — the direction is mined and deployed directly.

## The primitive

A `LatentConfounderAudit` struct + audit function, generic over an encoder `E: (&[f32],
&[f32]) -> &[f32]` (or a trait object). Three diagnostics:

### 1. Zero-transition response

```
R₀ = ‖E(x, x)‖₂ / (RMS(‖E(xᵢ, xᵢ′)‖₂) + ε)
```

Feed identical inputs through the encoder. The output norm should be ≈ 0 (a no-op input
produces a no-op latent). Normalized by the RMS of ordinary-transition norms so the metric
is scale-invariant.

**Applies to:** `evolve_hla` (identical obs → near-zero affect delta), `extract_functor`
(source == target → near-zero displacement), MAG direction (same activation → same
projection, trivially passes but the *delta under identical input* should be zero).

### 2. Shift-invariance response

```
R_shift = ‖E(x, T(x))‖₂ / (RMS(‖E(xᵢ, xᵢ′)‖₂) + ε)
```

Apply a nuisance transform `T` to one input (coordinate offset, tick drift, value shift).
The encoder output should be near-zero (the latent should be invariant to irrelevant
transforms).

**Applies to:** `evolve_hla` (irrelevant obs transform → no affect change),
`extract_functor` (translation of both source+target → same displacement), MAG direction
(irrelevant activation transform → same projection).

### 3. Shortcut leakage

```
L_shortcut = E[cos(zᵢ, zⱼ) | same-action, diff-context]
           − E[cos(zᵢ, zⱼ) | diff-action, same-context]
```

Same-decision/different-context pairs should have HIGHER cosine similarity than
different-decision/same-context pairs. If the gap is positive (or near zero), context is
dominating the latent structure — a confounder leak.

**Applies to:** HLA (same-emotion/diff-zone vs diff-emotion/same-zone), functor
(same-displacement/diff-scene vs diff-displacement/same-scene), MAG direction
(same-behavior/diff-context vs diff-behavior/same-context).

## API sketch

```rust
/// Three modelless forward-pass diagnostics auditing a conditioning latent
/// for action-irrelevant confounders.
///
/// Distilled from CD-LAM (arXiv:2607.09185) §III-B + Appendix A.
/// All fields are raw measurements (lower is better for R₀/R_shift;
/// more negative is better for L_shortcut). NOT probabilities — the
/// "Report the Floor" conformal rule does not apply.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LatentConfounderAudit {
    /// Zero-transition response. ≈ 0 = clean.
    pub zero_transition_response: f32,
    /// Shift-invariance response. ≈ 0 = clean.
    pub shift_invariance_response: f32,
    /// Shortcut leakage. < 0 = clean (action dominates context).
    pub shortcut_leakage: f32,
}

/// Audit an encoder function for confounder purity.
///
/// `encoder` is any function `(obs_a, obs_b) -> latent`. The caller supplies
/// test pairs organized by category. Zero-allocation in the hot path when
/// scratch buffers are pre-allocated.
pub fn audit_confounders<E>(
    encoder: &E,
    zero_transition_pairs: &[(&[f32], &[f32])],     // (x, x)
    shift_pairs: &[(&[f32], &[f32])],                // (x, T(x))
    ordinary_pairs: &[(&[f32], &[f32])],             // (x, x') for RMS normalization
    same_action_diff_context: &[((&[f32], &[f32]), (&[f32], &[f32]))],
    diff_action_same_context: &[((&[f32], &[f32]), (&[f32], &[f32]))],
    scratch: &mut AuditScratch,
) -> LatentConfounderAudit
where
    E: Fn(&[f32], &[f32]) -> &[f32];  // or returns a borrowed slice
```

## Target module

`katgpt-rs/crates/katgpt-core/src/latent_confounder_audit.rs` (new module).

Feature gate: `latent_confounder_audit` (opt-in, default-off).

## GOAT gate

| Gate | Criterion | Test |
|---|---|---|
| G1 | Correctness: diagnostics correctly identify confounders on a synthetic encoder with known confounders | Construct encoder `E(x, x') = A(x,x') + c·confounder(x)` with known `c`. Audit should detect non-zero R₀/R_shift/L_shortcut when `c > 0`, near-zero when `c = 0`. |
| G2 | Perf: sub-µs per check (O(d) norm + cosine) | Criterion bench on d=8 (HLA), d=32 (shard), d=64 (style_weights) |
| G3 | No-regression: new module, feature-gated, no existing code touched | `cargo check --workspace --all-features` clean |
| G4 | Alloc-free: pre-allocated `AuditScratch`, zero steady-state allocation | `CountingAllocator` audit across 100 audit calls |

## Consumers (force multiplier)

| Consumer | What it audits | When |
|---|---|---|
| MAG (Plan 418) | Mined direction vectors | Before deploying a mined direction — reject if confounders detected |
| TILR (Plan 425) | Refined trajectory-invariant directions | After refinement pass |
| Latent Field Steering (Plan 309) | Steering direction vectors | Before injecting a steering vector |
| Committed Personality Blend (321) | Archetype direction vectors | Before committing a blend |
| HLA evolve_hla | Per-NPC affect direction vectors | CI test: verify hand-constructed directions are clean |
| extract_functor | Functor displacement vectors | CI test: verify functor has translation invariance |

## Tasks

- [ ] **T1:** Implement `LatentConfounderAudit` struct + `audit_confounders` function in
  `crates/katgpt-core/src/latent_confounder_audit.rs`. Generic over encoder closure.
  Pre-allocated `AuditScratch` for zero-alloc hot path.
- [ ] **T2:** Feature gate `latent_confounder_audit` in `Cargo.toml` (opt-in).
- [ ] **T3:** GOAT G1 — synthetic encoder with known confounder coefficient `c`.
  Verify R₀/R_shift/L_shortcut detect `c > 0` and are near-zero for `c = 0`.
- [ ] **T4:** GOAT G2 — criterion bench on d=8/32/64. Target: sub-µs per check.
- [ ] **T5:** GOAT G3 — `cargo check --workspace --all-features` clean.
- [ ] **T6:** GOAT G4 — `CountingAllocator` audit, zero steady-state allocation.
- [ ] **T7:** If G1–G4 pass → promote `latent_confounder_audit` to default.

## Re-evaluation trigger

This is a diagnostic primitive, not a capability. It stays opt-in unless a consumer
(MAG/TILR/Steering) adopts it and benchmarks a quality gain (fewer misconfigured
directions deployed). Promotion to default requires a consumer showing the audit catches
real bugs that would otherwise ship.
