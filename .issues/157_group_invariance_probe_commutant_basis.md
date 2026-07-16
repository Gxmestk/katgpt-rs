# Issue 157: Add `commutant_basis` helper to `group_invariance_probe.rs`

**Date:** 2026-07-16
**Research:** [`katgpt-rs/.research/444_Invariant_Manifold_Inductive_Reasoning_IMIR.md`](../.research/444_Invariant_Manifold_Inductive_Reasoning_IMIR.md)
**Source paper:** [arXiv:2607.11875](https://arxiv.org/abs/2607.11875) — Musat et al., *Invariant Learning Dynamics of Transformers in Inductive Reasoning Tasks*
**Target:** `katgpt-rs/crates/katgpt-core/src/group_invariance_probe.rs`
**Feature flag:** existing `group_invariance_probe` (no new flag)
**Status:** Open — Gain (deferred)

---

## Context

Research 444 (IMIR) distilled the paper's training-dynamics theory to riir-train and extracted a small modelless residue: the **commutant construction** for symmetry-invariant operator bases. The paper's two concrete instances:

- **Binary-association permutation group** on centered token embeddings ⟹ commutant = `span{I⁽ᵗ⁾, C}` (identity + association matrix)
- **k-step shift group** on sinusoidal position embeddings ⟹ commutant = `span{I⁽ᵖ⁾, M, M², …, Mᵏ⁻¹}`

The currently-shipped `group_invariance_probe` (Plan 355, LieFlow) uses **sample-then-score** Monte Carlo on the hypothesis group: sample random `g ∈ G`, score `σ(β·(1−d(q, g·q)))`, classify subgroup by score concentration. This works but is O(n_samples · d²) and gives an MC estimate of the invariant subspace.

The commutant construction is the **closed-form alternative**: given the group action, compute the invariant-operator basis directly in O(d³) one-shot. For finite/permutation groups this is exact; for continuous groups it reduces to a small linear-algebra problem.

## Tasks

- [ ] **T1** Add `commutant_basis<U: GroupAction>(group: &U, d: usize) -> Vec<Vec<Vec<f32>>>` to `group_invariance_probe.rs`. Doc comment cites IMIR (Musat et al. 2026, R444) and LieFlow (R355).
- [ ] **T2** Implement the two concrete closed-form cases from the paper:
  - `commutant_binary_association(associations: &[(usize, usize)], d: usize) -> [Matrix; 2]` returning `{I⁽ᵗ⁾, C}`
  - `commutant_shift(k: usize, d: usize) -> Vec<Matrix>` returning `{I⁽ᵖ⁾, M, M², …, Mᵏ⁻¹}` (circulant powers)
- [ ] **T3** Unit tests: verify the constructions commute with their respective group actions on synthetic data.
- [ ] **T4** Doc cross-ref: add a paragraph to the module doc explaining when to use `commutant_basis` (closed form, finite/permutation/shift groups) vs `discover_subgroup_into` (MC sampling, continuous groups, general case).

## Not in scope

- No benchmark, no GOAT gate — the helper is a more-principled alternative, not a provable gain.
- No promotion to default. The existing `group_invariance_probe` feature flag stays opt-in (Plan 355 deferred promotion pending a downstream consumer — this issue does not change that).
- No transformer-substrate application. Our runtime has no transformer weights to project; the helper is generic linear algebra for future consumers.
- No LatCal commitment of commutant coefficients (speculative fusion in R444 §2.4 — not actionable today).
- No `MerkleFrozenEnvelope` commutant constraint (speculative fusion in R444 §2.3d — no shard layout has non-trivial automorphism today).

## Why Gain, not GOAT

The construction is one-paragraph linear algebra (commutant of permutation group = `{I, C}` after centering; commutant of shift group = `{I, M, …, Mᵏ⁻¹}`). It is a refinement of an existing shipped primitive, not a new capability. Our latent states (HLA 8-dim, NeuronShard 64-dim) have either trivial symmetry (BLAKE3-pinned basis) or named-axis symmetry (HLA valence/arousal/etc.), where the commutant is small and mostly already implicit in the axis structure. No provable gain on our substrate.

## Close condition

All four tasks done; `cargo clippy -p katgpt-core --features group_invariance_probe` clean; unit tests pass.
