# Issue 687: Orthogonal Factorization Primitives (orthonormalize + activity hinge + Parseval certificate)

> **Source:** Research 504 (arXiv:2608.20065 "Orthogonal JEPA") — Path 0 modelless extraction.
> **Repo:** katgpt-rs (open primitives, `katgpt-core`); consumers: riir-ai (affect/drive directions, k_selector reward axis, rollout certificate), riir-neuron-db (blend interference gate).
> **Feature:** `orthogonal_factorization = []` (opt-in; promotion requires GOAT + a consumer).
> **Filed:** 2026-08-24

## Why

The Orthogonal JEPA paper's *structure* is closed-form linear algebra with no gradient anywhere, and it fills documented gaps:

1. **Production direction sets are never orthogonalized.** The 5 HLA affect directions (`riir-ai/.../neuron_vessel_runtime.rs` `affect_directions`) and the 14 planner drive `dir_vec`s are extracted contrastively and may correlate; orthogonality is a TEST-ONLY assumption (`riir-games-civ/src/civ/emotion/tests.rs:803`). Cross-talk between "fear" and "despair" channels is the paper's monolithic-target failure mode wearing our vocabulary.
2. **No per-coordinate activity floor exists.** `effective_rank` is aggregate (a dead channel hides in a full-rank population — documented at `data_probe/gaussianity.rs:8-9`); gaussianity is distribution-shape. The hinge `max(0, γ−σ)` gives bounded, per-(factor,coordinate)-attributed variance-deficit — the missing third axis.
3. **No runtime Parseval invariant / exact truncation certificate.** With an orthonormal-complete basis: `‖z‖² = Σ_k‖B_k^Tz‖²` (structural check, O(d)) and dropped-energy truncation error is an identity, not an approximation.

## Tasks

- [ ] T1 `orthonormalize_into(vectors: &[[f32; D]], out: &mut [[f32; D]], defect: &mut f32)` — modified Gram–Schmidt, twice-reorthogonalized, fixed scratch, deterministic iteration order. Defect score = the paper's `L_orth` as diagnostic (`Σ‖B_k^TB_k−I‖²_F + Σ_{i<j}‖B_i^TB_j‖²_F`). Home: `katgpt-core/src/` (beside `cross_resolution.rs` / `spectral_pencil/`).
- [ ] T2 `factor_activity_hinge` — Welford accumulators over a population window, per-coordinate `max(0, γ−σ̂)`, mean over (k,j); γ schedule `≥ max(γ_min, c/√n)` (above the std-estimator's own sampling noise). Sibling to `data_probe/gaussianity.rs`.
- [ ] T3 `parseval_energy_check` — `|‖z‖² − Σ_k‖B_k^Tz‖²| ≤ ε·‖z‖²` + recompose identity; exact at f64 for Hadamard-style integer-core bases. Optional: `hadamard_factorize` for d=2^n (64-dim `style_weights`/HLA latent: 384 add/subs, zero multiplies, dyadic scale — cross-platform bit-identity).
- [ ] T4 Conditioning certificate — per-head `‖W_k‖₂ = √λ_max(W_k^T W_k)` via `spectral_pencil` (one constant matrix); composite rollout bound `Π_t max_k‖W_k‖₂`; commit κ/σmax as metadata at construction. (Orthonormal B ⇒ κ(B)=1 — the paper's conditioning caveat is void by construction; what remains certifiable is the heads.)
- [ ] T5 GOAT gates — G1 bit-identity across runs + platforms (fixed op order, no fast-math contraction); G2 µs-scale (GS < 5µs at d=64/K=14; hinge amortized O(N·d) at 1000-NPC scale); G4 zero steady-state alloc (fixed `[f32; 64]` scratch); G8 falsifiable negative controls: planted near-parallel pair ⇒ defect fires + GS decorrelates (|cos| < 1e-6); planted dead channel ⇒ hinge fires exactly on that coordinate.
- [ ] T6 Bench doc `.benchmarks/NNN_orthogonal_factorization_goat.md` + verdict table; promote/demote decision recorded.

## Non-goals / routing

- Learned/data-adaptive bases + dedicated trained heads → riir-train Plan 351 (this issue is the modelless half only).
- Consumer wiring (riir-ai affect orthogonalization A/B — a **gameplay owner call** per the CLR precedent; riir-neuron-db blend leakage gate / ShardCompactor merge criterion) = separate issues in those repos AFTER this GOAT passes.

## References

- Research 504 §Path 0 (signal-diffs) + §Distillation fusion items 1/2/3/4/7.
- Precedents: `subspace_phase_gate_goat.rs` (K orthogonal bases test fixture — the shape to productionize), Bench 494 dist-guard (advisory-audit integration template), Bench 691 q8kv (per-block scales → orthogonal blocks).
