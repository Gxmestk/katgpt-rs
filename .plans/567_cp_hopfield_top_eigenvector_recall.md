# Plan 567: CP^(d-1) Symmetric-Space Hopfield — Top-Eigenvector Recall Primitive

**Date:** 2026-08-04
**Research:** [katgpt-rs/.research/466](../.research/466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md)
**Private guide:** [riir-neuron-db/.research/304](../../riir-neuron-db/.research/304_Symmetric_Space_Hopfield_Super_GOAT_Guide.md)
**Source paper:** Victor Galitski — "High-Capacity Generalized Hopfield Networks" — [alphaXiv 2607.hopfield-networks](https://www.alphaxiv.org/abs/2607.hopfield-networks) (2026-07-31)
**Target:** `katgpt-rs/crates/katgpt-core/src/cp_hopfield/` (new module) + Cargo feature `cp_hopfield`
**Status:** Active — Phase 1 (skeleton)

---

## Goal

Ship the **open primitive** for CP^(d-1) symmetric-space Hopfield associative memory recall — the modelless, BBP-protected top-eigenvector recall operator distilled from Galitski (2026). The primitive is generic Lie-algebraic + Rayleigh-quotient math with no game/chain/shard IP. It unblocks Plan 276's documented "attractor needs training" blocker (Fusion A — the load-bearing G5 gate) and force-multiplies `ItemEmbedIndex` + vibe KG retrieval (Fusion B — G6 gate).

**Feature flag:** `cp_hopfield` (opt-in). Promotion to default-on requires G1–G7 all PASS, with G5 (Plan 276 unblock) and G7 (BBP gap at finite N) as the load-bearing gates.

**Honest framing:** this is a Super-GOAT *candidate*. The Super-GOAT verdict is contingent on G5 passing. If G5 fails, the primitive still ships as a GOAT (capacity gains via Fusion B/G6) but the headline "modelless belief unblock" selling point is gone.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create module `crates/katgpt-core/src/cp_hopfield/` with `mod.rs` declaring the public API surface. Add `cp_hopfield` feature to `katgpt-core/Cargo.toml` default-off.
- [ ] **T1.2** Implement `CpHopfieldRecaller<D>` struct (research note §2.1). Generic over `const D: usize` (complex dimension = `d` in CP^(d-1)). Fields: `memories: Vec<[Complex<f32>; D]>`, `structure_constants: &'static [[[f32; D2]; D2]; D2]` where `D2 = D*D - 1`.
- [ ] **T1.3** Implement SU(d) structure constants for d=2 (Pauli, `f_{abc} = ε_{abc}`), d=3 (Gell-Mann, Eq. 43 of paper), d=4, d=8. Hardcoded lookup tables — O(1) init.
- [ ] **T1.4** Implement `mattis_overlap_excluding(neuron_idx, mu) -> f32` — the `O_μ^(i)` computation. O(N) per call.
- [ ] **T1.5** Implement `build_memory_kernel(neuron_idx) -> [[Complex<f32>; D]; D]` — the `K_i = Σ_μ O_μ^(i) |ξ^μ_i⟩⟨ξ^μ_i|` construction. O(P·D²) per call.
- [ ] **T1.6** Implement `hermitian_top_eigenvector(k: &[[Complex<f32>; D]; D]) -> [Complex<f32>; D]` — power iteration (5–10 iters suffice for d ≤ 8). For d=2 use closed-form (Pauli matrix analytic roots); for d=3 use closed-form (cubic characteristic polynomial). O(D³) per call.
- [ ] **T1.7** Implement `bloch_projection(state: &[Complex<f32>; D]) -> [f32; D2]` — convert qudit to generalized Bloch vector via `s_a = ⟨ξ|λ_a|ξ⟩`. O(D·D2) per call.
- [ ] **T1.8** Implement `recall_step(neuron_idx, current_bloch: &[f32; D2]) -> [f32; D2]` — the full top-eigenvector recall step (build K_i → top evec → Bloch projection).
- [ ] **T1.9** Add G1 unit test: store 1 Haar-random memory on CP² (d=3), corrupt it 40%, recall → assert `m̄ ≥ 0.9` after 1 sweep.
- [ ] **T1.10** Add G1 unit test: store 10 Haar-random memories on CP² at α=0.1 (< α_c=0.62), corrupt memory 0, recall → assert `m̄_0 ≥ 0.9` after 1 sweep.
- [ ] **T1.11** Commit Phase 1 skeleton. Tag for the G5 PoC (Phase 5) to consume.

---

## Phase 2 — Manifold Constraint Enforcement

### Tasks

- [ ] **T2.1** Implement `project_to_manifold(bloch: &mut [f32; D2])` — enforce the non-linear CP^(d-1) constraint `d_{abc} s_b s_c = (2/3) s_a` via projected gradient (alternate normalization + constraint projection until convergence). O(D²) per call.
- [ ] **T2.2** Implement the symmetric `d_{abc}` tensor for d=3 (paper §VIII.C gives the explicit non-zero components). For d=2 all `d_{abc}=0` (no constraint beyond norm). For d=4, d=8 — derive from generalized Gell-Mann anticommutators.
- [ ] **T2.3** Add G4 unit test: `project_to_manifold` converges in ≤ 5 iterations for d=3; produces Bloch vector satisfying the constraint to `|d_{abc} s_b s_c − (2/3) s_a| < 1e-5` for all a.
- [ ] **T2.4** Add G4 unit test: `project_to_manifold` is sub-μs for d=3 (D=8) at criterion bench.

---

## Phase 3 — Generalized LLG Flow (Physical Recall)

### Tasks

- [ ] **T3.1** Implement `lie_bracket(s: &[f32; D2], b: &[f32; D2], f: &StructureConstants) -> [f32; D2]` — the `[s ×_f B]_c = f_{cab} s_a B_b` computation. O(D2²) per call.
- [ ] **T3.2** Implement `mean_field(neuron_idx, states: &[[f32; D2]; N]) -> [f32; D2]` — the `B_i = Σ_{j≠i} J_{ij} s_j = Σ_μ ξ^μ_i O_μ^(i)` computation. O(N·D2) per call.
- [ ] **T3.3** Implement `llg_flow_step(s: &mut [f32; D2], b: &[f32; D2], damping: f32, dt: f32)` — the generalized Landau-Lifshitz-Gilbert step: `ṡ = s ×_f B − λ [s ×_f [s ×_f B]]`. O(D2²) per call. Calls `project_to_manifold` after the step.
- [ ] **T3.4** Implement `llg_recall(recaller: &CpHopfieldRecaller, initial: &mut [f32; D2], damping: f32, dt: f32, max_steps: usize) -> RecallResult` — runs the LLG flow to fixed point, returns final state + energy trajectory + convergence step count.
- [ ] **T3.5** Add G1 unit test: LLG recall on CP² with 1 corrupted memory converges to `m̄ ≥ 0.99` within 10 damping times (paper Fig 9 shows ~3 damping times at λ=1).
- [ ] **T3.6** Add G1 unit test: LLG recall energy trajectory is monotonically non-increasing (`Ė = −λ Σ |s_i ×_f B_i|² ≤ 0`).
- [ ] **T3.7** Add G4 unit test: one LLG step is sub-μs for d=3 (D2=8) at criterion bench.

---

## Phase 4 — Capacity Measurement (G2)

### Tasks

- [ ] **T4.1** Implement `measure_capacity(d: usize, n: usize, alpha_range: &[f32], realizations: usize) -> CapacityCurve` — for each α in `alpha_range`, generate P=α·N Haar-random memories, corrupt a random target, recall, measure `m̄_0`. Average over `realizations`. Return α_c (where `m̄_0` crosses threshold 0.5).
- [ ] **T4.2** Add G2 benchmark: measure α_c for d=2, 3, 4 at N=64, 256, 1024. Compare to paper's asymptotic α_c (0.05, 0.62, 2.41). Document finite-N corrections.
- [ ] **T4.3** Add G2 benchmark: measure α_c at N=8 (our belief dim) for d=3. This is the critical finite-N test for Fusion A — if α_c(N=8, d=3) is much lower than the asymptotic 0.62, the Plan 276 unblock is at risk.
- [ ] **T4.4** Add G2 benchmark: measure α_c on CORRELATED memories (not Haar-random). Generate memories as `ξ^μ = cos(θ_μ) · v_base + sin(θ_μ) · v_orth` with varying `θ_μ` spread. Document how correlation reduces α_c.

---

## Phase 5 — Plan 276 G5 PoC (LOAD-BEARING)

### Tasks

- [ ] **T5.1** In `riir-ai/crates/riir-poc/benches/cp_hopfield_plan276_unblock.rs`, set up the three-competitor comparison:
  - **Baseline A:** random-init `AttractorKernel` (Plan 276 demoted family)
  - **Baseline B:** `LeakyIntegrator` (Plan 276 winner, flip count = 1)
  - **Candidate:** CP^(d=3) top-eigenvector recaller, loaded with `NeuronShard::style_weights[64]` as memories (freeze/thaw Path 1 — no training)
- [ ] **T5.2** Run all three on the Plan 276 G2.1 belief-flip benchmark (the ambiguous-window noise test). Measure flip count over 1000 ticks.
- [ ] **T5.3** **G5 PASS criterion:** Candidate flip count ≤ 10× LeakyIntegrator's flip count (≤ 10 flips). If candidate ≤ 10, Fusion A is validated → Super-GOAT confirmed. If candidate > 100, Fusion A is REFUTED → verdict drops to GOAT, document honestly in research note 466 §3.
- [ ] **T5.4** **G7 measurement:** at the operating point (N=64, d=3 or d=8), measure the eigenvalue gap `(λ_max − λ_2) / λ_max` of `K_i` at loads α = α_c/4, α_c/2, 3α_c/4. Document whether the gap is > 0.1 (BBP protection holds) or ≈ 0 (protection fails at finite N).
- [ ] **T5.5** Write the PoC addendum to research note 466 §"PoC Addendum": report raw numbers, state which claims (architectural / latency / quality) were confirmed vs refuted. Per §3.6, do NOT silently revise the verdict — record the refutation honestly and let the verdict stand on the confirmed axes.

---

## Phase 6 — Fusion B / G6 (KG Capacity, in riir-neuron-db follow-up)

### Tasks (deferred to a riir-neuron-db plan after Phase 5)

- [ ] **T6.1** Implement `NeuronShard` CP^(d-1) view — re-parameterize `style_weights[64]` as CP⁷ (d=8) Bloch vectors. Enforce non-linear constraint on read/write.
- [ ] **T6.2** Add `ItemEmbedIndex::query_cp` — top-eigenvector recall path alongside the existing cosine ANN path. Feature-gated.
- [ ] **T6.3** Add `vibe.rs` KG triple emission via top-eigenvector — feature-gated.
- [ ] **T6.4** G6 benchmark: per-shard KG triple capacity, cosine ANN vs CP^(d-1) recall, on correlated triples. PASS criterion: CP^(d-1) maintains precision@1 ≥ 0.9 at ≥ 3× the triple count where cosine ANN drops below 0.9.
- [ ] **T6.5** `MerkleFrozenEnvelope` commitment of CP^(d-1) memory sets + K_i kernels.

---

## Phase 7 — GOAT Gate + Promotion Decision

### Tasks

- [ ] **T7.1** Run G1–G7 full gate. Document results in `.benchmarks/567_*.md`.
- [ ] **T7.2** **Promotion decision:**
  - If G5 PASS AND G6 PASS AND G7 PASS → promote `cp_hopfield` to default-on. File riir-ai follow-up plan for Plan 276 AttractorKernel re-promotion under CP^(d-1) parameterization.
  - If G5 FAIL but G6 PASS → keep `cp_hopfield` opt-in. Verdict drops to GOAT. Document in research note 466 §3 that Fusion A was refuted; Fusion B (shard capacity) is the surviving value.
  - If G5 FAIL AND G6 FAIL → demote to experimental. Verdict drops to Gain. Document honestly.
- [ ] **T7.3** Update per-stack promote/demote ledger in research note 466 §3.
- [ ] **T7.4** Update `.docs/09_feature_catalog/` if promoted to default-on; update negative_results.md if demoted.
- [ ] **T7.5** Update Plan 276 benchmark note with the G5 PoC results (whether Fusion A unblocked it or not).

---

## Risk Register

| Risk | Mitigation |
|---|---|
| **G5 FAILS** (Plan 276 unblock refuted) | Verdict honestly drops to GOAT. Fusion B (shard capacity) may still hold. Document in §3.6 PoC addendum. Do NOT silently revise the Super-GOAT claim. |
| **Finite-N α_c much lower than asymptotic** | G2 measures this at N=8, N=64. If α_c(N=8, d=3) << 0.62, the capacity claim weakens. May restrict the primitive to d ≥ 4 where finite-N effects are smaller. |
| **Non-linear constraint projection too costly** | G4 measures `project_to_manifold` cost. If > 1μs for d=3, optimize (closed-form projection for CP²). If still too costly at d=8, restrict to d ≤ 4. |
| **Correlated memories behave very differently** | G2 + G1 test on correlated distributions. If α_c drops by > 2× on correlated vs Haar-random, document as a real-world capacity reduction. |
| **Shadow phenomenon causes personality bleed** | Characterize when shadow is desirable (KG retrieval — richer context) vs undesirable (personality recall — bleed). Add a `shadow_suppression` knob if needed. |
| **RSB corrections at d=8** | Our primary use case is d=3, d=4 (small). If we push to d=8 and α_c is much lower than the replica-symmetric prediction, RSB is the likely cause. Document; restrict to d ≤ 4 if needed. |
| **Integration cost (Phase 6)** | Re-parameterizing `style_weights[64]` is invasive. P1 shard bridge is a non-trivial migration. Time-box; if it exceeds budget, ship Phase 1–5 only (open primitive + PoC) and defer Phase 6. |

---

## References

- [Research 466](../.research/466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md) — the open primitive note
- [riir-neuron-db/.research/304](../../riir-neuron-db/.research/304_Symmetric_Space_Hopfield_Super_GOAT_Guide.md) — the private Super-GOAT guide
- [Plan 276 benchmark](../.benchmarks/276_micro_belief_goat.md) — the documented blocker (G5 load-bearing gate)
- [Research 455](../.research/455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md) — Hebbian Kernel Memory (construction-side cousin)
- [Research 317](../.research/317_Reasoning_As_Attractor_Dynamics_Gibbs_Retrieval.md) — Reasoning as Attractor (Gibbs retrieval cousin)
- Galitski 2026: [alphaXiv 2607.hopfield-networks](https://www.alphaxiv.org/abs/2607.hopfield-networks)
