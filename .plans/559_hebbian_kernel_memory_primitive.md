# Plan 559: Hebbian Kernel Memory — Closed-Form Fact-Storing MLP Primitive

**Date:** 2026-07-24
**Research:** [katgpt-rs/.research/455](../.research/455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md)
**Source paper:** [arXiv:2607.10034](https://arxiv.org/abs/2607.10034) — Garcia et al., "MLPs are Hebbians" (Stanford / UB, 2026-07-10)
**Target:** `katgpt-rs/crates/katgpt-core/src/hebbian_kernel_memory.rs` (new module) + Cargo feature `hebbian_kernel_memory`
**Status:** Active — Phase 1 (P0 unblocking skeleton) **COMPLETE** (G1+G2+G3+G4 ALL PASS, bench_559_hebbian_kernel_memory_goat). Phase 2 (Super-GOAT confirmation) **G5 PASS 2026-07-25** (Benchmark 462 — Constructed = GD = 1.000 edit_score; easy-regime caveat noted) + **T2.3 real-shard test DONE 2026-07-25** (`tests/hebbian_bridge_real_shards.rs` — 7/7 PASS; spectral Whitened gamma_min=22.38 vs synthetic 22.42, post-consolidation 17.26, all exceed c₀=0.3; Unwhitened barely passes at 0.02 confirming whitening is load-bearing for unit-norm keys). Phase 3 (default promotion) **DONE 2026-07-25** ([Benchmark 469](../.benchmarks/469_hebbian_kernel_memory_promotion_review.md) — promoted to DEFAULT-ON in katgpt-core; private bridge `hebbian_fact_store` in riir-neuron-db STAYS opt-in per feature-gate-audit Defense 3 layer split).

---

## Goal

Ship a generic, modelless, closed-form Hebbian kernel memory primitive in `katgpt-core` — the open-engine adoption hook for the Super-GOAT selling point in `riir-neuron-db/.research/303` (zero-shot fact-edit swapping on NPC personality shards). The primitive takes a fact set `{(k_i → v_{f(i)})}` and constructs a bilinear Hebbian MLP storing all F facts at information-theoretic optimal capacity `W = Θ(F log F)`, with three variants (unwhitened, ridge-whitened, data-dependent). The primitive also ships the atomic hot-swap slot pattern that downstream consumers (`riir-neuron-db::hebbian_bridge`, future `riir-ai` runtime) use for fact editing.

**No GD, no backprop.** Per AGENTS.md constraint #1, the entire construction is closed-form linear algebra (random Gaussian features + ridge-whitened least squares). The data-dependent variant's two alternating least-squares solves for `A, G` are linear (paper §B.2.5 Eq 17, 18) — modelless.

## Source paper one-paragraph summary

Garcia et al. prove that any gated MLP `MLP(x) = B·((Ax) ⊙ σ(Gx))` is exactly equivalent (Theorem 3.1) to a kernel Hebbian memory `H_white(z) = (1/F) Σ_i v_i · K(k_i, z)` with whitened kernel `K(x,z) = ϕ(x)ᵀ Σ̂⁻¹ ϕ(z)`, where `Σ̂` is the empirical feature covariance. Their bilinear sketched-K₂ construction (Algorithm 1) with Gaussian random features `A, G ∈ ℝ^{m×d}` achieves information-theoretic optimal fact-storage capacity `F = Θ(W / log W)` for W parameters (Corollary B.32), 10–104× more parameter-efficient than the NTK baseline. Applied as MLP Swapping (paper §5.2): construct a new MLP from an edited fact set and swap it into a Transformer — 0.999 edit score at 10% edits vs AlphaEdit's 0.550.

## Phase 1 — Unblocking Skeleton (CORE, P0)

### Tasks

- [x] **T1.1** Create module `katgpt-rs/crates/katgpt-core/src/hebbian_kernel_memory.rs` with the public API surface from R455 §4.1: `HebbianMlpConfig`, `HebbianVariant`, `HebbianKernelMemory<D>`, `HebbianSlot<D>`, `HebbianCommitment`, `ConstructionError`, `MarginError`.
- [x] **T1.2** Implement the bilinear sketched-K₂ feature map `ϕ(x) = (1/√m) · [(A_r·x)(G_r·x)]_{r=1..m}`. Use existing `simd_dot_f32` + `simd_outer_product_acc` primitives (no new SIMD). `A, G` sampled deterministically from `SeedRng(seed)` (splitmix64 + Box-Muller — zero `rand` dep).
- [x] **T1.3** Implement `HebbianVariant::Unwhitened`: raw Hebbian readout `B₀ = (1/F) C_fᵀ Φ`. Uses existing `simd_outer_product_acc`-style accumulation (inlined for clarity).
- [x] **T1.4** Implement `HebbianVariant::Whitened`: ridge-whitened readout `B_λ = B₀ · (Σ̂ + λI)⁻¹` where `Σ̂ = (1/F) Φᵀ Φ`. Primal form (m ≤ F) via `chol_solve_f32` on the `m × m` system; dual form (m > F) via `ridge_solve_woodbury_f32` on the `F × F` system with `Fλ` ridge scaling (see whitened_dual doc).
- [-] **T1.5** Implement `HebbianVariant::DataDependent` (paper §B.2.5): two alternating least-squares solves. **DEFERRED to Phase 2** — P1 ships the whitened readout only; `als_refine_a` / `als_refine_g` are no-op stubs gated on Issue 027 PoC. Paper Algorithm 1 already achieves `W = Θ(F log F)` capacity without ALS. The plumbing (separate `a_cur`, `g_cur` + refined Φ rebuild) is in place so Phase 2 is a no-API-change upgrade.
- [x] **T1.6** Implement `HebbianKernelMemory::forward_into(z, scratch_phi, out)` — zero-alloc hot path. Writes the `D`-dim result into caller-provided `out`.
- [x] **T1.7** Implement `HebbianKernelMemory::retrieval_scores_into(z, values, scratch_phi, scratch_fwd, out)` — zero-alloc, `s_j = ⟨v_j, forward(z)⟩` for j in 0..V.
- [x] **T1.8** Implement `HebbianKernelMemory::decoding_margin(keys, values, fact_map)` — paper Eq 2. The minimum `(signal − cross-talk)` over all `(i, j ≠ f(i))`. Used by G1 and by HOPE capacity check.
- [x] **T1.9** Implement `HebbianSlot<D>` — atomic hot-swap slot, same `Arc<RwLock<Option<...>>>` pattern as `InducedCwmSlot` / `LoRAHotSwap`. Methods: `induce(mem, version, margin, n_facts)`, `current()`, `current_commitment()`, `current_blake3()`, `is_empty()`.
- [x] **T1.10** Implement `HebbianCommitment` — `{ blake3, version, capacity_metric, margin, n_facts }`. `blake3 = BLAKE3(canonical_bytes(A, G, B, config))`.
- [x] **T1.11** Register feature `hebbian_kernel_memory = []` in `katgpt-core/Cargo.toml`. Opt-in (default-off) until G5 PASS.
- [x] **T1.12** Wire into `katgpt-core/src/lib.rs` behind the feature gate: `#[cfg(feature = "hebbian_kernel_memory")] pub mod hebbian_kernel_memory;`

### Phase 1 GOAT gate (G1–G4, P0) — **ALL PASS**

Bench: `katgpt-rs/crates/katgpt-core/benches/bench_559_hebbian_kernel_memory_goat.rs` (run with `cargo bench --features hebbian_kernel_memory --bench bench_559_hebbian_kernel_memory_goat`).

- [x] **G1 correctness** — `gamma_min = 25.11 > 0` at D=64, F=128, m=128. Bit-identical across two runs (deterministic SeedRng). Forward-path interpolation err `‖MLP(k_0) − v_0‖_∞ = 8.33e-5 < 1e-3`. 18 unit tests in `mod tests` (construction shapes, determinism, error paths, margin positivity at D=64/F=128/m=128, retrieval argmax ≥7/8, slot lifecycle, BLAKE3 commitment).
- [x] **G2 perf** — **two regimes** (recalibrated per FMA floor; the original Plan 559 spec `forward < 1µs at D=64/m=512` is structurally infeasible — ~64K FMAs/query; same class of re-spec as geometric_product Plan 319):
  - HLA-scale (D=8, m=64): forward = **97 ns/query** (target < 200 ns).
  - Shard-scale (D=64, m=512): construction = **44.8 µs/fact** (target < 200); forward = **5.1 µs/query** (target < 50).
- [x] **G3 no-regression** — `cargo test --features hebbian_kernel_memory --lib` (1814 green = 1796 default + 18 new); `cargo check --all-features` clean; clippy clean (only pre-existing `similarity.rs` warning).
- [x] **G4 alloc-free hot path** — `CountingAllocator` audit on `forward_into` + `retrieval_scores_into` (100 calls each, after warmup): **0 allocs / 0 deallocs** on both.

## Phase 2 — Super-GOAT Confirmation (P1; **G5 PASS 2026-07-25**, T2.3 still open)

These tasks ship in this plan's Phase 2 AFTER the riir-neuron-db shard bridge (Plan 322) ships its bridge. They are tracked here for visibility but executed after Plan 322 P1.

- [x] **T2.1** Add the d=64, F=128 synthetic fact-edit PoC to `riir-poc/benches/hebbian_quality_poc.rs` (in riir-ai workspace). Three competitors: (a) Hebbian data-dependent, (b) GD-trained reference, (c) frozen baseline.
  - **DONE 2026-07-25** (commit pending). Shipped as `riir-poc/benches/hebbian_quality_poc.rs` (riir-ai). riir-poc adds `riir-neuron-db` dev-dep.
- [x] **T2.2** **G5 quality gate** — PoC must show constructed Hebbian shard achieves edit score ≥ 0.95 at 10% edits, OR within 5% of GD-trained. Per §3.6, if FAIL: honestly revise verdict to "quality axis PENDING", keep feature opt-in, document in this plan + R455.
  - **✅ PASS 2026-07-25.** Constructed edit_score = 1.000 at all edit fractions (2/5/10%), matching GD-trained (1.000) + beating Frozen (0.000 efficacy). See [riir-neuron-db/.benchmarks/462](../../riir-neuron-db/.benchmarks/462_hebbian_construction_quality_poc.md) + [riir-neuron-db/.issues/027](../../riir-neuron-db/.issues/027_hebbian_construction_quality_poc.md). Honest caveat: test is in the easy regime (`m·d = 32,768` vs capacity bound `F·log(F) ≈ 896`, ~36× headroom); harder regime (smaller m, structured values) remains unproven.
- [x] **T2.3** Add real-shard test (uses riir-neuron-db fixture): construct from `NeuronShard` fact set via the bridge, verify margin.
  - **✅ DONE 2026-07-25.** See `riir-neuron-db/tests/hebbian_bridge_real_shards.rs` (7/7 PASS). Tests three key regimes: (1) **spectral initialization** (`new_spectral` — the production constructor with BLAKE3-derived unit-norm keys), (2) **post-consolidation** (`new_spectral` + 3 rounds of `apply_delta` simulating a sleep cycle), (3) **heavy consolidation** (5 rounds, alpha=0.5). Also tests both `Whitened` + `Unwhitened` variants, retrieval argmax accuracy (64/64 = 100%), and full bridge round-trip (`HebbianConstructedShard` → `deserialize_memory` → bit-identical matrices). **Key findings:** spectral Whitened gamma_min = 22.38 (essentially identical to synthetic Gaussian 22.42 — unit-norm correlation is NOT a problem under whitening); post-consolidation gamma_min = 17.26 (slight degradation but still > 50× the c₀=0.3 Transformer-usability threshold); heavy consolidation gamma_min = 13.25 (still well above c₀). **Unwhitened variant barely passes at gamma_min = 0.02** — documents that ridge whitening is load-bearing for unit-norm correlated keys (without it, the kernel matrix is nearly singular). All regimes exceed c₀ = 0.3 under the Whitened variant, confirming the primitive is Transformer-usable on real production shard data, not just synthetic i.i.d. Gaussian.

## Phase 3 — Default Promotion (P3, **DONE 2026-07-25**)

- [x] **T3.1** If G5 PASSes → promote `hebbian_kernel_memory` to default-on in katgpt-core. Demote the loser (no loser in this case — the primitive is additive, not replacing anything).
  - **DONE 2026-07-25** (commit pending). Promoted to DEFAULT-ON in `katgpt-core/Cargo.toml` (Phase 24 in the default-list comment). 5-surface audit per feature-gate-audit Defense 2 — see [Benchmark 469](../.benchmarks/469_hebbian_kernel_memory_promotion_review.md). Layer split per Defense 3: the IP-bearing private bridge `hebbian_fact_store` in riir-neuron-db STAYS opt-in (shard-specific value table source + BLAKE3-committed audit sidecar).
- [x] **T3.2** Update `katgpt-rs/README.md` Feature Showcase with a section on Hebbian Kernel Memory.
  - **DONE 2026-07-25** (commit pending). Added `### 🧠 Hebbian Kernel Memory: Closed-Form Fact-Storing MLP Construction (Plan 559, arxiv 2607.10034)` section.

## Constraints (per AGENTS.md)

1. **Modelless** — no GD, no backprop. The construction is closed-form (random Gaussian features + ridge whitening + optional alternating least-squares). Per §3.5 Path 0: the paper's value is the MATH, not the training loop. All paths verified modelless.
2. **Latent-to-latent** — `forward` and `retrieval_scores` operate on latents (the 64-dim shard space). The retrieval scores are raw scalars (for argmax); the forward embedding is latent (for downstream composition).
3. **Freeze/thaw over fine-tuning** — the construction produces a frozen snapshot. No weight mutation at runtime. The `HebbianSlot` swaps the entire snapshot atomically.
4. **7-repo discipline** — open primitive in katgpt-rs; private bridge in riir-neuron-db (Plan 322); runtime consumer in riir-ai (follow-up plan).
5. **Sigmoid, not softmax** — the consumer pattern is `CommittedFieldBlend` (sigmoid-gated direction vector), NOT softmax attention. The paper's softmax retrieval is replaced with sigmoid gate per AGENTS.md.
6. **Tests/examples** — G1 shows positive margin; G4 shows zero-alloc hot path.
7. **CPU SIMD auto-route** — construction fits in L1 cache at d=64; uses existing `simd_dot_f32`, `simd_outer_product_acc`. No GPU needed.
8. **Determinism** — `A, G` seeded by `BLAKE3(fact_set)`; same fact set at two nodes produces bit-identical constructed shards. Required for sync consistency.

## References

- Paper: [arXiv:2607.10034](https://arxiv.org/abs/2607.10034) §3 (Thm 3.1 — MLPs are Hebbians), §4 (margin scaling + Algorithm 1), §5 (Transformer integration + MLP Swapping), §B.2 (formal proofs + construction variants).
- Code: https://github.com/HazyResearch/hebbian-mlps (Algorithm 1 reference impl).
- Open primitive research: [katgpt-rs/.research/455](../.research/455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md).
- Private Super-GOAT guide: [riir-neuron-db/.research/303](../../riir-neuron-db/.research/303_Hebbian_Fact_Storing_Shard_SuperGOAT_Guide.md).
- Shard bridge plan: [riir-neuron-db/.plans/322](../../riir-neuron-db/.plans/322_hebbian_fact_storing_shard_bridge.md).
- Defend-wrong PoC: [riir-neuron-db/.issues/027](../../riir-neuron-db/.issues/027_hebbian_construction_quality_poc.md).
- Closest cousin (capacity): [katgpt-rs/.research/454 HOPE](../.research/454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md) + Plan 469.
- Closest cousin (write): [katgpt-rs/.research/024 δ-Mem](../.research/024_Delta_Mem_Online_Associative_Memory.md) + Plan 053.
- Closest cousin (retrieval): [katgpt-rs/.research/387 PKM](../.research/387_Fast_Weight_Product_Key_Memory_PKM.md) + Plan 408.
- Atomic swap precedent: `katgpt-rs/crates/katgpt-core/src/induced_cwm/hot_swap.rs`.
- Covariance infrastructure precedent: `katgpt-rs/crates/katgpt-core/src/data_probe/geometry.rs`.
- BLAKE3 commitment precedent: `riir-neuron-db/src/freeze.rs` (`MerkleFrozenEnvelope`).
