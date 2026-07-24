# Plan 559: Hebbian Kernel Memory — Closed-Form Fact-Storing MLP Primitive

**Date:** 2026-07-24
**Research:** [katgpt-rs/.research/455](../.research/455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md)
**Source paper:** [arXiv:2607.10034](https://arxiv.org/abs/2607.10034) — Garcia et al., "MLPs are Hebbians" (Stanford / UB, 2026-07-10)
**Target:** `katgpt-rs/crates/katgpt-core/src/hebbian_kernel_memory.rs` (new module) + Cargo feature `hebbian_kernel_memory`
**Status:** Active — Phase 1 (P0 unblocking skeleton)

---

## Goal

Ship a generic, modelless, closed-form Hebbian kernel memory primitive in `katgpt-core` — the open-engine adoption hook for the Super-GOAT selling point in `riir-neuron-db/.research/303` (zero-shot fact-edit swapping on NPC personality shards). The primitive takes a fact set `{(k_i → v_{f(i)})}` and constructs a bilinear Hebbian MLP storing all F facts at information-theoretic optimal capacity `W = Θ(F log F)`, with three variants (unwhitened, ridge-whitened, data-dependent). The primitive also ships the atomic hot-swap slot pattern that downstream consumers (`riir-neuron-db::hebbian_bridge`, future `riir-ai` runtime) use for fact editing.

**No GD, no backprop.** Per AGENTS.md constraint #1, the entire construction is closed-form linear algebra (random Gaussian features + ridge-whitened least squares). The data-dependent variant's two alternating least-squares solves for `A, G` are linear (paper §B.2.5 Eq 17, 18) — modelless.

## Source paper one-paragraph summary

Garcia et al. prove that any gated MLP `MLP(x) = B·((Ax) ⊙ σ(Gx))` is exactly equivalent (Theorem 3.1) to a kernel Hebbian memory `H_white(z) = (1/F) Σ_i v_i · K(k_i, z)` with whitened kernel `K(x,z) = ϕ(x)ᵀ Σ̂⁻¹ ϕ(z)`, where `Σ̂` is the empirical feature covariance. Their bilinear sketched-K₂ construction (Algorithm 1) with Gaussian random features `A, G ∈ ℝ^{m×d}` achieves information-theoretic optimal fact-storage capacity `F = Θ(W / log W)` for W parameters (Corollary B.32), 10–104× more parameter-efficient than the NTK baseline. Applied as MLP Swapping (paper §5.2): construct a new MLP from an edited fact set and swap it into a Transformer — 0.999 edit score at 10% edits vs AlphaEdit's 0.550.

## Phase 1 — Unblocking Skeleton (CORE, P0)

### Tasks

- [ ] **T1.1** Create module `katgpt-rs/crates/katgpt-core/src/hebbian_kernel_memory.rs` with the public API surface from R455 §4.1: `HebbianMlpConfig`, `HebbianVariant`, `HebbianKernelMemory<D>`, `HebbianSlot<D>`, `HebbianCommitment`, `ConstructionError`, `MarginError`.
- [ ] **T1.2** Implement the bilinear sketched-K₂ feature map `ϕ(x) = (1/√m) · [(A_r·x)(G_r·x)]_{r=1..m}`. Use existing `simd_dot_f32` + `simd_outer_product_acc` primitives (no new SIMD). `A, G` sampled deterministically from `BLAKE3(fact_set)` seed (paper requires Gaussian i.i.d. — we use a Gaussian-from-BLAKE3 generator).
- [ ] **T1.3** Implement `HebbianVariant::Unwhitened`: raw Hebbian readout `B₀ = (1/F) C_fᵀ Φ`. Uses existing `simd_outer_product_acc`.
- [ ] **T1.4** Implement `HebbianVariant::Whitened`: ridge-whitened readout `B_λ = (1/F) C_fᵀ Φ · (Σ̂ + λI)⁻¹` where `Σ̂ = (1/F) Φᵀ Φ`. Choose primal (m ≤ F) or dual (m > F) form based on shape. Use existing `data_probe::geometry::covariance` infrastructure for `Σ̂`. For the inverse, use Gauss-Jordan elimination on the small `m × m` matrix (m is typically ≤ 1024).
- [ ] **T1.5** Implement `HebbianVariant::DataDependent` (paper §B.2.5): two alternating least-squares solves — `G₁ = argmin ‖C_f − Φ(A₀, G) B₀ᵀ‖²_F`, then `A₁ = argmin ‖C_f − Φ(A, G₁) B₀ᵀ‖²_F`. Both are linear (paper Eq 17, 18) — use the same Gauss-Jordan solver.
- [ ] **T1.6** Implement `HebbianKernelMemory::forward(z)` — zero-alloc hot path. Returns `B · ϕ(z)` in a pre-allocated `[f32; D]` output.
- [ ] **T1.7** Implement `HebbianKernelMemory::retrieval_scores(z, values, out)` — zero-alloc, `s_j = ⟨v_j, forward(z)⟩` for j in 0..V.
- [ ] **T1.8** Implement `HebbianKernelMemory::decoding_margin(keys, values, fact_map)` — paper Eq 2. The minimum `(signal − cross-talk)` over all `(i, j ≠ f(i))`. Used by G1 and by HOPE capacity check.
- [ ] **T1.9** Implement `HebbianSlot<D>` — atomic hot-swap slot, same `Arc<RwLock<Option<...>>>` pattern as `InducedCwmSlot` / `LoRAHotSwap`. Methods: `induce(mem)`, `current()`.
- [ ] **T1.10** Implement `HebbianCommitment` — `{ blake3, version, capacity_metric, margin, n_facts }`. `blake3 = BLAKE3(canonical_bytes(A, G, B))`.
- [ ] **T1.11** Register feature `hebbian_kernel_memory = []` in `katgpt-core/Cargo.toml` and `katgpt-rs/Cargo.toml`. Opt-in (default-off) until G5 PASS.
- [ ] **T1.12** Wire into `katgpt-core/src/lib.rs` behind the feature gate: `#[cfg(feature = "hebbian_kernel_memory")] pub mod hebbian_kernel_memory;`

### Phase 1 GOAT gate (G1–G4, P0)

- [ ] **G1 correctness** — `tests/hebbian_g1.rs`: construct `HebbianKernelMemory<64>` from F=128 isotropic Gaussian fact set at m = ceil(F·log F / d) = 10. Assert `decoding_margin > 0` for all F facts. Assert bit-identical across two runs with same BLAKE3 seed.
- [ ] **G2 perf** — `benches/hebbian_g2.rs`: construction time < 50µs/fact at d=64, m=512; forward path < 1µs/query. Use `criterion`.
- [ ] **G3 no-regression** — `cargo test --all-features` passes. No clippy warnings on the new module.
- [ ] **G4 alloc-free hot path** — formal `CountingAllocator` audit on `forward` and `retrieval_scores`: 0 steady-state allocations.

## Phase 2 — Super-GOAT Confirmation (P1, DEFERRED to Plan 559 Phase 2 + Issue 027)

These tasks ship in this plan's Phase 2 AFTER the riir-neuron-db shard bridge (Plan 322) ships its bridge. They are tracked here for visibility but executed after Plan 322 P1.

- [ ] **T2.1** Add the d=64, F=128 synthetic fact-edit PoC to `riir-poc/benches/hebbian_quality_poc.rs` (in riir-ai workspace). Three competitors: (a) Hebbian data-dependent, (b) GD-trained reference, (c) frozen baseline.
- [ ] **T2.2** **G5 quality gate** — PoC must show constructed Hebbian shard achieves edit score ≥ 0.95 at 10% edits, OR within 5% of GD-trained. Per §3.6, if FAIL: honestly revise verdict to "quality axis PENDING", keep feature opt-in, document in this plan + R455.
- [ ] **T2.3** Add real-shard test (uses riir-neuron-db fixture): construct from `NeuronShard` fact set via the bridge, verify margin.

## Phase 3 — Default Promotion (P3, deferred)

- [ ] **T3.1** If G5 PASSes → promote `hebbian_kernel_memory` to default-on in katgpt-core. Demote the loser (no loser in this case — the primitive is additive, not replacing anything).
- [ ] **T3.2** Update `katgpt-rs/README.md` Feature Showcase with a section on Hebbian Kernel Memory.

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
