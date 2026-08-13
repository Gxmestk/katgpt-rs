# Issue 584: FlashMemory-style Periodic Sparse Attention Validation on 4090

**Filed:** 2026-08-13
**Source:** Research 436 (FlashMemory-DeepSeek-V4, arXiv:2606.09079) + PASS-Redirect from SparDA (arXiv:2606.04511)
**Status:** Open — Phase 1 mechanism landed (2026-08-13); Phase 2 scale test blocked on 4090 (Bench 456)
**Scope:** POC — validate FlashMemory's sigmoid-threshold periodic sparse attention mechanism + scale benefit on real hardware

---

## Motivation

Research 436 distilled FlashMemory's lookahead sparse attention (periodic batch-scoring every τ=64 steps + sigmoid threshold ≥0.5 + 90% KV reduction at 500K context) as a modelless inference paradigm. Originally PASSed because the stack's micro-transformer (n_layer=1, block_size=16) doesn't hit the long-context regime.

**Revised verdict (2026-08-13):** The stack has the hardware + models + substrate to validate this at real scale:

| Component | What | Why it matters |
|---|---|---|
| **4090 GPU** (24GB, Tailscale `100.85.179.44`) | CUDA inference + training | 16.2GB free (8.4GB used by Windows desktop + sibling Bench 456) |
| **Kimi-K3-0.40B** (395M, safetensors) | Architecture testbed | Same `kimi_k3` arch as full K3; stack already has `kimi_k3/decoder_layer.rs` (MLA+KDA+LatentMoE); MoE routing is `sigmoid` (matches AGENTS.md) |
| **Bonsai dspark-Q4_1** (1.95GB, GGUF) | Scale testbed | Qwen3.5 arch, 256K native context, ALL layers full attention (unlike Kimi K3's hybrid 6/8 linear). At 256K: KV cache ≈67GB >> 24GB → offloading/sparse attention IS the serving regime |
| **`VortexFlow` trait** (katgpt-attn) | Sparse attention substrate | 8 router impls (BlockTopK, Entmax, ValueEnergy, ChannelAware, MetaRouter, MSA MaxPool, MSA MaxStdDev, MSA PerGroup). `forward_indexer` takes Query — FlashMemory extends with periodic refresh + sigmoid threshold |

**The load-bearing math:** at 256K context on Qwen3.5-27B (Bonsai):
```
KV cache (FP16, GQA 8 KV heads, 64 layers):
  256K × 64 × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes ≈ 67 GB
```
67GB >> 24GB VRAM. FlashMemory's 90% reduction → 6.7GB → fits with 1.95GB weights (total ~8.7GB, 15GB headroom). **This is the winning formula for the 4090.**

---

## Two-model validation strategy

### Phase 1 — Mechanism test: Kimi-K3-0.40B (395M, 4K context)

**Goal:** Validate the FlashMemory mechanism (periodic sigmoid-threshold sparse selection) works with the existing `VortexFlow` substrate on Kimi K3's MLA layers.

**Caveat:** Kimi K3 is hybrid attention:
- `full_attn_layers: [4, 8]` → 2 MLA layers (full attention, has KV cache)
- `kda_layers: [1,2,3,5,6,7]` → 6 KDA layers (linear attention, fixed-size state, NO growing KV)

Sparse attention only applies to the **2 MLA layers**. This is a MECHANISM test, not a scale test. Validate:
- [x] Does `VortexFlow::forward_indexer` work with MLA's compressed latent KV (`kv_lora_rank: 128`)? **YES** — `FlashMemoryBlockCache::rebuild_from_cache` builds block centroids from `MlaKVCache::latent_kv_at()` + up-projects per-head via `W_UK`. Test: `q1_block_centroids_built_from_mla_latent_kv`.
- [x] Does the periodic refresh (every τ steps) amortize selection cost on the MLA layers? **YES** — `FlashMemorySelector::select()` caches the last decision; refreshes only when `step - last_refresh ≥ τ`. Test: `q2_periodic_refresh_amortizes_selection` (2 refreshes over 10 steps with τ=5).
- [x] Does sigmoid threshold (≥0.5) produce dynamic block counts (vs rigid top-k)? **YES** — `FlashMemorySelector` applies `sigmoid(score) ≥ threshold` per-head per-block; different queries select different block sets. Test: `q3_sigmoid_threshold_dynamic_block_counts`.
- [x] Does the selection preserve accuracy on a simple retrieval task at 4K context? **MECHANISM VALIDATED** — sparse forward (`mla_forward_token_flashmemory`) runs finite + stable at Kimi-K3-0.40B dims (d_c=128, d_h=64, 8 heads). Tests: `q4_sparse_forward_runs_without_panic`, `q4_kimi_k3_0_40b_config_smoke`. **Full retrieval-accuracy validation on real weights deferred to Phase 2 (needs GPU for Bonsai 256K context).**

**Phase 1 implementation:** `katgpt-attn/src/dash_attn/flashmemory_sparse.rs` (feature gate `flashmemory_sparse`, opt-in). 9 tests (Q1-Q4 mechanism validation). Sparse forward runs at production MLA dims; mechanism proven modellessly on M3 Metal.

**Model:** `riir-train/data/kimi-k3-0.40b/model.safetensors` (downloaded, ~1.5GB F32)

### Phase 2 — Scale test: Bonsai dspark-Q4_1 (1.95GB, 256K context)

**Goal:** Validate that FlashMemory's 90% KV reduction actually helps at long context on the 4090.

**Model:** `Ternary-Bonsai-27B-dspark-Q4_1.gguf` (1.95GB) from `prism-ml/Ternary-Bonsai-27B-gguf`

**Validations:**
- [ ] At 256K context: does FlashMemory's sparse selection reduce KV from ~67GB to ~6.7GB?
- [ ] Does the reduced KV fit in 24GB VRAM alongside the 1.95GB weights?
- [ ] Does the periodic refresh (τ=64) provide measurable latency improvement vs per-step scoring?
- [ ] Does the MRCR failure mode (dense global memory needed) manifest on game-AI-style workloads?
- [ ] Length generalization: does the indexer trained at shorter context transfer to 256K?

### Phase 3 — GOAT gate

- [ ] **G1 (correctness):** FlashMemory sparse selection preserves retrieval accuracy (RULER NIAH) vs dense baseline at matched context
- [ ] **G2 (perf):** decode latency with FlashMemory ≤ dense baseline at 256K context on 4090
- [ ] **G3 (no-regression):** existing VortexFlow tests pass with the periodic refresh extension
- [ ] **G4 (alloc-free):** periodic refresh path is alloc-free in steady state (reuse selection buffer)
- [ ] **G5 (memory):** KV cache footprint reduced ≥80% vs dense at 256K context (paper claims 90%)

---

## Training redirect → riir-train

FlashMemory's dual-encoder indexer training (BCE/Focal loss on pre-computed hidden states) is a training recipe → riir-train.

- [ ] File a riir-train Plan for the indexer training recipe scaled to a single 4090
- [ ] The paper's recipe: pre-compute hidden states offline (batch-by-batch at any context length), train indexer with BCE on pre-computed data → **more 4090-friendly than SparDA's KL-divergence approach** (which needs full forward at 65K context)
- [ ] GOAT gate: compare trained indexer vs modelless sigmoid threshold (FlashMemory's periodic batch-scoring works with ANY scorer — the trained indexer is an upgrade, not a requirement)

**SparDA comparison (PASS — stays secondary):** SparDA's trained Forecast head (KL divergence, 0.41% params) is a speed optimization (async CPU→GPU prefetch overlap) that only matters once you're already in the offloading regime. FlashMemory's memory reduction is the primary win for the 4090's 24GB constraint. SparDA's PCIe Gen4 benefit is halved vs H100 Gen5. SparDA PASS-Redirect logged in R436.

---

## 4090 status (2026-08-13)

```
GPU: RTX 4090, 8.4GB / 24.6GB used (16.2GB free)
GPU util: 4% (idle desktop + sibling Bench 456 training)
Active process: aurora_accuracy_parity_arm_c.exe (PID 17508)
Repo sync: DIVERGED — M3 at 9bd8e170, 4090 at 04e77fda (Bench 456 WIP)
```

**Blocker:** 4090 is running Bench 456 (sibling agent). Per AGENTS.md "dont use 4090 if any GPU is running there" — GPU experiments must wait for Bench 456 to complete + repos to sync.

**Unblocked tasks (can do on M3 now):**
- [x] Download Kimi-K3-0.40B to `riir-train/data/kimi-k3-0.40b/`
- [x] Download Bonsai dspark-Q4_1 to `riir-train/data/`
- [x] Wire `VortexFlow` sigmoid threshold selection (modelless, CPU-testable on M3) — **DONE** (commit below): `FlashMemorySelector` + `FlashMemoryBlockCache` + `mla_forward_token_flashmemory`
- [x] Write Phase 1 mechanism test (runs on M3 Metal at 4K context) — **DONE**: 9 tests validating Q1-Q4

**Blocked tasks (need 4090):**
- [ ] Phase 2 scale test at 256K context
- [ ] Indexer training (riir-train Plan)
- [ ] GOAT gate G2 (decode latency on 4090)

---

## Substrate map

| FlashMemory concept | Existing substrate | Gap |
|---|---|---|
| Periodic refresh every τ steps | NOT shipped — all scorers run per-step (R436 §2.1 gap) | **This issue's primary deliverable** |
| Sigmoid threshold selection | EGA sigmoid gate (R100), sigmoid margin (R061) | Pattern exists, needs wiring into VortexFlow |
| 3-layer union routing | Multi-head attention (implicit) | Optional — OR-mode is a conservative safety net |
| Block max-pool scoring | MSA (R225), VortexFlow centroid (R176) | ✅ Shipped |
| CPU cold / GPU hot tier | Plasma/Hot/Warm/Cold tiering | ✅ Conceptual |
| Compressed KV entries | OCTOPUS (R063), SpectralQuant (R039), Shard (R109) | ✅ Shipped |
| MLA compressed latent KV | `kimi_k3/decoder_layer.rs` (kv_lora_rank=128) | ✅ Shipped — Phase 1 test target |

---

## References

- [Research 436](../.research/436_FlashMemory_Lookahead_Periodic_Sparse_Attention.md) — FlashMemory distillation
- [Research 176](../.research/176_Vortex_Programmable_Sparse_Attention_Consolidated.md) — VortexFlow trait
- [arXiv:2606.09079](https://arxiv.org/abs/2606.09079) — FlashMemory-DeepSeek-V4
- [arXiv:2606.04511](https://arxiv.org/abs/2606.04511) — SparDA (PASS-Redirect, secondary)
- [Kimi-K3-0.40B](https://huggingface.co/inference-optimization/Kimi-K3-0.40B) — Phase 1 testbed
- [Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/tree/main) — Phase 2 testbed (dspark-Q4_1 = 1.95GB)
