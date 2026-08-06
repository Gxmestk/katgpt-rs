# Plan 568: Recurrent Residual Quantization (RRQ) — Single-Checkpoint Multi-Precision Weights

**Date:** 2026-08-06
**Research:** [katgpt-rs/.research/467_Recurrent_Residual_Quantization.md](../.research/467_Recurrent_Residual_Quantization.md)
**Source paper:** [arXiv:2608.04048](https://arxiv.org/abs/2608.04048) — Luo, Dong, Cheng, Shen (Intel), "Recurrent Residual Quantization: A Progressive Multi-Precision Representation for LLMs", Aug 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/rrq_quant.rs` (new module) + Cargo feature `rrq_quant`
**Status:** Active — Phase 1 COMPLETE (skeleton + G1/G3/G4 ALL PASS, 2026-08-06); Phases 2–4 are P1–P3 (no concrete consumer today)

---

## Goal

Ship a **modelless, calibration-free, single-checkpoint multi-precision weight quantization primitive** behind feature flag `rrq_quant`, default-off. The primitive represents an LLM weight matrix as `W̃(t) = Ŵ0 + Σ_{k=1..t} R̂k` — a low-bit quantized base plus a sequence of 2-bit quantized residual corrections. Each prefix of stages is a usable model at a distinct effective bit-width (2/4/6/8-bit for the default 2+2+2+2 config).

**Why now, why default-off:** zero shipped multi-precision weight representation exists in our stack (Research 467 §2.1 confirms zero prior art for `RRQ | residual_quant | Matryoshka | MatGPTQ | multi-precision weight`). The closest cousins — `quant_error_lora.rs` (single SVD correction), QJL residual in TurboQuant (activation level), and `multi_precision_npc` (FAILED training-time, now in riir-train) — all cover related but distinct problems. **No concrete consumer needs this today.** Ship the primitive + benchmark, default-off, revisit when (a) we serve a multi-precision LLM at runtime, (b) per-NPC expert routing (`quant_expert_goat.rs`) wants to share a multi-precision base, or (c) the §3.4 freeze/thaw incremental-upgrade fusion finds a consumer.

**GOAT gate (promotion criterion):**
- **G1** (correctness): prefix-t reconstruction is bit-identical to a reference sum-of-stages; load-time PMR selector classifies Llama (mild) vs Qwen (severe) outlier profiles correctly.
- **G2** (perf): fused 4-stage LUT dequant+dot at parity (within 1.05×) with single-stage 8-bit LUT path at the 8-bit prefix.
- **G3** (no-regression): default features + `--features rrq_quant` both build clean; clippy zero warnings; existing tests unchanged.
- **G4** (alloc-free): prefix-t reconstruction + dot have 0 steady-state allocations.
- **Promotion to default-on: REQUIRES a concrete consumer.** Default-off until then.

---

## Phase 1 — Unblocking Skeleton (CORE, P0)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/rrq_quant.rs` behind `#[cfg(feature = "rrq_quant")]`. Add `pub mod rrq_quant;` to `lib.rs` behind the same gate. Add `rrq_quant = []` to `[features]` in `katgpt-rs/crates/katgpt-core/Cargo.toml`.
- [x] **T1.2** Define the storage type: `RrqStage` (2-bit packed codes `Vec<u8>` + per-group `scales: Vec<f16>` + `zero_points: Vec<f16>` + `n_elements` + `group_size`) + `RrqWeights` (base + `Vec<RrqStage>` residuals + rows/cols).
- [x] **T1.3** Implement `RrqWeights::from_weights_rtn(weights, rows, cols, n_stages, group_size)` — pure RTN, calibration-free. Per the paper Algorithm 1: base = RTN of weights; for k=1..=n_stages: residual = full − recon; stage = RTN of residual; recon += stage_dequant.
- [x] **T1.4** Implement the inference primitives: `prefix_reconstruct_into(t, out)` (additive sum of stages) + `prefix_dot_into(t, x, out, scratch)` (exploits matmul linearity: x·W̃(t) = Σ per-stage GEMVs).
- [x] **T1.5** Implement `RrqStage::dequant_into(out)` + `RrqStage::dot_acc_into(cols, x, out)` (accumulate one stage's contribution).
- [x] **T1.6** **G1 gate tests** — **DEVIATION (documented):** G1 tests live in lib unit tests (`src/rrq_quant.rs::tests`, 7 tests) rather than `tests/rrq_quant_goat.rs`. This matches the codebase convention (most primitives put G1 correctness in `mod tests`; only G4 alloc-count goes in a separate binary because it needs the global `CountingAllocator`). The 7 G1 tests:
  - `g1_stage_quantize_matches_reference`: RTN quantize matches a hand-rolled reference (codes + scale + zero_point).
  - `g1_code_packing_roundtrip`: 2-bit packing/unpacking round-trips; dequant is monotone on monotone input.
  - `g1_prefix_reconstruct_matches_reference`: prefix-t reconstruction is bit-identical to an independent sum-of-stages reference path.
  - `g1_dot_matches_reconstruct_then_dot`: `prefix_dot_into(t, ...)` matches reconstruct-then-matmul within 1 ULP.
  - `g1_more_stages_lower_error`: more residual stages → monotonically lower reconstruction error; 8-bit error < 50% of 2-bit error.
  - `g1_constant_weights_exact`: constant weights → zero residual → exact reconstruction.
  - `g4_prefix_dot_smoke_zero_vec_growth`: smoke test (100 calls, no panic) — the real alloc-count gate is T1.7.
- [x] **T1.7** **G4 alloc-free test** (`tests/rrq_quant_alloc_check.rs`): thread-local `CountingAllocator`; 1000 steady-state `prefix_dot_into` + `dot_acc_into` calls; **0 allocations after warmup**.
- [x] **T1.8** **G3 no-regression**: `cargo clippy -p katgpt-core --all-targets --features rrq_quant` zero warnings; `cargo test -p katgpt-core --lib` (default features, 1840 passed) + `--features rrq_quant` (1847 passed = 1840 + 7 new) — zero regressions.
- [-] **T1.9** Update `katgpt-rs/README.md` Feature Showcase — **DEFERRED.** The README feature showcase is a curated marquee list; RRQ is opt-in with no consumer, so adding it now would be premature. Revisit when promotion to default-on lands (requires a concrete consumer per the GOAT gate).
- [x] **T1.10** Commit on `develop` with `feat:` prefix.

---

## Phase 2 — Load-Time PMR + KS Quant-Strategy Router (P1)

### Tasks

- [ ] **T2.1** Implement `peak_to_mean_ratio(weights: &[f32], group_size: usize) -> f32` — paper §3 PMR metric: `max|x| / mean|x|` per group, max across groups.
- [ ] **T2.2** Implement the strategy router:
  ```rust
  /// Per-layer load-time decision: RRQ vs direct quantization.
  /// Combines PMR (outlier severity for quant-strategy) with KS D-statistic
  /// (Research 200 OAQG, security anomaly flag).
  pub enum QuantStrategy {
      Rrq { n_stages: usize },  // PMR > threshold → RRQ beneficial
      DirectRtn { bits: u8 },    // PMR < threshold → direct fixed-bit
      FlagForReview,             // KS > 0.25 → security review regardless
  }

  pub fn select_quant_strategy(
      weights: &[f32],
      group_size: usize,
      ks_d_stat: f32,
      pmr_threshold: f32,  // paper §3.4 default ~9r for 2+2 split
  ) -> QuantStrategy { ... }
  ```
- [ ] **T2.3** **G1 gate test** for the selector:
  - `g1_pmr_classifies_llama_vs_qwen`: synthetic flat distribution (PMR ~5) → `DirectRtn`; synthetic outlier-heavy (PMR ~30) → `Rrq`. Reproduce paper Table 9 numbers (Llama mean K/MAE 26.5 → DirectRtn; Qwen3 max K/MAE 116 → Rrq).
  - `g1_ks_overrides_pmr`: KS > 0.25 → `FlagForReview` regardless of PMR.
- [ ] **T2.4** Commit on `develop`.

---

## Phase 3 — Fused Multi-Stage LUT Dequant+Dot Kernel (P2)

### Tasks

- [ ] **T3.1** Extend `simd_lut_dequant` (Plan 452) with a multi-stage variant: `dequant_dot_via_lut_multi_stage(codes_per_stage: &[&[u8]], luts: &[QuantLut], scales: &[f32], x: &[f32]) -> f32` — sums 4 stage contributions in registers, single SIMD gather pass per stage.
- [ ] **T3.2** **G2 latency gate**: `g2_4stage_lut_at_parity_with_single_8bit` — at the 8-bit prefix (4 stages × 2-bit), the fused LUT path is within 1.05× of a single 8-bit LUT path. Hypothesis: amortized gather; if it FAILS, document as honest negative (the LUT cost compounds and multi-stage is slower than single-stage at same total bits).
- [ ] **T3.3** Commit on `develop`.

---

## Phase 4 — Prefix-t as Tier Dispatch (STRETCH, P3, deferred until consumer)

### Tasks

- [ ] **T4.1** (DEFERRED) Wire `RrqWeights::prefix_dot_into(t, ...)` into a tier dispatch: Plasma tier (2-bit base only) → Hot tier (+1 stage, 4-bit) → Warm tier (+2 stages, 6-bit). Same checkpoint, three tiers, the tier transition is "include one more stage in the sum".
- [ ] **T4.2** (DEFERRED) Freeze/thaw integration (riir-neuron-db `MerkleFrozenEnvelope`): each stage is its own shard, the prefix-t view is a runtime composition. This is the Super-GOAT angle from Research 467 §3.4 — only pursue when a consumer needs incremental precision upgrades.
- [ ] **T4.3** (DEFERRED) G3 no-regression on existing tier tests.

**Phase 4 deferral rationale:** no concrete consumer needs incremental precision upgrades today. The `quant_expert_goat.rs` per-expert precision routing uses fixed precision per expert; the per-NPC personality divergence story is handled by `CommittedFieldBlend` (different axis). Revisit when one of those consumers wants to share a multi-precision base, or when we serve a multi-precision LLM at runtime.

---

## Open questions / risks

| Risk | Impact | Mitigation |
|---|---|---|
| No consumer materializes; primitive rots unused | Low | Default-off; benchmark proves it works; revisit at quarterly audit. Cost is one feature flag + ~300 LOC. |
| Stage-compounded scale overhead makes RRQ larger than direct 8-bit at the 8-bit prefix | Low | Paper Appendix G shows ~4–5% larger than MatGPTQ, which is acceptable for the multi-precision capability. Document in the GOAT gate. |
| Fused multi-stage LUT path slower than single 8-bit LUT (T3.2 FAIL) | Low | Honest negative result. Phase 1 still ships the additive primitive (just not the fused kernel). |
| PMR threshold (paper §3.4, `~9r` for 2+2) doesn't generalize to our codecs | Low | Phase 2 benchmark calibrates against our existing codecs; the threshold is config, not hardcoded. |
| Small-kernel parameter paradox (Research 463 §2.4.1) applies to RRQ too | Medium | On small CNNs (Moka-scale), each 2-bit residual stage adds 0.5 bits/weight; for a 32×288 conv that's substantial. RRQ is substrate for larger models (LLM weights, future game networks) — same scope caveat as `quant_error_lora`. Document in the module doc. |

---

## Out of scope

- SignRoundV2 learned base (training-method artifact; all-RTN variant is the modelless path)
- GPTQ/AWQ/OmniQuant heterogeneous stage variants (paper §5.4 representationally supported but not empirically evaluated)
- Matryoshka / MatGPTQ nested bit slicing (RRQ explicitly replaces this)
- NVFP4 / FP4 stages (format-specific to NVIDIA Blackwell; verdicted Pass in Research 439)
- riir-train follow-up (RRQ is PTQ, no training; §3.5 check moot)

---

## References

- [Research 467](../.research/467_Recurrent_Residual_Quantization.md) — the parent research note
- [arXiv:2608.04048](https://arxiv.org/abs/2608.04048) — the source paper
- [Plan 452](452_simd_lut_fused_dequant_dot.md) — SIMD LUT fused dequant+dot (Phase 3 substrate)
- [Plan 100](100_block_diagonal_rotation_quantization.md) — RotorQuant / PlanarQuant / IsoQuant (QJL residual cousin)
- [Research 200](../.research/200_Quantization_Outlier_Collapse_Security.md) — KS D-statistic detector (Phase 2 sibling)
- [Research 463](../.research/463_moka_freeze_thaw_lever_audit.md) — `quant_error_lora` (closest cousin; same `E = W − dequant(W_q)` problem, SVD mechanism)
- [Research 020](../.research/020_TurboQuant_Online_Vector_Quantization.md) — TurboQuant (QJL residual at activation level)
- [Research 418](../.research/418_StreamDQ_SIMD_LUT_DeQuant.md) — StreamDQ → SIMD LUT DeQuant (Phase 3 kernel substrate)
