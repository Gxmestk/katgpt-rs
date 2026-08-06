# Plan 568: Recurrent Residual Quantization (RRQ) — Single-Checkpoint Multi-Precision Weights

**Date:** 2026-08-06
**Research:** [katgpt-rs/.research/467_Recurrent_Residual_Quantization.md](../.research/467_Recurrent_Residual_Quantization.md)
**Source paper:** [arXiv:2608.04048](https://arxiv.org/abs/2608.04048) — Luo, Dong, Cheng, Shen (Intel), "Recurrent Residual Quantization: A Progressive Multi-Precision Representation for LLMs", Aug 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/rrq_quant.rs` (new module) + Cargo feature `rrq_quant`
**Status:** Active — Phase 1 (skeleton) pending; Phases 2–4 are P1–P3 (no concrete consumer today)

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

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/rrq_quant.rs` behind `#[cfg(feature = "rrq_quant")]`. Add `pub mod rrq_quant;` to `lib.rs` behind the same gate. Add `rrq_quant = []` to `[features]` in `katgpt-rs/crates/katgpt-core/Cargo.toml`.
- [ ] **T1.2** Define the storage type:
  ```rust
  /// One RRQ stage: 2-bit RTN quantized tensor with per-group scale + zero-point.
  /// Group size 128 (matches paper default + our existing codecs).
  #[derive(Clone)]
  pub struct RrqStage {
      /// 2-bit packed codes (4 codes per byte, group_size=128 → 32 bytes per group).
      pub codes: Vec<u8>,
      /// Per-group scale (f16 or f32 — pick f16 to halve metadata).
      pub scales: Vec<f16>,
      /// Per-group zero-point (f16). Symmetric variant stores 0.
      pub zero_points: Vec<f16>,
      /// Number of weights represented.
      pub n_elements: usize,
      /// Group size (128 default).
      pub group_size: usize,
  }

  /// A complete RRQ weight matrix: base + N residual stages.
  /// Default config: 1 base + 3 residuals → 2/4/6/8-bit prefixes.
  pub struct RrqWeights {
      pub base: RrqStage,
      pub residuals: Vec<RrqStage>,  // typically len=3
      pub rows: usize,
      pub cols: usize,
  }
  ```
- [ ] **T1.3** Implement the constructor — pure RTN, calibration-free:
  ```rust
  impl RrqWeights {
      /// Construct an all-RTN RRQ package from f32 weights.
      /// `n_stages` residual stages (default 3 → 2/4/6/8-bit prefixes).
      /// `group_size` default 128.
      pub fn from_weights_rtn(
          weights: &[f32],
          rows: usize,
          cols: usize,
          n_stages: usize,
          group_size: usize,
      ) -> Self { ... }
  }
  ```
  Inner loop (per the paper Algorithm 1):
  1. Quantize base: `codes_0 = rtn_quant(x, scales_0, zps_0)`; dequant to `x̂_0`.
  2. For k=1..=n_stages: `r^{k-1} = x − x̂^{k-1}`; quantize `rtn_quant(r^{k-1}, scales_k, zps_k)`; dequant to `r̂_k`; accumulate `x̂^k = x̂^{k-1} + r̂_k`.
- [ ] **T1.4** Implement the inference primitives:
  ```rust
  impl RrqWeights {
      /// Reconstruct weights at prefix-t precision into `out`.
      /// t=0 → base only (2-bit); t=1 → 4-bit; t=2 → 6-bit; t=3 → 8-bit.
      pub fn prefix_reconstruct_into(&self, t: usize, out: &mut [f32]) { ... }

      /// Compute `out = x · W̃(t)` at prefix-t precision, exploiting linearity.
      /// `out = x·Ŵ0 + Σ_{k=1..t} x·R̂k` — sum of per-stage GEMVs.
      /// `scratch` is a reusable buffer for the per-stage output.
      pub fn prefix_dot_into(
          &self,
          t: usize,
          x: &[f32],
          out: &mut [f32],
          scratch: &mut [f32],
      ) { ... }
  }
  ```
- [ ] **T1.5** Implement the helper:
  ```rust
  impl RrqStage {
      /// Dequantize all codes into `out`.
      pub fn dequant_into(&self, out: &mut [f32]) { ... }

      /// Compute `out += scale * x · dequant(codes)` for one stage.
      /// (Used by `prefix_dot_into` to accumulate stage outputs.)
      pub fn dot_acc_into(&self, x: &[f32], out: &mut [f32]) { ... }
  }
  ```
- [ ] **T1.6** **G1 gate test** (`katgpt-rs/crates/katgpt-core/tests/rrq_quant_goat.rs`):
  - `g1_prefix_reconstruct_matches_reference`: build an RRQ package from a known f32 matrix, reconstruct at t=0/1/2/3, assert bit-exact match against a hand-computed reference sum.
  - `g1_dot_matches_reconstruct_then_dot`: `prefix_dot_into(t, x, out)` matches `reconstruct → matmul` to f32 epsilon.
- [ ] **T1.7** **G4 gate test** (alloc-free hot path):
  - `g4_prefix_dot_alloc_free`: thread-local `CountingAllocator`; 100 steady-state `prefix_dot_into` calls; assert 0 allocations after warmup (the `Vec<u8>` codes are owned by `RrqWeights`, scratch buffers reused).
- [ ] **T1.8** **G3 no-regression**: `cargo clippy -p katgpt-core --all-targets --features rrq_quant` clean; `cargo test -p katgpt-core --lib` (default features) still passes.
- [ ] **T1.9** Update `katgpt-rs/README.md` Feature Showcase with a one-paragraph entry for RRQ (opt-in, gate results).
- [ ] **T1.10** Commit on `develop` with `feat:` prefix.

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
