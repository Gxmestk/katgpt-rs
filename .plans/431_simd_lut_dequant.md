# Plan 431: SIMD LUT DeQuantization — Software Analog of StreamDQ Near-Memory DQ

**Date:** 2026-07-13
**Research:** [katgpt-rs/.research/418_StreamDQ_SIMD_LUT_DeQuant.md](../.research/418_StreamDQ_SIMD_LUT_DeQuant.md)
**Source paper:** [arxiv 2607.08993](https://arxiv.org/abs/2607.08993) — StreamDQ (Jeong et al., SK Hynix, 2026-07-09)
**Target:** `katgpt-rs/crates/katgpt-core/src/simd_lut_dequant.rs` (new module) + Cargo feature `simd_lut_dequant`
**Status:** Active — Phase 1 COMPLETE (scalar reference shipped + clippy clean + 14 tests PASS). Phase 2 (SIMD inner loops) is next.

---

## Goal

Distill StreamDQ's "shared FP32 ALU + format-specific type-cast" pattern (paper §3.6) into a **generic software SIMD LUT-accelerated dequantize primitive**. The paper ships this in HBM base-die hardware (7.08× mpGEMM speedup on GPU); we ship it as a CPU SIMD primitive that replaces the integer-arithmetic INT→FP cast in our existing quant paths (Q4_K, future INT8/FP8) with a pre-computed LUT lookup.

**Honest scope:** the 7× hardware speedup is NOT replicable in software. The realistic target is a 1.0-1.5× microbench win on Q4_K dequant, plausibly a regression if SIMD gather is slow on a given platform. **The GOAT gate settles it honestly.** If the gate fails, the primitive stays opt-in and we ship the negative result.

**GOAT gate rule:** new feature flag `simd_lut_dequant` ships opt-in. Promote to default only if G1 (bit-exact) + G2 (latency ≥ 1.2× on aarch64 NEON) + G3 (no-regression) + G4 (alloc-free) + G6 (feature isolation) all pass. Demote the loser if the LUT path is slower than the arithmetic path.

**Why this is a Gain not a Super-GOAT:** LUT-based INT→FP is well-known HPC technique; not a new capability class; touches only the transformer-stack dequant slot. See Research 418 §3 for the Q1-Q4 novelty gate failure.

**Not UQ-bearing:** deterministic dequantization, no probability claim. Conformal-naive floor does NOT apply.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/simd_lut_dequant.rs` (feature-gated on `simd_lut_dequant`)
- [x] **T1.2** Define the format-tag enum + LUT trait:
  ```rust
  pub trait QuantLut {
      const LUT_LEN: usize;
      /// Build the LUT for this format at the given (scale, zero).
      fn build(scale: f32, zero: f32) -> Self;
      /// SIMD-friendly lookup; default impl uses scalar indexing, NEON/AVX2 impls override.
      fn lookup(&self, code: u8) -> f32;
  }

  pub struct Int4Lut([f32; 16]);       // 64 bytes = 1 cache line
  pub struct Int8Lut([f32; 256]);      // 1 KB
  pub struct UInt4Lut([f32; 16]);      // for unsigned nibble formats (Q4_K low nibble)
  ```

- [x] **T1.3** Define the generic shared-FP32-ALU dequantize primitive:
  ```rust
  pub fn dequant_via_lut<L: QuantLut>(
      codes: &[u8],    // packed quant codes (nibble-packed or byte-aligned per format)
      lut: &L,         // pre-built per-group LUT (paper §3.4 S/Z buffer analog)
      shift: u32,      // 0 for low nibble / byte, 4 for high nibble
      mask: u8,        // 0x0F for nibble, 0xFF for byte
      out: &mut [f32], // destination (length == codes.len())
  )
  ```
  The LUT bakes in `(x - z) * s` per code value; the inner loop is pure lookup, no FP32 ALU per element.

- [x] **T1.4** Scalar reference impl: builds the LUT, iterates `codes`, writes `lut.lookup((code >> shift) & mask)` to `out`. No SIMD yet.
- [x] **T1.5** Wire into `katgpt-core/src/lib.rs` behind `#[cfg(feature = "simd_lut_dequant")]`.
- [x] **T1.6** Add `simd_lut_dequant = []` to `katgpt-core/Cargo.toml` `[features]` table.

**Exit:** `cargo clippy -p katgpt-core --features simd_lut_dequant` clean. Scalar reference test passes. ✅ DONE (2026-07-13): clippy clean, 14 unit tests + 1 doctest PASS, default-feature clippy also clean (feature isolation). Added `dequant_arithmetic_ref` as the G1 correctness oracle (the comparator the Phase 4 G1 gate will call against). LUT convention: `lut[i] = (signed(i) - zero) * scale` where `zero` is in code units (standard asymmetric-quantization form). Q4_K caller converts FP-space `m0_val` to code units via `zero = m0_val / d_sc0` at build time.

---

## Phase 2 — SIMD Inner Loop

### Tasks

- [ ] **T2.1** Add NEON inner loop (`#[cfg(target_arch = "aarch64")]`):
  - Load 4 (or 8) packed codes via `vld1q_u8`
  - Shift+mask via `vshrq_n_u8` + `vandq_u8` (paper §3.6 sign-extension analog: zero-extend for nibbles)
  - Convert to u32 indices via `vmovl_u16` + `vmovl_u32`
  - Gather 4 (or 8) f32 values from the LUT using scalar indexing on the extracted u32 lanes (NEON has no native gather — fall back to scalar extraction OR use `vld1q_f32` with computed offsets if 4-aligned)
  - Write to `out` via `vst1q_f32`

- [ ] **T2.2** Add AVX2 inner loop (`#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]`):
  - Load 8 packed codes via `_mm_loadl_epi64`
  - Shift+mask via `_mm_srli_epi16` + `_mm_and_si128`
  - Use `_mm_i32gather_ps` (the native AVX2 gather the paper's hardware does physically)
  - Write to `out` via `_mm256_storeu_ps`

- [ ] **T2.3** Scalar fallback (`#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2"))))]`): use the T1.4 scalar loop.

- [ ] **T2.4** Add WASM SIMD128 fallback if straightforward (`#[cfg(target_arch = "wasm32")]`): WASM has no gather; use scalar extraction. Document this as a known slow path.

**Exit:** NEON + AVX2 + scalar paths all pass bit-exact test vs Phase 1 reference.

---

## Phase 3 — Fused DeQuant + Dot (the strongest fusion — Research 418 §2.3)

### Tasks

- [ ] **T3.1** Add `dequant_dot_via_lut<L>(codes, lut, x: &[f32], shift, mask) -> f32`:
  - Fuses dequant + dot product (paper's "fused DQ-GEMM" analog)
  - Dequant happens in registers, never spills to L1
  - Inner loop: gather → load `x[i]` → FMA → next iteration

- [ ] **T3.2** NEON impl using `vfmaq_f32` (FMA), AVX2 impl using `_mm256_fmadd_ps`.

- [ ] **T3.3** Scalar reference for testing.

**Exit:** Fused path passes bit-exact dot vs `(dequant then simd_dot_f32)` two-step.

---

## Phase 4 — GOAT Gate (the verdict)

### Tasks

- [ ] **T4.1** **G1 correctness bench** — `katgpt-rs/benches/simd_lut_dequant_goat.rs`:
  - Generate random Q4_K blocks (use `riir-ai`-compatible block layout OR inline the block decode)
  - Decode via (a) current arithmetic path, (b) new LUT path
  - Assert max abs diff == 0.0 (bit-exact)
  - PASS required for promotion.

- [ ] **T4.2** **G2 latency bench** — criterion microbench:
  - Workloads: single-block dequant (256 elements), full-row dequant (4096 elements), fused dequant+dot (4096 elements)
  - Compare: arithmetic path vs LUT path vs fused LUT+dot
  - Targets (aarch64 NEON, release build):
    - Single-block: LUT ≥ 1.0× (no regression — small workload, LUT overhead may dominate)
    - Full-row: LUT ≥ 1.2× (the win kicks in when amortizing LUT build cost)
    - Fused: ≥ 1.3× vs split (dequant to buffer + simd_dot)
  - If G2 FAILS on all three: document negative result, keep opt-in, do NOT promote. Honest.
  - If G2 PASSES on full-row or fused: promote to default-on for that path.

- [ ] **T4.3** **G3 no-regression** — `cargo clippy --workspace --all-features` clean + `cargo test -p katgpt-core --all-features` clean.

- [ ] **T4.4** **G4 alloc-free** — verify via `#[track_allocator]` or manual inspection: LUT is stack `[f32; N]`, hot loop has zero `Vec`/`Box`/`String`. Document in code comment.

- [ ] **T4.5** **G5 SIMD-level report** — print NEON instruction count (via `cargo asm` or godbolt) for the inner loop. Report whether the LUT lookup lowered to `vld1q_f32` (best case) or scalar extraction (fallback).

- [ ] **T4.6** **G6 feature isolation** — `cargo check --features simd_lut_dequant` clean without affecting default paths. The feature is purely additive.

- [ ] **T4.7** Write `.benchmarks/431_simd_lut_dequant_goat.md` with the gate result + decision (promote / demote / keep opt-in).

**Exit:** GOAT gate documented. If promote: bump `simd_lut_dequant` to default feature in `katgpt-core/Cargo.toml`.

---

## Phase 5 — Q4_K Integration (CONDITIONAL — only if Phase 4 G2 PASS)

**Defer this phase until Phase 4 verdict is in.** If G2 fails, this phase is cancelled and the primitive stays opt-in for future consumers (FP8 etc.).

### Tasks

- [ ] **T5.1** Open `riir-ai/.plans/NNN_q4k_lut_integration.md` (separate plan in private repo)
- [ ] **T5.2** Refactor `riir-ai/crates/riir-engine/src/quant/q4k.rs::dequantize_row_q4_k` to optionally use `katgpt_core::simd_lut_dequant::dequant_via_lut` when feature is enabled
- [ ] **T5.3** Add a feature gate in riir-engine that pulls `katgpt-core/simd_lut_dequant`
- [ ] **T5.4** Re-run the GOAT bench in riir-engine context (with real Q4_K blocks from a model file)
- [ ] **T5.5** If riir-engine bench confirms the win: promote in riir-engine too. Else: keep opt-in in riir-engine.

---

## Phase 6 — Future Format Coverage (DEFERRED)

### Tasks

- [-] **T6.1** FP8 → FP32 LUT (256 entries). Lower priority — we don't ship FP8 weights today.
- [-] **T6.2** INT8 → FP32 LUT for Q8_KV (`riir-ai/crates/riir-engine/src/quant/q8kv.rs`). Same pattern as INT4 but byte-aligned.
- [-] **T6.3** Runtime sideband-tag dispatch (paper §3.2): `dequant_dispatch(tag: QuantFormat, src, dst)` — currently compile-time via monomorphization; runtime dispatch would enable mixed-format batches. Open issue if a real consumer needs it.

---

## Per-stack promote/demote ledger

Per the §1.6 MOAT-gate discipline, this primitive lands in the **transformer-stack dequant slot**. The slot's current occupant is the arithmetic-cast path in `q4k.rs`. The promote/demote decision:

| Slot | Current occupant | Challenger | Verdict (post Phase 4) |
|---|---|---|---|
| Q4_K dequant (single-block) | arithmetic cast | LUT | TBD (likely no win — LUT overhead dominates) |
| Q4_K dequant (full-row) | arithmetic cast | LUT | TBD (the realistic win point) |
| Q4_K fused dequant+dot | split (dequant + simd_dot) | fused LUT+dot | TBD (strongest candidate) |
| INT8/FP8 dequant | (no current path) | LUT | N/A — LUT becomes the default when these ship |

If the LUT path loses on all slots: keep opt-in, document the negative result in `.benchmarks/431_*.md`. The infrastructure still serves future FP8 work.

---

## Constraints honored

- ✅ Modelless first — pure inference-time, no training, no riir-train deferral (§3.5 check passed)
- ✅ Feature flag discipline — `simd_lut_dequant` ships opt-in, promotes only on GOAT gate pass
- ✅ Alloc-free hot path (G4) — LUT is stack `[f32; N]`, no Vec/Box
- ✅ Pre-computed LUT (AGENTS.md optimization rule: "Pre-compute lookup tables once, store in config")
- ✅ Fixed-size arrays for bounded domains (AGENTS.md rule: "Use fixed-size arrays `[T; N]` when domain is bounded") — INT4 LUT is `[f32; 16]`, INT8 is `[f32; 256]`
- ✅ `cargo clippy` not `cargo check` (AGENTS.md rule)
- ✅ No deferral of benchmark task (AGENTS.md rule) — Phase 4 is mandatory before promotion
- ✅ 5-repo discipline — primitive in `katgpt-core` (public), integration tuning in `riir-ai` (private, Phase 5)
- ✅ Promote/demote tracked per stack (§1.6 katgpt-rs MOAT gate)

---

## TL;DR

Plan 431 ships a generic software SIMD LUT-accelerated dequantize primitive distilled from StreamDQ's hardware DQB design. The technique (LUT INT→FP + shared FP32 ALU) is well-known HPC; the win is platform-dependent and bounded by SIMD gather latency vs CVT instruction latency. **Realistic target 1.0-1.5× microbench, plausibly a regression on slow-gather platforms.** The GOAT gate (Phase 4) settles it honestly — if G2 fails, keep opt-in and ship the negative result. Not a Super-GOAT (Research 418 Q1-Q4 all NO), not UQ-bearing (no conformal floor), not a training dependency. Phase 5 (riir-ai Q4_K integration) is conditional on Phase 4 success. Phase 6 (FP8, INT8, runtime dispatch) deferred until a real consumer appears.
