# Research 418: StreamDQ — SIMD Analog of Near-Memory Weight DeQuantization

> **Source:** [StreamDQ: Near-Memory Weight DeQuantization in Custom HBM for Scalable AI Inference Acceleration](https://arxiv.org/abs/2607.08993) — Jeong et al., SK Hynix, 2026-07-09
> **Date:** 2026-07-13 (opened) · 2026-07-13 (resolved — both GOAT gates PASS)
> **Status:** RESOLVED — Gain confirmed by Plan 431 Phase 4 (raw primitive 4.58×) + Plan 486 Phase 3 (real Q4_K blocks 2.0–2.3×). Default-on in katgpt-core (`simd_lut_dequant`) and riir-engine (`simd_lut_q4k`).
> **Related Research:** 110 (Ciot ternary SIMD plasma tier), 218 (Breakeven routing — covers paper's kernel selection), 200 (Quantization outlier collapse), 202 (QAT infusion), 065 (RotorQuant), 020 (TurboQuant), 159 (KVarN)
> **Related Plans:** 431 (SIMD LUT DeQuant), 227 (async Q/DQ overlap — cousin), 218 (Breakeven router — covers fused/split kernel selection)
> **Classification:** Public
> **PASS-Redirects (synthesis):** Oh, Nam, Bhattacharjya, Chen, Das, Yun, Jang, Ding, Dutt, Imani [arXiv:2511.13676 "T-SAR: A Full-Stack Co-design for CPU-Only Ternary LLM Inference via In-Place SIMD ALU Reorganization"] (UC Irvine, DATE 2026) — PASS. T-SAR's diagnosis (memory-resident ternary LUTs make T-MAC/BitNet.cpp CPU inference memory-bound: TLUTs = 87.6% of memory transactions, 91.6% of exec time) names a bottleneck our tree doesn't have, and its fix is ISA silicon (TLUT/TGEMV instructions generating LUTs in the SIMD register file, +1.4% area — gem5-modeled, not shipping). Its algorithmic core — decompose a ternary dot into **two power-of-2 binary subset-sum tables + one subtract** (T-SAR: dense(±1) − sparse(0-mask); ours: pos − neg, algebraically the same 2×2^c split) — already ships as `gemv_ternary_plane_lut` (riir-gpu, Issue 606 T3c, `ternary_lut_gemv`): `Σ signᵢ·xᵢ = S[p] − S[q]`, 16-entry activation-keyed table built on-the-fly in **threadgroup memory** (on-chip, zero DRAM TLUT traffic — the property T-SAR needs new ISA for; GPUs have it structurally), one build amortized over all 32 workgroup rows. Distilled as kernel_opt B17 `split-sign-accumulators-defer-subtraction` + B20 `threadgroup-subset-sum-lut-keyed-by-activation`. CPU side needs nothing: the ternary dot is arithmetic (sign-mask bit-planes + 5-trit/byte `ternary_trit_pack`), no memory LUT to eliminate — and this note's documented standalone-LUT NEON regression (0.286×, no native gather) is precisely the CPU-silicon gap T-SAR patches with hardware, not software.

---

## TL;DR

StreamDQ is a **hardware** paper (DQBs in HBM base dies, 12nm CMOS, 7.08× mpGEMM speedup, 90.23% lower energy) — but its **dequantization techniques** distill cleanly into **software SIMD analogs** we already partially ship. The honest framing: we cannot replicate the 7× hardware speedup, but the paper exposes three real gaps in our SIMD dequant substrate. **Verdict: Gain — CONFIRMED.** Both GOAT gates now PASS (Plan 431 Phase 4 raw primitive 4.58× fused / 0.286× plain-dequant on NEON; Plan 486 Phase 3 real Q4_K blocks 2.0–2.3× fused CPU GEMV). The LUT path wins on the **fused dequant+dot** slot only; standalone dequant stays arithmetic (no native NEON gather). `simd_lut_dequant` is default-on in katgpt-core; `simd_lut_q4k` is default-on in riir-engine. Phase 6 (FP8/INT8/Q8_KV LUT + sideband dispatch) deferred `[-]` — open an issue if a consumer appears.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is a **polymorphic LUT-accelerated dequantize** — one shared FP32 affine stage `(x − z) · s` fed by format-specific LUT lookups that replace the slow integer→float arithmetic cast. The "sideband tag" becomes a `QuantFormat` enum dispatched at runtime. The "pseudo-channel-aware layout" is already shipped as cache-line alignment.

---

## 1. Paper Core Findings

StreamDQ integrates compact DeQuantization Blocks (DQBs) into the HBM base die. Each DQB sits on a pseudo-channel read path and performs on-the-fly dequantization using a 3-bit sideband tag attached to standard memory loads. Key results:

| Metric | Value |
|---|---|
| DQB area (12nm CMOS) | 0.127 mm² per DQB |
| DQB power | 0.355 W per DQB (20% toggle) |
| mpGEMM speedup | up to 7.08× over fused-kernel baselines |
| Energy reduction | up to 90.23% |
| End-to-end latency reduction | up to 54.68% |
| Decode throughput gain | up to 2.20× |
| Region-lookup-table size | 3.5–3.9 KB (one entry per quantized linear layer) |
| S/Z buffer per DQB | 4 KB |
| S/Z request table per DQB | 8 KB |

### 1.1 The five techniques (paper §3)

1. **Sideband tagging** (§3.2) — 3-bit tag per memory read selects bypass / INT4→FP16 / INT8→BF16 / FP8→FP16 / etc. No ISA change; tag rides on spare metadata bits.
2. **Pseudo-channel-aware layout** (§3.3) — offline transformation places each weight group + its S/Z metadata in the same HBM pseudo-channel, eliminating cross-PC communication. S/Z replication overhead ≤ 3.47% across practical configs.
3. **S/Z request generator + 4 KB S/Z buffer** (§3.4) — DQB computes `group_index = (WA − WBA) / group_width` and caches S/Z locally. Metadata reads are 1.16–7.25% of weight reads (Table 3).
4. **Shared FP32 ALU + format-specific type-cast** (§3.6) — one FP32 dequant arithmetic unit `(x − z) · s` reused across {INT4, INT8, FP8} → {FP16, BF16}. Format-specific logic only does the type cast.
5. **Wire-mapping FP-to-FP + LUT INT-to-FP** (§3.6):
   - **FP-to-FP** (FP8→FP16, FP32→FP16): bit-level wire mapping via 2:1 muxes for normalized values + zero. No ALU/shift for exponent bias.
   - **INT-to-FP**: single shared LUT for INT8→FP32 (256 entries), reused for INT4 via sign-bit extension. **Key optimization**: INT8→FP32 has no fractional part, so the lower 16 FP32 bits are hardwired zero → LUT stores only upper 16 bits (1 KB → 512 B).

### 1.2 The kernel-selection insight (§6.4)

- Fused DQ+GEMM kernel wins at small batches (memory-bound regime, shared memory avoids HBM traffic)
- Split DQ+GEMM kernel wins at large batches (compute-bound regime, but pays HBM write-back/reload for intermediate dequantized weights)
- GPTQ switching threshold: batch 64; AWQ: batch 256
- StreamDQ dominates at batch ≥ 8 because it fuses dequant into the memory read (no CUDA-core bottleneck, no HBM write-back)

---

## 2. Distillation — software SIMD analog

The paper's hardware framing obscures that **four of the five techniques have direct software SIMD analogs**, and we already ship parts of the pattern. The table below is the honest mapping.

| Paper technique (hardware) | Software SIMD analog | Current codebase state | Gap |
|---|---|---|---|
| Sideband tag dispatch | `QuantFormat` enum + match | Implicit (function names per format) | No runtime-polymorphic dispatch |
| Pseudo-channel-aware layout | Cache-line-aligned struct-of-data | **SHIPPED** `channel_simd.rs::AlignedWeightMatrix` (64-byte rows) | None |
| S/Z co-located buffer | L1 cache (block layout keeps S/Z near W) | **SHIPPED** `BlockQ4K { d, dmin, scales, qs }` is already struct-of-data | None for typical block sizes |
| Shared FP32 ALU | Generic `dequant_affine(x, z, s)` reused across formats | **MISSING** — `q4k.rs` hardcodes the INT4 logic inline | Real refactor target |
| Wire-mapping FP-to-FP | Bit-cast + SIMD shuffle for normalized values | **PARTIAL** — `f16::from_bits(block.d)` is wire-cast for scale load; no FP8 path | FP8 missing (low priority) |
| LUT-based INT-to-FP | Pre-computed `f32` LUT, SIMD gather by index | **MISSING** — `(qs[i] & 0x0F) as f32` is arithmetic cast | **Primary gap** |
| Fused DQ+GEMM kernel | Fused dequantize-matmul (dequant in registers, never spills) | **PARTIAL** — `simd_dot_f16_f32` fuses f16 load + dot; no fused Q4/K dot | Real fusion target |
| Split DQ+GEMM kernel | Dequant to buffer + separate matmul | **SHIPPED** as the default path in `q4k.rs` + `simd_dot_f32` | Already covered |
| Kernel-selection heuristic | Breakeven routing (memory-bound vs compute-bound threshold) | **SHIPPED** Plan 218 `breakeven/` | Already covered at the algorithmic level |
| Async double-buffered DQ | CPU dequant chunk N+1 while GPU computes chunk N | **SHIPPED** `crates/katgpt-kv/src/async_qdq.rs` (Plan 227) | Already covered |

### 2.1 The Q4_K hot path — what we do today

`riir-ai/crates/riir-engine/src/quant/q4k.rs:257-310`:

```rust
for pair in 0..4 {
    let (sc0, m0) = get_scale_min_k4(pair * 2, scales);
    let d_sc0 = d * sc0 as f32;           // shared FP32 ALU ✓
    let m0_val = dmin * m0 as f32;        // shared FP32 ALU ✓
    ...
    for c in 0..chunks {
        let l = c * 4;
        // ↓ ARITHMETIC INT→FP CAST — the gap
        dst[out_base + l]     = d_sc0 * (qs[qs_base + l]     & 0x0F) as f32 - m0_val;
        dst[out_base + l + 1] = d_sc0 * (qs[qs_base + l + 1] & 0x0F) as f32 - m0_val;
        dst[out_base + l + 2] = d_sc0 * (qs[qs_base + l + 2] & 0x0F) as f32 - m0_val;
        dst[out_base + l + 3] = d_sc0 * (qs[qs_base + l + 3] & 0x0F) as f32 - m0_val;
    }
    ...
}
```

Observations:
- The shared-FP32-ALU pattern (`d_sc0 * x - m0_val`) is **already there** — paper technique #4 ships implicitly.
- The INT4→FP32 cast (`& 0x0F as f32`) is the **bottleneck the paper eliminates via LUT**. In software, `int_to_float` is a CVT instruction per element; a LUT lookup is a single gather.
- The 4-element manual unroll hints at auto-vectorization but doesn't guarantee it.

### 2.2 The distilled primitive

```text
dequant_via_lut::<F: QuantFormat>(
    codes: &[u8],             // packed quantized weights
    lut:   &F::Lut,           // format-specific lookup table (16/256/256 entries)
    scale: f32,               // per-group s (shared FP32 ALU)
    zero:  f32,               // per-group z (shared FP32 ALU)
    out:   &mut [f32],        // destination
)
where
    F::Lut: LutLookup<Output = f32>;  // SIMD-gather abstraction
```

The LUT does the type conversion (format-specific); the FP32 affine `x * scale - zero` is shared. This is the paper's §3.6 "shared FP32 ALU + format-specific type-cast" pattern, lifted to software.

For Q4_K specifically, the LUT is 16 entries × 4 bytes = **64 bytes = one cache line**. It fits entirely in L1 for the duration of a block decode. The paper's "lower 16 bits are zero" optimization reduces the LUT storage but doesn't help us at INT4 (already tiny).

### 2.3 Fusion candidates (cross-pollination)

| Fusion | What it produces | Strength |
|---|---|---|
| StreamDQ LUT × Ciot ternary (R110) | Replace ternary `{-1, 0, +1} × scale` with 3-entry LUT — but ternary already does this implicitly via bit-planes | Weak (no gain) |
| StreamDQ sideband tag × Breakeven router (R218) | Runtime-polymorphic format dispatch + memory-bound/compute-bound routing → mixed-format batch inference | Medium — new capability, but unclear demand |
| StreamDQ fused DQ+GEMM × Q4_K | Fused Q4_K dequantize + FP32 dot product (dequant in registers, never spill to L1) | **Strong** — directly addresses the paper's large-batch win in software |
| StreamDQ LUT × async Q/DQ overlap (Plan 227) | LUT dequant + double-buffered overlap → faster CPU side of the overlap pipeline | Medium — incremental |
| StreamDQ LUT × LatCal fixed-point bridge | LUT as the fixed-point→float bridge for chain-committed weights | **Speculative** — LatCal weights aren't quantized this way today; would need design |

The strongest fusion is **fused Q4_K dequantize + FP32 dot** — the software analog of the paper's "fused DQ-GEMM wins at small batch" result, extended to "fused DQ-GEMM eliminates the L1 spill at all batch sizes" because we don't have CUDA cores to bypass.

### 2.4 Latent-space reframing (§1 workflow step 3 — mandatory)

The paper operates on raw quantized weights and S/Z metadata — not latent space. The latent reframing is weak but present:

- The **LUT is a bridge function** (raw quant code → FP32 latent) — the same structural role as the raw→latent bridges in the global AGENTS.md, just at the weight-format level rather than the HLA level.
- The shared FP32 ALU's `(x − z) · s` is an affine projection into FP32 latent space.
- The sideband tag dispatches which bridge function to apply per memory region — analogous to bridge selection by domain (physical raw / semantic latent).

This reframing doesn't unlock a Super-GOAT angle. The primitive stays in the "quant-aware inference" slot of the transformer stack (a katgpt-rs engine concern), not in the HLA / functor / shard / LatCal / DEC substrate.

---

## 3. Verdict

| Criterion | Assessment |
|---|---|
| **Q1 No prior art?** | NO — LUT-based INT→FP is well-known HPC; we already use `f16::from_bits` wire-cast for scale loads |
| **Q2 New capability class?** | NO — we already have Q4_K, Q8_KV, ternary, FP16 SIMD dequant paths |
| **Q3 Product selling point?** | NO — perf optimization of existing capability, not a new capability |
| **Q4 Force multiplier (≥2 pillars)?** | NO — touches the transformer-stack dequant slot only |

**Q1-Q4 = NO → not Super-GOAT.** No private guide required (§1.5).

### 3.1 Tier verdict: **Gain — CONFIRMED** (both GOAT gates PASS, 2026-07-13)

One-line reasoning: real gap in our SIMD substrate (no LUT-based INT→FP, no shared-FP32-ALU primitive), now closed by a fused dequant+dot primitive that wins 2.0–4.58× over the arithmetic baseline. The win is slot-specific: **fused dequant+dot = PROMOTE**, **standalone dequant = DEMOTE** (NEON has no native gather, so plain LUT dequant is 0.286× — arithmetic cast remains the GOAT for the standalone slot). Latency was benchmarked, not pre-claimed.

**Post-GOAT outcome ledger:**

| Slot | Verdict | Number |
|---|---|---|
| Fused dequant+dot (raw primitive, Plan 431 Phase 4) | ✅ PROMOTE — default-on in katgpt-core | 4.58× |
| Fused CPU GEMV on real Q4_K blocks (Plan 486 Phase 3) | ✅ PROMOTE — default-on in riir-engine | 2.0–2.3× |
| Standalone dequant, single-block (Plan 431 Phase 4) | ❌ DEMOTE — keep arithmetic | 0.575× |
| Standalone dequant, full-row (Plan 431 Phase 4) | ❌ DEMOTE — keep arithmetic | 0.286× |

### 3.2 MOAT gate per domain (§1.6)

| Domain | Bar | Verdict |
|---|---|---|
| `katgpt-rs` (public engine) | "Paper-derived fundamental / principle / base-foundation primitive that passes GOAT or Gain via fusion, with promote/demote tracked per stack" | ✅ IN SCOPE — quant-aware inference primitive, transformer-stack dequant slot. Promote/demote tracked against the existing arithmetic-cast path. |
| `riir-ai` | Pillar-level or Super-GOAT | N/A — perf optimization, not pillar-level |
| `riir-chain` / `riir-neuron-db` | Pillar-level or Super-GOAT | N/A — not a chain or shard concern |

Routing: **open primitive in katgpt-rs** (generic LUT dequant), **private tuning in riir-ai** (Q4_K integration, follow-up plan).

### 3.3 Modelless-first check (§3.5)

Not a training dependency. No gate to unblock. Pure inference-time primitive. No riir-train deferral.

### 3.4 Defend-wrong PoC check (§3.6)

Claims:
- **Architectural**: "the runtime analog exists" — proven by grep + read of `q4k.rs`, `channel_simd.rs`, `async_qdq.rs`. Sufficient.
- **Latency**: "LUT path is faster than arithmetic" — UNPROVEN, pending the GOAT gate benchmark. Explicitly marked unproven. Settles via criterion bench, not PoC.
- **Quality**: bit-exact by construction (LUT and arithmetic produce identical f32 outputs for the same INT input). No quality claim to defend.

§3.6 PoC does NOT trigger — quality is bit-exact by construction, latency claim is explicitly unproven and deferred to the GOAT gate, no "parity" claim against the paper's hardware numbers (we explicitly disclaim that — hardware wins are not replicable in software).

### 3.5 UQ-bearing primitive check ("Report the Floor" rule)

NOT UQ-bearing. Deterministic dequantization. No probability distribution, no predictive interval, no coverage guarantee. Conformal-naive floor does NOT apply.

---

## 4. What ships (Plan 431)

1. **Open primitive** in `katgpt-rs/crates/katgpt-core/src/` (or `katgpt-types/src/simd/`):
   - `QuantFormat` enum (sideband tag analog)
   - `dequant_via_lut::<F>(codes, lut, scale, zero, out)` generic function
   - `Int4Lut` (16 entries, 64 bytes), `Int8Lut` (256 entries, 1 KB), `Fp8Lut` (256 entries, 1 KB)
   - `simd_gather_lut` NEON/AVX2/scalar dispatch
   - Feature flag `simd_lut_dequant` (opt-in)

2. **GOAT gate** (the benchmark that decides promote-to-default):
   - **G1 correctness**: bit-exact match vs current arithmetic dequant on Q4_K blocks (must be 0.0 diff)
   - **G2 latency**: criterion bench on (a) single-block dequant, (b) full-row dequant, (c) fused dequant+dot — vs the current `dequantize_row_q4_k` arithmetic path. Target: ≥ 1.2× on aarch64 NEON. **If G2 fails: demote, keep feature opt-in, document.**
   - **G3 no-regression**: `--all-features` clippy + test clean
   - **G4 alloc-free**: zero allocations on the dequant hot path (LUT is stack-allocated or pre-built)
   - **G5 SIMD-level**: report NEON `vld1q_f32` gather throughput vs scalar
   - **G6 feature-isolation**: `--features simd_lut_dequant` builds clean without affecting default paths

3. **Conditional follow-up** (only if G2 PASS): `riir-ai/.plans/` integration to refactor `q4k.rs::dequantize_row_q4_k` onto the new primitive.

### 4.1 Honest expectation-setting

The paper's 7× speedup is a **hardware** number (DQBs eliminate CUDA-core instruction overhead and HBM write-back). In software SIMD, the win is bounded by:
- LUT gather latency vs CVT instruction latency (often comparable on modern CPUs)
- L1 cache pressure (LUT eats cache lines that arithmetic path doesn't)
- The dequant fraction of total inference time (paper's Fig 1: 40-80% on GPU; on CPU with SIMD fused dot, this is much smaller)

**Realistic target: 1.0×-1.5× on Q4_K dequant microbenchmarks, plausibly 0.9× (loss) if gather is slow.** The GOAT gate settles it honestly. We do NOT pre-claim a win.

**POST-GOAT (2026-07-13): the gate ran, and the prediction was an UNDERPROMISE on the fused slot.** Actual results exceeded the 1.0–1.5× target by 1.3–3×:
- Plan 431 Phase 4 raw primitive: **4.58× fused** (massively above target), **0.286× plain** (the predicted ~0.9× loss materialized, even worse than predicted — NEON gather is via scalar, not a single instruction).
- Plan 486 Phase 3 real Q4_K blocks: **2.0–2.3× fused CPU GEMV** (above target — per-block LUT rebuild cost ate ~half the raw win, but fusion + NEON `vfmaq_f32` FMA still dominates auto-vectorized arithmetic).

The honest lesson: the slot distinction (fused vs standalone) matters more than the technique. LUT wins when paired with FMA fusion; loses when standalone on gather-less ISAs. See `.benchmarks/432_simd_lut_dequant_goat.md` (katgpt-rs) and `riir-ai/.benchmarks/487_q4k_lut_gemv_goat.md` (riir-ai).

---

## TL;DR

StreamDQ is a hardware paper from SK Hynix (DQBs in HBM base dies, 7.08× mpGEMM speedup). The initial verdict (PASS, hardware-only) was **wrong** — we explicitly simulate hardware dequant techniques via software SIMD, and Research 110 documents this (Plasma = ternary SIMD, Cold = Q4_K dequant-on-read). The paper exposes three real gaps in our substrate: (1) no LUT-based INT→FP conversion, (2) no shared-FP32-ALU primitive generic over format, (3) no runtime sideband-tag dispatch. **Verdict: Gain — CONFIRMED by both GOAT gates (2026-07-13).** Plan 431 shipped the open primitive behind `simd_lut_dequant` (now default-on in katgpt-core); Plan 486 (riir-ai) wired it into a fused CPU Q4_K GEMV behind `simd_lut_q4k` (now default-on in riir-engine). Real numbers: 4.58× raw fused primitive, 2.0–2.3× on real Q4_K blocks (per-block LUT rebuild cost ate ~half the raw win). Standalone dequant stays arithmetic (0.286× on NEON — no native gather). Plan 431 Phase 6 (FP8/INT8/Q8_KV LUT + sideband dispatch) deferred `[-]`. Not a Super-GOAT (Q1-Q4 all NO: LUT INT→FP is known HPC, no new capability class, no moat). No private guide required. Not UQ-bearing (no conformal floor). Not a training dependency (no riir-train deferral).
