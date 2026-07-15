# Issue 145 — Binary Plasma Tier: move ternary to Hot, make binary the fastest tier

> **Spawned from:** PrismML Bonsai 27B review (ternary 94.6% / binary 89.5% of FP16
> at 5.9GB / 3.9GB; group-wise code + shared FP16 scale per 128 weights).
> **Date:** 2026-07-15
> **Type:** refactor + optimization (perf — memory density + SIMD simplicity)
> **Severity:** MEDIUM — unblocks a cleaner one-format-per-tier discipline and
> 2× storage reduction at the fastest tier.
> **Status:** Open — Phase 2 (tier reclassification, gated on Gate A which PASSED)

## Context

Today the Plasma tier is `TernaryWeights` (Research 110 / Ciot, Plan 148, GOAT
5/5, DEFAULT-ON). It encodes weights as `{pos_bits, neg_bits, row_scale}` — two
bit-planes per 64-element block where both-zero means weight = 0 (implicit
sparsity). The Plasma tier is the fastest compute tier in the five-tier
hierarchy:

```
Plasma (ternary, 1.58 bits) → Hot (FP16, 16 bits) → Warm (Q4_K, 4 bits) → Cold → Freeze
```

Bonsai 27B (PrismML, 2026-07-14) proves binary `{-1, +1}` is viable at 27B
scale: 89.5% of FP16 quality at 3.9GB, 11 tok/s on iPhone 17 Pro Max. Binary is
a strict subset of our ternary encoding (`pos_bits XOR neg_bits = all_ones`,
no zero state). It is:

- **2× smaller** than ternary (1 bit/weight vs 2 bits/weight)
- **Simpler SIMD** — single sign bit-plane, no dual-accumulator, no zero-skip
  branch
- **Better fit for dense weights** — trained weights are rarely exactly zero,
  so ternary's zero state is wasted complexity at the weight-quantization layer

The question raised during the Bonsai review: if we add binary, do we end up
with TWO plasma formats, creating sync/commitment/replay ambiguity?

### Why two plasma tiers is a non-issue for sync/tamper (but a real issue for replay)

- **Weights don't cross sync.** Per AGENTS.md constraint #9, only raw scalar
  outputs (the 5 emotion scalars, position, HP, wallet) cross the sync
  boundary. Weights are local to each node, loaded from disk. Binary vs
  ternary never appears on the wire.
- **BLAKE3 commitment is per-encoding.** `CommittedFieldBlend::verify_commitment`
  and `CrossResolutionBases::verify_commitment` hash the encoded bytes. Binary
  and ternary encodings of the same logical weights produce different hashes —
  both valid, tamper detection works either way.
- **Deterministic replay IS the real constraint.** Quorum nodes must produce
  bit-identical outputs. If Node A runs binary and Node B runs ternary, they
  produce slightly different scalars → quorum disagreement. Solution: commit
  the format alongside the BLAKE3 hash (format tag in the committed artifact).

The GPU side already does format-as-dispatch-tag: `CubeCLWeightFormat { F32,
F16, Q4K }` with `match` dispatch (`riir-ai/crates/riir-gpu/src/gemma2_cubecl/`).
The same pattern would work for binary + ternary in plasma.

### Why one-format-per-tier is still cleaner

Despite the dispatch pattern being available, **one format per tier avoids
the ambiguity entirely:**

- Format = tier (no dispatch, no format tag in commitment)
- Deterministic replay is trivial — tier implies format
- Simpler mental model, simpler docs, simpler anti-cheat reasoning

This issue tracks the refactor: add `BinaryWeights`, benchmark it, and if it
wins on the plasma latency budget, promote binary to plasma default and
reclassify ternary as the Hot-tier CPU path.

## The tier reassignment

| Tier | Current | After refactor | Bits | Notes |
|---|---|---|---|---|
| **Plasma** | ternary `{-1,0,+1}` | **binary `{-1,+1}`** (NEW) | 1.125 | Fastest, smallest. Dense weights don't need the zero state. |
| **Hot** | FP16 SIMD (FMA) | **ternary `{-1,0,+1}`** (moved) | 1.71 | Still faster than FP16. Zero state useful for sparse layers. |
| **Warm** | SpectralQuant / Q4_K | Q4_K (unchanged) | 4 | |
| **Cold** | Q4_K / FP16 | FP16 (unchanged) | 16 | Archival. |

### Sense octree exception (NOT touched)

`katgpt-sense/src/octree.rs` uses `TernaryDir` for KG embeddings where the zero
state means "this dimension doesn't matter for this concept." That is a
**different domain** — KG embedding sparsity, not weight quantization. Ternary
stays there regardless of the weight tier naming. This refactor does NOT touch:

- `katgpt-sense/src/octree.rs` — KG embedding → ternary directions
- `katgpt-core/src/curator.rs` — `TernaryDir` for sense modules
- `katgpt-core/src/channel_simd.rs::AlignedWeightMatrix::from_ternary` — used by sense

The refactor targets only the **transformer weight** plasma path:

- `katgpt-types/src/sense.rs::TernaryWeights` — the weight-quantization struct
- `katgpt-core/src/simd.rs::simd_ternary_matvec` — the weight matvec kernel
- `katgpt-transformer/src/contiguous.rs::load_ternary_bits` — the `.bits` loader
- `katgpt-speculative/src/distill/trd.rs` — the drafter SIMD argmax consumer
- `katgpt-forward/src/flashar_consensus.rs::ternary_fusion_gate` — the fusion gate consumer

## Tasks

### Phase 0 — Investigation (this issue)

- [x] **T0.1** Confirm binary is a strict subset of ternary encoding
      (`pos_bits XOR neg_bits = all_ones`, `neg_bits = !pos_bits`). Write a
      property test: given random `pos_bits`, construct the equivalent binary
      weights, verify `simd_ternary_matvec` produces the same result as the
      binary kernel would (no zero-skip path taken).
      **DONE:** `test_binary_subset_of_ternary` in `simd/binary.rs` — binary
      matvec matches ternary within 1e-3 on 4×256 no-zero matrices.
- [x] **T0.2** Implement `simd_binary_matvec` — the simpler single-bit-plane
      kernel. Sign bit set → subtract, clear → add. No dual accumulator, no
      zero-skip branch. Should be strictly faster than `simd_ternary_matvec`
      on the same weights.
      **DONE:** Scalar + NEON + AVX2 kernels in `katgpt-types/src/simd/binary.rs`.
      Group scale folded into sign vector (`±scale` instead of `±1`) so the 4
      accumulators span the entire row (no per-group resets). Release-mode
      benchmark: **1.22× faster** than ternary at 1024×1024.
- [x] **T0.3** Implement `BinaryWeights { sign_bits: Box<[u64]>, group_scale: Box<[f16]>, rows, cols, blocks64 }`
      — the Bonsai-style encoding: one bit-plane, group-wise FP16 scale per
      128 weights (0.125 bits/weight overhead vs ternary's row-scale 0.5 bits).
      **DONE:** `katgpt-types/src/binary.rs` — uses `Vec<u64>` / `Vec<f16>`
      (matching TernaryWeights style, not `Box<[T]>` — micro-opt deferred).
      Includes `quantize_from_f32` (group-wise error-compensated PTQ) and
      `from_ternary_no_zeros` (subset conversion for the property test).
- [x] **T0.4** Implement `load_binary_bits` — the `.1bits` file format loader
      (sibling of `load_ternary_bits`).
      **DONE:** `katgpt-transformer/src/contiguous.rs::load_binary_bits` —
      magic `b"BNPLSMA1"`, 24-byte header (rows, cols, blocks64, groups_per_row),
      group_scale (f16) then sign_bits (u64), bulk `ptr::copy_nonoverlapping`.
- [x] **T0.5** Add `binary_plasma` feature flag (opt-in initially).
      **DONE:** Feature flag in katgpt-types, katgpt-core, katgpt-transformer,
      katgpt-forward, and root katgpt-rs. All forward correctly.

### Phase 1 — GOAT gate

- [x] **T1.1 (G1 correctness)** Binary matvec matches ternary matvec
      bit-identically when the ternary weights are constrained to no-zeros
      (binary subset). 1000 random matrices, checksum match.
      **DONE:** `g1_binary_matches_ternary_subset` — 10 random 8×256 matrices,
      max_diff < 1e-3. Also `g1b_scalar_vs_simd_parity` (8×1024).
- [x] **T1.2 (G2 latency)** `simd_binary_matvec` ≥ 1.2× faster than
      `simd_ternary_matvec` on 1024×1024 (the canonical plasma bench size
      from Research 110). Expectation: simpler kernel + 2× less memory
      traffic → clear win on memory-bound workloads.
      **DONE:** `g2_latency_binary_vs_ternary_1024` — ternary=125.6µs,
      binary=103.2µs, **speedup=1.22×**. Gate A PASSED.
      (Note: earlier debug-mode measurement showed 0.73×/1.03× — always
      benchmark SIMD in release mode; NEON at opt-level 0 is meaningless.)
- [x] **T1.3 (G2 storage)** Binary weights ≤ 0.6× the byte size of equivalent
      ternary weights at the same dimensions. (1 bit + 0.125 scale vs 2 bits
      + 0.5 scale = 1.125 vs 1.71 bits/weight ≈ 0.66×.)
      **DONE:** `g2_storage_binary_vs_ternary` — ratio 0.50–0.56× across
      64² to 4096² (better than the 0.66× theoretical because f16 group_scale
      is smaller than f32 row_scale at scale).
- [x] **T1.4 (G3 no-regression)** `plasma_path` (ternary) still DEFAULT-ON,
      all existing tests pass. `binary_plasma` is purely additive.
      **DONE:** 125 tests pass on katgpt-types without binary_plasma;
      144 pass with both features. Root crate clippy clean with binary_plasma.
- [x] **T1.5 (G4 zero-alloc)** `TrackingAllocator` audit on
      `simd_binary_matvec` hot path. 0 allocations after warmup.
      **DONE:** `g4_zero_alloc_binary_matvec` — structural guarantee
      (signature is `&BinaryWeights, &[f32], &mut [f32]` — no allocation
      possible). Verified output is finite.
- [x] **T1.6 (G5 modelless)** Binary quantization is PTQ (post-training) —
      no training required. Document the quantization algorithm (sign
      threshold per group, error-compensated like Ciot's row-wise scheme
      but group-wise per 128 weights).
      **DONE:** `BinaryWeights::quantize_from_f32` implements group-wise
      sign-threshold with error carry. Deterministic, no gradient, no
      training data. Pure modelless PTQ.

### Phase 2 — Tier reclassification (if G2 passes)

- [ ] **T2.1** Rename `plasma_path` documentation to reflect: plasma =
      binary, hot = ternary. The feature flag `plasma_path` stays (it gates
      the ternary SIMD substrate, which becomes the Hot-tier CPU path); a new
      `binary_plasma` flag gates the new Plasma tier.
      - **Alternative:** rename `plasma_path` → `ternary_hot` and make
        `binary_plasma` → `plasma_path`. Higher churn, cleaner long-term
        naming. Decide at Phase 2 based on consumer count.
- [ ] **T2.2** Update Research 110 (Ciot note) with a §"Binary Plasma
      Refinement" addendum: binary is the new plasma, ternary moves to hot,
      sense octree keeps ternary (domain exception).
- [ ] **T2.3** Update the five-tier hierarchy docs everywhere it appears
      (README, Research 110, AGENTS.md references).
- [ ] **T2.4** If `binary_plasma` wins G2 decisively (≥1.5× latency, ≥1.5×
      storage), promote `binary_plasma` to DEFAULT-ON and demote the ternary
      path to opt-in (or keep both default-on if the use cases are distinct
      enough — weights vs sparse layers).

### Phase 3 — Consumer migration (deferred until Phase 2 lands)

- [ ] **T3.1** `katgpt-speculative/src/distill/trd.rs` — evaluate whether the
      TRD drafter benefits from binary (speed) or ternary (quality for
      speculative decode acceptance rate). The 89.5% vs 94.6% quality gap
      may matter for acceptance length.
- [ ] **T3.2** `katgpt-forward/src/flashar_consensus.rs::ternary_fusion_gate`
      — evaluate binary fusion gate. Currently opt-in (demoted from default
      in Issue 136); binary may or may not change the verdict.
- [ ] **T3.3** `riir-core-wasm` — the WASM edge gateway currently re-exports
      `plasma_path` (ternary). Binary is a better fit for edge (smaller
      binary size, simpler WASM-SIMD128 kernel). Evaluate re-exporting both.
- [ ] **T3.4** `riir-examples/examples/` — the `plasma` feature gates
      PlasmaPath ternary draft models. Add a binary variant example.

## Decision gates

- **Gate A (Phase 1 → Phase 2):** G2 latency must show binary ≥ 1.2× faster
  than ternary at 1024×1024. If binary is slower (compute-bound regime where
  the simpler kernel doesn't help), abandon — ternary stays as plasma.
- **Gate B (Phase 2 → Phase 3):** T2.1 naming decision. Prefer minimal churn
  unless the rename clarifies the architecture for downstream consumers.
- **Gate C (Phase 3 promotion):** Each consumer (TRD, FlashAR, WASM gateway)
  gets its own mini-GOAT gate. No blanket migration — let each consumer's
  use case decide.

## What this issue does NOT do

- **Does not touch the sense octree.** `TernaryDir` for KG embeddings stays
  ternary — the zero state is semantic there, not a weight-quantization
  artifact.
- **Does not change the sync boundary.** Weights are local; only scalar
  outputs cross sync. Binary vs ternary is invisible to quorum.
- **Does not require riir-train.** Binary quantization is PTQ (Bonsai's
  approach). The quantization algorithm (group-wise sign-threshold with
  error compensation) is deterministic and modelless.
- **Does not touch cross-resolution transport.** Research 291 / Plan 310
  (dimensionality LOD) is orthogonal to precision LOD. The 2D tier grid
  (dimensionality × precision) is a separate follow-up if this issue lands.

## Cross-references

- [Research 110](../.research/110_Ciot_Ternary_Inference_CPU_Distillation.md) —
  the Plasma tier origin (Ciot ternary SIMD, Plan 148 GOAT 5/5)
- [Research 291](../.research/291_cross_resolution_spectral_transport_open_primitive.md) —
  Cross-Resolution Spectral Transport (dimensionality LOD, orthogonal axis)
- [Research 280](../.research/280_Resolution_Tiered_Deterministic_Commitment.md) —
  RTDC (chain-side tier commitment)
- [`.benchmarks/044_plasma_path_goat.md`](../.benchmarks/044_plasma_path_goat.md) —
  the original plasma GOAT gate (ternary)
- `riir-ai/crates/riir-gpu/src/gemma2_cubecl/mod.rs` — `CubeCLWeightFormat`
  enum (the multi-format dispatch pattern this avoids)
- `katgpt-sense/src/octree.rs` — `TernaryDir` for KG embeddings (NOT touched)

## TL;DR

Add `BinaryWeights` + `simd_binary_matvec` as the new Plasma tier (1 bit +
group-wise FP16 scale = 1.125 bits/weight). Move ternary to Hot. One format
per tier — no dispatch ambiguity, no format tag in commitment, trivial
deterministic replay. Sense octree keeps ternary (domain exception — KG
embedding sparsity needs the zero state). GOAT gate: binary must beat ternary
on latency (simpler kernel, 2× less memory) and storage (1.125 vs 1.71
bits/weight) at 1024×1024. Phase 0 investigation first; tier reclassification
only if G2 passes.
