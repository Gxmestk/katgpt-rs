# Issue 578: Ternary tier with group-wise f16 scale (`Q2_0_g128` container)

**Date:** 2026-08-09
**Status:** LANDED, opt-in (2026-08-09). G1 partial / G2 PASS / G3 PASS / G4 PASS. Only AVX2 outstanding.
**Feature flag (proposed):** `ternary_group_scale` (opt-in; promotion needs the GOAT gate below)
**Filed by:** riir-train [Plan 333](../../riir-train/.plans/333_bitnet_ternary_moe_neuro_symbolic_poc.md) T3.1a
**Related:** [Issue 145](https://github.com/katopz/katgpt-rs) `binary_plasma` tier (got the scale
structure right, the alphabet wrong — see below), Plan 148 / Research 110 (`plasma_path` ternary SIMD)

> **Numbering note.** `.issues/.highwater` read `25` when this was filed, but git history shows 217
> issues allocated with a max of **577** (e.g. `577_general_sum_2player_cce_goat_closure.md`,
> `145_binary_plasma_tier_ternary_to_hot.md`). `25 + 1 = 26` would have recycled
> `026`, which history shows is already taken — the exact collision
> `121_monotonic_issue_numbering_recycled_collision.md` exists to prevent. This issue therefore
> takes **578** (`max(git history) + 1`) and `.highwater` is repaired to `578` in the same commit.
> **Do not trust the previous `.highwater` value; it was stale.**

---

## The gap (verified 2026-08-09)

[prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) ships as
GGUF **`Q2_0_g128`**: weights in `{-1, 0, +1}` stored in 2-bit slots, with **one f16 scale per 128
weights**. Neither shipped plasma tier can hold that:

| Tier | File | Alphabet | Scale granularity | Verdict |
|---|---|---|---|---|
| `TernaryWeights` | `crates/katgpt-types/src/ternary.rs:14` | ✅ `{-1, 0, +1}` | ❌ `row_scale: Vec<f32>` — one per **row** | wrong scale |
| `BinaryWeights` | `crates/katgpt-types/src/binary.rs:23` | ❌ `{-1, +1}`, no zero state | ✅ `group_scale: Vec<f16>`, `GROUP_SIZE = 128` | wrong alphabet |

**The obvious bridge does not work.** `BinaryWeights::from_ternary_no_zeros`
(`binary.rs:99`) returns `Option` and yields `None` the moment any weight is zero. Bonsai's zeros
are not incidental — ternary sparsity is the entire reason the model holds 94.6% of FP16 quality at
1.71 bits/weight. That constructor can never carry this model.

**Issue 145 came close.** Its own Cargo comment names the target verbatim — *"Binary `{−1,+1}`
plasma tier — single bit-plane, group-wise FP16 scale (Issue 145, Bonsai 27B)"*
(`crates/katgpt-core/Cargo.toml:141`) — and it landed `groups_per_row`, `group_scale: Vec<f16>`,
`GROUP_SIZE = 128`, and a `BNPLSMA1` ("Bonsai Plasma 1") loader. It got the **scale structure
exactly right** and dropped the zero state. This issue closes the remaining half.

---

## Required container shape

Mirror `BinaryWeights`' group-scale layout onto the ternary (two-bit-plane) alphabet:

```rust
/// Ternary {-1, 0, +1} bit-plane weights with group-wise f16 scale.
/// `group_scale[r * groups_per_row + g]` rescales each 128-weight group.
pub struct TernaryGroupWeights {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,       // cols.div_ceil(64)
    pub groups_per_row: usize, // cols.div_ceil(GROUP_SIZE)   // GROUP_SIZE = 128
    pub pos_bits: Vec<u64>,    // [rows * blocks64]   bit set -> +1
    pub neg_bits: Vec<u64>,    // [rows * blocks64]   bit set -> -1
    pub group_scale: Vec<f16>, // [rows * groups_per_row]
}
```

`pos_bits & neg_bits == 0` for every block is the representation invariant; a position clear in both
planes is the zero state.

**Footprint is identical to the GGUF deployed form — this is a repack, not an expansion.**
2 bit-planes = 2 bits/weight, plus 16 bits per 128 weights = **2.125 bits/weight**. At 27B params
that is 7.17 GB, matching Bonsai's documented ~7.2 GB deployed size (vs 5.9 GB "ideal" with native
2-bit slots). So the dequant path in T3.1b costs no extra memory over the file on disk.

### New tier vs. extending `TernaryWeights`

**Recommended: a new tier (as above).** Reasons:
1. Adding fields to `TernaryWeights` breaks its struct-literal construction in
   `crates/katgpt-transformer/src/contiguous.rs:317` (`load_ternary_bits`, the `CIOTBIT1` format,
   whose header carries `row_scale` and has no group-scale slot). The on-disk format would need a
   version bump.
2. Carrying both `row_scale` and `group_scale` on one struct creates an ambiguity — which wins, or
   do they multiply? — with no caller that wants both.
3. `binary_plasma` already established the house pattern: separate struct + separate kernel +
   separate loader magic + separate feature flag. A third tier is the consistent shape.

Cost: a third matvec kernel. Mitigate by generating it from the existing ternary kernel
(`crates/katgpt-types/src/simd/ternary.rs:527`) with the scale lookup moved from per-row to
per-group — the bit-plane popcount inner loop is unchanged.

### Work items

- [x] `TernaryGroupWeights` in `crates/katgpt-types/src/ternary_group.rs` behind
      `ternary_group_scale`: `new` / `set` / `get` / `scale_at` / `set_scale` / `checksum` /
      `encoded_bytes` / `invariant_holds` / `quantize_from_f32` (group-wise error feedback).
- [x] `ternary_group_matvec_scalar` + `simd_ternary_group_matvec` +
      `simd_ternary_group_matmul_batch` in `crates/katgpt-types/src/simd/ternary_group.rs`.
      **Scalar + NEON only — AVX2 deferred** (see Deviations).
- [x] Re-exported through `simd/mod.rs`, `katgpt-types/src/lib.rs`, and `katgpt-core` (feature
      forward `ternary_group_scale = ["katgpt-types/ternary_group_scale", "binary_plasma"]`).
- [x] Loader `load_ternary_group_bits` (`TGPLSMA1` magic) in
      `crates/katgpt-transformer/src/contiguous.rs`, following `load_binary_bits`' validation
      idiom + an added `pos & neg == 0` invariant check on load.
- [x] `TernaryGroupWeights::from_ternary` widening (broadcast `row_scale` across the row's
      groups) — always succeeds, unlike `from_ternary_no_zeros`.
- [ ] AVX2 kernel (deferred — see Deviations).
- [x] Perf bench for G2 + `CountingAllocator` harness for G4 —
      `crates/katgpt-types/tests/bench_578_ternary_group_goat.rs`.

---

## Implementation status (2026-08-09)

**Landed.** 9 unit tests in `simd/ternary_group.rs` + 2 loader tests in `contiguous.rs`, all
passing on aarch64/NEON:

| Test | Covers |
|---|---|
| `set_get_roundtrips_all_three_states` | all 3 states + overwrite clears the other plane |
| `group_geometry_matches_group_size` | `BLOCKS_PER_GROUP == 2`; 2.125 bits/weight footprint |
| `neon_matches_scalar_reference` | 6 shapes incl. ragged group / block / sub-4 tail, rel < 1e-6 |
| `zero_state_is_actually_skipped` | all-zero weights → exactly 0 at scale 3.0 |
| `per_group_scale_is_applied_not_per_row` | distinct per-group scales; a row-scale impl fails it |
| `group_scale_tracks_varying_magnitude_better_than_row_scale` | the format's reason to exist |
| `quantize_from_f32_holds_invariant_and_tracks_signs` | invariant + large magnitudes not zeroed |
| `batch_matches_per_row_calls` | batch == per-row, exactly |
| `widening_from_row_scale_ternary_preserves_result` | bit-planes verbatim, matvec within 1e-3 |
| `tgplsma1_roundtrips_through_the_loader` | wire format, ragged 300-col case, zero state survives |
| `loader_rejects_bad_magic_truncation_and_plane_overlap` | 3 corruption modes |

**Gate status — honest:**

| Gate | Status |
|---|---|
| G1 correctness | **PARTIAL.** Scalar/NEON parity, invariants, quantizer, loader, widening all pass. The **llama.cpp logit match (±1%) is NOT done** — it needs the real Bonsai tensor, so it stays riir-train Plan 333 T3.2. |
| G2 perf | **PASS.** Group-scale costs 1.29-1.31x the row-scale kernel across 512x512, 1024x1024, 512x5120 — under the 1.5x ceiling. See the table below. |
| G3 no-regression | **PASS.** `katgpt-types` 127 default tests unchanged, 155 with the feature; `katgpt-transformer` 21 default. Clean under default / `--no-default-features` / feature-on / feature-on-without-`plasma_path`. |
| G4 alloc-free | **PASS.** 0 allocs / 1000 calls (NEON matvec), 0 / 1000 (scalar path), 0 / 200 (sub-threshold batch), under a `CountingAllocator`. |

**Promotion to default-on is still NOT justified** — G1's load-bearing half (matching llama.cpp
on a real Bonsai tensor, Plan 333 T3.2) is untested, and x86_64 has no specialized kernel. Stays
opt-in.

### G2 measurements (M3 Max, aarch64/NEON, `--release`, median of 9 x 20 calls)

| Shape | group-scale ns | row-scale ns | dense f32 ns | vs row (gate ≤1.5x) | vs dense (info) |
|---|---|---|---|---|---|
| 512x512 | 41 500 | 31 719 | 14 467 | **1.31x** | 0.35x |
| 1024x1024 | 163 673 | 127 227 | 60 773 | **1.29x** | 0.37x |
| 512x5120 | 415 094 | 319 777 | 162 410 | **1.30x** | 0.39x |

The ~1.3x overhead is the extra `vmulq` per 4 lanes that folds the group scale into the sign
vector. Stable across a 10x range of shapes, so it is the arithmetic cost, not a cache artifact.

### Row-parallel kernel added 2026-08-10 (`simd_ternary_group_matvec_parallel`)

The G2 table above measures the **single-threaded** kernel, and until now that
was the only one this tier had: the sole rayon entry point,
`simd_ternary_group_matmul_batch`, parallelizes over **batch**, so at
`batch = 1` — every autoregressive decode step — there was no parallelism at
all. The dense tier has had `matmul_parallel` for exactly this all along.

Measured on real Ternary-Bonsai-27B geometry (riir-ai
[Bench 582](../../riir-ai/.benchmarks/582_ternary_bonsai_decode_throughput_preflight.md),
M3 Max, 16 threads):

| | s/token | tok/s | GMAC/s |
|---|---|---|---|
| serial | 3.995 | 0.25 | 6.41 |
| **row-parallel** | 0.554 | **1.80** | **46.22** |
| | | | **7.21× overall** |

The serial figure independently confirms this issue's own G2 prediction
(6.41 measured vs 6.3 GMAC/s).

GOAT: **G1** bit-identical to serial (rows are independent, so the
`par_chunks_mut` split is a pure partition — tested across ragged row counts
and a zeroed row); **G2** 7.21×, up to 7.77× on a 248320-row `lm_head`;
**G3** new function, serial path untouched; **G4** 0 allocations over 50 calls
(`katgpt-core/tests/ternary_group_parallel_alloc_check.rs`) — which is what let
riir-engine's `forward_ternary` adopt it without breaking its own
0-alloc-per-token gate.

Below `PARALLEL_ROW_MIN = 256` it delegates to serial. Bonsai's
`ssm_alpha`/`ssm_beta` are 48 rows and measure 1.01×, so the threshold is
exercised by the real model rather than only by tests.

**`vs dense` reproduces Benchmark 044's finding independently:** both ternary tiers run at
0.35-0.45x dense f32 NEON. Timings are wall-clock on a working developer machine — treat <15%
as noise; the 2.6-2.9x dense advantage is far outside that.

### Format VERIFIED against the real file + the PrismML fork source (2026-08-09)

Read from `PrismML-Eng/llama.cpp` @ `9ca265a` and from the real
`Ternary-Bonsai-27B-Q2_0.gguf` (7,165,121,600 B) via HTTP range requests.

**The container geometry is confirmed exactly.** The fork's on-disk block is:

```c
#define QK2_0 128
typedef struct {
    ggml_half d;            // f16 scale
    uint8_t   qs[QK2_0/4];  // 2 bits/weight -> 32 B
} block_q2_0;               // 34 B per 128 weights = 2.125 bits/weight
```

That is precisely `TernaryGroupWeights`' geometry (`groups_per_row = cols/128`,
`group_scale: Vec<f16>`), and 7.165 GB / (2.125/8) = **26.97B params** — the derivation in this
issue matched the real artifact before either was seen.

**Symmetry with Issue 145 confirmed:** the fork also defines `block_q1_0` = `f16 + qs[128/8]` =
18 B per 128 = **1.125 bits/weight**, which is exactly `BinaryWeights`. The two shipped plasma
tiers map 1:1 onto the fork's two custom types — Issue 145 modelled `Q1_0`, this issue models
`Q2_0`. Type IDs: `GGML_TYPE_Q1_0 = 41`, `GGML_TYPE_Q2_0 = 42`, `GGML_TYPE_COUNT = 43`.

**Decode mapping for T3.1b** (`dequantize_row_q2_0`), 2 bits LSB-first within each byte:

```c
q = (qs[j/4] >> ((j%4)*2)) & 0x03;   y[j] = ((int)q - 1) * d;
```

#### ⚠️ The format has a FOURTH state that two bit-planes cannot represent

`11` decodes to **+2**, not +1. `TernaryGroupWeights` stores `pos`/`neg` bit-planes and can
represent exactly `{-1, 0, +1}` — **a `+2` weight cannot be held.** Two facts make this safe for
this checkpoint but not unconditionally:

1. **Unreachable via the reference encoder.** `quantize_row_q2_0_ref` sets `d = amax` (block
   max-abs), so `w/d ∈ [-1,+1]` and `round(w*id)+1 ∈ {0,1,2}`. The `if (q > 3) q = 3` clamp is
   dead code. Any encoder choosing `d < amax` (e.g. an error-minimising fit) *would* reach it.
2. **Measured absent in the real weights.** Scanning 30,720,000 weights across two structurally
   different tensors:

| Tensor | `-1` | `0` | `+1` | **`+2`** |
|---|---|---|---|---|
| `blk.32.ffn_up.weight` | 35.183% | 29.788% | 35.029% | **0** |
| `output.weight` | 35.053% | 30.066% | 34.881% | **0** |
| **total (30.72M)** | 35.118% | 29.927% | 34.955% | **0 (0.000%)** |

**Verdict: the container is sufficient for Ternary-Bonsai-27B-Q2_0.** But T3.1b's parser MUST
reject code `11` loudly rather than silently folding it to `+1` — a future Prism-ML checkpoint, or
the `Q2_g64` / `PQ2_0` variants in the same repo, may use it. Added as a T3.1b requirement.

**Also note `Ternary-Bonsai-27B-Q2_g64.gguf` exists in the same repo** — group size **64**, not
128. `GROUP_SIZE` is a compile-time constant shared with `binary_plasma`; supporting g64 would
need it parameterised. Out of scope here, recorded so it is not discovered late.

### Bug found while writing the G4 harness (fixed here)

`simd_ternary_matmul_batch` was **broken for every `batch >= 2`** — both its sub-threshold loop
and its rayon path passed an open-ended `&x[x_off..]` into a matvec that asserts
`x.len() == w.cols`, panicking on the length check. `batch == 1` happened to work (offset 0, exact
length), which is why it survived since Plan 148. `simd_binary_matmul_batch` had the same bug in
its sub-threshold loop only; its rayon path already used the correct `zip(par_chunks)` form.

Fixed in all three tiers (exact slicing + `zip(par_chunks)`), with
`regression_sub_threshold_batch_does_not_panic` covering `batch` 1..=5 across the
`PARALLEL_BATCH_MIN = 4` boundary and asserting every slot equals the single-vector call.

### Deviations from the spec above

1. **AVX2 not implemented.** `simd_ternary_group_matvec` falls back to scalar on x86_64. The
   development machine is aarch64 (M3 Max), so a hand-written AVX2 intrinsics kernel could not be
   executed or validated here — an unvalidated AVX2 kernel is worse than a correct scalar one.
   The dispatcher has the arm shape ready for it.
2. **Scalar and NEON are close, not bit-identical** (~1e-6 relative). They associate the scale
   differently: scalar applies it once per group (matching how `Q2_0_g128` is defined and how
   llama.cpp dequantizes), NEON folds it per element so the 4 accumulators can span the whole row
   without per-group horizontal sums — the same trick Issue 145's binary kernel uses. Documented
   at the top of `simd/ternary_group.rs`.
3. **`ternary_group_scale` implies `binary_plasma`**, to reuse `GROUP_SIZE` as a single source of
   truth for the 128-weight group rather than declaring a second constant.

---

## Consumers blocked on this

**riir-train [Plan 333](../../riir-train/.plans/333_bitnet_ternary_moe_neuro_symbolic_poc.md) —
Phase 2 and Phase 3 both stall without it:**

| Blocked task | What it needs |
|---|---|
| **T3.1b** | Add the `Q2_0_g128` arm to `riir-ai/crates/riir-engine/src/gguf_loader.rs` — `GgmlType` enum (`:43`), `from_id` (`:66`), `tensor_bytes` (`:99`), dequant dispatch (`:400`–`:487`). **It has nothing to dequantize into until this container exists.** |
| **T3.1c** | Header validation mirroring `load_binary_bits`. Depends on the container's field set. |
| **T2.2** | `BitLinear` forward. Can proceed on `TernaryWeights` for a row-scale approximation, but cannot reproduce llama.cpp logits to the ±1% T3.2 tolerance without per-group scale. |
| **T3.3a/b** | LoRA-on-ternary forward — the frozen base is this container. |

Phase 1 (external PoC via the PrismML llama.cpp fork) is **not** blocked; it runs entirely outside
our stack. This issue gates the native-Rust port only.

---

## GOAT gate (promotion to default requires all four)

- **G1 correctness:** round-trip `set`/`get` over all three states; `pos_bits & neg_bits == 0`
  invariant held; group-scale matvec matches a scalar reference to 1e-6; loaded Bonsai tensor
  reproduces llama.cpp's logits within ±1% (this is Plan 333 T3.2).
- **G2 perf (RESPECIFIED 2026-08-09 — the original clause was wrong on arrival):**
  `simd_ternary_group_matvec` ≤ **1.5×** the row-scale kernel at equal dims. That overhead ceiling
  is the only throughput claim this tier controls.
  **The original "≥ 2× the dense f16 matvec on CPU" clause is struck.** It was copied from Plan
  333's G-PERF, which misreads [Benchmark 044](../.benchmarks/044_plasma_path_goat.md): that
  benchmark measured the row-scale ternary kernel at **16.12 Gop/s = 0.45× FP32 NEON's 36 Gop/s**
  and documented the gap as *fundamental* ("bit-decoding SWAR has higher opcode count than pure
  load+FMA"). Ternary is **slower** than dense f32 on NEON and always has been; its win is memory
  traffic. No group-scale kernel can satisfy a gate the tier it extends already fails.
- **G3 no-regression:** `--no-default-features`, default, and `--all-features` all clean; existing
  `plasma_path` / `binary_plasma` tests unchanged.
- **G4 alloc-free:** 0 allocations per call in steady state (`CountingAllocator`, matching the
  shipped kernels' contract of writing into a caller-owned `&mut [f32]`).

**Modelless by construction** — a container plus a closed-form kernel, no training, so the
"promotion requires modelless gain" rule is satisfied if G1–G4 pass.

---

## Non-goals

- **Native 2-bit-slot storage.** We repack into bit-planes because that is what the SIMD popcount
  kernel consumes. Closing the 2.125 → 1.71 bits/weight gap is a separate optimization.
- **Changing `TernaryWeights` or the `CIOTBIT1` format.** Additive tier only.
- **Training on this container.** Frozen base; the LoRA delta is riir-train's concern (Plan 333 T3.3).
