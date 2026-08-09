# Issue 578: Ternary tier with group-wise f16 scale (`Q2_0_g128` container)

**Date:** 2026-08-09
**Status:** OPEN — blocking riir-train Plan 333 Phase 2/3
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

- [ ] `TernaryGroupWeights` in `crates/katgpt-types/src/ternary.rs` (or a sibling module) behind
      `ternary_group_scale`, with `new` / `set` / `get` / `checksum` mirroring the existing tiers.
- [ ] `simd_ternary_group_matvec` + `simd_ternary_group_matmul_batch` in
      `crates/katgpt-types/src/simd/ternary.rs`, NEON + AVX2 + scalar, signature matching the
      shipped kernels: `(&TernaryGroupWeights, &[f32], &mut [f32])`.
- [ ] Re-export through `crates/katgpt-types/src/simd/mod.rs` under the matching `#[cfg(feature)]`,
      and through `katgpt-core` (feature forward, same shape as `plasma_path` / `binary_plasma`).
- [ ] Loader: follow `load_binary_bits`' validation idiom (`contiguous.rs:338`) — magic → dims →
      `blocks64` / `groups_per_row` cross-check → length check → bulk `copy_nonoverlapping`.
- [ ] `TernaryWeights → TernaryGroupWeights` widening (broadcast `row_scale` across that row's
      groups) — the lossless direction, unlike `from_ternary_no_zeros`.

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
- **G2 perf:** `simd_ternary_group_matvec` ≥ the row-scale kernel's throughput at equal dims (the
  per-group scale lookup must not cost more than the popcount it rides along with), and ≥ 2× the
  dense f16 matvec on CPU (Plan 333's G-PERF).
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
