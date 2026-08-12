# Issue 582: Trit-packed ternary tier — close the 2.125 → 1.725 bits/weight gap

**Date:** 2026-08-12
**Status:** **LANDED, opt-in (2026-08-12). G1 / G2 / G2b / G3 / G4 ALL PASS** — see
[Bench 582](../.benchmarks/582_trit_pack_goat.md). The perf axis came out a **gain,
not a trade**: **1.10–1.15× faster** than the bit-plane tier *and* 18.8% smaller.
(1.22–1.28× against the pre-Issue-583 baseline; that kernel then got 11% faster,
leaving the decode-only margin the attribution predicted.)
**Feature flag:** `ternary_trit_pack` (opt-in)
**Filed by:** the Issue 578 closure review (see
[`.docs/08_performance/ternary_group_q2_0_tier.md`](../.docs/08_performance/ternary_group_q2_0_tier.md)
§Non-goals — this is the named "plane interleaving / native slot storage" follow-up)
**Related:** Issue 578 (closed) `TernaryGroupWeights`, Issue 145 `binary_plasma`,
Bench 044 `plasma_path`

---

## The gap

`TernaryGroupWeights` stores two 1-bit planes = **2 bits/weight**, +16 bits per
128-weight group = **2.125 bits/weight**. At 27B params that is **7.17 GB**.

But a trit carries only `log2(3) = 1.585` bits. Two bit-planes waste the fourth
codepoint (`pos & neg`, forbidden by the invariant) — the same waste GGUF's
`Q2_0` has, where code `11` is unreachable in practice. Base-3 packing recovers
it: **5 trits fit in one byte** (`3^5 = 243 ≤ 256`).

| Layout | per 128-weight group | bits/weight | Bonsai-27B |
|---|---|---|---|
| two bit-planes (`TernaryGroupWeights`) | 32 B planes + 2 B scale = **34 B** | 2.125 | **7.17 GB** |
| `Q2_0` 2-bit slots (GGUF on disk) | 32 B codes + 2 B scale = **34 B** | 2.125 | 7.17 GB |
| **5-trits-per-byte + f16 scale** | 25.6 B trits + 2 B scale = **27.6 B** | **1.725** | **5.82 GB** |

**−18.8% footprint, and it beats the "~5.9 GB ideal" figure Issue 578 recorded**
as unreachable with 2-bit slots. (Filed as −17.6% / 1.75 bpw; that estimate
wrongly rounded each 128-weight group up to a whole 26 bytes. Rows pack
contiguously, so only the *row* tail rounds — measured 1.725 bpw.) Ternary GEMV is memory-bound on
every backend we ship (Metal GEMV sits at 45% of a 400 GB/s roofline; the 4090
dp4a kernel at 88.9% of HBM peak — both bandwidth-limited, not ALU-limited), so
an 18.8% traffic cut is the one remaining lever that is neither a kernel
micro-optimization nor a retraining dependency.

**Why this is not just a footprint story.** Two adjacent facts from the Issue 578
closure:

- riir-ai measured the remaining Metal GEMV gap vs llama.cpp as *architectural*:
  "2-array pos+neg vs llama.cpp's 1-array 2-bit codes" — two load streams where
  they have one.
- Bonsai-27B at 7.17 GB + KV cache is tight on a 24 GB 4090 when anything else
  is resident, and painful on 16 GB unified memory.

A single-array layout addresses both. Base-3 addresses both *and* takes 18.8%
off the traffic.

## Container shape

```rust
/// 5 trits per byte, base-3. `trits[i]` holds weights 5i..5i+5 as
/// `t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4` where each `t ∈ {0,1,2}` maps to
/// `{-1, 0, +1}` via `t - 1`.
pub struct TernaryTritWeights {
    pub rows: usize,
    pub cols: usize,
    pub bytes_per_row: usize,  // cols.div_ceil(TRITS_PER_BYTE)
    pub groups_per_row: usize, // cols.div_ceil(GROUP_SIZE)
    pub trits: Vec<u8>,        // [rows * bytes_per_row]
    pub group_scale: Vec<f16>, // [rows * groups_per_row]
}
```

**Alignment hazard (the reason this is not a trivial repack).** `GROUP_SIZE` is
128 and 128 is **not** divisible by 5 — so unlike the bit-plane tier, where a
group is exactly two whole `u64` blocks and a group boundary never splits a
word, **here a group boundary lands mid-byte**: group 0 covers weights 0..128,
i.e. bytes 0..25 plus 3 trits of byte 25. Every kernel and the quantizer must
handle a group whose first and last byte are shared with the neighbouring group.
This is the core implementation risk and the reason the decode is
group-scoped-with-carry rather than byte-parallel.

## Decode strategy

A 256-entry LUT `TRIT_LUT: [[i8; 8]; 256]` maps each byte to its 5 signed values
(padded to 8 for an aligned 8-byte copy; the last 3 lanes are zero). 2 KB, L1
resident, `const`-built.

Hot path per row: for each group, decode the group's byte span into a
stack-allocated `[i8; GROUP_SIZE + 8]` scratch (≤ 26 aligned 8-byte writes),
then run the existing `sign × x` accumulate over the scratch. Costs one decode
pass per group that the bit-plane kernel does not pay; buys 18.8% fewer bytes
loaded. **Which side wins on CPU is an open empirical question** — see G2.
(Answer, measured: the trit tier wins on *both* axes. See Measured outcome.)

Invalid bytes (`>= 243`) are a corruption signal, not a fourth state — the
loader must reject them.

## Work items

- [x] `TernaryTritWeights` in `crates/katgpt-types/src/ternary_trit.rs` behind
      `ternary_trit_pack`: `new` / `set` / `get` / `scale_at` / `set_scale` /
      `checksum` / `encoded_bytes` / `is_canonical` (all bytes < 243) /
      `from_group` (lossless repack from `TernaryGroupWeights`) / `to_group`.
- [x] `const TRIT_LUT` + `ternary_trit_matvec_scalar` +
      `simd_ternary_trit_matvec` (NEON) in
      `crates/katgpt-types/src/simd/ternary_trit.rs`.
- [x] Re-export through `simd/mod.rs` + `katgpt-types/src/lib.rs` + `katgpt-core`
      (feature forward `ternary_trit_pack = ["katgpt-types/ternary_trit_pack",
      "ternary_group_scale"]`).
- [x] GOAT bench `crates/katgpt-types/tests/bench_582_trit_pack_goat.rs` (4 gate
      tests) + 10 unit tests in `simd/ternary_trit.rs`.
- [x] Honest verdict in [`.benchmarks/582_trit_pack_goat.md`](../.benchmarks/582_trit_pack_goat.md)
      — including the wrong prediction, the attribution split, and the
      indexing bug the cross-tier gate caught.
- [x] Tier doc + opt-in feature catalog updated.

## GOAT gate

- **G1 correctness:** `from_group` → `to_group` round-trips bit-identically over
  all three states incl. ragged tails (`cols` ≢ 0 mod 5, mod 128); matvec
  matches `ternary_group_matvec_scalar` to ≤ 1e-6 relative on ≥ 6 shapes
  including a group boundary that splits a byte; every canonical byte < 243.
- **G2 footprint (the load-bearing gate, deterministic):**
  `encoded_bytes() ≤ 0.83 ×` the bit-plane tier's at equal dims — i.e. the
  −18.8% is actually realized, not eaten by padding. This gate cannot be noisy;
  it is arithmetic.
- **G2b perf (informational, honest):** measure `simd_ternary_trit_matvec` vs
  `simd_ternary_group_matvec` at 512×512 / 1024×1024 / 512×5120. **No pass
  threshold is set on CPU** — the decode pass may well lose, and a footprint
  tier is still shippable when it does (`binary_plasma` ships opt-in on exactly
  that logic: storage PASS, latency FAIL). Record the ratio either way; a
  *large* loss (> 2×) is a design reject, since the whole point is that memory
  traffic dominates.
- **G3 no-regression:** default / `--no-default-features` / feature-on /
  `--all-features` clean; `ternary_group_scale` tests unchanged.
- **G4 alloc-free:** 0 allocations per matvec call under `CountingAllocator` —
  the decode scratch must be a stack array, not a `Vec`.

**Modelless by construction** — a container plus a closed-form LUT decode. No
training, so promotion turns purely on G1–G4.

## Explicit non-goals

- **Promotion to default.** Same reasoning as Issue 578: it would transitively
  promote `binary_plasma`, and a footprint tier is opt-in by nature.
- **The Metal single-load-stream win.** Only riir-ai can measure the ~10% Metal
  GEMV claim (their kernel, their hardware harness). This issue ships the
  container + CPU evidence; the GPU leg is filed separately in riir-ai.
- **Changing `TGPLSMA1` or `TernaryGroupWeights`.** Additive tier, as with 578.
- **A GGUF loader arm.** Bonsai ships as `Q2_0`, not trit-packed; this tier is
  reached via `from_group` after the existing load, so there is nothing to parse
  until an upstream producer emits base-3.

---

## Measured outcome (2026-08-12)

| Gate | Result |
|---|---|
| G1 | **PASS** — scalar bit-identical to `ternary_group_matvec_scalar` (exact `assert_eq!`, not a tolerance); `from_group`→`to_group` lossless incl. ragged bytes/groups; `quantize_from_f32` agrees weight-for-weight and scale-for-scale; NEON within 1e-6 |
| G2 footprint | **PASS** — ratio 0.8118–0.8162 (gate ≤ 0.83); 1.725 bits/weight; Bonsai-27B 5.82 GB vs 7.16 GB |
| G2b latency | **PASS as a gain** — 0.87–0.91× the bit-plane kernel (1.10–1.15× faster) against the post-Issue-583 baseline; 0.78–0.82× against the pre-583 one. Stable across 3 runs; the gate only asked for ≤ 2× |
| G2c streaming | **measured, prediction refuted** — 0.872× at 32768×5120 (34.5 vs 42.5 MiB, both past L2), i.e. no better than cache-resident. Effective bandwidth 1.8 GB/s vs a ~400 GB/s roofline: CPU ternary GEMV is compute-bound, so bytes/weight cannot buy latency. The 18.8% is a **capacity** win; the 1.1× is a **decode** win |
| G3 | **PASS** — clippy clean on default / `--no-default-features` / feature-on / `--all-features`; 155 lib tests with the feature, 130 default |
| G4 | **PASS** — 0 allocs / 1000 calls, SIMD and scalar (`[i8; 160]` stack scratch) |

**The prediction in this issue was wrong.** It said "which side wins on CPU is an
open empirical question" and the kernel docs said a loss was expected because the
decode pass costs something the cache-resident regime cannot repay. The trit tier
wins outright, in the regime that should have favoured the incumbent.

**Attribution** (scalar-vs-scalar isolates decode from scale placement, since both
tiers' scalar paths already hoist the scale): base-3 LUT decode is ~10% cheaper
than SWAR bit extraction, and hoisting the group scale out of the inner loop is
worth another ~10%. **The second half is free for the existing bit-plane tier —
no format change — and is filed as Issue 583.**

**Promotion:** declined, opt-in. Same structural reason as Issue 578 — this
implies `ternary_group_scale` implies `binary_plasma`, which is opt-in by
deliberate Issue 145 T2.4 decision, and no default-path consumer holds ternary
weights. But **on CPU, trit-packed is now the better choice for a ternary
consumer**: smaller, faster, bit-identical scalar results.

**Still open (not blocking):** the AVX2 kernel is written and compile-verified but
**unmeasured** — Rosetta 2 has no AVX2, so it cannot execute on this host; queued
for the 4090 with Issue 583 T4. No real-model end-to-end (the GPU consumers need
riir-ai Issue 628). `from_group` is an O(rows·cols) scalar repack, fine at load
time only.
