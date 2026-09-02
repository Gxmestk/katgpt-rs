# Plan 586 — Issue 845 CPU scaffold: PoT-scale determinism for the ternary group tier

**Status:** IN PROGRESS — CPU-side half of riir-ai `.issues/845` (GPU-window-gated G1/G3 stay armed there).

Source: arXiv:2609.00363 (verified full-text, riir-clippy `.research/116`) — requantizing
quant scales to nearest powers of two makes scale application exact on every backend
(`rnd(x)·2^k = rnd(x·2^k)`), and kills the Vulkan `div.full` fault class AT THE SCALE SITE
by construction. Non-negotiables (the paper's +157% artifact): requantize from parents,
never reinterpret stored payloads; byte-exact reconstruction gate FIRST.

## T0 — Scale inventory (the "pin the inventory" discipline)

Every scale the Q2_0_g128 epilogue touches, and where it is snapped:

| Scale | Computed | Stored | Applied | In scope? |
|---|---|---|---|---|
| Ternary group scale (per-128) | CPU `quantize_from_f32` (mean-abs → f16) | f16 (`group_scale`) / f32 decode in `TernaryHandle` + AoS `TernaryBlockAoS` | GPU kernels by MULTIPLY (`sign·s`, dp4a dot·s, fma s0..s3; `gemv_ternary_cubecl.rs`, `gemm_ternary_*`, cudarc dp4a, embedding dequant) | **YES — this plan** |
| Scale-storage f16 rounding | — | f16 | — | **Eliminated by PoT** (PoT is exact in f16 for exp ∈ [-14, 15]) |
| Q8KV block scale (32-block) | CPU `Q8KVBuffers::quantize_kv_half` (block.d f16) at runtime quantize | f16 | kernel dequant MULTIPLY | YES, separate slice — **B55/Issue 716 territory, coordinate first** (deferred `[-]`) |
| ANE row requant `/127` | CPU `requant_per_row_int8` | f16 row scale | ANE int8 path | NO — fixed-format constant, CPU-side, not a cross-backend divergence site |
| Softmax `1/L`, rmsnorm `inv_rms` | runtime, data-dependent | f32 | DIV/rsqrt | NO — not quant scales; data-dependent, cannot be pre-snapped (out of the paper's construction; Bench 728's div.full there is covered by Issue 802-class tolerance gates, not this lever) |

Honest scoping (mirrors R116's caveat): PoT scales make the SCALE-APPLICATION contribution
to CPU↔GPU diffs exactly zero (multiply by PoT = exponent shift, zero rounding on both
sides). Accumulation-order faults (Bench 728's FMA-contraction class) are NOT scale faults
and remain tolerance-gated. G1's bit-identity claim is scoped to scale application sites,
exactly as riir-ai `.issues/845` worded it.

Bonus property (pinned by test below): PoT quantization is **exactly scale-covariant** —
quantizing `w·2^k` yields bit-identical payloads with exponent-shifted scales (every
intermediate scales by exact PoT factors; GROUP_SIZE=128=2^7 keeps the mean-abs divide
exact). The native tier only approximates this (f16 rounding of the scale breaks it).
Consequence: pre-scaling a weight tensor (e.g. folding a constant into weights at export)
no longer perturbs the quantized payload.

## Tasks

- [x] T0 scale inventory (this file; referenced from riir-ai `.issues/845`) — DONE at plan-write time (the table above is the deliverable)
- [x] `pot_scales = ["ternary_group_scale"]` feature in katgpt-types
- [x] `quantize_with_scale_rule` private core (snap=false = the legacy body VERBATIM;
      snap=true inserts `snap_f32_to_pow2` on the mean-abs before the f16 store) +
      `quantize_from_f32_pot` + private `snap_f32_to_pow2` (exact exponent math —
      bit-extract floor(log2), exact PoT divide to ratio ∈ [1,2), nearest in log space
      via `ratio² >= 2`, clamp to f16-normal PoT range [-14, 15]) + `all_scales_are_pow2` checker
- [x] Tests (`#[cfg(all(test, feature = "pot_scales"))]`): snap unit cases (exact-PoT stays,
      nearest picks √2 boundary correctly, out-of-range clamps, zero-group → 1.0 stays PoT);
      all-scales-PoT on random weights; scale-covariance for k ∈ [-6, 6] (payloads bit-identical,
      scales exactly ×2^k); f16 PoT round-trip exact for k ∈ [-14, 15]; native vs PoT SNR
      harness (deterministic distributions, printed table + pinned bound); native arm sanity
      (its scales are NOT all PoT — the arms genuinely differ)
- [x] Validation: `cargo clippy -p katgpt-types --all-targets` (default + pot_scales states)
      + `cargo test -p katgpt-types --lib` (default 132 ✓ / ternary_group_scale 151 ✓ /
      pot_scales 159 ✓ incl. 8 new pot_tests). SNR table (uniform/gaussish/sparse70/
      row-scaled): PoT rel-err ratio 0.988–1.012 vs native — the carry loop absorbs
      essentially all of the ≤√2 snap factor, far inside the pinned 1.35 bound.
- [ ] Commit + push (katgpt-rs develop)
- [ ] riir-ai `.issues/845` updated: CPU-half landed, this plan + commit referenced; G1/G3
      stay armed for the GPU-exclusive window
- [ ] riir-clippy queue snapshot updated (lever state: CPU half DONE)

## Deferred (this plan)

- [-] Trit-tier (`ternary_trit_pack`) PoT mirror — unblock: group-tier G1 PASS in the GPU
      window; the trit quantizer must mirror the same scale rule or the cross-tier
      `quantize_is_bit_identical_to_the_bit_plane_tier` contract would need a PoT-aware
      reformulation.
- [-] Q8KV snap-at-quantize slice — unblock: group-tier G1 PASS + B55/Issue 716 owner
      coordination (the issue's own entry rule).
- [-] GPU G1/G2/G3 gates — unblock: GPU-exclusive 4090 window (sibling L4 eval running at
      plan time; per the Issue 649 exclusivity rule these cannot run under contention).
