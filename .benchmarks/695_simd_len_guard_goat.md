# Bench 695 — SIMD `len`-family soundness guard (GOAT gate)

Status: COMPLETE — GOAT PASS (G1/G2/G3/G4). Fix landed; promoted unconditionally
(a soundness fix is not feature-gateable — see "Why no feature flag" below).

## The finding

`crates/katgpt-types/src/simd/` exposes 11 **safe** public fns that take an
explicit `len: usize` alongside their slices and then perform **unchecked**
pointer loads/stores up to `len`:

| fn | file | kind |
|---|---|---|
| `simd_dot_f32`, `simd_fma_row` | `dot.rs` | read |
| `simd_dot_f16_f32`, `simd_dot_f16_f16` | `dot.rs` | read |
| `simd_sum_sq`, `simd_dist_sq`, `simd_l_inf_distance_f32` | `research.rs` | read |
| `simd_fused_sub_acc`, `simd_fused_scale_acc`, `simd_fused_scale_acc_f16` | `research.rs` | **write** |
| `simd_gram_f32` | `research.rs` | derived (see below) |

9 of 11 had **no** `len` validation at all. Passing `len > slice.len()` from
100% safe code was therefore:

- **CWE-125 (OOB read)** for the read family — measured: `simd_dot_f32(&[1.0; 4],
  &[1.0; 4], 4096)` returned `NaN` on one run and `4` on another (allocator
  state dependent — nondeterministic, as OOB is). No panic, no diagnostic.
- **CWE-787 (OOB write)** for the `dst: &mut [f32]` family — measured with a
  contained neighbour-sentinel arena: `simd_fused_scale_acc(dst_16, &src, 2.0, 48)`
  corrupted **32 of 48** neighbouring floats (`-777.0 → -775.0`, i.e. the kernel
  applied `dst[i] += src[i]*2.0` past the end of `dst`).

The two fns that *did* carry a guard were **also unsound**:
`simd_l_inf_distance_f32` had `debug_assert_eq!(a.len(), b.len())`, which compares
the slices to **each other** and never to `len` (4 == 4 passes while `len` is
4096), and vanishes entirely in release. This is the load-bearing lesson: the
existing house idiom did not express the actual precondition.

No `get_unchecked` precondition check ever fired, because the SIMD bodies use raw
`as_ptr().add(i)` + `vld1q_f32`; the scalar tail that *does* use `get_unchecked`
is never reached when `len` is a clean multiple of the vector width. The bug was
therefore invisible to debug UB checks.

## The fix

Entry-point reslice on each public fn, e.g.

```rust
let (a, b) = (&a[..len], &b[..len]);
```

One bounds check per **call** (not per element). After it, every unchecked
access below is provably in range, and LLVM gains the length. `simd_fma_row`
inherits via its delegation to `simd_dot_f32` (no duplicate guard — DRY).
`simd_gram_f32` needs no new guard: it indexes safely (`&x[i*d_h..]`) and its
dot calls are now covered by the guarded `simd_dot_f32`.

## Why no feature flag

The repo's default is opt-in-behind-a-flag. That is wrong here and the rule
should not be applied mechanically: a flag would mean shipping a build in which
safe code can still corrupt the heap, and `--all-features` CI would be the only
sound configuration. Soundness is not a performance trade to be A/B'd — it is
promoted unconditionally. The *cost* is still gated by G2 below.

## G1 — correctness (numeric neutrality)

Bit-identical results, both arms, all sizes. The A/B binary prints an
accumulator sink; across 7 interleaved rounds × 5 sizes × 2 arms there is
exactly **1 distinct sink value per size** (`d=8 → 1759999.9`, `64 → 7039999.5`,
`256 → 14079999`, `1024 → 22527998`, `4096 → 22527998`). The guard clamps
nothing: a short `len` remains a valid prefix request (regression test
`short_len_is_a_valid_prefix_request`).

## G2 — perf (interleaved A/B, per-pair ratios)

Baseline = clean HEAD in a detached worktree; both arms built `--release`, and
the two binaries were checksum-verified to differ. 7 interleaved rounds,
alternating arms; ratio is the **median of per-pair ratios** (not
median(A)/median(B) — that form has flipped verdicts before).

| d | base ns/call | guard ns/call | ratio | Δ ns |
|---|---|---|---|---|
| 8 | 1.265 | 1.477 | **1.168** | +0.212 |
| 64 | 4.005 | 4.100 | 0.995 | +0.095 |
| 256 | 12.710 | 12.836 | 1.004 | +0.126 |
| 1024 | 52.367 | 51.810 | **0.989** | −0.557 |
| 4096 | 244.142 | 240.183 | **0.988** | −3.959 |

Read honestly: the guard costs a **fixed ~0.2 ns/call** (≈1 cycle). That is
+16.8% at `d=8`, noise at `d=64..256`, and a small *win* at `d≥1024` (LLVM
exploits the known length). **VERDICT: PASS.** Eliminating a safe-code heap
corruption primitive for ~1 cycle per call is the correct trade, and the
regression is confined to the smallest size, which is not the matmul-row shape
this kernel exists to serve. Callers that dot 8-element vectors in a tight loop
should hoist the check by slicing once outside the loop.

Reproduce: `cargo run --release -p katgpt-types --example simd_len_guard_bench_700`.

## G3 — no regression

Per-crate `--lib` across all **21** crates that consume the guarded fns:
**3304 passed, 0 failed** (katgpt-core 1992, katgpt-speculative 305, katgpt-dec
225, katgpt-types 132, katgpt-forward 125, katgpt-pruners 126, …).

Caveat, stated rather than hidden: `katgpt-attn`, `katgpt-attn-match`,
`katgpt-kv` and `katgpt-quant` report **0 tests** under default features — their
suites are cfg-gated, so this run does not exercise them. They are 0 at baseline
too, so this is not a regression, but it is not coverage either.

`cargo test --workspace --lib` was NOT used as the gate because it does not
compile on `develop` — see Issue 700 (pre-existing, reproduced at clean HEAD).

## G4 — alloc-free

The guard introduces no allocation: reslicing is a bounds check plus a fat-pointer
narrowing. No heap traffic added to any kernel.

## The second copy — `katgpt-dec` (found by the follow-up sweep)

`crates/katgpt-dec/src/simd.rs` ships its **own** `simd_dot_f32`, a near-verbatim
twin of katgpt-types' `scalar_dot_f32`, and carried the identical hole. The
duplication is deliberate and correct: `katgpt-dec` declares **zero
dependencies** so `katgpt-core` can re-export it as `katgpt_core::dec` without a
cyclic package dep. Consuming the guarded substrate twin would create exactly the
edge that design forbids, so the guard is applied in place instead. Pinned by
`crates/katgpt-dec/tests/simd_len_guard_695.rs`; `katgpt-dec` is 225 lib + 2 new,
0 failed.

## Cross-repo sweep — the class is confined to katgpt-rs

Scanned `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-game-sdk`, `riir-train`,
`riir-clippy`, `riir-dapps`, `katgpt-web` for safe `pub fn` taking an explicit
`len`-like param over a slice. Three candidates, **all sound** — each pins the
exact length with a real (not `debug_`) `assert_eq!` before any unchecked use:
`riir-infer-core/src/wall.rs:110 wall_prefix_prefill`,
`riir-train-engine/src/adapter_centroid.rs:258 split_adapter_weights`,
`riir-train-engine/src/maglev_drafter/joint.rs:976 make_shifted_targets`.
`katgpt-attn-match/src/score_matrix.rs:57 row_max` likewise asserts and indexes
safely. No sibling-repo fix required.

## Regression gate

`crates/katgpt-types/tests/simd_len_guard_700.rs` — 11 tests, all safe code, one
`#[should_panic]` per guarded fn plus the prefix-honoured test. **The pin was
verified to fire**: the identical file run against the unguarded HEAD worktree
does not merely fail, it **SIGSEGVs** (`signal: 11, SIGSEGV: invalid memory
reference`, "deadlock in SIGSEGV handler"). A future removal of the guard cannot
pass this suite quietly.
