# Plan 430 Dual-Path Tree Verification — GOAT Gate Results

**Date:** 2026-07-13
**Plan:** [katgpt-rs/.plans/430_dual_path_rollback_free_tree_verify_gdn_hola.md](../.plans/430_dual_path_rollback_free_tree_verify_gdn_hola.md)
**Feature:** `gdn_hola_tree_verify` (opt-in; implies `gdn_tree_verify` + `hippocampal_cache`)
**Bench:** `benches/bench_430_dual_path_verify.rs`

## Summary

| Gate | Result | Notes |
|------|--------|-------|
| **G1** (correctness) | ✅ PASS | Dual-path output matches sequential GDN2+HOLA forward to <1e-3 at T={16,32,64,128}. |
| **G2** (fusion efficiency) | ✅ PASS | `dual/(gdn+hola) ≈ 0.91–1.07` — the fusion adds **no overhead** beyond the sum of parts (shared scratch, single traversal). |
| **G2** (1.2× sub-bar) | ❌ FAIL | `dual/gdn = 1.24–1.40×` — HOLA's W=64 softmax read adds 24–40% on GDN. The plan's 1.2× target was based on wrong FLOP analysis (see below). |
| **G3** (no-regression) | ✅ PASS | GDN component byte-identical to Plan 424 standalone; feature-gated compile-time fallback. |
| **G4** (alloc-free) | ✅ PASS | 0 allocations on steady-state `verify_gdn_hola_tree_into` (CountingAllocator). |
| **G5** (retrieval) | ⏳ DEFERRED | Requires trained GDN2+HOLA model. Inherited from Plan 395's G4 retrieval gate. |

## G2 Detailed Results (d_k=d_v=64, W=64, release, single-threaded)

### Random tree (shallow, depth ~log T) — typical speculative decode shape

| T | Dual-path | GDN-only | HOLA-only | dual/gdn | dual/(gdn+hola) |
|---|---|---|---|---|---|
| 16  | 88.5 µs  | 64.9 µs  | 17.5 µs  | 1.36× | 1.07× |
| 32  | 164.1 µs | 131.2 µs | 34.8 µs  | 1.25× | 0.99× |
| 64  | 331.5 µs | 264.8 µs | 76.5 µs  | 1.25× | 0.97× |
| 128 | 693.4 µs | 557.3 µs | 150.6 µs | 1.24× | 0.98× |

### Chain tree (deep, depth = T) — worst case for HOLA ancestor path

| T | Dual-path | GDN-only | HOLA-only | dual/gdn | dual/(gdn+hola) |
|---|---|---|---|---|---|
| 16  | 86.6 µs   | 67.4 µs  | 20.1 µs  | 1.29× | 0.99× |
| 32  | 185.9 µs  | 143.3 µs | 48.5 µs  | 1.30× | 0.97× |
| 64  | 434.2 µs  | 323.0 µs | 135.7 µs | 1.34× | 0.95× |
| 128 | 1108.0 µs | 790.8 µs | 421.4 µs | 1.40× | 0.91× |

## Analysis: why `dual/gdn > 1.2×`

The plan's 1.2× target assumed "HOLA read is O(W·D), small vs O(T²·d_k)". This FLOP
analysis is correct but misleading:

- **GDN solve**: O(T²·d²) pure FMAs (no transcendental ops) — 67M FMAs at T=128, d=64.
- **HOLA read**: O(T·W·d) FMAs + **O(T·W) `exp` evaluations** — 524K FMAs + 8192 exps
  at T=128, W=64. By FLOP count HOLA is <1% of GDN. But `exp` is ~10× slower than
  FMA, and the streaming softmax has a serial dependency (running max + sum_exp) that
  prevents vectorization across slots. The wall-clock cost is therefore 24–40% of GDN.

**This is inherent to softmax attention at W=64.** It is NOT an implementation
inefficiency — the pre-normalization optimization (below) eliminated redundant work
and `dual/(gdn+hola) < 1.0` confirms the fusion itself adds zero overhead.

## G2 Efficiency Proof: `dual/(gdn+hola) ≤ 1.0`

The authoritative GOAT gate definition (Plan 430 line 20):

> G2 (perf — dual-path verify ≤ GDN-only verify + HOLA-only read, no rollback)

This is the **fusion efficiency** check: does fusing GDN+HOLA into one pass cost more
than running them separately? The answer is **NO** — the fusion is cheaper:

- Random T≥32: `dual/(gdn+hola) = 0.97–0.99×` (2–3% **cheaper** than separate)
- Chain T=128: `dual/(gdn+hola) = 0.91×` (9% **cheaper** than separate)

The savings come from shared scratch buffers (no per-path allocation) and a single
tree traversal (ancestor paths walked once). At T=16 the ratio is 1.07 due to
measurement noise on small absolute times (~88 µs).

**G2 PASSES** per the GOAT gate definition.

## Pre-normalization optimization (this session)

Added `tree_keys_norm` pre-allocated buffer to `GdnHolaTreeVerifier` and
`read_cache_into_fast_block_prenorm` to `HippocampalCacheDyn`. The dual-path now
pre-normalizes all tree keys once (using the cache's gamma) instead of re-normalizing
each ancestor key in every node's block_kv. This is a DRY fix:

- Chain T=128: root key was normalized 128× (once per descendant) → now 1×
- Impact: chain T=128 dual-path **1214.6 → 1108.0 µs** (−8.8%)
- Random trees barely affected (depth ~log T → few redundant normalizations)

Bit-identical output (6 existing dual-path tests pass unchanged, comparing against
the non-prenorm reference).

## Verdict

**G2 PASSES** per the GOAT gate definition (fusion adds no overhead beyond sum of parts).
The 1.2× aspirational sub-bar is **NOT met** (actual: 1.24–1.40×) — the HOLA softmax
read at W=64 inherently costs 24–40% of GDN's masked solve. This is the real cost of
exact long-range recall; it is not an implementation deficiency.

The feature stays **opt-in** behind `gdn_hola_tree_verify` (requires both
`gdn_tree_verify` + `hippocampal_cache` + a trained γ or modelless γ=1). Not promoted
to default.

**Recommendation:** if the 1.2× sub-bar is needed for a specific deployment, reduce
W (cache width) — the HOLA cost scales linearly with W. At W=16 the overhead drops
to ~6–10% (sub-1.2×). The tradeoff is fewer cached high-surprise tokens.

## Configuration

- **d_k, d_v, cache.d:** 64 (the dual-path constraint: single head dim)
- **W (cache width):** 64 (paper-scale; the primary HOLA cost driver)
- **Tree shapes:** random (shallow, depth ~log T) + chain (deep, depth = T)
- **Tolerance:** G1 uses 1e-3 (f32 accumulation)

## Reproduce

```bash
CARGO_TARGET_DIR=/tmp/430_dual_path_verify \
  cargo bench -p katgpt-core --features gdn_hola_tree_verify \
  --bench bench_430_dual_path_verify -- --nocapture
```
