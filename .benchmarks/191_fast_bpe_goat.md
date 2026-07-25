# Benchmark 191: `fast_bpe` GOAT Gate

**Date:** 2026-07-25
**Issue:** [191 — Fast BPE via Gigatoken](../.issues/191_fast_bpe_via_gigatoken.md)
**Research:** [456 — Gigatoken SIMD Pretokenization + Cache Hierarchy](../.research/456_Gigatoken_SIMD_Pretokenization_Cache_Hierarchy.md)
**Hardware:** Apple M-series (aarch64 Darwin), stable Rust 1.93.0, release profile
**Status:** Phase 1 + Phase 2 + Phase 2.5 DONE — gate **PASSES all four on the production path** (G4 was deferred at Phase 2, landed in Phase 2.5). Per-call `encode_fast` path is a documented regression on short inputs only.

---

## TL;DR

The vendored gigatoken BPE core (PairRankTable + heap+linked-list merge loop) ships behind the `fast_bpe` feature. The headline 1000× from upstream gigatoken **does NOT hold here** — that requires pretokenization + per-pretoken cache, which the katgpt `BpeTokenizer` does not do. The honest measured gain on the realistic use case (whole-text BPE encode) is:

- **86× speedup** on 64KB inputs (amortized encoder, release mode, Apple Silicon)
- **0.66× ratio** on 7-char inputs (amortized encoder is *faster* than `encode` even on short inputs)
- **82× speedup** on 64KB inputs (per-call `encode_fast` function — table rebuild amortized by algorithmic win)
- **764× regression** on 7-char inputs (per-call `encode_fast` — table rebuild dominates; documented, use `FastBpeEncoder`)

**Phase 3 verdict: DEFER promotion to default.** The amortized path PASSES the GOAT gate but the substrate (short-merge cores + pretoken cache) for the full gigatoken pipeline is shipped-but-unwired. Promoting to default before the substrate is wired would be advertising capability we don't have. Re-open Phase 3 when (a) pretokenization lands and the 1000× becomes real, OR (b) a downstream consumer (riir-data, riir-train) needs the 86× corpus-scale speedup.

---

## Phase 0 verdict — Option 1.5 (vendor)

See the [issue file](../.issues/191_fast_bpe_via_gigatoken.md) §"Phase 0 verdict" for the full decision. Summary: cargo dep on full gigatoken is **blocked** by (1) nightly `portable_simd`, (2) unconditional `pyo3`/`numpy`/`parquet`/`arrow` deps, (3) the workspace's stable-Rust + leaf-clean constraints. Vendoring the pure-Rust `bpe/` core (~2k LOC of MIT code) is the right path — proven portable to stable 1.93 + wasm32-unknown-unknown via the `/tmp/gigatoken-probe/` probe crate.

---

## Phase 1 deliverables

| File | Role | LOC |
|---|---|---|
| `crates/katgpt-tokenizer/src/fast_bpe/mod.rs` | Module root + re-exports + attribution | ~70 |
| `crates/katgpt-tokenizer/src/fast_bpe/token.rs` | `TokenId(u32)` newtype (bit-identical to upstream) | ~50 |
| `crates/katgpt-tokenizer/src/fast_bpe/pair_rank_table.rs` | `PairRankTable` + `MergeScratch` + branchless merge cores (small / short_scalar / short_neon) | ~730 |
| `crates/katgpt-tokenizer/src/fast_bpe/pretoken_cache.rs` | `ShortPretokenCache` (shipped, not wired — substrate for future pretokenization) | ~480 |
| `crates/katgpt-tokenizer/src/fast_bpe/pretokenize_keys.rs` | `pack_pretoken_key` + `pretoken_key_hash` (substrate for future pretokenization) | ~190 |
| `crates/katgpt-tokenizer/src/bpe.rs::FastBpeEncoder` | The amortized encoder — wraps `&BpeTokenizer` with cached `PairRankTable` + scratch + reusable `symbols` buffer | ~110 |
| `crates/katgpt-tokenizer/src/bpe.rs::BpeTokenizerImpl::encode_fast` | One-shot per-call convenience function (delegates to `FastBpeEncoder`) | ~5 |
| `crates/katgpt-tokenizer/tests/fast_bpe_goat.rs` | GOAT gate (G1 correctness + G2 perf smoke + G4 correctness floor) | ~280 |
| `crates/katgpt-tokenizer/tests/fast_bpe_goat_g4_alloc.rs` | G4 alloc-free audit (CountingAllocator — own file so the global counter is uncontended) | ~150 |

Vendored line count: ~1520 LOC. Adaptation delta vs upstream: ~50 LOC (module path + dropped SentencePiece variants + dropped `madvise_hugepage`).

---

## Phase 2 GOAT gate results

### G1 — correctness (bit-identical to `encode`)

| Test | Result |
|---|---|
| `g1_bit_identical_to_encode_small_vocab` (50-merge tokenizer on synthetic corpus) | ✅ PASS — both `encode_fast` and `FastBpeEncoder::encode` bit-identical to `encode` on 8 short texts including unknown chars + empty |
| `g1_bit_identical_to_encode_medium_vocab` (1024-merge tokenizer trained on `bpe.rs` source) | ✅ PASS — bit-identical on 4 short texts + the whole `bpe.rs` corpus (61 KB) |
| `g1_bit_identical_to_encode_with_table_fallback` (HashMap fallback path) | ✅ PASS — bit-identical on the fallback path (the `PairRankTable::build` refusal case) |
| `pair_rank_table_matches_map` (lib unit test, 100k random pairs) | ✅ PASS — flat table agrees with HashMap on every merge pair + the no-merge sentinel everywhere else |
| `short_merges_match_vec_merge_loop` (lib unit test, 2000-trial fuzz) | ✅ PASS — scalar + NEON short-merge cores produce identical output to the heap+linked-list loop |

**G1 verdict: ✅ PASS.**

### G2 — perf smoke

| Test | Input | Result |
|---|---|---|
| `g2_perf_smoke_amortized_no_regression_on_short_input` | "the cat" (7 chars), 1000 iters | ✅ PASS — **0.66× ratio** (FASTER than `encode` even on short inputs) |
| `g2_perf_smoke_amortized_gain_on_long_input` | `bpe.rs × 4` (64KB), 20 iters | ✅ PASS — **86.13× speedup** |
| `g2_perf_smoke_gain_on_long_input` (per-call) | `bpe.rs × 4` (64KB), 20 iters | ✅ PASS — **82.09× speedup** |
| `g2_perf_smoke_per_call_short_input_documented_regression` | "the cat" (7 chars), 100 iters | ✅ PASS (loose gate, 1000×) — **764× regression** documented; use `FastBpeEncoder` for short inputs |

**G2 verdict: ✅ PASS on the production path** (`FastBpeEncoder`). The per-call `encode_fast` is documented as a one-shot long-input function — its short-input regression is expected (the `PairRankTable::build` dense-grid allocation is ~16 MB regardless of merge count).

### G3 — no-regression

| Test | Result |
|---|---|
| `cargo test -p katgpt-tokenizer --all-features --release` | ✅ PASS — 70 lib tests + 7 GOAT gate tests + 0 doc tests = 77 pass, 0 fail, 1 ignored |
| `cargo test -p katgpt-tokenizer --lib` (without fast_bpe) | ✅ PASS — 6 existing tests still pass |
| `cargo check -p katgpt-tokenizer --no-default-features` | ✅ PASS |
| `cargo check -p katgpt-tokenizer --features fast_bpe --target wasm32-unknown-unknown` | ✅ PASS — `#[cfg(target_arch = "...")]` guards fall back to scalar paths |
| `cargo clippy -p katgpt-tokenizer --features fast_bpe --all-targets` | ✅ PASS — zero warnings |

**G3 verdict: ✅ PASS.**

### G4 — alloc-free steady state

**✅ PASS (Phase 2.5, 2026-07-25).** The new `FastBpeEncoder::encode_into` API writes its output into a caller-owned `&mut Vec<usize>` buffer, reusing the encoder's `symbols: Vec<TokenId>` scratch + `MergeScratch` across calls. After warmup, steady-state `encode_into` performs **zero heap allocations** on both the small-path (n ≤ 32 → stack-resident linked-list merge) and the long-path (n > 32 → BinaryHeap merge with drained-heap capacity reuse).

The audit lives in `tests/fast_bpe_goat_g4_alloc.rs` (its own file — the global `CountingAllocator` counter is uncontended only when the file has a single test). Both paths audited in `g4_zero_alloc_audit_combined`: small-path 0/100, long-path 0/20.

The per-call `BpeTokenizerImpl::encode_fast` + `FastBpeEncoder::encode` (which returns `Vec<usize>`) are NOT alloc-free — they allocate the return value per call. The zero-alloc contract is on `encode_into` only.

| Test | Result |
|---|---|
| `g4_zero_alloc_audit_combined` (small-path, 100 calls, n ≤ 32) | ✅ PASS — **0 allocations** in steady state |
| `g4_zero_alloc_audit_combined` (long-path, 20 calls, n > 32 → BinaryHeap) | ✅ PASS — **0 allocations** in steady state |
| `g4_encode_into_bit_identical_to_encode` (correctness floor) | ✅ PASS — `encode_into` bit-identical to `encode` across short + whole-corpus inputs |

**G4 verdict: ✅ PASS.**

---

## Phase 3 verdict — DEFER promotion

The gate **PASSES on the production path** (G1 ✅, G2 amortized ✅, G3 ✅, G4 ✅ — all four gates green). But promoting `fast_bpe` to default-on is **deferred** for three honest reasons:

1. **Substrate not yet wired.** The vendored module ships four substrate pieces (PairRankTable + merge cores + ShortPretokenCache + pretoken key utilities); only two are wired into `encode_fast` (PairRankTable + merge cores). The other two are explicitly substrate for future pretokenization work. Promoting to default before the substrate is wired would be advertising capability we don't have.

2. **No consumer needs it today.** The katgpt-tokenizer's existing callers are per-prompt encoders (riir-engine's Gemma2 tokenizer, riir-engine's `rove_perplexity_poc`, the `core_01_validator` example). None encode corpus-scale inputs. The 86× speedup is real but unlocked only when a corpus-scale consumer (riir-data, riir-train) lands.

3. **The headline 1000× claim is not honest.** Research 456 §3.1 flagged this from the start: the 1000× requires pretokenization + per-pretoken cache (~99% hit rate). Without pretokenization, the honest gain is 86× on long inputs. Promoting to default with a comment claiming 1000× would be dishonest.

**Phase 3 triggers** (any one):
- Pretokenization lands in katgpt-tokenizer (the SIMD GPT-2 regex replacement) → re-run GOAT gate, expect close to 1000× on corpus inputs.
- A downstream consumer (riir-data, riir-train) opens an issue requesting corpus-scale BPE → promote to default in that issue's plan.
- The cross-cutting cache-hierarchy port (Engram `ZipfianCacheHierarchy`, riir-neuron-db `ItemEmbedIndex`) lands and validates the pretoken cache pattern → promote alongside.

---

## Honest comparison: this vs upstream gigatoken

| Aspect | Upstream gigatoken | This crate (`fast_bpe`) |
|---|---|---|
| Headline speedup | 1000× vs HF `tokenizers` | 86× vs existing `encode` (long inputs), 0.66× ratio (short inputs, amortized) |
| Pretokenization | SIMD regex replacement (`portable_simd`, nightly) | **Not shipped** — would require nightly |
| Pretoken cache | Open-addressing + 2 MiB-aligned slots + hugepage | **Shipped, not wired** — substrate only |
| PairRankTable | Dense 16 MB grid + flat packed table | ✅ Shipped + wired |
| Merge cores | Heap+linked-list + small + short (scalar + NEON + AVX2/AVX512 reference) | ✅ Shipped + wired (heap+small+short_scalar+short_neon) |
| Python bindings | `pyo3` + `numpy` (unconditional) | **Not shipped** — pure Rust |
| Stable Rust | Nightly required | ✅ Stable 1.93 |
| wasm32 | Not validated | ✅ Compiles + scalar fallbacks |
| Attribution | MIT, Marcel Rød | MIT, Marcel Rød (vendored with header per file) |

The honest framing: **`fast_bpe` ships gigatoken's *BPE merge-core speedups*, not its *pipeline speedups*.** The pipeline (pretokenization + cache) is the substrate that unlocks the 1000×; it's vendored-but-unwired, waiting for someone to port the nightly `portable_simd` regex-replacement to stable `core::arch` intrinsics (a meaningful project, not a quick fix).

---

## Reproducing

```bash
# Stable Rust 1.93+ required (matches workspace).
cd katgpt-rs

# Run the GOAT gate (release mode for accurate perf measurement).
CARGO_TARGET_DIR=/tmp/katgpt-fast-bpe cargo test -p katgpt-tokenizer \
    --features fast_bpe --test fast_bpe_goat --release -- --nocapture

# Run the G4 alloc-free audit (own test file — the global CountingAllocator
# counter needs to be uncontended).
CARGO_TARGET_DIR=/tmp/katgpt-fast-bpe cargo test -p katgpt-tokenizer \
    --features fast_bpe --test fast_bpe_goat_g4_alloc --release -- --nocapture

# Run the lib unit tests (differential fuzz tests for the merge cores).
CARGO_TARGET_DIR=/tmp/katgpt-fast-bpe cargo test -p katgpt-tokenizer \
    --features fast_bpe --lib --release

# Verify no-regression on the existing path (without fast_bpe).
CARGO_TARGET_DIR=/tmp/katgpt-fast-bpe cargo test -p katgpt-tokenizer --lib

# Verify wasm32 compatibility.
CARGO_TARGET_DIR=/tmp/katgpt-fast-bpe cargo check -p katgpt-tokenizer \
    --features fast_bpe --target wasm32-unknown-unknown
```

---

## Cross-cutting follow-up (flagged, not scoped)

Per Issue 191 + Research 456 §2.2: the **pretoken cache hierarchy** technique (gigatoken's hardest piece) is structurally the same as Engram's `ZipfianCacheHierarchy` (Plan 299 P6) and riir-neuron-db's `ItemEmbedIndex`. The vendored `ShortPretokenCache` ships in `crates/katgpt-tokenizer/src/fast_bpe/pretoken_cache.rs` but is NOT wired into `encode_fast` (no pretokenization). If/when pretokenization lands, the long-tail cache-growth-management trick (Heaps-law sizing + 3/4 load growth threshold + 2 MiB alignment for dTLB) should be evaluated for retroactive porting to Engram's `ZipfianCacheHierarchy`. Open a separate issue then; don't scope-creep Issue 191.
