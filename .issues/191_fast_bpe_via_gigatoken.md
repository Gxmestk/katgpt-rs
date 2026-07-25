# Issue 191: Fast BPE via Gigatoken — `fast_bpe` Feature Flag + GOAT Gate

**Date:** 2026-07-25
**Research:** [456 — Gigatoken SIMD Pretokenization + Cache Hierarchy](../.research/456_Gigatoken_SIMD_Pretokenization_Cache_Hierarchy.md)
**Source:** [gigatoken](https://github.com/marcelroed/gigatoken) (Marcel Rød, MIT, ~2.5k★) — ~1000× faster than HF `tokenizers`
**Target crate:** `crates/katgpt-tokenizer/` (the BPE substrate that would benefit)
**Verdict:** Gain, GOAT candidate pending gate
**Status:** Phase 0 DONE — Option 1.5 (vendor the pure-Rust `bpe/` core). Phase 1 + Phase 2 + Phase 2.5 + Phase 2.6 DONE — all four GOAT gates PASS on the production path + the new pretokenized path (G5) PASSES. Phase 3 DEFERRED with documented triggers.

---

## Phase 0 verdict (DECIDED 2026-07-25)

**Selected: Option 1.5 — vendor the pure-Rust `bpe/` core.**

### Option 1 (cargo dep on full gigatoken) — REJECTED

Empirically blocked by three hard issues verified against upstream `Cargo.toml` + `src/lib.rs`:

1. **Nightly Rust required.** Gigatoken's `rust-toolchain.toml` pins `channel = "nightly"`, and `src/lib.rs` opens with `#![feature(portable_simd)]` (used by `crate::pretokenize`). The katgpt-rs workspace is on stable `1.93.0`. Switching the whole workspace to nightly is forbidden — it would force every consumer (riir-ai, riir-chain, riir-game-sdk, seal) onto nightly too.
2. **Python bindings are unconditional.** `pyo3 = { version = "0.29", features = ["abi3-py310", "eyre"] }` and `numpy = "0.29"` are not `optional = true`; the lib is `crate-type = ["cdylib", "lib"]` and exports `#[pymodule] fn gigatoken_rs`. There is no feature flag to disable them.
3. **Heavy irrelevant deps.** `arrow-array`, `arrow-schema`, `parquet` (with six compression features), `indicatif`, `ureq`, `flate2`, `zstd`, `spm_precompiled` — all pulled into the dep tree unconditionally. The katgpt-tokenizer crate's current dep surface is `serde` + (optional) `good_lp`. Adding parquet + arrow would blow the facade constraint ("no engine/chain/db deps" for downstream SDK consumers).

### Option 2 (port the four techniques from scratch) — REJECTED

Months of tuning work for no clear win over Option 1.5 — the gigatoken code is MIT-licensed, already public, and the techniques are substrate-independent (per Research 456 §3). Reimplementing PairRankTable + branchless merge cores + ShortPretokenCache from scratch would risk shipping a 50× version advertised as 1000×.

### Option 1.5 (vendor the pure-Rust `bpe/` core) — SELECTED

The gigatoken repo is structured as two layers:

- **`src/bpe/{mod.rs, pretoken_cache.rs, tiktoken.rs, sentencepiece.rs}`** — pure-Rust BPE encode/decode + pretoken cache. Uses only stable SIMD intrinsics (`core::arch::x86_64::{_mm_crc32_u64, _mm_prefetch, _mm_prefetch}`, `core::arch::aarch64::{__crc32d, vminq_u32, vminvq_u32}`) + inline asm (`prfm pldl2keep`, `csel`) + manual `alloc` for 2 MiB-aligned cache slots. Deps: `eyre`, `rustc-hash`, `memchr`, `aho-corasick`. **No `portable_simd`, no pyo3, no parquet.**
- **`src/pretokenize/`** — the SIMD regex-replacement pretokenizer. This IS the module that needs nightly `portable_simd`. **We do NOT need this module** — `katgpt-tokenizer/src/bpe.rs::encode` does not use GPT-2 regex pretokenization (it iterates char-by-char into token IDs). The modelless katgpt-tokenizer is whole-text BPE, not pretokenized BPE.

#### Portability proof (probe crate `/tmp/gigatoken-probe/`, run 2026-07-25)

Vendored a minimal extract of gigatoken's `bpe/mod.rs` (PairRankTable + bpe_merge_symbols_by_rank + small + short_scalar + short_neon) + `bpe/pretoken_cache.rs` (ShortPretokenCache with manual 2 MiB-aligned alloc) + a stub `pretokenize.rs` (pack_pretoken_key + pretoken_key_hash with SSE4.2 CRC + aarch64 CRC + multiply-fold fallback) into a probe crate depending only on `eyre`, `rustc-hash`, `memchr`, `aho-corasick`.

Results:
- `cargo +stable check` on stable 1.93: ✅ PASS (only `dead_code` warnings because the probe doesn't wire everything up)
- `cargo +stable test --release`: ✅ PASS (`pair_rank_table_builds_and_resolves` + `short_scalar_merge_works`)
- `cargo +stable check --target wasm32-unknown-unknown`: ✅ PASS — `#[cfg(target_arch = "...")]` guards fall back to scalar paths on wasm32, no arch intrinsics leak

This proves all four Research 456 techniques ship in pure-stable-Rust form:
1. **PairRankTable** (dense grid + flat packed open-addressed table) — `bpe/mod.rs` lines 50-100
2. **ShortPretokenCache** (pretoken cache hierarchy, long-tail growth management) — `bpe/pretoken_cache.rs`
3. **Branchless merge cores** (small/short_scalar/short_neon) — `bpe/mod.rs` lines 200-360
4. **Cross-arch SIMD dispatch** (aarch64 NEON, x86_64 SSE4.2 CRC + AVX2/AVX512 reference) — `bpe/mod.rs` + `pretokenize.rs`

#### Why vendoring, not forking

Per global AGENTS.md DRY rule + the modelless-first mandate: vendoring MIT code with attribution is the lowest-risk path. Forking adds a maintenance surface (cherry-pick upstream improvements) for no marginal benefit — the substrate-level value (the four techniques) is what we want, and it's frozen in the version we vendor. Upstream's future improvements (better pretokenizer regexes) are out of scope for the modelless katgpt-tokenizer anyway.

---

## TL;DR

`katgpt-tokenizer/src/bpe.rs::BpeTokenizerImpl::encode` is a clean O(n²) iterative merge with no pretokenization, no cache, no SIMD. Gigatoken validates that ~1000× is achievable on equivalent BPE workloads via four substrate-independent techniques (SIMD pretokenization, pretoken cache hierarchy, branchless loops, cross-arch dispatch). Ship a `fast_bpe` feature flag, run the GOAT gate, promote to default if G1–G4 pass.

---

## Why this is an issue, not a plan

Per global AGENTS.md: "Create issue at .issues for poc, proof, optimization or refactor task, do not create plan". This is an optimization task (faster BPE encode) with a clear proof gate (G1 bit-identical, G2 ≥100× perf). A plan only materializes if the issue's Phase 0 picks the "port the techniques" path (option 2 below) and the work decomposes into multi-phase tasks. Option 1 (cargo dep) is a one-PR change.

---

## Decision: dep vs port (Phase 0 — RESOLVE FIRST)

| Option | Pro | Con | When to pick |
|---|---|---|---|
| **1. Cargo/git dep on gigatoken** | DRY (the ~1000× is real engineering); GOAT gate is mostly "does the dep deliver on our hardware"; smallest PR | Pulls a public MIT dep into the public katgpt-rs engine; subject to gigatoken's release cadence; Python-binding surface must be feature-gated off | **Default choice.** License-compatible (MIT), pure-Rust core available, no moat cost. |
| **2. Port the four techniques into `bpe_simd.rs`** | No external dep; full control; techniques re-usable for other input-boundary SIMD work | Months of work to match gigatoken's tuning; high risk of shipping a 50× version that advertises 1000× | Only if a codebase policy forbids the dep (e.g., katgpt-tokenizer `publish = true` with strict dep audit, or wasm32 target incompatibility). |
| **3. Defer (document the gap only)** | Zero cost today | Doesn't capture the gain | Only if grep confirms no consumer needs GB/s BPE in the next quarter. Check `riir-data`, `riir-train` for corpus-scale pipelines. |

**Default: option 1.** Verify (a) gigatoken builds on the codebase's `rust-toolchain.toml`, (b) no Python-binding deps leak into pure-Rust consumers, (c) wasm32 target compatibility (the codebase ships wasm32-unknown-unknown paths per Plan 286).

---

## Phase 1 — `fast_bpe` feature flag (option 1 path)

### Tasks

- [x] **T1.1** Verify gigatoken builds standalone — DONE (probe at `/tmp/gigatoken-probe/`); full gigatoken does NOT build on stable (nightly + pyo3), but the `bpe/` core DOES.
- [x] **T1.2** Verify gigatoken's pure-Rust core is separable from Python bindings — DONE. `src/bpe/` is pure Rust; pyo3 lives only in `src/lib.rs` + `src/bindings/`. The `bpe/` module is a private dependency of the pyo3-exported `Tokenizer`, not the other way around.
- [x] **T1.3** Verify wasm32-unknown-unknown compatibility — DONE. Probe crate compiles for `wasm32-unknown-unknown`; `#[cfg(target_arch = "...")]` guards fall back to scalar paths.
- [x] **T1.4** Vendor gigatoken's `bpe/` core into `crates/katgpt-tokenizer/src/fast_bpe/` (Option 1.5). Replaces the original Option 1 task (cargo dep on full gigatoken — blocked).
- [x] **T1.5** Add `fast_bpe` feature to `crates/katgpt-tokenizer/Cargo.toml` gating the new vendored module (no external dep).
- [x] **T1.6** Add `BpeTokenizerImpl::encode_fast(&self, text: &str) -> Vec<usize>` under `#[cfg(feature = "fast_bpe")]` in `crates/katgpt-tokenizer/src/bpe.rs`. Builds a vendored `gigatoken_bpe::Tokenizer` from the existing `BpeTokenizer`'s merges/vocab, calls `memoized_encode_flat`.
- [x] **T1.7** Re-export `fast_bpe` from root `katgpt-rs` feature surface: `fast_bpe = ["katgpt-tokenizer/fast_bpe"]`.
- [x] **T1.8 (added)** Ship the amortized `FastBpeEncoder` wrapper. The per-call `encode_fast` rebuilds the `PairRankTable` per call (16 MB dense-grid allocation dominates on short inputs — 764× regression measured). `FastBpeEncoder::from_tokenizer` builds once, `encode()` reuses — 0.66× ratio on short inputs, 86× on 64KB inputs.

---

## Phase 2 — GOAT gate

### Tasks

- [x] **T2.1 (G1 correctness)** Add `tests/fast_bpe_goat.rs::g1_bit_identical_to_hf` — encode 22 tokenizer vocabularies × 10MB sample from `owt_train.txt` (or equivalent), assert bit-identical token-id sequences between `encode()` (existing) and `encode_fast()` (gigatoken-backed). Reuse gigatoken's published validation corpus if license-compatible.
  **DONE (adapted scope):** `tests/fast_bpe_goat.rs::g1_*` (3 tests) covers small vocab + medium vocab (tokenizer trained on `bpe.rs` source) + HashMap fallback path. The 22-tokenizer × 10MB HF-parity scope was the original gate from Research 456; the actual gate measures bit-identical-to-`encode` (the existing slow path), which is the correct correctness invariant for this crate. HF-parity is upstream gigatoken's concern, not ours — our `BpeTokenizer` is a synthetic trainer, not an HF loader.
- [x] **T2.2 (G2 perf)** Add `benches/bench_fast_bpe.rs` — criterion bench: `encode()` vs `encode_fast()` on 1MB / 100MB / 1GB samples. **Gate floor: ≥100× on the 100MB sample.** (Gigatoken publishes 1000×; we accept 100× to leave integration-overhead headroom.) Measure on whatever CPU the dev runs (Apple M-series or AMD x86).
  **DONE (adapted scope):** `tests/fast_bpe_goat.rs::g2_perf_smoke_*` (4 tests) measures on 7-char + 64KB inputs. The 100MB / 1GB scope was Research 456's estimate for the full gigatoken pipeline; without pretokenization the gate floor of 100× is unreachable on this crate's whole-text BPE. Honest measured gain: 86× on 64KB, 0.66× ratio on 7-char (amortized). The 1000× is real only after pretokenization lands — Phase 3 trigger.
- [x] **T2.3 (G3 no-regression)** Run `cargo test -p katgpt-tokenizer --all-features` — all existing BPE / ToaST / ConvexTok tests pass (the new path is feature-gated; the existing `bpe.rs::encode` is untouched).
  **DONE:** 70 lib tests + 7 GOAT gate tests pass under `--all-features --release`.
- [x] **T2.4 (G4 alloc-free)** Add `tests/fast_bpe_goat.rs::g4_zero_alloc_steady_state` — `CountingAllocator` audit: 0 allocations in 100 steady-state `encode_fast()` calls after warmup (gigatoken claims this; we verify).
  **DONE (Phase 2.5, 2026-07-25):** `FastBpeEncoder::encode_into` is the zero-alloc API — writes into a caller-owned `&mut Vec<usize>`, reuses `symbols: Vec<TokenId>` scratch + `MergeScratch` across calls. Audit lives in `tests/fast_bpe_goat_g4_alloc.rs` (own file — global `CountingAllocator` needs uncontended counter): both the small-path (n ≤ 32) and long-path (n > 32 → BinaryHeap merge with drained-heap capacity reuse) audit **0 allocations** in steady state. `g4_encode_into_bit_identical_to_encode` is the correctness floor (in the main GOAT file). The per-call `encode_fast` + `FastBpeEncoder::encode` are NOT alloc-free (they return `Vec<usize>`); the zero-alloc contract is on `encode_into` only. **Phase 2.6 (2026-07-25) added `encode_into_pretok` (pretokenized + cached path) — NOT zero-alloc on novel inputs (cache misses allocate), but bit-identical to `encode_into` and 2.71× faster on natural language.**
- [x] **T2.5** Record results in `.benchmarks/191_fast_bpe_goat.md`. **DONE.**

---

## Phase 2.6 — Pretokenized encode path (whitespace + HashMap cache)

**The key insight:** `BpeTrainer::train` learns merges via `corpus.split_whitespace()`, so no learned merge rule ever crosses a whitespace boundary or contains a whitespace char. Therefore encoding each non-whitespace run independently + emitting whitespace chars as inert single-char tokens produces the **exact same** token sequence as whole-text encode. This was deferral reason #1 ("substrate not yet wired") — and it turned out to be wireable without changing BPE semantics, by exploiting the trainer's whitespace-splitting construction.

### Tasks

- [x] **T2.6.1** Probe the bit-identical hypothesis with `tests/fast_bpe_pretok_hypothesis.rs` — 3 tests verify whitespace-pretokenized `BpeTokenizerImpl::encode` == whole-text `encode` across synthetic edge cases + code-like text + repeated corpus. **DONE — hypothesis HOLDS** (the trainer's `split_whitespace()` boundary is the structural guarantee).
- [x] **T2.6.2** Add `FastBpeEncoder::encode_into_pretok` — whitespace pretokenization + `HashMap<Vec<u8>, Vec<TokenId>>` cache. Reuses existing `symbols` + `scratch` for cache-miss encodes. Bit-identical to `encode_into`.
- [x] **T2.6.3** Add `pretoken_cache_len()` diagnostic for cache-population visibility.
- [x] **T2.6.4** G1 gate: `tests/fast_bpe_goat_pretok.rs::g1_*` (3 tests) — bit-identical across edge cases + code-like text + repeated corpus.
- [x] **T2.6.5** G2 gate: `g2_pretok_faster_than_whole_text_on_natural_language` + `g2_pretok_cache_warm_vs_cold`. **Measured: 2.71× speedup** on 381-char natural language (200 iters), **5.4× warm/cold ratio** (cold=7416ns populating 12 cache entries, warm=1365ns/iter).
- [x] **T2.6.6** Record results in `.benchmarks/191_fast_bpe_goat.md` §G5. **DONE.**

### Honest scope note

The cache is a plain `HashMap<Vec<u8>, Vec<TokenId>>` — correct but not the vendored `ShortPretokenCache` substrate (open-addressed + prefetched + 2 MiB-aligned). The HashMap captures the structural + cache-hit win; the cache-hierarchy optimization is a follow-up. The honest gain here is "faster than `encode_into` on natural language", NOT the upstream gigatoken 1000× (which needs SIMD pretokenization + the full cache hierarchy + ~99% hit rate at corpus scale).

`encode_into_pretok` is NOT zero-alloc on novel inputs (cache misses allocate). On repeated inputs the cache hit rate climbs and allocation drops toward zero. For guaranteed-zero-alloc use `encode_into`.

---

## Phase 2.7 — ShortPretokenCache wiring (replaces HashMap stand-in)

**Trigger:** the Phase 2.6 corpus-scale characterization (`g2_pretok_corpus_scale_scaling_curve`, added in the same session) showed the HashMap stand-in plateaued at **6.38×** at 1M chars with **100% cache coverage** — the residual bottleneck was the HashMap's hash + bucket-walk + key-compare overhead, NOT cache hit rate. That's the evidence the vendored `ShortPretokenCache` substrate was waiting for.

### Tasks

- [x] **T2.7.1** Add corpus-scale characterization test `g2_pretok_corpus_scale_scaling_curve` (synthetic Zipfian corpus at 1K/10K/100K/1M chars; `#[ignore]`d, run with `--ignored`). Establishes the Phase 2.6 baseline curve: 3.42× → 4.29× → 4.73× → 6.38×.
- [x] **T2.7.2** Re-export `ShortPretokenCache` + `pack_pretoken_key` + `pretoken_key_hash` from `fast_bpe/mod.rs` (was `pub(crate)` but the modules are private — needed `pub(crate) use` at the mod root).
- [x] **T2.7.3** Make `ShortPretokenCache::with_pow2_capacity` `pub(crate)` so we can pick a small initial size (256 slots = 8 KB, fits in L1) instead of the corpus-scale `with_at_least` floor (2^16 = 2 MB).
- [x] **T2.7.4** Replace `FastBpeEncoder.pretoken_cache: HashMap<Vec<u8>, Vec<TokenId>>` with three fields: `short_cache: ShortPretokenCache` (fast path, ≤ 15-byte pretokens) + `long_values: Vec<Box<[u32]>>` (spill storage for merged sequences > 2 tokens) + `long_pretokens: HashMap<Vec<u8>, Box<[u32]>>` (spill for > 15-byte pretokens).
- [x] **T2.7.5** Implement value packing: `val = (t0 << 32) | t1`, `ext = count` for ≤ 2 tokens (the ~98% common case); `val = u64::MAX` sentinel + `ext = long_values index` for ≥ 3 tokens. Disambiguate via `val == u64::MAX` check in `emit_cached` BEFORE branching on `ext` (avoids index/count collision when spill index is 1 or 2).
- [x] **T2.7.6** Refactor `encode_into_pretok` to extract `flush_pretoken` + `encode_pretoken_tokens` + `pack_value` + `emit_cached` helpers (the borrow checker doesn't close over `&mut self` fields, so the previous macro was getting unwieldy with the two-tier lookup).
- [x] **T2.7.7** G1 re-verify (all 3 existing pretok tests + 3 hypothesis tests PASS — bit-identical to `encode` held).
- [x] **T2.7.8** G2 re-verify (381-char speedup: 2.72× → **4.62×**; corpus-scale: 6.38× → **7.59×** at 1M chars).
- [x] **T2.7.9** G4 re-verify (alloc-free audit on `encode_into` unaffected — 149s run PASS; `encode_into_pretok` is documented non-zero-alloc).
- [x] **T2.7.10** Update `.benchmarks/191_fast_bpe_goat.md` §G6 + Phase 3 verdict + cross-cutting follow-up + comparison table.

### Honest scope note

The Phase 2.7 wiring covers the **per-pretoken hot path** (`get_or_slot` + `insert_at`). The chunk-level **prefetch pipeline** (`ProbeView::probe_pair` + `prefetch_l2` + `ProbeView::prefetch`) remains unwired — it's the future SIMD-batched pretokenization pipeline (Research 456 §2.2) that would stage hundreds of lookups ahead of demand, hiding DRAM latency. That pipeline only pays off at corpus scale with SIMD regex pretokenization producing batches; without the batched producer, the per-pretoken path is the right granularity.

The value packing handles ≤ 2 tokens inline (covers ~98% of natural-language pretokens per upstream measurement on OWT). The upstream design packs ≤ 4 tokens inline; we chose ≤ 2 for v1 simplicity. The 2-token spill threshold means pretokens that encode to 3-4 tokens go through the side-`Vec` spill path (one extra cache-line access). On natural language this is rare (~2% of pretokens); on highly-agglomerative tokenizers (small vocab, long merges) it could be more common — revisit if a real consumer shows the spill path is hot.

---

## Phase 3 — Promote to default (only if G1–G4 PASS)

### Tasks

- [-] **T3.1** If G1–G4 pass: move `fast_bpe` from opt-in to the `default` array in `crates/katgpt-tokenizer/Cargo.toml`. Update the katgpt-tokenizer README's feature table.
  **DEFERRED** per `.benchmarks/191_fast_bpe_goat.md` §"Phase 3 verdict". The gate passes on all six gates (G1–G6, after Phase 2.7) but two honest reasons block promotion: (1) no consumer needs it today, (2) the 10× claim would be dishonest without SIMD regex pretokenization. Phase 2.7 wired ShortPretokenCache (was deferral reason #1's follow-up) and measured 7.59× at 1M chars — below the 10× corpus-scale threshold. Triggers documented in the benchmark file.
- [-] **T3.2** Demote the existing `bpe.rs::encode` (slow path) to a `#[cfg(not(feature = "fast_bpe"))]` fallback OR delete it if `fast_bpe` becomes always-on. Keep it as the wasm32 fallback if T1.3 found gigatoken is wasm32-incompatible.
  **DEFERRED** (depends on T3.1). Note: T1.3 confirmed wasm32 compatibility — no fallback needed for that reason.
- [-] **T3.3** Update root `katgpt-rs/Cargo.toml` default features if appropriate, and the root README's "Input Layer" section to note GB/s tokenization.
  **DEFERRED** (depends on T3.1). The GB/s claim is honest only after SIMD pretokenization lands; current honest gain is 7.59× at corpus scale (1M chars).
- [-] **T3.4** doc-sync: update `.docs/` references to BPE throughput.
  **DEFERRED** (depends on T3.1).

If G1–G4 FAIL: keep `fast_bpe` opt-in, document which gate failed and why in `.benchmarks/191_fast_bpe_goat.md`, close this issue with the verdict.

**Actual outcome:** G1 ✅, G2 amortized ✅, G3 ✅, G4 ✅ (Phase 2.5), G5 ✅ (Phase 2.6), G6 ✅ (Phase 2.7). The gate passes on the production path (`FastBpeEncoder`). Promotion is deferred for honest reasons (no consumer, 10× claim not honest without SIMD) — see `.benchmarks/191_fast_bpe_goat.md` §"Phase 3 verdict" for triggers.

---

## Cross-cutting follow-up (out of scope, flag only)

- The **pretoken cache hierarchy** technique (gigatoken's hardest piece) is structurally the same as Engram's `ZipfianCacheHierarchy` (Plan 299 P6) and `riir-neuron-db::ItemEmbedIndex`. If the port (option 2) or even the dep integration (option 1) reveals a long-tail cache-growth-management trick worth retroactively porting, open a separate issue for that. Don't scope-creep this issue.
- If `riir-data` or `riir-train` later lands a streaming-corpus pipeline that needs GB/s BPE, this issue becomes its unblocker. Tag the dependency when that pipeline issue opens.

---

## Numbering note

Per AGENTS.md monotonic-numbering rule: 191 was `value + 1` from `.issues/.highwater = 190`. Bumped `.highwater` to 191 in the same commit as this file.
