//! `fast_bpe` — gigatoken-derived fast BPE encode path (Issue 191, Research 456).
//!
//! Vendored from [gigatoken](https://github.com/marcelroed/gigatoken) (MIT,
//! Marcel Rød, ~2.5k★) per the Issue 191 Phase 0 verdict. Only the pure-Rust
//! `bpe/` core is vendored — the upstream repo also ships a nightly-Rust
//! `pretokenize/` module (uses `#![feature(portable_simd)]`) and unconditional
//! `pyo3`/`numpy`/`parquet`/`arrow` deps that are incompatible with this
//! crate's stable-Rust + leaf-clean constraints. See `.issues/191_*.md` §"Phase
//! 0 verdict" for the portability proof.
//!
//! # What's here
//!
//! Two of gigatoken's four substrate-independent techniques (per Research 456):
//!
//! 1. **`PairRankTable`** — two-level merge-rank lookup replacing the existing
//!    `HashMap<(usize, usize), usize>` probe on every merge step. Dense
//!    `byte × byte` grid covers every round-1 lookup with one shift-or index
//!    and one L1/L2 load; a flat open-addressed packed table covers the rest.
//! 2. **Branchless merge cores** — small (≤32 symbols) + short (≤15 symbols)
//!    scalar + aarch64-NEON + x86_64-AVX2/AVX512 reference variants of the
//!    BPE merge loop, all with stack-resident doubly-linked lists.
//!
//! # What's NOT here (deferred to a follow-up issue if pretokenization lands)
//!
//! 3. **Pretoken cache hierarchy** — `ShortPretokenCache` is shipped in
//!    `pretoken_cache.rs` but not wired into `encode_fast`. The katgpt
//!    `BpeTokenizer` does whole-text BPE (no pretokenization); the cache
//!    only pays off when the same pretoken repeats across calls. Kept for
//!    the day someone adds a GPT-2 regex pretokenizer here or in a sibling
//!    crate. (Research 456 §2.2 names this as the genuinely novel technique;
//!    it stays vendored so the cross-cutting port to Engram's
//!    `ZipfianCacheHierarchy` is a one-import change when that lands.)
//! 4. **SIMD pretokenization** — out of scope. The upstream `pretokenize/`
//!    module is what needs nightly `portable_simd`. The katgpt-tokenizer
//!    is a modelless inference-time encoder; pretokenization is the
//!    pipeline's responsibility, not the tokenizer's.
//!
//! # Attribution
//!
//! Every file in this module records the upstream MIT license header. The
//! techniques ship verbatim where possible (the merge cores are bit-identical
//! to upstream so the `short_merges_match_vec_merge_loop` differential test
//! in upstream applies); adaptation is limited to (a) the `PairRankTable`
//! adapter from katgpt's `usize` IDs to gigatoken's `TokenId(u32)` and
//! (b) removal of the `madvise_hugepage` Linux-only hint from the cache slot
//! allocation path (the hint is a no-op off Linux anyway, and on Linux it
//! would force a `libc` dep this leaf-clean crate forbids).

mod pair_rank_table;
mod pretoken_cache;
mod pretokenize_keys;
mod token;

pub use pair_rank_table::{
    PairRankTable, PairRankTableBuildError, bpe_merge_symbols_by_rank,
    bpe_merge_symbols_by_rank_with_lookup,
};
pub use token::TokenId;

// Re-export the merge cores + scratch so `encode_fast` can drive them directly.
#[allow(unused_imports)] // SHORT_MERGE_MAX + short_scalar are substrate for future pretokenization work.
pub use pair_rank_table::{MergeScratch, SHORT_MERGE_MAX, bpe_merge_symbols_short_scalar};

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)] // substrate for future pretokenization work
pub use pair_rank_table::bpe_merge_symbols_short_neon;
