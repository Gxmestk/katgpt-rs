# Bench 663 — Bigram Markov head: the modelless sequential drafter primitive (Issue 659 T1–T3)

**Date:** 2026-08-17
**Repo:** katgpt-rs (`crates/katgpt-speculative/src/bigram_markov.rs`, feature `bigram_markov`, opt-in)
**Scope:** the PRIMITIVE gate only. The Bonsai consumer gate (acceptance rate G2, wall-clock G3 on Metal + 4090) belongs to riir-ai Plan 528 — deferred per Issue 659's ownership note. This bench records the primitive's correctness, cost, alloc, and memory gates.

## What shipped

- `BigramMarkovTable` — deterministic CSR top-`m` bigram transition table: `P(next | prev)` from corpus co-occurrence counts (Research 316 §3.5 path 2 — the modelless construction; no training).
- `BigramMarkovBuilder` — packed-u64 sort + two-pointer-pass build; `(count desc, next asc)` top-m; OOV pairs discarded; same corpus → bit-identical table.
- `bigram_predict` + `BigramMarginalBuffer` — zero-alloc greedy-chain marginal emission into the `dflash_predict_with` layout (`steps × vocab`, step-major), with touched-reset sparse writes (no O(vocab) per-cycle cost).
- `bigram_build_tree` — the `build_dd_tree` seam wiring (Issue 659 T3).

## GOAT

| Gate | Verdict | Evidence |
|---|---|---|
| G1 correctness | **PASS** | `g1_build_deterministic_bit_identical` (same corpus → identical table); `g1_top_m_matches_bruteforce_reference` — bit-equal to a HashMap brute-force reference at top_m ∈ {1,2,4,8}; `g1_oov_pairs_discarded`; `g1_tie_break_count_desc_then_token_asc`; `g1_row_offsets_sparse_prevs` (empty-row CSR boundaries). Emission semantics pinned by `g2_greedy_chain_follows_argmax_successor`, `g2_marginal_rows_sub_stochastic`, `g2_unseen_prev_emits_zero_row_and_chain_stalls` (the seam skips `prob <= 0` at every expansion site — zero rows propose nothing, verified end-to-end by `g3_build_tree_unseen_root_empty_tree`). |
| G2 cost (primitive axis) | **PASS** | `release_bonsai_scale_emission_cost_probe`: **181 ns/call = 23 ns/step** at vocab 131,072 / top_m 16 / lookahead 8, 1,000-call average (M3 Max, release, isolated target dir). Versus a separate 6-layer drafter forward at ~129 µs/step (Bench 661 E overhead scale) — **~5,600× cheaper per draft step**. This is Bench 656 failure-mode-2 avoidance made concrete: a table lookup does not pay a forward per step. |
| G3 no-regression | **PASS** | 305 lib tests default / 319 with `bigram_markov` (305 + 14 new), 0 failures. Clippy: zero diagnostics in the new file (the one `--all-targets` warning — `dd_tree/mod.rs` unused `HashMap` import — reproduces on the no-feature build; pre-existing, not this change). |
| G4 alloc-free | **PASS** | `g4_predict_zero_alloc_steady_state` — thread-local tracking `#[global_allocator]`, armed around the call: **0 allocations** steady state; `touched_len ≤ steps × top_m`. |
| G5 memory bound | **PASS** | `g5_memory_bytes_formula` — exact accounting pinned; worst case at Bonsai scale (`V=131,072`, `m=16`, every row full): **17,825,796 B ≈ 17 MB** vs dense `V×V` ~68 GB and DSpark low-rank `r=256` ~268 MB. |

## Honest caveats

- **G2 full (acceptance rate at equal draft depth) + G3 full (wall-clock on Metal AND 4090 with the Bonsai target) are NOT measured here** — they require the riir-ai Bonsai consumer (Plan 528) and the 4090 (currently occupied by a sibling's p336 bonsai parity run). The feature stays **opt-in** until that gate lands (Issue 659 T4).
- The probe corpus is synthetic xorshift with concentrated successor locality — the 23-touched count reflects that data; a real Bonsai corpus with full top_m rows costs proportionally more writes but stays O(steps × top_m).
- Corpus bigrams ≠ task-specific transitions (Research 316's own caveat): acceptance quality is a property of the corpus, not the primitive; that is the consumer gate's question.

## Commands

```bash
CARGO_TARGET_DIR=/tmp/katgpt-659-t13 cargo test -p katgpt-speculative --features bigram_markov --lib bigram_markov
CARGO_TARGET_DIR=/tmp/katgpt-659-t13 cargo test --release -p katgpt-speculative --features bigram_markov --lib release_bonsai_scale -- --nocapture
```
