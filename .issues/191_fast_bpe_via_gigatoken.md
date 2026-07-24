# Issue 191: Fast BPE via Gigatoken — `fast_bpe` Feature Flag + GOAT Gate

**Date:** 2026-07-25
**Research:** [456 — Gigatoken SIMD Pretokenization + Cache Hierarchy](../.research/456_Gigatoken_SIMD_Pretokenization_Cache_Hierarchy.md)
**Source:** [gigatoken](https://github.com/marcelroed/gigatoken) (Marcel Rød, MIT, ~2.5k★) — ~1000× faster than HF `tokenizers`
**Target crate:** `crates/katgpt-tokenizer/` (the BPE substrate that would benefit)
**Verdict:** Gain, GOAT candidate pending gate
**Status:** Open — Phase 0 (decision: dep vs port)

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

- [ ] **T1.1** Verify gigatoken builds standalone: `cargo build --manifest-path /tmp/probe/Cargo.toml` with `gigatoken = { git = "https://github.com/marcelroed/gigatoken" }`. If fails → fall back to option 2 or option 3.
- [ ] **T1.2** Verify gigatoken's pure-Rust core is separable from Python bindings (the codebase is pure Rust + wasm32; Python binding surface would be a leak).
- [ ] **T1.3** Verify wasm32-unknown-unknown compatibility (or document the wasm32 fallback to existing `bpe.rs`).
- [ ] **T1.4** Add `fast_bpe = ["dep:gigatoken"]` feature to `crates/katgpt-tokenizer/Cargo.toml`. Feature gates the gigatoken dep.
- [ ] **T1.5** Add `BpeTokenizerImpl::encode_fast(&self, text: &str) -> Vec<usize>` (and `encode_fast_batch`) under `#[cfg(feature = "fast_bpe")]` in `crates/katgpt-tokenizer/src/bpe.rs`. Delegates to gigatoken with the tokenizer's vocab/merges.
- [ ] **T1.6** Re-export `fast_bpe` from root `katgpt-rs` feature surface (`Cargo.toml [features]`): `fast_bpe = ["katgpt-tokenizer/fast_bpe"]`.

---

## Phase 2 — GOAT gate

### Tasks

- [ ] **T2.1 (G1 correctness)** Add `tests/fast_bpe_goat.rs::g1_bit_identical_to_hf` — encode 22 tokenizer vocabularies × 10MB sample from `owt_train.txt` (or equivalent), assert bit-identical token-id sequences between `encode()` (existing) and `encode_fast()` (gigatoken-backed). Reuse gigatoken's published validation corpus if license-compatible.
- [ ] **T2.2 (G2 perf)** Add `benches/bench_fast_bpe.rs` — criterion bench: `encode()` vs `encode_fast()` on 1MB / 100MB / 1GB samples. **Gate floor: ≥100× on the 100MB sample.** (Gigatoken publishes 1000×; we accept 100× to leave integration-overhead headroom.) Measure on whatever CPU the dev runs (Apple M-series or AMD x86).
- [ ] **T2.3 (G3 no-regression)** Run `cargo test -p katgpt-tokenizer --all-features` — all existing BPE / ToaST / ConvexTok tests pass (the new path is feature-gated; the existing `bpe.rs::encode` is untouched).
- [ ] **T2.4 (G4 alloc-free)** Add `tests/fast_bpe_goat.rs::g4_zero_alloc_steady_state` — `CountingAllocator` audit: 0 allocations in 100 steady-state `encode_fast()` calls after warmup (gigatoken claims this; we verify).
- [ ] **T2.5** Record results in `.benchmarks/191_fast_bpe_goat.md`.

---

## Phase 3 — Promote to default (only if G1–G4 PASS)

### Tasks

- [ ] **T3.1** If G1–G4 pass: move `fast_bpe` from opt-in to the `default` array in `crates/katgpt-tokenizer/Cargo.toml`. Update the katgpt-tokenizer README's feature table.
- [ ] **T3.2** Demote the existing `bpe.rs::encode` (slow path) to a `#[cfg(not(feature = "fast_bpe"))]` fallback OR delete it if `fast_bpe` becomes always-on. Keep it as the wasm32 fallback if T1.3 found gigatoken is wasm32-incompatible.
- [ ] **T3.3** Update root `katgpt-rs/Cargo.toml` default features if appropriate, and the root README's "Input Layer" section to note GB/s tokenization.
- [ ] **T3.4** doc-sync: update `.docs/` references to BPE throughput.

If G1–G4 FAIL: keep `fast_bpe` opt-in, document which gate failed and why in `.benchmarks/191_fast_bpe_goat.md`, close this issue with the verdict.

---

## Cross-cutting follow-up (out of scope, flag only)

- The **pretoken cache hierarchy** technique (gigatoken's hardest piece) is structurally the same as Engram's `ZipfianCacheHierarchy` (Plan 299 P6) and `riir-neuron-db::ItemEmbedIndex`. If the port (option 2) or even the dep integration (option 1) reveals a long-tail cache-growth-management trick worth retroactively porting, open a separate issue for that. Don't scope-creep this issue.
- If `riir-data` or `riir-train` later lands a streaming-corpus pipeline that needs GB/s BPE, this issue becomes its unblocker. Tag the dependency when that pipeline issue opens.

---

## Numbering note

Per AGENTS.md monotonic-numbering rule: 191 was `value + 1` from `.issues/.highwater = 190`. Bumped `.highwater` to 191 in the same commit as this file.
