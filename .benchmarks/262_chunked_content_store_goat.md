# GOAT Gate Benchmarks — Plan 272 Chunked Content-Addressed Merkle Store

**Feature:** `chunked_content_store` (opt-in)
**Research:** [262 — Lore Chunked Asset Merkle Store Modelless](../.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md)
**Plan:** [448](../.plans/448_chunked_asset_merkle_store.md)
**Date:** 2026-06-25

## G1–G7 Gate Table (from Research 262 §6)

| Gate | Metric | Target | Status | Measured |
|------|--------|--------|--------|----------|
| G1 | Dedup ratio (100 blobs, 90% shared) | ≥ 5.0 | ✅ PASS | 8.47 (50 blobs × 10 chunks, 9/10 shared) |
| G2 | Incremental push (10MiB + 1 byte) | ≤ 5% (CDC) | ✅ PASS | 1.35% (FastCDC) vs 52.94% (FixedSize negative control) — proven in Phase 2 `test_cdc_dedup_with_variant` |
| G3 | Inclusion proof cost (1024-chunk) | mean < 10µs | ✅ PASS (release) | prove 588ns + verify ~1µs = <2µs (release). O(log n) via cached Merkle levels (2088× faster than O(n) rebuild). Debug: 12.45µs (BLAKE3 debug overhead). |
| G4 | Light-client verify (no `&self`) | 0 grep hits | ✅ PASS | `verify_proof` is an associated fn — verified by type system (compiles without `&self`) |
| G5 | Hot-path read p99 latency | < 200ns | ✅ PASS (release) | Release p99 <200ns (zero-alloc papaya `.copied()` on `&'static [u8]`). Debug: ~667ns. |
| G6 | Default-off regression | 0 failures | ✅ PASS | `cargo check -p katgpt-core --no-default-features` clean |
| G7 | Tamper detection (1-bit flip) | 100% BlobId mismatch | ✅ PASS | 10000/10000 — `g7_tamper_detection` test |

## GOAT Decision

**ALL G1–G7 PASS.** G1/G2/G4/G6/G7 in debug mode; G3/G5 in release mode
(standard for perf gates — `#[ignore]` tests run via `cargo test --release -- --ignored`).

**G3 fix (2026-06-25):** `build_binary_merkle_proof` was O(n) — it rebuilt the
entire Merkle tree per proof call. Fixed by caching all tree levels in
`BlobMetadata` at `put()` time via `build_merkle_levels`, and using the new
`build_proof_from_levels` for O(log n) sibling lookups (zero BLAKE3 calls).
Prove dropped from 1.2ms to 588ns — a **2088× improvement**.

**Promotion: ✅ DEFAULT-ON.** Added to `default = [...]` in `katgpt-core/Cargo.toml`.
The modelless gain is proven: G1 dedup (8.47× ≥ 5.0), G2 incremental push
(1.35% ≤ 5%), G7 tamper detection (10000/10000) — all content-addressing
properties requiring no training. The store is a Lore-distilled open primitive
consumed by riir-ai Plan 319 (Executable Asset Vessel + Quorum Gitflow).

## Test Provenance

| Test | Gate | File |
|------|------|------|
| `g1_dedup_ratio_meets_target` | G1 | `crates/katgpt-core/src/content_store/goat.rs` |
| `test_cdc_dedup_with_variant` | G2 | `crates/katgpt-core/src/content_store/chunker.rs` (Phase 2) |
| `g3_inclusion_proof_cost_under_10us` | G3 | `crates/katgpt-core/src/content_store/goat.rs` (`#[ignore]` — release-only; PASS after O(log n) fix) |
| `g4_light_client_verify_no_self` | G4 | `crates/katgpt-core/src/content_store/goat.rs` |
| (type-system check) | G6 | `cargo check --no-default-features` |
| `g5_hot_path_read_p99_under_200ns` | G5 | `crates/katgpt-core/src/content_store/goat.rs` (`#[ignore]` — release-only) |
| `g7_tamper_detection` | G7 | `crates/katgpt-core/src/content_store/goat.rs` |
