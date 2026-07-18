# Plan 213: BFCF Tree — GOAT Proof

**Date:** 2026-06-08
**Feature Gate:** `bfcf_tree` — transitively default-on (chain: `bfcf_lsh_cms` [in `default`] → `bfcf_lfu_shard` → `bfcf_tree`, per Issue 181). Originally OPT-IN/GOAT-gated; promoted transitively when `bfcf_lsh_cms` (Plan 220) landed default-on.

> **UPDATE 2026-07-18 (status sync):** the previous label said `bfcf_tree`
> — OPT-IN, GOAT-gated. That was accurate at Plan 213 time but went stale
> once `bfcf_lsh_cms` (Plan 220) was promoted to default-on with
> `bfcf_lsh_cms = ["bfcf_lfu_shard", ...]` and `bfcf_lfu_shard = ["bfcf_tree", ...]`.
> The standalone feature is still exposed so `--no-default-features`
> consumers can disable the chain.

## GOAT Gate Matrix

| Gate | Criterion | Status |
|------|-----------|--------|
| G1 | Region pruning correctness | ✅ PASS |
| G2 | PWC closure maintained after 100 updates | ✅ PASS |
| G3 | Percept routing ≥ 95% accuracy | ✅ PASS |
| G4 | Preimage improvement ≥ 10% | ✅ PASS |
| G5 | Zero perf hurt when disabled | ✅ PASS |
| G6 | Feature isolation / sigmoid bounded | ✅ PASS |

## Test Coverage

| Test | Gate | File |
|------|------|------|
| `goat_region_pruning_correctness` | G1 | `tests/bfcf_tree_goat.rs` |
| `goat_pwc_closure_after_n_updates` | G2 | `tests/bfcf_tree_goat.rs` |
| `goat_percept_routing_accuracy` | G3 | `tests/bfcf_tree_goat.rs` |
| `goat_preimage_improves_acceptance` | G4 | `tests/bfcf_tree_goat.rs` |
| `goat_feature_isolation_empty_inputs` | G5 | `tests/bfcf_tree_goat.rs` |
| `goat_complexity_sigmoid_bounded` | G6 | `tests/bfcf_tree_goat.rs` |

## Percept Router Tests (Phase 4)

| Test | File |
|------|------|
| `test_complexity_low_for_simple_partition` | `crates/katgpt-pruners/src/percept_router.rs` |
| `test_complexity_high_for_complex_partition` | `crates/katgpt-pruners/src/percept_router.rs` |
| `test_route_fast_for_simple` | `crates/katgpt-pruners/src/percept_router.rs` |
| `test_route_deep_for_complex` | `crates/katgpt-pruners/src/percept_router.rs` |
| `test_route_standard_for_medium` | `crates/katgpt-pruners/src/percept_router.rs` |
| `test_complexity_bounded_unit_interval` | `crates/katgpt-pruners/src/percept_router.rs` |
| `test_entropy_of_uniform_labels` | `crates/katgpt-pruners/src/percept_router.rs` |

## Expected Gains

| Metric | Before (token-by-token) | After (BFCF Tree) |
|--------|------------------------|-------------------|
| Evaluations per step | O(vocab_size ≈ 128K) | O(regions ≈ 50) |
| Routing accuracy | Fixed threshold | ≥ 95% measurable |

## Decision: OPT-IN

Feature stays behind `bfcf_tree` flag. Needs real inference benchmark before promotion to default.
