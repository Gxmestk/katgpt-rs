# Issue 164 — Transformer.rs File Split (Issue 162 C1)

> **Source:** [Issue 162](../.issues/162_code_smell_audit.md) §C1 (Critical — >3200-line hard-limit violation)
> **Opened:** 2026-07-17
> **Type:** refactor (file-size hygiene)
> **Status:** DONE (T1–T3 complete)

---

## TL;DR

Split the monolithic `src/transformer.rs` (**5672 lines**, 77% over the 3200 hard
limit) into a `src/transformer/` module folder with 8 sub-modules. The public API
surface is preserved 1:1 — every item accessible via `crate::transformer::*` before
the split resolves identically after. GOAT gate **G1 + G3 PASS**: 200/200 default
tests pass bit-identically; clippy clean workspace-wide on default + `--all-features`.

The 3 pre-existing `--all-features` test failures
(`proof_qkv_interleave_forward`, `test_forward_paged_logits_match_forward`,
`test_no_lora_matches_existing_forward`) are **NOT caused by this split** — verified
by running the same 3 tests against the pre-split `transformer.rs` and observing
identical failure messages + numerical values.

## Motivation

Issue 162 §C1 flagged `src/transformer.rs` as a Critical file-size violation:

> Mixes RiM slots, forward passes (7 variants), generators, raven router, depth
> routing, paged + quantized paths in a single file. The top re-export block
> (lines 1–80) already shows the public API is a façade — the split is mechanical.

The file was already mostly a re-export façade over `katgpt_transformer` +
`katgpt_forward` crates, with ~2870 lines of composition-layer forward variants +
~2800 lines of tests. The split is purely mechanical — no logic changes.

## Module layout

| Module | Lines | Contents |
|--------|-------|----------|
| `mod.rs` | 173 | Façade: re-exports from `katgpt_transformer` / `katgpt_forward` + RiM helpers + module declarations |
| `variants.rs` | 571 | `forward_batched`, `forward_with_domain_latent`, `forward_looped` |
| `tf_loop.rs` | 488 | `forward_training_free_loop`, `depth_route_weights` (+ private helpers `forward_single_layer`, `depth_route`) |
| `prefill.rs` | 428 | `forward_prefill` |
| `raven.rs` | 430 | `tokens_to_string`, `raven_compute_router*`, `raven_update`, `raven_readout*`, `forward_raven` |
| `generators.rs` | 389 | `generate_with_prefill`, `generate_with_prefill_and_domain_latent`, `generate_with_collapse_detection`, `generate_into`, `generate`, `generate_batch` |
| `quantized.rs` | 234 | `forward_quantized`, `forward_turboquant` |
| `paged.rs` | 223 | `forward_paged` |
| `tests.rs` | 2799 | In-module test suite (2799 lines — tests exempt from soft limit) |

All implementation files are **well under the 2048 soft limit** (largest: 571 lines).
`tests.rs` at 2799 lines exceeds the soft limit but tests are explicitly exempt per
Issue 162 ("tests/benches/examples in the same band are lower priority").

## Design notes

- **Naming collision resolved:** the original plan named the batched-forward module
  `forward.rs`, but `forward` (the function, re-exported from `katgpt_forward`)
  already occupies that name in mod.rs's namespace. Renamed to `variants.rs` to
  avoid the `E0255: name defined multiple times` error.

- **Private helper visibility:** two private helpers moved with their callers:
  - `forward_single_layer` → stays private in `tf_loop.rs` (only called by
    `forward_training_free_loop` in the same file; not referenced by tests)
  - `depth_route` → made `pub(crate)` in `tf_loop.rs` + re-exported as
    `#[cfg(all(feature = "delta_routing", test))] pub(crate) use tf_loop::depth_route;`
    in mod.rs (exercised by the norm-stability test; dual-gated so non-test
    builds never see an unused-import warning)

- **Import hygiene:** `cargo fix --lib` auto-resolved 9 redundant import warnings.
  Sub-modules that use bare `types::Foo` syntax (`paged`, `quantized`, `raven`,
  `tf_loop`) retain `use crate::types::{self};`; sub-modules that only use glob-
  imported types (`variants`, `generators`, `prefill`) dropped the explicit import
  in favor of `use super::*;`.

## GOAT gate

| Gate | Target | Result |
|------|--------|--------|
| **G1** (correctness) | Bit-identical behavior: all existing tests pass | **PASS** — 200/200 default lib tests pass; 518/521 `--all-features` tests pass (3 failures are pre-existing, confirmed against pre-split code) |
| **G3** (no-regression) | No clippy warnings, no compile errors | **PASS** — `cargo clippy --lib` clean; `cargo clippy --all-features --lib --tests` clean; `cargo clippy --workspace` clean |

G2 (perf) and G4 (alloc-free) are **N/A** for a pure file split — no computational
change, no new allocations.

## Pre-existing failures (NOT caused by this split)

Three tests fail under `--all-features` both before and after the split, with
identical numerical values:

| Test | Failure message | Pre-split line | Post-split line |
|------|----------------|----------------|-----------------|
| `proof_qkv_interleave_forward` | `QKV interleave mismatch at logit[0]: sep=6.4559646, fused=-1.3949726, diff=7.850937` | `transformer.rs:5538` | `transformer/tests.rs:2666` |
| `test_no_lora_matches_existing_forward` | `forward and forward_base(None) differ at 8: 0.000007867813` | `transformer.rs:4207` | `transformer/tests.rs:1335` |
| `test_forward_paged_logits_match_forward` | `forward_paged logit 1 differs: 15.317682 vs 15.317859` | `transformer.rs:3549` | `transformer/tests.rs:677` |

These are feature-combo numerical interactions (likely `--all-features` enabling
mutually-exclusive attention variants), not structural failures. Tracked separately
from this refactor.

## Validation

```
cargo clippy --lib                                   → clean
cargo clippy --all-features --lib --tests            → clean
cargo clippy --workspace                             → clean
cargo test --lib                                     → 200/200 PASS
cargo test --all-features --lib                      → 518/521 PASS (3 pre-existing)
```

Hardware: Apple M3 Max (aarch64, NEON).
