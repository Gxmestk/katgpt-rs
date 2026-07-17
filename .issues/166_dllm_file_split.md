# Issue 166 — dllm.rs File Split (Issue 162 soft-limit band)

> **Source:** [Issue 162](../.issues/162_code_smell_audit.md) §"High-band soft-limit files"
> **Opened:** 2026-07-17
> **Type:** refactor (file-size hygiene)
> **Status:** DONE

---

## TL;DR

Split the monolithic `src/dllm.rs` (**3078 lines**, 50% over the 2048 `.rs`
soft limit from the global rule, under the 3200 hard cap) into a
`src/dllm/` module folder. The implementation stays in `mod.rs` (1924 lines,
under 2048 ✓); the test suite moved to `tests.rs` (1163 lines — tests are
exempt from the soft limit per Issue 162). The public API surface is
preserved 1:1 — every item accessible via `crate::dllm::*` before the split
resolves identically after. GOAT gate **G1 + G3 PASS**: 200/200 default lib
tests pass; clippy clean workspace-wide on default features.

This mirrors the Issue 164 (transformer.rs) and Issue 165 (dd_tree.rs) split
pattern: tests extraction is the minimal-mechanical split when the
implementation is cohesive (dLLM training is one research domain — noise
schedule + forward_save + backward + train loop + tests).

## Motivation

Issue 162 §"High-band soft-limit files" listed `src/dllm.rs` (3078 lines) as
the largest file in the 2048–3200 soft-limit band — explicitly low priority,
track-only, diminishing returns. This split closes it:

> Mixes noise schedule, forward_save, backward pass, training orchestration,
> and 1160 lines of tests in a single file. The implementation is cohesive
> (one research domain); the tests are the bulk of the size.

The file is dLLM (Discrete Diffusion Forcing) training infrastructure —
Plan 066 Phase 0 proof tasks. The forward INFERENCE substrate was already
moved to `katgpt-forward` (Plan 398); what remains in root `src/dllm.rs` is
the training loop + SGD backprop + noise schedule research code.

## Module layout

| File | Lines | Contents |
|------|-------|----------|
| `dllm/mod.rs` | 1924 | Implementation: `LossAveraging`, `NoiseSchedule`, `AdaptiveNoiseSchedule`, `corrupt_block*`, `ForwardActivations`, `ForwardSaveContext`, `TrainingGradients`, `BackwardContext`, `forward_save*`, `backward`, `rmsnorm_backward*`, `softmax_backward*`, `sgd_update`, `masked_loss*`, `evaluate_accuracy`, `generate_pattern_dataset`, `train_mini_dllm*`, `evaluate_set_causal_nelbo*` |
| `dllm/tests.rs` | 1163 | In-module test suite + `replaid_tests` submodule (gated `#[cfg(feature = "replaid_schedules")]`) |

`mod.rs` at 1924 lines is **under the 2048 soft limit** ✓. `tests.rs` at 1163
lines exceeds the soft limit but tests are explicitly exempt per Issue 162
("tests/benches/examples in the same band are lower priority").

## Design notes

- **Single-axis split (tests only):** unlike the transformer.rs split (Issue
  164) which split implementation along 7 functional axes, dllm's
  implementation is cohesive — it's one research domain (dLLM training) with
  tightly-coupled forward/backward/optimizer state. A functional split
  (noise/forward/backward/train) would scatter `ForwardSaveContext` fields
  across files and require awkward `pub(crate)` visibility for the 9
  cross-function private types. The test extraction is the principled minimal
  split.

- **Two test modules, one file:** the original had `mod tests` (always-on) +
  `mod replaid_tests` (gated `#[cfg(feature = "replaid_schedules")]`). Both
  moved to `tests.rs`. The outer `#[cfg(test)] mod tests;` in mod.rs gates
  the whole file; the inner `#[cfg(feature = "replaid_schedules")] mod
  replaid_tests { ... }` inside tests.rs preserves the feature gate.

- **Test path change (minor):** before the split, the replaid tests were at
  `dllm::replaid_tests::*` (sibling module). After the split, they're at
  `dllm::tests::replaid_tests::*` (nested). This doesn't affect correctness
  or CI filtering — the test names are unchanged, only the module path
  deepened by one level.

- **Import hygiene:** the test modules use `use super::*;` which brings in
  the parent module's items (including the top-level `use crate::types::*`
  imports). No changes needed — the pattern works identically with the
  extracted `tests.rs`.

## GOAT gate

| Gate | Target | Result |
|------|--------|--------|
| **G1** (correctness) | Bit-identical behavior: all existing tests pass | **PASS** — 200/200 default lib tests pass; 19/19 `--features dllm` tests pass; 24/24 `--features "dllm,replaid_schedules"` tests pass |
| **G3** (no-regression) | No clippy warnings, no compile errors caused by the split | **PASS** — `cargo clippy --lib` clean; `cargo clippy --features dllm --lib` clean; `cargo clippy --workspace` clean |

G2 (perf) and G4 (alloc-free) are **N/A** for a pure file split — no
computational change, no new allocations.

## Pre-existing failure (NOT caused by this split)

`cargo clippy --features dllm --lib --tests` surfaces one compile error in
`tests/issue_156_anytime_lt2_poc.rs` — a `forward_looped` argument count
mismatch in `src/transformer/variants.rs:158`. This is **pre-existing and
unrelated** — verified by `git diff --name-only HEAD` showing only
`src/dllm.rs` was touched. The error references `transformer/variants.rs`,
not `dllm`. Likely sibling-agent WIP on `forward_looped`'s signature
(the `elastic_override` parameter).

## Validation

```
cargo clippy --lib                                   → clean
cargo clippy --features dllm --lib                   → clean
cargo clippy --workspace                             → clean
cargo test --lib                                     → 200/200 PASS
cargo test --features dllm --lib dllm::              → 19/19 PASS
cargo test --features "dllm,replaid_schedules" --lib dllm::  → 24/24 PASS
```

Hardware: Apple M3 Max (aarch64, NEON).

## References

- [Issue 162](../.issues/162_code_smell_audit.md) — the code-smell audit
  that flagged the high-band soft-limit files.
- [Issue 164](../.issues/164_transformer_file_split_c1.md) — transformer.rs
  split (the pattern this mirrors).
- [Issue 165](../.issues/165_dd_tree_file_split_c2.md) — dd_tree.rs split
  (same pattern).
- Plan 066 — D2F Discrete Diffusion Forcing (the original research code).
- Plan 398 — forward substrate moved to `katgpt-forward` (the remaining root
  `dllm.rs` is training-only).
