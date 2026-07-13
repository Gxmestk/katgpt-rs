# Plan 433 — Weaver ↔ DFlash Pipeline Wiring

> **Date:** 2026-07-13
> **Spawned from:** Issue 131 (katgpt-rs) — runtime half complete, this is the
> "remaining riir-ai integration" the issue's TL;DR calls out.
> **Scope:** make Weaver correction transparent when `weaver_runtime` is on by
> providing a wrapper that combines DFlash predict + Weaver correct_marginals.

## Context

`WeaverCorrector::correct_marginals_with_scratch` is implemented and tested
(Issue 131 T3, G1-G4). Callers currently must invoke it explicitly after
`dflash_predict_with`. This plan adds a single-call wrapper so the spec decode
loop can opt into Weaver correction by passing `Some(&weaver_corrector)`.

### The blocker: hidden-state capture

`dflash_predict_with` overwrites `ctx.hidden_state` on each `forward_fn` call.
Weaver needs the per-step drafter hidden states `h_dflash[]` (D slices). We must
capture them during the dflash loop. Two options considered:

- **α: Modify `dflash_predict_with` to optionally capture.** Touches every
  caller signature (8+ sites across two repos).
- **β: Add a sibling `dflash_predict_with_capture` variant.** Zero churn for
  existing callers; the new function is opt-in.

**Decision: β.** Non-invasive, follows the existing `_ar_with` / `_conditioned_with`
/ `_with_fusion` sibling-variant pattern, and keeps the hot path unchanged when
Weaver is off.

## Tasks

- [x] **T1 (katgpt-speculative):** Add `hidden_state_slice()` to `DflashCtx` trait
  + impls in katgpt-forward and riir-engine.
- [x] **T2 (katgpt-speculative):** Add `dflash_predict_with_capture` sibling variant
  that captures per-step `hidden_state` into a caller-provided flat buffer.
- [x] **T3 (katgpt-forward unit test):** `dflash_predict_with_capture` matches
  `dflash_predict_with` marginals bit-for-bit + populates `h_dflat_captured`.
- [x] **T4 (katgpt-forward):** Add `dflash_predict_with_weaver` wrapper gated on
  `weaver_runtime` — runs `_capture`, builds `&[&[f32]]` view into
  `h_dflat_captured`, calls `WeaverCorrector::correct_marginals_with_scratch`.
- [x] **T5 (katgpt-forward unit test):** zero-weight Weaver is a no-op
  (marginals unchanged modulo the top-K truncation/renorm, bit-equiv on a
  vocab ≤ K test where truncation is a no-op); trained-shape Weaver runs
  without panic and preserves G1 (sum-to-1, finite).
- [x] **T6 (riir-engine mirror):** Mirror T1+T2+T4 wrappers in
  `riir-ai/crates/riir-engine/src/dflash.rs` (`_weaver` variant gated on
  `weaver_runtime` feature added to riir-engine's Cargo.toml).
- [x] **T7 (riir-engine unit test):** Mirror T5.
- [x] **T8 (clippy clean):** `cargo clippy -p katgpt-speculative`,
  `cargo clippy -p katgpt-forward --features weaver_runtime`,
  riir-engine equivalent. Fix before commit.
- [x] **T9 (G3 no-regression):** `weaver_runtime` OFF still compiles + tests pass
  in both repos.
- [x] **T10 (commit):** `docs:` for the plan, `feat:` for the wiring.
  - katgpt-rs develop: `1e92a496 feat: Plan 433 — Weaver ↔ DFlash pipeline wiring (dflash_predict_with_weaver)`
  - riir-ai develop:    `e46a1f06 feat: Plan 433 — mirror Weaver ↔ DFlash pipeline wiring in riir-engine`

## Out of scope (explicitly deferred)

- **Wiring into `speculative_step_qwen_deltanet_tree`** — that requires
  passing verifier hidden state + embedding + corrector through the spec step
  signature, which is a bigger surface change. Tracked as a follow-up; this
  plan only provides the building-block wrapper that such wiring would call.
- **GPU port** — Issue 131 G4 follow-up.
- **f16 weights** — Issue 131 future.

## TL;DR

Add `dflash_predict_with_capture` (hidden-state-capturing variant) to the
shared core, then a `_weaver` wrapper at the katgpt-forward and riir-engine
layers that combines predict + correct_marginals. Zero-churn for existing
callers; opt-in via the `weaver_runtime` feature.

**DONE 2026-07-14** — all 10 tasks complete. Both repos committed on
`develop` (not pushed).
