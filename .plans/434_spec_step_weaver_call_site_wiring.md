# Plan 434 — `speculative_step_qwen_deltanet_tree` Weaver Call-Site Wiring

**Status:** DONE ✅
**Started:** 2026-07-14
**Completed:** 2026-07-14
**Repo:** `riir-ai` (primary), `katgpt-rs` (doc updates only)
**Predecessor:** Plan 433 (Weaver ↔ DFlash pipeline wiring — DONE)
**Issue:** Follow-up to `katgpt-rs/.issues/131` (Weaver Runtime Integration, 7/7 PASS)

## Goal

Wire `dflash_predict_with_weaver` (the Plan 433 building block) into the
actual production spec decode hot path (`speculative_step_qwen_deltanet_tree`
in `riir-ai/crates/katgpt-attn/crates/katgpt-attn/src/gdn2/tree_forward.rs`). After this
plan, callers can opt into Weaver marginal correction for the DeltaNet
spec tree by calling the new sibling variant instead of the base function.

**Non-goal:** modifying the base `speculative_step_qwen_deltanet_tree`
signature (zero churn for the 3 existing tests + any external callers).

## Design — sibling variant + helper extraction (no DRY violation)

The base function is ~185 lines (805-990). Only the first line differs
between the weaver and non-weaver paths:

```
line 823:  dflash_predict_with(...)          // base
line 823': dflash_predict_with_weaver(...)   // weaver — same sctx mutation
```

After that, both paths consume `sctx.marginals_flat` + `sctx.steps_populated`
identically (build marginals view → DDTree → tree forward → p/q reject →
commit). Extract lines 824-989 into a private helper
`spec_step_deltanet_post_draft(sctx, tree_builder, target_weights, ...)`
and have both the base and weaver variants delegate to it.

This mirrors the established sibling-variant pattern from
`katgpt-forward/crates/katgpt-forward/src/step.rs` (`speculative_step` / `_rollback` /
`_rollback_with` / `_conditioned_with` / `_with_configurator`) and Plan 433
(`dflash_predict_with` / `_with_capture` / `_with_weaver`).

## Weaver input source mapping

| Weaver input | Source in spec step | Notes |
|---|---|---|
| `h_verifier` | `&target_scratch.hidden_copy[..target_config.n_embd]` | Populated by prior `forward_qwen_deltanet` (commit step) as the aliasing-avoidance copy. Zeros on cold-start — fine per Weaver no-harm contract. |
| `embedding` | `&target_weights.wte` | `[vocab_size, n_embd]` row-major — matches Weaver's expected layout. |
| `h_dflash` | `h_dflash_captured` (caller-allocated `[max_steps * n_embd]`) | Filled by `dflash_predict_with_capture` inside the weaver wrapper. |
| `vocab_size` | `draft_config.vocab_size` | Must equal `target_config.vocab_size` (Qwen3.5-DeltaNet has tied embedding). |
| `scratch` | Caller-provided `&mut WeaverScratch` | Allocated once per session. |

**Constraint:** Weaver `hidden_dim` must equal `target_config.n_embd`. The
drafter's `n_embd` must also match (else `h_dflash` slices have wrong width).
This holds for the Qwen3.5-DeltaNet + standard-transformer-drafter pairing.

## Tasks

- [x] **T1 — Extract `spec_step_deltanet_post_draft` helper** in
  `tree_forward.rs`. Moves lines 824-989 (everything after the
  `dflash_predict_with` call) into a private `fn`. Base function becomes:
  draft → call helper. No behavior change. All 3 existing tests must still
  pass.

- [x] **T2 — Add `speculative_step_qwen_deltanet_tree_with_weaver` sibling**
  gated `weaver_runtime`. Takes the base params + `h_dflash_captured:
  &mut [f32]`, `weaver: &WeaverCorrector`, `weaver_scratch: &mut
  WeaverScratch`. Body: extract `h_verifier` + `embedding`, call
  `dflash_predict_with_weaver`, delegate to helper.

- [x] **T3 — Feature-gate import + visibility.** Ensure `WeaverCorrector`,
  `WeaverScratch`, `dflash_predict_with_weaver` are reachable under
  `weaver_runtime`. The feature already exists in `riir-engine/Cargo.toml`
  (Plan 433 T6).

- [x] **T4 — Unit test: weaver variant matches base with zero-weight weaver.**
  Build a zero-weight `WeaverCorrector` (K > V early-return path — same as
  the Plan 433 zero-weight test). Run both variants on the same
  (draft, target, token, pos). Assert: identical `accepted` vec OR identical
  marginal shape (zero-weight weaver preserves marginals bit-identically per
  the no-harm contract; the RNG-driven accept/reject is deterministic with
  the same seed, so `accepted` should match exactly).

- [x] **T5 — Unit test: weaver variant runs end-to-end without panic on
  cold-start.** First-call case: `target_scratch.hidden_copy` is zeros.
  Weaver must early-return (K > V) or produce finite corrected marginals.
  Assert: at least one token accepted, all finite.

- [x] **T6 — `cargo clippy` clean with `weaver_runtime` on AND off.** G3
  no-regression on the feature-off path.

- [x] **T7 — Update `katgpt-rs/.issues/131`** with a note that the spec-step
  call-site wiring (the last deferred Plan 433 follow-up) has landed in
  riir-ai.

- [x] **T8 — Commit on `riir-ai/develop`** with `feat: Plan 434 — spec step
  Weaver call-site wiring (speculative_step_qwen_deltanet_tree_with_weaver)`.
  Update `.plans/.highwater` 433 → 434.

- [x] **T9 — Mark this plan DONE** with commit hash + DONE stamp.

## Out of scope

- **Modifying the base function signature.** Zero churn for existing callers.
- **Wiring other spec step variants** (`speculative_step_gdn_tree` in
  katgpt-rs, `speculative_step_gdn_hola_tree`). Those use GDN-2 verifier, not
  QwenDeltaNet — different hidden-state plumbing. Separate plans if pursued.
- **GOAT gate / promotion to default-on.** The `weaver_runtime` feature stays
  opt-in. Promotion requires a benchmarked modelless gain on a real workload
  (per AGENTS.md feature-flag discipline). This plan wires the call site; a
  future benchmark plan would gate promotion.
- **GPU port** (Issue 131 G4). CPU-side wiring only here.

## Risks

1. **`hidden_copy` vs `hidden`.** `scratch.hidden` is a transient MLP buffer
   (overwritten each layer); `scratch.hidden_copy` is the preserved copy made
   before the lm_head matmul. Must read from `hidden_copy`, not `hidden`.
   T1's tests will catch a misread (marginals would be garbage).

2. **Cold-start zeros.** On the first spec step (no prior commit),
   `hidden_copy` is zero-initialized. Weaver's no-harm contract handles this
   (zero hidden → zero residual → marginals unchanged). T5 verifies.

3. **Feature-off regression.** The base function is untouched; only the new
   sibling variant is gated. T6 verifies clippy clean both ways.
