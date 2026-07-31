# Plan 435 — GDN Tree Spec Step Weaver Call-Site Wiring

**Status:** done
**Started:** 2026-07-14
**Completed:** 2026-07-14
**Prerequisite:** Plan 434 (QwenDeltaNet Weaver wiring) — DONE
**Issue:** none (feature wiring, not optimization/poc/refactor)

## Goal

Wire the Weaver corrector into the GDN tree speculative step variants, mirroring
the Plan 434 pattern applied to `speculative_step_qwen_deltanet_tree`.

The prior session wired Weaver into `speculative_step_qwen_deltanet_tree` (Plan
434, commit `eca5dec2`) and `dflash_predict_with_weaver` (Plan 433). The GDN
tree variants (`speculative_step_gdn_tree`, `speculative_step_gdn_hola_tree`)
still call the uncorrected `dflash_predict_with`. This plan adds the
`_with_weaver` siblings.

## Background

### Weaver contract recap

`dflash_predict_with_weaver` takes:
- `h_dflash_captured` — drafter's per-step hidden states (written by DFlash)
- `h_verifier` — verifier's preserved hidden state from the prior commit
- `embedding` — token embedding matrix `[vocab_size, n_embd]` row-major
- `weaver` — trained corrector weights
- `scratch` — pre-allocated Weaver scratch

**No-harm contract:** zero Weaver weights → zero residual → marginals
bit-identical to the base path. This means the wiring is safe even before a
GDN-2-specific Weaver checkpoint is trained.

### Where `h_verifier` comes from in each variant

| Variant | Source | Populated by |
|---|---|---|
| QwenDeltaNet (Plan 434) | `target_scratch.hidden_copy[..n_embd]` | `forward_tree_qwen_deltanet` line 584-595 (copy before lm_head) |
| GDN tree (this plan) | `target_ctx.hidden_state[..n_embd]` | `forward_gdn2` line 252 (`ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n])`) |
| GDN HOLA tree (this plan) | `target_ctx.hidden_state[..n_embd]` | same `forward_gdn2` commit path |

**Cold start:** `hidden_state` is zero-init. Weaver no-harm contract handles
zero hidden → zero residual. No special-case needed.

## Tasks

- [x] T1: Extract shared post-draft pipeline into `gdn_tree_post_verify` helper
      (DRY — the marginals-view → DDTree → topology → tree forward → p/q reject
      → commit block is duplicated between `speculative_step_gdn_tree` and
      `speculative_step_gdn_hola_tree`). Both base functions delegate to it.
      Also extracted `build_marginals_view` for the shared marginals slicing.
- [x] T2: Add `speculative_step_gdn_tree_with_weaver` — gated behind
      `weaver_runtime`. Calls `dflash_predict_with_weaver` at step 1, then
      delegates to `gdn_tree_post_verify` with `forward_tree_gdn2`.
- [x] T3: Add `speculative_step_gdn_hola_tree_with_weaver` — gated behind both
      `weaver_runtime` AND `gdn_hola_tree_verify`. Calls
      `dflash_predict_with_weaver` at step 1, then delegates to
      `gdn_tree_post_verify` with `forward_tree_gdn2_hola`.
- [x] T4: Unit tests:
  - [x] T4.1: `test_speculative_step_gdn_tree_with_weaver_no_harm` — zero
        Weaver weights produce identical output to base path
  - [x] T4.2: `test_speculative_step_gdn_tree_with_weaver_returns_tokens` —
        non-zero Weaver weights still return ≥1 accepted token
  - [x] T4.3: `test_speculative_step_gdn_tree_with_weaver_cold_start` — fresh
        `hidden_state` (zeros) does not panic
  - [-] T4.4: `test_speculative_step_gdn_hola_tree_with_weaver_no_harm` —
        DEFERRED: the base `speculative_step_gdn_hola_tree` has no tests either
        (HOLA requires hippocampal cache setup). The HOLA-Weaver variant
        compiles clean with all features ON; test deferred until HOLA tests
        are added for the base variant.
- [x] T5: Validate — `cargo clippy` with `weaver_runtime` ON and OFF (G3
      no-regression), `cargo test` for the new tests — ALL PASS
- [x] T6: Added `weaver_runtime` feature forwarding to root `katgpt-rs`
      `Cargo.toml` (was missing — the root crate didn't forward the feature
      to `katgpt-forward`/`katgpt-speculative`, so the cfg gates in
      `step_gdn_tree.rs` would never activate)
- [x] T7: Commit on `develop` with `feat:` prefix

## Design decisions

### Why extract `gdn_tree_post_draft` (T1) — DRY

The post-draft pipeline (steps 2-7 in both functions) is ~150 lines of
identical code duplicated between `speculative_step_gdn_tree` and
`speculative_step_gdn_hola_tree`. Adding two more `_with_weaver` variants
without extracting would create 4 copies. The extraction is a prerequisite for
clean Weaver wiring.

The helper takes the tree-forward function as a parameter (or an enum flag) to
handle the GDN vs GDN-HOLA split. The cleanest approach: the helper takes the
already-computed `tree_logits: Vec<f32>` as input, and each caller runs its own
tree forward before delegating. This keeps the helper forward-agnostic.

### Feature gating

- `speculative_step_gdn_tree_with_weaver` — `#[cfg(feature = "weaver_runtime")]`
  (same as the QwenDeltaNet sibling)
- `speculative_step_gdn_hola_tree_with_weaver` —
  `#[cfg(all(feature = "weaver_runtime", feature = "gdn_hola_tree_verify"))]`
  (both gates required)

### Why this is NOT modelless-promotable

Same as Issue 131 §"Why this is NOT modelless-promotable": the Weaver adapter
is a trained artifact. The `weaver_runtime` feature stays opt-in permanently.
This plan adds call-site wiring, not a new primitive — no GOAT gate needed.

## Verification

```bash
# G3 no-regression: clippy with weaver_runtime OFF (default features)
cargo clippy -p katgpt-rs --lib

# Clippy with weaver_runtime ON
cargo clippy -p katgpt-rs --lib --features weaver_runtime

# New tests
cargo test -p katgpt-rs --lib --features weaver_runtime -- speculative::step_gdn_tree

# HOLA variant tests (needs both features)
cargo test -p katgpt-rs --lib --features weaver_runtime,gdn_hola_tree_verify -- speculative::step_gdn_tree
```
