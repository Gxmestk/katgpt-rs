# Issue 162 — Code Smell Audit — file size violations + stale TODOs

> **Source:** Code-smell audit pass (file-size sweep + stale TODO grep + softmax review).
> **Opened:** 2026-07-17
> **Type:** Audit / refactor task (per AGENTS.md — issue, not plan)
> **Verdict that opened it:** Hygiene gate. No primitive gain, no proof-of-correctness target — this issue enumerates findings that need **plans**, not quick fixes.

---

## TL;DR

Cross-repo code-smell sweep over `katgpt-rs`. Two files exceed the **3200-line hard limit** (one of them, `transformer.rs`, exceeds it by 77%). ~20 library files sit in the 2048–3200 soft-limit band. A handful of stale TODOs and a small number of softmax sites worth reviewing. Clean signals verified: zero `Uuid::new_v4()`, zero `Sha1`/`Sha256` imports (BLAKE3 used throughout), zero `Arc<RwLock<HashMap>>`.

**This issue does NOT propose fixes.** Each Critical/High finding needs its own plan (file split design is a refactor-with-goat-gate, not a one-shot patch).

---

## Critical (>3200 lines — hard limit violation)

### C1. `src/transformer.rs` — 5672 lines

Mixes RiM slots, forward passes (7 variants), generators, raven router, depth routing, paged + quantized paths in a single file. The top re-export block (lines 1–80) already shows the public API is a façade — the split is mechanical.

**Plan needed:**
- Split into `transformer/` module folder:
  - `forward.rs` — forward-pass variants
  - `generators.rs` — generators
  - `raven.rs` — raven router
  - `depth_route.rs` — depth routing
  - `paged.rs` — paged path
  - `quantized.rs` — quantized path
- Preserve the existing public API via `mod.rs` re-exports after split.
- GOAT gate: G1 (no behavior change — bit-identical `cargo test`), G3 (no perf regression on existing benchmarks).

### C2. `crates/katgpt-speculative/src/dd_tree.rs` — 4207 lines

Mixes dd-tree builders, `TreeBuilder` impl (lines 912→3006 — a **2100-line** impl block), SDE variants, residual/cross-scale trackers.

**Plan needed:**
- Extract `TreeBuilder` into `tree_builder.rs`.
- Extract scale configs into `scale_config.rs`.
- Keep `dd_tree.rs` as the data-type + small-impl home.

---

## High (2048–3200 lines — soft limit)

~20 files sit in this band. Library-code priorities (in descending impact):

- `src/dllm.rs` (3078)
- `src/pruners/bomber/players.rs` (2828)
- `crates/katgpt-speculative/src/weaver.rs` (2817)
- `crates/katgpt-core/src/karc.rs` (2597)
- `crates/katgpt-forward/src/dd_tree.rs` (2566)
- `crates/katgpt-core/src/parallax_attn.rs` (2524)
- `crates/katgpt-core/src/speculative/qmc.rs` (2516)
- `crates/katgpt-forward/src/d2f.rs` (2268)
- `crates/katgpt-percepta/src/wasm/lower.rs` (2248)
- `crates/katgpt-core/src/traits.rs` (2203)
- `crates/katgpt-core/src/manifold_bandit.rs` (2196)

(tests/benches/examples in the same band are lower priority — single-file fixtures; split only if they impede review.)

---

## Medium — stale TODOs

### M1. `crates/katgpt-attn/src/dash_attn/forward.rs` — Plan 173 Task 6 dead routing call

Lines 56, 192, 237, 241 — "sparse KV block selection TODO (Plan 173 Task 6)" appears 4×. Doc-comment at L192–198 says "the dead call ran `n_layer`…" — likely dead code left in place.

**Action:** verify Plan 173 status. If abandoned, remove the dead routing call + the TODO.

### M2. `crates/katgpt-backend/src/lib.rs:184` — stale `model_path` parameter

> "TODO: Remove `model_path` parameter — no longer needed with runtime compilation."

Stale signature — parameter is no longer consumed.

**Action:** remove the parameter (update call sites). Trivial.

### M3. `crates/katgpt-core/src/engram/commitment.rs:29` + `kernel.rs:33` — Phase X deferral

> "TODO (Phase X follow-on, deferred)".

Engram is `opt-in` feature-gated.

**Action:** confirm Phase status, or convert to a tracked issue with a concrete next step.

### M4. `crates/katgpt-core/src/linalg/` — three consolidation TODOs

- `geometric_product.rs:62`
- `mod.rs:13`
- `ridge_solve.rs:8`

All three reference "unify with peira's f64 path" / Plan 319 §Risks zero-pad variant.

**Action:** consolidate into a single decision (one plan or one canonical note).

---

## Medium — softmax review (NOT blanket replace)

Most softmax usage is mathematically required (token sampling, attention weights). The global rule (use sigmoid not softmax) applies to **latent-space projection / gating**, not to logit sampling. Three sites worth investigating for latent-domain misuse:

- `crates/katgpt-transformer/src/swir/strategy_adapter.rs:106, 192`
- `crates/katgpt-sense/src/reconstruction.rs:1236, 1237, 1264`
- `crates/katgpt-quant/src/octopus/forward.rs:438`

**Action:** audit each. If the softmax is over a direction-vector projection (latent domain), convert to sigmoid per global rule. If it's over logits or attention weights, leave in place with a one-line comment citing the exception.

---

## Low

- **L1.** `crates/katgpt-core/src/cgsp/types.rs:546–550` — hand-rolled UUID-v4-ish generator using `fastrand::u8(..)`. Consider `Uuid::now_v7().to_bytes_le()` if determinism permits (note: v7 is time-ordered; if the caller needs deterministic seeds, the fastrand path may be intentional — verify before changing).
- **L2.** `benches/cgsp_hint_receptivity_bench.rs:150` + `benches/sudoku_speculate_bench.rs:346` — `partial_cmp(...).unwrap()` on `f32`. Silent NaN bug risk (returns `Equal` ordering on NaN). Replace with an explicit NaN-handling comparator or assert `is_finite()` upstream.
- **L3.** Sampling DRY — `softmax_scaled(logits, 1.0/temp); sample_token_into(...)` appears 5+ times across call sites. Extract `sample_next_token(ctx, logits, temp, rng)` helper.

---

## Clean signals (verified)

- ✅ Zero `Uuid::new_v4()` in `src/` or `crates/`
- ✅ Zero `Sha1`/`Sha256` imports (BLAKE3 used throughout)
- ✅ Zero `Arc<RwLock<HashMap>>` — papaya or `&mut self` patterns used

---

## Recommended priority

1. **Split `transformer.rs`** (highest impact — 5672 lines, biggest review burden).
2. **Split `dd_tree.rs`** (4207 lines, contains a single 2100-line impl block).
3. **Remove stale `model_path` parameter** (trivial — unblock follow-on backend simplification).
4. **Audit Plan 173 dead routing call** (remove if abandoned — small but unblocks `dash_attn` review).

---

## Non-goals

- **NOT** a blanket "split every file over 2048 lines." Soft-limit files are flagged for awareness; split when they impede review or have a natural seam.
- **NOT** blanket softmax replacement. See Medium section above.
- **NOT** touching tests/benches/examples in the soft-limit band unless they block a code change.

## Cross-references

- Global `~/.agents/` rules — file-size limits (`< 3200` hard, `< 2048` soft for `.rs`), sigmoid-not-softmax, `Uuid::now_v7()`, blake3, papaya.
- `katgpt-rs/AGENTS.md` — feature-flag discipline, GOAT gate (G1 correctness / G3 no-regression apply to refactor splits).
