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

### C1. `src/transformer.rs` — 5672 lines ✅ DONE (Issue 164)

Mixes RiM slots, forward passes (7 variants), generators, raven router, depth routing, paged + quantized paths in a single file. The top re-export block (lines 1–80) already shows the public API is a façade — the split is mechanical.

**Resolved 2026-07-17 (Issue 164):** split into `transformer/` module folder with 8 sub-modules (`mod.rs` + `variants.rs` + `tf_loop.rs` + `prefill.rs` + `generators.rs` + `paged.rs` + `raven.rs` + `quantized.rs` + `tests.rs`). Public API preserved 1:1 via `mod.rs` re-exports. GOAT gate G1 + G3 PASS: 200/200 default tests bit-identical, clippy clean workspace-wide. Module naming differs slightly from the original proposal (batched-forward module is `variants.rs` not `forward.rs` to avoid collision with the re-exported `forward` fn; depth routing lives in `tf_loop.rs` with its sole caller `forward_training_free_loop`).

### C2. `crates/katgpt-speculative/src/dd_tree.rs` — 4207 lines ✅ DONE (Issue 165)

Mixes dd-tree builders, `TreeBuilder` impl (lines 912→3006 — a **2100-line** impl block), SDE variants, residual/cross-scale trackers.

**Resolved 2026-07-17 (Issue 165, commit `a4c6cbdb`):** converted to `dd_tree/` module folder with `TreeBuilder` extracted to `tree_builder.rs` (2091 lines). The scale-config structs (`LodestarConfig`, `WidthScaleConfig`, `CrossScaleConfig`, `ResidualTracker`) were NOT split into a separate `scale_config.rs` — they're tightly coupled to their builder functions and small enough to stay in `mod.rs`. Test file `dd_tree_tests.rs` moved into the module folder as `tests.rs`. Both `mod.rs` (2125) and `tree_builder.rs` (2091) are now under the 3200 hard limit (down from Critical); both remain in the 2048–3200 soft-limit band (High). GOAT gate G1 + G3 PASS: 305/305 + 200/200 + 1079/1079 (`--all-features`) tests pass. **Sibling-validated 2026-07-17:** the follow-on GRAPE feature re-test (`cargo test -p katgpt-core --lib --features grapem_rodrigues,position_group_action,grape_ap_vector`) returned **1605 passed / 0 failed / 3 ignored** — up from 1561 in the default-features run, confirming the opt-in GRAPE features added by Issues 159/160/161 compile and pass under the split. The sibling's self-admitted broken test from `dc192b3b` (fixed in the same push) is certified closed.

---

## High (2048–3200 lines — soft limit)

~20 files sit in this band. Library-code priorities (in descending impact):

- ~~`src/dllm.rs` (3078)~~ ✅ DONE (Issue 166, 2026-07-17): split to `dllm/` module folder — `mod.rs` (1924, under 2048 ✓) + `tests.rs` (1163, tests exempt). Single-axis tests-extraction split (implementation is cohesive dLLM training research code). G1+G3 PASS: 200/200 default + 19/19 `--features dllm` + 24/24 `--features "dllm,replaid_schedules"` tests pass; clippy clean.
- `src/pruners/bomber/players.rs` (2828) — **CONFIRMED needs functional split** (only 3% tests, 98 test lines; extracting tests leaves mod.rs at 2730, still over 2048). Out of scope for mechanical test-extraction passes.
- `crates/katgpt-speculative/src/weaver.rs` (2817) — **CONFIRMED stays over** (27% tests, 759 test lines; extracting tests leaves mod.rs at 2058, 10 lines over 2048). User explicitly said skip.
- `crates/katgpt-core/src/karc.rs` (2597) ✅ DONE (Issue 169, 2026-07-17): split to `karc/` module folder — `mod.rs` (**2008 lines, under 2048 ✓** — exactly as predicted, the only remaining split that lands under after test extraction) + `tests.rs` (589, tests exempt). Single-axis tests extraction. No `pub(super)` helpers, no path corrections. G1+G3 PASS: 19/19 `karc::tests::*` + 1587/1587 katgpt-core lib under `karc_forecaster` (vs 1558 default); clippy clean (lib + tests + workspace default + workspace lib under feature). Pre-existing unrelated failures documented (ht_chantry_diagnostic example needs `multi_agent_path`; issue_156_anytime_lt2_poc has the elastic_override argument mismatch from prior session) — both reproduce against unsplit original.
- `crates/katgpt-forward/src/dd_tree.rs` (2566) ✅ DONE (Issue 168, 2026-07-17): split to `dd_tree/` module folder — `mod.rs` (179, well under 2048 ✓) + `tests.rs` (2387, tests exempt). Single-axis tests extraction (file was 98% tests — only ~175 lines of impl: two feature-gated wrappers + the `pub use katgpt_speculative::dd_tree::*` re-export glob). No `pub(super)` helpers, no path corrections. G1+G3 PASS: 109/109 katgpt-forward default + 112/112 under `thinking_prune,sr2am_configurator,gdsd_distill` (49/49 in `dd_tree::tests::*`) + 200/200 root lib under same features; clippy clean workspace-wide. Pre-existing wiring note: katgpt-forward's `thinking_prune` feature alone does not forward `sr2am_configurator` — same error reproduces against the unsplit original; GOAT gate runs through the root crate (where forwarding is complete).
- ~~`crates/katgpt-core/src/parallax_attn.rs` (2524)~~ ✅ DONE (Issue 167, 2026-07-17): split to `parallax_attn/` module folder — `mod.rs` (973, well under 2048 ✓) + `tests.rs` (1559, tests exempt). Single-axis tests-extraction split (implementation is cohesive Parallax attention). 4 feature-gated test sections preserved with corrected cfg gates. G1+G3 PASS: 1558/1558 katgpt-core default + 24/24 `parallax_attn,sink_aware_attn,ssmax_temperature` tests pass; clippy clean.
- ~~`crates/katgpt-core/src/speculative/qmc.rs` (2516)~~ ✅ DONE (Issue 170, 2026-07-17): split to `qmc/` module folder — `mod.rs` (1085, well under 2048 ✓ — the file was 57% tests, not functional-split-required as the prior session's summary incorrectly stated) + `tests.rs` (1430, tests exempt). Single-axis tests extraction. Single `#[cfg(test)] mod tests` block, single `use super::*;`, zero `pub(super)` helpers, no path corrections. G1+G3 PASS: 119/119 qmc tests under `qmc_sampling` + 1558/1558 katgpt-core lib; clippy clean (lib + tests + workspace).
- ~~`crates/katgpt-forward/src/d2f.rs` (2268)~~ ✅ DONE (Issue 172, 2026-07-17): split to `d2f/` module folder — `mod.rs` (1783, under 2048 ✓ — the file was 21% tests, not functional-split-required as the prior session's summary incorrectly stated) + `tests.rs` (485, tests exempt). Single-axis tests extraction. Single `#[cfg(test)] mod tests` block, single `use super::*;`, zero `pub(super)` helpers, no path corrections. G1+G3 PASS: 18/18 d2f tests + 127/127 katgpt-forward lib + 200/200 root lib under `dllm` (d2f is dllm-gated); clippy clean (lib + tests + workspace). Same class of prior-session verdict error as Issues 170 + 171.
- ~~`crates/katgpt-percepta/src/wasm/lower.rs` (2248)~~ ✅ DONE (Issue 173, 2026-07-17): split to `wasm/lower/` module folder — `mod.rs` (1873, under 2048 ✓ — the file was 17% tests, not functional-split-required as the prior session's summary incorrectly stated) + `tests.rs` (375, tests exempt). Single-axis tests extraction. Single `#[cfg(test)] mod tests` block, single `use super::*;`, zero `pub(super)` helpers, no path corrections. G1+G3 PASS: 22/22 wasm::lower tests + 225/225 katgpt-percepta lib under `percepta_wasm` (wasm module is percepta_wasm-gated); clippy clean (lib + tests + workspace). Same class of prior-session verdict error as Issues 170 + 171 + 172.
- ~~`crates/katgpt-core/src/traits.rs` (2203)~~ ✅ DONE (Issue 174, 2026-07-17): split to `traits/` module folder — `mod.rs` (1495, under 2048 ✓ — the file was 32% tests across 5 independent modules, not functional-split-required as the prior session's summary incorrectly stated) + 5 sibling test files (`tests_leo.rs` 359 + `tests_spec_gen.rs` 50 + `tests_best_buddies.rs` 73 + `tests_reject_confidence.rs` 95 + `recursion_logits_tests.rs` 115 = 692 total, tests exempt). Single-axis tests extraction (5 modules). Zero `pub(super)` helpers, no path corrections. Note: `tests_leo` has no module-level `#[cfg(test)]` (pre-existing quirk — always compiled, inner `#[test]` gates emission); preserved exactly. G1+G3 PASS: 36/36 traits tests default + 39/39 under `recursion_logits` + 1561/1561 katgpt-core lib under feature; clippy clean (lib + tests + workspace). Same class of prior-session verdict error as Issues 170 + 171 + 172 + 173.
- ~~`crates/katgpt-core/src/manifold_bandit.rs` (2196)~~ ✅ DONE (Issue 171, 2026-07-17): split to `manifold_bandit/` module folder — `mod.rs` (1290, well under 2048 ✓ — the file was 41% tests, not functional-split-required as the prior session's summary incorrectly stated) + `tests.rs` (906, tests exempt). Single-axis tests extraction. Single `#[cfg(test)] mod tests` block, single `use super::*;`, zero `pub(super)` helpers, no path corrections. G1+G3 PASS: 34/34 manifold_bandit tests under feature + 1558/1558 katgpt-core lib; clippy clean (lib + tests + workspace). Same class of prior-session verdict error as Issue 170 (qmc.rs).

(tests/benches/examples in the same band are lower priority — single-file fixtures; split only if they impede review.)

---

## Medium — stale TODOs

### M1. `crates/katgpt-attn/src/dash_attn/forward.rs` — Plan 173 Task 6 dead routing call

Lines 56, 192, 237, 241 — "sparse KV block selection TODO (Plan 173 Task 6)" appears 4×. Doc-comment at L192–198 says "the dead call ran `n_layer`…" — likely dead code left in place.

**Status:** **RESOLVED 2026-07-17** (commit pending). Re-audit confirmed: the routing call itself was already removed (dead-compute elimination); what remained was 4 stale comments claiming "Plan 173 Task 6 is not yet implemented." Plan 173 Task 6 IS implemented — via `EntmaxRouter` (Plan 196 / `vortex_flow` feature gate), which composes `score_blocks_entmax` into a `VortexFlow` router. The block-level sparse KV selection lives in the VortexFlow dispatcher, not in this decode function. Updated the comments to point to `EntmaxRouter` and removed the stale "not yet implemented" claim.

### M2. `crates/katgpt-backend/src/lib.rs:184` — stale `model_path` parameter

> "TODO: Remove `model_path` parameter — no longer needed with runtime compilation."

Stale signature — parameter is no longer consumed.

**Status:** **DONE 2026-07-17** (commit `5d3324ec`). Removed `model_path: Option<&Path>` from `auto_backend` and `_model_path` from `try_ane_backend`. All 4 call sites (which already passed `None`) updated. TODO comment deleted. Validation: 2 lib tests + 11 bench/goat tests PASS.

### M3. `crates/katgpt-core/src/engram/commitment.rs:29` + `kernel.rs:33` — Phase X deferral

> "TODO (Phase X follow-on, deferred)".

Engram is `opt-in` feature-gated.

**Status:** **VERIFIED ACTIVE 2026-07-17 — keep.** Both TODOs document concrete deferred Plan 299 subtasks with explicit "file when first consumer needs it" guidance:
- `commitment.rs:29` — T5.1–T5.4 `EngramHotSwap` (AtomicPtr<Box<dyn EngramTable>> + reader closure, mirroring `sense/hotswap.rs`) + T5.7–T5.8 unit tests for hot-swap atomicity + G5 concurrent reader/writer gate.
- `kernel.rs:33` — T3.6 multi-branch `sigmoid_fuse_multi_branch_into` (M distinct gates sharing one `v`; default M=1, mHC backbone uses M=4) + T3.7 depthwise causal conv `conv_causal_into` (paper §2.3 eq 5).
These are well-scoped future-work notes with concrete acceptance criteria, NOT stale. Same verdict class as the 3 riir-ai TODOs verified ACTIVE in Issue 530.

### M4. `crates/katgpt-core/src/linalg/` — three consolidation TODOs

- `geometric_product.rs:62`
- `mod.rs:13`
- `ridge_solve.rs:8`

All three reference "unify with peira's f64 path" / Plan 319 §Risks zero-pad variant.

**Status:** **VERIFIED ACTIVE 2026-07-17 — keep, consolidated note below.** All three TODOs reference the same deferral: unifying the f32 linalg path with PEIRA's f64 path risks breaking PEIRA's bit-identical Plan 153 G4 reproducibility, and is correctly gated on a generic-over-`T: Float` Cholesky being proven bit-identical to the current f64 specialization. The `mod.rs:13` note is the canonical summary; `ridge_solve.rs:8` and `geometric_product.rs:62` (Plan 319 §Risks zero-pad variant) are local reminders pointing at the same future work. The three TODOs are consistent — NOT in conflict, NOT stale. **Consolidated decision:** leave in place; reopen as a plan if/when a concrete consumer needs the unified path (e.g. a new ridge-solver caller that wants both f32 perf and f64 numerical robustness from one entry point).

---

## Medium — softmax review (NOT blanket replace)

Most softmax usage is mathematically required (token sampling, attention weights). The global rule (use sigmoid not softmax) applies to **latent-space projection / gating**, not to logit sampling. Three sites worth investigating for latent-domain misuse:

- `crates/katgpt-transformer/src/swir/strategy_adapter.rs:106, 192`
- `crates/katgpt-sense/src/reconstruction.rs:1236, 1237, 1264`
- `crates/katgpt-quant/src/octopus/forward.rs:438`

**Action:** audit each. If the softmax is over a direction-vector projection (latent domain), convert to sigmoid per global rule. If it's over logits or attention weights, leave in place with a one-line comment citing the exception.

**Status: ALL 3 SITES AUDITED 2026-07-17 — KEEP, all are canonical logit/attention-domain softmax.**

1. **`swir/strategy_adapter.rs:106, 192`** — `softmax_into_scratch` converts **token logits → probability distribution** for soft-embedding accumulation (`EmitSoftEmbedding` path mixes vocab probabilities against the embedding matrix). Canonical logit-domain softmax. The doc comment at L99-105 explicitly justifies keeping it as a helper for SIMD drop-in. Not a latent projection.

2. **`katgpt-sense/src/reconstruction.rs:1236, 1237, 1264`** — `advantage_margin_hla` (Eq. 18, arxiv:2511.16886) computes `KL(π+ ‖ π̂)` via `log_softmax`. This is **information-theoretic KL divergence between two action distributions**, which mathematically requires log-softmax (KL = E_post[log π+ − log π̂]). The 6-element input is treated as logits over an action set, not as a latent embedding. Feature-gated under `self_advantage_gate`. Per AGENTS.md sigmoid-rule scope ("applies to latent-space projection / gating, not to logit sampling"), this is correctly softmax.

3. **`katgpt-quant/src/octopus/forward.rs:438`** — the cited line is a test-assertion message string (inside `#[test] fn test_attention_weights_normalized`), NOT a production softmax site. The production softmax is in `attention_octopus` (L88) — canonical softmax-over-keys for attention weights. KEEP.

No conversion to sigmoid needed at any of the 3 sites. The audit's flag was correct to surface them (the rule is non-obvious), but the verdict is uniformly KEEP with documented rationale.

---

## Low

- **L1.** `crates/katgpt-core/src/cgsp/types.rs:546–550` — hand-rolled UUID generator using `fastrand::u8(..)`.
  - **Status:** **VERIFIED ACTIVE 2026-07-17 — keep.** Re-audit corrected two mischaracterizations in the original entry: (1) this is a UUID**v7** generator (timestamp-prefixed, version=0x7, RFC 4122 variant) — NOT v4 as the original entry stated; (2) the `fastrand` choice is INTENTIONAL, not a bug. The function's own doc comment (L526-530) explains: "sufficient for ordering within a process without pulling in the `uuid` crate as a hard dependency here." `katgpt-core` deliberately does NOT depend on the `uuid` crate (it's a leaf-clean public primitive), so swapping to `Uuid::now_v7().to_bytes_le()` would either add a new dep to `katgpt-core` (against the leaf-clean rule) or move this caller out of `katgpt-core`. The current hand-rolled v7 layout is correct and intentional. No action.
- **L2.** `benches/cgsp_hint_receptivity_bench.rs:150` + `benches/sudoku_speculate_bench.rs:346` — `partial_cmp(...).unwrap()` on `f32`. Silent NaN bug risk (returns `Equal` ordering on NaN). Replace with an explicit NaN-handling comparator or assert `is_finite()` upstream.
  - **Status:** **DONE 2026-07-17** (commit `317534ec`). Replaced with `f32::total_cmp` (Rust 1.62+) which provides a total ordering. NaN now sorts as largest rather than silently comparing equal. No behavior change for the non-NaN inputs the benches actually produce. Validation: `cargo clippy --benches` → 0 warnings.
- **L3.** Sampling DRY — `softmax_scaled(logits, 1.0/temp); sample_token_into(...)` appears 5+ times across call sites. Extract `sample_next_token(ctx, logits, temp, rng)` helper.
  - **Status:** **DONE 2026-07-17** — the helper was already extracted as `ForwardContext::sample_next_token(&mut self, temperature: f32, rng: &mut Rng) -> usize` at `crates/katgpt-forward/src/lib.rs:303-307`. Its docstring (L287-291) explicitly states: "This fuses the two-line pattern that appeared 7+ times across the generator call sites (`softmax_scaled(logits, 1.0/temp); sample_token_into(&ctx.logits, rng, &mut ctx.cdf)`) into one intent-revealing call." Verified ZERO remaining call sites use the old 2-line pattern: a grep for `sample_token_into(&ctx.logits` across `src/` + `crates/` returns only the docstring comment itself. All generators (`generate_into`, `generate_gdn2_into`, `generate_hla_into`, `generate_ahla_into`, `InferenceRouter::generate_routed`, etc.) now call `ctx.sample_next_token(config.temperature, rng)`.

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
