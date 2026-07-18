# Code-Smell + File-Size Audit (2026-07-17)

> **What this is.** Point-in-time hygiene audit (Issue 162 + its 15 split
> sub-issues 164–178) that informed a wave of mechanical + functional file
> splits and a small set of correctness fixes. Kept here so future audits
> have a baseline: which files were split, which were deliberately kept,
> and which stale-TODO / softmax findings were closed (with rationale).
>
> **Source:** Issue 162 (parent, removed 2026-07-17 per noise rule) +
> Issues 164–178 (split sub-issues, removed same date). All verdicts
> preserved in git history; this doc is the canonical summary.

## TL;DR

Cross-repo code-smell sweep over `katgpt-rs`. Two files exceeded the
**3200-line hard limit** (one, `transformer.rs`, by 77%). ~20 library
files sat in the 2048–3200 soft-limit band. All Critical + most High
findings closed via Issues 164–178. Three softmax sites audited (all
KEEP — canonical logit/attention domain). Clean signals verified:
zero `Uuid::new_v4()`, zero `Sha1`/`Sha256` imports (BLAKE3
throughout), zero `Arc<RwLock<HashMap>>`.

## Critical (>3200 lines — hard limit violation) — ALL CLOSED

| File | Was | Now | Sub-issue |
|---|---|---|---|
| `crates/katgpt-percepta/src/transformer.rs` | 5672 | `transformer/` module folder (8 sub-modules, public API preserved 1:1 via `mod.rs` re-exports) | Issue 164 |
| `crates/katgpt-speculative/src/dd_tree.rs` | 4207 | `dd_tree/` module folder — `mod.rs` + `tree_builder.rs` + `lodestar.rs` + `tests.rs` | Issues 165 + 178 |

## High (2048–3200 lines — soft limit) — split outcomes

| File | Was | Verdict |
|---|---|---|
| `src/dllm.rs` | 3078 | ✅ Split (Issue 166) — `dllm/mod.rs` 1924 + `tests.rs` 1163 |
| `crates/katgpt-core/src/karc.rs` | 2597 | ✅ Split (Issue 169) — `karc/mod.rs` 2008 + `tests.rs` 589 |
| `crates/katgpt-forward/src/dd_tree.rs` | 2566 | ✅ Split (Issue 168) — `dd_tree/mod.rs` 179 + `tests.rs` 2387 (98% tests) |
| `crates/katgpt-core/src/parallax_attn.rs` | 2524 | ✅ Split (Issue 167) — `parallax_attn/mod.rs` 973 + `tests.rs` 1559 |
| `crates/katgpt-core/src/speculative/qmc.rs` | 2516 | ✅ Split (Issue 170) — `qmc/mod.rs` 1085 + `tests.rs` 1430 (57% tests) |
| `crates/katgpt-core/src/manifold_bandit.rs` | 2196 | ✅ Split (Issue 171) — `manifold_bandit/mod.rs` 1290 + `tests.rs` 906 (41% tests) |
| `crates/katgpt-forward/src/speculative/d2f.rs` | 2268 | ✅ Split (Issue 172) — `d2f/mod.rs` 1783 + `tests.rs` 485 (21% tests) |
| `crates/katgpt-percepta/src/wasm/lower.rs` | 2248 | ✅ Split (Issue 173) — `wasm/lower/mod.rs` 1873 + `tests.rs` 375 (17% tests) |
| `crates/katgpt-core/src/traits.rs` | 2203 | ✅ Split (Issue 174) — `traits/mod.rs` 1495 + 5 sibling test files (692 total) |
| `src/pruners/bomber/players.rs` | 2828 | ✅ Functional split (Issue 175) — `players/` folder: mod.rs 155 + helpers.rs 783 + 7 player-type files + tests.rs 102 |
| `crates/katgpt-pruners/src/vocab_channel_pruner.rs` | 2053 | ✅ Split (Issue 176 T1) — mod.rs 1183 (93% tests) |
| `crates/katgpt-percepta/src/legacy.rs` | 2124 | ✅ Split (Issue 176 T2) — mod.rs 910 (57% tests) |
| `crates/katgpt-core/src/funcattn.rs` | 2086 | ✅ Split (Issue 176 T3) — mod.rs 983 (53% tests) |
| `crates/katgpt-dec/src/sheaf_admm.rs` | 2109 | ✅ Split (Issue 176 T4) — mod.rs 1222 (42% tests) |
| `crates/katgpt-percepta/src/graph/types.rs` | 2055 | ✅ Split (Issue 176 T5) — `types/mod.rs` 1333 (35% tests) |
| `crates/katgpt-ruliology/src/bandit.rs` | 2178 | ✅ Functional split (Issue 177) — `bandit/` folder: mod.rs 1289 + environment.rs + session.rs + shared_stats.rs + randopt.rs |
| `crates/katgpt-speculative/src/dd_tree/mod.rs` | 2125 | ✅ Extracted Lodestar (Issue 178) — mod.rs 1866 + `lodestar.rs` 770 |
| `crates/katgpt-speculative/src/weaver.rs` | 2817 | **CONFIRMED KEEP** — user-explicit skip (759 test lines; extraction leaves impl at 2058, 10 over soft limit) |
| `crates/katgpt-speculative/src/dd_tree/tree_builder.rs` | 2091 | **CONFIRMED KEEP** (Issue 178 T4) — single `TreeBuilder` struct with tightly-coupled private state; splitting would require `pub(super)` fields or accessor boilerplate. 2% over soft limit, well under hard limit. Cohesion > marginal split benefit. |

**Lesson (Issue 170–174):** prior sessions mis-called several files as
"functional-split-required" when they were actually mechanical
test-extraction candidates (57%, 41%, 21%, 17%, 32% tests). Always
verify the test-fraction before declaring a file needs a functional
split — most impl-cohesive files just need their `#[cfg(test)] mod
tests` extracted.

## Medium — stale TODOs

| ID | Site | Verdict |
|---|---|---|
| **M1** | `crates/katgpt-attn/src/dash_attn/forward.rs` (4 stale "Plan 173 Task 6 not yet implemented" comments) | **RESOLVED** — routing call was already removed (dead-compute elimination); Task 6 IS implemented via `EntmaxRouter` (Plan 196 / `vortex_flow` feature). Comments updated to point at `EntmaxRouter`. |
| **M2** | `crates/katgpt-backend/src/lib.rs:184` — stale `model_path` parameter | **DONE** (commit `5d3324ec`) — removed from `auto_backend` + `try_ane_backend`; 4 call sites updated. |
| **M3** | `crates/katgpt-core/src/engram/commitment.rs:29` + `kernel.rs:33` — Phase X deferral TODOs | **VERIFIED ACTIVE — KEEP.** Both document concrete deferred Plan 299 subtasks (`EngramHotSwap` T5.1–T5.8; multi-branch sigmoid fuse + depthwise causal conv T3.6/T3.7) with explicit "file when first consumer needs it" guidance. |
| **M4** | `crates/katgpt-core/src/linalg/{geometric_product.rs:62, mod.rs:13, ridge_solve.rs:8}` — three "unify with peira f64 path" TODOs | **VERIFIED ACTIVE — KEEP, consolidated.** All three reference the same deferral: unifying the f32 linalg path with PEIRA's f64 path risks breaking PEIRA's bit-identical Plan 153 G4 reproducibility. Gated on a generic-over-`T: Float` Cholesky being proven bit-identical to the f64 specialization. `mod.rs:13` is the canonical summary; the other two are local reminders. Reopen as a plan if a concrete consumer needs the unified path. |

## Medium — softmax review (NOT blanket replace)

The global rule (use sigmoid not softmax) applies to **latent-space
projection / gating**, not to logit sampling or attention weights.
Three sites flagged for latent-domain misuse — all audited, all KEEP:

| Site | Verdict |
|---|---|
| `crates/katgpt-transformer/src/swir/strategy_adapter.rs:106, 192` | **KEEP** — `softmax_into_scratch` converts token logits → probability distribution for soft-embedding accumulation. Canonical logit-domain. |
| `crates/katgpt-sense/src/reconstruction.rs:1236, 1237, 1264` | **KEEP** — `advantage_margin_hla` (Eq. 18, arxiv:2511.16886) computes `KL(π+ ‖ π̂)` via `log_softmax`. KL divergence between action distributions mathematically requires log-softmax. Feature-gated `self_advantage_gate`. |
| `crates/katgpt-quant/src/octopus/forward.rs:438` | **KEEP** — cited line is a test-assertion message string, not production softmax. Production softmax (`attention_octopus` L88) is canonical softmax-over-keys for attention weights. |

## Low

- **L1.** `crates/katgpt-core/src/cgsp/types.rs:546–550` — hand-rolled UUID generator. **VERIFIED ACTIVE — KEEP.** Re-audit corrected two mischaracterizations: (1) this is a UUID**v7** generator (timestamp-prefixed), NOT v4; (2) the `fastrand` choice is INTENTIONAL — `katgpt-core` deliberately does not depend on the `uuid` crate (leaf-clean public primitive). Swapping to `Uuid::now_v7().to_bytes_le()` would add a dep or move the caller out of `katgpt-core`.
- **L2.** `benches/cgsp_hint_receptivity_bench.rs:150` + `benches/sudoku_speculate_bench.rs:346` — `partial_cmp(...).unwrap()` on `f32`. **DONE** (commit `317534ec`) — replaced with `f32::total_cmp`. NaN now sorts as largest rather than silently comparing equal.
- **L3.** Sampling DRY (`softmax_scaled` + `sample_token_into` 5+ sites). **DONE** — helper already extracted as `ForwardContext::sample_next_token` at `crates/katgpt-forward/src/lib.rs:303-307`. Verified zero remaining call sites use the old 2-line pattern.

## Clean signals (verified)

- ✅ Zero `Uuid::new_v4()` in `src/` or `crates/`
- ✅ Zero `Sha1`/`Sha256` imports (BLAKE3 used throughout)
- ✅ Zero `Arc<RwLock<HashMap>>` — papaya or `&mut self` patterns used

## Non-goals (still apply)

- **NOT** a blanket "split every file over 2048 lines." Soft-limit files
  are flagged for awareness; split when they impede review or have a
  natural seam.
- **NOT** blanket softmax replacement. See Medium section above.
- **NOT** touching tests/benches/examples in the soft-limit band unless
  they block a code change.

## See also

- [`loser_sweep_audit.md`](loser_sweep_audit.md) — Phase 0.5 loser-sweep audit (Proposal 003)
- [`claim_rubric_audit.md`](claim_rubric_audit.md) — research-note vs `Claim` fixture rubric (Plan 307 T4.2)
- [`cross_repo_consolidation_audit.md`](cross_repo_consolidation_audit.md) — riir-ai / riir-chain / riir-neuron-db consolidation
- Global `~/.agents/` rules — file-size limits (`< 3200` hard, `< 2048` soft for `.rs`), sigmoid-not-softmax, `Uuid::now_v7()`, blake3, papaya.
- `katgpt-rs/AGENTS.md` — feature-flag discipline, GOAT gate (G1 correctness / G3 no-regression apply to refactor splits).
