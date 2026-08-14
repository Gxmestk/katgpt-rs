# Issue 587: Distribution-Preserving Self-Speculation Acceptance (FLARE Eq 21/22) + Streaming Verify

**Date:** 2026-08-14
**Research:** [katgpt-rs/.research/480_FLARE_Hybrid_Diffusion_Dual_Trust_Decode.md](../.research/480_FLARE_Hybrid_Diffusion_Dual_Trust_Decode.md)
**Source paper:** [arXiv:2606.01774](https://arxiv.org/abs/2606.01774) — FLARE: Diffusion for Hybrid Language Model (§3.3, Appendix B.2)
**Target:** `crates/katgpt-forward/src/d2f_verifier.rs` (Phase 3) + optional `d2f/mod.rs` proposal-prob plumbing
**Feature gate:** stays under `tri_mode`; new `DraftAcceptPolicy` config enum (no new Cargo feature unless the gate grows)

## Problem

`D2fDrafterVerifier::speculate` Phase 3 uses **greedy argmax prefix-match acceptance**: accept `d_i` iff `d_i == argmax(p_i)`; on mismatch emit the target's argmax and stop. Per FLARE's proposal-consistency analysis this is distribution-biasing:

1. Every rejection emits the greedy token and every acceptance requires the greedy token → the merged output collapses toward the target mode; temperature/sampling semantics are destroyed (the bonus token is sampled correctly, but the accepted-prefix path is not).
2. Acceptance is strictly harder than rejection sampling (`u ≤ min(1, p/q)` accepts non-argmax drafts with the right probability) → likely lower acceptance rate AND wrong distribution.
3. Phase 2 materializes `p_distributions_flat: [(K+1) × vocab_size]` full softmax distributions per round (~5 MB/round at Gemma-2's 256K vocab, K=4) — FLARE's fused verify never forms these tensors.

The `LeviathanVerifier` already implements real p/q rejection for AR drafters — the machinery exists; it was never wired to the D2F drafter.

## Proposed fix (FLARE taxonomy)

```
enum DraftAcceptPolicy { PrefixMatch /*current, kept as fallback*/, SoftmaxArgmax, TruncatedArgmax /*+ Exact later*/ }
```

- **SoftmaxArgmax (Eq 21, P0):** requires argmax drafting. Accept `d_i ⟺ u ≤ min(1, p_i^full(d_i))`; correction `y* ~ p^full \ {d_i}`. Exact w.r.t. target at ~zero extra cost — Phase 3 loop change only, reusing the already-computed `p_dist`.
- **Exact-Truncated (Eq 8, P2):** our D2F drafter *samples* (temperature) rather than argmax-drafting, so exactness needs the draft-time proposal law `q_i` plumbed out of `d2f_decode_block` as compact top-k `(ids, probs)` and stored per round. True `min(1, p/q)` rejection; sparse k×k verify.
- **TruncatedArgmax (Eq 22, P1):** both sides top-k truncated; approximate (argmax drafts + truncated accept `min(1, p_trunc/q_trunc)`), correction `normalize(max(p−q, 0))`.

## Tasks

- [x] **T1 — SoftmaxArgmax policy (Eq 21).** Added `DraftAcceptPolicy` to `D2fDrafterVerifier` (+ `with_accept_policy` builder + `SelfSpecConfig` field); implemented in the fused Phase 2+3 loop: draw `u`, accept iff `u ≤ p_dist[draft_tok]`, else correction from `p_dist` renormalized over `≠ d` and break. Argmax drafting: `D2fDecodeConfig::greedy_draft` (new field) — the verifier FORCES it under SoftmaxArgmax (the policy's exactness precondition), documented on the enum. Users wanting sampled drafts get ExactQ (exact) instead.
- [x] **T2 — G1 exactness proof.** Two levels: (a) pure policy-level unit tests — 40k rounds on a fixed toy target, TV < 0.02 for SoftmaxArgmax + ExactQ, PrefixMatch FAILS (output is a constant — asserted as a point mass + TV > 0.7); (b) E2E through `speculate()` with the micro transformer — 8k rounds from a fixed anchor, first-token empirical distribution vs the reference `softmax(forward(anchor))`: SoftmaxArgmax TV = 0.0003, ExactQ TV = 0.0140, PrefixMatch collapses to 1 distinct token. UQ note: distribution-equivalence gate, not a calibrated-UQ claim — conformal floor not applicable (no interval/coverage claim).
- [x] **T3 — TruncatedArgmax (Eq 22).** Top-16 truncated p, accept `u ≤ p̃(d)` if `d` in support else reject, correction from top-k∖{d}. Unit test bounds TV by the discarded tail mass. Measured +10% latency vs PrefixMatch for exact-adjacent behavior — dominated by SoftmaxArgmax (kept as the taxonomy-complete approximate variant).
- [x] **T4 — Exact-Truncated (Eq 8).** Implemented as **ExactQ (full-width q)** — documented deviation from top-k: `d2f_decode_block_with_prompt_with_q` captures the true draft-time proposal law per position at commit time (point mass under greedy drafting, the actual sampled distribution otherwise); verification runs true `min(1, p/q)` + residual correction (`sample_residual_distribution_into`). Full-q instead of top-k because (a) the library already pays `[V]`-sized buffers, (b) full-q is EXACT rather than tail-approximate, (c) FLARE's top-k is a serving-memory optimization at 256K vocab — our hot-path scale doesn't need it. E2E TV = 0.0140 under sampled drafts.
- [x] **T5 — Streaming verify.** The `[(K+1)×V]` `p_distributions_flat` buffer is DELETED; one vocab-sized `probs_buf` streams position by position; Phases 2+3 fused (feed-then-test per position); scoring stops at the first rejection (early exit saves the remaining target forwards). Measured: SoftmaxArgmax 332.5µs vs PrefixMatch 352.4µs (−6%) at equal acceptance; memory 5×256K×4B → 1×256K×4B at Gemma-2 vocab, K=4. PrefixMatch kept as control on the same streaming path.
- [x] **T6 — GOAT gate.** G1 PASS (exactness, both levels); G2 PASS (self-spec: SoftmaxArgmax 1.98 tokens/round vs PrefixMatch 2.00, latency −6%; gate: ≥ acceptance at equal-or-better latency — see `.benchmarks/634`); G3 PASS (206 root lib + 154 katgpt-forward lib + 17 test_d2f_decode + 5 test_diffusion_sampler_goat + 10 test_d2f_verifier under tri_mode+dllm; clippy clean default/dllm/tri_mode/flashar combos); G4 PASS (streaming buffers constructor-fixed; accepted_buf realloc bounded by K+1 — the Vec-return ABI cost, pre-existing). **Promoted: `DraftAcceptPolicy::SoftmaxArgmax` is the `#[default]`** (enum + `D2fDrafterVerifier::new` + `SelfSpecConfig::default`); PrefixMatch demoted to explicit-opt-in control. ALSO FIXED: the pre-rewrite Phase 2/3 compared draft token i against the target distribution for position i+1 (off-by-one) — verification is now slot-aligned (matching LeviathanVerifier), which is why the legacy numbers were garbage-adjacent.
- [x] **T7 — FlashAR audit.** NOT distribution-preserving, by design — Plasma/Hot skip verification entirely (Plan 166's latency trade-off), Warm/Cold use argmax prefix-match (mode-biasing). Actionable slice (Warm/Cold → Eq 21 swap) filed as [Issue 651](651_flashar_cold_path_prefix_match.md); Plasma/Hot documented as a deliberate bias. No code change in this issue.
- [-] **T8 — deferred:** recurrent-state cache-and-scatter rewind for linear-attention backbones (FLARE App B.4). No hybrid-backbone speculative-decode consumer today; revisit when riir-ai's linear-attention lineage (HLA / Luce / KDA) grows an inference serving path. Recorded in Research 480 §2.3 D3.

## Notes

- Proposal-consistency subtlety (why Eq 8 needs stored q): parallel drafts under a masked block context are not sampled from the AR factorization, so recomputing `q_i` at verify time breaks distribution equivalence. Either argmax drafts (point-mass q, Eq 21 degenerates cleanly) or plumb the true q (T4).
- FLARE reports Diffusion-Trust (pure parallel denoise commit) is consistently weaker than AR-Trust on long structured outputs (code) — blocks commit without left-to-right syntactic verification. Consistent with our Plan 166 direction; no action.
