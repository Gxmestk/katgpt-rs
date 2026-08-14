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

- [ ] **T1 — SoftmaxArgmax policy (Eq 21).** Add `DraftAcceptPolicy` to `SelfSpecConfig`; implement in Phase 3: draw `u`, accept iff `u ≤ p_dist[draft_tok]`, else sample correction from `p_dist` renormalized over `≠ d_i` and break. Argmax drafting option for the D2F drafter (or accept the approximation under sampled drafts — document which).
- [ ] **T2 — G1 exactness proof.** Toy model + fixed seeds: empirical distribution of accepted tokens over ≥10k rounds must match direct target sampling (chi-square / total-variation bound). The PrefixMatch baseline must FAIL this test — that is the proof of gain. UQ note: this is a distribution-equivalence gate, not a calibrated-UQ claim — conformal floor not applicable (no interval/coverage claim).
- [ ] **T3 — TruncatedArgmax (Eq 22).** Top-k truncated p and q; sparse verify on k-wide support.
- [ ] **T4 — Exact-Truncated (Eq 8).** Plumb per-position top-k proposal probs out of `d2f_decode_block` (extend `D2fBlockResult`); store `(ids, probs)` per round; true p/q rejection.
- [ ] **T5 — Streaming verify.** Replace `[(K+1)×V]` materialization with two-pass logits streaming: pass 1 `(max, sum-exp)` + gather `p_i(d_i)`; on rejection pass 2 Gumbel-max over `[p−q]₊` (or renormalized-exclude for Eq 21) selecting the correction in one argmax sweep. Benchmark verify latency + peak memory at large vocab (Gemma-2 256K) vs the flat buffer; keep PrefixMatch path as control.
- [ ] **T6 — GOAT gate.** G1 = T2 exactness; G2 = acceptance rate + verify latency vs PrefixMatch (target: ≥ acceptance rate at equal-or-better latency); G3 = existing `tri_mode`/`dllm` test suites pass; G4 = steady-state alloc-free preserved (buffers pre-allocated, no per-round allocs). Promote winning policy to the `tri_mode` default if G1–G4 pass; demote PrefixMatch to fallback.
- [ ] **T7 — FlashARConsensus audit.** `FlashARConsensusVerifier` (Plan 166) replaced prefix-match with dual-path consensus — check whether its acceptance is distribution-preserving under the same analysis; file follow-up if not.
- [-] **T8 — deferred:** recurrent-state cache-and-scatter rewind for linear-attention backbones (FLARE App B.4). No hybrid-backbone speculative-decode consumer today; revisit when riir-ai's linear-attention lineage (HLA / Luce / KDA) grows an inference serving path. Recorded in Research 480 §2.3 D3.

## Notes

- Proposal-consistency subtlety (why Eq 8 needs stored q): parallel drafts under a masked block context are not sampled from the AR factorization, so recomputing `q_i` at verify time breaks distribution equivalence. Either argmax drafts (point-mass q, Eq 21 degenerates cleanly) or plumb the true q (T4).
- FLARE reports Diffusion-Trust (pure parallel denoise commit) is consistently weaker than AR-Trust on long structured outputs (code) — blocks commit without left-to-right syntactic verification. Consistent with our Plan 166 direction; no action.
