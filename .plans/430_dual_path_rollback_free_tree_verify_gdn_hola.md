# Plan 430: Dual-Path Rollback-Free Tree Verification — GDN Recurrent State × HOLA Hippocampal Cache

**Date:** 2026-07-13
**Research:** [katgpt-rs/.research/407_Trees_from_Marginals_GDN_Tree_Verify.md](../.research/407_Trees_from_Marginals_GDN_Tree_Verify.md) §2.2 (Fusion idea)
**Source papers:** [arXiv:2607.06763](https://arxiv.org/abs/2607.06763) §3.4 (GDN tree verify) + [arXiv:2607.02303](https://arxiv.org/abs/2607.02303) (HOLA hippocampal cache)
**Target:** `katgpt-rs/crates/katgpt-core/src/gdn_tree_verify/mod.rs` (extend existing module) + Cargo feature `gdn_hola_tree_verify` (implies `gdn_tree_verify` + `hippocampal_cache`)
**Status:** Active — Phase 1 (design)

---

## Goal

Fuse Plan 424 (GDN rollback-free tree verification) with Plan 395 (HOLA hippocampal exact KV cache) into a **dual-path tree verifier** that scores speculative draft tree nodes against BOTH the GDN recurrent state (via masked triangular solve) AND the HOLA hippocampal cache (via ancestor-masked softmax read) — with **zero rollback on either path**.

This is the open fusion from Research 407 §2.2:
> "GDN tree-verify × HOLA hippocampal cache × DDTree. The rollback-free verify algorithm composes with HOLA's surprise-evicted cache: verify speculative tree branches against the GDN recurrent state via the masked solve, AND against the HOLA hippocampal cache via a parallel ancestor-masked softmax path, with no rollback on either. This produces a speculative decoding path for GDN+HOLA models that has zero rollback overhead — a combination none of the three primitives alone achieves."

**Why this matters:** GDN2's fixed-size recurrent state compresses context but loses exact long-range recall. HOLA's hippocampal cache recovers exact recall for high-surprise tokens but is a flat top-w set with no tree-verification story. Fusing them at the speculative tree-verify layer gives both exact-recall recovery (HOLA) AND rollback-free tree scoring (GDN masked solve) — the hippocampal complement to the compressive recurrent state, extended to branching speculative drafts.

**GOAT gate:** G1 (correctness — dual-path output matches sequential GDN2+HOLA forward), G2 (perf — dual-path verify ≤ GDN-only verify + HOLA-only read, no rollback), G3 (no-regression — with `hippocampal_cache` off, byte-identical to Plan 424), G4 (alloc-free hot path), G5 (retrieval — synthetic needle test: dual-path recovers ≥80% where GDN-only recovers ≤30% at 8× length).

---

## Phase 1 — Dual-path verify skeleton (CORE)

### Tasks

- [ ] **T1.1** Add feature `gdn_hola_tree_verify = ["gdn_tree_verify", "hippocampal_cache"]` to `katgpt-core/Cargo.toml`. The feature implies both parents — no standalone code without them.
- [ ] **T1.2** Extend `GdnLayerParams` (or define a sibling `GdnHolaLayerParams`) to carry an optional `&HippocampalCache<D, W>` reference per layer. The cache is read-only during verify (same discipline as `S₀`).
- [ ] **T1.3** Implement `fn compute_out_hola(...)` — the hippocampal cache read path for tree-structured queries. For each node `i` in topological order:
  - Build `block_kv_i` = `[(k_j, v_j) for j in ancestors(i)]` from the tree's own key/value vectors (the draft tokens along the root→i path).
  - Call `cache.read_cache_into(q_i, gamma, &block_kv_i, &mut o_hola_i)`.
  - The ancestor masking is **by construction** — `block_kv_i` only contains ancestor tokens. No explicit bitmask needed on the cache path (unlike the GDN masked solve which needs the `X` interaction matrix).
  - Output: `O_hola[i]` per node, same shape as `O_gdn[i]`.
- [ ] **T1.4** Implement `fn verify_gdn_hola_tree_into(...)` — the top-level dual-path verify:
  - Step 1: run the existing GDN masked triangular solve (`verify_gdn_tree_into` from Plan 424) → `O_gdn[i]` per node.
  - Step 2: run the HOLA cache read path (T1.3) → `O_hola[i]` per node.
  - Step 3: residual-add `O[i] = O_gdn[i] + O_hola[i]` (the cache complements the recurrent state — HOLA §3.5 design).
  - Both steps are read-only (use `S₀` and the cache as-is). Zero rollback.
- [ ] **T1.5** Unit test: dual-path verify on a chain tree matches a sequential GDN2+HOLA forward pass (`test_dual_path_chain_matches_sequential`).

---

## Phase 2 — Dual-path commit-on-accept

### Tasks

- [ ] **T2.1** Implement `fn commit_accepted_dual(...)` — the commit path after Traversal verification picks the accepted leaf:
  - **GDN commit:** replay the delta-rule recurrence along the accepted path via `commit_accepted` (existing Plan 424). Updates `S₀`.
  - **HOLA commit:** for each token on the accepted path, call `cache.observe(k_t, v_t, beta_t, residual_norm_t)` (existing Plan 395 API). The `(beta_t, residual_norm_t)` come from the GDN delta-rule update at each step — they're already computed during the GDN commit. Pipe them to the cache.
  - Both commits are append-only / in-place-update — no rollback needed.
- [ ] **T2.2** Unit test: dual-path commit produces the same `S₀` and cache state as a sequential GDN2+HOLA forward over the accepted path (`test_dual_path_commit_matches_sequential`).
- [ ] **T2.3** Multi-head extension: `verify_gdn_hola_tree_multihead` + `commit_accepted_dual_multihead` — mirror the Plan 424 T4.1 multi-head API. The cache is shared across heads (paper form: one cache per layer, not per head); the GDN state is per-head.

---

## Phase 3 — Integration: hybrid QwenDeltaNet + HOLA speculative step

### Tasks

- [ ] **T3.1** Extend `speculative_step_gdn_tree` (Plan 424 T4.3) to `speculative_step_gdn_hola_tree` — the hybrid path:
  - Draft: DFlash produces marginals → DDTree builds the tree.
  - Verify: for each GDN layer, run `verify_gdn_hola_tree_into` (dual-path).
  - Accept: p/q rejection sampling along the best path.
  - Commit: `commit_accepted_dual` (GDN state + HOLA cache).
- [ ] **T3.2** riir-ai consumer: extend `forward_tree_qwen_deltanet` (Plan 424 T4.3c) to wire the hippocampal cache into the DeltaNet layers. The cache is a per-layer field on the QwenDeltaNet runtime state.
- [ ] **T3.3** Integration test: `speculative_step_gdn_hola_tree` produces valid tokens, deterministic for same seed, and the cache state is consistent across verify→commit cycles.

---

## Phase 4 — GOAT gate (benchmarks + promote decision)

### Tasks

- [ ] **T4.1 (G1 — correctness)** Test: `verify_gdn_hola_tree` on random trees (T=16,32,64,128) with a populated HOLA cache produces outputs within `1e-3` of a per-branch sequential GDN2+HOLA forward reference. **PASS bar: all 4 tree sizes within tol.**
- [ ] **T4.2 (G2 — perf)** Benchmark `benches/bench_430_dual_path_verify.rs`: dual-path verify time vs GDN-only verify (Plan 424) + HOLA-only read (Plan 395) summed. **PASS bar: dual-path ≤ 1.2× GDN-only verify time** (the HOLA read is O(W·D) per node, small vs the O(T²·d_k) masked solve at large T).
- [ ] **T4.3 (G3 — no-regression)** With `hippocampal_cache` feature OFF, `verify_gdn_hola_tree_into` must be byte-identical to `verify_gdn_tree_into` (Plan 424). Test: `test_dual_path_degrades_to_gdn_only_when_cache_disabled`. Plus `cargo test -p katgpt-core --features gdn_tree_verify --lib` (existing Plan 424 tests still pass).
- [ ] **T4.4 (G4 — alloc-free)** `verify_gdn_hola_tree_into` allocates 0 times on steady-state (CountingAllocator). The HOLA read path uses a stack-local logits buffer (per Plan 395 T1.4); the dual-path verify reuses the Plan 424 scratch buffers + adds a stack-local `o_hola` buffer.
- [ ] **T4.5 (G5 — retrieval gain)** Synthetic multi-key associative recall: 8 needles in a 4k-token stream. After verify→commit cycles, the dual-path (GDN+HOLA) recovers ≥80% of needles at 8× training length where GDN-only recovers ≤30%. This is HOLA's F1 fusion gate (Research 378 §2.4) extended to the tree-verify setting. **PASS bar: ≥80% dual-path vs ≤30% GDN-only at 8× length.**

### Promote decision

- [ ] **T4.6** If G1–G5 pass → `gdn_hola_tree_verify` stays **opt-in** (NOT default — requires both `gdn_tree_verify` (opt-in) + `hippocampal_cache` (opt-in) + a trained γ vector or modelless γ=1). Results documented in `.benchmarks/430_dual_path_verify_goat.md`.

---

## Key Design Decisions

1. **Ancestor masking by construction, not by bitmask.** The GDN masked triangular solve needs the explicit `X` interaction matrix with ancestor bitmasks (paper §3.4 Eq. 9) because the recurrent state couples all positions multiplicatively. The HOLA cache read is a softmax attention — ancestor masking is enforced by only passing ancestor tokens in `block_kv`. No bitmask needed on the cache path. This asymmetry is correct: the GDN solve is a global linear system; the HOLA read is a local attention.

2. **Residual-add, not replace.** The dual-path output is `O[i] = O_gdn[i] + O_hola[i]`. This matches HOLA §3.5: the hippocampal cache is a *complement* to the compressive recurrent state, not a replacement. The GDN state captures the smoothed/averaged context; the cache captures exact high-surprise tokens. Both contribute to the output.

3. **Commit pipes GDN residuals to HOLA.** The GDN delta-rule commit computes `(beta_t, residual_norm_t)` at each step — exactly what `HippocampalCache::observe` needs. Zero extra compute: the commit path was already computing these for the state update; we pipe them to the cache as a side effect. This is the "β·‖e‖ is free" insight from Research 378 §2.1.

4. **Read-only verify, dual-write commit.** Both the GDN state (`S₀`) and the HOLA cache are read-only during verify. The commit writes both: GDN via `commit_accepted` (state update along accepted path), HOLA via `observe` (append high-surprise tokens to cache). Zero rollback on either.

5. **Not a replacement for Plan 424 or Plan 395.** This plan composes them. With `hippocampal_cache` OFF, it degrades to Plan 424 (byte-identical). With `gdn_tree_verify` OFF, the HOLA cache read is still usable (Plan 395 standalone). The fusion only activates when both features are on.

6. **CPU-first.** Both parents are CPU SIMD. The dual-path verify maps to: (1) existing blocked SIMD matmul for the GDN solve, (2) existing stack-local softmax for the HOLA read. No GPU dependency.

---

## Out of scope (tracked elsewhere)

- **Traversal verification** (paper ref [10]) — the acceptance coupling scheme. This plan ships the dual-path verify primitive (per-node outputs); the acceptance policy is separate (Plan 012 DDTree has its own).
- **HOLA γ training** — the decoupled RMSNorm-γ is a model parameter. Plan 395 §5 Risk #2 identifies γ=1 (modelless) as the default; if retrieval fails at γ=1, a deterministically-constructed γ (e.g., `γ_i = √d / ‖k_i‖`) may close the gap. riir-train follow-up if needed.
- **GPU fused kernel** — CPU-first for correctness. riir-gpu task if throughput requires it.
- **HOLA × temporal_deriv fusion** (Research 378 F2) — a second surprise channel. Orthogonal to this plan (which fuses HOLA with GDN tree verify, not with temporal_deriv).

---

## References

- **Plan 424** (`katgpt-rs/.plans/424_gdn_tree_verification_primitive.md`) — GDN rollback-free tree verify (parent, COMPLETE, GOAT PASSED)
- **Plan 395** (`katgpt-rs/.plans/395_hippocampal_exact_kv_cache.md`) — HOLA hippocampal cache (parent, COMPLETE)
- **Research 407** (`katgpt-rs/.research/407_Trees_from_Marginals_GDN_Tree_Verify.md`) §2.2 — the fusion idea source
- **Research 378** (`katgpt-rs/.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md`) §2.4 F1 — GDN2 + HOLA cache fusion gate
- **Plan 012** — DDTree (tree structure source)
- **Plan 105** — GDN2 backbone (default-on)
- **Paper:** [arXiv:2607.06763](https://arxiv.org/abs/2607.06763) §3.4 (GDN tree verify) + [arXiv:2607.02303](https://arxiv.org/abs/2607.02303) (HOLA)
