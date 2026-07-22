# Plan 557: RoVE — Rotary Value Embeddings Attention (Modelless)

**Date:** 2026-07-22
**Research:** [katgpt-rs/.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md](../.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md)
**Source paper:** [arXiv:2606.11275](https://arxiv.org/abs/2606.11275) — García-Castellanos, Weiler, Bekkers, Jul 2026 (RoVE)
**Source code:** [github.com/AGarciaCast/RoVE](https://github.com/AGarciaCast/RoVE)
**Target:** `katgpt-rs/crates/katgpt-core/src/rotary_value_embedding.rs` (new module, sibling to `position_group_action.rs`) + Cargo feature `rotary_value_embedding` (re-exported from root `katgpt-rs/Cargo.toml` as `rotary_value_embedding = ["katgpt-core/rotary_value_embedding"]`). Wiring in `katgpt-rs/crates/katgpt-attn/src/` (Phase 3) + `katgpt-rs/crates/katgpt-attn-match/src/chunked.rs` (Phase 4).
**Status:** Active — Phase 1 DONE (commit pending).

> **Numbering note.** Research note 452 + Plan 557 deliberately use *different* numbers because `.research/` and `.plans/` are independent namespaces with independent highwater markers — `.research/` was at 451 (next free = 452), `.plans/` was at 556 (next free = 557). The number collision in `.plans/452` (`452_simd_lut_dequant.md` already exists) is what forces the plan to 557. The research note at `.research/452_*.md` is the design doc; this plan is the execution tracker.

---

## Goal

Distill Research 452 into a generic, modelless, MIT-licensed Rust module that applies the RoPE rotation family to attention **values** (in addition to the existing Q/K rotations) and inverse-rotates the aggregated output back into the query's frame. Concretely, given:
- the V projection `V[j] = attn_wv · x[j]` (already computed by every attention variant),
- the existing RoPE rotation family `{R_t}` (already computed via `RopeFreqs` / `MixedRopeSummarizer` / `RopeAction`),

RoVE produces:

```
# Per-position value rotation (before softmax aggregation):
V_rot[j] = R_j · V[j]                              ← rotate_values_into(j, V[j], &mut V_rot[j])

# Post-aggregation inverse rotation (after softmax(A) · V_rot):
ỹ[i] = R_{−i} · Σ_j softmax(A)_ij · V_rot[j]       ← inverse_rotate_output_into(i, aggregated, &mut ỹ[i])
```

The two-step composition collapses to `ỹ[i] = Σ_j A_ij · (R_{j−i} · W_V) · x[j] = Σ_j A_ij · ψ_{j−i} · x[j]` — an attentive convolution with offset-indexed block-Toeplitz kernel `ψ_δ = R_δ · W_V` (paper Eq. 3). Standard RoPE is recovered when both calls are no-ops (the feature-off path).

**Sibling to GRAPE Plan 446 Issue 160** (`PositionGroupAction` trait), not a replacement. RoVE is the **first concrete hot-path consumer** of that trait — turning GRAPE's documented "vocabulary bridge for cold-path tools" into a real attention variant. The trait provides `apply_at(n, x, out)` and `apply_inverse_at(n, x, out)`; RoVE adds the wiring that calls these on the V projection and the post-softmax output.

**Parameter-free, FlashAttention-compatible, `O(nd)` linear overhead.** The paper reports consistent gains over RoPE on in-context learning, OOD perplexity (64% reduction at 16k tokens), and long-context retrieval (RULER mean NLL 6.62 → 4.33 at 354M with YaRN) at both 124M and 354M scales.

**Honest scope caveat (Research 452 §3 "Honest caveat").** The paper validates RoVE as a training-time architectural choice (model trained from scratch with V rotation). It does NOT validate RoVE as an inference-time retrofit onto RoPE-trained checkpoints. Plan 557 ships the primitive for forward-compat (scenario 1: RoVE-trained upstream checkpoints) + benchmarks the retrofit (scenario 2) as a Phase 5 honest PoC. Promotion to default-on requires Phase 5 to show no regression.

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed with anything else)

Goal: a compiling, tested, feature-gated module that implements the two RoVE primitives (`rotate_values_into`, `inverse_rotate_output_into`) as zero-allocation wrappers around `RopeAction::apply_at` / `apply_inverse_at`. No attention forward path wiring yet (Phase 3).

### Tasks

- [x] **T1.1** Add feature flag `rotary_value_embedding = ["position_group_action"]` to `katgpt-rs/crates/katgpt-core/Cargo.toml` features section (near `position_group_action`). Add a root-level alias `rotary_value_embedding = ["katgpt-core/rotary_value_embedding"]` to `katgpt-rs/Cargo.toml` (mirror the GRAPE Issue 160 root-facade pattern). **CORRECTION (implementation-time):** the original plan claimed RoVE does NOT imply `position_group_action` — this was wrong. `RopeAction` lives inside the `position_group_action` module which is `#[cfg(feature = "position_group_action")]`-gated at the MODULE level (`pub mod position_group_action`), not at the re-export level. If the feature is off, `RopeAction` does not exist at all, so RoVE cannot compile without it. The feature therefore implies `position_group_action` (which transitively implies `grapem_rodrigues`). This was verified during implementation — the corrected dependency is in the Cargo.toml comment.
- [x] **T1.2** Add `#[cfg(feature = "rotary_value_embedding")] pub mod rotary_value_embedding;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` (near `position_group_action`).
- [x] **T1.3** Implement `RoVeConfig` struct in `rotary_value_embedding.rs`:
  - `theta: f32` (=10000.0 default — matches paper's RoPE base and our existing `RopeFreqs` / `MixedRopeSummarizer`).
  - No other config — RoVE is parameter-free. The config exists only to thread `theta` (for YaRN-style rescaling in future work, see paper Appendix C).
- [x] **T1.4** Implement `pub fn rotate_values_into(action: &RopeAction, pos: usize, values: &[f32], out: &mut [f32])`:
  - Wraps `action.apply_at(pos as f32, values, out)`.
  - **Zero allocation.** Caller-owned `out` buffer (length `d`).
  - **Semantics:** rotates the V projection at position `pos` into the global frame. Mathematically `V_rot[pos] = R_pos · V[pos]`.
- [x] **T1.5** Implement `pub fn inverse_rotate_output_into(action: &RopeAction, pos: usize, aggregated: &[f32], out: &mut [f32])`:
  - Wraps `action.apply_inverse_at(pos as f32, aggregated, out)`.
  - **Zero allocation.** Caller-owned `out` buffer (length `d`).
  - **Semantics:** rotates the softmax-aggregated output at query position `pos` from the global frame back into the query's local frame. Mathematically `ỹ[pos] = R_{−pos} · aggregated[pos]`.
- [x] **T1.6** Implement `pub fn batch_rotate_values_into(action: &RopeAction, positions: &[usize], values: &[f32], out: &mut [f32], dim: usize)`:
  - Loops over `positions.len()` tokens, calling `rotate_values_into` per token.
  - `values` and `out` are flat `[n * d]` row-major; per-token slice = `[token_idx * dim .. (token_idx + 1) * dim]`.
  - **Zero allocation** in the loop (per-token slice borrows only).
  - This is the API the attention forward path calls once per layer (Phase 3 wiring).
- [x] **T1.7** Implement `pub fn batch_inverse_rotate_output_into(action: &RopeAction, positions: &[usize], aggregated: &[f32], out: &mut [f32], dim: usize)`:
  - Same as T1.6 for the inverse direction.
- [x] **T1.8** Write unit tests in `rotary_value_embedding.rs` `mod tests`:
  - [x] **G1 mechanics (identity at pos 0):** `rotate_values_into(action, 0, v, out)` writes `v` to `out` (rotation by angle 0 is identity).
  - [x] **G2 mechanics (round-trip):** `rotate_values_into(action, p, v, tmp)` followed by `inverse_rotate_output_into(action, p, tmp, recovered)` recovers `v` to f32 precision.
  - [x] **G3 relativity check:** `rotate_values_into(action, j, v, v_at_j)` then `inverse_rotate_output_into(action, i, v_at_j, v_at_i)` produces `R_{j−i} · v` — equivalent to a single `action.apply_at((j − i) as f32, v, v_at_i)`. Verifies the offset-indexed kernel `ψ_{j−i} = R_{j−i} · W_V` claim from the paper (Eq. 3).
  - [-] **G4 zero-degradation when feature off:** architecturally unverifiable from within the feature-gated module (when the feature is off, the module doesn't compile, so no test can run). The guarantee is structural (the `#[cfg]` on the module declaration) and is verified by the Exit Criterion `cargo build` (default features) unchanged. Deferred — not a test, a structural property.
  - [-] **G5 zero-alloc:** `rotate_values_into` and `inverse_rotate_output_into` perform zero heap allocations. Unit test uses the code-inspection pattern (matching `phase_rotation.rs` — can't use `#[global_allocator]` in lib unit tests due to parallel test harness collisions). The empirical `CountingAllocator` audit is deferred to the Phase 2 bench (`benches/rotary_value_embedding_goat.rs`).
  - [x] **G6 batch correctness:** `batch_rotate_values_into` produces identical results to per-token `rotate_values_into` for every token.
  - [x] **G7 odd-dim safety:** RoPE requires even `dim`; `RoVeConfig::build_rope_action` delegates to `RopeAction::with_theta`, which panics on odd `dim`.
- [x] **T1.9** Document module in `rotary_value_embedding.rs` header with:
  - Paper reference (arXiv:2606.11275 + GitHub link).
  - Three-lens summary (convolution / matrix mixer / local frame) from Research 452 §1.2.
  - Sibling-relationship note with `position_group_action` (GRAPE Issue 160) — RoVE is the first hot-path consumer of `RopeAction::apply_at` / `apply_inverse_at`.
  - Inference-only caveat (paper validates training-time only; Phase 5 PoC settles retrofit).
  - FlashAttention-compat note: rotations act on V (pre-kernel) and aggregated output (post-kernel), never on the `n×n` score matrix.

### Phase 1 Exit Criteria
- [x] `cargo build --features rotary_value_embedding -p katgpt-core` compiles clean.
- [x] `cargo test --features rotary_value_embedding -p katgpt-core --lib rotary_value_embedding` passes (G1–G7: 9/9 tests).
- [x] `cargo clippy --features rotary_value_embedding -p katgpt-core --lib` zero warnings (pre-existing test warnings in `bench_453_*` are unrelated).
- [x] `cargo build` (default features) unchanged — RoVE is opt-in, no impact on existing paths.

---

## Phase 2 — GOAT Gate (CORE — required before any wiring)

Goal: prove the primitive is correct, fast, alloc-free, and FlashAttention-compatible. Promotion to default-on DEFERRED until Phase 5 (retrofit PoC) settles the open question.

### Tasks

- [x] **T2.1** Write `benches/bench_557_rotary_value_embedding_goat.rs` with GOAT benchmarks (named per repo convention `bench_NNN_*`, not `rotary_value_embedding_goat.rs`):
  - **G1 bit-identical to RoPE-when-disabled:** pos=0 identity is exact (cos=1, sin=0 in IEEE); round-trip at nonzero pos holds to f32 precision (1 ULP from library cosf/sinf, budget 1e-6). Feature surgical scope verified structurally (module cfg-gated in lib.rs). **PASS** (worst 1.79e-7).
  - **G2 latency overhead:** `batch_rotate_values_into` + `batch_inverse_rotate_output_into` per-layer cost at `n=1024, d=768`. Target: `< 5%` of `O(nd²)` V projection via `types::math::matmul`. **FAIL** (6.45%) — honest: scalar rotation (~0.7 GFLOP/s) vs SIMD matmul (~17 GFLOP/s) is a ~24× throughput gap that inflates the 0.13% FLOP ratio to 6.45% wall-clock. SIMD RoVE (Phase 3) is the unblock path. Gate NOT relaxed.
  - **G3 no-regression:** opt-in + additive; default build clean; 9/9 Phase 1 tests pass with feature on. **PASS**.
  - **G4 zero steady-state alloc:** `CountingAllocator` over 1000 calls on batch hot path — **PASS** (0/0).
  - **G5 FlashAttention output-equivalence:** two-path comparison (RoVE: rotate V → aggregate → inverse-rotate; reference: per-(i,j) R_{j−i} rotation) on n=16, d=32 random fixture. **PASS** (rel err 2.69e-8 < 1e-4 budget). Proves `(R_{−i} · Σ_j A_ij · R_j · V_j) = (Σ_j A_ij · R_{j−i} · V_j)`.
- [x] **T2.2** Run the GOAT gate. Record results in `.benchmarks/557_rotary_value_embedding_goat.md`. Honest reporting: G2 FAIL recorded as ❌ with the actual numbers (6.45% vs 5%) + documented root cause (scalar-vs-SIMD throughput gap). No target relaxation.

### Phase 2 Exit Criteria
- [x] G1, G3, G4, G5 PASS. G2 honest ❌ (6.45% vs 5%) with documented reason (scalar throughput gap) + deferral to Phase 3 SIMD work.
- [x] `.benchmarks/557_rotary_value_embedding_goat.md` written.
- [x] **Promotion deferred** — two independent blockers: G2 FAIL + Phase 5 retrofit PoC not done.

---

## Phase 3 — Hot-Path Wiring in `katgpt-attn` (CORE)

Goal: add an opt-in forward path that calls `rotate_values_into` and `inverse_rotate_output_into` from a real attention variant. Mirror the existing RoPE-on-QK call site in `dash_attn/forward.rs`.

### Tasks

- [ ] **T3.1** Identify the RoPE-on-QK call site in `katgpt-attn/src/dash_attn/forward.rs`. Per the codebase grep (Research 452 §2.1), RoPE is currently applied to Q and K but never to V. Document the exact line + helper function (`apply_rope_phase_shift` or in-place `RopeFreqs::apply`).
- [ ] **T3.2** Add an opt-in RoVE branch in the same forward path, gated by `#[cfg(feature = "rotary_value_embedding")]`:
  - After `types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kv_dim, n)` (line ~113), if RoVE is enabled: `rotate_values_into(&rope_action, pos, &ctx.v, &mut ctx.v_rot)`.
  - After the softmax-weighted sum is computed into `ctx.attn_out`, if RoVE is enabled: `inverse_rotate_output_into(&rope_action, pos, &ctx.attn_out, &mut ctx.attn_out_final)`.
  - `ctx.v_rot` and `ctx.attn_out_final` are pre-allocated scratch buffers in `ForwardContext` (zero-alloc per token; reused across tokens in the prefill loop).
- [ ] **T3.3** Repeat T3.2 for `forward_dash_attn_decode` (the decode path).
- [ ] **T3.4** Add a feature-gated `RoVeToggle` to the dash_attn `Config` (mirror the existing `wall_config: Option<WallConfig>` pattern from Plan 173). When `Some`, RoVE is active; when `None`, RoVE is off and the code path is identical to today.
- [ ] **T3.5** Write integration tests in `katgpt-attn`:
  - **G6 output change:** with RoVE on, the forward path output differs from RoVE off (the value rotation is non-trivial — `R_{j−i} ≠ I` for `j ≠ i`).
  - **G7 determinism:** two runs with identical input produce bit-identical output (RoPE rotations are pure float arithmetic; no RNG).
  - **G8 no-panic on edge cases:** single-token sequence (no aggregation; `R_{−0} · R_0 · V = V`), zero-position (identity), large-position (rotation by large angle — verify finiteness, not correctness).

### Phase 3 Exit Criteria
- [ ] `cargo build --features rotary_value_embedding,dash_attn -p katgpt-attn` compiles clean.
- [ ] `cargo test --features rotary_value_embedding,dash_attn -p katgpt-attn` passes.
- [ ] The dash_attn forward path supports RoVE as an opt-in toggle.

---

## Phase 4 — Attention Matching Fusion (modelless, optional)

Goal: when RoVE + Attention Matching both active, fit `C_V` in position-free V space. Mirror the existing key-side `apply_rope_phase_shift` pattern in `katgpt-attn-match/src/chunked.rs`.

### Tasks

- [ ] **T4.1** In `ChunkedCompactor::compact_text_based` (`katgpt-attn-match/src/chunked.rs`), the existing key-side path is:
  ```rust
  let pf = PositionFreeBridge::new(ROPE_THETA, d);
  let pos_free_keys = pf.un_rotate_f32(chunk_keys, chunk.start_pos);
  // ... compact pos_free_keys ...
  // ... re-rotate at compacted position ...
  ```
  Add a parallel value-side path gated by `#[cfg(feature = "rotary_value_embedding")]`:
  ```rust
  let pos_free_values = pf.un_rotate_f32(chunk_values, chunk.start_pos);
  // ... compact pos_free_values together with pos_free_keys ...
  // ... re-rotate at compacted position ...
  ```
- [ ] **T4.2** Update the `ChunkedCompactor` API to accept an optional `rove_active: bool` flag (or a `RoVeToggle`). When true, the compactor un-rotates V before compaction and re-rotates after; when false, the path is unchanged.
- [ ] **T4.3** Write integration tests:
  - **G9 compaction fidelity under RoVE:** when RoVE is active, the compacted `(Ck, Cv)` produces attention output within the same fidelity bound as without RoVE (≥0.991 cosine, matching Plan 297 Phase A's existing gate).
  - **G10 position-consistency:** the re-rotated `Cv` at the compacted position, when fed back into a RoVE-aware forward path, produces output consistent with the un-compacted RoVE-aware forward path (within the same fidelity bound).

### Phase 4 Exit Criteria
- [ ] `cargo test --features rotary_value_embedding,attention_matching -p katgpt-attn-match` passes.
- [ ] The two features compose cleanly with no fidelity regression.

---

## Phase 5 — Honest Retrofit PoC (HONEST CAVEAT — settles the open question)

Goal: settle whether RoVE as an inference-time retrofit onto RoPE-trained checkpoints helps, hurts, or is neutral. This is the question the paper does NOT answer (paper trains from scratch with RoVE; our engine serves upstream checkpoints).

**This phase is honest research, not a feature gate.** The output is a `.benchmarks/557_rove_retrofit_poc.md` document recording the result. No matter the outcome, Phase 5 informs the promotion decision.

### Tasks

- [ ] **T5.1** Train a toy GPT-2 (12 layers, 12 heads, d=768 — the paper's small config, ~124M params) on FineWebEdu-10B *without* RoVE. **This requires GPU training → riir-train.** Coordinate with riir-train to land a small training run; the resulting checkpoint is the "RoPE-trained baseline".
- [ ] **T5.2** At inference, benchmark the RoPE-trained checkpoint in three configurations:
  - **A) RoPE-only** (today's baseline).
  - **B) RoVE retrofit** (apply V rotation at inference, even though the model was trained without it).
  - **C) RoVE-trained from scratch** (control — train a second checkpoint WITH RoVE; this should match the paper's numbers and validates our RoVE impl).
- [ ] **T5.3** Measure on the same three benchmarks as the paper:
  - Core ICL accuracy (DCLM-Core few-shot).
  - OOD perplexity at 512, 1024, 2048, 4096 tokens.
  - RULER long-context retrieval at 4k (NIAH, Variable Tracking).
- [ ] **T5.4** Write `.benchmarks/557_rove_retrofit_poc.md` with:
  - A vs B comparison (retrofit effect on RoPE-trained model).
  - A vs C comparison (validates our RoVE impl against the paper).
  - Honest verdict: if B > A → retrofit helps, RoVE is a default-on candidate. If B ≤ A → retrofit is neutral or harmful, RoVE stays opt-in for forward-compat only.
- [ ] **T5.5** If T5.4 verdict is "retrofit helps" AND Phase 2 G1–G5 PASS → promote `rotary_value_embedding` to default-on. Update `katgpt-rs/Cargo.toml` default features, update README Feature Showcase, update `.benchmarks/557_*_promotion_session.md` with the promotion record.
- [ ] **T5.6** If T5.4 verdict is "retrofit neutral or harmful" → keep `rotary_value_embedding` opt-in. Document the rationale in `.benchmarks/557_rove_retrofit_poc.md` and in the README Feature Showcase entry.

### Phase 5 Exit Criteria
- [ ] `.benchmarks/557_rove_retrofit_poc.md` written.
- [ ] Promotion decision recorded (default-on vs opt-in) with honest justification.

---

## Constraints

- **Modelless only.** No new parameters; no gradient descent; no training in the primitive itself. (Phase 5 *uses* a trained checkpoint but only to benchmark the inference-time retrofit question — the primitive is still modelless.)
- **Zero allocation in hot paths.** `rotate_values_into` and `inverse_rotate_output_into` are pure float arithmetic over caller-owned buffers. The `ForwardContext` scratch buffers (`ctx.v_rot`, `ctx.attn_out_final`) are pre-allocated once per forward pass, reused across tokens.
- **FlashAttention-compat.** The rotations act on V (before the kernel call) and on the aggregated output (after the kernel call). Never on the `n×n` score matrix. Phase 2 G5 verifies this via the output-equivalence test.
- **Sigmoid, not softmax** (AGENTS.md global rule). RoVE does not introduce any new activation — the softmax is the existing attention softmax. No new scoring function; the value rotation is a linear operation.
- **Even dim only.** RoPE requires even `dim` (per-pair rotation). `RoVeConfig` panics on odd `dim`, mirroring `RopeAction::with_theta`.
- **No YaRN in this plan.** The paper composes RoVE with YaRN frequency interpolation for OOD contexts. YaRN is not shipped in katgpt-rs today (fixed `θ_0 = 10000`). Adding YaRN is a separate future plan; RoVE is fully functional without it (the paper's "RoVE (ours)" row in Table 1 is without YaRN).

---

## Honest caveats (carried from Research 452 §3)

1. **Retrofit is unvalidated by the paper.** Phase 5 is mandatory before any default-on promotion. The structural argument cuts both ways: RoVE makes the OV circuit offset-aware, but the model's `W_V` was trained under the offset-blind assumption. We do not know whether the retrofit helps or hurts until we measure.
2. **Substrate dependency on `position_group_action` feature gate.** Phase 1 T1.1 asserts RoVE does NOT imply `position_group_action`. This is because `RopeAction` is a concrete struct with inherent methods `apply_at` / `apply_inverse_at` (from `impl PositionGroupAction for RopeAction`); the trait dispatch is static. Verify this with `cargo build --features rotary_value_embedding --no-default-features` — it should compile with `position_group_action` off but `RopeAction` reachable.
3. **No new pillar.** RoVE touches transformer attention substrate only. It does not connect to HLA / latent_functor / cgsp_runtime / neuron-shard / LatCal. The verdict is GOAT (engine completeness), not Super-GOAT (game-AI moat).
4. **Phase 5 requires GPU training coordination.** T5.1 is a riir-train task (train the toy GPT-2 baseline). Block Plan 557 Phase 5 on riir-train availability; do not implement a CPU-only toy that wouldn't match the paper's setup.

---

## References

- **Research note:** [`.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md`](../.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md)
- **Source paper:** [arXiv:2606.11275](https://arxiv.org/abs/2606.11275) — García-Castellanos, Weiler, Bekkers.
- **Source code:** [github.com/AGarciaCast/RoVE](https://github.com/AGarciaCast/RoVE)
- **Closest cousin plans:**
  - [`.plans/446_GRAPE_Group_Representational_Position_Encoding.md`](446_GRAPE_Group_Representational_Position_Encoding.md) — provides the `PositionGroupAction` trait + `RopeAction` that RoVE consumes. **Plan 446 does not exist as a single file** — it landed as Issues 159/160/161/163 (all removed per noise rule; verdicts in `.benchmarks/457`/`458`/`459`/`460`). See Research 446 §4 for the full follow-up record.
  - [`.plans/271_attention_matching_compaction.md`](271_attention_matching_compaction.md) — KV compaction that preserves RoPE on keys; Phase 4 of this plan extends it to RoVE-aware V space.
  - [`.plans/173_wall_attention_diagonal_gate.md`](173_wall_attention_diagonal_gate.md) — Wall Attention (orthogonal axis; Wall replaces RoPE on QK, RoVE extends RoPE to OV).
- **Canonical format example:** [`.plans/271_attention_matching_compaction.md`](271_attention_matching_compaction.md)
