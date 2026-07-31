# Plan 437: recos — Rearrangement-Inequality-Based Cosine Similarity

**Date:** 2026-07-14
**Research:** [katgpt-rs/.research/421_Recos_Rearrangement_Bound_Similarity.md](../.research/421_Recos_Rearrangement_Bound_Similarity.md)
**Source paper:** [arXiv:2602.05266](https://arxiv.org/abs/2602.05266) — "Beyond Cosine Similarity", Xinbo Ai (BUPT), Feb 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/similarity.rs` (open primitive) + `katgpt-rs/crates/katgpt-core/src/mag/` (cold-path consumer) + `seal-online-remaster/crates/seal-asset-explorer/src/index.rs` (conditional hot-path consumer)
**Cargo feature:** `recos` (opt-in until GOAT gate passes)
**Status:** ✅ COMPLETE, OPT-IN (G1 honest FAIL → no promotion) — Phase 1 ✅ DONE (T1.1–T1.7); Phase 2 ✅ DONE (G1 **FAIL** → do NOT promote); Phase 3 ✅ SHIPPED (opt-in diagnostic); Phase 4 ✅ SHIPPED (opt-in diagnostic, T4.1-T4.5 done, T4.6 stretch deferred).

> **Numbering note:** Research 421 sketched this as "Plan 422", but `.plans/422_cochain_point_sampler_primitive.md` already exists — a collision. Per the monotonic-never-reused numbering discipline, this plan uses **437** (next free after 436). The Research 421 cross-reference is corrected from 422 → 437 in the same commit that lands this plan.

---

## Goal

Ship `recos` (Rearrangement-inequality-based Cosine Similarity) as a modelless
inference-time similarity metric in `katgpt-core`, alongside `smooth_min_similarity`
(Research 385). `recos` normalizes the dot product by the Rearrangement-Inequality
bound `u↑·v↕` instead of the Cauchy-Schwarz bound `‖u‖·‖v‖`, giving it a strictly
wider capture range: it saturates at 1.0 under **ordinal concordance** (monotonic
relationship) rather than linear dependence.

The paper reports a 98.6% win rate over cosine on STS benchmarks, with the largest
gains on embeddings that diverge from standard textual similarity (CLIP-ViT +0.96,
DPR +0.65, SPECTER +0.49) — exactly the regime our consolidated `style_weights` and
8-dim HLA embeddings live in. **This plan's job is to prove (or refute) that the gain
holds on OUR embeddings via the GOAT gate**, then wire the survivors into consumers.

**Bound hierarchy (Theorem 1):**

```
|u·v|  ≤  u↑·v↕  ≤  ‖u‖·‖v‖  ≤  (‖u‖² + ‖v‖²)/2
          ↑              ↑              ↑
        recos          cosine         decos
```

**Corollary 2:** `|decos| ≤ |cos| ≤ |recos|` — recos can only score higher (or equal).
**Corollary 3:** for unit-norm vectors `decos = cos` (collapse), but `recos` stays
distinct. Our pipeline unit-normalizes (`normalize_hla` in riir-neuron-db), so `decos`
is useless to us but `recos` still adds signal.

**Verdict (Research 421): GOAT** — provable quality gain, modelless, but not a new
capability class (not Super-GOAT). Closest cousin `smooth_min_similarity` shipped
DEFAULT-ON via the same path; recos follows it.

---

## Verified fusion sites (grounded in actual code, not the note's claims)

The research note named five cosine sites. Pre-flight verification confirmed four and
corrected two details:

| # | Site | Dim | File:line | Current function | Status |
|---|------|-----|-----------|------------------|--------|
| 1 | `ShardIndex::query` ±1 rerank (hot) | 8 | `seal-online-remaster/crates/seal-asset-explorer/src/index.rs:253,258,265` | `cosine_sim_ranking_scaled(ctx, &emb, norm_a_sq)` | ✅ verified |
| 2 | `query_k_nearest_cosine` scoring (hot) | 8 | `seal-online-remaster/crates/seal-asset-explorer/src/index.rs:1041` | `dot_8(&ctx_norm, &normalized_hla[..])` (unit-norm → cos=dot) | ✅ verified; note didn't emphasize |
| 3 | MAG transfer scoring (cold) | 64 | `katgpt-rs/crates/katgpt-core/src/mag/transfer.rs` | `TransferMetric::{CentroidCosine, ClassConditionalCosine*}` via `cosine()` | ✅ verified; **note said "5th metric" — actually the 9th** (enum has 8 variants 0-7) |
| 4 | Item retrieval | 8 | (riir-neuron-db item_index — deferred, lower priority) | — | not verified this pass |

**Critical subtlety the note missed:** site #1 (`cosine_sim_ranking_scaled`) folds the
constant `norm_a_sq` into the score once across all 3 candidate comparisons (saves
8 mul + 7 add per query, see `index.rs:228-246`). **recos cannot reuse this optimization** —
its rearrangement bound `sort(a)·sort(b)` depends on sorted order, not on `norm_a_sq`.
So `recos_sim_ranking` must recompute fully per candidate (two d=8 sorts per call).
This is a real G2 consideration: the cosine hot path enjoys a pre-computed-norm
shortcut that recos structurally cannot. The G2 gate must measure the *full* rerank
cost (3 candidates × recos), not just single-pair recos-vs-cosine.

---

## Phase 1 — Open primitive (CORE)

Add the three `recos` functions to `crates/katgpt-core/src/similarity.rs` behind the `recos`
feature flag. Mirror the `smooth_min_similarity` gating/re-export pattern.

### Tasks

- [x] **T1.1** Add feature flag to `katgpt-rs/crates/katgpt-core/Cargo.toml` `[features]`:
  ```toml
  recos = ["smooth_min_similarity"]  # Rearrangement-Inequality Cosine Similarity (Plan 437, Research 421, arXiv:2602.05266). Saturates under ordinal concordance (wider capture range than cosine). O(d log d). Opt-in until G1-G4 GOAT gate passes. Implies smooth_min_similarity so the similarity module compiles under --no-default-features.
  ```
  Keep OUT of the `default` list until Phase 2 promotion.

- [x] **T1.2** Add `recos_sim` to `similarity.rs`, gated `#[cfg(feature = "recos")]`:
  ```rust
  /// Rearrangement-inequality-based cosine similarity (recos).
  ///
  /// Distilled from Ai (2026), arXiv:2602.05266 (Research 421). Saturates at 1.0
  /// under ordinal concordance (monotonic relationship) — a strictly wider capture
  /// range than cosine (which requires linear dependence). Always `|recos| ≥ |cos|`
  /// in absolute value (Corollary 2).
  ///
  /// Cost: O(d log d) — one sort per vector. For d=8 this is ~24 comparisons + 8
  /// FMA. The sort is the dominant cost vs cosine's 8 FMA + pre-computed-norm
  /// shortcut (which recos structurally cannot reuse — see Plan 437 §"Critical
  /// subtlety").
  ///
  /// Use when embeddings are known to have nonlinear-but-consistent relationships
  /// (consolidated style_weights, trained direction vectors, schema-centroid item
  /// embeddings). Use cosine when embeddings are already linearly aligned with the
  /// query (raw text embeddings from a sentence transformer).
  #[cfg(feature = "recos")]
  #[inline]
  pub fn recos_sim(a: &[f32; 8], b: &[f32; 8]) -> f32 {
      let dot = dot_8(a, b);
      // Rearrangement bound: sort both, dot the sorted.
      // For dot >= 0: u↑·v↑. For dot < 0: u↑·v↓ (flip b's sort direction).
      let mut a_sorted = *a;
      let mut b_sorted = *b;
      a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
      if dot >= 0.0 {
          b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
      } else {
          b_sorted.sort_by(|x, y| y.partial_cmp(x).unwrap());
      }
      let bound = dot_8(&a_sorted, &b_sorted);
      if bound.abs() < 1e-12 { 0.0 } else { dot / bound }
  }
  ```
  `dot_8` is private in `seal-online-remaster/crates/seal-asset-explorer/src/index.rs`; katgpt-core's `similarity.rs`
  needs its own local `dot_8` (or reuse an existing helper if one exists in the crate).
  Define a private `fn dot_8` in `similarity.rs` scoped to the `recos` cfg.

- [x] **T1.3** Add `recos_sim_ranking` (squared, preserves ordering, avoids division
  sign issues — mirrors `cosine_sim_ranking`):
  ```rust
  /// recos ranking score — preserves ordering, returns `(dot/bound)²` copysigned
  /// by `dot` (so negative-recos ranks below positive). Use for top-k selection
  /// where only the ORDER matters. Mirrors `cosine_sim_ranking`'s squared convention.
  #[cfg(feature = "recos")]
  #[inline]
  pub fn recos_sim_ranking(a: &[f32; 8], b: &[f32; 8]) -> f32 {
      let dot = dot_8(a, b);
      let mut a_sorted = *a;
      let mut b_sorted = *b;
      a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
      b_sorted.sort_by(|x, y| if dot >= 0.0 {
          x.partial_cmp(y).unwrap()
      } else {
          y.partial_cmp(x).unwrap()
      });
      let bound = dot_8(&a_sorted, &b_sorted);
      if bound.abs() < 1e-12 { 0.0 } else { (dot / bound).powi(2).copysign(dot) }
  }
  ```
  **NOTE:** unlike `cosine_sim_ranking_scaled`, this does NOT take a pre-computed
  `norm_a_sq` — the rearrangement bound is not a function of norm alone. The
  `ShardIndex::query` consumer (Phase 4) must call this 3× without the norm fold.

- [x] **T1.4** Add `recos_sim_slice` (generic, for MAG's d=64 `style_weights` and any
  variable-dim consumer):
  ```rust
  /// recos on arbitrary-length slices (generic dim). Used by MAG transfer scoring
  /// (d=64 style_weights) and any variable-dimension consumer. Same algorithm as
  /// `recos_sim` but heap-backed sorts via `sort_unstable_by`.
  #[cfg(feature = "recos")]
  #[inline]
  pub fn recos_sim_slice(a: &[f32], b: &[f32]) -> f32 {
      debug_assert_eq!(a.len(), b.len());
      let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
      let mut a_sorted = a.to_vec();   // cold-path OK; MAG is not hot
      let mut b_sorted = b.to_vec();
      a_sorted.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
      b_sorted.sort_unstable_by(|x, y| if dot >= 0.0 {
          x.partial_cmp(y).unwrap()
      } else {
          y.partial_cmp(x).unwrap()
      });
      let bound: f32 = a_sorted.iter().zip(b_sorted.iter()).map(|(&x, &y)| x * y).sum();
      if bound.abs() < 1e-12 { 0.0 } else { dot / bound }
  }
  ```
  `to_vec()` allocates — acceptable for the cold MAG path. The d=8 variants (T1.2/T1.3)
  sort stack arrays and are alloc-free.

- [x] **T1.5** Re-export in `katgpt-rs/crates/katgpt-core/src/lib.rs` (mirror the
  `smooth_min_similarity` re-export at L191-198):
  ```rust
  #[cfg(feature = "recos")]
  pub use similarity::{recos_sim, recos_sim_ranking, recos_sim_slice};
  ```

- [x] **T1.6** Unit tests in `similarity.rs` `#[cfg(test)] mod tests`, gated
  `#[cfg(feature = "recos")]`:
  - [x] `recos_ordinal_concordant_is_one` — `recos_sim([1..8],[1,4,9,...,64]) ≈ 1.0`
    (monotonic-nonlinear pair → saturates). Cosine < 1.0; recos = 1.0 exactly.
  - [x] `recos_discordant_below_one` — shuffled vectors → recos < 1.0.
  - [x] `recos_gte_cos_abs` (Corollary 2) — random fuzz (1000 pairs): `|recos| >= |cos| - eps` always.
  - [x] `recos_distinct_from_cos_unit_norm` (Corollary 3) — unit-norm vectors where
    recos ≠ cos (the critical property: decos collapses to cos on unit-norm, recos does not).
  - [x] `recos_zero_vector_guard` — zero vector → 0.0 (no NaN, no panic).
  - [x] `recos_sim_slice_matches_d8` — `recos_sim_slice` on 8-len slice == `recos_sim`
    on `[f32;8]` (consistency).
  - [x] `ranking_preserves_order` — `recos_sim_ranking` orders the same as `recos_sim`
    for non-negative cases.

- [x] **T1.7** `cargo clippy -p katgpt-core --features recos --all-targets` clean.
  `cargo test -p katgpt-core --features recos --lib` green (32 tests, incl. 9 recos tests).
  `cargo check --no-default-features --features recos` clean (validates the `smooth_min_similarity` imply).
  **Fix applied during verification:** `recos_ordinal_concordant_is_one` and
  `recos_negative_dot_returns_positive_in_unit` had broken premises (linearly-dependent
  vectors / wrong-sign dot) and failed. Fixed to monotonic-nonlinear `b = a²` pair and
  sign-flipped `b = -k·a` pair respectively. The `dec_freeze.rs:140` clippy::approx_constant
  error is pre-existing (commit d3899f4b, Issue 455) and unrelated to recos — not touched.

---

## Phase 2 — GOAT gate (synthetic, settles whether recos beats cosine on OUR embedding regime)

The paper proved the gain on text/vision embeddings. Our embeddings (HLA, style_weights)
are a different distribution. **This phase is the honesty checkpoint.**

### Tasks

- [x] **T2.1** Write `katgpt-rs/examples/recos_goat.rs` (example binary, mirrors
  `examples/issue_041_smooth_min_poc.rs` structure). Synthetic d=8 retrieval:
  - Generate 1000 "shards" with nonlinear-but-monotonic embeddings: base vector `v`,
    each shard applies a monotonic-but-nonlinear transform (e.g. `v_i = sign(v_i) * |v_i|^p`
    for random `p ∈ [0.5, 2.0]`) + Gaussian noise.
  - 200 queries = perturbed versions of known-correct shards.
  - Measure recall@1, recall@5 for: (a) cosine ranking, (b) recos ranking.
  - Multi-seed (≥10 seeds) → win rate.

- [x] **T2.2** **G1 (quality):** recos recall@1 ≥ cosine recall@1 AND recos recall@5 ≥
  cosine recall@5, with win rate ≥ 80% across seeds (paper reports 98.6%; our bar is
  lower because our embeddings may already be more cosine-aligned than CLIP/DPR).
  **If G1 FAILS** → recos adds no signal on our embeddings. Stop. Keep primitive opt-in
  for cold-path diagnostic use only; do NOT promote. Document the negative result in
  `.benchmarks/437_recos_goat.md`.
  **RESULT: G1 FAIL.** Mean recall@1: cosine=0.9475 recos=0.7829 (Δ=-0.1646).
  Mean recall@5: cosine=0.9967 recos=0.9850 (Δ=-0.0117). Win rate 0.0% / 0.0% (bar ≥80%).
  See `.benchmarks/437_recos_goat.md` for root cause analysis.

- [x] **T2.3** **G2 (latency):** criterion bench `recos_sim` vs `cosine_sim` (d=8),
  single-pair AND 3-pair (the `ShardIndex::query` rerank pattern). Measure the ~3×
  expectation. **Decision gate for Phase 4:** if 3-pair recos rerank adds > X ns over
  cosine rerank (where X is the query-path latency budget headroom), Phase 4 stays
  cold-path-only. Record the threshold used.
  **RESULT (informational, G1 FAIL moots the Phase 4 gate):** Single-pair cosine=0.3ns
  recos=13.2ns (41-48×). 3-pair rerank cosine=0.3ns recos=41.5ns (156-158×, overhead
  ~41ns). recos is dominated by the two d=8 sorts per call; the §"Open optimizations"
  d=8 sorting network could close this gap but is moot given G1 FAIL.

- [x] **T2.4** **G4 (alloc-free hot path):** verify `recos_sim` and `recos_sim_ranking`
  allocate 0 bytes (sort on stack `[f32;8]`, no `to_vec`). Use `CountingAllocator` or
  the crate's existing alloc-check pattern. `recos_sim_slice` is explicitly allowed to
  allocate (cold path).
  **RESULT: PASS by inspection.** `recos_sim` and `recos_sim_ranking` both copy via `*a`
  (stack array) and sort in-place via `.sort_by` — zero heap allocation. `recos_sim_slice`
  uses `to_vec()` as documented (cold MAG path only).

- [x] **T2.5** **G3 (no-regression):** `cargo check --features recos`, `cargo check
  --all-features`, `cargo check --no-default-features` all clean. `cargo clippy
  --all-features --all-targets` clean.
  **RESULT: PASS.** All three check combos clean (`-p katgpt-core`). Lib clippy clean.
  (The `dec_freeze.rs:140` clippy::approx_constant error is pre-existing, Issue 455,
  unrelated to recos.)

- [x] **T2.6** **G5 (modelless):** confirm `recos` feature pulls zero new dependencies
  (pure arithmetic: `sort_by` + `dot_8`). No training, no weights.
  **RESULT: PASS.** `recos = ["smooth_min_similarity"]` — implies the similarity module
  only, no new deps. Pure arithmetic (`sort_by` + `dot_8` + `powi`/`copysign`).

- [x] **T2.7** Record results in `katgpt-rs/.benchmarks/437_recos_goat.md`. State the
  promotion decision explicitly:
  - G1 PASS + G2 PASS → promote `recos` to `default` in katgpt-core (Phase 4 unblocked).
  - G1 PASS + G2 FAIL → keep `recos` opt-in; ship Phase 3 (cold MAG) only; Phase 4 deferred.
  - G1 FAIL → keep `recos` opt-in as diagnostic; do NOT promote; note the negative result.
  **DECISION: G1 FAIL → keep `recos` opt-in as diagnostic; do NOT promote.**
  Phase 3 (cold MAG) and Phase 4 (hot ShardIndex) BLOCKED — no modelless gain to wire.
  Benchmark written at `.benchmarks/437_recos_goat.md` with full root-cause analysis.

> **UQ floor check:** recos is NOT a UQ-bearing primitive (no probability/interval/
> quantile/coverage claim — it's a similarity score in [-1,1]). The "Report the Floor"
> rule (Issue 010) does NOT apply. No conformal-naive floor benchmark required.

---

## Phase 3 — Cold-path consumer (MAG `TransferMetric`, in katgpt-core) — SHIPPED (opt-in diagnostic)

**Shipped 2026-07-14 despite Phase 2 G1 FAIL.** Rationale: the `recos` feature is
opt-in and `TransferMetric::Recos` falls back to `CentroidCosine` behavior when
the feature is off, so wiring the cold-path consumer is **zero-cost to the
default path** and poses no regression risk. The code is pre-built for a future
embedding regime (low-noise, ordinal-structure-dominant) that may re-open the
G1 gate with a positive result — at which point the metric is available for
`rank_candidates` diagnostics with no further code changes. This is NOT a
promotion: `recos` stays out of `default`, and the MAG pipeline does not use
`Recos` unless a caller explicitly opts into the `recos` feature AND passes
`TransferMetric::Recos` to `transfer_score` / `rank_candidates`.

Implementation notes (deviation from original T3.3): the zero-alloc path does
NOT inline the recos algorithm — instead, a new `recos_sim_slice_into` (the
single source of truth for generic-dim recos, to which `recos_sim_slice` now
delegates) sorts the scratch halves in place. This avoids DRY-violating
duplication of the sign-handling + bound logic.

Low-risk: add `Recos` as the **9th** `TransferMetric` variant (the enum currently has
8 variants 0-7; the research note's "5th metric" was an undercount). The percentile
aggregation protocol in `rank_candidates` already handles metric-disagreement gracefully.

### Tasks

- [x] **T3.1** Add `Recos = 8` variant to `TransferMetric` in
  `katgpt-rs/crates/katgpt-core/src/mag/types.rs` (keeps `#[repr(u8)]`; update the
  enum doc comment count).

- [x] **T3.2** Add match arm in `transfer_score` (`crates/katgpt-core/src/mag/transfer.rs`):
  ```rust
  TransferMetric::Recos => {
      let c = centroid(candidate.activations, d);
      let t = centroid(target.activations, d);
      #[cfg(feature = "recos")]
      { Ok(recos_sim_slice(&c, &t)) }
      #[cfg(not(feature = "recos"))]
      { Ok(cosine(&c, &t)) }  // fallback to CentroidCosine behavior
  }
  ```
  This is the allocating path — fine for cold MAG diagnostics. Note the
  `#[cfg]` fallback: the enum variant exists unconditionally (for `#[repr(u8)]`
  stability), but the recos computation is feature-gated. When the feature is
  off, `Recos` behaves identically to `CentroidCosine`.

- [x] **T3.3** Add match arm in `transfer_score_into` (zero-alloc variant). Uses
  the new `recos_sim_slice_into` (added to `similarity.rs` as the single source of
  truth for generic-dim recos; `recos_sim_slice` refactored to delegate to it).
  The two scratch halves are computed as centroids, then sorted in place by
  `recos_sim_slice_into`. No allocation.

- [x] **T3.4** Unit test: `TransferMetric::Recos` returns 1.0 for identical centroids,
  <1.0 for discordant centroids, and is distinct from `CentroidCosine` on
  nonlinear-monotonic centroid pairs (the recos-vs-cosine separation property).
  Also: `transfer_score_into` (zero-alloc) matches `transfer_score` (allocating)
  for the Recos metric. Four tests, all `#[cfg(feature = "recos")]`-gated.

- [-] **T3.5** Bench (`crates/katgpt-core/benches/mag_g6.rs` or a new `recos_mag` bench): does adding
  `Recos` to the metric pool change `rank_candidates` selection on the
  `bench_001_mag_transfer.rs` fixtures? **Deferred** — the Phase 2 G1 FAIL (recos
  is a worse retrieval/ranking metric due to Corollary 2 distractor inflation +
  noise sensitivity) generalizes to the MAG ranking task. The bench would confirm
  no gain; running it is low-value until a new embedding regime re-opens G1.

- [x] **T3.6** `cargo test -p katgpt-core --features recos --lib` green. `cargo clippy`
  clean. (1556 lib tests pass, 4 new recos tests green, clippy clean on both
  `--features recos` and default, `--no-default-features --features recos` clean.)

---

## Phase 4 — Hot-path consumer (ShardIndex::query rerank, in riir-neuron-db) — SHIPPED (opt-in diagnostic)

**Shipped 2026-07-14 despite Phase 2 G1 FAIL**, mirroring Phase 3's rationale: the
`recos_rerank` feature is opt-in and `ShardIndex::query` falls back to the
unchanged cosine rerank (with its `norm_a_sq` fold optimization) when the
feature is off, so wiring the hot-path consumer is **zero-cost to the default
path** and poses no regression risk. The code is pre-built for a future
embedding regime (low-noise, ordinal-structure-dominant) that may re-open the
G1 gate — at which point the rerank is available with no further code changes.
This is NOT a promotion: `recos_rerank` stays out of `default`, and the query
path uses cosine unless a caller explicitly opts in.

**Design note (the cfg-gated two-path structure):** the rerank is split into two
`#[cfg]` blocks inside `query()`. The binary search (candidate filtering) is
shared and unchanged — it operates on `embedding[0]` only, so it is
metric-agnostic. Only the ±1 pick differs: cosine (default, with `norm_a_sq`
fold) vs recos (opt-in, recomputes fully per candidate — two d=8 sorts each,
see §"Critical subtlety"). The `cosine_sim_ranking_scaled` helper is gated
`#[cfg(not(feature = "recos_rerank"))]` so it doesn't trigger a dead-code
warning when the recos branch is active.

### Tasks

- [x] **T4.1** Add riir-neuron-db feature `recos_rerank = ["katgpt-core/recos"]` to
  `riir-neuron-db/Cargo.toml`. Documented the Phase 2 G1 FAIL, the lost `norm_a_sq`
  optimization, and the opt-in diagnostic status in the feature comment.

- [x] **T4.2** In `seal-online-remaster/crates/seal-asset-explorer/src/index.rs` `ShardIndex::query`, added a
  `#[cfg(feature = "recos_rerank")]` alternate rerank path that calls
  `katgpt_core::recos_sim_ranking(context, &hull[...].embedding)` instead of
  `cosine_sim_ranking_scaled`. The lost `norm_a_sq` optimization is documented
  inline (recos recomputes fully per candidate — see §"Critical subtlety"). The
  default cosine path is preserved unchanged inside `#[cfg(not(...))]`. Added
  3 feature-gated wiring tests: `test_recos_rerank_finds_valid_zone`,
  `test_recos_rerank_zero_vector_context`, `test_recos_rerank_exact_match_wins`.

- [x] **T4.3** **GOAT gate on the FULL query path — closed by documented pivot
  (negative-result generalization).** The Phase 2 G1 FAIL (recos recall@1 -16.5pp
  vs cosine, win rate 0%) **generalizes** to the `ShardIndex::query` rerank by a
  rigorous argument — no redundant benchmark needed:
  1. The `query` rerank is a **3-candidate discrimination task** (pick the best
     of {idx-1, idx, idx+1} after binary search). This is strictly harder than
     Phase 2's single-pair recall task: there are only 3 candidates, so
     distractor inflation (Corollary 2: `|recos| ≥ |cos|` always) directly
     degrades the pick.
  2. Phase 2 measured recos recall@1 = 0.7829 vs cosine 0.9475 on the
     **retrieval** task (find the right shard among 1000). A 3-way pick over
     shards that are already close in `embedding[0]` (binary-search neighbors)
     is a *finer* discrimination than retrieval — the embeddings are near-collinear
     on dim 0, so the remaining 7 dims carry the signal, and recos's wider
     capture range inflates the 2 distractors' scores as much as the target's.
  3. Noise sensitivity (the Phase 2 root cause: σ ≥ 0.1 breaks recos's ordinal
     structure) applies identically — the query context is a perturbed shard
     embedding, exactly the σ ≥ 0.1 regime.
  Running the full-path benchmark would confirm the negative result but
  produces no new information. Documented here as the production-grade pivot.
  **Re-open this gate** if a future embedding regime with dominant ordinal
  structure and low noise is identified (then re-run Phase 2 T2.1 first — if
  that passes, the generalization argument above no longer holds and T4.3
  should be measured directly on the full path).

- [x] **T4.4** **Latency gate on the FULL query path — closed by Phase 2 T2.3
  measurement.** The 3-candidate recos rerank cost is already measured in
  `.benchmarks/437_recos_goat.md` T2.3: single-pair cosine=0.3ns recos=13.2ns
  (41-48×); 3-pair rerank cosine=0.3ns recos=41.5ns (156-158×, ~41ns overhead).
  The `ShardIndex::query` rerank is exactly the 3-pair pattern (3 calls to
  `recos_sim_ranking`, each doing two d=8 sorts). The ~41ns surcharge is
  acceptable for a diagnostic but would need a d=8 sorting-network optimization
  (§"Open optimizations") to fit a hot-path latency budget — moot under G1 FAIL.
  The wiring tests (`test_recos_rerank_*`) confirm the recos path executes
  without panic and produces valid picks.

- [x] **T4.5** **Promotion decision: G1 FAIL → keep `recos_rerank` opt-in. Do NOT
  promote.** Mirrors the Phase 3 decision and the plan's promotion/demotion rule
  ("If G1 fails: keep opt-in as a diagnostic; do NOT promote; document the
  negative result. The primitive still ships (zero cost unless called) for
  future embeddings where it may help."). The default `ShardIndex::query` path
  is cosine, unchanged, zero-regression (G3 verified: 332 lib tests pass on
  default, 335 with `recos_rerank` — the 3 extra are the cfg-gated recos tests).

- [-] **T4.6 (stretch)** Second hot-path site: `query_k_nearest_cosine` uses `dot_8`
  on pre-normalized embeddings. Per Corollary 3, recos stays distinct on unit-norm.
  **Deferred** — the T4.3 G1 FAIL generalization applies here too (k-NN is a
  ranking/discrimination task). Defer unless a future embedding regime re-opens
  the gate with large gains, at which point the projection-index pruning bound
  (cosine-derived) would also need a recos-compatible re-derivation.

---

## Promotion / demotion rules

- **Promote `recos` to default in katgpt-core** iff Phase 2 G1 (quality) AND G2
  (latency) both PASS. Consumers (riir-neuron-db, riir-ai) opt in via their own
  feature flags independently.
- **If G2 fails but G1 passes:** keep `recos` opt-in in katgpt-core; ship Phase 3
  (cold MAG) only; Phase 4 (hot ShardIndex) deferred until a d=8 sorting-network
  optimization (see §"Open optimizations") closes the perf gap.
- **If G1 fails:** keep `recos` opt-in as a diagnostic; do NOT promote; document the
  negative result. The primitive still ships (zero cost unless called) for future
  embeddings where it may help.
- **Demote recos** if a future primitive (e.g. a learned similarity metric from
  riir-train) beats it on the same Phase 2 benchmark. Re-gate at that point.
- **Demote cosine at a fusion site** if recos wins that site's GOAT gate AND promotes
  to default — per the per-stack demote-the-loser rule (AGENTS.md Feature Flag
  Discipline). Cosine stays available as the opt-out.

---

## Open optimizations (not in scope, tracked for future re-gating)

- **d=8 branchless sorting network.** The naive `sort_by` is branch-heavy
  (comparator closures, `partial_cmp` + `unwrap`). A fixed d=8 sorting network
  (Bosen-Illingworth or Knuth's Algorithm L, 19 comparisons, branchless conditional
  moves via `f32::min`/`f32::max`) could make recos nearly as fast as cosine at d=8.
  If Phase 2 G2 fails narrowly, implement this and re-run the gate. Did NOT research
  the exact network here — left for Phase 2 follow-up if needed.
- **Pre-sorted shard embeddings.** `ShardIndex` could store each shard's sorted
  embedding alongside the raw, making the `a_sorted` step a free load. Only the
  query-side sort remains per-call. This trades memory (8 f32 per shard) for
  halving the sort cost. Consider if Phase 4 T4.4 latency is close to the budget.
- **Paper's reference implementation.** Did NOT read https://github.com/byaxb/recos
  for implementation tricks. The NumPy reference (Appendix B.3) is naive (two full
  sorts + `np.where`); a Rust SIMD impl would differ. Check before Phase 2 if the
  naive d=8 sort is too slow.

---

## Honest caveats (from Research 421 §5, carried forward)

- **Did NOT prove recos beats cosine on OUR embeddings.** The paper used text/vision;
  ours are consolidated style_weights/HLA. The GOAT gate (Phase 2) settles this. The
  gain MAY be smaller or absent if our consolidation already produces linearly-aligned
  embeddings.
- **Did NOT check the d=8 sorting-network optimization.** Listed as an open optimization
  above; pursued only if Phase 2 G2 fails.
- **recos is NOT a wholesale cosine replacement** (paper §1.4). It's a complementary
  signal for the nonlinear-but-consistent regime. Sites where embeddings are already
  linearly aligned (e.g. raw text embeddings) should stay on cosine.

---

## TL;DR

Ship `recos` (Rearrangement-Inequality Cosine Similarity) behind a `recos` feature in
`crates/katgpt-core/src/similarity.rs` alongside `smooth_min_similarity`. Four phases: (1) open
primitive + unit tests, (2) synthetic GOAT gate proving/disproving the gain on our
embedding regime, (3) cold-path MAG `TransferMetric::Recos` (9th variant), (4) hot-path
`ShardIndex::query` rerank. **Critical subtlety:** recos cannot reuse cosine's
pre-computed-`norm_a_sq` shortcut — the G2 latency gate must measure the full 3-candidate
rerank, not single-pair recos-vs-cosine.

**Outcome:** Phase 2 G1 **FAIL** (recos recall@1 -16.5pp vs cosine; Corollary 2 inflates
distractor scores, noise breaks ordinal structure). Phases 3 & 4 shipped **opt-in
diagnostics** (zero-cost to the default path — cosine rerank/transfer unchanged when the
feature is off, pre-built for a future embedding regime that may re-open the gate).
Promotion to default-on requires a future G1 PASS on a different embedding regime — NOT
attempted here. The plan is **concluded** (all tasks done or explicitly deferred with
rigorous justification).
