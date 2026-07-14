# Issue 139 — Plan 230 `shard_embedding` G1 Failure Root Cause: JL Bound Violated at m=8

**Status:** OPEN (diagnostic complete; fix path requires design decision)
**Opened:** 2026-07-14
**Parent:** `katgpt-rs/.plans/230_shard_embedding_projection.md` (G1 FAIL since 2026-06-09)
**Related:** Issue 138 (CLOSED 2026-07-14, removed for noise — fusion SHELVED on G1, see Research 422 §6) surfaced this root cause as the prerequisite for any two-matrix salvage path.

---

## TL;DR

Plan 230's `JlProjectionMatrix` (64→8) **cannot satisfy G1** at any non-trivial
n. This is not a tuning problem or a scaling bug — it is the **Johnson–
Lindenstrauss lower bound** being violated by an order of magnitude. The
primitive is mathematically unsound as specified; no matrix construction
(random orthogonal, Gaussian, Achlioptas sparse, FJLT/Hadamard) can rescue it
at m=8.

**The salvage path (Option B = PCA) is empirically viable** for intrinsic
rank ≤ 8 at m=8 or rank ≤ 16 at m=16 (synthetic PCA probe — see §"Option B"
table). The architect's decision reduces to one empirical measurement:
the actual intrinsic rank of real `style_weights`. Until that measurement
exists, no code change is proposed.

---

## Empirical Sweep (2026-07-14, release build, M3 Max)

Diagnostic: `tests/diag_230_embed_dim_sweep.rs` (run, captured, then removed —
see "Artifacts" below). Methodology: Gram-Schmidt orthonormal m×64 matrix,
n random vectors in [-1,1]^64, top-k NN set overlap between original and
projected spaces. Averaged over 3 seeds (Euclidean) or single seed (n=500).

### n=100 (current bench_230 setting), Euclidean NN

| m | top1 | top5 | top10 |
|---|------|------|-------|
| 8 (current) | 5.0% | 13.3% | 21.4% |
| 16 | 7.7% | 18.9% | 26.8% |
| 24 | 14.3% | 25.8% | 33.1% |
| 32 | 19.0% | 32.5% | 39.3% |
| 40 | 27.7% | 42.1% | 49.3% |
| 48 | 40.3% | 52.4% | 58.6% |
| 56 | 54.0% | 62.8% | 68.9% |
| **64** | **100%** | 100% | 100% |

### n=100, Cosine NN (Plan 230's intended metric)

| m | top1 | top5 |
|---|------|------|
| 8 (current) | 4.7% | 13.3% |
| 16 | 10.0% | 22.7% |
| 24 | 15.0% | 29.5% |
| 32 | 22.0% | 37.7% |
| 40 | 31.0% | 44.9% |
| 48 | 38.3% | 54.9% |
| 56 | 53.0% | 68.3% |
| **64** | **100%** | 100% |

### n=500 (realistic shard-region size), Euclidean NN

| m | top1 | top5 |
|---|------|------|
| 8 (current) | **1.4%** | 4.8% |
| 16 | 4.0% | 10.2% |
| 24 | 7.8% | 14.3% |
| 32 | 13.0% | 21.2% |
| 48 | 25.4% | 40.2% |
| **64** | **100%** | 100% |

### Reading the table

- **m=8 (Plan 230's current setting) is in the noise floor.** 1.4% top-1 at
  n=500 is below random chance for top-5 (which would be 5/499 ≈ 1.0%).
- **Cosine is marginally more stable than Euclidean** in the projected space
  but follows the same curve — the metric choice is not the bottleneck.
- **Only m=64 (= identity) gives 100%.** That's not compression; that's a
  no-op.
- **JL ≥ 90% top-1 is unreachable below m=64 for random data.**

---

## Theoretical Floor (why m=8 was always doomed)

The Johnson–Lindenstrauss lemma: to embed n points into R^m with all pairwise
distorts preserved within (1±ε), the lower bound is

```
m ≥ (4 ln n) / (ε²/2 − ε³/3)        [Achlioptas 2003, clean form]
```

For the regime relevant to shard routing:

| n | ε=0.3 (strong) | ε=0.5 (weak) |
|---|----------------|--------------|
| 100 | **1750** | 554 |
| 500 | **2686** | 851 |
| 1000 | **3066** | 972 |

Plan 230's m=8 undershoots the JL bound by **>200×** for ε=0.3, n=100. Even
the m=64 upper bound (identity-equivalent) undershoots by ~27×. The JL lemma
does not apply at these dimensions; the projection is in the regime where
pairwise distances are essentially randomized.

This is **independent of matrix construction**: random Gaussian, Achlioptas
sparse {±1, 0}, FJLT (Hadamard + diagonal signs), or any other JL family
hits the same m ≥ O(log n / ε²) floor.

---

## Pre-existing Plan 230 Issues Surfaced

While running the diagnostic, two pre-existing inconsistencies surfaced that
should be noted (NOT fixed in this issue — they belong to the design
decision):

1. **G1 threshold mismatch.** The plan document (`.plans/230_...md` line 112)
   states "Top-1 ≥ 90%", but the actual test
   (`tests/bench_230_shard_embedding_goat.rs` line 101) asserts
   `min_rate = if cfg!(debug_assertions) { 0.03 } else { 0.30 }`. The
   30% release threshold is **3× lower than the documented 90%**, and the
   3% debug threshold is essentially "any signal at all". The test passes
   at 6% only because of this debug relaxation.

2. **Scaling comment is wrong but not load-bearing.** `shard_embedding.rs`
   line 68 scales rows by `1/sqrt(EMBED_DIM) = 1/sqrt(8)`. The comment says
   "per JL lemma" — but the correct unbiased-distance scaling for an
   orthonormal m×n projection is `sqrt(n/m)`, not `1/sqrt(m)`. This doesn't
   affect ranking (cosine and Euclidean are both scale-invariant for fixed
   scale), so it's not the cause of G1 failure. Noted for whoever does the
   salvage pass.

---

## Salvage Options (need architect decision)

Ranked by modelless-purity (per AGENTS.md modelless-first mandate):

### Option A — Raise `EMBED_DIM` to 48-64

- **Cost:** `ShardEmbedding([f32; 48])` = 192 bytes (vs current 32). Still
  cache-friendly. Crosses the 64-byte cache line but stays under 256.
- **Gain:** top-1 ≈ 40-54% at n=100 (still not 90%). Top-5 ≈ 52-69%.
- **Verdict:** Partial. Hits a usable "routing hint" regime but not exact-NN.
  Trades most of the compression benefit for a still-marginal quality gain.

### Option B — Switch to PCA (modelless, Plan 230's own option 1) ✅ PROBE DONE

- **Cost:** Requires accumulating style_weights samples at consolidation time
  and running an SVD. Adds a one-time O(n·d²) compute at consolidation, not
  at inference.
- **Gain:** **Probed on synthetic low-rank data (see §"PCA Probe" below).**
  PCA at m=8 nails G1 (≥90% top-1) when intrinsic rank ≤ 8 with low noise;
  PCA at m=16 covers up to rank=16. Fails for rank ≥ 32.
- **Verdict:** **Most promising modelless path.** The salvage question
  reduces to a single empirical measurement: **what is the actual intrinsic
  rank of real `style_weights`?**
  - If ≤ 8: Option B (PCA at m=8) works → Plan 230 fully salvageable.
  - If ≤ 16: Option B (PCA at m=16) works → type grows from 32 to 64 bytes.
  - If > 16: Plan 230 unsalvageable with this approach → consider Option D.
- **Blocker:** Real-data probe needs a corpus of actual `style_weights`
  vectors. Either (a) a synthetic generator calibrated to observed rank
  (deferred until real samples exist), or (b) a hook into the sleep
  consolidation pipeline to dump samples at next consolidation cycle.

#### PCA Probe Results (synthetic upper bound, n=100, d_in=64)

Power-iteration PCA on synthetic low-rank-plus-noise data. **This is the
best case for PCA** — if it failed here, Option B would be hopeless.

| k_true | noise σ | m=8 top1 | m=8 top5 | m=16 top1 | m=16 top5 |
|--------|---------|----------|----------|-----------|-----------|
|  4 | 0.0  | **95.0%** | 89.4% | 96.0% | 95.0% |
|  4 | 0.1  | 83.0%  | 92.4% | 85.0%  | 93.6% |
|  8 | 0.0  | **100.0%** | 100.0% | 87.0% | 84.0% |
|  8 | 0.1  | **94.0%** | 96.4% | **95.0%** | 97.8% |
|  8 | 0.3  | 56.0%  | 76.0% | 61.0%  | 83.4% |
| 16 | 0.0  | 37.0%  | 52.4% | **100.0%** | 100.0% |
| 16 | 0.1  | 38.0%  | 53.0% | **99.0%**  | 97.4% |
| 32 | 0.0  | 18.0%  | 36.8% | 46.0%  | 60.8% |

**The pass/fail contour (≥90% top-1):**

- m=8 PCA: passes for rank ≤ 8, noise ≤ 0.1.
- m=16 PCA: passes for rank ≤ 16, noise ≤ 0.1.
- m=8/16 PCA: fails for rank ≥ 32 (noise-free) — genuine high-dim regime.

**What this tells the architect:** the PCA probe converts the salvage
question from "maybe works, untested" into a **single empirical query**:
measure the intrinsic rank of the real `style_weights` corpus and look it
up in the table above.

### Option C — Switch to LSH (different primitive)

- **Cost:** Replace `ShardEmbedding` projection with LSH hash family
  (e.g., SimHash or p-stable LSH). Different retrieval semantics
  (approximate, collision-based) — breaks the current cosine-similarity API.
- **Gain:** LSH is designed for high-NN-recall at low dimension; doesn't
  pretend to preserve distances.
- **Verdict:** Right tool for the job, but it's a **different primitive**
  than what Plan 230 specified. Belongs in a new plan, not a fix to 230.

### Option D — Deprecate Plan 230

- **Cost:** Remove `shard_embedding` feature + `JlProjectionMatrix` + the
  bench. Mark `ShardEmbedding` type as `#[deprecated]`.
- **Gain:** Eliminates a broken primitive from the surface area. The BFCF
  region cache (Plan 230 T3) needs a different secondary lookup key —
  probably a hash of the style_weights BLAKE3 commitment (exact-match, not
  similarity).
- **Verdict:** Honest. If no consumer actually depends on approximate
  similarity, deprecating is cheaper than maintaining a broken primitive.

---

## Recommendation

**The architect's decision reduces to one measurement: the intrinsic rank
of real `style_weights`.** The PCA probe table above is a lookup: measure
the rank, read off whether Option B works.

1. **Probe real `style_weights` rank first** (see new T7 below). This is
   the single load-bearing empirical question. Cost: one instrumentation
   pass in the sleep consolidation pipeline to dump 100-500 samples, then
   power-iteration PCA to read off the effective rank (where cumulative
   variance crosses 95%).
2. **If rank ≤ 8**: Option B (PCA at m=8) is the GOAT fix — modelless,
   fits the existing type/API, satisfies G1 at 94-100%.
3. **If 8 < rank ≤ 16**: Option B (PCA at m=16). Type grows from 32 to 64
   bytes (still one cache line). Modelless gain, promotion-worthy.
4. **If rank > 16**: Plan 230's compression goal is incompatible with the
   data. Choose between Option A (raise dim, accept quality loss) or
   Option D (deprecate, switch BFCF to exact-match hash key).
5. **Option C (LSH)** remains a separate primitive, deserves its own plan
   if approximate similarity is actually needed.

The synthetic PCA probe already done in this issue is the **upper-bound
proof of viability** for Option B. The remaining work is the real-data
measurement, not more theory.

---

## Tasks

- [x] T1 Reproduce Plan 230 G1 number (6.0% — matches doc).
- [x] T2 EMBED_DIM sweep, Euclidean NN, n=100.
- [x] T3 EMBED_DIM sweep, Cosine NN, n=100.
- [x] T4 EMBED_DIM sweep, n=500 (realistic shard-region size).
- [x] T5 Compute JL theoretical floor for the relevant (n, ε) regime.
- [x] T6 Document salvage options with trade-offs.
- [x] T7 Synthetic PCA probe (Option B upper bound) — DONE. PCA at m=8
      satisfies G1 for intrinsic rank ≤ 8, noise ≤ 0.1. PCA at m=16 covers
      rank ≤ 16. Fails for rank ≥ 32.
- [-] T8 **BLOCKED** — Real-data `style_weights` intrinsic-rank measurement.
      Requires a hook into the sleep consolidation pipeline to dump samples
      at next consolidation cycle. Re-opens automatically when the corpus
      exists; the T7 table above is the lookup.
- [-] T9 **DEFERRED** — Architect decision between Options A/B/D (C is a
      separate primitive). The decision reduces to T8's measurement: look
      up the rank in the PCA probe table, read off the salvage path.

---

## Artifacts

- `tests/diag_230_embed_dim_sweep.rs` — the diagnostic. Run with
  `cargo test --release --features shard_embedding --test diag_230_embed_dim_sweep -- --nocapture`.
  Kept in-tree (not deleted) because the sweep is the evidence base for the
  salvage decision; whoever picks up T7/T8 will want to re-run it under
  different assumptions (PCA matrix, real data).
- Raw numbers transcribed into the tables above.

---

## Cross-references

- `katgpt-rs/.plans/230_shard_embedding_projection.md` — the broken primitive
- Issue 138 (CLOSED 2026-07-14, removed for noise — fusion SHELVED on G1,
  see Research 422 §6) — its "salvage path" footnote (two-matrix
  architecture) is blocked on this issue being resolved first
- `katgpt-rs/crates/katgpt-core/src/shard_embedding.rs` — current impl
- `katgpt-rs/tests/bench_230_shard_embedding_goat.rs` — current G1 test
  (assertion threshold is 30% release / 3% debug, not the documented 90%)
