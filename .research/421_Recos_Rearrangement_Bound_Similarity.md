# Research 421: recos — Rearrangement-Inequality-Based Cosine Similarity

> **Source:** "Beyond Cosine Similarity" — Xinbo Ai (BUPT), arXiv:2602.05266 (Feb 2026)
> **Date:** 2026-07-14
> **Status:** Active
> **Related Research:** 385 (SoftMatcha2 smooth-min — closest cousin, also a similarity variant that shipped DEFAULT-ON), 144 (Functional Emotions Linear Representations — direction-vector ranking), 296 (Stokes/DEC vocabulary crosswalk — bound-hierarchy framing)
> **Related Plans:** 422 (recos open primitive — to be opened)
> **Classification:** Public

---

## TL;DR

`recos` (Rearrangement-inequality-based Cosine Similarity) is a pure inference-time
similarity metric that normalizes the dot product by the **Rearrangement-Inequality
bound** `u↑·v↕` instead of the Cauchy-Schwarz bound `‖u‖·‖v‖`. It saturates at 1 under
**ordinal concordance** (monotonic relationship) rather than linear dependence, giving
it a strictly wider capture range than cosine. Empirically beats cosine on 71/72 STS
model-dataset pairs (98.6% win rate, p<0.001), with the largest gains on embeddings
that diverge from standard textual-similarity assumptions (CLIP-ViT +0.96, DPR +0.65,
SPECTER +0.49).

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is the **bound hierarchy** itself:

```
|u·v|  ≤  u↑·v↕  ≤  ‖u‖·‖v‖  ≤  (‖u‖² + ‖v‖²)/2
          ↑              ↑              ↑
        recos          cosine         decos
```

Each bound induces a similarity metric with a different saturation condition
(recos = ordinal concordance, cosine = linear dependence, decos = identity).
The tighter the bound, the wider the capture range. `recos` costs O(d log d)
(the sort) vs O(d) for cosine, but for our d=8 HLA/shard embeddings the sort is
~24 comparisons — cheap. The value is **retrieval recall on nonlinear-but-consistent
embeddings**, exactly the regime our trained direction vectors and consolidated
style_weights live in.

---

## 1. Paper Core Findings

### 1.1 The bound hierarchy (Theorem 1)

For any `u, v ∈ ℝᵈ` with `u·v ≠ 0`:

| Bound | Tightness | Saturation condition | Metric |
|-------|-----------|---------------------|--------|
| `u↑·v↕` | Tightest | Ordinal concordance (monotonic) | `recos` |
| `‖u‖·‖v‖` | Middle (Cauchy-Schwarz) | Linear dependence (`v = ku`) | `cos` |
| `(‖u‖²+‖v‖²)/2` | Loosest (AM-QM) | Identity (`u = ±v`) | `decos` |

Where `u↑` = u sorted ascending, `v↕` = v sorted to match u's direction
(`v↑` if `u·v > 0`, else `v↓`). The Rearrangement Inequality guarantees
`u↑·v↑ ≥ u·v` for any permutation, with equality iff u, v are similarly ordered.

**Key corollary (Corollary 2 — metric hierarchy):** `|decos| ≤ |cos| ≤ |recos|`.
recos is *always ≥ cosine in absolute value*. It can't score lower; it can only
score higher (or equal) by recognizing ordinal structure that cosine misses.

**Key corollary (Corollary 3 — norm identity):** For unit-norm vectors,
`decos = cos` (they collapse). But `recos` stays distinct because its denominator
depends on ordinal structure, not norms. **This is the critical property for us:**
our pipeline normalizes embeddings to unit norm before cosine (see
`ShardIndex::normalized_hla` in riir-neuron-db), which means `decos` would be
useless but `recos` still adds signal.

### 1.2 Empirical results (Table 1, 77 model-dataset pairs)

- **Win rate:** 71/72 non-tied comparisons (98.6%), Wilcoxon p<10⁻¹³, effect size r=0.835 (large).
- **Average gain:** +0.29 Spearman ρ points over cosine.
- **Largest gains:** CLIP-ViT (+0.96), DPR (+0.65), SPECTER (+0.49) — models with
  representation spaces that diverge from standard textual similarity.
- **Smallest gains:** Word2Vec, BGE, E5 (+0.02–0.08) — already cosine-aligned.
- **Single loss:** −0.31 (BGE on STS13) — rare, model-specific.

### 1.3 Complexity

- `recos`: O(d log d) — one sort of each vector + O(d) dot products.
- `cos`: O(d).
- For d=8 (our HLA/shard dimension): sort ≈ 24 comparisons vs 8 mults. ~3× slower
  per comparison, but absolute cost is nanoseconds.

### 1.4 What recos is NOT

The paper is explicit: recos is not a wholesale cosine replacement. It's a
**complementary signal** that captures ordinal/monotonic structure. The recommendation
is to use it where embeddings are known to have nonlinear-but-consistent relationships.

---

## 2. Distillation

### 2.1 Why this matters here

Our codebase runs cosine similarity in **five load-bearing latent-space sites**:

| Site | Dim | Path | Cosine function |
|------|-----|------|-----------------|
| Shard retrieval (hot) | 8 | `riir-neuron-db/src/index.rs` `ShardIndex::query` | `cosine_sim_ranking_scaled` (squared cosine, no sqrt) |
| Shard KNN (cold) | 8 | `riir-neuron-db/src/index.rs` `query_k_nearest_cosine` | `cosine_sim` (full cosine) |
| Item retrieval | 8 | `riir-neuron-db/src/item_index.rs` `ItemEmbedIndex::query` | cosine on schema-centroid embeddings |
| MAG transfer scoring | 64 | `riir-neuron-db/src/consolidation/mod.rs` | `CentroidCosine` / `ClassConditionalCosineMalicious/Benign` |
| Multi-token rerank | var | `katgpt-rs/crates/katgpt-attn-match/src/rerank.rs` | `cosine_score_into` |

Every one of these operates on embeddings that are **trained or consolidated**,
not raw text — exactly the regime where the paper shows recos gains are largest
(DPR, SPECTER, CLIP-ViT; +0.49 to +0.96). Our `style_weights[64]` and 8-dim HLA
embeddings are consolidated through Raven/δ-Mem sleep cycles; they are NOT linearly
proportional to query context by construction. This is the paper's sweet spot.

### 2.2 The open primitive

`recos` belongs in `katgpt-rs/crates/katgpt-core/src/similarity.rs` alongside
`smooth_min_similarity` (Research 385). Both are similarity variants; both are
modelless; both are inference-time. The module is already the canonical home.

**Proposed API (d=8 fixed, matching `ShardIndex`'s `[f32; 8]` convention):**

```rust
/// Rearrangement-inequality-based cosine similarity (recos).
///
/// Distilled from Ai (2026), arXiv:2602.05266. Saturates at 1.0 under ordinal
/// concordance (monotonic relationship), a strictly wider capture range than
/// cosine (which requires linear dependence). Always |recos| ≥ |cos| in
/// absolute value (Corollary 2).
///
/// Cost: O(d log d) — one sort per vector. For d=8 this is ~24 comparisons
/// + 8 FMA. The sort is the dominant cost vs cosine's 8 FMA.
///
/// Use when embeddings are known to have nonlinear-but-consistent relationships
/// (consolidated style_weights, trained direction vectors, schema-centroid
/// item embeddings). Use cosine when embeddings are already linearly aligned
/// with the query (raw text embeddings from a sentence transformer).
#[inline]
pub fn recos_sim(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let dot = dot_8(a, b);
    // Rearrangement bound: sort both, dot the sorted.
    // For dot > 0: u↑·v↑. For dot < 0: u↑·v↓ (flip b's sort direction).
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

**Ranking-only variant** (mirrors `cosine_sim_ranking` — avoids sqrt, preserves order):

```rust
/// recos ranking score — preserves ordering, avoids division.
/// Returns (dot / bound)² like cosine_sim_ranking returns (dot/denom)².
/// Use for top-k selection where only the ORDER matters.
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

**Generic variant** (for MAG's d=64 `style_weights`):

```rust
pub fn recos_sim_slice(a: &[f32], b: &[f32]) -> f32 { ... }
```

### 2.3 Fusion opportunities

#### Fusion A — recos rerank in `ShardIndex::query` (STRONGEST)

`ShardIndex::query` does binary search on `embedding[0]` (using the Cauchy-Schwarz
upper-bound `|c₀−q₀|` for pruning), then checks ±1 neighbors ranked by
`cosine_sim_ranking_scaled`. The binary-search PRUNING step can't use recos
(recos's bound doesn't have a clean 1D sort key), but the ±1 RERANK step can.
Replace `cosine_sim_ranking_scaled` with `recos_sim_ranking` in the rerank.
This is a one-line change with a feature flag; the binary search still uses
the cosine bound for candidate filtering, but the final 3-way pick uses recos.

**Why this is the strongest fusion:** the ±1 rerank is where the metric's capture
range matters most — we're choosing among 3 candidates that all passed the bound.
recos's wider capture range means we're less likely to dismiss a truly-similar
shard because its embedding isn't linearly proportional to the query.

#### Fusion B — recos as 5th MAG transfer metric

`ConsolidationPipeline::sleep_transfer` already aggregates 4 metrics
(`CentroidCosine`, `Euclidean`, `Correlation`, `ClassConditionalCosine{Malicious,Benign}`)
via percentile protocol. Adding `Recos` as a 5th `TransferMetric` variant is trivial
(it's the same shape — dot product + normalization) and gives the percentile
aggregation a signal that captures ordinal structure the other 4 miss. This is a
clean Gain with zero risk (the percentile protocol already handles
metric-disagreement gracefully).

#### Fusion C — recos as HLA direction-vector projection quality check

Research 144 (Functional Emotions) and the Lean proof on sigmoid ranking preservation
(`RiirAiProof`) establish that direction vectors must preserve the intended ranking
of affect dimensions. recos is literally a ranking-preservation metric: it saturates
iff the direction vector and the projection target are ordinally concordant. We could
use `recos_sim(direction, observed)` as a **projection quality gate** — if recos is
high but cosine is low, the direction is "shape-correct but scale-wrong", which is
a different failure mode than "wrong direction entirely". This connects to the
freeze/thaw integrity envelope (a frozen direction vector that has decayed in scale
but preserved ordinal structure should be healable, not replaced).

#### Fusion D — recos in `funcattn_scorer` for item retrieval

`FuncAttnItemScorer` (riir-neuron-db, Plan 362 Phase 4) denoises item embeddings
before dot-product scoring. recos could be an alternative scoring function that's
robust to the denoiser's nonlinear transforms. Lower priority — the FUNCATTN
denoiser is already designed to make cosine work well.

### 2.4 What does NOT transfer

- **The STS benchmark methodology** — we don't have human-judged similarity labels
  for game embeddings. Our GOAT gate must use synthetic recall benchmarks (like
  the `smooth_min_similarity` PoC did) or downstream task metrics (MAG transfer
  accuracy, shard retrieval hit rate).
- **The 11-model evaluation** — our "models" are the HLA kernel, the style_weights
  consolidator, and the item embedder. We benchmark on those, not on BERT/CLIP.
- **The statistical analysis apparatus** (Wilcoxon, mixed-effects) — overkill for
  a d=8 retrieval benchmark. We use recall@k and cosine-alignment-to-target, same
  as the TEMP and MAG gates.

---

## 3. Verdict

### Tier: **GOAT**

**One-line reasoning:** Provable quality gain (paper's 98.6% win rate, +0.29 avg,
up to +0.96 for nonlinear embeddings) on a modelless inference primitive
(similarity function), with clear fusion targets in 5 load-bearing latent-space
sites. NOT a new capability class (it's a better similarity metric, same operation),
so it doesn't clear the Super-GOAT bar (new behavior + new pillar).

**Comparison to closest cousin:** `smooth_min_similarity` (Research 385) is also a
similarity variant that shipped as DEFAULT-ON after a +50.5pp recall gain on
multi-token retrieval. recos has smaller headline gains (+0.29 STS avg) but broader
applicability (every cosine site, not just multi-token). Both are GOAT-tier
similarity-stack primitives; both ship behind a feature flag and earn default-on
via the GOAT gate.

### MOAT gate: **katgpt-rs in-scope (strengthens engine, neutral on private moats)**

- **katgpt-rs (this repo):** ✅ Paper-derived fundamental similarity primitive.
  Belongs in `similarity.rs` alongside `smooth_min_similarity`. Per-stack
  promote/demote tracking applies (similarity stack: `smooth_min_similarity`,
  `cosine_score`, `recos_sim`). The GOAT gate decides default-on vs opt-in.
- **riir-neuron-db (consumer):** Neutral. `ShardIndex::query` and MAG transfer
  are consumers via katgpt-core dep. No new shard IP.
- **riir-ai (consumer):** Neutral. HLA projections and latent functor ops are
  consumers. No new runtime IP.
- **riir-chain / riir-train:** Out of scope.

### Novelty gate Q1–Q4

| Q | Answer | Evidence |
|---|--------|---------|
| Q1 No prior art? | **YES** | Grep for `recos\|rearrangement\|ordinal.*(concord\|similar\|monoton)` across all repos + .md files = ZERO relevant hits. `similarity.rs` ships `smooth_min_similarity` + `edit_penalty` only. No rearrangement-based variant exists. |
| Q2 New class of behavior? | **NO** | It's a better similarity metric (wider capture range), not a new capability. Same operation (similarity), different bound. |
| Q3 Product selling point? | **NO** | "Our NPCs retrieve shards via ordinal concordance" is a stretch — the gain is retrieval quality, not a new feature. |
| Q4 Force multiplier? | **PARTIAL** | Broad reach (5 cosine sites) but doesn't connect to ≥2 pillars or enable a new pillar. |

**Q2 NO → not Super-GOAT.** Proceed to GOAT plan.

### Perf caveat (must be measured in GOAT gate)

The paper acknowledges O(d log d) overhead. For our d=8 hot path (`ShardIndex::query`),
this is ~3× slower per comparison (24 sort comparisons vs 8 FMA). The sort is
branch-heavy (branchless sort networks exist for d=8 — see the `sort_by` →
sorting-network optimization path). The GOAT gate MUST measure:

1. **G1 (quality):** recall@k gain on synthetic shard retrieval (correct shard
   ranked in top-k more often with recos rerank than cosine rerank).
2. **G2 (latency):** `recos_sim` vs `cosine_sim` on d=8 — must be < 100ns (the
   `ShardIndex::query` budget is ~10ns for the 3-way rerank; recos can be 30ns
   and still fit). Consider a branchless d=8 sorting network.
3. **G3 (no-regression):** `--all-features` clean, existing `ShardIndex::query`
   tests pass unchanged when feature is off.
4. **G4 (alloc-free):** sort in-place on stack arrays, zero heap allocs.

If G2 fails (recos too slow for hot path), **downgrade Fusion A to cold-path only**
(MAG transfer, KNN heal) and keep the hot path on cosine. The cold-path fusions
(B, C) are valuable on their own.

---

## 4. Plan sketch (Plan 422)

**Target:** `katgpt-rs/crates/katgpt-core/src/similarity.rs` (add `recos_sim`,
`recos_sim_ranking`, `recos_sim_slice`) + feature flag `recos` (opt-in).

**Phases:**

- **Phase 1 — Open primitive.** Add the three functions behind `#[cfg(feature = "recos")]`.
  Unit tests verify: (a) `|recos| ≥ |cos|` always (Corollary 2), (b) recos = 1.0 for
  ordinally-concordant vectors, (c) recos < 1.0 for discordant vectors, (d) recos
  distinct from cos for unit-norm vectors (Corollary 3).
- **Phase 2 — GOAT gate (synthetic).** Benchmark on synthetic d=8 retrieval:
  generate 1000 shards with nonlinear-but-monotonic embeddings, query with
  perturbed versions, measure recall@1 and recall@5 for cosine vs recos rerank.
  Target: recos recall ≥ cosine recall (paper's 98.6% win rate suggests this holds).
- **Phase 3 — Cold-path consumer (MAG transfer).** Add `Recos` as a 5th
  `TransferMetric` variant in katgpt-core's MAG module. Wire into
  `ConsolidationPipeline::sleep_transfer` behind the same feature flag. GOAT gate:
  does adding recos to the metric pool change the selected subset on any test fixture?
  (If yes and the change is toward the target, that's the gain.)
- **Phase 4 — Hot-path consumer (ShardIndex rerank, CONDITIONAL on Phase 2 G2 pass).**
  If the d=8 latency gate passes, replace `cosine_sim_ranking_scaled` with
  `recos_sim_ranking` in the ±1 rerank step of `ShardIndex::query`. Feature-flagged,
  off by default. GOAT gate: recall@1 on synthetic shard retrieval with the FULL
  query path (binary search + recos rerank) vs (binary search + cosine rerank).

**Promotion rule:** If Phase 2 G1 passes AND Phase 2 G2 passes → promote `recos`
to default-on in katgpt-core. Consumers (riir-neuron-db, riir-ai) opt in via their
own feature flags. If Phase 2 G2 fails → keep `recos` opt-in, document the
hot-path-too-slow caveat, ship Phase 3 (cold-path MAG) only.

**Demotion rule:** If a future d=8 sorting-network optimization makes recos as fast
as cosine, re-run the gate and promote. If a future primitive (e.g., a learned
similarity metric from riir-train) beats recos on the same benchmark, demote recos.

---

## 5. What I did NOT do (honesty log)

- **Did not run the GOAT gate.** This is a research note + plan sketch, not an
  implementation. The gate is Phase 2 of Plan 422.
- **Did not prove recos beats cosine on OUR embeddings.** The paper proves it on
  STS benchmarks with text/vision embeddings. Our embeddings (HLA, style_weights,
  item schema-centroids) are a different distribution. The gain MAY be smaller or
  absent if our consolidation already produces linearly-aligned embeddings. The
  GOAT gate settles this.
- **Did not check the d=8 sorting-network optimization.** The naive `sort_by` is
  branch-heavy; a branchless d=8 sorting network (e.g., Bosen-Illingworth) could
  make recos nearly as fast as cosine. This is a Phase 2 optimization, not
  researched here.
- **Did not read the paper's GitHub repo** (https://github.com/byaxb/recos) for
  implementation tricks. The NumPy reference implementation in Appendix B.3 is
  naive (two full sorts + np.where). A Rust SIMD implementation would differ.

---

## TL;DR

`recos` is a modelless inference-time similarity metric with a provably wider
capture range than cosine (ordinal concordance vs linear dependence). It belongs
in `katgpt-rs/crates/katgpt-core/src/similarity.rs` alongside `smooth_min_similarity`.
**Verdict: GOAT** — provable quality gain, modelless, but not a new capability class
(not Super-GOAT). Closest cousin `smooth_min_similarity` shipped DEFAULT-ON; recos
follows the same path (feature flag → GOAT gate → promote if wins). Strongest
fusion: recos rerank in `ShardIndex::query` (conditional on d=8 latency gate) and
recos as 5th MAG transfer metric (cold-path, low-risk). Plan 422 to follow.
