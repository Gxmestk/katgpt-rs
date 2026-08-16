# Benchmark 658 — Issue 657: clustered LM head, admissible bound + D² init

**Date:** 2026-08-16
**Device:** Apple M3 Max
**Harness:** `tests/bench_657_clustered_lm_head_bound.rs`
**Config:** vocab=32768, n_embd=512, cluster_size=128 (≈256 clusters), 200 probes
**Supersedes:** `.benchmarks/657_clustered_lm_head_goat.md` (same primitive, two fixes applied)

**Verdict:** **G2b now PASSES — recall 0.675 → 1.0000.** But **Issue 657's own
diagnosis was wrong**: the scoring objective was not the defect. A degenerate
k-means initialization was. The bound Issue 657 proposed is a *worse* ranker
than the thing it replaced; its value is elsewhere — it makes an **exact**
(recall-1.0-by-construction) stop rule possible.

**Promotion: still NOT default-on.** G3 inverts to a **0.08× loss** on the
unstructured control, and no real checkpoint has been measured. See
§Promotion.

---

## Why there were two fixes

Plan 574 failed G2b at 0.675 recall. Issue 657 attributed that to the **scoring
objective**: stage 1 ranks by `⟨h, centroid_c⟩`, which is the cluster's *mean*
logit, when the question is which cluster holds the *max*. That produced **fix
A** — the Cauchy–Schwarz radius bound.

Implementing it surfaced an unrelated second defect the diagnosis had not
considered. K-means seeded its centres at token IDs `0, stride, 2·stride, …`
with `stride = vocab / k`. Benchmark 657's fixture assigns token `t` to planted
group `t % n_groups` with `n_groups == k`, so every seeded centre came from
group `(c·stride) mod k` — which for `stride = vocab/k` collapses to **two
distinct groups**. Lloyd cannot recover 256 planted groups from centres drawn
out of two. That is **fix B**: deterministic k-means++ (D²) seeding.

Two candidate causes for one failed gate is exactly the setup that yields a
confident wrong attribution, so this bench measures the full **2×2** rather
than applying both and claiming the result.

## Results — argmax recall at matched active budget

Recall = fraction of probes where the pruned head returns the token the full
head would have. Compared at equal **active fraction**, not equal `topk`
(Benchmark 657 §Methodology). `topk` in parentheses. Deterministic — bit-identical
across 3 runs.

### Structured, groups == clusters (the verdict regime)

| arm | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| round-robin + mean | 0.0150 (5) | 0.0550 (12) | 0.1200 (25) | 0.4900 (64) |
| strided init + mean — *Plan 574's measurement* | 0.0700 (4) | 0.1600 (15) | 0.2300 (88) | 0.6750 (102) |
| strided init + BOUND — *fix A alone* | 0.0000 (0) | 0.0350 (1) | 0.0700 (2) | 0.4100 (5) |
| **D² init + mean — *fix B alone*** | **1.0000 (5)** | **1.0000 (13)** | **1.0000 (26)** | **1.0000 (64)** |
| D² init + BOUND — *both* | 0.0000 (2) | 0.9950 (7) | 1.0000 (20) | 1.0000 (60) |

### Structured, 64 groups vs 256 clusters (split penalty)

| arm | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| round-robin + mean | 0.0350 (5) | 0.1800 (12) | 0.3250 (25) | 0.5450 (64) |
| strided init + mean | 0.2000 (1) | 0.2350 (3) | 0.4900 (6) | 0.9350 (223) |
| strided init + BOUND | 0.0000 (0) | 0.0000 (0) | 0.2300 (1) | 0.8050 (3) |
| **D² init + mean** | 0.9950 (4) | **1.0000 (12)** | **1.0000 (27)** | **1.0000 (68)** |
| **D² init + BOUND** | **1.0000 (4)** | **1.0000 (12)** | **1.0000 (27)** | **1.0000 (68)** |

### Random control (no structure — the honest bound)

| arm | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| round-robin + mean | 0.0000 (5) | 0.0450 (12) | 0.1050 (25) | 0.4200 (64) |
| strided init + mean | 0.0300 (5) | 0.0900 (13) | 0.2050 (26) | 0.5000 (64) |
| D² init + mean | 0.0300 (6) | 0.0700 (14) | 0.1700 (28) | 0.4350 (66) |
| D² init + BOUND | 0.0100 (5) | 0.0250 (12) | 0.0850 (25) | 0.3200 (62) |

No arm beats ~0.5 with no structure to find. Correct, and the point of the
control: **the primitive buys nothing on a geometrically flat LM head.**

## What the 2×2 says — and it is not what Issue 657 predicted

1. **Fix B did all of it.** D² init alone takes recall 0.675 → **1.0000 at a 2%
   budget** (`topk=5` of 255 clusters). The failed gate was an *initialization*
   defect, not an objective defect.
2. **Fix A alone makes things worse.** Strided + bound is *below* strided +
   mean at every budget (0.675 → 0.410 at 25%). The bound adds
   `‖h‖ · radius_c`, and since `radius_c` varies little across clusters while
   `‖h‖` is shared, the term is near-constant noise that swamps the signal at
   tight budgets. As a *ranking function* the bound is a downgrade.
3. **The bound's real value is a different product.** It is *admissible*, so it
   supports an exact stop rule that the mean score cannot.

The diagnosis in Issue 657 was reasoned from the right evidence ("102 of 252
clusters admitted and still missing the argmax 32% of the time — so the
information is there and the scoring is wrong") and reached a conclusion that
the measurement refutes. The information was indeed there; what was wrong was
the *clustering*, not the *scoring* of it.

## Admissible stop — recall is 1.0 by construction, cost is the result

Visit clusters in descending bound order; stop when the next bound falls at or
below the best **exact** logit already found. Every unvisited cluster provably
holds nothing larger, so the argmax cannot be missed. Recall 1.0 is *asserted*
in the harness, not measured — a regression there is a correctness bug.

| regime | active | FLOP ratio vs full head |
|---|---|---|
| structured, groups == clusters | **7.30%** | 13.70× |
| structured, 64 groups | **6.79%** | 14.72× |
| random control | **99.99%** | 1.00× |

This is the honest shape of the primitive: on a clustered head it proves the
exact argmax after touching 7% of the vocabulary; on a flat head it degrades to
touching all of it.

## G3 latency — interleaved protocol, 3 runs

Measured with riir-ai's interleaved protocol (4 warmup pairs discarded, 20
measure pairs, **alternating** A→B / B→A within each pair, median of **per-pair**
ratios). An earlier revision of this bench used median(A)/median(B) and returned
**2.44× on one run and 1.42× on the next from identical deterministic inputs** —
neither number was trustworthy. That is the same failure mode as riir-ai Bench
666 (1.19× win → 0.87× loss) and Issue 658 (1.24× → 0.95×).

| regime | run 1 | run 2 | run 3 | per-pair spread (run 1) |
|---|---|---|---|---|
| structured | 2.66× | 2.12× | 2.89× | 0.95–3.57 |
| random control | **0.08×** | **0.08×** | **0.08×** | 0.05–0.09 |

Absolute medians, run 1: structured `standard 0.6092 ms` vs
`admissible 0.3750 ms`; random `standard 0.6413 ms` vs `admissible 9.2830 ms`.

### Why 2–3× and not 13.7×

`standard_lm_head` dispatches through `matmul_parallel` — rayon across all
32 768 rows. The clustered path walks scattered token IDs on **one thread**. So
the wall-clock win is always far below the FLOP ratio, and on the control regime
(where the bound cannot prune) the same FLOPs run serially against a parallel
baseline and the ratio inverts to 0.08× — a **12× regression**.

That gap is a fixable implementation limit, not a property of the method: a
rayon-parallel or gather-then-`matmul_parallel` stage 2 should recover most of
the 13.7×. Filed as Issue 661. The structured spread (0.95–3.57 over 20 pairs)
is also wide enough that only the *direction* is established here, not the
magnitude.

## Promotion

**NOT promoted to default-on.** G2b passes and G1 holds, but:

- **G3 fails on the control regime** (0.08×). A primitive whose cost depends on
  input geometry cannot be defaulted on without knowing which regime production
  is in.
- **Both regimes are synthetic.** Planted-Gaussian and uniform-random are the
  two extremes; a real LM head sits somewhere between, and *where* is precisely
  the number that decides this. Unmeasured — Issue 662.
- The measured operating point is a **serial** stage 2. Re-gate after Issue 661.

What did land, unconditionally: **D² seeding replaces strided seeding as the
default** in `cluster_map_from_embeddings`. That is a strict improvement with no
regime dependence (it changes only which map is built, not what the hot path
costs), and the strided variant survives solely as `ClusterInit::Strided` so
this bench can keep attributing.

## Caveats

- Synthetic LM heads, not a real checkpoint — the load-bearing caveat, carried
  forward from Benchmark 657 and now the gating one (Issue 662).
- The strided-init pathology is **amplified by the fixture**: Benchmark 657
  deliberately interleaves groups by token ID to defeat round-robin, which is
  exactly the adversarial ordering strided seeding cannot survive. On a real
  vocabulary strided seeding is arbitrary rather than adversarial, so fix B's
  share of the gain here is an **upper bound** on what it buys in production.
  This does not weaken the correction — the recorded 0.675 was still a fixture
  artifact, not a property of the primitive — but it does mean "D² init fixes
  recall" must be re-established on real weights.
- 200 probes ⇒ recall resolution ±0.005.
- Recall and active% are bit-identical across all 3 runs (deterministic);
  only the latency ratios vary.
