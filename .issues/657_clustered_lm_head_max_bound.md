# Issue 657 — Clustered LM head: replace mean-logit ranking with an admissible max bound

**Status:** RESOLVED — implemented, and the **diagnosis below was refuted by
its own fix**. Recall 0.675 → 1.0000, but the radius bound did not cause the
gain. Read §Outcome first; the rest is preserved as the (wrong) reasoning.
**Opened:** 2026-08-16
**Resolved:** 2026-08-16
**Owner:** katgpt-rs
**Blocked:** Plan 574 T6 (promotion) — now blocked on Issues 661/662 instead
**Evidence:** `.benchmarks/658_clustered_lm_head_admissible_goat.md` (supersedes
`.benchmarks/657_clustered_lm_head_goat.md`)

## Outcome (2026-08-16)

The fix shipped and **G2b passes: 0.675 → 1.0000 at a 2% active budget.** The
attribution is not what this issue predicted.

Implementing the bound surfaced a second, unrelated defect: k-means seeded its
centres at token IDs `0, stride, 2·stride, …` with `stride = vocab/k`, and
Benchmark 657's fixture assigns group `t % n_groups` with `n_groups == k` — so
every seeded centre came from **two distinct groups**. Benchmark 658 measured
the 2×2 rather than applying both fixes and claiming the result:

| arm | recall @ 25% budget |
|---|---|
| strided init + mean (Plan 574's measurement) | 0.6750 |
| strided init + **bound** — *this issue's fix alone* | **0.4100** (worse) |
| **D² init** + mean — *the init fix alone* | **1.0000** |

**The scoring objective was not the defect; the initialization was.** As a
*ranking function* the bound is a downgrade — `radius_c` varies little across
clusters while `‖h‖` is shared, so the added term is near-constant noise that
swamps the signal at tight budgets.

What the bound *is* good for is the thing the mean score cannot do at all: an
**admissible** stop rule. Visiting clusters in descending bound order and
stopping when the bound falls below the best exact logit proves the argmax after
touching **7.30%** of the vocabulary — recall 1.0 by construction, not by
measurement. That shipped as `ClusterStop::Admissible`.

The reasoning below was sound from the evidence available ("102 of 252 clusters
admitted and still missing the argmax 32% of the time ⇒ the information is
there, the scoring is wrong"). The information was indeed there. What was wrong
was the *clustering*, not the scoring of it.

Promotion is still blocked, on different grounds: Issue 661 (stage 2 is serial,
so the 13.7× FLOP win measures as 2–3× and inverts to 0.08× on flat data) and
Issue 662 (no real checkpoint measured).

---

## Original diagnosis (preserved — refuted above)

## Problem

Plan 574's clustered LM head passes G2a (k-means beats round-robin at every
matched compute budget) and G3 (4.68× speedup), but **fails G2b**: best argmax
recall is **0.675** against a 0.99 target, and only **0.16** at a usable 5%
budget. Promotion is blocked — a 4.68× speedup that returns the wrong argmax is
not a modelless gain (AGENTS.md).

## Diagnosis: the objective is wrong, not the input

Stage 1 ranks clusters by `dot(hidden, centroid_c)`, which equals the cluster's
**mean logit**. The question it must answer is "which cluster contains the
**max** logit". A cluster holding one spike among many low values has a poor
mean and gets pruned even though it owns the argmax.

This is a *scoring* defect, not an information defect. The evidence: at a 25%
budget the selector already admits **102 of 252 clusters** and still misses the
argmax ~32% of the time. If the hidden state lacked information, admitting 40%
of clusters would not leave a third of the mass unreachable.

## Proposed fix (modelless, admissible)

Decompose `w_t = centroid_c + r_t`. Then for any `t` in cluster `c`:

```text
logit[t] = dot(h, centroid_c) + dot(h, r_t)
         <= dot(h, centroid_c) + ||h|| * max_{t in c} ||r_t||
```

Store one extra scalar per cluster — `radius_c = max ||r_t||` — and score with
that upper bound instead of the bare centroid.

This converts a heuristic into an **exact** algorithm: keep every cluster whose
bound exceeds the best exact logit found so far and the argmax provably cannot
be missed. Budget becomes a tunable early-stop rather than a silent correctness
risk.

Cost: `num_clusters` extra f32s (negligible vs the classifier itself) and one
fused multiply-add per cluster in stage 1. Deterministic from shipped weights —
**modelless**, no training.

## Tasks

- [x] Add `radius_c` to the classifier artifact (or a parallel `Vec<f32>`).
      → `cluster_radii_from_map` in `crates/katgpt-forward/src/cluster_build.rs`.
- [x] Bound-based stage-1 scoring in `clustered_lm_head`, behind the existing
      opt-in path. → `clustered_lm_head_bounded` in
      `crates/katgpt-forward/src/cluster_head.rs` (`ClusterStop::TopK` /
      `::Admissible`); the shipped `clustered_lm_head` is unchanged and now
      shares its stage-2 loop.
- [x] Re-run — target recall >= 0.99 at <= 10% active. **Met at 2%**, but by the
      D² init, not the bound. `.benchmarks/658`.
- [x] Admissible-mode recall is 1.0 by construction; it costs **7.30% active**
      on the structured fixture and **99.99%** on the unstructured control.
      That spread, not a single number, is the real operating point.

## Note on the random control

The bench's no-structure control also shows k-means beating round-robin (0.50 vs
0.42 at 25%). That is probably a **norm effect** — k-means groups high-norm rows
together and high-norm tokens are likelier to be argmax — not semantic
structure. The radius bound would make that signal explicit rather than
incidental.

**Measured (Benchmark 658):** the norm-effect reading holds. On the control the
bound arm is *worse* than the mean arm at every budget (0.32 vs 0.435 at 25%),
and the admissible stop needs 99.99% active — i.e. with no geometry the bound
has nothing to exclude. The control is the honest lower bound on the primitive
and is why promotion is still blocked (Issue 662).
