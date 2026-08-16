# Issue 657 — Clustered LM head: replace mean-logit ranking with an admissible max bound

**Status:** Open
**Opened:** 2026-08-16
**Owner:** katgpt-rs
**Blocks:** Plan 574 T6 (promotion)
**Evidence:** `.benchmarks/657_clustered_lm_head_goat.md`, commit `df072ca2`

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

- [ ] Add `radius_c` to the classifier artifact (or a parallel `Vec<f32>`).
- [ ] Bound-based stage-1 scoring in `clustered_lm_head`, behind the existing
      opt-in path.
- [ ] Re-run `.benchmarks/657` — target: recall >= 0.99 at <= 10% active.
- [ ] If admissible-mode recall hits 1.0, report the active% it costs; that is
      the real operating point.

## Note on the random control

The bench's no-structure control also shows k-means beating round-robin (0.50 vs
0.42 at 25%). That is probably a **norm effect** — k-means groups high-norm rows
together and high-norm tokens are likelier to be argmax — not semantic
structure. The radius bound would make that signal explicit rather than
incidental.
