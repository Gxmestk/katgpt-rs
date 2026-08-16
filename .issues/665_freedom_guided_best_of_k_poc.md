# Issue 665: PoC — freedom-guided best-of-K selection mode (extension-count criterion)

**Date:** 2026-08-16
**Research:** [katgpt-rs/.research/486_Freedom_Of_Function_Extension_Count_Selection.md](../.research/486_Freedom_Of_Function_Extension_Count_Selection.md)
**Source:** Bennett, "Why the Third Axis Is Freedom" (arXiv:2608.05423 / Zenodo 10.5281/zenodo.21965230); XM = arXiv:2607.27372
**Priority:** low (research-distilled, no blocking consumer)

---

## Problem

Bennett proves best-of-K selection should prefer, among candidates within a loss gate of the winner, the one that **opens an unoccupied output region** (Δ log extension count) — freedom of function provably orders generalization (future compatibility ∝ freedom in unseen contexts), and the criterion beat standard XM on ImageNet (FID −3.7%, recall +11.7% at fixed K=25). Our best-of-K substrates select by relevance (`BestQ`), frequency (`mode@K`), residual (`Top1Converged`), or stability (`best_of_n_stability`) — **none consume occupancy/extension structure**. Workspace audit (2026-08-16) found zero extension-count/least-commitment machinery anywhere in the 7 repos.

The paper's own evidence has an unresolved confound: no random-near-best control (gain may come from merely relaxing the min-loss choice, not from freedom). Our PoC must separate the two.

## Falsifiable question

Does near-best selection by Δ-log-freedom beat (a) strict min-loss, (b) random-near-best, and (c) the shipped selection modes, on a controlled toy where ground-truth coverage is enumerable — at matched candidate budget K?

## Tasks

- [ ] **T1** `katgpt-core` extension-count module (feature `freedom_selection`, opt-in): `log_freedom(&[u32]) -> f32` (Σ log(2^a − 1), a = occupied-cell counts per context over a declared finite partition), `freedom_gain(occupancy, candidate_cell) -> f32`, near-best loss gate (absolute or relative tolerance). Zero-alloc, `#[cfg(test)]` pin: product formula matches brute-force enumeration on small vocabularies.
- [ ] **T2** Wire as a selection mode on ONE existing substrate (cheapest: `renoise_ce::best_of_n_stability` sibling or `dd_tree` `WidthSelectionMode::FreedomGain`) — selection-only change, no training.
- [ ] **T3** PoC in `riir-poc` (defend-wrong shape, ≥3 competitors, no training, `CARGO_TARGET_DIR=/tmp/...`, clean up): toy generator with declared output partition + occupancy table; arms = min-loss / random-near-best / BestQ-or-stability (shipped analog) / FreedomGain; metrics = hit probability under a declared distribution shift (child→parent, Exp-2 shape) + end-state log-freedom. Print verdict table.
- [ ] **T4** Gate: FreedomGain beats min-loss AND random-near-best (the confound control) → open plan + GOAT gate (feature flag + bench); otherwise record negative result in Research 486 §PoC Addendum and close.
- [ ] **T5** If promoted: Theorem-7 allocation formula ([1−(λ/(K·p_j))^{1/(K−1)}]+) evaluated as a second primitive for exploration-budget priority — separate gate, do not bundle.

## Notes

- Finite vocabulary + admissibility rule is load-bearing (raw support saturates useless under softmax/dense mass — paper §13). Use a thresholded prototype/codebook occupancy, the Exp-3 controller shape.
- Fusion leads that must NOT be bundled into this PoC (each needs its own consumer + gate): Thm-6 demonstration priority for coverage_curiosity; freedom-tiered `SlackCorePartition` in neuron-db; per-NPC plasticity scalar in riir-ai — see Research 486 §3.
- Vocabulary hazard: `speculative/qmc` "min freedom" means sample independence — unrelated; do not reuse the bare word in API names (`freedom_gain` ok, bare `freedom` module-ambiguous — prefer `extension_count` where visible).
