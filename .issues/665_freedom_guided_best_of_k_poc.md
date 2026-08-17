# Issue 665: PoC — freedom-guided best-of-K selection mode (extension-count criterion)

**Date:** 2026-08-16
**Research:** [katgpt-rs/.research/486_Freedom_Of_Function_Extension_Count_Selection.md](../.research/486_Freedom_Of_Function_Extension_Count_Selection.md)
**Source:** Bennett, "Why the Third Axis Is Freedom" (arXiv:2608.05423 / Zenodo 10.5281/zenodo.21965230); XM = arXiv:2607.27372
**Priority:** low (research-distilled, no blocking consumer)
**Status:** T1–T4 DONE (2026-08-17) — T4 gate **PASS** (64/64 per-seed wins vs
both min-loss AND the random-near-best confound control; see Research 486
§PoC Addendum). Primitive shipped opt-in `freedom_selection`; stays opt-in
until a production consumer A/B + GOAT gate (the switch_cost / Issue-663
precedent).

---

## Problem

Bennett proves best-of-K selection should prefer, among candidates within a loss gate of the winner, the one that **opens an unoccupied output region** (Δ log extension count) — freedom of function provably orders generalization (future compatibility ∝ freedom in unseen contexts), and the criterion beat standard XM on ImageNet (FID −3.7%, recall +11.7% at fixed K=25). Our best-of-K substrates select by relevance (`BestQ`), frequency (`mode@K`), residual (`Top1Converged`), or stability (`best_of_n_stability`) — **none consume occupancy/extension structure**. Workspace audit (2026-08-16) found zero extension-count/least-commitment machinery anywhere in the 7 repos.

The paper's own evidence has an unresolved confound: no random-near-best control (gain may come from merely relaxing the min-loss choice, not from freedom). Our PoC must separate the two.

## Falsifiable question

Does near-best selection by Δ-log-freedom beat (a) strict min-loss, (b) random-near-best, and (c) the shipped selection modes, on a controlled toy where ground-truth coverage is enumerable — at matched candidate budget K?

## Tasks

- [x] **T1** `katgpt-core` extension-count module (feature `freedom_selection`, opt-in): `log_freedom(&[u32]) -> f32` (Σ log(2^a − 1), a = occupied-cell counts per context over a declared finite partition), `freedom_gain(occupancy, candidate_cell) -> f32`, near-best loss gate (absolute or relative tolerance). Zero-alloc, `#[cfg(test)]` pin: product formula matches brute-force enumeration on small vocabularies.
  - Landed as `src/extension_count.rs` (+ `ExtensionOccupancy` O(1) state struct + `LossGate`). Documented deviations: `freedom_gain` takes the partition as a 3rd arg (2-arg form can't distinguish fresh vs occupied cell); empty contexts excluded from the product; first-activation pinned to `FIRST_ACTIVATION_GAIN = 2.0 > ln 3` (raw increment +∞). 8 tests incl. the brute-force enumeration pin.
- [x] **T2** Wire as a selection mode on ONE existing substrate (cheapest: `renoise_ce::best_of_n_stability` sibling or `dd_tree` `WidthSelectionMode::FreedomGain`) — selection-only change, no training.
  - Landed as `renoise_ce::best_of_n_freedom` (freedom sibling of `best_of_n_stability`): drift-gate + max-Δ-log-extension-count over a caller-owned occupancy table. 4 tests. Default-state no-regression verified (1893 default / 1905 feature-on, clippy 0 both states).
- [x] **T3** PoC in `riir-poc` (defend-wrong shape, ≥3 competitors, no training, `CARGO_TARGET_DIR=/tmp/...`, clean up): toy generator with declared output partition + occupancy table; arms = min-loss / random-near-best / BestQ-or-stability (shipped analog) / FreedomGain; metrics = hit probability under a declared distribution shift (child→parent, Exp-2 shape) + end-state log-freedom. Print verdict table.
  - Landed as `examples/freedom_best_of_k.rs` (example, not bench — avoids compiling sibling-dirty dev-dep trees). Both substrate arms call the REAL substrates; all arms replay identical pools (matched budget). G1 determinism asserted in-binary.
- [x] **T4** Gate: FreedomGain beats min-loss AND random-near-best (the confound control) → open plan + GOAT gate (feature flag + bench); otherwise record negative result in Research 486 §PoC Addendum and close.
  - **PASS**: parent-hit 0.7075 vs 0.4453 (min-loss) vs 0.5156 (random-near-best); 64/64 per-seed wins vs both. Decomposition: relaxation buys +0.070, freedom guidance buys +0.192 more (73% of the gain is the freedom signal). Honest findings + full table in Research 486 §PoC Addendum. Promotion (plan + GOAT + consumer A/B) remains a separate owner decision — the feature stays opt-in.
- [-] **T5** If promoted: Theorem-7 allocation formula ([1−(λ/(K·p_j))^{1/(K−1)}]+) evaluated as a second primitive for exploration-budget priority — separate gate, do not bundle.
  - Defer: gated on promotion (T4 pass opened the path but no consumer exists). Reopen with the promotion plan.

## Notes

- Finite vocabulary + admissibility rule is load-bearing (raw support saturates useless under softmax/dense mass — paper §13). Use a thresholded prototype/codebook occupancy, the Exp-3 controller shape.
- Fusion leads that must NOT be bundled into this PoC (each needs its own consumer + gate): Thm-6 demonstration priority for coverage_curiosity; freedom-tiered `SlackCorePartition` in neuron-db; per-NPC plasticity scalar in riir-ai — see Research 486 §3.
- Vocabulary hazard: `speculative/qmc` "min freedom" means sample independence — unrelated; do not reuse the bare word in API names (`freedom_gain` ok, bare `freedom` module-ambiguous — prefer `extension_count` where visible).
