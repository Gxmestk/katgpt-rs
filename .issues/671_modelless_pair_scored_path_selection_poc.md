# Issue 671 — Modelless pair-scored path selection (DFlash 2 selector over dflash marginals + bigram table)

**Research:** [katgpt-rs/.research/490_DFlash2_Pair_Scored_Path_Selection.md](../.research/490_DFlash2_Pair_Scored_Path_Selection.md)
**Source:** DFlash 2 blog (Inco AI, 2026-08-18) — "choosing is cheaper than predicting": +2.0M params / +0.6% latency path selector beats DSpark's sequential Markov correction (40× fewer params, 16× lower overhead) by scoring adjacent candidate pairs and walking one coherent path.
**Upstream lineage:** Issue 659 (bigram head) + Issue 670 (TreePath, RESOLVED) + riir-ai Issue 717 / Bench 693/694 (consumer gates).

## Problem

Bench 694 measured the selection headroom on our seam: the candidate structure contains the target path ~89% of the time (tree acceptance 0.8785–0.8940 at budget 256) but the greedy bigram chain realizes 0.2984 tok/cycle. The G3a wall-clock FAIL is verify-side economics (K sequential verify forwards + rollback per cycle) — break-even needs acceptance ≈ 0.5–0.7 at K small. The tree path to that band requires a batched tree-verify wall-clock harness (armed, Issue 717); the **selector path reaches the same band at chain-verify cost** and needs no new verify infrastructure.

Neither shipped selection mechanism consumes both scoring signals: `dd_tree::extract_best_path` scores nodes by marginal only (no transition term); the bigram chain conditions on transitions only (no per-position context evidence). DFlash 2's selector is exactly the composition — and both ingredients ship here.

## Hypothesis

`S_t(a,b) = U_t(b) + λ_t · log P_bigram(b|a)` — per-position candidates from `dflash_predict_parallel` marginals (`U`), adjacent-pair coherence from `BigramMarkovTable` (replacing DFlash 2's trained rank-256 bilinear `A·H·B` — the deterministic-construction precedent of Issue 659), per-position entropy/forecast sigmoid gate for `λ_t` (the `H(h_t)` analog) — walked greedily from the last verified token, lifts chain acceptance from 0.2984 toward the 0.5–0.7 break-even band without training and without tree verification.

## Tasks

- [ ] **T1** Lattice scorer: `pair_score(a, b, t) = U_t(b) + λ_t · log P(b|a)` over per-position top-m candidates from parallel marginals (m ∈ {8, 16}); zero-alloc, feature-gated under the `bigram_markov` family (opt-in).
- [ ] **T2** Gate `λ_t`: entropy-sigmoid (`λ₀ · σ(−κ·H(p_t))`) or margin-sigmoid; λ sweep {0, 0.5, 1, 2}; λ=0 must reduce to argmax-of-marginals (property test).
- [ ] **T3** Walk: greedy best-successor from last verified token; optional max-product Viterbi variant O(K·m²) (record whether it beats greedy — DFlash 2 ships greedy only).
- [ ] **T4** Offline acceptance harness (Bench 694 P1 rig): arms = argmax-of-marginals chain / greedy bigram chain (0.2984 baseline) / pair-scored selection / tree ceiling (0.8785–0.8940 upper bound) / pure-unigram-`U` negative control.
- [ ] **T5** Gate G1 (correctness): λ=0 ≡ argmax-of-marginals bit-identical; lossless chain verify unchanged (LeviathanVerifier path).
- [ ] **T6** Gate G2 (the headline): offline acceptance at depth 8 ≥ 0.5 on the proxy corpus → unblocks the G3a Metal wall-clock re-run; report the λ/m matrix either way.
- [ ] **T7** If G2 passes: G3a wall-clock re-run on the Bench 694 harness (M3 Metal, GPU-exclusive per AGENTS rules) — selected chain vs no-drafter baseline at K ∈ {2, 4}.
- [ ] **T8** Record: does the entropy gate beat flat λ? (the `H(h_t)` modelless-analog question — informs whether the gate carries signal or dead weight)

## Non-goals

- Trained selector weights (the whole point is the modelless construction; a trained variant belongs to the Weaver redirect lineage, Research 407).
- Two-tap conv / suffix-decay fixes — training-side, no trained parallel drafter exists in the stack (Research 490 §3 redirect).
- Tree-verify harness (the Issue 717 armed path stays independent; this is the chain-seam alternative, not a replacement).

## References

- Research 490 (this line's note) · Research 316 (DSpark contract; §3.5 path 2 precedent) · Research 407 (acceptance ceiling — the headroom framing) · Research 177 (Domino, the sequential-correction baseline shape)
- Bench 693/694 (riir-ai) — headroom + G3a economics · Issue 659 (bigram head) · Issue 670 (TreePath fix)
- DFlash 2: https://inco.ai/blog/dflash2/ · DFlash: arXiv:2602.06036 · DSpark: arXiv:2607.05147 · EAGLE-2: arXiv:2406.16858 (concept-level selection prior art)
