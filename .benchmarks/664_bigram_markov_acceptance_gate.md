# Bench 664 — Bigram Markov head: the acceptance gate (Issue 659 T4, hardware-independent half)

**Date:** 2026-08-17
**Repo:** katgpt-rs (`crates/katgpt-speculative/src/bigram_markov.rs`, feature `bigram_markov`, opt-in)
**Device:** M3 Max (measurement is hardware-independent — acceptance counts, not wall-clock)
**Predecessor:** [Bench 663](663_bigram_markov_head_primitive.md) (the primitive gate: correctness, cost, alloc, memory)
**Test:** `g2_acceptance_bigram_vs_factorized_floor_heldout`

## What this closes, and what it does not

Issue 659 T4 bundled two things that separate cleanly:

| half | status here |
|---|---|
| **acceptance rate at equal draft depth** (G2) | **MEASURED** (this bench, scoped — see below) |
| **wall-clock on Metal AND 4090 vs the Bonsai target** (G3) | **still deferred** to riir-ai — needs the GGUF runner |

The G2 half needs no GPU and no target model: acceptance is a count of matching
tokens. It was deferred alongside G3 only because the two were written as one
task. It is measured now.

**The floor arm is NOT trained DFlash.** Real DFlash needs Prism-ML's trained
low-rank weights and a target model to verify against. What is measurable
modellessly is the *structural* property DFlash has: it is a **factorized**
head — its per-depth marginals do not condition on the drafted prefix (the
deep-position dilution Plan 424 T6.2 records). So the baseline here is a
**factorized floor**: position-independent unigram marginals from the same
train split. It is labelled a floor everywhere, and the trained-DFlash arm
remains the consumer gate's job.

## Metric correction (the first attempt was wrong)

The initial harness scored each arm by the **highest-scoring** root-to-leaf
chain and produced an apparently damning result: the tree *lost* to the bare
greedy chain as `top_m` grew (0.81 → 0.76). That metric is wrong for a tree
drafter. A tree verifier checks **every** path in one batched forward and
commits the **longest match**. Scoring only the best-scored chain throws away
the tree's whole reason for existing. The corrected metric —
`tree_acceptance` = longest root-to-leaf path matching the held-out
continuation from depth 0 — reverses the finding (0.70 → 2.02).

## Setup

- Corpus: 4,293 bytes of fixed English prose, **embedded in the test** so the
  numbers stay reproducible (a repo file would drift commit to commit).
- Split: first 80% fits the table, last 20% is held out. The tokenizer is built
  over the whole corpus; only the *table* sees train. Held-out prevs the train
  split never saw become zero rows — the honest production behaviour.
- Equal draft depth (`lookahead = 8`), equal `tree_budget`, equal vocabulary
  across all arms.
- Metric: mean tokens a lossless verifier commits per draft cycle.
- Arms: **B** = bigram head via `bigram_build_tree`; **C** = greedy argmax
  chain, no tree (isolates conditioning from tree branching); **F** =
  factorized floor.

## Result — byte-level (vocab 256, well-fitted: 0.1% zero rows)

| budget | top_m | B (bigram) | C (chain) | F (floor) | B/F |
|---|---|---|---|---|---|
| 16 | 1 | 0.7024 | 0.7024 | 0.9106 | 0.77× |
| 16 | 4 | 1.1129 | 0.7024 | 0.9106 | 1.22× |
| 16 | **16** | **1.2671** | 0.7024 | 0.9106 | **1.39×** |
| 64 | 1 | 0.7024 | 0.7024 | 1.3859 | 0.51× |
| 64 | 4 | 1.2741 | 0.7024 | 1.3859 | 0.92× |
| 64 | **16** | **1.6753** | 0.7024 | 1.3859 | **1.21×** |
| 256 | 1 | 0.7024 | 0.7024 | 1.8482 | 0.38× |
| 256 | 4 | 1.3447 | 0.7024 | 1.8482 | 0.73× |
| 256 | **16** | **2.0235** | 0.7024 | 1.8482 | **1.09×** |

**G2 verdict: PASS, scoped.** At its intended operating point (`top_m = 16`)
the head beats the factorized floor at every budget. Two findings matter more
than the headline:

1. **Tree branching, not conditioning, carries most of the gain.** The bare
   greedy chain accepts 0.7024 regardless of budget; the tree reaches 2.0235.
   The consumer must use the `build_dd_tree` seam, not the bare chain.
2. **MEASURED SCOPE LIMIT — the head's edge shrinks as budget grows**
   (1.39× → 1.21× → 1.09×), and at `top_m = 1` the floor *wins* outright. The
   floor is a **coverage** strategy: it spends budget enumerating the globally
   most frequent tokens at every depth, so its acceptance scales with budget,
   while a narrow head proposes the same few chains no matter the budget.
   **Consequence for the consumer:** run this head with a wide `top_m`, and
   expect it to be most valuable under a TIGHT verification budget — which is
   the Metal batch-1 regime it is proposed for. Both directions are pinned by
   assertions so the scope note cannot silently rot.

## Result — word-level: DATA-STARVED, no verdict drawn

A word-level arm was added to probe whether vocabulary *sparsity* changes the
picture (Bonsai's vocab is 131,072; 256 bytes is unrepresentatively dense).
It does not answer that question:

| budget | top_m | B | C | F | B/F | 0-row% |
|---|---|---|---|---|---|---|
| 16 | 16 | 0.1400 | 0.1000 | 0.3467 | 0.40× | **36.0** |
| 256 | 16 | 0.1467 | 0.1000 | 0.6467 | 0.23× | **36.0** |

636 training words over a 356-word vocabulary cannot fit a bigram model:
**36% of held-out prevs were never seen**, so the head emits a zero row and
proposes nothing, while the floor needs no conditioning and is unaffected. This
arm measures **corpus size, not vocabulary sparsity**, so **no quality
conclusion is drawn from it** — the head is not shown to be weak on sparse
vocabularies; it is shown to be unfittable on 636 tokens. The confound itself
is pinned by an assertion (zero-row rate > 25%) so this cannot later be
misread as a measured verdict.

Answering the sparse-vocab question needs a Bonsai-scale corpus — which is the
riir-ai consumer gate, not a katgpt-rs unit test.

## Gate summary

| Gate | Verdict | Note |
|---|---|---|
| G1 correctness | **PASS** (Bench 663) | unchanged |
| G2 acceptance | **PASS, scoped** | this bench: 1.09–1.39× over the factorized floor at `top_m=16`; loses at `top_m=1`; trained-DFlash arm still open |
| G3 wall-clock (Metal + 4090) | **NOT MEASURED** | needs riir-ai's Bonsai GGUF runner |
| G4 alloc-free | **PASS** (Bench 663) | unchanged |
| G5 memory bound | **PASS** (Bench 663) | 17 MB at Bonsai scale |

**Promotion: NO.** `bigram_markov` stays **opt-in**. G3 is the headline
motivation of Issue 659 (the Bench 656 mode-2 claim) and is unmeasured; a
promotion on G2 alone would promote an unproven wall-clock win.

## Structural invariants pinned (both regimes)

- Tree acceptance ≥ bare-chain acceptance (the tree contains the chain).
- At `top_m = 1` the tree **is** the chain (bit-equal means).
- At a budget that admits them, `top_m = 16` ≥ `top_m = 1` (extra successors
  cannot remove paths).

## Commands

```bash
CARGO_TARGET_DIR=/tmp/katgpt-659-fix cargo test -p katgpt-speculative \
  --features bigram_markov --lib g2_acceptance -- --nocapture
```

Suite: 320 passed / 0 failed with `bigram_markov` (305 default + 15).
Clippy: zero diagnostics in `bigram_markov.rs`. The one remaining crate
warning (`dd_tree/mod.rs` unused `HashMap`) is pre-existing and cross-feature
(the import *is* used at `mod.rs:1017` under a different gate) — untouched,
same as Bench 663 recorded.
