# Issue 581: sigmoid argmaxability bottleneck — audit our low-rank sigmoid projections

**Date:** 2026-08-10
**Type:** proof + poc
**Research:** [472](../.research/472_Embedding_Retrieval_Dimension_Capacity_Limit.md) §1.6
**Source:** Grivas, Vergari & Lopez, "Taming the Sigmoid Bottleneck: Provably Argmaxable Sparse Multi-Label Classification", AAAI 2024 — [arXiv:2310.10443](https://arxiv.org/abs/2310.10443)
**Status:** Open

---

## Problem

The repo mandates **sigmoid, not softmax**, for projections onto direction vectors. Research 472 surfaced a published theorem that constrains exactly this construction:

> A low-rank output layer with sigmoid activation makes **exponentially many label combinations unargmaxable** — they cannot be produced as the prediction for *any* input, regardless of weights or training.

Their fix is a **DFT output layer**, which guarantees every ≤`k`-sparse label combination is argmaxable, while training faster and using up to 50% fewer trainable parameters at equal F1@k.

`argmaxable | rank bottleneck | sign-rank | sigmoid bottleneck` returns **zero hits** across all `.md` in all 7 repos, and zero hits in `*.rs`. This is a genuine gap — and unlike the main capacity bound (Research 472), it is *not* covered by Research 123.

## Why it lands on us

Candidate exposed sites — all are low-rank → sigmoid, and several are default-on:

| Site | Shape | Note |
|---|---|---|
| HLA affect projection | `[f32; 8]` → **5** scalars (valence/arousal/desperation/calm/fear) via sigmoid-dot bridge | The 5 synced scalars are the *only* thing crossing the sync boundary — an unargmaxable affect combination would be a permanently unreachable NPC state |
| `ItemEmbedIndex` | 8-D, sigmoid-gated paths | Already indicted by Plan 410's Linking-Fold theorem for a *different* reason |
| `neighbor_heal::sigmoid_gated_weights` | `riir-neuron-db/src/neighbor_heal.rs:142` | Heal weight selection |
| `riir-rag` `graph_score` | `1/(1+exp(λ·d))`, λ=1.5 | Bounded (0,1], additive fusion |
| `vortex_flow` `BlockTopKRouter` | centroid + dot + sigmoid | Routing, default-on |

The affect-scalar bridge is the highest-stakes one: 8 → 5 is a low-rank sigmoid multi-label layer by any reading, and its outputs are chain-committed.

## Precedent for the right response

Plan 410 (Linking-Fold) is the template: a published impossibility theorem about monotonic activations was absorbed as **(a) a diagnostic** and **(b) a closed-form modelless correction** (`|x| = x + 2·ReLU(−x)`), not a rewrite or a deferral to training. Note that Linking-Fold's theorem *also* covers sigmoid (coordinate-wise monotonic), so these two results are siblings and the audit should check whether the existing fold already mitigates part of this.

Convenient alignment: the DFT fix is not foreign machinery here — `ModellessEmbedder` already uses a DFT magnitude in its 8-D construction (`riir-ai/crates/riir-rag/src/embedder.rs:119-121`), and `katgpt-spectral` ships.

## Tasks

- [ ] **T1** Read the paper and the reference implementation (`github.com/andreasgrv/sigmoid-bottleneck`). Extract the exact argmaxability test — the paper gives a decision procedure for whether a given label combination is achievable under a given weight matrix.
- [ ] **T2** Implement that test as a modelless diagnostic in `katgpt-rs` (feature-gated, opt-in). Input: the projection matrix + a candidate output combination. Output: argmaxable / not. Must be allocation-free for the small shapes we care about (8→5).
- [ ] **T3** Run it over the HLA affect bridge: enumerate all `2^5 = 32` affect sign combinations and report which, if any, are unargmaxable under the shipped direction vectors. **32 cases is exhaustive — this is a complete answer, not a sample.**
- [ ] **T4** Repeat for `ItemEmbedIndex` equip-eligibility-adjacent sigmoid paths and `neighbor_heal::sigmoid_gated_weights`.
- [ ] **T5** If any unargmaxable combination is found: evaluate the DFT output layer as a closed-form, deterministically-constructed replacement (no training — this must stay modelless per the mandate). Gate behind a feature flag; GOAT gate is "previously-unargmaxable combinations become reachable with no regression on the existing affect benchmarks."
- [ ] **T6** Check the interaction with Plan 410's Linking-Fold — does the fold already restore argmaxability for some combinations? Record the overlap either way.

## Expected outcome

Plausibly a **negative result** — at 8→5 the layer is only mildly low-rank, and the theorem bites hardest when labels ≫ features (the paper's setting is thousands of labels). If T3 finds all 32 combinations argmaxable, that is a clean, cheap, exhaustive proof of non-exposure and the issue closes as an honest negative. That outcome is worth having documented, because the rank margin shrinks if the affect vocabulary ever grows beyond 8 dimensions of input.

## Done when

T3 returns an exhaustive verdict for the affect bridge, T4 covers the other two sites, and the result is recorded — as a closed honest negative, or as a DFT-correction gate with a benchmark.
