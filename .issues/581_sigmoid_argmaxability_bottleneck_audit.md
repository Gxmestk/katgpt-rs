# Issue 581: sigmoid argmaxability bottleneck — audit our low-rank sigmoid projections

**Date:** 2026-08-10
**Type:** proof + poc
**Research:** [472](../.research/472_Embedding_Retrieval_Dimension_Capacity_Limit.md) §1.6
**Source:** Grivas, Vergari & Lopez, "Taming the Sigmoid Bottleneck: Provably Argmaxable Sparse Multi-Label Classification", AAAI 2024 — [arXiv:2310.10443](https://arxiv.org/abs/2310.10443)
**Status:** T1–T5' DONE 2026-08-10. **The conditional RESOLVED — and it resolved NEGATIVE:** the shipped directions are **rank 2 of 6**, so Benchmark 575's clean verdict does not transfer ([Benchmark 577](../.benchmarks/577_emotion_direction_rank.md)). T6 remains.

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

- [x] **T1 DONE (2026-08-10)** — extracted the decision procedure: combination `y` is argmaxable iff `diag(2y−1)·W·x > 0` is feasible. Two-tier: `rank(W) == L` proves ALL combinations achievable (right inverse ⇒ `W·x = 2y−1` solvable with unit margin); otherwise per-combination perceptron search, reported as *unresolved* rather than *impossible* when no witness is found. Original: Read the paper and the reference implementation (`github.com/andreasgrv/sigmoid-bottleneck`). Extract the exact argmaxability test — the paper gives a decision procedure for whether a given label combination is achievable under a given weight matrix.
- [x] **T2 DONE (2026-08-10)** — `matrix_rank`, `argmaxable_witness`, `audit_argmaxable` + `ArgmaxAudit` in `katgpt-types/src/simd/research.rs` (feature `sigmoid_margin`), re-exported via `katgpt-core`. Clippy clean. Original: Implement that test as a modelless diagnostic in `katgpt-rs` (feature-gated, opt-in). Input: the projection matrix + a candidate output combination. Output: argmaxable / not. Must be allocation-free for the small shapes we care about (8→5).
- [x] **T3 DONE (2026-08-10)** — **EXHAUSTIVE: all 32 combinations argmaxable, `rank(W) = 5 = L`.**
      Also clean at `L=6` with `anger` (64/64). Verified across 16 independently
      seeded direction sets. This is the cheap definitive negative this issue
      predicted — 32 combinations *is* the whole space, so it is a proof, not a sample.

      **Detector validity asserted** so the pass is not vacuous: forcing one
      collinear row drops rank to 4 and makes **12 of 32** unreachable; the paper's
      own regime (`L=12`, `d=3`) leaves **99% (4044/4096)** unreachable.

      **Conditional, though.** The audit uses random direction matrices, so it
      establishes the *structural* claim "any `L ≤ d` with linearly independent
      directions is safe" — not a check of the directions that actually ship
      (`extract_emotion_directions`, derived from recorded HLA state, which could
      come out near-degenerate). See T5'. Original: Run it over the HLA affect bridge: enumerate all `2^5 = 32` affect sign combinations and report which, if any, are unargmaxable under the shipped direction vectors. **32 cases is exhaustive — this is a complete answer, not a sample.**
- [x] **T4 DONE (2026-08-10)** — `neighbor_heal::sigmoid_gated_weights`, `riir-rag` `graph_score` and the `ItemEmbedIndex` sigmoid paths are all **single-output** gates (`L = 1`), so `rank ≥ 1` holds trivially for any non-zero direction. The bottleneck is inherently a multi-label phenomenon; these are structurally non-exposed. Original: Repeat for `ItemEmbedIndex` equip-eligibility-adjacent sigmoid paths and `neighbor_heal::sigmoid_gated_weights`.
- [-] **T5 NOT NEEDED (2026-08-10)** — no unargmaxable combination exists at our
      shape, so the DFT output layer is unnecessary. Deferred rather than done.
      Original: If any unargmaxable combination is found: evaluate the DFT output layer as a closed-form, deterministically-constructed replacement (no training — this must stay modelless per the mandate). Gate behind a feature flag; GOAT gate is "previously-unargmaxable combinations become reachable with no regression on the existing affect benchmarks."
- [x] **T5' DONE (2026-08-10) — and it FAILS: the shipped directions are rank 2 of L=6.**
      [Benchmark 577](../.benchmarks/577_emotion_direction_rank.md).
      `riir-ai/crates/riir-games-civ/src/civ/emotion/rank_audit.rs` behind
      `direction_rank_audit`, consuming `katgpt_core::matrix_rank` (not reimplemented),
      called from `extract_emotion_directions`.

      **This does not upgrade T3's conditional to unconditional — it refutes the
      precondition.** Three structural causes:

      1. **`fear` and `anger` are hardcoded `vec![0.0; embed_dim]`** — only 4 scenarios
         exist (no near-threat / under-attack). A zero direction projects to a constant
         `sigmoid(0) = 0.5`: those two scalars carry **no information at all**.
      2. **`arousal ≡ −valence` exactly** (`cos = −1.000000`) — the formulas use the same
         two scenario groups with the sign swapped, so they are collinear *by
         construction*; no dataset could separate them.
      3. **`desperation` is anti-parallel to `calm`** (`cos = −1.000000`), norm ~400×
         smaller — the recorded scenario means lie essentially on one line.

      **Both dependent pairs are ANTI-collinear, which is the consequential case:** since
      `sigmoid(⟨s,−v⟩) = 1 − sigmoid(⟨s,v⟩)`, such a pair is perfectly *complementary*, so
      **high-valence + high-arousal is unreachable**, as is high-desperation + high-calm.
      Unreachable sign combinations are exactly what Benchmark 575 was hunting.

      Precisely what is / isn't overturned: 575's *structural* claim stands (independent
      directions at `L ≤ d` are safe) and its arithmetic was right; the **precondition**
      fails. The bridge is not "clean because the 8→5 shape is safe" — it is degenerate
      for an unrelated reason, namely how the directions are derived.

      **Deliberate deviation from the literal wording:** the guard *reports* instead of
      asserting. A hard `assert!(rank == L)` would fire on every civ run, since the
      degeneracy is shipped reality with two acknowledged placeholders. Following
      riir-neuron-db's `capacity_audit` precedent it is latched, `debug_assertions`-only
      and feature-gated (zero release cost), and it separates **acknowledged** degeneracy
      (zero rows) from **unexpected** degeneracy (`rank < informative_directions` — two
      *non-zero* dependent rows, `true` on the shipped set). Silently weakening the check
      to make it pass, or breaking every run, were both worse options.

      Rank is tolerance-sensitive (4 at 1e-9, 3 at 1e-6, **2** at every tol ≥ 1e-5), so
      the audit scales its pivot tolerance by `max|W|` and a test pins that stability.
- [ ] **T6** Check the interaction with Plan 410's Linking-Fold — does the fold already restore argmaxability for some combinations? Record the overlap either way. **Re-scoped by T5' (2026-08-10):** the question is no longer "does the fold help an already-clean bridge" but "does it recover any of the axes lost to rank-2 degeneracy". Note the fold cannot fix causes 1 and 2 — a hardcoded zero row and an exact sign-flip are gone before any fold sees them.

- [ ] **T7** (opened by T5') Fix the derivations rather than the symptom: (a) derive `arousal` from a contrast that is not valence-with-the-sign-flipped (e.g. high- vs low-activity states, orthogonalised against valence); (b) give `desperation` a contrast independent of `calm`; (c) either add the near-threat / under-attack scenarios `fear` and `anger` need, or remove those permanently-0.5 fields from the public struct so consumers stop reading them as signal. Then re-run Benchmark 577 — the audit is already wired to report the improvement.

- [ ] **T8** (opened by T5') Audit the affect *consumers*: anything treating the five scalars as five independent axes is mistaken — there are ~2 axes of real information. Check before any sixth scalar is added.

## Expected outcome

Plausibly a **negative result** — at 8→5 the layer is only mildly low-rank, and the theorem bites hardest when labels ≫ features (the paper's setting is thousands of labels). If T3 finds all 32 combinations argmaxable, that is a clean, cheap, exhaustive proof of non-exposure and the issue closes as an honest negative. That outcome is worth having documented, because the rank margin shrinks if the affect vocabulary ever grows beyond 8 dimensions of input.

## Done when

T3 returns an exhaustive verdict for the affect bridge, T4 covers the other two sites, and the result is recorded — as a closed honest negative, or as a DFT-correction gate with a benchmark.

**Amended 2026-08-10 (T5').** The T1–T4 result was recorded as a closed honest negative, but
T5' showed that verdict rested on a precondition the shipped code violates. So the issue does
**not** close here: the bottleneck is not an intrinsic property of our 8→5 shape (575 stands on
that), yet the shipped affect bridge *does* have unreachable combinations — via direction
degeneracy rather than the shape. Closing now requires T7 (fix the derivations) and T8 (audit
the consumers). T6 is re-scoped accordingly.

**The transferable lesson:** an argmaxability/expressivity audit that assumes full-rank
projections proves nothing about a system whose projections are derived from recorded data
unless the rank is actually checked. The check cost ~30 lines and overturned a shipped
conclusion.
