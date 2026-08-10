# Benchmark 575: sigmoid argmaxability audit — affect bridge (Issue 581 T3)

**Date:** 2026-08-10
**Issue:** [581](../.issues/581_sigmoid_argmaxability_bottleneck_audit.md)
**Source:** Grivas, Vergari & Lopez, *Taming the Sigmoid Bottleneck: Provably Argmaxable Sparse Multi-Label Classification*, AAAI 2024 — [arXiv:2310.10443](https://arxiv.org/abs/2310.10443)
**Research:** [472 §1.6](../.research/472_Embedding_Retrieval_Dimension_Capacity_Limit.md)
**Harness:** `crates/katgpt-types/tests/sigmoid_argmaxability.rs` (feature `sigmoid_margin`)
**Reproduce:**
```bash
cargo test -p katgpt-types --features sigmoid_margin \
    --test sigmoid_argmaxability -- --nocapture
```

---

## Verdict: **honest negative — no exposure, conditional on `rank(W) = L`**

The affect bridge is not vulnerable to the sigmoid bottleneck at its current
shape, and the audit is **exhaustive** rather than sampled: `L = 5` means 32
combinations, which is the entire space.

## Result

| Shape | rank(W) | combinations | argmaxable | verdict |
|---|---|---|---|---|
| **L=5 (synced affect) from d=8** | **5 = L** | 32 | **32 / 32** | ✅ no exposure |
| L=6 (with `anger`) from d=8 | 6 = L | 64 | 64 / 64 | ✅ no exposure |
| L=5, d=8, one collinear row | 4 < L | 32 | 20 / 32 | ⚠️ 12 unreachable |
| L=12, d=3 (the paper's regime) | 3 | 4096 | 52 / 4096 | ❌ **99% unreachable** |

## Why the clean shapes are safe — a proof, not a measurement

When `rank(W) = L`, `W` has a right inverse, so `W·x = 2y − 1` is solvable
*exactly* for every target combination `y`, giving `diag(2y−1)·W·x = 1 > 0` with
unit margin. Every combination is therefore argmaxable by construction — no
search required, and the audit short-circuits with `full_rank_proof = true`.

Full row rank holds whenever `L ≤ d` and the direction vectors are linearly
independent. We project 5 (or 6) emotions from an 8-dimensional HLA, so we have
dimensions to spare. `matrix_rank` confirms rank `= L` for every `L ∈ 1..=8` at
`d = 8`, and saturates at `d` beyond that.

## The detector is not vacuous

A pass on the affect bridge would be worthless if the audit could not detect a
real bottleneck, so two positive controls are asserted:

- **Collinearity:** setting row 4 := row 0 + row 1 drops the rank to 4 and makes
  **12 of 32** combinations unreachable. The mechanism is elementary — if
  `a₄ = a₀ + a₁` then `⟨a₄,x⟩ = ⟨a₀,x⟩ + ⟨a₁,x⟩`, so demanding
  `a₀ > 0, a₁ > 0, a₄ < 0` is unsatisfiable for any `x`.
- **The paper's regime:** `L = 12` labels from `d = 3` features leaves **99%**
  (4044 / 4096) of combinations unreachable, reproducing the "exponentially many
  unargmaxable combinations" claim.

## Honest caveat — what this does *not* establish

The audit runs on **16 independently seeded random direction matrices**, not on
the direction vectors that ship. So the result is a **structural** one:

> For any `L ≤ d` with linearly independent directions, every combination is
> argmaxable.

Our shipped directions come from `extract_emotion_directions` /
`extract_emotion_directions_for_map` (riir-ai `riir-games-civ/src/civ/emotion/`),
which derive them from recorded HLA state. Derived directions *could* in principle
come out collinear or near-degenerate — the third row of the table above is
exactly what that would cost. Nothing currently checks for it.

**Therefore the actionable output of this audit is a guard, not a fix:** assert
`matrix_rank(W) == L` at direction-extraction time. That converts the conditional
result into an unconditional one at negligible cost, and it is the only follow-up
this issue needs. Filed as Issue 581 T5' (replacing the original T5, since the DFT
output layer Grivas et al. propose is unnecessary at our shape).

## Also confirmed as non-exposed

- `neighbor_heal::sigmoid_gated_weights`, `riir-rag` `graph_score` and
  `ItemEmbedIndex`'s sigmoid paths are **single-output** gates (`L = 1`).
  `rank ≥ 1` is trivially satisfied for any non-zero direction, so argmaxability
  is not a concern there — the bottleneck is inherently a *multi*-label
  phenomenon.

## Relationship to Plan 410 (Linking-Fold)

Both results constrain the same sigmoid projections but on different axes, and
they do not overlap:

- **Linking-Fold (Theorem 4.7):** coordinate-wise monotonic activations preserve
  linking number, so linked class manifolds cannot be separated regardless of
  depth. A statement about *which decision boundaries* are reachable.
- **This audit (Grivas):** a low-rank sigmoid layer cannot emit certain *output
  combinations* at all. A statement about *which outputs* are reachable.

A full-rank layer can still be defeated by linked manifolds; the fold does not
raise rank. Issue 581 T6 tracks confirming the two are independent in practice.
