# Issue 697: Numeric-Deviation Contextualization Probe (arXiv:2405.02803)

**Status:** Open — unowned, POC/proof task (probe primitive + reference-band gate)
**Date:** 2026-08-28
**Research:** [`.research/515_Is_Flash_Attention_Stable_Numeric_Deviation_Contextualization.md`](../.research/515_Is_Flash_Attention_Stable_Numeric_Deviation_Contextualization.md)
**Source:** [arXiv:2405.02803](https://arxiv.org/abs/2405.02803) "Is Flash Attention Stable?" (Meta FAIR + Harvard, 2024)
**Consumers:** riir-ai gate layer (Bench 773 successor form; Issue 753 f16-KV long-ctx tolerance schedule), riir-train Issue 492 (drift probe + divergence ledger)

---

The stack's kernel numeric gates all pin hand-picked absolute/relative bands (q8kv `5e-3`, parity `1e-2`/`2e-2`, Bench 773's argmax+max_abs certifiable form). The paper's contextualization acceptance rule replaces arbitrary bands with dominance against references the system demonstrably tolerates: R1 = two-random-init divergence, R2 = precision-change divergence (modelless proxy: quantize→dequant round-trip, labeled a single-step lower bound). `Wasserstein1d` already ships in `katgpt-core::mag::transfer` (Plan 418) — reuse it, do not duplicate.

## Phase 1 — Probe primitive (`katgpt-core`, feature `numeric_stability`, opt-in)

- [ ] **T1.1** `truncate_mantissa(f64, bits)` format emulator + known-value vectors + idempotence test (truncate twice == once).
- [ ] **T1.2** `DeviationReport { max_diff, wasserstein_1d }` computed over tensor pairs; delegate 1-D Wasserstein to `mag::transfer::wasserstein1d`; sorted-quantile determinism (no HashMap-order leakage).
- [ ] **T1.3** Acceptance rule: `accept(reports, refs: &ReferenceBands, margin) -> {Accept, Reject, Inconclusive}`; margin configurable, NO default derived from the paper's context-specific 2–5×.
- [ ] **T1.4** Reference builders: `R1` from two seeds of the init distribution; `R2` proxy from round-trip quantize→dequant — with a doc-truth tripwire test pinning the lower-bound labeling (house precedent: EMPTY_HASH preimage test).
- [ ] **T1.5** Scope-limit tripwire: API docs + test assert the protocol bounds divergence similarity, NOT training stability (the paper's explicit anti-claim; 2510.04212 owns the mechanism).

## Phase 2 — Perturbable reference attention lab (host numeric, no GPU dep)

- [ ] **T2.1** Tiled online-softmax attention with knobs: tile shape (Bc, Br), tile dimension order, mantissa width (via T1.1), seq length. G1: in f64 it equals naive attention exactly (golden identity).
- [ ] **T2.2** Ordering-law gates: max-diff non-increasing in mantissa bits across ≥2 context lengths; Spearman rank correlation between `R = ⌈S/T⌉−1` ordering and measured max-diff over an (S, T) grid (ordinal gate — the first-order predictor is explicitly not an absolute bound); tile-area ordering reproduced at ≥2 formats; dim-order-swap ordering; square-tile row pinned as the negative control.

## Phase 3 — Falsifiability + consumers

- [ ] **T3.1** Planted-deviation gate: deviations at 0.1× / 1.0× / 10× of the reference band must land Accept / margin-line / Reject (a gate that cannot fail proves nothing).
- [ ] **T3.2** `tol(S)` schedule helper: offline fit `tol(S) = tol(S₀)·f(S/S₀)` from the lab, emitted as a pinned constant table (determinism: hash the fit inputs); two-length probe — a kernel passing tol(S₀) must not flip verdict class at 8×S₀. First consumer: the Issue 753 f16-KV path (80K ctx vs its fixed-shape validation).
- [ ] **T3.3** Consumer follow-ups filed on first consumption (substrate-first: riir-ai gate layer consumes; do not fork the probe). riir-train side is Issue 492.

**Constraints:** zero-alloc hot paths (`_into` variants per Plan 418 pattern); sigmoid/none needed; no new deps.
