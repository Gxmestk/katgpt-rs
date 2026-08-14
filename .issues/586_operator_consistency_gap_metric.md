# Issue 586: Operator Consistency-Gap Metric (rule-application consistency gate)

> **Source paper:** [arXiv:2608.09888](https://arxiv.org/abs/2608.09888) "BDH-CQ: In-Context Learning with Recurrent Latent Reasoning" — §6.4 within-task consistency
> **Research:** [katgpt-rs/.research/479](../.research/479_BDH_CQ_In_Context_Learning_Recurrent_Latent_Reasoning.md)
> **Filed:** 2026-08-14
> **Type:** PoC / proof task (no plan per global rule)

## Context

BDH-CQ reports an 18.5-point gap between test-pair accuracy (77.9%) and strict-task accuracy (59.4%) on ConceptARC: 52/160 tasks produce one or two correct outputs but are never solved as a whole. The paper's interpretation: a correctly induced rule should transfer to every test input; partial success means the transformation is **not applied consistently** — and isolated correct outputs are indistinguishable from a narrower rule applied completely.

Our stack infers reusable operators (`bisimulation/operator.rs::infer_operators` → `OperatorSchema`, BLAKE3-committed) and binds conditional associations (δ-mem, engram), but has **no shipped measure of whether a bound/inferred rule is applied consistently across repeated or parallel applications**. Grep this session: zero cousins (closest is bisimulation G2 edge-coverage soundness — a different property: it checks the quotient covers edges, not that applications agree).

This metric is the missing half of "demonstration-conditioned operator binding": binding without a consistency gate can promote a flaky operator to committed state (engram / chain commitment) with no signal that it only works on half the inputs.

## Proposal

Add a modelless, zero-alloc consistency measure over operator applications:

- `rule_consistency(applications: &[ApplicationOutcome]) -> ConsistencyReport` — fraction agreement across N applications of the same inferred `OperatorSchema`, with the strict/partial split (all-correct vs some-correct vs none) as a 3-bin histogram, plus a structure-preservation breakdown (correct-output-shape vs localized-error vs construction-failure) mirroring the paper's execution-vs-extrapolation signature.
- Sigmoid confidence on the gap (paper's Wilson-interval analog) so small-N applications don't produce overconfident "inconsistent" verdicts — **not** softmax; per constraint #2.

## Consumers (after PoC proves signal)

1. Promotion gate: inferred → committed operator requires consistency above threshold (ties into engram write policy / freeze gates).
2. Per-NPC integrity signal (riir-ai Cognitive Integrity Layer, Research 129) — feeds riir-ai Issue 672's curiosity targeting: high gap + failures clustered at a complexity level → seek one exemplar at that level.
3. Bench diagnostic: classify failures by structure preservation (execution vs extrapolation) in operator-inference benches.

## Tasks

- [ ] T1: Define `ConsistencyReport` shape (3-bin strict/partial/none + structure-preservation breakdown + sigmoid-gapped confidence) in `katgpt-core` near the bisimulation operator module; `types.rs` style, no deps.
- [ ] T2: Implement `rule_consistency` — fixed-size arrays, zero-alloc, `#[cfg(test)]` unit tests including the paper's anchor cases (19/30 pairs but 2/10 tasks → "partial, gap high").
- [ ] T3: PoC (defend-wrong §3.6 — quality claims need head-to-head, not architectural reasoning): synthetic operator domain where a known operator is applied to K inputs with injected failure regimes (i.i.d. noise vs complexity-clustered misses). Show the metric separates (a) consistent application, (b) random flakiness, (c) complexity-clustered inconsistency — (c) is the signal that triggers exemplar-seeking in riir-ai Issue 672.
- [ ] T4: Feature-gate decision: if T3 signal is clean, wire as opt-in gate on `infer_operators` output (`operator_consistency` feature) + bench note in `.benchmarks/`; if not, record negative result in Research 479 addendum and close.

## Acceptance

- Metric distinguishes the three regimes in T3 with no training and no allocation in the hot path.
- Verdict recorded either way (GOAT gate: G1 regime separation, G2 sub-µs at N≤64, G3 no regression, G4 alloc-free).
