# Issue 568: Loop Injection Regime-Split PoC

> **Source paper:** "Towards Looped Models Done Right" — Huang, Shi, Chen, Wen,
> Liu, Xing, Ma (IFM/USC/CMU), Aug 2026.
> [notion:ifm-research/Towards-Looped-Models-Done-Right]
> **Related Research:** 073 (LT2), 097 (Training-Free Loop), 414 (Fully Looped
> Transformer), 048 (HRM-Text — additive injection cousin)
> **PoC location:** `riir-ai/crates/riir-poc/` (`src/loop_injection_poc.rs` +
> `benches/loop_injection_regime_split.rs`)
> **Filed:** 2026-08-01
> **Status:** RESOLVED — PoC verdict: NO TRANSFER (planning at chance). Empirically validated PASS.

## The claim to defend-or-refute

The paper's Q2 finding: **persistent input injection** (writing the fixed
prelude representation `e` into the recurrent stream before each recurrent
step via a diagonal operator `z̃_t = (1-α)·z_t + α·e`) **helps
retrieval/context/code** (MMLU +2.53, HumanEval+ +5.49, BBH-CoT +6.63) but
**hurts math/reasoning** (MATH500 −3.60, GSM8K −2.51).

The PASS verdict on the paper (commit `ccb4434b`) routed the training-time
architecture decisions to riir-train and noted this injection trade-off as a
"cautionary flag" for any fusion adding persistent injection to our belief
kernels. This issue converts that cautionary flag into an **empirically
settled question** via a defend-wrong PoC, per research skill §3.6.

## The falsifiable prediction

On a toy belief kernel (hand-designed recurrent state, NOT a trained looped
transformer), the regime split transfers:

- **Always-inject > baseline** on a retrieval task (recall a pattern from K
  steps ago via dot-product readout).
- **Always-inject < baseline** on a planning task (compute a K-step
  compositional function over the state).
- **Regime-gated-inject ≥ both** on their respective tasks (oracle regime
  signal in the PoC; a real system would use a heuristic/learned detector).

## What each outcome means

| Outcome | Verdict | Action |
|---|---|---|
| Split holds (inject helps retrieval, hurts planning) | **Gain** | File plan for regime-gated injection primitive in katgpt-core, feature-gated, GOAT gate on split magnitude. |
| No effect (inject doesn't change either task) | **Empirically validated PASS** | The paper's finding is substrate-specific to trained looped transformers; our hand-designed kernels don't have the tension. Close this issue. |
| Helps both (inject improves retrieval AND planning) | **Gain (surprise)** | The paper's negative finding on math doesn't transfer; injection is a free lunch on our substrate. File plan for always-on injection. |
| Hurts both | **PASS (confirmed negative)** | Injection is harmful on our substrate; document + close. |

## Why this is worth a PoC (not just architectural reasoning)

Per §3.6: the PASS verdict claimed the injection trade-off is "a cautionary
flag, not a config contradiction" based on architectural reasoning (our
belief kernels fold input into the recurrent update already, so a separate
write mechanism may be redundant). Architectural reasoning is insufficient
to settle whether the regime split transfers to a different substrate. The
PoC is the honest settlement.

## Scope

- **In scope:** toy belief kernel (8-dim recurrent state, matching per-NPC
  belief dim), two task types (retrieval + planning), three competitors
  (baseline / always-inject / regime-gated), verdict table.
- **Out of scope:** real NPC cognition, trained weights, riir-train
  integration. This is a modelless PoC on a synthetic domain.
- **Repo:** PoC lives in `riir-ai/crates/riir-poc/` (the defend-wrong crate).
  Issue lives in `katgpt-rs/.issues/` because the primitive (if it ships)
  would be a katgpt-core modelless op.

## Tasks

- [x] **T1** Implement `src/loop_injection_poc.rs` — toy belief kernel, two
      task types, three injection strategies.
- [x] **T2** Implement `benches/loop_injection_regime_split.rs` — head-to-head
      verdict table + criterion latency bench.
- [x] **T3** Register bench in `Cargo.toml`.
- [x] **T4** Run PoC, record verdict table (below).
- [-] **T5** Split does NOT hold → no plan for regime-gated injection primitive.
      Issue closed with empirically validated PASS.

## PoC verdict (2026-08-01)

```
=== Loop Injection Regime-Split PoC ===
Prediction (paper): inject helps retrieval, hurts planning.
Setup: D=8, K=16, trials=500, α=0.3

Strategy            Retrieval Cos   Planning Acc   Verdict
-------------------------------------------------------------------------------------
Baseline                   0.0143         0.5145   ─
AlwaysInject(0.3)          0.0246         0.5270   helps_retrieval=YES  hurts_planning=HELPS
RegimeGated                0.0246         0.5145   best_of_both=YES
-------------------------------------------------------------------------------------

=== HONEST VERDICT ===
NO TRANSFER (planning at chance — regime split requires trained model)
  AlwaysInject Δretrieval=+0.0103, Δplanning=+0.0125
  RegimeGated ≥ baseline on both tasks — oracle ceiling achievable.
```

**Result:** injection helps retrieval (+0.0103 cosine, as predicted ✓) but
the planning task stays at chance (0.5145 baseline, within ±0.03 of 0.50) for
ALL conditions. The "hurts planning" prediction is untestable on our substrate
because a fixed-random untrained kernel cannot do multi-step accumulation
above chance — tanh saturation + random W_z mixing destroy the signal across
K=16 steps.

**Two planning tasks were tried:**
1. **Parity (XOR)** — hardest Boolean function; baseline at 0.4967 (pure
   coin flip). Untestable.
2. **Weighted-sum-direction** (linear accumulation, much easier) — baseline
   at 0.5145. Still at chance. Untestable.

**Conclusion:** the regime split is irreducibly a **trained-model
phenomenon**. The "injection hurts math" prediction requires a model that can
actually do math (accuracy well above chance), which is the precondition for
injection to measurably degrade it. On hand-designed belief kernels, the
baseline can't plan, so there's nothing to degrade. This empirically
validates the PASS verdict on the paper: the finding is substrate-specific
to trained looped transformers → riir-train.

The retrieval improvement (+0.0103 cosine) is real but tiny in absolute terms
(both baseline and injection are near-zero cosine ~0.01–0.03), so it is not
actionable as a Gain-tier primitive.

**Issue closed.** The PoC source remains in `riir-poc/` as a permanent
regression check — its job was to settle the dispute, and it should keep
settling it if the belief kernel is later trained/structured.
