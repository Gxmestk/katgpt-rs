# Issue 691: Stale-Residual Speculative Layer Pipelining POC (arXiv:2608.23841 §6.3)

**Repo:** katgpt-rs (primitive + analyzer) — simulator consumer may live in riir-train
**Research:** [katgpt-rs/.research/508](../.research/508_Pipeline_Native_Transformers_CPU_Decode_CoDesign.md)
**Source:** [arXiv:2608.23841](https://arxiv.org/abs/2608.23841) §6.3 (Approach A + B — the paper's own UNTESTED hypotheses)
**Filed:** 2026-08-26
**Cost estimate:** T1 zero-GPU (offline analyzer on saved traces); T2 zero-GPU (simulator); T3 optional GPU bench

---

## The falsifiable question

For **standard** (non-delay-rewritten) transformer checkpoints, does residual dominance
(`‖δℓ‖/‖x_in^ℓ‖ ≪ 1`) hold strongly enough that layer ℓ+1 can begin on the **stale**
residual `x_in^ℓ` while layer ℓ is still computing — accept-with-correction when the
layer contribution lands small, rollback-and-recompute when it doesn't — yielding a net
wall-clock win from overlapping weight I/O with compute?

The paper proves the vertical-pipeline schedule math for *rewritten* architectures and
proposes speculative recovery as the path for *standard* checkpoints — **without running
it**. No prior art found for stale-residual layer speculation (LayerSkip/early-exit are
different mechanisms: variable depth / conditional compute, not stale-input execution).
We hold checkpoints + trace tooling + rollback machinery → we can produce the first
measured verdict.

## Why the stack can test it cheaply

- Offline analyzer needs only saved per-layer activations (norm in / norm out per layer).
- The rollback machinery pattern ships: GDN tree-verify (rollback-free S₀), token-level
  `rollback_speculative_gpu` (riir-train), KV page fast path (bench_414).
- Distinct from `HydraSkipPlan` (skips layers on cumulative-DE — different mechanism;
  the signal-diff is documented in Research 508 §2.1 #8).

## Tasks

- [ ] **T1 — Residual-dominance analyzer** (katgpt-core, zero-GPU): per-layer
  `‖δℓ‖/‖x_in^ℓ‖` distributions over held activations for 2–3 real checkpoints we
  already run (Gemma-2-2B class, Bonsai-27B class, K3-0.4B class). Paper's own
  viability bar: >50% of layers with ratio < 0.05. Output: per-layer safety table.
- [ ] **T2 — Trace simulator** (zero-GPU): replay saved traces executing layer ℓ+1 on
  stale `x_in^ℓ`; measure accept-rate vs threshold sweep + top-1 divergence (KL on
  logits) + net latency model `(C+IO)/max(C, IO_eff)` (Research 508 §1.2) at our
  stream ratios.
- [ ] **T3 — Approach B probe** (folds in): closed-form least-squares router-logit→
  FFN-delta predictor (no GD — pseudo-inverse on saved activations); does corrected
  speculative input lift accept-rate materially (paper targets R² > 0.7)?
- [ ] **T4 — Verdict + gate decision**: if T1+T2 pass the paper's bar, file the
  wall-clock POC plan (feature flag + bench, G1 bounded-error / G8 accept-rate / G2
  net wall-clock); if they fail, record the negative in Research 508 and close.

## Honest scope notes

- **No quality-parity claim is made here** — this issue TESTS the hypothesis (§3.6
  defend-wrong discipline: the POC defends or refutes; either outcome is a result).
- Ternary-regime caveat (Research 508 §2.0): at 1.58 bits/weight we are only ~2× below
  machine balance — the overlap payoff shrinks; T2's latency model must use OUR stream
  ratios, not the paper's Q4 numbers.
- Attention layers update KV cache — a stale attention input writes stale K/V; T2 must
  model this (the paper flags it as the open hazard for attention-path delays).
