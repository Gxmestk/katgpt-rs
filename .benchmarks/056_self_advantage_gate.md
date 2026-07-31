# Bench 056: Self-Advantage Recursion Gate GOAT (Plan 283 Phase 4)

**Date**: 2026-06-16
**Feature Gate**: `self_advantage_gate` (promoted to **default-on** after GOAT PASS)
**Source**: [arxiv:2511.16886](https://arxiv.org/abs/2511.16886) — "Latent Reasoning in TRMs is Secretly a Policy Improvement Operator"
**Research**: [250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md](../.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md)
**Benchmark**: `benches/self_advantage_gate_bench.rs`

## Summary

GOAT proof for the self-advantage recursion gate primitives. The `AdvantageMarginGate`
detects dead-compute in latent recursion loops by comparing pre/post recursion logits
(no teacher, no oracle). **4/4 GOAT gates PASS.** Forward-pass reduction ranges from
**2.68× (vocab=1024) to 6.76× (vocab=8)** at **100% argmax quality preservation** with
threshold=0.01. Per-call latency is **41–500 ns** for vocab ≤ 128 (game AI action spaces).

## Structural Note: EarlyStopGate Comparison

Plan T4.1 called for A/B against `EarlyStopGate`. This is **structurally impossible**
as a drop-in comparison:

| Gate | Trait | Signal | Role |
|------|-------|--------|------|
| `EarlyStopGate<P>` | `ScreeningPruner` | `(depth, token_idx, parent_tokens)` | Tree-path expansion screening |
| `AdvantageMarginGate` | standalone | `(pre_logits, post_logits, candidate)` | Recursion-loop continuation gating |

They operate at different abstraction layers and are **complementary**, not competitive.
`EarlyStopGate` has no logits access and does not gate recursion loops. The honest
baseline is **no-gate** (always run `max_steps`).

**No demotion of `EarlyStopGate`** — it serves a different role (DDTree path screening)
and remains default-on for its purpose.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | Geometric blend `logits ← 0.5·logits + 0.5·target` per step |
| Max steps | 20 (budget the no-gate baseline always exhausts) |
| Threshold | 0.01 (practical default — see Finding #1 below) |
| Cases per vocab | 200 (deterministic xorshift64, seed `0xA5A5_0000 \| vocab`) |
| Vocab sizes | 8, 32, 128, 1024 |
| Build | Release (`--release`, `bench` profile) |
| Platform | macOS (aarch64) |
| Timing | Best-of-200 after 50 warmup iters (`std::time::Instant`) |

## GOAT Gate Results

### G1: Forward-Pass Reduction (≥ 2×)

| Vocab | Baseline steps | Gated steps | Reduction | Status |
|-------|---------------|-------------|-----------|--------|
| 8     | 4000          | 592         | **6.76×** | ✅ PASS |
| 32    | 4000          | 759         | **5.27×** | ✅ PASS |
| 128   | 4000          | 1026        | **3.90×** | ✅ PASS |
| 1024  | 4000          | 1493        | **2.68×** | ✅ PASS |

All vocabs exceed the 2× target. Reduction decreases with vocab size because larger
action spaces require more steps to converge (more tokens to suppress).

### G2: Argmax Quality Preservation (≥ 95%)

| Vocab | Match rate | Status |
|-------|-----------|--------|
| 8     | 100.0%    | ✅ PASS |
| 32    | 100.0%    | ✅ PASS |
| 128   | 100.0%    | ✅ PASS |
| 1024  | 100.0%    | ✅ PASS |

**Zero quality loss** across all vocabs. The gate stops only after the dominant token's
argmax is locked in. The well-separated target distribution (dominant at +6..+8, others
at −2..−1.5) ensures convergence is unambiguous before the gate fires.

### G3: Latency (< 1 µs for vocab ≤ 128)

| Vocab | Latency (µs) | Status |
|-------|-------------|--------|
| 8     | 0.041       | ✅ PASS |
| 32    | 0.125       | ✅ PASS |
| 64    | 0.209       | ✅ PASS |
| 128   | 0.500       | ✅ PASS |

**Informational (not gated — O(vocab) scaling):**

| Vocab | Latency (µs) | Note |
|-------|-------------|------|
| 256   | ~1.0        | Borderline — game AI rarely exceeds vocab=128 |
| 1024  | ~4.0        | Still < 1% of a ~500µs forward pass → 125× ROI on first skip |

### G4: Robustness Across Vocab Sizes

All vocabs {8, 32, 128, 1024} pass G1 + G2. ✅ PASS.

## Threshold Sensitivity (vocab=32)

| Threshold | Gated steps | Reduction | Argmax match | G1 | G2 |
|-----------|------------|-----------|-------------|-----|-----|
| 0.000     | 4000       | 1.00×     | 100.0%      | ✗   | ✓   |
| 0.001     | —          | —         | —           | —   | —   |
| 0.005     | 888        | 4.50×     | 100.0%      | ✓   | ✓   |
| **0.010** | **759**    | **5.27×** | **100.0%**  | ✓   | ✓   |
| 0.050     | 583        | 6.86×     | 100.0%      | ✓   | ✓   |
| 0.100     | 486        | 8.23×     | 100.0%      | ✓   | ✓   |
| 0.500     | 400        | 10.00×    | 100.0%      | ✓   | ✓   |
| 1.000     | 345        | 11.59×    | 100.0%      | ✓   | ✓   |

## Key Findings

### Finding #1: threshold=0.0 Never Fires (Critical Design Note)

At threshold=0.0 (the mathematical KL-centered criterion from Eq. 18), the gate
**never stops early** — 1.00× reduction across all vocabs. Root cause:

> `margin(candidate) = A(candidate) − KL(π+ ‖ π̂)`
>
> At threshold=0.0: accept iff candidate's advantage ≥ average. For converging
> recursion where the candidate IS the convergence target, it *always* benefits
> above average → the gate is effectively disabled.

**Fix**: Use threshold=0.01 as the practical default. This means "only continue if the
candidate benefits *meaningfully* more than average" — the correct interpretation for
dead-compute detection. The `Default` impl still uses 0.0 (mathematically correct per
Eq. 18), but all examples and the benchmark use 0.01.

**Action item**: Consider changing `AdvantageMarginGate::default()` to 0.01 in a future
commit. Current default (0.0) is technically correct but practically useless for
dead-compute detection in convergent scenarios.

### Finding #2: Latency is O(vocab) with Good Constants

The function performs 7 passes over the data (2× log_softmax at 3 passes each + 1
subtraction pass). Latency scales linearly: ~4ns/element. For game AI action spaces
(vocab ≤ 128), this is sub-µs. For LLM-scale vocabs (32k+), it would be ~130µs —
still negligible vs a ~500µs–50ms forward pass.

### Finding #3: Quality is Perfect at All Thresholds

Even at threshold=1.0 (11.59× reduction), argmax match remains 100%. This is because
the geometric blend model converges monotonically — once the dominant token is locked
in, further steps only sharpen probabilities without changing argmax. Real models with
non-monotonic convergence might show quality degradation at high thresholds.

## GOAT Gate Summary

| # | Gate | Criterion | Result | Status |
|---|------|-----------|--------|--------|
| G1 | Forward-pass reduction | ≥ 2× all vocabs | 2.68×–6.76× | ✅ PASS |
| G2 | Argmax quality | ≥ 95% match | 100% all vocabs | ✅ PASS |
| G3 | Latency | < 1µs vocab ≤ 128 | 41–500 ns | ✅ PASS |
| G4 | Vocab robustness | G1+G2 all vocabs | all pass | ✅ PASS |

**Overall: 4/4 GOAT PASS.**

## Recommendation

**Promote `self_advantage_gate` to default-on** (done — added to `default` feature list).
The `AdvantageMarginGate` primitive is validated for dead-compute detection in recursion
loops. No quality loss, significant forward-pass savings, negligible overhead.

**Do NOT demote `EarlyStopGate`** — it serves a different role (tree-path screening, not
recursion-loop gating). They are complementary.

**Keep `product_policy_sharpen` opt-in** — not separately benchmarked. It's a utility
wrapper for controllable log-space interpolation, not a gate.

**Deferred (Phase 2)**: Wiring into `LoopMode::WeightShared` (T2.2) and
`SpeculativeGenerator` trait (T2.3) — these touch the hot inference path and require
broader design review. The gate primitive itself is validated and available by default.

## Commands to Reproduce

```bash
# Run the GOAT benchmark (now works with default features)
cargo bench --bench self_advantage_gate_bench

# Exit code 0 = PASS, 2 = FAIL (CI-gatable)
```

## Files

| File | Role |
|------|------|
| `benches/self_advantage_gate_bench.rs` | GOAT gate benchmark (G1–G4) |
| `crates/katgpt-pruners/src/self_advantage.rs` | Primitive implementation (Phases 1–3) |
| `.benchmarks/056_self_advantage_gate.md` | This file |

## Related

- `.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md`
- `.plans/283_self_advantage_recursion_gate.md`
- `.benchmarks/011_sdpg_bandit_arena.md` (SDPG cross-validation baseline)

## TL;DR

4/4 GOAT PASS. `AdvantageMarginGate` at threshold=0.01 gives 2.68×–6.76× forward-pass
reduction at 100% argmax quality, 41–500ns latency (vocab ≤ 128). Promoted to default-on.
threshold=0.0 (mathematical default) never fires for convergent recursion — use 0.01
practically. `EarlyStopGate` not demoted (different role).
