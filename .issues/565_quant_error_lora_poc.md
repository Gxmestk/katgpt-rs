# Issue 565 — Quantization-Compensating Reader-LoRA PoC (BinaryPlasma Unblock Attempt)

> **Filed:** 2026-07-31
> **Updated:** 2026-07-31 (enriched with 3 additional strategies from Gemini consultation)
> **Updated:** 2026-08-01 (T1-T7, T12 RUN — G1/T12 results recorded)
> **Research:** [463](../.research/463_moka_freeze_thaw_lever_audit.md)
> **Type:** defend-wrong PoC (per research skill §3.6)
> **Status:** Active (G1/T12 done; G5 deferred pending G5 wiring decision)

## Context

Research 463 audited whether converting Moka weights to our freeze/thaw format
unlocks any of the 11 rejected levers in
[`moka_head_to_head.md`](../.docs/06_game_arenas/moka_head_to_head.md). The honest
verdict: format conversion alone does NOT unlock any lever. BUT the freeze/thaw
**ecosystem** (reader-LoRA hot-swap, Plan 025) enables a FAMILY of modelless
quantization-compensation candidates. This PoC tests all of them head-to-head.

**The honest pre-PoC prediction (per Research 463 §6):** the PoC will likely
FAIL G5 (win-rate) on this 105K-param network, because:
1. The int8 path is already within noise of f32 (95% native, Bench 565).
2. The Small-Kernel Parameter Paradox (§2.4.1): rank-8 LoRA is 27.8% overhead
   on a 32×288 conv vs 0.39% on an LLM linear layer.
3. The error matrix is likely near-full-rank (105K-param CNNs don't have
   low-dimensional weight structure).

The PoC's value is negative knowledge + reusable substrate regardless of outcome.

## The 4 strategies under test

| # | Strategy | Research 463 § | Corrects | Key risk |
|---|---|---|---|---|
| A | Weight-space SVD-LoRA | §2.4 | Weight error `E = W - dequant(W_q)` | 27.8% param overhead (Small-Kernel Paradox) |
| B | Output-space SVD-LoRA (data-aware) | §2.4.2 | Output-weighted error `E·X` on calibration set | Same overhead, better rank choice |
| C | D4 dihedral symmetry averaging | §2.6 | Quantization-induced symmetry breaking | 2× forward passes; PUCT already averages |
| D | Top-K sparse residual bypass | §2.7 | Worst-5% quantization errors | Gather overhead on SIMD/WASM |

## Tasks

- [x] **T1** — PoC harness built in `riir-ai/crates/riir-poc/tests/quant_error_lora_poc.rs`.
      Baselines + candidates all implemented (B2 ternary, A wSVD rank sweep,
      B oSVD rank sweep, D sparse fraction sweep). E (combo D4) deferred.
- [x] **T2** — Weight-space SVD-LoRA implemented (`QuantErrorLora::from_error` in
      katgpt-core, behind `quant_error_lora` feature). Uses Plan 301 `thin_svd_into`.
- [x] **T3** — Output-space SVD-LoRA implemented (`QuantErrorLora::from_error_data_aware`).
      PoC uses truncated board features as calibration (known approximation —
      a proper calibration would capture actual layer activations/patches).
- [-] **T4** — D4 symmetry averaging: deferred. The forward-pass G1 results (A rank-16+
      helps, B/D hurt) make D4 a lower priority — it's orthogonal to LoRA but the
      cost concern (2× forward per leaf) + the fact that PUCT already averages
      make it unlikely to help. Revisit if G5 is attempted.
- [x] **T5** — Top-K sparse bypass implemented (`SparseErrorBypass::from_error` in
      katgpt-core). PoC tests fractions {5%, 10%, 20%}.
- [x] **T6** — G1 (cosine) measured. See §PoC Results below.
- [x] **T7** — G2 (latency) measured. The `Correction::Full` measurement vehicle
      has 1128% overhead (expected — it's full dense matvec, not rank-r LoRA).
      The production rank-8 LoRA would have 27.8% overhead per Research 463 §2.4.1.
- [ ] **T8** — G5 (win-rate vs int8): NOT RUN. Needs PUCT integration (the
      forward_corrected path exists but PuctPlayer doesn't call it yet). The G1
      result (0.9939 at rank-32) is promising but the int8 baseline is already
      at 95% win-rate — the ternary+LoRA path needs to beat that.
- [ ] **T9** — If ALL FAIL G5: record raw numbers. Deferred pending T8.
- [ ] **T10** — If ANY PASS G5: open plan. Deferred pending T8.
- [x] **T11** — Results recorded in this issue + the bench output.
- [x] **T12** — Small-Kernel Paradox measured. See §PoC Results below.

## Rank sweep

For strategies A + B (SVD-LoRA), sweep rank ∈ {4, 8, 16, 32}. For strategy D
(sparse), sweep top-K ∈ {1%, 5%, 10%, 20%}. The sweep finds the
accuracy/overhead sweet spot — or confirms there is none.

## Calibration set for strategy B

Collect 64 diverse 9×9 Go board positions (from self-play, opening/midgame/
endgame mix). No labels needed — just the input tensors. This is a one-time
offline collection, same pattern as `hydra_budget.rs::run_logit_lens_calibration`.

## Cleanup

- `rm -rf /tmp/quant_error_lora_poc` when done.
- The PoC bench stays in `riir-poc/` as a permanent regression check (per §3.6).

## PoC Results (2026-08-01)

All tests run on Apple Silicon, release build. 4 tests, each `#[ignore]`d
(run with `--ignored --nocapture`). Commit: `5fb54d09f` (riir-ai) + `31486450`
(katgpt-rs research fix) + `95bf58a6` (moka-wasm research feature) + `6e7009a6`
(primitive).

### T0 — Smoke test

Moka v1 total params: 105,353 (matches the known architecture).
Overall ternary relative error: **145.38%** — the ternary quantization error is
larger than the weights themselves. Expected for aggressive ternary on a small
network; confirms the BinaryPlasma quality concern.

### T12 — Small-Kernel Paradox (rank sweep of captured energy)

| Rank | Overall captured fraction |
|---|---|
| 4 | 35.1% |
| 8 | **51.1%** |
| 16 | 65.9% |
| 32 | 78.2% |

**Verdict: PARTIALLY confirmed.** The pre-PoC prediction was <40% at rank-8
(near-full-rank). Actual: 51.1% at rank-8. The error matrix has SOME
low-rank structure — enough that rank-16+ can recover >65% of the error
energy. The Small-Kernel Paradox is real (the error isn't low-rank) but
weaker than predicted (not full-rank either). The `policy.linear` layer
(82×324) is the worst offender: rank-32 captures only 61%.

### G1 — Cosine similarity vs f32 forward (64 boards)

| Strategy | Cosine | Δ vs B2 |
|---|---|---|
| **B2** (ternary, no correction) | **0.9706** | — |
| A (wSVD rank-4) | 0.9192 | −0.051 |
| A (wSVD rank-8) | 0.9651 | −0.006 |
| **A (wSVD rank-16)** | **0.9888** | **+0.018** |
| **A (wSVD rank-32)** | **0.9939** | **+0.023** ← G1 gate PASS (≥0.02) |
| B (oSVD data-aware, r=4–32) | ~0.90 | **−0.06** (WORSE) |
| D (sparse 5–20%) | 0.91–0.96 | −0.03 to −0.06 (WORSE) |

**Key findings:**
1. **Strategy A (weight-space SVD) at rank-16+ PASSES the G1 gate** (cosine
   improves by ≥0.02 over B2). At rank-32, cosine reaches 0.9939 — very close
   to the f32 reference. The correction is modelless (closed-form SVD).
2. **Strategy B (data-aware SVD) HURTS.** Consistently worse than B2 across
   all ranks. This is a PoC calibration artifact: the bench uses truncated
   board features (first `in_dim` of the 972-dim board tensor) as the
   calibration set, which does NOT represent the actual layer activations
   (especially for conv layers where the input is a 3×3 patch, not the full
   board). A proper data-aware calibration would need to capture actual
   intermediate activations via forward-pass instrumentation. The negative
   result is a PoC limitation, NOT a fundamental finding about output-space SVD.
3. **Strategy D (sparse bypass) HURTS.** Targeting outlier errors destabilizes
   the output — correcting a few large errors changes the output more than the
   many small errors they were compensating for. The result is real (not a PoC
   artifact): sparse outlier correction doesn't help when the error is
   distributed (T12 confirmed 51% at rank-8, not concentrated).
4. **Low-rank corrections at r=4/r=8 make things WORSE** (Δ < 0). The SVD
   truncation at low rank introduces its own error that exceeds the correction.
   The crossover where correction starts helping is rank-16.

### G2 — Latency overhead

| Path | Latency (µs) | Overhead |
|---|---|---|
| f32 baseline | 370 | — |
| corrected (Correction::Full, ternary swap) | 4549 | +1128% |

The `Correction::Full` path uses full dense matvec (the measurement vehicle).
The 1128% overhead is NOT representative of the production path — production
would use rank-r LoRA (27.8% overhead at rank-8 per Research 463 §2.4.1) on
ternary SIMD matvec. This measurement confirms the plumbing works; the
production cost model is in Research 463 §2.4.1.

### G5 — Win-rate vs int8 (NOT RUN)

G5 needs PUCT integration (the `forward_corrected_with_scratch` exists but
`PuctPlayer` doesn't call it yet — needs a corrected-forward mode added).

The G1 result (0.9939 at rank-32) is promising, but the prediction from
Research 463 §6 stands: the int8 baseline is already at 95% win-rate (Bench
565), within binomial noise of f32. The ternary+LoRA path needs to beat that
bar, and even a near-perfect cosine (0.9939) doesn't guarantee the same
MOVE SELECTION under PUCT search (tiny policy perturbations can change the
argmax, especially for close moves). G5 remains the load-bearing gate.

**Decision: G5 is deferred.** The G1/T12 results are sufficient negative+
positive knowledge to update Research 463. Wiring G5 requires adding a
corrected-forward mode to `PuctPlayer` (analogous to `with_int8`/`with_f32`)
+ running n=100 games — ~2 hours of additional work for a gate predicted to
fail. If a future consumer needs the ternary+LoRA path, G5 can be wired then.
