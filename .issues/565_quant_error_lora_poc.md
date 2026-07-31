# Issue 565 — Quantization-Compensating Reader-LoRA PoC (BinaryPlasma Unblock Attempt)

> **Filed:** 2026-07-31
> **Updated:** 2026-07-31 (enriched with 3 additional strategies from Gemini consultation)
> **Updated:** 2026-08-01 (T1-T7, T12 RUN — G1/T12 results recorded)
> **Updated:** 2026-08-01 (G1-B proper-calibration Strategy B — negative result confirmed as REAL)
> **Updated:** 2026-08-01 (G5 RUN — DECISIVELY NEGATIVE: ternary path unviable under PUCT, LoRA doesn't fix it)
> **Research:** [463](../.research/463_moka_freeze_thaw_lever_audit.md)
> **Type:** defend-wrong PoC (per research skill §3.6)
> **Status:** CLOSED — all gates run, G5 DECISIVELY NEGATIVE (ternary+LoRA = 0% win-rate vs f32's 100%). The modelless quant-error-compensating LoRA approach is confirmed unviable for the ternary path; the trained-projection path (riir-train) is the only remaining option, same verdict as Issue 566.

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
      **G1-B follow-up (2026-08-01):** proper activation-based calibration
      landed (`forward_collecting_activations` in katgpt-moka-wasm +
      `g1b_cosine_strategy_b_proper_calibration` test). Proper calibration
      improved B by ~0.02 cosine over truncated-features, but B STILL HURTS
      (Δ≈−0.05). The negative result is REAL, not a PoC artifact: data-aware
      SVD overfits to calibration distribution on small networks.
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
- [x] **T8** — G5 (win-rate vs greedy f32): RUN. Both B2 (ternary-only) and
      A rank-32 (ternary+LoRA) scored **0/20 (0%)** vs f32's **20/20 (100%)**
      through the same corrected-forward harness. The G1 cosine gate (0.9939)
      does NOT translate to PUCT parity — the residual 0.6% error is amplified
      by the softmax priors into a catastrophic strength collapse. See §G5
      Results below.
- [x] **T9** — ALL FAILED G5: recorded. See §G5 Results below.
- [-] **T10** — N/A (no strategy passed G5, so no plan to open). The
      trained-projection path (riir-train) is the only remaining option,
      same verdict as Issue 566.
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

### G1-B — Strategy B proper calibration (2026-08-01)

The initial Strategy B negative result (Δ≈−0.06) was suspected to be a PoC
artifact: truncated board features are a poor approximation of actual layer
inputs (especially for conv layers where the input is a 3×3 im2col patch).

**The fix:** `forward_collecting_activations` (katgpt-moka-wasm commit
`0bd2e29b`) runs the same forward arithmetic but captures each layer's
actual input vectors. The PoC collects 26,688 calibration vectors across
60 layers (81 positions × 64 boards, subsampled to max 512/layer) and builds
the data-aware SVD from REAL activations.

| Strategy | Rank 8 | Rank 16 | Rank 32 |
|---|---|---|---|
| B-OLD (truncated features) | −0.066 | −0.071 | −0.064 |
| B-PROPER (actual activations) | **−0.050** | **−0.046** | **−0.050** |
| A (weight-space SVD, reference) | −0.006 | **+0.018** | **+0.023** |

**Verdict:** proper calibration IMPROVED Strategy B by ~0.02 cosine over
truncated features — confirming the artifact hypothesis. But Strategy B
STILL HURTS even with proper calibration (Δ≈−0.05). The negative result is
now REAL, not an artifact. **Strategy A (weight-space SVD) remains the
winner.**

**Why data-aware SVD underperforms weight-space SVD:** the data-aware SVD
optimizes for `||E·X||²` on the calibration set — it captures the principal
directions of the output error projected through inputs. On small networks
(105K params), the weight structure dominates: the intrinsic error
structure of the weight matrix (what weight-space SVD captures) generalizes
better to unseen inputs than the calibration-conditioned output error.
Data-aware SVD overfits to the calibration distribution. This is the
opposite of what happens in large LLMs (where data-aware approaches like
GPTQ/OBQ outperform weight-only quantization), suggesting the Small-Kernel
Parameter Paradox applies to the data-aware axis too.

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

### G5 — Win-rate vs greedy f32 (RUN, DECISIVELY NEGATIVE)

The corrected-forward mode was wired into `PuctPlayer` (behind the `research`
feature: `with_corrected` constructor + `forward_leaf` branch). The G5 test
runs n=20 games at budget=50 vs greedy f32 Moka, same seed convention as
`native_puct_winrate.rs` so the games are directly comparable.

| Strategy | Win-rate (n=20) | Verdict |
|---|---|---|
| **f32** (zero-corr harness) | **100% (20/20)** | Harness verified correct |
| B2 (ternary-only) | **0% (0/20)** | Ternary path catastrophically bad |
| A rank-32 (ternary+LoRA) | **0% (0/20)** | LoRA correction does NOT help |

**Root cause:** The G1 cosine gate (0.9939 at rank-32) is a NECESSARY but NOT
SUFFICIENT condition for PUCT parity. PUCT uses softmax priors to guide search
exploration — even tiny logit differences (total abs diff ≈ 45 across 82 moves
= 0.55/move) change the exploration distribution. Over a budget=50 search,
these perturbations compound: the search explores slightly different branches,
and the value head's small errors accumulate. The ternary PUCT player passes
excessively (26+ passes per game vs ~17 moves), indicating it evaluates its
positions as losing — consistent with the value head's residual error.

**The key lesson:** cosine similarity 0.9939 is NOT sufficient for PUCT parity.
The bar for PUCT-usable forward-output fidelity is higher than for greedy-move
parity (where int8's 0.97 cosine suffices for 85-95% win-rate). This is because
PUCT amplifies small policy perturbations through search, while greedy move
selection only depends on the single argmax.

**Comparison to int8:** The int8 path achieves 85-95% win-rate with 0.97
cosine. The ternary+LoRA path achieves 0% with 0.9939 cosine. The difference:
int8's error is UNIFORM (small, symmetric noise) while ternary's error is
STRUCTURED (large, biased — 145% relative error), and the LoRA correction
removes the bias but leaves residual structure that PUCT amplifies.

**Decision:** G5 FAILS. The modelless quant-error-compensating LoRA approach
is unviable for the ternary path. Issue 565 is CLOSED. The trained-projection
path (riir-train) is the only remaining option, same verdict as Issue 566.
