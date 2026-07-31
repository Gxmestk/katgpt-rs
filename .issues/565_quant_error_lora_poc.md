# Issue 565 — Quantization-Compensating Reader-LoRA PoC (BinaryPlasma Unblock Attempt)

> **Filed:** 2026-07-31
> **Updated:** 2026-07-31 (enriched with 3 additional strategies from Gemini consultation)
> **Research:** [463](../.research/463_moka_freeze_thaw_lever_audit.md)
> **Type:** defend-wrong PoC (per research skill §3.6)
> **Status:** Active

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

- [ ] **T1** — Build the PoC harness in `riir-ai/crates/riir-poc/` (the
      defend-wrong R&D crate). Baselines + candidates:
      - **B0:** PUCT + f32 Moka (reference, 98% win rate, Bench 205)
      - **B1:** PUCT + int8 Moka (DEFAULT-ON, 95% native, Bench 565)
      - **B2 (control):** PUCT + ternary Moka WITHOUT correction (the BinaryPlasma rejection)
      - **A:** PUCT + ternary Moka + weight-space SVD-LoRA (rank ∈ {4,8,16,32})
      - **B:** PUCT + ternary Moka + output-space SVD-LoRA (rank ∈ {4,8,16,32}, 64-board calibration set)
      - **C:** PUCT + ternary Moka + D4 symmetry averaging (2 random transforms per leaf)
      - **D:** PUCT + ternary Moka + top-5% sparse bypass
      - **E (combo):** PUCT + ternary Moka + D4 averaging + top-5% sparse bypass
- [ ] **T2** — Implement weight-space SVD-LoRA (`QuantErrorLora::from_error`):
      compute `E = W - dequant(W_q)`, then rank-r truncated SVD of `E`.
- [ ] **T3** — Implement output-space SVD-LoRA (`QuantErrorLora::from_error_data_aware`):
      compute `E_out = E·X` on 64-board calibration set, SVD, project `A = U_r`,
      `B = U_r^T · E`. Follows the established calibration-set pattern
      (`rt_turbo/calibration.rs`, `fpcg_goat_gate.rs`).
- [ ] **T4** — Implement D4 symmetry averaging: pick 2 random `g ∈ D4` per leaf,
      transform board → forward → inverse-transform policy → average.
- [ ] **T5** — Implement top-K sparse bypass: select top-5% `|E[i,j]|`, store as
      COO, sparse matvec correction.
- [ ] **T6** — Measure G1 (cosine): forward-pass output similarity vs f32
      reference for each strategy. Does the correction reduce the cosine gap?
- [ ] **T7** — Measure G2 (latency): overhead of each correction vs the
      quantized forward. Target: < 20% overhead for the LoRA/sparse paths;
      document the 2× cost of D4 averaging.
- [ ] **T8** — Measure G5 (the load-bearing gate): win rate of each candidate
      vs B1 (int8 baseline), n=100, same protocol as Bench 205 (PUCT budget=200,
      vs Moka greedy). Target: ≥ 90% to justify the ternary+correction path
      over the simpler int8 path.
- [ ] **T9** — If ALL strategies FAIL G5 (predicted): record raw numbers as a
      §"PoC Addendum" in Research 463. Confirm the quantization floor for small
      CNNs. Ship the best-performing correction primitive as opt-in substrate
      for larger models.
- [ ] **T10** — If ANY strategy PASSES G5 (surprise): open a plan for that
      primitive + GOAT promotion gate. BinaryPlasma unblocked.
- [ ] **T11** — Update `moka_head_to_head.md` rejection table with the PoC
      result (pass or fail per strategy), linking to this issue.
- [ ] **T12** — Measure the Small-Kernel Parameter Paradox empirically: what is
      the actual rank of the error matrix `E` for Moka's conv layers? If it's
      near-full-rank (rank ≈ 32), that confirms the structural failure mode.

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
