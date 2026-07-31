# Issue 565 — Quantization-Compensating Reader-LoRA PoC (BinaryPlasma Unblock Attempt)

> **Filed:** 2026-07-31
> **Research:** [463](../.research/463_moka_freeze_thaw_lever_audit.md)
> **Type:** defend-wrong PoC (per research skill §3.6)
> **Status:** Active

## Context

Research 463 audited whether converting Moka weights to our freeze/thaw format
unlocks any of the 11 rejected levers in
[`moka_head_to_head.md`](../.docs/06_game_arenas/moka_head_to_head.md). The honest
verdict: format conversion alone does NOT unlock any lever. BUT the freeze/thaw
**ecosystem** (reader-LoRA hot-swap, Plan 025) enables one modelless attempt to
unblock BinaryPlasma — recovering the quantization error that PlasmaPath
(Research 110 / Plan 148) currently discards, via a deterministically
constructed (SVD closed-form) low-rank reader adapter.

**The honest pre-PoC prediction (per Research 463 §6):** the PoC will likely
FAIL G5 (win-rate) on this 105K-param network, because the int8 path is already
within noise of f32, and the LoRA matvec overhead will eat the ternary speedup.
The PoC's value is negative knowledge + reusable substrate.

## Tasks

- [ ] **T1** — Build the PoC harness in `riir-ai/crates/riir-poc/` (the
      defend-wrong R&D crate). Three competitors minimum:
      - **A:** PUCT + f32 Moka (the reference baseline, 98% win rate)
      - **B:** PUCT + int8 Moka (the current DEFAULT-ON path, 95% native)
      - **C:** PUCT + ternary Moka + quant-error reader-LoRA (the candidate)
      - **D (control):** PUCT + ternary Moka WITHOUT LoRA (the BinaryPlasma
        rejection baseline — should be the worst)
- [ ] **T2** — Implement `QuantErrorLora::from_error(w_ref, w_quant, out_dim, in_dim, rank)`:
      compute `E = W - dequant(W_q)`, then rank-r truncated SVD of `E`.
      Use `CARGO_TARGET_DIR=/tmp/quant_error_lora_poc` per AGENTS.md.
- [ ] **T3** — Measure G1: cosine similarity of forward-pass output vs f32
      reference, for each of {int8, ternary, ternary+LoRA-r8, ternary+LoRA-r16}.
      Does the LoRA reduce the cosine gap?
- [ ] **T4** — Measure G2: latency overhead of the LoRA matvec vs the
      quantized forward. Target: < 20% overhead.
- [ ] **T5** — Measure G5 (the load-bearing gate): win rate of C vs B
      (n=100, same protocol as Bench 205). Does ternary+LoRA beat int8?
      Target: ≥ 90% (to justify the ternary+LoRA path over the simpler int8).
- [ ] **T6** — If G5 FAILS (predicted): record raw numbers as a §"PoC Addendum"
      in Research 463. Confirm the quantization floor for small CNNs. The
      `quantization_error_lora` primitive still ships as substrate (opt-in
      feature `quant_error_lora`) for larger models where the error manifold
      might be genuinely low-rank.
- [ ] **T7** — If G5 PASSES (surprise): open a plan for the primitive +
      GOAT promotion gate. BinaryPlasma unblocked.
- [ ] **T8** — Update `moka_head_to_head.md` rejection table with the PoC
      result (pass or fail), linking to this issue.

## Rank sweep

The SVD rank is the key hyperparameter. Run T3-T5 at rank ∈ {4, 8, 16, 32} to
find the accuracy/overhead sweet spot. On a 105K-param net, even rank-32 is
a small fraction of the weight count.

## Cleanup

- `rm -rf /tmp/quant_error_lora_poc` when done.
- The PoC bench stays in `riir-poc/` as a permanent regression check (per §3.6).
