# Issue 694: LycheeDecode chain-overlap PoC — measure adjacent-layer same-head top-k overlap on real weights

**Status:** OPEN
**Date:** 2026-08-27
**Research:** [katgpt-rs/.research/514_LycheeDecode_Hybrid_Head_Chain_Propagation.md](../.research/514_LycheeDecode_Hybrid_Head_Chain_Propagation.md)
**Source paper:** [arXiv:2602.04541](https://arxiv.org/abs/2602.04541) — LycheeDecode (ICLR 2026)
**Target:** PoC harness (no production code) — gates a potential `fm_chains` mode on `flashmemory_sparse`
**Repo:** katgpt-rs

---

## Problem

`flashmemory_sparse` (Issue 584, Research 436) — the ACTIVE long-context sparse decode path (riir-ai Bench 671: 1.8× decode @64K, 256K Bonsai on 4090) — amortizes per-head block selection over **time** (τ-step refresh) but recomputes selection at **every layer**. LycheeDecode (Research 514) shows selection can additionally amortize over **depth**: head h's top-k set at layer l reused by head h at layer l+1 — IF the adjacent-layer same-index overlap is high for that head. The paper's Fig. 2 (Llama3): overlap varies **0–100% per head** — chain reuse is head-conditional.

Before implementing a chain mode, measure the overlap on OUR weights. This is the falsifiable gate: cheap (calibration forward passes), decisive (kills or greenlights `fm_chains`).

## Tasks

- [ ] **T1** Harness: one forward pass over calibration prompts (passkey-style needles in long synthetic context, the Issue 584 / RTPurbo calibration recipe), capture per-layer per-head top-k attention index sets (k ∈ {block-equivalent 64-token granularity, and raw k=2048/4096}) at decode-step query positions. Model: Kimi-K3-0.40B local (MLA — matches flashmemory's latent-KV substrate; heads = Q heads over shared KV, the paper's GQA avg-pool note applies).
- [ ] **T2** Statistic: per (l, h) overlap `|topK_h^l ∩ topK_h^{l-1}| / k`, aggregated per head across layers 1..L-1 + per layer across heads; report the distribution (median/quartiles), the head-count above thresholds {0.5, 0.7, 0.9}, and the layer-depth trend (does overlap grow/shrink with depth).
- [ ] **T3** Second model for generality: Bonsai-27B on the 4090 (GPU-exclusive window per the AGENTS.md gate; attention-probe capture through the existing bench harness paths). If 4090 unavailable in a reasonable window, record Kimi-K3-only + mark generality unproven.
- [ ] **T4** Verdict table (the go/no-go):
  - **KILL** — median overlap < ~0.5 on the measured model(s) → chains dead; close issue; note the negative record in Research 514.
  - **GO** — majority head-population ≥ ~0.7 → open the `fm_chains` plan (feature flag; GOAT: G1 selection-cosine ≥ 0.95 vs per-layer refresh at equal budget, G2 selection-compute reduction, G3 no-regression on flashmemory's 169 tests, G4 alloc-free chain reuse).
  - **SPLIT** — bimodal distribution → heterogeneous assignment (high-overlap heads chain, low-overlap refresh per layer) — the expected outcome per the paper's heterogeneity; the overlap statistic joins `HeadCalibration` as the chain-affinity axis.
- [ ] **T5** Record: benchmark note `.benchmarks/` with the overlap tables + the verdict; update Research 514 §Actionable with the outcome; close or transition this issue.

## Notes

- The overlap statistic is modelless (offline forward pass, no training) — this PoC does NOT need HardKuma or any distillation (that's the riir-train recipe line in Research 514 §2.5).
- Secondary observation to capture while probing (free): whether high-overlap heads correlate with high retrieval-score heads (rt_turbo calibration) — if correlated, the chain axis may come free from existing calibration without a separate statistic.
- Do NOT touch production decode paths — PoC harness only (`CARGO_TARGET_DIR=/tmp/...`, clean up after).
