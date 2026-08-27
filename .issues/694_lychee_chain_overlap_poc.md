# Issue 694: LycheeDecode chain-overlap PoC — measure adjacent-layer same-head top-k overlap on real weights

**Status:** RESOLVED — KILL (Bench 686); T3 [-] generality unproven
**Date:** 2026-08-27
**Research:** [katgpt-rs/.research/514_LycheeDecode_Hybrid_Head_Chain_Propagation.md](../.research/514_LycheeDecode_Hybrid_Head_Chain_Propagation.md)
**Source paper:** [arXiv:2602.04541](https://arxiv.org/abs/2602.04541) — LycheeDecode (ICLR 2026)
**Target:** PoC harness (no production code) — gates a potential `fm_chains` mode on `flashmemory_sparse`
**Repo:** katgpt-rs

---

## Problem

`flashmemory_sparse` (Issue 584, Research 436) — the ACTIVE long-context sparse decode path (riir-ai Bench 671: 1.8× decode @64K, 256K Bonsai on 4090) — amortizes per-head block selection over **time** (τ-step refresh) but recomputes selection at **every layer**. LycheeDecode (Research 514) shows selection can additionally amortize over **depth**: head h's top-k set at layer l reused by head h at layer l+1 — IF the adjacent-layer same-index overlap is high for that head. The paper's Fig. 2 (Llama3): overlap varies **0–100% per head** — chain reuse is head-conditional.

Before implementing a chain mode, measure the overlap on OUR weights. This is the falsifiable gate: cheap (calibration forward passes), decisive (kills or greenlights `fm_chains`).

> **PREMISE CORRECTION (found at T1, 2026-08-28):** the issue assumed 8 full-attention MLA layers. Kimi-K3-0.40B is a **hybrid** — `kda_layers [1,2,3,5,6,7]` are linear (delta-rule) attention, only `full_attn_layers [4,8]` = layer_idx 3 and 7 are full attention (MLA). **Zero adjacent full-attn layer pairs exist**; the only hostable chain is L3→L7 (span 4), which is what the PoC measured.

## Tasks

- [x] **T1** Harness: one forward pass over calibration prompts (passkey-style needles in long synthetic context, the Issue 584 / RTPurbo calibration recipe), capture per-layer per-head top-k attention index sets (k ∈ {block-equivalent 64-token granularity, and raw k=2048/4096}) at decode-step query positions. Model: Kimi-K3-0.40B local (MLA — matches flashmemory's latent-KV substrate; heads = Q heads over shared KV, the paper's GQA avg-pool note applies).
  - *Resolution:* 9 fixed prompts (3 depths × 3 needles), 3,637-token contexts, 48 greedy decode steps each, CPU float32 eager. Captured at the only two full-attention layers (3, 7 — see premise correction). k grid: raw {512, 1024, 2048} + block-64 {8, 16, 32} blocks; raw k=4096 degenerate at this context (3,637 < 4,096 → trivially 1.0), excluded. CPU path: fla kernels are triton-only on macOS → pure-torch `KimiDeltaAttention.forward` reimplementation (validated against the shipped `ref_logits_bos.npy`: cosine 0.999999, argmax match) + `eager_attention_forward` wrapper for capture. Probe: `.benchmarks/686_lychee_overlap_probe.py`.
- [x] **T2** Statistic: per (l, h) overlap `|topK_h^l ∩ topK_h^{l-1}| / k`, aggregated per head across layers 1..L-1 + per layer across heads; report the distribution (median/quartiles), the head-count above thresholds {0.5, 0.7, 0.9}, and the layer-depth trend (does overlap grow/shrink with depth).
  - *Resolution:* with one full-attn pair, the population is 8 heads × 1 pair. Medians: raw512 **0.135** / raw1024 **0.236** / raw2048 0.527; blk8 0.147 / blk16 0.287 / blk32 0.522. **Every value sits at or below the k/n chance baseline** (0.140 / 0.280 / 0.559 / 0.140 / 0.281 / 0.561). Head-counts ≥0.5: 0/8 (raw512), 0/8 (raw1024), 8/8 (raw2048 — chance artifact, k = 55% of context), 5/8 (blk32 — chance artifact); ≥0.7: 0/0/0/1 (chance). Depth trend N/A (single pair); decode-step trend flat (first8 0.2596 vs last8 0.2542); needle-depth trend flat (0.2491/0.2489/0.2489). Bonus: same-head margin negative 6/8 heads; within-layer off-diagonal 0.574/0.680 ≫ cross-layer same-head 0.236 (clusters by layer, not head identity). Contrast: time-axis step-to-step overlap L7 0.753 / L3 0.381 vs 0.280 chance — harness detects real structure where it exists.
- [-] **T3** Second model for generality: Bonsai-27B on the 4090 (GPU-exclusive window per the AGENTS.md gate; attention-probe capture through the existing bench harness paths). If 4090 unavailable in a reasonable window, record Kimi-K3-only + mark generality unproven.
  - *Resolution:* 4090 reachable via Tailscale but **busy** (sibling compute process `riir_poc-*` resident, 10.3 GB RSS — owner's GPU-exclusivity rule); additionally no attention-prob tap exists on the Bonsai Rust GGUF/cudarc path (Issue 717's layer capture is hidden-state only) — a new harness, not a cheap path. Disposition: **Kimi-K3-only, generality unproven** (the issue's own escape hatch).
- [x] **T4** Verdict table (the go/no-go):
  - **KILL** — median overlap < ~0.5 on the measured model(s) → chains dead; close issue; note the negative record in Research 514.
  - **GO** — majority head-population ≥ ~0.7 → open the `fm_chains` plan (feature flag; GOAT: G1 selection-cosine ≥ 0.95 vs per-layer refresh at equal budget, G2 selection-compute reduction, G3 no-regression on flashmemory's 169 tests, G4 alloc-free chain reuse).
  - **SPLIT** — bimodal distribution → heterogeneous assignment (high-overlap heads chain, low-overlap refresh per layer) — the expected outcome per the paper's heterogeneity; the overlap statistic joins `HeadCalibration` as the chain-affinity axis.
  - *Resolution:* **KILL.** Median overlap < 0.5 at every budget ≤1024 tokens, and stronger than the criterion: overlap is at/below chance at every k (no head above chance anywhere ≤1024); the paper's 0–100% head heterogeneity is entirely absent (distributions unimodal, tight) — SPLIT's bimodality does not exist on this model. The paper's adjacent-layer mechanism itself remains untested for dense models (see T3); this kills `fm_chains` for the Kimi-K3 family and documents that this architecture cannot host adjacent-layer chains at all.
- [x] **T5** Record: benchmark note `.benchmarks/` with the overlap tables + the verdict; update Research 514 §Actionable with the outcome; close or transition this issue.
  - *Resolution:* [Bench 686](../.benchmarks/686_lychee_chain_overlap_poc.md) + probe artifact [`.benchmarks/686_lychee_overlap_probe.py`](../.benchmarks/686_lychee_overlap_probe.py); Research 514 outcome addendum added under §Actionable; determinism verified (two full passes byte-identical). Honest limitation recorded: the 0.4B model fails passkey retrieval at these context lengths (identical non-retrieval continuations; BOS control no better), so the retrieval-head secondary is not measurable on this model — L7 Pearson −0.005, L3 −0.709 driven by one outlier at n=8.

## Notes

- The overlap statistic is modelless (offline forward pass, no training) — this PoC does NOT need HardKuma or any distillation (that's the riir-train recipe line in Research 514 §2.5).
- Secondary observation to capture while probing (free): whether high-overlap heads correlate with high retrieval-score heads (rt_turbo calibration) — if correlated, the chain axis may come free from existing calibration without a separate statistic.
  - *Outcome:* not measurable on this model — the 0.4B fails passkey retrieval outright (identical non-retrieval continuations across all needles; ~1K-token BOS control no better), so retrieval-y heads cannot be identified behaviorally. Correlations recorded (L7 r = −0.005; L3 r = −0.709, single-outlier-driven at n=8) carry no signal.
- Do NOT touch production decode paths — PoC harness only (`CARGO_TARGET_DIR=/tmp/...`, clean up after).
  - *Honored:* zero cargo invocations; Python-only probe; `data/` read-only.
