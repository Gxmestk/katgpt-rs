# Benchmark 656 — MTP Metal Batch-Width Floor

**Date:** 2026-08-16
**Device:** Apple M3 Max (~400 GB/s peak bandwidth)
**Harness:** `crates/katgpt-backend/examples/bench_mtp_metal_batch_floor.rs`
**Feature:** `gpu_inference`
**Verdict:** **ARTIFACT** — batched speculative verify is affordable on Metal at
shallow depth. llama.cpp's Apple-Silicon MTP loss is an implementation issue,
not a hardware limit. `mtp+ddtree` is viable on M3, in the **N ≤ 4** band only.

---

## Why this ran

A proposed pivot from `dflash+ddtree` to `mtp+ddtree` (native Qwen MTP heads
feeding `build_dd_tree`) was blocked on one question: MTP is reported as a *net
loss at every configuration* on Apple Silicon.

[llama.cpp #23752](https://github.com/ggml-org/llama.cpp/issues/23752) (open,
unconfirmed, filed 2026-05-27) — M1 Max, Qwen3.5-9B-Q4_K_M, 2048 tokens:

| config | tok/s | acceptance |
|---|---|---|
| baseline (no MTP) | 25.3 | — |
| `--spec-draft-n-max 0` | 22.4 | **100%** |
| `--spec-draft-n-max 2` | 21.9 | 73–76% |
| `--spec-draft-n-max 6` | 19.3 | 41–44% |

The load-bearing row is the second: **11% slower while drafting nothing and
accepting everything.** No amount of better tree-building can recover that,
because DDTree's lever is acceptance rate and acceptance is already 100%. If
that penalty were fundamental to Metal, `mtp+ddtree` would be capped below
baseline on this box and MTP would be a CUDA-only play.

Two candidate explanations:

- **ARTIFACT** — llama.cpp evaluates the MTP path unconditionally / adds a sync
  point. A from-scratch Metal implementation would not inherit it.
- **FUNDAMENTAL** — batched verify is genuinely expensive on Metal, so
  speculation can never pay regardless of implementation.

## What was measured

Speculative decoding replaces one width-1 decode step with one width-`N` verify
pass (`N` = draft depth + 1). Decode is dominated by weight-matrix multiplies,
so the question reduces to `cost(N) / cost(1)`. Speculation pays iff that ratio
falls below `E`, the mean accepted tokens per step.

`E` is taken conservatively as **1.7** at `n-max 2`. (If llama.cpp's "73–76%"
is per-token acceptance `p`, the true figure is `1 + p + p² ≈ 2.30`; if it is
already a mean accepted length, ~1.7. The conclusion below holds against the
lower, stricter reading.)

Kernel is written as a real verify pass would be: one thread per output row,
each weight loaded **once** and all `N` columns accumulated in registers, with
`BATCH` baked in as a compile-time constant per pipeline so the accumulator
unrolls into registers instead of spilling. Activations stored transposed
(`[in_dim][BATCH]`) for coalesced reads.

## Results — 3 runs, min-of-40 iterations

`cost(N)/cost(1)`, all three runs:

| shape | threads | GB/s @ N=1 | N=2 | **N=3** | N=4 | N=7 | N=16 |
|---|---|---|---|---|---|---|---|
| `attn_qkv` [4096, 4096] | 4 096 | 127 | 1.19–1.29 | **1.35–1.55** | 1.40–1.51 | 1.95–1.97 | 3.07–3.12 |
| `ffn_up` [11008, 4096] | 11 008 | 235–251 | 0.97–1.03 | **1.19–1.29** | 1.28–1.38 | 1.72–1.81 | 2.54–2.77 |
| `lm_head` [32000, 4096] | 32 000 | 299–315 | 1.01–1.04 | **0.98–1.06** | 1.00–1.11 | 1.26–1.33 | 1.53–1.57 |

Worst-case `cost(3)/cost(1)` per run: **1.35×, 1.39×, 1.55×** — all below the
conservative 1.70× breakeven, in every run.

### Two findings

**1. At speculative depth, batched verify is affordable.** At `N=3` — the exact
config llama.cpp measured as 13% slower — the widest matrix (`lm_head`) batches
for **free** (0.98–1.06×) and even the narrowest costs only ~1.4×, against
≥1.7 tokens won back. The width-N verify cost cannot explain llama.cpp's loss.
Combined with the fact that `n-max 0` — drafting *zero* tokens, requiring no
verify batch at all — is already 11% slower, the penalty is implementation.

**2. The viable band is N ≤ 4.** Cost grows faster than acceptance can pay for
beyond that: at `N=7`, ratios reach 1.95× (`attn_qkv`) against ~1.7 expected
tokens at 41–44% acceptance → a loss. This independently reproduces llama.cpp's
ordering (`n-max 6` worse than `n-max 2`) from kernel cost alone, and is the
part of their result that *is* real.

### Why narrow matrices cost more

Batching gets cheaper as the matrix gets wider — the opposite of a compute
limit, and the signature of an occupancy limit:

- `lm_head` (32 000 threads) saturates bandwidth at ~300 GB/s of a ~400 GB/s
  peak → weights dominate, extra columns ride along free.
- `attn_qkv` (4 096 threads) reaches only 127 GB/s → too few threads to fill the
  GPU, so per-thread work (`in_dim × BATCH` FLOPs) becomes the bottleneck.

The `attn_qkv` penalty is therefore a limitation of this deliberately naive
one-thread-per-row kernel, not of Metal. A tuned kernel (SIMD-group reduction,
multiple threads per row) would raise its occupancy — so the real-world ratio is
**better than measured**, strengthening the ARTIFACT verdict.

## Limitations

- Measures **weight-matmul cost only**. Attention over N positions, the MTP head
  evaluation itself, and KV-cache handling for rejected tokens are unmeasured
  and add cost.
- `attn_qkv` retains 13–17% baseline drift (vs 0.4–4.0% for the saturated
  shapes) — its ratios are ±0.2 at best.
- f32 throughout. Production decode is quantized; quantization raises arithmetic
  intensity, which shifts the balance further toward "batching is free."
- Verdict margin at `N=3` is real but not vast (1.35–1.55× vs 1.70×). It is the
  *direction* that is robust across 3 runs, not the third decimal.

### Harness notes (two corrections that changed the answer)

The first two attempts produced a **wrong verdict** and are worth recording:

1. **Cold GPU.** Without a global warmup the first timed config absorbed shader
   compile + clock ramp, inflating the `N=1` baseline. This *understates* every
   ratio and biased toward "free" — a 64 MB matvec measured slower than a 172 MB
   one, and ratios came back below 1.0. Fixed by `warm_gpu()`.
2. **Per-config weight reallocation.** Allocating and CPU-filling a fresh weight
   buffer (up to 500 MB) per configuration page-faulted between measurements and
   drove baseline drift to 75%. Weights are now allocated once per shape and
   reused — as a real decoder keeps them resident. Drift fell to 0.4–4.0% on the
   saturated shapes.

Estimator is **min-of-40**, not median: every perturbation is strictly additive,
so the minimum is the least-contaminated estimate of true kernel cost.

## Independent confirmation (2026-08-16, follow-up search)

The ARTIFACT verdict was subsequently confirmed from outside, with an explicit
mechanism — and the mechanism is **not** the one this benchmark measured.

**Root cause is per-step kernel dispatch, not matmul width.** llama.cpp
launches a separate Metal GPU kernel for each draft-verify step; on Apple
Silicon that dispatch cost exceeds the speculative gain. MLX instead **fuses
verification into the same compute graph as the main forward pass**, removing
the per-token dispatch penalty — and running the identical speculative idea
through MLX "flips it from a loss to a large win."

**Working Apple-Silicon implementations exist.** [MTPLX](https://github.com/youssofal/MTPLX)
is an MLX-native MTP runtime using the model's own heads with no external
drafter, via "NAX verify kernels + compiled verify":

| hardware | speedup |
|---|---|
| M4 Mac mini (16 GB) | 1.6× |
| M5 Max | 2.24× |
| M4 Mac mini, 9B, depth 1 | 14.4 → 23.0 tok/s |

It supports Qwen 3.8 27B, 3.6 (27B / 35B MoE), and 3.5 (4B / 9B), and
auto-tunes over draft depths **D1/D2/D3** — independently reproducing this
benchmark's `N ≤ 4` viable band.

**llama.cpp's MTP is broken on every backend, not just Metal**, which settles
the artifact question:

- [#23011](https://github.com/ggml-org/llama.cpp/issues/23011) — Qwen3.6-35B-A3B
  self-MTP much slower on Metal **at 95.6% acceptance**. Definitively not an
  acceptance-rate problem.
- [#23698](https://github.com/ggml-org/llama.cpp/issues/23698) — MTP regression
  on the **CPU** backend (44.5 → 21.2 tok/s).
- [RTX 3090 sweep](https://hackmd.io/ODXuOQNzSiyUITz7g9mtBw) — no llama.cpp
  speculative mode beats baseline on consumer Ampere either.

### This repo's Metal backend has the same defect

`crates/katgpt-backend/src/gpu.rs` allocates **9 command buffers and issues 9
`wait_until_completed()` CPU syncs per forward pass**, 8 of them *inside* the
per-layer loop (lines 538–737) — roughly `8 × n_layer` CPU↔GPU round trips per
token. That is precisely the anti-pattern behind llama.cpp's loss.

Building MTP on the backend as it stands would reproduce the failure exactly.
The prerequisite is therefore not merely a batched forward but a **single-submit
(graph-fused) forward**. That work stands on its own: collapsing the per-layer
syncs should speed up ordinary width-1 decoding too, independent of MTP.

## Consequence for the pivot

- **MTP can be gated on Metal**, not CUDA-only. It is not disqualified on the
  primary dev box.
- **Draft depth must stay shallow (N ≤ 4)** on Metal, and should be *adaptive* —
  which is precisely what the existing DDTree budget substrate already does
  (`caddtree_budget` GOAT 7/7, `corr_budget` GOAT 10/10, `entropy_truncate_horizon`).
  The ddtree half of `mtp+ddtree` is what makes the Metal case viable at all;
  it is not the part to pivot away from.
- **The real blocker is not Metal.** Prerequisites, in dependency order:
  1. **Single-submit forward** — collapse `gpu.rs`'s 9 command buffers / 9 CPU
     syncs per pass into one graph. Without this, MTP reproduces llama.cpp's
     loss. Independently valuable for width-1 decode.
  2. **Batched forward** — `InferenceBackend::forward` takes
     `token: usize, pos: usize`; MTP verify needs width-N.
  3. **`nextn`/`mtp.*` tensor loading** — `grep` over `crates/` → zero hits.
  4. Wire MTP marginals into `build_dd_tree` at the existing seam.

## Target model (corrected 2026-08-16)

**Qwen3.8 shipped 2026-08-14**, and is the correct target — an earlier note in
this file's review cycle wrongly concluded it did not exist, from a search that
surfaced only 3.5/3.6.

- **Qwen3.8-27B** (dense) — linear attention on 48 of 64 layers, vision tower,
  **built-in BF16 MTP draft head**, 262K native context (extensible to 1M).
- **Qwen3.8-2.4T-A95B** (MoE) — 512 experts, 92 layers, hybrid linear/full
  attention, MTP trained across multiple steps.
- MTP-bearing quantizations already published (e.g.
  [`lued/Qwen3.8-27B-INT8-W8A16-MTP`](https://huggingface.co/lued/Qwen3.8-27B-INT8-W8A16-MTP)),
  and MTPLX already supports Qwen 3.8 27B on Apple Silicon.

## Related

- `crates/katgpt-transformer/src/mtp.rs` — **this is real MTP substrate**, not a
  name collision (an earlier revision of this file wrongly called it one). It is
  **Gemma 4-flavour MTP** per Research 026, whose three mechanisms all ship:
  1. **Target activations** — `project_target_activation()` (Plan 016/055).
  2. **Shared KV cache** — `mtp_shared_kv_prompt_threshold`.
  3. **Clustered LM head** — `mtp_cluster_classifier` / `mtp_cluster_map` /
     `mtp_cluster_topk`, live at `forward.rs:752` and `forward.rs:1323`.

  The distinction from Qwen-native MTP is real but narrower than stated: Gemma 4
  MTP conditions a *separate small drafter*; Qwen ships the draft head *inside*
  the model. The **clustered LM head is orthogonal to both and helps either** —
  it cuts the final matmul from `[vocab, hidden]` to `[cluster, hidden]` (~100×
  at BPE scale per Research 026). That matters directly here: `lm_head` is the
  dominant matrix in this benchmark (500 MB, 1.7–2.5 ms), and a width-N verify
  computes `N × vocab` logits, so clustering compounds with MTP rather than
  competing with it.

  **Status: implemented and wired, never fed weights** (`mtp_cluster_*` are
  always `None`). Round-robin cluster assignment is the shipped baseline;
  K-means over existing embeddings is planned (Plan 056). Both are deterministic
  → activating this is a **modelless** task, no riir-train dependency.
- `.research/407_Trees_from_Marginals_GDN_Tree_Verify.md` — DFlash-TfM measures
  **4.37× over AR decoding** (24.7% over tuned DFlash) on **Qwen3.6-27B**, the
  exact dflash+ddtree combination this repo already invested in. Flags the GDN
  rollback-free tree-verify as modelless "gold" (masked triangular solve, no
  training) and the acceptance-ceiling result as "a one-line DDTree tuning
  change" (argmax-of-marginal beats full-marginal sampling at depth). Plan 424 +
  `bench_424_dd_tree_deep_argmax.rs` already exist.
- `.research/078_MTP_Cluster_Top_K_Efficient_Embedder.md`,
  `.research/243_Bebop_Entropy_Bounded_MTP_Acceptance_Adaptive_Gamma.md` —
  further MTP clustering / acceptance-adaptive work already distilled.
- `crates/katgpt-speculative/src/dd_tree/` — `build_dd_tree(marginals: &[&[f32]], config)`
  takes distributions and is origin-agnostic; zero `dflash` references in
  production code. MTP marginals drop in at this existing seam.
- `dflare_fusion` / `dflare_kv_routing` / `dflare_progressive_budget` — all three
  already tombstoned in `Cargo.toml` (🪦 IMPROVEMENT GOAT FAILED, research-only).
