# Issue 662 — Clustered LM head: measure a real checkpoint

**Status:** Open
**Opened:** 2026-08-16
**Owner:** riir-ai (see §Why not katgpt-rs)
**Blocks:** Plan 574 promotion — this is the gating measurement
**Evidence:** `.benchmarks/658_clustered_lm_head_admissible_goat.md`

## The question

Benchmark 658 measures the clustered LM head at the two extremes of a spectrum
and gets opposite answers:

| fixture | admissible active% | wall-clock vs full head |
|---|---|---|
| planted-Gaussian clusters | 7.30% | 2.1–2.9× **win** |
| uniform random rows | 99.99% | 0.08× **loss** |

A real LM head sits between them. **Where** is the entire promotion decision —
there is no way to read it off the synthetic runs, and no amount of further
synthetic work will produce it.

## What to measure

For a real `output.weight` / `token_embd.weight`:

1. `cluster_map_from_embeddings` (D² init) at Gemma 4's shipping ratio
   (`num_clusters ≈ vocab / 128`).
2. `cluster_radii_from_map`, then `ClusterStop::Admissible` over hidden states
   sampled from an actual forward pass — **not** random probes. Real hidden
   states are not isotropic, and the bound's tightness depends on `‖h‖`.
3. Report: mean active%, argmax recall of `ClusterStop::TopK` at 2/5/10/25%
   budgets, and wall-clock under the interleaved protocol.

Real hidden states matter as much as real weights here. Random probes were fine
for a relative comparison between arms; they are not fine for an absolute
operating point.

## Why not katgpt-rs

katgpt-rs has **no GGUF reader** (`grep -ril gguf crates src` → only
`contiguous.rs`, `config.rs`, `enums.rs`, `ternary_trit.rs`, none of them a
parser). The reader is `riir_engine::gguf_loader::{GgufFile, dequant_f16_to_f32,
dequant_tensor_row}` in riir-ai, and riir-ai already path-depends on
`katgpt-core` / `katgpt-hla` / `katgpt-pruners` — so the bench belongs there,
consuming `katgpt-forward`'s builders. katgpt-rs is the public repo and must not
depend on a private sibling.

## Candidate checkpoints

`riir-train/data/`:

- `MiniCPM5-1B-F16.gguf` (2.1 GB) — smallest, start here.
- `gemma-2-2b-it-f16.gguf` (5.2 GB) — `token_embd.weight` ≈ 256000 × 2304 f16
  → ~2.4 GB dequantized f32. K-means at `k≈2000` is ~3.3e10 FLOPs/iteration ×
  10 iterations single-threaded; budget for tens of minutes or reduce `k`.
- `Qwen3.6-27B-UD-Q4_K_XL.gguf` — the MTP-relevant target (Benchmark 656), but
  Q4_K requires `dequant_tensor_row` per row rather than a bulk f16 convert.

## Tasks

- [ ] riir-ai bench: load a real LM head, build the map/classifier/radii.
- [ ] Sample real hidden states from a forward pass; do not use random probes.
- [ ] Report active% + recall + interleaved-protocol latency.
- [ ] Feed the result back into Plan 574 T6 and `.benchmarks/658`.
- [ ] If active% lands near the random control, record Plan 574 as a permanent
      negative and stop — do not tune the fixture until it agrees.

## 2026-08-17 update — harness AUTHORED (riir-ai Bench 688), run deferred

`riir-ai/crates/riir-engine/tests/bench_662_clustered_lm_head_real_checkpoint.rs`
(#[ignore]d, 2 tests: `smoke_real_pipeline_and_exactness` +
`measurement_real_checkpoint`; compile + clippy verified CPU-only). Everything
per spec above: real tied wte head, D² clustering at the shipping ratio 128,
`after_final_norm` probes from `forward_gemma2_trace` over 6 real prompts
(chat-templated + plain), TopK budget curve + packed Admissible exactness
(asserted) + interleaved latency. Full record:
`riir-ai/.benchmarks/688_clustered_lm_head_real_checkpoint_harness.md`.

**Why the run waits:** the 32 GB 4090 box hosts the sibling bonsai training
job (~7.3 GB resident); the bench peaks at ~13 GB host RAM (f32 model + the
+2.4 GB packed permuted copy). RAM-gated, not GPU-gated — run in the same
exclusive window as riir-ai Issue 714 (when bonsai ends), smoke first
(cluster_size=4096, minutes), then the measurement (cluster_size=128 — the
serial PROJ_SEED-pinned k-means over 256000×2304 at k≈2000 is multi-hour CPU;
`RIIR_662_CLUSTER_SIZE=256` halves it).
