# Bench 683 — Stale-Residual Speculative Layer Pipelining POC (Issue 691)

**Verdict: NEGATIVE — the paper's premise (Approach A) fails on every architecture class we hold, by ~an order of magnitude. Approach B (router-logit predictor) fails the R² > 0.7 bar too (0.18–0.45 held-out). The paper's viability criterion is arithmetically unreachable for practical LLM depths.**

- **Source:** arXiv:2608.23841 §6.3 (Approach A + B — the paper's own UNTESTED hypotheses; this is the first measured verdict we know of)
- **Date:** 2026-08-26 (M3 Max, pure CPU, zero GPU)
- **Harness:** `benches/bench_683_stale_residual_poc.rs` (real K3-0.40B weights + real tiktoken prompts) + `riir-ai/crates/riir-engine/examples/stale_residual_trace_dump.rs` (Bonsai-27B / Gemma-2-2B "SRTR" v1 traces, pure CPU, consuming the EXISTING capture surfaces: `forward_qwen_deltanet_ternary_with_capture` + `forward_gemma2_trace`)
- **Primitives:** `katgpt-core/src/stale_residual.rs` (opt-in `stale_residual` feature) + `src/kimi_k3/stale_residual.rs` (the K3 simulator: snapshot/restore, per-layer capture, replay-from-layer-ℓ+1)

## T1 — Residual dominance (the paper's viability bar: >50% of layers with median ‖δℓ‖/‖x_in^ℓ‖ < 0.05)

| Model | Layers | Median-ratio range | Layers < 0.05 | Verdict |
|---|---|---|---|---|
| Kimi-K3-0.40B (real text, 768 pos) | 8 | 0.36 – 54.2 | **0/8 (0%)** | FAIL |
| **Ternary-Bonsai-27B** (Q2_0, 32 pos) | 64 | 0.15 – 13.1 | **0/64 (0%)** | FAIL |
| **Gemma-2-2B-it** (f16, 32 pos) — the paper's assumed vanilla-residual class | 26 | 0.28 – 1.65 | **0/26 (0%)** | FAIL |

The load-bearing evidence is the two PRODUCTION checkpoints (Bonsai-27B, Gemma-2-2B); K3-0.40B is the stack's test-architecture model (per AGENTS.md "dumb, use for test arch only") and only corroborates.

**Why it fails (the measured law):** real per-layer contributions scale as `ratio ≈ k/√L` with `k ≈ 1.5–3` — consistent with a residual stream whose norm grows ~√L while each layer adds an O(1)-norm block. The measured medians match this almost exactly (Bonsai L=64: ~0.25 measured vs 1.5/8 = 0.19 predicted; Gemma L=26: ~0.35 vs 1.5/5.1 = 0.29; K3 L=8: ~0.5–0.9 vs 1.5/2.8 = 0.53). To pass the paper's 0.05 bar a network needs `L ≥ (k/0.05)²` ≈ **900–3600 layers**. The "residual dominance ≪ 1" intuition comes from hundred-layer vision ResNets, not from 8–64-layer LLMs. K3's attn-res block-restart (block_size 4: the stream is zeroed at layers 0 and 4) makes its ratios worse (boundary layers 0/4 are its maxima: 54.2 / 1.60).

## T2 — Speculative replay sweep (K3, 5376 executions, real text)

θ-sweep (accept iff ratio < θ), averaged over delay layers 0–6:

| θ | accept-rate | top-1 preserved \| accept | mean KL(true‖spec) |
|---|---|---|---|
| 0.01–0.05 (paper's bar) | **0.0%** | — (nothing accepted) | — |
| 0.10 | 0.0% | 14.3% | 0.000 |
| 0.20 | 0.6% | 57.1% | 0.092 |
| 0.50 | 18.0% | 62.8% | 0.100 |

At the paper's threshold nothing qualifies (T1 said so). Even forcing acceptance at θ=0.5, **~37% of accepted executions flip the argmax** — stale-input execution is not quality-preserving at any threshold where acceptance actually happens.

**Persistent-hazard arm** (16-token greedy trajectories, stale-written KV/KDA state persists on accept; best delay layer 6):

- θ=0.05: 0 accepts / 256 rejects → trajectory 100% identical (the gate is honest — it refuses everything).
- θ=0.50: 54 accepts / 202 rejects → **95.7% token agreement, 2/16 trajectories diverge (positions 9, 12)** — the KV/KDA compounding corruption the paper flagged as its open hazard is real and measurable.

## T3 — Approach B (closed-form router-logit → δ predictor; paper target R² > 0.7)

Held-out R² (fit on 8 prompts, evaluated on 8 disjoint prompts, per delay layer):

| delay | router-logit R² | x_in-linear R² |
|---|---|---|
| 1 | 0.185 | −1.29 |
| 2 | 0.182 | −0.33 |
| 3 | 0.201 | −0.07 |
| 4 | **0.445** | **0.774** |
| 5 | 0.262 | 0.123 |
| 6 | 0.309 | 0.318 |

Best router predictor explains 44.5% of δ variance (paper needs >70%). Corrected replay at delay 4 (unconditional, 384 test execs): top-1 93.5% → 97.4%, KL 0.501 → 0.122 — **real signal, insufficient to make stale ≈ true**, and the corrected arm still accepts 0% at the paper's θ.

## Latency model (extraction #3) at measured accept-rate (0% @ paper's θ)

| Regime | paper (C+IO)/max(C,IO) | pair speedup (accept-adj) |
|---|---|---|
| M3 Max RAM-resident f32 (shared bus) | 1.53× | **1.00×** |
| Disk-resident Q4, NVMe (hideable IO) | 1.26× | 1.66× |
| Disk-resident ternary 1.58 b/w (hideable) | 1.76× | 1.40× |
| GPU H2D cold-thaw (hideable IO) | 1.20× | 1.71× |

Structural finding (encoded in `OverlapLatency` + unit tests): in the **shared-bus RAM-resident regime** (all our current deployments) speculation cannot create bandwidth — the pair model correctly degenerates to 1.00×. The wall-clock win exists ONLY in the hideable-IO regimes (disk-resident >RAM models, GPU H2D Cold-thaw), and — the model's non-obvious corollary — **in the IO-bound disk regime rejection is nearly latency-free** (the speculative compute rode the IO shadow; rollback is a cache-hit recompute) while **in compute-bound regimes rejection repays the full compute**. The quality gate failing (T1/T2) means the accept-rate-adjusted win never materializes at honest thresholds.

## Gates

- G1a determinism (double run, bit-identical captures): **PASS**
- G1b capture ≡ `kimi_k3_forward_token` (bit-identical): **PASS**
- G1c true-input replay ≡ true logits (bit-identical, all delays; unit test): **PASS**
- G1d snapshot restore roundtrip mid-sequence (bit-identical; unit test): **PASS**
- Pre-existing failure NOT ours: `transformer::tests::test_cluster_map_from_embeddings_collapses_identical_rows` fails at clean HEAD `b9fa4a07` (verified via stash; untouched by this change).

## Honest scope notes

- Bonsai/Gemma traces use a deterministic common-token walk (BOS + low-vocab region), not real text — the K3 arm covers the real-text case; the three classes agree, and the measured k/√L law is architecture-driven, not token-distribution-driven.
- 32 positions for the 27B/2B traces (CPU capture cost); K3 has 768 positions with the same picture.
- The x_in-linear predictor is underdetermined at this sample count (n≈384 train vs d=1024) — its held-out R² is a lower bound, not a ceiling; the ROUTER predictor (9 features, well-determined) is the paper's actual proposal and fails on its own.
- K3-0.40B per AGENTS.md is the test-arch model; the verdict rests on Bonsai-27B + Gemma-2-2B.

## Disposition (T4)

Record the negative in Research 508; close Issue 691. No wall-clock POC plan (its precondition — T1+T2 passing the paper's bar — failed). The landed, reusable pieces: the `stale_residual` analysis primitives (ratios/gate/KL/sweep/latency model, unit-tested), the K3 replay simulator (bit-exact machinery + persistent-hazard arm — reusable for any future stale/approximate-layer experiment), the SRTR trace format + riir-ai dumper, and the measured k/√L law — the answer to hand anyone who re-proposes residual-dominance layer speculation.

**Reopen triggers:** a ≥1000-layer checkpoint class, or an architecture explicitly trained for residual dominance (a δℓ-norm regularizer) — either would need T1 re-run first (one command).
