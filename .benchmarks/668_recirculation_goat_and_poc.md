# Benchmark 668 — Recirculation operator GOAT + defend-wrong PoC (Issue 673)

**Date:** 2026-08-19
**Feature:** `recirculation` (opt-in — **the PoC quality axis FAILED on our substrate; default stays OFF**)
**Source:** Research 492 / arXiv:2608.17981 (Recirculation, Mozer et al., Google DeepMind).
**Modules:** `crates/katgpt-core/src/recirculation.rs` + `tests/recirculation_alloc_check.rs` (katgpt-rs); `crates/riir-poc/tests/recirculation_poc.rs` (riir-ai, the Phase 2 PoC); riir-engine enablers (`PostLayerHook` pub + `forward_one_token_hooked`).

## Phase 1 — operator GOAT gates

| Gate | Evidence | Verdict |
|---|---|---|
| **G1 determinism** | Fixed α + fixed buffers ⇒ **bit-identical repeat** (`deterministic_bit_identical_repeat`, 50 composed steps × D=128); the `l2_norm` 8-way unroll is a compile-time-constant association order (deterministic, latency-chained broken). | **PASS** |
| **G2 overhead** | `g2_step_mixture_under_1us_at_d2048` (release): **553.7 ns/step at D=2048** — 1.8× under the ≤1µs gate. (First implementation measured 3068.6 ns — a single-accumulator serial dependency chain; the fixed 8-way unroll cut it 5.5× with determinism preserved.) | **PASS** |
| **G3 no-regression** | Default lib suite 1897/0/6 exact baseline (module cfg'd away); `--all-features` clean; clippy 0 in default + feature + tests states. | **PASS** |
| **G4 alloc-free** | `recirculation_alloc_check` (CountingAllocator, release): capture+mix cross-step loop, 50,000 steps at D=2048 = **0 allocations** (the `RecircBuffer` is a one-time fixed-D allocation). | **PASS** |

Operator unit tests: convex+norm-match boundedness (triangle-inequality class), norm-match identity (`x/x == 1.0` exactly), ramp schedule + step-0 bit-identical no-op, β=1 non-convex exact-fp recurrence, cross-step composition boundedness over 200 synthetic steps, paper anchor pairs {11,4}/{18,9}/{35,16} round-trip at 26/34/48 layers.

## Phase 2 — defend-wrong PoC on gemma-2-2b-it (riir-poc)

**Setup:** `Gemma2PatchedForward::from_gguf(gemma-2-2b-it-f16.gguf)` + SentencePiece; serial token-by-token windows (the honest recirculation prefill mode); paper layer pair for 26 layers src=11/dst=4 (`RecircPair::for_depth`); ramp 10; NLL over each position's next token. Two local text registers as datasets (honestly labeled): **A** = katgpt-rs `.research/*.md` (arXiv-style prose), **B** = riir-ai `.docs/**/*.md` (long-form technical prose — PG19-substitute; PG19 not on disk).

**Run 1** (2×96 tokens/arm/dataset; dst_hook=4):

| dataset | baseline | α=0.07 | α=0.10 | α=0.15 | overwrite | temp-1.2 |
|---|---|---|---|---|---|---|
| A | 91.32 | 99.66 (−9.1%) | 104.80 (−14.8%) | 114.28 (−25.1%) | 2.2e8 (**clobbered**) | 68.38 (+25.1%) |
| B | 311.20 | 329.88 (−6.0%) | 342.66 (−10.1%) | 374.97 (−20.5%) | 5.4e8 (**clobbered**) | 197.16 (+36.7%) |

**Run 2 — injection-point robustness** (3×160 tokens/arm/dataset; `RECIRC_POC_DST_SHIFT=-1` → dst_hook=3, the pre-dst input-injection equivalent):

| dataset | baseline | α=0.07 | α=0.10 | α=0.15 | overwrite | temp-1.2 |
|---|---|---|---|---|---|---|
| A | 339.03 | 361.97 (−6.8%) | 380.10 (−12.1%) | 432.03 (−27.4%) | 4.4e9 (**clobbered**) | 233.87 (+31.0%) |
| B | 342.44 | 370.27 (−8.1%) | 389.42 (−13.7%) | 439.43 (−28.3%) | 8.5e9 (**clobbered**) | 233.11 (+31.9%) |

### Verdict

- **Gate (b) FAILS: ppl reduction > 0 on ≥2 datasets — recirculation INCREASES ppl on our substrate**, dose-dependently in α, at BOTH injection points, on BOTH registers (12/12 recirc cells harmful). The paper's Gemma3-family training-free gains do **not** transfer to gemma-2-2b-**it** under this harness.
- **Safety check PASSES: recirc is strictly safer than the R417 overwrite** at equal pairs (mixture −7…−28% vs overwrite −10⁸% — the clobbering catastrophe, reproducing Plan 431's failure mode on a real model, the strongest possible contrast arm). The operator's *relative* semantics are exactly as designed.
- **temp-1.2 control: ppl IMPROVES** (+25…+37%) — same direction as the paper's temp control (−8.5% alone there), so the harness measures real distribution effects; recirculation's harm is not a temperature artifact.

### Honest caveats (recorded, binding for any reopen)

1. **Session-scaled counts** (192–480 tokens/arm; paper-scale is ~500×1024): the direction is consistent and monotone in α across 12 cells, but the magnitudes are small-sample.
2. **Model register:** gemma-2-2b-**it** (instruction-tuned), not a base Gemma2/Gemma3; the paper validated base models. IT-tuned residual-stream statistics may differ.
3. **Data register:** local markdown corpora (code fences, tables), not wikitext/PG19 prose.
4. **Reopen conditions:** paper-scale re-run (`RECIRC_POC_WINDOWS=50 RECIRC_POC_WINDOW_TOKENS=512`) + a base-model check (the 4090 + a base Gemma-family GGUF) before citing this negative as final at scale.

## Phase decisions (per the issue's gates)

- **Phase 3 (decode-stack integration): DOES NOT PROCEED** — gated on Phase 2 PASS; it failed.
- **Phase 4 promotion: default stays OFF** — G1–G4 pass at the operator level, but the quality axis failed on our substrate. The operator ships opt-in as substrate (the mixture semantics are measurably the safe member of the relocation family).
- **Phase 5 (belief-recirculation guide): DOES NOT TRIGGER** (quality axis unproven).
- **Cost accounting (recorded per the issue):** decode = 2 stack instances/step (~2× FLOPs serial, 2× KV footprint); prefill serial. The PoC measured ~8.7 tok/s serial prefill on M3 CPU for the 2B f16 — the KV doubling was never reached (integration not warranted).

## Run

```bash
# Operator GOAT
cargo test -p katgpt-core --features recirculation --lib recirculation::
cargo test --release -p katgpt-core --features recirculation --lib g2_step_mixture -- --nocapture
cargo test -p katgpt-core --features recirculation --test recirculation_alloc_check --release
# PoC (real model; ~5-13 min per config on M3)
GEMMA2_2B_GGUF=... TOKENIZER_MODEL=... cargo test -p riir-poc --test recirculation_poc --release -- --nocapture --ignored
```
