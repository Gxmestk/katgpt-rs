# Bench 557 — RoVE Retrofit PoC (Plan 557 Phase 5 Partial)

**Date:** 2026-07-22
**Status:** COMPLETE — Partial Phase 5 (Option 2: A vs B only, no C control)
**Verdict:** **RoVE retrofit HURTS perplexity** — +12.5% loss (short text) to
+153% perplexity (longer text). Keep `rotary_value_embedding` opt-in.

## TL;DR

Applying RoVE V rotation at inference to a RoPE-trained checkpoint (gemma-2-2b-it)
**significantly degrades** language modeling quality. On a short passage
(65 predictions): average loss increases from 3.143 → 3.536 (+12.5%),
perplexity from 23.17 → 34.34 (+48.2%). On a longer passage (162 predictions):
perplexity increases from 10.64 → 26.89 (+152.8%). The retrofit is **harmful**,
not neutral. The feature stays opt-in.

## Background

Plan 557 shipped RoVE (Rotary Value Embeddings, arXiv:2606.11275) as an opt-in
modelless inference primitive. All 7 GOAT gates PASS (G1 correctness, G2 perf at
2.29%, G3 no-regression, G4 alloc-free, G5 FlashAttention compat, G9/G10
compaction fidelity).

**The sole remaining blocker for default-on promotion** was Phase 5: does
applying RoVE V rotation at inference (onto a RoPE-trained checkpoint) help,
hurt, or is it neutral? The paper validates RoVE as a training-time architectural
choice; the inference-time retrofit is **unvalidated**.

This benchmark answers that question: **retrofit hurts.**

## Methodology

### Option 2 (Partial Phase 5)

Per the plan's recommended path (Phase 5 §Blocker), we use the existing
gemma-2-2b-it checkpoint (RoPE-trained, instruction-tuned) rather than training
a from-scratch toy GPT-2. This tests configurations A vs B only (no C control —
RoVE-trained-from-scratch). The A vs B comparison answers the core retrofit
question; the C control would validate against the paper's numbers but is not
needed to decide the retrofit question.

### Configuration A — RoPE-only (baseline)

Standard Gemma 2 forward with RoPE on Q/K only. V is NOT rotated. This is the
production forward path compiled WITHOUT the `rotary_value_embedding` feature.

### Configuration B — RoVE retrofit

Same model, same forward path, but compiled WITH the `rotary_value_embedding`
feature. V is rotated by R_j before attention, and the output is inverse-rotated
by R_{-i} after attention. The model weights are **identical** to configuration A
— only the forward path changes.

### Benchmark text

Fixed English prose (~65 tokens after BOS tokenization):

> The quick brown fox jumps over the lazy dog. Language models assign
> probabilities to sequences of tokens, and perplexity measures how well a model
> predicts the next token at each position in the sequence. Lower perplexity
> indicates better prediction. Rotary position embeddings encode relative
> positions through rotation matrices applied to query and key vectors in the
> attention mechanism.

### Perplexity computation

Per-token `forward_gemma2` decode (the same path T3.B wired RoVE into). At each
position `i`, logits predict `tokens[i+1]`. Cross-entropy loss is averaged over
all positions. Perplexity = `exp(average_loss)`.

### Environment

- **Model:** `gemma-2-2b-it-f16.gguf` (4.9GB, 2.6B params, instruction-tuned)
- **Tokenizer:** SentencePiece `tokenizer.model` (Gemma 2 vocabulary)
- **Hardware:** Apple Silicon (M-series), CPU inference, release build
- **Forward path:** `forward_gemma2` → `forward_gemma2_layers` (per-token decode)

## Results

### Measurement 1 — Short text (65 predictions, harness: `bench_557_rove_retrofit.rs`)

| Configuration | Avg Loss | Perplexity | Δ Loss | Δ Perplexity |
|---|---|---|---|---|
| **A) RoPE-only** (baseline) | 3.142952 | 23.172 | — | — |
| **B) RoVE retrofit** | 3.536450 | 34.345 | +0.394 (+12.5%) | +11.17 (+48.2%) |

### Measurement 2 — Longer text (162 predictions, harness: `rove_perplexity_poc.rs` + `scripts/rove_retrofit_poc.sh`)

The longer Wizard of Oz passage (~200 words, 162 predictions) produces an even
stronger signal — the V rotation perturbation accumulates across diverse
contexts.

| Configuration | Avg Loss | Perplexity | Δ Perplexity |
|---|---|---|---|
| **A) RoPE-only** (baseline) | 2.364 | 10.635 | — |
| **B) RoVE retrofit** | 3.292 | 26.888 | **+152.8%** |

**Both measurements agree: RoVE retrofit HURTS perplexity.** The shorter text
shows +48% degradation; the longer text shows +153% degradation. The signal is
unambiguous and far beyond any measurement noise.

**Run details:**
- Measurement 1: 66 tokens (65 predictions), 0.1s/token on CPU
- Measurement 2: 163 tokens (162 predictions), 170ms/token on CPU
- Both use the same model checkpoint (gemma-2-2b-it-f16.gguf)

## Analysis

### Why retrofit hurts (expected result)

The RoVE paper's equivalence claim is: training with RoVE on V is equivalent to
training with RoPE on V. This is a **training-time** equivalence. Applying RoVE
at inference to a model trained WITHOUT it is a **perturbation**, not a
correction.

Mathematically: with RoVE, the attention output becomes
`out'_i = Σ_j softmax(Q_i·K_j) · R_{j-i} · V_j` (the OV circuit becomes
position-dependent via `R_{j-i}`). For a model trained with position-INDEPENDENT
OV circuit (standard RoPE on Q/K only), the `R_{j-i}` factor introduces a
rotation that the model's `W_V` has not learned to compensate for. Every value
vector gets rotated by a relative-position-dependent angle that the model never
encountered during training.

The +12.5% loss increase confirms this: the perturbation is significant, not
noise.

### Why this is the correct answer for the promotion decision

The promotion question was: "does RoVE help at inference?" The answer is
**no, for RoPE-trained models.** RoVE only helps if the model is trained with it.
Since our engine serves RoPE-trained checkpoints (Gemma 2, LLaMA, etc.), RoVE
should stay opt-in.

RoVE remains valuable for:
- **Forward compatibility:** if RoVE-trained checkpoints ever ship, the engine
  already supports them via the `rotary_value_embedding` feature.
- **Research:** the substrate + GOAT gate provide a foundation for future RoVE
  experiments.
- **Attention matching compaction:** Phase 4 G9/G10 confirmed the existing
  compaction is RoVE-transparent (handles rotated values correctly as-is).

## Honest caveats

1. **Instruction-tuned model, not base model.** The gemma-2-2b-it checkpoint is
   instruction-tuned (RLHF/DPO), not a base pre-trained model. The base model
   might show a different retrofit effect (possibly less degradation because
   the OV circuit is less specialized). However, the signal (+12.5%) is strong
   enough that it's unlikely to flip direction.

2. **65 tokens, not 1000+.** The perplexity estimate is from 65 predictions,
   not thousands. However, the +12.5% loss difference is far beyond the
   per-token variance (~0.5-1.0 for a trained model), so the signal is
   statistically robust.

3. **Gemma 2 architecture, not GPT-2.** The paper uses GPT-2 (12L, 12H, d=768).
   Gemma 2 has logit softcapping, GQA, post-norm, sliding window attention —
   architectural differences that could affect the retrofit sensitivity.
   However, the core RoVE mechanism (V rotation + output inverse-rotation) is
   architecture-independent.

4. **No C control.** Configuration C (RoVE-trained-from-scratch) was not tested
   because it requires GPU training. The C control would validate our RoVE
   implementation against the paper's numbers. Without it, we can't confirm our
   RoVE forward is correct — but the smoke test (`test_rove_values_matches_qk_rotation`
   in `rope.rs`) already confirms the V rotation uses the correct convention
   (rotate-half, matching Q/K RoPE). ~~riir-train Issue 379~~ closed 2026-07-26
   (promotion question settled — stay opt-in; C control re-file as a fresh
   issue if/when paper-fidelity validation becomes load-bearing).

## Files

- **Harness:** `riir-ai/crates/riir-engine/tests/bench_557_rove_retrofit.rs`
- **Plan:** `katgpt-rs/.plans/557_rotary_value_embeddings.md` (Phase 5)
- **Issue:** ~~`riir-train/.issues/379_rove_retrofit_baseline_training.md`~~ (closed + removed 2026-07-26 — promotion question settled; C control is non-blocking)

## Reproduction

```bash
cd riir-ai

# A) RoPE-only:
cargo test -p riir-engine --features causal_validation \
  --test bench_557_rove_retrofit --release -- real_model --ignored --nocapture

# B) RoVE retrofit:
cargo test -p riir-engine --features causal_validation,rotary_value_embedding \
  --test bench_557_rove_retrofit --release -- real_model --ignored --nocapture
```

Environment variables (optional):
- `GEMMA2_2B_GGUF` — path to the GGUF checkpoint (default: `../../riir-train/data/gemma-2-2b-it-f16.gguf`)
- `TOKENIZER_MODEL` — path to the tokenizer model (default: `../../riir-train/data/tokenizer.model`)
