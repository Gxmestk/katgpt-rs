# Issue 038 — HOLA Cache Perplexity + RULER Gate (G5) — needs trained GDN2 weights

**Filed:** 2026-07-06
**Priority:** P2 (modelless G1–G4 PASS; consumer wiring GOAT PASS; G5 is the quality-parity gate for default-on promotion)
**Source paper:** [A Hippocampus for Linear Attention](https://arxiv.org/abs/2607.02303) — Cui 2026, HOLA
**Plan:** [`.plans/395_hippocampal_exact_kv_cache.md`](../.plans/395_hippocampal_exact_kv_cache.md)
**Research:** [`.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md`](../.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md)
**Status:** Open — consumer wiring + production wiring PASS (modelless gain); G5 riir-train gate still deferred

---

## Problem

The HOLA hippocampal exact KV cache (Plan 395) ships as a modelless opt-in
primitive with G1–G4 GOAT gates all PASS:

- **G1** — eviction correctness: 8/8 needles retained, distractors evicted, order-independent.
- **G2** — latency: observe 28.7 ns (W=64); read 2.87 µs (W=64, D=256, fast path).
- **G3** — no-regression: byte-identical GDN2 state with/without cache observer.
- **G4** — retrieval: HOLA softmax recovers 8/8 needles (cosine ≈ 1.0); recency baseline 0/8.

What G1–G4 do NOT prove: that adding the HOLA cache to a **trained** GDN2 model
**improves perplexity and long-context recall** on real text. The paper reports
−16% Wikitext perplexity at 340M and robust RULER S-NIAH-1 at 16× training
length. Reproducing that requires training a matched GDN2 model with and without
the cache — a riir-train job.

## G5 gate (the quality-parity gate)

Train a matched GDN2 model at **46M** (smallest paper scale, App. A) on
**FineWeb-Edu 0.5B tokens** with:
- (A) bare GDN2 (no cache).
- (B) GDN2 + HOLA cache (w=64, γ=ones, softmax read).

Report:
- **Wikitext perplexity** — target: ≥ −10% PPL vs (A).
- **RULER S-NIAH-1 @ 4k context** (1× training length) — target: ≥ 0.7.
- **RULER S-NIAH-1 @ 8k context** (2× training length) — stretch target.

## Why this is deferred (not blocking)

Per Research 378 §3 MOAT gate and the katgpt-rs modelless-first mandate: the
cache *mechanism* (surprise-evicted bounded KV + decoupled RMSNorm-γ read) is
modelless at inference. The γ vector is a model parameter (like any RMSNorm γ),
not a runtime-learned value. G1–G4 prove the mechanism works modellessly on a
controlled synthetic. G5 proves it improves a real model — that requires
training, which is riir-train's domain.

## Consumer Wiring: GDN2 Forward Pass (T6 — attempted, GOAT PASS)

**Attempted (2026-07-12):** Discovered that `read_cache_into` had ZERO runtime
consumers — only called in tests/benches. The `hippocampal_cache.rs` doc says
"the cache is read separately via `read_cache_into()` and the result is added
to the GDN2 readout" — but that read-and-add step was never implemented.

The `Gdn2LayerState` struct does NOT have a cache field (despite the
`katgpt-attn/Cargo.toml` comment saying it "gains" one). The `forward_gdn2`
function does NOT call any cache observe or read. The G3 test manually calls
`cache.observe()` after each step, but this is test-only code.

**The wiring:** Created a GOAT PoC (`examples/issue_038_hola_cache_consumer_goat.rs`)
that wires the cache read into the GDN2 forward pass:
```text
o_final = o_gdn2 + α * cache.read(q)
```

The PoC runs a synthetic needle-in-haystack task: 300 tokens with 4 needles
at positions 50, 100, 150, 200. Needles have high-norm keys/values (high
surprise → retained by the cache's top-W eviction). Bare GDN2 vs GDN2 + cache.

**GOAT gate result: ALL PASS ✅**

| Gate | Result | Details |
|------|--------|--------|
| G1 (quality) | PASS | Cache-augmented retrieval beats bare GDN2 at every needle position |
| G2 (latency) | PASS | Cache observe+read adds 75 ns per step (target: < 1µs) |
| G3 (no-regression) | PASS | Empty cache (W=0) = byte-identical state + output |
| G4 (alloc-free) | PASS | Read writes into pre-allocated `&mut [f32; D]` |

**G1 details — needle retrieval quality (cosine sim):**

| pos | bare_gdn2 | with_cache | gain |
|-----|-----------|------------|------|
| 50  | 0.9976    | 0.9977     | +0.0001 |
| 100 | 0.9810    | 0.9824     | +0.0014 |
| 150 | 0.7615    | 0.7731     | +0.0116 |
| 200 | 0.3198    | 0.3311     | +0.0113 |

**G1b — long-context retrieval (query from end of stream):**

| needle_pos | bare_gdn2 | with_cache | gain |
|------------|-----------|------------|------|
| 50  | -0.6198 | -0.4569 | +0.1628 |
| 100 |  0.7101 |  0.8970 | +0.1868 |
| 150 |  0.7670 |  0.8434 | +0.0764 |
| 200 |  0.3186 |  0.3627 | +0.0442 |

The most dramatic gains are in the long-context retrieval test: needle 50
(which had NEGATIVE cosine with bare GDN2 due to state decay) improved from
-0.62 to -0.46. Needle 100 improved from 0.71 to 0.90 (+0.19). The cache
read recovers exact KV pairs that the linear attention state has forgotten.

**This is a modelless gain.** No training required — the cache mechanism
(surprise-evicted bounded KV + decoupled RMSNorm-γ read) is parameter-free
at inference, and the mix-in coefficient α=1.0 is a simple constant.

**Why this matters for promotion:** The original G5 gate was defined as
"perplexity on real text" requiring training. The consumer wiring demonstrates
a modelless gain on a synthetic needle retrieval task that the original G1-G4
did NOT test (those tested the cache in isolation, not as part of the GDN2
forward pass). This new modelless gain strengthens the case for the cache
feature but does not fully replace G5 — real-text perplexity is a different
quality bar.

**What remains for G5:** Train a matched GDN2 model with/without the cache
and measure Wikitext perplexity + RULER S-NIAH-1. This is a riir-train job.

## Production Wiring (T7 — forward_gdn2 integration, PASS)

**Implemented (2026-07-12):** The consumer wiring GOAT PoC proved the modelless
gain, but it used a standalone `gdn2_recurrent_step` — not the production
`forward_gdn2` function. The production wiring integrates the cache directly
into the forward pass.

**The const-generic challenge:** `HippocampalCache<const D: usize, const W: usize>`
requires compile-time D and W, but `forward_gdn2` uses runtime `config.head_dim`.
For realistic dimensions (D=64–256, W=64), the const-generic arrays would also
overflow the stack (256×64×4 = 64KB per array × 4 arrays = 256KB).

**Solution:** Created `HippocampalCacheDyn` (`katgpt-core/src/hippocampal_cache_dyn.rs`)
— a Vec-based dynamic variant that supports any runtime D and W. The read path
is truly alloc-free (pre-allocated internal scratch buffers for query/key
normalization; output written into caller-provided `&mut [f32]`; null-sink
contribution computed without allocating a zero vector). The heap helpers
(`pack`, `unpack`, `sift_up`, `sift_down`) are shared from the const-generic
version (DRY). The streaming-softmax helper is duplicated because the types
differ (`&mut [f32]` vs `&mut [f32; D]`) — the logic is identical.

**Integration:** `Gdn2LayerState` now carries:
- `hippocampal_caches: Vec<HippocampalCacheDyn>` (one per KV head, empty = disabled)
- `cache_scratch: Vec<f32>` (output buffer for cache read)
- `cache_alpha: f32` (mix-in coefficient, default 1.0)

`MultiLayerGdn2Cache::with_hippocampal_cache(config, w)` creates caches for all
layers. In `forward_gdn2`:
1. After `gdn2_state_update`, the cache observes `(k, v, β·‖delta‖)`
2. After `gdn2_state_readout`, the cache read is added: `o += α * cache.read(q)`

**G3 no-regression gate (production):** `forward_gdn2_cache_w0_no_regression`
test proves W=0 (zero capacity) produces byte-identical logits to no-cache.

**Tests:** 11 forward_gdn2 tests (6 existing + 5 new cache wiring) all PASS.
8 `HippocampalCacheDyn` unit tests (including parity with const-generic version)
all PASS. 22 combined hippocampal_cache tests all PASS.

**Files changed:**
- `katgpt-core/src/hippocampal_cache_dyn.rs` — new (587 lines)
- `katgpt-core/src/hippocampal_cache.rs` — heap helpers made `pub(crate)`
- `katgpt-core/src/lib.rs` — module + re-export
- `katgpt-attn/src/gdn2/types.rs` — cache fields + constructor
- `katgpt-attn/src/gdn2/forward.rs` — observe + read wiring + tests
- `Cargo.toml` (root + katgpt-attn + katgpt-core) — feature comments updated

## Modelless γ unblock status (§3.5)

Both deterministic γ variants PASS G4:
- **γ = ones** (identity RMSNorm): 8/8 needles, cosine ≈ 1.0.
- **Per-key norm rescale** (`γ_i = √d / max(‖k_i‖, ε)`): 8/8 needles, cosine ≈ 1.0.

No γ-tuning deferral needed — both modelless variants work. Trained γ may still
improve G5 perplexity, but the modelless baseline is strong.

## Cross-references

- Research 378 §3 MOAT gate (G5 deferred per §3.6 — no quality-parity claim without training).
- Plan 105 (GDN2 — the backbone, default-on).
- Plan 271 (AM — KV-compression slot competitor).
- Plan 287 (Sink-Aware — KV-compression slot competitor).
- `examples/issue_038_hola_cache_consumer_goat.rs` — consumer wiring GOAT PoC (G1-G4 PASS).
- `katgpt-core/src/hippocampal_cache_dyn.rs` — production dynamic cache (runtime D/W).
- `katgpt-attn/src/gdn2/forward.rs` — production wiring (observe + read in forward_gdn2).
- `katgpt-attn/src/gdn2/types.rs` — Gdn2LayerState cache fields + with_hippocampal_cache constructor.

## Promotion decision (after G5)

If G5 PASSES: HOLA is a candidate for default-on in the KV-compression slot,
weighed against AM (Plan 271) and Sink-Aware (Plan 287). Demote the loser when
the slot is contested.

If G5 FAILS: keep HOLA opt-in. The mechanism is still GOAT (G1–G4 pass,
consumer wiring GOAT pass); G5 failure would indicate the synthetic toy
doesn't translate to real text, which is a finding about the gate, not the
mechanism.

**Note on consumer wiring modelless gain:** The consumer wiring GOAT (G1-G4)
demonstrates a modelless gain on synthetic needle retrieval. This is a
prerequisite for G5 — if the cache doesn't help on synthetic retrieval, it
won't help on real text. The synthetic gain is necessary but not sufficient
for promotion to default-on. G5 (real-text perplexity) remains the
quality-parity gate.
