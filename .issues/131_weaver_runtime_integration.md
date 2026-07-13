# Issue 131 — Weaver Runtime Integration (inference-only SpeculativeGenerator adapter)

> **Spawned from:** `riir-train/.plans/314_weaver_adapter_training.md` Phase 6
>   ("Open as a katgpt-rs issue when Phase 6 passes" — Phase 6 passed 2026-07-10)
> **Date:** 2026-07-10
> **Status:** UNBLOCKED-PATH-IDENTIFIED (2026-07-13 update) — the original "no
>   base model weights" blocker (below) was REFUTED: 4 verifier GGUFs exist in
>   `riir-train/data/`, and DFlash LoRA training methodology ships as Plan 143.
>   The real remaining work is RUNNING the pipeline (precompute + training),
>   not sourcing methodology or weights. See "§ Unblock path (2026-07-13
>   revision)" below.
> **Priority:** Medium (was Low — upgraded because the blocker was refuted)
> **Feature gate (proposed):** `weaver_runtime` (opt-in)

## TL;DR

riir-train Plan 314 shipped the **Weaver adapter** — a 56.7M-param
autoregressive residual transformer that corrects the DFlash drafter's
top-K=512 marginal distributions toward the verifier's distribution. Training
loop, data pipeline, backward pass, and synthetic GOAT gate are all DONE in
riir-train. The paper reports **+77% mean acceptance length over chain DFlash,
+32% over DDTree**.

This issue tracks the **katgpt-rs runtime half**: an inference-only adapter
that loads trained Weaver weights (via freeze/thaw) and applies the residual
correction to DFlash draft logits at decode time. **Blocked until trained
weights exist** — no code work can start until the 300k completion training
run produces a `weaver_v1.safetensors` checkpoint with non-trivial gain.

### Why katgpt-rs (not riir-ai)

Per Research 402 §4: katgpt-rs ships the **top-K constrained projection +
residual-add** as an inference-only adapter (no training). riir-ai wires it
into its speculative decode pipeline (`forward_qwen_deltanet` path) — that
is a riir-ai runtime task, separate from this issue.

### The vocabulary-projection efficiency trick

Weaver **never projects to the full vocabulary**. It reads only K=512 rows
of the vocabulary projection matrix (selected as the DFlash top-K tokens),
adds its residual logits to the DFlash output logits, and normalizes over
the candidate set. This avoids the memory-bandwidth bottleneck of a
standard autoregressive drafter while restoring the conditional coupling
that independently-predicted marginals destroy.

## Blocker chain (why this is blocked)

```
Trained DFlash drafter weights ──────────────────────────┐
Trained verifier (base LLM) weights ─────────────────────┤
                                                         ▼
                              300k-completion precompute (riir-train T4.1 real)
                                         │
                                         ▼
                              Weaver training run (riir-train Phase 5)
                                         │
                                         ▼
                              weaver_v1.safetensors (BLAKE3-checked)
                                         │
                                         ▼
                              THIS ISSUE (katgpt-rs runtime integration)
```

**~~No trained DFlash/verifier transformer weights exist in any of the 5 repos~~**
**REFUTED 2026-07-13.** The original audit (above) was wrong or out-of-date.
The actual state of `riir-train/data/` (verified 2026-07-13):

| File | Size | Role |
|---|---|---|
| `gemma-2-2b-it-f16.gguf` | 5.2 GB | Verifier candidate (Gemma 2 2B IT, f16) |
| `MiniCPM5-1B-F16.gguf` | 2.1 GB | Verifier candidate (MiniCPM5 1B, f16) |
| `llama-3.2-3b-instruct-q8_0.gguf` | 3.4 GB | Verifier candidate (Llama 3.2 3B, q8_0, added Jul 13) |
| `qwen2.5-3b-instruct-q8_0.gguf` | 3.6 GB | Verifier candidate (Qwen 2.5 3B, q8_0, added Jul 13) |

**4 verifier candidates exist.** The DFlash drafter training methodology
ships as **Plan 143** (`dflash_training` feature, COMPLETE, all 8 tasks
T1-T8 done). Proposal 018 §3.2 confirms: "DFlash ✅ Plan 143 — No gap".

The real remaining work is **running the pipeline**, not sourcing weights
or methodology. See the revised unblock path below.

The Weaver adapter's value proposition (distilling the verifier's top-K
distribution into the drafter's marginal) requires a real verifier→drafter
quality gap to close. At the repo quintet's current scale (game-domain
micro-GPT), this gap may not exist. The paper's gain is measured on
production LLMs (Nemotron V2 verifier, DFlash/DDTree drafters).

## What the integration looks like (when unblocked)

### Architecture (from riir-train Plan 314 / Research 402)

```
                    ┌──────────────────────────────────────┐
                    │   Frozen DFlash / DDTree drafter      │
                    └──────────────┬───────────────────────┘
                                   │ top-K=512 token ids + draft logits
                                   ▼
                    ┌──────────────────────────────────────┐
                    │   Weaver adapter (THIS ISSUE)         │
                    │                                       │
                    │   uᵢ = W_c · RMSNorm(hᵢ_dflash) + pᵢ  │  # conditioning
                    │   causal self-attn over draft path    │
                    │   SwiGLU MLP                          │
                    │   ℓ_weaver = top-K projection         │
                    │                                       │
                    │   ℓ_final[topk] = ℓ_dflash[topk]      │  # residual add
                    │                     + ℓ_weaver[topk]  │
                    │   renormalize over K candidates       │
                    └──────────────┬───────────────────────┘
                                   │ corrected top-K marginals
                                   ▼
                    ┌──────────────────────────────────────┐
                    │   Verifier (base LLM)                 │
                    └──────────────────────────────────────┘
```

### Integration point in katgpt-rs

The Weaver correction slots into the **DFlash predict pipeline** as a
post-draft logit corrector, between `dflash_predict_with` producing
`DraftResult.marginals` and the verifier's acceptance check.

**Two integration options (decide when unblocked):**

1. **Marginal corrector (lighter):** Modify `DraftResult.marginals` in-place
   after `dflash_predict_with` returns. The Weaver forward reads the DFlash
   hidden states + top-K token ids, produces the residual, and the marginals
   are renormalized. This is a post-processing step — no DFlash internal
   changes.

2. **Logit corrector (heavier, matches paper):** Intercept the DFlash draft
   logits before marginalization, apply the Weaver residual, then
   renormalize. This requires exposing the raw draft logits from the DFlash
   forward context.

**Recommendation:** option 1 (marginal corrector) for the initial integration —
it's non-invasive and the marginal is what the verifier ultimately consumes.

### Proposed task breakdown (when unblocked)

- **T1: Checkpoint loader** — load `weaver_v1.safetensors` into a runtime
  `WeaverWeights` struct. The format is defined in
  `riir-train-engine/src/weaver_train.rs` (`weights_to_safetensors_bytes`).
  katgpt-rs needs a read-only mirror (duplicate the struct for now — small,
  ~12 fields; extract to a shared crate later if more consumers appear).
- **T2: Top-K constrained forward pass** — (1) gather K rows from the vocab
  embedding using the DFlash top-K token ids; (2) run the single-layer Weaver
  transformer (conditioning → causal attention → SwiGLU MLP → top-K gather
  projection); (3) add the Weaver residual logits to the DFlash output logits;
  (4) renormalize over K candidates (sigmoid, not softmax — per global rule).
  Mirrors `riir-train-engine/src/weaver.rs::weaver_forward` but inference-only
  (no backward, no autograd cache).
- **T3: SpeculativeGenerator adapter** — wrap T1+T2 as a
  `SpeculativeGenerator` adapter that receives DFlash/DDTree top-K marginals,
  applies the Weaver correction, and returns corrected logits to the
  tree-builder / verifier.
- **T4: Feature gate** — ship behind `weaver_runtime` (opt-in). The adapter
  loads weights lazily — if no checkpoint is found, it falls back to the
  uncorrected drafter (zero overhead).

### Weight loading (freeze/thaw)

Trained weights ship as `weaver_v1.safetensors` with a BLAKE3 manifest
(riir-train Plan 314 T5.2). The runtime loads via the freeze/thaw envelope
(consistent with riir-neuron-db's `MerkleFrozenEnvelope`):

- `WeaverWeights` struct (from riir-train `weaver.rs`) is `#[repr(C)]` Pod.
- Load: `load_checkpoint(path) -> Result<WeaverWeights, Blake3Mismatch>`.
- Feature-gated as `weaver_runtime` (opt-in) — the weights are a trained
  artifact, not modelless-promotable.

### Feature gate

`weaver_runtime = []` (opt-in, default-OFF). The feature:
- Adds the `WeaverCorrector` adapter struct.
- Gates the DFlash post-processing hook.
- Depends on `katgpt-core` types (`DraftResult`) but NOT on riir-train
  (weights load via safetensors, no training dependency at runtime).

**Promotion:** stays opt-in permanently (trained-weight dependency, not
modelless). Unlike modelless primitives, a trained adapter cannot be
default-on because it requires a checkpoint file to exist on disk.

## Acceptance criteria (when unblocked)

- [ ] `WeaverCorrector` struct: holds `WeaverWeights`, implements the forward
      pass (conditioning → causal attn → SwiGLU → top-K projection → residual
      add → renormalize).
- [ ] Load path: `WeaverCorrector::from_checkpoint(path)` reads
      `weaver_v1.safetensors`, verifies BLAKE3, returns the corrector.
- [ ] Integration hook: DFlash `DraftResult.marginals` are corrected when the
      `weaver_runtime` feature is on and a corrector is registered.
- [ ] G1 (correctness): corrected marginals sum to 1.0 over top-K, no NaN/Inf.
- [ ] G2 (gain): mean acceptance length(corrected) > mean acceptance length(raw)
      on the real verifier (not synthetic). This is the real acceptance
      benchmark — the synthetic +134% from riir-train Phase 6 is not
      transferable without real weights.
- [ ] G3 (no-regression): when `weaver_runtime` is OFF, DFlash behavior is
      bit-identical to the current default (zero-cost abstraction).
- [ ] G4 (latency): Weaver forward adds < X µs per draft step (TBD — the
      single-layer model is lightweight, but the top-K=512 projection reads
      4 MiB of weights; needs measurement).

## Why this is NOT modelless-promotable

The Weaver adapter is a **trained** artifact by construction:
- Its 56.7M parameters encode the verifier→drafter distillation.
- Zero-init weights produce zero residual (no-op) — the value IS the trained
  weights.
- No freeze/thaw, raw/lora hot-swap, or latent projection can substitute —
  the correction is a learned nonlinear mapping from drafter context to
  logit residuals.

This is a legitimate riir-train dependency. The modelless mandate
(AGENTS.md §3.5) does not apply — the modelless path was never the question
for Weaver (unlike Research 400 / Issue 428, where the modelless path was
prematurely declared exhausted).

## Unblock path (2026-07-13 revision)

The original blocker chain (above) assumed weights + methodology were missing.
Both assumptions are now refuted. The revised unblock path:

### What EXISTS (no work needed)

1. **Verifier weights** — 4 GGUFs in `riir-train/data/` (gemma-2-2b-it,
   MiniCPM5-1B, llama-3.2-3b, qwen2.5-3b). Pick one as the frozen verifier.
2. **DFlash drafter training methodology** — Plan 143 (`dflash_training`,
   COMPLETE). Trains LoRA adapters for the DFlash bidirectional draft model
   conditioned on target hidden states.
3. **Weaver training methodology** — riir-train Plan 314 (`weaver_adapter_training`,
   COMPLETE incl. synthetic GOAT gate). LK loss + Muon + top-K=512 vocabulary
   projection constraint.
4. **Weaver checkpoint format** — `weaver_v1.safetensors` writer in
   `riir-train-engine/src/weaver_train.rs` (`weights_to_safetensors_bytes`).
5. **katgpt-rs DFlash inference** — `dflash_predict_with` in
   `katgpt-speculative/src/dflash.rs` (the integration point).

### What NEEDS DOING (the actual work)

| Step | Task | Owner repo | Dependency |
|---|---|---|---|
| **S1** | Pick verifier (recommend MiniCPM5-1B — smallest, fastest training) | riir-train | none |
| **S2** | Warm-start DFlash base from verifier weights (standard EAGLE/DFlash practice — initialize draft Wq/Wk/Wv/MLP from verifier) | riir-train | S1 |
| **S3** | Run Plan 143 (`dflash_training`) to train DFlash LoRA adapter on the warm-started base | riir-train | S2 |
| **S4** | Produce frozen DFlash drafter checkpoint (BLAKE3-hashed) | riir-train | S3 |
| **S5** | Run 300k-completion precompute (Plan 314 T4.1 real) — generate verifier logits + DFlash lookaheads on 300k completions | riir-train | S1 + S4 |
| **S6** | Run Weaver training (Plan 314 Phase 5) on the precomputed data | riir-train | S5 |
| **S7** | Produce `weaver_v1.safetensors` checkpoint (BLAKE3-hashed) | riir-train | S6 |
| **S8** | **THIS ISSUE** — katgpt-rs runtime integration (T1-T4 below) | katgpt-rs | S7 |

### Why the original audit was wrong

The 2026-07-10 audit found "only LoRA artifacts + bandit states" and concluded
"no base model weights". This missed the 4 GGUF files in `riir-train/data/`
(which were already present — gemma-2-2b-it and MiniCPM5-1B since May 2026).
The audit likely searched for `.safetensors` or weight-tensor files, not GGUF
quantized model files. GGUF is the canonical runtime format for these models
(loadable via `gguf_loader.rs` in riir-engine).

### Risk: scale mismatch

The paper's gain (+77% MAL, +32% over DDTree) is measured on Qwen3.6-27B
(production scale). Our verifier candidates are 1B-3B (game-domain scale).
The verifier→drafter quality gap may be smaller at this scale, reducing
Weaver's correction magnitude. **Mitigation:** the synthetic GOAT gate
(riir-train Plan 314 Phase 6, +134%) validates the methodology; the real-data
gate (S8 G2) confirms the gain transfers. If it doesn't transfer, the
runtime integration is still correct (just lower gain) — no regression.

### Revised priority

**Medium** (was Low). The blocker is refuted; the path is clear. The work is
GPU-training-bound (S3, S5, S6 are multi-hour GPU jobs), not code-blocked.
Once S7 produces a checkpoint, S8 (this issue) is ~2 days of code work
(T1-T4 in "Proposed task breakdown" below).

---

## Non-goals

- **Training** — stays in riir-train. katgpt-rs is inference-only.
- **GPU fused kernel** — CPU-first for correctness. riir-gpu task if
  throughput requires it. The top-K projection is `O(K·d)` = 62.5× smaller
  than a full-vocab projection (K=512 vs 32k vocab), so bandwidth is not
  the bottleneck — the G3 no-regression gate should hold by construction.
- **Traversal verification** (paper ref [10]) — separate algorithm, not
  Weaver.

## Cross-references

- **riir-train Plan 314** — the training plan (DONE, synthetic validation)
- **riir-train Research 402** — the paper distillation (Weaver = residual
  over top-K marginals)
- **katgpt-core `DraftResult`** (`crates/katgpt-core/src/speculative/types.rs`)
  — the integration point (marginals field)
- **katgpt-speculative `dflash_predict_with`** (`crates/katgpt-speculative/src/dflash.rs`)
  — where the correction hooks in
- **riir-neuron-db `MerkleFrozenEnvelope`** — the freeze/thaw weight-loading
  pattern to mirror
