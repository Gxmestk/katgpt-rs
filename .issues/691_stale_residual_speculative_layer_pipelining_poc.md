# Issue 691: Stale-Residual Speculative Layer Pipelining POC (arXiv:2608.23841 §6.3)

**Repo:** katgpt-rs (primitive + analyzer) — simulator consumer may live in riir-train
**Research:** [katgpt-rs/.research/508](../.research/508_Pipeline_Native_Transformers_CPU_Decode_CoDesign.md)
**Source:** [arXiv:2608.23841](https://arxiv.org/abs/2608.23841) §6.3 (Approach A + B — the paper's own UNTESTED hypotheses)
**Filed:** 2026-08-26
**Cost estimate:** T1 zero-GPU (offline analyzer on saved traces); T2 zero-GPU (simulator); T3 optional GPU bench

---

## The falsifiable question

For **standard** (non-delay-rewritten) transformer checkpoints, does residual dominance
(`‖δℓ‖/‖x_in^ℓ‖ ≪ 1`) hold strongly enough that layer ℓ+1 can begin on the **stale**
residual `x_in^ℓ` while layer ℓ is still computing — accept-with-correction when the
layer contribution lands small, rollback-and-recompute when it doesn't — yielding a net
wall-clock win from overlapping weight I/O with compute?

The paper proves the vertical-pipeline schedule math for *rewritten* architectures and
proposes speculative recovery as the path for *standard* checkpoints — **without running
it**. No prior art found for stale-residual layer speculation (LayerSkip/early-exit are
different mechanisms: variable depth / conditional compute, not stale-input execution).
We hold checkpoints + trace tooling + rollback machinery → we can produce the first
measured verdict.

## Why the stack can test it cheaply

- Offline analyzer needs only saved per-layer activations (norm in / norm out per layer).
- The rollback machinery pattern ships: GDN tree-verify (rollback-free S₀), token-level
  `rollback_speculative_gpu` (riir-train), KV page fast path (bench_414).
- Distinct from `HydraSkipPlan` (skips layers on cumulative-DE — different mechanism;
  the signal-diff is documented in Research 508 §2.1 #8).

## Substrate check (substrate-first skill, 2026-08-26)

- Searched for: `stale_residual`, residual dominance, layer capture, `prefix_sum_in`, per-layer hidden, `forward_token_with_layer_capture`, least-squares / pseudo-inverse / Cholesky.
- **Found (consume, do not rebuild):**
  - K3 per-layer capture + re-entry: `kimi_k3_forward_token_saved` (`layers[].prefix_sum_in`), `kimi_k3_forward_token_hidden` (caller-set `runtime.hidden`), pub `kimi_decoder_layer_forward` — `src/kimi_k3/`.
  - K3 real checkpoint + tokenizer: `loader::load_kimi_k3` + `TiktokenTokenizer` (data/kimi-k3-0.40b/model.safetensors, 1.58 GB, on disk).
  - Bonsai-27B CPU forward **with per-layer capture**: `riir_engine::deltanet::forward_qwen_deltanet_ternary_with_capture` (pure CPU — the bench_604 diag path).
  - Gemma-2-2B CPU forward with a generic per-layer hook: `forward_gemma2_layers` + `PostLayerHook::after_layer(layer_idx, residual)` (production path, no duplication).
  - Closed-form OLS substrate (T3): `katgpt-attn-match::value_fitter::fit_cv_least_squares` (blocked Cholesky + jitter escalation + ridge, multi-RHS) — consumed by the K3 simulator, not reimplemented.
  - Rollback machinery (context): GDN tree-verify S₀, `rollback_speculative_gpu` — different layer (token-level), not consumed here.
- **Not found (build new, minimal):** the ratio/analyzer math, the stale-replay simulator (snapshot/restore + replay-from-layer-ℓ+1), the overlap latency model, the trace-file handoff. Nothing in the stack runs a layer on a stale residual (HydraSkip *drops* layers; spec-decode speculates *tokens* — different mechanisms, confirmed by grep).
- Architectural rules checked: modelless-first (T3 is closed-form OLS — no GD) ✓; katgpt-rs is upstream of riir-ai (trace files, not deps) ✓; sigmoid-not-softmax n/a; feature-flag discipline (opt-in `stale_residual` in katgpt-core, opt-in module in root) ✓.

## Implementation plan (added at execution)

- `katgpt-core` opt-in feature `stale_residual`: `residual_dominance` (T1 ratios + paper bar), `overlap_latency` (extraction #3 `(C+IO)/max(C,IO_eff)`), unit tests.
- root `src/kimi_k3/stale_residual.rs` (feature `kimi_k3_loader`): runtime snapshot/restore (MLA prefix+seq_len, KDA clone, block_state), instrumented true forward (per-layer x_in + mid-token block_state capture), speculative replay from layer ℓ+1 on stale/corrected input, KL/top-1 metrics, router-logit + x_in OLS predictors (consume `fit_cv_least_squares`).
- `benches/bench_683_stale_residual_poc.rs` (required-features `kimi_k3_loader`): real K3-0.40B T1+T2+T3+latency; reads Bonsai/Gemma trace files when present (env `STALE_RESIDUAL_TRACES` dir).
- riir-ai `crates/riir-engine/examples/stale_residual_trace_dump.rs`: pure-CPU Bonsai (`with_capture`) + Gemma (`PostLayerHook`) per-layer trace dumper → binary trace v1 ("SRTR").

## Tasks

- [x] T0 — Substrate-first gate (recorded above)
- [x] **T1 — Residual-dominance analyzer** — **VERDICT: FAIL on all three classes** (Bench 683): K3-0.40B 0/8 layers, Bonsai-27B 0/64, Gemma-2-2B 0/26 under the 0.05 bar. Measured law: ratio ≈ k/√L (k≈1.5–3) → passing needs ≥~1000 layers. Primitives in `katgpt-core/src/stale_residual.rs` (opt-in `stale_residual`); cross-model traces via `riir-ai` `stale_residual_trace_dump` (SRTR v1).
- [x] **T2 — Trace simulator** — **VERDICT: gate-accepting regimes destroy quality** (Bench 683): 0% accept at the paper's θ (T1's consequence); at θ=0.5 (forced), 18% accept but top-1 preserved only 62.8% among accepted; persistent-hazard arm (stale KV/KDA persists on accept) → 95.7% greedy-token agreement, 2/16 trajectories diverge. Simulator in `src/kimi_k3/stale_residual.rs` (bit-exact: G1a–G1d PASS).
- [x] **T3 — Approach B probe** — **VERDICT: FAIL vs paper's R² > 0.7**: best router-logit held-out R² = 0.445 (delay 4); corrected replay improves top-1 93.5%→97.4% / KL 0.50→0.12 (real signal, insufficient; still 0% gate accept at θ=0.05). x_in-linear ceiling also low (≤0.77 at one layer, elsewhere ≪). OLS substrate consumed (`katgpt-attn-match::value_fitter`) — no new solver.
- [x] T4 — Verdict + gate decision — **NEGATIVE, recorded + closed.** No wall-clock POC plan (precondition failed). Record: [Bench 683](../.benchmarks/683_stale_residual_poc.md); Research 508 updated with the measured law. Reusable landed pieces: analysis primitives, the K3 replay simulator, SRTR trace format + dumper, the k/√L law. **Reopen triggers:** ≥1000-layer checkpoint class, or an architecture explicitly trained for residual dominance (re-run T1 first — one command).

## Honest scope notes

- **No quality-parity claim is made here** — this issue TESTS the hypothesis (§3.6
  defend-wrong discipline: the POC defends or refutes; either outcome is a result).
- Ternary-regime caveat (Research 508 §2.0): at 1.58 bits/weight we are only ~2× below
  machine balance — the overlap payoff shrinks; T2's latency model must use OUR stream
  ratios, not the paper's Q4 numbers.
- Attention layers update KV cache — a stale attention input writes stale K/V; T2 must
  model this (the paper flags it as the open hazard for attention-path delays).
