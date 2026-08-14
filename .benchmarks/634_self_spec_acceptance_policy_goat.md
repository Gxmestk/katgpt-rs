# Bench 634 — Self-Speculation Distribution-Preserving Acceptance (Issue 587 / FLARE Eq 8/21/22)

**Date:** 2026-08-14
**Issue:** [587](../.issues/587_self_spec_exact_acceptance_policy.md) — executing [Research 480](../.research/480_FLARE_Hybrid_Diffusion_Dual_Trust_Decode.md) (FLARE, arXiv:2606.01774 §3.3 + App B.2)
**Target:** `crates/katgpt-forward/src/d2f_verifier.rs` + `d2f/mod.rs` (+ root re-export shims, `SelfSpecConfig`)
**Feature gate:** `tri_mode` (unchanged — the policy is a config enum, not a Cargo feature)
**Machine:** M3 Max (Apple Silicon), debug-mode cargo test harness. GPU: idle (CPU-only change; no GPU exclusivity concern).
**Headline:** `DraftAcceptPolicy::SoftmaxArgmax` promoted to default — exact at −6% latency.

---

## What changed

1. **Acceptance policy taxonomy** (`DraftAcceptPolicy`): `PrefixMatch` (legacy control) / `SoftmaxArgmax` (Eq 21, exact) / `TruncatedArgmax` (Eq 22, approximate) / `ExactQ` (Eq 8 analog, exact under sampled drafts).
2. **Off-by-one fix**: the pre-rewrite Phase 2/3 compared draft token `i` against the target distribution for position `i+1`. Verification is now slot-aligned (matching LeviathanVerifier's `offset = i * vocab`). This fix alone makes ALL policies' acceptance meaningful; the legacy numbers were comparing misaligned distributions.
3. **Streaming verify (T5)**: `p_distributions_flat [(K+1)×V]` deleted; one `[V]` `probs_buf`; Phases 2+3 fused (score one position ahead only while accepting); early exit on first rejection skips the remaining target forwards.
4. **Greedy drafting** (`D2fDecodeConfig::greedy_draft`): deterministic argmax over valid (pruner-respecting, relevance-weighted) tokens — the point-mass proposal law Eq 21 exactness requires. No RNG consumption.
5. **q-capture** (`d2f_decode_block_with_prompt_with_q`): per-position draft-time proposal law captured at commit time (the actual sampled distribution; point mass under greedy). Caller-owned out-buffer → zero-alloc hot path.

## G1 — correctness (exactness)

Pure policy level (fixed toy target p over 32 tokens, 40k rounds, pinned seeds; bounds are the test assertions):

| Policy | TV from target | Bound | Verdict |
|---|---|---:|---|
| SoftmaxArgmax (argmax draft) | < 0.02 (asserted) | 0.02 | **PASS** |
| ExactQ (sampled draft + stored q) | < 0.02 (asserted) | 0.02 | **PASS** |
| TruncatedArgmax (top-16) | ≤ (1 − z₁₆) + 0.05 | tail-bound | PASS (approximate by design) |
| PrefixMatch (control) | **> 0.7 — output is a constant** | fail-required | **FAILS exactness (the gain proof)** |

E2E through `speculate()` (micro_dllm config, self-speculation: draft == target weights, 8000 rounds from a fixed anchor; reference = `softmax(forward(anchor))` at target temperature):

| Policy | First-token TV | Distinct outputs |
|---|---:|---:|
| SoftmaxArgmax (forced greedy drafts) | **0.0003** | 27 (full support) |
| ExactQ (sampled drafts + stored q) | 0.0140 | full support |
| PrefixMatch (control) | 0.0140 | **1 — total mode collapse** |

Note on the two 0.0140s: for PrefixMatch the value is `1 − p_max` (p_max ≈ 0.986 at target temp 0.5) with a *single distinct output token* — collapse. ExactQ's 0.0140 is f32 correction-path noise (residual cumsum accumulation), 4× under the 0.06 bound, deterministic under the pinned seed.

**G1 verdict: PASS.** The exactness claim is modelless (a distribution-equivalence identity, not a trained-quality claim) — no conformal floor applies (no interval/coverage/UQ claim being made).

## G2 — acceptance + latency (self-speculation, untrained weights, K=4, 400 rounds)

| Policy | tokens/round | latency µs/round | vs PrefixMatch |
|---|---:|---:|---|
| PrefixMatch | 2.00 | 352.4 | control |
| **SoftmaxArgmax** | **1.98** | **332.5** | −1% acceptance, **−6% latency** |
| TruncatedArgmax | 1.99 | 387.8 | +10% latency |
| ExactQ | 2.00 | 441.0 | +25% latency |

Gate: `SA tokens ≥ PM − 0.1` ✔ (1.98 ≥ 1.90), `SA latency ≤ PM × 1.25 + 5` ✔ (332.5 ≤ 445.5). **G2 PASS.**

Latency wins for SoftmaxArgmax come from the streaming rewrite (no `[(K+1)×V]` softmax+copy; early exit on rejection; greedy drafting skips the sampler's cumulative scan). ExactQ's +25% is the q-capture write (`[V]` per position) + residual scan — the price of exactness under sampled drafts.

Memory (Gemma-2 256K vocab, K=4): p-flat `(5 × 256K × 4B) = 5.0 MB` → `probs_buf 1.0 MB` (SoftmaxArgmax / TruncatedArgmax); ExactQ adds `[K × V]` q = 4.0 MB (captured at draft time, reused).

Caveat (honest): acceptance numbers are untrained-weight relative comparisons. A trained-model acceptance study is riir-train territory (Research 480's cross-refs); the gate here is non-regression vs the control, which holds.

## G3 — no-regression

- `cargo test --features "tri_mode dllm" --lib` (root): **206 passed**
- `cargo test -p katgpt-forward --features "dllm tri_mode" --lib`: **154 passed** (incl. the 3 legacy shape tests + 8 new Issue 587 unit tests)
- `cargo test --features tri_mode --test test_d2f_verifier`: **10 passed** (5 legacy proofs unchanged + 5 new: 3× G1 E2E, G2, G4)
- `cargo test --features tri_mode --test test_d2f_decode`: **17 passed**
- `cargo test --features tri_mode --test test_diffusion_sampler_goat`: **5 passed**
- `cargo clippy` clean on: default, `dllm`, `tri_mode+dllm` (--all-targets), `dllm+tri_mode+flashar_consensus` (--all-targets), `--no-default-features`

**G3 PASS.** (Existing tests assert shape/determinism/boundedness — robust to the policy default change + off-by-one fix; no test pinned the old token streams.)

## G4 — alloc-free steady state

- `probs_buf [V]`, `q_distributions_flat [K×V]`, `residual_buf [V]`: constructor-fixed capacities, never grown at runtime.
- `accepted_buf`: one bounded realloc per `speculate()` call (`std::mem::take` for the `Vec`-return ABI, capacity ≤ K+1) — pre-existing, unchanged, documented as the ABI cost.
- Streaming verify performs zero heap allocation in the accept loop (correction paths walk existing buffers).
- E2E guard: 220-call window, output length ∈ {1, 2} stable for both SoftmaxArgmax and ExactQ.

**G4 PASS** (with the documented `Vec`-return ABI exception, identical to the pre-rewrite behavior).

## Promotion

`DraftAcceptPolicy::SoftmaxArgmax` is now `#[default]` (enum default, `D2fDrafterVerifier::new`, `SelfSpecConfig::default`). PrefixMatch demoted to explicit opt-in control. Justification: G1 exact (the only non-PrefixMatch policy that is exact at *lower* latency than the control), G2 non-regressing, G3/G4 clean. The gain is **modelless** (distribution identity + streaming memory) per the promotion rule.

ExactQ stays available (not default) for consumers that want temperature-diverse drafting with exactness — its +25% latency is the honest price.

## Deviations from the issue text (documented)

1. **T4 top-k → full-q**: FLARE's Exact-Truncated stores top-k `(ids, probs)` to bound serving memory at 256K vocab. We store the full `[K×V]` q — exact (no tail approximation), zero-alloc via a caller-owned out-buffer, and affordable at library scale. Named `ExactQ` to make the semantics explicit.
2. **T5 streaming shape**: FLARE fuses the verify into the LM-head pass (streaming `(max, sum-exp)` + Gumbel-max correction, never materializing logits). Our forward returns a logits slice per position, so the natural streaming point is per-position softmax into one reused `[V]` buffer + fused score-accept-advance loop + early exit — same `[(K+1)×V]` elimination, different fusion boundary.
3. **Off-by-one fix discovered mid-implementation** (not in the original issue): documented above; it is the reason legacy acceptance numbers were near-garbage and why G2's control numbers moved.

## FlashAR audit (T7)

Not distribution-preserving by design: Plasma/Hot accept unverified (Plan 166's latency trade-off), Warm/Cold use argmax prefix-match (mode-biasing — the exact failure class fixed here). Actionable slice filed as [Issue 650](../.issues/650_flashar_cold_path_prefix_match.md). No code change in this issue.
