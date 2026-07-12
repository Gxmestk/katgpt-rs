# Issue 136: Demote `flashar_consensus` from default-on — GOAT G1 FAIL (quality)

> **Created:** 2026-07-12
> **Status:** Open
> **Priority:** P1
> **Blocked:** No
> **Depends:** Plan 485 benchmark (complete)
> **Related:** [Plan 485](../../riir-ai/.plans/485_consensus_tree_verify_poc.md), [Benchmark 485](../../riir-ai/.benchmarks/485_consensus_tree_verify_poc.md), [Proposal 018 §4.6-4.7](../../riir-ai/.proposals/018_unique_runtime_training_methodology.md), [Issue 453](../../riir-ai/.issues/453_consensus_tree_verify_proof.md)

## Problem

The Plan 485 defend-wrong PoC benchmark quantified `flashar_consensus`'s
quality cost: **KL divergence 2.9–6.5 nats** at accepted positions across 3
game domains (bomber/Go/FFT), vs the Leviathan baseline's KL ≈ 0.03. That's
a **100× quality regression** — the consensus-skip mechanism introduces
severe distribution bias because two imperfect drafters agreeing on a token
does NOT mean they agree on the RIGHT token.

The DSpark modelless hybrid (entropy-based skip-verify) outperforms FlashAR
on **both** axes:

| Domain | Metric | FlashAR (consensus) | DSpark (entropy) | Advantage |
|--------|--------|--------------------|-----------------|-----------|
| Bomber | KL | 2.9364 | **0.1182** | 25× lower |
| Bomber | tok/cycle | 4.70 | **5.00** | 6% faster |
| Go | KL | 5.2856 | **0.2451** | 22× lower |
| Go | tok/cycle | 4.73 | **5.00** | 6% faster |
| FFT | KL | 4.7606 | **0.1504** | 32× lower |
| FFT | tok/cycle | 4.72 | **5.00** | 6% faster |

DSpark wins on quality by 22–32× AND on speed by 6%. FlashAR has no axis
where it dominates.

## Root Cause

FlashAR's consensus-skip is fundamentally flawed:

1. **Two imperfect drafters agreeing ≠ correct.** The D2F drafter's
   distribution diverges from the target's (KL > 0). When D2F and AR agree,
   they agree on a token biased away from the target argmax.
2. **Entropy is the direct signal.** DSpark measures the target's own
   entropy — low entropy means the argmax is very likely correct, so
   skipping verification is safe. Consensus measures drafter agreement,
   which is an indirect (and biased) proxy for correctness.
3. **Thermal routing doesn't fix it.** FlashAR's Hot/Warm/Cold paths still
   accept biased tokens at plasma positions. The consolidation design
   (Proposal 018 §4.7, Plan 485) tried to fix this with DDTree+Leviathan
   at disputed positions — but it made things WORSE (ConsensusTree KL
   4.8–6.5 > FlashAR KL 2.9–5.3).

## GOAT Gate Assessment

| Gate | Criterion | Pass? | Evidence |
|------|-----------|-------|----------|
| G1 (quality) | KL ≤ baseline (Leviathan 0.03) | **FAIL** | KL 2.9–6.5 (100× worse) |
| G2 (speed) | tok/cycle ≥ baseline | PASS | 4.7 ≥ 1.0 |
| G3 (no regression) | No competitor better on BOTH axes | **FAIL** | DSpark better on both |

**G1 and G3 FAIL.** The feature should not be default-on.

## Demotion Plan

### Step 1: Demote from default-on (this issue)

Remove `flashar_consensus` from the `default` feature list in
`katgpt-rs/Cargo.toml`. The feature stays available as opt-in
(`--features flashar_consensus`) for callers who explicitly want the
speed-over-quality tradeoff.

### Step 2: Check downstream breakage

`flashar_consensus` pulls in `tri_mode` and `plasma_path`. Demoting it
means:
- `FlashARConsensusVerifier` is no longer compiled by default
- `katgpt_rs::speculative::FlashARConsensusVerifier` import fails unless
  the feature is explicitly enabled
- Downstream crates (riir-ai, riir-poc) that use FlashAR must add
  `features = ["flashar_consensus"]` explicitly

### Step 3: Document the replacement

The DSpark modelless hybrid (entropy-based skip-verify) is the recommended
replacement. A production implementation would use:
- DFlash drafting (existing)
- Bigram Markov head (to be implemented — see Research 316 §1.2)
- Bebop entropy confidence (existing — `AcceptanceForecast`)
- Entropy threshold for skip-verify (ln(2) ≈ 0.693)

The PoC DSpark hybrid in `riir-poc/benches/consensus_tree_verify_poc.rs`
proves the concept. A future plan should implement the full DSpark
architecture as a katgpt-rs feature.

## Acceptance Criteria

- [ ] `flashar_consensus` removed from `default` in `katgpt-rs/Cargo.toml`
- [ ] `cargo check` passes (fix downstream breakage or add explicit features)
- [ ] Downstream crates that need FlashAR add `features = ["flashar_consensus"]`
- [ ] GOAT gate verdict documented in this issue
- [ ] Issue 453 (consolidation proof) cross-referenced

## References

- [Benchmark 485](../../riir-ai/.benchmarks/485_consensus_tree_verify_poc.md) — full 4-way benchmark results
- [Plan 485](../../riir-ai/.plans/485_consensus_tree_verify_poc.md) — defend-wrong PoC
- [Proposal 018 §4.6-4.7](../../riir-ai/.proposals/018_unique_runtime_training_methodology.md) — DDTree vs FlashAR tradeoff + consolidation design (REFUTED)
- [Research 316](../.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md) — DSpark analysis
- [Plan 166](../.plans/166_flashar_consensus_tri_mode.md) — FlashAR Consensus original plan
