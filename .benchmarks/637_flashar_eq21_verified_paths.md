# Bench 637 — FlashAR Warm/Cold Eq 21 Verified Paths (Issue 651)

**Date:** 2026-08-15
**Machine:** M3 Max (CPU-only; no GPU used — exclusivity check N/A)
**Source:** Issue 651 (removed per noise-reduction rule — resolved 2026-08-15; this bench is the record) — the Issue 587 T7 audit's actionable slice
**Source paper:** FLARE ([arXiv:2606.01774](https://arxiv.org/abs/2606.01774) §3.3) — Eq 21/22 acceptance taxonomy
**Feature:** `flashar_consensus` (opt-in, unchanged — root feature already implies `tri_mode` + `plasma_path`; katgpt-forward's flag now implies `tri_mode` for the d2f_verifier import)
**Run:** `cargo test -p katgpt-rs --features flashar_consensus --test bench_166_flashar_consensus_goat -- --nocapture` + the new exactness test in `katgpt-forward/src/flashar_consensus.rs`

## Verdict: GOAT PASS (G1 exactness + G3 no-regression + memory win)

```
G1 exactness (NEW — the 587 T2 E2E harness applied to FlashAR's verified
   paths; all-Warm/Cold config: plasma/hot thresholds forced > 1.0):
   first-token empirical marginal vs reference p_0 = softmax(forward(anchor)),
   n=8000 rounds, micro transformer, vocab 64, temp 0.5:
     SoftmaxArgmax (Eq 21):  TV = 0.0046  — exact (≈ sampling noise)
     PrefixMatch (control):  TV = 0.3256  — collapses to the mode point mass
   → the verified paths are distribution-preserving; the legacy check was
     mode-biasing. (UQ note: distribution-equivalence gate, not a
     calibrated-UQ claim — conformal floor N/A, same as 587 T2.)

G3 no-regression:
   bench_166_flashar_consensus_goat: 9/9 PASS (G1-G7 + T9 both benches)
   katgpt-forward lib (dllm+tri_mode+flashar_consensus): 165/165
   root lib (flashar_consensus+tri_mode): 206/206
   test_d2f_verifier 17/17 + test_d2f_decode 10/10 (the 587 suite — the
     shared step helpers are untouched, only visibility widened)
   clippy clean (katgpt-forward lib; root tests)

T9 metrics delta (micro fixture, draft_width=4, n=100):
   avg tokens/call:  5.00 → 5.00  (unchanged — see routing analysis)
   time per token:   0.452ms → 0.404ms  (−11%)

Memory (the 587 T5 streaming pattern applied):
   p_flat (MAX_DRAFT_WIDTH+1) × vocab + forward_scratch (vocab) deleted
   → single vocab-sized probs_buf. At Gemma-2's 256K vocab + MAX_DRAFT_WIDTH=64:
   65 × 256K × 4B ≈ 67 MB → 1 MB. Target forwards stop at the first rejection
   (early exit) instead of scoring all positions upfront.
```

## What changed

1. **Slot alignment (the load-bearing fix).** The pre-651 Phase 2 had the
   same off-by-one Issue 587 fixed in `D2fDrafterVerifier`: it tested draft
   token `i` against the target's position-`i+1` distribution *conditioned on
   the draft token being tested* (fed `v_0`, then the H tokens). H-win
   positions auto-accepted trivially (`h_i` WAS that argmax); V-win positions
   compared against a self-conditioned argmax. The verified paths were
   structurally degenerate — which is why the old T9 showed a saturated
   5.00 tokens/call. The loop now tests `d_i` against
   `p_i = P(· | anchor, accepted d_0..d_{i-1})` and feeds the ACCEPTED token
   after each acceptance (the Leviathan/D2f streaming pattern).
2. **Eq 21 acceptance on Warm/Cold** (the issue's T1): `ConsensusConfig`
   gains `accept_policy: DraftAcceptPolicy` (default `SoftmaxArgmax` — the
   587 precedent). The step helpers are shared with `d2f_verifier`
   (`pub(crate)` visibility — no duplication). `ExactQ` is honored as its
   point-mass identity (the consensus winner is a deterministic proposal;
   Eq 8 with q = δ_d degenerates to Eq 21). `TruncatedArgmax` (Eq 22) is
   wired through the shared helper.
3. **Plasma/Hot remain distribution-biased BY DESIGN** (T3 — header doc):
   they skip verification entirely; that skipping IS the latency feature.
   Consumers needing sampled output force Warm/Cold (thresholds > 1.0).
4. **DRY refactor**: per-position kernels `ternary_consensus_one` +
   `route_one` extracted; the array-based `compute_ternary_consensus` /
   `route_thermal_paths` delegate to them (the aligned `speculate` loop
   interleaves the kernels — the array forms can't, because H must be
   conditioned on the accepted prefix, which is inherently sequential).

## Honest notes

- **The 5.00 avg tokens/call is unchanged — and the reason changed.** On the
  micro fixture, `p_0(argmax) ≈ 0.67` (derived: PrefixMatch TV = 1 − p₀(am)),
  so most positions clear the Hot/Plasma thresholds and SKIP verification —
  the default config is skip-heavy by design, and both the old (degenerate
  auto-accept) and new (skip-heavy routing) mechanisms saturate the counter.
  Under the forced-verified config the exactness harness shows real
  accept/reject behavior (Eq 21 mean acceptance ≈ Σ p² ≈ 0.5 at position 0).
  The G5 gate (≥ 1.0) passes either way; the number carries no quality
  signal on this fixture.
- **Latency −11%** (0.452 → 0.404 ms/token) from the streaming restructure
  (no upfront full-width scoring pass; early exit) — measured on the micro
  fixture, informational.
- **The PrefixMatch control's semantics changed** (aligned conditioning) —
  its old outputs were produced by the off-by-one law; nothing pinned them
  (tests check determinism/bounds, not token identity), and the control
  exists to demonstrate mode collapse, which it still does (TV 0.3256).

## Trigger status (from the issue)

The issue was deferred pending "a consumer of flashar_consensus [needing]
sampled (temperature) output rather than mode-seeking decode". No such
consumer has appeared; this lands the well-scoped slice anyway because the
off-by-one discovered during implementation meant the verified paths were
degenerate (H-win auto-accept), which is a correctness defect in the
opt-in feature regardless of the distribution-fidelity trigger. The
Plasma/Hot skip-bias trade-off is untouched.
