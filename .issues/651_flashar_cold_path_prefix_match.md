# Issue 651: FlashAR Consensus Cold/Warm paths use mode-biasing acceptance (587 T7 audit)

**Date:** 2026-08-14
**Source:** [Issue 587](587_self_spec_exact_acceptance_policy.md) T7 — FlashARConsensusVerifier audit
**Target:** `crates/katgpt-forward/src/flashar_consensus.rs`
**Feature gate:** `flashar_consensus` (opt-in, unchanged)

## Audit verdict (T7)

`FlashARConsensusVerifier` (Plan 166) is **not distribution-preserving**, by design:

| Path | Acceptance | Distribution-preserving? |
|---|---|---|
| Plasma | both paths agree + conf > τ_p → accept unverified | ✗ — trusts the consensus token; output biased toward the drafter mode |
| Hot | winner conf > τ_h → accept unverified | ✗ — same, plus prefers the higher-confidence path |
| Warm | single-position spot-check vs cached `target_argmax` | ✗ — argmax prefix-match: accept iff `d == argmax(p)`, correct with argmax → collapses toward target mode (the exact failure Issue 587 fixed in `D2fDrafterVerifier`) |
| Cold | segment flush via prefix-match against `target_argmax` | ✗ — same class |

This is a **deliberate design tension, not a bug**: Plan 166's GOAT was speed
(verification-skipping via consensus), not distribution fidelity. Making
Plasma/Hot distribution-preserving would require adding the rejection
sampling they exist to skip — that would defeat their purpose.

## The actionable slice

Warm + Cold already materialize the target distributions (`p_flat`, the
`target_argmax` cache) — swapping their checks from `d == target_argmax` to
the FLARE Eq 21 rule (`u ≤ p(d)`, correction `p∖{d}`) is a ~15-line change
using the policy-step helpers landed in Issue 587 (`d2f_verifier.rs`). It
would make the verified paths exact while leaving the Plasma/Hot
verification-skipping trade-off intact (documented as such).

Deferring because:
1. `flashar_consensus` is opt-in (default-off), unlike `tri_mode`.
2. The swap changes FlashAR's output distribution — its tests + Bench 166
   numbers were measured under prefix-match; re-gating is required.
3. No consumer has asked for distribution fidelity from the consensus path.

## Tasks

- [ ] T1 — Swap Warm spot-check + Cold flush to Eq 21 acceptance (reuse the
  Issue 587 policy-step helpers; `DraftAcceptPolicy` on `ConsensusConfig`).
- [ ] T2 — Re-run the Plan 166 bench (latency + consensus rates) + the Issue
  587 G1 exactness harness on the Warm/Cold paths; document the delta.
- [ ] T3 — Doc note in the FlashAR header: Plasma/Hot remain
  distribution-biased BY DESIGN (verification-skipping is the feature).

## Trigger

Revisit when a consumer of `flashar_consensus` needs sampled (temperature)
output rather than mode-seeking decode.
