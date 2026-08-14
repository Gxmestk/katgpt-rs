# Issue 653 — MOP G4 bidirectionality follow-up (tie-break symmetry + γ regime)

> **Source:** [Issue 585](585_mop_defend_wrong_poc.md) G4 refutation → [riir-ai Bench 677](../../riir-ai/.benchmarks/677_mop_defend_wrong_poc.md)
> **Type:** POC refinement / falsification-cause isolation
> **Filed:** 2026-08-14
> **Status:** Open
> **Priority:** Normal — gates the G4 axis of the MOP Super-GOAT verdict (currently 3/4)

## The refutation being followed up

Bench 677's G4 gate: MOP's pooled CW-move ratio on the prey-predator ring =
**0.6117**, 1.9pp outside the [0.4, 0.6] band, at paper defaults (α=1, β=0,
γ=0.95). Per-seed strongly bimodal (mean 0.556, sd 0.270, min 0.067 /
max 0.903): individual episodes are *directional*; the policy itself does not
collapse. Two identified causes, neither yet isolated by experiment:

1. **Arena tie-break asymmetry (spec'd in Issue 585's ring design):** the
   predator's pursuit direction at the antipode (offset 8) is a fixed CW
   tie-break. This breaks mirror symmetry of the transition kernel — every
   post-prey-move offset-8 state transitions as if CW were canonical, tilting
   chase-region entry toward the CW-hold side.
2. **γ regime:** the γ-diagnostic is monotone (γ=0.80 → 0.4006 in-band;
   γ=0.90 → 0.4489 in-band; γ=0.95 → 0.6176; γ=0.99 → 0.8150). At γ=0.95 the
   softmax π* strongly favors hold/escape (V ramps ~3→11 nats from d=1 to
   d=7), episodes are short (~24 steps), and the pooled ratio is dominated by
   which chase region each episode enters.

## Tasks

- [ ] T1 — Add a symmetric tie-break variant (predator breaks antipode ties
  randomly with a seeded coin, or alternates) to `PreyPredatorRing`; re-run
  the G4 measurement at paper defaults. Isolates cause 1.
- [ ] T2 — Re-run G4 at γ ∈ {0.80, 0.90} with the symmetric tie-break.
  If in-band at both, the refutation is fully attributable to the spec's
  tie-break + γ regime (an arena/parameter artifact), and the gate can be
  re-marked PASS-with-caveat on the symmetric arena.
- [ ] T3 — If still out-of-band with symmetric tie-break at γ=0.95: record
  the honest boundary — MOP's within-episode directionality at high γ is a
  real property, and the G4 axis stays refuted for the paper-default regime.
  Do NOT tune α/β (β is inert on deterministic kernels; α is π*-invariant
  when β=0).
- [ ] T4 — Optionally: longer-episode variant (cap 50 000, matching G2) to
  test whether within-episode direction alternates over long horizons, which
  would move the pooled ratio toward band without any arena change.
- [ ] T5 — Update Bench 677 (append the isolation result) + Issue 585's G4
  entry with the outcome; close or re-mark the gate accordingly.

## Non-goals

- Re-tuning parameters to pass. The only knobs touched are the tie-break
  (arena bug class) and γ (paper-default vs documented alternative regime) —
  both with the measurement recorded either way.
- Touching G1–G3 (all PASS with margin).

## Related

- [Issue 585](585_mop_defend_wrong_poc.md) — the parent PoC
- [riir-ai Bench 677](../../riir-ai/.benchmarks/677_mop_defend_wrong_poc.md) — full record
- `riir-ai/crates/riir-poc/src/mop_poc.rs` — `PreyPredatorRing::env()` is the
  tie-break site; `g4_diagnostic_gamma_sensitivity` is the γ-sweep harness
