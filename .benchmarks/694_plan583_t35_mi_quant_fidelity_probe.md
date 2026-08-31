# Bench 694 — Plan 583 T3.5: offline MI quant-fidelity probe (KVarN, audit-only)

**Status:** COMPLETE — probe GOAT PASS (ordering / retention / non-vacuity control / determinism; audit-only, no gate flip per the plan's own rule).
**Date:** 2026-08-31
**Plan:** [`.plans/583_mi_est_modelless_mi_estimator.md`](../.plans/583_mi_est_modelless_mi_estimator.md) T3.5
**Module:** [`.benchmarks/693_mi_est_modelless_mi_estimator_goat.md`](693_mi_est_modelless_mi_estimator_goat.md) (module GOAT); consumer sibling: riir-train `.benchmarks/568_plan583_t34_mi_audit_axis.md` (T3.4)
**Surface:** `crates/katgpt-kv/src/kvarn/` (KVarN round-trip) + `crates/katgpt-kv/tests/bench_694_mi_quant_fidelity_probe.rs`
**Feature:** `katgpt-kv/mi_probe = ["kvarn", "katgpt-core/mi_est"]` (opt-in, test-support only, never a default)

---

## What landed

An offline **I(W; Ŵ)** probe that runs the SHIPPED KVarN production path
(`KVarNKVCache::store_key` → tile quantize → `dequantize_key_into`) on a
factor-structured key population (n=64 × d=64, one complete tile; rows =
shared latent direction × per-row scale + noise — the low-dimensional
structure real K/V populations carry) and reports, per bit width:

- **dCor² dependence magnitude + permutation p** — the verdict/magnitude pair
  (`PermTest::run_dcor`, B=128, reseeded);
- **DV effect-size tuple** — FrozenProj LOO + SMILE-LOO (τ=0.025) + 8-fold
  spread, REPORTED with the gauge caveat, never asserted against;
- **the existing reconstruction metrics** (`pseudo_decode_eval` tile MSE +
  cosine) for the same population — no duplication.

## Measured record (seed-fixed, deterministic)

| bits | dCor² | p | DV loo (gauge-caveated) | MSE | cosine |
|---|---|---|---|---|---|
| 8 | 1.00000 | 0.0078 | −41.68 nats | 2.7e-6 | 0.99999 |
| 4 | 0.99982 | 0.0078 | −42.25 nats | 8.2e-4 | 0.99623 |
| 2 | 0.99891 | 0.0078 | −40.86 nats | 4.2e-3 | 0.98079 |

- **retention**: dependence significant at 8, 4 AND 2 bits (2-bit + VarNorm
  is the shipped production setting — the probe confirms dependence
  RETAINED there, quantitatively).
- **ordering**: dCor² strictly decreases 8 → 4 → 2.

## The en-route instrument lesson (why the magnitude axis is dCor², not DV)

The first draft asserted STRICT ORDERING on the DV LOO value. First seed
passed (−80.1 → −81.4 → −84.3); a seed/population change VIOLATED it
(2-bit arm read −40.9, ABOVE the 4-bit arm's −42.2). Root cause: the DV
value of a fixed critic carries a null gauge that depends on the population
AND the noise level through the score variance — the T1.4 calibration's
critic-dependent gauge, extended here: **the gauge's noise dependence is not
even signed a priori** (variance explosion drives the logmeanexp Q-term
either way), so cross-width DV ordering is not a law. dCor² has no gauge
(a distance-matrix correlation in [0,1]) and orders with tiny estimator
variance. The probe now pins ordering on dCor² and reports the DV tuple
honestly — the failure is itself the instrument-choice record. This also
sharpens the module's honesty contract for every future consumer: *value +
spread + p for reporting; magnitude questions go to dCor²; significance
questions go to p.*

## Gates (all PASS)

| Gate | Result |
|---|---|
| ordering | dCor² 1.00000 > 0.99982 > 0.99891 (strict) |
| retention | p = 0.0078 (= 1/129, minimal) at 8/4/2 bits |
| non-vacuity control | constant-row "dequantized" population → p ≥ 0.05 (dependence lost) — the probe can fail |
| determinism | bit-identical records across runs (dCor, DV, MSE, cosine all pinned by `.to_bits()`) |

Clippy 0 in touched files (`-p katgpt-kv --features mi_probe --all-targets`).
`cargo test -p katgpt-core --features mi_est --lib` 2033/0/7i (module surface
unchanged by the probe). Default builds untouched (opt-in feature).

## Honest scope

- The probe is an AUDIT instrument: it does not gate KVarN promotion or
  regression — a decision-value re-gate (does I(W;Ŵ) catch a regression the
  MSE/cosine columns miss?) is the explicitly recorded precondition for any
  gate flip.
- Population shape is fixture-level (factor-structured synthetic), not a
  trained-model KV dump; the probe measures the INSTRUMENT + the round-trip,
  not a production checkpoint's fidelity.
- One off-by-construction note: `pseudo_decode_eval` takes keys+values, so
  the probed keys ride both slots — the MSE/cosine columns are key-tile
  metrics, matching what the MI axis measures (keys only).
