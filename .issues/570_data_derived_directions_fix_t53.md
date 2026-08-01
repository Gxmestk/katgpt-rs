# Issue 570 — T5.3b: Data-Derived Directions Fix the Concentration-of-Measure Failure

**Status:** RESOLVED (2026-08-02) — PASS. Layer 4 validated on synthetic data.

**Parent:** [Proposal 011](../.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md) Phase 5 T5.3.
**Prior:** [Issue 569](569_swe_trajectory_geometry_synthetic_poc.md) (resolved, T5.3 CONDITIONAL FAIL).

## Context

Issue 569's T5.3 found that `CommittedFieldBlend::commit` produces degenerate
sigmoid gates (max 0.4879 < 0.6 threshold) when direction vectors are sampled
randomly. The documented design constraint was: "real freezer needs data-derived
directions, not random." This issue tests that constraint directly.

## The deeper finding (pre-implementation analysis)

The T5.3 failure had TWO compounding causes, not one:

1. **Random directions** (concentration of measure): random unit vectors in the
   summary space are nearly orthogonal to any fixed summary → near-zero dots →
   sigmoid ≈ 0.5 → degenerate. This is the documented cause.

2. **Summary encoding mismatch** (NEW, discovered during this issue): the
   synthetic trajectories use **random targets per seed** (random attractors,
   random drift directions, random convergence targets). This means **endpoint
   positions do NOT cluster by failure mode** — oscillation-seed-0's endpoint
   is just as far from oscillation-seed-1's endpoint as it is from
   committed-wrong-seed-0's endpoint. The discriminative signal lives in the
   **geometry** (curvature, length, step-to-step cosine), not the position.

Consequence: even with data-derived directions, **endpoint-position summaries
will NOT cluster by failure mode** → centroid-based directions will be
near-zero → still degenerate. The fix requires BOTH data-derived directions
AND a summary encoding that captures the discriminative geometry.

## Hypothesis

When direction vectors are derived from cluster centroids of **geometry-encoded**
trajectory summaries, FAME produces non-degenerate gates (> 0.6) that correctly
identify the matching archetype.

Dual-strategy test to isolate the two causes:
- **Strategy A (geometry summary):** encode `(length, curvature, cosine, n_steps)`
  as the summary → clusters by failure mode → data-derived directions should
  produce discriminative gates.
- **Strategy B (endpoint summary):** endpoint position padded to 32-D → does NOT
  cluster by failure mode (random targets) → data-derived directions should
  STILL be near-degenerate.

The **contrast** between A and B is the finding: the SweTrajectoryFreezer's
summary encoder must capture failure-mode-discriminative geometry, not just
raw latent position.

## Protocol

1. Build M=5 trajectories per failure mode (seeds 0..4) for K=3 modes:
   oscillation, committed-wrong, converged-correct.
2. Split: seeds 0..2 → train (derive directions); seeds 3..4 → test (probes).
3. For each trajectory, compute the summary via each strategy.
4. **Derive directions:** centroid_k = mean(train summaries for mode k);
   global = mean(centroid_k for all k); direction_k = normalize(centroid_k - global).
5. **Probe:** for each test trajectory, FAME-commit with derived directions.
   Record the gate for each archetype.
6. **Verdict:** matching archetype gate > 0.6 AND is the max gate.

## Falsifiable threshold

≥ 80% of test probes must have their matching archetype's gate > 0.6 AND be the
argmax gate. Below 80% → FAIL.

## Outcome-action table

| Strategy A (geometry) | Strategy B (endpoint) | Action |
|---|---|---|
| PASS | FAIL | **Expected.** Documents the design constraint: summary encoder must be geometry-aware. T5.3 upgraded to PASS. T5.5 proceeds with geometry-summary encoder. |
| PASS | PASS | Unexpected — positional summaries DO cluster. Investigate why (possibly low effective dimensionality). |
| FAIL | FAIL | Data-derived directions insufficient even with geometry summaries. T5.5 blocked; consider riir-train LoRA fallback (Layer 4b). |
| FAIL | PASS | Contradiction (shouldn't happen — geometry is more discriminative than position). Re-examine protocol. |

## Verdict (2026-08-02)

**Strategy A (geometry): ✅ PASS — 100% accuracy (6/6 probes correct).**

| Probe mode | gate_0 | gate_1 | gate_2 | argmax | correct |
|---|---|---|---|---|---|
| oscillation | 0.879 | 0.333 | 0.125 | 0 | ✅ |
| oscillation | 0.862 | 0.334 | 0.145 | 0 | ✅ |
| committed-wrong | 0.574 | 0.611 | 0.363 | 1 | ✅ |
| committed-wrong | 0.541 | 0.611 | 0.400 | 1 | ✅ |
| converged-correct | 0.393 | 0.510 | 0.618 | 2 | ✅ |
| converged-correct | 0.396 | 0.510 | 0.614 | 2 | ✅ |

All matching-archetype gates > 0.6. Non-matching gates are clearly suppressed
(< 0.6, often < 0.4). The FAME blend produces a dominant archetype in every
probe — exactly the non-degenerate behavior T5.3 failed to achieve.

**Strategy B (endpoint): ❌ FAIL — 17% accuracy (1/6 probes correct).**

| Probe mode | gate_0 | gate_1 | gate_2 | argmax | correct |
|---|---|---|---|---|---|
| oscillation | 0.472 | 0.528 | 0.472 | 1 | ❌ |
| oscillation | 0.454 | 0.546 | 0.454 | 1 | ❌ |
| committed-wrong | 0.999 | 0.001 | 0.999 | 2 | ❌ |
| committed-wrong | 0.331 | 0.669 | 0.332 | 1 | ✅ (by luck) |
| converged-correct | 0.472 | 0.528 | 0.472 | 1 | ❌ |
| converged-correct | 0.455 | 0.545 | 0.455 | 1 | ❌ |

Endpoint positions don't cluster by failure mode (random targets scatter
positions). Gates are near 0.5 — degenerate. The concentration-of-measure
issue persists even with data-derived directions when the summary doesn't
encode discriminative signal.

## Verdict

**Expected outcome achieved: Strategy A PASS + Strategy B FAIL.**

This confirms the design constraint documented in the pre-implementation
analysis: **the SweTrajectoryFreezer's summary encoder must capture
failure-mode-discriminative geometry, not just raw latent position.** The
T5.3 failure had two compounding causes:
1. Random directions (concentration of measure) — fixed by data-derived directions.
2. Summary encoding mismatch (position doesn't encode geometry) — fixed by
   geometry-aware summary encoder.

**T5.3 is upgraded from CONDITIONAL FAIL to PASS** (on the data-derived +
geometry-summary condition). Layer 4 is validated on synthetic data:
T5.1 (geometry discriminates) + T5.2 (CUCG fires) + T5.3b (data-derived
FAME blend works) all PASS.

**Design constraint for T5.5:** the `SweTrajectoryFreezer` must use a
summary encoder that extracts trajectory geometry features (length,
curvature, step-to-step cosine, etc.) — not just the final latent state.
This is the load-bearing design decision for the freezer's summary stage.
