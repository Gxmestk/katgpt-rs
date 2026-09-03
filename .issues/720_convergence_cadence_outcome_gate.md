# Issue 720 — Convergence-cadence outcome gate: churn plateau = escalate (damp/deliberate/restart), decay = commit; not just halt

**Status:** OPEN — filed 2026-09-03, poc/proof task (analysis-sourced; no consumer wiring yet)

> **Source:** Rodrigues & Kang, "Dissecting Hierarchical Reasoning Models: A Mechanistic Study" (ICML 2026 Mech Interp Workshop; alphaxiv `2609.dissecting-hierarchical-reasoning-models`) — distilled in `.research/529_HRM_Mechanistic_Dissection.md`. **Consumers:** `GainCostLoopHalter` (Plan 304 / Research 282 — halt semantics extension), `forward_looped` deep runs (Issue 717 T1/T2 harness — the detector 717's damping knob lacks), per-NPC belief loops (`katgpt-sense` `evolve_belief` — deliberation/settle consumers in riir-mmorpg-examples Issue 054 L2).

## Why

The paper's Finding 4: on HRM's recurrent refinement, the windowed update-magnitude trajectory classifies outcome long before the final step — solved runs' ‖Δz_H‖ decays (0.30 by step 7-8, cos → 0.998, state norm GROWS to 9.4) while failed runs plateau HIGH (1.46, ~4.9×, cos stalls 0.97, norm stalls 6.8). n=93/107. Signal-diff (Research 529 §table):

- `GainCostLoopHalter` consumes step-size ‖Δh‖ but only for **HALT** (decay = concavity stop; growth = expansion stop). It has no OUTCOME read and no escalation arm. A run that plateaus at high ‖Δh‖ eventually halts, but the caller cannot distinguish "nothing left to gain" from "stuck churning — abort and try something else".
- Issue 717 ships (T3/T4, when landed) damping + tangential knobs with the rule "don't damp unless inference already degrades" — but has no degradation DETECTOR. Cadence is the detector.
- `surprise_norm` / `DerivativeCuriosity` read churn as *novelty* (explore), never as *failure* (abstain/escalate).
- riir-neuron-db `can_freeze` gates consolidation on output convergence (validated by this paper's finding) but is measure-time, not a live predictor.

The gap in one sentence: **"plateau-high" is diagnosable from deltas we already compute, and it warrants different ACTIONS per consumer — damp (lt2), deliberate (NPC), restart-with-new-conjecture (CGSP) — none of which a halt-only signal can express.**

Constraints carried from the source + cousins:
- ABSOLUTE update magnitude, never relative (Issue 717 T6 growing-denominator trap).
- Windowed trajectory shape (decay vs plateau), not a single-step threshold.
- Near-orthogonal successive updates (cos_updates ≈ 0 in the paper) → the churn is rotational; pair with 717-T4 tangential decomposition before choosing radial damping.
- Classification ≠ anti-cheat: this is a think-brain signal, never a sync/raw surface (AGENTS.md domain rules).

## Tasks

- [x] T1: `ConvergenceCadence` probe (katgpt-core, feature-gated `cadence_gate`): zero-alloc ring of last-K update norms (‖Δh‖ or ‖Δbelief‖, caller-fed — the halter and `evolve_belief` both already have the delta in hand), emits `Settled { mag } | Churning { mag, plateau_len }` from decay-ratio + plateau detection. G1: bit-identical when feature off; G4 alloc-free. *(LANDED 2026-09-03, katgpt-rs `99920de2` — 12/12 feature-on tests incl. paper-shape fixtures + G4 0-alloc hot path + shuffled non-vacuity; default 1992/0 bit-unchanged; clippy 0 both states.)*
- [ ] T2: Falsifiable A/B on a controlled loop (defend-wrong, riir-poc): three arms on `forward_looped` T=64 — (a) no gate, (b) halt-only (shipped halter), (c) halt + cadence-escalation (on plateau-high: apply 717 damping / restart from perturbed state). Metric: accuracy-at-equal-or-less compute + abort-precision (cadence verdict vs ground-truth solved/failed on a suite with known outcomes). Non-vacuity: gate must FAIL when fed shuffled cadences.
- [ ] T3: NPC consumer sketch (riir-mmorpg-examples, Issue 054 L2 deliberation): belief-churn over the think window as an ALTERNATIVE/ADDITIONAL stuck trigger (indecision detection, generalizes position-stuck), + settled-belief early-commit (skip think cycle when no new evidence and cadence settled). Gated, default-off; measure think-tick savings + deliberation precision on the 1000-NPC harness.
- [ ] T4: Doc pins: (a) GainCostLoopHalter docs gain the outcome-semantics note (halt ≠ classify); (b) Research 529's three-law combo (absolute Δ / windowed shape / tangential-first) recorded at the probe site; (c) cross-link Issue 717 (its detector) and riir-neuron-db `can_freeze` (its consolidation-side sibling).
- [ ] T5: `- [-]` deferred unless T2 lands positive: CGSP restart-with-new-conjecture arm (DerivativeCuriosity owns the explore axis; only add the *abandon* axis if T2 shows explore-alone is insufficient).

## References

- Research 529 (this paper's distill), Issue 717 (damping/tangential/f32 + residual trap — T1 here feeds its T3/T4), Plan 304/Research 282 (halter), Plan 277 (surprise/derivative curiosity), Plan 108 (forward_looped), riir-neuron-db `can_freeze`, riir-mmorpg-examples Issue 054 (L2 deliberation)

## Summary

**(1) Original task:** file the convergence-cadence extraction from the HRM dissection paper.
**(2) Accomplished:** issue filed with signal-diff against halter/surprise/can_freeze/717, constraints (absolute Δ, windowed, tangential-first), T1-T5.
**(3) What remains:** T2 (riir-poc A/B), T3 (mmorpg Issue 054 L2 consumer), T4 doc pins (module-level half DONE in `99920de2` — the three laws + signal-diff + cross-links live at the probe site; halter-doc note (a) + can_freeze cross-link (c) remain), T5 deferred.
**(4) Active plan state:** this issue (OPEN — T1 DONE); Research 529 (RECORD); Issue 717 (OPEN — its detector; sibling lane landing T1-T4).
