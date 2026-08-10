# Benchmark 579: Scenario generator Fearful/Aggressive moods (Issue 581 T10 resolution)

> Closes the T10 deferred follow-up from katgpt-rs Issue 581 (sigmoid argmaxability
> bottleneck audit). Implementation tracked in riir-ai Issue 582.

## What T10 asked for

Enrich the scenario generator to emit Fearful/Aggressive moods so `fear` and
`anger` directions become derivable. This was the only path to raising the
recorded state rank above 2 — the hard ceiling on direction independence.

## Three root causes found (measured, not assumed)

| # | Root cause | Evidence |
|---|---|---|
| 1 | **Tick mismatch** | `extract_emotion_directions_for_map` uses `ticks: 200`, but Betrayal's Aggressive shift was at tick **500** → unreachable. Histogram: Calm 2000, Joyful 1000, Urgent 1000 — Aggressive never occurs. |
| 2 | **Label/state disconnect** | Betrayal called `environmental_mood_triples(false, true, false, …)` which produces **Urgent + Worried** triples. The `mood_type` label said Aggressive, but the HLA state reflected Urgent+Worried — the Aggressive role embedding (slot 0) was never injected. |
| 3 | **No Fearful scenario** | `environmental_mood_triples` never emits Aggressive or Fearful triples. No scenario produced Fearful mood at all. |

A fourth gap: v2 reported mood availability but never **extracted** fear/anger
directions even when moods were present.

## Fix (riir-ai Issue 582)

1. **Tick scaling.** `scenario_mood_context`, `tick_morality`, `scenario_energy`
   now take `total_ticks: u64`. Phase transitions use `total_ticks / 2` instead
   of hardcoded 500. Fixes root cause #1.

2. **Betrayal injection fix.** Post-midpoint Betrayal now injects an actual
   `MoodTriple { mood_type: Aggressive, intensity: 0.8 }` instead of
   `environmental_mood_triples(…)`. Fixes root cause #2.

3. **New `Scenario::MonsterRaid`.** Emits sustained Fearful mood with a Fearful
   MoodTriple (slot 1). High desperation, declining energy. Fixes root cause #3.

4. **v2 extraction of fear/anger.** When Fearful/Aggressive moods are present,
   v2 now extracts:
   - `fear ← mean(Fearful records) − mean(Calm records)`
   - `anger ← mean(Aggressive records) − mean(Calm records)`

5. **`extract_emotion_directions_for_map`** now runs 5 scenarios (added MonsterRaid).

## Measured results

| Metric | Before T10 | After T10 |
|---|---|---|
| v2 direction rank | 2 of 6 | **4 of 6** |
| Zero rows | `fear`, `anger` | **none** |
| Collinear pairs (v2) | valence~arousal (−1.0), desperation~calm (−1.0) | **none** |
| Recorded state rank | 2 of 16 | **4 of 32** |

The rank doubled (2 → 4) and ALL collinear pairs vanished. The v2 rederivations
(arousal from energy, desperation from desperation-scalar) combined with T10's
new moods eliminated every degeneracy the rank audit could detect.

The remaining gap (4 of 6, not 6 of 6) is the **recorded state rank ceiling** —
the HLA state varies through 4 independent mood-injection patterns (Calm,
Joyful, Urgent, Fearful+Aggressive share dimensions). Reaching rank 6 would
require even more diverse scenarios, but 4 of 6 with zero collinearity is a
major structural improvement.

## GOAT gate

| Gate | Status |
|---|---|
| G1 (correctness) | ✅ PASS — 1454 lib tests (was 1451; +3 new T10 tests) |
| G2 (perf) | ✅ PASS — `extract_emotion_directions_for_map` cost: ~5 × 5 × 200 = 5000 HLA updates (was 4000); negligible vs map generation |
| G3 (no-regression) | ✅ PASS — all existing tests pass unchanged; `test_run_betrayal_scenario_shift` still verifies the midpoint transition |
| G4 (alloc-free) | ✅ N/A — scenario generator runs once at map init, not per-tick |

## Tests shipped

- `fearful_and_aggressive_moods_appear_in_extraction_data` — regression guard
  for root causes #1 + #3
- `v2_extracts_nonzero_fear_and_anger_directions` — verifies fear/anger
  directions are non-zero + reported as derived
- `rank_audit_no_longer_reports_fear_anger_as_zero_rows` — verifies rank ≥ 4 +
  zero collinear pairs

## What this does NOT do

- **v1 `extract_emotion_directions` unchanged.** Fear/anger stay zero in v1.
  This is intentional — v1 is the safe default; v2 is opt-in
  (`emotion_directions_v2` feature). Promotion of v2 to default-on remains a
  separate decision (changes output for every caller).
- **No new HLA dimension.** The embed_dim is unchanged; the improvement is
  purely from scenario diversity + correct mood injection.

## Files changed (riir-ai)

- `crates/riir-games-civ/src/civ/emotion/mod.rs` — Scenario enum, scenario
  generator functions, extract_emotion_directions_for_map
- `crates/riir-games-civ/src/civ/emotion/directions_v2.rs` — v2 fear/anger
  extraction + tests
- `crates/riir-games-civ/src/civ/emotion/tests.rs` — updated call sites for
  new function signatures
