# Benchmark 578: Linking-Fold interaction (T6) + affect consumer audit (T8) — Issue 581 closeout

**Date:** 2026-08-10
**Issue:** [581](../.issues/581_sigmoid_argmaxability_bottleneck_audit.md) T6 + T8 (closeout)
**Sibling records:** [Benchmark 575](575_sigmoid_argmaxability_audit.md) (T1–T4), [Benchmark 577](577_emotion_direction_rank.md) (T5')
**Code touched:** `riir-ai/crates/riir-games-civ/src/civ/emotion/mod.rs` — fixed a stray
`#[cfg(feature = "emotion_mux")]` on `pub mod rank_audit;` (it was accidentally masked by
`emotion_mux` being default-on; the module has no `emotion_mux` dependency).

---

## T6 — Does Plan 410's Linking-Fold restore any lost argmaxability?

**No. Zero combinations are recovered.** The fold and the bottleneck act on different objects.

### Why, precisely

Plan 410's coordinate-fold (`fold_projection_into`) is a **state-space** operation — it
maps the latent vector in place:

```
state[i] ← center[i] + |state[i] − center[i]|
```

The sigmoid argmaxability bottleneck (Benchmark 575 / 577) is a **direction-matrix**
property — it depends on the rank and pairwise collinearity of the rows of `W`, the
projection matrix whose rows are `valence, arousal, desperation, calm, fear, anger`.

A state-space map `f` (folded or not) cannot change the collinearity of the direction
rows. For any state transformation `f: ℝ^d → ℝ^d` and any two direction rows `v_a, v_v`:

```
v_a = −v_v  ⟹  ⟨f(s), v_a⟩ = ⟨f(s), −v_v⟩ = −⟨f(s), v_v⟩
```

The two projections remain **exact negatives** regardless of `f`. The sigmoid identity
`σ(−x) = 1 − σ(x)` then forces `arousal = 1 − valence` for every state, folded or not.
The "both high" combination is unreachable either way. The same argument applies to
the `desperation ≡ −calm` pair.

For the zero rows (`fear`, `anger`): a zero direction projects to zero on every state,
folded or not. `σ(0) = 0.5` is the constant; the fold does not give the zero row a
non-zero projection.

### The three causes from Benchmark 577, re-checked against the fold

| Cause | Fold can fix? | Why |
|---|---|---|
| `fear`, `anger` hardcoded zero rows | **No** | Zero dot-product is zero regardless of state transform |
| `arousal ≡ −valence` (exact sign-flip) | **No** | Collinearity is in the direction vectors, not the state; any `f` preserves `⟨f(s), −v⟩ = −⟨f(s), v⟩` |
| `desperation` anti-parallel to `calm` | **No** | Same argument as above |

The fold's job (Plan 410) is to break topological linking between two latent clusters
by creating a local extremum in the activation — unlinking manifolds so a monotonic
classifier can separate them. That is a different problem from making a rank-deficient
projection matrix full-rank. They share only the broad theme of "monotonic activation
limitations"; the mechanisms and the fixes are disjoint.

### What the fold *does* do in this stack

The fold ships default-on (`linking_fold_fold` in katgpt-core) as a pre-projection
latent correction when the linking detector fires. It is complementary to the v2
direction fix (Issue 581 T7), not a substitute: v2 fixes the **directions**, the fold
fixes the **state**. A production NPC tick could apply both — fold the state, then
project onto v2 directions. Neither makes the other redundant.

---

## T8 — Affect consumer audit: who reads the five scalars as independent?

**Scope.** The argmaxability bottleneck (Benchmark 577) limits what the
**direction → sigmoid → scalar** projection layer can express. After projection, the
runtime emotion systems (feeling-brain heal-back, grudge bridge, emotion MUX, animal
bridges) modify the scalars via non-direction paths. Consumers of the **runtime
scalars** are therefore reading a mix of (degenerate direction projection) +
(non-direction runtime inputs). The audit below covers both the direction-only and
the full-runtime paths.

### The affect scalar types in the stack

| Type | Crate | Fields | Fear/Anger source |
|---|---|---|---|
| `EmotionDirections` | riir-games-civ | `valence, arousal, desperation, calm, fear, anger` (6) | **direction projection** (zero → σ(0)=0.5) |
| `EmotionProfile` | riir-games-civ | same 6 + `love, trust, gratitude` (MUX) | **direction projection** + `set_anger_from_grudge` + `decay_anger` |
| `EmotionReading` | katgpt-pruners | `valence, arousal, desperation, calm` (4) | no fear/anger — not exposed |
| `NpcEmotionScalars` | riir-engine | `valence, arousal, desperation, calm, fear` (5) | runtime state (multiple input paths) |
| `AffectScalars` | riir-cognition-sdk | `valence, arousal, desperation, calm, fear` (5) | cross-boundary wire format (5 × f32 LE) |

### Consumer-by-consumer verdict

**1. `NpcEmotionScalars::curiosity_drive` (riir-engine/cgsp_runtime/types.rs)**
```rust
0.30 * arousal + 0.25 * valence + 0.20 * calm − 0.15 * fear
```
Uses 4 of 5 scalars in a weighted sum. Since the direction projection makes
`arousal ≡ 1 − valence` and `desperation ≡ 1 − calm`, the first three terms collapse
to `0.30·(1−v) + 0.25·v + 0.20·c = 0.30 + (−0.05)·v + 0.20·c` — two independent axes
(valence, calm), not four. The `−0.15·fear` term: fear from the direction path is 0
(NpcEmotionScalars stores the raw dot-product, not the sigmoid), so it contributes
nothing via the direction path. Non-direction runtime paths (feeling-brain baseline,
animal bridge) CAN set fear to non-zero, in which case the term is live.

**Verdict: not broken, but the direction-projection path contributes only 2 independent
axes where the formula implies 4. The scalar arithmetic is correct; the *information
content* is lower than the formula suggests.**

**2. `BeliefStateSpace::bucket` (riir-engine/cce_runtime/state_space.rs)**
```rust
w0·valence + w1·arousal + w2·desperation + w3·calm + w4·fear
```
Discretises the emotion state into a belief bucket. Same collapse as above: the linear
combination of 5 scalars has at most 2 degrees of freedom from the direction path.
Buckets that differ only in the collapsed dimensions are indistinguishable.

**Verdict: the bucketing has fewer effective bins than the 5-scalar type signature
implies. Not a correctness bug (the arithmetic is fine), but a capacity limit — the
belief space is 2-D at the direction layer, not 5-D.**

**3. `EmotionScalarsSpace::latent_distance` (riir-engine/cgsp_runtime/interpolation_geometry.rs)**
```rust
√(Δval² + Δaro² + Δdes² + Δcalm² + Δfear²)
```
Euclidean distance over 5 scalars. Anti-collinear pairs contribute their squared sum:
`(Δv + Δ(1−v))² = 1` — a constant that inflates every distance by √2 without adding
discriminative information. Zero-valued fear contributes nothing.

**Verdict: distances are inflated by the complementary-pair constants. Not wrong, but
the effective metric is 2-D, and the constant inflation is dead weight.**

**4. `EmotionAxis::project` (riir-engine/cgsp_runtime/stokes_validator.rs)**
Match arm over the 5 axes. A consumer selecting `EmotionAxis::Fear` to project onto
gets a constant (0 from direction). Selecting `Arousal` gets `−valence`.

**Verdict: any Stokes-validator consumer selecting Fear or Arousal as an independent
projection axis is reading a constant or a duplicate.**

**5. `EmotionProfile::set_anger_from_grudge` + `decay_anger` (riir-games-civ)**
These SET anger from the grudge memory, NOT from the direction projection. This is a
**non-direction bridge** — anger carries real information about accumulated grudge,
independent of the direction matrix rank.

**Verdict: anger is NOT dead at the runtime level — it has a grudge bridge. The
direction projection contributes σ(0)=0.5 (a constant floor), but `set_anger_from_grudge`
overrides it with real signal. This is the one case where a "zero direction" scalar is
rescued by a non-direction input path.**

**6. `compute_animal_emotions` (riir-games-civ)**
Animal fear comes from boids force projection (`sigmoid(dot(force_state, fear_dir))`),
NOT from the HLA emotion direction matrix. The `species_fear_direction` is a separate
6-D vector specific to the animal's boids state.

**Verdict: animal fear is a completely separate path — not affected by the HLA direction
rank deficiency at all.**

**7. `ReviewMetrics::record_emotion` (katgpt-pruners)**
Records `valence, arousal, desperation, calm` (4 scalars, no fear/anger) from the
pruner's `EmotionReading`. Uses fixed-point quantisation (`× 10000`) into atomics.

**Verdict: records 4 scalars but only ~2 carry independent info via direction
projection. The review metrics aggregate over many ticks, so the constant/anti-collinear
structure averages out. Not broken.**

### The honest summary

| Claim | Status |
|---|---|
| "5 affect scalars are 5 independent axes" | **False** at the direction layer — 2 independent axes (valence, calm) + 2 anti-collinear duplicates (arousal=1−valence, desperation=1−calm) + 1 constant (fear=0) |
| Consumers that sum/average the scalars are broken | **No** — the arithmetic is correct; the *information content* is lower than the type signature implies |
| Consumers that threshold on fear as an independent signal are mistaken | **Yes** via the direction path — `NpcEmotionScalars.fear` from projection is always 0; only non-direction bridges (feeling-brain, animal) make it non-zero |
| Anger is dead | **No** — `set_anger_from_grudge` is a non-direction bridge that populates anger from the grudge memory |
| Adding a 6th scalar would help | **Only if it has a non-direction input path or a direction derived from an independent contrast** — otherwise it collapses like arousal/desperation did |

### What was done about it

1. **v2 directions** (Issue 581 T7, opt-in `emotion_directions_v2`): fixes the
   `arousal` sign bug (anti-correlation → positive correlation with energy) and
   rederives `desperation` from its own scalar. Both changes are semantic corrections,
   not rank inflation — the rank stays 2 because the recorded HLA state is rank 2
   (3 distinct moods only). Promotion to default is a separate decision (changes
   output for every caller).
2. **`DirectionSourceReport`** (v2 module): explicitly reports `fear` and `anger` as
   **unavailable** — no contrast exists in the recorded data. Consumers can check
   `report.has_unavailable()` before treating a scalar as signal.
3. **Rank audit** (T5', `direction_rank_audit` feature): latched debug-only report
   that fires when the direction matrix is rank-deficient. Catches future regressions
   if the scenario generator (T10 follow-up) enriches the mood space and directions
   gain rank.
4. **No fields removed.** Removing `fear`/`anger` from the public struct would break
   the 5-scalar wire format (`AffectScalars::to_le_bytes`) and the `EmotionProfile`
   API. The fields stay, documented as permanently-constant via the direction path,
   with `anger` rescued by the grudge bridge and `fear` awaiting Fearful-mood scenarios
   (T10).

---

## Issue 581 closeout

All tasks resolved:

| Task | Status | Record |
|---|---|---|
| T1 — argmaxability decision procedure | ✅ DONE | `katgpt-types/src/simd/research.rs` (`sigmoid_margin`) |
| T2 — modelless diagnostic impl | ✅ DONE | `matrix_rank`, `argmaxable_witness`, `audit_argmaxable` in katgpt-core |
| T3 — exhaustive 32-combination audit (structural) | ✅ DONE | Benchmark 575 |
| T4 — other sigmoid sites (L=1, structurally safe) | ✅ DONE | Benchmark 575 |
| T5 — DFT output layer | − NOT NEEDED | No unargmaxable combination at full-rank shapes; deferred, then T5' showed the precondition fails |
| T5' — rank audit of shipped directions | ✅ DONE (NEGATIVE) | Benchmark 577 — rank 2 of 6 |
| T6 — Linking-Fold interaction | ✅ DONE (this benchmark) | Fold operates on state; collinearity is in directions → zero combinations recovered |
| T7 — fix the derivations | ✅ DONE (a, b) + PARTIAL (c) | (a)(b) v2 module fixes arousal sign + desperation; (c) fear/anger stay zero — reported unavailable, anger rescued by grudge bridge, Fearful-mood scenarios are T10 follow-up |
| T8 — affect consumer audit | ✅ DONE (this benchmark) | 2 independent axes at direction layer; consumers using sums/distances are correct but information-poor; fear-from-direction is constant; anger has grudge bridge |

**Deferred follow-up:** T10 — enrich the scenario generator to emit Fearful/Aggressive
moods so `fear` and `anger` directions become derivable. This is the only path to
raising the recorded state rank above 2, which is the hard ceiling on direction
independence.
