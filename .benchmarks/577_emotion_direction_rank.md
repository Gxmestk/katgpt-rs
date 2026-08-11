# Benchmark 577: the shipped emotion directions are rank 2 of 6 — Benchmark 575's verdict does not transfer (Issue 581 T5')

> **✅ SUPERSEDED 2026-08-11 — the shipped directions are now rank 4 of 6, zero collinear pairs.**
> The rank-2 finding below was accurate at measurement time (2026-08-10) against
> the v1 `extract_emotion_directions`. Two landings raised the rank and eliminated
> both anti-collinear pairs:
> 1. **Benchmark 584** (T10, riir-ai Issue 582) added Fearful/Aggressive
>    scenarios → recorded state rank 2 → 4; `fear`/`anger` became non-zero.
> 2. **Commit `f07bb097`** (riir-ai, Issue 581 T7 closeout) promoted the v2
>    corrections (arousal ← energy median split; desperation ← desperation-scalar
>    median split) into the **canonical** `extract_emotion_directions`. The
>    `emotion_directions_v2` feature now gates only a diagnostic wrapper.
>
> | metric | v1 (this benchmark) | canonical (post-`f07bb097`) |
> |---|---|---|
> | rank | 2 of 6 | **4 of 6** |
> | zero rows | `fear`, `anger` | **none** |
> | collinear pairs | valence~arousal (−1.0), desperation~calm (−1.0) | **none** |
> | arousal vs energy | −0.88 (wrong sign) | **+0.91** |
> | desperation vs scalar | +0.81 | **+0.96** |
>
> The analysis below remains valid as the **historical record** of the v1 defect
> and the diagnosis that drove the T7 + T10 fixes. The "actionable consequences"
> (§1–5) are all now resolved. The remaining gap (4 of 6, not 6 of 6) is the
> recorded-state ceiling (4 independent mood-injection patterns), not a
> derivation defect — raising it requires richer scenarios (Tired/Worried/Friendly
> moods), which is content authoring.

**Date:** 2026-08-10
**Issue:** 581 T5' (removed 2026-08-10 per noise-reduction rule — DONE; full content preserved in Benchmarks 575 + 577 + 578)
**Supersedes the conditional in:** [Benchmark 575](575_sigmoid_argmaxability_audit.md)
**Harness:** `riir-ai/crates/riir-games-civ/src/civ/emotion/rank_audit.rs` +
`riir-ai/crates/riir-games-civ/tests/emotion_direction_rank.rs`, feature `direction_rank_audit`
**Reproduce:**
```bash
cargo test -p riir-games-civ --features direction_rank_audit \
    --test emotion_direction_rank -- --nocapture
```
Consumes `katgpt_core::matrix_rank` (shipped under `sigmoid_margin`) — rank is not
reimplemented. No GPU, no training, no new deps.

---

## The finding

Benchmark 575 audited the HLA affect bridge for the sigmoid argmaxability
bottleneck and returned an **exhaustive negative** over all `2^5 = 32` affect sign
combinations. That result was **conditional**: it established the structural claim
*"any `L ≤ d` with linearly independent directions is safe"* — it did not check the
directions that actually ship. Issue 581 T5' existed precisely because those are
derived from recorded HLA state and *"could come out near-degenerate."*

They do. **The shipped direction matrix is rank 2 of L = 6.**

```
emotion directions: rank 2 of L=6 (d=16); zero rows: fear, anger;
valence~arousal cos=-1.0000; desperation~calm cos=-1.0000
```

| direction | ‖·‖ |
|---|---|
| valence | 1.048564 |
| arousal | 1.048564 |
| desperation | 0.006398 |
| calm | 2.530599 |
| fear | **0.000000** |
| anger | **0.000000** |

## Three causes, all structural rather than data accidents

**1. `fear` and `anger` are hardcoded zero.** The extractor literally writes
`fear: vec![0.0; embed_dim]` and `anger: vec![0.0; embed_dim]` — only four
scenarios exist (Prosperity, Crisis, CalmRoutine, Betrayal), with no near-threat or
under-attack scenario. `anger` is documented as a placeholder; `fear` carries only
an inline comment. A zero direction projects to a constant `sigmoid(0) = 0.5`, so
**both scalars carry no information at all** — they are not weak signals, they are
absent ones.

**2. `arousal ≡ −valence` exactly** (`cos = −1.000000`). The two formulas use the
same two scenario groups with the sign swapped:

```
valence = mean(prosperity + calm) − mean(crisis + betrayal)
arousal = mean(crisis + betrayal) − mean(calm + prosperity)
```

This is collinear *by construction*, not by data coincidence — no dataset could
make these independent.

**3. `desperation` is anti-parallel to `calm`** (`cos = −1.000000`), with a norm
~400× smaller (0.0064 vs 2.53). `desperation = mean(late crisis) − mean(early
crisis)` comes out as a tiny multiple of `calm = mean(calm) − mean(crisis)`, which
indicates the recorded HLA scenario means lie essentially on one line in the 16-D
space.

## Why it matters: anti-collinear ≠ merely redundant

Both dependent pairs have **negative** cosine, and for a sigmoid gate that is the
consequential case:

```
sigmoid(⟨s, −v⟩) = 1 − sigmoid(⟨s, v⟩)
```

So an anti-collinear pair is perfectly **complementary**, not just duplicated:

- **high valence + high arousal is unreachable** — the pair sums to 1 by identity.
- **high desperation + high calm is unreachable**, likewise.
- `fear` and `anger` are pinned at exactly 0.5 forever.

Unreachable sign combinations are exactly the phenomenon Benchmark 575 set out to
look for. **575's clean verdict therefore does not transfer to the shipped
directions** — it was a correct statement about a full-rank matrix, applied to a
matrix that is rank 2.

To be precise about what is and isn't overturned: 575's *structural* claim stands
(independent directions at `L ≤ d` are safe), its arithmetic was not wrong, and the
bottleneck is still not an intrinsic property of the 8→5 shape. What fails is the
**precondition**. The affect bridge is not "clean because the shape is safe"; it is
degenerate for an unrelated reason — how the directions are derived.

## Rank is tolerance-sensitive, so the verdict is measured across tolerances

| absolute pivot tol | rank |
|---|---|
| 1e-9 | 4 |
| 1e-7 | 3 |
| 1e-6 | 3 |
| **1e-5 … 1e-2** | **2** |

The 3s and 4s are float residue from the near-zero rows, not real axes. Every
tolerance at or above `1e-5` agrees on **2**. Because `desperation`'s norm is ~400×
smaller than `calm`'s, an *absolute* tolerance is actively misleading here, so
`rank_audit` scales its pivot tolerance by the largest matrix entry
(`RANK_REL_TOL = 1e-4 × max|W|`), making the verdict scale-invariant. Pinned by a
dedicated test.

## What shipped (T5')

`EmotionDirections::rank_audit()` → `DirectionRankAudit { rank, n_directions,
n_features, zero_rows, collinear_pairs }`, plus `audit_directions_once()` called at
the end of `extract_emotion_directions`.

**It reports rather than panics, and that is a deliberate deviation from T5's
literal wording** ("assert `matrix_rank(W) == L`"). A hard assert would fire on
every civ run, since the degeneracy is shipped reality including two acknowledged
placeholders. Following the `capacity_audit` precedent in riir-neuron-db, the guard
is latched (once per process), `debug_assertions`-only, and feature-gated — zero
release cost. The alternative temptations were both worse: silently weakening the
check to make it pass, or breaking every run.

The audit separates **acknowledged** degeneracy (zero placeholder rows) from
**unexpected** degeneracy (`rank < informative_directions`, i.e. two *non-zero*
directions that are dependent). On the shipped set that flag is `true` — rank 2 vs
4 informative rows — because the valence/arousal and desperation/calm collinearity
is nowhere documented as intentional. That is the actionable signal, and conflating
it with the known placeholders would have buried it.

## Actionable consequences

1. **`arousal` should not be derived as the negation of `valence`.** As written it
   is not a second axis. A genuine arousal direction needs a contrast that is not
   the valence contrast with its sign flipped (e.g. high-activity vs low-activity
   states, orthogonalised against valence).
2. **`desperation` needs a contrast independent of `calm`.** Its current
   late-vs-early-crisis split yields a near-zero vector parallel to `calm`.
3. **`fear` and `anger` need their scenarios** (near-threat, under-attack) or should
   be removed from the public struct — shipping a field that is permanently 0.5
   invites consumers to treat it as signal.
4. **Any consumer reading all five affect scalars as independent is mistaken.**
   There are ~2 axes of real information. Worth checking the affect consumers
   before adding a sixth scalar.
5. **Benchmark 575 needs its conditional marked resolved-negative**, which this
   record does. The rank margin was never the risk; the derivation was.

## Scope limits

- Measured at `extract_emotion_directions_for_map(n_layers=2, embed_dim=8)` → `d=16`.
  Causes 1 and 2 are structural (hardcoded zeros; formula symmetry) and cannot vary
  with dimension. Cause 3 is data-dependent and could in principle differ at other
  dimensions or with different scenario data, though the identical `cos = −1.000000`
  suggests a one-dimensional scenario manifold rather than a coincidence.
- The audit reports rank and pairwise collinearity. It does **not** enumerate which
  of the 32 sign combinations are unreachable — that would be the natural follow-up
  now that the precondition is known to fail, and it belongs with Benchmark 575's
  harness rather than here.
- No claim is made that this degrades any measured downstream behaviour. It bounds
  *expressivity*: two anti-collinear pairs and two dead scalars mean the bridge
  cannot represent combinations the type signature implies it can. Whether any
  consumer needs those combinations is unmeasured.
