# Issue 188 — DualLeo: commit-then-fuse (sigmoid-gated) vs average-then-argmax

**Filed:** 2026-07-21
**Repo:** katgpt-rs
**Branch:** develop
**Status:** OPEN — design proposed, implementation in progress
**Parent:** [Proposal 007](../.proposals/007_qgf_dual_leo_oracle.md) (QGF DualLeoOracle)
**Related:**
- [Plan 467](../.plans/467_qgf_dual_leo_oracle.md) — `DualLeoOracle` (current average-then-argmax oracle)
- [Bench 553](../../riir-ai/.benchmarks/553_qgf_dual_leo_oracle_g5.md) — synthetic G5 FAIL (dual 0.00% vs single 0.50%)
- [Bench 558](../../riir-ai/.benchmarks/558_civ_qgf_dual_leo_oracle_g5_real_network.md) — civ real-network G5 FAIL (dual +2.69% vs ≥3% gate)
- [Research 322](../../riir-ai/.research/322_civ_alternative_critic_post_stop_rule_verdict.md) — closed the alternative-critic escape hatch
- [Plan 459](../.plans/459_flow_field_dual_leo_mixer_fusion.md) — pre-max fusion FAIL (the `ActingMode::Max` washout lesson)

## Problem

`DualLeoOracle::q_gradient_at` (Plan 467) fuses LEO teacher + UVFA student via a **linear α-blend then argmax** at the QGF gradient layer:

```text
grad_mix[a] = α · Q_leo[a] + (1-α) · Q_uvfa[a]
action*     = argmax_a grad_mix[a]
```

This is structurally the wrong fusion. Three failure modes (per Bench 553 / 558 evidence):

1. **Argmax-of-average ≠ average-of-argmax.** Linear blend produces a *compromise* Q-vector whose argmax can be an action *neither* head prefers. Worst-of-both-worlds at the decision boundary. (The Plan 459 `ActingMode::Max` lesson: any nonlinearity between the mix and the argmax breaks the α-weighting.)
2. **Noise-on-signal.** When one head is untrained (Bench 558's xavier-init UVFA), α-blending is *literally Gaussian noise injection* on the trained teacher's Q. Argmax over noisy-Q is a coin flip.
3. **No confidence gating.** A fixed α has no notion of "trust the head that's currently confident". If the teacher is sharp on this state and the student is diffuse, α=0.3 still injects 0.7·(diffuse student signal) into the mix.

G5 has FAILED on every measured axis: synthetic Go (−100%), untrained-real civ postmax (+3.3% vs ≥30% gate), trained-LEO-only-real civ postmax (+3.6% vs ≥30% gate), trained-LEO-only-real civ QGF (+2.69% vs ≥3% gate). Four-axis stop rule triggered; Research 322 closed the alternative-critic escape hatch.

## Hypothesis

Replace **average-then-argmax** with **commit-then-fuse**:

```text
a_t = argmax Q_teacher           # each head commits independently
a_s = argmax Q_student
Δconf = max(Q_teacher) - max(Q_student)
gate = σ(β · Δconf)              # sigmoid gate, NOT linear α
# action chosen from {a_t, a_s} only — never a compromise action
```

The β parameter spans both intuitions:

- **β = 0**: `gate = 0.5` always → reduces to current 0.5·α-mix (regression sanity check)
- **0 < β < ∞**: sigmoid-gated soft switch — user proposal #1 ("sigmoid instead of average")
- **β → ∞**: hard commit-then-fuse — user proposal #2 ("wrap up later instead of each")

Two structural advantages over the current oracle:

1. **Bounded worst case.** The chosen action is always either `a_t` or `a_s` — never a compromise action neither head prefers. Worst case ≤ `max(error_teacher, error_student)`.
2. **Saturating gate.** When teacher is much more confident than student (`Δconf >> 0`), `gate → 1` — fully trust teacher. Symmetric in the other direction. Linear α has no such saturation; it always blends.

This is **not** the same as `ActingMode::Max` (Plan 459's failed pre-max fusion). `Max` does *element-wise max of Q-vectors*, which still admits compromise actions. Commit-then-fuse does *argmax-then-pick-between-committed-actions*, which by construction cannot.

## Why this is not category-confused (vs the Plan 556 rejection)

[Plan 556](../.plans/556_karc_mitigations_open_primitives.md) §"What this plan does NOT do" rejected "dual LEO+KARC fusion" as category-confused: `DualLeoOracle` is a Q-gradient oracle, KARC is an HLA forecaster — they produce different objects and cannot be blended at the gradient layer.

This issue does NOT propose fusing KARC + LEO. It proposes a **different fusion *mechanism*** between two homogeneous LEO heads (LEO + UVFA, both produce `R^{G×A}` Q-tables). The category is consistent. The mechanism change is from "linear blend Q-vectors" to "commit each head's argmax, sigmoid-gate between them".

## Implementation plan

### T1 — `CommitDualLeoOracle<H1, H2>` (katgpt-rs)

New oracle in `crates/katgpt-core/src/qgf/oracles.rs`, sibling to `DualLeoOracle`. Gated on `leo_all_goals + dual_leo` (same as `DualLeoOracle`).

```rust
pub struct CommitDualLeoOracle<H1: LeoHead, H2: LeoHead> {
    head_leo: H1,
    head_uvfa: H2,
    beta: f32,        // gate sharpness (0 = average, ∞ = hard commit)
    goal_idx: usize,
}

impl<H1: LeoHead, H2: LeoHead> QGradientOracle for CommitDualLeoOracle<H1, H2> {
    // q_gradient_at: pick head via sigmoid(max(Q_t) - max(Q_s)),
    // return that head's full Q-slice (per-action) as the gradient.
    // This preserves the QGF consumer contract: gradient = per-action Q-slice.
}
```

**Plan 460 invariant preservation:** no operator between the gate decision and the QGF consumer. The gate picks *which head's* Q-slice to return — the slice itself is untouched. The consumer sees a pure `q_leo` or `q_uvfa`, never a blend.

**Confidence:** Still 1.0 (deterministic lookup + sigmoid). Matches `DualLeoOracle`'s "determinism = quality" lie (Proposal 007 caveat 5).

### T2 — G1 correctness test (katgpt-rs)

- `β = 0` reduces to current `DualLeoOracle` at α=0.5 (within f32 epsilon).
- `β → ∞` with `Q_t == Q_s` → bit-identical to single-head `LeoHeadOracle`.
- Out-of-range goal_idx → empty gradient (mirrors existing oracle's edge case).

### T3 — G5 measurement on synthetic Bench 553 setup (riir-ai)

Reuse the T9 harness (`bench_leo_game_training.rs` T9 test). Add a T11 variant using `CommitDualLeoOracle` with β ∈ {0, 1, 4, 16, 64}. Compare solve rate vs single-head baseline.

**Pass criterion:** commit-then-fuse solve rate ≥ single-head solve rate (no regression), AND ≥ current DualLeo rate (dominance).

**Stretch pass:** ≥ 0.5% absolute solve-rate gain over single-head on synthetic.

### T4 — G5 measurement on civ real-network Bench 558 setup (riir-ai)

Reuse the T10 harness. Add a T12 variant. β-sweep as above.

**Pass criterion:** dual ≥ single + 3% (the gate current DualLeo misses by 0.31pp).

## What this does NOT do

- ❌ **Does NOT fuse KARC + LEO.** Plan 556's category-confusion rejection stands. This is purely a different fusion mechanism between two LEO-family heads.
- ❌ **Does NOT close the unmeasured paired-trained-UVFA cell.** Bench 558's setup (trained LEO + xavier UVFA) is reused as-is. If commit-then-fuse clears G5 on that setup, great; if not, the paired-trained-UVFA cell remains the only unmeasured axis.
- ❌ **Does NOT change `DualLeoOracle` itself.** `CommitDualLeoOracle` is a sibling, not a replacement. If it dominates, we can deprecate `DualLeoOracle` later.

## Decision tree

- **T3 PASS + T4 PASS** → `CommitDualLeoOracle` is the new recommended dual path. Document in Proposal 007 addendum. Keep `DualLeoOracle` as legacy opt-in.
- **T3 PASS + T4 FAIL** → commit-then-fuse is correct in principle but doesn't rescue civ. Investigate whether paired-trained-UVFA (the unmeasured cell) is the blocker — if so, that's a riir-train task.
- **T3 FAIL** → commit-then-fuse is also structurally wrong. Stop. The dual-LEO mechanism is dead and should be archived.

## Acceptance

- [ ] T1: `CommitDualLeoOracle` impl lands, gated on `leo_all_goals + dual_leo`.
- [ ] T2: G1 correctness tests pass (β=0 reduces to average, β→∞ reduces to commit, edge cases handled).
- [ ] T3: Bench 553 synthetic re-run with β-sweep, results recorded.
- [ ] T4: Bench 558 civ real-network re-run with β-sweep, results recorded.
- [ ] Decision tree resolved; outcome recorded in this issue + Proposal 007 addendum.

## TL;DR

Current `DualLeoOracle` does `argmax(α·Q_t + (1-α)·Q_s)` — structurally a *compromise* action that can be worse than either head's top pick. User's intuition: use sigmoid gate (not linear α) and wrap up *after* commit (not blend *before* argmax). New `CommitDualLeoOracle` picks between `argmax(Q_t)` and `argmax(Q_s)` via `σ(β·Δconf)` — bounded worst case, saturating gate, β sweeps both proposals. File + try; if synthetic + civ both still fail, dual-LEO is dead and we archive it honestly.
