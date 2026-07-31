# Plan 467: QGF DualLeoOracle — Test-Time LEO+UVFA Q-Gradient Fusion

**Status:** DONE — G1–G4 PASS (mechanistic); G5 measured FAIL on synthetic data (Bench 553 in riir-ai, 2026-07-18) AND on civ real networks (Bench 558 in riir-ai, 2026-07-19). **POST-PLAN-500 UPDATE (2026-07-18):** riir-ai [Plan 500](../riir-ai/.plans/500_dual_leo_trainer_backprop_fix.md) fixed [Issue 554](../riir-ai/.issues/554_dual_leo_trainer_backprop_noop.md) — LEO last-layer now does real per-sample SGD. Re-ran T9/Bench 553 with the fix: same 0.00% / 0.50% numbers, but the meaning changes from "frozen-noise LEO behavior" to "real negative finding — DualLeo α-mix is actively harmful on synthetic data". Plan 500's T12 proves LEO loss decreases 32% on a learnable task. **POST-PLAN-507 UPDATE (2026-07-19):** the real-network G5 verdict has now LANDED — riir-ai [Plan 507](../riir-ai/.plans/507_civ_qgf_dual_leo_oracle_g5_real_network.md) + [Bench 558](../riir-ai/.benchmarks/558_civ_qgf_dual_leo_oracle_g5_real_network.md) measured QGF DualLeoOracle on civ real networks with trained CivLeoNet (Plan 505 v7) + untrained CivLeoUvfa: dual +2.69% vs single (35.68% → 36.64%) on civ action-prediction, missing the ≥3% gate by 0.31pp — fourth-axis stop rule triggered. [Research 322](../riir-ai/.research/322_civ_alternative_critic_post_stop_rule_verdict.md) then closed the "alternative critic" escape hatch (category-confused). The civ dual-LEO investigation is FULLY CLOSED. See [`.benchmarks/467_qgf_dual_leo_oracle_goat.md`](../.benchmarks/467_qgf_dual_leo_oracle_goat.md) + [riir-ai `.benchmarks/553`](../riir-ai/.benchmarks/553_qgf_dual_leo_oracle_g5.md) + [riir-ai `.benchmarks/558`](../riir-ai/.benchmarks/558_civ_qgf_dual_leo_oracle_g5_real_network.md) + [riir-ai Plan 500](../riir-ai/.plans/500_dual_leo_trainer_backprop_fix.md) + [Proposal 007 addendum](../.proposals/007_qgf_dual_leo_oracle.md).
**Branch:** `develop`
**Repo:** `katgpt-rs`
**Proposal:** [007 QGF DualLeoOracle](../.proposals/007_qgf_dual_leo_oracle.md)
**Fusion of:** Plan 268 (QGF `LeoHeadOracle`) × Plan 155 (`DualLeoMixer`) × Plan 460 (postmax lesson — linear-in-grad mix)
**Started:** 2026-07-18
**Completed:** 2026-07-18

## Context (the gap Plan 268 leaves)

Plan 268 ships two `QGradientOracle` impls in `crates/katgpt-core/src/qgf/oracles.rs`:

| Oracle | Feature | What it wraps |
|---|---|---|
| `LeoHeadOracle<H: LeoHead>` | `leo_all_goals` | A single LEO head; emits the per-action Q-slice for the selected goal. |
| `FlowFieldOracle` | `flow_field_nav` | An owned `FlowField`; emits the `(dx, dy)` flow vector at the queried cell. |

When both a LEO teacher AND a UVFA student are available, **there is no QGF oracle that uses both.** The dual-fusion machinery (`DualLeoMixer` trait, Plan 155) exists, but only `FlowFieldCache::get_or_compute_dual_postmax` consumes it — that's a different pipeline (potential-field navigation, not Q-gradient guidance). Proposal 007 closes this gap by adding a third `QGradientOracle` impl.

## Design invariant (Plan 460 lesson encoded by construction)

Plan 459 (pre-max dual fusion) failed because `max_a(·)` sat between the α-mix and the FFT consumer:

```
max_a (α·Q_leo[a] + (1-α)·Q_uvfa[a])  ≠  α·max_a Q_leo[a] + (1-α)·max_a Q_uvfa[a]
```

Plan 460 fixed this for flow fields by blending post-max potentials (linear in the FFT's input).

**Plan 467 encodes the same lesson as a design invariant for QGF.** The QGF `LeoHeadOracle` has no max-pool — the gradient IS the per-action Q-slice (`∇_a Q(s, a)[i] = Q(s, a_i)`). So a gradient-level α-mix is a pure linear combination:

```
grad_mix[i] = α · Q_leo(s, a_i) + (1-α) · Q_uvfa(s, a_i)
            = α · grad_leo[i] + (1-α) · grad_uvfa[i]
```

**No operator sits between the `DualLeoMixer::combine_into` and the QGF consumer.** This is encoded in the `DualLeoOracle` doc-comment.

## The primitive

```rust
#[cfg(all(feature = "leo_all_goals", feature = "dual_leo"))]
pub struct DualLeoOracle<H1, H2, M>
where
    H1: LeoHead, H2: LeoHead, M: DualLeoMixer,
{
    head_leo: H1,
    head_uvfa: H2,
    mixer: M,
    alpha: f32,
    goal_idx: usize,
}
```

Mirrors `LeoHeadOracle<H>` API (`new`, `head_leo`/`head_uvfa` accessors, `goal_idx`, `alpha`). `QGradientOracle` impl dispatches both heads' `all_goals_q` → `q_for_goal` slices into `mixer.combine_into`. Confidence inherits `LeoHeadOracle`'s default (1.0 — deterministic cached lookup).

## Feature gates

- `qgf_oracle` (always required for the `oracles` module)
- `leo_all_goals` (for `LeoHead` trait)
- `dual_leo` (for `DualLeoMixer` trait; also implies `leo_all_goals`)

Both `leo_all_goals` and `dual_leo` are already default-on in `katgpt-core`'s `default` feature set (see `Cargo.toml`); `qgf_oracle` stays opt-in (Plan 268 stance). The oracle compiles when all three are enabled.

## Constraint Check

- **Modelless mandate (katgpt-rs/AGENTS.md):** Affine α-blend of two cached Q-slices is deterministic, modelless, no gradients, no training. ✅
- **Sync-boundary rule:** The gradient is a latent Q-value vector; it feeds QGF's local drafter tilt and does NOT cross any sync boundary. ✅
- **Alloc-free hot loop (G4):** `q_gradient_into` writes into the caller-provided `&mut [f32]`; the `combine_into` call is also in-place. ✅
- **Feature gate discipline:** New module ships behind existing `leo_all_goals + dual_leo` features (both already in `default`). The `qgf_oracle` parent feature gates the whole `oracles` module. No new feature flag. ✅
- **`LeoHeadOracle` stays landed.** `DualLeoOracle` is a sibling, not a replacement. Single-head callers continue using `LeoHeadOracle`. ✅

## Tasks

- [x] T1: Add `dual_leo_oracle` module to `crates/katgpt-core/src/qgf/oracles.rs` after `FlowFieldOracle` (struct + `new` + accessors + `QGradientOracle` impl + doc-comment encoding the Plan 460 invariant).
- [x] T2: Add `pub use dual_leo_oracle::DualLeoOracle;` re-export, gated on `leo_all_goals + dual_leo`.
- [x] T3: Mirror `LeoHeadOracle`'s test suite + add dual-specific tests (α=1.0 bit-identity with `LeoHeadOracle` on the same LEO head; α=0.0 bit-identity with `LeoHeadOracle` wrapping UVFA head; Lc α=0.5 produces exactly `[α·leo[i] + (1-α)·uvfa[i]]`; out-of-range `goal_idx` → empty gradient + zero-filled buffer; `q_gradient_at`/`q_gradient_into` agree; confidence is 1.0).
- [x] T4: Update the module-level tier table in `crates/katgpt-core/src/qgf/oracles.rs` (lines 8-16) to add `DualLeoOracle` row.
- [x] T5: Update `qgf/mod.rs` tier table (around lines 38-42) to add `Hot | DualLeoOracle | dual_leo | 1.0` row.
- [x] T6: Run `cargo test -p katgpt-core --features qgf_oracle,leo_all_goals,dual_leo --lib` — all tests pass.
- [x] T7: Run `cargo clippy -p katgpt-core --features qgf_oracle,leo_all_goals,dual_leo --all-targets` — no new warnings.
- [x] T8: Write GOAT bench report `.benchmarks/467_qgf_dual_leo_oracle_goat.md` (G1–G4 results; G5 deferred).

## GOAT gate

- **G1 (correctness — bit-identity):**
  - α=1.0 (LeoOnly) → `DualLeoOracle` produces bit-identical gradients to `LeoHeadOracle` with the same LEO head + goal.
  - α=0.0 (UvfaOnly) → bit-identical to `LeoHeadOracle` wrapping the UVFA head.
  - Both verified by unit tests.
- **G2 (perf):** `DualLeoOracle` query ≤ 1.5× `LeoHeadOracle` query (median-of-3 trials of 1000-iteration batches, per Plan 460 perf-honesty lesson).
- **G3 (no-regression):** `cargo test -p katgpt-core --lib` passes (default features); `cargo test -p katgpt-core --features qgf_oracle,leo_all_goals,dual_leo --lib` passes (+ new tests).
- **G4 (alloc-free hot path):** `q_gradient_into` writes into the caller's buffer; no `Vec::new` in the into-path (verified by code inspection).
- **G5 (downstream task gain — MEASURED FAIL on both synthetic AND civ real-network data; investigation closed):** Attempted on riir-ai T7 Go puzzle harness (Bench 553, Issue 553, 2026-07-18): QGF+DualLeoOracle scored **0.00%** vs QGF+LeoHeadOracle **0.50%** — dual is WORSE. The G1 correctness invariant (b ≡ a) held bit-identically, confirming the mechanism is correct. **POST-ISSUE 554 CORRECTION (2026-07-18):** originally mis-attributed to "synthetic data has no signal"; actual root cause is riir-ai Issue 554 — `DualLeoTrainer::apply_leo_last_layer_update` is a `let _ = (grad, lr);` no-op stub, so LEO weights are bit-identical before vs after training (proven by T11 diagnostic test). The 0.50% baseline is argmax over frozen-noise LEO. **POST-PLAN-507 UPDATE (2026-07-19):** the real-network G5 measurement has now landed — riir-ai [Bench 558](../riir-ai/.benchmarks/558_civ_qgf_dual_leo_oracle_g5_real_network.md) measured QGF DualLeoOracle on civ real networks with trained CivLeoNet (Plan 505 v7) + untrained CivLeoUvfa: dual +2.69% vs single (35.68% → 36.64%) on civ action-prediction, missing ≥3% gate by 0.31pp — fourth-axis stop rule triggered. [Research 322](../riir-ai/.research/322_civ_alternative_critic_post_stop_rule_verdict.md) then closed the "alternative critic" escape hatch (category-confused — UQ primitives produce state forecasts, not per-action Q-gradients). The civ dual-LEO investigation is FULLY CLOSED. The oracle ships as opt-in and is documented as "mechanism complete, downstream gain unproven across synthetic + civ real-network regimes." Three re-open triggers: seal integration gain, new game domain positive G5, or Q-vs-forecast research breakthrough.

## What ships now (katgpt-rs) vs deferred

### Ships now — katgpt-core

- `DualLeoOracle<H1, H2, M>` struct in `crates/katgpt-core/src/qgf/oracles.rs` (new module `dual_leo_oracle`)
- `QGradientOracle` impl with `q_gradient_at` + `q_gradient_into` (inherits default `confidence = 1.0`)
- Module-level + `qgf/mod.rs` tier table entries
- Unit tests mirroring `LeoHeadOracle`'s test suite + dual-specific bit-identity tests
- Doc-comment encoding the Plan 460 "no operator between mix and consumer" invariant

### Deferred — riir-ai

- Consumer-side adapter wrapping real UVFA nets as `LeoHead` (Proposal 007 caveat 4) — **DONE in bench 553** as `UvfaAsLeoHead` (bench-local).
- Downstream task-quality gate G5 (Sudoku / DDTree / Bomber) — **MEASURED FAIL on synthetic Go puzzles (Bench 553) AND civ real networks (Bench 558).** Investigation closed per Research 322.
- Tuning α per consumer.
- Promotion decision from opt-in to "documented as recommended" — **BLOCKED on positive G5 measurement** (none in hand across two measurement regimes; reopen triggers: seal integration gain, new game domain positive G5, or Q-vs-forecast research breakthrough).

### Explicitly NOT shipped

- Per-goal runtime switching (`set_goal`) — Proposal 007 caveat 3; defer until consumer demand.
- Civ flow-field navigation (Proposal 028).
- UVFA network architecture / training.

## Connection to existing GOAT-proved work

| Plan / Issue | Status | Connection |
|---|---|---|
| Plan 155 (LEO All-Goals) | ✅ DEFAULT-ON SUPER GOAT | Source of `LeoHead` + `DualLeoMixer` traits. Plan 467 adds a **4th consumer** of `DualLeoMixer` (was 3: `QuestLeoScorer` + `get_or_compute_dual` + `get_or_compute_dual_postmax`; now 4). |
| Plan 268 (QGF) | ✅ (opt-in) | Source of `LeoHeadOracle` + `QGradientOracle` trait. Plan 467 adds `DualLeoOracle` as a sibling. |
| Plan 460 (post-max dual fusion) | ✅ DEFAULT-ON (recommended dual path) | Source of the "no operator between mix and consumer" design invariant. Plan 467 encodes it as a doc-comment. |
| Proposal 007 (this plan's parent) | ✅ SHIPPED Phase 1-2 (this plan); G5 deferred | Status updated from "draft" to "shipped Phase 1-2 (Plan 467); G5 still deferred to riir-ai". |

## TL;DR

Shipped `DualLeoOracle` as QGF's 3rd oracle — LEO+UVFA Q-gradient fusion at the gradient level. Plan 460's max-pool washout lesson is encoded as a design invariant (no operator between the `DualLeoMixer::combine_into` and the QGF consumer). G1–G4 PASS mechanistically; **G5 measured FAIL on both synthetic data (Bench 553) AND civ real networks (Bench 558)** — across two measurement regimes the dual-α-mix did not cross the quality gate. The correctness invariant (b ≡ a) held bit-identically on both, confirming the mechanism is correct. Research 322 closed the alternative-critic escape hatch (category-confused). Primitive stays opt-in with documented unproven G5; civ dual-LEO investigation FULLY CLOSED 2026-07-19.
