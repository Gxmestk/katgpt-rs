# Issue 197: HLA Naming Collision — Residual Per-NPC Constants (Follow-up to 195)

## Context

Issue 195 (CLOSED) renamed per-NPC "HLA" → "belief" across `katgpt-rs` + `riir-ai`
for **fields, methods, and local identifiers**. The rename was scoped to the
grep-confusing identifiers (the `evolve_hla` / `.hla()` / `hla_state` class
that collided with Transformer-attention `forward_hla` / `HlaLayerState`).

Issue 195's summary explicitly listed CamelCase per-NPC types
(`HlaKarcConfig`, `HlaProjection`, etc.) as "out of original scope — compile
clean, don't cause the collision." This follow-up tracks the **residual
per-NPC SCREAMING_SNAKE_CASE constants** that were missed in the same scope
decision and remain stale today.

## Discovery path

This session (2026-07-28) discovered that `tests/bench_277_unified_surprise_bus.rs`
**failed to compile** under its feature combo (`temporal_deriv sense_composition
delta_mem collapse_aware_thinking cgsp`) because it still called
`state.inject_hla_delta(delta)` and `state.evolve_hla()` — the methods were
renamed but the test was missed. Fixed inline (commit below); the residual
constant renames are tracked here.

## What remains (per-NPC, SCREAMING_SNAKE_CASE, public API)

### Family A — `katgpt_core::latent_steering` (the 8-dim affect axis constants)

These ARE per-NPC belief vocabulary — valence/arousal/desperation/calm/fear
are exactly the 5 synced affect scalars from AGENTS.md. They were missed
because the Issue 195 grep keyed on `evolve_hla` / `.hla()` shapes, not on
the `HLA_*` constant family.

| Current | Proposed | Consumers |
|---|---|---|
| `pub const HLA_VALENCE: usize = 0` | `VALENCE_AXIS` / `BELIEF_VALENCE` | re-exported at `katgpt_core::` root |
| `pub const HLA_AROUSAL: usize = 1` | `AROUSAL_AXIS` / `BELIEF_AROUSAL` | re-exported at `katgpt_core::` root |
| `pub const HLA_DESPERATION: usize = 2` | `DESPERATION_AXIS` / `BELIEF_DESPERATION` | re-exported at `katgpt_core::` root |
| `pub const HLA_CALM: usize = 3` | `CALM_AXIS` / `BELIEF_CALM` | re-exported at `katgpt_core::` root |
| `pub const HLA_FEAR: usize = 4` | `FEAR_AXIS` / `BELIEF_FEAR` | re-exported at `katgpt_core::` root |
| `pub const HLA_DIM: usize = 8` | `BELIEF_DIM` (8-affect) | re-exported at `katgpt_core::` root; also referenced in `conformal/mod.rs` doc comment + `conformal_alloc_check.rs` test comment |

**Naming collision risk:** `BELIEF_DIM` already exists in
`riir_engine::poincare_bridge::BELIEF_DIM` (renamed from `HLA_DIM` in Issue
195 — that's the 8-dim Poincaré adapter belief dim, same semantic). If both
are renamed to `BELIEF_DIM`, they would shadow at the crate root if both are
re-exported. Resolution options:
- (a) Prefix: `BELIEF_AFFECT_DIM` (latent_steering) vs `BELIEF_DIM` (poincare)
- (b) Put `latent_steering::BELIEF_DIM` behind a submodule re-export only,
  keeping the root re-export name distinct
- (c) Unify: the Poincaré `BELIEF_DIM` and latent_steering `HLA_DIM` are both
  8 — they may be the same concept (the 8-dim per-NPC belief vector). If so,
  both should reference one canonical constant.

**Recommendation:** investigate (c) first — if the two are semantically the
same 8-dim vector, unify them under `BELIEF_DIM` in one place and re-export.
If they are distinct (different consumers, different invariants), use (a).

### Family B — `katgpt_core::cgsp::types::DEFAULT_HLA_DIM` (the CGSP 64-dim curiosity direction)

| Current | Proposed | Consumers |
|---|---|---|
| `pub const DEFAULT_HLA_DIM: usize = 64` | `DEFAULT_BELIEF_DIRECTION_DIM` / `DEFAULT_CURIOSITY_DIM` | re-exported at `katgpt_core::cgsp::` + `katgpt_core::` root; mirrored in `riir_engine::engram_runtime::local_cache::DEFAULT_HLA_DIM` with an explicit cross-repo invariant test (`cross_repo_default_hla_dim_matches`) |

**Cross-repo invariant:** the local_cache mirror has a test that asserts
parity with the katgpt-core source-of-truth. Renaming requires coordinated
changes in both repos + updating the test name + comment.

This is a **public API breaking change** — downstream consumers
(`riir-engine`, `riir-games`, etc.) reference `DEFAULT_HLA_DIM` directly in
30+ places (struct array sizes, function signatures, bench constants).

### Family C — CamelCase per-NPC types (carry-over from Issue 195 summary)

Not the focus of this issue, but listed for completeness. ~30+ types embed
"Hla" in CamelCase (`HlaKarcConfig`, `HlaProjection`, `HlaSleepTimeScratch`,
`HlaAxis`, `HlaAugmentedTick`, `HlaDeltaStepLocalizer`, `HlaTilrConfig`,
`HlaTilrState`, `HlaSnapshot`, `HlaCommittedBlend`, `HlaStateSpace`,
`HlaCuriosityDirection`, `MotorGatedHlaConfig` [renamed to
`MotorGatedBeliefConfig` in 195], etc.). These are per-NPC types that would
benefit from renaming for full consistency. They don't cause the grep
collision (CamelCase, not the `hla` field/method), but they're lexical noise
against the rename.

## Scope decision for THIS issue

This issue tracks the **discovery** of the residual rename work. It does
NOT schedule the work — the renames are:

1. **Public API breaking changes** (constants re-exported at crate roots)
2. **Cross-repo coordinated** (DEFAULT_HLA_DIM has an invariant test)
3. **Semantic investigation needed** (BELIEF_DIM collision in Family A)

The right next step is a **plan** (not direct implementation) that:
- Audits each constant's consumers (grep across all 7 repos)
- Decides the naming scheme (option a/b/c above for Family A)
- Schedules the cross-repo rename as one atomic change
- Updates the cross-repo invariant test
- Updates all docs that reference the old names

## What this session actually fixed (commit `TBD`)

- `tests/bench_277_unified_surprise_bus.rs` — fixed compile regression:
  `inject_hla_delta` → `inject_belief_delta`, `evolve_hla` → `evolve_belief`,
  + 4 comment/print-string updates (HLA → belief in the F1 section header,
  the per-tick comment, the section divider, and the summary println). This
  was a real regression — the test would not compile under its feature combo.
- `tests/claim_rubric_test.rs::r287_s4_hla_evolve_is_l1` — updated claim
  text + doc comment to reference `evolve_belief` (the current name).
  Test function name kept as-is for report continuity (it maps to R287 §4
  row 6, a historical research note reference).
- `tests/bench_307_claim_rubric_goat.rs` — same claim text update in the
  GOAT gate fixture.

## Verification (this session)

- `cargo check --workspace --all-features` — clean
- `cargo check --tests --features 'temporal_deriv sense_composition delta_mem collapse_aware_thinking cgsp'` — clean (the previously-broken combo)
- `cargo test --test claim_rubric_test` — 17/17 pass
- `cargo test --test bench_307_claim_rubric_goat` — 1/1 pass
- `cargo test --features '...' --test bench_277_unified_surprise_bus` — 1/1 pass

## Status

**OPEN** — tracking issue for the residual rename. Not scheduled for
immediate implementation; requires a plan with cross-repo coordination +
public API migration strategy.

## References

- Issue 195 (CLOSED + REMOVED) — the original rename. See commit history
  on `develop` in both repos for the full change set.
- AGENTS.md §"Latent vs Raw Space Rules" — the 5 synced affect scalars
  (valence/arousal/desperation/calm/fear) that `HLA_VALENCE` etc. index.
