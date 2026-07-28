# Issue 198 — Residual `*_HLA_DIM` constants (Issue 197 follow-up)

## Origin

Follow-up to Issue 197 (closed 2026-07-28). Issue 197 renamed the bare
`HLA_DIM` / `DEFAULT_HLA_DIM` / `HLA_VALENCE` / `HLA_AROUSAL` /
`HLA_DESPERATION` / `HLA_CALM` / `HLA_FEAR` constants to `BELIEF_*` /
`DEFAULT_BELIEF_DIRECTION_DIM` across the 7-repo stack and verified
"zero remaining `HLA_DIM`-family references in `.rs` files."

That verification was **technically correct for the bare tokens** but
**missed the suffixed / prefixed variants** — local constants named
`<CONTEXT>_HLA_DIM` or `HLA_DIM_<CONTEXT>` that derive from the same
8-dim per-NPC belief/affect vector. A re-audit on 2026-07-28 found 38
genuine source-level occurrences in riir-ai that Issue 197 did not touch.

## Scope (verified 2026-07-28 via grep excluding `target/`)

### In scope — `*_HLA_DIM` / `HLA_DIM_*` local constants (riir-ai, 38 occurrences)

| File | Constant(s) | Count | Notes |
|---|---|---|---|
| `crates/riir-engine/src/latent_functor/crowd_regime_post_tick.rs` | `BELIEF_DIM as CROWD_HLA_DIM` (test alias) | 11 | Test-only import alias; rename to `CROWD_BELIEF_DIM` |
| `crates/riir-engine/src/cognitive_branches_runtime/metamemory_bridge.rs` | `PIPELINE_HLA_DIM = 8` | 6 | Local const; rename to `PIPELINE_BELIEF_DIM` |
| `crates/riir-engine/src/committed_blend/functor_bridge.rs` | `_HLA_DIM_REF` | 1 | Compile-time guard; rename to `_BELIEF_DIM_REF` |
| `crates/riir-engine/src/sheaf_cce_bridge.rs` | `_HLA_DIM_ASSERT` | 1 | Compile-time guard; rename to `_BELIEF_DIM_ASSERT` |
| `crates/riir-games-civ/src/civ/map_builder.rs` | `FLAT_HLA_DIM = 64` | 3 | 64-dim CGSP direction (matches `DEFAULT_BELIEF_DIRECTION_DIM`); rename to `FLAT_BELIEF_DIRECTION_DIM` |
| `crates/riir-games-civ/src/civ/map_tick/cgsp_curiosity.rs` | `MAG_HLA_DIM = 8` + comments | 7 | 8-dim MAG direction; rename to `MAG_BELIEF_DIM` |
| `crates/riir-games-civ/src/civ/map_tick/mag_mining.rs` | `MAG_HLA_DIM = 8` + `npc_to_hla`/`hla_to_npc` fn names + comments | 9 | Same const + two helper fns; rename const + helpers |

**Naming decisions:**
- `*_HLA_DIM` where the value is 8 (per-NPC affect axis) → `*_BELIEF_DIM`
- `*_HLA_DIM` where the value is 64 (CGSP direction space) → `*_BELIEF_DIRECTION_DIM`
- Helper fns `npc_to_hla` / `hla_to_npc` → `npc_to_belief` / `belief_to_npc`
- Test alias `CROWD_HLA_DIM` → `CROWD_BELIEF_DIM`

### Out of scope — historical doc comments (riir-train, 5 occurrences)

These are comments explicitly documenting the Issue 197 rename ("Renamed
from `HLA_DIM` (Issue 197)"). They are **intentional historical
references** and MUST NOT be edited — removing them would erase the
audit trail. Files:

- `crates/riir-train-engine/tests/issue_307_archetype_library_gates.rs:17`
- `crates/riir-train-engine/src/dsom_structure_validation.rs:708`
- `crates/riir-train-engine/src/sense_goat_proof.rs:26`
- `crates/riir-train-engine/src/archetype_library_train.rs:86`
- `crates/riir-train-engine/src/sense_reconstruction_batch.rs:23`

### Out of scope — Family C (CamelCase `Hla*` types + lowercase `hla_*`)

**Verdict: NOT DOING.** Recorded here so future agents don't re-litigate.

The CamelCase `Hla*` vocabulary is the **established domain name** for
the per-NPC affect/belief subsystem across the 7-repo stack:

- `HlaCuriosityDirection` (495 occurrences) — CGSP direction vector type
- `HlaKarcConfig` (150) — KARC forecaster config
- `HlaAnticipatedQueryDir` (114) — sleep-time anticipator query direction
- `HlaProjectionGuide` (78) — CGSP quality guide (public API in katgpt-core)
- ~45 other types, totaling **2000+ occurrences across 7 repos**

These compile clean. They are **not collisions** — `Hla` is a consistent
prefix referring to the same subsystem (Higher-order Linear Attention /
latent affect). Renaming them would:

1. Break public API in `katgpt-core` (`HlaProjectionGuide` is re-exported)
2. Require coordinated cross-repo commits touching 2000+ sites
3. Produce zero functional gain — the names already compile and read fine
4. Conflict with established documentation (`.docs/`, `.research/`, plan
   files all reference `Hla*` types by name)

The `hla_*` lowercase identifiers (`hla_arousal` trait method,
`hla_dim` doctest vars) are similarly established vocabulary. The
Issue 195/197 renames targeted **fields, methods, and constants that
were genuinely ambiguous or stale**; the surviving `Hla*` CamelCase
types are **unambiguous domain vocabulary** and were correctly left
alone.

**Re-open only if:** a future architectural split makes "HLA" actively
ambiguous (e.g., a second subsystem adopts the same acronym), or a
consumer reports concrete confusion. Lexical noise alone is not
justification.

## Tasks

- [x] T1: Rename `*_HLA_DIM` constants + helper fns in riir-ai (38 sites + 1 stale comment in bench_460)
- [x] T2: Verify `cargo check --workspace` + targeted feature-gated checks
      - `cargo check -p riir-engine -p riir-games-civ` (default features) — CLEAN
      - `cargo check -p riir-engine --features crowd_attention --tests` — CLEAN
      - `cargo test -p riir-engine --features latent_functor,mean_field_regime,crowd_attention --lib crowd_regime_post_tick` — 37 PASS
      - `cargo clippy -p riir-engine -p riir-games-civ --lib` — CLEAN
      - `cargo clippy -p riir-engine --features crowd_attention --lib` — CLEAN
      - Note: `--all-features --tests` has a pre-existing failure in `katgpt-micro-belief/src/leaky.rs` (`belief_default` not found) — unrelated to this rename, confirmed via git stash.
- [x] T3: Re-grep to confirm zero remaining `*_HLA_DIM` source tokens (the 1 hit in riir-train is an intentional historical comment)
- [x] T4: Commit + close this issue

## GOAT gate

Not applicable — pure rename, no behavior change. Validation =
`cargo check --workspace` clean + targeted feature-gated compile paths
clean (the lesson from Issue 197's `grudge_field` regression).
