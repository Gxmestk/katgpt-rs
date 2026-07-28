# Issue 195 — HLA naming collision (Transformer attention vs per-NPC affect)

**Type:** Refactor / code smell.
**Severity:** Medium (concrete cost already paid — see "Evidence" below).
**Repos:** `katgpt-rs` (primary), `riir-ai` (downstream consumer of the per-NPC field).

## The smell

Two unrelated concepts share the acronym "HLA" in the 7-repo stack:

| Concept | Path | Expansion | Paper anchor | Architectural role |
|---|---|---|---|---|
| **Higher-order Linear Attention** | `katgpt-hla` crate, re-exported as `katgpt_core::hla`, mirrored in `riir-engine/src/hla/` | "Higher-order Linear Attention" (documented in crate docstring L1) | Zhang, Qin, Wang, Gu (2026) — `.research/28_*.md` | Transformer attention layer replacement (O(1) per-token recurrent state) |
| **per-NPC latent affect/belief state** | `katgpt-sense::ReconstructionState.hla: [f32; 8]` + `evolve_hla()` + `inject_hla_delta()` + `hla_learning_rate` + `max_hla_delta` | **none documented** — the field is just named `hla`; the crate itself is "KG Latent Octree Sense" (no HLA in crate name) | none | per-NPC evolving latent state (valence/arousal/desperation/calm/fear + 3), drives 6-module affect projection via dot+sigmoid |

These two mechanisms have nothing to do with each other. They live at different layers (Transformer forward pass vs game-runtime NPC cognition), use different math (recurrent matrix second moments vs leaky-integrated evidence projection), and serve different consumers.

## Evidence the smell has real cost

1. **Already burned an audit.** During a verification pass on 2026-07-28, an agent grep'd `HLA`, found only the per-NPC `evolve_hla` field, and produced a wrong verdict claiming "HLA is not a Transformer attention replacement." The user had to push back. The agent's mistake was the canonical paper-vs-codebase vocabulary failure mode — but the failure was *enabled* by the collision: a grep for "HLA" returning two unrelated mechanisms is the root cause.

2. **Fossilized in docs.** `riir-ai/AGENTS.md` describes `riir-engine/src/hla/` as "per-NPC 8-dim latent state (valence/arousal/desperation/calm/fear + 3)". That path is actually **Transformer attention code** (re-exports `katgpt-hla` kernels + adds `*_role_aware` variants per its own `forward.rs` L1-28). The wrong description in AGENTS.md is the same confusion, frozen into the repo's authoritative context file.

3. **Leaked into bench comments.** `katgpt-rs/crates/katgpt-core/benches/bench_337_tropical_perf.rs` L91-95 has to defensively disambiguate: "D=8 (HLA-scale) dense matvec is a cold-path curiosity — the actual HLA use case uses sparse DEC wrappers." Which HLA? The reader has to guess. `bench_425_tilr_goat.rs` L307 uses `TARGET_HLA_NS` for "d=8, r=3" — that's the per-NPC scale, unrelated to the Transformer attention HLA's actual latency profile.

4. **Asymmetric anchoring.** The Transformer side has a paper citation, a published crate name, and a documented acronym. The per-NPC side has none of these — it's just a field name with no expansion, inside a crate named after something else. The asymmetry is the tell: one side is anchored, the other is ad-hoc.

## The rename

**Keep `katgpt-hla` (Transformer attention). Rename the per-NPC side.**

The Transformer attention side is anchored to a paper (Zhang et al. 2026), published as a standalone crate, and matches the literature's standard expansion. Renaming it would break the public API + the literature mapping. The per-NPC side has no anchor — no paper, no acronym expansion, no published crate name — so its rename cost is bounded and local.

### Proposed rename: `hla` → `belief` / `belief_state`

Rationale:
- `belief` is already the codebase's vocabulary (AGENTS.md "Spatial Cognition (Two-Brain Model)" uses "SpatialBelief"; bridge docs use "belief manifold"; `riir-engine/src/latent_functor/` uses "belief state" throughout).
- Zero collision with anything in the 7-repo stack (grepped: no other `belief` field competes with this one).
- Accurately describes what the field does: an evolving latent state that drives downstream projections. The field's own docstring calls it a "cue" — `belief` is the same idea, more standard.
- Reads clean: `evolve_belief()`, `inject_belief_delta()`, `belief_learning_rate`, `max_belief_delta`, `ReconstructionState::belief()`, `belief_mut()`.

### Alternative candidates considered

| Candidate | Verdict |
|---|---|
| `affect` / `affect_state` | Semantically accurate (matches valence/arousal/desperation/calm/fear), but "affect" is jargon and would collide conceptually with `ai::AffectField` in the SDK. |
| `latent` / `latent_state` | Too generic — every embedding is "latent". Loses the per-NPC semantics. |
| `cue` | Already used in the field's own docstring, but too short / ambiguous on its own. |
| `belief_state` (full) | Cleaner than `belief` for type positions; same vocabulary win. |

## Scope of the rename

### `katgpt-rs` (primary)

- [ ] **T1** `crates/katgpt-sense/src/reconstruction.rs`: rename field `hla: [f32; 8]` → `belief: [f32; 8]`. Rename methods `evolve_hla` → `evolve_belief`, `evolve_hla_simd` → `evolve_belief_simd`, `inject_hla_delta` → `inject_belief_delta`, `hla()` → `belief()`, `hla_mut()` → `belief_mut()`. Rename config fields `hla_learning_rate` → `belief_learning_rate`, `max_hla_delta` → `max_belief_delta`. Update all doc comments. Preserve byte-identical behavior (this is a rename, not a behavior change).
- [ ] **T2** `crates/katgpt-sense/src/reconstruction_depth_invariance.rs`: rename `evolve_hla_regularized` → `evolve_belief_regularized` (mirrors T1).
- [ ] **T3** Audit `katgpt-core/benches/*.rs` for "HLA-scale" / "D_HLA" / "hla_state" comments referencing the per-NPC dimension. Update comments to say "belief-scale" / "D_BELIEF" / "belief_state". Leave Transformer-attention HLA references untouched.
- [ ] **T4** Audit `katgpt-core/examples/*.rs` for `HlaCacheProxy` references in comments — clarify or update to reflect that this is per-NPC belief, not Transformer attention.
- [ ] **T5** Audit `katgpt-core/src/{babel_codec,branching,bridge,cce}/*.rs` for "HLA state" / "HLA bucket" comments referencing per-NPC semantics. Update to "belief state" / "belief bucket".
- [ ] **T6** Add a one-line note at the top of `katgpt-sense/src/lib.rs` clarifying: "Note: `belief` (per-NPC affect state) is unrelated to `katgpt-hla` (Higher-order Linear Attention, Transformer attention layer). The shared 'HLA' acronym was a naming collision, resolved in Issue 195."

### `riir-ai` (downstream consumer)

- [ ] **T7** `crates/riir-engine/benches/reconstruction_bench.rs`: ~10 sites calling `evolve_hla()` directly. Rename to `evolve_belief()`.
- [ ] **T8** `crates/riir-engine/benches/bench_497_poincare_mcts_g9.rs`, `self_advantage_hla_bench.rs`: rename `src_hla` → `src_belief` where it refers to per-NPC state. (The bench *file names* `*_hla_*` are out of scope — they may legitimately reference Transformer attention; check each.)
- [ ] **T9** Audit `crates/riir-engine/src/` for direct `evolve_hla` / `ReconstructionState.hla` callers. Rename.
- [ ] **T10** **Fix the AGENTS.md error**: `riir-ai/AGENTS.md` describes `riir-engine/src/hla/` as "per-NPC 8-dim latent state (valence/arousal/desperation/calm/fear + 3)". That's wrong — `riir-engine/src/hla/` is Transformer attention (re-exports `katgpt-hla` + role-aware variants per its own `forward.rs` L1-28). Replace that table row with the correct description: "Forward pass implementations for Higher-order Linear Attention (re-exports `katgpt-hla` kernels; adds `*_role_aware` variants behind `hla_role_aware` feature). Paper: Zhang et al. 2026."

### Validation gates

- [ ] **G1** — byte-identical behavior. The per-NPC `belief` field is a pure rename. Run existing tests:
  - `cargo test -p katgpt-core --features sense_composition --lib` (the sense tests)
  - `cargo test -p katgpt-core --features temporal_deriv --lib`
  - `cargo test -p riir-engine --lib` (downstream consumer tests)
  All must pass without numeric changes.
- [ ] **G2** — no regression in `reconstruction_bench.rs` numbers (the leaky integrator is unchanged).
- [ ] **G3** — `cargo check --workspace --all-features` clean across both repos.
- [ ] **G4** — grep `katgpt-rs/` and `riir-ai/` for residual `\bevolve_hla\b|\.hla\b|max_hla_delta|hla_learning_rate|inject_hla_delta` after the rename — must return ZERO hits in the per-NPC sense context (Transformer-attention `HlaLayerState` / `MultiLayerHlaCache` etc. are NOT in scope and remain untouched).

## Out of scope

- The Transformer attention side (`katgpt-hla`, `HlaLayerState`, `MultiLayerHlaCache`, `forward_hla`, `forward_ahla`, `HlaMode`, `HlaVariant`) — UNTOUCHED. Anchored to a paper + published crate name.
- The `riir-engine/src/hla/` directory path itself — UNTOUCHED (it correctly hosts Transformer attention code; the AGENTS.md description was wrong, not the path).
- Bench *file names* containing `_hla_` — case-by-case. Many legitimately reference Transformer attention.

## Non-goals

- Renaming the Transformer attention `katgpt-hla` crate. That would break the public API + a literature-mapped name.
- Renaming the `katgpt-sense` crate itself. "KG Latent Octree Sense" is accurate and doesn't collide with anything.

## References

- The conversation that surfaced this: 2026-07-28 verification pass — agent initially produced wrong verdict claiming "HLA is not Transformer attention" because grep returned only the per-NPC `evolve_hla`. User pushback surfaced the collision.
- `katgpt-rs/crates/katgpt-hla/src/lib.rs` L1-38 — Transformer attention HLA, paper-anchored.
- `katgpt-rs/crates/katgpt-sense/src/reconstruction.rs` — per-NPC HLA field, no paper anchor.
- `riir-ai/crates/riir-engine/src/hla/forward.rs` L1-28 — proof that `riir-engine/src/hla/` is Transformer attention, contradicting the riir-ai AGENTS.md description.
- Research skill SKILL.md §"Vocabulary translation tips" — the protocol that would have caught this if applied before the original verdict.
