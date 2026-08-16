---
name: boundary-guard
description: Audit + enforce game-stack boundary rules across the 10-repo workspace (katgpt-rs, riir-ai/-chain/-neuron-db/-train/-game-sdk/-mmorpg-examples/-unity/-viewbridge/-clippy, future riir-bevy). Use when adding a new System impl, game logic, vocabulary type, FFI surface, or view-layer code; reviewing PRs touching game systems or view/FFI boundaries; when a violation is suspected ("why is this logic here?" or "why is this game logic in C#"); or quarterly as a boundary-hygiene gate. Covers 7 surfaces; consumer src/ (no generic game logic), SDK facade (no engine deps), SDK root-vs-members (root clean), leaf vocabulary (engine deps feature-gated), view C#/Bevy (render only), FFI bridge (raw physical, no latent), dev tools (no game coupling). Enforces via grep checks + ci_boundary_guard.sh. Sibling to feature-gate-audit + goat-audit + doc-sync.
---

# Boundary Guard

Generic game logic → substrate (`riir-games`). View renders state, doesn't compute it. FFI moves raw bytes only.

## Seven surfaces

| # | Surface | Rule | Grep check |
|---|---------|------|------------|
| 1 | Consumer `src/` (riir-mmorpg-examples, seal) | Thin glue only — no `impl System` with loops/math; no hardcoded constants; no duplicated helpers | `grep -rn 'fn distance_2d\|const.*FEAR' src/` |
| 2 | SDK root crate (`riir-game-sdk/src/`) | Facade only — no engine/chain/db deps in the DEFAULT build. Sanctioned opt-in exceptions (Issue 053 Part 2 + `auth_impl`/`gm` pattern; all `optional = true`, heavy ones target-gated native): `auth_impl`/`identity_impl` (riir-auth, riir-chain/ssh_key), `gm` (katgpt-core, hoisted `InferenceBackend`), `static_data_impl`/`warm_tier_impl` (riir-neuron-db, neuron-db-sdk). Scan BOTH `[dependencies]` AND `[target.'cfg(...)'.dependencies]` — the Issue 053 deps live in the target-gated section | main + target sections, filter out `optional = true` lines |
| 3 | SDK root vs workspace members | Root crate clean; members (`crates/riir-viz`, etc.) MAY depend on engine | `sed -n '/\[dependencies\]/,/^\[/p' riir-game-sdk/Cargo.toml \| grep katgpt` |
| 4 | Leaf-clean vocabulary (`riir-games-shared`) | No engine deps unless feature-gated — **at the dep level too**: dep line `optional = true` AND its feature carries `dep:<name>`. Half-gated (module cfg-gated, dep non-optional) = violation — every no-features build pays the engine tree (Issue 682: katgpt-core non-optional pulled rustfft/postcard/half into a default-`[]` crate) | `grep -E 'katgpt-core\|riir-engine' riir-ai/crates/riir-games-shared/Cargo.toml \| grep -v 'optional = true'` |
| 5 | View consumers (`riir-unity` C#, `riir-bevy`) | Rendering + input only — NO game logic (AI, combat, physics, sync). Documented deliberate debt in an OPEN issue + cross-language contract doc (the `WireProtocol.cs` precedent: riir-unity Issue 002 Phase A, contract at `.docs/03_game_client/wire_protocol.md`) = record as such, don't re-file | `grep -rn 'sigmoid\|dot_product\|class.*System' riir-unity/**/*.cs \| grep -v Showcase\|Benchmark\|Camera` |
| 6 | FFI bridge (`riir-viewbridge`) | Raw physical only (`pos[3]`, `rot[4]`) — NO latent state crosses FFI | `grep -rn 'emotion\|fear\|mood\|curiosity' riir-viewbridge/crates/*/src/` |
| 7 | Dev tools (`riir-clippy`) | Zero game-domain coupling in the DEFAULT build (only `katgpt-core`, the public primitives crate). Sanctioned opt-in arms where reimplementation would duplicate whole substrates: `ternary_inference` (riir-engine + riir-gpu), `latent_retrieval` (riir-rag) | `grep -E 'riir-games\|riir-chain' riir-clippy/Cargo.toml` (must be empty; engine/rag allowed only `optional = true` behind their features) |

All greps should return **empty** (clean), modulo the sanctioned opt-in exceptions noted per-surface.

**Methodology lesson (2026-08-15 run):** exclusion filters can hide exactly what you're looking for — the "who enables feature X" grep returned zero because the forwarder lines contain `katgpt-core` and were killed by `grep -v katgpt-core`. Vocabulary-translation care applies to filters, not just search terms.

## Failure pattern

"Helper" in consumer → wrapped in `System impl` → grows loops + math → stuck in consumer. Same applies to C# view code reimplementing substrate logic.

## Extraction checklist

Before adding to consumer `src/` or view C#/Bevy:

1. Is this generic game behavior? → substrate (`riir-games`)
2. Does substrate already have it? → grep `riir-games/src/{swarm,motivation,combat}/`
3. Can it be parameterized? → trait (`ThreatSource`) or config struct
4. Is the consumer/view just data + wiring? → if loops/math/constants present, STOP

If unsure → file an issue, don't add the code.

## Filing violations

1. `.issues/NNN_boundary_*.md` in the repo
2. Reference which surface (1–7)
3. Include file:line + grep output
4. Propose extraction target (substrate module + trait)
5. Don't fix in same commit

## Running

```bash
cd riir-mmorpg-examples && ./scripts/ci_boundary_guard.sh   # exit 0 = clean
```

For other repos, adapt the SRC_DIR + patterns. Or as pre-commit: `exec ./scripts/ci_boundary_guard.sh`

## Run log

| Date | Runner | Verdict |
|---|---|---|
| 2026-08-17 | idle-queue routine check (riir-clippy skill item 6; S1 script run, repo = riir-mmorpg-examples) | **1 actionable finding — S1 `ci_boundary_guard.sh` exit 1: 8 E1/E2 facade leaks, all new since the 2026-08-16 clean run, from Plan 022 phases F/H/K.** 5 sites landed (karma.rs:48/791 + warm_tier.rs:480/490/491 — `riir_neuron_db::karma_history` types reaching past the facade), 1 site in ACTIVE sibling WIP (shared_world.rs:73 `riir_simloop::SimLoop` — uncommitted). Detection-only per discipline: filed as riir-mmorpg-examples Issue 061 (commit `f3ac749`), fix route = SDK `karma_history` re-export (Issue 053 Part 2 precedent) + a `riir_simloop` facade-or-sanction decision; shared_world site deferred to the Plan 022 owner (do not touch sibling WIP). No fixes in the detection commit. |
| 2026-08-15 | boundary-guard quarterly gate (agent session, katgpt-rs handoff) | 6/7 surfaces CLEAN; 1 actionable finding — Surface 4 half-gated dep: `katgpt-core` non-optional in riir-games-shared while its only consumers (`grudge_field`, `sleep_time_reload`) are cfg-gated (Issue 682). **Fixed same session** (commit `e982ef7d2`, riir-ai develop): dep optional + `dep:katgpt-core` in both features + implicit `katgpt-core/sleep_time_anticipation` need made explicit; no-features check 15.45s→5.76s, zero katgpt-core artifacts; all feature paths/forwarders/wasm32/reverse-deps/clippy/514 lib tests green; workspace Cargo.lock unchanged. Bonus pre-existing fix: stats doctest used the pre-move `riir_games::` path (commit `20abe832d` — no pinned suite runs `-p riir-games-shared --doc`, the "which gate runs this?" blind spot). Surfaces verified clean: 1 (`ci_boundary_guard.sh` exit 0), 2/3 (all heavy SDK deps optional + feature-gated + documented), 5 (WireProtocol.cs = documented open-issue debt, not re-filed), 6 (raw-physical only), 7 (opt-in arms documented, default = katgpt-core only). Skill checks refined this run: S2/S4/S7 grep notes now encode the sanctioned exceptions + the half-gated rule. |
| 2026-08-15 (evening) | idle-queue routine check (riir-clippy skill item 6, Issue 019 T8; grep-only run during the Plan 337 32K measurement — no builds) | **7/7 surfaces CLEAN, zero violations, zero new issues.** Delta since the morning quarterly gate: only sibling WIP (riir-ai qv_lora/riir-gpu) + riir-clippy fix_verify WIP — none touch a boundary surface. S1 `ci_boundary_guard.sh` exit 0. S2/S3 SDK root: heavy deps all `optional = true` (comments-only grep hits). S4 Issue 682 fix holding (`dep:katgpt-core`/`dep:riir-engine` gating intact). S5 user C# zero sigmoid/dot_product/class-System hits (PackageCache hits excluded as toolchain, not user code; WireProtocol.cs remains documented open debt = riir-unity Issue 002 Phase A, user-gated). S6 viewbridge: only the guard's own doc-comments. S7 riir-clippy: no game deps; engine/gpu/rag `optional = true` behind features (sibling Cargo.toml WIP does not add coupling). First idle-queue invocation of the check — T8 closes on this run + the morning gate. |
| 2026-08-16 | idle-queue routine check (riir-clippy skill item 6; grep-only + S1 script run) | **7/7 surfaces CLEAN, zero violations, zero new issues.** Delta since 2026-08-15 evening: Issue 599 derive fix (neuron-db-derive attr parser + mmorpg SpawnZoneRow attribute trim — no boundary surface), Issue 430 G2/G3 validation (riir-train bench example — not a surface), sibling riir-ai GPU WIP (705/706 — not surfaces). S1 `ci_boundary_guard.sh` exit 0. S2/S3 comments-only hits (all heavy deps optional). S4 gating intact. S5 zero user-C# hits. S6 clean. S7 clean. |
