# katgpt-rs — boundary contract

> The single source of truth for what may live in and depend on this repo.
> Audited by the `boundary-guard` skill + two scripts:
> `../riir-ai/scripts/ci_boundary_contract.sh` (workspace — is the dep graph
> what this file says, and is this file still honest?) and
> `../riir-mmorpg-examples/scripts/ci_boundary_guard.sh` (per-repo — is the
> CODE in the right repo?). Cross-repo rules LINK to their one
> canonical home — never copied. Rollout record:
> `../riir-ai/.docs/01_orientation/737_boundary_contract_rollout_record.md`.

## Owns

- **Public modelless inference primitives** (`katgpt-core` + member crates) —
  the public funnel per Research 003. No training, no backprop, no gradient
  descent; runtime weight mutations limited to freeze/thaw, deterministic
  raw/lora hot-swap, latent-space updates.
- Workspace-level agent skills in `.agents/skills/` (boundary-guard,
  substrate-first, feature-gate-audit, goat-audit, proposal, research) — these
  are workspace *tooling*, not crate surface.

## Does not own

Everything downstream is a consumer, never a dependency:

| Concern | Correct home |
|---|---|
| Game runtime / cognition wiring | `../riir-ai` |
| Chain / storage / training / SDK / dApps | the respective `../riir-*` repos |

## May depend on

| Crate | Location | Condition |
|---|---|---|
| — | — | **Nothing workspace-internal.** katgpt-rs is UPSTREAM of riir-ai; any riir dep here is a dependency cycle. |

## Inherited boundaries (links)

- Cross-repo dep direction: `../riir-ai/BOUNDARY.md`
- Chain admission (three-test): `../riir-chain/BOUNDARY.md`

## Drift ledger (target vs actual)

None. (Clean at last guard run.)

## WASM / wasmi compatibility contract (audited 2026-08-23)

- Targets: `wasm32-unknown-unknown` (browser + CF Worker via wasm-bindgen) and
  `wasm32-wasip2` (wasmtime-run benches). Both verified clean via
  `cargo check -p katgpt-core --target <t>` with `+simd128` (2026-08-23).
- **simd128 rule (the Issue 205 lesson)**: every wasm build that runs perf
  kernels MUST pass `-C target-feature=+simd128` — without it the SIMD paths
  silently compile the scalar fallback (~16× slower; encoded in
  `scripts/build-moka-wasm.sh`) — and `wasm-opt --enable-simd` at the
  optimize step (without it SIMD ops are stripped).
- `getrandom` wasm backends (0.2 `js` + 0.3 `wasm_js`) stay pinned at
  workspace level — transitive bevy/uuid consumers break without them.
- **wasmi hosts carry `features = ["simd"]`** so they load modules built
  under any sibling workspace's `+simd128` rustflag. One version-aligned
  wasmi 1.x in every tree ("one wasmi" rule).
- Footgun: env `RUSTFLAGS` REPLACES `.cargo/config.toml` rustflags (they do
  not merge) — include every needed flag when overriding.
