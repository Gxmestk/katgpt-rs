# katgpt-rs — boundary contract

> The single source of truth for what may live in and depend on this repo.
> Audited by the `boundary-guard` skill. Cross-repo rules LINK to their one
> canonical home — never copied. Handoff tracker: `../riir-ai/.issues/737`.

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
