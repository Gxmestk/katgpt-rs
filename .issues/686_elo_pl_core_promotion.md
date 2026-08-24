# Issue 686 — Promote the ELO/Plackett-Luce rating primitives to `katgpt-core`

> **Opened:** 2026-08-24 · **Status:** OPEN
> **Requesting side:** `riir-clippy` Issue 039 (ELO-rated rules over the unified trajectory store)
> **Substrate:** `katgpt-pruners/src/arena/types.rs` (EloCalculator, base 1200) + `katgpt-pruners/src/proof/plackett_luce.rs` (PL→Elo via Gibbs, Plan 128 T4)

## Ask

Move the rating math — the batch Plackett-Luce → Elo conversion (order-free,
replayable) and the `EloCalculator` scale conventions (base 1200, 400 per
log10 unit) — into `katgpt-core` as public modelless primitives, with
`katgpt-pruners` re-exporting/consuming from core instead of owning its own
copies.

## Why

Three in-stack consumers, the third one blocked on the promotion:

| Consumer | Use |
|---|---|
| `katgpt-pruners` arena | live `Ranking { elo }` + leaderboard (existing owner) |
| `katgpt-pruners` proof | PL→Elo conversion (existing owner, same crate) |
| **`riir-clippy` Issue 039 T2** | batch PL→Elo over the trajectory store's real verdicts (rule-vs-shape ratings). `riir-clippy` depends on `katgpt-core` NON-OPTIONALLY today — if the primitives live in core, 039 consumes them with zero new deps; if they stay in `katgpt-pruners`, 039 must either adapt-locally (a third copy of the math) or add a `katgpt-pruners` dep. |

## Domain test

Passes: pure rating arithmetic (Gibbs sampling over win/loss/draw conventions
already the house standard — `katgpt-core/induced_cwm/tournament.rs` L110
uses win=1/loss=0/draw=0.5), no `riir-*` deps, modelless, no training. This
is exactly the "modelless inference primitive" class this repo exists to host.

## Scope sketch

1. `katgpt-core`: new module (e.g. `rating`) — `EloConfig { base: 1200, scale: 400 }`,
   the PL batch → Elo conversion (f64 math, no_std-friendly where cheap),
   unit-gated against `katgpt-pruners`'s own fixtures (the cross-check 039's
   T2 requires anyway).
2. `katgpt-pruners`: delegate both existing sites to core (the arena
   calculator + the PL conversion), tests unchanged in expectation.
3. `riir-clippy` 039 T2: consume via the existing core dep.

## Non-goals

- No online/incremental ELO API (039 computes batch-on-load by design —
  its hot fix loop stays allocation-stable).
- No selection-semantics change anywhere (ratings are reported axes until a
  measured challenge gate says otherwise — riir-clippy Issue 026 precedent).

## Until this lands

`riir-clippy` 039 T2 proceeds adapt-locally with citation (the
`ruliology_search.rs` precedent — reuse the algorithm pattern, no new dep),
and switches to the core primitive when this issue closes. Filed per 039's
T1 instruction ("file the katgpt-rs ELO→core promotion issue at T1 time").
