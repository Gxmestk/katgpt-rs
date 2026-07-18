---
name: feature-gate-audit
description: Audit feature-gate status claims across the 7-repo stack (katgpt-rs, riir-ai, riir-chain, riir-neuron-db, riir-train, riir-game-sdk, seal-online-remaster). Use when a doc/plan/issue/commit-message claims a "promotion discrepancy", "not yet wired", or "Default-off until..." state, when fixing stale feature-gate comments, before promoting or demoting any feature flag, or quarterly as a feature-gate-hygiene gate. Enforces three defenses (1) source-code verification of every wiring claim — grep the production tick path, don't trust the doc; (2) multi-surface grep for stale comments across 5 documentation surfaces (source .rs, lib.rs module doc, Cargo.toml default block, downstream Cargo.toml, .benchmarks/*_promotion_review.md); (3) layer-split awareness — engine-layer DEFAULT-ON + game-layer OPT-IN is a deliberate pattern when each layer gates a different concern (generic runtime vs IP-bearing content), NOT a missed propagation. Sibling to goat-audit + doc-sync.
---

# feature-gate-audit — Verify feature-gate claims against production code

This skill exists because of a caught error chain: session `f92a7ffe` in
`riir-ai` (2026-07-18) framed `npc_sleep_time`'s engine-DEFAULT-ON /
civ-OPT-IN state as a "promotion discrepancy" and claimed the catalog
was "wired but not yet exercised in production civ tick". Session
`f865913e` (same day) verified both claims against production code and
found them **factually wrong**:

- The split is a **deliberate layer split** (engine = generic runtime;
  civ = IP-bearing catalog + graceful-degradation hook into
  `SleepConsolidateNode::tick` Gate 3).
- The catalog IS exercised in production via
  `crates/riir-games-civ/src/civ/daily_loop/mod.rs:1071-1078`, which
  calls `ctx.sleep_runtime.sleep_cycle_tick(&hla, &self.dirs)`.

The three defenses below would have caught both errors before they
landed. Apply them to every feature-gate claim you encounter.

## When to use

- A doc, plan, issue, commit message, or session summary claims a
  "promotion discrepancy" between two crates' feature-gate status
- A doc claims a feature is "wired but not yet exercised in production"
- You encounter a `// Default-off until G1–GN GOAT gate passes` comment
  in source code
- Before promoting a feature flag to default-on (or demoting to opt-in)
  in any of the 7 repos
- Quarterly as a feature-gate-hygiene gate (alongside `doc-sync` and
  `goat-audit`)

## The three defenses

### Defense 1 — Verify every "discrepancy" / "not yet wired" claim against production code

**NEVER** propagate a doc/plan/session-summary's framing of feature-gate
state without first grepping the actual production wiring. Plausible-but-
wrong framings are the canonical failure mode — they read as reasoned
verdicts, get propagated through summaries, and solidify into "facts"
that downstream sessions trust.

**Protocol:**

1. Identify the production host(s) — the function that would invoke the
   gated code path on the hot tick. Common hosts in this stack:
   - `tick_proactive_salience` (salience gate)
   - `tick_cgsp_curiosity` (CGSP runtime)
   - `tick_motivation` (motivation system)
   - `SleepConsolidateNode::tick` (sleep-time, daily loop)
   - `evolve_feeling_brain` (emotion systems)
2. `grep` for the feature name across **all** `.rs` files in the crate
   tree, not just the file the doc cited:
   ```sh
   grep -rn "FEATURE_NAME" crates/ --include="*.rs"
   ```
3. Read the body of every host function whose signature imports a type
   from the gated module.
4. If a `#[cfg(feature = "FEATURE_NAME")]` block is reached along the
   tick path, the feature IS exercised in production — regardless of
   what the doc says.
5. If no host reaches the gated block, **then** the "not yet wired"
   claim may be accurate — but verify with a call-graph grep, not just
   a single-file read.

**Anti-pattern — trust the summary:** "the prior session said X, and
the prior session is in git history, so X must be true." Prior sessions
make mistakes. Git commits are not authority — production code is.

### Defense 2 — Stale gate-status comments live in MULTIPLE locations; fix all of them

When a feature is promoted from opt-in to DEFAULT-ON, the promotion
must propagate through **five documentation surfaces**. Missing any one
leaves a stale comment that misleads future readers.

| # | Surface | What to check |
|---|---|---|
| 1 | The feature's source `.rs` file (top-of-module `//!` doc) | "Default-off until..." → "DEFAULT-ON since..." |
| 2 | The crate's `lib.rs` `pub mod` declaration comment | The `// Default-OFF until...` line above `#[cfg(feature = "...")]` |
| 3 | The crate's `Cargo.toml` `[features] default = [...]` block | Feature name should appear, OR have an explicit sibling comment explaining intentional opt-in |
| 4 | Every downstream consumer's `Cargo.toml` `[features]` block | Forwarding features don't need to be default-on, but their comments must not cite a pending gate that has already passed |
| 5 | `.benchmarks/NNN_<feature>_promotion_review.md` | Must exist + reference the GOAT gate evidence (G1–GN) |

**Canonical failure (riir-ai, 2026-07-18):** the `npc_sleep_time` comment
`// Default-off until G1–G5 GOAT gate passes (Plan 341 Phase 7)` existed
in **both** `crates/riir-engine/src/lib.rs:629` **and**
`crates/riir-games-civ/src/npc/sleep_time_catalog.rs:48`. A prior session
only caught the latter; the engine `lib.rs` surface went unfixed.

**Protocol — the 5-surface grep (run for every feature under audit):**

```sh
# In each repo that touches the feature:
grep -rn "Default-off until\|Default-OFF until" crates/ --include="*.rs"
grep -n "FEATURE_NAME" Cargo.toml crates/*/Cargo.toml
grep -n "FEATURE_NAME" crates/*/src/lib.rs
ls .benchmarks/*FEATURE_NAME* 2>/dev/null
```

**Fix discipline:** all stale surfaces go in **one commit**, not
piecemeal. The commit message should enumerate which surfaces were
fixed and cite the canonical benchmark (`Promotion Review` NNN) as
authority.

### Defense 3 — Layer splits are deliberate, not bugs

**Asymmetric feature-gate status across crates is OFTEN a deliberate
design, not a missed propagation.** Before claiming a "discrepancy",
verify that each layer is gating the SAME concern. If they're gating
**different** concerns, the asymmetry is a layer split — update the
comments to explain the split; do NOT propagate the promotion.

**Established layer-split patterns in this codebase:**

| Feature | Engine layer | Game layer | Why split |
|---|---|---|---|
| `npc_sleep_time` | DEFAULT-ON (generic runtime: `HlaSleepTimeOp`, codec, BLAKE3 commitment) | OPT-IN (`riir-games-civ`, `riir-games-shared`, `riir-games`) | Civ layer gates the IP-bearing catalog (per-NPC-type direction vectors) + graceful-degradation hook into `SleepConsolidateNode` |
| `cognitive_branches_runtime` | DEFAULT-ON (plasma-tier core) | OPT-IN sub-features (`_engram`, `_closure`, `_replay`, `_entity_cognition`) | Engine ships the core; heterogeneous subsystems stay opt-in |
| `committed_personality_runtime` | DEFAULT-ON (plasma-tier core) | OPT-IN `committed_blend_freeze` (ArchetypeBlendShard persistence layer) | Engine ships the core; persistence layer stays opt-in |
| `arg_runtime` | DEFAULT-ON (plasma-tier core) | OPT-IN `_manifold`, `_action`, `_offline`, `_replay` | Engine ships the core; heterogeneous Steps 4/5/7/8 stay opt-in |
| `cwm_runtime` | DEFAULT-ON (IP-free core with mock LLM) | OPT-IN `cwm_gemini` / `cwm_local_llm` / `cwm_wasm_loader` | Engine ships the modelless core; concrete LLM backends stay opt-in |

**Established propagate-the-promotion patterns (NOT splits):**

| Feature | Engine | Game | Why propagated |
|---|---|---|---|
| `karc_runtime` | DEFAULT-ON | DEFAULT-ON (`riir-games-civ`) | Both layers gate generic machinery; no IP split |
| `cgsp_runtime` | DEFAULT-ON | DEFAULT-ON | Generic machinery; no IP split |
| `proactive_salience` | n/a (no engine dep) | DEFAULT-ON | Self-contained at the civ layer |

**Decision rule:**

- **Both layers gate generic, IP-free machinery** → propagate the
  promotion to all 5 surfaces per Defense 2.
- **One layer gates IP-bearing content** (catalogs, direction vectors,
  game-design data, trained weights) → keep the IP layer opt-in,
  document the split with a comment explaining the rationale.
- **One layer gates heavy optional deps** (`riir-neuron-db`,
  `riir-chain`, GPU crates) → keep that layer opt-in, document the
  dep-cost rationale.

**Protocol — before claiming a discrepancy:**

1. Read both crates' `Cargo.toml` comments. Do they describe the SAME
   concern or DIFFERENT concerns?
2. If different concerns → it's a layer split. Update the comments on
   both sides to explain the split; do NOT propagate the promotion.
3. If same concern → it's a real discrepancy. Propagate the promotion
   and update all 5 surfaces per Defense 2.

## Output format

After running an audit, produce a per-feature verdict table:

| Feature | Surface 1 (src .rs) | Surface 2 (lib.rs) | Surface 3 (Cargo.toml default) | Surface 4 (downstream) | Surface 5 (.benchmarks) | Verdict |
|---|---|---|---|---|---|---|
| `npc_sleep_time` | ✅ L44-66 | ✅ L629-642 | ✅ engine default-on | ✅ civ opt-in by design | ✅ bench 341 | deliberate-split |
| `<feature>` | ✅/❌ + line | ✅/❌ + line | ✅/❌ + line | ✅/❌ + line | ✅/❌ + file | see verdicts |

**Verdicts:**

- **clean** — all 5 surfaces accurate; no action.
- **fix-N-surfaces** — N surfaces have stale comments; fix in one commit
  per Defense 2.
- **deliberate-split** — asymmetric gate status is intentional per
  Defense 3; update comments on both sides to explain the split.
- **real-discrepancy** — same concern, different gate status across
  layers; propagate the promotion per Defense 2.
- **pending-gate** — gate has not yet passed; comment is accurate, no
  action (verify the gate is still pending by checking the latest plan
  phase status).

## Scope — the 7 workspace repos

```
katgpt-rs/              ← engine primitives; mostly DEFAULT-ON at this layer
riir-ai/                ← multi-crate workspace; engine/SDK/games layers
                          (the most common site of layer splits)
riir-chain/             ← chain lib + daemon
riir-neuron-db/         ← leaf crate (re-exported by riir-chain)
riir-train/             ← training-method research
riir-game-sdk/          ← game-vocabulary facade over riir-ai
seal-online-remaster/   ← consumer (seal-core / mmorpg)
poc-maxman/             ← consumer (pacman-like reasoning POC)
```

For each repo, **read its `AGENTS.md` first** — it documents the
canonical feature-gate layout and the layer-split conventions for that
repo. The repo-local rules override the general guidance here when they
conflict.

## Common failure patterns

### "Plausible but wrong"

The most common failure: a doc/plan/session-summary claims state X
based on a partial read of the code, and the claim is propagated
through downstream summaries without re-verification. **Always** grep
the production wiring before propagating the claim. The cost of
verification is ~30 seconds; the cost of a wrong claim landing in a
commit is a future audit to undo it.

### "Single-surface fix"

Fixing one stale comment but missing the other four. A reader who finds
a stale comment in `lib.rs` and a correct comment in the source file
concludes "the comment is just stale" — eroding trust in the whole
documentation surface. **Always** do the 5-surface grep; fix all stale
surfaces in one commit.

### "Reflexive discrepancy framing"

Seeing asymmetric gate status and assuming it's a bug. This is the
single most common framing error in this codebase because the layer
split is the **dominant** pattern (5 of the 9 major runtimes ship as
engine-core-default-on + sub-features-opt-in). **Always** check whether
each layer gates the same concern before claiming a discrepancy.

### "Authoritative-sounding stale comment"

A `// Default-off until G1–G5 GOAT gate passes` comment in source code
**looks** authoritative — it cites the plan, the phase, the gates.
Readers trust it. But the comment was written **before** the gate
passed, and nobody updated it when the promotion landed. Treat any
"until... passes" comment as a candidate for staleness, and verify
against the `.benchmarks/NNN_*_promotion_review.md` (which is the
authoritative record of the gate's status).

## See also

- `~/.agents/skills/doc-sync/SKILL.md` — quarterly doc hygiene gate
  (sibling skill; feature-gate-audit focuses specifically on gate-status
  accuracy, doc-sync covers the broader `.docs/` + `README.md` sync)
- `katgpt-rs/.agents/skills/goat-audit/SKILL.md` — cross-repo GOAT
  cherry-pick audit (sibling skill; focuses on primitive propagation,
  this skill focuses on feature-flag status accuracy)
- `katgpt-rs/AGENTS.md` §"Feature Flag Discipline" — the GOAT promotion
  rule (the policy this skill enforces)
- `riir-ai/.benchmarks/341_npc_sleep_time_promotion_review.md` —
  canonical example of a complete promotion review (G1–G5 + modelless
  + latent boundary + β-tuning finding)
- Commit `f865913e` on `riir-ai/develop` (2026-07-18) — the canonical
  example of this skill's three defenses catching a prior session's
  errors
