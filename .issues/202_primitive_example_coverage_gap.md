# Issue 202 — Primitive Example Coverage Gap

**Filed:** 2026-07-29
**Origin:** Plan 561 second-session re-audit (the prior session's "lone exception" claim was incorrect).
**Severity:** Documentation gap (no behavior impact).
**Re-audited:** 2026-07-29 (third session) — **corrected a flawed audit methodology.**

## Problem

Multiple public primitives in `katgpt-core` ship **zero example harnesses**. The
prior session (commit `dfecdeef`) claimed `transformer_inversion` was "the lone
exception" — that claim was wrong. An independent re-audit found at least 9
genuine user-facing primitives with no example coverage at all.

Every public primitive should have at least one reference example showing its
API surface + intended use case — this is established practice (CNA has 3
examples, EGA has 4, MUX has 4, ATTN_MATCH has 5; 224 total example files in
the repo root + 25 in `crates/katgpt-core/examples/`).

## ⚠️ Methodology correction (2026-07-29, third session)

The first version of this issue claimed **9 primitives** lacked examples,
with `conformal` as the #1 priority. That was wrong. The original audit grep
only searched the **root** `examples/` directory:

```bash
# FLAWED — misses crates/katgpt-core/examples/:
grep -rl "katgpt_core::$mod" examples/
```

The repo has **two** example directories:
- `examples/` (root, 224 files) — the main catalog
- `crates/katgpt-core/examples/` (25 files) — primitive-specific demos

Re-auditing against **both** directories revealed:

| Primitive | First-verdict | Correct verdict | Examples found |
|---|---|---|---|
| `conformal` | no examples ❌ | **HAS examples** ✅ | `conformal_airpassengers.rs` (248 LOC, "Report the Floor" reference), `conformal_karc_overlay.rs` |
| `qgf` | no examples ❌ | **HAS examples** ✅ | `qgf_01_guided_drafter.rs`, `qgf_02_adaptive_weight.rs`, `qgf_03_tier_routing.rs` |
| `transformer_inversion` | (built prior session) | HAS example ✅ | `transformer_inversion_01_forensics.rs` |
| `best_belief` | no examples | **no examples** ✅ confirmed | — |
| `ssmax` | no examples | **no examples** ✅ confirmed | — |
| `poincare` | no examples | **no examples** ✅ confirmed | — |
| `newton_schulz` | no examples | **no examples** ✅ confirmed | — |
| `faithfulness` | no examples | likely no examples (opt-in, lower priority) | (only CGSP's internal "faithfulness gate" matches — different concept) |

**Real gap: 4 DEFAULT-ON primitives + 1 opt-in, not 9.** The `conformal` #1
priority was completely wrong — the mandated UQ baseline already has a 248-line
example that demonstrates the exact API I was going to build.

### Lesson

Audit greps for example coverage MUST search every `examples/` directory in
the workspace, not just the root one. The repo's two-tier example layout
(root catalog + per-crate primitive demos) is easy to miss if you only
check the obvious path. Future audits: `find . -path '*/examples/*.rs'
-not -path '*/target/*'`.

## Affected primitives (verified zero example coverage, re-audited)

### DEFAULT-ON primitives (highest priority — shipped but undocumented)

| Primitive | Feature gate | Plan | Why it matters |
|---|---|---|---|
| `best_belief` | `best_belief` (DEFAULT-ON) | Plan 336 | ε-quantile Beta lower bound for conservative selection. Grandfathered UQ primitive per Issue 010. Pairs with the conformal "Report the Floor" story — Thompson sampling explores, best_belief exploits. |
| `ssmax` | `ssmax_temperature` (DEFAULT-ON) | Plan 411 | Length-aware log-N attention temperature (SSMax). Part of the parallax attention family. |
| `poincare` | `poincare_navigator` (DEFAULT-ON) | Plan 449 | Poincaré navigator (hyperbolic geometry). Closed-form latent navigation. |

### Opt-in primitives (lower priority — not in default build)

| Primitive | Feature gate | Plan |
|---|---|---|
| `newton_schulz` | `newton_schulz` | Plan 152 (Muon optimizer) |
| `faithfulness` | `faithfulness_probe` | Plan 244 (FaithfulnessProbe) |

## Proposed resolution

Build example harnesses prioritized by:

1. ~~**`best_belief`**~~ — **DONE (2026-07-29).** Example landed at
   `crates/katgpt-core/examples/best_belief_01_conservative_selection.rs`.
   Demonstrates: monotonicity invariants (RQGM Prop. 4), Thompson-explores /
   best_belief-exploits contrast, a realistic 5-candidate freeze/thaw promotion
   scenario, incumbent tie preference (anti-churn), and the LUT hot path vs
   closed-form cold path. 5 sections, ~270 LOC. Clippy clean, runs clean,
   18 best_belief unit tests still pass. Wired in `Cargo.toml` behind
   `required-features = ["best_belief"]`.
2. **`ssmax`** + **`poincare`** — DEFAULT-ON, deserve at least a minimal demo.
3. Opt-in primitives (`newton_schulz`, `faithfulness`) as time permits.

Each example should follow the established pattern: module doc comment with
"what this proves / what this does NOT prove", runnable demonstration of the
core API, honest scope note.

## Not included (internal substrate — examples not needed)

`alloc`, `linalg`, `traits`, `freeze`, `dec_freeze`, `proof_cache`,
`delta_mem`, `mcts_state_action_cache`, `thinking_mode`, `shard_embedding`,
`simd_lut_dequant`, `set_diffusion_schedule`, `content_store`, etc. — these
are infrastructure consumed by other primitives/modules, not standalone
user-facing primitives.

## Verification method (corrected)

```bash
# MUST search BOTH example directories:
for mod in best_belief ssmax poincare newton_schulz faithfulness; do
  count=$(find . -path '*/examples/*.rs' -not -path '*/target/*' -name "*${mod}*" 2>/dev/null | wc -l)
  echo "$mod: $count examples (filename match)"
done
```

## Out of scope

- Building ALL examples in one pass — each is a separate focused task.
- Modifying any primitive's API — examples document existing API, don't change it.
- Plan 561 (`transformer_inversion`) — already has its example (commit `dfecdeef`).
