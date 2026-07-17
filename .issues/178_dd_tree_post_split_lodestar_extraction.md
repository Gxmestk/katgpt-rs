# Issue 178 — Post-split growth: `dd_tree/mod.rs` (2125) + `tree_builder.rs` (2091)

## Context

`crates/katgpt-speculative/src/dd_tree.rs` was a 4207-line CRITICAL (>3200)
violation in Issue 162 (C2). Issue 165 split it into `mod.rs` + `tests.rs` +
`tree_builder.rs`. Both impl files have since crept back over the 2048 soft
limit (mod.rs at 2125, tree_builder.rs at 2091) due to feature-gated
additions (Lodestar Plan 207, Belief-Drafter Plan 217, RecFM Plan 168,
DendriticGate Plan 260, AND-OR Plan 190, dflare/corr/nf_flow budget variants).

These were NOT in the original Issue 162 audit (they were under the limit
at audit time) — they grew past it after the audit closed. This issue
tracks whether further splitting is warranted.

## Plan

### mod.rs (2125 lines) — extract Lodestar (765 lines)

`mod.rs` has clear `// ──`-delimited feature-gated sections. The biggest
is Lodestar (L148-913, 765 lines, `lodestar` feature). Extracting it
alone brings mod.rs to 1360 — well under 2048 ✓.

| File | Lines | Sections |
|---|---|---|
| `dd_tree/mod.rs` | ~1360 | core builders + small feature-gated wrappers |
| `dd_tree/lodestar.rs` | ~770 | LodestarConfig + build_dd_tree_lodestar + a_star_score + find_forced_token (Plan 207, Research 183) |

Other sections (Belief-Drafter 301, RecFM 214, DendriticGate 174, AND-OR 137)
are kept in mod.rs — mod.rs is already under the limit after Lodestar extraction.
Further splitting would create many tiny files (50-200 lines each) with
navigation overhead that outweighs the soft-limit benefit.

### tree_builder.rs (2091 lines) — verdict: KEEP (43 lines over)

`tree_builder.rs` is a **single `TreeBuilder` struct** with 12 methods that
share tightly-coupled private state (`heap`, `chain_nodes`, etc. are private
fields). Splitting methods across files would require either:
- Making private fields `pub(super)` — encapsulation hit
- Adding accessor methods — runtime overhead + boilerplate

For 43 lines of soft-limit reduction, neither trade-off is justified.
**Verdict: keep as-is.** The file is 1% over the soft limit, well under the
3200 hard limit, and the struct's cohesion outweighs the marginal split benefit.

## Tasks

- [x] T1. Extract Lodestar section (L148-421 of original mod.rs, 274 lines) from `dd_tree/mod.rs` → `dd_tree/lodestar.rs`. **Note:** the section header `// ── Lodestar ──` spanned L148-913, but only L148-421 were actually Lodestar-specific (LodestarConfig + build_dd_tree_lodestar + a_star_score + find_forced_token). L422-913 are core functions (build_dd_tree_screened, build_dd_tree_balanced, merge_retrieved_branches, inject_sde_noise_into, build_slices_view, extract_*, find_valid_sequence, par_find_*) that were placed under the same header region — NOT Lodestar. Initial extraction attempt swept up all of L148-913 and broke mod.rs (those core functions were missing). Corrected to L148-421 only.
- [x] T2. Added `#[cfg(feature = "lodestar")] mod lodestar;` + `pub use lodestar::*;` to mod.rs. Removed unused `CompletionHorizon` import from mod.rs (now in lodestar.rs).
- [x] T3. GOAT G1+G3. **PASS:**
  - clippy clean under default + lodestar + --all-features
  - 305/305 katgpt-speculative tests under lodestar
  - 1079/1079 under --all-features
  - Workspace sweep: 6519 passed (same count as pre-split)
  - Workspace clippy clean
  - Pre-existing flaky perf gates (jacobian_svd_r8x8_latency_gate, workflow_lattice bench) reproduce the same pass-in-isolation/fail-under-load behavior as documented in prior sessions — NOT caused by this split.
- [x] T4. tree_builder.rs verdict: **KEEP** (43 lines over). Single TreeBuilder struct with tightly-coupled private state (`heap`, `chain_nodes`, etc.). Splitting methods across files would require making private fields `pub(super)` (encapsulation hit) — not justified for 43 lines of soft-limit reduction.

## Final file sizes

- `dd_tree/mod.rs`: 1866 (was 2125) — under 2048 ✓
- `dd_tree/lodestar.rs`: 274
- `dd_tree/tests.rs`: 902 (tests, exempt)
- `dd_tree/tree_builder.rs`: 2091 — **KEEP** (43 lines over; single struct, private state, not worth encapsulation hit)

## Out of scope

- tree_builder.rs split (verdict: KEEP — single struct, private state, 43 lines over)
- weaver.rs (user-explicit skip)
