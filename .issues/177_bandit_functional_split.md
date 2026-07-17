# Issue 177 — Functional split: `bandit.rs` (2178) → `bandit/` module folder

## Context

`crates/katgpt-pruners/src/bandit.rs` (2178 lines) was missed by Issue 162's
audit. The file's own `// ──`-style delimiter comments mark **14 natural
functional seams** (Strategy / Stats / Pruner / Beta / Environment / Event /
Result / Session / AbsorbCompress / SharedBanditStats / RandOpt) — exactly
the pattern that Issue 175 used for `players.rs`. The 3-line "Tests" section
is just a `#[path = "bandit_tests.rs"] mod tests;` reference (no inline
test code to extract).

**This is a functional split, not a mechanical test extraction** (the file
has 0% inline tests — third-party test file already exists).

## Plan

| File | Lines (est.) | Sections |
|---|---|---|
| `bandit/mod.rs` | ~1300 | imports + Strategy + Stats + Pruner + Beta sampling + AbsorbCompress Integration + re-exports + mod declarations + `#[cfg(test)] #[path = "../bandit_tests.rs"] mod tests;` (path correction needed) |
| `bandit/environment.rs` | ~155 | BanditEnv trait + BernoulliEnv + GaussianEnv (L1243-1396) |
| `bandit/session.rs` | ~480 | BanditEvent + BanditResult + BanditSession (L1398-1880). Needs `use super::*;` + `pub(super) fn make_stats` in mod.rs. |
| `bandit/shared_stats.rs` | ~170 | SharedBanditStats + BanditSnapshot (L1911-2076) — `#[cfg(feature = "bandit")]`-gated |
| `bandit/randopt.rs` | ~100 | solution_density + spectral_discordance + select_arms_top_p (L2078-2172). `select_arms_top_p` is `bandit_top_p`-gated. |

**Path adjustment for tests:** the original `#[path = "bandit_tests.rs"]`
in `bandit.rs` resolved to `src/bandit_tests.rs`. After moving to
`bandit/mod.rs`, the same path attribute would resolve to
`src/bandit/bandit_tests.rs`. To keep the test file in place, use
`#[path = "../bandit_tests.rs"]` (or move the file in-tree).

## External API surface (must be preserved)

From `crates/katgpt-pruners/src/lib.rs:169-189`:
- `BanditEnv`, `BanditEvent`, `BanditPruner`, `BanditResult`, `BanditSession`,
  `BanditStats`, `BanditStrategy`, `BernoulliEnv`, `GaussianEnv`,
  `SharedBanditStats` (all under `#[cfg(feature = "bandit")]`)
- `select_arms_top_p` (under `#[cfg(feature = "bandit_top_p")]`)

Strategy: each sibling file uses `pub use ...` to surface its types, then
mod.rs has a single `pub use {environment::*, session::*, shared_stats::*,
randopt::*};` to flatten the namespace. External callers see no change.

## Tasks

- [x] T1. Create `bandit/` directory + move `bandit.rs` → `bandit/mod.rs`
- [x] T2. Extract environment.rs (BanditEnv + Bernoulli + Gaussian) — 162 lines
- [x] T3. Extract session.rs (BanditEvent + BanditResult + BanditSession) — 499 lines. Marked `make_stats` as `pub(super)` in mod.rs. Note: imports of `ReviewMetrics`, `SafePhasedState`, `TrialLog`, `TrialRecord` in session.rs use `crate::` prefix (they're at the crate root, not in `bandit::`).
- [x] T4. Extract shared_stats.rs (SharedBanditStats + BanditSnapshot, `bandit`-gated) — 172 lines
- [x] T5. Extract randopt.rs (Plan 121 diagnostics + select_arms_top_p) — 101 lines
- [x] T6. Update mod.rs: added `mod environment; pub use environment::*;` etc. + fixed `#[path = "../bandit_tests.rs"]` for tests
- [x] T7. GOAT G1+G3. **PASS:**
  - 43/43 `bandit::tests::*` under `bandit` feature
  - 197/197 katgpt-pruners lib tests under `bandit`
  - 197/197 under `bandit,bandit_top_p`
  - Workspace sweep: 6519 passed / 0 failed under default features
  - Workspace `cargo clippy --lib` clean
  - All relevant feature combos compile: default + bandit + bandit,bandit_top_p + bandit,safe_bandit + bandit,skill_lifecycle + bandit,idea_divergence + bandit,dynamic_rank + --all-features
- [x] T8. Issue 162 updated with completion record.

## Final file sizes (all under 2048 ✓)

- `bandit/mod.rs`: 1289 (was 2178)
- `bandit/environment.rs`: 162
- `bandit/session.rs`: 499
- `bandit/shared_stats.rs`: 172
- `bandit/randopt.rs`: 101

## Notes

- Path correction: `#[path = "bandit_tests.rs"]` → `#[path = "../bandit_tests.rs"]` because the path resolves relative to `bandit/mod.rs` (now in the `bandit/` directory), but the actual test file lives at `src/bandit_tests.rs`.
- Imports in `session.rs`: `ReviewMetrics`, `SafePhasedState`, `TrialLog`, `TrialRecord` use `crate::` prefix (not `super::`) because they live at the crate root, not in the `bandit` module.
- `make_stats` is the only function that needed visibility change (`fn` → `pub(super) fn`) — it's used by both `BanditPruner` (mod.rs) and `BanditSession` (session.rs).
- Pre-existing unrelated failure under `--all-features`: `workflow_lattice::tests::test_bench_lattice_vs_noop` (flaky perf gate; passes in isolation). Same class as the documented `jacobian_svd_r8x8_latency_gate` flake.

## Out of scope

- The `bandit_tests.rs` file itself (already a sibling; just keeps living where it is).
- Issue 178 (dd_tree post-split growth — separate investigation).
