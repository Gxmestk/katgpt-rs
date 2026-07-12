# Issue 036 — Feature-Promotion Struct Bloat in BanditPruner/AbsorbCompressLayer

**Filed:** 2026-07-03
**Priority:** P2 (perf — not blocking, but accounts for remaining ~17% gap)
**Related:** `.benchmarks/372_crate_refactor_lto_regression_fix.md`

## Problem

Between May 29 and June 12, 2026, several features were promoted to default-on
that add fields to `BanditPruner` and `AbsorbCompressLayer`:

| Feature | Fields added to `BanditPruner` |
|---------|-------------------------------|
| `partial_scoring` | `partial_scorer: Option<Box<dyn PartialScorer>>` |
| `idea_divergence` | `idea_divergence: Option<IdeaDivergence>` + `arm_score_vectors: Vec<Vec<f32>>` |
| `skill_lifecycle` | `memory: PrunerMemory` |
| `bandit` | `shared_stats: Option<Arc<SharedBanditStats>>` |

Each field is individually justified (passed GOAT G1–G7 quality gates). But
collectively, they push `BanditPruner` from ~3 fields (May 29) to 13+ fields,
spreading the struct across multiple cache lines. The `Bandit update()`
benchmark (tight inner loop touching only `self.stats`) regressed ~30% from
the struct layout change alone (separate from the LTO fix in Bench 372).

## Root Cause

The GOAT gate checks the feature being promoted, but does NOT check whether
adding fields to a shared struct degrades OTHER benchmarks that don't use the
feature. This is a systemic blind spot.

## Proposed Fix (deferred — not blocking)

Group all optional/feature-gated extension fields behind a single `Box<Extensions>`:

```rust
pub struct BanditPruner<P: ScreeningPruner> {
    inner: P,
    strategy: BanditStrategy,
    stats: BanditStats,           // ← hot path: update(), relevance()
    thompson_cache: Vec<f32>,     // ← hot path: prepare_episode()
    // ── cold path: grouped behind one indirection ──
    extensions: Option<Box<BanditExtensions>>,
}

struct BanditExtensions {
    shared_stats: Option<Arc<SharedBanditStats>>,
    dual_cutoff: f32,
    soft_route: bool,
    soft_route_tau: f32,
    partial_scorer: Option<Box<dyn PartialScorer>>,
    idea_divergence: Option<IdeaDivergence>,
    arm_score_vectors: Vec<Vec<f32>>,
    memory: PrunerMemory,
    soft_route_scores: Option<Mutex<Vec<f32>>>,
    soft_route_weights: Option<Mutex<Vec<f32>>>,
}
```

This keeps the hot-path fields (`inner`, `strategy`, `stats`, `thompson_cache`)
in one cache line, and pushes all rarely-used extension fields behind a single
pointer indirection. The `Option<Box<>>` is 8 bytes when `None` (the fast path
for benchmarks that don't use extensions).

### Expected Gain

Bandit update() should recover the remaining ~17% gap (415M → ~500M) by
keeping `BanditStats` in a hotter cache line.

## Tasks

- [-] Benchmark `BanditPruner` struct size before/after the `Box<Extensions>` refactor — DEFERRED (2026-07-04 verification attempt: workspace build blocked by sibling agent's speculative-crate refactor cycle; decision is stay-deferred regardless — see "Verification attempt" below).
- [-] Implement `BanditExtensions` grouping — DEFERRED (gain within measurement noise; refactor risk not justified).
- [-] Verify all 130 katgpt-pruners tests pass — DEFERRED (blocked on T1).
- [-] Run full bench suite, confirm Bandit update() ≥ 480M (within 5% of peak) — DEFERRED (re-evaluation trigger documented below).
- [-] Apply same pattern to `AbsorbCompressLayer` if bench shows gain — DEFERRED (AbsorbCompress already *exceeds* May-29 peak after Bench 372 fixes: 60.1M > 57.4M).

## Deferral Rationale

This is P2 because:
1. The LTO + lazy-Mutex + Vec-compress fixes (Bench 372) already recovered the
   biggest regressions (Bandit +69%, AbsorbCompress +190%).
2. The remaining 17% gap is within run-to-run thermal variance (~25%).
3. The `Box<Extensions>` refactor touches every constructor and every field
   access — higher risk than the Bench 372 fixes.

## Verification attempt (2026-07-04; re-checked 2026-07-09)

Attempted to re-benchmark to get current numbers. **Blocked at the time** — the workspace build was broken by a sibling agent's in-progress speculative-crate refactor (cyclic dependency: `katgpt-pruners` → `katgpt-speculative` → `katgpt-pruners`).

**Update (2026-07-09): that blocker is RESOLVED.** The cycle was broken by Plan 388
Phase 3, which moved the `katgpt-pruners` reference in `katgpt-speculative` under
`[dev-dependencies]` (`crates/katgpt-speculative/Cargo.toml:296-303`) — dev-deps
do not propagate to dependents, so there is no real lib cycle. Verified by grep:
the only `katgpt-pruners` reference in `katgpt-speculative/Cargo.toml` is the
dev-dep on line 303. The build is therefore no longer broken *by this issue's
concerns* (the working tree is currently dirty with unrelated sibling WIP in
`crates/katgpt-core/src/*.rs` and benches, which is out of scope here).

**Decision: stay deferred.** Even if the build were green, the case for doing this refactor now is weak:
1. The 502M "peak" (May 29) was partially thermal-inflated — `cooldown()` was a no-op before commit `ef78b555` (2026-06-12), per Bench 372 §"Remaining Gap". The real regression target is lower than 502M.
2. Run-to-run variance was 321M–415M (~25%) on the *same binary* — the 17% gap (415M vs 502M) is within that noise band.
3. The refactor touches 31 field-access sites + all constructors (high risk for a gain that may be unmeasurable).
4. AbsorbCompress compress() — the other benchmark in this family — already *exceeds* the May-29 peak (60.1M > 57.4M) after the Bench 372 fixes.

**Re-evaluation trigger:** revisit if profiling on non-thermal-throttled hardware (e.g., a dedicated benchmark machine with consistent cooling) shows Bandit update() consistently below 420M across 5+ runs. Until then, the expected gain is within measurement noise and the refactor risk is not justified.

## Verification attempt (2026-07-12)

Ran the re-evaluation trigger check via `cargo run --release --example
issue_036_bandit_bench --features bandit` (5 runs, 5000 iters, 100 arms,
matching the `bench_bandit()` harness in `src/benchmark/distillation.rs`):

| Run | Throughput (ops/s) |
|-----|-------------------:|
| 1   |          363,636,364 |
| 2   |          375,009,375 |
| 3   |          370,370,370 |
| 4   |          369,221,681 |
| 5   |          375,009,375 |
| **Mean** |  **370,649,433** |

All 5 runs are below 420M → the re-evaluation trigger's *numeric*
condition is met.

**However, decision: stay deferred.** Reasons:
1. **Thermal caveat unmet.** The trigger explicitly requires
   "non-thermal-throttled hardware." This Mac (Apple Silicon) cannot
   guarantee that — the 370M mean is below the 415M post-fix number from
   Bench 372, consistent with thermal throttling rather than a new
   regression.
2. **Within documented variance.** The 370M mean falls squarely within the
   321M–415M variance range documented in the issue (25% spread on the
   *same binary*). No new regression is demonstrated.
3. **Cache-line analysis suggests minimal gain.** With `P = NoScreeningPruner`
   (ZST, the benchmark config), `BanditStats` (48 bytes: 2 × Vec) starts at
   offset ~1 and fits within the first 64-byte cache line. The cold fields
   start at offset ~73 (second cache line) but `update()` never accesses
   them, so the second cache line is not loaded. Grouping cold fields behind
   `Box<Extensions>` replaces 10 fields with one 8-byte pointer at offset 73
   — this does not change the first-cache-line layout that `update()`
   depends on.
4. **No new struct bloat since Bench 372.** Git log confirms no new fields
   were added to `BanditPruner` since commit `92c8f90f` (Bench 372, 2026-07-03).
   The commits since are perf optimizations and file-splitting refactors, not
   field additions.

The benchmark example (`examples/issue_036_bandit_bench.rs`) is kept for
future re-evaluation on dedicated hardware.

## TL;DR

Feature promotions (May 29 → June 12) bloated `BanditPruner` from 3 → 13
fields, causing cache-line sprawl. Fix: group cold fields behind
`Box<Extensions>`. Deferred as P2 — the acute regressions are fixed in Bench 372.
