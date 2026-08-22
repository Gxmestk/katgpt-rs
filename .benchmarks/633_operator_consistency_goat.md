# Bench 633 — Operator Consistency Metric GOAT gate (Issue 586)

**Session:** 2026-08-14 · executing `katgpt-rs/.issues/586` (BDH-CQ Research 479 follow-up)
**Feature:** `operator_consistency = ["bisimulation_operator_inference"]` (opt-in)
**Module:** `crates/katgpt-core/src/bisimulation/consistency.rs`
**Bench:** `cargo bench -p katgpt-core --features operator_consistency --bench bench_586_operator_consistency_goat`
**Machine:** M3 Max arm64, release (`bench` profile). CPU-only (GPU-exclusivity rule N/A).

## What shipped

`rule_consistency(&[ApplicationOutcome]) -> ConsistencyReport` — the modelless,
zero-alloc rule-application consistency metric distilled from BDH-CQ §6.4
(arXiv:2608.09888): the paper's 18.5-point test-pair (77.9%) vs strict-task
(59.4%) gap, decomposed into:

- **3-bin task histogram** (strict / partial / none) + `application_accuracy`
  + `strict_task_accuracy` + `gap`.
- **Sigmoid-guarded gap**: `gap_shrunk = gap·n/(n+4)` + `gap_confidence =
  sigmoid(0.5·(n_tasks−4))` — small samples don't produce overconfident
  "inconsistent" verdicts (sigmoid, not softmax, per global rule).
- **Structure-preservation breakdown**: localized errors (extrapolation
  signature) vs construction failures (execution signature) +
  `extrapolation_share`.
- **`ConsistencyRegime`**: `Consistent` / `NoisyFlaky` / `ComplexityClustered
  { level }` / `Ambiguous` — cluster detection = clean prefix (fail rate ≤
  0.25) + broken suffix (fail rate ≥ 0.75, ≥ 2 suffix failures).
- **`promotion_verdict`** gate on `infer_operators` output: `Promote` /
  `SeekExemplar { level }` (checked FIRST — coverage failures present as low
  accuracy with a cheap targeted repair) / `Hold` / `Reject`.

## GOAT gate results — ALL PASS (G1 + G2 + G4)

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| **G1** regime separation | 3 fixtures → 3 regimes + 3 verdicts | consistent→`Consistent`/`Promote` (acc 0.990); flaky→`NoisyFlaky`/`Hold` (gap 0.292); clustered→`ComplexityClustered{4}`/`SeekExemplar{4}` (acc 0.667) | **PASS** |
| **G2** latency @ N≤64 | < 1 µs | N=16: 84ns · N=64: 208ns (4.8× under) · N=256: 583ns (report-only) | **PASS** |
| **G3** no-regression | existing suite unaffected | 14 new module tests pass; 1936 existing pass with feature on | **PASS** (see caveat) |
| **G4** alloc-free | 0 steady-state allocs | 0 allocations / 100 calls @ N=64 (CountingAllocator) | **PASS** |

Paper anchors pinned in unit tests: 19/30 pairs vs 2/10 tasks (gap 0.433,
partial-heavy); nesting 19/24 → `ComplexityClustered{5}`; multi-level suffix
(6,7,8 broken) → boundary reported at the LOWEST failing level (6);
single stray failure ≠ cluster; graded degradation → `Ambiguous`; empty
input → `Ambiguous` + `Hold` (no evidence is "gather more", not "discard").

### G3 caveat (unrelated, pre-existing)

`subspace_phase_gate::tests::jacobian_svd_r8x8_latency_gate` failed during
the G3 run (debug-mode latency guard: 101µs vs its threshold). Verified
**pre-existing and unrelated**: it fails identically on default features
(my module does not compile into that build). Load-sensitive (sibling cargo
processes were running). Not caused by this change; left unfixed per the
don't-fix-unrelated rule.

## T4 verdict: stays opt-in

`operator_consistency` implies `bisimulation_operator_inference` and mirrors
its **opt-in-forever policy** — the parent is an opt-in primitive by design
("promotion to default-on is not planned", Plan 324), and a submodule of an
opt-in primitive cannot be default-on coherently (the parent would not even
compile in the default build). No katgpt-rs `default` change.

## Design notes (recorded for Issue 672 / Research 479)

1. **Lookup-binding divergence**: the paper's trained model partially
   transfers a single exemplar at level c to neighboring levels (ordering
   length 8: 0/24 → 13/24 from one demo at 8). The modelless analog is
   demonstrated-value lookup — one exemplar repairs exactly level c. The
   policy equivalent is **re-target after re-measure**: seek at the cluster
   boundary, re-run the metric, seek at the new boundary. Issue 672's PoC
   implements exactly that loop.
2. **Flaky ≠ coverage**: `NoisyFlaky` (uniform i.i.d. failure) must NOT
   trigger exemplar-seeking — a demonstration does not repair noise. The
   gate returns `Hold` (re-estimate reliability — CLR `should_write_memory`
   territory) even when accuracy is above the floor. Pinned by the
   `regime_b_random_flakiness` test.
3. **Clustered checks precede the accuracy floor** in `promotion_verdict`:
   the paper's coverage failures present as low overall accuracy (ordering
   family: 0% at the failing levels), which would otherwise hit `Reject`
   before the cheap targeted repair could fire.

## Consumers

- riir-ai Issue 672 (`demo_coverage_curiosity` feature in `riir-games/swarm`)
  — the exemplar-seeking policy keyed on `ComplexityClustered { level }`.
- Future: engram write policy / freeze gates (promotion requires consistency
  above threshold); operator-inference bench diagnostics (execution vs
  extrapolation classification).
