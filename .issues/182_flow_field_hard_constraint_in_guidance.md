# Issue 182: Flow Field as Hard Constraint in Guidance (Implement Proposal 006)

> **Type:** optimization / refactor
> **Priority:** P2
> **Filed:** 2026-07-18
> **Status:** OPEN
> **Proposal:** [006](../.proposals/006_flow_field_hard_constraint_in_guidance.md)
> **Closes:** [Issue 546 (riir-ai)](../../../riir-ai/.issues/546_lacam_multistep_escalation_ht_chantry.md) — if G1 ht_chantry ≥ 0.30 lands
> **Repo:** katgpt-rs (substrate is `crates/katgpt-core/src/multi_agent_path/`)

## Context

Proposal 006 (this session) analyzed the existing `GridFlowField` (Plan 440 Issue 149/150)
and identified **three independent root causes** for its near-zero effect on throughput
(Benchmark 440 §Issue 150: −0.6% to +1% across 4 maps):

1. **A\* guidance ignores the flow field.** The path planner
   (`SpaceTimeGuidance::astar_for_agent` in `local_guidance.rs:459-555`) never queries the
   flow field. Agents are *routed* through corridors in the wrong direction before PIBT runs.
2. **`flow_mismatch` is position 2 in the cost tuple**, behind `guidance_mismatch`. In a
   head-on corridor, both agents have `guidance_mismatch = 0` for the forward move, so the
   flow tiebreak is never consulted.
3. **All corridors are assigned `sign = +1`.** Any agent whose goal is in the `−` direction
   has no flow-legal corridor to use.

The Issue 546 deadlock-chain diagnostic (`katgpt-rs` commit `2a8c378d`) confirmed the
consequence: P95 max-cluster-size = 8 on ht_chantry. Multi-step LaCAM cannot close the gap at
any tractable depth. The fix must come from the guidance layer — this issue.

## The work

Implement Proposal 006 in three phases. Each phase is independently shippable.

### Phase 1: Bi-directional corridor pairing (root cause #3)

**Files:** `crates/katgpt-core/src/multi_agent_path/flow.rs`

- [ ] **P1.1** Add `sign = 0` (bidirectional / dual) variant. Update `FlowDirection::mismatch`
  to return `0` for `sign = 0` (any direction allowed).
- [ ] **P1.2** In `GridFlowField::from_map`, for **2-wide corridors**: assign one cell
  `sign = +1`, the partner cell `sign = −1`. This creates a two-lane highway.
- [ ] **P1.3** For **1-wide corridors**: walk each maximal chain between two junctions,
  number corridors in BFS order from map centroid, alternate `sign = +1 / −1` by parity.
- [ ] **P1.4** Add articulation-point detection (Tarjan's bridge-finding or a
  simpler SCC-based check). At 1-wide corridor segments that are the only path between two
  regions, assign `sign = 0` (accept deadlock risk to preserve reachability).
- [ ] **P1.5** Tests: bi-directional 2-wide corridor test, 1-wide parity test, articulation-
  point fallback test. Update existing flow field tests that assumed `sign = +1` everywhere.

### Phase 2: Flow-respecting A\* (root cause #1)

**Files:** `crates/katgpt-core/src/multi_agent_path/local_guidance.rs`, `mod.rs`

- [ ] **P2.1** Add `set_flow_field` method to the `LocalGuidanceSource` trait (default no-op).
- [ ] **P2.2** `SpaceTimeGuidance` stores an optional `Arc<dyn FlowField<P>>` (default
  `NoFlow`). Add `.with_flow_field(flow)` builder method mirroring `LifelongLaCam::with_flow_field`.
- [ ] **P2.3** In `astar_for_agent` and `astar_for_agent_flat`, add a hard pruner in the
  neighbor-expansion loop:
  ```rust
  if self.flow_field.mismatch(&pos, &neighbor) == 1 {
      continue; // Flow-forbidden neighbor — skip.
  }
  ```
- [ ] **P2.4** `LifelongLaCam::tick` passes its flow field to guidance via the new
  `set_flow_field` setter before `compute_guidance`. (Both PIBT and guidance now consult the
  same flow field.)
- [ ] **P2.5** Tests: a synthetic corridor map where A\* previously routed through the
  corridor against flow must now route around. Verify path is flow-legal end-to-end.

### Phase 3: Demote `flow_mismatch` in PIBT cost tuple (root cause #2)

**Files:** `crates/katgpt-core/src/multi_agent_path/pibt.rs`

- [ ] **P3.1** In `Candidate::lexicographic_cmp`, move `flow_mismatch.cmp(...)` from position 2
  to position 4 (after `hindrance`).
- [ ] **P3.2** Update the doc comment on `Candidate` and the module-level comment in
  `pibt.rs:14-15` to reflect the new ordering.
- [ ] **P3.3** Re-run all existing PIBT tests. Any test that asserted flow_mismatch dominance
  over goal_dist needs updating (these tests encoded the bug).

### Phase 4: GOAT gate

**Files:** `crates/katgpt-core/benches/bench_440_lllg_paper_repro.rs`,
`crates/katgpt-core/examples/ht_chantry_deadlock_chain_diagnostic.rs` (re-run)

- [ ] **P4.1** Run bench_440 with the redesigned flow field on all 4 maps. Targets:
  - empty-48-48: ≥ 0.68 (no regression — was 0.68)
  - random-64-64-10: ≥ 0.69 (no regression — was 0.69)
  - warehouse: ≥ 0.30 (target — was 0.41, should stay or improve)
  - ht_chantry: **≥ 0.30** (close Issue 546 — was 0.27-0.28)
- [ ] **P4.2** Re-run the deadlock-chain diagnostic. Target (G-flow): P95 max-cluster-size
  drops by ≥ 50% (8 → ≤ 4).
- [ ] **P4.3** Re-run bench_453 (one-step LaCAM GOAT). Verify G6c/G-col collision-freedom
  stays perfect (1.000 / 0.0%).
- [ ] **P4.4** Latency: median tick ≤ 100ms at 800 agents on non-maze maps; ht_chantry ≤ 500ms.

### Phase 5: Promotion / verdict

- [ ] **P5.1** If P4.1 ht_chantry ≥ 0.30 AND P4.2 cluster-size drops: **close Issue 546** (update
  `riir-ai/.issues/546_*` to RESOLVED, point to this issue + Proposal 006).
- [ ] **P5.2** If P4.1 ht_chantry < 0.30 BUT P4.2 cluster-size drops: **mark proposal
  architecturally correct but insufficient** — accept the marginal G1 fail as steady-state
  (Issue 546 path 2). Document the verdict in `.benchmarks/440_lllg_paper_repro_goat.md`.
- [ ] **P5.3** If P4.1 ht_chantry < 0.30 AND P4.2 cluster-size unchanged: **REVERT** — the
  proposal's mechanism hypothesis is wrong.
- [ ] **P5.4** Update `.benchmarks/440_lllg_paper_repro_goat.md` with the new G1 results per
  map. Note whether 4/4 PASS or 3/4 PASS + ht_chantry steady-state.
- [ ] **P5.5** Update `riir-ai/.issues/546_*` status (RESOLVED if 4/4, or note "flow-field
  redesign landed but insufficient — accepted marginal fail" if 3/4).

## Out of scope

- **Dynamic flow reversal.** Future extension. Out of scope here (would break replay determinism).
- **Global flow-balanced assignment** (LP / min-cost-flow). Future enhancement if (A)+(B)+(C)
  insufficient.
- **Multi-step LaCAM** (Issue 546 original plan). Permanently deferred per the diagnostic.

## Acceptance criteria (overall)

- [ ] All 4 maps run cleanly with the redesigned flow field (no panics, no test failures).
- [ ] P95 max-cluster-size on ht_chantry drops by ≥ 50% vs the Issue 546 baseline.
- [ ] No regression on empty/random/warehouse throughput.
- [ ] Either ht_chantry ≥ 0.30 (close Issue 546) OR a documented verdict in
  `.benchmarks/440_lllg_paper_repro_goat.md` explaining the residual gap.
- [ ] All clippy + lib tests pass under `--features lacam_escalation` (the full multi_agent_path
  feature set).

## References

- [Proposal 006](../.proposals/006_flow_field_hard_constraint_in_guidance.md) — the architectural argument
- [Issue 546 (riir-ai)](../../../riir-ai/.issues/546_lacam_multistep_escalation_ht_chantry.md) — DEFERRED, this issue closes it if G1 lands
- [Benchmark 440](../.benchmarks/440_lllg_paper_repro_goat.md) — the parent G1 gate
- [Benchmark 453](../.benchmarks/453_lacam_escalation_goat.md) — one-step LaCAM GOAT (must not regress)
- [Plan 440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md) — LLLG substrate
