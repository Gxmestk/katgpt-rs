# Plan 453 — Bounded One-Step LaCAM Escalation (Phase 5)

**Date:** 2026-07-15
**Status:** OPEN
**Repo:** katgpt-rs (substrate)
**Feature flag:** `lacam_escalation` (opt-in; promotion candidate)
**Blocks:** Issue 154 (DEFERRED → this plan), riir-ai Issue 516 T5, riir-ai Proposal 023
**Research:** [424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md) (LLLG), this plan Phase 1 (LaCAM paper distillation)
**Prior art:** Issues [140](../.issues/140_pibt_priority_inheritance_and_warmstart_integration.md) (PI revert), [143](../.issues/143_lacam_escalation_full_pibt.md) (shuffled retry), [154](../.issues/154_pibt_vertex_collisions_priority_inheritance.md) (PI prior-art finding)
**Paper:** Okumura 2023, "LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding", AAAI 2023 — [project page](https://kei18.github.io/lacam/), [code](https://github.com/Kei18/lacam)
**Benchmark:** [440](../.benchmarks/440_lllg_paper_repro_goat.md) (substrate G1–G4 gate; Phase 5 next steps §lines 743–766)

---

## TL;DR

Replace the fake "LaCAM escalation" (shuffled-priority retries of greedy PIBT,
`DEFAULT_LACAM_RETRIES = 2` in `pibt.rs`) with a **bounded one-step LaCAM** —
the constraint-tree search from Okumura 2023, applied to a single tick. The
current code is NOT real LaCAM: it shuffles priority orderings and picks the
result with fewest stuck agents. Real LaCAM does **systematic backtracking
over joint configurations** via a constraint tree, using recursive PIBT as the
configuration generator.

**The critical insight from reading the reference implementation** (`Kei18/lacam/src/planner.cpp`):
LaCAM **does** use recursive PIBT (`funcPIBT` calls itself for priority
inheritance) — but it works because the **constraint tree** bounds the
recursion and systematically explores assignments. Issues 140/143 collapsed
throughput because they used recursive PI **without** the constraint tree.
The constraint tree is the missing piece. This is why "PI is prior art"
(Issue 154) and "LaCAM is not prior art" can both be true: LaCAM = recursive
PIBT + constraint tree, and only the recursive PIBT half was tried.

**Honest risk:** Research 424 §1.5 documents that LaCAM* anytime refinement
*degrades* lifelong throughput. But that finding is about **optimizing the
windowed plan** (multi-step LaCAM* on the planning horizon). This plan
implements **one-step** LaCAM (find a collision-free joint action for the
current tick), which is a different objective. The throughput risk remains
and is addressed by the bounded budget + fallback to greedy PIBT.

**Promotion criteria:** if G6c (collision-freedom) improves to ≥ 0.50
**without** collapsing G1 (throughput ratio ≥ 0.30 on all maps) or G4
(latency < 500ms), promote `lacam_escalation` to default-on. If G1 collapses
(like Issues 140/143), stay opt-in and document the negative result.

---

## 1. The Gap: Fake LaCAM vs Real LaCAM

### 1.1 What the current code does (fake LaCAM)

`pibt_step` in `pibt.rs:201-308`:

1. Run greedy PIBT once → get `(moves, stuck)`.
2. If `stuck.len() >= MIN_STUCK_FOR_RETRY` (20), retry up to
   `DEFAULT_LACAM_RETRIES` (2) times with **shuffled priority orderings**
   (`shuffle_order`).
3. Return the result with the fewest stuck agents.
4. Remaining stuck agents are forced to wait in place → **vertex collisions**
   (Issue 154 root cause).

This is **priority shuffling**, not LaCAM. It cannot find collision-free
configurations that require changing *which cell* an agent goes to, only
*which order* agents are processed.

### 1.2 What real LaCAM does (from the reference implementation)

`Kei18/lacam/src/planner.cpp` — `Planner::solve()` + `get_new_config()`:

```
High-level: DFS over configurations (OPEN stack + CLOSED set)
  For each configuration S:
    Low-level: constraint tree (BFS queue of Constraint objects)
      Pop constraint M from S->search_tree
      If M->depth < N:
        Agent i = S->order[M->depth]  (next agent in priority order)
        For each candidate cell u in neighbors(C[i]) ∪ {C[i]}:
          Push Constraint(M, i, u)  (force agent i to cell u)
      get_new_config(S, M):
        Apply constraints from M (force specific agents to specific cells)
        Run funcPIBT for unconstrained agents (RECURSIVE PIBT with PI)
        If PIBT succeeds and config is collision-free → new config
        Else → reject, try next constraint
```

**The constraint tree** is the key mechanism. It systematically explores
assignments: agent 0 tries cell A, then cell B, then cell C; for each, agent 1
tries its candidates; etc. This is exponential in theory but bounded in
practice by (a) the deadline and (b) PIBT resolving most agents without
constraints.

**Recursive PIBT (`funcPIBT`)** is used *inside* `get_new_config` for agents
not fixed by constraints. It DOES use priority inheritance (`funcPIBT(ak)`
recursive call when agent `ai` wants a cell occupied by `ak`). But the
recursion is bounded by the constraint tree — the constraint tree provides
the backtracking that makes PI safe.

### 1.3 Why Issues 140/143 collapsed throughput

Issues 140/143 implemented **recursive PIBT with priority inheritance** but
**without the constraint tree**. Without the constraint tree, the recursion
has no backtracking — a single push can cascade (A pushes B, B pushes C, ...)
and stall the entire system. The constraint tree is what makes recursive PI
safe: when a push cascade would stall, the constraint tree backtracks and
tries a different assignment for the root agent.

**This is why "PI is prior art" (Issue 154) and "LaCAM is not prior art" can
both be true.** PI alone was tried and reverted. LaCAM = PI + constraint
tree, and only the PI half was tried.

---

## 2. Design: Bounded One-Step LaCAM

### 2.1 Scope: one-step, not multi-step

Full LaCAM is a **multi-step planner** (finds complete paths from start to
goal via high-level config search). Our use case is **one-step**: given the
current configuration `Q_t`, find a collision-free next configuration
`Q_{t+1}` for the current tick. This is exactly what LaCAM's `get_new_config`
does — we don't need the high-level config search.

**What we keep:** the constraint tree + recursive PIBT (LaCAM's low-level
search).

**What we drop:** the high-level config search (OPEN/CLOSED over
configurations). We only need one next config, not a sequence.

### 2.2 The algorithm (bounded one-step LaCAM)

```
fn lacam_escalation_step(config, guidance, goals, priorities, hindrance, flow, neighbors, rng, budget):
    # Phase A: try greedy PIBT (fast path — current code)
    (moves, stuck) = greedy_pibt_pass(...)
    if stuck.is_empty():
        return Ok(JointAction::new(moves))  # no stuck agents, done

    # Phase B: constraint-tree search (the real LaCAM mechanism)
    # Build the constraint tree, bounded by budget.max_nodes
    constraint_tree = ConstraintQueue::new()
    constraint_tree.push(Constraint::empty())

    order = compute_priority_order(n, priorities)  # process order

    nodes_explored = 0
    while let Some(constraint) = constraint_tree.pop():
        nodes_explored += 1
        if nodes_explored > budget.max_nodes:
            break  # budget exhausted, fall back to greedy result

        # Expand constraint: if depth < N, push children
        if constraint.depth < n:
            i = order[constraint.depth]
            candidates = neighbors(config.pos(i)) ∪ {config.pos(i)}
            for u in shuffled(candidates, rng):  # randomize for diversity
                constraint_tree.push(constraint.child(i, u))

        # Try to build a collision-free config with this constraint
        match get_new_config(config, &constraint, guidance, goals, ...):
            Ok(moves) if is_collision_free(&moves, config):
                return Ok(JointAction::new(moves))  # found!
            _ => continue  # constraint rejected, try next

    # Phase C: budget exhausted — fall back to greedy PIBT result
    # (current behavior: stuck agents wait in place, may vertex-collide)
    place_stuck_as_wait(&mut best_moves, &best_stuck, config)
    Ok(JointAction::new(best_moves))
```

### 2.3 `get_new_config` (the core LaCAM primitive)

Adapted from `Kei18/lacam/src/planner.cpp:get_new_config`:

```
fn get_new_config(config, constraint, guidance, goals, hindrance, flow, ...):
    # Apply constraints: force specific agents to specific cells
    occupied_next = HashSet::new()
    for (agent_i, cell_u) in constraint.who_where():
        # Check vertex collision
        if occupied_next.contains(cell_u):
            return Err(ConstraintRejected)
        # Check swap collision
        if let Some(j) = current_to_agent.get(cell_u):
            if occupied_next maps j to config.pos(agent_i):
                return Err(ConstraintRejected)  # swap
        occupied_next.insert(cell_u)
        moves[agent_i] = Some(cell_u)

    # Run recursive PIBT for unconstrained agents
    for i in order:
        if moves[i].is_none():
            if not func_pibt_recursive(i, ...):
                return Err(ConstraintRejected)  # PIBT failed

    return Ok(moves)
```

### 2.4 `func_pibt_recursive` (LaCAM's recursive PIBT with PI)

Adapted from `Kei18/lacam/src/planner.cpp:funcPIBT`. This is the piece
Issues 140/143 implemented standalone (without the constraint tree) and
collapsed throughput. **Here it is safe** because the constraint tree
backtracks when the recursion stalls.

```
fn func_pibt_recursive(agent_i, config, moves, occupied_next, ...):
    current = config.pos(agent_i)
    candidates = neighbors(current) ∪ {current}
    # Sort by lexicographic cost (same as current greedy PIBT)
    candidates.sort_by(lexicographic_cost(agent_i, guidance, goals, ...))

    for u in candidates:
        # Vertex collision check
        if occupied_next.contains(u):
            continue
        # Swap collision check
        if let Some(j) = current_to_agent.get(u):
            if moves[j] == Some(current):
                continue

        # Reserve cell
        occupied_next.insert(u)
        moves[agent_i] = Some(u)

        # Empty cell or staying → success
        let occupant = current_to_agent.get(u)
        if occupant.is_none() || u == current:
            return true

        # Priority inheritance: push the occupant
        let k = occupant.unwrap()
        if moves[k].is_none():
            if func_pibt_recursive(k, ...):
                return true  # occupant moved, we get the cell
            else:
                continue  # occupant couldn't move, try next candidate

        return true

    # Failed — stay in place
    occupied_next.insert(current)
    moves[agent_i] = Some(current)
    return false
```

### 2.5 Budget bounds (the real-time guarantee)

The constraint tree is exponential in theory. We bound it with:

| Bound | Default | Rationale |
|---|---|---|
| `max_nodes` | 1000 | Cap on constraint-tree nodes explored per tick. At ~1μs/node, this is ~1ms overhead. |
| `max_stuck_ratio` | 0.0 | Only escalate when stuck agents exist (always true if we reach Phase B). |
| `time_budget_us` | 5000 (5ms) | Wall-clock cap. Checked every 64 nodes to reduce branch overhead. |

When the budget is exhausted, fall back to the greedy PIBT result (current
behavior). This guarantees the real-time tick is never blocked.

**Tuning plan:** the defaults are conservative. Phase 3 benchmarks sweep
`max_nodes ∈ {100, 500, 1000, 5000}` to find the knee — the point where more
search doesn't improve collision-freedom but starts hurting latency.

---

## 3. Phased Implementation

### Phase 1 — Research distillation (paper → concrete spec) ✅ DONE

- [x] **T1.1** Fetch and distill the LaCAM paper (Okumura 2023, AAAI). Source:
      [project page](https://kei18.github.io/lacam/),
      [reference code](https://github.com/Kei18/lacam/blob/master/lacam/src/planner.cpp).
      Focus: §3 (the two-level search), §4.1 (`get_new_config`), §4.2
      (`funcPIBT`). Output: [`.research/441_lacam_constraint_tree_distillation.md`](../.research/441_lacam_constraint_tree_distillation.md)
      documenting the algorithm shape, data structures, and the
      constraint-tree → recursive-PIBT composition. Cross-ref Research 424
      §1.5 (the LaCAM* anytime-refinement negative result — confirmed our
      one-step scope avoids it, see Research 441 §2.3).
- [x] **T1.2** Vocabulary-translate the paper against the codebase. Grep
      `crates/katgpt-core/src/` for: `constraint`, `backtrack`, `recursive`,
      `priority inheritance`, `search_tree`, `constraint_tree`. Result:
      `constraint` appears only in `arg/policy.rs` (`PolicyConstraints` —
      governance, unrelated to MAPF). The constraint tree is NOT shipped
      under any name (Research 441 §3 documents the full vocabulary table).
- [x] **T1.3** Document the prior-art distinction: PI alone (Issues 140/143,
      reverted) vs PI + constraint tree (LaCAM, this plan). Issue 154 updated
      with cross-ref to Plan 453 (commit `a6cf51c2`). The key sentence:
      "PI is prior art; LaCAM is PI + constraint tree, and only the PI half
      was tried." Research 441 §5 has the full prior-art comparison table.

### Phase 2 — Implement bounded one-step LaCAM

- [ ] **T2.1** Create `crates/katgpt-core/src/multi_agent_path/lacam.rs` behind
      `feature = ["lacam_escalation"]`. Module structure:
      - `Constraint { who: Vec<usize>, where_cells: Vec<P>, depth: usize }`
      - `ConstraintQueue` (FIFO queue of constraints — LaCAM uses BFS-style)
      - `EscalationBudget { max_nodes, time_budget_us }`
      - `lacam_escalation_step(...)` — the public entry point (§2.2)
      - `get_new_config(...)` — §2.3
      - `func_pibt_recursive(...)` — §2.4
- [ ] **T2.2** Wire `lacam_escalation_step` into `pibt_step` behind the
      feature flag. When `lacam_escalation` is ON and stuck agents exist,
      call `lacam_escalation_step` instead of the shuffled-retry loop. When
      OFF, keep the current shuffled-retry behavior (back-compat).
- [ ] **T2.3** Reuse the existing `Candidate` struct and lexicographic cost
      from `pibt.rs` — do NOT duplicate the cost function. The recursive
      PIBT uses the same `⟨guidance_mismatch, flow_mismatch, goal_dist,
      hindrance, ε⟩` tuple.
- [ ] **T2.4** Reuse the O(1) collision-detection structures from Issue 516
      T1g (`current_to_agent: HashMap<P, usize>`, `committed_dests:
      HashSet<P>`). The constraint tree adds a third structure:
      `constrained_agents: HashSet<usize>` (agents fixed by the current
      constraint, skipped in the recursive PIBT loop).
- [ ] **T2.5** Add `EscalationBudget` to `GuidanceConfig` (or a new
      `LacamConfig`) with the defaults from §2.5. Consumers can tune via the
      existing config seam.
- [ ] **T2.6** Unit tests (`tests.rs`):
      - `test_lacam_resolves_stuck_agent` — construct a scenario where
        greedy PIBT produces a stuck agent, verify LaCAM finds a
        collision-free config.
      - `test_lacam_budget_fallback` — set `max_nodes = 0`, verify it falls
        back to greedy PIBT (no panic, returns a valid JointAction).
      - `test_lacam_collision_free_on_congested_grid` — the G6c-style
        scenario (60 agents, bottleneck), verify collision-free rate
        improves vs greedy PIBT.
      - `test_lacam_no_regression_on_open_map` — empty-48-48, verify
        throughput unchanged (LaCAM fast-path returns immediately when no
        stuck agents).
      - `test_func_pibt_recursive_bounded` — verify the recursion terminates
        (no infinite loop) on a synthetic deadlock.

### Phase 3 — Benchmark (G6c + G1 + latency)

- [ ] **T3.1** Port the riir-ai G6c scenario (`riir-ai/.benchmarks/516_g6c_collision_freedom_delta.md`)
      to a substrate-level benchmark. 60 agents, 20×20 grid, 6-cell bottleneck
      gap, 200 ticks. Measure: vertex collision rate, edge collision rate,
      stuck-agent count per tick. Run with `lacam_escalation` ON vs OFF.
- [ ] **T3.2** Re-run the G1 throughput benchmark (Benchmark 440 G1 section)
      with `lacam_escalation` ON. All 4 maps (empty-48-48, random-64-64-10,
      warehouse, ht_chantry), 800 agents. Measure: throughput ratio vs the
      0.30 threshold. **Critical: verify ht_chantry improves** (the maze
      deadlock case — this is where LaCAM's constraint tree should help
      most, per Benchmark 440 §"Next steps").
- [ ] **T3.3** Latency sweep: `max_nodes ∈ {100, 500, 1000, 5000}` on
      empty-48-48 (1000 agents, 100 ticks). Measure median + max per-tick
      latency. Find the knee — the point where latency exceeds the 500ms
      budget or stops improving collision-freedom.
- [ ] **T3.4** Write `.benchmarks/453_lacam_escalation_goat.md` with the full
      results: G6c collision-freedom table, G1 throughput table, latency
      sweep table, and the collision-free-vs-throughput tradeoff analysis.

### Phase 4 — GOAT gate

- [ ] **T4.1** **G1 (throughput):** all 4 maps ≥ 0.30 ratio. Target: ht_chantry
      improves from 0.01 → ≥ 0.30 (the maze case). If ht_chantry stays at
      0.01, LaCAM didn't help G1 — document why (constraint tree may not
      resolve multi-step corridor deadlocks in one step).
- [ ] **T4.2** **G-col (collision-freedom, NEW gate):** vertex collision rate
      ≤ 10% on the G6c scenario (vs current 62.5%). Target: G6c delta ≥ 0.50
      (the riir-ai Proposal 023 threshold). If collision rate stays > 50%,
      the constraint tree can't resolve the bottleneck in one step —
      document the physics (genuine bottleneck, no collision-free config
      exists without throughput collapse).
- [ ] **T4.3** **G3 (no-regression):** `cargo test -p katgpt-core --all-features`
      passes; `cargo clippy -p katgpt-core --all-features` clean. All
      existing PIBT tests pass with `lacam_escalation` ON (the feature
      replaces the shuffled retry, not the greedy fast path).
- [ ] **T4.4** **G4 (latency):** median per-tick ≤ 500ms on empty-48-48 (1000
      agents). Stretch: ≤ 100ms. The constraint tree adds overhead only when
      stuck agents exist; on open maps, the fast path is unchanged.
- [ ] **T4.5** **G-PI (no throughput collapse):** explicitly verify the
      Issue 140/143 failure mode does NOT recur. empty-48-48 throughput
      ratio stays ≥ 0.60 (vs the 0.02 collapse in Issue 140). This is the
      gate that proves the constraint tree makes recursive PI safe.

### Phase 5 — Promotion decision

- [ ] **T5.1** If G1 + G-col + G3 + G4 + G-PI all PASS: promote
      `lacam_escalation` to default-on. Update `Cargo.toml` default features.
      Update `pibt.rs` module docs (remove the "Collision profile" caveat —
      collisions are now prevented by construction when the budget suffices).
      Re-open riir-ai Proposal 023 → re-run G6c → promote `crowd_motion_lllg`.
- [ ] **T5.2** If G-col FAILS (collision rate still high): the substrate
      genuinely can't guarantee collision-freedom in this regime. Stay
      opt-in. Document the negative result. The riir-ai consumer should use
      the occupied-set baseline for guaranteed collision-freedom. Close
      Issue 154 as "won't fix — physics constraint".
- [ ] **T5.3** If G1 FAILS (ht_chantry still 0.01): LaCAM one-step can't
      resolve multi-step corridor deadlocks. Document and defer to a
      future multi-step LaCAM plan (the full high-level config search).
      The collision-freedom improvement (G-col) may still stand on its own.
- [ ] **T5.4** If G-PI FAILS (throughput collapse recurs): the constraint
      tree did NOT make recursive PI safe. This would be a surprising
      negative result — document the root cause. Stay opt-in. This would
      confirm that lifelong MAPF fundamentally trades collision-freedom
      for throughput on congested maps.

---

## 4. Honest Caveats and Risks

### 4.1 The throughput risk (the big one)

The entire history of this substrate (Issues 140, 143, 144, 154) shows that
**every mechanism that forces agents to yield collapses throughput on
congested maps**. LaCAM's constraint tree forces yielding (to achieve
collision-freedom). There is a real risk that G-PI fails — the constraint
tree doesn't make recursive PI safe enough, and throughput collapses again.

**Mitigation:** the bounded budget + greedy fallback. If the constraint tree
can't find a collision-free config quickly, we fall back to greedy PIBT
(collisions but throughput). The worst case is "no improvement" (same as
today), not "throughput collapse" (worse than today). The budget ensures we
never spend more than ~5ms/tick on the search.

**The honest expectation:** LaCAM should improve collision-freedom on
*reorderable* congestion (ht_chantry maze deadlocks — G1 target) but may NOT
improve it on *genuine bottlenecks* (G6c scenario — the 6-cell gap is
physically too narrow for 60 agents). The plan's Phase 5 has explicit
branches for each outcome.

### 4.2 The LaCAM* anytime-refinement negative result (Research 424 §1.5)

Research 424 documents that LaCAM* anytime refinement *degrades* lifelong
throughput. **This plan does NOT implement LaCAM* anytime refinement.** The
scope is one-step collision-freedom (find a collision-free joint action for
the current tick), not multi-step plan optimization. The negative result
applies to optimizing the windowed plan, which is a different objective.

T1.1 explicitly cross-checks this distinction against the paper.

### 4.3 The one-step limitation

Full LaCAM searches over *sequences* of configurations (multi-step paths).
This plan implements *one-step* LaCAM (the next configuration only). Some
congestion patterns require multi-step coordination (e.g., agents need to
back up two cells to clear a corridor). One-step LaCAM can't resolve these.
If G1 (ht_chantry) fails, this is likely why — T5.3 handles this case by
deferring to a future multi-step LaCAM plan.

### 4.4 The deterministic-replay constraint

The constraint tree uses a seeded RNG for candidate shuffling (like the
existing PIBT `ε` tiebreak). Deterministic replay is preserved: same seed +
same config → same constraint-tree exploration → same joint action. T2.6
includes a determinism test.

### 4.5 Allocation discipline

The constraint tree allocates `Constraint` objects. On the hot path (open
maps, no stuck agents), the constraint tree is never entered — zero
allocation overhead. On the congested path, allocations are bounded by
`max_nodes`. The `ConstraintQueue` uses `Vec<Constraint>` with
`Vec::with_capacity(max_nodes)` upfront. Per the global rule: no allocation
inside hot loops — the constraint tree is NOT a hot loop (it only runs when
stuck agents exist, which is the congested minority of ticks).

---

## 5. File Impact

| File | Change |
|---|---|
| `crates/katgpt-core/src/multi_agent_path/lacam.rs` | **NEW** — constraint tree, recursive PIBT, escalation step (behind `lacam_escalation` feature) |
| `crates/katgpt-core/src/multi_agent_path/pibt.rs` | Wire `lacam_escalation_step` into `pibt_step` behind the feature flag; keep shuffled-retry as the OFF-path fallback |
| `crates/katgpt-core/src/multi_agent_path/mod.rs` | Export `lacam_escalation_step` + `EscalationBudget`; update module docs |
| `crates/katgpt-core/src/multi_agent_path/tests.rs` | Add LaCAM-specific tests (T2.6) |
| `crates/katgpt-core/Cargo.toml` | Add `lacam_escalation` feature (opt-in); add to `multi_agent_path` feature deps |
| `.research/441_lacam_constraint_tree_distillation.md` | **NEW** — T1.1 output |
| `.benchmarks/453_lacam_escalation_goat.md` | **NEW** — T3.4 output |
| `.issues/154_*.md` | Update: reference this plan as the non-PI path forward; keep DEFERRED status until T5 |

---

## 6. References

- **LaCAM paper:** Okumura 2023, "LaCAM: Search-Based Algorithm for Quick
  Multi-Agent Pathfinding", AAAI 2023. [Project page](https://kei18.github.io/lacam/),
  [code](https://github.com/Kei18/lacam).
- **LLLG paper:** Arita & Okumura 2026, "Lifelong LaCAM with Local Guidance
  for Lifelong MAPF", AAAI 2026. arXiv:2605.16855. → Research 424.
- **PIBT paper:** Okumura et al. 2022, "PIBT: Scalable and Prioritization
  Planning for Multi-Agent Pathfinding", AIJ.
- **Prior art (reverted):** Issues 140 (recursive PI), 143 (shuffled retry),
  144 (swap technique), 154 (PI prior-art finding).
- **Substrate gate:** Benchmark 440 (G1–G4), §"Next steps" lines 743–766.
- **Consumer gate:** riir-ai Proposal 023 (G6c), riir-ai Issue 516 T5,
  riir-ai Benchmark 516 G6c.

---

## TL;DR

The current "LaCAM escalation" is fake — it shuffles priorities. Real LaCAM
(Okumura 2023) uses a **constraint tree** + **recursive PIBT**. The critical
insight from reading the reference code: LaCAM **does** use recursive PI
(the piece Issues 140/143 tried and reverted), but it works because the
**constraint tree** bounds the recursion and provides backtracking. Issues
140/143 collapsed throughput because they used recursive PI **without** the
constraint tree. This plan ships the constraint tree — the missing half.

**Scope:** one-step LaCAM (find a collision-free joint action for the
current tick), bounded by a node/time budget, with greedy-PIBT fallback.

**Risk:** the throughput collapse may recur (G-PI gate). The bounded budget
+ fallback ensures the worst case is "no improvement", not "collapse".

**Promotion:** if G-col (collision-freedom) + G1 (throughput) + G-PI (no
collapse) all pass → promote `lacam_escalation` to default-on, re-open
riir-ai Proposal 023, promote `crowd_motion_lllg`. If G-PI fails → stay
opt-in, document the negative result, the substrate genuinely trades
collision-freedom for throughput on congested maps.
