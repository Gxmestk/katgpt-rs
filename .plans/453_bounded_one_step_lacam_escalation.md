# Plan 453 — Bounded One-Step LaCAM Escalation (DONE)

**Date:** 2026-07-15
**Status:** ✅ DONE (all 5 phases complete; feature stays opt-in per T5.3)
**Repo:** katgpt-rs (substrate)
**Feature flag:** `lacam_escalation` (opt-in; T5.3 decision: stay opt-in)
**Resolves:** ~~Issue 154~~ (closed as fixed — G-col = 0.0%, vertex collisions eliminated)
**Consumer gates:** riir-ai Issue 516 T5 (unblocked for opt-in use), riir-ai Proposal 023 (stays REJECTED — substrate not default-on)
**Research:** [424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md) (LLLG), [441](../.research/441_lacam_constraint_tree_distillation.md) (Phase 1 distillation)
**Prior art:** Issues 140 (PI revert, removed), 143 (shuffled retry, removed), 154 (PI prior-art finding, closed as fixed by this plan)
**Paper:** Okumura 2023, "LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding", AAAI 2023 — [project page](https://kei18.github.io/lacam/), [code](https://github.com/Kei18/lacam)
**Benchmark:** [440](../.benchmarks/440_lllg_paper_repro_goat.md) (substrate G1–G4 gate), [453](../.benchmarks/453_lacam_escalation_goat.md) (this plan's GOAT gate)

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
      `constraint` appears only in `crates/katgpt-core/src/arg/policy.rs` (`PolicyConstraints` —
      governance, unrelated to MAPF). The constraint tree is NOT shipped
      under any name (Research 441 §3 documents the full vocabulary table).
- [x] **T1.3** Document the prior-art distinction: PI alone (Issues 140/143,
      reverted) vs PI + constraint tree (LaCAM, this plan). Issue 154 updated
      with cross-ref to Plan 453 (commit `a6cf51c2`). The key sentence:
      "PI is prior art; LaCAM is PI + constraint tree, and only the PI half
      was tried." Research 441 §5 has the full prior-art comparison table.

### Phase 2 — Implement bounded one-step LaCAM ✅ DONE

- [x] **T2.1** Created `crates/katgpt-core/src/multi_agent_path/lacam.rs` behind
      `feature = ["lacam_escalation"]`. Module structure:
      - `Constraint { who: Vec<usize>, where_cells: Vec<P> }` (depth = `who.len()`)
      - `ConstraintQueue` (FIFO `VecDeque<Constraint<P>>` — LaCAM BFS-style)
      - `EscalationBudget { max_nodes, time_budget_us }`
      - `lacam_escalation_step(...)` — the public entry point (§2.2)
      - `get_new_config(...)` — §2.3
      - `PibtState::func_pibt_recursive(...)` — §2.4 (state-bundled for manageable recursion signature)
- [x] **T2.2** Wired `lacam_escalation_step` into `pibt_step` behind the
      feature flag. When `lacam_escalation` is ON and stuck agents exist,
      delegates to `lacam_escalation_step`. The legacy shuffled-retry loop
      extracted to `legacy_shuffled_retry` (cfg-gated OFF), kept for back-compat
      and as the GOAT-gate baseline.
- [x] **T2.3** Reused the existing `Candidate` struct and `lexicographic_cmp`
      from `pibt.rs` — `Candidate` and its fields made `pub(super)`. The
      recursive PIBT uses the same `⟨guidance_mismatch, flow_mismatch, goal_dist,
      hindrance, ε⟩` tuple. No cost function duplication.
- [x] **T2.4** Reused the O(1) collision-detection structures from Issue 516
      T1g (`current_to_agent: HashMap<P, usize>`, `occupied_next: HashSet<P>`).
      Added `constrained_agents: HashSet<usize>` (agents fixed by the current
      constraint, skipped in the recursive PIBT loop).
- [x] **T2.5** `EscalationBudget` added with the defaults from §2.5
      (`max_nodes = 1000`, `time_budget_us = 5000`). Currently the budget is
      hardcoded in `pibt_step`'s delegation (`EscalationBudget::default()`);
      a future T2.5b can add it to `GuidanceConfig` or a new `LacamConfig` if
      consumers need to tune it. For now, the default is the GOAT-gate
      configuration.
- [x] **T2.6** Unit tests added in `tests.rs::lacam_escalation_tests` (5 tests):
      - `test_lacam_resolves_stuck_agent` — 4 agents converging on center,
        collisions ≤ 5/50 (verifies LaCAM resolves most stuck agents).
      - `test_lacam_budget_fallback` — 20 ticks, no panic (budget exhaustion
        fallback path).
      - `test_lacam_no_regression_on_open_map` — 4 agents on 10×10, 0 collisions
        (fast path = greedy PIBT).
      - `test_func_pibt_recursive_bounded` — 1-wide corridor deadlock,
        50 ticks, no hang (recursion terminates).
      - `test_escalation_budget_default` — default budget is non-zero.

### Phase 3 — Benchmark (G6c + G1 + latency) ✅ DONE

- [x] **T3.1** Port the riir-ai G6c scenario to a substrate-level benchmark.
      60 agents, 20×20 grid, 6-cell bottleneck gap, 200 ticks. G6c = 1.000
      (100% collision-free). **Critical fix discovered:** the `pibt_step`
      threshold (`MIN_STUCK_FOR_RETRY = 20`) gated the LaCAM constraint tree
      too aggressively — lowered to 1 when `lacam_escalation` is ON.
- [x] **T3.2** Re-ran G1 throughput benchmark with `lacam_escalation` ON.
      ht_chantry improved 0.01 → 0.28 (28×). 3/4 maps PASS (ht_chantry
      marginal at 0.28 < 0.30). See `.benchmarks/453_lacam_escalation_goat.md`.
- [x] **T3.3** Latency sweep: `max_nodes ∈ {100, 500, 1000, 5000}` on the G6c
      scenario (60 agents, 200 ticks). Median latency flat at ~40µs across all
      budgets — the constraint tree converges fast and rarely exhausts its
      budget at this scale.
- [x] **T3.4** Wrote `.benchmarks/453_lacam_escalation_goat.md` with full
      results: G6c table, G1 table, latency sweep, GOAT gate summary, and
      Phase 5 promotion decision (stay opt-in — defer to multi-step LaCAM).

### Phase 4 — GOAT gate ✅ DONE (2026-07-15)

All data collected in Phase 3; formally marked here. Full results in
`.benchmarks/453_lacam_escalation_goat.md`.

- [x] **T4.1** **G1 (throughput):** ⚠ **3/4 maps PASS** (ht_chantry marginal).
      empty 0.69 ✅, random 0.69 ✅, warehouse 0.40 ✅, ht_chantry **0.28 ❌**
      (target ≥ 0.30). ht_chantry improved 0.01 → 0.28 (28×) but one-step
      LaCAM cannot plan multi-step maze detours — exactly the §4.3 limitation.
- [x] **T4.2** **G-col (collision-freedom, NEW gate):** ✅ **PASS.** Vertex
      collision rate = **0.0%** (target ≤ 10%); G6c delta = **1.000** (target
      ≥ 0.50). The constraint tree resolves the bottleneck collision-free via
      recursive PIBT priority inheritance — no genuine physics constraint,
      just the missing constraint-tree half.
- [x] **T4.3** **G3 (no-regression):** ✅ **PASS.** `cargo test --features
      lacam_escalation`: 1616/1616. `cargo test --features multi_agent_path`:
      1611/1611. `cargo clippy` clean on both feature sets.
- [x] **T4.4** **G4 (latency):** ✅ **PASS.** Median 14-19ms at 800 agents
      (target ≤ 500ms, stretch ≤ 100ms met). Latency sweep at 60 agents is
      budget-insensitive (flat ~40µs across `max_nodes ∈ {100,500,1000,5000}`)
      — the constraint tree converges in the first few nodes.
- [x] **T4.5** **G-PI (no throughput collapse):** ✅ **PASS.** empty-48-48
      throughput ratio = **0.69** (target ≥ 0.60; Issue 140 collapsed to
      0.02). This is the gate that confirms the Plan 453 thesis: the
      constraint tree is the missing half that makes recursive PIBT safe.

### Phase 5 — Promotion decision ✅ DONE (2026-07-15)

**T5.3 applies:** G1 marginally FAILS on ht_chantry (0.28 < 0.30). The
other four gates (G-col, G-PI, G3, G4) all PASS, and the collision-freedom
improvement stands on its own as a genuine win. Per the AGENTS.md promotion
rule, modelless gain requires all gating criteria to pass — ht_chantry is
2pp under the G1 threshold, so the feature cannot be promoted to default-on
in this cycle.

- [-] **T5.1** (NOT REACHED) — G1 did not fully pass, so promotion is blocked.
- [-] **T5.2** (NOT REACHED) — G-col passed.
- [x] **T5.3** **G1 FAILS marginally (ht_chantry 0.28 < 0.30):** LaCAM
      one-step can't resolve multi-step corridor deadlocks. **Decision:
      STAY OPT-IN.** The constraint tree is a genuine improvement over the
      legacy shuffled retry:
      - Collision-freedom: 37.5% → 100% (G6c scenario; G-col = 0.0%)
      - ht_chantry throughput: 0.01 → 0.28 (28×)
      - No throughput collapse (G-PI PASS — the Issue 140/143 failure mode
        is fixed by construction)
      Defer ht_chantry G1 parity (0.28 → ≥ 0.30) to a future **multi-step
      LaCAM plan** building on Plan 453's one-step constraint tree
      foundation. The collision-freedom improvement (G-col) stands on its
      own; Issue 154 is closed as fixed (the remaining throughput gap is a
      multi-step planning problem, not a collision-freedom problem).
- [-] **T5.4** (NOT REACHED) — G-PI passed (the Plan 453 thesis confirmed).

**Re-opening riir-ai Proposal 023** (G6c gate, originally rejected at
G6c = 0.360) is **NOT triggered** by T5.3: Proposal 023 gates on the
consumer-side `crowd_motion_lllg` feature promotion, which depends on
`lacam_escalation` being default-on in the substrate. Since the substrate
stays opt-in, the consumer feature stays opt-in. Proposal 023 can be
re-opened once the future multi-step LaCAM plan closes the ht_chantry G1
gap and `lacam_escalation` promotes to default-on.

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

**Phase 3 result (2026-07-15):** G6c = 1.000 (PASS), G-col = 0.0% (PASS),
G-PI = 0.69 (PASS), G1 = 3/4 maps PASS (ht_chantry marginal at 0.28 < 0.30).
G-col and G-PI pass; G1 marginally fails on ht_chantry. Decision: stay opt-in
— the constraint tree fixes collisions and improves ht_chantry 28×, but
full G1 parity needs multi-step LaCAM. See `.benchmarks/453_lacam_escalation_goat.md`.
