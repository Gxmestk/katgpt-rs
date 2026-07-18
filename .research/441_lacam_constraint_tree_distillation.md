# Research 441: LaCAM Constraint-Tree Distillation — the Missing Half of Issues 140/143

> **Source:** Okumura 2023, "LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding", AAAI 2023.
> [Project page](https://kei18.github.io/lacam/) · [Reference code](https://github.com/Kei18/lacam/blob/master/lacam/src/planner.cpp) (`Kei18/lacam`, C++).
> **Date:** 2026-07-15
> **Status:** Active — grounds [Plan 453](../.plans/453_bounded_one_step_lacam_escalation.md) (bounded one-step LaCAM escalation).
> **Related Research:** [424](424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md) (LLLG — the consumer paper), 296 (Stokes Calculus DEC Vocabulary Crosswalk — the vocabulary-translation lesson).
> **Prior art (reverted):** Issues 140 (recursive PIBT, -92% throughput), 143 (shuffled retry, same collapse), 154 (PI prior-art finding).
> **Classification:** Public (katgpt-rs/MIT). The algorithm is published prior art; the one-step adaptation is the novel scope.

## TL;DR

LaCAM (Lazy Constraints Addition Search for MAPF) is a **two-level search**.
The critical finding from reading the reference implementation
(`Kei18/lacam/src/planner.cpp`): **LaCAM uses recursive PIBT with priority
inheritance** (`funcPIBT` calls itself), but it does NOT collapse throughput
because the **constraint tree** bounds the recursion and provides systematic
backtracking. Issues 140/143 implemented recursive PIBT **without** the
constraint tree — that is why they collapsed throughput. **LaCAM = recursive
PIBT + constraint tree**, and only the recursive PIBT half was tried. Plan
453 ships the constraint tree — the missing half.

**Scope for Plan 453:** one-step LaCAM (find a collision-free joint action
for the current tick), NOT the full multi-step config search. This avoids
the LaCAM* anytime-refinement negative result (Research 424 §1.5) which
applies to multi-step plan optimization, not one-step collision-freedom.

---

## 1. The LaCAM Algorithm (from the reference implementation)

### 1.1 Two-level search structure

```
High-level: DFS over configurations
  OPEN: stack of configuration nodes
  CLOSED: set of visited configurations
  For each node S (LIFO from OPEN):
    Low-level: constraint tree (FIFO queue of Constraint objects)
      Pop constraint M from S->search_tree
      If M->depth < N (number of agents):
        Expand: for the next agent in priority order, push one child
        constraint per candidate cell (neighbor ∪ stay)
      get_new_config(S, M):
        Apply M's constraints (force specific agents to specific cells)
        Run funcPIBT for unconstrained agents
        Return new config if collision-free, else reject
    If get_new_config succeeded → push new config node to OPEN
    If S->search_tree empty → pop S (backtrack)
```

The **constraint tree** is per-configuration-node. Each configuration node
owns its own constraint queue. When a configuration's constraint queue is
exhausted, the high-level search backtracks (pops the configuration).

### 1.2 The Constraint data structure

From `planner.cpp`:

```cpp
Constraint::Constraint(Constraint* parent, int i, Vertex* v)
    : who(parent->who), where(parent->where), depth(parent->depth + 1)
{
  who.push_back(i);    // which agent is constrained
  where.push_back(v);  // which cell it's forced to
}
```

A constraint is a **chain**: `(agent_0 → cell_0, agent_1 → cell_1, ...)` up
to `depth`. The root constraint is empty (`depth = 0`). Each child extends
the parent by one `(agent, cell)` assignment. The agent at each depth is
determined by the configuration's priority order (`S->order[depth]`).

### 1.3 `get_new_config` — applying constraints + PIBT

```cpp
bool Planner::get_new_config(Node* S, Constraint* M) {
  // 1. Clear occupied_now/occupied_next caches, set occupied_now from S->C
  // 2. Apply constraints from M:
  for (k = 0; k < M->depth; ++k) {
    i = M->who[k];        // constrained agent
    l = M->where[k]->id;  // forced cell
    if (occupied_next[l] != nullptr) return false;  // vertex collision
    // swap collision check
    if (occupied_next[l_pre] != nullptr && occupied_now[l] != nullptr &&
        occupied_next[l_pre]->id == occupied_now[l]->id) return false;
    A[i]->v_next = M->where[k];
    occupied_next[l] = A[i];
  }
  // 3. Run funcPIBT for unconstrained agents (in priority order)
  for (k : S->order) {
    if (A[k]->v_next == nullptr && !funcPIBT(A[k])) return false;
  }
  return true;
}
```

Key: constraints are applied FIRST (fixed assignments), then PIBT resolves
the remaining agents. If PIBT fails for any agent, the whole constraint is
rejected (`return false`), and the next constraint is tried.

### 1.4 `funcPIBT` — recursive PIBT with priority inheritance

This is the piece Issues 140/143 implemented standalone. In LaCAM it is
safe because the constraint tree backtracks when it fails.

```cpp
bool Planner::funcPIBT(Agent* ai) {
  i = ai->id;
  // Get candidates: neighbors ∪ {current}, sorted by dist-to-goal + ε
  for (k = 0; k < K+1; ++k) {
    u = C_next[i][k];
    if (occupied_next[u->id] != nullptr) continue;      // vertex collision
    ak = occupied_now[u->id];
    if (ak != nullptr && ak->v_next == ai->v_now) continue;  // swap collision
    occupied_next[u->id] = ai;
    ai->v_next = u;
    if (ak == nullptr || u == ai->v_now) return true;   // empty cell or stay
    // Priority inheritance: push the occupant
    if (ak->v_next == nullptr && !funcPIBT(ak)) continue;  // RECURSIVE CALL
    return true;
  }
  // Failed — stay in place
  occupied_next[ai->v_now->id] = ai;
  ai->v_next = ai->v_now;
  return false;
}
```

**The recursion** (`funcPIBT(ak)`) is the priority inheritance: when agent
`ai` wants cell `u` occupied by `ak`, `ai` pushes `ak` to find another cell.
If `ak` succeeds, `ai` gets `u`. If `ak` fails, `ai` tries its next
candidate.

**Why this collapses throughput without the constraint tree (Issue 140/143):**
a single push can cascade — `ai` pushes `ak`, `ak` pushes `al`, `al` pushes
`am`, ... Each agent in the chain is forced away from its goal. On dense
maps, this cascade stalls the entire system (empty-48-48 throughput: 18.6 →
0.47, -92%).

**Why it's safe WITH the constraint tree:** the constraint tree provides
backtracking at the configuration level. When `funcPIBT` fails (returns
false), `get_new_config` returns false, and the constraint tree tries a
different constraint (a different assignment for the root agent). The
cascade is bounded by the constraint tree's exploration — it doesn't spiral.

---

## 2. The One-Step Adaptation (Plan 453 scope)

### 2.1 What we keep vs drop

| LaCAM component | Plan 453 | Rationale |
|---|---|---|
| **Constraint tree** (low-level search) | **KEEP** | The missing half — systematic backtracking over assignments |
| **Recursive PIBT** (`funcPIBT`) | **KEEP** | The configuration generator; safe within the constraint tree |
| **High-level config search** (OPEN/CLOSED) | **DROP** | We only need one next config, not a sequence of configs |
| **Deadline-based termination** | **KEEP** (as `EscalationBudget`) | Real-time guarantee — bounded nodes + time |
| **Goal-condition check** (`is_same_config`) | **DROP** | One-step scope — no goal to reach within the search |

### 2.2 The one-step algorithm

```
fn lacam_escalation_step(config, ...):
    # Fast path: greedy PIBT (current code)
    (moves, stuck) = greedy_pibt_pass(...)
    if stuck.is_empty(): return Ok(moves)

    # Constraint-tree search (bounded)
    queue = ConstraintQueue::with_capacity(budget.max_nodes)
    queue.push(Constraint::empty())
    order = compute_priority_order(n, priorities)
    best = (moves, stuck.len())  # fallback = greedy result

    while let Some(constraint) = queue.pop():
        if nodes_explored > budget.max_nodes: break

        # Expand: push children for the next agent in priority order
        if constraint.depth < n:
            i = order[constraint.depth]
            for u in shuffled(neighbors(config.pos(i)) ∪ {config.pos(i)}):
                queue.push(constraint.child(i, u))

        # Try to build a collision-free config
        match get_new_config(config, &constraint, ...):
            Ok(moves) => return Ok(moves)  # collision-free!
            Err(_) => continue  # try next constraint

    # Budget exhausted — return greedy fallback
    Ok(best.moves)
```

### 2.3 Why this avoids the LaCAM* negative result (Research 424 §1.5)

Research 424 §1.5: "applying LaCAM* anytime refinement to LLLG's windowed
plan **degrades** lifelong throughput... the LaCAM* f-value (`g + h` on the
`w_Π`-step window) is misaligned with the lifelong throughput metric."

**This plan does NOT implement LaCAM* anytime refinement.** The scope is:
- **One-step** (find the next joint action), not multi-step plan optimization
- **Collision-freedom** objective, not windowed-plan-quality objective
- **Bounded** (budget caps the search), not anytime (no iterative improvement)

The negative result applies to optimizing the windowed plan (`g + h` over
`w_Π` steps), which is misaligned with lifelong throughput. Our objective
(find a collision-free single step) is different — it doesn't try to optimize
plan quality, only to avoid collisions.

**Residual risk:** forcing collision-freedom may still reduce throughput on
genuinely congested maps (some agents must wait). This is addressed by the
G-PI gate (Plan 453 T4.5) and the bounded-budget fallback (if the search
can't find a collision-free config quickly, fall back to greedy PIBT with
collisions).

---

## 3. Vocabulary Translation (R296 lesson applied)

Per the R296 lesson (paper vocabulary may differ from code vocabulary), I
grepped the codebase for both paper and code vocabulary:

| Paper term | Codebase grep | Found? |
|---|---|---|
| `constraint` | `crates/katgpt-core/src/` | YES — but only in `crates/katgpt-core/src/arg/policy.rs` (`PolicyConstraints` — governance, unrelated to MAPF) |
| `backtrack` | `crates/katgpt-core/src/` | NO |
| `search_tree` | `crates/katgpt-core/src/` | NO |
| `constraint_tree` | `crates/katgpt-core/src/` | NO |
| `priority inheritance` / `funcPIBT` | `crates/katgpt-core/src/multi_agent_path/pibt.rs` | NO (the module docs *discuss* PI but the code uses greedy PIBT, not recursive) |

**Conclusion:** the constraint tree is NOT shipped under any name. The
vocabulary-translation gate passes — Plan 453 is implementing genuinely
new machinery.

---

## 4. Data Structures for Plan 453

### 4.1 `Constraint<P>`

```rust
struct Constraint<P: Position + Clone> {
    /// Chain of (agent_index, forced_cell) pairs.
    who: Vec<usize>,
    where_cells: Vec<P>,
    /// Depth = who.len(). Root constraint has depth 0.
    depth: usize,  // redundant with who.len() but cached for fast comparison
}

impl<P: Position + Clone> Constraint<P> {
    fn empty() -> Self { Self { who: vec![], where_cells: vec![], depth: 0 } }

    fn child(&self, agent: usize, cell: P) -> Self {
        let mut c = Constraint {
            who: self.who.clone(),
            where_cells: self.where_cells.clone(),
            depth: self.depth + 1,
        };
        c.who.push(agent);
        c.where_cells.push(cell);
        c
    }
}
```

**Allocation note:** `child()` clones the vectors. For the hot path (depth
0-3, the vast majority of useful constraints), this is 0-3 clones of small
vecs. The `ConstraintQueue` pre-allocates `Vec::with_capacity(max_nodes)`.
Per the global rule: no allocation inside hot loops — the constraint tree is
NOT a hot loop (only runs when stuck agents exist).

### 4.2 `ConstraintQueue<P>`

```rust
struct ConstraintQueue<P: Position + Clone> {
    queue: VecDeque<Constraint<P>>,
    max_nodes: usize,
}
```

LaCAM uses FIFO (BFS-style) for the constraint queue. This explores
shallow constraints first (fewer forced assignments), which is more likely
to succeed (less constraining).

### 4.3 `EscalationBudget`

```rust
struct EscalationBudget {
    max_nodes: usize,      // default 1000
    time_budget_us: u64,   // default 5000 (5ms)
}
```

Checked every 64 nodes (branch-free fast path). When exhausted, fall back to
greedy PIBT result.

---

## 5. The Prior-Art Distinction (the key argument)

### 5.1 Why "PI is prior art" and "LaCAM is not prior art" can both be true

| Implementation | Recursive PIBT? | Constraint tree? | Throughput | Status |
|---|---|---|---|---|
| **Issue 140** | YES | NO | -92% (collapsed) | REVERTED |
| **Issue 143** | YES | NO (shuffled retry ≠ constraint tree) | -92% (collapsed) | REVERTED |
| **LaCAM (Okumura 2023)** | YES | **YES** | Scalable (paper-reported) | Published |
| **Plan 453** | YES | **YES** | TBD (G-PI gate) | OPEN |

The constraint tree is the differentiator. Without it, recursive PIBT
cascades and collapses throughput. With it, the cascade is bounded by
systematic backtracking.

### 5.2 The honest counter-argument

A skeptic could argue: "the constraint tree just delays the collapse — on a
genuinely congested map, no assignment is collision-free without massive
waiting, so the constraint tree will exhaust its budget and fall back to
greedy PIBT (collisions)."

This is possible. Plan 453's Phase 5 has explicit branches:
- **T5.2:** if G-col fails (collision rate still high) — the substrate
  genuinely can't guarantee collision-freedom. Stay opt-in.
- **T5.4:** if G-PI fails (throughput collapse recurs) — the constraint tree
  didn't help. Stay opt-in.

The bounded budget ensures the worst case is "no improvement" (fall back to
greedy), not "throughput collapse" (worse than today).

---

## 6. Open Questions for Plan 453 Phase 1

- [ ] **Q1:** Does the one-step constraint tree resolve ht_chantry maze
      deadlocks (G1), or does it need multi-step coordination? Answer: T3.2
      benchmark.
- [ ] **Q2:** Does the one-step constraint tree improve G6c (bottleneck),
      or is the bottleneck genuinely too narrow? Answer: T3.1 benchmark.
- [ ] **Q3:** What is the right `max_nodes` default? Answer: T3.3 latency
      sweep.
- [ ] **Q4:** Does the recursive PIBT need a depth limit (beyond the
      constraint tree's implicit bound)? The reference implementation doesn't
      have one — the constraint tree bounds it. But our budget may need an
      explicit recursion depth cap for safety. Answer: T2.6 stress test.

---

## 7. References

- **LaCAM paper:** Okumura 2023, "LaCAM: Search-Based Algorithm for Quick
  Multi-Agent Pathfinding", AAAI 2023.
  [Project page](https://kei18.github.io/lacam/),
  [code](https://github.com/Kei18/lacam).
- **LLLG paper:** Arita & Okumura 2026, arXiv:2605.16855. → Research 424.
- **PIBT paper:** Okumura et al. 2022, AIJ.
- **Reference implementation:** `Kei18/lacam/src/planner.cpp` — the
  `Planner::solve()`, `get_new_config()`, and `funcPIBT()` functions are
  the primary sources for Plan 453's algorithm shape.
- **Prior art:** Issues 140, 143, 144, 154 (katgpt-rs).
- **Vocabulary lesson:** Research 296 (Stokes Calculus DEC Vocabulary
  Crosswalk) — paper vocabulary vs code vocabulary.

---

## TL;DR

LaCAM = recursive PIBT + constraint tree. Issues 140/143 tried recursive
PIBT without the constraint tree → throughput collapsed (-92%). Plan 453
ships the constraint tree — the missing half. Scope: one-step (find a
collision-free joint action for the current tick), bounded by a node/time
budget, with greedy-PIBT fallback. The LaCAM* anytime-refinement negative
result (Research 424 §1.5) does NOT apply — that's about multi-step plan
optimization, not one-step collision-freedom. Vocabulary grep confirms the
constraint tree is not shipped under any name. Open questions (does it
resolve ht_chantry? does it help G6c?) are answered by Plan 453 Phase 3
benchmarks.
