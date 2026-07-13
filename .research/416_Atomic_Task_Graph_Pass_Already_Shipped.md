# Research 416: Atomic Task Graph (ATG) — Agentic Planning/Execution Framework (PASS — already shipped)

> **Source:** [Atomic Task Graph: A Unified Framework for Agentic Planning and Execution](https://arxiv.org/abs/2607.01942) — Zhang, Chen, Huang, Cui, Ji, Wang (SCUT + Tsinghua Shenzhen), arXiv:2607.01942v1, 2 Jul 2026
> **Date:** 2026-07-13
> **Status:** Done — closed.
> **Classification:** Public
> **Related Research:** 060 (LEAP Blueprint DAG — game-AI decomposition), 170 (LEAP — same family), 289 (RecursiveMAS — canonical PASS pattern), 169 (Agent-Native Memory — canonical PASS pattern), 300 (CUCG — closed-unit compaction), 407 (GDN tree verify — masked triangular solve)
> **Related Plans:** 190 (AND-OR DDTree — shipped), 223 (Lean4Agent: TrajectoryDoctor + HoarePruner + WorkflowLattice + LLMExecGuard — shipped), 333 (CUCG — default-on), 424 (GDN tree verify — shipped), 014+ (PagedKVCache fork-based rollback — shipped)
> **Verdict: PASS.** ATG is an agent-control-framework paper. All three of its mechanisms ship in our quintet under different vocabulary — most at higher fidelity (game-AI substrate, latent-state reframing). The paper's value is its **LLM-orchestration process** (LLM-as-decomposer, LLM-as-thought-experiment-validator, LLM-as-subgraph-repairer), which is LLM-dependent at every step and is the canonical R133/R169/R289 NO-GAIN failure class. No file, no plan, no guide created beyond this classification note.

---

## TL;DR

ATG frames LLM-agent task solving as a DAG of atomic tool-use units with explicit input-output dependencies, and ships three mechanisms: (1) interface-preserving recursive graph compilation that refines a coarse task into atomic nodes while preserving each parent's I/O interface, (2) dependency-aware parallel execution with a "thought experiment" pre-execution validator, (3) minimal necessary subgraph repair that traces a failed node to its lowest-common-ancestor in refinement history and repairs only that subgraph while freezing validated regions. Evaluated on ALFWorld / WebShop / ScienceWorld with 7B–8B open-source backbones, ATG beats GPT-4-ReAct on ALFWorld + WebShop and reduces hallucinatory actions from 42.86% (ReAct) to 12.14% on ALFWorld.

**Distilled for katgpt-rs (modelless, inference-time):** nothing not already shipped. ATG's three mechanisms map directly onto shipped primitives in our quintet — see the architecture-class mapping below. The paper explicitly states "ATG operates purely at inference time: it does not require task-specific fine-tuning" (training-free), but every step (decompose, validate, repair) is an LLM call — placing ATG firmly in the R169 / R289 / R133 LLM-orchestration failure class, not the R368 decision-structure class.

---

## 1. Paper Core Findings (verified by full read)

### 1.1 The three mechanisms (§4)

| Mechanism | What it does | LLM dependency |
|---|---|---|
| **Interface-Preserving Recursive Graph Compilation** (§4.1) | Recursively refines a coarse task into a DAG of atomic tool-use units. Each refinement step preserves the parent node's external input-output interface, so the surrounding graph stays structurally stable. Records the refinement history (sequence of coarse-to-fine graphs). | LLM does the decomposition at every step. No modelless substrate can decompose arbitrary user tasks ("generate a sales analysis report") into subtasks. |
| **Dependency-Aware Execution** (§4.2) | Executes atomic nodes in topological order. Independent branches run in parallel. Includes a "thought experiment" pre-execution validator that checks consistency, missing-step detection, tool appropriateness, dependency validity, and constraint satisfaction before real execution. | Topological scheduling is modelless; the thought-experiment validator is LLM-driven (semantic checks). |
| **Minimal Necessary Subgraph Repair** (§4.3) | On failure (from thought experiment or runtime), localizes to a failed atomic node or small node set, traces them back to their lowest-common-ancestor `a_f` in refinement history, constructs a minimal repair subgraph covering the failed node + affected upstream/downstream, freezes the rest, and repairs only that subgraph. | LLM generates the repaired subgraph. |

### 1.2 The empirical claim (§5)

- 7B–8B backbones with ATG beat GPT-4-ReAct on ALFWorld + WebShop; ATG-Llama-3-8B hits 63.65 / 68.36 / 56.79 on ALFWorld / WebShop / ScienceWorld.
- Hallucinatory action rate: 42.86% (ReAct) → 12.14% (ATG) on ALFWorld — a 71.7% relative reduction.
- Average steps reduced: 31.42 → 18.36 on ALFWorld (parallel branches counted as one step).
- Ablations (§5.2): removing the thought experiment costs ~4 points; removing subgraph repair costs ~7 points.

### 1.3 What the paper does NOT claim (honest caveats)

- ATG "depends on the backbone LLM's decomposition ability" (§7 Limitations) — the framework amplifies but cannot substitute for backbone capability.
- Failure localization "can be difficult under noisy observations or long-range dependencies" (§7).
- "ATG also introduces extra overhead for simple tasks" (§7).
- All three benchmarks are text-based; multimodal and real-world settings are unvalidated.

---

## 2. Architecture-Class Mapping — why this is PASS, not GOAT

Every ATG mechanism has a shipped equivalent in our quintet. The mapping below follows the canonical R169/R289 PASS pattern.

| ATG mechanism | Shipped equivalent (modelless) | Where | Higher fidelity? |
|---|---|---|---|
| **Recursive graph compilation with interface preservation** | **AND-OR DDTree** (`AndOrNode<G,S>` enum, `AndOrBuilder` relevance regions, `BlueprintPass` argmax plan, `DecompositionReviewer` novelty-based dead-end detection, `ProofGoalCache` BLAKE3 memo) | `katgpt-rs/crates/katgpt-core/src/and_or/types.rs`, `katgpt-rs/src/speculative/{and_or_builder,blueprint,decomp_reviewer}.rs` (Plan 190, Bench 040) | ✅ — our version adds a BLAKE3-addressed proof-goal cache for cross-call reuse, which ATG's refinement history does not provide. LEAP Blueprint DAG (R060) is the game-AI decomposition framing. |
| **Thought experiment / pre-execution validator** | **LLMExecGuard** (entropy-gated 3-tier routing) + **ScreeningPruner** (relevance gate) + **ConstraintPruner** (validity gate) + **DecompositionReviewer** (novelty-based dead-end detection) + **HoarePruner** (predicate propagation) | `katgpt-pruners` crate (Plans 223, 223 Phase 2, 190, 223 Phase 2) | ✅ — our version is a layered pruner cascade with entropy gating (LLMExecGuard), not a single LLM-driven pre-execution check. |
| **Minimal necessary subgraph repair (failure localization to node + repair)** | **`TrajectoryDoctor` trait** with `localize_failure(tokens, depth_limit) -> Option<FailureSite>` + `FailureEpisodeStore` + `HoareTrajectoryDoctor` (predicate-based) + `BracketTrajectoryDoctor` (bracket-depth) + **`ReestimationScheduler`** (coherence-driven re-estimation when `coherence < tau_reest`) | `katgpt-rs/crates/katgpt-pruners/src/trajectory_doctor.rs` (Plan 223 Phase 3, shipped) + `riir-ai/crates/riir-engine/src/latent_functor/reestimation/` | ✅ — `FailureSite { depth, token_idx, violated_predicate, alternatives }` IS the ATG "lowest-common-ancestor in refinement history" (depth in decomposition tree). The `ReestimationScheduler` extends the pattern to per-NPC latent state — a capability ATG lacks entirely. |
| **Freeze validated regions, repair only affected subgraph** | **`PagedKVCache::fork`** + `rollback(pos)` (copy-on-write, prefix-shared) + **`MerkleFrozenEnvelope`** + **`GdnTreeVerifier`** masked triangular solve (commit-on-accept, no rollback of recurrent state) | `katgpt-rs` (Plan 014+) + `riir-neuron-db/src/freeze.rs` + `katgpt-core/gdn_tree_verify` (Plan 424, R407) | ✅ — our version is bit-committed (BLAKE3) and works for delta-rule recurrences (GDN tree verify), not just attention KV-cache. |
| **DAG-of-tool-calls with explicit I/O dependencies** | **AND-OR DDTree node type** (goal `G`, state `S`, child set) + **DEC cochain flow edges** (`exterior_derivative` d, `DecFlowField`) | `katgpt-core/src/and_or/` + `katgpt-core/src/dec/` (Plans 190, 251) | ✅ — our DAG substrate is generic AND-OR + typed cochain edges (not just tool-use nodes). |
| **Parallel execution of independent branches** | rayon parallel iteration (heavily used across the codebase) + `latent_functor/zone_scheduling.rs` | many | ✅ — commodity. |
| **Graph evolution history (refinement sequence)** | **Trajectory compaction records** + **snapshot/freeze versioning** (`MerkleFrozenEnvelope` versioned, `Uuid::now_v7`) | `katgpt-core/closed_unit_compaction` (Plan 333) + `riir-neuron-db/src/freeze.rs` | ≈ — our version stores versioned snapshots but does not explicitly thread them as a "refinement history" data structure for backward tracing. The capability is implicit; ATG makes it explicit. |
| **Closed-unit / sub-goal boundary (related)** | **`ClosedUnitCompactionGate` (CUCG)** — rubric-gated trajectory compaction primitive (default-on, Plan 333, R300) + per-NPC sub-goal rubric (riir-ai R155) | `katgpt-core/closed_unit_compaction` + `riir-engine/cce_runtime/` | ✅ — strictly more capable (rubric is sigmoid-gated on latent features, not LLM-judged from verbatim quotes). |

### 2.1 Failure mode common to all ATG mechanisms

Every ATG mechanism is LLM-driven at runtime: decomposition is an LLM call, the thought experiment is an LLM call, subgraph repair is an LLM call. At a 20Hz NPC tick this is 20 LLM calls/sec/NPC — orders of magnitude over the modelless-first mandate. This is the identical NO-GAIN pattern documented for FluxMem (R133), AgentMemBench (R169), and RecursiveMAS (R289).

The structural substrate (DAG execution, topological scheduling, snapshot/rollback, localized repair, freeze/thaw, BLAKE3 memo) is modelless and ships. The LLM-dependent orchestration layer does not.

---

## 3. Distillation

### 3.1 What's LLM-dependent → not distillable here

- **Recursive decomposition of arbitrary user tasks** (LLM generates the subtask list at each refinement step). No modelless substrate can decompose "produce accurate analysis and insightful visualizations" into a DAG of tool calls.
- **Semantic thought experiment** (LLM checks for missing steps, invalid dependencies, tool-appropriateness, interface mismatches). These are semantic / NL checks; no modelless substrate computes them.
- **Subgraph repair generation** (LLM replaces incorrect tools, inserts missing nodes, adjusts dependencies). The repair *content* is LLM-generated.

### 3.2 What's modelless but already shipped

| ATG modelless substrate | Shipped cousin | Plan / Research |
|---|---|---|
| DAG-of-tool-calls scheduling | AND-OR DDTree (`AndOrNode<G,S>`), DEC cochain edges | P190, P251 |
| Recursive decomposition with memoization | AND-OR DDTree + BlueprintPass + ProofGoalCache (BLAKE3) | P190, Bench 040 |
| Pre-execution validation cascade | LLMExecGuard + ScreeningPruner + ConstraintPruner + DecompositionReviewer + HoarePruner | P223 Phases 1–2, P190 |
| Failure localization (depth in decomposition tree) | `TrajectoryDoctor::localize_failure` + `FailureSite { depth, token_idx, violated_predicate }` + `FailureEpisodeStore` | P223 Phase 3 |
| Coherence-triggered localized re-estimation | `ReestimationScheduler` (coherence < `tau_reest`), `CceReestimationTrigger`, `cwm_runtime::trigger` | P303, R303 |
| Freeze validated regions + rollback affected | `PagedKVCache::fork`/`rollback`, `MerkleFrozenEnvelope`, `GdnTreeVerifier` masked solve | P014+, P424/R407 |
| Closed-unit boundary detection | `ClosedUnitCompactionGate` (default-on, Super-GOAT G7) | P333/R300, riir-ai R155 |

### 3.3 Fusion — none novel

The closest potential fusion is "interface preservation as an explicit invariant during refinement" — a contract-based design pattern where each refinement step must preserve the parent node's external I/O interface. This is **not shipped** as a named primitive (grep for `interface_preserv|refinement_history|recursive_compil|atomic_task_graph` returns zero hits across all five repos). However:

1. The capability is **implicit** in the AND-OR DDTree (`AndOrNode` carries goal + state; children must collectively satisfy the parent goal — that IS interface preservation).
2. As a standalone primitive, "interface-preserving graph rewrite" is a software-engineering pattern (refinement calculus), not a modelless inference primitive. It belongs in `katgpt-rs` only if a concrete inference consumer needs it; no such consumer exists today.
3. The "graph evolution history" as a versioned data structure for backward tracing is the same capability `MerkleFrozenEnvelope` + `snapshot` provide for forward integrity; the difference is read direction, not data structure.

There is no fusion of ATG × existing-primitive that produces a capability none of them has alone. The prior-art surface is dense.

### 3.4 Latent vs raw boundary (mandatory check)

Not applicable — ATG is entirely at the orchestration layer; no new boundary-crossing behavior. Our shipped equivalents already enforce the boundary discipline (raw scalars cross sync, latent state stays local).

### 3.5 Latent-space reframing check (mandatory per skill — primary framing)

- **HLA framing:** ATG = "decompose task → execute in topological order → repair on failure." Per-NPC equivalent: NPCs already decompose goals via latent_functor + cgsp + AND-OR DDTree, execute per-tick with coherence-gated re-estimation, and repair via `ReestimationScheduler`. ATG adds nothing to this loop.
- **Latent functor framing:** each "atomic tool-use unit" = one functor application. The interface-preservation property is exactly the `AndOrNode` parent-goal satisfaction contract. Already shipped.
- **DEC Stokes framing:** "I/O dependency edge" = `exterior_derivative` d on a typed cochain. Already shipped as the DEC substrate.
- **Neuron-shard framing:** "freeze validated regions" = `MerkleFrozenEnvelope` atomic Arc swap. Already shipped.
- **Adapter routing framing:** not applicable (ATG has no adapter concept).

No Super-GOAT framing survives the latent reframing check. The defaulting-to-adapter-routing symptom (per skill §1 step 3) does not even arise — ATG has no adapter angle to fall back to.

---

## 4. Verdict

**Tier: PASS.** LLM-orchestration framework paper; every mechanism ships under different vocabulary; the paper's value is its LLM-dependent process (R133/R169/R289 class).

### Novelty gate (§1.5) — honest scoring

| Q | Criterion | Honest answer |
|---|---|---|
| **Q1** No prior art? | **FAIL.** Every mechanism ships in our quintet. `TrajectoryDoctor::localize_failure` + `FailureSite { depth }` is exactly ATG's "lowest-common-ancestor in refinement history" failure localization. `ReestimationScheduler` extends the pattern to latent state. AND-OR DDTree (Plan 190) is the recursive decomposition substrate. CUCG (Plan 333) is the closed-unit boundary detector. GdnTreeVerifier (Plan 424) is the freeze-validated-regions + minimal-necessary-verify primitive. |
| **Q2** New class of behavior? | **FAIL.** "DAG-based agent control" = AND-OR decomposition (shipped) + topological scheduling (commodity) + localized repair (shipped as TrajectoryDoctor) + freeze/thaw (shipped as PagedKVCache + MerkleFrozenEnvelope). No new capability class. |
| **Q3** Product selling point? | **FAIL for new selling point.** "Decompose → execute in parallel → repair locally" IS the latent_functor + TrajectoryDoctor selling point — and our version is modelless + per-NPC + bit-committed, none of which ATG is. |
| **Q4** Force multiplier? | **NO** — only as a redescription of capabilities we already compose. |

### MOAT gate (§1.6) per domain

- **`katgpt-rs` (public engine):** ATG contributes no new modelless inference primitive. The shipped primitives (AND-OR DDTree, TrajectoryDoctor, CUCG, GdnTreeVerifier, PagedKVCache) already cover the modelless substrate.
- **`riir-ai` (private runtime):** ATG contributes no pillar-level amplification. The latent_functor + TrajectoryDoctor + ReestimationScheduler composition is already a strictly stronger runtime than ATG's LLM-orchestration loop.
- **`riir-chain` / `riir-neuron-db`:** out of scope — ATG has no chain or shard angle.

A great primitive in the wrong repo dilutes the moat; a redundant primitive in any repo adds noise. This is the latter.

### Parity claim check (§3.6 defend-wrong PoC)

Not applicable. This note makes NO quality-parity claim ("our TrajectoryDoctor matches ATG's subgraph repair on ALFWorld"). The verdict is architectural-only: the mechanism class ships. Whether TrajectoryDoctor matches ATG's 12.14% hallucinatory action rate on a controlled toy benchmark is an empirical question that would require a PoC — but since the verdict is PASS (not "our primitive beats ATG"), no PoC is required. A future implementer who wants to upgrade this note to a quality-parity claim would need to run head-to-head on a toy agent benchmark in `riir-poc`.

---

## 5. Why not an `.issues/` entry (declined)

ATG does not identify a measurable defect in any shipped primitive (no failing test, no perf regression, no quality gap). It is a control-framework paper whose three mechanisms all ship under different names. Filing an issue would generate noise without a concrete acceptance criterion.

If a future implementer wants the explicit "interface-preservation contract" as a named primitive (currently implicit in `AndOrNode` parent-goal satisfaction), that is a refactor task, not an ATG-derived task — and it should be filed only when a concrete consumer needs the named invariant, not preemptively.

---

## 6. Cross-references

- **Canonical PASS precedent (LLM-orchestration class):** `katgpt-rs/.research/133_FluxMem_Connectivity_Evolving_Memory.md`, `riir-ai/.research/169_Agent_Native_Memory_Benchmark_PASS.md`, `katgpt-rs/.research/289_RecursiveMAS_Pass_Already_Shipped.md`. Consult before re-evaluating any agent-control-framework paper whose mechanism lives at the LLM-orchestration layer.
- **Shipped decomposition substrate:** `katgpt-rs/.plans/190_*.md` (AND-OR DDTree), `katgpt-rs/.benchmarks/040_and_or_dtree_goat.md`, `riir-ai/.research/060_LEAP_Blueprint_DAG_Game_Strategy.md`.
- **Shipped failure-localization:** `katgpt-rs/.plans/223_lean4agent_formal_verification_fusion.md` (Phase 3 TrajectoryDoctor), `katgpt-rs/crates/katgpt-pruners/src/trajectory_doctor.rs`.
- **Shipped coherence-driven re-estimation:** `riir-ai/crates/riir-engine/src/latent_functor/reestimation/`, `riir-ai/crates/riir-engine/src/cce_runtime/reestimation_trigger.rs`.
- **Shipped freeze-validated-regions + minimal-verify:** `katgpt-rs/.plans/424_gdn_tree_verification_primitive.md`, `katgpt-rs/.research/407_Trees_from_Marginals_GDN_Tree_Verify.md`.
- **Shipped closed-unit boundary:** `katgpt-rs/.research/300_Closed_Unit_Compaction_Gate_Rubric_Gated.md`, `riir-ai/.research/155_Per_NPC_Sub_Goal_Compaction_Guide.md`.
- **Re-evaluation guard:** this note exists to prevent a future agent from re-running the full mandatory pre-flight + 5-repo fusion search on the same paper. If you arrived here from a grep, the verdict is PASS; do not re-distill unless the paper has a new version with a novel mechanism.

## TL;DR

ATG is an LLM-agent control-framework paper. All three of its mechanisms (recursive graph compilation, dependency-aware parallel execution + thought experiment, minimal necessary subgraph repair) ship in our quintet under different vocabulary — `AndOrNode` + `BlueprintPass` (decomposition), `LLMExecGuard` + pruner cascade (validation), `TrajectoryDoctor::localize_failure` + `FailureSite { depth }` + `ReestimationScheduler` (localized repair), `PagedKVCache::fork` + `MerkleFrozenEnvelope` + `GdnTreeVerifier` (freeze validated regions). The paper's value is its LLM-dependent orchestration process (LLM-as-decomposer, LLM-as-validator, LLM-as-repairer) — the canonical R133/R169/R289 NO-GAIN class. No file, no plan, no guide created beyond this classification note.
