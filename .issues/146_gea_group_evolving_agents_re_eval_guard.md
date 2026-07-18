# Issue 146 — Group-Evolving Agents (GEA): Re-Evaluation Guard

**Filed:** 2026-07-15
**Priority:** P4 (track-only — PASS verdict recorded; this issue exists only to prevent redundant re-distillation)
**Related:** `.research/425_AIDE2_Recursive_Self_Improvement_PASS.md` (governing precedent — same verdict class), `.research/368_AutoMem_Metamemory_LLM_Orchestration_PASS.md` (the decision-structure test), `.research/021_G-Zero_Self-Play_Open-Ended_Generation.md`, `.research/289_RecursiveMAS_Pass_Already_Shipped.md` (multi-agent self-improvement — PASS by a *different* reason; see cousins table)
**Origin:** Research verdict on arXiv:2602.04837 ("Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing", Weng et al., UCSB, Feb 2026) — delivered PASS in-thread 2026-07-15. No `.research/` note was created (PASS = no file per research skill protocol); this issue is the discoverable breadcrumb.

## Context

GEA proposes open-ended self-improvement for a **population** of LLM agents via four
modules: (1) **Reflection** — each agent post-hoc analyzes its own trajectories, (2)
**Evolution** — a Performance-Novelty algorithm (Alg 1) selects which agents survive
into the next round, (3) **Acting** — selected agents execute environment tasks, (4)
**Experience Sharing** — successful agents' trajectories/insights are consolidated into a
shared pool that seeds the next generation.

The mechanism is open-ended in the Quality-Diversity sense (MAP-Elites over a
behavior-characteristic archive), but the **archive stores agent harnesses** (prompt +
tool-set + reasoning framework), not latents, shards, or direction vectors.

## Verdict (recorded in-thread; PASS)

**PASS.** This is the R425 (AIDE²) class — LLM-dependent **process** (reflection generates
text, framework code-patches are synthesized, selection is over agent harnesses) — not
the R368 (AutoMem) class (a **decision structure** that distills modellessly into a
freeze/thaw or routing table).

Three of four modules have no modelless analog on our substrate:

| Module | LLM-as-mechanism? | Modelless analog on our substrate? |
|---|---|---|
| Reflection (post-hoc analysis) | Yes — LLM generates the reflection text | No — no probe/draft/pruner computes "reflect on this trajectory" |
| Evolution (Performance-Novelty selection) | Selection rule is modelless, but **selects over agent harnesses** (an artifact class we do not produce) | No substrate — we have no "agent harness" artifact to archive |
| Acting (task execution) | Yes — LLM executes the task | No |
| Experience Sharing (consolidation) | The *consolidation* is modelless, but it operates on LLM-generated text/experience blobs | Already shipped differently (see below) |

The one modelless module (Performance-Novelty, Alg 1) is textbook MAP-Elites / novelty
search — selecting among **agent harnesses**. That artifact class is absent from our
substrate: we archive `NeuronShard`s, direction vectors, and freeze/thaw snapshots, not
prompt+tool+framework bundles.

### What already ships under different vocabulary (the latent-state reframing)

The GEA pitch — "group experience sharing consolidates diversity into a shared,
re-usable pool that seeds the next generation" — reframed into our substrate is already
covered by the consolidation + coordination substrate:

- **Raven / δ-Mem consolidation** (`katgpt-rs/src/sleep/consolidation.rs`) — the
  sleep-cycle that consolidates wake-events into a frozen `NeuronShard`, gated by
  `FreezeGateReport`. This IS "experience → shared archive entry", just over latent
  wake-events rather than LLM text.
- **Neighborhood heal** — diversity-preservation across the shard index, the
  Quality-Diversity analog that GEA's novelty term is reaching for.
- **Sheaf coordination** — cross-entity latent consistency (the gluing/sheaf
  restriction maps), which is "experience sharing" in the sense GEA means it, but
  operating on latent state rather than agent-harness text.

The latent-state reframing produces no new primitive because the substrate for it
already exists. GEA's contribution is the LLM-side machinery (reflection prompts,
harness code-patching, archive-over-harnesses), which is out of scope for a modelless
inference engine.

## §3.5 Modelless-unblock check (recorded)

All three modelless paths fail for the LLM-dependent modules:

1. **Freeze/thaw** — no frozen snapshot computes "reflect on this trajectory" or
   "synthesize a framework code-patch". The selection step has nothing to freeze.
2. **Raw/lora hot-swap** — a deterministically-constructed LoRA cannot replace the
   "LLM generates reflection text" step; there is no deterministic construction of a
   reflection.
3. **Latent-space updates** — direction-vector projections cannot stand in for code
   generation. The one place latents appear (the archive) already ships as shards.

→ Not a modelless-unblock candidate. Correctly excluded from `riir-train` routing
because the blocker is not "needs gradient descent" — it is "needs an LLM at runtime",
which is outside the 5-repo quintet's product surface (we are a modelless inference
engine + private runtimes, not an LLM-orchestration framework).

## §3.6 PoC requirement — not triggered

PASS-by-scope-exclusion (the mechanism lives in an artifact class we don't have) is not
a PASS-by-parity-claim. No "we already do this" claim is made that would require a PoC
to defend. The claim is "we do the latent-state analog differently, under different
vocabulary" — which is a framing statement, not a parity proof.

## Closest cousins (for future-novelty grep)

| Note | Class | Why a cousin |
|---|---|---|
| **R425 AIDE²** | Identical class | LLM-dependent code-level RSI via harness rewrite. GEA is the population/multi-agent generalization of the same class. **If GEA ever re-evaluates to non-PASS, R425 almost certainly does too, and vice versa.** |
| **R021 G-Zero** | Self-play, verifier-free | Shares the "no external ceiling" framing, but G-Zero's reward (Hint-δ) IS modelless and distilled productively. GEA has no analog of Hint-δ. |
| **R289 RecursiveMAS** | Multi-agent self-improvement (DIFFERENT PASS reason) | Closest prior population-scale multi-agent self-improvement paper evaluated. **Important distinction:** R289 is PASS because every modelless primitive is *already shipped at higher fidelity* (parity verdict); GEA is PASS because its modules are *LLM-dependent with no modelless analog* (scope-exclusion verdict). R289's cross-agent latent comms already ship as the NPC mind-reading bus (Plan 311); GEA's reflection/evolution/acting have no shipped analog because they can't have one modellessly. Cousins by topic (multi-agent), **not** by verdict class. |

## Re-evaluation trigger

This issue exists to prevent a future agent from re-running the full mandatory pre-flight
+ 5-repo fusion search on this paper or its descendants. If you arrived here from a grep,
the verdict is **PASS**; do not re-distill unless **ALL** of the following hold:

1. **A descendant paper strips the LLM code-generation dependency** from the
   reflection/evolution/acting modules — i.e., the "rewrite harness" or "generate
   reflection" step is replaced by a deterministic construction (a freeze/thaw swap, a
   raw/lora hot-swap, or a latent-space projection). Without this, the mechanism is
   fundamentally LLM-orchestration, which is outside the 5-repo product surface.
2. **The Performance-Novelty selection targets a substrate we actually have** —
   latents (`NeuronShard`, `HlaCacheProxy`), direction vectors, or freeze/thaw
   snapshots — rather than agent harnesses (prompt + tool-set + reasoning framework).
   Today we produce no "agent harness" artifact, so MAP-Elites-over-harnesses has no
   archive to populate.
3. **A sub-mechanism is identified that is NOT already covered** by Raven/δ-Mem
   consolidation (`katgpt-rs/src/sleep/consolidation.rs`), neighborhood heal, or sheaf
   coordination — the existing latent-state analogs of "group experience sharing".

A weaker re-evaluation signal (does NOT alone justify re-distillation, but worth a grep):
the paper releases code and the code contains a modelless novelty/diversity-preserving
selector that benchmarks a gain over our existing neighborhood heal. In that case, open
an issue for the *selector primitive* specifically — not for GEA as a whole.

## Tasks

- [-] **T1:** (DEFERRED — only if trigger 1+2+3 all hold) Re-run the research skill
  mandatory pre-flight on the triggering descendant paper. G1 novelty gate, §3.5
  modelless-unblock, §3.6 PoC requirement. DEFERRED — trigger conditions not met.
- [-] **T2:** (DEFERRED — blocked on T1) If T1 produces a non-PASS verdict, create the
  `.research/` note under the next free number and open the corresponding plan/issue.
  DEFERRED — blocked on T1.

## TL;DR

GEA (arXiv:2602.04837) is the R425 AIDE² class — LLM-dependent process (reflection +
framework code-patch synthesis + selection over agent harnesses), not the R368 AutoMem
decision-structure class. Three of four modules are LLM-as-mechanism with no modelless
analog; the one modelless module (Performance-Novelty) is MAP-Elites over agent
harnesses, an artifact class absent from our substrate. The latent-state reframing
("group experience sharing consolidates diversity") already ships as Raven/δ-Mem
consolidation + neighborhood heal + sheaf coordination. Verdict PASS; revisit only if a
descendant strips the LLM dependency AND targets latents/shards/direction vectors.
