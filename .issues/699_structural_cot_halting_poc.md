# Issue 699: Structural CoT halting PoC — answer-space cycle detection (TRACE heuristics, arXiv:2510.07880)

**Status:** Open — PoC queued (Gain verdict per Research 525; GOAT gate required before any promotion)

> **Research:** [katgpt-rs/.research/525_TRACE_Structural_Overthinking_Halting.md](../.research/525_TRACE_Structural_Overthinking_Halting.md)
> **Source paper:** [arXiv:2510.07880](https://arxiv.org/abs/2510.07880) — TRACE (Google DeepMind/UMich): structural analysis of LLM overthinking + two ground-truth-free real-time halting heuristics.
> **Why:** the stack ships ZERO text/discourse-level CoT halting — every halter consumes numeric signals (entropy, residual, gain/cost, patience-on-score). The two paper heuristics (self-loop-K termination; backtrack-revisit termination) form a new signal class: halting on the trace's own answer-space structure, which works black-box (no logits/hidden states) — the only halt family usable on API models, post-hoc trace monitoring, and wasm/Unity consumers. Published prior art: no real-time method ships either heuristic (ES-CoT arXiv:2509.14004 is adjacent for self-loop via *elicited* answer stability; backtrack-revisit has no analog anywhere).

## Mechanism to build (modelless, zero-LLM-rater)

`StructuralTraceMonitor` — consumes a stream of answer-bearing reasoning steps:

- **Answer ring** — normalized distinct answers (bounded ring, e.g. 8 entries; exact/normalized string match for the text variant, embedding-similarity via existing substrate for the latent variant).
- **Transition classification** (answer-space events): `verify` (answer unchanged, shift ≈ 0 → `verify_run += 1`) · `correct` (answer changed → reset run, push ring) · `backtrack_revisit` (new answer ∈ ring reached after an abandonment) → cycle detected.
- **Halting policies:** `SelfLoopHalt { k }` (halt at k consecutive verifies post-answer; paper K=2 default) · `BacktrackRevisitHalt` (halt on revisit-on-backtrack).
- **Pattern-conditional policy selection (the fusion):** classify Explorer vs Late Landing modellessly from the answer histogram — `collision_purity(π)=Σπ²` (shipped, Plan 294 `ict/`) + positional mass (first-half vs last-half concentration) — then select policy: Explorer → backtrack trigger; Late Landing → self-loop K=3. The paper hand-tunes K per model; we derive it.
- **Halt votes:** compose as a third independent signal family beside hidden-state residual (FPRM 266) and gain/cost (282/304) via the existing `GainCostLoopHalter::halt_decision` arbiter shape.
- **Bonus (cheap):** revisit-count as a credibility/vote weight (paper §5.1: a revisited answer ≈ two independent derivations) → CLR / BoMSampler weighting.

## Tasks

- [ ] **T1** `StructuralTraceMonitor` core (ring + transition classifier + `verify_run` + revisit predicate), unit tests on synthetic traces; feature `structural_cot_halt` (opt-in).
- [ ] **T2** Halting policies `SelfLoopHalt` / `BacktrackRevisitHalt` + plug points: agentic/llmexec path (`ThinkingController` trace stream) + `dd_tree/tree_builder.rs` patience loop + `mcts.rs` budget loop (answer-space variant).
- [ ] **T3** Pattern classifier (collision purity + positional mass → policy/K selection) + unit gate pinning the Explorer/Late-Landing → policy mapping.
- [ ] **T4** Defend-wrong PoC in `riir-ai/crates/riir-poc/` (per skill §3.6 — quality claims need head-to-head evidence): ≥3 arms — structural halter vs no-halt baseline vs numeric halter (`GainCostLoopHalter`-analog) — on controlled toy traces + real agentic traces; print verdict table; falsifiable A/B.
- [ ] **T5** GOAT gate: feature flag + benchmark — token savings ≥30% at ≤1% accuracy delta vs no-halt baseline (paper reports 40–60% savings at ≤3 acc delta on Qwen3/R1 traces; we must measure OUR traces); G1 determinism (bit-identical halt decisions per seed); G4 alloc-free steady state; halt-vote composition must never regress accuracy vs the numeric-only arbiter.
- [ ] **T6** (optional, behind same flag) revisit-count vote weighting for CLR/BoMSampler + regression test.

## Boundaries

- Black-box text variant needs NO model access; latent variant MAY reuse `ModellessEmbedder`-class similarity but must not require logits.
- No default promotion without T4+T5; demote the loser if the numeric halters win on the same traces (per-stack ledger rule).
- riir-games consumer (early-exit for `SwarmDeliberationSystem` stuck-NPC search) defers until T5 passes — do not wire game code from this issue.

## References

- Research 525 (verdict + prior-art tables) · Research 282 (gain/cost — the convergence point at latent granularity; externally validated by this paper's utility curves) · Research 266 (FPRM residual patience) · Research 270 (collision purity) · Research 343 (modelless depth/step cousin table) · Plans 294/304/223/026/231/194.
