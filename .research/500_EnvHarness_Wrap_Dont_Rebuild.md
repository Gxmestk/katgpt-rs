# Research 500: EnvHarness — Wrap, Don't Rebuild (Fusion Refinement of 496)

> **Paper:** [EnvHarness: Awakening Static Worlds for Agent Learning](https://arxiv.org/abs/2608.19880) — Huang, Wang, Han et al. (Google Research), 2026-08-20
> **Code:** https://github.com/google-research/envharness (Apache-2.0, Python; Setup/Rule/Link + designer loop + GRPO via verl-agent)
> **Date:** 2026-08-22
> **Verdict:** Gain — the architectural COMPLEMENT to Research 496 (SPADE). Not Super-GOAT: Q1 fails (the wrapper-over-frozen-env axis is established prior art — Gymnasium JMLR 2024, ACCEL ICML 2022, ALP-GMM JMLR 2020).
> **Status:** RECORD — P1 fold DECIDED (YES — owner-delegated perf+sec call, 2026-08-23) + executed: Guide 340 §"Wrapper composition" (c11875ded) + Plan 576 folded (this commit). P0 → riir-train Plan 348 Item A; P2 rides PoC 677.
> **Label-anchoring hazard (avoided, same class as 496):** "Agent Learning" + GRPO vocabulary routes wholesale to riir-train. The adversarial panel (both advocates, this session) shows the load-bearing deltas are interface algebra + closed-form statistics + one sampling policy. Only the recipe shell is frontier-scale.

## TL;DR

SPADE (496) generates NEW environments code-first; EnvHarness **transforms a frozen environment through composable wrapper layers** — three component types at the standard `reset`/`step` interface: **Setup** (reshape initial state via committed action-prefix replay), **Rule** (reshape interaction: allowed actions, their effects, observations), **Link** (compose another env's tasks in). The env's verifier stays **untouched** — every reshaped env inherits the original trusted verifier. **EnvRigger** is the automation: read the policy's trajectories → **diagnose a named weakness** → write components targeting it → validate on fresh rollouts → revise. Results: at equal budget it beats BOTH original envs and domain-specific generation pipelines (GenEnv/VeriEnv/SWE-smith all lost); band coverage 6%→80% purely through interface constraints; skills transfer back to the UNTOUCHED benchmark (up to +9.0 OOD, −9.8% steps); GRPO signal improves; gains compound over designer rounds.

**Load-bearing for us (three things 496 does not carry):**
1. **The wrap-vs-generate axis.** Our quest_grammar drafter is SPADE-shaped (generate + score). We ALSO own the exact thing EnvHarness wraps: a FROZEN quest table (`QuestTemplateRow`, 10 rotating templates) + a static restock FSM. The pending Plan 576/Guide 340 rollout can be refined BEFORE implementation: quest **modifier composition** over the frozen table instead of (beside) new-quest generation — cheaper, verifier-inherited by construction, and composable.
2. **Diagnosis-first targeting.** SPADE scores post-hoc (generate → regret-score); EnvRigger diagnoses pre-hoc (name the weakness → rig for it). Modellessly, the diagnosis is closed-form over observables we already record (see Path 0 rows 4–7) — no LLM needed.
3. **Zero-advantage group filtering** — a training item with NO SPADE overlap, verified absent in `riir-train-gpu/src/loss_grpo.rs` (group=16, z-scored advantage; an all-success/all-fail group carries no gradient and wastes 16 rollouts; no task-saturation filter ships). Cheapest item in the whole distillation; lives in riir-train Plan 348.

## The axis (signal-diff vs 496 / Guide 340)

| | SPADE (496) | EnvHarness (this) | Our substrate |
|---|---|---|---|
| Environment supply | generate new MDPs as code | transform frozen env via 5 interception hooks (reset / action-filter / action-transform / obs-shape / task-set) | quest_grammar generates; quest table is frozen; NO transform layer ships |
| Verifier | authored per env (designer writes it) | **inherited untouched** — the whole argument for wrap > generate at equal budget | quest completion predicate (fixed); raw-sync/bit-identical discipline is the kinship |
| Targeting | post-hoc: regret-score what was generated | pre-hoc: diagnose the named weakness, then rig | coverage_curiosity re-targets (agent-side); nothing env-side |
| Difficulty control | reward anchor band [0.4, 0.6] win rate | same band, achieved through **interface constraints only** (6%→80% in-band coverage) | 576's sigmoid band-pass gate (pending) |
| Evaluation | improvement on generated pool | **transfer back to the untouched original** (held-out) | scenario-runner fronts (the native A/B shape) |

## Path 0 delta decomposition (panel-verified, both advocates)

Rows marked "exists/partial" signal-diffed per §3.6. Delta only — band/regret/triage/Beta-LCB-ordering are 576's territory, not re-extracted.

| # | EnvHarness component | Math / mechanism | Coverage in stack | Extraction (modelless) |
|---|---|---|---|---|
| 1 | Wrapper algebra: Setup/Rule/Link as env endomaps | H: Env→Env closed over 5 hooks; Setup replay = free-monoid action of A* on S (prefixes compose by concatenation); Rule = `ρ_post ∘ step ∘ ρ_pre`; stack = closure ("a wrapper IS an env") | **No** — quest_grammar generates; grep confirms no modifier/harness surface in riir-games-quest (only Bevy `Resource` wrappers, different concept) | YES — trait delegation + stacking combinator; the stack is committed DATA (BLAKE3-able, freeze/thaw-able) |
| 2 | Verifier untouchability | type-level (components never see `verify`) + runtime `BLAKE3(canonical(verifier))` asserted invariant across wraps | **Partial** — raw-sync/bit-identical discipline is the same VALUE, not an enforced wrapper contract | YES — separation + O(1) hash check per episode |
| 3 | Setup replay as curriculum artifact | committed `Vec<u32>` action prefix; raw, replayable bit-identically; prefix-length on mastered segments = difficulty dial with planted direction | **No** | YES — pure determinism |
| 4 | Per-slice Beta-LCB weakness ordering | partition episodes by content slice (quest kind, rank band, element, zone); weak iff `LCB_Beta < P̂_global − margin` | **Partial** — BetaPosterior (riir-clippy) gates CANDIDATES, not content slices; PlanDiag records replan reasons as data, consumes nothing | YES — counting + Beta quantiles. **UQ floor rule applies** |
| 5 | Coverage-hole vs brittleness decision law | hole `V∖A ≠ ∅` → exposure (needs Link/Setup); brittle `n_s ≥ m ∧ fail` → practice (needs Rule) — one set-difference | **Partial** — pets `covered_mask` is exactly this law at per-rank granularity (u8); not generalized to content atoms, not env-side | YES — bitmask set-diff |
| 6 | Failure-signature clustering | `σ(ep) = (terminal_phase, last_failed_precondition, death_cause)` → `HashMap<σ, count>` — exact counting, no learned clustering | **Partial** — the raw material ships as data (PlanDiag replan reasons, karma events); nothing reads it as diagnosis | YES |
| 7 | Composition-deficit detector Δ_AB | `min(P_A, P_B) − P_{A→B} > θ` flags sequencing weakness (can do both halves, not the chain) | **No** — chained quests (tame→hunt) ship as STRUCTURE, not as a diagnosed weakness | YES — three Beta counters |
| 8 | Weakness→hook routing table | closed total function: hole→Link/Setup, slice-deficit→Rule+Setup, Δ_AB→Link, early-death→Setup prefix-skip, obs-brittleness→Rule obs-hook | **No** | YES — enum→enum lookup (the "designer" is a router) |
| 9 | Transfer-back A/B protocol | arms adapt under H(E) vs E, identical budgets, both evaluated on frozen E; verifier-hash identical across arms | **Partial** — scenario-runner fronts are the native shape; no transfer-back arm exists | YES — measurement design; **this IS the §3.6 instrument** |
| 10 | Fault channel: bad component = recorded trace | totalized `run(c) ∈ Ok ∪ Err{Panic,Budget,Malformed,VerifierTamper}`; k strikes → Withdrawn (absorbing) | **Partial** — EvidenceTier absorbing transitions are the same law for fix trajectories | YES — `catch_unwind` + budget caps + schema validation |
| 11 | Band-tracking selection over compositions | per-composition `Beta(p_w)`; select `argmin_w |mean(p_w) − c|` — co-evolution = posterior + selection rule, no gradient | **Partial** — DualPool/HintDelta bandits select over ITEMS; arms-as-compositions is the delta | YES. **UQ floor rule applies** |
| 12 | Monotone-chain calibration walk | planted difficulty directions (mask strictness ↑, chain depth ↑, prefix-skip ↑=easier); order test on posterior means; violations demote chain to poset; O(log n) calibration probes | **No** | YES — pairwise counting + order statistic |
| 13 | Zero-advantage group filtering (RL batch side) | drop tasks with rolling SR outside (≈0.1, 0.9) BEFORE spending K rollouts; all-same group ⇒ σ=0 ⇒ no gradient ⇒ 16 wasted rollouts (paper §F.2 round-3: ~1/3 of tasks contributed nothing) | **NO — verified absent** in `loss_grpo.rs` | YES — a sampling policy; **negative GPU cost**; riir-train Plan 348 Item A |

**Funnel result:** every delta row extracts modellessly or composes with pending 576 substrate → the runtime half is MODELLESS-VALIDABLE (no deferral). Training deltas → riir-train Plan 348 (composing with, not duplicating, Plan 346).

## Adversarial panel highlights (§3.5)

- **No-GD advocate (14 items, all YES):** the decisive observation — EnvHarness's interface is CLOSED (5 hooks, 1 invariant), so every "learned" artifact in EnvRigger decomposes into a table + a posterior + a threshold. The paper's free-form Python `_Rules` authoring is one (unsafe, unverifiable) encoding of a choice from a five-hook space; a bounded component vocabulary + composition spans the same reachable set. Honest boundary stated: whether bounded vocabulary reaches the SAME CEILING as LLM authoring is empirical — row 9's protocol is the instrument, parity NOT claimed here.
- **Model-based advocate (8 recipe items, prioritized 13→3→1→2→6→5→8→7):** the key distinction preserved throughout — **SPADE shapes the REWARD; EnvHarness shapes the ENVIRONMENT DISTRIBUTION while the reward stays the raw verifier signal.** Different levers on the same GRPO machine; a single A/B must never move both. Scale table: Stage prefix-replay GO (scale-free, envs merely need deterministic+resettable — ours are by the raw-sync rule); Contract action-masking GO and *stronger* at small scale (weak policies exploit reward shapes more; masking leaves no reward-hacking surface); diagnosis→data-curation GO adapted (programmatic failure-cluster conditioning replaces the paper's 228M-token LLM diagnosis); zero-advantage filter GO FIRST (negative cost); skill-bank RAFT GO-capped (kill if inside noise of Plan 346 A); invalid-action penalty 0.1 GO (one constant + ScoreDiag guards already shipped); Chain budget-sharing last; full GRPO-in-env recipe 2B-LoRA-only (paper ran Qwen3-8B on 8×H100 — 40-80× our rollout budget).
- **Discarded-advocate-findings audit:** none discarded; both briefs adopted with their own honest scale verdicts.

## Prior-art verdicts (§4 searches, agent-verified)

1. **Wrapper taxonomy over frozen envs: clearly established** — Gymnasium (JMLR 2024) formalizes composable wrappers (`TransformReward/Observation/Action`, `TimeLimit`); DI-engine et al. make stacking ambient practice. A named Setup/Rule/Link taxonomy is vocabulary, not mechanism.
2. **Transformation-vs-generation axis: clearly established** — ALP-GMM (JMLR 2020) reparameterizes one env; **ACCEL (ICML 2022) edits previously discovered levels** (the strongest single preempt for "modify, don't generate"); ICML 2026 parameter-change UED continues the axis.
3. **Diagnosis-driven targeting: adjacent-but-distinct** — DDA models scalar skill (Hunicke & Chapman 2004; Zook & Riedl 2012); ITS targets knowledge deficits via item selection; regret-based UED is weakness-targeted for RL agents. The LM-diagnosis→env-transformation loop for LLM agents is the plausibly-novel cell; our closed-form instantiation is novel in OUR stack.
4. **Verifier preservation: adjacent-but-distinct** — Ng/Harada/Russell 1999 policy-invariance under reward transformation is the classic analog; train-on-curriculum/eval-on-frozen is standard UED protocol. The wrapper CONTRACT formalization is the contribution.
5. **Skill induction + transfer-back: established on the agent side** (Voyager, ExpeL, ReasoningBank — same org, lineage not independent prior art); env-side is the complement cell.

**Net:** the COMBINATION (diagnosis-driven + verifier contract + composable named classes over frozen LLM-agent benchmarks) is plausibly novel in the literature; no single part is. Q1 therefore fails for a mechanism-novelty claim; our Super-GOAT slot for this problem class is already held by 496 (the curriculum loop), which this note refines architecturally.

## Fusion (§Distillation)

**1. Guide 340 / Plan 576 refinement (the headline — both still PENDING, fold before implementing):** the quest-center consumer becomes **Setup/Rule/Link modifier composition over the FROZEN `QuestTemplateRow` table**:
- *Setup* — spawn configuration, monster rank/element draw, resource endowment, committed prefix (curriculum artifact).
- *Rule* — interaction constraints: no-heal rounds, elemental-counter-required, denser fog, time limits. Maps to existing levers: `monster_predation` FOV profiles, `WEAKNESS_MATRIX` counters, quest `kind_filter`.
- *Link* — chained quests (tame→hunt prerequisites ALREADY ship as structure — Link is half-built).
- Verifier = quest completion predicate, untouched + BLAKE3-pinned (row 2 contract). Difficulty steering = 576's band gate consuming wrapper-selection as its lever (row 11) instead of a scalar knob.
- Diagnosis-first loop: weakness taxonomy from recorded observables (`taught_mask` holes, PlanDiag replan reasons, karma events, per-kind quest credit, elemental death causes) → routing table (row 8) → modifier selection. Deterministic, modelless, ~µs.

**2. PoC 677 refinement (one extra arm, one extra assert):** add the **wrap arm** (modifier-rigged variants) beside regret-gated-generated and uniform; add the **transfer-back evaluation** (all arms scored on the UNTOUCHED quest table, verifier-hash asserted identical across arms). This is the paper's protocol transplanted — and it directly tests the bounded-vocabulary question (caveat 2).

**3. riir-train Plan 348:** zero-advantage group filter first, then Stage/Contract augmentation arms, diagnosis→data curation, invalid-action penalty. Explicit no-double-count rule: the band mechanism appears in exactly ONE arm per A/B (Plan 346's reward-anchor OR 348's batch-filter, never both).

## Honest caveats

1. **Quality parity UNPROVEN (§3.6).** +9.0 OOD is Gemini-3.5-Flash designer + policy scale. Our modelless diagnosis→routing→modifier loop is architecturally plausible and latency-cheap; the head-to-head (wrap vs generate vs uniform, transfer-back) is exactly PoC 677's extended rig. Not claimed as PASS here.
2. **Bounded-vocabulary ceiling.** Free-form LLM rule-writing may reach reshapes a closed modifier vocabulary cannot. The paper's own round-3 record shows gains shrinking as flaws grew local (expect 2–3 productive rounds, not 10) — the PoC should measure vocabulary saturation, not just closure rate.
3. **Overlap discipline.** Band/regret/triage/Beta-LCB are Plan 576's — rows here COMPOSE (routing consumes 576's gates; bandit arms supplied by wrapper algebra), never re-implement.
4. **RL-signal claims are frontier-scale** (Qwen3-8B, 8×H100, verl-agent). Only the scheduling insight (env/distribution quality substitutes for step count) transfers to our 2B-LoRA regime — and it must be A/B'd, not assumed.

## Priority

- **P0:** riir-train Plan 348 Item A — zero-advantage group filtering (negative GPU cost, no SPADE overlap, improves every `loss_grpo.rs` consumer).
- **P1:** fold the wrapper-composition + diagnosis-first design into Guide 340 §"Wrapper composition" + Plan 576 BEFORE implementation (both pending — the fold is nearly free now, expensive after).
- **P2:** PoC 677 gains the wrap arm + transfer-back evaluation (caveat 1's instrument).
- **P3 (PoC-gated):** katgpt-core open-primitive candidates — `harness` (wrapper algebra, verifier-hash invariant) + `traj_diag` (slice LCB, signatures, Δ_AB). NOT opened now: the closed 5-hook space is small enough to live game-side first; promote to open primitive only if the PoC vindicates generality.

## PASS-Redirects (synthesis)

None (Gain verdict). Closest cousins updated instead: Research 496 (wrap-axis cross-ref appended below), Guide 340 (§"Wrapper composition" added), riir-train Plan 348 (training deltas).
