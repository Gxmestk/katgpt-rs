# Research 496: SPADE — Hint-Regret Frontier Curriculum (Modelless Distillation)

> **Paper:** [SPADE: Self-Play in Adaptive Synthetic Executable Environments](https://arxiv.org/abs/2608.19197) — Liu, Yu, Jiang et al. (UW/Stanford/NEU/CMU/MIT/NUS/SNU), 2026-08-19
> **Code:** https://github.com/spade-rl/spade
> **Date:** 2026-08-21
> **Verdict:** Super-GOAT (4/4 novelty gate; quality-parity PROVEN at PoC scale — §PoC Addendum, Issue 677 DEFEND)
> **Status:** RECORD — PoC 677 defended the quality-parity claim (2026-08-24, riir-mmorpg-examples `c56f072`); Plan 576 remains the open primitive, now unblocked for Guide 340 P2 wiring
> **Mandatory outputs:** Guide [riir-ai `.research/340`](../../riir-ai/.research/340_hint_regret_frontier_curriculum_guide.md) · Plan 576 (katgpt-rs open primitive) · Issue 677 (defend-wrong PoC) · riir-train Plan 346 (training-recipe backlog)
> **Label-anchoring hazard (avoided):** the abstract's "self-play RL framework / GRPO / 30B" vocabulary routes wholesale to riir-train. Path 0 decomposition shows EVERY load-bearing signal is a Monte-Carlo average, band gate, posterior quantile, or limit detector — all modelless-extractable. What needs GD is only *training a neural designer to maximize* the signal; nothing needs GD to *compute, use, or act on* it.

## TL;DR

SPADE trains one LLM in two roles: an **Environment Designer** writing executable Gym-style Python environments and a **Reasoning Agent** solving them. The designer's reward is **hint-based regret** — `rD(e) = r̄A(e|h) − r̄A(e)`, the gap between the agent's return WITH vs WITHOUT a privileged hint the designer also writes. High regret = environment at the learning frontier (solvable with the hint, not without). At 30B-A3B: +8.1 over base, +5.3 over the strongest fixed-environment baseline, +13.9 ACEBench-Agent — margins that GROW with scale while fixed pools saturate.

**The load-bearing finding for OUR stack is a signal-diff against shipped substrate:** `katgpt-core` CGSP's reward `(1 − solve_rate) · guide_score` (verified: `cgsp/derivative_curiosity.rs` L10) is an *aggregate difficulty* signal that **conflates intractable with frontier** — a task the solver can NEVER solve also drives solve_rate → 0 and receives MAX reward. SPADE's hint arm separates the two: **frontier** = high regret (hint helps), **mastered** = low regret + high return, **intractable** = low regret + low return (the hint can't help either). This three-regime triage is the missing discriminator in our curiosity loop, and it is a paired-rollout estimator away — no gradient descent involved in using it.

**Fusion (Super-GOAT composition):** regret-scored frontier curriculum — quest_grammar (designer) seeded by diverse corpus retrieval → paired-rollout regret scoring (hint = one demonstration, the pets substrate) → three-regime triage → mastered content freezes (`MerkleFrozenEnvelope`), frontier content re-ranked by Beta-LCB, intractable content evicted. Connects ≥2 pillars (cgsp + quest_grammar + coverage-curiosity/pets + freeze/thaw) with a falsifiable seed already measured (pets A/B: targeted closure 1.000 vs generic 0.667, 32/32 seeds, ~37 ns/tick).

## Paper Core

**Problem.** Training-environment supply is the bottleneck for agentic RL. Hand-curated, statically-synthesized, and frozen-verifier pools all keep the goal distribution fixed as the learner scales; the agent exhausts them and stops improving.

**Solution.** (1) *Code-as-environment*: the designer writes complete MDPs (state transitions, reward functions, verification) as Python with `reset()`/`step()` — any computable MDP is expressible, unbounding the design space. (2) *Hint-based regret*: the designer also emits a privileged hint (strategy sketch / key observation, never the answer); the agent plays each environment G times with and G times without it; the return gap rewards the designer. (3) *Corpus grounding*: each generation round conditions on a freshly sampled pretraining document (15k pool). (4) *Environment memory*: 200-record buffer of past environments with regret scores + skill tags, conditioning future generation.

**Key measured findings** (all from the paper, our anchors):
- Corpus grounding is **THE diversity mechanism**: Vendi/n 0.68 with corpus vs **0.04 without** (the no-corpus run emitted the same rotating-maze task 41 consecutive times). A FROZEN designer + corpus still holds 0.70 — training adds difficulty-targeting, not diversity.
- Frozen designer + no memory = **9.7 BELOW base** — badly-configured self-play is worse than no training.
- Regret beats EMA learning-potential (the cheap alternative reusing existing rollouts) but EMA captures **~70% of the gain** (+5.7 of +8.1).
- The deployed reward blends floored regret (weight 0.4, fixed scale 0.15) with a flat-top difficulty anchor (weight 0.6, band [0.4, 0.6] win rate, ramp 0.25) — the ANCHOR is the majority weight.
- Learnable share (fraction of envs with win rate ∈ [0.2, 0.8]) rises 0.16 → 0.31 over training — the signature metric of a working frontier curriculum.
- Equilibrium (Appendix B): at every pure Nash equilibrium regret → 0 and hints become vacuous — a convergence/stopping law.

## Path 0 decomposition (mandatory inventory)

| # | SPADE component | Math | Coverage in stack | Extraction (modelless) |
|---|---|---|---|---|
| 1 | Hint-based regret `rD` | VoI of hint; paired-rollout estimator, CRN-variance-reducible, Hoeffding-boundable | **Partial** — CGSP `r_synth` is aggregate difficulty (conflates intractable); SDAR `Δt` is the same gap family but per-token + consumed by a distillation LOSS (training-side) | Paired-sampling estimator + arithmetic. Zero GD to compute/use |
| 2 | Flat-top difficulty anchor / learnable band | Band-pass on win rate: `σ(κ(w−w_lo))·σ(κ(w_hi−w))` | **Partial** — `CgspConfig { solve_rate_floor: 0.05, solve_rate_ceiling: 0.95 }` ships the wide band; no reward-shaped anchor | Closed-form sigmoid band gate |
| 3 | Corpus-grounded generation (anti-collapse) | Sample doc → deterministic seed → generate; Vendi/effective-rank as diversity metric | **Partial** — `SealQuestCorpus` is a frozen corpus (drafter samples it); no fresh-doc-grounded GENERATION; `ShardIndex::retrieve_diverse` enforces diversity at retrieval | Hash-seeded generation + effective-rank tripwire |
| 4 | Environment memory (regret-scored, oldest-first eviction) | Scored replay buffer with skill tags | **Partial** — `LatentFixMemory` (3-axis retrieval, `EvidenceTier`, BLAKE3 commitment) is the same shape for FIX trajectories; not pointed at content generation | Buffer algebra; tier transitions are pure functions |
| 5 | Equilibrium: regret → 0 ⇒ stop/regenerate | Limit law | No — our collapse detection is entropy-based (`tau_low`), not VoI-based | Threshold on estimator + return-level disambiguation |
| 6 | Three-regime triage (frontier / mastered / intractable) | 2-threshold 2D partition of (r̂, R⁻) | **NO** — nothing separates intractable from frontier (CGSP's conflation) | Closed-form 2D gate |
| 7 | Learnable-share metric | Counting + Wilson CI | No | Trivial statistic |
| 8 | Hint injection as search-time prior | `P_h(a) ∝ P(a)·exp(β·1[a ∈ A_h])` | **YES (the proof it works)** — demonstration-teachable pets: hint = hero kill demo, coverage mask = unlocked ranks, falsifiable A/B measured | Already shipped as game substrate; needs the estimator + designer loop around it |

**Funnel result:** all components either have analogs (with real signal-diffs) or extract modellessly → **MODELLESS-VALIDABLE**. Not a wholesale riir-train redirect. Training-side residue (the genuinely-applicable items) → riir-train Plan 346.

## Signal-diffs (§3.6 discipline — every "covered" row defended)

| Cousin (read) | Consumes | SPADE component consumes | Verdict |
|---|---|---|---|
| CGSP `(1−solve_rate)·guide_score` (`cgsp/derivative_curiosity.rs`) | aggregate solve rate + guide quality — **cannot distinguish intractable from frontier** | paired with-hint/without-hint differential — query-conditional VoI | **Real gap.** The conflation is the bug-shape: unsolvable candidates get max reward |
| SDAR `Δt = log π(·\|s⁺) − log π(·\|s)` (Research 038, `loss_sdar.rs`) | per-token privileged-context gap → distillation LOSS (train the student) | per-task return gap → GENERATOR reward / content selection (train/steer the designer) | Same gap family, **different consumer + granularity**. Training side covered; designer side absent |
| `LatentFixMemory` (`riir-clippy/src/memory.rs`) | fix-trajectory success/fail + tier, 3-axis retrieval | task-regret scores + skill tags conditioning generation | Same buffer algebra, **different content + direction** (ours records outcomes of fixes; SPADE's seeds future generation) |
| `SealQuestCorpus` + `AdaptiveQuestDrafter` | frozen corpus → hash-seeded draft (42 ns) | fresh external doc per round → generate NEW content | Ours samples existing; SPADE's grounds creation. **Partial** |
| Pets demonstration learning (Issue 672 / Bench 674 / mmorpg Bench 013) | hint arm (demo) + coverage mask — measured 1.000 vs 0.667 | the SAME hint arm + the missing regret estimator + designer loop | **The hint arm ships and is proven; the designer-side loop is the gap** |

## Adversarial panel (§3.5, both advocates run)

**No-GD advocate** extracted 16 modelless components (full brief in session log; highlights): paired CRN estimator with Hoeffding schedule + analytic VoI oracles (reveal-the-arm bandits, hinted shortest-path); sigmoid band-pass gate (one-line Lean extension of the shipped `sigmoid_bounded` family); regret-scored memory with absorbing-tier eviction; equilibrium stop-law; three-regime partition (property-testable); hint-as-MCTS-prior (β→∞ returns the demo trajectory bit-exactly — G1 oracle); homeostatic difficulty controller (`θ += η(ŵ − w*)` — 1-D proportional control replacing "the designer learns to target the band"); Beta-LCB curriculum ordering (composes shipped `SelectionMode::BetaPosterior`).

**Model-based advocate** extracted 10 recipe items with honest scale verdicts (full brief in session log; highlights): (a) EMA learning-potential = the measured 70%-for-zero-rollouts reward — best ROI item; (b) asymmetric clip 0.20/0.28 + per-role advantage normalization — one-day `loss_grpo.rs` patch; (c) corpus grounding is a **data-pipeline rule, not a training technique** — the strongest transferable finding, with the clippy-L4 unblock (Plan 336's 0/60 FAIL; addressable 9/29 per Bench 037) as the concrete consumer: grounded synthetic span generation over our ~1M lines of in-domain Rust; (d) the frozen-designer configuration (27B ternary inference-only + corpus + memory) recovers ~35% of full SPADE and holds diversity at ceiling — our exact scale fit; (e) the full two-role GRPO replica is **MARGINAL below 4B** (paper's own 4B regret went noisy-negative; our 0.4B–2B sit below the measured regime) — POC-only on Kimi 0.4B with a restricted env DSL.

**Discarded-advocate-findings audit:** none discarded — the Model-based advocate's honest scale verdicts are adopted wholesale (the two-role replica is defer-marked in Plan 346, not scheduled).

## Novelty gate (§1.5)

1. **No prior art?** ✓ In-repo: verified by grep (`hint_regret|difficulty_anchor|environment_design` → only pet-rank `learnable_band`; `regret` → FeedbackBandit cost-model + CGSP aggregate formula) + code reads above. Published: regret-based env design is the PAIRED/ACCEL/PLR line (established); SPADE's own contribution is the hint-as-regret-estimator + code-as-env. Our claim is NOT the mechanism's novelty — it is the **modelless runtime instantiation + the fusion**, which nothing in our corpus articulates.
2. **New behavior class?** ✓ Content/challenges curated at each learner's exact frontier with intractable/frontier separation, at runtime, zero weights. Pets ship the hint arm; no designer loop.
3. **Product selling point?** ✓ "Our quests and pet-training auto-calibrate to each learner's edge — challenges where ONE demonstration unlocks the next step, and never-winnable content is recognized and retired instead of farmed."
4. **Force multiplier?** ✓ cgsp (katgpt-core) + quest_grammar (riir-ai) + demonstration-pets (riir-games/mmorpg) + freeze/thaw (mastered → `MerkleFrozenEnvelope`) + `LatentFixMemory` pattern. ≥2 pillars.

**All 4 YES → Super-GOAT.** Mandatory outputs landed this session (header links).

## Fusion (§Distillation)

**Regret-scored frontier curriculum loop** — the composition none of the parts has alone:

```
diverse corpus retrieval (ShardIndex::retrieve_diverse / fresh-doc grounding)
        ↓ seeds
quest_grammar drafter + ConstraintPruner  ←—— environment memory (LatentFixMemory
        ↓ generate candidates                  pattern: regret-scored, skill-tagged,
paired-rollout regret scoring                oldest-first eviction)
  arm A: solver WITHOUT hint  ── G rollouts
  arm B: solver WITH hint (= one demonstration, pets substrate) ── G rollouts
  r̂ = mean(B) − mean(A)   [CRN shared seeds; Hoeffding K(ε,δ) schedule]
        ↓
three-regime triage:
  frontier    (r̂ ≥ τ_r)                → keep active, Beta-LCB re-rank
  mastered    (r̂ < τ_r, R⁻ ≥ τ_R)     → freeze (MerkleFrozenEnvelope), retire from rotation
  intractable (r̂ < τ_r, R⁻ < τ_R)     → evict from memory, reseed from corpus
        ↓
learnable-share metric (Wilson CI) — the signature: share ∈ [0.2, 0.8] band must rise
```

What none of the cousins alone can do: CGSP has the loop but not the discriminator; pets have the hint arm but not the estimator or the designer; quest_grammar has the designer but not the scoring; freeze/thaw has the retirement but not the trigger. The loop composes all four.

**Secondary fusion (riir-train Plan 346):** corpus-grounded synthetic span generation for the clippy L4 fallback (the 9/29 addressable misses) + EMA learning-potential as the zero-cost arena reward + asymmetric clipping.

## Honest caveats

1. **Quality parity UNPROVEN (the §3.6 obligation).** SPADE's +5.3/+13.9 are *trained-designer* results. The claim "a modelless selection loop over the same signal recovers meaningful gains" is architecturally plausible and latency-cheap, but the head-to-head (modelless regret-gated curriculum vs uniform vs aggregate-difficulty CGSP reward) has not been run. PoC = Issue 677, hosted in the pets harness (the existing falsifiable A/B rig) — NOT claimed as PASS here.
2. **Estimator noise at small G.** The paper's own 4B/8B regret estimates dip below zero for long stretches (finite-sample noise; regret is non-negative only at the optimum). Mitigation shipped in the plan: CRN pairing, Hoeffding-scheduled K, the anchor majority-weight blend (0.6) carrying the load at low G, and the Beta-LCB ordering refusing to promote thin evidence.
3. **The 70% alternative.** If the PoC shows the paired-rollout overhead (2× rollouts) buys less than the EMA-potential baseline at our scale, the primitive demotes to the triage gate only (component 6 — still uniquely valuable against the CGSP conflation).
4. **What genuinely needs GD:** training a neural designer to *generate* frontier content beyond the drafter's parameterized space (SPADE's open-endedness claim). That is the invisible-leash boundary — our frozen drafter + corpus stays inside the leash; SPADE's trained designer pushes it. The riir-train POC (restricted DSL, Kimi 0.4B) is the honest probe of whether that matters at our scale.

## Validation protocol (GOAT gates — full detail in Plan 576)

- **G1 (correctness):** analytic VoI oracles — reveal-the-arm bandit (hint exposes μ*: `r = μ* − max_j μ̂_j`) + hinted-shortest-path (β→∞ returns the demo trajectory's return bit-exactly); estimator within 2× Hoeffding bound at prescribed K; three-regime partition property test (exactly one cell per (r̂, R⁻), boundaries pinned).
- **G2 (perf):** CRN variance ratio ≥ 2× vs independent seeds; per-pair scoring alloc-free; O(K) rollouts.
- **G3 (no-regression):** cgsp default suite unchanged with the gate off; feature-gated `hint_regret`.
- **G4 (alloc):** scratch-buffer paired estimator, zero steady-state alloc.
- **G8 (behavior):** learnable-share rises under the loop vs no-gate control (the paper's own 0.16 → 0.31 signature, replicated modellessly); mastered fraction freezes; intractable content evicted rather than farmed.

## Priority

- **P0:** katgpt-core `hint_regret` module (estimator + band gate + triage) behind `hint_regret` feature — Plan 576.
- **P1:** PoC Issue 677 in the pets harness (defend-wrong: modelless loop vs uniform vs aggregate-difficulty).
- **P2:** riir-ai quest-center consumer (frontier-weighted quest offering) — after PoC verdict.
- **P3:** riir-train Plan 346 items (corpus-grounded clippy-L4 spans first — it unblocks an existing measured failure, not a new capability).

## PASS-Redirects (synthesis)

None (Gain verdict). Closest shipped cousins updated by this note's signal-diff table instead: Research 038 (SDAR — designer-side gap noted), Research 240 (SGS/CGSP — conflation documented), riir-ai `.research/126` (CGSP guide — three-regime extension pointer in Guide 340).

> **Follow-up (2026-08-22):** [EnvHarness arXiv:2608.19880] — the architectural COMPLEMENT to this note: transform a FROZEN env via composable Setup/Rule/Link wrapper layers (verifier inherited untouched) instead of generating new MDPs; diagnosis-first targeting (name the weakness, then rig) vs this note's score-first loop. Distilled as Research 500 (katgpt-rs) with the wrap-axis signal-diff + the delta Path 0 table; refines Plan 576/Guide 340 before implementation.

## PoC Addendum (Issue 677 — defend-wrong verdict, 2026-08-24)

**VERDICT: DEFEND.** All pre-registered gates pass; the quality-parity
caveat (§Honest caveats #1) is discharged at PoC scale. Harness:
riir-mmorpg-examples `c56f072` (`src/pet_teaching_677.rs`, extending the
Bench-013 rig). 32 seeds, release, bit-identical double-run; every arm ran
the REAL systems (`CombatSystem` + `PetAiSystem` + the pets teaching
substrate), identical boot (2× rank-1 warmup) + CRN sites per seed, and the
validated `TargetedExemplarSeeking` attention policy — the CONTENT-SELECTION
policy was the only variable.

### The rig (what "content selection" means here)

The Bench-013 rig had a fixed encounter script; here each training slot
(4-encounter block, Bench-013 density) has its monster rank chosen by the
arm's policy, decided at slot end on the live pet state. Content pool =
ranks 1..=6: 1..=4 the pet's world (rank-2 pet: natural {0,1}, learnable
{2,3,4}); **5..=6 model boss content beyond the species ceiling — a
demonstration there can never translate into pet performance** (the
eligibility gate never opens past `MAX_SKILL_LEVEL`=4): intractable by
domain construction, discoverable only empirically.

- **Arm 1 RegretGated** — the inline Plan-576-style estimator (~60 LOC;
  Plan 576 is NOT landed — the PoC did not block on the primitive): paired
  probe of every candidate over shadow skill states,
  `r̂ = mean(with-demo) − mean(without-demo)`, K=4 pairs per arm per
  candidate; two-threshold triage (τ_r = τ_R = 0.5) keeps frontier, freezes
  mastered, evicts intractable; offers the top frontier item; STOPS
  (budget unspent) when no frontier remains.
- **Arm 2 Uniform** — rank uniform over the full pool 1..=6.
- **Arm 3 AggregateDifficulty** — CGSP-shaped `(1−solve_rate)·guide`,
  guide = 1 (uniform guide quality — nothing rigged), solve rate = Laplace
  posterior over the pet's REAL per-rank swing record (new `rank_stats` on
  `PetTeachingState`); argmax, least-recently-offered tie-break; no hint arm.

### Measured (raw, 32 seeds)

| Metric (mean) | 1 regret | 2 uniform | 3 aggregate |
|---|---|---|---|
| closure @ 3 slots | **1.000** | 0.313 | 0.667 |
| closure @ 6 slots | **1.000** | 0.635 | 1.000 |
| closure @ 12 slots | **1.000** | 0.875 | 1.000 |
| time-to-frontier ticks (first cover) | **252** (32/32) | 555 (32/32) | 252 (32/32) |
| ticks-to-full-closure | **632** (32/32) | 1621 (**20/32**) | 1087 (32/32) |
| wasted intractable encounters | **0.0** | 15.1 | **24.0** |
| slots spent (of 12) | **3** | 12 | 12 |
| regret probe overhead | 88 intractable samples/seed (192 total, 4 rounds) | — | — |

**Gates (pre-registered, not tuned):** wins_primary (arm 1 strictly beats
arm 2 to full closure) = **32/32** (gate ≥ 28/32) ✓; wins_ttf = 32/32 ✓;
waste gate arm1 0.0 < arm3 24.0/2 = 12.0 ✓. Regret's ticks-to-full range
604–661 across seeds; uniform never closes on 12/32 seeds (coupon-collector
misses: 12 uniform draws over a 6-rank pool miss one of {2,3,4} with the
expected ~31% probability — the measured 12/32 matches).

The 12 uniform non-closures are the closure-per-budget story in one number:
at ANY budget ≥ 3 slots the regret arm holds closure 1.000 while uniform's
mean is 0.313/0.635/0.875 at 3/6/12. The paired-rollout 2× overhead
(the REFUTE axis) is measured at 192 probe samples/seed of scheduler-side
compute — the whole 96-run sweep executes in 0.07 s release; there is no
regime in this domain where it could buy nothing.

### Honest findings (the parts that were not designed)

1. **The estimator rediscovers the hand-tuned curriculum.** Arm 1 offers
   exactly `[2, 3, 4]` on every seed — the Bench-013 script minus warmups,
   which was hand-derived from the cluster-density finding. The modelless
   selection loop produces the expert-authored sequence deterministically,
   then stops (3 of 12 slots spent).
2. **Prefix-blocked ≈ intractable to the probe.** At boot, ranks 3/4
   (learnable, but the pet cannot attempt them until 2 is covered) carry
   the same (r̂=0, R⁻=0) signature as ranks 5/6 (never learnable). The
   estimator cannot distinguish them — it does not need to: per-round
   re-probing reclassifies 3/4 as frontier the moment the prefix completes,
   and no non-frontier item is ever OFFERED. 88 of the 192 probe samples
   land on intractable-classified candidates (including those early 3/4
   probes). This is the domain forcing the loop to be stateful, exactly as
   SPADE's frontier moves with the learner.
3. **The conflation costs closure SPEED and WASTE, not frontier-hitting.**
   Arm 3 reaches the frontier as fast as arm 1 (ttf 252 = 252) — the
   aggregate signal is not blind to difficulty, only to the
   intractable/frontier DISTINCTION. Its losses: (a) full-closure speed —
   the Laplace solve-rate rises slowly, so it re-drills each rank twice
   ([2,2,3,3,4,4], 1087 vs 632 ticks); (b) the permanent tail — after the
   band completes it farms 5/6 for 6 of 12 slots (24 encounters, and 3
   wasted exemplar writes polluting the taught-mask with bits that can
   never matter). The issue's prediction ("this arm should farm intractable
   content") holds; the refinement is that it also gets there slower.
4. **Uniform wastes on TWO axes**: intractable slots (15.1 encounters) AND
   mastered re-offers (mean ~21 encounters at already-covered ranks) — the
   freeze-mastered half of the triage is worth as much as the evict half.

### Harness adaptations (recorded per the issue's instruction)

- **Learnable share := closure over the learnable band.** SPADE's metric
  (fraction of pool with win rate ∈ [0.2, 0.8]) is not expressible in this
  domain: `combat_tick` hits are CERTAIN (no miss chance), so per-item
  solve is bimodal {0, 1} and nothing ever SITS in the band. Closure is
  the domain-faithful analog — content moved from unsolvable to solvable,
  rising 0 → 1.0 under arm 1 exactly as the 0.16→0.31 signature demands
  directionally; the difference (SPADE counts content AT the frontier, we
  count content CROSSED) is inherent to binary lookup-binding skills.
- **Budget = training slots** (encounter blocks). Probe samples are
  scheduler-side arithmetic (the modelless loop's actual overhead —
  measured above), not training engagements; both are reported so the
  waste gate can be recomputed under stricter accountings. Under the
  strictest reading (probe samples counted as attempts at ~10 pet swings
  per encounter-equivalent), arm 1's intractable-attempt total would be
  88/10 ≈ 9 encounter-equivalents vs arm 3's 24 — the gate (0 < 12)
  passes on the slot accounting, and would FAIL on that strictest one;
  the honest claim is the slot-level one, since probes move no game
  entity. Recorded so a skeptic can re-derive either way.
- **Determinism of the sweep**: the 192 per-seed report lines are
  bit-identical across independent release runs (verified by output diff
  across three runs, one spanning the print-only fix); the debug
  determinism gate additionally asserts full `ArmReport` equality on
  5 seeds × 3 arms double-run.

### Disposition

- **Guide 340 P2 (riir-ai quest-center consumer): UNBLOCKED** — the loop
  ships as the validated selection policy (the issue's DEFEND branch).
- **Plan 576 (the primitive): still worth landing** — the PoC ran the
  estimator INLINE (~60 LOC) because the primitive was not landed; the
  primitive adds the CRN/Hoeffding K schedule + Wilson CI + Beta-LCB
  ordering that this domain's deterministic swings made vacuous here but
  any stochastic-solve consumer needs. The "Report the Floor" gate
  (triage accuracy vs single-arm banding at matched budget) remains the
  primitive's own obligation (Plan 576 Phase 4 G-Floor).
- **The REFUTE branch (primitive demotes to the triage gate alone) did
  NOT fire** — the full loop beat uniform 32/32 with measured-negligible
  overhead.
- Issue 677 closed + removed (noise-reduction rule); this addendum is the
  record.
