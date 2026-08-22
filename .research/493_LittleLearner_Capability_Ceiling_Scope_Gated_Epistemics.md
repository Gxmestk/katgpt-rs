# 493 — LittleLearner: The Capability-Ceiling Law + Scope-Gated Epistemics

**Status:** IMPLEMENTED (opt-in POC) — Issue 674 T1–T4 shipped behind `contrastive_scope`, GOAT G1–G4 + T4 battery PASS ([Bench 669](../.benchmarks/669_contrastive_scope_poc_goat.md)); T5 verdict recorded (no consumer adopted yet — promotion deferred, issue closed per its own rule) (tracking: katgpt-rs Issue 674 + riir-train Issue 469)
**Verdict:** Gain (not Super-GOAT — Q1 fails: every extracted technique is textbook; the value is the law as planning gate + the fusion placements)
**Paper:** Li, Zeller, Prada-Corral, Wiedemer, Mayilvahanan, Cotterell, Brendel — arXiv:2608.13545 "LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure" (MPI-IS / ETHZ, Aug 2026)

## TL;DR

LittleLearner is a 5B LLM trained from scratch on **LittleCurriculum** — an 88B-token corpus filtered from FineWeb-Edu to *only* US K-5 (elementary school) content — creating a controlled-exposure sandbox with an interpretable knowledge boundary. Their suite of interventions on the boundary yields one law and one failure-mode signature:

1. **The capability-ceiling law**: the pretraining distribution — not the intervention — sets the effective capability ceiling. Scaling 0.6B→1.3B→5B improves in-scope only (and partially at the boundary, Grades 6-7); SFT+GRPO post-training boosts in-scope but fails to recover Beyond-K-5 *even when post-trained on out-of-scope data*; ICL few-shot steers output format but does not unlock out-of-scope reasoning. At pass@1024 the Grade-8 plateau persists — it is a capability ceiling, not a sampling artifact.
2. **The out-of-scope failure signature**: the model does *not* express uncertainty. It "systematically projects unfamiliar concepts onto familiar reasoning patterns learned during training" — coherent but incorrect (Schrödinger's cat → "a cat with two faces"; E=mc² → "a character from Star Wars"). Confidence is *miscalibrated exactly where it matters*; the corrective signal must be **external and corpus-derived, not model-reported**.

**Why this matters here (triply corroborated on our own stack):**

| Corroboration | Where |
|---|---|
| Paper | LittleLearner scaling/post-training/ICL experiments (§4) |
| Our L4 generative fallback | Bench 465 + 467: **0/60** usable fixes, tied with the frozen-backbone control |
| Our Gemma2 direction source | Bench 571: reads real semantic signal but ranks a JSDoc pass above a critical security vuln |
| Our in-scope training WIN | Issue 717 G2-full: trained DSpark drafter **+17.4%** true acceptance (1.689 vs 1.439 tok/cyc) — in-scope amplification works |

The paper promotes our repeated measurement from anecdote to law: **training amplifies what the substrate contains; it does not create what it lacks.** This is the strongest external validation yet of the three-track thesis (modelless inference + self-adaptive latent updates operate *within* the frozen distribution; only base-model selection moves the boundary).

## Paper mechanics (what they actually did)

- **Filtering cascade** (cost-ordered, precision-first): AoA (Age-of-Acquisition) pre-filter with Zipf-frequency imputation → LLM-as-judge annotation (Gemini Flash, OpenEvolve+DSPy/GEPA-optimized prompts, majority vote, ties toward *higher* grade) → FastText cheap classifier → ModernBERT expensive classifier (only on the residual band) → symbolic regex filter for math notation → **frequency sampling** with a contrastive blocklist. Full-corpus LLMJ would cost ~$46M — the cascade is the economics.
- **Validation**: BPB on CLEAR/CoMTA rises with out-of-scope difficulty (LittleLearner only); Jeopardy science split by NGSS grade collapses at the boundary; MathCAMPS pass@1 drops disproportionately.
- **Contrastive frequency-ratio score**: `score(w) = log2(rate_Beyond-K5(w) / rate_K5(w))` — terms disproportionately associated with out-of-scope material form a rank-ordered blocklist for final sampling.
- **Small-model specialization**: at 0.6B the restricted model *outperforms* the unfiltered control in-scope — no advanced content competing for capacity. Disappears by 5B.
- **Non-human acquisition order**: LittleLearner performs *better* on multi-digit division than single-digit division — capability acquisition does not follow the human curriculum prerequisite DAG.

## Path 0 decomposition (training-target → modelless components)

| Paper component | Coverage in stack | Extraction |
|---|---|---|
| Contrastive frequency-ratio score table | **No analog** (BM25/IDF is single-corpus; all "contrastive" code is latent-direction contrast: TILR/CNA/MAG) | **YES — closed form**: two streaming count passes + log2, additive smoothing; BLAKE3-committable table |
| Document-level scope score | **No analog** (relevance gates check query-conditional relevance, not input-distribution membership) | **YES — closed form**: `D(x) = dot(count_vec, log_ratio_vec)` = Naive Bayes log-LLR = a sparse GEMV our ternary SIMD substrate accelerates |
| Scope-conditioned confidence haircut | **No analog** (EvidenceTier = history; engram gate + Issue 030 = query-conditional relevance; neither is input-scope) | **YES — closed form**: `ĉ = c · sigmoid(−κ·D(x))`; decline/demote when `D(x) > θ` |
| OOS probe battery (ceiling law as gate) | **Partial** — "Report the Floor" (conformal-naive floor) exists; no OOS axis | **YES — methodology**: add `cov_in − cov_out` axis; flat-OOS null for capability claims |
| GRPO pool re-banding (1-15/16 solves) | **No analog** (verified: zero re-banding code in riir-train; `group_size: 16` already ships — exact alignment) | Recipe (riir-train Issue 469) — advantage collapse is a known GRPO failure the banding prevents |
| Curation cascade (cheap→expensive classifier ordering) | **No analog** (riir-data verified zero curation code) | Recipe — methodology for future game-log corpora |
| AoA developmental tiering | **No analog** | Partial — vendors an external lexicon; low consumer pull today |
| Acquisition-order DAG over freeze/thaw snapshots | **Partial** (chained quests + DemonstratedSkill ship fixed ranks; measurement doesn't ship) | Fusion idea — consumer pull weak (pets learn by fixed rank gates) |
| Capability-ceiling law | **Partial** (Bench 465/467/571 measured it; never encoded as a rule) | **YES — planning gate** (riir-train Issue 469 T2) |
| Capacity-competition router sizing | **No analog** | Weak — qualitative law only; constants must be self-measured |

## Distillation

### 1. The law as a planning gate (highest ROI, zero GPU)

Every proposed training task classifies as:
- **In-scope amplification** (base ~solves it; we amplify consistency/format/latency) → train (Issue 717 proves this wins)
- **Boundary expansion** (base can't do it; SFT/GRPO/ICL won't create it per the law + our 0/60) → route to base-model selection at the router, or a modelless substrate, **never GPU-hours**

Encoded in riir-train Issue 469 T2. Optional 5-10 GPU-h negative control: SFT a 0.5B on out-of-scope data, show held-out out-of-scope eval flat — our own micro-LittleLearner.

### 2. Scope-gated epistemics (fusion: paper × Issue 030 × EvidenceTier)

The Issue 030 lesson was "a syntactic check (`parses`) is not a relevance check". LittleLearner generalizes it one step further: **a relevance check is not a scope check.** An LLM/engram/drafter can be perfectly relevant to an input and still be confidently wrong, because the input is outside the distribution that shaped it. The failure signature (coherent projection onto familiar patterns) is invisible to relevance gates — it *is* relevant-looking output.

The gate is two-dimensional: **rule-coverage × input-scope**, both closed-form:
```
score(w)  = log2((c_B(w)+α)/(N_B+αV)) − log2((c_I(w)+α)/(N_I+αV))
D(x)      = Σ_w tf_x(w) · score(w)          // sparse GEMV vs precomputed direction
ĉ         = c · sigmoid(−κ·D(x))            // epistemic haircut
D(x) > θ  ⇒ decline / EvidenceTier demotion // decline is a CORRECT answer off-distribution
```
Consumers: riir-clippy L4 reachability (2D gate), riir-ai engram gates, UQ primitives' OOS axis. Tracked in katgpt-rs Issue 674 as a POC — **fusion idea, novelty TBD** (the math is textbook NB; the gate placement is the novel-for-us part; consumer pull is moderate, not urgent).

### 3. GRPO re-banding (riir-train Issue 469 T1)

Re-band the training pool between GRPO segments to problems the policy solves **1-15 out of 16** — mastered (16/16) and impossible (0/16) items produce zero group-normalized advantage (pure wasted compute). Our shipped `group_size: 16` aligns exactly. ~0.5 GPU-h per re-band segment; 30-60 GPU-h for the A/B. Addresses advantage collapse our own `GrpoMetrics.mean_advantage` can instrument.

### 4. OOS probe battery as a Report-the-Floor extension

UQ-bearing primitives gain a second mandatory axis: report `cov_in − cov_out` (or pass-rate delta) on a paired in/out-of-distribution probe set. Honest primitives widen intervals or decline OOS; dishonest ones project confidently — the paper's exact failure signature. Corollary for our hot-swap system: `scope(overlay) ⊆ scope(base)` — deterministic LoRA overlays are *more* bounded than SFT+GRPO, so a regression test asserting overlays add zero OOS lift catches accidental scope-claims. Rules change adopted via this note + Issue 674; enforcement at each affected primitive's next re-gate (grandfathered, same adoption path as the original floor rule).

## What does NOT transfer (stated plainly)

- 5B-from-scratch, 88B tokens, 8×B200×100h (~4,000-8,000 4090-equivalent-hours) — two-to-three orders beyond our practice
- 8-way Muon optimizer sharding (single 4090; sharding is where the Issue 679 shared-uniform bug class lives)
- MXFP8 block-scaled compute (B200-specific; Q4/Q8 QLoRA is our shipped analog)
- Custom tokenizer (no from-scratch consumer today; LoRA keeps tokenizers fixed — honest defer)
- The $46M absolute economics (local judges are free at the margin; wall-clock is our binding constraint)

## Prior art (searched, not novel-claimed)

Knowledge-boundary surveys (Li et al. 2025; Wen et al. 2024 — cited by the paper), BabyLM (quantity restriction, not conceptual scope), Talkie (temporal cutoff, not developmental), CurricuLLM (curriculum *generation*, not exposure control), MATH-B (evaluation-side control, not training-side). The contrastive log-odds document score is textbook Naive Bayes — no novelty claimed for the math. The paper's novelty (the sandbox) is not ours to claim; our extractions are placement-fusions.

## Routing

| Artifact | Where |
|---|---|
| This note (primary distillation) | katgpt-rs (public — the ceiling law validates the public engine's modelless thesis) |
| Scope-gate POC (contrastive table + NB doc score + haircut + OOS axis) | katgpt-rs Issue 674 → primitive would land in `katgpt-core` (generic math, leaf-clean) |
| GRPO re-banding + in-scope-amplification gate + cooloff mixture | riir-train Issue 469 |
| Curation cascade | Recorded in Issue 469 as a methodology reference for future game-log corpus builds (no active blocker) |

## References

- Paper: arXiv:2608.13545 (LittleLearner / LittleCurriculum)
- Our corroboration: riir-clippy Bench 465/467 (L4 0/60), Bench 571 (Gemma2 directions GOAT FAIL), katgpt-rs Issue 717 (in-scope +17.4%)
- Adjacent in-stack: riir-clippy Issue 030 (relevance gate — "syntactic ≠ relevance"; this note adds "relevance ≠ scope"), EvidenceTier (riir-clippy Issue 021), Report-the-Floor (katgpt-rs Issue 010, Plan 340), engram relevance gate (riir-ai Research 147 + katgpt-rs Issue 656)
