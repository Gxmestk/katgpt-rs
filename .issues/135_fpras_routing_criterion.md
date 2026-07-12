# Issue 135 — FPRAS Routing Criterion for Self-Reducible #P Problems

**Filed:** 2026-07-12
**Priority:** P3 (track-only — blocked on a missing self-reducibility detector)
**Related:** `.research/411_CoT_vs_Latent_Thought_Formal_Comparison.md`, Plan 281 (BoMSampler), `.research/367_quasimotto.md`, `.research/281_bomsampler.md`, Issue 134 (parent fusion)
**Origin:** Research 411 §2.3 (genuinely novel insight) + §3 (routing table) — deferred T8

## Context

Research 411 distilled [arXiv:2509.25239](https://arxiv.org/abs/2509.25239) (Xu & Sato,
ICML 2026), which proves (Theorems 4.3–4.5) that **CoT with stochastic decoding admits
FPRAS** (fully polynomial-time randomized approximation schemes) for self-reducible `#P`
counting problems — and that **no deterministic latent thought with polynomially many
iterations can match this**.

This is the **first formal separation in favor of CoT**, and it is **genuinely novel for
our corpus**: zero grep hits across all 5 repos for `TC^k|FPRAS|FPTAS|approximate
counting|self-reducib` (Research 411 §2.3). No prior note or shipped code frames
stochastic sampling as FPRAS-eligible for #P problems.

## The Insight

For self-reducible `#P` counting problems (SAT counting, DNF counting, graph colorings,
partition functions), **stochastic sampling is provably more powerful than deterministic
latent iteration**. The mechanism: CoT explicitly samples intermediate tokens, inducing
stochastic computation that emulates randomized algorithms (Monte Carlo, MCMC). Latent
thought performs only deterministic transformations in latent space.

Our shipped stochastic samplers:
- **`BoMSampler`** (Plan 281) — K-hypothesis single-pass belief sampling.
- **`QuasiMoTTo`** (Research 367) — quasi-Monte Carlo belief sampling.

These are FPRAS-eligible arms for self-reducible #P problems, but we do not currently
route to them on that basis. Today they are invoked by mode-switching heuristics
(SwiR's entropy trend) or explicit caller choice, not by a complexity-class criterion.

## The Routing Criterion

Add a routing rule: **if the problem is detected as self-reducible #P, route to
`BoMSampler` / `QuasiMoTTo` stochastic sampling instead of deterministic latent
iteration.**

This is the #P-detection half of the Issue 134 complexity-class classifier. Where Issue
134 is the broader fusion (TC^K → latent + #P → stochastic + cost + DAG substrate),
this issue tracks just the FPRAS routing criterion — the narrower, independently-shippable
rule that self-reducible #P problems should go to the stochastic arm.

## Blocker

The criterion requires a **self-reducibility detector** — a runtime classifier that
determines whether the current problem exhibits self-reducibility structure (the
recursive decomposition property that enables FPRAS via self-reduction + stochastic
sampling).

The paper proves the *theorem* (self-reducible #P → FPRAS via CoT) but provides **no
detector**. Detecting self-reducibility at inference time is non-trivial:
- Self-reducibility is a structural property of the counting relation, not a runtime
  signal.
- Candidate heuristics (problem-shape matching, entropy-based proxies, learned
  classifiers) are all approximate and would need their own GOAT gate.

## Relationship to Issue 134

This issue is a **sub-component of Issue 134** (the complexity-class-gated routing
fusion). Issue 134's classifier needs to detect both TC^K-parallelizable AND
#P-self-reducible; this issue tracks only the #P-self-reducible detection half.

**This issue can ship independently of Issue 134.** A self-reducibility detector that
routes #P problems to BoM/QuasiMoTTo does not require the full fusion (Breakeven cost
dimension, k_selector TC^K arm, DEC substrate). It is the smaller, more tractable
slice — and if it ships first and demonstrates value, it strengthens the case for the
full Issue 134 fusion.

## Novelty gate (honest)

| Q | Criterion | Answer | Notes |
|---|---|---|---|
| Q1 | No prior art? | **YES (for the criterion)** | Zero grep hits for FPRAS/self-reducibility across all 5 repos. The *samplers* exist (BoM, QuasiMoTTo); the *routing criterion* (route #P → stochastic because FPRAS) does not. |
| Q2 | New class of behavior? | **NO** | A routing rule is a better switching signal, not a new capability. BoM/QuasiMoTTo already sample stochastically; the criterion adds a provably-correct trigger. |
| Q3 | Product selling point? | **Partial** | "Our samplers are FPRAS-optimal for #P" is a nice line, but the samplers already ship. |
| Q4 | Force multiplier? | **YES** | Connects BoMSampler + QuasiMoTTo + SwiR (the router) = 3 systems. |

**Q2 fails → GOAT candidate only** — and only if the detector ships and benchmarks a
provable gain over the current entropy-trend routing on self-reducible #P workloads.

## Tasks

- [-] **T1:** Design a self-reducibility detector (runtime classifier for #P
  self-reducible structure). DEFERRED — no known general-purpose detector; candidate
  heuristics each need their own GOAT gate. This is the hard research problem.
- [-] **T2:** If T1 produces a candidate detector, add a routing rule that sends
  detected #P problems to BoMSampler/QuasiMoTTo. DEFERRED — blocked on T1.
- [-] **T3:** GOAT-gate the routing rule against SwiR-alone on a #P workload
  (e.g., DNF counting, SAT counting proxy). G1: does stochastic sampling beat
  deterministic latent on accuracy/coverage? G2: perf. G3: no-regression on non-#P
  workloads. DEFERRED — blocked on T2.
- [-] **T4:** If GOAT passes AND the gain is modelless → promote to default. DEFERRED
  — blocked on T3. Note: if the detector requires training (riir-train), the gain is
  NOT modelless and the rule stays opt-in.

## Deferral Rationale

This is P3 track-only because:

1. The paper provides the FPRAS theorem but **no self-reducibility detector**. The
   routing rule cannot ship without one.
2. A self-reducibility detector is itself a research problem. If it requires training,
   the whole criterion becomes a riir-train dependency and cannot be promoted to
   default-on (per the modelless-first mandate).
3. The samplers (BoM, QuasiMoTTo) already ship and are already invoked by SwiR's
   entropy-trend switch. The criterion would be a *better trigger*, not a new
   capability — the gain is theoretical optimality for #P, not a measurable new
   behavior on general workloads.
4. The FPRAS insight's value today is **prophylactic**: it prevents a future agent from
   routing self-reducible #P problems to deterministic latent iteration under the
   mistaken belief that "latent thought is always more efficient" (Research 411 §4
   failure mode #3). The note documents this; the issue tracks the future implementation.

## Re-evaluation trigger

Revisit if any of the following lands:
- A paper ships a runtime self-reducibility detector (not just theorems).
- A modelless heuristic detector is developed (e.g., problem-shape matching against
  known self-reducible problem templates) and GOAT-gated to beat entropy-trend routing
  on a #P benchmark.
- A concrete #P workload emerges in the game/AI runtime where the FPRAS gap becomes
  measurable (i.e., where deterministic latent iteration demonstrably fails and
  stochastic sampling demonstrably succeeds).

## TL;DR

Research 411 proves stochastic sampling (CoT) admits FPRAS for self-reducible #P
problems that deterministic latent thought provably cannot solve. Our `BoMSampler` and
`QuasiMoTTo` are FPRAS-eligible arms, but we don't route to them on that basis. The
routing criterion is blocked on a self-reducibility detector the paper does not
provide. Track only; this is the #P-detection half of Issue 134's broader fusion and
can ship independently if a detector lands.
