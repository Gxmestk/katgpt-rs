# Issue 134 — Complexity-Class-Gated Mode Router Fusion

**Filed:** 2026-07-12
**Priority:** P3 (track-only — blocked on a missing complexity-class classifier)
**Related:** `.research/411_CoT_vs_Latent_Thought_Formal_Comparison.md`, `.research/241_swir.md` (Plan 275), `.research/218_breakeven_complexity_router.md` (Plan 250), `.research/344_implicit_fp_rnn.md`, Plan 318 (k_selector), Plan 251 (DEC operators), Plan 281 (BoMSampler), `.research/367_quasimotto.md`
**Origin:** Research 411 §2.6 (fusion candidate) + §3 (routing table) — deferred T7. **Issue 135 (FPRAS routing criterion) consolidated into this issue 2026-07-25** — both are blocked on the same open research problem (a runtime complexity-class / self-reducibility detector); maintaining two parallel P3-track-only issues was noise without signal.

## Context

Research 411 distilled [arXiv:2509.25239](https://arxiv.org/abs/2509.25239) (Xu & Sato, ICML 2026) — a
theoretical paper proving two formal separations between chain-of-thought (CoT) and
latent thought:

1. **Latent thought exactly captures TC^k** with `log^k n` iterations (Theorem 3.12);
   CoT with the same steps is bounded by `TC^{k-1}` (Lemma 3.13). Latent wins on
   **parallelizable** problems.
2. **CoT admits FPRAS** for self-reducible `#P` counting problems (Theorems 4.3–4.5)
   that deterministic latent thought provably cannot match. CoT wins on **approximate
   counting**.

The paper ships no new mechanism — it proves theorems about paradigms we already ship
under different vocabulary (SwiR mode switching, `LoopMode::WeightShared`,
`LatentThoughtKernel`, HLA recurrent belief, DEC operators). Research 411 verdict was
**Gain** (Q2 fails: no new capability class; the value is the formal foundation).

## The Fusion Candidate

Fuse five existing shipped components into a single router that classifies the problem's
complexity class and routes to the provably-correct paradigm:

| Component | Source | Role in fusion |
|---|---|---|
| Mode switch controller | SwiR (Research 241, Plan 275) | Switches between latent and explicit/stochastic modes |
| Cost-amortization signal | Breakeven Bandit (Research 218, Plan 250) | Routes by compute tier economics |
| Rank-k (= TC^K) selection | k_selector (Plan 318) | Chooses the complexity class of latent iteration |
| DAG depth/size substrate | DEC operators (Plan 251) | The cell complex on which depth-bounded iteration runs |
| Stochastic sampling arm | BoMSampler (Plan 281) / QuasiMoTTo (Research 367) | The FPRAS-eligible arm for self-reducible #P problems |
| **Complexity-class classifier** | **NEW — not shipped** | Detects whether the problem is TC^K-parallelizable or #P-self-reducible |

## What this fusion produces that none alone can

Today, SwiR switches on **entropy trend** (a runtime confidence signal that says "am I
confident right now?"). The fusion would switch on **complexity class** (a structural
signal that says "is this problem parallelizable or does it need stochastic sampling?"):

- For parallelizable problems (TC^K) → route to latent iteration with K from `k_selector`.
- For self-reducible `#P` problems → route to BoM/QuasiMoTTo stochastic sampling (the
  FPRAS-eligible arm — formerly tracked as a separate Issue 135, now consolidated here).
- The Breakeven Bandit adds the cost dimension.
- The DEC substrate provides the DAG on which depth-bounded iteration runs.

## Blocker

The fusion requires a **complexity-class classifier** — a runtime detector that
determines whether the current problem is:
- **TC^K-parallelizable** (→ latent iteration), or
- **#P-self-reducible** (→ stochastic sampling).

The paper proves the *theorems* about these classes but provides **no classifier**.
Building one is a non-trivial research problem in itself — there is no known
general-purpose runtime complexity-class detector for inference-time queries. Candidate
approximations (entropy trend, problem-shape heuristics, learned classifiers) are all
heuristic and would need their own GOAT gate to prove they beat the existing SwiR
entropy signal.

## Novelty gate (honest, from Research 411 §2.6)

| Q | Criterion | Answer | Notes |
|---|---|---|---|
| Q1 | No prior art? | **Partial** | The *combination* (complexity-class-gated routing) has no prior art. The *components* are all shipped. The *theoretical result* (TC^k vs FPRAS) is from the paper. |
| Q2 | New class of behavior? | **NO** | A router that switches on complexity class rather than entropy is a *better switching signal*, not a new capability. SwiR already switches modes; the fusion adds a provably-correct criterion. Incremental. |
| Q3 | Product selling point? | **Partial** | "Our NPCs route reasoning by complexity class" is nice, but SwiR's "adaptive alternating" already covers the selling point. |
| Q4 | Force multiplier? | **YES** | Connects SwiR + Breakeven + k_selector + DEC + BoM/QuasiMoTTo = 5 systems across 2 pillars (P8 Reasoning Pack, P4 Frame-Sampling). |

**Q2 fails → not Super-GOAT. GOAT candidate only** — and only if the classifier ships
and benchmarks a provable gain over SwiR's entropy-trend switch.

## Tasks

- [-] **T1:** Design a complexity-class classifier (detect TC^K-parallelizable vs
  #P-self-reducible at runtime). DEFERRED — no known general-purpose detector; candidate
  heuristics (entropy trend, problem-shape, learned classifier) each need their own
  GOAT gate. This is the hard research problem the paper does not solve.
- [-] **T2:** If T1 produces a candidate classifier, fuse SwiR × Breakeven × k_selector
  × DEC × BoM into a router gated by it. DEFERRED — blocked on T1.
- [-] **T3:** GOAT-gate the fused router against SwiR-alone (G1 correctness, G2 perf,
  G3 no-regression, G4 alloc-free). DEFERRED — blocked on T2.
- [-] **T4:** If GOAT passes AND the gain is modelless → promote to default. DEFERRED —
  blocked on T3. Note: if the classifier itself requires training (riir-train), the
  gain is NOT modelless and the fusion stays opt-in per the promotion rule.

## Deferral Rationale

This is P3 track-only because:

1. The paper provides the theoretical foundation (the TC^k and FPRAS theorems) but
   **no classifier**. The fusion cannot ship without one.
2. A complexity-class classifier is itself a research problem with no obvious modelless
   solution. If it requires training, the whole fusion becomes a riir-train dependency
   and cannot be promoted to default-on (per the modelless-first mandate).
3. SwiR's entropy-trend switch is already DEFAULT-ON and covers the mode-switching
   capability. The fusion would be a *better switching signal*, not a new capability —
   the gain is theoretical optimality, not a measurable new behavior.
4. The retrospective recognition (Research 411 §2.6) that `k_selector`'s
   `K_OPTIONS = [1,2,4,8,16]` IS TC^K selection means the latent-iteration arm is
   already complexity-class-aware in practice, just without the formal label.

## Re-evaluation trigger

Revisit if any of the following lands:
- A paper ships a runtime complexity-class classifier (not just theorems).
- A modelless heuristic classifier is developed and GOAT-gated to beat SwiR's entropy
  signal on a representative benchmark suite.
- A FPRAS routing rule for self-reducible `#P` problems is independently shippable
  (BoMSampler / QuasiMoTTo are FPRAS-eligible arms; route to them when a #P
  self-reducibility detector fires). This was formerly tracked as Issue 135; it
  consolidates back here because the same detector gap blocks both — a #P
  self-reducibility detector is just the #P-half of the complexity-class classifier.
  See Research 411 §2.3 + T8 (deferred).

## TL;DR

Research 411 proves latent thought captures TC^k (parallelizable) and CoT captures
FPRAS (#P counting). The natural fusion — route by complexity class via SwiR ×
Breakeven × k_selector × DEC × BoM — is a GOAT candidate, but is blocked on a
complexity-class classifier the paper does not provide. Track only; revisit if a
modelless classifier lands. (Issue 135 — the FPRAS arm — consolidated here
2026-07-25: same detector gap blocked both.)
