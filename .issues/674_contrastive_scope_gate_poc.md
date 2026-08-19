# Issue 674 — Contrastive scope-gate POC (two-corpus log-odds + epistemic haircut)

**Source:** Research 493 (arXiv:2608.13545 LittleLearner) — fusion idea, novelty TBD (the math is textbook Naive Bayes; the gate placement — input-distribution scope as a second gate axis — is the novel-for-us part).
**Classification:** POC / proof task. Consumer pull is moderate (L4 2D gate, engram gates, UQ OOS axis) — not urgent; this issue tracks the falsifiable evaluation, not a commitment to promote.

## Problem

LittleLearner's out-of-scope failure signature: models do not express uncertainty off-distribution — they emit *coherent but incorrect* projections onto familiar patterns (confidence miscalibrated exactly where it matters). Our shipped gates cannot catch this class:

| Shipped gate | Signal consumed | Blind spot |
|---|---|---|
| Identity guard (Issue 010) | identifier overlap fix↔bug | wrong-but-overlapping fixes pass |
| Issue 030 relevance gate | rule ∈ corpus coverage | in-corpus rule, out-of-corpus INPUT passes |
| EvidenceTier (Issue 021) | history (3-strike) | first-offense OOS passes |
| Engram gate | query-conditional relevance σ(dot(q,k)/τ) | relevant-looking OOS passes |

The missing axis: **input-distribution membership** — is the input inside the distribution that shaped the responder? Issue 030's lesson generalizes: *a relevance check is not a scope check.*

## Proposed primitive (modelless, closed-form)

1. `ContrastiveScoreTable` — two streaming count passes (in-corpus / out-corpus) → `score(w) = log2((c_B+α)/(N_B+αV)) − log2((c_I+α)/(N_I+αV))`, additive smoothing, papaya lock-free reads, BLAKE3-committed + freeze/thaw-able. Lands in `katgpt-core` (generic math, leaf-clean, no game semantics).
2. Document scope score — `D(x) = dot(count_vec_x, log_ratio_vec)` — a sparse GEMV (Naive Bayes log-LLR); accelerates on the existing ternary SIMD matvec substrate.
3. Epistemic haircut — `ĉ = c · sigmoid(−κ·D(x))`; `D(x) > θ` ⇒ decline / EvidenceTier demotion. Decline is a *correct* answer off-distribution (the L4 contract already treats declining as safe).
4. OOS probe axis — paired in/out fixture battery; report `cov_in − cov_out` (Report-the-Floor extension).

## Tasks

- [ ] T1: `ContrastiveScoreTable` in katgpt-core behind feature flag `contrastive_scope` (opt-in) + G1 toy-corpus parity test incl. zero-count smoothing edges + α→0 limit
- [ ] T2: document scope score `D(x)` — G1 bit-identical vs scalar loop (fixed-order accumulation), G2 µs/doc at 10⁴ tokens, G4 zero-alloc caller scratch
- [ ] T3: haircut gate `ĉ = c·sigmoid(−κ·D(x))` + decline wiring — G3 **in-distribution bit-identical** (haircut ≈ 1 in-scope — the load-bearing no-regression half); authored OOS fixtures must be discounted or declined
- [ ] T4: OOS probe battery — known-positive control (deliberately scope-restricted fixture corpus shows flat OOS; a seeded leak is caught)
- [ ] T5: POC verdict — if G-gates pass AND a consumer adopts (riir-clippy L4 2D gate or riir-ai engram), promote per GOAT discipline; else record negative result and close

## Honest caveats

- κ/θ constants must be pinned from OUR benches (paper's numbers are regime-specific, not portable)
- The two-corpus split needs an out-corpus: for riir-clippy the natural pair is (healer corpus, adversarial/clean spans — the Issue 017 eval fixture set already has both); for riir-ai, (engram shard corpus, OOD zone embeddings)
- If no consumer adopts by T5, close as negative — do not promote a gate nothing consumes
