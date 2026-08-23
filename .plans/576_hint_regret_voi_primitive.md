# Plan 576: Hint-Regret VoI Primitive (katgpt-core)

> **Source:** Research 496 (SPADE distillation) · Research 500 (EnvHarness wrap-refinement, 2026-08-22) · Guide 340 (riir-ai)
> **Date:** 2026-08-21 (folded 2026-08-23 — owner decision, perf+sec)
> **Repo:** katgpt-rs (public, MIT) — generic math, no game semantics
> **Feature:** `hint_regret` (opt-in; promotion only after G1–G4+G8)

## Goal

Ship the modelless hint-regret primitive SPADE (arXiv:2608.19197) distills to: a paired-rollout value-of-information estimator + sigmoid band-pass difficulty gate + three-regime triage + Beta-LCB frontier ordering. This is the open half of the Super-GOAT; the game-side guide is riir-ai `.research/340`.

## Design constraints

- Zero-alloc hot path; scratch buffers; fixed-size where bounded.
- Sigmoid, never softmax (band gate = product of two sigmoids).
- UQ-bearing (the estimator claims a gap estimate with a confidence bound) → **"Report the Floor" rule**: benchmark the estimator's decision quality against a naive floor (single-arm win-rate banding without the hint arm) — the paired estimator must beat the floor on triage accuracy at matched rollout budget, or the gate FAILS.
- Deterministic under seed (CRN pairing is load-bearing for G2 variance reduction).
- **Composition discipline (Research 500 fold, 2026-08-23):** the game-side difficulty LEVER is modifier composition (Setup/Rule/Link over the frozen quest table — Guide 340 §"Wrapper composition"), never a scalar knob. The primitives below COMPOSE with that wrapper algebra: the band gate consumes the per-composition Beta posterior mean as `w`; the bandit arms ARE modifier compositions. The wrapper algebra itself stays game-side (riir-ai) — promote to an open primitive only if PoC 677 vindicates generality (Research 500 P3).

## Phase 1 — Estimator core (`crates/katgpt-core/src/hint_regret.rs`)

- [ ] `HintRegretEstimator` struct: paired arms, CRN shared seeds, rolling K with Hoeffding `K(ε,δ) = ⌈(b−a)²/(2ε²)·ln(2/δ)⌉`, sequential stopping on CI half-width.
- [ ] `estimate() -> RegretEstimate { r_hat, ci_half_width, n_pairs, arm_means }` — pure arithmetic over recorded pairs, alloc-free.
- [ ] Analytic oracles as test fixtures: reveal-the-arm bandit (hint exposes μ*; `r = μ* − max_j μ̂_j`), hinted shortest path (β→∞ returns demo return bit-exactly).
- [ ] G1 gate: estimator within 2× Hoeffding bound at prescribed K across 10³ seeds; coverage of the δ-guarantee ≥ nominal.

## Phase 2 — Band gate + triage (`hint_regret/gate.rs`)

- [ ] `learnable_band_gate(w, w_lo, w_hi, kappa) -> f32` — `σ(κ(w−w_lo))·σ(κ(w_hi−w))`; monotone ↑↓, strict (0,1), peaks at band center. Property tests + one-line Lean extension of the shipped `sigmoid_bounded` family (`.proofs/KatgptProof`). **`w` = per-composition Beta posterior mean** (Research 500 row 11) — the gate ranks modifier compositions, and when `w` leaves the band the consumer swaps composition, not a scalar.
- [ ] `Regime` enum { Frontier, Mastered, Intractable } + `triage(r_hat, r_floor, unhinted_return, tau_r, tau_R) -> Regime` — 2-threshold 2D partition; property test: exactly one cell per input, boundaries pinned.
- [ ] Wilson CI on the learnable-share statistic (UQ honesty on the signature metric).

## Phase 3 — Frontier ordering + memory seam

- [ ] `beta_lcb_order(scores: &[(successes, fails)]) -> Vec<usize>` — ε-quantile of Beta(1+S, 1+F) descending (mirror of shipped `SelectionMode::BetaPosterior`; extract or re-implement leaf-clean — check katgpt-core first, DRY). **Expose the per-entry quantile (`beta_lcb`) so consumers can order BOTH directions** — descending for frontier ordering, ascending for weakness-slice diagnosis (Research 500 row 4: weakest slice = lowest LCB; Guide 340 diagnosis-first loop). One primitive, two consumers.
- [ ] `RegretMemoryEntry { content_hash, r_hat, ci, skill_tag_bits, last_seen_tick }` + salience `r_hat · σ(−λ·Δt)` (staleness decay — same family as `decay_confidence`).
- [ ] Oldest-first eviction at capacity; absorbing-eviction for Intractable-classified entries.

## Phase 4 — GOAT gate + bench

- [ ] `benches/bench_576_hint_regret_goat.rs`:
  - G1: oracle calibration (Phase 1) + triage partition properties.
  - G2: CRN variance ratio ≥ 2× vs independent-seed estimation (1000 reps); per-pair cost sub-µs alloc-free.
  - G3: default feature set untouched (gate compiles out); cgsp suite count-identical with `hint_regret` off.
  - G4: counting-allocator zero steady-state allocs over 10⁴ pairs.
  - G-Floor (Report the Floor): triage accuracy vs single-arm banding floor at matched budget — paired arm must win or FAIL.
  - G8 (simulated): learnable-share rises under regret-gated selection vs uniform on a synthetic curriculum (the 0.16→0.31 signature, modelless).
- [ ] `.benchmarks/576_hint_regret_goat.md` record.
- [ ] Promotion decision: default-on only if G-Floor and G8 pass modellessly; else stays opt-in with the verdict recorded.

## Phase 5 — Consumer wiring (defer-marked, owner-gated)

- [ ] [-] riir-ai quest-center frontier weighting **via Setup/Rule/Link modifier composition over the frozen `QuestTemplateRow` table** (Guide 340 §"Wrapper composition" — folded BEFORE implementation per Research 500 P1; verifier BLAKE3-pinned invariant across wraps is a P2 gate condition; after PoC Issue 677 verdict which now includes the wrap arm + transfer-back evaluation).
- [ ] [-] CGSP conflation fix evaluation: whether `(1−solve_rate)·guide_score` gains an intractable-separation term behind the feature (behavior change to shipped substrate — needs its own gate run).
- [ ] [-] edge_lora arena opponent selection (riir-train Plan 346 dependency).

## Non-goals

- The wrapper algebra itself (Setup/Rule/Link combinators, verifier-hash contract, EnvRigger-style diagnosis loop) — game-side (Guide 340 §"Wrapper composition", riir-ai); promote to an open primitive only if PoC 677 vindicates generality (Research 500 P3).
- Training a neural environment designer (SPADE proper) — that is riir-train Plan 346's POC-only item.
- Executable-environment generation (code-as-env) — our designer stays the parameterized quest-grammar drafter; the invisible-leash boundary is documented in Research 496 §Honest caveats.
