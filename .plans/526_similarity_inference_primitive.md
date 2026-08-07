# Plan 526: Similarity Inference Primitive — Open Modelless Math

**Date:** 2026-08-07
**Research:** [katgpt-rs/.research/471_Similarity_Inference_Embedded_Equilibrium.md](../.research/471_Similarity_Inference_Embedded_Equilibrium.md)
**Private guide:** [riir-ai/.research/335_Similarity_Inference_Emergent_Cooperation_Guide.md](../../riir-ai/.research/335_Similarity_Inference_Emergent_Cooperation_Guide.md)
**Source paper:** [arXiv:2608.03958](https://arxiv.org/abs/2608.03958) — Meulemans et al., Google Paradigms of Intelligence, 4 Aug 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/similarity_inference/` (new module) + Cargo feature `similarity_inference`
**Status:** Active — Phase 1 (skeleton)

---

## Goal

Ship a generic, modelless, leaf-clean open primitive that maintains a **similarity posterior** `ω ∈ [0,1]` between a focal decision-maker and each partner, updated from joint-action history, and a **cooperation gate** (`embedded_best_response`) that switches from competitive-best-response to cooperative-best-response when `ω` crosses a payoff-derived threshold. The primitive composes a Bayesian posterior update + a sigmoid cooperation threshold + a best-response comparator — zero game semantics, zero entity-kind assumptions, pure math.

This is the open half of the Super-GOAT pair (R471 + riir-ai R335). The closed-form math is from arXiv:2608.03958 §H + §I; the modelless composition is the invention.

**GOAT gate (G1–G7):**
- G1 closed-form reproduction (`ω_T` matches `α/(α+(1−α)·2^(−T))` to f32 epsilon).
- G2 emergent-cooperation PoC (N=64 entities, shared-shard pairs cooperate >80%, random-shard pairs <20% — the §3.6 defend-wrong PoC).
- G3 no-regression (workspace `cargo test` passes).
- G4 alloc-free steady state (0 allocs after construction).
- G5 indirect inference (zero-shot cooperation from third-party observation).
- G6 crowd-scale (1000 entities × 1000 ticks, <5ms/tick).
- G7 UQ floor comparison (`ω` beats `ω_floor = sigmoid(dot(history_summary, identity_direction))` on Brier score — the "Report the Floor" rule).

If G2 fails (cooperation does not emerge, or emerges for random pairs too), the Super-GOAT verdict is honestly revised per skill §3.6 — the architectural coverage stands, the quality claim is downgraded.

---

## Phase 1 — Skeleton + Closed-Form Math (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/similarity_inference/mod.rs` with module doc + feature gate `similarity_inference`.
- [ ] **T1.2** Define `JointActionHistory` trait — `push(self_a: &[f32], partner_a: &[f32], situation: &[f32])` + `window(t: usize)`.
- [ ] **T1.3** Define `SimilarityPosterior` struct — `{ prior_alpha: f32, log_w_independent: f32, last_omega: f32 }`. Implement `new(prior_alpha)`, `observe(...)`, `omega()`, `predictive_similarity(contemplated)`.
- [ ] **T1.4** Implement the closed-form update per paper §H.2: `ω_T = α / (α + (1−α)·W(æ_<T))` where `W(æ_<T) = Π_t P(a_i_t, a_j_t | situation_t)` under independent-policy marginal. Use log-space accumulator to avoid underflow.
- [ ] **T1.5** Define `embedded_best_response(omega, payoff_table, partner_predicted) -> u8` per paper §H.3. Compute `Q(C) − Q(D)` and return the argmax. The threshold is payoff-table-derived at runtime (canonical PD collapses to 0.5).
- [ ] **T1.6** Add `PayoffTable<2>` adapter (or reuse existing if katgpt-core ships one — grep first per substrate-first).
- [ ] **T1.7** Unit tests: closed-form `ω_T` matches analytical `α/(α+(1−α)·2^(−T))` to f32 epsilon for T=0..50, α=0.1. (G1.)
- [ ] **T1.8** Unit tests: `embedded_best_response` cooperates iff `ω > 0.5` for canonical PD. Defects otherwise.
- [ ] **T1.9** `cargo clippy -p katgpt-core --features similarity_inference` clean.
- [ ] **T1.10** Wire feature into `katgpt-core/Cargo.toml` `[features]` block (opt-in).

---

## Phase 2 — Emergent Cooperation PoC (G2 — the load-bearing gate)

### Tasks

- [ ] **T2.1** Create `katgpt-rs/crates/katgpt-core/src/similarity_inference/poc.rs` (gated `#[cfg(test)]`).
- [ ] **T2.2** Build a synthetic crowd: N=64 entities. Half are "shared-shard" pairs (same deterministic policy `π`); half are "random-shard" pairs (independent random policies). Each entity has a `SimilarityPosterior` per AOI-neighbor.
- [ ] **T2.3** Simulate T=50 info-gathering rounds (random 2×2 matrix games per round, perfect monitoring). Each entity observes its partner's action + the situation.
- [ ] **T2.4** At round T+1, terminal Prisoner's Dilemma. Each entity runs `embedded_best_response`. Record cooperation rate per pair type.
- [ ] **T2.5** **G2 assertion**: shared-shard pairs cooperate at >80%; random-shard pairs cooperate at <20%.
- [ ] **T2.6** If G2 FAILS: honestly record the numbers in `.benchmarks/526_similarity_inference_goat.md`, do NOT silently revise. Per skill §3.6, the verdict is downgraded: architectural coverage stands (the math is correct), quality claim is "unproven on this domain".

---

## Phase 3 — Indirect Inference (G5)

### Tasks

- [ ] **T3.1** Extend `SimilarityPosterior` with `observe_third_party(self_a, partner_a_in_same_situation, situation)` — updates `ω` from parallel third-party encounters without direct interaction.
- [ ] **T3.2** Build synthetic indirect-inference setup: 2 primary entities + 3 shared NPC entities. Primary entities never interact directly during info-gathering; each plays the 3 NPCs concurrently.
- [ ] **T3.3** After T=50 info-gathering rounds, primary entities meet for terminal PD.
- [ ] **T3.4** **G5 assertion**: shared-policy primary entities cooperate at >70%; random-policy primary entities cooperate at <25%.
- [ ] **T3.5** Test the staleness window: third-party encounters must be within K ticks to count as evidence.

---

## Phase 4 — Alloc-Free + Crowd-Scale (G4 + G6)

### Tasks

- [ ] **T4.1** Audit `SimilarityPosterior::observe` for allocations. The `log_w_independent` accumulator must be incremental (no replay of full history). Use a fixed-size scratch buffer if needed.
- [ ] **T4.2** **G4 assertion**: `observe` allocates 0 bytes after construction (use `CountingAllocator` pattern from Plan 011 G4 tests).
- [ ] **T4.3** Crowd-scale bench: 1000 entities × 20 AOI-neighbors each = 20K pairwise `ω` updates per tick. Measure wall-clock per tick.
- [ ] **T4.4** **G6 assertion**: <5ms total per tick for the 20K pairwise updates on Apple Silicon. Sub-µs per individual update.

---

## Phase 5 — UQ Floor Comparison (G7 — "Report the Floor" rule)

### Tasks

- [ ] **T5.1** Implement the conformal-naive floor: `omega_floor = sigmoid(dot(history_summary, identity_direction))` where `history_summary` is a fixed-length EMA of recent joint-action embeddings and `identity_direction` is a fixed random direction (deterministic via BLAKE3 seed per AGENTS.md).
- [ ] **T5.2** Build a held-out test set: 1000 (entity_pair, true_identity_label) tuples after T=50 info-gathering. `true_identity_label = 1` if shared-shard, else 0.
- [ ] **T5.3** Compute Brier score + log-loss for both `omega` (Bayesian posterior) and `omega_floor` (single-direction projection).
- [ ] **T5.4** **G7 assertion**: `omega` Brier score < `omega_floor` Brier score by ≥10% relative. If `omega` does NOT beat the floor, the primitive is not adding value over a single dot-product — the GOAT gate FAILS and the primitive stays opt-in with documented limitation.

---

## Phase 6 — Documentation + Promotion Decision

### Tasks

- [ ] **T6.1** Write `.benchmarks/526_similarity_inference_goat.md` with all G1–G7 results (pass or fail, honestly).
- [ ] **T6.2** Update `katgpt-rs/README.md` feature table with `similarity_inference` (opt-in initially).
- [ ] **T6.3** If ALL gates pass (G1–G7): promote `similarity_inference` to `default` in `katgpt-core/Cargo.toml`. Record promotion in the benchmark file.
- [ ] **T6.4** If G2 (emergent cooperation) FAILS: keep opt-in, document the failure in the benchmark, do NOT promote. The architectural coverage (closed-form math is correct) stands; the quality claim (emergent cooperation on this domain) is unproven.
- [ ] **T6.5** If G7 (UQ floor) FAILS: keep opt-in, document that the Bayesian posterior does not beat a single-direction projection on this domain. Consider whether a richer prior (beyond the paper's constructed one) would help — but that's a follow-up, not this plan.
- [ ] **T6.6** Cross-ref: add a one-line note to `katgpt-rs/.research/274` (CCE Moderator) pointing to this primitive as the *similarity-inferred* cousin.
- [ ] **T6.7** Commit on `develop` (per AGENTS.md global rule — commit at task completion).

---

## Non-Goals

- **Game-runtime wiring** (per-NPC `ω` sparse map, KG encounter log extension, crowd spectral clustering, CCE moderator endogenous switch) — these are riir-ai tasks, tracked in R335 §7. This plan ships ONLY the open math primitive.
- **Lean 4 formal verification** of the cooperation threshold theorem `T > log_2((1−α)/α)` — P3 follow-up, separate plan if pursued.
- **Cross-model partial-similarity validation** (Flash-Lite vs Flash analog) — P3 follow-up.
- **Pet-owner bond via `ω` accumulation** — riir-ai task (Plan 016/017 follow-up).
- **AI-vs-human asymmetry narrative validation** (G8 in R335) — riir-ai task.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| G2 fails — cooperation does not emerge on our toy domain | Honestly record numbers; downgrade quality claim per §3.6. The closed-form math (G1) still stands as a correct primitive. |
| G7 fails — Bayesian `ω` doesn't beat single-direction floor | The paper's constructed prior may be too simple for our action embeddings. Try a 2-direction floor (identity + anti-identity) as a stronger baseline. If still fails, keep opt-in with documented calibration limitation. |
| Indirect inference (G5) fails — zero-shot cooperation doesn't emerge | The staleness window K may be too tight. Sweep K. If still fails, document that indirect inference requires denser shared encounters than our toy setup provides. |
| Alloc-free (G4) blocked by history replay | The closed-form `W(æ_<T) = Π_t P(...)` is a product — accumulate in log-space incrementally. No replay needed. |
| Crowd-scale (G6) blows the 5ms budget | The pairwise `ω` update is O(D). For D=32, 20K updates = 640K ops = sub-ms on SIMD. If it blows, profile and SIMD-vectorize. |

---

## Source Paper Citation

Meulemans, Wołczyk, Weis, Nasser, Rocca, Kobayashi, Lajoie, Steger, Richards, Hutter, Manyika, Saurous, Sacramento, Agüera y Arcas. "A game theory for foundation models shows new paths to rational cooperation through similarity inference." [arXiv:2608.03958](https://arxiv.org/abs/2608.03958). 4 Aug 2026. §H (direct similarity analysis) + §I (indirect similarity analysis) + §F.2 (evidential information formalization) + §G (equilibrium convergence).
