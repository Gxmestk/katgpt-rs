# Bench 684 — Prover-Selection GOAT: D+Al vs Strength (Issue 692 T5)

**Verdict: PASS — ranking provers by Theorem 3.1's complementarity bound γ·(D+Al) instead of by strength picks the prover that actually delivers a wired gain, on a controlled PAV harness, at every paper α (16 seeds). `prover_selection` promoted to DEFAULT-ON in katgpt-core (commit `1b65662f`).**

- **Source:** arXiv:2410.08146 (Setlur et al., "Rewarding Progress") via [Research 509](../.research/509_Rewarding_Progress_PAV_Prover_Advantage.md) §5's defend-wrong obligation
- **Date:** 2026-08-27 (4090 box, pure CPU — seeded arithmetic, zero GPU)
- **Harness:** `benches/bench_684_prover_selection_goat.rs` (root feature `prover_selection`, `harness = false`)
- **Primitives under test:** `katgpt_core::prover_selection::{distinguishability, alignment, theorem_bound, selection_gate}` — T1's estimators consumed on logged Bernoulli means, exactly as shipped

## The three arms (Research 509 §5's exact shape)

| Arm | Selection surface | Wiring |
|---|---|---|
| **A0** frozen baseline / shipped analog | none — retention by raw Q̂^π alone | the strength-only selector the stack ships everywhere (dd_tree `BestQ`, drafter mean-acceptance, Elo) |
| **A1** strength arm | argmax mean logged success | wired: r_eff = Q̂^π + α·A^μ |
| **A2** paper arm | argmax `theorem_bound(D, Al, γ)` from the SAME logs | wired identically at equal cost |

## Controlled harness

- Ground truth θ(s,a) uniform iid, 64 states × 8 actions; base log Q̂^π = mean of n_mc=16 Bernoulli(θ) draws.
- Prover logs (independent RNG streams, same (s,a) support):
  - `strong_flat` p=0.95 const — the strength trap: top strength, A^μ ≈ 0 everywhere (the paper's "too strong ⇒ no distinguishability").
  - `peer_independent` p=θ — equal-competence peer, fresh MC noise.
  - `intermediate_ranked` p=0.30+0.50·within-state-θ-rank — weaker overall (~0.55), complementary profile.
  - `anti_aligned` p mirrored (0.80 − 0.50·rank).
- End task: retain top-32 of the 512 (s,a) pairs by arm score; quality = mean true θ of the retained set — **cross-state** retention, the direction where per-state centering is NOT rank-invariant (Research 509 §2.2).

## Results (aggregate over 16 seeds; per-seed tables in the bench log)

Prover stats (seed 42, representative — margins are stable ±10% across seeds):

| prover | strength | D | Al | bound | gate |
|---|---|---|---|---|---|
| strong_flat | **0.9479** | 0.0029 | 0.0009 | 0.0037 | 0.5009 |
| peer_independent | 0.4949 | **0.0812** | **0.0711** | **0.1523** | 0.5380 |
| intermediate_ranked | 0.5574 | 0.0366 | 0.0386 | 0.0752 | 0.5188 |
| anti_aligned | 0.5527 | 0.0384 | **−0.0420** | **−0.0037** | 0.4991 |

Retained-beam mean θ (32 of 512 pairs):

| α | A0 baseline | A1 strength→strong_flat | A2 bound→peer_independent | A2>A1 cells | A2>A0 cells |
|---|---|---|---|---|---|
| 0.2 | 0.9381 | 0.9397 | **0.9445** | 12/16 | 12/16 |
| 0.4 | 0.9381 | 0.9384 | **0.9444** | 13/16 | 12/16 |
| 0.6 | 0.9381 | 0.9352 | **0.9425** | 14/16 | 10/16 |

## Gates

- **G1 determinism** — bit-identical re-run per seed. PASS.
- **G2 the inversion is real** — strength-pick == `strong_flat` at every seed AND the bound ranks it last (bound(strong) < 0.25·bound(pick), measured ~40× below). The pre-gate correctly flags the strength winner as ≈no-gain. PASS per-seed.
- **G3 the headline** — mean quality(A2) > mean quality(A1) at every α (+0.0048/+0.0060/+0.0073), win-rate ≥ 75% of cells. PASS.
- **G4 no-harm** — mean quality(A2) > mean quality(A0) at α ∈ {0.4, 0.6} (0.9444/0.9425 vs 0.9381). PASS.
- **G5 direction sensitivity** — bound(anti) < bound(peer) per-seed; measured stronger: anti's bound is **negative** (Al ≈ −D), `selection_gate` < 0.5 — the pre-gate REJECTS anti-aligned provers outright. PASS.

## Honest findings

1. **The α=0.2 noise floor.** Per-(seed × α) cells carry 32-slot selection quantization noise (~±0.003); at α=0.2 a peer-strength prover's n_mc=16 MC noise is not paid back cell-wise (A1 also ≤ A0 there on most seeds) — G4 is gated at α ∈ {0.4, 0.6} and the α=0.2 row is reported. The aggregate mean still clears at α=0.2 (0.9445 > 0.9381). Practical pre-gate lesson: wire a prover only when the tilt weight clears the prover's own log-noise floor.
2. **The strength pick is a wash, not a disaster.** Wiring `strong_flat` is neutral at α ≤ 0.4 (mean 0.9397/0.9384 ≈ baseline) and harmful at α=0.6 (0.9352) — a flat prover's tilt is pure MC noise; the damage scales with α. The paper's prediction (no gain from an indistinguishable prover) reproduces.
3. **Scope.** A controlled harness validates the MECHANISM — the bound's ranking predicts wired gain and strength's does not — not real-world drafter superiority. Margins (+0.005–0.007 mean θ on a top-6% retention task) are the controlled-task scale, not the paper's >8% trained-prover accuracy (their provers are amortized MC, far less noisy than an n_mc=16 log). Real-log head-to-heads (drafter ranking) are riir-train / riir-poc territory per Research 509's routing.
4. **The showcase holds as the paper states it**: the bound-pick (`peer_independent`, strength 0.49) is *weaker* than the strength-pick (`strong_flat`, 0.95) yet delivers the gain — complementarity, not ceiling (Prop F.1).

## Promotion (T1 → DEFAULT-ON)

Gate passed modellessly (pure seeded arithmetic, no training) → per Issue 692's acceptance rule and the house GOAT discipline (G1 exhaustive unit-gated; G2 zero-cost-unless-invoked pure f32; G4 allocation-free by construction — no allocation sites in the kernels; G3 katgpt-core default lib suite **1978/0/7 green** post-promotion, the module's 27 tests now default-included; clippy 0 on all touched files):

- `crates/katgpt-core/src/lib.rs` — module ungated (the `rating` precedent).
- `crates/katgpt-core/Cargo.toml` — `prover_selection = []` stays as an inert alias, comment records DEFAULT-ON + the gate.
- Root `Cargo.toml` — `prover_selection = ["katgpt-core/prover_selection"]` forward + the `[[bench]]` entry.

Commit: `1b65662f` (bench + promotion, 2026-08-27).

**Companion verdict — Issue 692 T4 (dd_tree `BestAdvantage`) REFUTED BY MECHANISM, no code shipped:** scoring rollout i by `Q_i − mean_j Q_j` cannot change selection — `mean_j Q_j` is rollout-independent, so argmax is identical to `BestQ` at every seed (the same rank-invariance Research 509 §2.2 used to reject the QGF and riir-clippy consumers: the K rollouts of one `best_of_k_rollouts` call form a single within-state pool sharing the same marginals; a per-depth-centered variant also sums to a constant shift since all rollouts share the depth count). Fusion D corrected in Research 509; the enum arm is NOT added (no-op API surface, the "no vocabulary with zero consumers" rule).
