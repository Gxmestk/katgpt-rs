# Issue 692 — Prover-selection statistics + cross-state advantage centering (open primitives from Research 509)

**Filed:** 2026-08-27
**Source:** [Research 509](../.research/509_Rewarding_Progress_PAV_Prover_Advantage.md) (arXiv:2410.08146, Setlur et al.)
**Kind:** open-primitive POC + optimization (modelless; no training)
**Repo:** katgpt-rs (host) — consumers: dd_tree (in-tree), riir-clippy (post-densification), future QGF consumers

## Problem

arXiv:2410.08146 proves per-step supervision should be **advantages under a complementary prover policy**, and that the right way to *select* a prover/critic/verifier is **not strength** but distinguishability + alignment:

- D(μ) = E_s Var_{a~π}[A^μ(s,a)]  (distinguishability)
- Al(μ) = E_s E_{a~π}[A^μ(s,a)·A^{π}(s,a)]  (alignment — a dot product, house style)
- Theorem 3.1: improvement ≳ γ·(D + Al) — a **predicted-gain pre-gate** computable offline from logged Bernoulli outcomes.

Our stack selects inference components by strength everywhere: drafters by mean acceptance, clippy rules by Elo (`katgpt_core::rating`), QGF oracles by head quality. The paper's result inverts this ranking — and predicts a *weaker but complementary* component can beat a stronger one. None of D/Al/cross-state-centering ships anywhere (grep-verified, Research 509 §2).

## Tasks

- [x] T1 `katgpt-core` open primitive: `prover_selection` module (beside `rating`/`bandit`) — `distinguishability(base_outcomes, prover_outcomes)`, `alignment(...)`, `theorem_bound(D, Al, γ)`; zero-alloc, f32, sigmoid-gated exposure per house rules. Exhaustive unit tests on synthetic Bernoulli grids. **DONE 2026-08-27 (`fcfb5f8c`): all three fns + `selection_gate` (the sigmoid-gated exposure — monotone in the bound, so ranking by gate ≡ ranking by bound) landed in the T2 module behind the same opt-in `prover_selection` feature. Estimator: per-(s,a) Bernoulli means, entry multiplicity = the π-weight, two-pass stable form, NaN propagates (caller screens). 13 new tests: exact hand-computed quarters; the paper's too-strong/too-weak/μ=π collapse pins; entry-multiplicity reweighting; Popoviciu D ≤ 0.25 + Cauchy–Schwarz |Al| ≤ √(D(μ)·D(π)) asserted over the exhaustive 121-combo grid. Feature-on lib suite 1972/0/7, clippy 0, default build untouched. No riir-clippy twin exists for the D/Al fns (T6 deferred the consumer side; the T2 twin-parity pin covers `first_pit` only).**
- [x] T2 `katgpt-core`: `first_pit(q_seq: &[f32], eps: f32) -> Option<usize>` changepoint kernel (first index where Q̂ < ε) + tests. (Consumer wiring — fix-verify blame ordering, kill-credit — is consumer-side, separate.) **DONE 2026-08-27: `prover_selection` module + the kernel behind the opt-in `prover_selection` feature; 8 tests incl. the twin-parity pin (`matches_the_riir_clippy_twin_bit_for_bit` — the swap-to-substrate contract for riir-clippy's pav_data twin). The riir-clippy twin swaps to this import on its next katgpt-core dep bump.**
- [x] T3 K\* law validation gate: exhaustive sweep asserting the closed form K\* = ln(ln(1−Q)/ln(1−V))/ln((1−V)/(1−Q)) matches the empirical argmax of A(K) = (1−V)^K − (1−Q)^K over a (Q,V) grid (skip degenerate Q≈V). Pure math — no PoC needed. **DONE 2026-08-27 (`983c6fb8`): `k_star` (the closed form) + `bok_advantage` (A(K), `powi`-based) landed beside the T1/T2 kernels behind the same opt-in `prover_selection` feature. The gate sweeps BOTH halves of the square (the law maximizes |A| — Research 509 §2.1 — symmetric under Q↔V while the gap is antisymmetric): an exhaustive 0.02-step full-square grid (2,256 off-diagonal-band pairs, K-sweep ≤ 1024 — ≥ 38× beyond every grid peak) + a 0.005-step near-diagonal refinement box [0.30, 0.60]² (3,540 pairs — the flattest-peak zone, sweep 256), skipping the degenerate |Q−V| bands. The assertion is EXACT, not tolerance: unimodality is structural (((1−V)/(1−Q))^K strictly monotone in K ⇒ dA/dK has a unique root), so argmax|A| ∈ {floor(K\*), ceil(K\*)} (floor clamped to the sweep's 1.. domain for K\* < 1). 6 tests: the two-grid gate, far-tail extremes (incl. the largest-K\* corner Q=.06/V=.02 → K\*≈26.9, peak 27), note anchors, dyadic bit-exactness + power identities, antisymmetry/symmetry/degenerate-diagonal, NaN totality. **ERRATUM found in Research 509 §2.1's anchors**: anchor 1 (Q=.5,V=.3 → K\*≈1.9747, peak 2, A(1)=.20/A(2)=.24/A(3)=.218) confirmed; anchor 2's "Q=.2, V=.1 → K\*≈3.4" is arithmetically wrong — the true closed form gives K\*≈6.371 with the empirical peak at K=6 (A(6)=.2693 > A(7)=.2686), exactly what floor/ceil predicts. The LAW holds everywhere on the grid; only the note's anchor prose was wrong (status line added to the note). Bit-exact pins: (1−Q)=(1−V)² ⇒ K\*=1.0 exactly in f32 (round(2x)=2·round(x) makes ln(0.25)=2·ln(0.5) bit-identical, so both terms land on ln(2.0)); m=3/2 identity ⇒ K\*=log₂(3/2) exact. En route: fixed the garbled lib.rs module comment block left by the T1 session. Feature-on lib suite 1978/0/7 (+6), clippy 0, fmt clean, default build untouched.**
- [ ] T4 `katgpt-speculative` dd_tree: add `WidthSelectionMode::BestAdvantage` (score rollout i by Q_i − mean_j Q_j; cross-rollout centering = the paper's cross-state fix). G8 gate: path diversity + downstream quality vs `BestQ` at equal K, ≥2 seeds.
- [ ] T5 GOAT bench for T1: head-to-head prover selection on a controlled harness (e.g., speculative drafter ranking or the Go-puzzle scorer): strength-ranked vs D+Al-ranked prover, frozen baseline, shipped analog — per the defend-wrong rule (Research 509 §5). Promote T1 to default only if the gate passes modellessly.
- [-] T6 riir-clippy lift-axis selection — DEFERRED by evidence: within-pool ranking is center-invariant (shared state), cross-rule pools are thin-evidence-starved (Issue 039: 98.5% unseen; Issue 026 doctrine). Revisit at Issue 039's densification trigger.

## Honest negatives (do not re-litigate without new evidence)

- QGF/`DualLeoOracle` is NOT a consumer: per-state tilt is argmax-invariant under centering (T9/T10 correctness checks pin it), and the civ critic axis is closed (riir-ai Research 322).
- The trained PAV (amortized MC) is riir-train territory — riir-train Plan 356.

## Acceptance

T1–T3 land behind a feature flag (`prover_selection`), tests green, clippy clean; T4 behind `WidthScaleConfig` arm with its gate; T5 verdict recorded in `.benchmarks/` with the three-arm table. Re-gate per stack-slot rules if promoted.
