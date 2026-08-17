# Issue 668: `effective_degree` open primitive — modelless function-space simplicity metric

**Source:** [Research 488](../.research/488_Effective_Degree_Polynomial_Simplicity.md) (arXiv:2605.29823, ICML 2026).
**Consumer:** riir-neuron-db Issue 602 (freeze-gate PoC) — the gating consumer; must land first.
**Repo:** katgpt-rs (open, generic math — no game semantics).

## Why

The stack has no function-space complexity metric (Research 488 §3: greps clean; KARC ships the Chebyshev basis for *forecasting*, `output_flatness` consumes the *weight spectrum*). ED — Σ|c_k|·k over Chebyshev coefficients fitted along data-pair interpolation paths — is a distribution-aware, reparameterization-invariant simplicity probe computable on any frozen function. Two of three components already ship (`karc::ChebyshevBasis`, ridge solve); the metric + node sampler are ~150 LOC.

## Tasks

- [x] T1: `crates/katgpt-core/src/effective_degree.rs` behind feature `effective_degree` (opt-in):
  - `EdConfig { resolution: usize, max_degree: usize, damping: f32, n_pairs: usize, seed: u64 }` + sensible preset `EdConfig::cheap()` (r=4, K=3) and `EdConfig::precise()` (r=15, K=7) mirroring the paper's efficiency/performance configs.
  - `randomized_cosine_nodes(r, seed, out: &mut [f32])` — stratified θ_i ~ U[(i−1)π/r, iπ/r], α = (1−cosθ)/2 (paper Eq. 8; deterministic given seed).
  - `effective_degree_along_path(outputs: &[f32], nodes: &[f32], cfg, scratch) -> EdResult { ed, ed_norm, coeffs }` — build design matrix T[i,k] = T_k(2α_i−1) reusing `karc::ChebyshevBasis`, solve damped normal equations `(TᵀT + εI)c = Tᵀy` (K+1 ≤ 8 → fixed-size array solve, no deps), return Σ|c_k|·k and normalized variant. Zero-alloc: caller scratch.
  - Generic driver `ed_over_pairs<F: Fn(&[f32], &mut [f32])>(f, pairs, cfg, scratch)` — samples α nodes, calls decode f at interpolated points, averages ED. The decode closure is consumer-supplied (a shard readout, an adapter, a policy) — katgpt-core stays game-agnostic.
- [x] T2: G1 order-preservation test (paper Appendix I protocol): synthetic polynomial targets of algebraic degree {1, 2, 5} (paper's Tasks 1–3) → ED strictly ordered, stable across Chebyshev/Legendre-style basis perturbation and ×2 output scaling for `ed_norm`. Also: constant function → ED ≈ 0; pure-affine → ED ≈ |c₁|.
- [x] T3: G2 latency bench: per-path cost at r=4/K=3 (target: sub-µs, it is a (K+1)² solve + r·(K+1) basis evals) and r=15/K=7; document pair-count scaling.
- [x] T4: G4 alloc-free steady state with reused scratch (CountingAllocator pattern).
- [x] T5: Docs — module doc citing Research 488 + the data-manifold caveat (endpoints must be real-data pairs, not random noise — paper C.1) + scale-dependence note (prefer `ed_norm` or normalized outputs).
- [x] T6: GOAT gate record in `.benchmarks/`; promotion decision (default-off until the Issue 602 PoC lands a consumer verdict — the no-default-consumer rule).

## Status — SHIPPED opt-in, CLOSED (2026-08-17)

All six tasks complete. GOAT record: [`.benchmarks/665_effective_degree_goat.md`](../.benchmarks/665_effective_degree_goat.md).

- Module: `crates/katgpt-core/src/effective_degree.rs`; feature `effective_degree = ["karc_forecaster"]` (opt-in, **not** in `default`).
- Gates: G1a–G1g + G2 + G3 + G4 **ALL PASS**. Highlights — order preservation
  monotone over the full degree 1..5 chain (stronger than the paper's 3-point
  protocol) and stable across 8 node seeds + a Legendre basis swap; `ed` scales
  ×2.0000 with outputs while `ed_norm` is invariant to 1e-4; **195.9 ns/path**
  at the cheap config; **0 allocs** in steady state; lib tests 1893 → 1905
  (+12, 0 regressions).
- Substrate consumed, not rebuilt: `karc::ChebyshevBasis<8>` +
  `linalg::{cholesky_f64, chol_solve_f64}`. Only the ED reduction and the
  randomized-cosine sampler are new.
- **Honest finding (Bench 665 §1):** `ed_norm` is a degree-weighted mean over
  *all* coefficients including `k = 0`, so a DC offset drags it below the
  algebraic degree (deg-5 fixture reads 1.15; with `c₀` zeroed, 1.63). Ordering
  is unaffected — but `ed_norm` is comparative-only, never an absolute degree
  read. An offset-free arm is one line via `ed_from_coeff_norms`.
- **Promotion: default-OFF and it stays there.** ED is **not** UQ-bearing (it
  emits a complexity scalar, not a distribution/interval/coverage claim), so the
  Issue 010 "Report the Floor" rule does not apply.

**Consumer verdict — riir-neuron-db Issue 602 CLOSED 2026-08-17: SCOPE-LIMITED,
no gate change.** ([Bench 484](../../riir-neuron-db/.benchmarks/484_ed_vs_flatness_freeze_gate_poc.md),
Research 488 §10.) Over 360 shard states, `ed_norm` out-correlates the incumbent
`output_flatness` **12.6×** pooled (0.598 vs 0.047, control 0.032, permutation
floor 0.042) and beat it 4/4 scenarios — but the **sign inverts between grains**
(pooled +0.598, within-regime all four negative), a Simpson reversal on 3
disjoint seed sets, so no threshold wires it as the proposed one-sided freeze
gate. **This issue's deferral trigger is the one that fired:** the primitive
stays as a diagnostic surface and is NOT deleted.

Two consequences folded back into the docs:
- **The Bench 665 DC finding changed the consumer's conclusion.** Carried as an
  `ed_ac` arm, zeroing `coeff_norms[0]` collapses the correlation 0.598 → +0.122
  (below flatness-plus-noise), so ED's power on that substrate lives in the DC
  term — event *alignment*, not shape complexity. The paper's actual thesis is
  therefore **not** confirmed at shard scale; the weaker claim (data-anchored
  function-space beats data-blind parameter-space) is.
- **New documented risk: grain-dependent sign** (module-doc caveat 4). Consumers
  must state their grain and verify the sign there; the paper measures only the
  across-model grain.

Externally confirmed here: `EdConfig::cheap()` is sufficient (0.598 vs precise's
0.623 for ~5× less work), 195.9 ns/path held on real substrate, and Theorem
3.1's path-averaging shows **0 ranking flips** across `n_pairs ∈ {1..64}` with
per-path spread falling **8.8×**. Surviving promotion axis is **cross-regime
triage** (the KARC regime-mismatch probe, Research 488 §4) — not freeze timing.

## Non-goals

- The differentiable regularizer (training-only → Research 488 §7, riir-train).
- PCA output compression (paper C.5: not the source of the gains; consumers can pre-reduce).

## Deferral triggers

If Issue 602's PoC refutes ED > flatness on the consolidation substrate, the primitive stays as a diagnostic-only surface (KARC regime-mismatch probe still consumes it) — do not delete; record the refutation.
