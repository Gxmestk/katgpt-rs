# Plan 580: Certified Frontier — Open Primitive

**Status:** OPEN — pending owner decision (not yet scheduled)
**Date:** 2026-08-27
**Research:** [katgpt-rs/.research/510_ActFlow_Certified_Frontier_Expansion.md](../.research/510_ActFlow_Certified_Frontier_Expansion.md)
**Source paper:** [arXiv:2606.08802](https://arxiv.org/abs/2606.08802) — De Santi et al., *Active Flow Expansion for Out-of-Distribution Discovery*, 2026 (safe-set expansion operator = SAFEOPT lineage, Sui et al. 2015/2018)
**Target:** `katgpt-rs/crates/katgpt-core/src/certified_frontier.rs` (new module) + Cargo feature `certified_frontier = []` (std-only, zero deps)
**Private consumer:** riir-ai `cgsp_runtime` (curiosity frontier) + riir-games `swarm/coverage_curiosity` (certified mask) — no game semantics here.

## Goal

Ship the modelless half of ActFlow's guarantee apparatus: a **certified frontier** primitive that grows a monotone set of provably-valid latent cells from (a) a query buffer of binary verifier outcomes, (b) a closed-form uncertainty model, and (c) a Lipschitz reachability budget — then acquires the next query target on the frontier's edge and says when to stop. Generic math, no game/chain/shard semantics: buffers are `&[[f32; D]]`, labels `&[bool]`, the "verifier" is caller-side.

The paper's entire theory (soundness, monotone growth, reachability coverage, halting) is proven about exactly these GD-free parts; the primitive makes them executable.

## What ships (7 functions + 2 metrics + 1 type)

1. `posterior_variance_linear(x, buffer_feats, lambda, scratch) -> f32` — Eq 10 exact: `σ²(x) = k(x,x) − k(x,X)(K+λI)⁻¹k(X,x)`, `k = dot`. Incremental rank-1 Cholesky of `(K+λI)⁻¹` per appended observation (never a re-solve). Companion `append_observation` maintains it.
2. `beta_mean_variance(valid: u32, invalid: u32) -> (f32, f32)` — the honest closed-form μ substitute (paper's kernel-logistic μ_t needs a convex solve; Beta-Bernoulli per-cell is exact-free and house-consistent). Alt: `ridge_mean(x, X, y, λ)` — `k(x,X)(K+λI)⁻¹y`.
3. `confidence_schedule(t, delta, lambda, kappa, b_rkhs) -> f32` — Eq 31/37: `β_t = 4·L_s·B + 2·L_s·√(2κ/λ·(γ_t + log(1/δ)))` with `L_s = 1/4` and `κ = 1/(s(B)(1−s(B)))` closed-form for the sigmoid. Monotone in t (pinned property).
4. `reachability_dilation(cells, hop_budget) -> Cells` — Eq 15 as a grid/bitmap morphological dilation: admit neighbor z iff `∃z' ∈ S: margin(z') ≥ L·d(z,z')` where `margin(z') = s(μ(z')) − β̂·σ(z') − h` and `L = L_s·L_g`. H-fold = iterate.
5. `expand_certified(frontier, buffer, cfg) -> AppendList` — Eq 32: the certified-set update (LCB + Lipschitz decay, monotone union). THE core op.
6. `acquire_frontier_target(frontier) -> Option<cell>` — safe uncertainty sampling (Eq 33 / approx Eq 14 with factor α): `argmax_{z∈S} σ(z)`. The where-to-look answer.
7. `should_advance(t_since_certified, beta, gamma, epsilon) -> bool` — the halting law: a certified hop is guaranteed once `σ ≤ ε/(2β̂)`; `T ≳ 8α²β̂²γ/ε²`. The stop-looking answer.
8. Metrics: `sphere_exclusion_coverage(samples, threshold)` (greedy, order-pinned — the coverage scoreboard) + `vendi_diversity(kernel_eigs)` (`exp(−Σλᵢ log λᵢ)` — the diversity scoreboard).
9. `CertifiedFrontier<const MAX_CELLS: usize>` — fixed-capacity cell set: latent `[f32; D]`, Beta counts `(u32, u32)`, certified bit (`covered_mask` storage pattern). Zero-alloc by construction.

## Fusion (why this and not just math helpers)

- **Grow-then-navigate**: `CertifiedFrontier` cells feed `build_safe_manifold_graph` (Plan 312) as the node source — the missing acquisition half of the VMG stack.
- **EVPI composition** (riir-ai side): straddling gate `LCB − L·δ < h ≤ UCB` prunes deep-inside + far-outside cells to zero queries — the when-to-look gate gets a certified frontier instead of a disc.
- **Prop 1 bounds** (`spherical_cap_bound`, `laurent_massart_radius`) ship beside the module as pure fns — the design law + the pre-registered G8 prediction (passive vs frontier-targeted separation factor `exp((m−1)cos²φ/2)`).
- **DEC resonance** (documented, not load-bearing): frontier = `∂S` via `exterior_derivative` on the cell cochain; expansion flow divergence via `codifferential`.

## Phase 0 — Self-contained PoC (no feature gate)

- [ ] **T0.1** `examples/certified_frontier_01_basic.rs` — self-contained (std only, LCG RNG): 2D checkerboard-validity world (the paper's own illustrative setup), seed set = one valid cell, verifier = ground-truth predicate queried ONLY through the buffer. Run 500 rounds of acquire→query→expand; print ASCII map of certified vs true-valid region + violation count.
- [ ] **T0.2** Verify the headline separation: passive random querying vs frontier acquisition on a *sparse-frontier* variant (valid corridor of opening angle φ) — expect the Prop-1-predicted exponential gap in coverage@budget.
- Exit: example reproduces certified growth with ZERO violations on the dense world + visible separation on the sparse world.

## Phase 1 — Module skeleton

- [ ] **T1.1** Feature `certified_frontier = []` in katgpt-core + root passthrough (no deps — pure std).
- [ ] **T1.2** `certified_frontier.rs` with module doc citing R510 + arXiv:2606.08802 + SAFEOPT lineage.
- [ ] **T1.3** Types: `CertifiedFrontier`, `FrontierConfig { lambda, delta, b_rkhs, h, lipschitz: f32, alpha }`, `FrontierScratch` (fixed-capacity Cholesky factor + kernel column).
- [ ] **T1.4** Fns 1–8 above + `#[inline]` hot paths. Export behind cfg in lib.rs.

## Phase 2 — Core correctness

- [ ] **T2.1** `posterior_variance_linear`: pinned against a dense-solve reference (nalgebra-free: hand-rolled small Cholesky, or compare rank-1 incremental vs batch solve at N=64 — must agree to 1e-6).
- [ ] **T2.2** `expand_certified` soundness property test (Lemma E.2 as a test): plant a known validity fn, adversarial query sequences (order-shuffled, corrupted labels calibrated by the model), assert **zero uncertified-valid→actually-invalid admissions** at the configured δ across ≥1000 seeds.
- [ ] **T2.3** Monotonicity property: certified set never shrinks across arbitrary query sequences.
- [ ] **T2.4** Confidence schedule: monotone in t; β_0 sanity; κ/L_s closed-form spot-checks.
- [ ] **T2.5** Halting law: once `should_advance` fires, one `reachability_dilation` hop admits no violations (the Lemma E.4/F.7 contract, executed).
- [ ] **T2.6** Sphere-exclusion: order-pinned determinism (fixed order → bit-identical cluster count); Vendi on planted eigenvalues.

## Phase 3 — GOAT gates

- [ ] **T3.1 G2 perf**: batch acquisition + expansion at crowd scale — 1000 frontier queries (one per NPC) with buffer N=256, D=8; target < 1 µs/query amortized (rank-1 updates, precomputed inverse); release-only gate.
- [ ] **T3.2 G4 alloc-free**: `FrontierScratch` capacity stable across 1000 expand/acquire cycles (tracking allocator).
- [ ] **T3.3 G3 no-regression**: feature-off build clean; default surface untouched until promotion.
- [ ] **T3.4 UQ floor** (Report-the-Floor adaptation — this primitive claims a coverage guarantee): bench certified-growth × violation-rate against the naive floor = **adjacency-only expansion** (certify any neighbor of a valid-labeled cell, no uncertainty model). The primitive must dominate the floor on the product metric (growth ⋅ (1 − violation rate)); if it cannot, demote to documented-negative.
- [ ] **T3.5 Bench doc**: `.benchmarks/580_certified_frontier_goat.md` with the floor table.

## Phase 4 — Fusion surface

- [ ] **T4.1** `CertifiedFrontier → SafeManifoldGraph` adapter fn (nodes from certified cells; edges via existing kNN + midpoint check): one example running grow-THEN-navigate end-to-end.
- [ ] **T4.2** Straddling-gate helper `query_is_decision_relevant(lcb, ucb, h, cell_diam) -> bool` (the EVPI-shape prune) — pure fn + unit tests.
- [ ] **T4.3** riir-poc four-arm gate (riir-ai side, follow-up issue there): certified-frontier vs curiosity-only vs passive vs never-look at equal perception budget; 16 seeds; Prop-1 separation pre-registered as the PASS prediction.

## Phase 5 — Promotion

- [ ] **T5.1** If G1–G4 + floor PASS → add to `default = [...]` (both Cargo.tomls) + README showcase row + `.docs/01_orientation/overview.md` Feature Flags row.
- [ ] **T5.2** If floor FAILS → keep opt-in, document the regime where adjacency-only wins (dense-frontier worlds), demote honestly in R510 footer.

## Risk register

| Risk | Mitigation |
|---|---|
| Beta mean substitute breaks calibration → false certifications | Soundness test (T2.2) runs against the SUBSTITUTE, not the paper's μ; floor gate (T3.4) catches it if uncertainty adds nothing over adjacency |
| Buffer-depth O(t²) blowup at long horizons | Cap buffer (ring) + document the information-gain plateau (γ ~ d_eff·log T → uncertainty is eliminable); halting law bounds t per cell anyway |
| 8-D fine, 64-D shards slow (curse of dim) | Scope: zone-grid cells + 8-D HLA latents; document d_eff caveat (R510 caveat 6) |
| Known-art operator (SAFEOPT) | Honest framing everywhere: we ship it as we ship bandits — operator known, fusion + domain novel |
| Two selection-mode negatives (Bench 035/042) suggest acquisition layers can lose | Those modified starved per-candidate pool scoring; this is input-space epistemic variance with 8K-observation substrate (healer) or ground-truth verifier (PoC) — and the floor gate is the honest kill switch |

## Cross-references

- [R510](../.research/510_ActFlow_Certified_Frontier_Expansion.md) — source distillation + Path 0 table + signal-diffs
- [Plan 312](312_viable_manifold_graph_primitive.md) — the navigation half (grow-then-navigate fusion)
- riir-ai Issue 738 EVPI gate (when-to-look), riir-games Issue 672 `covered_mask` (storage pattern)
- Sibling tracks: [riir-train Plan 357](../../riir-train/.plans/357_actflow_discrete_expansion.md) (training half), [riir-clippy Issue 048](../../riir-clippy/.issues/048_gp_acquisition_layer.md) (healer acquisition)
