# Research 520: On Connected Sublevel Sets in Deep Learning — No Bad Local Valleys, Dead Units, Align-then-Lerp

**Status:** GAIN — theory lens + 2 actionable items filed (riir-train Issue 494: product-space LoRA merge + run-start feature-rank probe). NOT Super-GOAT (Q1/Q3 fail on mode-connectivity / model-merging published prior art). One verdict per track: modelless track Gain (certificate is open-glue over shipped erank substrate — consumer gap only), model-based track Gain (riir-train Issue 494). **LANDING ANNOTATIONS (riir-train, 2026-08-31, commits `4ede5770`/`15caadde`/`c8565cd7`/`cbfb175c`):** (1) Phase 1's prescribed product-space merge was DEMOTED by its own gate — the rank-r SVD re-factorization re-canonicalizes A's row space and the sigmoid does not survive it (probe parity 0.77–0.87 vs 0.999+); align-then-average landed instead. GAUGE NARROWING: for sigmoid-gated edges the ONLY exact function gauge is the PERMUTATION group — general orthogonal rotations mix gate arguments pre-nonlinearity (`σ(Qᵀx) ≠ Qᵀσ(x)`) and sign flips break σ; this §'s orthogonal-align framing narrows accordingly. (2) T2.2 real-data gate: alignment retained (as-trained cross-seed members share a near-gauge, aligned/naive 1.0010; gauge-twin corroboration on a real-trained seed: naive parity 0.0–0.945 vs aligned ≥0.99999). (3) T4's prescribed warning `feature_rank < N` is a MATHEMATICAL TAUTOLOGY — mean-centering puts the all-ones vector in the centered matrix's left null space, so erank ≤ N−1 IDENTICALLY (pinned: 8 exact basis directions → erank exactly 7.0); the landed warning is the paper's actual structural condition `hidden_width < N`, with the refined `target_rank > model_rank` comparison separating converged floors 133× in the T4.3 sweep.

> **Source:** Quynh Nguyen, "On Connected Sublevel Sets in Deep Learning", ICML 2019, arXiv:1901.07417 (v2).
> **Date:** 2026-08-30
> **Workflow:** research skill — adversarial panel run (No-GD + Model-based advocates, one spawn round), §4 prior-art verified via paper refs + arXiv API, Path 0 inventory complete, §3.6 signal-diffs run on both "already ships" candidates.
> **Related:** riir-train Issue 494 (the actionable half); Plan 415 / bench_415_within_class_erank_goat (`effective_rank` substrate); riir-train `edge_lora_dist_guard` (Issue 743, `activation_population_erank`); riir-clippy Research 081 / Bench 013 (`SortKeyCache`); katgpt-rs `memory_soup_lora.rs`; Bench 558 (predecessor-init harm).

---

## TL;DR

Nguyen (ICML 2019) proves that for deep fully-connected nets with convex output loss and piecewise-linear (or strictly-monotone onto-ℝ, e.g. LeakyReLU) activations, **over-parameterization makes the loss geometry benign**: with one hidden layer of width `n_k ≥ N` (N = training-set size), the loss has **no bad local valleys** and every valley is unbounded; with `n₁ ≥ 2N`, **every sublevel set is connected** (unique valley, all global minima connected). The proofs are constructive and the constructions — not the inequalities — are the transferable cargo for this stack:

1. **Closed-form realization solve**: the first layer can drive the network output to ANY target matrix exactly (`W₁ = [X,𝟏]† σ⁻¹(...)`) — no gradient descent, pseudoinverse algebra.
2. **`rank(F_k) = N` is the load-bearing quantity**: the hidden-activation matrix's rank vs sample count is the realization-capacity certificate; rank < N is repairable via free bias directions.
3. **Align-then-lerp**: use loss-invariant paths (absorb one checkpoint's units into the other's kernel/dead directions) to make layers coincide, then linear interpolation of the last layer is convexity-bounded. Naive whole-net lerp has NO such guarantee.
4. **Dead units are alignment capacity**: kernel directions scale to infinity with output unchanged — the mechanism behind both unboundedness and checkpoint absorption.
5. **§7 honesty**: connected sublevel sets do NOT ensure GD succeeds (bad init stalls forever; saddles exist). A path existing ≠ optimization finding it.

**Verdict logic:** the theorem class is published prior art (the paper's own references: Venturi et al. 2018, Draxler 2018, Garipov 2018, Nguyen & Hein 2017/2018) and the adjacent interpolate-and-merge technique is a dense literature (LMC / Git Re-Basin / model soups — verified via arXiv API, see §4). Q1 novelty FAIL, Q3 selling point weak → not Super-GOAT. But two items are concretely actionable in our stack: **the consolidation merge in riir-train's `sleep_consolidation.rs` averages LoRA A/B factors separately (gauge-undefined — the exact hazard the align-then-lerp machinery forbids)**, and **no module in the stack compares activation rank vs sample count** (the paper's certificate is a NEW consumer over our shipped erank substrate). → Gain, filed as riir-train Issue 494.

## The theorems (compact)

| Theorem | Conditions | Conclusion |
|---|---|---|
| 3.2 | `rank(X)=N`, widths decreasing `n₁>…>n_L`, σ invertible (LeakyReLU) | Every sublevel set connected; every level-set component unbounded |
| 4.2 | one hidden layer `n_k ≥ N`, decreasing after | No bad local valleys; every local valley unbounded |
| 5.1 | `n₁ ≥ 2N`, decreasing after | Unique valley — all global minima connected |
| 6.1 | ReLU (no invertibility), ALL hidden layers `≥ N` (or `≥ 2N`) | Same conclusions |
| §7 | — | Connected sublevel sets ⇏ GD success (init stalls, saddles) |

**Activation-class honesty (mandatory caveat for every consumer):** the theorem class is PWL (ReLU, LeakyReLU) or strictly-monotone-onto-ℝ. Our stack's activations are sigmoid-family (strictly monotone but NOT onto ℝ — violates Assumption 2.1) and SiLU/GELU (smooth, not PWL). The three transformers (Bonsai/qwen/grammar) are outside the literal class. Also `N` = unique training samples — at our corpus sizes (thousands) `n₁ ≥ 2N` is unattainable. **The value is directional structure (alignment-before-interpolation, rank conditioning, dead-unit freedom), never the literal inequalities.**

## Vocabulary crosswalk (paper → stack)

| Paper term | Stack home (verified by grep) |
|---|---|
| `rank(F_k) = N` (activations full row rank) | `katgpt-core::data_probe::geometry::effective_rank` / `within_class_effective_rank` (Plan 415); riir-train `dist_guard::activation_population_erank` (Issue 743) — **no existing consumer does the vs-N comparison** |
| closed-form target solve `F_k(h(...))=A` | `katgpt-attn-match::value_fitter::fit_cv_least_squares` (Cholesky/Pinv, linear case = σ=id special case); `katgpt-core::poincare` (W_pinv adapter fits); `riir-engine::latent_functor::arithmetic::rank_k` (dual ridge operator fit) |
| align-then-lerp merge | `katgpt-core::memory_soup_lora.rs` (SoupCheckpoint, delta interpolation); `katgpt-kv::kv_share::merge_kv_weights` (naive mean — NO alignment, NO certificate); riir-train `sleep_consolidation.rs::consolidate` (averages A and B **separately** — the live hazard) |
| dead units / kernel directions | `katgpt-core::group_invariance_probe::null_space`; `gauge_invariant_demo.rs` (output-preserving LoRA rebalance A·Bᵀ) |
| gauge orbit (`W₁* + ker`, equivalent nets) | same group_invariance machinery; Bench 558's `predecessor_init` harm is the measured cautionary tale (raw weight copy ≠ loss-invariant path) |
| no bad valleys / valley structure | riir-train `degrpo::CollapseMonitor`, muon absorbing-fixed-point stall docs, `arm_c` early-stop-patience |

## Path 0 inventory (modelless components — No-GD advocate, coordinator-adjudicated)

| # | Component | Ships without GD? | Coverage in stack | Disposition |
|---|---|---|---|---|
| 1 | Closed-form layer-1 realization solve (Thm 3.2 h-map) | YES — deterministic | PARTIAL: `fit_cv_least_squares` / `poincare` / `rank_k` solve min‖XC−Y‖ **without σ⁻¹ inside the design** (linear readout = σ=id special case). Delta = activation-inverse composition for invertible activations. | Thin delta; recorded, not filed (auditable: existing solvers cover the linear case; our activations are sigmoid — outside the invertible-onto-ℝ class anyway) |
| 2 | **Realization-capacity certificate: rank(F_k) ≥ N** | YES — offline SVD/Jacobi rank | NO consumer does vs-N (checked: erank consumers judge population health/collapse, not realization capacity). | **FILED — riir-train Issue 494 T4** (run-start probe over existing `data_probe` substrate; no new katgpt-rs code needed) |
| 3 | Bias-direction rank-raising bound (Thm 4.2 step 1) | YES — count distinct preactivation rows | `module_rank_profile.rs` measures variance energy, no row-distinctness | Doc-level only (auditable: subsumed by #2's probe in practice) |
| 4 | Width ordering law `n₁>…>n_L` | YES — integer compare | n/a | Discard (auditable: literal inequalities unattainable at our scales; transformers outside class) |
| 5 | Valley-structure predictors `n_k ≥ N`, `n₁ ≥ 2N` | YES | n/a | Discard as literal check; retained as directional guidance in this note |
| 6 | **Align-then-lerp certified merge** | YES — greedy unit matching + last-layer lerp, bound verifiable by evaluating loss at a few t | `memory_soup_lora` + `kv_share::merge_kv_weights` do naive merge with NO alignment and NO certificate; `sleep_consolidation` has the gauge hazard | **FILED — riir-train Issue 494 T1–T3** |
| 7 | Dead-unit kernel-direction invariance paths | YES — nullspace + rescale | `group_invariance_probe::null_space` + gauge demo cover the mechanism | Covered (auditable: the LIVE consumer of this theory is #6's merge hazard, which is what gets fixed) |
| 8 | Layer-1 solution set = W₁* + ker (gauge orbit) | YES | `group_invariance_probe` commutant/gauge machinery | Covered |
| 9 | Unboundedness ⇒ huge-norm checkpoint = invariance orbit (offline audit) | YES | `data_probe` / `numeric_stability` could host | Idea only — no consumer demand; novelty TBD if a checkpoint-hygiene audit ever materializes |
| 10 | §7 caveat (connectivity ≠ GD success) | YES — doc + pinned test string | Missing from our stall/restart docs | **FILED — riir-train Issue 494 T5** (doc-truth pin) |

## Three-track panel — Model-based advocate (coordinator-adjudicated)

| # | Recipe | Disposition |
|---|---|---|
| 1 | **Merge adapters in ΔW = B·A product space, never factor space** — `sleep_consolidation.rs` L226–253 averages A and B separately; `(cB)(A/c) ≡ BA` and rank-axis permutations make factor-space averaging gauge-undefined | **FILED — Issue 494 T1** (the highest-ROI item; ~0 GPU-h) |
| 2 | **Gauge-align members to cluster seed before merging** (reuse `distill_select.rs` CCA as misalignment meter) | **FILED — Issue 494 T2** |
| 3 | **Same-run checkpoint soup licensed; cross-run NOT** (shared lineage ⇒ non-adapter layers bit-identical) | **FILED — Issue 494 T3** (flag-gated, GOAT vs best checkpoint, ≥3 seeds) |
| 4 | **Restart-on-plateau** — §7 is the theory behind muon's documented absorbing fixed point (`b_init_std = 1/sqrt(rank)` re-draw) | Doc-only for now (auditable: arm_c has patience flags; adding the re-init action is a behavior change needing its own bench — recorded as candidate in Issue 494 §Non-goals) |
| 5 | **Probe feature rank at fine-tune start: rank(F_k) vs N** | **FILED — Issue 494 T4** (same as Path 0 #2) |
| 6 | Dead/low-activation units are alignment budget — keep below `min_active_magnitude` instead of dropping pre-cluster | Folded into T1's design notes (auditable: same consolidation site, one change-set) |
| 7 | Early-width placement for small FC trainers (`n_k ≥ N` kills valleys) | Doc-only (auditable: applies to future small trainers, not any in-flight pipeline; also explains Bench 558's predecessor-init harm — comment pin in T5) |
| 8 | **Cross-game edge merging NOT licensed** — different losses, no interpolation guarantee | **FILED — Issue 494 T2** (eval gate on cross-game clusters) |

## §4 Published prior art (verified)

The paper's own references kill class novelty on the geometry: Venturi et al. 2018 (spurious valleys, 2-layer), Draxler et al. 2018 + Garipov et al. 2018 (empirical mode connectivity), Nguyen & Hein 2017/2018 (loss surface of deep/wide nets), Liang et al. 2018 (adding one neuron eliminates bad local minima), Sohl-Dickstein & Kawaguchi 2019.

arXiv API check (2026-08-30) on the adjacent merge/interpolate technique confirms a dense active landscape:
- 2210.06671 Wasserstein-barycenter model fusion + LMC
- 2406.16300 "Landscaping Linear Mode Connectivity" (barrier-height predictor, ICML 2024 HiLD)
- 2310.18769 LMC in sparse networks (NeurIPS 2023 UniReps)
- 2506.22712 Generalized LMC for Transformers (NeurIPS 2025 **oral** — four symmetry classes, width-heterogeneous alignment)
- 2405.14596 LMC in differentiable tree ensembles (ICLR 2025)

Plus the known Git Re-Basin (Ainsworth et al.) / model soups (Wortsman et al.) / task-arithmetic (Ilharco et al.) family. **Q1 novelty: FAIL at class level.** The only stack-local open claim — "rank(F_k)-vs-N realization-capacity certificate has no consumer in our stack" — was verified by the No-GD advocate's grep (all erank consumers judge population health) and is a CONSUMER gap over shipped substrate, not a published-artifact claim.

## §1.5 Novelty gate scoring

| Question | Score | Reason |
|---|---|---|
| Q1 No prior art | **FAIL** | Geometry + LMC/merging literature dense (above) |
| Q2 New behavior class | partial | A capacity certificate + certified merge are behaviors our stack lacks, but both are known technique classes externally |
| Q3 Product selling point | weak | "certified adapter soup" is an infra nicety, not a customer-facing capability |
| Q4 Force multiplier | yes | erank substrate, value_fitter, memory_soup, consolidation, dist_guard |

**All 4 YES? NO → not Super-GOAT. Not a candidate (banned escape hatch). Verdict: GAIN** — actionable improvements exist and are filed.

## Fusion (recorded even though unplanned)

paper (align-then-lerp + dead-unit absorption) × `memory_soup_lora` × `sleep_consolidation` clustering = **certified adapter fusion**: cluster members → gauge-align to seed → merge in product space → re-factorize rank-r → verify with a few-point loss/eval sample → freeze as one BLAKE3-committed edge. The §7 caveat and the activation-class mismatch must ride the consumer docs (the certificate says a bounded path EXISTS, not that optimization finds it, and our sigmoid gates are outside the literal theorem class — the construction is directional, verified empirically per-merge by the eval sample, which is exactly what T1's gate does).

Second fusion (unfiled, novelty TBD): `rank(F_k) vs N` × `numeric_stability` × checkpoint hygiene = "is this checkpoint sitting on an invariance orbit?" audit (huge norm, unchanged loss). No consumer demand today.

## GOAT gates (for Issue 494's implementation)

- **G1**: merged-edge function parity — product-space merge vs members on a fixed probe set (cosine/MSE), plus bit-stability across 3 runs.
- **G2**: consolidation wall-clock delta ≈ 0 (SVDs at edge scale are µs).
- **G3**: no-regression — `arena_eval` win-rate not worse than the current separate-average baseline; existing suite counts unchanged.
- **G4**: merge path alloc-stable (grow-only scratch; `papaya`/`vec` discipline).
- **G8** (falsifiable): T4's probe correlates start-time feature rank with stall/final-loss on ≥20 logged runs (warning-only until then); T2's alignment A/B must show aligned-merge ≥ naive-merge or the alignment step is demoted.

## P0–P3

| Priority | Task | Repo | Status |
|---|---|---|---|
| P0 | T1 product-space merge fix + regression pin | riir-train | Issue 494 T1 |
| P1 | T2 gauge alignment + cross-game eval gate | riir-train | Issue 494 T2 |
| P1 | T3 same-run soup behind flag + GOAT | riir-train | Issue 494 T3 |
| P2 | T4 run-start rank-vs-N probe (warning-only) | riir-train | Issue 494 T4 |
| P2 | T5 §7 + Bench 558 doc-truth pins | riir-train | Issue 494 T5 |
| P3 | closed-form realization variant w/ σ⁻¹ (only if an invertible-activation consumer appears) | katgpt-rs | idea only |

## Honest caveats (pin against re-litigation)

1. **Theorem class ≠ our activations.** Sigmoid: strictly monotone but range (0,1), violates Assumption 2.1 (needs onto ℝ). GELU/SiLU: not PWL. The constructions are *directional* for our nets; every consumer must verify empirically per-merge (which T1's eval gate does by construction).
2. **N unattainable.** `n₁ ≥ 2N` never holds at corpus scale; the inequalities are guidance, not gates. Never encode `width ≥ 2N` as a CI check.
3. **§7 is load-bearing.** Connected sublevel sets ≠ optimization success. Any doc citing this paper for "merging is safe" must carry the §7 caveat and the empirical per-merge verification requirement.
4. **Bench 558 is the measured cautionary tale**: predecessor-init (raw weight copy across positions) cost 2×/2–5× — a weight copy is not a loss-invariant path. NeuMeta's same-position copies preserve function. This is the paper's alignment machinery showing up in our own data, in reverse.
