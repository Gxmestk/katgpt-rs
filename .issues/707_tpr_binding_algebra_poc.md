# Issue 707: `tpr` binding-algebra PoC — bind/unbind/surgery/project + validation harness (Research 527)

**Status:** Phases 1–3 LANDED, GOAT **ALL PASS**, feature stays **opt-in** — 2026-09-02. `katgpt_core::tpr` ships behind `tpr = []` with 26 lib tests + `.benchmarks/698_tpr_binding_algebra_goat.md` (G1/G2/G3/G4/G8 all PASS, three consecutive runs). Promotion is blocked by the **no-default-consumer rule**, not by any gate: T-Promote needs a real consumer (riir-clippy Issue 062 T4 or riir-ai Issue 847). Nothing is left unchecked; the five `[-]` rows each name a non-engineering unblock (a consumer, a training run, or an owner window). Everything the primitive can prove alone is proven.

**Research:** [katgpt-rs/.research/527_TPR_Emergent_Symbolic_Structure_Binding_Algebra.md](../.research/527_TPR_Emergent_Symbolic_Structure_Binding_Algebra.md)
**Sibling family:** R299 Clifford wedge (`geometric_product_wedge_into`), R495 spectral pencil, R491/R389 steering. The rank-m generalization of the single-direction-vector latent ops.

## Goal

Ship the modelless TPR (Tensor Product Representation) algebra as an opt-in `tpr` feature in katgpt-core: compose structured latent states from role-filler bindings, unbind with a computable error bound, edit via closed-form constituent surgery, denoise via structural projection, and validate any latent-state corpus for systematic binding with the three-part harness. The DISCOVER fit runs as ridge-ALS (closed form, monotone-certifiable) — no gradient descent anywhere.

## Phase 1 — Core algebra (modelless)

- [x] **T1** `tpr_bind`: per-role pre-sliced `W_r ∈ R^{D×d}` skinny GEMV; `bind(f, r) = W_r·f`. Caller-owned scratch, zero alloc.
- [x] **T2** `tpr_unbind`: `f̂ᵢ = M·r̂ᵢ` from the summed core matrix `M = Σⱼ fⱼrⱼᵀ`; orthonormalized role basis (pivoted QR, offline); ship the coherence bound `‖Δf‖ ≤ μ(m−1)·max‖fⱼ‖` (`μ = max pairwise |⟨r̂ᵢ,r̂ⱼ⟩|`) as a sigmoid-gate input.
- [x] **T3** `surgery_delta`: `e' = e + W_r(f_new − f_old)`; role-crossing form `e' = e − W_{r1}f + W_{r2}f`. One GEMV + axpy, bit-additive.
- [x] **T4** `core_project`: structural denoise `ê = W·C⁻¹Wᵀ(e−b)+b` with cached Cholesky of `C = WᵀW + λI`; residual + a-priori certificate (tail eigenmass of the state covariance, offline).
- [x] **T5** Ridge-ALS fit (offline calibration): 4 closed-form blocks (W,b / cores / fillers / roles), deterministic HOSVD/QR init, monotone-certificate assert; L2,1 via prune + exact refit (fallback: reweighted-ridge MM). Output = frozen BLAKE3-committed artifact (`W_r` slices, orthonormal `R`, filler table, μ, scheme label, fit residual) thawable at runtime.

## Phase 2 — Validation harness (the GOAT gate instruments)

- [x] **T6** Three-part binding test: (a) fit-residual band + monotone certificate; (b) planted-TPR positive control (unbind cosine ≥ 0.999, surgery bit-additive) + intervention battery through a real downstream consumer; (c) withheld-(role,filler)-pair OOD vs the **atomic-dictionary null** (a per-pair lookup must FAIL OOD by construction — TPR beating it is the systematicity certificate) + shuffled-role control.
- [x] **T7** Diagnostics: BoW-null structure router (m=1 shared-role fit; `r_bow/r_full < 1+ε` ⇒ state family carries no binding structure ⇒ skip structured machinery) + role-scheme BIC selection (`score(S) = N·log(RSS_S/N) + p_S·log N`, argmin = frozen structure label).

## Phase 3 — GOAT gate (per Research 527 §6)

- [x] **T-G1** Planted-TPR recovery + double-run bit-identical artifacts + monotone certificate.
- [x] **T-G2** Surgery p99 sub-µs at 64–768-D entity dims; projection ≤ 2× dot-product floor; ALS ≤ GD-baseline wall-clock.
- [x] **T-G3** Default features untouched (opt-in `tpr`, kill-switch env); existing readouts bit-identical with feature off.
- [x] **T-G4** Zero steady-state allocs on bind/unbind/surgery/project (tracking allocator, exact counts).
- [x] **T-G8** Withheld-pair accuracy > atomic-dictionary null by a registered margin; intervention battery ≥ registered bar; BoW task-dependence reproduces on a structure-free control family; denoise-readout adopted only where it wins. — **PASS on three of four clauses** (100.0% vs 0.0% at a 20 pp bar with the null verified informative ID; BoW router + shuffled-role control both reproduce on the order-free family; projection adopted only behind its own floor gate). The **intervention-battery clause needs a consumer** and is tracked as `T6b-consumer` below rather than being counted here.
- [-] **T-Promote** Promote only after ALL gates pass AND a consumer exists (no-default-consumer rule). Demote-loser if a simpler op wins the same slot.
  **First consumer arrived and did not unblock this — it hardened the harness instead.** riir-clippy Bench 063 (2026-09-02) fitted the primitive over 325 real clippy/rustc lint states and measured its 8-D consumer state **FLAT**: `bow_router` 1.14 / `cross_state` 1.07 on the 256-D control vs `r_bow < r_full` in the 8-D space the reranker actually scores in. So 062 T4 is re-scoped to a scalar-evidence bridge and still does not make `tpr` a default-path dependency. What the consumer DID produce is Issue 710: `shuffled_role_control` could not fail on a single-binding corpus, the shape every retrieval consumer has. Fixed in `af3bca25` — vacuity is now a reported quantity on all four validation instruments (`ShuffledRoleReport::vacuous`/`moved`/`mode`, `BowRouterReport::vacuous`, `candidate_pool_coverage`, and the pre-existing `AtomicNull::coverage`), with the cross-state arm resolving automatically. **This is the T-G8 blind spot, found by the only means that could find it:** G8 ran the role control on multi-binding synthetic data, where the defect is invisible. Read that as a caution on the remaining `[-]` rows — each is unblocked by a real consumer, and a real consumer is what measures what synthetic fixtures cannot.

## Landed — what shipped, and where it diverges from this issue's text

**Code:** `crates/katgpt-core/src/tpr/{mod,types,als,validate,tests}.rs`,
`crates/katgpt-core/benches/bench_707_tpr_binding_goat.rs`, feature `tpr = []`.
**Evidence:** `.benchmarks/698_tpr_binding_algebra_goat.md`.

Four places where the shipped primitive is NOT what the task text above says.
Each was a measurement, not a preference:

1. **T4 ships the explicit `(WᵀW + λI)⁻¹`, not the Cholesky factor.** The fit
   still factors the Gram (`linalg::spd_inverse_f32`, so definiteness is still
   checked); the artifact carries the inverse because a K=32 triangular solve
   is a 32-step serial dependency chain and put the whole projection at 2.14×
   its floor against a 2× bar. One K×K matvec: 1.33×.
2. **T5's "deterministic HOSVD/QR init" is a seeded filler table + an exact
   `W,b` solve.** HOSVD was not needed: the `W,b` block is closed-form, so the
   init only has to be deterministic. The pivoted QR that DID ship is the
   unbind basis (T2), where it is load-bearing. `tucker_factorization` (which
   ships HOSVD) is deliberately not a dep.
3. **The monotone certificate is ENFORCED, not counted.** Blocks 3–4 minimize a
   core-space surrogate, so an uphill sweep is possible — one was measured. The
   fit rejects the proposal and rolls back, so the artifact is the minimum of
   the recorded trajectory; `monotone_violations` counts rejections, and the
   checkable claim is `fit_objective == min(ssr_per_sweep)`.
4. **T6(b)'s "intervention battery through a real downstream consumer" did NOT
   ship** — there is no consumer yet, which is the same fact that blocks
   T-Promote. What shipped is the planted positive control, the withheld-pair
   OOD arm vs the atomic null, and the shuffled-role control. See the deferred
   row below.

Also worth knowing: the encoder layout in `TprArtifact::w` is a **measured**
choice with a table of the three layouts tried (2708 → 2625 → 333 ns for a
D=768 bind), and the G8 harness had two bugs that each produced a plausible
wrong number before being fixed — both documented in Bench 698.

- [-] **T6b-consumer** Intervention battery through a real downstream consumer.
  Unblock: the first consumer lands (riir-clippy Issue 062 T4 structured
  retrieval, or riir-ai Issue 847 quest_grammar). Same unblock as T-Promote —
  filed separately so the gate table is not read as complete.

## Deferred — training-track items (unblock conditions; per Research 527 §2 rows 9–10)

> **First real-world consumer (F5):** riir-clippy `.issues/62` — TPR × healer corpus fusion. Its T1 (withheld-pair OOD bench) has NO dependency on this primitive and produces the OOD baseline that gates its Phase 2; its T4 (structured retrieval) is the designated real consumer for this issue's T-G8 intervention battery — the healer corpus beats a synthetic micro-model as the G8 validation dataset.
>
> **That baseline now EXISTS — measured 2026-09-02, riir-clippy `.benchmarks/062_withheld_pair_ood.md`**
> (Phase 1 T1+T2 landed, riir-clippy `ce9d395` preregistration + `cf7d356` results).
> The consumer's OOD numbers T-G8 must beat, and the caveats that bound the claim:
>
> | corpus | ID top-1 | OOD top-1 | paired Δ | chance floor | atomic null (ID/OOD) |
> |---|---:|---:|---:|---:|---:|
> | healer, pure retrieval | 52.0% | **4.0%** | +48.0 pp | 2.2% | 0.0% / 0.0% |
> | healer, shipped `Structural` rerank | 92.0% | 52.0% | +40.0 pp | 2.2% | 0.0% / 0.0% |
> | `rustc_errors` | 97.7% | **48.8%** | +48.8 pp | 12.5% | 79.1% / 0.0% |
>
> Three constraints this puts on T-G8, all measured rather than assumed:
>
> 1. **Use `rustc_errors`, not the healer corpus, as the atomic-null arm.** The
>    healer null is **VACUOUS** — 0.0% on the ID arm too, because no two healer
>    bindings share a normalized token structure, so the memorizer cannot fit even
>    its own training set. A null that fails in-distribution certifies nothing.
>    `rustc_errors` (79.1% ID / 0.0% OOD) is the informative one.
> 2. **Scope the claim to retrieved-but-demoted rows.** OOD accuracy is a **step
>    function in the role's remaining filler count**: E0597 (6 fillers) generalizes
>    at 100.0%, E0596 (2 fillers) collapses to 0.0% *while keeping the true code at
>    rank 2–7*. A role with one remaining filler has nothing to compose from — any
>    gain there should be read as **leakage, not systematicity**.
> 3. **The corpora are nearly role/filler-COLLINEAR** (32/45 clippy roles and 6/8
>    error codes occur under exactly ONE shape-class), so most unseen combinations
>    do not exist to be tested. Widening fillers-per-role is a CORPUS task and a
>    precondition for a strong T-G8, not something the primitive can fix.
>
> Also note for T5/BoW: `rustc_errors` being 6/8 single-filler predicts *little*
> residual gain from the m=1 shared-role fit there, while `clippy_lints` has 13
> multi-filler roles. That is a prior to test, not a result.

>
> **First consumer by priority (F6, #1 surface):** riir-ai `.issues/847` — quest_grammar role-move variants over S-V-O (paper Fig 7.4C: the agreement cascade is `SealGrammarAnnotation.verb_forms`; `GrammarValidator` is the downstream re-validation). Its T1 (withheld-(noun, slot)-pair OOD bench) is likewise ungated; its T3–T6 (surgery variant ops + cascade re-validation) consume this issue's T1–T5.

- [-] **T8** OOD withheld-pair eval protocol + L2,1 arm A/B for trained artifacts (edge_lora / hypernet / KG-embedding tables / direction tables). Unblock: next training run touching any matrix that must compose systematically; protocol is cheap and should land as a pinned bench then. Consumers span riir-train, riir-ai (quest_grammar training), riir-clippy.
- [-] **T9** TPR-surrogate interpretability of our own artifacts (riir-clippy drafter first; Bench-694 markov-head wrong-contract question as the concrete hook). Unblock: owner approves a small offline analysis window; never a runtime path.
- [-] **T10** riir-ai runtime guide (NPC personality surgery over the R158 committed-personality surface + F3 unbinding→KG emission). Unblock: a production consumer materializes (no-default-consumer rule).
