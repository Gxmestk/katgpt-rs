# Issue 707: `tpr` binding-algebra PoC — bind/unbind/surgery/project + validation harness (Research 527)

**Status:** Open — PoC/proof task filed from Research 527 (arXiv:2608.29530, McCoy/Soulos/Linzen/Smolensky 2026). Gain verdict; feature-gated, no default promotion until gates pass AND a consumer exists.

**Research:** [katgpt-rs/.research/527_TPR_Emergent_Symbolic_Structure_Binding_Algebra.md](../.research/527_TPR_Emergent_Symbolic_Structure_Binding_Algebra.md)
**Sibling family:** R299 Clifford wedge (`geometric_product_wedge_into`), R495 spectral pencil, R491/R389 steering. The rank-m generalization of the single-direction-vector latent ops.

## Goal

Ship the modelless TPR (Tensor Product Representation) algebra as an opt-in `tpr` feature in katgpt-core: compose structured latent states from role-filler bindings, unbind with a computable error bound, edit via closed-form constituent surgery, denoise via structural projection, and validate any latent-state corpus for systematic binding with the three-part harness. The DISCOVER fit runs as ridge-ALS (closed form, monotone-certifiable) — no gradient descent anywhere.

## Phase 1 — Core algebra (modelless)

- [ ] **T1** `tpr_bind`: per-role pre-sliced `W_r ∈ R^{D×d}` skinny GEMV; `bind(f, r) = W_r·f`. Caller-owned scratch, zero alloc.
- [ ] **T2** `tpr_unbind`: `f̂ᵢ = M·r̂ᵢ` from the summed core matrix `M = Σⱼ fⱼrⱼᵀ`; orthonormalized role basis (pivoted QR, offline); ship the coherence bound `‖Δf‖ ≤ μ(m−1)·max‖fⱼ‖` (`μ = max pairwise |⟨r̂ᵢ,r̂ⱼ⟩|`) as a sigmoid-gate input.
- [ ] **T3** `surgery_delta`: `e' = e + W_r(f_new − f_old)`; role-crossing form `e' = e − W_{r1}f + W_{r2}f`. One GEMV + axpy, bit-additive.
- [ ] **T4** `core_project`: structural denoise `ê = W·C⁻¹Wᵀ(e−b)+b` with cached Cholesky of `C = WᵀW + λI`; residual + a-priori certificate (tail eigenmass of the state covariance, offline).
- [ ] **T5** Ridge-ALS fit (offline calibration): 4 closed-form blocks (W,b / cores / fillers / roles), deterministic HOSVD/QR init, monotone-certificate assert; L2,1 via prune + exact refit (fallback: reweighted-ridge MM). Output = frozen BLAKE3-committed artifact (`W_r` slices, orthonormal `R`, filler table, μ, scheme label, fit residual) thawable at runtime.

## Phase 2 — Validation harness (the GOAT gate instruments)

- [ ] **T6** Three-part binding test: (a) fit-residual band + monotone certificate; (b) planted-TPR positive control (unbind cosine ≥ 0.999, surgery bit-additive) + intervention battery through a real downstream consumer; (c) withheld-(role,filler)-pair OOD vs the **atomic-dictionary null** (a per-pair lookup must FAIL OOD by construction — TPR beating it is the systematicity certificate) + shuffled-role control.
- [ ] **T7** Diagnostics: BoW-null structure router (m=1 shared-role fit; `r_bow/r_full < 1+ε` ⇒ state family carries no binding structure ⇒ skip structured machinery) + role-scheme BIC selection (`score(S) = N·log(RSS_S/N) + p_S·log N`, argmin = frozen structure label).

## Phase 3 — GOAT gate (per Research 527 §6)

- [ ] **T-G1** Planted-TPR recovery + double-run bit-identical artifacts + monotone certificate.
- [ ] **T-G2** Surgery p99 sub-µs at 64–768-D entity dims; projection ≤ 2× dot-product floor; ALS ≤ GD-baseline wall-clock.
- [ ] **T-G3** Default features untouched (opt-in `tpr`, kill-switch env); existing readouts bit-identical with feature off.
- [ ] **T-G4** Zero steady-state allocs on bind/unbind/surgery/project (tracking allocator, exact counts).
- [ ] **T-G8** Withheld-pair accuracy > atomic-dictionary null by a registered margin; intervention battery ≥ registered bar; BoW task-dependence reproduces on a structure-free control family; denoise-readout adopted only where it wins.
- [ ] **T-Promote** Promote only after ALL gates pass AND a consumer exists (no-default-consumer rule). Demote-loser if a simpler op wins the same slot.

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
