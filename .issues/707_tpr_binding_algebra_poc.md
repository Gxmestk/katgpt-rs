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

- [-] **T8** OOD withheld-pair eval protocol + L2,1 arm A/B for trained artifacts (edge_lora / hypernet / KG-embedding tables / direction tables). Unblock: next training run touching any matrix that must compose systematically; protocol is cheap and should land as a pinned bench then. Consumers span riir-train, riir-ai (quest_grammar training), riir-clippy.
- [-] **T9** TPR-surrogate interpretability of our own artifacts (riir-clippy drafter first; Bench-694 markov-head wrong-contract question as the concrete hook). Unblock: owner approves a small offline analysis window; never a runtime path.
- [-] **T10** riir-ai runtime guide (NPC personality surgery over the R158 committed-personality surface + F3 unbinding→KG emission). Unblock: a production consumer materializes (no-default-consumer rule).
