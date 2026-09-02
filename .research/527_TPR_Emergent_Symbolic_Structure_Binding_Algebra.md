# Research 527: TPR Emergent Symbolic Structure — Role-Filler Binding Algebra

> **Source:** [The Emergent Symbolic Structure of Artificial Neural Networks](https://arxiv.org/abs/2608.29530) — R. Thomas McCoy, Paul Soulos, Tal Linzen, Paul Smolensky. arXiv:2608.29530v1 [cs.CL], 30 Aug 2026. 59 pp.
> **Date:** 2026-09-02
> **Status:** Active — **Gain** verdict; PoC issue 707 filed (`tpr` binding algebra + validation harness)
> **Related Research:** 299 (Clifford geometric product — multiplicative binding, sibling family), 495 (spectral pencil affine gate — rank-1 direction-gate sibling), 144 (functional emotions linear representations — the concept-direction framing TPR generalizes with roles), 257 (functional attention transport operator), 324 (trajectory geometry of transformer layers — layer-profile diagnostics), 103 (state distribution view), 389/491/382/505 (steering corpus — additive-only, the delta target); riir-ai 010 (KG × HLA × Role Transport — closest shipped cousin), 123 (latent functor runtime guide), 158 (per-NPC committed personality blend — the surgery application surface), 156 (Clifford wedge emotional complementarity), 141 (KG triple typology)
> **Related Plans:** 319 (geometric product latent interaction); riir-ai 151 (KG role transport); **707 (this — PoC, filed same day)**; riir-clippy **62** (F5 consumer — TPR × healer corpus: structured retrieval axis + withheld-pair OOD bench)
> **Classification:** Public

---

## TL;DR

McCoy et al. show that the vector representations of MLPs, GRUs, Transformers, and 7 LLMs (incl. GPT-OSS-20b) are closely approximated by **linearly-transformed Tensor Product Representations**: `E = W(Σᵢ fᵢ ⊗ rᵢ) + b` — filler embeddings bound to role embeddings via tensor products, summed, affine-transformed. The approximations are causally load-bearing: **constituent surgery** (`e' = e − TPR(r:f_old) + TPR(r:f_new)`) edits LLM behavior as intended (0.903 avg across 31 intervention types), DISCOVER generalizes to withheld role-filler pairs (systematic binding, not atomic memorization), and — the sleeper result — downstream decoders read **better from the idealized TPR approximation than from the true noisy encodings** (0.71 → 0.96 on GPT-OSS middle layer).

**Distilled for katgpt-rs (modelless, inference-time):** the entire payload is algebra, not optimization. The DISCOVER fit is multilinear least squares ⇒ **ridge-ALS closed form** (4 block solves, monotone-certifiable, no autodiff). What ships: `tpr_bind` / `tpr_unbind` (with a computable coherence-error bound feeding a sigmoid gate) / `surgery_delta` (one skinny GEMV + axpy) / `core_project` (Eckart–Young-optimal structural denoise) / the **BoW-null structure router** / the **role-scheme BIC diagnostic** / the three-part **binding-validation harness** (fit residual, surgery causality, withheld-pair OOD vs the atomic-dictionary null). All frozen-artifact (BLAKE3-committed, freeze/thaw-compatible), zero-alloc on the hot path, latent-only (sync boundary untouched).

**Verdict: Gain** (not Super-GOAT — Q1 fails: TPR/DISCOVER/surgery are published prior art; the fusion and the runtime primitive family are the contribution). PoC filed as `.issues/707`.

---

## 1. Paper Core Findings

1. **TPR structure emerges broadly.** Bidirectional roles (left+right position pairs) approximate single-vector encoders (MLP/GRU/bottleneck-Transformer) at ≥0.97–1.0 across copy/reverse/interleave tasks; bag-of-words (the structureless null) fails on all structure-dependent tasks.
2. **Structure tracks task demands.** Train the same architectures to SORT (order-irrelevant) and bag-of-words now fits well — representation structure is a response to task structure, not an architectural given.
3. **LLM period encodings are sentence TPRs.** The period token encodes the preceding list/sentence; bidirectional roles fit well, syntactic roles fit worse — period encodings prioritize linear order over syntax (confirmed even when the downstream decoder is trained on parsing outputs, App. G).
4. **GPT-OSS symbolic tasks (arithmetic, syllogisms, code execution, passivization, tense reinflection, question formation): `task-specific (all)` roles win on 6/6.** Each token's representation encodes itself AND all preceding tokens, each as a filler bound to a **pair role** (concatenation of source and host task positions, e.g. `subject_adj-object_noun`). Approximation accuracy lands within 2.36pp of GPT-OSS's own task accuracy.
5. **Causal interventions (constituent surgery).** Edits = subtract `W(f_old ⊗ r)` + add `W(f_new ⊗ r)` applied across every token/layer predicted to carry the pair. Role moves work: an adjective moved from object-modifier to subject-modifier flips GPT-OSS's tense output as if the sentence had been rewritten. 0.903 avg over 31 intervention types. Local (single-token) edits suffice for filler swaps but degrade sharply for role edits — **identity is local, structure is distributed**.
6. **OOD generalization to withheld role-filler pairs** beats the atomic-dictionary chance baseline (1/n!) — evidence of systematic binding. **L2,1 regularization on the filler/role embedding matrices is load-bearing** (unregularized 0.00 → regularized 0.86–0.90 on several settings; 0.54 → 0.97 in one) — it kills whole embedding dimensions, preventing degenerate per-pair memorization. White-box controls confirm the inference: a model with systematic binding supports OOD generalization; an atomic-pair-embedding model does not.
7. **The idealization beats the truth.** Period-unpacking decoders trained on real LLM encodings decode MORE accurately from DISCOVER's clean TPR approximations than from the encodings they were trained on. Interpretation ("limitivism"): the network approximately-but-not-precisely realizes the symbolic structure; the decoder learns to rely on the structure; the fitted approximation realizes it precisely ⇒ more robust readout.
8. **All major VSAs (HRR, MAP, BSC) are special cases of linearly-transformed TPRs** — so a successful DISCOVER fit certifies membership in the whole role-filler/multiplicative-binding family.

---

## 2. Distillation — Path 0 inventory (coverage × extraction)

Per §3.5: every row marked (coverage: analog exists / partial / none) received the signal-diff check before "covered".

| # | Paper component | Coverage in stack | Modelless extraction | Disposition |
|---|---|---|---|---|
| 1 | TPR bind `W(f⊗r)` | partial — Clifford wedge (R299) is pairwise interaction, not role composition; role transport (riir-ai 010) conditions attention keys | `tpr_bind`: per-role pre-sliced `W_r` GEMV. **Signal diff vs R010:** role transport applies an operator to attention keys `k_j = W_k·R_{s(j)}·x_j` at attention time; bind composes STATE vectors. Different surface, both alive. | Issue 707 T1 |
| 2 | Unbinding + error bound | none — no role-filler readout anywhere | `f̂ᵢ = M·r̂ᵢ` exact under orthonormal roles; bound `‖Δf‖ ≤ μ(m−1)·max‖fⱼ‖` with `μ = max|⟨r̂ᵢ,r̂ⱼ⟩|` computable offline → principled sigmoid gate input | Issue 707 T2 |
| 3 | Constituent surgery | none — steering corpus (491/389/382/505) is additive-only; **signal diff:** additive steering cannot express role MOVES (`−W_{r1}f + W_{r2}f` crosses two role slices) | `surgery_delta` = one skinny GEMV + axpy, closed form | Issue 707 T3 |
| 4 | Structural denoise readout | partial family — hodge_decompose/spectral projections; **signal diff:** projection onto the FITTED role-filler affine manifold, Eckart–Young-optimal for fitted W, with a residual certificate | `core_project`: `ê = W·C⁻¹Wᵀ(e−b)+b`, 2 GEMVs + triangular solve | Issue 707 T4 |
| 5 | DISCOVER fit | none (no decomposition of our own state corpora into role-filler form) | **ridge-ALS**: 4 closed-form blocks (W·b / cores / fillers / roles), monotone-certifiable; L2,1 via prune+exact-refit (recommended) or reweighted-ridge MM | Issue 707 T5 |
| 6 | Three-part validation harness | partial — Dirichlet energy (riir-ai 264) measures alignment, not binding systematicity; no withheld-pair OOD test exists for any of our latent primitives | fit-residual gate / planted-TPR positive control + intervention battery / withheld-pair OOD vs atomic-dictionary null | Issue 707 T6 |
| 7 | BoW-null structure router | none — we have no closed-form test of "does this state family carry binding structure" | fit m=1 shared-role ALS; residual ratio `r_bow/r_full < 1+ε` ⇒ skip structured machinery | Issue 707 T7 |
| 8 | Role-scheme BIC diagnostic | none | `score(S) = N·log(RSS_S/N) + p_S·log N`; argmin = frozen structure label stamped in the artifact | Issue 707 T7 |
| 9 | L2,1 recipe for matrices that must compose | not applied — our trained embedding/adapter matrices have **zero OOD-generalization measurement** | withheld-pair eval protocol on edge_lora / hypernet / KG-embedding tables; L2,1 arm A/B | Issue 707 T8 (defer-marked, training track) |
| 10 | TPR surrogate interpretability of our artifacts | none | surrogate over riir-clippy drafter / Bench-694 markov head (wrong-contract question) | Issue 707 T9 (defer-marked, training track) |

**Path 0 verdict: MODELLESS-VALIDABLE.** Rows 1–8 are closed-form or eval-protocol; the only GD in the paper is the fit, and the fit is bilinear least squares. Rows 9–10 are genuine training-track items → tracked as defer-marked issue tasks with unblock conditions (per one-verdict-per-track: the training-track verdict is ALSO Gain-conditional, secondary to the modelless track by serving-envelope fit — TPR surrogates are offline microscopes, never runtime).

---

## 3. Fusion

**Fusion F1 (primary): TPR binding × Clifford wedge (R299) × freeze/thaw.** The paper's §9.2: all major VSAs are linearly-transformed TPR special cases — so R299's shipped `geometric_product_wedge_into` and a `tpr_bind/unbind` family are the same algebra viewed from two angles (wedge = pairwise orthogonality signal; TPR = multi-slot compositional state). The fusion: NPC belief/personality states composed as TPRs (filler = entity/emotion concept, role = social position), wedge as the filler↔filler structural-divergence signal *within* the bound state, whole artifact BLAKE3-frozen and hot-swappable. None of the three alone gives editable, certifiable, versioned structured state.

**Fusion F2: surgery × steering corpus (491/389).** Existing steering adds concept directions; surgery moves concepts BETWEEN roles (adjective subject↔object move class). For the riir-ai selling point: "move a grudge from faction-A to faction-B role", "swap which entity fills the threat slot" — targeted personality edits (R158's committed personality blend is the state surface) with causal validation from the harness. Named follow-up, NOT filed as a guide: no-default-consumer rule — the runtime guide (riir-ai) waits for a consumer to materialize.

**Fusion F3: unbinding × KG emission (riir-ai 010/141, vibe.rs).** KG triple (s,p,o) IS a role-filler binding. Unbind `(s,p,o)` roles from a latent state, similarity-classify fillers against the frozen vocabulary, threshold → triples derived from binding structure rather than raw dot-product heuristics. Upgrades an existing latent→KG path; same disposition as F2 (consumer-gated).

**Fusion F4: BoW router × thermal tiering.** The residual-ratio router decides whether a state family pays the structured-op cost at all — the same gate shape as `breakeven` (R218) applied to representation structure rather than compute routing.

**Fusion F5 (consumer, riir-clippy — filed as riir-clippy `.issues/62`): the healer corpus IS a role-filler dataset.** Every corpus entry binds role (rule/lint identity) to filler (code shape); the retrieval stack's own history is the paper's thesis in miniature — the Issue 017/T3.5 misses were spans sharing the filler (generic fn shape) while differing in the ROLE (operator pattern), which the RustOpTokenizer fixed by injecting role-ish signal into tokens; TPR makes that axis structural. Four sub-items: (a) **withheld-(rule, shape)-pair OOD bench** — the one axis nothing measures (every retrieval floor and fixture set is in-distribution by construction); the paper predicts atomic memorization is the default failure, and the bench converts that prediction into our measurement BEFORE building the axis, with the verbatim-lookup null as the atomic floor — this half has NO dependency on the primitive; (b) ALS fit of the corpus into shape⊗rule form, with `self_evolve`'s EMA domain directions as the already-runtime-updated rank-1 role embeddings (thaw from the same freeze/thaw cycle, don't fork); (c) TPR-structured retrieval behind `tpr_retrieval` — query-side structural denoise (paper finding 7 applied to matching) + role-conditioned rerank, with the honest claim discipline that in-distribution gains are NOT the claim (floors already 89.6–100 top-1); the claim axes are the OOD delta and kernel_opt long-tail crowding (426 rules); (d) counterfactual surgery on `LatentFixMemory` trajectories — closed-form "would domain X have healed this miss?" complementing Bench 032's re-run-based switch-cost measurement. This also supplies issue 707's G8 "real downstream consumer": the healer corpus beats a synthetic micro-model as the first validation dataset.

---

## 4. Novelty gate (§1.5)

**Pinned claims (pre-search, per the TTPO rule):**
- **A:** "Constituent surgery (closed-form ±role-filler edits from a fitted TPR decomposition) as a structure-aware steering primitive for game-NPC latent states, distinguished from additive steering (R491/R389) by role-crossing edits and from riir-ai role transport (R010) by operating on the representation rather than attention keys."
- **B:** "The three-part binding-systematicity harness (fit residual / surgery causality / withheld-pair OOD vs atomic-dictionary null) applied to our own per-NPC latent primitives."
- **C:** "Structural projection onto the fitted TPR manifold as a gated denoising readout."

**§4 searches:** (1) "constituent surgery tensor product representation intervention steering LLM role-filler" → zero relevant hits (group-representation-theory noise only) — no competing published work applying surgery as an agent-steering primitive. (2) "DISCOVER TPR interpretability" → confirms the published lineage: McCoy 2019 (TPDN), Soulos 2020 (role-learning networks / constituent surgery origin), AID (ICLR 2024), Discrete Dictionary Decomposition (NeurIPS 2024), TPNN survey. No game/agent applications found.

| Question | Verdict |
|---|---|
| Q1 no prior art? | **NO.** TPR (Smolensky 1990), DISCOVER/TPDN (McCoy 2019), constituent surgery (Soulos 2020), and this paper's own results are published prior art for the mechanism. The *application fusion* and the ALS-closed-form runtime family are not published — but that is GOAT-tier fusion, not mechanism novelty. |
| Q2 new behavior class? | YES for the stack: role-crossing surgery edits, certified unbinding, and the binding-systematicity harness are capabilities no shipped primitive has (signal-diffs above). Rests on a published mechanism ⇒ not a moat-creating class. |
| Q3 product selling point? | YES: "NPC latent personalities are surgically editable — move a grudge between factions, swap a threat-slot filler, denoise belief states before readout — with causally-validated closed-form edits." |
| Q4 force multiplier? | YES: ≥2 pillars (HLA, latent functor, KG, Clifford, freeze/thaw). |

**All-4-YES fails on Q1 → NOT Super-GOAT. No "candidate" escape hatch invoked.**

**Tier: Gain.** Routing: open primitive family → katgpt-rs (issue 707, feature-gated `tpr`); training-track recipes → defer-marked tasks in the same issue with unblock conditions; riir-ai runtime guide (F2/F3) → named follow-up, consumer-gated (no-default-consumer rule).

**MOAT gate (katgpt-rs):** PASS — fundamental/base primitive via fusion (latent-op slot: a rank-m generalization of the shipped single-direction-vector ops: spectral_pencil's one-direction gate → TPR's role-multiplexed bind/unbind). No game/chain/shard IP in the primitive. Demote/promote per the per-stack ledger after the GOAT gate.

---

## 5. Three-track adversarial panel (merged)

**No-GD advocate (feasible, high confidence):** 13-item inventory, all closed-form. Ridge-ALS with 4 block solves; `W,b` block shares one cached Cholesky; filler/role blocks embarrassingly parallel; monotone certificate (`obj_new ≤ obj_prev + ε`) that GD cannot cheaply provide; deterministic HOSVD/QR init. L2,1 three GD-free routes — reweighted-ridge MM (monotone), proximal soft-threshold, **prune + exact refit (recommended: exact zeros ⇒ sparser GEMV, same dead-dim identification)**. Honest caveat recorded: ALS reaches the same fixed-point *class*, not the same point — mitigated by the monotone certificate + residual gate + the fact that the paper's claims are downstream-validated (surgery causality, OOD pairs), which is exactly what the gates measure. Runtime cost at entity dims (64–768-D, k≤16 roles): sub-µs surgery, trivial ALS offline.

**Model-based advocate (4 of 5 recipe items consumable):** (1) **OOD withheld-pair eval protocol FIRST** — near-zero cost, and the honest admission it forces: every trained artifact we own (edge_lora, hypernet, drafters, qwen38 heads) is evaluated in-distribution only; the paper predicts the default outcome is atomic memorization, and the protocol converts that prediction into our measurement. (2) L2,1 on LoRA A/B matrices (learned sparsity complementing the structural `active-index-grad-scatter`), hypernet adapter emitters, and a small KG-embedding run for `riir-neuron-db` (the cleanest role-filler fit in the stack: (s,p,o) *is* a binding). (3) TPR surrogates as interpretability microscopes — first target riir-clippy drafter (minutes on CPU/Metal), the Bench-694 markov-head wrong-contract question as a concrete hook; Bonsai-27B explicitly out of budget by default. (4) Role-scheme design as architecture prior — conditional on the grammar drafter ever hitting an OOD wall. (5) Anti-memorization curriculum. Kill-switches pre-registered: if current artifacts already generalize at ceiling, items 2–5 demote.

**Coordinator merge:** the tracks agree on the spine — measure first (harness), then the modelless algebra ships as a frozen artifact, then training-track items fire only if the OOD baseline shows the memorization failure mode. Discard audit: no advocate finding discarded; rows 9–10 carry auditable unblock conditions rather than silent drops.

---

## 6. GOAT gate sketch (for issue 707)

- **G1:** planted-TPR micro-model — unbind cosine ≥ 0.999, surgery bit-additive, ALS monotone certificate, double-run bit-identical artifact.
- **G2:** surgery p99 sub-µs at entity dims; projection ≤ 2× dot-product floor; ALS wall-clock ≤ the GD baseline it replaces.
- **G3:** default feature set untouched (`tpr` opt-in, kill-switch); existing readouts bit-identical with the feature off.
- **G4:** zero steady-state allocs on bind/unbind/surgery/project (tracking allocator, exact counts).
- **G8 (behavioral):** withheld-pair accuracy beats the atomic-dictionary null by a registered margin; intervention battery ≥ registered bar through a real downstream consumer; BoW task-dependence reproduces (structure-free family shows `r_bow ≈ r_full`); denoise-readout adoption only where it wins.
- **UQ note:** the denoise/coherence gates produce confidence scalars, not predictive intervals — not UQ-bearing under the Report-the-Floor rule; if coverage claims are ever added, the conformal floor applies.
- **Promotion:** opt-in until all gates pass AND a consumer exists (no-default-consumer rule); demote-loser discipline applies.

## 7. References

- McCoy, Soulos, Linzen, Smolensky. *The Emergent Symbolic Structure of Artificial Neural Networks.* arXiv:2608.29530, 2026. (Code: github.com/tommccoy1/discover)
- Smolensky 1990 (TPR); McCoy et al. 2019 (RNNs implicitly implement TPRs, ICLR); Soulos et al. 2020 (role-learning networks, constituent surgery, BlackboxNLP).
- Internal: R299/Plan 319 (Clifford), riir-ai R010/Plan 151 (role transport), riir-ai Bench 264 (Dirichlet + transport interference), R495 (spectral pencil), R491/R389 (steering), R158 (personality blend), riir-ai R141 (KG typology).
