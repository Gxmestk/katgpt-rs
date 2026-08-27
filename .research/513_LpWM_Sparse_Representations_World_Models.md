# Research 513: LpWM — Sparse Representations in World Models (support/magnitude factorization)

> **Source:** [LpWM: A Case for Sparse Representations in World Models](https://arxiv.org/abs/2608.22764) — Kuang, Dagade, Le Lidec, Maes, Balestriero, LeCun (NYU AMI Labs / Duke / Mila / Brown), arXiv:2608.22764v1, 2026-08-24. Code: [github.com/YilunKuang/lpworldmodel](https://github.com/YilunKuang/lpworldmodel) (MIT).
> **Date:** 2026-08-27
> **Status:** DISTILLED — pending owner decision (katgpt-rs #693 primitive + riir-ai R352 guide)
> **Related Research:** 504 (Orthogonal JEPA — the sibling geometry claim: orthogonal vs sparse codes; same Gain shape), 498 (LeVLJEPA/SIGReg — LeWM's regularizer lineage), 426 (Temporal Straightening — slowness-family prior, cited by LpWM), 360 (AdaJEPA), 138 (LeJEPA), 270 (ICT distributional branching — the OTHER regime/decision detector; complementary signal, see §Signal-diff), 192 (NextLat belief states), 245 (latent spatial memory video WM)
> **Related Research (riir-ai):** 352 (support-regime events guide — the private half), 142 (ICT branching guide), 151 (latent magnitude hygiene)
> **Related Issues:** katgpt-rs #693 (support-instability + mode-factored accessors primitive)
> **Domain:** katgpt-rs (open primitive: support-instability operator over the shipped `iou` kernel) + riir-ai (regime-gated cognition, KG regime events) + riir-neuron-db (consolidation timing, shard support comparison — noted, not filed)

---

## TL;DR

LpWM is a JEPA world model whose encoder/predictor outputs pass a (Rep)ReLU link and whose per-timestep latent marginals are matched (2-Wasserstein over random projections, RDMReg) to a **Rectified Laplace** target — yielding **non-negative, exactly sparse** latent codes. Claims: (1) *sparse geometry simplifies dynamics* — a one-hot covering theorem (Prop 1: Lipschitz controlled dynamics on a compact space admit finite one-hot encodings with EXACTLY linear action-conditioned latent transitions, error O(N^{-1/d})) motivates distributed sparse codes as the relaxation; empirically sparse codes let **linear/MLP predictors plan where dense codes need Transformers** (+24–57% on PushT at intermediate capacity). (2) The learned codes are **mode-factored**: the binary support decodes the discrete dynamics regime (zone 94–99%, contact), the magnitudes decode continuous within-regime state (position R²=0.94). (3) A **Temporal-Jaccard slowness prior** (soft-Jaccard = Ruzicka 1958, `Σmin/Σmax`) on adjacent-frame supports steers support instability from raw motion (r≈0.87 effector) to regime events (r≈0.80 cube contact).

**Verdict: GAIN (not Super-GOAT).** Prior art is crowded on every component except the mode-factored finding (no prior art found for support=regime/magnitude=state emerging in a single distributed non-negative code — the paper's genuine novelty). For us the kernel coverage is remarkably high: `functional_substitution::iou` IS Ruzicka similarity exactly (different consumer), `stiff_anomaly::stability` is window-Jaccard instability (different domain), `DemonstratedSkill::covered_mask` is a shipped support/magnitude split, `edge_lora::SigmoidGate` matches the LTV gated-correction shape exactly, `crowd_attention` already runs per-zone linear operators. The genuinely missing composition — **consecutive-tick support-overlap on per-entity non-negative latent state as a free regime-transition event detector**, feeding cognition-tier escalation + KG emission + consolidation timing — is filed as #693 + R352.

---

## Key Mechanisms

| # | Mechanism | Formal statement | Training-only? |
|---|---|---|---|
| 1 | RDMReg sparse regularization | match per-timestep marginal of `z` to `ReLU(GN_p)`, p=1 → Rectified Laplace; sliced 2-Wasserstein over `c ~ Unif(S^{d-1})` | Training loss (diagnostics analog: gaussianity probe R498) |
| 2 | (Rep)ReLU link → exact zeros | `RepReLU(x) = sg(ReLU(x)) + GeLU(x) − sg(GeLU(x))` — forward = exact zeros, backward = GeLU gradient | Link is modelless; in-loop use is training |
| 3 | One-hot linearization (Prop 1) | compact X, Lipschitz f ⇒ ∃ finite one-hot `E: X→R^N`, per-action 0/1 matrices `P(a)` with `z' = P(a)z` exactly linear, decoded error ≤ (L+1)ε; grid corollary O(N^{-1/d}), rollout ≤ ε·Σ L^k | **Pure theorem — modelless lens** |
| 4 | Mode-factored code | `supp(z) ∈ {0,1}^D` → discrete regime (94–99% zone decode); `z ⊙ supp(z)` magnitudes → continuous state (R²=0.94 position) | **Structural insight — modelless** |
| 5 | Soft Jaccard (Ruzicka) | `J(x,y) = Σᵢmin(xᵢ,yᵢ) / Σᵢmax(xᵢ,yᵢ)` on `x,y ≥ 0` | **Modelless — kernel already ships as `iou`** |
| 6 | Support instability | `1 − J(z_t, z_{t+1})` between consecutive frames — regime-flip detector (contact r 0.05→0.61 with TJ) | **Modelless — the missing composition** |
| 7 | Temporal-Jaccard slowness prior | `L_TJ = mean over t of (1 − J_S(z_t, z_{t+1}))` — support stability across adjacent frames | Training loss; runtime duals ship (`decay_confidence`, `persistence_eta`) |
| 8 | MLP∘LTV predictor | `A_i(z) = A_i + U_i diag(σ(Gz)) V_iᵀ` — base linear + low-rank sigmoid-gated correction, zero-init U (LoRA-style) | Shape ships as `edge_lora::SigmoidGate` (training side); runtime analog `latent_functor/rank_k` |

---

## Path 0 Decomposition (component → coverage + signal-diff → extraction)

| Component | Coverage (analog ships?) | Signal-diff vs cousin | Extraction | Verdict |
|---|---|---|---|---|
| Soft-Jaccard kernel | **EXACT** — `katgpt-core/src/functional_substitution/iou.rs`: `Σmin/Σmax` SIMD, zero-alloc | iou consumes **two attention rows at the same timestep** (real head vs surrogate — functional-equivalence gate for head substitution, τ≈0.4). LpWM consumes **the same entity's state at consecutive ticks** — a temporal signal. Kernel identical, semantic disjoint | Kernel ships; new consumer = #693 | #693 |
| Support instability as regime event | **PARTIAL, wrong domain** — `katgpt-spectral/stiff_anomaly/stability.rs` (EigenvalueTracker: window-Jaccard over **eigenvalue sets** vs frozen baseline, z-score gate); KARC surprise (`karc_surprise_spikes_on_regime_change` — **forecast-error** based) | stiff_anomaly: spectral aggregate vs baseline, not per-entity consecutive supports. KARC: prediction residual, needs a running forecaster. ICT/R142 branching: **JS-divergence of K sampled action distributions** — policy-side decision points, costs K samples/tick. LpWM: state-side regime flips, costs ONE iou call. Four detectors, four different signals — the cheap state-side one is absent | **YES** — `1 − iou(z_t, z_{t+1}) > θ` (+ hysteresis/debounce) | **#693** |
| Mode-factored state layout | **NEAREST** — `DemonstratedSkill::covered_mask` (u8 bitmask support + separate magnitude fields, wire-persisted v2, BLAKE3-committed); TJS `SpecialistMask` (trained support-size); `entmax_1p5` (sparse non-negative attention); `SparseTaskVector` | covered_mask is an AUTHORED bitmask over discrete ranks — support chosen by design, not emergent from a non-negative code. No per-NPC *continuous* latent state (HLA/emotion/belief) is read through a support/magnitude split today; `NpcEmotionScalars` is 5 dense scalars | **YES** — accessors `support()`/`magnitudes()` over any non-neg f32 state (ReLu/clamp bridge for signed states) | #693 |
| One-hot linearization theorem | **EMPIRICAL PRECEDENT** — `crowd_attention.rs` per-zone linear W tick + affine region feedback; `zone_gating` per-zone (τ,β) tiers; BFCP per-region rank-1; `ConPwl` piecewise-affine value | Our zone-linear dynamics shipped without the covering-radius footing; Prop 1 + Corollary O(N^{-1/d}) is the *justification lens* (and the curse-of-dimensionality caveat matches our d≤3 DEC guidance) | Lens, not mechanism — cite in docs | doc-note |
| Sparse-code training (RDMReg) | **PARTIAL** — riir-train `tjs_sparse_loss` (trains support SIZE toward target), dist-guard erank/gaussianity audits; `gaussianity_probe` (R498) | tjs pins the support *size*; RDMReg matches the full marginal *shape* (Rectified Laplace). Different tools, same intent | Training half — optional, gated (see §Trained half) | not filed |
| TJ slowness prior | **RUNTIME DUALS SHIP** — `decay_confidence(Δt,λ)=σ(−λΔt)` (belief fade), `persistence_eta` low-pass (1/η timescale), EMA sites | Paper's loss is training-time; our slowness is runtime decay. The *insight* — support should be stable within a regime, flip AT transitions — is the consolidation-timing dual (R352 F4) | runtime composition | R352 |
| LTV low-rank gated correction | **EXACT SHAPE** — `edge_lora::SigmoidGate` (`Σ σ(uᵢ·h)·edgeᵢ(h)`, the house sigmoid-not-softmax rule); `latent_functor/rank_k` affine runtime ops | covered; LpWM's twist (zero-init U ⇒ LTV ⊇ LTI at init) is a training detail | — | covered |

**Reverse-grep (documented gaps this fills):** none of the four detector gaps above are TODO-pinned (they are absent, not deferred) — this note + #693 is the record. The closest documented deferral is `multi_tick_band.rs` orthogonal-signal note (consumed by R504).

---

## Novelty Gate (published prior art — searches run via subagent, reference list ground-truthed)

- **RDMReg** = the authors' own Rectified LpJEPA (arXiv:2602.01456, Feb 2026) — cited; LpWM is the world-model application.
- **Prop 1** ≈ Pola/Girard/Tabuada 2008 approximately-bisimilar symbolic models (cited as "closely related"); **switched Koopman** (CDC 2023, uncited) already embodies "discrete mode indicator → linear dynamics"; Korda–Mezić Koopman MPC (cited).
- **Discrete/sparse WM latents**: DreamerV2 categorical states (cited), Variational Sparse Gating (cited) — known art.
- **Soft-Jaccard loss**: standard segmentation practice (Jaccard Metric Losses, NeurIPS 2023, uncited); measure = Ružička 1958 (cited).
- **Slowness prior**: Slow Feature Analysis 2002 (**uncited root** — a scholarship gap in the paper); temporal straightening 2603.12231 (cited); PLDM smoothness (cited).
- **Mode-factored support/magnitude**: discrete+continuous *factorized latents* are crowded (Joint-DVAE 2017, DisCo-Diff, HyAR, THICK — all as separate heads/groups, all uncited). **No prior art found** for the mode/continuous split emerging in a single distributed non-negative code with no categorical grouping — the paper's genuine novelty.
- **Concurrent crowd** (uncited, within ~6 months): SCALE (2608.16287, same month), RC-aux (2605.07278), Temporal-Distance JEPA (2607.25337), GeoWorld (2602.23058), VLWM (2606.21775).

**Q1 fails for the components; Q3 (product selling point for the paper's own claims in our domain) is modest. Not Super-GOAT.** Our *fusion* (below) is novel-to-us but is a cheap composition over shipped parts — honestly a Gain.

---

## Distillation & Fusion

1. **Support-instability regime detector (katgpt-rs #693, the modelless half).** `instability(z_t, z_{t+1}) = 1 − iou(z_t, z_{t+1})` over any non-negative per-entity latent state, with debounce (N-tick window) and hysteresis — a THIRD regime-transition mechanism beside authored FSMs and forecast-error surprise (KARC), and the cheap state-side sibling of ICT's policy-side branching (R142/R270, which costs K samples/tick). Composes the already-shipped `iou` kernel with a consecutive-tick consumer. Quality claim (does it fire on real regime flips in OUR states?) **unproven — the #693 PoC exists to answer exactly that** (§3.6 low-confidence path).
2. **Regime-gated cognition (riir-ai R352, the private half).** The paper's headline — sparse codes let LINEAR predictors suffice — inverts into a compute gate: while an NPC's latent support is STABLE, its dynamics are locally linear → the cheap cognition tier suffices; escalation (CLR vote, HLA evolve, re-estimation, MCTS) fires exactly on support flips. This is the missing-cheap-signal complement to R142's expensive JS-divergence gate and the thermal-LOD/think-budget family (R136/R339).
3. **Mode-factored reading of existing state (structural).** Read `NeuronShard.style_weights[64]` / HLA codes through support/magnitude: support = which facets active (discrete, Jaccard-comparable, cheap prefilter before Clifford-wedge KNN), magnitudes = intensity (cosine). Merge criterion for `ShardCompactor`: merge most support-aligned pairs first. Doc-level guidance in R352.
4. **Consolidation timing (riir-neuron-db, noted).** TJ's insight dualized: support instability LOW over a window = regime settled = the right moment for a sleep-cycle consolidation/freeze; support flip = invalidate pending consolidation. Noted in R352; needs a consumer before filing.
5. **Zone-linear footing (doc-note).** Prop 1 gives `crowd_attention`'s per-zone W a covering-radius error bound and names the curse (d≤3 fine, high-dim shards no) — consistent with our DEC guidance. Cite where zone-linear docs live.
6. **Trained half (optional, gated).** "Sparsity lowers drafter capacity" could ride NextLat belief-drafter training (add RDMReg-style marginal matching; compare drafter capacity). NOT filed: evidence domain is pixel world models; `tjs_sparse_loss` already trains support sparsity in-repo; no consumer pull. Reopen trigger: NextLat capacity becomes the binding cost.

**Honest scope notes.** Our latent states are not guaranteed non-negative today (HLA scalars are sigmoid ∈ (0,1) — dense, fine; DEC cochains signed — need clamp/|x| bridge); the paper's codes are 30–65% active (not ultra-sparse) at D=192–4096; PushT/Wall/Piecewise are pixel domains — transfer to symbolic/3D-vector game state is the open question the PoC addresses. Prop 1's O(N^{-1/d}) is a curse-laden idealization (the paper says so).
