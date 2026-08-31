# Research 521: MINE — Mutual Information Estimation, the Modelless Fixed-Critic Extraction

**Status:** GAIN — modelless track PRIMARY (fixed-critic variational MI evaluator + permutation calibration; plan 583 filed, feature `mi_est`), model-based track SECONDARY (trained-T collapse detector + MI-max regularizer; riir-train plan 365 filed). NOT Super-GOAT (Q1: all math is richly published; Q3: no product selling point — instrumentation, not behavior).

> **Source:** Belghazi, Baratin, Rajeswar, Ozair, Bengio, Courville, Hjelm — "MINE: Mutual Information Neural Estimation", ICML 2018, [arXiv:1801.04062](https://arxiv.org/abs/1801.04062)
> **Date:** 2026-08-31
> **Workflow:** research skill — pre-flight (5 READMEs/pipelines + 5 `.research/` listings + 4 src-tree listings + §4 web prior-art + training-code grep), adversarial panel (No-GD + Model-based advocates, one spawn round, both read live code), Path 0 inventory complete, §3.6 signal-diffs run on 7 shipped candidates.
> **Related Research:** 501 (SVCCA), 498 (SIGReg → `sketched_gaussianity`), 314 (f-divergences), 315-riir-ai (membership-inference "MI" — DIFFERENT MI, see the disambiguation box), 414 (code-LLM RLVR — unrelated "MI" collision source)
> **Related Plans:** katgpt-rs 583 (modelless `mi_est`), riir-train 365 (trained-T detector + MI-max campaign)

---

## TL;DR

MINE trains a statistics network `T_θ(x,y)` by gradient ascent on the **Donsker–Varadhan bound** `I(X;Y) ≥ E_P[T] − log E_{P⊗Q}[e^T]`, with an EMA on the log-mean-exp term, and applies MI-max/IB penalties to generative + supervised training. **The training loop is the least transferable part.** The transferable cargo is the *skeleton*: every variational MI bound (DV, NWJ, InfoNCE, JS) is a **sample-evaluable functional of critic scores** — fix the critic (dot/cosine/frozen-seeded projection, all zero-parameter) and you get a modelless, one-pass, zero-alloc **MI-bound evaluator in nats**, upgradeable by a **permutation test** that is distribution-free and finite-sample exact. The stack ships MI-shaped instruments that are all *bounded and scale-free* (retrieval-accuracy proxy ∈ [0,1] in `riir-train/training/mi_proxy.rs`; sigmoid-of-InfoNCE ∈ (0,1) in `katgpt-band`; cosine in `peira_loss`) — none reports a **magnitude in nats** or a **calibrated significance**, and the retrieval proxy is structurally blind to collapse onset (reads 1.0 while MI has already fallen). The extraction: `katgpt-core` `mi_est` (DV+LOO, bound ladder, K-ladder tightness diagnostic, permutation calibration, Gaussian closed-form arm gated by the shipped `sketched_gaussianity`), consumed as the third audit axis in the dist-guard family and as an information-fidelity probe for quantization/compression surfaces (InfoQ/Nie-2025 precedent; KV-cache audit space essentially unoccupied per §4).

---

## 1. Paper core (compact)

- **Objective:** `max_θ E_{P_XY}[T_θ] − log E_{P_X⊗P_Y}[e^{T_θ}]` — gradient ascent on the DV variational form of KL; `I = D_KL(P_XY ‖ P_X⊗P_Y)`.
- **Stabilizer:** EMA of the log-mean-exp (partition-function) term — tames minibatch variance *of the training signal*.
- **Claims:** linear scaling in dim and sample size (compute-linear — NOT accuracy-linear; statistical sample complexity still grows with d), strong consistency (requires the critic family to become dense — i.e. requires the training).
- **Applications in paper:** MI-max regularization to improve adversarial generative models; Information Bottleneck (`min I(X;T) s.t. I(T;Y)`) for supervised classification.
- **Known pathologies (follow-up literature, see §4):** plug-in bound value biased **upward** ~O(eff-critic-dof/2N) (log-Jensen) — "detects MI on the null" at small N; variance explodes at high true MI; neural estimators violate self-consistency (data-processing inequality) at high dim (Song & Ermon ICLR 2020; SMILE clipping is the variance fix; Poole et al. taxonomy).

> **MI disambiguation (grep hazard recorded):** this note's MI = **mutual information** (information theory). Workspace greps for "MI" also hit **membership inference** (privacy attacks — riir-ai Research 315, Yeom 2018 loss-gap theory on KarcShard) and **morphological inflection/MI-abbreviated** terms. The two "MI"s are unrelated concepts; Research 315 is *adjacent* (information-leakage measurement) but ships no estimator and is NOT coverage for this note.

---

## 2. Path 0 inventory (training-target decomposition → disposition)

| # | Component | Modelless extraction? | Coverage in stack (signal-diff §3.6) | Disposition |
|---|---|---|---|---|
| 1 | DV bound **evaluation** with fixed T (dot/cosine/frozen-seeded proj) | YES — one pass, O(N) | NONE for the DV family; InfoNCE-family test ships (different bound, different scale) | → plan 583 Phase 1 (core) |
| 2 | LOO logmeanexp + λ-family + EMA-as-streaming-stat | YES — running statistics | none | → plan 583 Phase 1 |
| 3 | Bound ladder: NWJ, InfoNCE-K, JS from the same score matrix | YES | InfoNCE **conditional** test ships in `katgpt-band` (sigmoid-scale, caller-supplied negatives, H0-test purpose); NWJ/JS/absolute-nats InfoNCE absent | → plan 583 Phase 2 |
| 4 | K-ladder tightness diagnostic `Î(K)` saturation | YES | none (the "how much MI can this critic family even see" statistic) | → plan 583 Phase 2 |
| 5 | Permutation test (block/circular/stratified) — distribution-free p | YES — exact finite-sample under H0 | none | → plan 583 Phase 2 |
| 6 | Gaussian closed-form MI `−½Σlog(1−ρᵢ²)` under a distributional gate | YES — gate = shipped `sketched_gaussianity` (Mardia alternative deferred) | Gate ships (Issue 681 probe); the MI closed form itself does not; `svcca` computes canonical correlations but emits a similarity meter, not nats (measured 270× worse dynamic range on low-rank edges — R520 annotation) | → plan 583 Phase 3 |
| 7 | Frozen IB ratio `Î(T;Y)/Î(X;T)` as a selection criterion | YES | none | → plan 583 Phase 3 `[-]` pending consumer |
| 8 | KSG k-NN nonparametric arm | YES (d≤~8 estimator, detector beyond) | binned MI exists **test-only** in qmc tests | → `[-]` referee-role only; not a plan phase |
| 9 | Trained-T DV estimator (MINE proper) + EMA | NO — needs GD | `mi_proxy.rs` = retrieval-accuracy proxy (bounded, saturation-blind); `peira_loss.rs` = cosine MI-max (bilinear, gradient-death near its own 0.95 stop) | → riir-train plan 365 Phase 1 (instrument), Phase 2 (regularizer) |
| 10 | MI-max regularization of a generator | Partially | peira is the in-stack MI-max shape (cosine form) | → plan 365 Phase 2 (DV-critic upgrade w/ entropy-bonus control arm) |
| 11 | IB penalty training | Partially | `loss_cd_lam.rs` has only the keep side; the penalize side `I(z;ctx)` has no term and no modelless estimator | → plan 365 Phase 3 `[-]` |
| 12 | GAN improvement via MI-max | NO analog needed | no adversarial generator exists in-tree (dllm = diffusion, drafters = LoRA; grep: no GAN/discriminator training pipeline in riir-train/riir-ai) — the transferable half of this row IS row 10's recipe | **audited discard** |

Every row ends in a plan phase, a defer with condition, or an audited discard — no silent drops.

---

## 3. §3.6 Signal-diffs on the "already ships" candidates

1. **`katgpt-band::conditional_dependence_infonce`** — InfoNCE lower bound on **conditional** MI with a frozen critic fn-pointer. Diff: bound family (InfoNCE vs DV/NWJ), scale (sigmoid(nce) ∈ (0,1) vs nats), inputs (caller-supplied negatives vs permutation), purpose (H0 independence test inside band conditioning vs magnitude measurement). One level up: no caller anywhere reports MI magnitude; the containing system only needs a test statistic. **Partial coverage; the gap (nats + DV/NWJ + calibration) is real.**
2. **`riir-train/training/mi_proxy.rs`** (Plan 234) — MI proxy = in-batch retrieval **accuracy** ∈ [0,1], alert when < 0.3. Diff: rank statistic vs magnitude; structurally blind to **onset** (saturates at 1.0 while MI falls) and to **critic family** (cosine-axis-only). Model-based advocate read the code and confirmed both failure modes fall through. **The gap is documented by the instrument's own contract.**
3. **`riir-train/peira_loss.rs`** — MI-max across views via cosine of 8-dim projections; early stop at cos ≥ 0.95. Diff: bilinear critic family; gradient death exactly where it stops; no nats curve for plateau tests. Training-track consumer of plan 365, not coverage.
4. **`katgpt-core::data_probe::gaussianity` (`sketched_gaussianity`)** — projection-KS normality gate (Issue 681, SIGReg-distilled, zero-alloc). Diff: it is the **gate**, not the estimator; the Gaussian MI arm CONSUMES it (DRY — no Mardia re-implementation needed for v1; Mardia = alternative gate, deferred `[-]`).
5. **`katgpt-core::data_probe::cca` (SVCCA)** — subspace similarity; canonical correlations are the Gaussian-MI ingredient, but the emitted quantity is a bounded similarity with measured dynamic-range collapse on low-rank populations (R520 close-out #7). Diff: meter vs gated nats estimate. Complement, not coverage.
6. **qmc binned-MI (katgpt-core `speculative/qmc/tests.rs`)** — histogram MI in test code only. Diff: not shipped API, no bias correction, no calibration. Lead, not coverage.
7. **riir-ai Research 315 (KarcShard "MI" audit)** — membership-inference MI (Yeom loss-gap), no code. Different concept (see disambiguation box). Adjacent art for *leakage* audits; zero overlap with dependence estimation.

---

## 4. Published prior-art search (§4, two web agents, citations)

**Bound landscape (all trained-critic literature):** NWJ (arXiv:0809.0853 — lower variance, higher bias, dies ≳3 nats); InfoNCE/CPC (arXiv:1807.03748 — lowest variance, hard log-K ceiling); f-GAN/JS (arXiv:1606.00709); SMILE (arXiv:1906.03309 — clip-the-ratio variance fix); Poole et al. taxonomy (arXiv:**1905.06922** — all variational bounds degrade when true MI is large; interpolated I_α family). Self-consistency violations of neural MI estimators: Song & Ermon ICLR 2020 (arXiv:1910.06222); estimator benchmark: Czyż et al. NeurIPS 2023 (arXiv:2306.11078) — no estimator dominates, neural estimators degrade off-Gaussian and at high dim.

**Novelty probe (the load-bearing one):** no published work found that evaluates DV/NWJ with a **strictly fixed, never-optimized** statistics function as a single-shot diagnostic. Closest adjacencies, all distinguishable: Tschannen et al. ICLR 2020 (simple critics, still *trained* objectives); random-feature variational divergences (fixed basis, *optimized coefficients*); Zhelezniak et al. ACL 2020 (rank-correlation MI on word embeddings — different estimator family, STS purpose); MILE (logdet family); CKA family (similarity, not MI).

**Application landscape (MI as compression/quantization audit):** verdict **(b) proposed but contested, KV-cache case essentially absent.** InfoQ (AAAI 2026, arXiv:2508.04753 — MI-change layer selection for mixed precision); Nie et al. 2025 (representation MI as compression-quality metric, niche venue, no tooling); IB theory contested (Tishby 1503.02406; Shwartz-Ziv 1703.00810; Saxe ICLR 2018 — compression phase = binned-estimator artifact critique; Goldfeld 1810.05728 — the canonical *fixed* SP estimator + binning-artifact quantification); fixed-estimator training monitors exist at low dim (latent-MI dynamics 2025); RankMe (arXiv:2210.02885) = the mainstream *spectral* collapse monitor — our erank dist-guard is in this family, the MI axis is its complement. The one mature MI-fidelity field is **image QA** (IFC/VIF, IEEE TIP 2005/2006) — precedent never ported to weight/KV quantization.

**Nonparametric + closed-form (components, fully published):** KSG (Kraskov PRE 69, 066138; degrades ≳ d≈8–12, Gao AISTATS 2015); binned/adaptive-partition MI with Miller–Madow; Gaussian MI via covariance determinant / canonical correlations (textbook).

**Net:** every *component* is published; the **combination** (fixed-critic variational evaluation in nats + permutation calibration + K-ladder + quant/shard audit application) has no found prior art. That is composition-level novelty → GOAT ceiling at most, and with no product selling point (Q3 fail): **Gain.**

---

## 5. Adversarial panel (merged verdicts)

**No-GD advocate** (top extractions, with honest ledger): (1) permutation-calibrated LOO-DV/InfoNCE dot-critic dependence statistic — the p-value is the strongest export (exact, distribution-free; power, not validity, degrades with dim); (2) Mardia-gated Gaussian MI — the only arm with true ground-truth accuracy; gate must hard-fire on the zero-correlation `Y = X²` control (nonzero MI, zero correlation) or the gate is decorative; (3) frozen IB ratio as a *selection* criterion among frozen representations (noise-dim injection must strictly decrease the ratio). Ledger: plug-in up-bias O(dof/2N) → LOO default + null-calibration curve shipped with the module; DV variance explodes at high MI → report bound *spread* + K-ladder, never a bare number; EMA is display-smoothing only (bias-preserving, autocorrelation-inducing — bootstrap over raw batch values); block/circular permutation mandatory for tick streams; everything in **nats** (`MiNats` newtype — bits/nats mixing is the silent-bug class).

**Model-based advocate** (read the live files; honest waste table): trained-T wins exactly where fixed instruments structurally cannot — collapse **onset** (unbounded nats vs saturated [0,1] proxy), dependence **off the bilinear axis** (MLP critic ⊃ cosine critics), **magnitude thresholds** (IB cap in nats), **rotating critic subspaces** (PEIRA views). Wasted motion: `go_predictor` (n≈100 — critic = overfit machine; sigmoid margin already GOAT-proven superior), `loss_cd_lam`'s per-pair shaping loss (wrong objective class; SogLR deliberate). Top 2: MI-max **remedial** regularizer on the documented RAGEN-2 collapse signature (MI↓ + entropy↑ — entropy bonuses provably cannot address it; 8–16 GPU-h with the pre-registered entropy-bonus control arm), trained-T DV **detector** + EMA (4–6 GPU-h; the shared instrument unblocking #1, the PEIRA re-gate, and the IB hinge). House-rule check: the DV log-mean-exp is a max-subtracted LSE, not a softmax over a batch — composes with the sigmoid-not-softmax rule.

**Coordinator merge:** the model-based track is ranked SECONDARY per serving-envelope fit (the decision layer of the primary value — fixed-critic evaluation — runs inside the stack's µs-ms audit/diagnostic path; the trained critic is a GPU-h campaign outside it). The critic campaign CONSUMES `mi_est`'s DV core (DRY — no re-implementation), which is why the modelless plan lands first.

---

## 6. Verdict

| Question | Answer |
|---|---|
| Q1 no prior art? | Math: NO (richly published — MINE is the ancestor). Combination (fixed-critic variational diagnostic + calibration + audit application): yes at composition level. |
| Q2 new behavior class? | Marginal — a new *measurable axis* (nats + calibrated significance) over bounded proxies; not a new NPC/system capability class. |
| Q3 product selling point? | **NO** — dev-facing instrumentation; cannot finish "our NPCs do X no competitor can". |
| Q4 force multiplier? | YES — composes `sketched_gaussianity` + erank dist-guard + quant/compaction audit surfaces (≥2 pillars). |

**Tier: Gain** (not Super-GOAT: Q1-composition-only + Q3 fail). Per-track: **modelless = PRIMARY** (plan 583, katgpt-rs, feature `mi_est`, GOAT-gated consumers); **model-based = SECONDARY** (riir-train plan 365, explicitly marked, gated on 583's DV core).

**MOAT gate:** katgpt-rs — in scope (base dependence statistic next to `svcca`/`gaussianity`/`effective_rank` in `data_probe`; public primitive, no game semantics). riir-train — in scope (active moat; training recipes with GOAT gates vs the modelless baseline). riir-neuron-db / riir-ai — consumers only (reconstruction/kvarn audits; per-NPC diagnostics), cross-ref, no new notes there.

**Discards (audited):** GAN-transfer row — mechanism (MI-max on an adversarial generator) has no substrate in-tree and its transferable half is row 10; Mardia gate — DRY defer behind the shipped sketched-gaussianity gate (revisit only if the gate shows false-accepts on copula fixtures); KSG arm — referee-role only in validation tests, not a shipped estimator (curse of dimensionality, and the permutation test supersedes its "detector" use).

**Reopen/upgrade triggers:** a production consumer adopts `mi_est` for a *behavioral* decision (not just audit) → re-run the four-question gate; an IB-style cap ships in a training loss (plan 365 Phase 3) → the audit axis becomes load-bearing → re-gate for GOAT.
