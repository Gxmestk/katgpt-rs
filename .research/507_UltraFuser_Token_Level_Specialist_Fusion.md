# Research 507: UltraFuser — Token-Level Output Fusion of Specialized Models

> **Source:** "Mastering Text, Code and Math Simultaneously via Fusing Highly Specialized Language Models" — [arXiv:2403.08281](https://arxiv.org/abs/2403.08281) — Ding, Chen, Cui, Lv, Zhao, Xie, Zhou, Liu, Sun (Tsinghua), Mar 2024
> **Date:** 2026-08-25
> **Status:** Active — distillation complete; training half filed as riir-train Plan 353
> **Related Research:** 126 (MoA token-level activation gates), 161 (dMoE block vs token routing), 246 (manifold power-iteration router), 253/254 (SwiR continuous router fusion / MUX bandit arm), 302 (FAME per-entity fixed-weight MoE), 453 (variable-rank domain experts / `pick_domain`), 059 (MoE + speculative decode co-design); riir-ai 128 (zone functor gating), 158 (committed personality blend)
> **Related Plans:** riir-train Plan 353 (edge_lora gate training, two-stage ULTRAFUSER recipe)
> **Cross-ref (riir-train):** Plan 353 — the model-based half (edge_lora Phase 4 gate training)
> **Classification:** Public

---

## TL;DR

ULTRAFUSER fuses three *already-specialized* 13B LLMs (text/code/math) at the **output-logit level** through a small per-token gating net, trained in two stages: gate-only first (specialists frozen), then joint fine-tune, with batch-level class-balanced sampling replacing MoE balance losses. For this stack the paper's value is threefold: (1) the strongest published **external evidence for the freeze/thaw-over-fine-tuning mandate** (directly fine-tuning specialists on mixed data *severely degrades them* — CodeLlama drops on every benchmark); (2) a **granularity law** — gate cadence must match signal interleaving cadence (domains interleave *within* one sequence; sample-level routing loses); (3) a **training recipe that unblocks edge_lora Phase 4** (gates are defined but never trained today) — two-stage warm-up + freeze-mask + stratified sampling, affordable at <1 GPU-hr on the 4090.

**Distilled for katgpt-rs (modelless, inference-time):** the trained gate is redundant — the paper shows it converges to domain-affinity weights with no explicit specialization mechanism, so it can be **constructed** (`w_k = sigmoid(dot(z, d_k)/τ)` over committed domain directions) rather than trained, in the exact `CommittedFieldBlend` committed-once shape. The paper's own co-activation data (on math data `w_code = 0.39` vs `w_math = 0.43`) is evidence **for** the house sigmoid-additive rule and **against** its softmax gate: the domains genuinely co-activate, and softmax forces them to compete.

---

## 1. Paper Core Findings

1. **Token-level output-logit gating.** A shared 2-layer linear+ReLU gate `g` reads each specialist's last hidden state per token; `w(i) = Softmax(g(h_text) : g(h_code) : g(h_math))`; final logits `= w(i)·(o_text : o_code : o_math)` per token. All specialists stay dense at inference (vLLM multi-instance logits fusion).
2. **Two-stage training.** Stage 1 trains ONLY the gate (N1=400 steps, specialists frozen) — protects specialist capability from early-stage bad gradients. Stage 2 joint fine-tune (N2 steps, cosine lr 2e-5).
3. **Data-level balancing, not loss-level.** Every batch carries equal instances per domain (n=64/32). Hypothesis: a highly specialized model already produces lower loss on its domain, so balance the data, not the loss. Measured: improves both performance AND checkpoint-to-checkpoint stability (std 2.80→1.96 TruthfulQA, 4.92→2.74 HumanEval).
4. **Token-level beats sample-level.** Domains interleave within one sequence (code inside text docs; math ≈ code) — a sample-level selector cannot exploit this. Case studies show per-token weight shifts between prose and equation spans.
5. **The negative result (most load-bearing for us).** Directly further-tuning a highly-specialized model on mixed-domain data *severely harms its expertise*: CodeLlama+FurtherTune degrades on every benchmark (AlpacaEval 69.2→17.9); the fusion structure (frozen specialists + output-level composition) is what protects capability.
6. **Emergent domain-affinity gates.** With NO explicit specialization mechanism, the gate converges to domain-aligned weights (text data: 0.45/0.29/0.26; code: 0.23/0.59/0.18; math: 0.18/0.39/0.43) — math and code co-activate almost equally.
7. Fused model beats every specialist on average (47.48 vs 36.78 best single) and beats every further-tuned single specialist (MT-Bench overall 7.02 vs 6.62 best).

## 2. Vocabulary Translation

| Paper term | Codebase equivalent(s) |
|---|---|
| token-level gating | per-op / per-tick gate evaluation (vs `pick_domain`'s per-entity argmax); `FusionArm` per-tick emission fusion |
| specialist | frozen archetype field (`CommittedFieldBlend` `f_k`); edge_lora cross-game edge; zone expert bundle |
| fused logits | additive sigmoid blend `Σ σ(π_k/τ)·f_k(z)`; `FusionArm` max/sum/mean/spectral |
| two-stage gate training | evidence warm-up gate (`KarcRegimeMux` cold-start); Beta-posterior trust before opening a route |
| balanced sampler | stratified trajectory mix (self_evolve store); game-stratified mini-batches (edge_lora trainer — absent today) |
| output-level fusion (not parameter merge) | freeze/thaw + adapter composition (house mandate); NOT model soups |

## 3. Path 0 Decomposition (training-target decomposition)

| Component | Coverage | Extraction (computable without GD?) |
|---|---|---|
| Gated output fusion `o = Σ w_k·o_k` | **Partial** — `CommittedFieldBlend` (per-entity, deterministic π), edge_lora `sigmoid_gate.rs` (per-call, trained-but-unwired), `FusionArm` (per-tick) | **YES** — dot+sigmoid domain-affinity gate over committed directions; sigmoid-additive per house rule |
| Trained gate net g | None trains one at inference | **YES** — replaced by construction (paper's own finding 6: converges to affinity) |
| Two-stage gate warm-up | Partial — `KarcRegimeMux` evidence warm-up; FAME commit-once | Modelless analog = evidence-before-trust (Beta posterior); the *trained* form is edge_lora Phase 4 → Plan 353 |
| Class-balanced sampling | None in edge_lora trainer (sequential chunks) | **YES** as principle — stratified round-robin; also a Plan 353 recipe item |
| Token vs sample granularity law | Partial — dMoE note 161 discusses; nothing ships per-op | **YES** — ordering law: gate cadence ≥ signal interleaving cadence |
| 3×13B dense logit fusion | No (doesn't fit 24GB; not our product shape) | NO — honest scope: transfer is the *recipe*, not the scale |

## 4. Distillation — modelless components (ranked)

1. **Domain-affinity gate via dot+sigmoid** (open-primitive candidate, katgpt-core shape). `w_k = sigmoid(dot(z, d_k)/τ)` over BLAKE3-committed domain directions — the deterministic construction the paper's finding 6 proves sufficient. Hosts beside `CommittedFieldBlend` (default-on) / `pick_domain` (argmax form, opt-in). GOAT if built: routing parity vs `pick_domain` on labeled traces + zero-alloc. *No consumer today — fusion idea, novelty TBD; do not build without one (Issue 528 precedent).*
2. **Granularity law.** Gate evaluation cadence must match signal interleaving cadence — per-NPC per-tick satisfies the law for cognition (sub-ops within a tick already route through distinct substrates); per-*sample* (per-query) does not for interleaved streams. Design principle for any future routing surface; external evidence for the dMoE note-161 trade-off.
3. **Co-activation prior.** The measured math↔code co-activation (0.39 vs 0.43) justifies initializing blend priors from domain-centroid dot products rather than one-hot — a committed table, not a learned one. Also the strongest external data point **for** sigmoid-additive over softmax: co-activating specialists are exactly what softmax suppresses (codified as riir-clippy kernel corpus rule `additive-sigmoid-gates-not-softmax-routing`).
4. **Fusion-not-finetune protection (cite, don't implement).** Finding 5 is the published proof of the freeze/thaw-over-fine-tuning mandate: output-level composition of frozen specialists ≥ destructive mixed fine-tuning, measured across 7 benchmarks. Reference evidence for the constraint doc, not new code.
5. **Class-balanced scheduling over balance losses.** For self-adaptive loops (self_evolve trajectory mix, EMA direction updates): balance the *evidence mix*, add no regularization. The paper's stability numbers (std halves) are the supporting citation.
6. **Softmax-gate ceiling → spectral interpolation.** The paper's softmax gate sits between winner-take-all and mean; our `FusionArm::Spectral` (`λmax(diag + c·offdiag) − c`) is the continuous knob over the same spectrum with the sandwich bound `mean ≤ fused ≤ max` — structurally immune to the above-ceiling failure the raw-Sum arm measured (riir-mmorpg Bench 027). Fusion partner, not a port.

## 5. Fusion (paper × shipped substrate)

- **× `CommittedFieldBlend` + Research 302 (FAME):** the paper is the trained-gate counterpart of our committed blend — same shape (`Σ w_k·f_k` over frozen specialists), different gate provenance (trained vs committed-once). FAME's "per-entity MoE with fixed routing weights" gains published evidence that the *fixed* form captures most of the value (finding 6).
- **× edge_lora `sigmoid_gate.rs`:** the paper's two-stage recipe is exactly what edge_lora Phase 4 needs — and the paper's negative result predicts the current warm-start mutation hazard (specialist edges copied then mutated under mixed data). → **riir-train Plan 353**.
- **× `FusionArm` (Bench 027/028):** paper's softmax gate ≈ a point on the spectral arm's c-axis; the Sum-arm failure we measured is the paper's implicit "why not sum" — softmax normalization was their fix, the spectral bound is ours.
- **× dMoE (161) / `pick_domain` (453):** the granularity ladder now has external evidence: sample-level (MUX bandit, `pick_domain`) → per-tick (FusionArm) → per-token (paper). Our per-NPC-per-tick default is defensible; per-query routing of interleaved streams is the measured loser.

## 6. Adversarial Panel (three-track, mandated — abstract carries fine-tuning)

- **No-GD advocate:** 7 extractable components (§4); headline = "the trained components are redundant — the gate converges to domain-affinity, a dot product recovers it"; highest-value pair = dot+sigmoid gate + freeze/thaw citation.
- **Model-based advocate (code ground truth verified):** `sigmoid_gate.rs` L22 — "Joint training with edge weights is Phase 4 — here we only define storage and the compose kernel"; `sgd_step` touches only `trainer.edges`; `training.rs` uses sequential unstratified `chunks`; warm-start copies specialist edges then mutates them under mixed data (the paper's degradation exposure, live). Recipe → Plan 353: gate-only warm-up (<1 GPU-hr, unblocks Phase 4 correctly), freeze-mask as control arm, game-stratified sampler (<1 GPU-hr), per-domain templates deferred (no live consumer).
- **Coordinator merge:** both tracks win — modelless half recorded here (§4), model-based half filed as Plan 353. No advocate finding discarded.

## 7. Prior Art (§4 search — published)

Combination novel as of Mar 2024 (agent-verified): closest priors are **Branch-Train-Merge/LM-Blend** (NeurIPS'22 — logit ensembling but *fixed* coefficients, shared-seed branches), **LLM-Blender** (ACL'23 — sequence-level generative fusion, extra decode pass), **FuseLLM** (ICLR'24 — parameter-level, specialists discarded), classic **MoE gating** (token-level but intra-model over homogeneous FFN experts with balance losses). Follow-ups confirm the seam: DLLG (arXiv:2606.04378, token-wise fusion over frozen experts), FusionRoute (arXiv:2601.05106), Weak-to-Strong logits fusion (arXiv:2406.15480). Novelty for *us* is therefore not the mechanism (our cousins cover the shape) but the recipe + evidence.

## 8. Verdict

**Gain** — incremental, useful; actionable delta filed.

- **Not Super-GOAT:** Q1 fails (strong shipped cousins: `CommittedFieldBlend`, edge_lora gates, FusionArm; published prior art above). Q2 fails (per-op fusion is a granularity refinement, not a new capability class). Q3 fails ("NPCs fuse specialist outputs per-op" is not a distinct selling point vs the committed blend). Q4 partial.
- **Not Pass:** the core mechanism does NOT ship at the paper's granularity with a trained gate, AND the model-based advocate found a documented gap the paper unblocks (edge_lora Phase 4 unwired — `sigmoid_gate.rs` L22). Actionable = Gain.
- **MOAT gate:** katgpt-rs note (generic primitive + evidence) ✔; training recipe → riir-train (active moat) ✔; no riir-ai guide (no Super-GOAT).

**Dual-track contribution:** (a) modelless — dot+sigmoid domain-affinity gate construction, granularity law, co-activation prior, freeze/thaw citation (this note); (b) model-based — two-stage gate training + freeze-mask + stratified sampling for edge_lora (riir-train Plan 353, <1 GPU-hr core / 2–4 hrs with self-play data gen on the 4090).

**Honest scope:** 3×13B dense logit fusion does not fit 24GB and is not our product shape. The transfer is the recipe and the gate math at edge_lora scale, where the gating composition is the same shape.
