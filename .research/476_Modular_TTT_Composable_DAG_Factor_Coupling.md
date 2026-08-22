# Research 476: Modular TTT — Composable DAG Framework + Factor Coupling Theorem for Deep Fast Weights

> Source: [Modular TTT: Rethinking Test-Time Training as Composable Modules](https://arxiv.org/pdf/2608.07110)
> Authors: Bohao Tang, Zhen Qin (project lead), Yuqi Pan, Zheng Li, Pengfei Liu, Ya Zhang
> Affiliation: Shanghai Jiao Tong University · Shanghai Innovation Institute · ByteDance Seed
> Date: 2026-08-10
> Code: https://github.com/ByteDance-Seed/Modular-TTT

## TL;DR

Modular TTT factorizes Test-Time Training (TTT) into a DAG of composable primitives (Linear, Gate, Norm, Act, Add, Mul + 4 loss functions) and automatically composes their train-view forward / train-view backward / causal query-view rules into the full graph-level TTT computation. Systematic ablation at 160M / 410M / 1.45B on 10B–100B tokens yields a clear empirical picture of the TTT design space, and a theorem explaining why deep fast-weight networks are hard to train.

**For our stack:** the empirical sweet spot (small-lr init η₀=10⁻³, scalar decay, single SiLU nonlinearity, shallow linear learner) **confirms** choices we already make across GDN2 (Plan 070), HLA (Plan 028), DendriticGate (Plan 260), and our sigmoid-everywhere discipline. The genuinely novel + actionable contribution is the **factor coupling theorem** (Proposition 5 + Appendix C.4) — a documented mathematical constraint against multiplicative composition of fast weights, which warns against future designs that would stack LoRA adapters multiplicatively (`W₁·W₂`) or build deep product-form fast-weight MLPs.

**Verdict: Gain.** No new primitive to ship (we don't ship TTT inner loops — modelless constraint #1). Two actionable items: (1) document the config sweet spot as defaults for any future TTT/HLA distillation (riir-train Plan 066), (2) record the factor coupling theorem as a design constraint against product-form weight compositions anywhere in the 7-repo stack.

---

## Paper Summary

### The Modular TTT Framework

A TTT layer is a sequence mixer whose hidden state is the **fast weights** of an inner learner, updated by an inner learning rule (one or more gradient steps on a self-supervised objective). Prior TTT variants (TTT-Linear, TTT-MLP, Titans, LaCT, ATLAS, TNT) hard-code each variant separately, making it hard to isolate the role of each component.

Modular TTT represents the inner learner as a **directed acyclic graph** `G = (V, E)` where each node is a primitive (linear map, activation, residual add, normalization) and each edge is a tensor dependency. For each primitive, three rules are registered:

| Rule | What it computes |
|---|---|
| `ϕ_train` (train-view forward) | Forward pass on the key `K`, producing reconstruction `V̂` |
| `ϕ_train-bwd` (train-view backward) | Analytic gradient + local parameter update `Δθ` (no nested autodiff) |
| `ϕ_query` (query-view forward) | Causal readout on query `Q` using train-view activations + parameter updates |

The framework automatically composes these rules over the DAG in topological order (Algorithm 1). This eliminates hand-deriving a new global update rule for every TTT variant.

### Primitive Operators (Table 1)

| Primitive | Train-view | Backward | Query-view |
|---|---|---|---|
| **Linear** | `V̂ = KW` | `dW = K̂ᵀdV̂` | `O = QW − Tril(QK̂ᵀ ⊙ M) dV̂`, `W ← W − dW` |
| **Gate** | `V̂ = K diag(W)` | `dW = sum(K̂ ⊙ dV̂)` | `O = Q diag(W) − Q ⊙ cumsum(K̂ ⊙ dV̂)` |
| **Norm** (RMSNorm) | `v̂ = k/σ` | analytic (see Eq. 10) | shared |
| **Act** | `V̂ = f(K)` | `dK = dV̂ ⊙ f'(K)` | shared |
| **Add** | `V̂ = K₁+K₂` | `dK₁=dK₂=dV̂` | shared |
| **Mul** | `V̂ = K₁⊙K₂` | `dK₁=dV̂⊙K₂`, etc. | shared |

Loss functions: Inner Product (`−⟨V̂,V⟩/s`), MSE (`‖V̂−V‖²/(2s)`), L1, RMSE.

---

## Key Empirical Findings (160M / 410M, 10B tokens)

The systematic ablation produces a clear, robust picture:

### 1. Loss function (Table 2)
- **MSE ≈ Inner Product** (both competitive; differ by < 0.001 across 5 seeds)
- L1 and RMSE substantially worse (remove magnitude information from gradient)

### 2. Learning-rate initialization (Table 3)
- **Small-lr init** (`η₀ = 10⁻³`, achieved via `b = log(√(p/(1−p)))`, `p = η₀/2 ≈ 5×10⁻⁴`, `b ≈ −7.60`) consistently beats standard init (`η₀ ≈ 1`)
- Mechanism (Proposition 2): for MSE, the homogeneous update matrix `A = I − Kᵀ diag(η) K` has eigenvalues `1 − λᵢ(H)` where `H = Kᵀ diag(η) K ⪰ 0`. If `λ_max(H) > 2`, some eigenvalue of `A` leaves the unit disk → instability. Small `η₀` bounds `λ_max(H) ≤ η₀ ‖K‖²₂`.
- **Mean scaling** (`s = c`, divides gradient by chunk length) is an alternative that achieves the same initial scale control but locks the factor fixed; small-lr init lets the learned lr predictor adapt per-token.

### 3. Decay (Table 4)
- **Scalar decay** recovers most of the gain from vector decay at negligible cost
- Vector decay: ~25% throughput drop, +3 GB peak memory
- No decay: clearly weakest (no forgetting of stale context)

### 4. Nonlinearity (Table 5)
- **SiLU and GELU consistently improve** over plain Linear (gated write `ΔW = Kᵀ(D_o ⊙ σ'(U))`)
- Norm (RMSNorm) is mixed — the `1/σ(z)` factor in its gradient amplifies noise when `σ(z)` is small (Eq. 10)
- Best trade-off: Linear + SiLU

### 5. Deep fast-weight memory (Table 6) — THE NEGATIVE RESULT
- **Deeper graph learners do NOT surpass the shallow frontier**
- Two-layer linear: 3.1265 vs shallow Linear-SiLU 3.0205 (+0.106 gap)
- Linear-SiLU-Linear-SiLU: 3.1156 (still worse)
- Norm-containing deep variants either diverge or stay behind
- Residual connections and gating (SwiGLU) provide little measurable benefit

### Scale-up (410M, 1.45B, 100B tokens)
- Modular TTT (Linear + SiLU, small-lr, scalar decay) achieves performance **comparable to Gated DeltaNet (GDN)** on perplexity + multiple-choice benchmarks
- Containment-style tasks (SWDE, SQuAD, FDA) remain challenging — fixed-state TTT still has limitations for precise recall
- Throughput: 2.2×–3.3× improvement over official TTT implementation

---

## The Factor Coupling Theorem (Proposition 5 + Appendix C.4)

This is the paper's genuinely novel mathematical contribution — a structural explanation for the deep fast-weight negative result.

**Setup:** Consider a product-form fast learner `Y = XW⁽¹⁾W⁽²⁾`. Define `W = W⁽¹⁾W⁽²⁾` as the effective fast weight. Note `W⁽¹⁾W⁽²⁾ = (cW⁽¹⁾)(c⁻¹W⁽²⁾)` for any nonzero scalar `c` — the represented function `f(X) = XW` is invariant under this rescaling.

**The update is NOT invariant:** Under one-step TTT, the induced update on the effective fast weight becomes (Eq. 14):

```
(W⁽¹⁾ − Δ[W⁽²⁾]ᵀ)(W⁽²⁾ − [W⁽¹⁾]ᵀΔ)
  = W⁽¹⁾W⁽²⁾
    − Δ[W⁽²⁾]ᵀW⁽²⁾ − W⁽¹⁾[W⁽¹⁾]ᵀΔ
    + Δ[W⁽²⁾]ᵀ[W⁽¹⁾]ᵀΔ
```

Under rescaling `W⁽¹⁾ → cW⁽¹⁾`, `W⁽²⁾ → c⁻¹W⁽²⁾`, the dominant terms change:
- Large `c`: dominated by `−c² W⁽¹⁾[W⁽¹⁾]ᵀΔ`
- Small `c`: dominated by `−c⁻² Δ[W⁽²⁾]ᵀW⁽²⁾`

**Conclusion:** Two factorizations representing the **same effective fast weight** induce **substantially different TTT update directions**. Optimization must not only learn the correct effective `W`, but also discover a factorization that induces favorable update dynamics — an additional degree of freedom that makes deep TTT memory hard to optimize.

**Zero-init trap (Proposition 5):** For depth-`L ≥ 2` product-form linear learners, if all fast factors initialize to zero, the train-view gradient of every factor is exactly zero (since `ΔWⱼ = Aⱼ₋₁ᵀ D Bⱼ₊₁ᵀ` and either `Aⱼ₋₁` or `Bⱼ₊₁` contains a zero factor). The zero state persists across chunks unless externally perturbed. By contrast, a single linear fast learner with `W = 0` has update `ΔW = KᵀD`, which is nonzero.

This is a generalization of classical deep linear network theory (Arora, Cohen, Hu, Luo — cited [1, 2, 4, 25]) to the TTT setting.

---

## Mapping to Our Stack

### What This Paper CONFIRMS (Already-Shipped Design Choices)

| Paper finding | Our shipped equivalent | Status |
|---|---|---|
| **Sigmoid learning rate** (`η = 2σ(β+b)`) | Sigmoid everywhere (AGENTS.md mandate; `fast_sigmoid`, all gates, all projections) | ✅ Already aligned |
| **Scalar decay** best efficiency/perf trade-off | GDN2 scalar decay (Plan 070 E2 "Channel-Wise Decay") | ✅ Already shipped |
| **Single SiLU nonlinearity** after linear | SiLU in channel mixers, DendriticGate voltage sensitivity | ✅ Already aligned |
| **Shallow > Deep** fast-weight networks | Single-layer GDN2 fast weights; flat `style_weights[64]` in NeuronShard | ✅ Already aligned |
| **Avoid deep product-form** fast-weight MLPs | NeuronShard dendritic branches are NOT product-form — they're a single flat `[f32; 64]` reinterpreted as `[16 proximal, 24 intermediate, 24 distal]` with branch-level gating (see `riir-neuron-db/src/shard/dendritic/`) | ✅ Safe by construction |
| **MSE ≈ Inner Product** for fast-weight loss | We don't train inner loops (modelless constraint #1); n/a | ✅ N/A |

### What This Paper WARNS Against (Design Constraint for Future Work)

The **factor coupling theorem** is a documented mathematical constraint against:

1. **Multiplicative LoRA composition** (`W₁ · W₂` instead of `W₁ + W₂`). Our `PersonalityWeightedComposition` (Plan 297) and `CommittedFieldBlend` (Plan 321) are **additive** compositions — safe. **If a future plan proposes multiplicative composition, the factor coupling theorem predicts optimization difficulty.**

2. **Deep product-form fast-weight MLPs** in any new shard/branch design. Our current dendritic branches are flat (single-layer reinterpretation). **If a future plan proposes a 2+ layer product-form shard layout, the factor coupling theorem predicts training instability** (zero-init trap if factors start at zero, factor-coupled updates even with Gaussian init).

3. **Stacked fast-weight layers** (multi-layer TTT). We don't ship these. **If a future plan proposes stacking GDN2/HLA layers multiplicatively, this theorem applies.**

### Config Defaults for Future TTT/HLA Training (riir-train Plan 066)

If/when riir-train distills HLA or trains TTT-style layers (Plan 066 SDPA→HLA distillation, Plan 059 G-Zero GRPO/DPO), the paper's empirical sweet spot is the canonical starting recipe:

| Hyperparameter | Recommended default | Paper evidence |
|---|---|---|
| Inner learning rate init | `η₀ = 10⁻³` (via `b ≈ −7.60` in `η = 2σ(β+b)`) | Table 3: 0.16–0.49 loss reduction |
| Decay | Scalar | Table 4: recovers most of vector-decay gain at ~0 cost |
| Nonlinearity | Single SiLU after linear | Table 5: +0.015–0.011 loss reduction |
| Depth | **Shallow (1 layer)** | Table 6: every deep variant worse |
| Loss | MSE or Inner Product (equivalent) | Table 2, Table 17 (5-seed) |
| Chunk size | 256 (robust across 128–512) | Table 25 |
| Fast-weight init | Gaussian (std 0.02), NOT zero | Table 27 (zero-init traps deep learners) |
| Mean scaling | Disabled (small-lr init handles scale) | Table 15 |

This is directly applicable to **riir-train Plan 066** (`distill_attention.rs` — SDPA→HLA distillation) and any future TTT-style training in the model-based track.

---

## Distillation / Fusion Ideas

### Fusion A: Factor Coupling as a Composition Safety Gate

The factor coupling theorem could become a **static design constraint checker** in `katgpt-rs` — a small utility that, given a composition spec (additive vs multiplicative, depth), returns whether the factor coupling warning applies. This would be a modelless design-time tool, not a runtime primitive.

- **Where:** `katgpt-rs/src/composition/` (new module) or extend `PersonalityWeightedComposition` with a `verify_no_factor_coupling()` method
- **Value:** Prevents future plans from accidentally introducing multiplicative LoRA composition or deep product-form shards
- **Priority:** P3 (defensive; we don't currently violate this)

### Fusion B: Modular DAG Framework for Linear Attention Variants

The DAG composition pattern (register primitives + auto-compose train-view/query-view) is a methodology we could adopt for organizing our linear attention family (HLA, AHLA, GDN, GDN2, Mamba, RWKV). Currently each variant has its own kernel; a modular framework would let us ablate them systematically the way Modular TTT does.

- **Where:** `katgpt-rs/crates/katgpt-linear/` (hypothetical new crate) or extend `katgpt-core::hla`
- **Value:** Faster iteration on linear attention variants; systematic ablation
- **Priority:** P3 (engineering improvement; not a perf or quality gain)
- **Caveat:** This is methodology, not a primitive. The paper itself frames Modular TTT as a framework, not a model.

### Fusion C: Cross-Reference to riir-train Plan 066 Recipe

The paper's config sweet spot should be recorded in `riir-train/.docs/02_pipelines/` as the canonical TTT/HLA training recipe, cross-referenced from Plan 066. This is a documentation task, not a code task.

---

## What Does NOT Map

| Paper concept | Why it doesn't apply |
|---|---|
| **TTT inner gradient loop** (n-step inner GD on fast weights) | Modelless constraint #1 — we don't do backprop at inference. Our fast-weight updates are deterministic δ-rule (GDN2) or freeze/thaw swaps, not inner GD. |
| **Training 410M / 1.45B models on 100B tokens** | That's riir-train territory (Plan 318 SFT, Plan 066 HLA distillation). The findings transfer as config defaults, not as a runtime primitive. |
| **Chunkwise parallel training (WY form)** | Training-time optimization; our inference path uses the recurrent form. |
| **Contained retrieval limitations (RULER NIAH at 8k)** | Known limitation of fixed-state TTT; we use full KV cache for precise recall when needed. |
| **LaCT SwiGLU + sliding-window hybrid** | Different architecture; we don't ship LaCT. |

---

## Verdict: Gain

**Tier:** Gain (actionable improvements, no new primitive).

**Rationale:**
- The paper's empirical sweet spot **confirms** our existing design choices (sigmoid lr, scalar decay, SiLU, shallow). No regression risk.
- The **factor coupling theorem** is a genuinely novel mathematical contribution that provides a documented design constraint against future product-form weight compositions. This is actionable as a design rule, even though we don't currently violate it.
- The config defaults are directly applicable to riir-train Plan 066 (HLA distillation) when that pipeline runs.
- No new primitive to ship in katgpt-rs (we don't ship TTT inner loops — modelless constraint).

**Why not GOAT:** No measurable perf/quality gain on our shipped primitives. The findings are confirmatory + cautionary, not a new capability.

**Why not Super-GOAT:** No new behavior class, no product selling point, no force multiplier across ≥2 pillars. This is a methodology paper with engineering findings.

**Why not PASS:** The factor coupling theorem is novel and actionable as a design constraint. The config sweet spot is a useful documented default for future training. Both belong in the research record.

---

## Actionable Items (Tracked)

1. **[Design constraint — documented here]** The factor coupling theorem (Proposition 5 + Appendix C.4) is recorded as a standing design rule: **do not compose fast weights multiplicatively** (`W₁·W₂`); prefer additive composition (`W₁+W₂`). Applies to: LoRA composition, shard layout, fast-weight stacking. Current stack is compliant (verified: `PersonalityWeightedComposition`, `CommittedFieldBlend`, NeuronShard dendritic branches are all additive or flat).

2. **[Cross-reference to riir-train]** The config sweet spot table above should be recorded in `riir-train/.docs/02_pipelines/` when Plan 066 (HLA distillation) or any TTT-style training runs. The recipe: small-lr init (η₀=10⁻³), scalar decay, single SiLU, shallow (1 layer), MSE or Inner Product loss, chunk size 256, Gaussian fast-weight init (std 0.02).

3. **[No code change required]** Our shipped primitives (GDN2, HLA, DendriticGate, NeuronShard dendritic branches) are already aligned with the paper's findings. No feature flag, no benchmark, no GOAT gate needed.

---

## Relationship to Existing Research

| Note | Relationship |
|---|---|
| **R019 (TTT-Discover)** | Different paper (discovery RL via test-time LoRA). Same TTT family but different concern — R019 is about RL training at test time; this paper is about TTT as a sequence model architecture. |
| **R028 (HLA)** | Sibling architecture — HLA is higher-order linear attention, TTT is inner-learning-based. Both are linear-attention-family. The paper's findings about scalar decay + SiLU + shallow apply to HLA distillation. |
| **R070 (GDN2)** | The paper's primary baseline. Modular TTT achieves "comparable to GDN" — confirms GDN2 is a strong baseline. Our GDN2 scalar decay (Plan 070 E2) is exactly what the paper recommends. |
| **R073 (LT2 Looped Transformers)** | Related — looped transformers are a form of implicit depth. The factor coupling theorem explains why explicit depth in fast weights is hard; LT2's looping is a different escape from the same constraint. |
| **R097 (Training-Free Looped Transformers)** | Modelless analog of the looped-transformer idea. Confirms our modelless-first approach. |
| **R230 (Semiseparable State Space Duality)** | SSD is another linear-attention-family member. The Modular TTT framework could organize SSD alongside HLA/GDN/TTT. |
| **R260 (DendriticGate)** | Our NMDA-inspired gate uses entropy × coincidence × sigmoid — exactly the "gated write" pattern the paper finds beneficial (`ΔW = Kᵀ(D_o ⊙ σ'(U))`). Confirms the design. |
| **Plan 066 (riir-train, HLA distillation)** | The config sweet spot table is the canonical starting recipe for this pipeline. |

---

## References

```bibtex
@article{tang2026modularttt,
  title   = {Modular TTT: Rethinking Test-Time Training as Composable Modules},
  author  = {Tang, Bohao and Qin, Zhen and Pan, Yuqi and Li, Zheng and Liu, Pengfei and Zhang, Ya},
  journal = {arXiv preprint arXiv:2608.07110},
  year    = {2026},
  note    = {ByteDance Seed + SJTU; code at https://github.com/ByteDance-Seed/Modular-TTT}
}
```

Key prior art cited by the paper:
- **TTT (Sun et al. 2024)** — arXiv:2407.04620 — the original TTT-Linear/TTT-MLP
- **Titans (Behrouz et al. 2025)** — arXiv:2501.00663 — neural memory at test time
- **LaCT (Zhang et al. 2025)** — arXiv:2505.23884 — "Test-time training done right"
- **ATLAS (Behrouz et al. 2025)** — arXiv:2505.23735 — optimal memorization
- **TNT (Li et al. 2025)** — arXiv:2511.07343 — chunkwise TTT training
- **Gated DeltaNet (Yang et al. 2025)** — the baseline Modular TTT matches
- **Deep linear network theory (Arora et al. 2018, 2019)** — the factor coupling basis
