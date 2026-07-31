# Research 463: Moka Weights → Freeze/Thaw — Does Format Conversion Unlock the Rejected Levers?

> **Source:** internal question (2026-07-31), grounded in
> [`moka_head_to_head.md`](../.docs/06_game_arenas/moka_head_to_head.md) §"Investigated and rejected" (lines 249-264).
> **Date:** 2026-07-31
> **Status:** Done — verdict Gain, PoC tracked in Issue 565.
> **Related Research:** 110 (Ciot / PlasmaPath ternary), 132 (LoRAPrune), 202 (QAT Infusion), 229 (ProgramAsWeights F4 SpecAdapter), 062 (SHINE — context-to-LoRA hypernetwork)
> **Related Plans:** 148 (PlasmaPath ternary SIMD matvec), 025 (LoraPair raw/lora hot-swap), 563 (Go Moka baseline PoC — native port), 565 (Moka WASM browser + wasmi comparison)
> **Related Docs:** [`moka_head_to_head.md`](../.docs/06_game_arenas/moka_head_to_head.md) (the complete record — architecture, PUCT results, int8 path, rejection table), [`go_arena.md`](../.docs/06_game_arenas/go_arena.md) (Go Arena overview — 6 player tiers, Plan 565 real-browser results, wasmi ladder)
> **Related Benchmarks:** 204 (Opening Book vs Moka — negative, monotone hurt), 205 (PUCT search vs Moka — 98% win at budget=200, the headline result), 044 (PlasmaPath GOAT — ternary cosine 0.77 random / ≥0.92 real NN), 565 (int8×int8 vs f32 — 95% native / 85% wasmi, 1.39× faster, DEFAULT-ON)
> **Related Issues:** 565 (defend-wrong PoC for quantization-compensating reader-LoRA), 564 (Moka ANE/CoreML — 4.66× slower, negative), 206 (int8×int8 dot investigation), 207 (int8 default-on promotion — resolved+removed, commit `7da5cf76`)
> **Moka weight source:** vendored at `crates/katgpt-pruners/assets/moka/` (`go-model.bin` 113,648 bytes + `go-model.json` manifest, sha256-verified, MIT license from `github.com/millionco/moka`). Native port: `crates/katgpt-pruners/src/go/moka_net.rs`. WASM port: `crates/katgpt-moka-wasm/`.
> **PASS-Redirects (synthesis):** HyperLoRA [arXiv:2606.06154 "Amortizing Federated Adaptation: Hypernetwork Driven LoRA for Personalized Foundation Models"] — federated LoRA TRAINING method (all three components — hypernetwork generator G_φ, product-space synthesizer S_ψ, residual corrector C_ω — are LEARNED via meta-objective Eq 18). → riir-train. Does NOT apply to Moka (single network, no federation, no aggregation problem). Does NOT replace the SVD quant-error-LoRA in this note (different problem: federated aggregation bias vs single-network quantization error). The one transferable math insight (Proposition 1: factor-wise averaging produces chimeric cross-terms B_i·A_j) is already moot in our stack because our LoRA merge is ADDITIVE (`W + B·A`, confirmed in `forward_coda`), not factor-averaged. §3.5 Path 0 decomposition: G_φ has NO modelless analog (genuinely needs learning); S_ψ decomposes to SVD (= FlexLoRA baseline, which the paper beats) + learned statefulness; C_ω decomposes to least-squares residual fit + learned conditioning. Closest shipped cousin: Research 062 (SHINE — context-to-LoRA hypernetwork, also verdicted model-based/training).
> **Classification:** Public

---

## TL;DR

**Question 1 — can we convert Moka weights to our freeze/thaw format?** YES, trivially.
`NeuronShard` is a generic `#[repr(C)]` Pod; Moka's int8 weights + per-channel f32
scales fit as a blob. `MerkleFrozenEnvelope` wraps it with BLAKE3. This is mechanical
repackaging — no new capability.

**Question 2 — does conversion unlock the rejected levers?** **NO, not directly.**
Every rejection in the head-to-head table is fundamental:

| Rejection class | Why format conversion can't help |
|---|---|
| Architecture mismatch (HLA/AHLA, LT2) | CNN ≠ attention; the weights ARE conv weights, you can't apply attention math to them regardless of how the bytes are packaged |
| Can't manufacture strength (MUX) | Routing between weak players stays weak; freeze/thaw of the SAME weights produces the SAME outputs |
| Different model / needs training (LEO/QGF/DualLeoMixer) | Format conversion can't manufacture trained weights for a different architecture |
| Wrong domain shape (DDTree/Poincaré/FlowField) | Go board ≠ continuous navigation ≠ pathfinding; the domain doesn't change with storage format |
| Quantization floor (BinaryPlasma) | 1-2 bit matvec wrecks int8 accuracy; the quantization MATH is substrate-independent |
| Hardware dispatch (ANE) | Fixed dispatch overhead; storage format doesn't change silicon |
| Already better (Opening Book) | Moka's policy already plays better openings; format is irrelevant |

Freeze/thaw is a **storage + integrity mechanism** (BLAKE3 commitment, versioned
snapshot, atomic hot-swap). It changes HOW weights are stored and committed, not
WHAT they compute. A CNN forward pass is a CNN forward pass whether the weights
live in a raw `.bin` blob or a `NeuronShard`.

**BUT — the freeze/thaw ECOSYSTEM (not just the format) enables mechanisms that
were NOT in the rejected list.** One strong candidate emerges: **Quantization-
Compensating Reader-LoRA** — a modelless attempt to unblock BinaryPlasma by
recovering the quantization error that PlasmaPath currently discards, via a
deterministically constructed (SVD-closed-form, not trained) low-rank reader
adapter. This fuses PlasmaPath (Research 110) × LoraPair reader hot-swap
(Plan 025) × NeuronShard freeze. **Verdict: Gain.** A defend-wrong PoC
(Issue 565) is worth running even though the honest prediction is that it
will NOT beat the int8 path ([Bench 565](../.benchmarks/): int8×int8 is 1.39× faster
than f32 at 95% native win rate, DEFAULT-ON since commit `7da5cf76`) on this
specific network — the PoC's value is negative knowledge (confirming or
refuting the quantization floor) plus the substrate (a reusable
`quantization_error_lora` primitive).

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable insight is NOT "freeze/thaw unlocks levers" — it is "the
error signal that PlasmaPath currently DISCARDS is a free, modelless
correction substrate." Every row-wise error-compensated quantizer computes
`carry[i] = adjusted[i] - q[i]·scale` per element and throws the carry away.
A rank-r truncated SVD of the cumulative carry matrix `E = W - dequant(W_q)`
yields `E ≈ A·B` (A: [out×rank], B: [rank×in]), applicable at inference as a
reader-LoRA: `y = W_q·x + α·B·(A·x)`. This is closed-form (no backprop),
deterministic, and the corrected weights freeze as a shard. The question the
PoC settles: does the accuracy recovery justify the LoRA matvec overhead?

---

## 1. The Question, Precisely

> **Full Moka record:** [`moka_head_to_head.md`](../.docs/06_game_arenas/moka_head_to_head.md)
> (architecture diagram, all player results, int8 path, rejection table) +
> [`go_arena.md`](../.docs/06_game_arenas/go_arena.md) (Go Arena overview, Plan 565
> real-browser results, wasmi ladder). Native port: [Plan 563](../.plans/563_go_moka_baseline_poc.md).
> WASM browser comparison: [Plan 565](../.plans/565_moka_wasm_browser_wasmi_comparison.md).
> Weight source: `crates/katgpt-pruners/assets/moka/` (`go-model.bin` + `go-model.json`,
> MIT license from `github.com/millionco/moka`). Native port code:
> `crates/katgpt-pruners/src/go/moka_net.rs`. WASM port: `crates/katgpt-moka-wasm/`.

The user observed the head-to-head rejection table
([`moka_head_to_head.md`](../.docs/06_game_arenas/moka_head_to_head.md) L249-264,
also mirrored in [`go_arena.md`](../.docs/06_game_arenas/go_arena.md) §"Investigated
and rejected") and asked two things:

1. **Can we convert Moka weights to our freeze/thaw format?**
2. **After conversion, can we somehow use all the levers?**

The user explicitly wants a "go wild" research pass — "breakthrough and find new
thing even failed poc is still worth learning." This note honors that mandate
while being honest about what the format conversion actually enables.

## 2. Distillation — Honest Lever-by-Lever Audit

### 2.1 The rejected levers (does freeze/thaw change the verdict?)

| Lever | Original rejection | Does freeze/thaw change it? | New angle? |
|---|---|---|---|
| **HLA / AHLA** | Wrong architecture (CNN) | **No.** HLA is a linear-attention kernel for transformer attention layers. Moka has zero attention layers. Converting conv weights to a `NeuronShard` doesn't transmute them into attention weights. The mathematical bridge (3×3 conv ≈ degenerate local attention) exists but doesn't enable HLA's O(n) recurrence win — conv is already O(n·k²). | See §2.3 (conv-as-attention curiosity) |
| **LT2 T-pass loop** | Wrong architecture | **No.** Weight-shared loops need attention layers to iterate over. CNN has no sequence dimension to loop. | No |
| **MUX / MUX-Latent** | Can't manufacture strength | **No.** The original rejection stands. Routing between snapshots of the SAME weights produces the SAME outputs. Freeze/thaw enables VERSIONED snapshots, but without diverse perturbations there's only one snapshot. Multi-snapshot ensemble needs diverse error profiles → see §2.2. | See §2.2 (multi-snapshot PUCT) |
| **LEO / GoLeoNet / DualLeoMixer** | Different model, needs training | **No.** These need different TRAINED weights for a different network topology. Format conversion cannot manufacture trained weights. → riir-train. | No |
| **QGF** | Wrong model | **No.** Same — needs LEO/UVFA network + training. → riir-train. | No |
| **AND-OR DDTree** | Wrong domain shape | **No.** Go has no subgoal decomposition. The domain shape doesn't change with storage format. | No |
| **Poincaré Navigator** | Wrong domain shape | **No.** Continuous pose navigation, not board games. | No |
| **FlowField** | Wrong domain shape | **No.** Civ pathfinding, not Go. | No |
| **BinaryPlasma / PlasmaPath** | Would lose quality (1-2 bit wrecks int8) | **PARTIALLY.** The format doesn't help, but the freeze/thaw ECOSYSTEM (reader-LoRA hot-swap) might. See §2.4. | **YES — the one strong candidate** |
| **Apple Neural Engine (CoreML)** | 4.66× slower ([Issue 564](../.issues/564_moka_ane_coreml_inference.md)) | **No.** Hardware dispatch overhead. Storage format doesn't change silicon. | No |
| **Opening Book** | Hurts monotonically ([Bench 204](../.benchmarks/204_opening_book_vs_moka_negative.md)) | **No.** Moka's policy already plays better 9×9 openings. | No |

**Bottom line on question 2:** format conversion does NOT unlock any of the
rejected levers. 10/11 are unchanged. The 11th (BinaryPlasma) has a PARTIAL
new angle that doesn't come from the format itself but from the reader-LoRA
hot-swap ecosystem.

### 2.2 Multi-snapshot PUCT (MUX reborn) — honest Pass

The MUX rejection: "routing between 0%-scoring players still scores 0%." But Moka
is the 50% baseline, not 0%. Could freeze/thaw produce multiple diverse snapshots
for PUCT leaf routing?

**The honest problem:** we only have ONE set of Moka weights. To get diverse
snapshots, we'd perturb them deterministically (different quantization
granularities, channel dropout, scale jitter). Deterministic perturbations of
the same weights produce **correlated** error profiles. An ensemble of
correlated-error networks doesn't diversify — it averages toward the mean.

More importantly: **PUCT already extracts ~98% of the available strength**
([Bench 205](../.benchmarks/205_puct_search_vs_moka_win.md): PUCT budget=200
beats Moka greedy 98.0% native, n=100). The marginal gain from leaf-routinging
between 2-3 perturbed snapshots is likely within noise of the single-snapshot
baseline. The PUCT search itself IS the ensemble mechanism — it averages over
200 simulations per move.

**Verdict: Pass.** PUCT is the ensemble; adding snapshot routing is redundant.

### 2.3 Conv-as-Local-Attention (HLA bridge) — mathematical curiosity, no win

A 3×3 convolution over an H×W feature map IS a degenerate attention layer:
- 9 "virtual heads" (one per kernel position)
- Each head has fixed Q (the conv weight for that position), K=V=input features
- Attention weights are uniform (1/9) within the 3×3 window, 0 outside

HLA replaces softmax attention with a linear recurrence. Could we apply HLA's
machinery to this "attention interpretation" of Moka's convs?

**The honest answer: there's no win.** HLA's value is reducing O(n²) global
attention to O(n) via linear recurrence. But conv is already O(n·k²) where
k=3 — it's already local and linear. Reinterpreting it as attention doesn't
make it faster or better; it's the same computation wearing different clothes.

The only way HLA would help is if we could REPLACE the local 3×3 conv with a
GLOBAL linear attention that captures long-range dependencies the convs miss.
But that requires different weights (trained for global attention) → riir-train.

**Verdict: Pass.** Mathematical curiosity, zero modelless win.

### 2.4 Quantization-Compensating Reader-LoRA (BinaryPlasma unblock) — the one strong candidate

**The rejection:** "1-2 bit matvec would wreck int8 net accuracy (int8×int8
with per-channel scale is the floor)."

**The freeze/thaw ecosystem angle:** Plan 025 ships `LoraPair { reader, writer }`
— a raw/lora hot-swap where a **deterministically constructed** (not trained)
adapter is applied at inference time. The freeze/thaw rule
(`AGENTS.md` §modelless-first-mandate) explicitly lists raw/lora hot-swap as
one of the three allowed weight mutations.

**The insight:** PlasmaPath (Research 110, Plan 148) does row-wise
**error-compensated** ternary quantization:

```
scale = mean(|row|)
threshold = 0.5 * scale
for each weight:
    adjusted = value + carry
    if adjusted > threshold → q = +1
    elif adjusted < -threshold → q = -1
    else → q = 0
    carry = adjusted - (q * scale)   ← THIS IS DISCARDED
```

The `carry` is the per-element quantization error. PlasmaPath compensates the
NEXT element with it (error diffusion, like Floyd-Steinberg dithering), then
throws it away. **What if we kept a low-rank approximation of the cumulative
error matrix and applied it as a reader-LoRA?**

**The math:**
1. `W_q = quantize(W)` (int8 or ternary, with error-compensated carry)
2. `E = W - dequant(W_q)` (the per-element residual error matrix, [out_dim × in_dim])
3. `E ≈ A · B` where A is [out_dim × rank], B is [rank × in_dim] (truncated SVD — closed-form, no gradient descent)
4. At inference: `y = W_q · x + α · B · (A · x)` — the LoRA correction

This is **deterministically constructed** (SVD is closed-form). It's modelless
per §3.5 Path 2 (raw/lora reader hot-swap). The corrected weights freeze as a
`NeuronShard` + `MerkleFrozenEnvelope`.

**Why this might work:** the error matrix `E` for a well-trained network is
typically LOW-RANK (the trained weights live near a low-dimensional manifold —
this is the inductive bias behind LoRA itself). A rank-8 or rank-16 SVD
correction might recover most of the accuracy lost to aggressive quantization.

**Why it might NOT work (the honest prediction):**
1. **At int8 (current floor):** the error is already tiny. The LoRA overhead (2 extra matvecs per layer) likely exceeds the accuracy gain. The int8×int8 path is already 1.39× faster than f32 at 95% native win rate ([Bench 565](../.benchmarks/), [Issues 206+207](../.issues/206_int8_int8_dot_investigation.md), DEFAULT-ON since `7da5cf76`) — adding LoRA would eat that margin.
2. **At ternary/binary (where BinaryPlasma was rejected):** the error is large. PlasmaPath measured cosine 0.77 on random weights (≥0.92 on real NN weights, [Bench 044](../.benchmarks/044_plasma_path_goat.md)). A rank-r correction might recover to ~0.95, but the LoRA matvec costs real FLOPs — potentially negating the ternary speedup (ternary is ~5× faster per MAC, but the LoRA is f32 matmul).
3. **The CNN is tiny (105K params).** LoRA compensation shines on large models where the error manifold is genuinely low-rank. On a 105K-param CNN, the error might be full-rank (no low-rank structure to exploit).

**The PoC's value is negative knowledge either way.** If it works → BinaryPlasma
unblocked, a genuine modelless gain. If it fails → we've empirically confirmed
the quantization floor for this network class, and the `quantization_error_lora`
primitive still ships as reusable substrate for larger models where it might
matter more.

**Verdict: Gain.** Open primitive + defend-wrong PoC. Tracked in Issue 565.

### 2.5 DEC-Enriched Input via Stem-Conv Reader-LoRA — honest Pass (→ riir-train)

Idea: enrich Moka's 12 input planes with DEC-computed features (stone density
divergence, influence curl, harmonic territory component from
`katgpt-core/src/dec/`) and absorb the new channels via a reader-LoRA on the
stem conv (12→32 becomes 16→32, with the 4 new channel weights as the LoRA).

**The honest problem:** the "correct" weights for the new input channels are
unknown without training. A zero-initialization LoRA (new channels contribute
nothing) is a no-op. A PCA-projection LoRA (project DEC features onto the
existing output space) is arbitrary — it doesn't correspond to any learned
association between DEC features and good moves.

This angle genuinely needs riir-train to construct meaningful weights for the
new channels. The modelless path (zero-init or arbitrary projection) doesn't
add capability.

**Verdict: Pass for modelless.** The trained version (learn the 4 new channel
weights via riir-train) is a legitimate research direction but out of scope
for this workflow.

## 3. Verdict

**Tier: Gain** (for the one strong candidate — Quantization-Compensating Reader-LoRA).

The format conversion itself is **Pass** (mechanical repackaging, no new capability).
The freeze/thaw ecosystem enables one Gain-tier research candidate that attempts
to unblock a specifically-rejected lever (BinaryPlasma). The PoC's value is
honest negative knowledge + reusable substrate.

**One-line reasoning:** converting Moka weights to freeze/thaw format does NOT
unlock any of the 11 rejected levers (the rejections are fundamental —
architecture, domain, training, quantization math), BUT the reader-LoRA
hot-swap ecosystem enables one modelless attempt to recover the quantization
error PlasmaPath currently discards, which is worth a defend-wrong PoC even
though the honest prediction is failure on this specific 105K-param network.

### Why this is NOT Super-GOAT

- **Q1 (no prior art?):** NO — quantization-error LoRA compensation is
  well-known in the LLM world (QLoRA, QA-LoRA, GPTQ, AWQ all use learned
  corrections; the modelless SVD variant is a known closed-form technique).
- **Q2 (new capability class?):** NO — it's an optimization on an existing
  capability (int8 inference), not a new class of behavior.
- **Q3 (product selling point?):** WEAK — "we recover quantization error
  modellessly" is a perf claim, not a capability claim. PUCT's 98% win rate
  ([Bench 205](../.benchmarks/205_puct_search_vs_moka_win.md)) is the
  headline; this doesn't beat it.
- **Q4 (force multiplier?):** NO — it touches one primitive (PlasmaPath) and
  one consumer (Moka). Doesn't connect to ≥2 pillars.

### MOAT gate (katgpt-rs domain)

The `quantization_error_lora` primitive IS in-scope for katgpt-rs (generic
quantization-aware inference primitive, no game IP). But it's a neutral Gain
(in-scope, not pillar-level). Ship behind feature flag, track promote/demote,
do NOT overclaim moat.

## 4. The Primitive (if the PoC proceeds)

```rust
// crates/katgpt-core/src/quant_error_lora.rs (sketch)

/// A deterministically-constructed low-rank reader-LoRA that compensates
/// for the quantization error of a weight matrix.
///
/// Given W (f32 reference) and W_q (quantized), compute E = W - dequant(W_q),
/// then E ≈ A·B via truncated SVD. At inference: y = W_q·x + α·B·(A·x).
///
/// Modelless: SVD is closed-form (no gradient descent). The corrected weights
/// freeze as a NeuronShard via the freeze/thaw ecosystem.
pub struct QuantErrorLora {
    /// Down-projection [rank × in_dim]
    pub a: Vec<f32>,
    /// Up-projection [out_dim × rank]
    pub b: Vec<f32>,
    /// Scaling factor (tunable; default 1.0)
    pub alpha: f32,
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl QuantErrorLora {
    /// Construct from a reference weight matrix + its quantized form.
    /// Computes the error matrix, then its rank-r truncated SVD.
    pub fn from_error(w_ref: &[f32], w_quant: &[f32], out_dim: usize, in_dim: usize, rank: usize) -> Self { ... }

    /// Apply the correction: y += alpha * B · (A · x)
    pub fn apply_correction_into(&self, x: &[f32], y: &mut [f32]) { ... }
}
```

**Feature flag:** `quant_error_lora` (opt-in until PoC settles).

**GOAT gate (if PoC proceeds to promotion):**
- G1: does the LoRA correction reduce the cosine gap between f32 and quantized forward pass? (target: cosine(quant+lora) > cosine(quant) by ≥0.02)
- G2: latency overhead of the LoRA matvec vs the quantized forward (target: < 20% overhead)
- G3: no-regression (default + all-features clean)
- G4: alloc-free hot path (pre-allocated A/B slices)
- G5 (the load-bearing gate): win-rate of PUCT + ternary-Moka + quant-error-LoRA vs PUCT + int8-Moka (target: ≥ 90% to justify the ternary+LoRA path over the simpler int8 path). Protocol: same as [Bench 205](../.benchmarks/205_puct_search_vs_moka_win.md) (n=100, PUCT budget=200, vs Moka greedy).

## 5. What I Did NOT Propose (and why)

- **No Super-GOAT guide.** This is Gain-tier, not Super-GOAT. No private selling-point doc needed.
- **No plan.** Issue 565 tracks the PoC. If the PoC passes G1-G4, THEN open a plan for the primitive + promotion gate.
- **No riir-train deferral.** The SVD LoRA is genuinely modelless (closed-form). The only riir-train dependency is the DEC-enriched-input angle (§2.5), which is honestly Pass for modelless.
- **No Conv-as-Attention primitive.** Mathematical curiosity, zero modelless win (§2.3).

## 6. Honest Prediction (pre-PoC)

The PoC will **likely fail G5** (win-rate) on this specific 105K-param network, because:

1. The int8 path is already within noise of f32 (95% native / 85% wasmi vs 100% f32 — within the n=20 binomial band, [Bench 565](../.benchmarks/)).
2. The ternary path's quality loss is too large for a rank-r LoRA to fully recover on a tiny network ([Bench 044](../.benchmarks/044_plasma_path_goat.md): cosine 0.77 random / ≥0.92 real NN).
3. The LoRA matvec overhead eats the ternary speedup.

**But the PoC is still worth running** because:
- If it surprises (G5 passes) → BinaryPlasma unblocked, a genuine modelless gain.
- If it fails as predicted → we've empirically confirmed the quantization floor for small CNNs, and the `quantization_error_lora` primitive ships as reusable substrate for larger models (LLM weights, future game networks) where the error manifold is genuinely low-rank.
- The PoC artifact stays as a permanent regression check (per §3.6 defend-wrong protocol).

**This is the honest "go wild" answer:** one strong candidate, honestly
predicted to fail, worth running for negative knowledge + substrate. The
other 10 levers stay rejected for fundamental reasons that freeze/thaw
cannot fix.
