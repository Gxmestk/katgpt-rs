# Research 463: Moka Weights → Freeze/Thaw — Does Format Conversion Unlock the Rejected Levers?

> **Source:** internal question (2026-07-31), grounded in
> [`moka_head_to_head.md`](../.docs/06_game_arenas/moka_head_to_head.md) §"Investigated and rejected" (lines 249-264).
> **Date:** 2026-07-31
> **Status:** Done — verdict Gain, PoC tracked in Issue 565.
> **Related Research:** 110 (Ciot / PlasmaPath ternary), 132 (LoRAPrune), 202 (QAT Infusion), 229 (ProgramAsWeights F4 SpecAdapter), 062 (SHINE — context-to-LoRA hypernetwork)
> **Related Plans:** 148 (PlasmaPath ternary SIMD matvec), 025 (LoraPair raw/lora hot-swap), 563 (Go Moka baseline PoC — native port), 565 (Moka WASM browser + wasmi comparison)
> **Related Docs:** [`moka_head_to_head.md`](../.docs/06_game_arenas/moka_head_to_head.md) (the complete record — architecture, PUCT results, int8 path, rejection table), [`go_arena.md`](../.docs/06_game_arenas/go_arena.md) (Go Arena overview — 6 player tiers, Plan 565 real-browser results, wasmi ladder)
> **Cross-ref (riir-ai):** [Proposal 008](../../riir-ai/.proposals/008_go_gemma_arena.md) (Go Gemma Arena — Gemma 2 2B as a Go player; the concrete "larger model" where the quant-error-LoRA primitive might actually work, per §6), [Proposal 006](../../riir-ai/.proposals/006_gemma_latent_steering_bridge.md) (latent steering bridge — a DIFFERENT modelless mechanism: residual-stream direction injection, not weight quantization compensation)
> **Cross-ref (riir-train):** [Plan 084](../../riir-train/.plans/084_go_lora_training_to_benchmarks.md) (Go LoRA training pipeline — `train_go.rs` GPU backprop + AdamW for the katgpt-rs transformer stack; the TRAINED counterpart to this note's MODELLESS approach. Currently weak: 4% Top-1 move accuracy, 979 ELO on 20-game self-play. Cannot apply to Moka — different architecture: transformer linear layers vs CNN conv kernels, zero shared code paths confirmed by grep. This is the riir-train infrastructure that the "better weights" path from `moka_head_to_head.md` defers to.)
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

#### 2.4.1 The Small-Kernel Parameter Paradox (why rank-r LoRA fails on small CNNs)

*Source: Gemini consultation, 2026-07-31. The math below is verified correct.*

Low-rank error compensation behaves radically differently on small CNNs vs
LLMs because of the **parameter overhead ratio**. The rank-r LoRA adds
`r·(out_dim + in_dim)` parameters to correct a weight matrix of
`out_dim × in_dim` parameters. The overhead ratio is:

`overhead = r·(out + in) / (out × in)`

| Layer type | Weight shape | Weight params | Rank-8 LoRA params | Overhead |
|---|---|---|---|---|
| **LLM linear** (4096→4096) | [4096 × 4096] | 16,777,216 | 65,536 | **0.39%** |
| **Moka conv** (32→32, 3×3 kernel) | [32 × 288] | 9,216 | 2,560 | **27.8%** |

At 27.8% overhead, the rank-8 LoRA adds **two extra dense matvecs** per layer
(`B·x` is [8×288]·[288] = 2,304 MACs, then `A·(B·x)` is [32×8]·[8] = 256 MACs
= 2,560 total) to compensate a conv that itself costs 9,216 MACs. That's a
**27.8% compute overhead** on top of the ternary speedup (which is ~5× per
MAC). The net effect: ternary (5× faster) − LoRA overhead (1.28× slower) ≈
3.9× net speedup, but with significant accuracy risk. Compare to int8 (1.39×
faster, 95% win rate, already DEFAULT-ON) — the ternary+LoRA path needs to
beat that bar.

**This is the structural reason the PoC is predicted to fail on Moka.** The
low-rank inductive bias ("trained weights live near a low-dimensional manifold")
is weak on a 105K-param CNN — the error matrix `E` is likely near-full-rank
because the network is too small to have a low-dimensional weight structure.

#### 2.4.2 Output-Space SVD vs Weight-Space SVD (data-aware correction)

*Source: Gemini consultation, 2026-07-31. Refinement of the §2.4 SVD-LoRA.*

Naive weight-space SVD (§2.4 above) minimizes `||E - A·B||_F` — treating all
weight errors equally. But weights with high magnitude might multiply
activations near zero (irrelevant), while small weights might multiply huge
activation spikes (critical). The **output-space** formulation is strictly
better: minimize the error on actual outputs.

**Data-aware reduced-rank regression** (Izenman 1975):

Given a calibration set `X` [in_dim × N_cal] of N_cal=64 board positions
(no labels needed — just inputs):

1. Compute output error on calibration: `E_out = E · X` [out_dim × N_cal]
2. SVD: `E_out = U · Σ · V^T`, take top-r: `U_r` [out_dim × r]
3. Project the full error onto the top-r output directions:
   - `A = U_r` [out_dim × r]
   - `B = U_r^T · E` [r × in_dim]
4. At inference: `correction(x) = A · (B · x) = U_r · U_r^T · E · x`

This projects `E·x` onto the principal directions of output error **as measured
on the calibration distribution**. It's the optimal rank-r correction in the
output L2 sense, not the weight L2 sense.

**Modelless compliance:** zero backprop, zero optimizer states — one matmul
(`E·X`) + one SVD + one matmul (`U_r^T·E`), all closed-form. The calibration
set is a one-time ~10ms linear algebra pass over 64 board states. Our codebase
already has this pattern: `rt_turbo/calibration.rs`, `fpcg_goat_gate.rs::
build_calibration_set`, `hydra_budget.rs::run_logit_lens_calibration`,
`causal_head_importance` — all use offline calibration passes. This fits
naturally.

**Why it's better than weight-space SVD:** captures activation-weighted error
spikes. If a weight error is in a direction the calibration inputs never
activate, output-space SVD correctly assigns it zero rank budget. Weight-space
SVD wastes rank budget on irrelevant directions.

**Why it still faces the Small-Kernel Parameter Paradox (§2.4.1):** the overhead
ratio is the same (27.8% at rank-8). Output-space SVD chooses the rank budget
*better* but doesn't reduce the *amount* of budget needed. On a near-full-rank
error matrix, even optimally-chosen rank-8 might not recover enough accuracy.

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

### 2.6 D4 Dihedral Symmetry Ensembling at PUCT Leaves — modelless alternative

*Source: Gemini consultation, 2026-07-31.*

The 9×9 Go board has 8-fold dihedral symmetry (`D4`: 4 rotations × 2
reflections). A well-trained Go policy SHOULD be approximately D4-invariant:
if you rotate the board 90°, the policy should rotate accordingly. Quantization
noise (especially aggressive ternary/binary) breaks this invariance slightly —
the policy becomes subtly biased toward certain orientations.

**The technique (test-time symmetry averaging):**
1. At a PUCT leaf on state `s`, pick 2 random group elements `g1, g2 ∈ D4`.
2. Compute `π1 = g1⁻¹ · f(g1 · s)` and `π2 = g2⁻¹ · f(g2 · s)`
   (transform board → forward → inverse-transform policy).
3. Average: `π_leaf = (π1 + π2) / 2`.

This dampens quantization jitter without adding model parameters or breaking
the modelless mandate. It's the Go-board analog of test-time augmentation (TTA)
in computer vision.

**Grep result:** NOT shipped in our Go code. The `D4` hits in `rat_bridge/
dilated_kv.rs` are KV-cache dilation strides (`DilationConfig::D1/D4/D16/D64`),
NOT board dihedral symmetry. No hits for `dihedral`, `rotate.*board`,
`reflect.*board`, `symmetry_averag`. This is genuinely novel for our stack.

**Why it might work:** error-compensated quantization (PlasmaPath) has a
DETERMINISTIC error pattern per row — it's not random noise, it's structured
bias. D4 averaging exploits the fact that the TRUE policy is D4-symmetric while
the quantization bias is NOT (it depends on the weight row ordering, which has
no reason to respect board symmetry). Averaging over orientations cancels the
asymmetric component of the bias.

**Why it might NOT work (honest concerns):**
1. **PUCT already averages.** Budget=200 does 200 leaf evaluations. If the D4
   symmetry breaking is random across positions, PUCT's multi-leaf averaging
   already dampens it. D4 per-leaf averaging only helps if the bias is
   SYSTEMATIC within a single leaf evaluation (which it may be).
2. **Cost: 2× forward passes per leaf.** Either double the total compute
   (~80ms → ~160ms/move at budget=200) or halve the effective search depth
   (200 leaves → 100 unique leaves). Both hurt.
3. **Moka was trained with D4 augmentation.** The policy is already approximately
   D4-invariant at f32 precision. Quantization adds a small perturbation, but
   it may be within the policy's natural noise tolerance.

**Verdict: Gain (alternative candidate).** Worth testing in the PoC as a
SEPARATE strategy from LoRA — it's orthogonal (corrects symmetry breaking, not
weight error). Could combine with LoRA (LoRA corrects weight error + D4
averaging corrects residual symmetry breaking). The PoC should measure D4
alone, LoRA alone, and D4+LoRA combined.

### 2.7 Top-K Sparse Residual Bypass — third modelless alternative

*Source: Gemini consultation, 2026-07-31.*

Because 3×3 conv error matrices are small and often full-rank (per §2.4.1),
SVD is inefficient — it spends rank budget on all directions equally. An
alternative: store only the **top-K worst-quantized elements** explicitly and
do a sparse matvec.

**The technique:**
1. Compute per-element error: `E[i,j] = W[i,j] - dequant(W_q)[i,j]`
2. Select top-5% by `|E[i,j]|` (the worst quantization errors).
3. Store as COO/CSR sparse matrix `S` [out_dim × in_dim, 5% non-zero].
4. At inference: `y = W_q · x + S · x` (dense quantized matvec + sparse correction).

**Parameter comparison (Moka 32×288 conv, rank-8 vs top-5%):**

| Strategy | Params | MACs | Overhead vs conv |
|---|---|---|---|
| Rank-8 LoRA (dense) | 2,560 | 2,560 | 27.8% |
| Top-5% sparse (COO) | ~460 values + 460 row/col indices = ~1,380 elements | 460 MACs + 460 gathers | 5.0% MACs |

The sparse path has **5.6× fewer MACs** than the dense LoRA. BUT it has
gather/scatter overhead that dense LoRA doesn't — on ARM SIMD and WASM,
random-access gathers into the input vector `x` are significantly slower than
contiguous reads. The crossover point depends on SIMD width and cache behavior.

**Grep result:** NOT shipped. The `top_k` hits in our codebase are all KV-cache
block routing (`BlockTopKRouter`, `PerGroupTopKRouter`), not sparse weight
matvec. No COO/CSR matvec in the weight path. Genuinely novel.

**Why it might work:** for small CNNs where the error is concentrated in a few
outlier weights (the classic "activation outlier" problem in LLM quantization,
but for weights), a sparse correction targeting just those outliers might
recover most of the accuracy at 5% the MAC cost. This is the weight-space
analog of GPTQ's outlier-aware quantization.

**Why it might NOT work:**
1. **Gather overhead.** On ARM NEON, a gather operation (`x[col_idx]`) costs
   ~5-10 cycles vs 1 cycle for a contiguous read. At 460 gathers, that's
   ~2,300-4,600 cycles of overhead, potentially exceeding the 2,100 MAC savings.
2. **The error might not be concentrated.** If the error is uniformly spread
   (no outlier structure), top-5% captures only 5% of the total error — not
   enough to matter.
3. **WASM is worse for sparse.** WASM's SIMD model has no native gather; the
   JIT must emulate it with scalar loads, making sparse even slower relative
   to dense.

**Verdict: Gain (third candidate).** Worth testing in the PoC alongside LoRA
and D4 averaging. The honest prediction: dense LoRA wins on SIMD hardware
(contiguous access), sparse wins only if the error has strong outlier structure
AND the hardware has fast gather (which ARM NEON partially does, WASM doesn't).

### 2.8 CNN→Transformer Latent Bridge — the deeper gap (not covered by this note)

*Identified during the Research 463 review conversation (2026-08-01). This is
the root cause of WHY Research 463 exists: we can't load Moka's CNN into the
transformer, so this note investigated freeze/thaw format conversion as a
workaround. The bridge itself is a distinct research direction, tracked here
as a gap, not a candidate.*

**The architectural reality:**

katgpt-rs provides TWO layers to Moka:

| Layer | What | Moka uses it? |
|---|---|---|
| **SIMD primitives** (`katgpt_types::simd::simd_dot_f32`) | Hand-written NEON/AVX2 dot product kernel | ✅ YES — 8.7× speedup, the "katgpt primitive" in the head-to-head doc |
| **Transformer engine** (`TransformerWeights`, `forward`, attention, KV cache) | Token-sequence inference engine | ❌ NO — Moka is a CNN, has zero attention layers |

Moka's forward pass (`moka_net.rs::forward_with_scratch`) is **self-contained**:
it borrows `simd_dot_f32` for conv/linear acceleration, but the computation
graph (conv→relu→residual→pool→linear→tanh) has zero overlap with the
transformer graph (embedding→attention→mlp→norm→KV cache). The CNN and
transformer are completely disjoint code paths.

**The gap:** `forward_with_scratch` returns only `[policy, value]` — the
82-dim policy logits and 1-dim value scalar. The **intermediate feature
maps** (e.g., the `[32, 9, 9]` activation after block 6) are computed and
**thrown away**. These intermediate maps ARE latent vectors that COULD be
bridged into the transformer's residual stream — but currently aren't exposed.

**The bridge (LLaVA-for-Go pattern):**

```
Moka CNN intermediate feature map [32, 9, 9]
      ↓ reshape to [81, 32]  (81 spatial positions × 32 channels = 81 "tokens")
      ↓ project to [81, d_model]  (the bridge)
      ↓ inject into transformer residual stream (forward_with_steering)
Transformer/Gemma does attention + reasoning on Moka's "Go vision"
      ↓
Policy [82] + Value [1]  (or natural-language move reasoning)
```

This is structurally the LLaVA / Flamingo pattern: vision encoder (Moka CNN)
→ projection → language model (Gemma/transformer). The projection has two paths:

| Path | Method | Modelless? | Status |
|---|---|---|---|
| Trained (LLaVA-style) | Learn projection via backprop (riir-train) | ❌ → riir-train | Not started |
| Deterministic | PCA / random projection / `cross_resolution_transport` (Plan 310, DEFAULT-ON) | ✅ modelless | Not started |

**Why this is NOT a freeze/thaw mechanism (and thus out of scope for this note's PoC):**

- Freeze/thaw (§2.4-2.7) operates at the WEIGHT level — correcting quantization
  errors in Moka's conv kernels. It keeps Moka's CNN as a self-contained black box.
- The latent bridge operates at the ACTIVATION level — tapping intermediate
  feature maps and projecting them into a different architecture's latent space.
  It opens the black box.

These are different problems. The quant-error-LoRA PoC (Issue 565) tests the
weight-level corrections; the latent bridge is a separate research question.

**What would be needed to pursue this:**

1. **Expose intermediate feature maps** — modify `moka_net.rs` to optionally
   return block-N output alongside `[policy, value]` (gated behind a `research`
   feature, off for production WASM build).
2. **Projection layer** — `cross_resolution_transport` (Plan 310) projects
   between different-dimensional latent spaces. A deterministic PCA projection
   on a calibration set of Go positions gives a modelless initial bridge.
3. **Consumer** — `forward_with_steering` (Proposal 006) injects vectors into
   the residual stream. The projected Moka features become "steering fields."
4. **Quality PoC** — does the transformer/Gemma actually produce BETTER Go
   decisions when given Moka's intermediate features vs just the policy/value
   output? This is the load-bearing question.

**Verdict: separate research direction, not a Research 463 candidate.** This
note's scope is weight-space freeze/thaw mechanisms. The latent bridge is an
activation-space mechanism that requires opening Moka's black box. Tracked as
a gap here so it's not lost; a future Research note (or Issue) should formalize
the PoC if a consumer materializes (e.g., Proposal 008's Gemma Go Arena needs
better Go understanding than text-based parse-fallback provides).

## 3. Verdict

**Tier: Gain** (for the candidate family — 4 modelless quantization-compensation strategies).

The format conversion itself is **Pass** (mechanical repackaging, no new capability).
The freeze/thaw ecosystem enables a FAMILY of modelless quantization-compensation
candidates worth a defend-wrong PoC:

| # | Strategy | Corrects | Overhead | Prediction |
|---|---|---|---|---|
| A | Weight-space SVD-LoRA (§2.4) | Weight error | 27.8% params + 2 dense matvecs | Likely FAIL (Small-Kernel Paradox) |
| B | Output-space SVD-LoRA (§2.4.2) | Output-weighted error | Same overhead, better rank choice | Marginal improvement over A |
| C | D4 symmetry averaging (§2.6) | Symmetry breaking | 2× forward passes per leaf | Uncertain — PUCT already averages |
| D | Top-K sparse bypass (§2.7) | Worst-element errors | 5% MACs + gather overhead | Uncertain — depends on outlier structure |

The PoC (Issue 565, updated) tests all 4 strategies head-to-head against the
int8 baseline. The honest prediction: none individually beats int8 on this
105K-param network, but the COMBINATION (e.g., D4 averaging + sparse bypass)
might surprise. Either way, the PoC's value is negative knowledge + reusable
substrate.

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
- **No CNN→Transformer Latent Bridge.** The deepest unexplored direction (§2.8) — expose Moka's intermediate feature maps + project into the transformer residual stream via `cross_resolution_transport`. This is an activation-space mechanism (opening the CNN black box), not a weight-space mechanism (the scope of this note). It's the root cause of why Research 463 exists at all: we can't load the CNN into the transformer, so this note tried freeze/thaw format conversion instead. The bridge is a separate research question that needs its own PoC if a consumer materializes.

## 6. Honest Prediction (pre-PoC)

The PoC will **likely fail G5** (win-rate) on this specific 105K-param network, because:

1. The int8 path is already within noise of f32 (95% native / 85% wasmi vs 100% f32 — within the n=20 binomial band, [Bench 565](../.benchmarks/)).
2. The ternary path's quality loss is too large for a rank-r LoRA to fully recover on a tiny network ([Bench 044](../.benchmarks/044_plasma_path_goat.md): cosine 0.77 random / ≥0.92 real NN).
3. The LoRA matvec overhead eats the ternary speedup.

**But the PoC is still worth running** because:
- If it surprises (G5 passes) → BinaryPlasma unblocked, a genuine modelless gain.
- If it fails as predicted → we've empirically confirmed the quantization floor for small CNNs, and the `quantization_error_lora` primitive ships as reusable substrate for larger models where the error manifold is genuinely low-rank. The concrete instance: **Gemma 2 2B** ([riir-ai Proposal 008](../../riir-ai/.proposals/008_go_gemma_arena.md), the Go Gemma Arena) — a 2B-param LLM where the Small-Kernel Parameter Paradox (§2.4.1) does NOT apply (rank-8 LoRA on a 4096×4096 linear layer = 0.39% overhead vs Moka's 27.8%). If Gemma is ever aggressively quantized (Q4 → Q2 for edge deployment), the output-space SVD-LoRA (§2.4.2) is the right compensation tool. That's a speculative cross-repo follow-up, not actionable until (a) the primitive ships + (b) a consumer needs aggressive Gemma quantization.
- The PoC artifact stays as a permanent regression check (per §3.6 defend-wrong protocol).

**This is the honest "go wild" answer:** one strong candidate, honestly
predicted to fail, worth running for negative knowledge + substrate. The
other 10 levers stay rejected for fundamental reasons that freeze/thaw
cannot fix.

## 7. PoC Addendum (2026-08-01)

The Issue 565 PoC ran T1-T7 + T12 (G1 cosine, G2 latency, Small-Kernel Paradox
rank sweep). G5 (win-rate) deferred — needs PUCT integration (see Issue 565 §PoC Results). Results below revise the pre-PoC prediction.

### T12 — Small-Kernel Paradox: PARTIALLY confirmed (not as bad as predicted)

Pre-PoC prediction: rank-8 captures <40% of error energy (near-full-rank).
**Actual: rank-8 captures 51.1%** overall (energy-weighted). The error matrix
has SOME low-rank structure — enough that rank-16+ recovers >65%. The Paradox
is real (not low-rank) but weaker than predicted (not full-rank either).

### G1 — Strategy A (weight-space SVD) PASSES the G1 gate at rank-16+

| Strategy | Cosine vs f32 | Δ vs B2 |
|---|---|---|
| B2 (ternary, no correction) | 0.9706 | — |
| A (wSVD rank-16) | 0.9888 | +0.018 |
| A (wSVD rank-32) | 0.9939 | **+0.023** ← G1 gate PASS |
| B (data-aware SVD) | ~0.90 | −0.06 (WORSE — PoC calibration artifact) |
| D (sparse bypass) | 0.91–0.96 | −0.03 to −0.06 (WORSE — real result) |

**Surprise finding:** Strategy A at rank-16+ genuinely improves the ternary
forward pass beyond the G1 ≥0.02 target. The pre-PoC prediction ("rank-8 LoRA
fails on small CNNs") was too pessimistic — at rank-16+ the correction works.
The cost is 27.8% param overhead at rank-8 (higher at rank-16/32), which is
the Small-Kernel Paradox manifesting as a COST issue, not a QUALITY issue.

**Strategy B's negative result is now CONFIRMED REAL** (G1-B follow-up,
2026-08-01). The initial PoC used truncated board features as calibration
(known artifact). A proper activation-based calibration — capturing actual
im2col patches + flat activations via `forward_collecting_activations`
(26,688 vectors across 60 layers, subsampled to 512/layer) — improved
Strategy B by ~0.02 cosine over truncated features. But B STILL HURTS
(Δ≈−0.05 vs ternary baseline). The negative result is no longer a PoC
artifact. **Strategy A (weight-space SVD, calibration-free) remains the
winner.**

The finding contradicts the §2.4.2 prediction that data-aware SVD would be
strictly better than weight-space SVD. On small networks (105K params), the
weight structure dominates: the intrinsic error structure of the weight
matrix generalizes better to unseen inputs than calibration-conditioned
output error. Data-aware SVD overfits to the calibration distribution. This
is the opposite of what happens in large LLMs (GPTQ/OBQ), suggesting the
Small-Kernel Parameter Paradox applies to the data-aware axis too: small
networks lack the redundancy that makes data-aware compression effective.

**Strategy D's negative result is real.** Sparse outlier correction
destabilizes the output when the error is distributed (T12 confirmed 51%
at rank-8, not concentrated in outliers). The gather-overhead concern
(§2.7) is moot — the quality issue dominates.

### What the PoC ships

1. **`QuantErrorLora` primitive** in katgpt-core (`quant_error_lora` feature,
   opt-in) — `from_error` (weight-space SVD) + `from_error_data_aware`
   (output-space SVD) + `SparseErrorBypass` (top-K COO). Reusable substrate
   for larger models. 7 unit tests PASS.
2. **`research` feature** in katgpt-moka-wasm — weight accessors + corrected
   forward pass. Native-only (off for WASM build).
3. **4-test PoC bench** in riir-poc (`tests/quant_error_lora_poc.rs`) —
   permanent regression check per the defend-wrong protocol.

### Revised prediction for G5

The G1 result (0.9939 at rank-32) is more promising than the pre-PoC prediction
(0.95 at best). But G5 (win-rate vs int8 at 95%) is still predicted to FAIL
because: (1) near-perfect cosine ≠ identical move selection under PUCT search
(tiny policy perturbations change the argmax for close moves); (2) the ternary
path offers no speedup over int8 on this hardware (int8×int8 SDOT is native;
ternary SIMD would need a custom kernel + the rank-32 LoRA is f32 matmul);
(3) even if G5 passes, the ternary+LoRA path is more complex than int8 for
zero net gain. G5 is deferred — wire it only if a future consumer needs the
ternary+LoRA path for a reason int8 doesn't satisfy (e.g., extreme model
compression for edge deployment where ternary's 16× storage win matters).
