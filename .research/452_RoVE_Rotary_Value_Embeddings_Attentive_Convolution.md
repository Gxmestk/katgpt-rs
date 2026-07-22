# Research 452: RoVE — Rotary Value Embeddings → Attentive Convolution

> **Source:** Alejandro García-Castellanos, Maurice Weiler, Erik J. Bekkers — *RoVE: Rotary Value Embeddings Attention for Relative Position-dependent Value Pathways* — [arXiv:2606.11275](https://arxiv.org/abs/2606.11275), Jul 2026 (AMLab UvA + MIT CSAIL)
> **Code:** [github.com/AGarciaCast/RoVE](https://github.com/AGarciaCast/RoVE)
> **Date:** 2026-07-22
> **Status:** Active — verdict GOAT; plan opened (Plan 557)
> **Related Research:** 233 (Attention Matching — KV compaction; different problem, same OV-circuit neighborhood), 431 (Wall Attention — replaces RoPE rather than extends it to values), 446 (GRAPE — provides the `PositionGroupAction` trait that RoVE consumes as its hot-path bridge), 070 (GDN2 — value pathway under linear attention), 109 (Shard — asymmetric K/V treatment), 181 (Compositional Muon — OV-circuit LoRA training), 213/271 (StillKV / Attention Matching — `apply_rope_phase_shift` on keys only)
> **Related Plans:** 173 (Wall Attention), 233→271 (Attention Matching), 322 (phase rotation), 446→159/160/161/163 (GRAPE trilogy — Rodrigues + `PositionGroupAction` + GRAPE-AP + GL(d+2) lift), 557 (this note's plan — renumbered from 452 to avoid `.plans/452_simd_lut_dequant.md` collision; `.plans/` and `.research/` are independent namespaces)
> **Classification:** Public

---

## TL;DR

RoVE is a **parameter-free modification** of RoPE attention that closes a structural asymmetry: RoPE makes the QK circuit (attention scores) position-relative but leaves the OV circuit (value pathway) position-blind — `W_V` is applied identically regardless of the token's offset from the query. RoVE additionally rotates each value `W_V x_j` by `R_j` before aggregation and inverse-rotates the output by `R_{−i}` to put it in the query's frame:

```
ỹ_i = R_{−i} Σ_{j∈N(i)} A_ij(X) · R_j · W_V · x_j
    = Σ_{j∈N(i)} A_ij(X) · (R_{j−i} W_V) · x_j
    = Σ_{j∈N(i)} A_ij(X) · ψ_{j−i} · x_j          ← attentive convolution
```

The effective kernel `ψ_δ = R_δ W_V` is an **offset-indexed block-Toeplitz family** — the same rotation family RoPE already uses on QK, now applied to the OV circuit. This converts the RoPE attention mixer from Kronecker structure (`A ⊗ W_V`) to **gated block-Toeplitz structure** (`Σ_δ (A ⊙ S_δ) ⊗ (W_U W_O R_δ W_V W_E)`) — the signature of an **attentive convolution** (Romero 2020, Fuchs 2020). Standard RoPE is recovered by the degenerate choice `ψ_δ ≡ W_V`.

**Distilled for katgpt-rs (modelless, inference-time):** the entire mechanism is parameter-free (zero new weights), FlashAttention-compatible (rotations act on V and the post-softmax output, never on the score matrix), and adds only `O(nd)` linear compute on top of the existing `O(nd²)` QKV projections — negligible. It reuses the **exact rotation family RoPE already applies to Q/K**, just applied to V and post-aggregation output. The closed form `ψ_δ = R_δ W_V` is identical to what `RopeAction::apply_at(δ, ·, ·)` already computes (Plan 446 / Issue 160). What is missing is the **hot-path wiring** that calls `apply_at` on V (and `apply_inverse_at` on the aggregated output) — GRAPE Research 446 explicitly shipped `PositionGroupAction` as a *vocabulary bridge for cold-path tools*, not a hot-path attention variant. RoVE is that hot-path variant. Plan 557 ships it.

---

## 1. Paper Core Findings

### 1.1 The asymmetry RoVE fixes

Standard RoPE (Su et al. 2024) rotates queries `q_i` by `R_i` and keys `k_j` by `R_j` before the inner product, producing `score_ij = (W_Q x_i)^T R_{j−i} (W_K x_j) / √d` — depending on positions only through the offset `δ = j − i`. The QK circuit is **shift-equivariant**. But the OV circuit is untouched: the value map `W_V` is applied identically to every key regardless of offset. Two tokens assigned equal attention weight contribute identical `W_V x_j` regardless of distance.

In Elhage et al. (2021) transformer-circuits vocabulary: RoPE biases the QK circuit toward relative positions while leaving the OV circuit **position-blind**. Unlike a convolution kernel, the channel map `W_V` carries no information about where `x_j` lies relative to the query.

### 1.2 RoVE — rotate values into the query's frame

Definition 1 (paper Eq. 3):

```
ỹ_i = R_{−i} Σ_{j∈N(i)} A_ij(X) · R_j · W_V · x_j
```

Three equivalent lenses:

| Lens | What it says |
|---|---|
| **Convolution** | RoVE turns RoPE attention into an attentive convolution (Romero 2020, Fuchs 2020) with kernel `ψ_δ = R_δ W_V`. Standard RoPE is the degenerate `ψ_δ ≡ W_V`. |
| **Matrix mixer** (Hwang et al. 2024) | RoPE mixer = `A ⊗ W_V` (Kronecker). RoVE mixer = `Σ_δ (A ⊙ S_δ) ⊗ (W_O R_δ W_V)` — gated block-Toeplitz. Each block diagonal shares the same rotated value kernel, modulated by content-dependent scalar gate `A_ij`. |
| **Local frame** (Miyato et al. 2024) | Each value `W_V x_j` is rotated from its local frame into a shared global frame by `R_j`, contributions are aggregated in the global frame, the result is rotated into the query's local frame by `R_{−i}`. The effective kernel is the frame-change composed with the learned channel map. |

### 1.3 Why it's free

- **Zero new parameters.** RoVE reuses the exact RoPE rotation family `{R_t}_{t∈ℕ}` already constructed from the same frequency schedule `ω_m = θ_0^{−2m/d}`.
- **`O(nd)` linear overhead.** Rotations act independently on `n` tokens and `d/2` channel pairs; cost is dominated by `O(nd²)` linear and `O(n²d)` attention.
- **FlashAttention-compatible.** RoVE touches V (before the kernel call) and the aggregated output (after). It never modifies the `n×n` score matrix, so `flash_attn(Q, K, V_rot)` and a post-hoc `R_{−i}` rotation on the output preserve the IO-aware kernel pattern.
- **Same inverse-rotation mechanics as `apply_rope_phase_shift`** in our `katgpt-attn-match/src/chunked.rs` — RoVE just runs the inverse *on values* in addition to keys.

### 1.4 Empirical results (paper §4 + Appendix A)

GPT-2-style transformers at 124M and 354M parameters, trained one epoch on FineWebEdu-10B at 1024-token context. RoVE is the only architectural difference vs RoPE baseline (value pathway only; all other hyperparameters held fixed).

**In-distribution (≤1024 tokens):**
| Scale | Method | Core ICL ↑ | PPL @ 512 | PPL @ 1024 |
|---|---|---|---|---|
| 124M | RoPE | 0.1375 | 25.23 | 22.37 |
| 124M | RoVE | **0.1416** | **25.05** | **22.30** |
| 354M | RoPE | 0.1664 | 17.68 | 15.64 |
| 354M | RoVE | **0.1856** | **17.52** | **15.52** |

**Out-of-distribution (up to 16× training context):**
| Scale | Method | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 354M | RoPE | 293.24 | 840.10 | 1412.76 | 1630.72 |
| 354M | RoVE | **133.68** | **311.38** | **458.15** | **583.84** |
| 354M | RoPE + YaRN | 16.05 | 48.61 | 154.61 | 270.98 |
| 354M | RoVE + YaRN | **15.78** | **18.40** | **35.95** | **124.82** |

RoVE cuts 16k-token perplexity by **~64%** without YaRN, and the gap *persists* with YaRN (270.98 → 124.82) — RoVE and YaRN are **complementary**, not redundant. YaRN fixes low-frequency OOD angles; RoVE fixes the QK/OV mismatch that arises when QK extrapolates but OV stays put.

**Long-context retrieval (RULER, 4k/8k, NLL ↓):**
| Scale | Method | CWE 4k | NIAH 4k | QA 4k | VT 4k | Avg |
|---|---|---|---|---|---|---|
| 354M | RoPE + YaRN | 4.31 | 7.61 | 6.40 | 4.53 | 6.62 |
| 354M | RoVE + YaRN | 4.39 | **3.63** | 5.61 | **2.11** | **4.33** |

Largest gains on **Variable Tracking** (4.53 → 2.11) and **multi-key NIAH** (7.61 → 3.63) — tasks that require information to be **maintained and recombined** across the context, not merely detected. This is exactly what the OV-circuit position-awareness story predicts: standard RoPE selects *which* token to retrieve but transforms it identically regardless of distance; RoVE additionally controls *how* each retrieved feature is realigned before recombination.

### 1.5 Independent re-discoveries (the mechanism is robust)

RoVE-style value rotation has been independently arrived at across four communities:

- **Miyato et al. 2024 (GTA)** — multi-view novel-view synthesis; the frame-change motivation.
- **Wu et al. 2026 (RayRoPE), Li et al. 2026** — computer vision extensions.
- **Klee et al. 2026 (RAVEN)** — robotics equivariant learning.
- **DeepSeek-V4** — compressed shared-KV architecture; positional information leaks from keys into values, and the inverse output rotation `R_{−i}` is applied as a *corrective measure* to maintain relative position.

Each community motivated the mechanism on application-specific grounds; the paper provides the first **structural account** (attentive-convolution + matrix-mixer unification) and the first evaluation as a **standalone module in standard (non-shared-KV) language models**.

### 1.6 What the paper does NOT claim

- **No new state-of-the-art.** RoVE is a structural bias that improves RoPE across multiple axes, not a benchmark winner over the latest long-context methods.
- **No theoretical extrapolation guarantee.** OOD perplexity improves but does not vanish; YaRN is still needed at very long contexts.
- **No new parameters.** RoVE adds zero weights — the value pathway already has `W_V`; RoVE just rotates its output.
- **The mechanism behind OOD gains is not fully pinned down.** The paper's working hypothesis (Appendix D): RoVE induces a more coherent extrapolation regime because QK and OV share the same rotation family and therefore drift together at unseen offsets, whereas standard RoPE's QK extrapolates while OV stays put.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper ↔ codebase)

| Paper term | Codebase equivalent | Ships? |
|---|---|---|
| "RoPE rotation family `{R_t}`" | `RopeAction` (`katgpt-core/src/position_group_action.rs`, Plan 446 Issue 160) | ✅ — `apply_at(n, x, out)` does per-pair 2D rotation; `apply_inverse_at(n, x, out) = apply_at(−n, x, out)` |
| "Value pathway" / "OV circuit" / "`W_V x_j`" | `attn_wv` projection + `attn_wo` output projection (`katgpt-attn/src/dash_attn/forward.rs` lines 113, 115) | ✅ — but **not rotated** anywhere |
| "Frame-change" / "rotate into query's frame" | `apply_rope_phase_shift` (`katgpt-attn-match/src/chunked.rs`) | ✅ on **keys** during compaction; never on values |
| "Position-free compaction" | `PositionFreeCompactor` (`katgpt-kv/src/still_kv/position_free.rs`); `PositionFreeBridge::un_rotate_f32` | ✅ on keys |
| "MixedRoPE summarizer" | `MixedRopeSummarizer::from_rope_theta` (`katgpt-core`, used by HGA Plan 397) | ✅ on keys |
| "Block-Toeplitz kernel `ψ_δ`" / "attentive convolution" | (no codebase analog) | ❌ — no value-side offset-indexed kernel ships |
| "YaRN" / "frequency interpolation" | (no codebase analog — our RoPE paths use a fixed `θ_0 = 10000`) | ❌ — open opportunity, but out of scope for this note |
| "Matrix mixer" / "Kronecker vs gated block-Toeplitz" | (theoretical lens; not a codebase abstraction) | n/a |

### 2.2 What ships vs what RoVE adds

| Mechanism | Ships in katgpt-rs? | RoVE's added value |
|---|---|---|
| RoPE rotation family `{R_t}` (canonical pairs, log-uniform θ) | ✅ `RopeAction`, `RopeFreqs`, `MixedRopeSummarizer`, `apply_rope_phase_shift` | None — RoVE reuses the exact same family |
| RoPE applied to Q and K (the standard attention score path) | ✅ in every attention variant (`dash_attn`, `gdn2`, `ega`, `hga`) | None |
| **RoPE applied to V (value rotation before aggregation)** | ❌ — grep confirms no `apply_rope` / `rotate` call touches the V projection in any attention forward path | **NEW: the structural mechanism** |
| **Inverse RoPE applied to the aggregated output (`R_{−i} · aggregated`)** | ❌ — `apply_inverse_at` exists in `RopeAction` but is never called on attention output | **NEW: completes the frame-change** |
| Closed-form rank-2 Rodrigues `exp(L)` (GRAPE-M generalization of RoPE) | ✅ Plan 446 Issue 159 (`grapem_rodrigues` feature) | None — RoVE uses canonical-pair RoPE; could *optionally* use a GRAPE-M general plane in a future fusion |
| Unified `PositionGroupAction` trait | ✅ Plan 446 Issue 160 (`position_group_action` feature) | None — RoVE consumes this trait as its hot-path bridge (turning GRAPE's "vocabulary bridge" into a real attention variant) |
| Wall Attention (RoPE replacement via diagonal gates) | ✅ Plan 173 / Research 431 | Orthogonal — Wall *replaces* RoPE on QK; RoVE *extends* RoPE to OV. They could compose (Wall on QK + RoVE-style value rotation with the Wall gate substituted for `R_δ`), but that is a future fusion, not this primitive. |
| Attention Matching KV compaction (preserves RoPE-aware keys) | ✅ Plan 271 / Research 233 | Composable — AM compacts `(K, V)`; with RoVE, the compacted `V` should be **un-rotated** before compaction and **re-rotated** at the compacted position (mirror `apply_rope_phase_shift` for keys). Open fusion hook (Phase 4 of Plan 557). |

### 2.3 Prior-art surface — verified grep + read

**Three-layer check (notes + code + vocabulary translation) — mandatory per skill §1.5:**

1. **Notes grep** (`katgpt-rs/.research/*.md` + `.plans/*.md` + `riir-*/.research/*.md`) for `RoVE|rotary.*value|rotate.*value|value.*rotat|value.*position|position.*value|OV.*circuit|attentive.*conv|block.toeplitz` → **ZERO** hits for the value-rotation mechanism. The two `OV circuit` hits (Research 181 Compositional Muon, Research 450 Algorithmic Syntactic Causal ID) are about *training-time* LoRA optimization and *mech-interp* head importance respectively — neither touches the inference-time value rotation.
2. **Code grep** (`**/*.rs`) for `rotate_value|value.*rotat|apply_rope.*value|R_\{?j\}?|inverse_rotat` → **ZERO** hits for value-side rotation. The value projection (`attn_wv`, `w_v`) appears in every attention forward path but is **never** passed through any rotation call. `apply_rope_phase_shift` and `PositionFreeBridge::un_rotate_f32` are called on **keys only**.
3. **Vocabulary translation** — paper says "rotate the value at position `j` by `R_j`"; codebase equivalent would be `RopeAction::apply_at(j as f32, &w_v_x_j, &mut v_rotated)`. The trait method **exists** (Plan 446 Issue 160), is unit-tested (`RopeAction_inverse_roundtrip`), and is documented as a "vocabulary bridge" — but is never called on the V projection in any production attention path. The gap is *wiring*, not substrate.

**Closest cousins (4) — and why RoVE is NOT redundant:**

| Cousin | What it does | Why RoVE is different |
|---|---|---|
| **GRAPE `PositionGroupAction` trait (R446, P159/160/161/163)** | Abstracts RoPE/ALiBi/FoX/Wall as one group-action family with `apply_at`/`apply_inverse_at`. Explicitly documented as a **vocabulary bridge for cold-path code** ("does NOT replace `PositionFreeCompactor` or `WallDiagonalGate` internally — those stay as-is for hot-path performance"). Research 446's novelty table row "Composed `GL(d+2)` block-diagonal" lists status as "❌ Wall *replaces* RoPE; they are not composed". | GRAPE provides the *abstraction*; RoVE is the *first concrete hot-path attention variant that applies the rotation to V*. RoVE is the consumer that turns GRAPE's vocabulary bridge into a real attention mechanism. |
| **Attention Matching KV compaction (R233, P271)** | Mass-preserving compaction of `(K, V)` via NNLS-fit `β` and OLS-fit `C_V`. Preserves RoPE on keys via `apply_rope_phase_shift`. | Different problem (KV cache size reduction, not value-pathway position-awareness). Composes with RoVE: AM's `C_V` should be fit in position-free space when RoVE is active. |
| **Wall Attention (R431, P173)** | Replaces RoPE with data-dependent diagonal forget gates on QK. Value pathway unchanged. | Orthogonal axis. Wall fixes QK's data-independence; RoVE fixes OV's position-blindness. Could compose (Wall-QK + RoVE-OV), but that is a future fusion. |
| **Compositional Muon (R181)** | Partner-whitened optimizer for QK and OV circuits during LoRA *training*. | Training-time (→ riir-train territory). RoVE is inference-time and parameter-free. |

### 2.4 Fusion candidates (cross-repo — none strong enough for Super-GOAT)

The mandatory latent-space reframing (skill §1 step 3) across the 7 Super-GOAT factory modules yields these candidates:

1. **RoVE × GRAPE-M learned rotation planes** (`katgpt-core` + `katgpt-attn`): RoVE as described uses canonical-pair RoPE. A future fusion could substitute GRAPE-M's rank-2 Rodrigues (`grapem::Rank2Plane`, Plan 446 Issue 159) for the canonical pairs, giving **per-layer learned value-rotation planes** — a modelless analog of OFT applied to the OV circuit. **Latent-to-latent** (operates on the V projection's output). Stay-public primitive; quality gain needs head-to-head bench vs canonical-pair RoVE.
2. **RoVE × HLA per-NPC belief state** (`riir-engine/src/hla/`): HLA already rotates between `[valence, arousal, desperation, calm]` halves via `phase_rotation_gate_into` (Plan 322). RoVE's value-side rotation is structurally similar but operates on the attention V projection, not on HLA state directly. **Weak connection** — HLA's rotation is intra-entity belief-state; RoVE's is inter-token attention. No clear Super-GOAT fusion; flagged for separate consideration if a per-NPC attention variant lands.
3. **RoVE × Wall Attention** (`katgpt-attn`): Wall on QK + RoVE-style value rotation using the Wall gate (substituting `diag(f_t^n)` for `R_t`). Would give a **fully position-aware attention** where both score and value pathways carry relative-position information without RoPE at all. **Latent-to-latent** (operates on V). Public katgpt-rs fusion candidate; needs separate novelty gate before claiming.
4. **RoVE × Attention Matching** (`katgpt-attn-match`): when RoVE is active, AM's `C_V` fit should happen in position-free space (un-rotate V at the original position, fit, re-rotate at the compacted position — mirror the existing key-side `apply_rope_phase_shift`). **Public katgpt-rs fusion candidate; Phase 4 of Plan 557.**

None of these connect to ≥2 pillars + a product selling point, so the verdict stays at GOAT (see §3).

### 2.5 What stays public vs private

- **Public (`katgpt-rs`):** the RoVE attention primitive itself — generic transformer math, parameter-free, MIT-appropriate. Reuses `RopeAction` (already public). Adds a feature flag `rotary_value_embedding` (opt-in → default-on after GOAT gate).
- **Private (`riir-ai`):** any per-NPC personality-specific value rotation plane fusion (Fusion 2 above, if it ever materializes).
- **Private (`riir-chain`):** n/a — the rotation is local to the attention layer; nothing crosses the sync boundary.
- **Private (`riir-neuron-db`):** n/a — shard internals are unaffected.

---

## 3. Verdict

**Verdict: GOAT.**

RoVE is a **provable gain over RoPE** (latency-equivalent — `O(nd)` linear overhead dominated by `O(nd²)` linear layers; consistent quality gains across ICL, OOD perplexity, and long-context retrieval at both 124M and 354M scales), **parameter-free** (zero new weights), and **FlashAttention-compatible** (rotations act on V and post-softmax output, never on the score matrix). It closes a structural asymmetry that has been independently discovered across four communities (vision, robotics, multi-view synthesis, DeepSeek-V4). The closest shipped substrate (GRAPE's `PositionGroupAction` trait) explicitly ships as a "vocabulary bridge for cold-path code" — RoVE is the first concrete hot-path attention variant that applies the rotation to V, turning that bridge into a real mechanism.

### Why not PASS

Per skill §1.55: PASS requires "no actionable improvements". RoVE produces one concrete, in-scope, modelless actionable primitive — the value-side rotation wiring. The substrate exists (`RopeAction::apply_at` / `apply_inverse_at`); the wiring does not. That is the canonical shape of a Gain-or-higher verdict.

### Why not Super-GOAT (novelty gate Q1–Q4)

| Q | Answer |
|---|---|
| Q1 No prior art? | **YES.** Three-layer check confirms no value-side rotation ships in any `.research/`, `.plans/`, or `.rs` file. The closest cousin (GRAPE) ships the *abstraction* (`PositionGroupAction`) but explicitly not the *application* (V rotation). |
| Q2 New class of behavior? | **Partial.** RoVE produces a new mixer structure (gated block-Toeplitz vs Kronecker), but it is a *parametric enrichment* of an existing capability (RoPE attention), not a new capability class. Standard RoPE is the degenerate `ψ_δ ≡ W_V`. |
| Q3 Product selling point? | **NO (for our stack).** Our modelless inference engine serves upstream checkpoints — the model's `W_V` is fixed by the upstream weights. RoVE improves quality only if the upstream checkpoint was trained with RoVE (or if we adopt it as a runtime retrofit, which the paper does NOT validate). For pure inference of RoPE-trained checkpoints, RoVE-as-retrofit is unvalidated — applying it at inference to a model trained without it would change the OV-circuit semantics in a way the upstream training did not anticipate. |
| Q4 Force multiplier? | **NO.** RoVE touches one pillar (transformer attention substrate). It does not connect to HLA / latent_functor / cgsp_runtime / neuron-shard / LatCal. The four fusion candidates in §2.4 are speculative; none is a slam-dunk cross-pillar multiplier. |

→ 1.5 of 4 YES → GOAT, no Super-GOAT guide created. The honest reason this is not Super-GOAT for *our* stack: we ship a modelless inference engine for upstream checkpoints, and RoVE's quality gains require the upstream checkpoint to be trained with RoVE. The mechanism is GOAT for the **engine completeness** story (RoVE-aware inference for RoVE-trained checkpoints), not for our game-AI moat.

### MOAT gate per domain (§1.6)

- **`katgpt-rs` (this repo):** **Strengthens moat** — RoVE-aware inference completes the PositionGroupAction story. GRAPE shipped the vocabulary bridge (Plan 446); RoVE is the concrete attention variant that justifies the bridge. A new transformer substrate primitive that passes GOAT, with a clear "RoVE-trained checkpoint" consumer story for downstream adopters. ✅ In scope.
- **`riir-ai` fusion candidate:** HLA per-NPC value rotation plane — weak connection (see §2.4 F2); not a Super-GOAT today.
- **`riir-chain` / `riir-neuron-db`:** no clear fusion — the rotation is local to the attention layer.

### Honest caveat — the inference-only retrofit question

RoVE is published as a **training-time architectural choice** (the model is trained from scratch with V rotation in place). The paper does NOT validate RoVE as an inference-time retrofit onto RoPE-trained checkpoints. Three scenarios for our stack:

1. **RoVE-trained upstream checkpoint** (future-proof): when an open-weight RoVE-trained model appears, our engine should support it natively. The primitive is required for forward-compat. **This is the strongest case for shipping it.**
2. **RoPE-trained checkpoint + RoVE retrofit at inference** (unvalidated): would the value rotation help or hurt? The paper does not say. The structural argument cuts both ways — RoVE makes the OV circuit offset-aware, but the model's `W_V` was trained under the offset-blind assumption. A modelless PoC (Phase 5 of Plan 557) could settle this on a toy GPT-2 by training without RoVE, then benchmarking with-RoVE vs without-RoVE at inference. **Honest verdict: unknown, needs PoC.**
3. **Fine-tune with RoVE, deploy with RoVE** (riir-train territory): training a LoRA adapter that *expects* RoVE-style value rotation. This is a training-method question → riir-train.

Plan 557 ships the primitive (scenario 1) + benchmarks the retrofit (scenario 2) as a Phase 5 honest PoC. If scenario 2 fails (RoVE retrofit hurts RoPE-trained models), the primitive stays opt-in for forward-compat only; if scenario 2 succeeds (free quality gain at inference), it becomes a default-on candidate.

---

## 4. Actionable follow-ups — Plan 557 opened

**Plan:** [`.plans/557_rotary_value_embeddings.md`](../.plans/557_rotary_value_embeddings.md)

| Phase | Goal | Feature gate |
|---|---|---|
| 1 | Skeleton — `RoVeConfig`, `rotate_values_into`, `inverse_rotate_output_into` in a new `katgpt-core` module. Zero-dep, zero-alloc. | `rotary_value_embedding` (opt-in) |
| 2 | G1–G5 GOAT gate — bit-identical to canonical RoPE when disabled; correct rotation when enabled; zero steady-state alloc; `O(nd)` overhead < 5% of `O(nd²)` linear layer; FlashAttention-compat (output-equivalent to manual rotation). | (gate) |
| 3 | Wiring — opt-in forward path in `katgpt-attn` that calls `rotate_values_into` after the V projection and `inverse_rotate_output_into` after the softmax-weighted sum. Mirror the existing RoPE-on-QK call site. | (gate) |
| 4 | AM fusion — when RoVE + Attention Matching both active, fit `C_V` in position-free V space (un-rotate at original pos, fit, re-rotate at compacted pos). | `attention_matching + rotary_value_embedding` |
| 5 | Honest retrofit PoC — train a toy GPT-2 (124M-equivalent at small scale) *without* RoVE, then benchmark perplexity with-RoVE vs without-RoVE at inference. Documents whether scenario 2 (retrofit gain) is real or a quality regression. | (bench only; no feature gate) |

**Promotion rule:** if Phase 2 G1–G5 PASS + Phase 5 PoC shows no regression → promote `rotary_value_embedding` to default-on (it is parameter-free and FlashAttention-compatible; the only risk is the unvalidated retrofit, which Phase 5 settles). If Phase 5 shows regression on RoPE-trained models → keep opt-in for forward-compat (scenario 1) only.

**Issue tracker:** none needed at this stage — the plan tracks everything. If Phase 5 reveals a deeper mech-interp question (e.g. "which RoPE frequency bands benefit most from value rotation?"), open an `.issues/` entry then.

---

## 5. References

- **Paper:** [arXiv:2606.11275](https://arxiv.org/abs/2606.11275) — García-Castellanos, Weiler, Bekkers, Jul 2026.
- **Code:** [github.com/AGarciaCast/RoVE](https://github.com/AGarciaCast/RoVE)
- **Prior art in stack:**
  - [`crates/katgpt-core/src/position_group_action.rs`](../crates/katgpt-core/src/position_group_action.rs) — `PositionGroupAction` trait + `RopeAction` (Plan 446 Issue 160). The substrate RoVE consumes.
  - [`crates/katgpt-core/src/grapem.rs`](../crates/katgpt-core/src/grapem.rs) — rank-2 Rodrigues (Plan 446 Issue 159). Optional fusion for non-canonical rotation planes.
  - [`crates/katgpt-attn-match/src/chunked.rs`](../crates/katgpt-attn-match/src/chunked.rs) — `apply_rope_phase_shift` (the key-side analogue of what RoVE does to values).
  - [`crates/katgpt-kv/src/shard_kv/kv_cache.rs`](../crates/katgpt-kv/src/shard_kv/kv_cache.rs) — `RopeFreqs` (the per-pair frequency schedule reused by RoVE).
  - [`.research/446_GRAPE_Group_Representational_Position_Encoding.md`](446_GRAPE_Group_Representational_Position_Encoding.md) — GRAPE distillation.
  - [`.research/431_Wall_Attention_Diagonal_Gate_RoPE_Replacement.md`](431_Wall_Attention_Diagonal_Gate_RoPE_Replacement.md) — Wall Attention (orthogonal axis).
  - [`.research/233_Attention_Matching_KV_Compaction.md`](233_Attention_Matching_KV_Compaction.md) — Attention Matching (composes via Phase 4).
- **Independent re-discoveries cited in paper:**
  - Miyato et al. 2024 (GTA) — multi-view novel-view synthesis, the frame-change motivation.
  - DeepSeek-V4 (2026) — inverse output rotation as corrective measure for shared-KV leakage.
  - Klee et al. 2026 (RAVEN) — robotics equivariant learning.
  - Wu et al. 2026 (RayRoPE), Li et al. 2026 — computer vision extensions.
- **Structural theory:**
  - Romero et al. 2020 (Attentive Group Equivariant Convolutional Networks) — the attentive convolution primitive RoVE recovers.
  - Fuchs et al. 2020 (SE(3)-Transformers) — tensorial messages in equivariant attention.
  - Elhage et al. 2021 (Transformer Circuits) — QK/OV circuit decomposition RoVE completes.
  - Hwang et al. 2024 (Hydra) — matrix mixer unification (Kronecker vs gated block-Toeplitz).
