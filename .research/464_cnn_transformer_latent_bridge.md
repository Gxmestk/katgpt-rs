# Research 464: CNN→Transformer Latent Bridge — Modelless LLaVA-for-Go Design

> **Filed:** 2026-08-01
> **Type:** Design formalization (gap → candidate)
> **Origin:** Research 463 §2.8 identified this as the root-cause gap motivating
> the entire freeze/thaw weight-space workaround family. Issue 565 G1-B confirmed
> the weight-space workarounds have hit their ceiling (Strategy A expensive at
> 27.8% overhead, Strategy B overfits, Strategy D wrong model). This note
> formalizes the activation-space alternative.
> **Consumer:** Proposal 008 (Go Gemma Arena) — 🟡 PARTIALLY DELIVERED, 100%
> parse-fallback without priming, strength claim awaits Go understanding.

## TL;DR

Moka's CNN forward pass computes rich intermediate feature maps (`[32, 9, 9]`
after each residual block) and **throws them away** — only `[policy(82), value(1)]`
crosses the boundary. These intermediate maps ARE latent vectors representing
Moka's "Go vision." Projecting them into Gemma's residual stream (the LLaVA /
Flamingo pattern) could give the transformer actual board understanding —
unblocking Proposal 008's 100% parse-fallback problem modellessly.

**The modelless bridge has three pieces, all with existing substrate:**

| Step | What | Substrate | Status |
|---|---|---|---|
| 1. Tap | Capture Moka's trunk after block N | New `forward_tapping_trunk` fn (mirrors `forward_collecting_activations`) | Not started |
| 2. Project | 32-dim Moka features → d_model Gemma residual | `CrossResolutionBases` (Plan 310, DEFAULT-ON) + `transport_cross_resolution_into` | ✅ shipped |
| 3. Inject | Projected features → transformer residual stream | `forward_with_steering` + `ResidualField` (Proposal 006) OR direct sequence prepend | ✅ shipped |

**Honest prediction:** the modelless bridge will provide PARTIAL improvement
(parse-fallback rate drops, Gemma starts "seeing" the board), but will NOT
reach Moka-native Go strength without training the projection (riir-train).
The value is: (a) proving the bridge wiring works end-to-end modellessly;
(b) establishing the substrate for a future trained projection; (c) negative
knowledge about how far deterministic projection can go.

## 1. The Architectural Reality (from Research 463 §2.8)

katgpt-rs provides TWO layers to Moka:

| Layer | What | Moka uses it? |
|---|---|---|
| SIMD primitives (`simd_dot_f32`) | Hand-written NEON/AVX2 dot product kernel | ✅ YES — 8.7× speedup |
| Transformer engine (`TransformerWeights`, `forward`, attention, KV cache) | Token-sequence inference engine | ❌ NO — Moka is a CNN |

Moka's `forward_with_scratch` is **self-contained**: it borrows `simd_dot_f32`
for conv/linear acceleration, but the computation graph (conv→relu→residual→
pool→linear→tanh) has zero overlap with the transformer graph (embedding→
attention→mlp→norm→KV cache). The CNN and transformer are completely disjoint.

**The gap:** `forward_with_scratch` returns only `[policy, value]`. The
intermediate feature maps (e.g., `[32, 9, 9]` trunk after block 6) are computed
and **discarded**. These maps ARE latent vectors that COULD be bridged into
the transformer's residual stream.

## 2. The Two Injection Strategies

### 2.1 Aggregate Injection (simplest — recommended for first PoC)

```
Moka trunk after block N: [32, 9, 9]  (32 channels × 81 positions)
      ↓ global mean-max pool → [64]  (32 mean + 32 max, same as Moka's global branch)
      ↓ project [64] → [d_model]  via CrossResolutionBases (modelless PCA)
      ↓ inject as single ResidualField at one Gemma layer
Gemma forward with steering → logits
```

**Pros:**
- Simplest wiring — `forward_with_steering` already accepts `ResidualField`
- The global mean-max pool is already computed inside Moka's global blocks
- Single vector injection is the proven path (CWM judge uses it)

**Cons:**
- Loses spatial information — Gemma sees "what's on the board" but not "where"
- A 64-dim summary can't capture the spatial patterns that matter in Go

### 2.2 Full LLaVA Injection (stronger — harder)

```
Moka trunk after block N: [32, 9, 9]
      ↓ reshape → [81, 32]  (81 spatial positions = 81 "Go vision tokens")
      ↓ project each [32] → [d_model]  via CrossResolutionBases
      ↓ prepend 81 projected tokens to Gemma's input sequence
Gemma attends to 81 visual + N text tokens → logits
```

**Pros:**
- Preserves spatial information — Gemma can reason about "where"
- The true LLaVA/Flamingo pattern — proven in vision-language models
- Attention can learn which board positions matter for the current decision

**Cons:**
- Requires bypassing the embedding lookup (the 81 projected vectors are
  residual-stream-level, not token-ID-level)
- Needs a new forward variant or modification to `forward_gemma2_layers`
- 81 extra tokens add ~81×(attention cost) per Gemma forward — significant
  for a 2B model (but Go is turn-based, cold-tier, uncapped latency)

**Structural note:** `forward_with_steering` injects a single d_model vector
per layer per token — it does NOT add new sequence positions. The full LLaVA
pattern needs a different injection mechanism. The simplest approach: prepend
the projected vectors directly into the residual stream before the first
attention layer (they enter as "pre-computed embeddings").

## 3. The Modelless Projection (CrossResolutionBases)

`katgpt-core::cross_resolution::CrossResolutionBases` (Plan 310, DEFAULT-ON,
G1-G4 PASS) projects between different-dimensional latent spaces via a k-dim
spectral intermediary:

```
src_state [d_src] → phi_src [d_src × k] → spectral [k] → psi_dst [d_dst × k] → dst_predicted [d_dst]
```

For the CNN→Transformer bridge:
- `d_src` = 32 (Moka TRUNK_CHANNELS) for per-position, or 64 for aggregate
- `d_dst` = d_model (Gemma 2 2B: 2304; the exact dim from `Config::n_embd`)
- `k` = spectral rank (start with k=16, sweep k ∈ {8, 16, 32})

**The bases are caller-supplied frozen PCA/SVD artifacts** computed offline
on a calibration set of paired (Moka features, Gemma activations). The
calibration is a one-time offline step — the runtime projection is pure
linear algebra, no GD.

**Calibration set:** collect N Go positions, run Moka forward (capturing
trunk at block N) + Gemma forward (capturing residual at layer L) on the
same board encoding. Fit PCA on the paired set. Same pattern as
`hydra_budget.rs::run_logit_lens_calibration` and the Issue 565 G1-B
activation collection.

**Honest limitation:** without paired training data (what SHOULD the Gemma
residual look like given this Moka trunk?), the PCA projection is fitting
to the DISTRIBUTION of Moka features, not to a target Gemma activation.
This is a "blind" projection — it preserves variance but not semantics.
The trained path (riir-train) would learn the semantically correct mapping.

## 4. Which Block to Tap

Moka has 12 residual blocks. The trunk evolves from raw board features
(block 0) to high-level strategic understanding (block 11). The choice of
tap point trades detail vs abstraction:

| Tap point | What it captures | Information density |
|---|---|---|
| After stem (block 0) | Raw conv features — local stone patterns | High spatial, low strategic |
| After block 3 | Early tactical patterns (ladders, captures) | Medium |
| After block 6 (mid) | Mid-game strategic features (influence, territory) | Balanced |
| After block 9 | Late strategic features (life/death, endgame) | Low spatial, high strategic |
| After block 11 (final) | Pre-policy features — almost the policy head input | Closest to move selection |

**Recommendation for first PoC:** tap after block 6 (mid-network). This is
the "sweet spot" in CNN feature hierarchies — enough abstraction to be
meaningful, enough spatial detail to be useful. Sweep this in the PoC.

**Signal-availability check (2026-08-01):** the tapped trunk after block 6
was measured across 4 diverse board positions (empty / 1 center stone / 5
scattered stones / 10 clustered corner stones). Pairwise cosine similarity:
**min=0.7988, max=0.9258** — well below the 0.99 "no signal" threshold.
The trunk DOES differentiate board positions. This is a necessary
precondition for the bridge (if trunks were nearly identical, no projection
could extract position-discriminating signal). It is NOT a sufficiency
proof — the cross-modal projection (T2-T4) still needs to carry this signal
into Gemma's residual stream in a way Gemma can interpret.

**T4 calibration results (2026-08-01):** the cross-modal correlation was
measured on 24 Go positions (16 train + 8 test) with real Gemma 2 2B.
Key findings:

1. **Cross-covariance is non-zero** (σ_1=1.31 at layer 13) — linear
   cross-modal structure EXISTS.
2. **Blind PCA fails** (R²=0.0015) — independent per-modality PCA does NOT
   capture cross-modal structure. This confirms the §3 "blind projection"
   limitation: preserving variance in each space ≠ preserving correlation.
3. **CCA-aware projection captures real signal:** centered test R²=0.277 at
   k=2 — 27.7% of the board-specific variance generalizes to held-out boards.
4. **Board-specific variance is tiny** (0.008% of total) — the prompt template
   dominates 99.992% of the residual. Even a perfect bridge injects a small
   perturbation relative to the prompt-driven signal.
5. **Overfitting at k≥8** (test cR² drops to 0.091, gap grows to 0.47). k=1-3
   is the stable regime.

**Verdict:** the modelless bridge carries WEAK BUT REAL cross-modal signal.
Not enough for full Go understanding (73% of board-specific variance
uncaptured), but sufficient to justify Phase 1 wiring (T5-T7) — the
aggregate bridge is worth testing for parse-fallback reduction.

**Phase 1 verdict (T5-T7, 2026-08-01): NEGATIVE.** The aggregate bridge is
confirmed unviable as a parse-fallback fix despite the non-zero cross-modal
signal (cR²=0.277). Results:

| Scale | Mean P(coord) shift | Argmax coord 18/20 → |
|---|---|---|
| 1.0 | **-0.0004** | 17/20 (worsened) |
| 10.0 | +0.0001 | 17/20 |
| 100.0 | +0.0001 (saturated) | 17/20 |

The signal exists but is too weak to be useful. The 0.008% board-specific
variance fraction means even a perfect linear projection injects a tiny
perturbation relative to the prompt-driven signal. Scaling the injection
doesn't help because the alpha parameter saturates at 1.0. **Phase 2
(per-position LLaVA tokens) is the only remaining modelless path** — it
preserves spatial information and uses attention (not addition) to integrate
board features, which may overcome the signal-to-noise problem.

## 5. The Consumer: Proposal 008 (Go Gemma Arena)

Proposal 008's state (as of 2026-08-01):
- 🟡 PARTIALLY DELIVERED — demo wiring DONE
- **100% parse-fallback without priming** — Gemma produces coordinate-shaped
  output only with a one-shot example; without it, it says "This looks like
  chess/checkers"
- Strength claim awaits Phase 5 LoRA training

**Why the latent bridge could help:** Gemma 2 2B has vast world knowledge
(including some Go knowledge from training data) but NO visual board
perception. When given a board as text (FEN/SGF), it can't "see" the spatial
patterns. Injecting Moka's intermediate features as residual steering gives
Gemma a "Go vision" modality it currently lacks.

**The honest comparison:** this is NOT a replacement for Phase 5 LoRA training.
It's a modelless FIRST step that could:
1. Drop the parse-fallback rate (Gemma has board context to reason about)
2. Improve move quality even without training (Moka's features carry signal)
3. Establish the substrate for future trained projection

If the modelless bridge drops parse-fallback from 100% to <50%, that's a
GOAT-quality win for Proposal 008 — it proves the bridge carries real signal.

## 6. Honest Prediction (pre-PoC)

| Outcome | Probability | Actual |
|---|---|---|
| Aggregate bridge drops parse-fallback significantly | Medium | **❌ NO (T6: delta ≈ 0, argmax worsened)** |
| Full LLaVA bridge drops parse-fallback significantly | Higher | TBD (T8-T9, deferred) |
| Either bridge reaches Moka-native Go strength | Low | TBD |
| Either bridge beats int8 Moka (Issue 565 G5) | Very low | TBD |

**Phase 1 lesson (2026-08-01):** the cross-modal signal EXISTS (cR²=0.277)
but is too weak to be useful via aggregate injection. The root cause is the
0.008% board-specific variance fraction — the prompt template dominates
99.992% of the layer-13 residual. Adding a board-specific perturbation via
residual addition is like whispering in a hurricane. The full LLaVA path
(prepending 81 attention-attended tokens) may overcome this because attention
can ROUTE around the prompt dominance — it doesn't have to fight the common
mode at the residual level.

**The load-bearing question:** does Gemma, given Moka's Go vision, produce
BETTER Go decisions than Gemma without it? This is a quality PoC, not a perf
PoC. Go is turn-based (cold-tier, uncapped latency), so the 81-token
attention overhead is acceptable.

## 7. Relationship to Issue 565 (Weight-Space Workarounds)

Issue 565 tested four weight-space quantization-compensation strategies:
- **Strategy A** (weight-space SVD): works at rank-32 (+0.023 cosine), but
  costs 27.8% param overhead — the Small-Kernel Paradox
- **Strategy B** (data-aware SVD): FAILS even with proper calibration —
  small networks overfit to the calibration distribution
- **Strategy D** (sparse bypass): FAILS — error is distributed, not
  outlier-concentrated

**The connection:** weight-space workarounds operate at the WEIGHT level
(correcting quantization errors). They keep Moka as a black box. The latent
bridge operates at the ACTIVATION level (tapping intermediate features).
It opens the black box.

The G1-B finding (weight-space workarounds have diminishing returns) does
NOT prove the activation-space bridge will work — but it removes the
"weight-space is sufficient" argument. If the bridge also fails, the honest
conclusion is that Moka's CNN and Gemma's transformer are fundamentally
incompatible at this scale without training.

## 8. Scope and Deferrals

**In scope for this note:**
- The modelless bridge design (aggregate + full LLaVA)
- The substrate inventory (all pieces exist)
- The consumer justification (Proposal 008)
- The honest prediction

**Out of scope (deferred to riir-train if modelless fails):**
- Trained projection (LLaVA-style learned linear layer)
- Fine-tuning Gemma on Go game records
- Any backpropagation or gradient descent

**Out of scope (not needed for this PoC):**
- Multi-modal training (joint Moka+Gemma training)
- Real-time bridge (Go is turn-based, cold-tier)
- Anti-cheat implications (the bridge is read-only on Moka's features)

## 9. Verdict

**Tier: Gain** (for the modelless bridge candidate — a new capability
demonstration, not a pillar primitive).

The bridge is a modelless activation-space mechanism that opens Moka's black
box. If it works, it unblocks Proposal 008's strength claim without training.
If it fails, it provides negative knowledge about CNN→Transformer
compatibility at this scale. Either way, the substrate (tap + project + inject)
is reusable for any future vision-to-language bridge.

**One-line reasoning:** since the weight-space workarounds (Issue 565) have
hit their ceiling and the substrate exists, the activation-space bridge is
the natural next modelless attempt — it gives Gemma a "Go vision" modality
that addresses Proposal 008's 100% parse-fallback problem directly.

**Issue:** 566 (PoC plan — tap + project + inject + quality gate).
