# Issue 566 — CNN→Transformer Latent Bridge PoC (Modelless LLaVA-for-Go)

> **Filed:** 2026-08-01
> **Research:** [464](../.research/464_cnn_transformer_latent_bridge.md)
> **Origin:** Research 463 §2.8 (gap tracking) → Research 464 (design formalization).
> Issue 565 G1-B confirmed weight-space workarounds have hit their ceiling,
> strengthening the case for the activation-space bridge.
> **Consumer:** Proposal 008 (Go Gemma Arena) — 100% parse-fallback, needs Go understanding.
> **Type:** PoC (defend-wrong — predicted partial improvement, not full strength)
> **Status:** CLOSED — both phases NEGATIVE. Phase 1 (aggregate): signal too
> weak (KL=0.0001, delta≈0). Phase 2 (per-position): signal actively harmful
> (P(coord) 0.49→0.002 — out-of-distribution embeddings). The modelless bridge
> is confirmed unviable; the cross-modal linear projection captures statistical
> correlation but not semantic alignment. The trained projection path (riir-train)
> is the only remaining option for the CNN→Transformer bridge.

## Context

Moka's CNN computes rich intermediate feature maps (`[32, 9, 9]` trunk after
each residual block) and discards them — only `[policy(82), value(1)]`
crosses the boundary. Research 464 formalizes the LLaVA-for-Go design:
tap Moka's intermediate features, project them to d_model via
`CrossResolutionBases` (modelless PCA), inject into Gemma's residual stream.

**Why now:** Issue 565 G1-B confirmed Strategy B (data-aware SVD) fails even
with proper calibration — weight-space workarounds have hit their ceiling.
The activation-space bridge is the natural next modelless attempt. Proposal
008's 100% parse-fallback is the concrete consumer.

## Substrate Inventory (all exists)

| Piece | Location | Status |
|---|---|---|
| `forward_collecting_activations` | `katgpt-moka-wasm/src/research.rs` | ✅ captures layer INPUTS (for calibration) |
| `CrossResolutionBases` + `transport_cross_resolution_into` | `katgpt-core/src/cross_resolution.rs` (Plan 310, DEFAULT-ON) | ✅ G1-G4 PASS |
| `forward_with_steering` + `ResidualField` | `riir-engine/src/latent_steering_bridge.rs` (Proposal 006) | ✅ proven end-to-end |
| `GemmaGoPlayer` (the consumer) | Proposal 008 / Plan 393 | ✅ demo wiring DONE, strength NOT proven |

**What's missing:** a function to tap Moka's trunk OUTPUT after block N
(existing `forward_collecting_activations` captures layer INPUTS, not block
OUTPUTS). Plus the wiring to project + inject.

## The PoC Plan (two phases)

### Phase 1 — Aggregate Bridge (simplest, recommended first)

**Design:** global mean-max pool Moka's trunk after block N → 64-dim →
project to d_model via `CrossResolutionBases` → inject as single
`ResidualField` at one Gemma layer.

This is the simplest bridge — it loses spatial information but proves the
wiring works end-to-end. If it drops parse-fallback even slightly, the
concept is validated.

### Phase 2 — Full LLaVA Bridge (stronger, if Phase 1 shows signal)

**Design:** reshape Moka's trunk to [81, 32] → project each position to
d_model → prepend 81 "Go vision tokens" to Gemma's input sequence.

This preserves spatial information (the true LLaVA pattern) but needs a new
forward variant that bypasses the embedding lookup for the prepended tokens.

## Tasks

> **Design note on T2-T4 (calibration):** `CrossResolutionBases` projects
> via a k-dim spectral intermediary using frozen PCA bases fit on PAIRED data.
> For cross-game HLA transfer (the existing use case), both domains are HLA
> (affect/emotion) states — comparable semantic spaces. For the CNN→Transformer
> bridge, the source (Moka trunk = visual board patterns) and destination
> (Gemma residual = language features) are DIFFERENT modalities. PCA finds the
> LINEAR correlation between them, but there may be little linear correlation
> to find across modality boundaries. T2-T4 will measure this empirically —
> the correlation IS the signal. If it's near-zero, the modelless bridge is
> confirmed unviable without training (the honest prediction). If it's
> non-trivial, the PCA bridge carries real signal worth wiring.

- [x] **T1** — Add `forward_tapping_trunk` to `katgpt-moka-wasm/src/research.rs`.
      Mirrors `forward_with_scratch` but copies the trunk buffer after a
      caller-specified block into an output slice. Gated behind `research`
      feature (off for production WASM build). **DONE 2026-08-01** — 4 unit
      tests pass: (1) tapping forward matches production within 1e-3 epsilon,
      (2) each block index 0..11 produces non-trivial trunk, (3) out-of-range
      tap is a no-op, (4) **signal-availability check: tapped trunks
      differentiate board positions** — cosine 0.80-0.93 across 4 diverse
      boards (empty / 1 stone / 5 scattered / 10 clustered), well below the
      0.99 threshold. The trunk carries position-discriminating signal —
      the bridge's necessary precondition is met.
- [x] **T2** — Calibration set: collect N=16 Go positions, run
      `forward_tapping_trunk` to capture trunk after block 6. This gives
      the Moka-side of the paired calibration data. **DONE 2026-08-01** —
      `collect_moka_aggregates` in `cnn_transformer_bridge_poc.rs`. Global
      mean-max pool → 64-dim aggregates. Pairwise min cosine 0.79 (PASS).
- [x] **T3** — Gemma-side calibration: run `GemmaGoPlayer` on the same N
      positions, capture residual at layer L. This gives the paired
      (Moka features, Gemma residual) set for PCA. **DONE 2026-08-01** —
      `forward_collecting_residual` added to `latent_steering_bridge.rs`
      (5 G1 tests: g1h-g1k). `collect_gemma_residuals` uses
      `go_gemma_data::render_prompt_body` (the arena text format) + Gemma 2
      2B at layer 13. **Key finding:** pairwise min cosine 0.9999 — residuals
      barely differ across boards (prompt template dominates at this layer).
- [x] **T4** — Fit `CrossResolutionBases` via PCA on the paired set. Offline
      one-time step. Store the bases as a frozen artifact. **DONE 2026-08-01** —
      Four measurement tests (T4/T4b/T4c/T4d/T4e) in
      `cnn_transformer_bridge_poc.rs`. **Key findings:**
      - T4: cross-covariance SVD has non-zero singular values (σ_1=1.31 at
        L13) — linear cross-modal structure EXISTS.
      - T4b (blind PCA): mean R²=0.0015 — independent PCA does NOT capture
        cross-modal structure (confirms Research 464 §3 limitation).
      - T4c (CCA-aware): train R²=0.9999 — BUT misleading (common mode).
      - T4d (cross-validation): test R²=0.9999 — generalizes, but still the
        common mode (99.992% of variance is prompt template).
      - **T4e (centered R² — the TRUE signal test):** board-specific variance
        is only 0.008% of total. CCA captures **test cR²=0.277 at k=2** —
        27.7% of the board-specific variance generalizes to held-out boards.
        **Real cross-modal signal exists.** Overfitting at k≥8 (test cR² drops
        to 0.091). k=1-3 is the stable regime.
- [x] **T5** — Phase 1 wiring: aggregate bridge. Global mean-max pool →
      project → `ResidualField` → `forward_with_steering`. **DONE 2026-08-01** —
      `CnnTransformerBridge` struct + `fit_cca_bases` + `project` +
      `to_residual_field` in `cnn_transformer_bridge_poc.rs`. The bridge wires
      Moka trunk aggregate → CCA project → ResidualField → `forward_with_steering`.
      **Result: wiring WORKS** — KL divergence is non-zero (mean 0.0001 bits,
      max 0.0004 bits), top-token changed on 1/8 test boards. The bridge
      measurably shifts Gemma's output, but the perturbation is tiny (projected
      norm ~1.03; board-specific variance = 0.008% of total).
- [x] **T6** — Phase 1 quality PoC: measure coordinate-token probability shift
      on N=20 held-out boards. **DONE 2026-08-01** — Logit-level proxy for
      parse-fallback rate: P(coordinate letter tokens) at the first decoded
      position, with vs without the bridge. Swept scale ∈ {1.0, 10.0, 100.0}.
      **Result: NEGATIVE** — the aggregate bridge does NOT shift probability
      toward Go coordinate tokens:
      - scale=1.0: mean P(coord) 0.5227 → 0.5223 (delta **-0.0004**),
        argmax coord 18/20 → **17/20** (worsened)
      - scale=10.0: delta +0.0001, argmax coord 18/20 → **17/20**
      - scale=100.0: delta +0.0001 (identical to scale=10.0 — alpha clamps
      at 1.0, so magnitude saturates above ~1.0 projected norm)
      The bridge carries measurable signal (T4e: cR²=0.277) but NOT enough to
      shift Gemma's Go output at the logit level.
- [x] **T7** — Phase 1 verdict: **NEGATIVE — aggregate bridge does NOT reduce
      parse-fallback.** The cross-modal signal (27.7% of board-specific
      variance) exists but is too weak to be useful. The board-specific variance
      itself is only 0.008% of the total residual — even a perfect bridge would
      inject a tiny perturbation relative to the prompt-driven signal. Scaling
      the injection (scale=10/100) doesn't help because alpha saturates at 1.0.
      **Phase 2 (full LLaVA, per-position tokens) is the only remaining
      modelless path.** The aggregate bridge loses spatial information — a
      64-dim summary can't carry the position-discriminating patterns that Go
      requires. Per-position injection (81 tokens × d_model) would give Gemma
      spatial board vision, which is what the parse-fallback problem actually
      needs.
- [x] **T8** — Phase 2 wiring (full LLaVA): DONE 2026-08-01. Prepends 81
      per-position vision tokens to Gemma's input sequence via
      `forward_gemma2_with_embedding` (bypasses embedding lookup). New code:
      `collect_moka_per_position_flat`, `fit_cca_bases_per_position`,
      `project_per_position`, `forward_with_vision_prefix`. **Result: wiring
      works STRONGLY** — mean KL divergence 4.46 bits at scale=1.0 (44,000×
      larger than Phase 1's 0.0001 bits). Top-token changed on 8/8 boards.
      Scaling works monotonically (KL grows with scale, no saturation unlike
      Phase 1). However, the top token changed FROM coordinate letters ('C')
      TO non-coordinates ('It', 'This') — the vision prefix DOMINATES the
      output, not nudges it.
- [x] **T9** — Phase 2 quality PoC: DONE 2026-08-01. **Result: DECISIVELY
      NEGATIVE — worse than Phase 1.** The vision prefix DESTROYS coordinate
      probability at ALL scales (even scale=0.01):
      - scale=0.01: P(coord) 0.4936 → 0.0016 (delta **-0.492**), argmax 6/8 → 0/8
      - scale=0.1: P(coord) 0.4936 → 0.0015 (delta **-0.492**), argmax 6/8 → 0/8
      - scale=0.5: P(coord) 0.4936 → 0.0011 (delta **-0.492**), argmax 6/8 → 0/8
      - scale=1.0: P(coord) 0.4936 → 0.0009 (delta **-0.493**), argmax 6/8 → 0/8
      The 81 vision tokens are out-of-distribution embeddings — they enter at
      layer 0 as d_model vectors projected from layer-13 residual space (where
      the CCA was calibrated). The attention mechanism treats them as strong
      signals, but the signal is semantically WRONG — it pushes Gemma AWAY
      from coordinate tokens. This is a deeper failure than Phase 1: Phase 1
      injected a weak signal that was ignored; Phase 2 injects a STRONG signal
      that is actively HARMFUL.
- [x] **T10** — Results recorded in this issue + Research 464 §4.
- [ ] **T11** — Cleanup: `rm -rf /tmp/cnn_transformer_bridge_poc` when done.

## Honest Pre-PoC Prediction

| Outcome | Probability | Actual |
|---|---|---|
| Aggregate bridge drops parse-fallback significantly | Medium | **❌ NO — delta ≈ 0 across all scales; argmax coord count dropped** |
| Full LLaVA bridge drops parse-fallback significantly | Higher | **❌ NO — P(coord) DESTROYED (0.49→0.002) at all scales; vision tokens are out-of-distribution embeddings** |
| Either bridge reaches Moka-native Go strength | Low | **❌ NO — both phases negative** |
| Either bridge beats int8 Moka (Issue 565 G5) | Very low | **❌ NO — both phases negative** |

**Post-T4 calibration finding (2026-08-01):** The cross-modal linear
structure IS non-zero — CCA captures 27.7% of the board-specific variance
on held-out data (test cR²=0.277 at k=2). This is weak but real, matching
the 'partial improvement' prediction. The signal is sufficient to justify
Phase 1 wiring (T5-T7) — the aggregate bridge is worth testing for
parse-fallback reduction.

**Post-T6 Phase 1 verdict (2026-08-01): NEGATIVE.** Despite the non-zero
cross-modal signal (cR²=0.277), the aggregate bridge does NOT improve
parse-fallback rate. P(coordinate letter) shift is essentially zero
(delta -0.0004 to +0.0001 across scales 1/10/100). The argmax coordinate
count actually dropped (18/20 → 17/20). The root cause: board-specific
variance is only 0.008% of the total residual — even a perfect linear
projection injects a tiny perturbation that gets drowned out by the
prompt-driven signal. The aggregate bridge is confirmed unviable as a
parse-fallback fix. Phase 2 (per-position LLaVA tokens) is the only
remaining modelless path.

**Key caveat from T4e:** board-specific variance is only 0.008% of the
total Gemma residual variance at layer 13 (the prompt template dominates).
This means even a perfect bridge would inject a tiny perturbation relative
to the prompt-driven signal. Layer choice matters — a later layer (20+)
where board content has more influence might carry more exploitable signal.

The load-bearing question: does Gemma, given Moka's Go vision, produce
BETTER Go decisions than Gemma without it? Go is turn-based (cold-tier,
uncapped latency), so the attention overhead is acceptable.

## GOAT Gate

- **G1** (correctness): the bridge must not crash + must produce valid
  Gemma logits. Trivially satisfied by construction.
- **G2** (perf): N/A — Go is cold-tier, uncapped latency. The bridge adds
  ~81 token attention cost for the full LLaVA path, acceptable for turn-based.
- **G3** (no-regression): default build unaffected — all behind `research`
  feature in katgpt-moka-wasm + `latent_steering_bridge` in riir-engine.
- **G5** (quality): **FAILED — both phases.** Phase 1: delta ≈ 0 (no
  improvement). Phase 2: delta = -0.49 (catastrophic regression — P(coord)
  drops from 0.49 to 0.002). The modelless bridge cannot pass the quality
  gate on either architecture.

## Final Verdict (T8-T9, 2026-08-01)

**Both phases of the modelless CNN→Transformer bridge are NEGATIVE.**

| Phase | Architecture | KL (wiring) | P(coord) delta | Verdict |
|---|---|---|---|---|
| 1 (T5-T7) | Aggregate (64-dim → residual add @ L13) | 0.0001 bits | -0.0004 | Signal too weak |
| 2 (T8-T9) | Per-position (81×32-dim → embedding prepend @ L0) | 4.46 bits | -0.492 | Signal actively harmful |

**Root cause:** the CCA cross-modal projection captures STATISTICAL
LINEAR correlation (27.7% of board-specific variance) but NOT SEMANTIC
ALIGNMENT. In Phase 1, the projected signal is so weak it's drowned by
the prompt. In Phase 2, the projected embeddings are out-of-distribution
for Gemma's layer 0 — they have non-trivial magnitude (norm ~0.72) but
wrong statistical properties, causing the attention mechanism to derail
the output entirely.

**The modelless bridge is confirmed unviable.** The honest pre-PoC
prediction ("partial improvement") was wrong on both axes — neither phase
achieves even partial improvement. The trained projection path (riir-train)
is the only remaining option for the CNN→Transformer bridge.

**Lessons learned:**
1. Cross-modal linear correlation (CCA cR²=0.277) does NOT imply usable
   signal for steering. Statistical correlation ≠ semantic alignment.
2. Embedding-layer injection (Phase 2) is MORE sensitive to distributional
   mismatch than residual-layer injection (Phase 1). A weak wrong signal
   (Phase 1) is ignored; a strong wrong signal (Phase 2) is harmful.
3. The aggregate bridge's failure mode (too weak) and the per-position
   bridge's failure mode (too strong + wrong direction) bracket the space
   of modelless injection strategies. There is no modelless "sweet spot."

## What This PoC Does NOT Prove

- It does NOT prove Gemma can play Go at Moka's level — that needs training.
- It does NOT replace Proposal 008 Phase 5 (LoRA training) — it's a modelless
  first step that could unblock or complement it.
- It does NOT address Issue 565 G5 (win-rate vs int8) — that's a separate
  question about the ternary+LoRA path.

## Cleanup

- The PoC bench stays in `riir-poc/` as a permanent regression check (per
  defend-wrong protocol §3.6).
- `rm -rf /tmp/cnn_transformer_bridge_poc` when done.
