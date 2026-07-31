# Issue 566 — CNN→Transformer Latent Bridge PoC (Modelless LLaVA-for-Go)

> **Filed:** 2026-08-01
> **Research:** [464](../.research/464_cnn_transformer_latent_bridge.md)
> **Origin:** Research 463 §2.8 (gap tracking) → Research 464 (design formalization).
> Issue 565 G1-B confirmed weight-space workarounds have hit their ceiling,
> strengthening the case for the activation-space bridge.
> **Consumer:** Proposal 008 (Go Gemma Arena) — 100% parse-fallback, needs Go understanding.
> **Type:** PoC (defend-wrong — predicted partial improvement, not full strength)
> **Status:** Active

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
      feature (off for production WASM build). **DONE 2026-08-01** — 3 unit
      tests pass: (1) tapping forward matches production within 1e-3 epsilon,
      (2) each block index 0..11 produces non-trivial trunk, (3) out-of-range
      tap is a no-op.
- [ ] **T2** — Calibration set: collect N=64 Go positions, run
      `forward_tapping_trunk` to capture trunk after block 6. This gives
      the Moka-side of the paired calibration data.
- [ ] **T3** — Gemma-side calibration: run `GemmaGoPlayer` on the same N
      positions, capture residual at layer L. This gives the paired
      (Moka features, Gemma residual) set for PCA.
- [ ] **T4** — Fit `CrossResolutionBases` via PCA on the paired set. Offline
      one-time step. Store the bases as a frozen artifact.
- [ ] **T5** — Phase 1 wiring: aggregate bridge. Global mean-max pool →
      project → `ResidualField` → `forward_with_steering`.
- [ ] **T6** — Phase 1 quality PoC: run `GemmaGoPlayer` with vs without
      the bridge on N=20 games. Measure parse-fallback rate + move quality
      (does Gemma produce legal, non-pass moves more often?).
- [ ] **T7** — Phase 1 verdict: does the aggregate bridge drop parse-fallback
      from 100%? If yes → Phase 2. If no → record negative result.
- [-] **T8** — Phase 2 wiring (full LLaVA): DEFERRED pending Phase 1 signal.
      Requires new forward variant bypassing embedding lookup for prepended
      tokens. Pursue only if Phase 1 shows the bridge carries signal.
- [-] **T9** — Phase 2 quality PoC: DEFERRED pending T8.
- [ ] **T10** — Results recorded in this issue + Research 464 §"PoC Addendum".
- [ ] **T11** — Cleanup: `rm -rf /tmp/cnn_transformer_bridge_poc` when done.

## Honest Pre-PoC Prediction

| Outcome | Probability |
|---|---|
| Aggregate bridge drops parse-fallback significantly | Medium |
| Full LLaVA bridge drops parse-fallback significantly | Higher |
| Either bridge reaches Moka-native Go strength | Low |
| Either bridge beats int8 Moka (Issue 565 G5) | Very low |

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
- **G5** (quality): **the load-bearing gate.** Parse-fallback rate drop is
  the primary metric. Move quality (legal, non-pass, sensible) is secondary.

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
