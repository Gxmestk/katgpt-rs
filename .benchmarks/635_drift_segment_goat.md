# Bench 635 — DriftSegmentStore GOAT gate (Issue 652 / Research 482)

**Date:** 2026-08-15
**Machine:** M3 Max (CPU-only bench; no GPU used — exclusivity check N/A)
**Source paper:** "Dynamic Linear Attention" ([arXiv:2606.10650](https://arxiv.org/abs/2606.10650)) — modelless adaptation
**Feature:** `drift_segment` (katgpt-kv) — **opt-in**, promotion unblocked at next re-gate
**Run:** `cargo bench --bench bench_635_drift_segment_goat --features drift_segment`

## Verdict: GOAT PASS (G1 + G2 + G3 + G4)

```
G1 needle recall (mean over 16 seeds, paired streams)
  stream          single   fixed-LFU   drift    (c)-(b)
  change-point     0.172      0.117   0.578    +46.09pp
  stationary       0.133      0.180   0.930    +75.00pp
  G1: PASS (target: change-point >= +10pp, stationary >= -2pp)

diag (seed 0, change-point): tokens=8382 slots=30 boundaries=53 merges=23 recall=0.625
G2 latency (ns/token, release)
  single=18  fixed-LFU=6  drift=12  readout=307 ns/query
  drift/single ratio = 0.69x (target: small constant — O(d + K)/token)

G4 alloc-free: 0 allocations across 1000 steady tokens (observe+readout) — PASS

GOAT verdict: PASS
```

## What this measured (the Area-7 training-free claim)

The paper proves the **co-trained** variant (soft gating during pre-training,
50B tokens, 4×A100). The published open lane (Research 482 §2.5, Area 7) is
the **training-free post-hoc** variant — nobody had measured it. This bench
does, with zero training: three arms at **matched budget** (K=30 slots ×
identical `DriftSlot` representation × identical `sigmoid_gated_readout`)
over synthetic streams, so the comparison isolates the **slot policy**:

| Arm | Boundary policy | Capacity policy |
|---|---|---|
| (a) SingleState | none (one accumulator) | none |
| (b) FixedLFU (SegmentStore policy, Plan 223b) | fixed 128-token tiles | LFU evict |
| (c) DriftSegmentStore | rising-edge drift ≥ τ | adjacent-density merge |

Streams: change-point (~32 orthogonal regimes, 100–400 tokens each,
~8400 tokens) and stationary (one 8000-token regime), both with 8
**32-token needle spans** at windowed positions; needle j's value = basis
vector e_j (orthogonal identification space); recall = probe with the clean
needle key, identified iff argmax over e-components is the true needle.
16 deterministic seeds, paired (same stream feeds all arms).

### Why the mechanism wins (measured, not asserted)

- **(a) 0.17/0.13**: the needle's contribution to a whole-stream accumulator
  is diluted to ~32/8000 — chance-level identification (1/8 = 0.125).
- **(b) 0.12/0.18**: fixed tiles straddle change points (the paper's
  Theorem 3.1 — within-block heterogeneity dominates error) AND the needle's
  value is diluted 1/128 within its tile; on 8400 tokens, 62 tiles > K=30
  forces 32 content-blind evictions (write-only LFU degenerates to FIFO).
- **(c) 0.58/0.93**: the needle is a local distribution shift → drift spike
  → its own slot (needle density ≈ 0.7 vs regime ≈ 0.15); the
  adjacent-density merge preferentially fuses boring regimes and preserves
  the needle slot at the same budget.

Diag (seed 0): 53 boundaries for 40 planted changes (+13 transient
re-fires), 23 merges under K=30 pressure — capacity + merge machinery both
exercised.

## Honest caveats

1. **Synthetic-only.** Same class of limitation as segment_checkpoint's
   blocked T10.1 NIAH (needs real model KV). The claim validated here is the
   **policy** comparison at matched budget on controlled streams — not
   end-to-end LM quality.
2. **Arm (b) LFU degeneration.** On a write-only fill stream (probes happen
   after fill) all access counts are 0, so LFU ties break FIFO-of-oldest —
   content-blind either way. Interleaved-read LFU is a different (routing)
   workload, not the paper's regime.
3. **Multi-token needles (span 32) are paper-faithful** — NIAH/MQ-NIAH
   needles are multi-token spans. Single-token needles are an adversarial
   case the paper does not claim: pair-density is n-weighted, so a
   singleton's high density barely protects it against merging with a long
   neighbor.
4. **τ=0.35 recalibrated** for the key-relative drift score (measured
   floors at D=16: σ=0.08→~0.11 mean; orthogonal-regime first-token jump
   ~0.56 rising to ~1.0). The paper's τ=0.6 is for its state-relative
   Frobenius form — different scale.
5. **Hysteresis trap (found by the unit tests):** a regime whose noise floor
   exceeds τ (σ≥0.3 → floor ≥0.39) permanently disarms the rising-edge
   detector — boundaries stop firing after it. τ must sit above the
   haystack noise floor; this is a deployment calibration constraint, not a
   defect (documented in `drift_segment/mod.rs` tests).
6. **G2 ratio 0.69×** is an artifact of the single-state arm's unoptimized
   scalar accumulate; the meaningful number is **12 ns/token absolute**
   (O(d + K) per token, d=32, K=30) and 307 ns/query readout.

## Gate table

| Gate | Target | Result |
|---|---|---|
| G1 change-point | (c)−(b) ≥ +10pp | **+46.09pp** PASS |
| G1 stationary | (c)−(b) ≥ −2pp | **+75.00pp** PASS |
| G2 latency | O(d+K)/token, reported | 12 ns/token; 307 ns/query PASS |
| G3 no-regression | segment_checkpoint + all-features unchanged | 38 + 253 lib tests PASS |
| G4 alloc-free | 0 steady-state allocs | 0 across 1000 tokens PASS |

Not UQ-bearing (recall metric, no probability/interval claim) — conformal
floor N/A per the Report-the-Floor rule.

## Promotion

**Stays opt-in (`drift_segment`)** per Issue 652 T5: promotion is evaluated
at the **next re-gate**, not immediately — this bench is synthetic-only and
the feature has no consumer wiring yet (F2/F3 below). The GOAT is
modelless-valid and unblocks that evaluation.

## Follow-ups filed (per Issue 652 T5)

- **F2 — riir-ai Issue 677**: per-NPC episodic belief memory (surprise opens
  an episodic slot, K bounded per NPC at 20 Hz; "NPCs remember the moments
  that matter, at bounded cost").
- **F3 — riir-neuron-db Issue 595**: consolidation density-aware adjacent
  merge (wake-buffer uniform average → density-aware adjacent merge; online
  cousin of `sleep_diverse`; `hope_compactor` adjacency variant).

## Numbering note

The issue text prescribed `.benchmarks/482_drift_segment_goat.md`; the
bench number was allocated as **635** per the `.benchmarks/.highwater`
discipline (482 is the *Research* number — the bench namespace had reached
634).
