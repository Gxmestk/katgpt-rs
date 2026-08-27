# Issue 693: Support-instability regime detector + mode-factored state accessors (open primitive)

> **Repo:** katgpt-rs (primitive) — consumer guides: riir-ai R352
> **Filed:** 2026-08-27, from Research 513 (LpWM, arXiv:2608.22764)
> **Status:** OPEN
> **Priority:** P2 — cheap composition over shipped parts; quality unproven until T3

## Why

LpWM (arXiv:2608.22764) shows that in non-negative sparse latent codes, the binary
**support** carries the discrete dynamics regime (zone/contact decode 94–99%) and
**support instability** `1 − J(z_t, z_{t+1})` (soft-Jaccard / Ruzicka) detects
regime transitions at near-zero cost. Our workspace has the exact kernel
(`functional_substitution::iou` — SIMD `Σmin/Σmax`, zero-alloc) but uses it only
for same-timestep head-substitution gating. Three other regime/change detectors
ship with different, more expensive signals: KARC (forecast error — needs a
running forecaster), `stiff_anomaly` (eigenvalue windows vs frozen baseline),
ICT/R142 branching (JS-divergence over K sampled action distributions — costs K
samples/tick). The **state-side consecutive-tick** detector — the free one — is
absent.

## What ships (tasks)

- [ ] **T1 — `support_instability` primitive** in `katgpt-core` beside `iou`
  (likely `functional_substitution` or a small new `regime` module; no new deps,
  no allocation): `instability = 1.0 − iou(z_t, z_{t+1})` for `&[f32]` non-negative
  pairs + a **debounced detector** struct (ring of the last K instability values,
  fixed-size array, fire when mean-over-window > θ_fire and previous state was
  calm — hysteresis via θ_calm < θ_fire). Config consts mirrored on the
  `decay_confidence` pattern (tick-indexed, no wall-clock).
- [ ] **T2 — mode-factored accessors**: `support_mask(&[f32], eps) -> u64/u128
  words` (bitmask, popcount active fraction) + `magnitudes(&[f32])` iterator +
  `support_jaccard(mask_a, mask_b) -> f32` (hard/exact Jaccard for the discrete
  half). These make the support/magnitude split a named, tested operation usable
  on ANY non-negative state (documented clamp/`|x|` bridge for signed states —
  the bridge is caller-side, documented, zero-cost).
- [ ] **T3 — falsifiable PoC gate (the §3.6 quality defense)**: on a controlled
  toy (piecewise-affine zone dynamics in the LpWM `Piecewise` shape — per-zone
  bias fields over a 2D room, deterministic), three arms: (a) a non-negative
  latent state proxy per zone occupancy, (b) the debounced detector, (c) ground
  truth zone id. Gate: ≥90% of zone transitions detected within ≤2 ticks at ≤10%
  false-fire rate on within-zone motion, at < 100 ns/entity/tick (release bench,
  the `g2_*` debug-ignore pattern). If the gate FAILS on our state shapes, record
  the negative result — that is the answer, not a tuning loop.
- [ ] **T4 — feature gate + GOAT**: opt-in feature (`support_regime`), bench vs
  the two shipped detector cousins on the same fixture (KARC-surprise needs its
  forecaster — include its cost honestly), G1 determinism (tick-indexed, no
  wall-clock), G4 alloc-free (TrackingAllocator). Promotion to default only if
  T3 passes AND a riir-ai consumer lands (the no-default-consumer rule; the
  `evpi_gate` precedent).

## Non-goals

- No training, no RDMReg port, no JEPA (the trained half is gated at R513 §6).
- No changes to `iou`'s hot path (new consumers only).
- No riir-ai wiring in this issue (that's R352's consumer plan, post-GOAT).

## Evidence pointers

- Kernel: `crates/katgpt-core/src/functional_substitution/iou.rs` (eq. 3,
  arXiv:2606.19317 lineage; SIMD min/max; both-zero → 0.0 convention).
- Cousins + signal-diffs: R513 §Path-0 table (iou / stiff_anomaly / KARC / ICT).
- Paper anchors: LpWM Fig 2 (support Jaccard recovers zone structure, zone
  decode 0.97 via support alone), Fig 4 (instability tracks contact with TJ,
  r 0.05→0.61–0.80), Table 3 (codes 30–65% active — not ultra-sparse; pick θ
  accordingly).
