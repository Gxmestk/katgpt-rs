# Issue 693: Support-instability regime detector + mode-factored state accessors (open primitive)

> **Repo:** katgpt-rs (primitive) — consumer guides: riir-ai R352
> **Filed:** 2026-08-27, from Research 513 (LpWM, arXiv:2608.22764)
> **Status:** LANDED 2026-08-28 (T1–T4; T3 quality gate FAIL-honest at the pre-registered config — signal perfect, debounce mismatched; ships opt-in)
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

- [x] **T1 — `support_instability` primitive** in `katgpt-core` beside `iou`
  (likely `functional_substitution` or a small new `regime` module; no new deps,
  no allocation): `instability = 1.0 − iou(z_t, z_{t+1})` for `&[f32]` non-negative
  pairs + a **debounced detector** struct (ring of the last K instability values,
  fixed-size array, fire when mean-over-window > θ_fire and previous state was
  calm — hysteresis via θ_calm < θ_fire). Config consts mirrored on the
  `decay_confidence` pattern (tick-indexed, no wall-clock).
  *Resolution: `functional_substitution::support_instability` (mod gated on
  `support_regime` which implies `functional_substitution_gate` — the parent
  module is itself feature-gated, so an empty dep list would gate a dead
  module). `support_instability()` fn + `SupportInstabilityDetector` ([f32; 8]
  ring, `DetectorState #[repr(u8)]`, consts THETA_FIRE=0.6 / THETA_CALM=0.35 /
  DEFAULT_WINDOW=3 / MAX_WINDOW=8, `with_params` clamps — no panic path).
  Incomparable/both-empty inputs → instability 1.0 (inherited iou
  conventions). 25 unit tests incl. the single-spike semantics pinned both
  ways.*
- [x] **T2 — mode-factored accessors**: `support_mask(&[f32], eps) -> u64/u128
  words` (bitmask, popcount active fraction) + `magnitudes(&[f32])` iterator +
  `support_jaccard(mask_a, mask_b) -> f32` (hard/exact Jaccard for the discrete
  half). These make the support/magnitude split a named, tested operation usable
  on ANY non-negative state (documented clamp/`|x|` bridge for signed states —
  the bridge is caller-side, documented, zero-cost).
  *Resolution: `SupportMask` (2×u64 words ≤ 128 dims, `from_state -> Option`,
  strict `> eps` bit test) + `jaccard()` (popcount, both-empty → 0.0) +
  `magnitudes()` filtered iterator; mode-factoring round-trip pinned by test
  (mask popcount == magnitudes count; magnitudes reconstruct the state).
  `active_fraction` is capacity-relative (×128 — the struct carries no dim
  count; documented on the method).*
- [x] **T3 — falsifiable PoC gate (the §3.6 quality defense)**: on a controlled
  toy (piecewise-affine zone dynamics in the LpWM `Piecewise` shape — per-zone
  bias fields over a 2D room, deterministic), three arms: (a) a non-negative
  latent state proxy per zone occupancy, (b) the debounced detector, (c) ground
  truth zone id. Gate: ≥90% of zone transitions detected within ≤2 ticks at ≤10%
  false-fire rate on within-zone motion, at < 100 ns/entity/tick (release bench,
  the `g2_*` debug-ignore pattern). If the gate FAILS on our state shapes, record
  the negative result — that is the answer, not a tuning loop.
  *Resolution: **GATE FAIL (honest negative), recorded in Bench 685.** Detect
  0/310 = 0.0% at the pre-registered defaults (0 fires — a hard support swap is
  a 1-tick spike ≈ 0.95 and the window-3 mean over {low, low, spike} ≈ 0.32 <
  θ_fire 0.6); false-fire 0.00% PASS; raw-signal diagnostic 100.0% (310/310
  episodes spike >0.5 at the flip tick, all in the 0.9–1.0 bin). Post-hoc
  sensitivity (labeled, not the verdict): θf=0.3/w=3, θf=0.5/w=1, θf=0.3/w=2
  each → 100% detect @ 0% false — the primitive's signal is perfect, the
  pre-registered DEBOUNCE is mismatched to 1-tick spike width. One void run
  documented (generator gave zones 2/3 no primary block under the issue's
  16-dim-block arithmetic which does not close at D=64; corrected to 8-dim
  blocks, re-run once, no detector parameter changed). Per-tick cost < 100 ns
  held (G2 below).*
- [x] **T4 — feature gate + GOAT**: opt-in feature (`support_regime`), bench vs
  the two shipped detector cousins on the same fixture (KARC-surprise needs its
  forecaster — include its cost honestly), G1 determinism (tick-indexed, no
  wall-clock), G4 alloc-free (TrackingAllocator). Promotion to default only if
  T3 passes AND a riir-ai consumer lands (the no-default-consumer rule; the
  `evpi_gate` precedent).
  *Resolution: `support_regime = ["functional_substitution_gate"]` (opt-in).
  G1 PASS (bit-identical instability streams + fire timelines across
  independent runs; LCG-only). G2 PASS — **29.29 ns/entity/tick** release
  (64 × 2000 ticks, D=64, budget 100 ns). G3 PASS (default check clean;
  `--no-default-features --features support_regime` also compiles; clippy 0
  warnings in the new files, both feature states). G4 PASS — 0 allocs via the
  repo TrackingAllocator (10_000-push loop, lib test) + the full PoC loop in a
  separate single-test CountingAllocator binary (the bench_680 convention —
  parallel sibling tests pollute a shared global counter). Cousin table
  (same streams): support 26.0 ns vs KARC D=8-canonical 377.4 ns (~15×) vs
  KARC same-width D=64 9658.7 ns (~372×); stiff_anomaly + ICT
  cited-not-measured (downstream crates; R513 §Path-0). **Stays opt-in** — T3
  did not pass at the pre-registered config and no consumer has landed; R352
  owns the spike-width tuning + re-gate on real state.*

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
