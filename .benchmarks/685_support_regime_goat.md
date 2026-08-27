# Bench 685: Support-Instability Regime Detector — GOAT + Pre-Registered T3 PoC

**Status:** T3 QUALITY GATE **FAIL (honest negative — the pre-registered debounce config is mismatched to 1-tick support swaps; the raw signal is PERFECT: 100% spike / 0% false)**; G1 PASS, G2 PASS (29.29 ns/entity/tick), G3 PASS, G4 PASS. Ships opt-in; promotion blocked on a riir-ai consumer (R352).

> **Issue:** katgpt-rs #693 · **Research:** 513 · **Paper:** LpWM, [arXiv:2608.22764](https://arxiv.org/abs/2608.22764) (Kuang et al. 2026-08-24) · **Feature:** `support_regime` (implies `functional_substitution_gate`) · **Date:** 2026-08-28 · **Box:** M3 Max (release numbers)

---

## What shipped (T1 + T2)

`crates/katgpt-core/src/functional_substitution/support_instability.rs` — beside `iou`, consuming it temporally:

| Item | Description |
|---|---|
| `support_instability(z_t, z_t1) -> f32` | `1.0 − iou(z_t, z_{t+1})` (Ruzicka complement). Incomparable/both-empty → `1.0` (inherited iou conventions). Non-negative contract; signed-state bridge (`max(0.0)`/`abs()`) is caller-side, documented, zero-cost. |
| `SupportInstabilityDetector` | Zero-alloc debounced detector: `[f32; 8]` ring + head/len, `DetectorState {Calm=0, Firing=1}` (`#[repr(u8)]`), hysteresis (fire on window mean > θ_fire from Calm; calm on mean < θ_calm). Tick-indexed (the `decay_confidence` pattern — `Instant` appears nowhere in the logic). Cold detectors trust their partial window (documented). `with_params` clamps window to `1..=8` and repairs inverted thresholds — no panic path. |
| Consts | `THETA_FIRE = 0.6`, `THETA_CALM = 0.35`, `DEFAULT_WINDOW = 3`, `MAX_WINDOW = 8`. |
| `SupportMask` | T2 discrete half: 2×`u64` words + popcount `active`, `D ≤ 128` (`from_state` → `None` on empty/over-long; bit set iff `z[i] > eps` strictly). `jaccard` = exact popcount Jaccard (both-empty → 0.0, iou convention). `active_fraction` is capacity-relative (×128 — the struct carries no dim count; documented). |
| `magnitudes(&[f32])` | T2 continuous half: zero-alloc filtered iterator over `(i, z[i])` for `z[i] > 0.0`. Mode-factoring round-trip pinned by test (mask popcount == magnitudes count; magnitudes reconstruct the state). |

Module docs carry the signal-class table (this module = state-side consecutive-tick; KARC = forecast-error; `stiff_anomaly` = spectral-window vs frozen baseline; ICT = policy-side JS over K samples) and the mode-factoring citation (support = discrete regime, 94–99% zone decode; magnitudes = continuous state, R² = 0.94).

## The toy world (T3 fixture)

LpWM `Piecewise` shape, fully deterministic (LCG-only; no wall-clock, no HashMap iteration anywhere):

- 2D room `[0,1]²`, K=4 quadrant zones; 32 entities × 2000 ticks; ~10 episodes each (310 total, all within the 8–16 design band).
- Trajectory per entity: dwell 120–220 ticks (wiggle step 0.02, margin-clamped inside the zone — dwells cannot cross), then cross to an **adjacent** zone (base step 0.03 toward the target + the same 0.02 wiggle — the only chatter source, emergent).
- Latent proxy: `z = clamp0(W_zone · [x, y, 1])`, `D = 64`. Dims `0..32` = shared background (weights 0.01–0.15, active in all zones); dims `32..64` = four 8-dim per-zone primary blocks (weights 0.5–2.0; zone k owns block k).
- **Spec deviation (documented):** the task sketch said 16-dim primary blocks, whose arithmetic (32 bg + 4×16 = 96) does not close at the pinned D=64. The first generator draft implemented that sketch literally and silently gave zones 2/3 **no primary block** (their `(i−32)/16 == z` condition is unsatisfiable for z ∈ {2,3} within dims 32..64) — 73/310 transitions spiked at only ~0.26 (background-only latents). That run is void as a **bug run**; the fixture was corrected to 8-dim blocks (4×8=32, preserving D=64 + the half-background structure) and re-run. No detector parameter was touched between the two runs.

Measured signal shape on the corrected fixture: within-zone instability ≈ 0.01–0.03 (wiggle), crossing-pre-boundary ≈ 0.02–0.05, **transition spike ≈ 0.94–0.97 for a single tick** (hard primary-mass swap at the boundary tick; the toy's crossings are clean single flips — chatter did not materialize at step 0.03 vs wiggle 0.02).

## T3 — pre-registered gate (ONE run at defaults)

**Pre-registered before evaluation** (test file header, committed with the run): detect = ≥90% of episodes fire within ≤2 ticks of the episode's first zone-change tick; false-fire = ≤10% of fires outside every ±2-tick window of any zone-change tick; population ≥32 entities × 2000 ticks; detector at `THETA_FIRE=0.6 / THETA_CALM=0.35 / window=3`. Assertion policy: detect-rate is printed + recorded here (experimental result — the issue's own negative-result path), the false-fire safety axis is asserted, G1/G2/G4 are asserted invariants.

| Axis | Threshold | Measured | Verdict |
|---|---|---|---|
| Detect rate | ≥ 90% | **0 / 310 = 0.0%** (0 fires at defaults) | **FAIL** |
| False-fire rate | ≤ 10% | **0.00%** (0 fires, 0 false) | **PASS** |
| Mean latency | ≤ 2 ticks | n/a (no fires) | n/a |
| Raw signal at flip (diagnostic, NOT the gate) | — | **100.0%** (310/310 episodes spike >0.5 at the flip tick; all in the 0.9–1.0 bin) | — |

**The mechanism, decomposed honestly.** The failure is entirely in the DEBOUNCE, not the signal. A hard support swap is a 1-tick instability spike (~0.95); the pre-registered window-3 mean over {low, low, spike} ≈ 0.32 < θ_fire 0.6 → the detector never leaves Calm. Every config that lets a single tick's evidence reach the threshold recovers the signal perfectly:

| Config (post-hoc, ≤4 pts, diagnostic only) | detect% | false% | fires |
|---|---|---|---|
| θf=0.6, θc=0.35, w=3 (pre-registered) | 0.0 | 0.0 | 0 |
| θf=0.3, θc=0.15, w=3 | 100.0 | 0.0 | 310 |
| θf=0.5, θc=0.2, w=1 | 100.0 | 0.0 | 310 |
| θf=0.3, θc=0.15, w=2 | 100.0 | 0.0 | 310 |

The single-spike semantics are pinned by unit tests both ways (`detector_single_spike_does_not_fire_at_defaults`, `detector_with_params_window_one_fires_on_single_spike`), so this trade is a documented property, not a surprise.

**Actionable for the consumer (R352):** the debounce window must be matched to the expected spike WIDTH of the regime flips in the consumer's state. For 1-tick hard swaps: window=1 (θ_fire ≈ 0.5) or window=3 with θ_fire ≈ 0.3. For multi-tick drifts (LpWM's smoothed codes flip over ~2–4 frames), the pre-registered defaults are the right shape. The consumer tunes `with_params` to its state; the shipped consts stay the pre-registered record.

**Verdict recorded as FAIL at the pre-registered config.** The primitive still ships opt-in: the signal axis (100% detection at 26 ns, zero false fires at every tested config) is exactly the LpWM claim reproduced on our construction; the negative is about one specific debounce configuration, pre-registered and honestly recorded — not tuned past (the sensitivity table exists to inform R352, not to relabel the gate).

## GOAT gates

| Gate | Result |
|---|---|
| **G1 determinism** | **PASS** — two independent runs of the whole PoC (generation + detection) produce bit-identical instability streams (`to_bits` compared) and identical fire timelines, 32 entities × 2000 ticks. LCG-only construction; debug and release runs produced identical verdict tables. |
| **G2 perf** | **PASS — 29.29 ns/entity/tick** (release, best-of-3, 64 entities × 2000 ticks, D=64, full path: `iou` + ring update + fire logic; budget < 100 ns, 3.4× headroom; observed 25.9–29.3 across two release runs on this box). Debug variant is `#[cfg_attr(debug_assertions, ignore)]`-locked (the `g2_*` house pattern). |
| **G3 no-regression** | **PASS** — `cargo check -p katgpt-core` (default) clean; the feature is opt-in and implies `functional_substitution_gate`, so default builds are untouched by construction. `--no-default-features --features support_regime` also compiles (both lib + test target). `cargo clippy` default + `--features support_regime --all-targets`: 0 warnings in the new files. |
| **G4 alloc-free** | **PASS** — two independent checks: (a) lib unit test via the repo `TrackingAllocator` — 10_000 pushes + the `iou` calls feeding them = **0 allocations**; (b) separate single-test binary (`bench_685_support_regime_alloc_check`, the `bench_680` CountingAllocator convention — split from the GOAT binary because parallel sibling tests pollute the global counter) — the full PoC loop (projection → iou → push) over 2000 ticks = **0 allocations**. |

## Cousin cost table (same fixture, release)

| Detector | Signal class | ns/entity/tick | Source |
|---|---|---:|---|
| **support-instability (this)** | state-side consecutive-tick support overlap | **31.2** | measured (bench_685 cousin arm, 8 entities × 1599 ticks, D=64; 26.0–31.2 across two release runs) |
| KARC surprise, same fixture (D=64 M=8 K=4, d_h=2048) | forecast error (fitted forecaster: forecast + observe + ‖x−û‖₂) | **12127.3** | measured (same latent streams; fit on 400 ticks/entity, λ=1e-2 for numerical stability on the rank-deficient toy features — fit quality is not the claim, cost is) |
| KARC surprise, canonical HLA shape (D=8 M=8 K=4, d_h=256) | forecast error | **396.8** | measured (8-dim slice of the same streams — cost-only arm) |
| `stiff_anomaly` (katgpt-spectral) | eigenvalue window vs frozen baseline | **cited-not-measured** | downstream crate (no dev-dep added); its own GOAT (`.benchmarks/037`) is correctness-only with no latency axis; structurally an eigendecomp + window buffering — a heavier spectral-window class |
| ICT branching (riir-ai R142/R270) | policy-side JS-divergence over K sampled action distributions | **cited-not-measured** | R513 §Path-0; costs K samples/tick |

Headline (same-run ratios): on identical latent streams the support-instability detector is **~12.7× cheaper than KARC at its canonical config and ~389× cheaper than KARC at the fixture's own width**, while the T3 run shows its raw detection quality on hard regime swaps is 100% at 0% false fire. The cousins carry strictly more information (KARC predicts, ICT reads the policy) — the cost row is about the regime-transition EVENT detection task specifically.

## Honest findings / limitations

1. **The T3 negative is a debounce-config mismatch, not a primitive defect** — but it is recorded as the gate verdict per the pre-registration, and the shipped consts remain the failing config. Any consumer must tune `with_params` to its spike width (or land a consumer that motivates re-pinning the defaults).
2. **Toy proxy, not real game state.** `clamp0(W_zone·[x,y,1])` with hard quadrant assignment produces *perfectly clean* 1-tick support swaps. Real latents (HLA `style_weights`, shard codes) flip over multiple ticks with partial overlap — closer to LpWM's 30–65%-active codes, where the pre-registered defaults are the right shape. The toy's cleanliness is precisely why the debounce missed.
3. **D=64, 4 zones, single geometry.** No claim about other widths/densities; `SupportMask` caps at 128 dims.
4. **Chatter did not materialize** at the toy's step/wiggle ratio (0.03/0.02): crossings are single-flip (fires == episodes at every firing config). The chatter-tolerance path is exercised only by unit tests, not the PoC.
5. **`active_fraction` is capacity-relative** (×128) because the pinned struct shape carries no dim count — documented on the method; compare `active()` counts or rescale at call sites.
6. **One void run documented:** the first T3 execution hit the 16-vs-8-dim generator bug (zones 2/3 background-only, 73/310 low spikes); fixture corrected, re-run once. No detector parameter changed between runs; both runs' numbers preserved here.
7. **KARC fit λ raised to 1e-2** on the toy streams (rank-deficient features); fit quality is irrelevant to the cost row but noted for reproducibility.

## Promotion verdict

**Stays opt-in** (`support_regime = ["functional_substitution_gate"]`), per the no-default-consumer rule — the `evpi_gate` precedent: T3 did not pass at the pre-registered config, and no consumer has landed. Consumer plan: **riir-ai R352** (regime-gated cognition — support-stable ⇒ cheap tier suffices, escalate on flips), which owns the spike-width tuning decision and the re-gate on real state.

Run it:

```bash
cargo test -p katgpt-core --features support_regime --test bench_685_support_regime_goat --release -- --nocapture
cargo test -p katgpt-core --features support_regime --test bench_685_support_regime_alloc_check
cargo test -p katgpt-core --features support_regime --lib functional_substitution::support_instability
```
