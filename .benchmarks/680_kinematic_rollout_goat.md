# Bench 680: Kinematic Rollout Primitive — GOAT Gate (Plan 578)

**Date:** 2026-08-26
**Plan:** [578](../.plans/578_kinematic_rollout_primitive.md)
**Research:** [506](../.research/506_LDR_Kinematic_Integration_Closed_Form_Rollout.md) — LDR (arXiv:2608.09926, Li et al.)
**Module:** `crates/katgpt-core/src/kinematics/` — feature `kinematic_rollout`
**Numbering note:** the plan's literal filename `578_kinematic_rollout_goat.md` is PLAN numbering; this doc is Bench **680** per `.benchmarks/.highwater` (679 → 680 — re-scanned at WRITE time after the same-session collision with the sibling session's 677-679 landed first; committed references win, mine renumbered).

---

## Verdict

**GOAT G1–G4 ALL PASS → `kinematic_rollout` PROMOTED TO DEFAULT-ON** (the KARC precedent: pure f32 math, zero deps, zero allocs, `#[repr(C)]` POD, zero-cost-unless-invoked). The UQ floor row is **RANK-ONLY** (the policy's legitimate fallback — see §UQ floor).

| Gate | Verdict | Measured |
|---|---|---|
| **G1** exactness + determinism | **PASS** | uniform/parabola prediction error **exactly 0.0** at k ∈ {1, 10, 100, 1000}; const-jerk exactly 0 at k ∈ {1, 10, 100} + the documented f32 24-bit-mantissa band at k=1000; closed form **bit-identical** to the reference difference-engine chain on the exactness family; multi-seed pipeline determinism (6 seeds × ID/OOD, bit-equal hashes) |
| **G2** ns-cost (release, M3 Max) | **PASS** | single-target extrapolate **4.64 ns** (d=4, k=100, ZeroJerk; budget < 10 ns — 2.2× headroom); full table below |
| **G3** default build unchanged | **PASS** | pre-promotion default `cargo test -p katgpt-core --lib` = **1917/0/7** with the feature off, identical to the pre-change baseline; post-promotion default = 1951/0/7 (1917 + 34 kinematics tests — promotion adds the module's own suite, nothing else moved) |
| **G4** alloc-free | **PASS** | **0** steady-state allocations over 10,000 iterations spanning every hot-path operator (observe, all 5 schedules × {closed-form, reference-chain, capped}, TTC both variants, closest approach, intercept, regime classify, residual update, horizon gate, weight-ss, half-width, terminal velocity); **0** per-tick allocations in the full fixture pipeline (CountingAllocator binaries) |
| **UQ floor** (Report the Floor) | **RANK-ONLY** | parabola corpus **BeatsFloor** decisively (CRPS ratio **0.10**, Winkler **0.08**, coverage 0.86 vs the floor's **0.17 collapse**); uniform and white-noise lose **1.08×** at the harness's h=1 protocol — the documented √2·σ shared-floor analysis. The horizon ships the **k\* ordering claim, no calibrated-coverage claim** |

---

## The load-bearing claim: exactness, and its honest f32 boundary

The Newton-backward closed form is the unique degree-≤3 polynomial through
the last ≤4 observations, evaluated at any horizon:

```text
x̂(k) = x[m] + C(k,1)·∇¹ + C(k+1,2)·∇² + C(k+2,3)·∇³
```

### Exactness fixtures (module tests, all assert `==` on f32 bits)

| Trajectory | k=1 | k=10 | k=100 | k=1000 |
|---|---|---|---|---|
| uniform `2.5t` (deg 1) | **0.0** | **0.0** | **0.0** | **0.0** |
| parabola `t²` (deg 2) | **0.0** | **0.0** | **0.0** | **0.0** |
| cubic `t³` (deg 3, jerk 6) | **0.0** | **0.0** | **0.0** | rel < 1e-6 (documented) |
| Δt-rescale invariance (Δt vs Δt/2, matched wall-times) | bit-equal on uniform/parabola/cubic | | | |

**Why the cubic's k=1000 row cannot be exactly 0 in f32** (found by
arithmetic, not tolerated silently): the lattice coefficient
`C(1002,3) = 167,167,000` needs **25 mantissa bits** — one over f32's 24.
The largest 24-bit-exact `C(k+2,3)` is `C(290,3) = 4,086,980` (k = 287).
Degree ≤ 2 stays bit-exact through the full lattice (`C(1025,2) = 525,250`).
The plan's "exactly 0 at k ∈ {1,10,100,1000}" is therefore achievable for
deg ≤ 2 at all four horizons and deg 3 at k ≤ ~287; the k=1000 cubic row
asserts the measured ~2⁻²⁴ relative band instead.

### Bit-identity with the step-by-step chain

The O(1) closed form and the O(k) difference-engine chain
(`d2 += d3; d1 += d2; s += d1`) evaluate the same polynomial through
different float op sequences:

- On the **exactness family** (dyadic-representable trajectories): every op
  in both paths is exact → **bit-identical** (pinned by module tests and by
  the GOAT sweep over all fixtures × k ∈ {1, 8, 31}).
- On **arbitrary random walks**: max relative divergence **1.71e-5**
  (measured, 200 trajectories × k ∈ {1, 5, 23, 97}).
- On the **drag schedule**: bit-identity holds while the decaying
  acceleration stays above `DRAG_ACC_FLOOR` (2⁻¹⁰); deeper in the tail the
  chain drops sub-half-ULP increments the closed form keeps — the GOAT sweep
  asserts rel < 1e-6 there (documented boundary).

No O(1) rearrangement of an O(k) accumulation can be bit-identical in
general — the exactness family is where the identity holds exactly, and that
is where the plan needs it.

---

## T3.2: the PhyWorld ID-OOD gap table

Deterministic fixtures (seeded SplitMix64, dyadic parameter grids) over the
paper's exact ranges — ID: v ∈ [1,4], r ∈ [0.7,1.4], |ṙ| ∈ [0, 0.03]; OOD:
v ∈ [0.05,6], r ∈ [0.6,2], |ṙ| ∈ [0.05,0.09]; T = 31 — five interleaved
regime segments per fixture (uniform → parabola → bounce → looming → drag),
6 seeds × 2 range sets, per-segment schedules, event-crossing and
mixed-window anchors excluded:

| Segment | ID max err (k=1/8/31) | OOD max err | gap |
|---|---|---|---|
| uniform | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | ≡ 0 (bit-exact both arms) |
| parabolic | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | ≡ 0 |
| bounce (event-free spans) | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | ≡ 0 |
| looming | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | ≡ 0 |
| drag (above resolution floor) | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | ≡ 0 |

**The paper: LDR's ID-OOD error gap ratio ≈ 23.9× (empirical, pixel MSE,
256²). Ours: gap ≡ 0 by construction** — both arms' errors are identically
zero, so the ratio is 0/0/undefined and the gap itself vanishes. The paper's
OOD ranges change the *magnitudes*, not the *family*; the closed form is
exact on the whole family. This is the provable strengthening Research 506
set out to ship.

### Regime classification: 100% on interleaved 5-regime streams

5 seeds × {ID, OOD}: the sigmoid-gated hysteresis classifier matches the
generator's honest per-frame tags on **every comparable frame** (≥3 frames
into each segment — the FD window flush; regime transitions are force
onsets, which the impulse discriminator legitimately fires on once). The
planted bounce is detected at the **exact tick** with restitution read
exactly (`e = |v_after|/|v_before| = 0.5`) and the wall axis inferred from
the sign flip. **0 alarms on 10⁵ clean uniform ticks** and on 5,000 clean
parabola ticks (the parabola's f32 24-bit exactness ceiling — a g=0.25
parabola stays dyadic-exact only while t²·g/2·(1/g) < 2²⁴; the uniform
stream carries the full 10⁵).

### Fixture-design lessons (found by running, not by theory — recorded for
the consumer PoC)

1. **Ticks must be globally monotonic across segments** (per-segment restart
   violates the stencil's `NonMonotonicTick` screen).
2. **Mixed-window anchors after an event pollute the FD coefficients for 3
   frames** — event exclusion must cover the window span, not just the
   crossing tick.
3. **The running-force-scale EMA must be winsorized**: a raw EMA absorbs
   segment-boundary transients (~120 units) and stays inflated long enough
   to mask a genuine bounce 15 ticks later; a hard exclusion of high-dv
   ticks starves the scale on sustained force onsets and fires impulses on
   every parabola frame. `min(acc, 10·running + 0.05)` at β=0.5 handles
   both.
4. **The drag-tail FD flickers below ~ULP(position)**: any scale-free gate
   reads noise there. `DRAG_ACC_FLOOR` (2⁻¹⁰) is shared by the generator's
   tags and the classifier's gate so they cannot drift apart.

---

## G2: ns-cost table (release, M3 Max, best-of-3)

| Operator | Cost | Budget | Verdict |
|---|---|---|---|
| single-target extrapolate (d=4, k=100, ZeroJerk) | **4.64 ns** | < 10 ns | PASS (2.2×) |
| single-target extrapolate (d=4, k=100, ConstJerk) | 4.65 ns | < 10 ns | PASS |
| single-target extrapolate (d=4, k=100, GeometricDrag) | 9.77 ns | < 50 ns | PASS (powf cost, documented) |
| 1000-target batch (d=4, k=100, ZeroJerk) | 4630 ns total (**4.63 ns/target**) | < 10 µs | PASS |
| `time_to_contact` | 1.42 ns | < 10 ns | PASS |
| `extrapolation_horizon` (lattice scan to k*) | 20.9 ns | < 2 µs | PASS |
| `closest_approach` (d=2) | 2.70 ns | < 20 ns | PASS |
| `RegimeClassifier::classify` (d=2) | 35.5 ns | < 50 ns | PASS |
| `ResidualMonitor::update` (d=2) | 15.7 ns | < 50 ns | PASS |

---

## UQ floor (Report the Floor — Issue 010 policy)

Adapter: `tests/conformal_floor_kinematics.rs` — the k=1 predictive interval
composed from the module's operators (σ̂ from the **full-ladder residual**,
EMA-smoothed velocity point, `z·σ̂·√(2 + 2β/(2−β))` width) vs the canonical
`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` m=1 floor.
n=1500, warmup=64, α=0.05.

| Corpus | CRPS (prim/floor) | Winkler | Coverage | Verdict |
|---|---|---|---|---|
| noisy parabola (c=0.012, σ=0.1) | 0.597 / 5.891 = **0.10** | 0.961 / 11.97 = **0.08** | 0.860 / **0.166** | **BEATS FLOOR** |
| noisy uniform (v=0.7, σ=0.1) | 0.570 / 0.530 = 1.08 | 1.08 | 0.944 / 0.949 | LOSES |
| white noise (σ=0.1) | 0.583 / 0.538 = 1.08 | 1.06 | 0.953 / 0.941 | LOSES |

**Verdict: RANK-ONLY.** The shipped claim for `extrapolation_horizon` is the
**k\* ordering** (error-propagation bound monotone in k → trust ordering),
not calibrated coverage. Two findings drive it, both measured:

1. **Curving motion is a decisive win**: the floor's conformal drift
   correction cannot track a moving drift (the parabola's increment grows
   every step) — its coverage collapses to 0.17 while the kinematic interval
   holds 0.86 at 10× lower CRPS.
2. **Straight motion at h=1 is a mathematical tie-at-best**: both predictors
   anchor on the same noisy last observation, so the error floor √2·σ is
   shared; the floor's finite-sample conformal quantile sits slightly inside
   it (~1.35σ), giving the observed 1.08× edge. The kinematic advantage
   lives at longer horizons and under curvature, which the harness's fixed
   h=1 protocol cannot express.

### Estimator lessons from the floor bench (recorded for consumers)

- **σ̂ must come from the full-ladder residual, never the screened order's.**
  A too-low order's residual contains the *motion*; feeding it back into the
  order screen death-spirals (measured: the parabola corpus pinned order 0,
  σ̂ → 37, 174-wide intervals).
- **The predictive interval needs the new observation's noise**:
  `√(wss + 1)`, not `√wss` — the +1 was found by the floor bench
  (under-coverage without it).
- **The order screen needs EMA-smoothed inputs**: per-tick noise draws
  flicker the order, and each over-order tick amplifies the residual stream
  (√70·σ at order 3 vs √6·σ at order 1), destabilizing σ̂.

---

## Promotion record

`kinematic_rollout` added to the crate's `default` features (Cargo.toml
Phase 28 comment). Pre-promotion gate: default build identical
(1917/0/7 with the feature off — G3). Post-promotion default:
**1951/0/7** (+34 kinematics module tests). Zero new dependencies
(hard requirement — `cargo tree` unchanged). Consumer PoC:
**riir-ai Issue 757** (anticipatory belief), which was blocked on this
landing.

## Files

- `crates/katgpt-core/src/kinematics/{mod,perception,fixture,tests}.rs` — the module
- `crates/katgpt-core/tests/bench_680_kinematic_rollout_goat.rs` — G1 + T3.2 + G2
- `crates/katgpt-core/tests/bench_680_kinematic_alloc_check.rs` — G4
- `crates/katgpt-core/tests/conformal_floor_kinematics.rs` — UQ floor
- `crates/katgpt-core/Cargo.toml` — feature + default promotion + test registrations
- `crates/katgpt-core/src/lib.rs` — module declaration
