# Plan 460: FlowField × DualLeoMixer Post-Max Fusion

**Status:** NOT STARTED
**Branch:** `develop`
**Repo:** `katgpt-rs`
**Predecessor:** [Plan 459](459_flow_field_dual_leo_mixer_fusion.md) (pre-max Q-slice fusion — G1–G4 PASS, G5 FAIL)
**Started:** —
**Completed:** —

## Context (the gap Plan 459 left open)

Plan 459 fused `DualLeoMixer` into `FlowFieldCache` by mixing **raw Q-slices**
(`α·Q_leo[a] + (1-α)·Q_uvfa[a]` per action) *before*
`LeoPotentialGrid::from_q_values` applies its nonlinear `max-over-actions`
step. The fusion is correct (G1 bit-identity, G2 ≤1.13× perf, G3 no-regression,
G4 alloc-free) but **does not improve navigation quality** at any α ∈ {0.1..0.9}
on a synthetic 2D landscape (G5 best case: 25.9% stuck-NPC reduction at α=0.1,
short of the 30% gate; paper default α=0.3 only 3.7%).

Root cause (from
[`.benchmarks/459_flow_field_dual_leo_mixer_goat.md`](../.benchmarks/459_flow_field_dual_leo_mixer_goat.md)
§"Root cause"):

```
max_a (α·Q_leo[a] + (1-α)·Q_uvfa[a])  ≠  α·max_a Q_leo[a] + (1-α)·max_a Q_uvfa[a]
```

The α-mix on raw Q-slices is washed out by the max-pool *before* the FFT
sees it. The LEO decoy peak survives the max even at low α because the mix
is per-action, pre-max.

**Plan 460 attacks the root cause directly.** Instead of mixing pre-max
Q-slices, build two complete `LeoPotentialGrid`s (each with max-pool already
applied) and mix the resulting **post-max potentials**:

```
potential_mixed[x,y] = α · potential_leo[x,y] + (1-α) · potential_uvfa[x,y]
```

This is **linear in the field the FFT sees**. If the UVFA student's post-max
potential is genuinely cleaner (no LEO-style decoy peaks — the synthetic
`BenchUvfaStudent` from Plan 459's bench fits this shape: sharp + unimodal),
low-α post-max mixing should wash out the LEO decoy **linearly**, which is
exactly the regime where the FFT low-pass preserves the effect.

## Hypothesis (what we expect to prove)

On the same 64×64 synthetic landscape used by Plan 459's bench (broad
multimodal LEO teacher with a decoy peak at (16,16) + sharp unimodal UVFA
student), post-max mixing should:

1. **H1 (quality gain — the gate Plan 459 missed):** Stuck-NPC reduction
   ≥30% vs LEO-only baseline at some α ∈ {0.1..0.9}. Predicted best α ≈ 0.2–0.3
   (low-UVFA-weight regime where the decoy washes out but the goal peak
   survives). Mechanism: linear blend of two post-max potentials, FFT is
   linear, so the α-mix transfers cleanly to the smoothed gradient field.
2. **H2 (perf overhead):** Post-max dual path costs ≤1.5× the LeoOnly
   single-head cache-miss latency. Two `from_q_values` builds + one cell-wise
   blend + one FFT — bounded by `O(2·cells·actions + cells + cells·log(cells))`,
   vs single-head `O(cells·actions + cells·log(cells))`. The FFT dominates
   so the ratio should stay near Plan 459's 1.11×.
3. **H3 (bit-identity):** `get_or_compute_dual_postmax` with `ActingMode::LeoOnly`
   produces bit-identical `FlowField` to `get_or_compute` with the same head.
   (`α=1.0` mix is identity; `UvfaOnly` is `α=0.0`.)
4. **H4 (modelless):** α is a deterministic scalar; blend is a cell-wise
   affine combination. SATISFIED BY CONSTRUCTION.

## Constraint Check

- **Modelless mandate (katgpt-rs/AGENTS.md):** Affine blend of two
  post-max potentials is deterministic, modelless, no gradients. ✅
- **Sync-boundary rule:** `FlowField` is local latent structure, never
  synced. ✅
- **Manifold geometry / Stokes:** Same 2D regime as Plan 459 (d=2).
  FFT is a volume op on the post-max potential grid; boundary-vs-volume
  perf rule doesn't apply. ✅
- **Alloc-free hot loop (G4):** The blend must write into a pre-allocated
  buffer (`blend_into`), not allocate a fresh `Vec` per cache-miss. The
  cache already owns `potential_buf`; reuse it. ✅
- **Feature gate discipline:** New API ships behind existing `dual_leo` +
  `flow_field_nav` features. No new feature flag — composition of two
  default-on primitives, same as Plan 459.
- **Plan 459 stays landed.** This plan adds a *sibling* API
  (`get_or_compute_dual_postmax`), not a replacement. Callers choose
  pre-max (Plan 459) or post-max (this plan) based on which fusion point
  fits their heads. Doc-comments on both must cross-reference.

## Design decision: API shape (Option C — primitive + cache consumer)

Two-layer API, mirroring the `LeoPotentialGrid` / `FlowFieldCache` split:

1. **Primitive:** `LeoPotentialGrid::blend_into(&self, other: &Self, alpha: f32, out: &mut [f32])`
   — writes `α·self.potential + (1-α)·other.potential` into a caller-provided
   buffer. Pure, unit-testable, no allocation. Acts only on `potential`;
   `blocked` is OR'd in the cache layer (an obstacle on either side is an
   obstacle).
2. **Cache consumer:** `FlowFieldCache::get_or_compute_dual_postmax<H1: LeoHead, H2: LeoHead, M: DualLeoMixer>`
   — mirrors the Plan 459 `get_or_compute_dual` signature. Builds two grids,
   blends potentials into `self.potential_buf`, then runs the existing FFT +
   gradient tail.

**Why not just one method:** the primitive is the testable invariant
(linearity, identity at α=1.0, identity at α=0.0); the cache method is the
consumer-facing surface. Splitting them keeps the unit tests fast (no FFT,
no cache state) and the integration tests honest (full pipeline).

**Refactor prerequisite (T1):** Extract `compute_from_grid(&mut self, grid: LeoPotentialGrid, goal_id, tick)`
from the existing `compute_from_q_slice` (everything after `from_q_values`:
obstacle inflate → FFT → gradient → cache insert). This lets both pre-max
(`compute_from_q_slice` keeps its current shape) and post-max
(`compute_from_potential_blend` calls `compute_from_grid`) share the FFT +
gradient tail without duplication. Pure refactor, no behavior change — G3
must verify bit-identity against the pre-refactor field on the same inputs.

## Tasks

- [ ] T1: Refactor — extract `compute_from_grid` helper from `compute_from_q_slice`
      (everything after `LeoPotentialGrid::from_q_values`). No behavior change.
      Verify G3 (existing tests, including Plan 459's 5 dual tests) still pass.
- [ ] T2: Add `LeoPotentialGrid::blend_into(&self, other: &Self, alpha: f32, out: &mut [f32])`
      — pure primitive. Asserts `out.len() >= self.cells()`. Writes
      `α·self.potential[i] + (1-α)·other.potential[i]`. No allocation.
- [ ] T3: Add `FlowFieldCache::get_or_compute_dual_postmax<H1, H2, M>` —
      signature mirrors `get_or_compute_dual`. Internally: build two grids,
      OR their blocked bitfields, blend potentials into `self.potential_buf`,
      then call `compute_from_grid` on the assembled grid.
- [ ] T4: Unit tests
  - [ ] T4.1: `blend_into` at α=1.0 writes `self.potential` bit-identically.
  - [ ] T4.2: `blend_into` at α=0.0 writes `other.potential` bit-identically.
  - [ ] T4.3: `blend_into` at α=0.5 produces elementwise mean.
  - [ ] T4.4: `blend_into` asserts on length mismatch (panics on `out.len() < cells`).
  - [ ] T4.5: `get_or_compute_dual_postmax(LeoOnly)` is bit-identical to
        `get_or_compute` with the same head, same state, same goal.
  - [ ] T4.6: `get_or_compute_dual_postmax(UvfaOnly)` matches the UVFA-head-only
        `get_or_compute` field.
- [ ] T5: Extend `benches/dual_flow_field_bench.rs` — reuse the Plan 459
      200-NPC gradient-follow simulator + mock heads. Add:
  - [ ] T5.1: `dual_postmax_lc_alpha_03` quality + perf measurement (paper default).
  - [ ] T5.2: `dual_postmax_alpha_sweep` — α ∈ {0.1, 0.2, 0.3, 0.5, 0.7}, Lc mode.
  - [ ] T5.3: Print side-by-side vs Plan 459's pre-max numbers for direct
        comparison in the bench output.
- [ ] T6: Run GOAT gate. Capture results in
      `.benchmarks/460_flow_field_dual_leo_postmax_goat.md`. Side-by-side
      table vs Plan 459 must be included (this is the whole point — prove
      or disprove that the pipeline-stage change moves the needle).
- [ ] T7: Promotion decision
  - If G5 passes (≥30% stuck-NPC reduction at some α) → promote
    `get_or_compute_dual_postmax` as the **recommended** dual path in the
    doc-comment, demote Plan 459's pre-max path to "compatibility only".
  - If G5 fails again → keep both APIs landed (both correct + cheap), update
    doc-comments to say honestly "neither pre-max nor post-max dual fusion
    improves navigation on synthetic 2D grids; the gain — if any — requires
    real CivLeoNet + UVFA evidence". Do **not** open a third pipeline-stage
    variant (QGF `DualLeoOracle` is a different axis — test-time, not
    geometry).

## GOAT Gate Definition

| Gate | Criterion | Target | Method |
|------|-----------|--------|--------|
| **G1** | Correctness | `get_or_compute_dual_postmax(LeoOnly)` produces bit-identical `FlowField` to `get_or_compute` with same LEO head, same state, same goal | Unit test bit-compare |
| **G2** | Perf overhead | Cache-miss latency postmax/LeoOnly-single ratio | ≤ 1.5× on 64×64 grid (same `std::time::Instant` harness as Plan 459) |
| **G3** | No-regression | All existing `flow_field_nav` + `dual_leo` tests pass (Plan 459's 5 dual tests included) | `cargo test -p katgpt-core --features flow_field_nav,dual_leo --lib` |
| **G4** | Alloc-free hot path | `blend_into` writes into pre-allocated `&mut [f32]`; no `Vec::new` in compute path | Code inspection |
| **G5** | **Quality gain** (the gate Plan 459 missed) | Stuck-NPC reduction ≥30% vs LEO-only baseline at some α ∈ {0.1..0.9} | New bench, 200 NPC starts, 64×64 grid, same landscape + mock heads as Plan 459 |

**Promotion rule:** If G1–G4 pass AND G5 shows ≥30% stuck-NPC reduction at
any swept α → promote `get_or_compute_dual_postmax` as the **recommended**
dual path. Update the Plan 459 doc-comment to mark `get_or_compute_dual`
(pre-max) as "compatibility / parity with QGF pre-max mix". Stays opt-in
— `get_or_compute` (single-head) remains the lowest-latency path for
callers without a UVFA student.

**Demotion rule (honest stop):** If G5 fails across the full α sweep, the
plan is DONE as a negative result. We do NOT open a third pipeline-stage
variant. The conclusion becomes: "synthetic 2D landscape cannot
distinguish LEO+UVFA fusion from LEO-only — real-network evidence
required (riir-games-civ wiring, separate plan in riir-ai)." Doc-comment
must say so on both `get_or_compute_dual` and `get_or_compute_dual_postmax`.

## Predicted outcome (the honest pre-registration)

Based on the Plan 459 root-cause analysis, post-max mixing is the **strongest
available attack** on G5 within the modelless constraint:

- If the UVFA mock's post-max potential is genuinely cleaner than the LEO
  mock's post-max potential (it should be — the mock was designed that way:
  sharp + unimodal, no decoy), then linear α-blend at low α will preserve
  the UVFA goal peak and attenuate the LEO decoy peak proportionally. The
  FFT, being linear, preserves that ratio.
- The predicted best α is somewhere in 0.2–0.4 — low enough that the UVFA
  field dominates the mix, high enough that the LEO field's broad coverage
  still contributes.
- **Plausible pass probability:** moderate-to-high (60–75%). The root-cause
  analysis predicted this exact pipeline change as the fix; if it still
  fails, the synthetic landscape itself may be too easy (the UVFA mock is
  *so* clean that `UvfaOnly` already achieves 0% stuck in Plan 459's run —
  any α < 1.0 should help).

**If it fails despite the mechanism prediction:** that's a meaningful
negative result. It would mean the post-max nonlinear pipeline (FFT
low-pass + finite-difference gradient + unit-length normalization) is itself
enough to wash out the linear α-mix. That would be strong evidence that
flow-field navigation is the wrong target for LEO fusion — the right target
is test-time (QGF `DualLeoOracle`) or per-action policy mixing, not
potential-field mixing. Document and stop.

## Honest caveats up front

1. **Bench heads are still synthetic.** Same mock LEO/UVFA pair as Plan 459.
   A G5 pass demonstrates the *mechanism* works on a designed-adversarial
   landscape; it does NOT prove the gain holds for real trained networks.
   Real-network evidence (riir-games-civ wiring) remains a separate follow-up.
2. **No Lean proof.** This is a perf/quality gate, not a correctness
   invariant. The Lean 4 instances in `.proofs/` are unaffected — `blend_into`
   is a pure arithmetic op, not in any existing spec.
3. **`dual_leo` is already default-on.** This plan does not change default
   features; it adds a *sibling consumer* of an existing default-on primitive.
4. **Two failed gates is the stop rule.** If post-max also fails G5, the
   verdict is "flow-field navigation is not a quality-fusion target for LEO"
   — not "try a third pipeline stage". The QGF `DualLeoOracle` (test-time,
   different axis) remains open as a sibling plan if a LEO-fusion quality
   gain is still desired.
5. **This is a Plan, not an Issue, by explicit user instruction.** Per the
   global rule, optimization/proof tasks normally go to `.issues/`. The user
   explicitly requested "Plan 460" — if the user prefers, this can be
   converted to an `.issues/` file before implementation starts. The plan
   number (460) is consumed either way per the monotonic-numbering rule.

## Follow-up (out of scope)

- **Real-network evidence** — wire `get_or_compute_dual_postmax` into
  `riir-games-civ` (CivLeoNet + a UVFA wrapper). Separate plan in riir-ai.
  Pre-condition: G5 passes here (otherwise there's nothing to validate).
- **QGF `DualLeoOracle`** — test-time fusion (different axis from
  potential-field fusion). Sibling to `LeoHeadOracle` (Plan 268). Separate
  plan in katgpt-rs. Open regardless of this plan's G5 outcome — QGF fusion
  is at a different pipeline stage and might pass G5 even if potential-field
  fusion cannot.
- **Pre-allocate `q_uvfa` + `goal_q` in `FlowFieldCache` scratch space**
  (carried forward from Plan 459 follow-up — currently allocates per
  cache-miss; not a hot path so acceptable for v1).

## Connection to existing GOAT-proved work

| Plan / Issue | Status | Connection |
|---|---|---|
| Plan 155 (LEO All-Goals) | ✅ DEFAULT-ON SUPER GOAT | Source of `LeoHead` + `DualLeoMixer` traits. Plan 460 adds a 3rd consumer of `DualLeoMixer` (was 2: QuestLeoScorer + Plan 459 `get_or_compute_dual`; now 3). |
| Plan 242 (Fourier Flow Fields) | ✅ DEFAULT-ON | Source of `FlowFieldCache::get_or_compute`. Plan 460 adds the post-max dual sibling `get_or_compute_dual_postmax` + a `LeoPotentialGrid::blend_into` primitive. |
| Plan 268 (QGF) | ✅ (opt-in) | `LeoHeadOracle` consumes `LeoHead`. A future `DualLeoOracle` would be a sibling — out of scope here (see Follow-up). |
| Plan 459 (pre-max dual fusion) | ✅ DONE — honest demotion | Plan 459's pre-max `get_or_compute_dual` stays landed. Plan 460 is the post-max sibling attacking the same root cause from a different pipeline stage. Side-by-side bench comparison is mandatory in T6. |
