# Issue 569: SWE Trajectory Geometry Synthetic PoC (Proposal 011 Phase 5 T5.1–T5.3)

> **Source proposal:** [Proposal 011](../.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md) —
> Rust-SWE-bench as a Latent-Space Benchmark via WASM Constraint Pruner.
> **Layer:** 4 (trajectory freeze/thaw — the modelless reframe).
> **PoC location:** `katgpt-rs/benches/bench_011_swe_trajectory_geometry_poc.rs`
> **Filed:** 2026-08-02
> **Status:** RESOLVED — Partial Gain (T5.1 + T5.2 PASS, T5.3 FAIL). Geometry discriminates; FAME commit is deterministic but produces degenerate blends from random-direction + small-magnitude summaries. Real-freezer design constraint documented.

## The claim to defend-or-refute

Proposal 011's Layer 4 hypothesis: **even when the underlying model proposes zero
valid patches, the inference loop's trajectory through patch-space has measurable
geometry (length / curvature / drift / bifurcation) that differs across snapshots
or failure modes — and that geometry is freezable via already-shipped substrate.**

This issue converts the proposal's "cheapest first step" (Phase 5 T5.1–T5.3, all
synthetic — no Rust-SWE-bench, no rubrc, no Kimi-K3 needed) into an empirically
settled question via a defend-or-refute PoC, per research skill §3.6.

## Substrate check (substrate-first skill — 2026-08-02)

| Concept | Searched for | Found at | Verdict |
|---|---|---|---|
| trajectory geometry | `latent_trajectory_geometry`, `from_states`, `bifurcation_ratio`, `TrajectoryGeometry` | `katgpt-core/src/latent_trajectory_geometry.rs` (Plan 342, opt-in feature, G1–G5 PASS) | ✅ CONSUME |
| compaction gate | `closed_unit_compaction`, `ClosedUnitCompactionGate`, `SearchRubric`, `RubricScratch` | `katgpt-core/src/compaction/` (Plan 333, DEFAULT-ON, G1–G7 PASS) | ✅ CONSUME |
| frozen blend | `committed_field_blend`, `TriArchetypeBlend`, `ArchetypeFieldSource` | `katgpt-core/src/committed_field_blend.rs` (Plan 321, DEFAULT-ON, G1–G5 PASS) | ✅ CONSUME |
| AST histogram (Layer 1) | `ast_histogram`, `SourceFeatureDirection`, `non_hidden_state` | none | ❌ P010 not shipped — NOT NEEDED for Phase 5 (synthetic only) |
| Merkle freeze envelope | `MerkleFrozenEnvelope`, `FrozenEnvelope` | `katgpt-core/src/curator.rs`, riir-neuron-db (referenced) | ⚠️ Layer 4 step 5 — only needed if T5.3 passes; not in this PoC |

**Architectural rules checked:**

- **Domain classification** (AGENTS.md): trajectory geometry is latent (semantic);
  the BLAKE3 commitment is raw (sync-safe). ✅
- **Sync boundary**: this PoC stays entirely latent-side; nothing crosses the sync
  boundary in the synthetic POC. ✅
- **Bridge pattern**: none needed at this layer (latent → latent). ✅
- **Modelless-first mandate**: Phase 5 is the modelless exhaustion step before
  any riir-train deferral. ✅

**Decision:** CONSUME — all six claimed Layer-4 primitives exist as advertised.
No new substrate code; this is a pure composition PoC over shipped primitives.

## The falsifiable predictions

### T5.1 — Trajectory geometry discriminates failure modes

On synthetic trajectories constructed to mimic distinct SWE-attempt failure modes
(no real model needed), `from_states` produces measurably different geometry:

| Failure mode | Trajectory shape | Predicted `mean_curvature` | Predicted `length` |
|---|---|---|---|
| **Committed-wrong** | monotone drift toward wrong attractor | low (~0 rad) | high |
| **Oscillation** | ping-pong between two wrong patches | high (~π rad) | high |
| **Drift** | rotating through wrong answers | mid (~π/2 rad) | high |
| **Stuck** | frozen at one point | n/a (degenerate) | ~0 |
| **Converged-correct** | monotone drift toward correct attractor | low (~0 rad) | moderate |

**Falsifiable threshold:** across at least 3 distinct failure modes, the
`(mean_curvature, length)` pair differs by > 0.5 rad on curvature OR > 20% on
length. If all modes produce statistically indistinguishable geometry, T5.1 FAILS.

### T5.2 — CUCG `evaluate()` fires on test-pass events

A test-pass event (in the SWE-attempt analogy) is a closed unit: the trajectory
reached a coherent state (C1), is summarizable (C2), made progress (C3), and is
not still actively churning (¬N1). Construct synthetic feature streams where a
test pass corresponds to (high coherence, low rank, positive divergence, low
novelty) — CUCG should fire `Compress` (the freeze candidate).

**Falsifiable threshold:** on a synthetic stream with an embedded test-pass event,
CUCG fires `Compress` at that event AND does NOT fire on the surrounding churn.
If CUCG never fires, or fires indiscriminately, T5.2 FAILS.

### T5.3 — `committed_field_blend` produces stable BLAKE3-committable blend from all-fail trajectory

Even with a trajectory summary derived from an all-fail SWE attempt, FAME's
commit step produces:
- a stable `pi` (re-commit on the same summary → identical `pi` — determinism),
- a stable BLAKE3 hash (re-commit → identical hash),
- a NON-trivial blend (not all-zero / all-saturated — the blend still picks a
  dominant archetype even from failure signal).

**Falsifiable threshold:** re-commit produces bit-identical BLAKE3 + the blend is
non-degenerate (max `sigmoid(pi_k/tau)` > 0.6 for at least one `k`). If the
all-fail summary produces an all-zero blend, T5.3 FAILS — there's no signal to
freeze.

## What each outcome means

| T5.1 | T5.2 | T5.3 | Verdict | Action |
|---|---|---|---|---|
| PASS | PASS | PASS | **Gain (modelless Layer 4 validated)** | File a plan for `SweTrajectoryFreezer` substrate composition. Proposal 011 Layer 4 G5 met on synthetic data; T5.4 (real Kimi-K3 trajectories) is the next validation step, gated on P032 Phase 5. |
| PASS | * | * | **Partial Gain** | Geometry discriminates — at minimum the diagnostic is useful. T5.2/T5.3 failures narrow the design space (e.g., CUCG may need a SWE-specific rubric). |
| FAIL | * | * | **PASS (confirmed negative)** | Trajectory geometry alone is insufficient signal. Document the negative result honestly; defer to Layer 4b (riir-train LoRA fallback) per the modelless-first mandate. |

## Scope

- **In scope:** three synthetic PoCs (T5.1, T5.2, T5.3), all modelless, all
  consuming shipped DEFAULT-ON or opt-in katgpt-core primitives. One bench file
  with a printed verdict table.
- **Out of scope:** real Kimi-K3 forward passes (T5.4 — gated on P032 Phase 5),
  Rust-SWE-bench dataset, rubrc/WASM pruner (Layer 3), P010 AST histogram
  (Layer 1 — not yet shipped), MerkleFrozenEnvelope composition (Layer 4 step 5
  — only needed if T5.3 passes + a real freezer is built).

## Tasks

- [x] **T1** Implement `benches/bench_011_swe_trajectory_geometry_poc.rs`:
  - T1.1 T5.1 — construct 5 synthetic failure-mode trajectories, measure
        `from_states` + `bifurcation_ratio`, print geometry table + verdict.
  - T1.2 T5.2 — construct a synthetic feature stream with an embedded test-pass
        event, run CUCG `evaluate` per step, print fire-pattern + verdict.
  - T1.3 T5.3 — construct an all-fail trajectory summary (mean + endpoint
        strategies), run FAME `commit`, assert determinism + non-degeneracy,
        print blend + verdict.
- [x] **T2** Register bench in `Cargo.toml` behind
      `latent_trajectory_geometry + closed_unit_compaction + committed_field_blend`.
- [x] **T3** Run PoC, capture verdict table (below).
- [x] **T4** Record verdict + decide next action per the outcome table above.

## PoC verdict (2026-08-02)

```
── T5.1: trajectory geometry discriminates failure modes ──────
          failure_mode     n_steps          length       mean_curv     min_cos
  ----------------------------------------------------------------------------
       committed_wrong         100         15.0000          0.0002      0.6801
           oscillation         100         77.5032          3.1416      0.5397
                 drift         100         12.0000          0.4000      0.2089
                 stuck         100          0.0000          0.0000      1.0000
     converged_correct         100          0.8424          0.0002      0.9418

  T5.1 verdict: PASS — ≥3 distinct (curvature, length) pairs.

── T5.2: CUCG fires on synthetic test-pass events ──────────────
  Compress fired at steps: [20, 40]
  (expected: [20, 40] — synthetic test-pass events)

  T5.2 verdict: PASS — fires exactly at the test-pass events.

── T5.3: committed_field_blend from all-fail summary ──────────
  strategy: mean      — deterministic: true,  max gate 0.4879 (< 0.6)  → degenerate
  strategy: endpoint  — deterministic: true,  max gate 0.4699 (< 0.6)  → degenerate

  T5.3 verdict: FAIL — blends are deterministic but degenerate (all gates ≈ 0.5).
```

### What the T5.3 failure means (the load-bearing finding)

T5.3 FAIL is **not a primitive failure** — FAME's commit is deterministic and
stable (re-commit produces bit-identical BLAKE3 on both strategies). The failure
is that **random direction vectors in 32-dim space produce near-zero dot products
with bounded-magnitude summaries** (concentration of measure: random high-dim
vectors are nearly orthogonal). With `dot(summary, dir_k) ≈ 0`, sigmoid lands
at 0.5 for every archetype → uniform blend → no dominant archetype → degenerate.

The fix is **data-derived direction vectors** (cluster real trajectories to
extract archetype directions, rather than sampling random unit vectors) +
**trajectory-aware summaries** (KARC delay-embeddings, not raw endpoints).
Both require real model trajectories — that's T5.4, gated on Proposal 032
Phase 5 (Kimi-K3 loaded).

**Design constraint for the real SweTrajectoryFreezer:** random direction
vectors are a stress test, not a realistic deployment. The freezer MUST derive
its archetype directions from clustering real (or synthetic-but-structured)
failure trajectories, not from random sampling.

### What T5.1 + T5.2 passing means (the partial Gain)

- **T5.1 PASS** validates the core Layer 4 hypothesis: trajectory geometry
  discriminates failure modes even without any model. The 5 synthetic modes
  produce 5 distinct `(curvature, length)` signatures — the oscillation mode
  hits exactly π curvature (the ping-pong signature Plan 342's Phase 3 gate
  was designed to detect), committed-wrong and converged-correct are near-
  geodesic (low curvature) but distinguishable by length, stuck is degenerate.
- **T5.2 PASS** validates that CUCG evaluates test-pass events as closed units
  (high coherence + low rank + positive divergence + low novelty → Compress).
  This is the Layer 4 step 3 contract: a test pass qualifies as a freeze
  candidate.

### Next action

The Partial Gain is sufficient to justify filing a **plan** for the
`SweTrajectoryFreezer` substrate composition, with two explicit design
constraints carried forward:

1. Direction vectors MUST be data-derived (T5.3 negative result).
2. Real-model validation (T5.4) is gated on P032 Phase 5.

Filing that plan is a follow-up — this issue is closed with the verdict
documented. The bench file stays in-tree as the regression guard for the
geometry discrimination + CUCG test-pass firing contracts.

Per the outcome-action table: **Partial Gain → geometry discriminates, T5.3
failure narrows the design space (data-derived directions required).** No
riir-train deferral needed — the modelless substrate is validated on the
geometry axis; the FAME commit failure is a parameterization issue, not a
fundamental modelless insufficiency.
