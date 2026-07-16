# Plan 454 — 3D NCA GOAT Gate (G1a/G1b/G2/G3/G4/G5/G6) Results

> **Issue 155 close-out (2026-07-16):** Implementation complete via Plan 454 T1–T6 (committed). GOAT gate ran — G1a/G2/G3/G4a/G5/G6 PASS, G1b (morphology) + G4b (latency) FAIL → `grid_3d` stays opt-in. T4 (civ engine wiring) deferred to riir-ai, gated on T8 promotion. Issue file removed; this benchmark is the close-out record.

**Date:** 2026-07-16
**Plan:** [katgpt-rs/.plans/454](../.plans/454_3d_cellcomplex_grid_stochastic_birth_death.md)
**Issue:** [katgpt-rs/.issues/155](../.issues/155_3d_cellcomplex_grid_stochastic_birth_death.md)
**Bench:** `crates/katgpt-dec/benches/bench_454_3d_nca_goat.rs`
**Status:** ❌ **GAIN FAILS (G1b) — `grid_3d` stays opt-in; reach + regeneration gains hold, branched morphology needs a learned update rule**

---

## Gate-by-gate results

| Gate | Criterion | Target | Measured | Verdict |
|------|-----------|--------|----------|---------|
| **G1a** | Growth reach: competitor 4 reach ≥ 3× competitor 3 reach | ≥ 3× | 12 vs 2 = **6.0×** | ✅ PASS |
| **G1b** | Structural complexity: size-normalized roughness ratio ≥ 1.5× | ≥ 1.5× | 1.301 vs 1.568 = **0.83×** (best across 420-combo sweep) | ❌ FAIL |
| **G2** | Regeneration: ≥ 80% of destroyed-alive voxels regrown after 40 steps | ≥ 80% | **100.0%** | ✅ PASS |
| **G3** | No-regression: clippy clean + existing 2D tests pass | clean | Clean (verified separately) | ✅ PASS |
| **G4a** | Latency: 3D stencil per-vertex ≤ 2× 2D stencil | ≤ 2× | **1.73×** | ✅ PASS |
| **G4b** | Latency: birth/death overhead < 20% on top of Laplacian | < 20% | **123.7%** | ❌ FAIL |
| **G5** | Zero-alloc: 0 allocations in steady state (100+ ticks) | 0 | **0 allocs** | ✅ PASS |
| **G6** | Determinism: bit-identical across 10 runs (same seed) | bit-exact | **bit-identical** | ✅ PASS |

**GAIN gates (G1a + G1b + G2 + G6):** ❌ FAIL (G1b blocks)
**Engineering gates (G3 + G4 + G5):** ❌ FAIL (G4b blocks)
**Verdict:** Do NOT promote `grid_3d` to default. Keep opt-in.

---

## Ablation table (24³ grid, 100 steps)

| Competitor | Volume | Surface | Roughness | Reach |
|---|---|---|---|---|
| Frozen (seed only) | 1 | — | — | 0 |
| Det 3D (diffusion + source) | 33 | 78 | 1.568 | 2 |
| NCA 3D (full birth/death) | 13824 | 3456 | 1.241 | 12 |

The NCA 3D fills the entire 24³ grid (13824 voxels = all voxels) → solid block →
roughness 1.241 (the cube/sphere surface ratio). Det 3D produces a small 33-voxel
blob with naturally-higher roughness (1.568) for its size. The NCA's solid block
is smoother (lower roughness) than the det 3D blob — the opposite of what G1b
requires for branched morphology.

---

## G1b analysis — why branched morphology fails

### The sweep

The G1b gate sweeps 420 parameter combinations:
- `birth_rate ∈ {0.05, 0.10, 0.15, 0.20}` (4 values)
- `consumption_rate ∈ {0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30}` (7 values)
- `dropout_prob ∈ {0.0, 0.3, 0.5}` (3 values)
- `alive_threshold ∈ {0.5, 0.6, 0.7, 0.8, 0.9}` (5 values — added beyond the
  plan's original 3-parameter spec because threshold directly controls growth
  selectivity)

**Best ratio found: 0.83×** (NCA roughness 1.301 vs det3D roughness 1.568).
No parameter regime produces a branched structure.

### Root cause

The modelless birth/death update rule has a binary growth character:
- **Low consumption / low threshold** → growth fills the entire grid (solid block,
  roughness ≈ 1.24, the cube/sphere ratio). Any positive morphogen crosses the
  gate, so once diffusion reaches a voxel it becomes alive.
- **High consumption / high threshold** → growth dies entirely (volume < 10,
  filtered out by the sweep's trivially-small guard).

There is no intermediate regime that produces branching. Branched/coral morphology
(as in the Sudhakaran paper's visual results) requires a **competition mechanism**
where voxels fight for morphogen and only some win. The paper achieves this with a
**learned update rule** (the NCA has trained weights that create selective growth).

Our modelless update rule (diffusion + autocatalysis + sigmoid gate) cannot express
this competition — it's a simple threshold system, not a learned growth policy.

### Honest verdict

Per the plan's T8 fallback rule:
> If G1b cannot be made to pass (no parameter regime produces branched structures)
> → keep grid_3d opt-in, note that morphology needs a learned update rule
> (→ riir-train Issue 004 follow-up). The reach + regeneration gain still holds;
> only the branched-morphology claim is refuted.

**The mechanism CLASS is proven** (6× growth reach, 100% regeneration). **The
branched-morphology claim is refuted** under the modelless constraint — it needs
a learned update rule (riir-train follow-up).

---

## G4b analysis — birth/death overhead

The G4b gate measures `(t_bd - t_lap) / t_lap` where `t_bd` is the full
`stochastic_birth_death_step` (Laplacian + 5 update passes) and `t_lap` is the
bare `graph_laplacian_into`.

**Measured: 123.7% overhead** — the 5 non-Laplacian passes (dropout mask,
diffusion apply, reaction, alive gate, dead decay) take about as long as the
Laplacian itself.

This is expected: the birth/death step does 6 full-grid passes total (1 Laplacian
+ 5 updates), each touching all `n_vertices × dim` elements. The 20% gate was
written before the T4 implementation revealed the 5-step update structure; 20%
is unrealistic for 5 additional full-grid passes.

**Optimization opportunity (not urgent):** the 4 post-Laplacian passes (diffusion
apply, reaction, alive gate, dead decay) have data dependencies but could be
fused into a single pass per voxel, reducing the overhead to ~2 passes total
(Laplacian + fused update). This would bring the overhead to ~50% — still above
20% but a significant improvement. Deferred until the morphology question is
resolved (no point optimizing a primitive that doesn't produce the target behavior).

---

## What passes

- **G1a (growth reach):** NCA grows 6× farther than pure diffusion (12 vs 2
  Chebyshev distance). The linear-wave growth mechanism is confirmed.
- **G2 (regeneration):** 100% regrowth after 8×8×8 center destruction. The NCA's
  headline self-repair property is confirmed.
- **G4a (stencil ratio):** The 3D 7-point stencil is 1.73× slower per-vertex than
  the 2D 5-point stencil (6 neighbor reads vs 4). Well within the 2× gate.
- **G5 (zero-alloc):** 0 allocations across 100 ticks. The scratch-buffer design
  works as intended.
- **G6 (determinism):** Bit-identical across 10 runs with the same seed. The
  quorum-safety contract holds.

---

## Parameters

| Parameter | Value |
|---|---|
| Grid | 24×24×24 = 13,824 voxels |
| Steps | 100 |
| Dim | 2 (alive + morphogen) |
| Seed position | (12, 12, 12) — center |
| PRNG seed | 7 (G1a/G2/G5), 99 (G6), 42 (G4) |
| Competitor 3 threshold | 0.1 (morphogen > 0.1 → alive) |
| Competitor 3 diffusion_dt | 0.1 |
| Competitor 4 params | `BirthDeathParams::paper_defaults()` |

---

## TL;DR

Plan 454 T7 GOAT gate: **G1a ✅, G1b ❌, G2 ✅, G3 ✅, G4a ✅, G4b ❌, G5 ✅, G6 ✅**.
G1b fails because the modelless update rule cannot produce branched morphology —
it either fills the grid solid or kills growth. The reach (6×) and regeneration
(100%) gains are confirmed; the branched-morphology claim needs a learned update
rule (riir-train follow-up). `grid_3d` stays opt-in per the plan's T8 fallback.
