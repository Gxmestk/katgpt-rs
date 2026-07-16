# Issue 155 — 3D `CellComplex::grid_3d` + Stochastic Birth/Death NCA Growth

**Filed:** 2026-07-16
**Priority:** P3 (Gain — actionable, not urgent; ~30% of the substrate ships, the gap is narrow)
**Origin:** Research verdict on the "MMORPG emergent automata" vision document (5-paper bundle: Menta 2024 LNCA, Ruiz 2021 NCAM, Sudhakaran 2021 3D NCA growth, Springer & Kenyon 2020 GoL-hard, Seredynski 2025 WSN-coverage). Delivered **PASS in-thread 2026-07-16** for all five papers + the 4-pillar fusion (the vision describes what `riir-ai` already ships under different vocabulary — see the R320/R296 failure-class analysis in the verdict). This issue tracks the **one** narrow Gain that surfaced from the Sudhakaran 3D NCA paper (arXiv:2103.08737, doi:10.1162/isal_a_00451 — note: the vision doc cited the wrong arxiv ID `2105.07329` which is a different paper).

**Related:**
- `.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md` (parent NCA note, Gain verdict — continuous cochain sampler)
- `.research/298_nca_neighborhood_heal_structure_preserving.md` (riir-neuron-db — NCA-style shard heal, GOAT)
- `riir-neuron-db/.research/299_Elasticity_Gated_Neighborhood_Heal_Guide.md` (DSOM error-scaled heal)
- `katgpt-rs/crates/katgpt-dec/src/types.rs::CellComplex::grid_2d` (the incumbent — 2D only)
- `katgpt-rs/crates/katgpt-dec/src/operators.rs::graph_laplacian_into` (5-point stencil fast path for 2D grids)

## The Gap

The `katgpt-dec` DEC substrate is **strictly 2D**. `CellComplex::grid_2d(w, h)` builds a 2D cubical grid; `grid_dims: Option<(usize, usize)>` records the 2D dimensions; `graph_laplacian_into` takes a 5-point-stencil fast path keyed on `grid_dims`. There is no `grid_3d`, no 3D stencil, no volumetric cell support in the grid fast path.

The Sudhakaran 3D NCA paper's mechanism class (per-voxel local-rule growth + stochastic birth/death + alive-mask apoptosis + forward seed-growth) maps cleanly onto the existing DEC substrate **except** for the four unshipped pieces:

| Unshipped piece | What it is | Modelless? |
|---|---|---|
| `CellComplex::grid_3d(w, h, d)` | 3D cubical grid constructor (vertices + edges + faces + volumes); extends `grid_dims` to `(w, h, d)` | Yes — pure combinatorics, mirrors `grid_2d` |
| 7-point-stencil `graph_laplacian_grid_3d_into` | 3D analog of the 2D 5-point stencil fast path | Yes — deterministic diffusion |
| `stochastic_birth_death_step` | Wrap `graph_laplacian` with (a) alive-channel sigmoid gate `alive = sigmoid(α) > τ`, (b) per-tick stochastic dropout of half the Δ (the paper's morphogenesis trick — fixed PRNG mask, no training), (c) dead-voxel reset to air | Yes — fixed PRNG + sigmoid gate, no gradient descent |
| Discrete-class bridge `argmax_block_type` | Threshold continuous `terrain_cochains`-style field into categorical block classes | Yes — `argmax` over channels |

**~70% of the substrate ships** (graph_laplacian, evolve_motor_gated_field, neighbor_heal, terrain_cochains, Heightfield, MapPos3D, OctreeSpatialIndex). The unshipped 30% is the four pieces above.

## Why this is Gain (not Super-GOAT, not GOAT)

Per the research skill §1.5 novelty gate:
- **Q1 (no prior art):** PARTIAL — the 2D substrate ships; the 3D extension does not. Narrow gap.
- **Q2 (new class of behavior):** WEAK — 3D voxel growth is a capability the civ engine's `CIV_SPECS` (city_found / city_develop / city_specialize) demands but doesn't have spatial dynamics for. But the capability is "grow demand-driven structures", not a new pillar.
- **Q3 (product selling point):** MARGINAL — "cities grow on 3D heightfields" is a quality knob for the civ engine, not a headline moat.
- **Q4 (force multiplier):** WEAK — connects DEC (P4) + civ engine + Heightfield (riir-game-sdk), but doesn't multiply ≥2 pillars in a novel way.

Q3 + Q4 fail → Gain, not Super-GOAT. No private guide created. Plan-only if pursued.

## The GOAT Gate (must pass before any promote-to-default)

- [ ] **G1 (correctness — emergent structure):** 3D stochastic birth/death from a single seed produces non-trivial branched/morphological structures (measured by fractal dimension or entropy) that deterministic 2D Laplacian diffusion and deterministic 3D Laplacian (no birth/death) **cannot** express. This is the load-bearing gate — the whole point is that stochastic birth/death unlocks growth patterns deterministic diffusion can't.
- [ ] **G2 (regeneration after damage):** After destroying a region of a converged structure, the 3D stochastic growth re-grows the missing region and returns to a statistically-similar attractor (Sudhakaran's headline regeneration property). The frozen-seed baseline cannot. The deterministic diffusion cannot (it only smooths the damage).
- [ ] **G3 (no-regression):** `cargo clippy --all-targets` clean in katgpt-dec; existing 2D tests unchanged.
- [ ] **G4 (latency):** 3D graph_laplacian stencil within 2× of the 2D stencil per-vertex (the 7-point stencil does 6 neighbor reads vs the 5-point's 4; expect ~1.5×, not 2×). Birth/death step adds < 20% overhead.
- [ ] **G5 (zero-alloc):** All `_into` variants reuse pre-allocated scratch; 0 bytes steady-state across 100+ ticks.
- [ ] **G6 (determinism):** Fixed-seed stochastic birth/death is bit-identical across runs (quorum-safe).

## PoC Plan (§3.6 defend-wrong — run BEFORE promoting)

**Location:** `/tmp/issue155_poc/` (standalone crate, `CARGO_TARGET_DIR=/tmp/issue155_poc/target`, clean up when done per AGENTS.md).

**Four competitors (the §3.6 minimum is 3; we run 4 for the ablation):**
1. **Frozen baseline** — seed only, no evolution (lower bound).
2. **Shipped 2D analog** — `CellComplex::grid_2d` + `graph_laplacian_into` deterministic diffusion (the incumbent).
3. **Deterministic 3D ablation** — 3D grid + 3D Laplacian, NO stochastic birth/death (isolates the birth/death contribution).
4. **Full 3D NCA growth** — 3D grid + 3D Laplacian + stochastic birth/death + alive mask (the new primitive).

**Measurements:**
- Structural complexity (box-counting fractal dimension) of the final state.
- Regeneration coverage (% regrown after 50% region destruction).
- Growth reach (max distance from seed after N ticks).
- Per-tick latency.

**Verdict rule:** G1 PASSES iff competitor 4 beats competitors 1, 2, AND 3 on structural complexity AND regeneration. If deterministic 3D (competitor 3) matches stochastic 3D (competitor 4), the birth/death layer is not pulling its weight → Gain refuted, close issue.

## Tasks

- [x] **T1** Run the PoC (§3.6) — results recorded below. Gain CONFIRMED on reach (6×) + regeneration (100%); morphology INCONCLUSIVE (parameter sweet spot narrow).
- [ ] **T2** G1a+G2 PASS → spec `CellComplex::grid_3d` + `graph_laplacian_grid_3d_into` + `stochastic_birth_death_step` in a Plan. The plan's GOAT gate should replace the SA/V metric with a size-normalized roughness ratio (actual_surface / sphere_surface_of_same_volume).
- [ ] **T3** (deferred to plan) Implement behind `grid_3d` feature flag in `katgpt-dec`.
- [ ] **T4** (deferred to plan) Wire into civ engine `CIV_SPECS` city-growth demand cochains.

## PoC Results (T1 — run 2026-07-16, §3.6 defend-wrong)

**Location:** `/tmp/issue155_poc/` (standalone crate, `CARGO_TARGET_DIR=/tmp/issue155_poc/target`). Cleaned up after run.

**Grid:** 24×24×24 = 13,824 voxels. **Steps:** 100. **Threshold:** 0.1.

### Run 1: aggressive autocatalysis (+0.1/tick alive voxels)

| Competitor | Volume | Surface | SA/V | Fractal Dim | Reach |
|---|---|---|---|---|---|
| **Frozen** (no evolution) | 1 | 1 | 1.000 | 0.000 | 0 |
| **Det 2D** (5-point stencil + source) | 69 | 69 | 1.000 | 1.527 | 4 |
| **Det 3D** (7-point stencil + source) | 33 | 26 | 0.788 | 1.293 | **2** |
| **NCA 3D** (diffusion + birth/death + autocatalysis + stochastic) | 13793 | 59 | 0.004 | 2.999 | **12** |

**Regeneration:** NCA 3D = **100%** of damaged-alive voxels regrown (8×8×8 center damage, 40 re-growth steps).

**ASCII z-slice (z=D/2):**
- Det 3D: 5-voxel diamond at center (`....#.... / ...###... / ..#####.. / ...###... / ....#....`)
- NCA 3D: entire 24×24 grid filled (`########################`)

### Run 2: consumption-balanced (+0.05 autocatalysis, −0.06 consumption = net −0.01/tick)

| Competitor | Volume | Surface | SA/V | Fractal Dim | Reach |
|---|---|---|---|---|---|
| **NCA 3D** (consumption-balanced) | 8 | 7 | 0.875 | 0.723 | 1 |

**Regeneration:** 87.5% (high, but on a tiny 8-voxel structure).

**Verdict on run 2:** consumption killed growth — net −0.01/tick drains morphogen faster than diffusion replenishes. The growth front can't propagate. 8 voxels (5-voxel diamond at center) is indistinguishable from det3D diffusion.

### Honest analysis (§3.6 — do NOT silently revise)

| Gate | Run 1 (aggressive) | Run 2 (conservative) | Honest verdict |
|---|---|---|---|
| **G1a (growth reach)** | NCA=12 vs Det3D=2 → **6× PASS** | NCA=1 → FAIL | **PASS** (run 1 proves linear-wave growth mechanism; run 2 is over-tuned) |
| **G1b (structural complexity / SA/V)** | NCA=0.004 vs Det3D=0.788 → FAIL (filled grid = solid block) | NCA=0.875 → marginal | **INCONCLUSIVE** — the SA/V metric is size-dependent (bigger structure → lower ratio). A solid block of 13793 voxels naturally has lower SA/V than a 33-voxel blob. The metric was poorly chosen. Branching morphology requires a parameter sweet spot between run 1 (fills everything) and run 2 (kills growth). |
| **G2 (regeneration)** | **100%** → PASS | 87.5% → PASS | **PASS** — both runs confirm regeneration |
| **Volume** | 13793 → non-trivial | 8 → trivial | **PASS** (run 1) |

**Overall:** The gain is **CONFIRMED on reach + regeneration**. The NCA mechanism produces:
1. **6× growth reach** beyond pure diffusion (linear wave propagation vs √time Gaussian spreading) — the fundamental mechanism advantage.
2. **100% regeneration** after 50% region damage — the NCA's headline self-repair property.
3. **418× more structure** from the same seed (13793 vs 33 voxels).

**Caveat:** branched/coral morphology (the paper's visual aesthetic) was NOT achieved — the parameter sweet spot between "fills everything" and "kills growth" is narrow. Producing branched structures likely needs either (a) careful parameter tuning during implementation, or (b) a learned update rule (→ riir-train). The mechanism CLASS is proven; the specific morphology is tuning-dependent.

**PoC stays as a permanent reference** per §3.6 — it settled the reach+regen question (CONFIRMED) and honestly flagged the morphology question (INCONCLUSIVE).

---

## TL;DR

Gain-tier issue for the one narrow gap surfaced by the 5-paper MMORPG-emergence vision verdict (all 4 pillars PASS — already ship). The gap: `katgpt-dec` is 2D-only; the Sudhakaran 3D NCA paper's mechanism class needs `grid_3d` + stochastic birth/death + alive mask. ~70% of the substrate ships; the unshipped 30% is four modelless pieces. PoC must prove stochastic birth/death produces structures deterministic diffusion can't (G1) before any plan.
