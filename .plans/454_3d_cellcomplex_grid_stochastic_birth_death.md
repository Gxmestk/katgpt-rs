# Plan 454: 3D `CellComplex::grid_3d` + 7-Point Stencil + Stochastic Birth/Death

> **Issue:** 155 (`katgpt-rs/.issues/155_3d_cellcomplex_grid_stochastic_birth_death.md`)
> **Origin paper:** Sudhakaran et al., "3D NCA Growth", ALIFE 2021 (arXiv:2103.08737, doi:10.1162/isal_a_00451)
> **Origin verdict:** Research PASS-in-thread 2026-07-16 on the 5-paper MMORPG-emergence vision (all 4 pillars already ship under different vocabulary). This plan tracks the **one** Gain that surfaced — `katgpt-dec` is strictly 2D; the 3D extension + stochastic birth/death is the unshipped 30%.
> **PoC:** Run 2026-07-16 (Issue 155 T1). Gain CONFIRMED on growth reach (6×) and regeneration (100%); morphology INCONCLUSIVE due to a size-dependent SA/V metric — this plan replaces SA/V with a size-normalized roughness ratio (§GOAT G1b).
> **Feature gate:** `grid_3d` (default-OFF until GOAT G1–G6 pass)
> **Priority:** P3 — Gain, actionable, not urgent. ~70% of the substrate ships; the gap is narrow.
> **Date:** 2026-07-16

---

## Summary

Close the one narrow gap surfaced by the 5-paper verdict. The `katgpt-dec` substrate currently ships `CellComplex::grid_2d` + a 5-point-stencil `graph_laplacian_into` fast path. The Sudhakaran 3D NCA paper's mechanism class (per-voxel growth + stochastic birth/death + alive-mask apoptosis) maps cleanly onto the existing DEC surface **except** for four modelless pieces. This plan ships all four behind a `grid_3d` feature flag:

1. `CellComplex::grid_3d(w, h, d)` — 3D cubical grid constructor (vertices + 3 edge orientations + 3 face orientations + volumes). Mirrors `grid_2d` exactly; the rank-4 substrate (`MAX_RANK = 3`) already exists.
2. `graph_laplacian_grid_3d_into` — 7-point-stencil fast path keyed on the 3D grid dims. Mirrors the 5-point path.
3. `stochastic_birth_death_step` — wraps `graph_laplacian` with (a) an alive-channel sigmoid gate, (b) per-tick fixed-PRNG stochastic dropout of half the Δ (the paper's morphogenesis trick — no training), (c) dead-voxel reset to air. Zero-alloc via pre-allocated scratch + a modelless PRNG (SplitMix64 seeded once).
4. `argmax_block_type` — discrete-class bridge: threshold the continuous field into categorical block classes (the alive-mask → block-class step the civ engine's `CIV_SPECS` consumes).

All four are modelless (no gradient descent, no learned weights — just a fixed PRNG mask and a sigmoid gate). The growth mechanism is a deterministic function of the seed and the parameters, which keeps G6 (determinism / quorum-safety) tractable.

---

## Design: `grid_dims` extension

The existing `grid_dims: Option<(usize, usize)>` field records 2D grid dims and gates the 5-point-stencil fast path. 3D needs a third dimension. The clean extension (minimal churn, no 2D call-site changes):

```rust
// types.rs — NEW enum, replaces Option<(usize, usize)>
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridDims {
    /// 2D regular grid (produced by `grid_2d`).
    Dim2 { w: usize, h: usize },
    /// 3D regular grid (produced by `grid_3d`).
    Dim3 { w: usize, h: usize, d: usize },
}

// CellComplex field change:
//   grid_dims: Option<(usize, usize)>
// becomes:
//   grid_dims: Option<GridDims>
```

**Back-compat accessor strategy** (the "changes minimal and focused" rule):
- `grid_dims()` keeps its existing signature `Option<(usize, usize)>` — returns `Some((w, h))` for `Dim2`, `None` for `Dim3` or non-grid. **Zero 2D call-site changes.**
- NEW `grid_dims_3d() -> Option<(usize, usize, usize)>` — returns `Some((w, h, d))` for `Dim3`, `None` otherwise.
- NEW `grid_dims_full() -> Option<GridDims>` — the full discriminated accessor for new code that needs to dispatch on dimensionality.

The `graph_laplacian_into` dispatch becomes:

```rust
pub fn graph_laplacian_into(cx, potential, output) {
    match cx.grid_dims_full() {
        Some(GridDims::Dim2 { w, h }) => graph_laplacian_grid_into(w, h, potential, output),
        Some(GridDims::Dim3 { w, h, d }) => graph_laplacian_grid_3d_into(w, h, d, potential, output),
        None => graph_laplacian_edge_list_into(cx, potential, output),
    }
}
```

`invalidate_coboundary_cache` (called by every `remove_face` / `remove_cell` mutation) sets `grid_dims = None` — already correct for both `Dim2` and `Dim3` (the `merkle_root` lesson: a grid with a removed cell is no longer regular, so the stencil would be wrong at the gap).

---

## Tasks

### T1: `GridDims` enum + `grid_dims` back-compat accessors
- [ ] Add `pub enum GridDims { Dim2 { w, h }, Dim3 { w, h, d } }` to `types.rs` (derive `Clone, Copy, Debug, PartialEq, Eq`)
- [ ] Change `CellComplex::grid_dims` field type from `Option<(usize, usize)>` to `Option<GridDims>`
- [ ] Update `CellComplex::new` to initialize `grid_dims: None` (one-line change)
- [ ] Update `grid_2d` to set `grid_dims = Some(GridDims::Dim2 { w, h })`
- [ ] Keep `grid_dims() -> Option<(usize, usize)>` signature: returns `Some((w, h))` for `Dim2`, `None` otherwise (back-compat — zero 2D call-site changes)
- [ ] Add `grid_dims_3d() -> Option<(usize, usize, usize)>`: returns `Some((w, h, d))` for `Dim3`, `None` otherwise
- [ ] Add `grid_dims_full() -> Option<GridDims>`: the full discriminated accessor
- [ ] Update `invalidate_coboundary_cache` (still sets `grid_dims = None` — no change needed beyond the field type)
- [ ] Existing 2D tests must pass unchanged (regression guard)

### T2: `CellComplex::grid_3d(w, h, d)` constructor
- [ ] New constructor behind `#[cfg(feature = "grid_3d")]` in `types.rs`
- [ ] **Cell counts** (cubical grid topology):
  - Vertices: `w * h * d`
  - Edges: x-aligned `(w-1)*h*d` + y-aligned `w*(h-1)*d` + z-aligned `w*h*(d-1)`
  - Faces: xy-planes `(w-1)*(h-1)*d` + xz-planes `(w-1)*h*(d-1)` + yz-planes `w*(h-1)*(d-1)`
  - Volumes: `(w-1)*(h-1)*(d-1)`
- [ ] **Vertex indexing**: `vidx(x, y, z) = (z * h + y) * w + x` (row-major, z-slowest — matches `CochainField` flat layout)
- [ ] **Edge indexing** (3 orientations, contiguous ranges):
  - x-edges: `e_x(x, y, z) = (z * h + y) * (w - 1) + x` for `x ∈ [0, w-1)`
  - y-edges: `e_y(x, y, z) = n_x_edges + (z * (h - 1) + y) * w + x` for `y ∈ [0, h-1)`
  - z-edges: `e_z(x, y, z) = n_x_edges + n_y_edges + (z * w + y) * w + x` — **wait, z-edges connect `(x,y,z) ↔ (x,y,z+1)`, so the index is over `z ∈ [0, d-1)` and the stride is `w*h` per z-slice: `e_z(x, y, z) = n_x_edges + n_y_edges + z * (w * h) + y * w + x`**
- [ ] **Face indexing** (3 orientations, contiguous ranges):
  - xy-faces (normal = z): `f_xy(x, y, z) = (z * (h - 1) + y) * (w - 1) + x` for `x ∈ [0, w-1), y ∈ [0, h-1), z ∈ [0, d)`
  - xz-faces (normal = y): offset by `n_xy_faces`, indexed over `x ∈ [0, w-1), y ∈ [0, h), z ∈ [0, d-1)`
  - yz-faces (normal = x): offset by `n_xy_faces + n_xz_faces`, indexed over `x ∈ [0, w), y ∈ [0, h-1), z ∈ [0, d-1)`
- [ ] **Volume indexing**: `vol(x, y, z) = (z * (h - 1) + y) * (w - 1) + x` for `x ∈ [0, w-1), y ∈ [0, h-1), z ∈ [0, d-1)`
- [ ] **B₁ (vertex→edge)**: for each edge, push `(tail, e, -1)` and `(head, e, +1)` — orientation convention: tail = lower-index corner, head = higher-index corner (matches `grid_2d`)
- [ ] **B₂ (edge→face)**: each face is bounded by 4 edges in a consistent orientation; pick the right-handed normal convention (xy-faces normal = +z, etc.). Push 4 entries per face with `±1` signs matching the orientation.
- [ ] **B₃ (face→volume)**: each volume is bounded by 6 faces; push 6 entries per volume with `±1` signs. **This is the new rank-3 boundary matrix — `grid_2d` leaves it empty, `grid_3d` populates it.**
- [ ] Pre-allocate boundary vectors to exact capacity (`reserve_exact`) — mirrors the `grid_2d` no-realloc pattern
- [ ] Set `grid_dims = Some(GridDims::Dim3 { w, h, d })`
- [ ] **Assert** `w >= 2 && h >= 2 && d >= 2` (a 1-cell-thick grid has zero volumes — degenerate; mirror the implicit `grid_2d` contract where `w=1` produces zero faces)
- [ ] Unit tests: cell counts match closed-form for `(3,3,3)` and `(4,3,2)`; B₁ entries per vertex match the 7-point stencil neighbor count (3 at corner, 4 at edge, 5 at face, 6 at interior); `boundary_entries(0)` length = `2 * n_edges`; B₃ populated

### T3: `graph_laplacian_grid_3d_into` — 7-point stencil fast path
- [ ] New private fn in `operators.rs` behind `#[cfg(feature = "grid_3d")]`
- [ ] Mirror `graph_laplacian_grid_into` (the 5-point path) exactly: raw pointer arithmetic, branch-free interior, explicit boundary handling
- [ ] **Interior** (`1 <= x < w-1`, `1 <= y < h-1`, `1 <= z < d-1`): deg = 6, 6 neighbor reads (±x, ±y, ±z), direct write `6*center - Σ neighbors`
- [ ] **Boundary**: 6 face planes (deg 5), 12 edges (deg 4), 8 corners (deg 3) — handle with the same `has_left/has_right/has_up/has_down/has_front/has_back` flag pattern as the 2D boundary path
- [ ] **Stride math**: z-stride = `w * h * dim`, y-stride = `w * dim`, x-stride = `dim`. Same `unsafe` raw-pointer pattern as the 2D path (offsets are only dereferenced when the corresponding `has_*` flag is true).
- [ ] Update `graph_laplacian_into` dispatch to the 3-arm `match` on `grid_dims_full()` (see Design section)
- [ ] `graph_laplacian` (the allocating variant) works unchanged — it delegates to `_into`
- [ ] Unit tests:
  - `Δ(linear) = 0` at interior vertices (the load-bearing DEC identity — must hold exactly)
  - Stencil path matches edge-list path on the same 3D grid within 1 ULP (mirror the 2D `graph_laplacian_grid_matches_edge_list_*` tests)
  - Multi-channel (dim=16) equivalence
  - Symmetry: `Δ` at `(x,y,z)` equals `Δ` at the grid-reflected point when the potential is mirror-symmetric

### T4: `stochastic_birth_death_step` — NCA growth wrapper
- [ ] New module `birth_death.rs` behind `#[cfg(feature = "grid_3d")]`
- [ ] **Signature** (zero-alloc, `_into` convention — mirrors `evolve_motor_gated_field`):
  ```rust
  pub fn stochastic_birth_death_step(
      cx: &CellComplex,
      field: &mut CochainField,          // [alive, morphogen, ...] — channel 0 = alive, channel 1 = morphogen
      params: &BirthDeathParams,
      rng: &mut SplitMix64,               // fixed-seed PRNG — modelless, no training
      scratch_lap: &mut CochainField,     // reused across ticks (G5 zero-alloc)
      scratch_dropout: &mut [u8],         // reusable dropout mask buffer, len = n_vertices
  )
  ```
- [ ] **`BirthDeathParams`** (plain struct, `Copy`):
  - `diffusion_dt: f32` — morphogen diffusion timestep (feeds `graph_laplacian_grid_3d_into`)
  - `alive_threshold: f32` — sigmoid gate threshold τ (alive iff `sigmoid(α) > τ`)
  - `birth_rate: f32` — autocatalytic morphogen gain for alive voxels
  - `consumption_rate: f32` — morphogen drain for alive voxels (the paper's consumption term)
  - `dropout_prob: f32` — per-tick stochastic dropout probability (paper: ~0.5 — kill half the Δ)
  - `decay_rate: f32` — morphogen decay for dead voxels
- [ ] **Algorithm** (per tick, per voxel — branch-free interior where possible):
  1. Diffuse morphogen: `scratch_lap = graph_laplacian_grid_3d(field.morphogen)`; `field.morphogen += diffusion_dt * scratch_lap`
  2. Autocatalysis + consumption on alive voxels: `field.morphogen[v] += birth_rate - consumption_rate` if alive, else `-= decay_rate`
  3. Stochastic dropout: for each voxel, draw `r = rng.next_u32() as f32 / u32::MAX as f32`; if `r < dropout_prob`, halve the Δ applied this tick (the paper's morphogenesis trick — write the half-updated value). Reusable mask buffer.
  4. Alive gate: `field.alive[v] = sigmoid(field.alive[v] * α_scale) > alive_threshold ? 1.0 : 0.0` (sigmoid, NOT softmax per the global rule)
  5. Dead-voxel reset: if `!alive`, `field.morphogen[v] *= decay_rate` (gradual, not instant — matches the paper)
- [ ] **PRNG choice**: `SplitMix64` (a single `u64` state, deterministic, no deps). Seed once at the start of the growth run; advance once per voxel per tick. Bit-identical across runs (G6 determinism).
- [ ] **No allocations**: all scratch buffers passed in by the caller; the function borrows `&mut`. The growth loop pre-allocates once and reuses across all ticks.
- [ ] Unit tests:
  - Determinism: same seed → bit-identical field after N ticks (the quorum-safety gate)
  - Birth propagates: a single seed voxel grows to >1 voxel after N ticks (the core mechanism)
  - Death resets morphogen: a voxel that drops below `alive_threshold` sees its morphogen decay toward 0
  - Zero-alloc: `GlobalAlloc` counter stable across 100 ticks (G5)

### T5: `argmax_block_type` — discrete-class bridge
- [ ] New fn in `birth_death.rs` (or `bridge.rs` if the module grows) behind `#[cfg(feature = "grid_3d")]`
- [ ] **Signature**: `pub fn argmax_block_type(field: &CochainField, n_classes: usize, out: &mut [u8])`
- [ ] For each voxel, `out[v] = argmax over channels of field.data[v*dim + c]` — threshold the continuous field into categorical block classes (air/dirt/stone/water/...). The civ engine's `CIV_SPECS` consumes categorical block types, not continuous morphogen values.
- [ ] Unit tests: known field → known block classes; ties broken by lowest channel index (deterministic)

### T6: Feature flag + Cargo.toml
- [ ] Add `grid_3d = []` to `[features]` in `katgpt-dec/Cargo.toml` (default-OFF — single-line, mirrors `motor_gated_field` / `cochain_point_sampler`)
- [ ] Gate T2/T3/T4/T5 code with `#[cfg(feature = "grid_3d")]`
- [ ] Do NOT add to `default` — promotion requires the GOAT gate below

### T7: GOAT gate — replace SA/V with size-normalized roughness ratio
- [ ] New bench `benches/bench_454_3d_nca_goat.rs` behind `required-features = ["grid_3d"]`
- [ ] **Four competitors** (the §3.6 minimum is 3; run 4 for the ablation, mirroring the Issue 155 PoC):
  1. Frozen baseline (seed only, no evolution — lower bound)
  2. Deterministic 2D diffusion (`grid_2d` + 5-point stencil — the incumbent)
  3. Deterministic 3D diffusion (`grid_3d` + 7-point stencil, NO birth/death — isolates the birth/death contribution)
  4. Full 3D NCA (`grid_3d` + 7-point stencil + `stochastic_birth_death_step`)
- [ ] **Grid**: 24×24×24 = 13,824 voxels (matches the Issue 155 PoC — comparability)
- [ ] **Steps**: 100 (matches the PoC)
- [ ] **Gates** (all must pass to promote to default-on):

  - [ ] **G1a (growth reach):** competitor 4 reach ≥ 3× competitor 3 reach (deterministic 3D diffusion). PoC showed 6× (12 vs 2). **PASS threshold: ≥ 3×.** Reach = max Chebyshev distance from seed after N ticks.
  - [ ] **G1b (structural complexity — corrected metric):** competitor 4 **size-normalized roughness ratio** ≥ 1.5× competitor 3. **This replaces the size-dependent SA/V metric that was INCONCLUSIVE in the PoC.** The corrected metric:
    - `roughness = actual_surface_area / sphere_surface_area_of_same_volume`
    - where `sphere_surface_area = 4π · r²` and `r = (3·volume / (4π))^(1/3)`
    - A solid block has `roughness ≈ 1.0` (minimal surface for its volume — a sphere is the theoretical min, a cube is ~1.24×). A branched/coral structure has `roughness >> 1.0`. **Size-normalized**: comparing roughness across competitors controls for total volume, so a 13793-voxel solid block (PoC run 1) and a 33-voxel blob (competitor 3) are compared on shape, not size.
    - **PASS threshold: ≥ 1.5×.** If competitor 4 fills the grid solidly (PoC run 1: roughness ≈ 0.8× competitor 3 because the block is bigger and smoother), the gate FAILS — parameter tuning is needed (lower `birth_rate`, higher `consumption_rate`, or higher `dropout_prob`) to hit the branched regime. The PoC's run 2 (consumption-balanced) is the parameter target.
  - [ ] **G2 (regeneration after damage):** destroy an 8×8×8 region at the center of a converged competitor-4 structure, run 40 re-growth steps, measure `% of originally-alive voxels regrown`. PoC showed 100%. **PASS threshold: ≥ 80%.** Competitor 3 (deterministic diffusion) cannot regenerate — it only smooths the damage (PASS requires competitor 4 >> competitor 3).
  - [ ] **G3 (no-regression):** `cargo clippy --all-targets --features grid_3d` clean in katgpt-dec; existing 2D tests + the `GridDims` back-compat accessor tests pass unchanged.
  - [ ] **G4 (latency):** 3D 7-point stencil per-vertex latency within 2× of the 2D 5-point stencil per-vertex (the 7-point does 6 neighbor reads vs 5-point's 4; expect ~1.5×). Birth/death step adds < 20% overhead on top of the Laplacian. Measure with `criterion` on a 32³ grid.
  - [ ] **G5 (zero-alloc):** `stochastic_birth_death_step` scratch buffers reused across 100+ ticks — 0 bytes steady-state allocation (use the `GlobalAlloc` counter pattern from the `evolve_motor_gated_field` zero-alloc test).
  - [ ] **G6 (determinism):** fixed-seed `SplitMix64` → bit-identical `field.data` across 10 runs (the quorum-safety gate — the whole point of modelless + fixed PRNG).

- [ ] **Verdict rule:** G1a + G1b + G2 + G6 must ALL pass for the Gain to hold. G3/G4/G5 are engineering gates (must pass for promotion, but don't refute the gain if they fail — they just block promotion). If G1b FAILS at the default parameters, sweep `birth_rate ∈ [0.01, 0.20]` × `consumption_rate ∈ [0.0, 0.10]` × `dropout_prob ∈ [0.0, 0.5]` to find the branched regime before declaring the gain refuted. The PoC proved the mechanism class; the morphology is parameter-tuning-dependent.

### T8: Promotion decision (post-GOAT)
- [ ] If G1a + G1b + G2 + G3 + G4 + G5 + G6 ALL pass → add `grid_3d` to `default` in `katgpt-dec/Cargo.toml`
- [ ] Update `katgpt-dec/Cargo.toml` feature-flag comment with the GOAT result
- [ ] Update `katgpt-dec/README.md` with a 3D-grid section
- [ ] If G1b cannot be made to pass (no parameter regime produces branched structures) → keep `grid_3d` opt-in, note in the Cargo.toml comment that morphology needs a learned update rule (→ riir-train Issue 004 follow-up). The reach + regeneration gain still holds; only the branched-morphology claim is refuted.

### T9 (deferred — out of scope for this plan): civ engine wiring
- [ ] **Tracked in Issue 155 T4, NOT here.** Wire `grid_3d` + `stochastic_birth_death_step` into the civ engine's `CIV_SPECS` city-growth demand cochains (`riir-ai/crates/riir-engine/...`). The civ engine consumes `argmax_block_type` output (categorical block classes) as the spatial dynamics for `city_found` / `city_develop` / `city_specialize`. This is a consumer-side change in `riir-ai`, gated on T8 promotion.

---

## Optimization Alignment

Per the AGENTS.md hot-loop rules and the existing `graph_laplacian_grid_into` pattern:

- **Pre-allocate boundary vectors** in `grid_3d` via `reserve_exact` — mirrors `grid_2d`, avoids re-allocations during push (✅ T2)
- **Raw-pointer stencil** in `graph_laplacian_grid_3d_into` — branch-free interior, explicit boundary. Same `unsafe` discipline as the 2D path: offsets computed unconditionally, dereferenced only when the `has_*` flag is true (✅ T3)
- **Zero-alloc `_into` variants** — `stochastic_birth_death_step` takes `&mut` scratch buffers, reuses across ticks (✅ T4, G5)
- **Fixed PRNG, no HashMap** — `SplitMix64` is a single `u64` state, advances in O(1), no allocations (✅ T4)
- **Chunked interior loop** — write the 7-point interior in z-slice-major order to help LLC locality (z-slice = `w*h*dim` contiguous f32s); the 2D path's row-major order is the natural 3D extension (✅ T3)
- **No `Mutex` in hot loops** — the growth step is single-threaded by design; parallelism (z-slice parallel) is a T8 follow-up if G4 latency gate is tight

---

## Feature Gate

```toml
[features]
# 3D cubical grid + 7-point stencil Laplacian + stochastic birth/death NCA growth
# (Plan 454, Issue 155 — Sudhakaran 3D NCA, arXiv:2103.08737).
# DEFAULT-OFF until GOAT G1–G6 pass.
grid_3d = []
```

**Default: OFF** until T7 GOAT gate passes. If G1a + G1b + G2 + G6 all pass, promote to `default` per T8.

---

## Anti-Patterns Avoided

- **No softmax** — the alive gate uses sigmoid per the global rule. (`alive = sigmoid(α) > τ`, not `softmax(...)`)
- **No gradient descent** — all four primitives are modelless (fixed PRNG + sigmoid gate + deterministic stencil). The growth is a deterministic function of seed + params, keeping G6 tractable.
- **No latent encoding of position** — `grid_3d` vertex indices are raw integers; the morphogen field is continuous but the alive mask is a hard threshold. No raw↔latent round-trip across a sync boundary.
- **No `merkle_root`-class field omission** — `grid_3d` populates ALL four boundary matrices (B₁/B₂/B₃); the `merkle_root` lesson audits every constructor.
- **No 2D regression** — the `GridDims` enum + back-compat `grid_dims()` accessor means zero 2D call-site changes. T7 G3 guards this.

---

## References

- [Issue 155](../.issues/155_3d_cellcomplex_grid_stochastic_birth_death.md) — the parent issue with the PoC results
- [Sudhakaran et al. 2021](https://arxiv.org/abs/2103.08737) — "3D NCA Growth", ALIFE (doi:10.1162/isal_a_00451)
- [`graph_laplacian_grid_into`](../crates/katgpt-dec/src/operators.rs) — the 5-point-stencil path this plan extends
- [`evolve_motor_gated_field`](../crates/katgpt-dec/src/motor_gated.rs) — the zero-alloc scratch-buffer pattern `stochastic_birth_death_step` mirrors
- [Plan 357](../.plans/357_motor_gated_dec_field.md) — the G5 latency fix that introduced the 2D stencil fast path (the pattern T3 extends)
- [Research 404](../.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md) — the parent NCA research note (Cells2Pixels, Gain verdict)

---

## TL;DR

Ship the one narrow Gain from the 5-paper MMORPG-emergence verdict: 3D `CellComplex::grid_3d` + 7-point-stencil `graph_laplacian_grid_3d_into` + zero-alloc `stochastic_birth_death_step` + `argmax_block_type` bridge, all behind a `grid_3d` feature flag (default-OFF). The `GridDims` enum extends `grid_dims` to 3D without touching any 2D call site. GOAT gate replaces the PoC's size-dependent SA/V metric with a size-normalized roughness ratio (G1b). Gain already CONFIRMED on reach (6×) + regeneration (100%); morphology is parameter-tuning-dependent (the gate sweeps the parameter space before declaring refutation). Modelless throughout — fixed SplitMix64 PRNG, no gradient descent, quorum-safe determinism.
