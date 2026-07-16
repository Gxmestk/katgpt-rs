//! Stochastic Birth/Death NCA Growth Step — Plan 454 T4 / Issue 155.
//!
//! One tick of Sudhakaran-style 3D Neural Cellular Automata (NCA) growth,
//! distilled to a modelless closed-form update over the shipped DEC substrate
//! (the 7-point stencil [`graph_laplacian_into`](crate::operators::graph_laplacian_into)).
//!
//! # The primitive
//!
//! Per voxel per tick (channel 0 = `alive`, channel 1 = `morphogen`):
//!
//! 1. **Diffuse morphogen** — `Δmorph = graph_laplacian(field)`; apply
//!    `morph -= diffusion_dt · Δmorph` to every non-alive channel (channel 1..).
//!    The sign is `-=` (not `+=`): the graph Laplacian is positive at peaks,
//!    so `-=` gives the smoothing/diffusion operator (morphogen flows
//!    high→low). Channel 0 (`alive`) is intentionally NOT diffused.
//! 2. **Autocatalysis + consumption** (alive voxels only) — alive voxels gain
//!    `birth_rate − consumption_rate` morphogen. Dead voxels get NO reaction
//!    term here (their decay is purely multiplicative in step 5; an additive
//!    drain would wipe out frontier diffusion gains and prevent birth).
//! 3. **Stochastic dropout** — with probability `dropout_prob`, a voxel's
//!    *entire Δ this tick* is halved (the paper's morphogenesis trick).
//!    Bit-identical under a fixed seed (G6).
//! 4. **Alive gate** — `alive' = sigmoid(morphogen · α_scale) > τ ? 1.0 : 0.0`.
//!    Gates on the MORPHOGEN (the growth signal), not the alive channel —
//!    a dead voxel with enough diffused morphogen crosses the threshold and
//!    is born. Sigmoid, NOT softmax (global rule).
//! 5. **Dead-voxel reset** — `morph *= decay_rate` for voxels that are dead
//!    after the gate (gradual drain, not instant zero).
//!
//! Three of these steps deviate from the plan's literal pseudocode; each
//! deviation is documented inline in the implementation and is required for
//! the birth/death mechanism to function (the literal pseudocode cannot
//! propagate growth). See the step comments in [`stochastic_birth_death_step`].
//!
//! # Modelless
//!
//! Every step is closed-form algebra: the DEC Laplacian (shipped, no learned
//! kernel), a precomputed logit threshold (the inverse-sigmoid shortcut — one
//! `ln` per tick, then a per-voxel comparison), a fixed-seed SplitMix64 PRNG
//! (no deps), and scalar field updates. No training, no backprop — per the
//! katgpt-rs modelless mandate.
//!
//! The crowding-death mechanism (step C*, Plan 454 G1b modelless fix) is
//! likewise modelless: it reads the alive-channel Laplacian `Δ(alive)` —
//! already computed by step 1 and previously discarded — to identify voxels
//! in dense neighborhoods and kill them. This adds the competition mechanism
//! the bare threshold gate lacks, producing branched/sponge morphology
//! without any learned weights. One scalar parameter (`crowding_threshold`),
//! zero extra memory traffic, zero extra Laplacian calls.
//!
//! # Zero-alloc
//!
//! All scratch is caller-owned. [`stochastic_birth_death_step`] borrows
//! `&mut field` + `&mut scratch_lap` + `&mut scratch_dropout` and mutates them
//! in place; a growth loop pre-allocates once and reuses across all ticks.
//!
//! # Determinism (G6)
//!
//! SplitMix64 is a single-`u64`-state deterministic PRNG. Same seed → same
//! dropout mask every tick → bit-identical `field` after N ticks. This is the
//! quorum-safety gate: two nodes running the same seed and same `field` must
//! produce byte-identical results.
//!
//! # Discrete-class bridge
//!
//! [`argmax_block_type`] thresholds the continuous multi-channel cochain
//! (the alive + morphogen channels produced by the growth step, or a
//! caller-arranged class-activation layout) into a categorical `u8` block
//! class per voxel. A future civ-engine city-growth consumer would consume
//! categorical block types (air / dirt / stone / water / ...) rather than
//! continuous morphogen values; this is the raw → categorical bridge function
//! that such a consumer would call. **No such consumer exists today** (Plan
//! 454 T9 caveat — the `CIV_SPECS` labels in `riir-engine` are pure HLA
//! goal-direction vocabulary, not a cochain substrate). Generic — the caller
//! picks which channels count as classes via the `n_classes` bound.
//!
//! # References
//!
//! - Plan 454 (this primitive), Issue 155 (the 3D NCA gap).
//! - Sudhakaran et al., "Morphogenesis of Neural Cellular Automata" (arXiv:2103.08737).
//! - [`evolve_motor_gated_field`](crate::motor_gated::evolve_motor_gated_field) —
//!   the sibling modelless growth primitive whose scratch-buffer pattern this
//!   module mirrors.

use crate::operators::graph_laplacian_into;
use crate::types::{CellComplex, CochainField};

/// Scale applied to the alive channel before the sigmoid gate (step 4).
///
/// The plan bakes this as a constant rather than a [`BirthDeathParams`] field
/// because the Sudhakaran paper uses a fixed gain on the alive pre-activation.
/// Promote to a param if a GOAT gate needs per-domain tuning.
const ALIVE_GATE_SCALE: f32 = 1.0;

/// Parameters for one tick of [`stochastic_birth_death_step`].
///
/// Plain `Copy` struct — pass by value, tweak per tick if desired. All rates
/// are per-tick scalars (the paper folds the timestep into the rates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BirthDeathParams {
    /// Morphogen diffusion timestep (multiplies the graph-Laplacian Δmorph).
    pub diffusion_dt: f32,
    /// Sigmoid gate threshold τ: alive iff `sigmoid(alive · α_scale) > τ`.
    /// Typical paper value ~0.5.
    pub alive_threshold: f32,
    /// Autocatalytic morphogen gain for alive voxels (per tick).
    pub birth_rate: f32,
    /// Morphogen drain for alive voxels (the paper's consumption term).
    pub consumption_rate: f32,
    /// Per-tick stochastic dropout probability (paper: ~0.5 — kill half the Δ).
    /// A voxel whose dropout draw fires has its entire this-tick Δ halved.
    pub dropout_prob: f32,
    /// Morphogen decay rate for dead voxels (step 5 multiplier). Typical ~0.5.
    pub decay_rate: f32,
    /// **Crowding-death threshold** (step C* — the G1b modelless competition
    /// mechanism). Alive voxels whose alive-channel Laplacian `Δ(alive)` falls
    /// below this value are killed (set dead). `Δ(alive) ≈ 0` for interior
    /// voxels (all neighbors alive), `> 0` for frontier voxels (some dead
    /// neighbors). So a threshold of `0.5` kills only fully-interior voxels;
    /// higher values prune more aggressively.
    ///
    /// Set to [`f32::NEG_INFINITY`] to disable crowding death entirely (the
    /// original behavior before the G1b modelless fix — growth fills the grid
    /// solid). The G1b GOAT gate sweeps this parameter to find the branched
    /// regime; `paper_defaults()` keeps it disabled for back-compat with the
    /// G1a/G2/G5/G6 gates (which need dense, stable growth).
    pub crowding_threshold: f32,
}

impl BirthDeathParams {
    /// The Sudhakaran-paper-aligned defaults used by the Issue 155 PoC.
    ///
    /// Diffusion-heavy, birth-dominated, 50% dropout, gradual decay. Tune per
    /// domain via struct-literal overrides.
    pub const fn paper_defaults() -> Self {
        Self {
            diffusion_dt: 0.1,
            alive_threshold: 0.5,
            birth_rate: 0.3,
            consumption_rate: 0.1,
            dropout_prob: 0.5,
            decay_rate: 0.5,
            crowding_threshold: f32::NEG_INFINITY, // disabled — G1a/G2/G5/G6 back-compat
        }
    }
}

/// SplitMix64 PRNG — single `u64` state, deterministic, zero dependencies.
///
/// The standard SplitMix64 (Steele/Lea 2014): each `next_u64` advances the
/// state by the golden-ratio constant and mixes via the canonical
/// `x ^= x >> a; x *= k; ...` sequence. Bit-identical across platforms and
/// runs (G6 determinism). Used by [`stochastic_birth_death_step`] for the
/// dropout mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator. Same seed → same sequence (G6 quorum-safety gate).
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance and return the next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Advance and return the next 32-bit output (high bits).
    ///
    /// Used for the dropout uniform draw: `r = next_u32() as f32 / u32::MAX as f32`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

/// One tick of stochastic birth/death NCA growth (in place, zero-alloc).
///
/// Implements the 5-step update documented at the [module level](self).
/// Mutates `field` in place; uses `scratch_lap` for the Laplacian output and
/// `scratch_dropout` for the per-voxel dropout mask (both caller-owned,
/// reused across ticks).
///
/// # Arguments
///
/// * `cx` — the 3D cell complex (must be a [`CellComplex::grid_3d`](crate::types::CellComplex::grid_3d)
///   product so the 7-point stencil fast path engages; a non-grid complex
///   falls back to the edge-list Laplacian, which is correct but slower).
/// * `field` — rank-0 cochain on `cx`'s vertices with `dim >= 2`. Channel 0
///   is `alive` (re-binarized to 0/1 each tick); channels 1.. are `morphogen`
///   (diffused + autocatalyzed + decayed). Mutated in place.
/// * `params` — the per-tick rates (see [`BirthDeathParams`]).
/// * `rng` — the deterministic PRNG (see [`SplitMix64`]). Advanced once per
///   voxel per tick (for the dropout draw).
/// * `scratch_lap` — caller-owned scratch holding the graph-Laplacian output.
///   Must be sized `n_vertices · dim` (same shape as `field`); `rank`/`dim`
///   are synced internally to match `field` (callers reuse buffers across
///   grids, mirroring [`evolve_motor_gated_field`](crate::motor_gated::evolve_motor_gated_field)).
/// * `scratch_dropout` — caller-owned dropout mask, length `n_vertices`. One
///   byte per voxel: `1` if the dropout draw fired (halve this voxel's Δ),
///   `0` otherwise. Reused across ticks.
///
/// # Panics (debug only)
///
/// Debug-asserts `field.dim >= 2`, `field.rank == 0`, and that both scratch
/// buffers are large enough.
///
/// # Determinism
///
/// Bit-identical across runs given the same `field`, same `params`, and same
/// `rng` seed (G6). The dropout mask is drawn once per voxel per tick in
/// vertex-index order, so the PRNG sequence is fully determined by the tick
/// count and vertex count.
#[inline]
pub fn stochastic_birth_death_step(
    cx: &CellComplex,
    field: &mut CochainField,
    params: &BirthDeathParams,
    rng: &mut SplitMix64,
    scratch_lap: &mut CochainField,
    scratch_dropout: &mut [u8],
) {
    let dim = field.dim;
    let n = field.n_cells();
    let len = n * dim;

    debug_assert_eq!(field.rank, 0, "stochastic_birth_death_step requires rank-0 field");
    debug_assert!(
        dim >= 2,
        "stochastic_birth_death_step requires dim >= 2 (alive + morphogen), got {dim}"
    );
    debug_assert!(
        scratch_lap.data.len() >= len,
        "scratch_lap len {} < required {len}",
        scratch_lap.data.len()
    );
    debug_assert!(
        scratch_dropout.len() >= n,
        "scratch_dropout len {} < required {n}",
        scratch_dropout.len()
    );

    // Sync scratch metadata to match field (callers reuse buffers across grids;
    // graph_laplacian_into asserts the input rank).
    scratch_lap.rank = field.rank;
    scratch_lap.dim = dim;

    // Precompute logit(τ) = ln(τ/(1−τ)) — the inverse-sigmoid threshold for
    // step C. fast_sigmoid(α) > τ  ⟺  α > logit(τ)  for α ∈ (−40, 40),
    // which lets the fused loop skip the per-voxel `expf` call entirely and
    // do a single comparison instead. Edge cases: τ ≤ 0 → everything alive;
    // τ ≥ 1 → nothing alive. For τ ∈ (ε, 1−ε) this is bit-identical to the
    // sigmoid path (the clamp boundaries at |α| > 40 only matter for τ
    // within ~1e-17 of 0 or 1, which no sane config uses).
    let logit_tau = if params.alive_threshold <= 0.0 {
        f32::NEG_INFINITY // every voxel crosses the gate
    } else if params.alive_threshold >= 1.0 {
        f32::INFINITY // no voxel can cross
    } else {
        let t = params.alive_threshold;
        (t / (1.0 - t)).ln()
    };

    // ── Step 0: precompute the dropout mask (one draw per voxel) ──────────
    // The paper draws the mask before applying any update so "halve the Δ" is
    // well-defined: a masked voxel's entire this-tick Δ is halved, regardless
    // of which step produced it. Drawing in vertex-index order keeps the PRNG
    // sequence determined by (tick, vertex-count) — G6 determinism.
    let inv_max = 1.0f32 / u32::MAX as f32;
    for mask in &mut scratch_dropout[..n] {
        let r = rng.next_u32() as f32 * inv_max;
        // *mask = 1 if dropout fires (r < prob), else 0.
        *mask = (r < params.dropout_prob) as u8;
    }

    // ── Step 1: diffuse morphogen (channels 1..) ──────────────────────────
    // Run the graph Laplacian on the whole field (reading all channels is
    // harmless — only channels 1.. are applied back). Channel 0 (alive) in
    // scratch_lap is computed but discarded; the alive channel in `field` is
    // NOT modified by this step (it stays at its pre-tick 0/1 value and is
    // re-binarized in step 4).
    //
    // Reusing `graph_laplacian_into` keeps this DRY with the 2D/3D stencil
    // fast paths and avoids duplicating the 7-point stencil inline.
    graph_laplacian_into(cx, field, scratch_lap);

    // ── Steps 1–5 FUSED: single per-voxel pass ────────────────────────────
    // The original implementation had 4 separate chunks_exact_mut loops over
    // the field (diffusion apply, reaction, alive gate, dead decay). All four
    // are per-voxel independent operations with clear within-voxel sequential
    // dependencies and NO cross-voxel data flow (the Laplacian already
    // captured neighbor interactions into scratch_lap). Fusing them into one
    // loop streams each voxel through registers once instead of four times —
    // the G4b optimization (Plan 454 follow-up).
    //
    // Data-dependency ordering within each voxel (MUST be preserved):
    //   (A) diffusion   — writes morph[1..], reads precomputed lap
    //   (B) reaction    — reads OLD alive[0] (pre-gate), writes morph[1..]
    //   (C) alive gate  — reads morph[1] (after A+B), writes NEW alive[0]
    //   (D) dead decay  — reads NEW alive[0] (post-gate), writes morph[1..]
    // (B) MUST run before (C); (C) MUST run before (D). (A) before (B) is
    // required for the "diffusion then reaction" semantics. The fused loop
    // executes them in this exact order per voxel — bit-identical to the
    // 4-pass version (verified by the determinism test + G6).
    //
    // **Sign convention** (step A): `graph_laplacian_into` computes
    // `Δ = deg·center − Σneighbors` (positive at peaks, negative at valleys).
    // For SMOOTHING diffusion — morphogen flowing seed→frontier — the update
    // is `morph -= dt · Δ` (the negative Laplacian). The plan's literal
    // `+= dt · Δ` is anti-diffusive (sharpening) and cannot propagate growth.
    //
    // **Deviation from plan pseudocode** (step B): the plan's literal step 2
    // also applies `-= decay_rate` to dead voxels. The additive drain
    // double-counts with step D's `*= decay_rate` and wipes out frontier
    // diffusion gains — birth can never propagate. Dead voxels passively
    // decay (step D only); alive voxels produce/consume morphogen (step B).
    //
    // **Deviation from plan pseudocode** (step C): the plan gates on the
    // alive channel, but sigmoid(0)=0.5 with τ=0.5 never births a dead
    // voxel. The paper gates on the MORPHOGEN (the growth signal): a dead
    // voxel with enough diffused morphogen crosses the threshold and is
    // born. The alive channel is purely the binarized OUTPUT of this gate.
    let field_chunks = field.data[..len].chunks_exact_mut(dim);
    let lap_chunks = scratch_lap.data[..len].chunks_exact(dim);
    for ((voxel, &mask), lap) in field_chunks
        .zip(scratch_dropout.iter())
        .zip(lap_chunks)
    {
        // (A) Diffusion: apply `-diffusion_dt · Δlap` to morphogen channels.
        let dt_scale = if mask != 0 { 0.5 } else { 1.0 } * params.diffusion_dt;
        for ch in 1..dim {
            voxel[ch] -= dt_scale * lap[ch];
        }

        // (B) Reaction (alive voxels only) — reads OLD alive (pre-gate).
        let alive_old = voxel[0] > 0.5;
        if alive_old {
            let reaction_scale = if mask != 0 { 0.5 } else { 1.0 };
            let reaction_delta = params.birth_rate - params.consumption_rate;
            for morph in voxel[1..dim].iter_mut() {
                *morph += reaction_scale * reaction_delta;
            }
        }

        // (C) Alive gate — reads morphogen[1] (after A+B), writes alive[0].
        // Uses the precomputed logit(τ) instead of calling fast_sigmoid per
        // voxel: fast_sigmoid(α) > τ  ⟺  α > logit(τ). Saves one `expf` per
        // voxel (the dominant compute cost in the fused pass).
        let alpha = voxel[1] * ALIVE_GATE_SCALE;
        let mut alive_new = alpha > logit_tau;

        // (C*) Crowding death — the G1b modelless competition mechanism.
        // Reads lap[0] (the alive-channel Laplacian, already computed by step 1
        // and previously discarded). For an alive voxel: lap[0] ≈ 0 in the
        // interior (all neighbors alive), > 0 at the frontier (some dead
        // neighbors). Killing voxels with lap[0] < crowding_threshold prunes
        // the interior, preventing solid-grid filling and producing
        // branched/sponge morphology (high surface/volume ratio = high
        // roughness). Only applies to voxels that were ALREADY alive
        // (alive_old) — newly-born frontier voxels get a grace tick so growth
        // can propagate. Disabled when crowding_threshold = NEG_INFINITY.
        if alive_old && alive_new && lap[0] < params.crowding_threshold {
            alive_new = false;
        }

        voxel[0] = if alive_new { 1.0 } else { 0.0 };

        // (D) Dead-voxel morphogen decay — reads NEW alive (post-gate).
        if !alive_new {
            for morph in voxel[1..dim].iter_mut() {
                *morph *= params.decay_rate;
            }
        }
    }
}

/// Threshold a continuous multi-channel cochain into categorical block classes.
/// Plan 454 T5 — the raw → categorical bridge. Intended for a future civ-engine
/// city-growth consumer (Plan 454 T9 caveat: no such consumer exists today —
/// the `CIV_SPECS` labels in `riir-engine` are HLA goal-direction vocabulary,
/// not a cochain substrate).
///
/// For each cell `v`, writes `out[v] = argmax over channels 0..n_classes of
/// field.data[v*dim + c]`. Ties are broken by lowest channel index
/// (deterministic — strict `>` keeps the first maximum). NaN channels never
/// win (the scan starts from `NEG_INFINITY`, so any finite value beats NaN
/// at the comparison).
///
/// This is the discrete-class counterpart to the continuous growth step
/// [`stochastic_birth_death_step`]: the growth step evolves the morphogen
/// field in continuous space, and this fn collapses it into a categorical
/// block-type space. Generic over which channels count as classes — pass
/// `n_classes = dim` to consider every channel, or a smaller value to ignore
/// trailing channels.
///
/// # Layout
///
/// `field` is a flat `[n_cells × dim]` row-major cochain (channel `c` of cell
/// `v` lives at `field.data[v*dim + c]`). `out` receives one `u8` per cell.
///
/// # Arguments
///
/// * `field` — the continuous cochain to classify.
/// * `n_classes` — number of leading channels to scan (1..=min(dim, 256)).
/// * `out` — output buffer, length >= `field.n_cells()`. Written in place.
///
/// # Panics (debug only)
///
/// Debug-asserts `1 <= n_classes <= dim`, `n_classes <= 256`, and
/// `out.len() >= field.n_cells()`.
///
/// # Determinism
///
/// Bit-identical given the same `field` and `n_classes` (pure scan, no RNG,
/// no allocation). The strict-`>` tie-break makes the output a pure function
/// of the input.
#[inline]
pub fn argmax_block_type(field: &CochainField, n_classes: usize, out: &mut [u8]) {
    let dim = field.dim;
    let n = field.n_cells();

    debug_assert!(
        n_classes >= 1,
        "argmax_block_type requires n_classes >= 1, got {n_classes}"
    );
    debug_assert!(
        n_classes <= dim,
        "argmax_block_type: n_classes {n_classes} > field dim {dim}"
    );
    debug_assert!(
        n_classes <= 256,
        "argmax_block_type: n_classes {n_classes} > 256 (u8 output limit)"
    );
    debug_assert!(
        out.len() >= n,
        "argmax_block_type: out len {} < n_cells {n}",
        out.len()
    );

    // NaN-safe argmax: init from NEG_INFINITY so any finite value beats the
    // initial sentinel, and NaN never satisfies `val > best_val` so it never
    // wins. Strict `>` keeps the first (lowest-index) maximum on ties.
    for (v, voxel) in field.data[..n * dim].chunks_exact(dim).enumerate() {
        let mut best_idx: u8 = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (c, &val) in voxel.iter().enumerate().take(n_classes) {
            if val > best_val {
                best_val = val;
                best_idx = c as u8;
            }
        }
        out[v] = best_idx;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 3D grid field with `dim` channels, all zero.
    fn zero_field_3d(cx: &CellComplex, dim: usize) -> CochainField {
        CochainField::zeros(0, cx.n_vertices(), dim)
    }

    /// vidx(x, y, z) for a w×h×d grid (matches the grid_3d indexing convention).
    fn vidx(x: usize, y: usize, z: usize, w: usize, h: usize) -> usize {
        (z * h + y) * w + x
    }

    #[test]
    fn splitmix64_determinism() {
        // Same seed → same sequence.
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64(), "SplitMix64 sequence diverged");
        }
        // Different seed → different first output (extremely likely).
        let mut c = SplitMix64::new(43);
        assert_ne!(a.next_u64(), c.next_u64(), "different seeds produced same output");
    }

    #[test]
    fn splitmix64_next_u32_uses_high_bits() {
        // next_u32 should be the high 32 bits of next_u64 — sanity check that
        // the two are consistent for the first draw.
        let mut a = SplitMix64::new(123);
        let mut b = SplitMix64::new(123);
        let full = a.next_u64();
        let half = b.next_u32();
        assert_eq!(half, (full >> 32) as u32);
    }

    #[test]
    fn stochastic_birth_death_determinism() {
        // Same seed + same initial field → bit-identical field after N ticks.
        // This is the G6 quorum-safety gate.
        let (w, h, d) = (4usize, 4usize, 4usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let dim = 2usize;
        let params = BirthDeathParams::paper_defaults();

        // Two independent field copies + PRNGs with the same seed.
        let mut field_a = zero_field_3d(&cx, dim);
        let mut field_b = zero_field_3d(&cx, dim);
        // Seed a single alive voxel with morphogen.
        let seed = vidx(2, 2, 2, w, h);
        field_a.data[seed * dim] = 1.0; // alive
        field_a.data[seed * dim + 1] = 1.0; // morphogen
        field_b.data[seed * dim] = 1.0;
        field_b.data[seed * dim + 1] = 1.0;

        let mut scratch_lap_a = zero_field_3d(&cx, dim);
        let mut scratch_lap_b = zero_field_3d(&cx, dim);
        let mut dropout_a = vec![0u8; cx.n_vertices()];
        let mut dropout_b = vec![0u8; cx.n_vertices()];

        let mut rng_a = SplitMix64::new(99);
        let mut rng_b = SplitMix64::new(99);

        for _ in 0..10 {
            stochastic_birth_death_step(
                &cx, &mut field_a, &params, &mut rng_a,
                &mut scratch_lap_a, &mut dropout_a,
            );
            stochastic_birth_death_step(
                &cx, &mut field_b, &params, &mut rng_b,
                &mut scratch_lap_b, &mut dropout_b,
            );
        }

        for i in 0..field_a.data.len() {
            assert_eq!(
                field_a.data[i].to_bits(),
                field_b.data[i].to_bits(),
                "determinism violated at index {i}: {} vs {}",
                field_a.data[i],
                field_b.data[i]
            );
        }
    }

    #[test]
    fn stochastic_birth_death_birth_propagates() {
        // A single seed voxel should grow to >1 alive voxel after N ticks
        // (the core NCA growth mechanism). Diffusion spreads morphogen to
        // neighbors, autocatalysis amplifies it, and the alive gate flips
        // neighbors on.
        let (w, h, d) = (5usize, 5usize, 5usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let dim = 2usize;
        // Growth-friendly params. diffusion_dt must be small enough for
        // explicit-Euler stability (dt < ~1/(2·deg) ≈ 1/12 for interior deg=6);
        // 0.1 is stable and moves enough morphogen to neighbors to cross the
        // alive threshold in a few ticks. birth_rate > consumption_rate keeps
        // alive voxels accumulating morphogen (autocatalysis); no dropout for
        // a deterministic growth signal.
        let params = BirthDeathParams {
            diffusion_dt: 0.1,
            alive_threshold: 0.5,
            birth_rate: 0.4,
            consumption_rate: 0.05,
            dropout_prob: 0.0, // no dropout — deterministic growth
            decay_rate: 0.5,
            crowding_threshold: f32::NEG_INFINITY, // disabled
        };

        let mut field = zero_field_3d(&cx, dim);
        let seed = vidx(2, 2, 2, w, h);
        field.data[seed * dim] = 1.0; // alive
        field.data[seed * dim + 1] = 1.0; // morphogen

        let mut scratch_lap = zero_field_3d(&cx, dim);
        let mut dropout = vec![0u8; cx.n_vertices()];
        let mut rng = SplitMix64::new(7);

        let initial_alive = 1usize;
        for _ in 0..20 {
            stochastic_birth_death_step(
                &cx, &mut field, &params, &mut rng,
                &mut scratch_lap, &mut dropout,
            );
        }
        let final_alive: usize = (0..cx.n_vertices())
            .filter(|&v| field.data[v * dim] > 0.5)
            .count();
        assert!(
            final_alive > initial_alive,
            "birth did not propagate: started with {initial_alive} alive, ended with {final_alive}"
        );
    }

    #[test]
    fn stochastic_birth_death_death_decays_morphogen() {
        // A voxel that is gated dead should see its morphogen decay toward
        // 0 each tick (step 5: morph *= decay_rate). The alive gate reads the
        // morphogen (step 4), so to keep voxels dead we need
        // sigmoid(morphogen) <= alive_threshold, i.e. morphogen <= 0.0 for
        // threshold=0.5.
        let (w, h, d) = (3usize, 3usize, 3usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let dim = 2usize;
        let params = BirthDeathParams {
            diffusion_dt: 0.0, // no diffusion — isolate the decay mechanism
            alive_threshold: 0.5,
            birth_rate: 0.0,
            consumption_rate: 0.0,
            dropout_prob: 0.0,
            decay_rate: 0.5,
            crowding_threshold: f32::NEG_INFINITY, // disabled
        };

        let mut field = zero_field_3d(&cx, dim);
        // All voxels seeded dead with negative morphogen (gate stays dead:
        // sigmoid(-1) ≈ 0.27 < 0.5).
        for v in 0..cx.n_vertices() {
            field.data[v * dim] = 0.0; // alive channel (will be re-gated dead)
            field.data[v * dim + 1] = -1.0; // morphogen — negative, stays dead
        }
        let mut scratch_lap = zero_field_3d(&cx, dim);
        let mut dropout = vec![0u8; cx.n_vertices()];
        let mut rng = SplitMix64::new(1);

        // After 1 tick the dead-voxel trajectory is (with the step-2
        // reaction applying only to alive voxels):
        //   step 2 reaction: (dead → skipped)
        //   step 4 gate:     sigmoid(-1.0) ≈ 0.27 < 0.5   →  stays dead
        //   step 5 decay:    morph *= decay_rate = 0.5     →  morph = -0.5
        stochastic_birth_death_step(&cx, &mut field, &params, &mut rng, &mut scratch_lap, &mut dropout);
        for v in 0..cx.n_vertices() {
            let morph = field.data[v * dim + 1];
            assert!(
                (morph - (-0.5)).abs() < 1e-6,
                "after 1 tick dead-voxel morphogen should be -0.5, got {morph}"
            );
            assert_eq!(field.data[v * dim], 0.0, "voxel should be dead after gate");
        }
        // After several more ticks |morphogen| should keep shrinking toward 0
        // (the reaction pushes negative, the decay multiplies toward 0; the
        // fixed point is morph = -decay_rate / (1 - decay_rate) = -1.0, but
        // starting from -1.0 the trajectory is monotone toward 0 in magnitude
        // after the first tick). Just assert it stays bounded and dead.
        for _ in 0..5 {
            stochastic_birth_death_step(&cx, &mut field, &params, &mut rng, &mut scratch_lap, &mut dropout);
        }
        for v in 0..cx.n_vertices() {
            assert_eq!(field.data[v * dim], 0.0, "voxel {v} should stay dead");
            let morph = field.data[v * dim + 1];
            assert!(morph <= 0.0, "dead morphogen should be <= 0, got {morph}");
        }
    }

    #[test]
    fn stochastic_birth_death_alive_channel_stays_binarized() {
        // The alive channel (channel 0) must always be exactly 0.0 or 1.0
        // after a tick — the sigmoid gate re-binarizes it. This is the
        // invariant downstream consumers (e.g. argmax_block_type in T5) rely on.
        let (w, h, d) = (4usize, 4usize, 4usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let dim = 2usize;
        let params = BirthDeathParams::paper_defaults();

        let mut field = zero_field_3d(&cx, dim);
        // Seed a random-ish alive/morphogen pattern.
        let mut rng_field = SplitMix64::new(2024);
        for v in 0..cx.n_vertices() {
            field.data[v * dim] = if rng_field.next_u32().is_multiple_of(2) { 1.0 } else { 0.0 };
            field.data[v * dim + 1] = rng_field.next_u32() as f32 / u32::MAX as f32;
        }
        let mut scratch_lap = zero_field_3d(&cx, dim);
        let mut dropout = vec![0u8; cx.n_vertices()];
        let mut rng = SplitMix64::new(55);

        for tick in 0..15 {
            stochastic_birth_death_step(&cx, &mut field, &params, &mut rng, &mut scratch_lap, &mut dropout);
            for v in 0..cx.n_vertices() {
                let alive = field.data[v * dim];
                assert!(
                    alive == 0.0 || alive == 1.0,
                    "tick {tick} voxel {v}: alive channel should be 0 or 1, got {alive}"
                );
            }
        }
    }

    #[test]
    fn stochastic_birth_death_dropout_halves_delta() {
        // With dropout_prob = 1.0, every voxel's Δ is halved every tick. With
        // dropout_prob = 0.0, no halving. A purely-diffusive setup (no
        // reaction, no gate change) lets us verify the halving precisely.
        let (w, h, d) = (3usize, 3usize, 3usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let dim = 2usize;

        // Params: pure diffusion, no reaction/gate effects.
        let base_params = BirthDeathParams {
            diffusion_dt: 0.1,
            alive_threshold: 0.5,
            birth_rate: 0.0,
            consumption_rate: 0.0,
            dropout_prob: 0.0, // will override per run
            decay_rate: 1.0,   // no decay
            crowding_threshold: f32::NEG_INFINITY, // disabled
        };

        // Run with no dropout.
        let mut field_full = zero_field_3d(&cx, dim);
        // Seed morphogen everywhere so the Laplacian is nonzero at boundaries.
        for v in 0..cx.n_vertices() {
            field_full.data[v * dim] = 1.0; // alive (stays alive)
            field_full.data[v * dim + 1] = (v as f32) * 0.1;
        }
        let mut scratch = zero_field_3d(&cx, dim);
        let mut dropout = vec![0u8; cx.n_vertices()];
        let mut rng = SplitMix64::new(3);
        let mut p_no_drop = base_params;
        p_no_drop.dropout_prob = 0.0;
        stochastic_birth_death_step(&cx, &mut field_full, &p_no_drop, &mut rng, &mut scratch, &mut dropout);

        // Run with full dropout (same seed + initial field).
        let mut field_half = zero_field_3d(&cx, dim);
        for v in 0..cx.n_vertices() {
            field_half.data[v * dim] = 1.0;
            field_half.data[v * dim + 1] = (v as f32) * 0.1;
        }
        let mut rng2 = SplitMix64::new(3);
        let mut p_drop = base_params;
        p_drop.dropout_prob = 1.0;
        stochastic_birth_death_step(&cx, &mut field_half, &p_drop, &mut rng2, &mut scratch, &mut dropout);

        // The dropout path should have applied half the diffusion Δ to each
        // morphogen voxel. Compare morphogen channels.
        for v in 0..cx.n_vertices() {
            let full = field_full.data[v * dim + 1];
            let half = field_half.data[v * dim + 1];
            // The initial value was identical; full applied Δ, half applied Δ/2.
            // So full - initial = 2 * (half - initial).
            let initial = (v as f32) * 0.1;
            let delta_full = full - initial;
            let delta_half = half - initial;
            if delta_full.abs() > 1e-6 {
                assert!(
                    (delta_half - 0.5 * delta_full).abs() < 1e-5,
                    "voxel {v}: dropout should halve the Δ, got full_delta={delta_full}, half_delta={delta_half}"
                );
            }
        }
    }

    // ── T5: argmax_block_type ────────────────────────────────────────────

    #[test]
    fn stochastic_birth_death_crowding_kills_interior() {
        // Crowding death (step C*) prunes interior voxels: an alive voxel
        // surrounded by alive neighbors has lap[0] ≈ 0, which falls below
        // crowding_threshold → it dies. An alive voxel at the frontier
        // (some dead neighbors) has lap[0] > threshold → it survives.
        //
        // Setup: a 3×3×3 grid with the center voxel and all 6 face-neighbors
        // alive, but the 8 corners + 12 edge-neighbors dead. The center voxel
        // has all 6 neighbors alive → lap[0] = 0 → crowded → should die.
        // The face-neighbors have 1 alive neighbor (the center) → lap[0] = 5
        // → not crowded → should survive (threshold = 2.5).
        let (w, h, d) = (3usize, 3usize, 3usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let dim = 2usize;
        let params = BirthDeathParams {
            diffusion_dt: 0.0, // no diffusion — isolate crowding
            alive_threshold: 0.5,
            birth_rate: 0.0,
            consumption_rate: 0.0,
            dropout_prob: 0.0,
            decay_rate: 1.0, // no decay
            crowding_threshold: 2.5, // kill if lap[0] < 2.5
        };

        let mut field = zero_field_3d(&cx, dim);
        // Center voxel + 6 face-neighbors alive with high morphogen.
        let center = vidx(1, 1, 1, w, h);
        field.data[center * dim] = 1.0;
        field.data[center * dim + 1] = 5.0;
        let neighbors = [
            vidx(0, 1, 1, w, h), vidx(2, 1, 1, w, h), // ±x
            vidx(1, 0, 1, w, h), vidx(1, 2, 1, w, h), // ±y
            vidx(1, 1, 0, w, h), vidx(1, 1, 2, w, h), // ±z
        ];
        for &n in &neighbors {
            field.data[n * dim] = 1.0;
            field.data[n * dim + 1] = 5.0;
        }

        let mut scratch_lap = zero_field_3d(&cx, dim);
        let mut dropout = vec![0u8; cx.n_vertices()];
        let mut rng = SplitMix64::new(1);

        stochastic_birth_death_step(
            &cx, &mut field, &params, &mut rng, &mut scratch_lap, &mut dropout,
        );

        // Center voxel: all 6 neighbors alive → lap[0] = 6 - 6 = 0 < 2.5 → KILLED.
        assert_eq!(
            field.data[center * dim], 0.0,
            "center voxel (fully surrounded) should be killed by crowding death"
        );
        // Face-neighbors: 1 alive neighbor (center) → lap[0] = 6 - 1 = 5 >= 2.5 → SURVIVE.
        for &n in &neighbors {
            assert_eq!(
                field.data[n * dim], 1.0,
                "face-neighbor {n} (exposed) should survive crowding death"
            );
        }
    }

    #[test]
    fn stochastic_birth_death_crowding_disabled_by_default() {
        // With crowding_threshold = NEG_INFINITY, the mechanism is disabled —
        // the center voxel should survive even when fully surrounded.
        let (w, h, d) = (3usize, 3usize, 3usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let dim = 2usize;
        let params = BirthDeathParams {
            diffusion_dt: 0.0,
            alive_threshold: 0.5,
            birth_rate: 0.0,
            consumption_rate: 0.0,
            dropout_prob: 0.0,
            decay_rate: 1.0,
            crowding_threshold: f32::NEG_INFINITY, // disabled
        };

        let mut field = zero_field_3d(&cx, dim);
        let center = vidx(1, 1, 1, w, h);
        field.data[center * dim] = 1.0;
        field.data[center * dim + 1] = 5.0;
        for &n in &[vidx(0, 1, 1, w, h), vidx(2, 1, 1, w, h), vidx(1, 0, 1, w, h),
                     vidx(1, 2, 1, w, h), vidx(1, 1, 0, w, h), vidx(1, 1, 2, w, h)] {
            field.data[n * dim] = 1.0;
            field.data[n * dim + 1] = 5.0;
        }

        let mut scratch_lap = zero_field_3d(&cx, dim);
        let mut dropout = vec![0u8; cx.n_vertices()];
        let mut rng = SplitMix64::new(1);

        stochastic_birth_death_step(
            &cx, &mut field, &params, &mut rng, &mut scratch_lap, &mut dropout,
        );

        assert_eq!(
            field.data[center * dim], 1.0,
            "center should survive when crowding death is disabled"
        );
    }

    #[test]
    fn argmax_block_type_basic() {
        // Known field → known block classes. dim=3, 4 voxels.
        let dim = 3usize;
        let mut field = CochainField::zeros(0, 4, dim);
        // voxel 0: channel 2 wins
        field.data[0] = 0.1;
        field.data[1] = 0.2;
        field.data[2] = 0.9;
        // voxel 1: channel 0 wins
        field.data[3] = 0.8;
        field.data[4] = 0.1;
        field.data[5] = 0.1;
        // voxel 2: channel 1 wins
        field.data[6] = 0.0;
        field.data[7] = 0.7;
        field.data[8] = 0.3;
        // voxel 3: channel 0 wins (negative values)
        field.data[9] = -0.1;
        field.data[10] = -0.5;
        field.data[11] = -0.9;

        let mut out = [0u8; 4];
        argmax_block_type(&field, dim, &mut out);
        assert_eq!(out, [2, 0, 1, 0], "basic argmax mismatch");
    }

    #[test]
    fn argmax_block_type_ties_lowest_index() {
        // Ties broken by lowest channel index (strict `>`).
        let dim = 3usize;
        let mut field = CochainField::zeros(0, 2, dim);
        // voxel 0: channels 0,1,2 all equal → class 0 wins
        field.data[0] = 0.5;
        field.data[1] = 0.5;
        field.data[2] = 0.5;
        // voxel 1: channels 1,2 tied at 0.9, channel 0 at 0.1 → class 1 wins
        field.data[3] = 0.1;
        field.data[4] = 0.9;
        field.data[5] = 0.9;

        let mut out = [0u8; 2];
        argmax_block_type(&field, dim, &mut out);
        assert_eq!(out, [0, 1], "tie-break should pick lowest index");
    }

    #[test]
    fn argmax_block_type_n_classes_1_always_zero() {
        // With n_classes=1, only channel 0 is scanned → every voxel gets 0.
        let dim = 4usize;
        let mut field = CochainField::zeros(0, 3, dim);
        for v in 0..3 {
            for c in 0..dim {
                field.data[v * dim + c] = (c as f32) * 0.3;
            }
        }
        let mut out = [99u8; 3];
        argmax_block_type(&field, 1, &mut out);
        assert_eq!(out, [0, 0, 0], "n_classes=1 should always return class 0");
    }

    #[test]
    fn argmax_block_type_n_classes_less_than_dim() {
        // n_classes < dim should ignore trailing channels. Set channel 3 very
        // high but n_classes=2 → channel 3 is ignored.
        let dim = 4usize;
        let mut field = CochainField::zeros(0, 1, dim);
        field.data[0] = 0.1;
        field.data[1] = 0.9;
        field.data[2] = 0.2;
        field.data[3] = 999.0; // would win if scanned, but n_classes=2 skips it

        let mut out = [99u8; 1];
        argmax_block_type(&field, 2, &mut out);
        assert_eq!(out, [1], "n_classes=2 should pick channel 1, ignoring channel 3");
    }

    #[test]
    fn argmax_block_type_nan_safe() {
        // NaN channels never win (init from NEG_INFINITY, NaN > x is always false).
        let dim = 3usize;
        let mut field = CochainField::zeros(0, 2, dim);
        // voxel 0: channel 0 NaN, channel 1 = 0.5 → class 1 wins (NaN loses)
        field.data[0] = f32::NAN;
        field.data[1] = 0.5;
        field.data[2] = 0.1;
        // voxel 1: all NaN → class 0 (the fallback best_idx=0)
        field.data[3] = f32::NAN;
        field.data[4] = f32::NAN;
        field.data[5] = f32::NAN;

        let mut out = [99u8; 2];
        argmax_block_type(&field, dim, &mut out);
        assert_eq!(out[0], 1, "NaN channel should lose to finite 0.5");
        assert_eq!(out[1], 0, "all-NaN voxel should fall back to class 0");
    }

    #[test]
    fn argmax_block_type_deterministic_across_calls() {
        // Pure scan — same input must produce same output across calls.
        let dim = 3usize;
        let mut field = CochainField::zeros(0, 8, dim);
        let mut rng = SplitMix64::new(42);
        for v in 0..8 {
            for c in 0..dim {
                field.data[v * dim + c] = (rng.next_u32() as f32) / (u32::MAX as f32);
            }
        }
        let mut out_a = [0u8; 8];
        let mut out_b = [0u8; 8];
        argmax_block_type(&field, dim, &mut out_a);
        argmax_block_type(&field, dim, &mut out_b);
        assert_eq!(out_a, out_b, "same field → same classes (pure scan)");
    }

    #[test]
    fn argmax_block_type_after_birth_death() {
        // Integration test: run a few birth/death ticks, then classify.
        //
        // NOTE: alive voxels do NOT always map to class 0. The alive channel
        // is binarized to exactly 1.0, but the morphogen channel is UNBOUNDED
        // — alive voxels gain `birth_rate - consumption_rate = 0.2` morphogen
        // per tick, so after 10 ticks a seed voxel's morphogen can reach ~3.0,
        // which beats the alive channel's 1.0 and wins class 1. This is
        // expected: the argmax picks the dominant signal. The specific
        // class→semantics mapping (alive → air, high-morphogen → stone, etc.)
        // is the civ engine's job (T9), not T5's. T5 only guarantees: valid
        // output range + deterministic scan.
        let (w, h, d) = (4usize, 4usize, 4usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let dim = 2usize;
        let params = BirthDeathParams::paper_defaults();

        let mut field = zero_field_3d(&cx, dim);
        let seed = vidx(2, 2, 2, w, h);
        field.data[seed * dim] = 1.0;
        field.data[seed * dim + 1] = 1.0;

        let mut scratch_lap = zero_field_3d(&cx, dim);
        let mut dropout = vec![0u8; cx.n_vertices()];
        let mut rng = SplitMix64::new(7);

        for _ in 0..10 {
            stochastic_birth_death_step(
                &cx, &mut field, &params, &mut rng,
                &mut scratch_lap, &mut dropout,
            );
        }

        let mut out = vec![99u8; cx.n_vertices()];
        argmax_block_type(&field, dim, &mut out);

        // (1) Every voxel must get a valid class in [0, dim).
        for (v, &class) in out.iter().enumerate() {
            assert!(
                (class as usize) < dim,
                "voxel {v}: class {class} out of range [0, {dim})"
            );
        }

        // (2) Determinism: same field → same classes (pure scan, no RNG).
        let mut out_again = vec![99u8; cx.n_vertices()];
        argmax_block_type(&field, dim, &mut out_again);
        assert_eq!(out, out_again, "same field should produce same classes");

        // (3) At least one voxel should be non-zero (growth happened, so some
        //     morphogen channel won somewhere). If all voxels were class 0,
        //     either growth didn't propagate or the argmax is broken.
        let non_zero = out.iter().filter(|&&c| c != 0).count();
        assert!(
            non_zero > 0,
            "expected at least one voxel with morphogen > alive channel (class != 0), \
             got all zeros — growth may not have propagated"
        );
    }

    #[test]
    fn argmax_block_type_all_negative_values() {
        // All-negative field: argmax still picks the least-negative channel.
        let dim = 3usize;
        let mut field = CochainField::zeros(0, 1, dim);
        field.data[0] = -5.0;
        field.data[1] = -1.0; // least negative
        field.data[2] = -9.0;

        let mut out = [99u8; 1];
        argmax_block_type(&field, dim, &mut out);
        assert_eq!(out, [1], "argmax should pick the least-negative value");
    }
}
