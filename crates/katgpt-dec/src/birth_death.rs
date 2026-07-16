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
//! kernel), elementwise sigmoid (shipped [`fast_sigmoid`](crate::simd::fast_sigmoid)),
//! a fixed-seed SplitMix64 PRNG (no deps), and scalar field updates. No
//! training, no backprop — per the katgpt-rs modelless mandate.
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
//! # References
//!
//! - Plan 454 (this primitive), Issue 155 (the 3D NCA gap).
//! - Sudhakaran et al., "Morphogenesis of Neural Cellular Automata" (arXiv:2103.08737).
//! - [`evolve_motor_gated_field`](crate::motor_gated::evolve_motor_gated_field) —
//!   the sibling modelless growth primitive whose scratch-buffer pattern this
//!   module mirrors.

use crate::operators::graph_laplacian_into;
use crate::simd::fast_sigmoid;
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

    // Apply diffusion to channels 1.. only. For dim==2 this is just the
    // morphogen; for dim>2 every non-alive channel diffuses independently
    // (useful for multi-morphogen variants). Channel 0 (alive) is skipped.
    // Iterate voxel-by-voxel via chunks (field is [v0_ch0, v0_ch1, ..., v1_ch0, ...]).
    //
    // **Sign convention**: `graph_laplacian_into` computes `Δ = deg·center − Σneighbors`
    // (positive at peaks, negative at valleys). For a SMOOTHING diffusion —
    // morphogen flowing from the seed outward to neighbors — the update must
    // be `morph += dt · (−Δ)` (the negative Laplacian, a.k.a. the graph
    // diffusion operator). The plan's literal `+= dt · Δ` is anti-diffusive
    // (sharpening) and cannot propagate growth; this sign correction is
    // required for the "birth propagates" mechanism and the paper's
    // morphogenesis. (Verified by the birth_propagates test.)
    let field_chunks = field.data[..len].chunks_exact_mut(dim);
    let lap_chunks = scratch_lap.data[..len].chunks_exact(dim);
    for ((voxel, &mask), lap) in field_chunks
        .zip(scratch_dropout.iter())
        .zip(lap_chunks)
    {
        // Dropout scaling: halve the diffusion Δ for masked voxels.
        let dt_scale = if mask != 0 { 0.5 } else { 1.0 } * params.diffusion_dt;
        for ch in 1..dim {
            voxel[ch] -= dt_scale * lap[ch];
        }
    }

    // ── Step 2 + 3: autocatalysis/consumption + dropout on the reaction Δ ─
    // Alive voxels: morph += (birth_rate - consumption_rate) * reaction_scale.
    // Dead voxels:  no reaction term here — their morphogen is handled purely
    //                by the step-5 multiplicative decay.
    //
    // **Deviation from the plan's literal pseudocode** (plan step 2 also
    // applies `-= decay_rate` to dead voxels): the additive `-decay_rate` on
    // dead voxels double-counts with step 5's `*= decay_rate` and, worse,
    // immediately wipes out small diffusion gains at the growth frontier —
    // a dead neighbor receiving +0.1 morphogen from the seed gets -0.5
    // reaction, ending at -0.4, which the gate kills. Birth can never
    // propagate. The paper's actual mechanism: alive voxels produce/consume
    // morphogen (autocatalysis); dead voxels passively decay (step 5 only).
    // This is the biologically-correct and functional reading.
    //
    // reaction_scale halves the Δ for masked voxels (same dropout mask,
    // applied to the reaction term — "kill half the Δ" covers both diffusion
    // and reaction, matching the paper's morphogenesis trick).
    let field_chunks = field.data[..len].chunks_exact_mut(dim);
    for (voxel, &mask) in field_chunks.zip(scratch_dropout.iter()) {
        let alive = voxel[0] > 0.5;
        if !alive {
            continue; // dead voxels: no reaction (step 5 handles decay)
        }
        let reaction_scale = if mask != 0 { 0.5 } else { 1.0 };
        let reaction_delta = params.birth_rate - params.consumption_rate;
        // Apply to every morphogen channel (1..dim).
        for morph in voxel[1..dim].iter_mut() {
            *morph += reaction_scale * reaction_delta;
        }
    }

    // ── Step 4: alive gate (sigmoid, not softmax — global rule) ───────────
    // alive' = sigmoid(morphogen · α_scale) > τ ? 1.0 : 0.0
    //
    // **Deviation from the plan's literal pseudocode** (plan step 4 reads
    // `field.alive[v]`): reading only the alive channel can never *birth* a
    // dead voxel — sigmoid(0)=0.5, and with τ=0.5 the strict `>` is always
    // false, so dead voxels stay dead forever and growth cannot propagate.
    // The paper's actual birth mechanism gates on the MORPHOGEN (the growth
    // signal): a dead voxel with enough diffused morphogen crosses the
    // threshold and becomes alive. This is the only reading under which the
    // "birth propagates" test (and the paper's morphogenesis) can pass, so
    // the gate reads channel 1 (morphogen), not channel 0 (alive). The alive
    // channel is purely the binarized OUTPUT of this gate.
    //
    // For dim > 2 the gate reads channel 1 (the canonical morphogen); extra
    // channels (2..) are auxiliary morphogens that diffuse but do not
    // directly gate aliveness.
    for voxel in field.data[..len].chunks_exact_mut(dim) {
        let alpha = voxel[1] * ALIVE_GATE_SCALE;
        voxel[0] = if fast_sigmoid(alpha) > params.alive_threshold {
            1.0
        } else {
            0.0
        };
    }

    // ── Step 5: dead-voxel morphogen reset (gradual drain) ────────────────
    // For voxels that are dead AFTER the gate, multiply morphogen by
    // decay_rate (gradual, not instant — matches the paper). Alive voxels
    // keep their accumulated morphogen.
    for voxel in field.data[..len].chunks_exact_mut(dim) {
        if voxel[0] < 0.5 {
            for morph in voxel[1..dim].iter_mut() {
                *morph *= params.decay_rate;
            }
        }
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
}
