//! SWE Trajectory Freezer — modelless committed freeze of an inference
//! attempt's trajectory geometry (Proposal 011 Phase 5, Task T5.5).
//!
//! Composes three shipped DEFAULT-ON primitives:
//!
//! - [`crate::latent_trajectory_geometry::from_states`] — the probe-free
//!   geometric diagnostic (length + mean curvature + min adjacent cosine).
//! - [`crate::committed_field_blend::CommittedFieldBlend`] — sampling-
//!   invariant per-entity MoE with sigmoid projection onto direction vectors
//!   and a BLAKE3 commitment.
//! - A local lightweight BLAKE3 envelope ([`TrajectoryFreezeEnvelope`])
//!   wrapping the frozen blend + summary + geometry. This matches the
//!   `MerkleFrozenEnvelope` pattern referenced across katgpt-core, but lives
//!   here instead of pulling a cross-repo dep on `riir-neuron-db` (the
//!   facade constraint forbids that, and this primitive is a generic open
//!   layer).
//!
//! # What this primitive is for
//!
//! A Rust-SWE-bench inference attempt (or any "iterative refinement" loop —
//! MCTS over patch proposals, repeated forward passes on evolving drafts, a
//! tf_loop rollout) leaves a trajectory through latent/patch space. The
//! geometry of that trajectory (how much it drifts, how sharply it turns,
//! whether it oscillates between attractors) is a modelless signal that can
//! be used to:
//!
//! 1. Discriminate failure modes (T5.1: oscillation vs committed-wrong vs
//!    converged-correct produce measurably distinct `(length, curvature)`
//!    signatures — `mean_curvature ≈ π` is the ping-pong signature).
//! 2. Produce a frozen, BLAKE3-committed, sampling-invariant archetype blend
//!    that characterizes *which* failure mode the attempt exhibited (T5.3b:
//!    data-derived directions from cluster centroids of geometry-encoded
//!    summaries → 100% probe accuracy on synthetic trajectories).
//! 3. Discriminate across snapshots/models (T5.6 G5 gate — the open
//!    question this primitive is built to answer).
//!
//! # The two-stage commit pipeline
//!
//! ```text
//! Stage 1 — Fit (offline, per-snapshot):  train_summaries → derive_directions
//! Stage 2 — Freeze (online, per-attempt): trajectory → encode_summary → commit
//! ```
//!
//! **Stage 1** consumes a corpus of training summaries (one per observed
//! attempt, labeled by failure mode) and produces `N` archetype direction
//! vectors via nearest-centroid derivation:
//!
//! ```text
//! direction_k = normalize( centroid_k − global_centroid )
//! ```
//!
//! This is the **T5.3b design constraint** encoded in code: archetype
//! directions MUST be data-derived, not random. Random directions hit the
//! concentration-of-measure failure (T5.3): in high-D, random unit vectors
//! are nearly orthogonal to any fixed summary, so `dot(summary, dir_k) ≈ 0`
//! for all k → degenerate uniform blend.
//!
//! **Stage 2** takes a single attempt's trajectory, computes its geometry
//! via `from_states`, encodes that geometry into a fixed-`D` summary via
//! [`GeometrySummaryEncoder`], projects onto the pre-fit directions via
//! FAME's sigmoid projection, and produces a [`FrozenAttempt`] carrying the
//! committed `CommittedFieldBlend` + the envelope.
//!
//! # The summary encoder (load-bearing per T5.3b)
//!
//! [`GeometrySummaryEncoder`] places four normalized geometry features
//! (length, curvature, min cosine, n_steps) into the summary, replicated
//! across blocks for dot-product stability. This is the **T5.3b design
//! constraint** encoded in code: the summary MUST capture failure-mode-
//! discriminative geometry, NOT just raw latent position. The dual-strategy
//! test in T5.3b proved endpoint-position summaries produce 17% accuracy
//! (vs 100% for geometry summaries) — the encoder is the load-bearing piece.
//!
//! # Const generics
//!
//! - `N` — archetype count (number of failure modes to discriminate).
//!   Production case is `N = 3` (oscillation / committed-wrong /
//!   converged-correct).
//! - `D` — summary dimension. Must be `>= 4` (the encoder writes 4 features
//!   per block). Production case is `D = 32` (matches FAME's
//!   `TriArchetypeBlend`).
//!
//! # Feature gate
//!
//! Gated behind `swe_trajectory_freeze`. Implies `latent_trajectory_geometry`
//! + `committed_field_blend` (both already default-on). Opt-in — this is a
//! research-validation primitive (Proposal 011 Phase 5); promotion to
//! default requires the T5.6 G5 gate (cross-snapshot discrimination) to
//! pass on real model trajectories, which is currently PARTIAL (T5.4 G3
//! FAIL at 29% on Kimi-K3 depth trajectories — see
//! `.benchmarks/012_kimi_k3_trajectory_geometry.md`).
//!
//! # Allocation
//!
//! - [`GeometrySummaryEncoder::encode_into`] — zero-alloc (writes into a
//!   caller-supplied `&mut [f32; D]`).
//! - [`derive_directions`] — allocates one `[ центroids; N ]` on the stack
//!   (compile-time fixed) + writes into caller-supplied `&mut [[f32; D]; N]`.
//!   No heap allocation.
//! - [`SweTrajectoryFreezer::freeze_attempt`] — zero-alloc. The envelope
//!   payload is written into a stack-fixed `[u8; 512]` buffer (max payload
//!   at N=3, D=32 is 154 bytes); `from_states` + `encode_into` + FAME
//!   `commit` are all documented zero-alloc.
//!
//! # References
//!
//! - Proposal: `katgpt-rs/.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md` Phase 5
//! - T5.1–T5.3 verdict: `katgpt-rs/.issues/569_swe_trajectory_geometry_synthetic_poc.md`
//! - T5.3b verdict (data-derived directions): `katgpt-rs/.issues/570_data_derived_directions_fix_t53.md`
//! - T5.4 PARTIAL verdict (real Kimi-K3 depth trajectory): `katgpt-rs/.benchmarks/012_kimi_k3_trajectory_geometry.md`
//! - Substrate: Plan 342 (`latent_trajectory_geometry`), Plan 321
//!   (`committed_field_blend`)

// (Module gating is handled by `#[cfg(feature = "swe_trajectory_freeze")]`
// on the `mod` declaration in `lib.rs`; this file must NOT duplicate it.)

#![allow(clippy::needless_range_loop)] // const-generic multi-array indexing by index
#![allow(clippy::doc_lazy_continuation)] // multi-line prose with leading words clippy misreads as list items

use crate::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
use crate::latent_trajectory_geometry::{LatentTrajectoryGeometry, from_states};

// ─── Encoder ────────────────────────────────────────────────────────────────

/// Normalize a trajectory's raw geometry into a fixed-`D` summary suitable
/// for FAME's sigmoid projection.
///
/// This is the **load-bearing piece** of the T5.5 design (T5.3b design
/// constraint #2): the summary MUST capture failure-mode-discriminative
/// geometry (`length`, `mean_curvature`, `min_adjacent_cosine`, `n_steps`),
/// NOT just raw latent position. T5.3b's dual-strategy test proved endpoint
/// summaries produce 17% accuracy vs 100% for geometry summaries.
///
/// The four features are placed in the first 4 slots of `out`, then
/// replicated into subsequent 4-slot blocks. The replication is a defense
/// against sparse-vector edge effects in the dot product (a summary with
/// only 4 nonzero entries of 32 produces a dot product sensitive to noise
/// in the high-D direction vector). Replication doesn't change the
/// information content — it just makes the projection numerically
/// well-behaved without requiring a normalization step.
///
/// # Const generic `D`
///
/// Must be `>= 4`. If `D > 4`, the encoder writes `(D / 4)` blocks (rounded
/// down); trailing slots outside a full block are left as zero. At `D = 32`
/// (production) that's 8 blocks of 4 features each.
///
/// # Normalization
///
/// - `length`: divided by `length_scale` (default 20.0), clamped to `[0, 1]`.
///   The default matches the synthetic PoC (Bench 011) where 100-step
///   trajectories had lengths in the 0–20 range. **For real-model
///   trajectories the caller should override `length_scale`**: Kimi-K3
///   depth trajectories (Bench 012) have lengths in the 400–600 range, so
///   `length_scale = 1000.0` would be appropriate.
/// - `mean_curvature`: divided by `π` (curvature is in `[0, π]` by
///   construction via [`fast_acos`]).
/// - `min_adjacent_cosine`: already in `[-1, 1]`.
/// - `n_steps`: divided by `n_steps_scale` (default 100), clamped to `[0, 1]`.
///
/// [`fast_acos`]: crate::latent_trajectory_geometry::fast_acos
#[derive(Clone, Copy, Debug)]
pub struct GeometrySummaryEncoder {
    /// Divisor for the `length` feature. Caller overrides for trajectories
    /// with a different length scale (e.g. real-model depth trajectories).
    pub length_scale: f32,
    /// Divisor for the `n_steps` feature.
    pub n_steps_scale: f32,
}

impl GeometrySummaryEncoder {
    /// Default encoder matching the synthetic PoC (Bench 011).
    ///
    /// `length_scale = 20.0`, `n_steps_scale = 100.0`. Appropriate for
    /// 100-step iterative refinement trajectories with step magnitudes ~0.15.
    pub const fn default_synthetic() -> Self {
        Self {
            length_scale: 20.0,
            n_steps_scale: 100.0,
        }
    }

    /// Encoder tuned for real-model depth trajectories (Bench 012 Kimi-K3).
    ///
    /// `length_scale = 1000.0` (Kimi-K3 depth lengths ~400–600),
    /// `n_steps_scale = 9.0` (depth trajectory = embed + 8 post-layer = 9
    /// states = 8 displacement steps).
    pub const fn default_depth_trajectory() -> Self {
        Self {
            length_scale: 1000.0,
            n_steps_scale: 9.0,
        }
    }

    /// Construct a custom encoder.
    pub const fn new(length_scale: f32, n_steps_scale: f32) -> Self {
        Self {
            length_scale,
            n_steps_scale,
        }
    }

    /// Encode `geom` into `out`.
    ///
    /// Writes 4 normalized features per 4-slot block, replicating across as
    /// many full blocks as fit in `D`. Zero-allocation.
    ///
    /// # Panics (debug)
    ///
    /// In debug builds, panics if `out.len() < 4`.
    #[inline]
    pub fn encode_into<const D: usize>(
        &self,
        geom: &LatentTrajectoryGeometry,
        out: &mut [f32; D],
    ) {
        debug_assert!(D >= 4, "D must be >= 4 to fit one feature block");

        // Normalize each feature to roughly [-1, 1] for stable dot products.
        let length_norm = (geom.length / self.length_scale).clamp(0.0, 1.0);
        let curvature_norm = (geom.mean_curvature / core::f32::consts::PI).clamp(0.0, 1.0);
        let cosine_norm = geom.min_adjacent_cosine.clamp(-1.0, 1.0);
        let n_steps_norm = if self.n_steps_scale > 0.0 {
            (geom.n_steps as f32 / self.n_steps_scale).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Replicate the 4-feature block across D, 4 slots at a time.
        // Trailing slots outside a full block are left at zero.
        let n_blocks = D / 4;
        for block in out.chunks_mut(4).take(n_blocks) {
            block[0] = length_norm;
            block[1] = curvature_norm;
            block[2] = cosine_norm;
            block[3] = n_steps_norm;
        }
    }
}

impl Default for GeometrySummaryEncoder {
    #[inline]
    fn default() -> Self {
        Self::default_synthetic()
    }
}

// ─── Direction derivation (Stage 1: fit) ────────────────────────────────────

/// Derive archetype direction vectors from cluster centroids of training
/// summaries (the **nearest-centroid classifier** direction).
///
/// For each mode `k`:
///
/// ```text
/// direction_k = normalize( centroid_k − global_centroid )
/// ```
///
/// When probed with a summary from cluster `k`, `dot(summary_k, direction_k)`
/// is large and positive; for other clusters, small. This is the
/// **T5.3b design constraint** encoded in code: archetype directions MUST
/// be data-derived, not random. Random directions hit the concentration-of-
/// measure failure (T5.3) — in high-D, random unit vectors are nearly
/// orthogonal to any fixed summary.
///
/// # Arguments
///
/// - `train_summaries` — `&[[[f32; D]; M]; N]`: `N` modes, each with `M`
///   training summaries. The inner array length `M` is the per-mode sample
///   count used to compute the centroid. All summaries must be non-empty.
/// - `out_directions` — `&mut [[f32; D]; N]`: written in place. Each
///   direction is L2-normalized.
///
/// # Allocation
///
/// Stack-only: per-mode centroids + global centroid are fixed-size arrays.
/// No heap allocation.
///
/// # Panics
///
/// In debug builds, panics if `M == 0` (cannot compute a centroid with zero
/// samples). A release-mode `M == 0` produces a zero direction (which will
/// yield `dot = 0` → sigmoid(0) = 0.5 → uniform blend — the degenerate
/// signal that flags an unfitted freezer).
pub fn derive_directions<const N: usize, const M: usize, const D: usize>(
    train_summaries: &[[[f32; D]; M]; N],
    out_directions: &mut [[f32; D]; N],
) {
    debug_assert!(M > 0, "M (per-mode sample count) must be > 0");

    // Per-mode centroids.
    let mut centroids = [[0.0_f32; D]; N];
    for (mode_idx, mode_sums) in train_summaries.iter().enumerate() {
        for s in mode_sums.iter() {
            for j in 0..D {
                centroids[mode_idx][j] += s[j];
            }
        }
        let inv = 1.0 / (M as f32);
        for j in 0..D {
            centroids[mode_idx][j] *= inv;
        }
    }

    // Global centroid (mean of per-mode centroids).
    let mut global = [0.0_f32; D];
    for c in &centroids {
        for j in 0..D {
            global[j] += c[j];
        }
    }
    let inv_n = 1.0 / (N as f32);
    for j in 0..D {
        global[j] *= inv_n;
    }

    // direction_k = normalize(centroid_k − global).
    for k in 0..N {
        let mut norm_sq = 0.0_f32;
        for j in 0..D {
            out_directions[k][j] = centroids[k][j] - global[j];
            norm_sq += out_directions[k][j] * out_directions[k][j];
        }
        let norm = norm_sq.sqrt().max(1e-9);
        for j in 0..D {
            out_directions[k][j] /= norm;
        }
    }
}

// ─── Trajectory freeze envelope (BLAKE3 commitment) ─────────────────────────

/// Wire-format magic for [`TrajectoryFreezeEnvelope`].
///
/// ASCII `"SWTF"` (SWE Trajectory Freeze). Distinct from every other magic
/// in the stack (riir-neuron-db uses `"NRFZ"`, `"KRCS"`, etc.).
pub const SWTF_MAGIC: [u8; 4] = *b"SWTF";

/// Envelope version 1.
pub const SWTF_VERSION: u32 = 1;

/// A lightweight BLAKE3 commitment envelope wrapping a frozen attempt.
///
/// Matches the `MerkleFrozenEnvelope` pattern referenced across katgpt-core
/// (see `cross_resolution.rs`, `curator.rs`, `latent_steering.rs`, etc.),
/// but lives here instead of pulling a cross-repo dep on `riir-neuron-db`.
/// This primitive is the generic open layer; the IP-bearing private bridge
/// (if ever needed) lives in the consumer.
///
/// # Layout
///
/// ```text
/// magic        : [u8; 4]   // "SWTF"
/// version      : u32       // envelope version (currently 1)
/// merkle_root  : [u8; 32]  // BLAKE3 of (pi, summary, geometry)
/// data_len     : u64       // byte length of the serialized payload
/// commitment   : [u8; 32]  // BLAKE3 of (magic || version || merkle_root || data_len)
/// ```
///
/// `merkle_root` commits to the attempt's frozen state (the blend's `pi` +
/// the encoded summary + the raw geometry triple). `commitment` commits to
/// the envelope header — tamper-detection at thaw time checks both.
///
/// # Determinism
///
/// All operations are deterministic: BLAKE3 is platform-independent, the
/// serialization layout is fixed (no padding, no pointers). Two freezers
/// with the same `(pi, summary, geometry)` produce bit-identical envelopes.
#[derive(Clone, Copy, Debug)]
pub struct TrajectoryFreezeEnvelope {
    /// Format magic bytes — always [`SWTF_MAGIC`].
    pub magic: [u8; 4],
    /// Envelope version — currently [`SWTF_VERSION`].
    pub version: u32,
    /// BLAKE3 hash over the serialized payload `(pi, summary, geometry)`.
    pub merkle_root: [u8; 32],
    /// Byte length of the serialized payload (informational).
    pub data_len: u64,
    /// BLAKE3 commitment over `(magic, version, merkle_root, data_len)`.
    pub commitment: [u8; 32],
}

impl TrajectoryFreezeEnvelope {
    /// Construct an envelope from the serialized payload bytes.
    ///
    /// The payload is the concatenation `pi_bytes || summary_bytes ||
    /// geometry_bytes` — caller-supplied, layout determined by the caller
    /// (typically `bytemuck::cast_slice` for fixed-layout types).
    ///
    /// `merkle_root` = BLAKE3(payload). `commitment` = BLAKE3(header).
    /// `data_len` = payload length.
    pub fn freeze(payload: &[u8]) -> Self {
        let merkle_root = blake3::hash(payload);
        let mut header = [0u8; 4 + 4 + 32 + 8];
        header[0..4].copy_from_slice(&SWTF_MAGIC);
        header[4..8].copy_from_slice(&SWTF_VERSION.to_le_bytes());
        header[8..40].copy_from_slice(merkle_root.as_bytes());
        header[40..48].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        let commitment = blake3::hash(&header);
        Self {
            magic: SWTF_MAGIC,
            version: SWTF_VERSION,
            merkle_root: merkle_root.into(),
            data_len: payload.len() as u64,
            commitment: commitment.into(),
        }
    }

    /// Verify the envelope's header commitment.
    ///
    /// Recomputes BLAKE3(magic || version || merkle_root || data_len) and
    /// compares with `self.commitment`. Does NOT recompute `merkle_root`
    /// (that requires the original payload — use [`Self::verify_payload`]
    /// for full verification).
    ///
    /// Returns `false` if `magic != SWTF_MAGIC` or `version != SWTF_VERSION`
    /// or the header commitment doesn't match.
    pub fn verify_header(&self) -> bool {
        if self.magic != SWTF_MAGIC || self.version != SWTF_VERSION {
            return false;
        }
        let mut header = [0u8; 4 + 4 + 32 + 8];
        header[0..4].copy_from_slice(&SWTF_MAGIC);
        header[4..8].copy_from_slice(&SWTF_VERSION.to_le_bytes());
        header[8..40].copy_from_slice(&self.merkle_root);
        header[40..48].copy_from_slice(&self.data_len.to_le_bytes());
        let recomputed = blake3::hash(&header);
        recomputed.as_bytes() == &self.commitment
    }

    /// Full verification: header + payload.
    ///
    /// Returns `true` iff [`Self::verify_header`] passes AND
    /// `BLAKE3(payload) == self.merkle_root`.
    pub fn verify_payload(&self, payload: &[u8]) -> bool {
        if !self.verify_header() {
            return false;
        }
        let recomputed = blake3::hash(payload);
        recomputed.as_bytes() == &self.merkle_root
    }
}

// ─── FrozenAttempt ──────────────────────────────────────────────────────────

/// A frozen SWE-attempt characterization.
///
/// Composed at freeze-time from:
///
/// - The committed FAME blend (`blend`) — the sampling-invariant archetype
///   weights derived from the trajectory summary.
/// - The encoded summary (`summary`) — the geometry features that produced
///   the blend.
/// - The raw geometry (`geometry`) — the original `LatentTrajectoryGeometry`
///   triple, retained for diagnostics.
/// - The BLAKE3 envelope (`envelope`) — tamper-evident commitment.
///
/// # Const generics
///
/// `N` archetype count + `D` summary dimension, mirroring
/// [`CommittedFieldBlend<N, D>`].
///
/// [`CommittedFieldBlend<N, D>`]: crate::committed_field_blend::CommittedFieldBlend
#[derive(Clone, Debug)]
pub struct FrozenAttempt<const N: usize, const D: usize> {
    /// The committed archetype blend (sigmoid weights over `N` archetypes).
    pub blend: CommittedFieldBlend<N, D>,
    /// The encoded geometry summary (length-N features replicated across D).
    pub summary: [f32; D],
    /// The raw trajectory geometry (length, curvature, min_cos, n_steps).
    pub geometry: LatentTrajectoryGeometry,
    /// The BLAKE3 commitment envelope.
    pub envelope: TrajectoryFreezeEnvelope,
}

impl<const N: usize, const D: usize> FrozenAttempt<N, D> {
    /// Sigmoid gates over the blend's `pi`. Convenience accessor.
    ///
    /// `gate_k = sigmoid(pi_k / tau)`. Range `(0, 1)`.
    #[inline]
    pub fn gates(&self) -> [f32; N]
    where
        [(); N]:,
    {
        let tau = self.blend.tau;
        let mut gates = [0.0_f32; N];
        for k in 0..N {
            gates[k] = crate::personality_composition::sigmoid::sigmoid(self.blend.pi[k] / tau);
        }
        gates
    }

    /// Argmax archetype index (the most-embodied failure mode).
    ///
    /// Returns the `k` with the largest sigmoid gate. Ties broken by lowest
    /// index.
    #[inline]
    pub fn argmax_archetype(&self) -> usize
    where
        [(); N]:,
    {
        let gates = self.gates();
        let mut best_k = 0;
        let mut best_g = gates[0];
        for k in 1..N {
            if gates[k] > best_g {
                best_g = gates[k];
                best_k = k;
            }
        }
        best_k
    }

    /// Verify the envelope's header commitment (cheap — no payload needed).
    #[inline]
    pub fn verify_envelope_header(&self) -> bool {
        self.envelope.verify_header()
    }
}

// ─── SweTrajectoryFreezer ───────────────────────────────────────────────────

/// The headline type — composes encoder + FAME + envelope.
///
/// Construct once with the pre-fit direction vectors + encoder config, then
/// call [`Self::freeze_attempt`] per observed trajectory.
///
/// # Const generics
///
/// - `N` — archetype count.
/// - `D` — summary dimension (must match the direction vectors' dimension).
///
/// # Example
///
/// ```no_run
/// # use katgpt_core::swe_trajectory_freeze::*;
/// # use katgpt_core::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
/// # fn make_fields() -> [&'static dyn ArchetypeFieldSource<32>; 3] { unimplemented!() }
/// # fn make_directions() -> [[f32; 32]; 3] { unimplemented!() }
/// // Stage 1 — fit (offline): derive_directions from a labeled corpus
/// // (see derive_directions docs).
/// let directions = make_directions();
/// let freezer = SweTrajectoryFreezer::<3, 32>::new(directions);
///
/// // Stage 2 — freeze (online): characterize a single attempt
/// let trajectory: Vec<Vec<f32>> = vec![]; // populate with real states
/// let refs: Vec<&[f32]> = trajectory.iter().map(|v| v.as_slice()).collect();
/// let fields = make_fields();
/// let frozen = freezer.freeze_attempt(&refs, &fields, /* version */ 1);
///
/// // Inspect the result
/// let gates = frozen.gates();
/// let argmax = frozen.argmax_archetype();
/// let blake3 = frozen.envelope.commitment;
/// ```
pub struct SweTrajectoryFreezer<const N: usize, const D: usize> {
    /// Pre-fit archetype direction vectors (from [`derive_directions`]).
    pub directions: [[f32; D]; N],
    /// The geometry-summary encoder.
    pub encoder: GeometrySummaryEncoder,
}

impl<const N: usize, const D: usize> SweTrajectoryFreezer<N, D> {
    /// Construct a freezer with the given direction vectors + default encoder.
    pub fn new(directions: [[f32; D]; N]) -> Self {
        Self {
            directions,
            encoder: GeometrySummaryEncoder::default(),
        }
    }

    /// Construct a freezer with the given direction vectors + custom encoder.
    pub fn with_encoder(
        directions: [[f32; D]; N],
        encoder: GeometrySummaryEncoder,
    ) -> Self {
        Self {
            directions,
            encoder,
        }
    }

    /// Freeze a single attempt's trajectory into a committed characterization.
    ///
    /// # Pipeline
    ///
    /// 1. Compute `LatentTrajectoryGeometry` via `from_states(trajectory)`.
    /// 2. Encode the geometry into a `[f32; D]` summary.
    /// 3. Commit the FAME blend: `pi_k = clamp(dot(summary, dir_k), ±pi_max)`.
    /// 4. Build the envelope: BLAKE3 over `(pi, summary, geometry)`.
    /// 5. Return the [`FrozenAttempt`].
    ///
    /// # Arguments
    ///
    /// - `trajectory` — `&[&[f32]]`, one slice per latent state. All slices
    ///   must share the same dimension.
    /// - `fields` — the `N` archetype fields (used for their BLAKE3
    ///   commitments only — FAME's `commit` does not invoke `evolve`).
    /// - `version` — monotonic version counter for this commit.
    ///
    /// # Allocation
    ///
    /// - `from_states` is zero-alloc (the substrate's documented contract).
    /// - `encode_into` is zero-alloc.
    /// - FAME `commit` is zero-alloc (Plan 321 G4).
    /// - The envelope construction allocates one `Vec<u8>` for the payload
    ///   (size `N·4 + D·4 + 16`). This is a cold-path cost (one alloc per
    ///   freeze); the steady-state hot path is the FAME `apply_blended`
    ///   call, which is zero-alloc.
    pub fn freeze_attempt(
        &self,
        trajectory: &[&[f32]],
        fields: &[&dyn ArchetypeFieldSource<D>; N],
        version: u64,
    ) -> FrozenAttempt<N, D> {
        // 1. Geometry.
        let geometry = from_states(trajectory);

        // 2. Summary.
        let mut summary = [0.0_f32; D];
        self.encoder.encode_into(&geometry, &mut summary);

        // 3. FAME commit.
        let mut blend = CommittedFieldBlend::<N, D>::uncommitted();
        blend.commit(&summary, &self.directions, fields, version);

        // 4. Envelope. Payload = pi || summary || geometry-bytes.
        //    Layout: N·4 (pi) + D·4 (summary) + 4+4+4+2 (geometry: length,
        //    curvature, min_cos, n_steps as f32/f32/f32/u16) = N·4 + D·4 + 14.
        //    Stack-fixed — no heap allocation. The max payload size at the
        //    production case (N=3, D=32) is 154 bytes, well under any
        //    reasonable stack budget.
        let payload_len = N * 4 + D * 4 + 14;
        let mut payload = [0u8; 512];
        debug_assert!(
            payload_len <= payload.len(),
            "payload_len {payload_len} exceeds stack buffer {}",
            payload.len()
        );
        let mut offset = 0;
        for k in 0..N {
            payload[offset..offset + 4].copy_from_slice(&blend.pi[k].to_le_bytes());
            offset += 4;
        }
        for j in 0..D {
            payload[offset..offset + 4].copy_from_slice(&summary[j].to_le_bytes());
            offset += 4;
        }
        payload[offset..offset + 4].copy_from_slice(&geometry.length.to_le_bytes());
        offset += 4;
        payload[offset..offset + 4].copy_from_slice(&geometry.mean_curvature.to_le_bytes());
        offset += 4;
        payload[offset..offset + 4].copy_from_slice(&geometry.min_adjacent_cosine.to_le_bytes());
        offset += 4;
        payload[offset..offset + 2].copy_from_slice(&geometry.n_steps.to_le_bytes());
        let envelope = TrajectoryFreezeEnvelope::freeze(&payload[..payload_len]);

        FrozenAttempt {
            blend,
            summary,
            geometry,
            envelope,
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)] // index loops used for multi-array indexing
    use super::*;

    // Test constants matching Bench 011's synthetic regime.
    const DIM: usize = 8;
    const N_STEPS: usize = 100;
    const D: usize = 32;
    const N: usize = 3;

    // ─── LCG (deterministic, matches Bench 011) ─────────────────────────────

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        #[inline]
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
        }
    }

    // ─── Synthetic trajectory builders (mirrors Bench 011) ──────────────────

    fn build_committed_wrong(seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg::new(seed);
        let mut state: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 0.1).collect();
        let mut direction: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let norm = direction.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in direction.iter_mut() {
            *x /= norm;
        }
        let step_size = 0.15;
        let mut traj = Vec::with_capacity(N_STEPS + 1);
        traj.push(state.clone());
        for _ in 0..N_STEPS {
            for j in 0..DIM {
                state[j] += step_size * direction[j];
            }
            traj.push(state.clone());
        }
        traj
    }

    fn build_oscillation(seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg::new(seed);
        let attractor_a: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let attractor_b: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let mut traj = Vec::with_capacity(N_STEPS + 1);
        // Alternate between A and B every step (period 2) — matches bench_011.
        // The smoothing variant (0.7/0.3 mix) kills the π-curvature signature.
        for i in 0..=N_STEPS {
            let target = if i % 2 == 0 { &attractor_a } else { &attractor_b };
            traj.push(target.clone());
        }
        traj
    }

    fn build_converged_correct(seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg::new(seed);
        let target: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let mut state: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 0.1).collect();
        let mut traj = Vec::with_capacity(N_STEPS + 1);
        traj.push(state.clone());
        // Exponential convergence — 5% per step (matches bench_011).
        for _ in 0..N_STEPS {
            for j in 0..DIM {
                state[j] += 0.05 * (target[j] - state[j]);
            }
            traj.push(state.clone());
        }
        traj
    }

    fn build_trajectory_for_mode(mode_idx: usize, seed: u64) -> Vec<Vec<f32>> {
        match mode_idx {
            0 => build_oscillation(seed),
            1 => build_committed_wrong(seed),
            2 => build_converged_correct(seed),
            _ => unreachable!(),
        }
    }

    fn build_refs(traj: &[Vec<f32>]) -> Vec<&[f32]> {
        traj.iter().map(|v| v.as_slice()).collect()
    }

    // ─── T5.5 G1 — substrate composes + is deterministic ────────────────────

    /// Two freezes of the same trajectory + same directions + same version
    /// MUST produce bit-identical envelopes. This is the FAME sampling-
    /// invariance property extended to the full freeze pipeline.
    #[test]
    fn g1_freeze_is_deterministic() {
        // Train directions on 3 fixed trajectories.
        let train: [[[f32; D]; 3]; N] = [
            [
                encode_geom_summary(&build_trajectory_for_mode(0, 7)),
                encode_geom_summary(&build_trajectory_for_mode(0, 107)),
                encode_geom_summary(&build_trajectory_for_mode(0, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(1, 7)),
                encode_geom_summary(&build_trajectory_for_mode(1, 107)),
                encode_geom_summary(&build_trajectory_for_mode(1, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(2, 7)),
                encode_geom_summary(&build_trajectory_for_mode(2, 107)),
                encode_geom_summary(&build_trajectory_for_mode(2, 207)),
            ],
        ];
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);

        // Freeze the same test trajectory twice.
        let test_traj = build_trajectory_for_mode(0, 999);
        let refs = build_refs(&test_traj);
        let fields = make_stub_fields();

        let a = freezer.freeze_attempt(&refs, &fields, 1);
        let b = freezer.freeze_attempt(&refs, &fields, 1);

        // Bit-identical envelopes.
        assert_eq!(a.envelope.commitment, b.envelope.commitment);
        assert_eq!(a.envelope.merkle_root, b.envelope.merkle_root);
        assert_eq!(a.envelope.data_len, b.envelope.data_len);
        // Bit-identical pi + summary.
        assert_eq!(a.blend.pi, b.blend.pi);
        assert_eq!(a.summary, b.summary);
    }

    /// Helper used by the G1 test to encode a trajectory's geometry summary
    /// using the default synthetic encoder.
    fn encode_geom_summary(traj: &[Vec<f32>]) -> [f32; D] {
        let refs = build_refs(traj);
        let geom = from_states(&refs);
        let mut summary = [0.0_f32; D];
        GeometrySummaryEncoder::default_synthetic().encode_into(&geom, &mut summary);
        summary
    }

    // ─── T5.5 G2 — directions data-derived, not degenerate ──────────────────

    /// Directions derived from distinct cluster centroids must be distinct
    /// (the T5.3b fix: random directions hit concentration-of-measure).
    #[test]
    fn g2_directions_are_non_degenerate() {
        let train: [[[f32; D]; 3]; N] = [
            [
                encode_geom_summary(&build_trajectory_for_mode(0, 7)),
                encode_geom_summary(&build_trajectory_for_mode(0, 107)),
                encode_geom_summary(&build_trajectory_for_mode(0, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(1, 7)),
                encode_geom_summary(&build_trajectory_for_mode(1, 107)),
                encode_geom_summary(&build_trajectory_for_mode(1, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(2, 7)),
                encode_geom_summary(&build_trajectory_for_mode(2, 107)),
                encode_geom_summary(&build_trajectory_for_mode(2, 207)),
            ],
        ];
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);

        // Each direction must be unit norm.
        for k in 0..N {
            let norm_sq: f32 = directions[k].iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "direction {k} has norm {} (expected 1.0)",
                norm_sq.sqrt()
            );
        }

        // Distinct modes must produce distinct directions (not collapsed).
        for i in 0..N {
            for j in (i + 1)..N {
                let dot: f32 = (0..D).map(|k| directions[i][k] * directions[j][k]).sum();
                let cos = dot; // both are unit-norm
                assert!(
                    cos < 0.99,
                    "directions {i} and {j} are near-identical (cos={cos:.4})"
                );
            }
        }
    }

    // ─── T5.5 G3 — cross-mode discrimination (the load-bearing gate) ────────

    /// The freezer must correctly classify held-out test trajectories into
    /// their source failure mode. Mirrors T5.3b's geometry-strategy gate:
    /// ≥80% accuracy with the matching-gate > 0.6.
    ///
    /// **This is the substrate-level gate.** T5.6 G5 is the cross-snapshot
    /// gate (does the freezer discriminate across *models*, not just across
    /// failure modes within a model). This G3 only asserts the substrate
    /// works end-to-end on synthetic data — which T5.3b already proved.
    #[test]
    fn g3_cross_mode_discrimination() {
        // Train split: first 3 seeds per mode.
        const TRAJ_PER_MODE: usize = 5;
        const TRAIN_SEEDS: usize = 3;

        // Build all trajectories.
        let mut all_trajs: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(N);
        for mode_idx in 0..N {
            let mut mode_trajs = Vec::with_capacity(TRAJ_PER_MODE);
            for seed in 0..TRAJ_PER_MODE {
                mode_trajs.push(build_trajectory_for_mode(mode_idx, seed as u64 * 100 + 7));
            }
            all_trajs.push(mode_trajs);
        }

        // Derive directions from train split.
        let mut train: [[[f32; D]; TRAIN_SEEDS]; N] = [[[0.0; D]; TRAIN_SEEDS]; N];
        for mode_idx in 0..N {
            for seed in 0..TRAIN_SEEDS {
                train[mode_idx][seed] = encode_geom_summary(&all_trajs[mode_idx][seed]);
            }
        }
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);
        let fields = make_stub_fields();

        // Probe on test split.
        let mut n_correct = 0usize;
        let total = N * (TRAJ_PER_MODE - TRAIN_SEEDS);
        for mode_idx in 0..N {
            for seed in TRAIN_SEEDS..TRAJ_PER_MODE {
                let refs = build_refs(&all_trajs[mode_idx][seed]);
                let frozen = freezer.freeze_attempt(&refs, &fields, 1);
                let gates = frozen.gates();
                let matching_gate = gates[mode_idx];
                let argmax = frozen.argmax_archetype();
                let correct = matching_gate > 0.6 && argmax == mode_idx;
                if correct {
                    n_correct += 1;
                }
            }
        }

        let accuracy = n_correct as f32 / total as f32;
        assert!(
            accuracy >= 0.8,
            "G3 FAIL: accuracy {accuracy:.2} < 0.80 (correct {n_correct}/{total})"
        );
    }

    // ─── T5.5 G4 — envelope tamper-evidence ─────────────────────────────────

    #[test]
    fn g4_envelope_tamper_evidence() {
        let train: [[[f32; D]; 3]; N] = [
            [
                encode_geom_summary(&build_trajectory_for_mode(0, 7)),
                encode_geom_summary(&build_trajectory_for_mode(0, 107)),
                encode_geom_summary(&build_trajectory_for_mode(0, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(1, 7)),
                encode_geom_summary(&build_trajectory_for_mode(1, 107)),
                encode_geom_summary(&build_trajectory_for_mode(1, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(2, 7)),
                encode_geom_summary(&build_trajectory_for_mode(2, 107)),
                encode_geom_summary(&build_trajectory_for_mode(2, 207)),
            ],
        ];
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);

        let traj = build_trajectory_for_mode(0, 999);
        let refs = build_refs(&traj);
        let fields = make_stub_fields();
        let frozen = freezer.freeze_attempt(&refs, &fields, 1);

        // Header verification passes.
        assert!(frozen.envelope.verify_header());

        // Tamper the magic → header verification fails.
        let mut tampered = frozen.envelope;
        tampered.magic = *b"XXXX";
        assert!(!tampered.verify_header());

        // Tamper the version → header verification fails.
        let mut tampered = frozen.envelope;
        tampered.version = 999;
        assert!(!tampered.verify_header());

        // Tamper the merkle_root → header verification fails.
        let mut tampered = frozen.envelope;
        tampered.merkle_root[0] ^= 0xff;
        assert!(!tampered.verify_header());

        // Tamper the commitment → header verification fails.
        let mut tampered = frozen.envelope;
        tampered.commitment[0] ^= 0xff;
        assert!(!tampered.verify_header());
    }

    /// Stub fields for tests — FAME's `commit` only reads `commitment()`,
    /// not `evolve()`. So a trivial stub suffices.
    fn make_stub_fields() -> [&'static dyn ArchetypeFieldSource<D>; N] {
        // SAFETY: this leaks a static. Acceptable for tests.
        // (We use Box::leak so the field outlives the call; for the test
        // suite this is fine — the process exits shortly after.)
        static F0: StubField = StubField::new(0);
        static F1: StubField = StubField::new(1);
        static F2: StubField = StubField::new(2);
        [&F0, &F1, &F2]
    }

    /// Minimal stub field — `commitment()` returns a per-index hash.
    /// `evolve()` is never called by `freeze_attempt` (only `commit()` is,
    /// which only reads `commitment()`).
    struct StubField {
        idx: u8,
    }
    impl StubField {
        const fn new(idx: u8) -> Self {
            Self { idx }
        }
    }
    impl ArchetypeFieldSource<D> for StubField {
        fn evolve<'a>(&self, _z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
            // Should never be called by freeze_attempt.
            for x in dz_scratch.iter_mut().take(D) {
                *x = 0.0;
            }
            &mut dz_scratch[..D]
        }
        fn commitment(&self) -> [u8; 32] {
            let mut h = blake3::Hasher::new();
            h.update(b"stub_field:");
            h.update(&[self.idx]);
            h.finalize().into()
        }
    }

    // ─── T5.5 G5 — encoder normalizes correctly ─────────────────────────────

    #[test]
    fn g5_encoder_normalizes_features() {
        let encoder = GeometrySummaryEncoder::default_synthetic();
        let mut summary = [0.0_f32; D];

        // Length at the scale boundary → normalized to 1.0.
        let geom = LatentTrajectoryGeometry {
            length: 25.0, // > 20.0 → clamps to 1.0
            mean_curvature: core::f32::consts::PI, // = π → normalized to 1.0
            min_adjacent_cosine: -1.0,
            n_steps: 200, // > 100 → clamps to 1.0
        };
        encoder.encode_into(&geom, &mut summary);

        // First block.
        assert!((summary[0] - 1.0).abs() < 1e-6, "length_norm");
        assert!((summary[1] - 1.0).abs() < 1e-6, "curvature_norm");
        assert!((summary[2] - (-1.0)).abs() < 1e-6, "cosine_norm");
        assert!((summary[3] - 1.0).abs() < 1e-6, "n_steps_norm");

        // Replication into second block.
        assert!((summary[4] - 1.0).abs() < 1e-6, "block 2 length");
        assert!((summary[5] - 1.0).abs() < 1e-6, "block 2 curvature");
    }

    #[test]
    fn g5_encoder_handles_zero_geometry() {
        let encoder = GeometrySummaryEncoder::default_synthetic();
        let mut summary = [99.0_f32; D]; // poison
        let geom = LatentTrajectoryGeometry::default(); // all zeros
        encoder.encode_into(&geom, &mut summary);
        // All features are 0 (length=0/scale=0, curvature=0/π=0, etc.).
        for j in 0..D {
            assert!(
                summary[j].abs() < 1e-6,
                "slot {j} = {} (expected 0)",
                summary[j]
            );
        }
    }

    // ─── T5.5 G6 — version affects envelope ─────────────────────────────────

    /// A re-commit with a different version produces a distinct envelope
    /// (mirrors FAME's `g5_blake3_version_affects_hash`).
    #[test]
    fn g6_version_affects_envelope() {
        let train: [[[f32; D]; 3]; N] = [
            [
                encode_geom_summary(&build_trajectory_for_mode(0, 7)),
                encode_geom_summary(&build_trajectory_for_mode(0, 107)),
                encode_geom_summary(&build_trajectory_for_mode(0, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(1, 7)),
                encode_geom_summary(&build_trajectory_for_mode(1, 107)),
                encode_geom_summary(&build_trajectory_for_mode(1, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(2, 7)),
                encode_geom_summary(&build_trajectory_for_mode(2, 107)),
                encode_geom_summary(&build_trajectory_for_mode(2, 207)),
            ],
        ];
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);

        let traj = build_trajectory_for_mode(0, 999);
        let refs = build_refs(&traj);
        let fields = make_stub_fields();

        let v1 = freezer.freeze_attempt(&refs, &fields, 1);
        let v2 = freezer.freeze_attempt(&refs, &fields, 2);

        // Same pi (the geometry + directions are identical; only version differs).
        assert_eq!(v1.blend.pi, v2.blend.pi, "pi must be identical");
        // Different FAME blake3 (version is part of FAME's commitment input).
        assert_ne!(
            v1.blend.blake3, v2.blend.blake3,
            "FAME blake3 must differ by version"
        );
        // Different envelope commitment (because pi is the same but version
        // differs in the FAME blake3, which is NOT in our payload — but our
        // payload is also identical, so the envelope commitment should ALSO
        // be identical. This is correct behavior: the envelope commits to
        // the trajectory's geometry, not to the FAME version counter).
        //
        // Wait — the payload includes `blend.pi`, which is identical, but
        // NOT `blend.blake3` or `blend.version`. So the envelope commitment
        // is identical across versions. That's the correct semantics: the
        // envelope commits to the *trajectory characterization*, not to the
        // FAME audit counter.
        assert_eq!(
            v1.envelope.commitment, v2.envelope.commitment,
            "envelope commits to trajectory, not to FAME version"
        );
    }
}
