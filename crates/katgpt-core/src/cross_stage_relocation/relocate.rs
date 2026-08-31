//! Cross-Stage Residual Relocation Operator (Plan 431 Phase 2, Research 417).
//!
//! The applied (risky) half of the Knowing-Using Gap distillation. Where the
//! [`crate::cross_stage_relocation`] diagnostic scans all stage pairs to
//! *locate* stranded representations, this module ships the
//! *operator* that performs the relocation: snapshot an anchor token's
//! residual state at one stage and overwrite at another during a forward pass.
//!
//! # The paper's fixed two-pair heuristic
//!
//! arXiv:2607.08393 §5.5 shows that two predetermined layer pairs per
//! architecture — `(⌊0.82L⌉ → ⌊0.45L⌉)` and `(⌊0.10L⌉ → ⌊0.45L⌉)` — recover
//! 58–75% of oracle headroom across 6 models × 2 domains, with no per-instance
//! search. [`RelocatePair::LateEarly`] encodes this default.
//!
//! # Promotion gate
//!
//! The 58–75% recovery is a quality claim on the paper's LLM substrate. Our
//! substrate (latent functors, HLA, neuron shards) does not have the same
//! early/late MLP structure, so the transfer MUST be PoC-verified (Plan 431
//! Phase 3 in `riir-ai/crates/riir-poc/`) before any promotion to
//! default-on. Until then this module ships opt-in diagnostic-only.

/// A single cross-stage residual relocation.
///
/// During a forward pass with this op active, the anchor token's residual
/// state at `src_stage` is snapshotted and overwrites the anchor's residual
/// state at `dst_stage`. The host's forward pass must implement
/// [`RelocatingForward`] to expose the snapshot/overwrite hooks.
///
/// # Fields
///
/// - `src_stage` — the stage to snapshot from (paper's "source layer").
/// - `dst_stage` — the stage to overwrite at (paper's "target layer").
/// - `anchor_token_idx` — which token in the sequence is the anchor (the
///   paper's "head-entity position" — the entity whose representation is
///   stranded).
///
/// No new sync-boundary data: `src_stage` / `dst_stage` / `anchor_token_idx`
/// are configuration `usize`s, not gameplay state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelocateOp {
    pub src_stage: usize,
    pub dst_stage: usize,
    pub anchor_token_idx: usize,
}

impl RelocateOp {
    /// Apply this relocation to a host implementing [`RelocatingForward`].
    ///
    /// Orchestrates: snapshot the anchor's state at `src_stage` into
    /// `scratch`, then overwrite the anchor's state at `dst_stage` from
    /// `scratch`. The host owns the actual forward-pass machinery; this
    /// method only coordinates the snapshot+overwrite via the trait hooks.
    ///
    /// # Zero-alloc guarantee
    ///
    /// The snapshot is written into the caller-supplied `scratch` buffer; no
    /// `Vec` growth occurs. `scratch.len()` must be `>=` the host's per-stage
    /// residual width (typically the model's hidden dim).
    ///
    /// # Panics
    ///
    /// Panics if `scratch.len()` is smaller than the host's residual width,
    /// or if `src_stage` / `dst_stage` are `>= host.n_stages()`. The host's
    /// `snapshot_anchor_at` / `overwrite_anchor_at` define the precise
    /// behavior; this method only forwards the indices.
    #[inline]
    pub fn apply_into<F: RelocatingForward + ?Sized>(&self, host: &mut F, scratch: &mut [f32]) {
        // Snapshot src → scratch, then scratch → dst. Two memcpys; the host
        // owns the forward-pass machinery and the actual residual buffers.
        // This method orchestrates only.
        host.snapshot_anchor_at(self.src_stage, self.anchor_token_idx, scratch);
        host.overwrite_anchor_at(self.dst_stage, self.anchor_token_idx, scratch);
    }
}

/// The paper's fixed two-pair default + a custom variant.
///
/// arXiv:2607.08393 §5.5 shows that `(⌊0.82L⌉ → ⌊0.45L⌉) +
/// (⌊0.10L⌉ → ⌊0.45L⌉)` recovers 58–75% of oracle headroom. The
/// [`RelocatePair::LateEarly`] variant encodes this default;
/// [`RelocatePair::Custom`] lets the caller specify arbitrary source
/// fractions and a shared destination fraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RelocatePair {
    /// The paper's two-pair default (§5.5): `(⌊0.82L⌉, ⌊0.45L⌉) +
    /// (⌊0.10L⌉, ⌊0.45L⌉)`. Recovers 58–75% of oracle headroom on the KU Gap
    /// benchmark across 6 models × 2 domains.
    LateEarly,
    /// Custom pair — caller specifies both source fractions and the shared
    /// destination fraction. Fractions are in `[0, 1]`; `to_ops` rounds to
    /// the nearest stage index.
    Custom { src_a: f32, src_b: f32, dst: f32 },
}

impl RelocatePair {
    /// Convert to the two [`RelocateOp`]s for a stack of `n_stages` stages.
    ///
    /// Uses round-to-nearest on the fractional stage indices (paper §5.5
    /// notation `⌊·⌉`). For `n_stages == 0` or `1`, returns ops that index
    /// stage 0 (degenerate — the host will typically reject via
    /// [`RelocatingForward::n_stages`]).
    ///
    /// The two ops share `anchor_token_idx`; the caller may apply them
    /// sequentially via [`RelocateOp::apply_into`].
    pub fn to_ops(&self, n_stages: usize, anchor_token_idx: usize) -> [RelocateOp; 2] {
        let (src_a_frac, src_b_frac, dst_frac) = match self {
            Self::LateEarly => (0.82, 0.10, 0.45),
            Self::Custom { src_a, src_b, dst } => (*src_a, *src_b, *dst),
        };
        let dst_stage = frac_to_stage(dst_frac, n_stages);
        [
            RelocateOp {
                src_stage: frac_to_stage(src_a_frac, n_stages),
                dst_stage,
                anchor_token_idx,
            },
            RelocateOp {
                src_stage: frac_to_stage(src_b_frac, n_stages),
                dst_stage,
                anchor_token_idx,
            },
        ]
    }
}

impl Default for RelocatePair {
    /// Defaults to the paper's [`RelocatePair::LateEarly`] heuristic.
    fn default() -> Self {
        RelocatePair::LateEarly
    }
}

/// Host's forward pass with snapshot/overwrite hooks.
///
/// The primitive itself does NOT own forward-pass machinery — same contract
/// as Plan 358's `direct_effect_importance` caller-supplied closure pattern.
/// The host (typically a transformer-style forward in riir-engine /
/// riir-games) implements this trait to expose the snapshot/overwrite hooks
/// that [`RelocateOp::apply_into`] coordinates.
///
/// # Local-only data
///
/// The anchor's residual state at any stage is **local-only (latent, never
/// synced)** — it does not cross `SyncBlock → ChainConsensus`. This trait
/// operates on local activation buffers only.
pub trait RelocatingForward {
    /// Snapshot the anchor's residual state at `stage` into `out`.
    ///
    /// `out.len()` must be `>=` the host's per-stage residual width.
    /// Implementations should `out[..width].copy_from_slice(...)` and leave
    /// the tail untouched (or zero it — caller-agnostic).
    fn snapshot_anchor_at(&self, stage: usize, anchor_idx: usize, out: &mut [f32]);

    /// Overwrite the anchor's residual state at `stage` with `state`.
    ///
    /// `state.len()` must match the host's per-stage residual width. The
    /// overwrite is a pure substitution (no blending) — the paper's
    /// self-patching is a hard overwrite, not a residual mix. A blended
    /// variant would be a host-side responsibility (this trait stays
    /// minimal).
    fn overwrite_anchor_at(&mut self, stage: usize, anchor_idx: usize, state: &[f32]);

    /// Number of stages in the forward pass (e.g., n_layers for an LLM).
    fn n_stages(&self) -> usize;
}

/// Map a fractional position `frac ∈ [0, 1]` to a stage index for a stack of
/// `n` stages, using round-to-nearest (paper §5.5 notation `⌊·⌉`).
///
/// The paper writes `⌊fL⌉` for a model with `L` layers; we compute
/// `round(frac * n)` and clamp to `[0, n-1]`. This is the literal reading
/// of the paper's formula (for `L=10`, `⌊0.82L⌉ = round(8.2) = 8`).
///
/// - `frac = 0.0` → stage 0 (earliest).
/// - `frac = 1.0` → stage `n-1` (latest; `round(n)` clamps to `n-1`).
/// - `n == 0` → 0 (degenerate; the host will reject via `n_stages`).
/// - `n == 1` → 0 (only one stage).
#[inline]
fn frac_to_stage(frac: f32, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let scaled = (frac.clamp(0.0, 1.0) * n as f32).round();
    // round() can produce n on frac=1.0 fp noise; clamp to n-1.
    (scaled as usize).min(n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── frac_to_stage ──────────────────────────────────────────────────────

    #[test]
    fn frac_to_stage_paper_examples() {
        // Paper §5.5 notation ⌊fL⌉ = round(f * L), computed for L=10:
        //   ⌊0.82 * 10⌉ = ⌊8.2⌉ = 8
        //   ⌊0.10 * 10⌉ = ⌊1.0⌉ = 1
        //   ⌊0.45 * 10⌉ = ⌊4.5⌉ = 5 (Rust rounds half away from zero)
        assert_eq!(frac_to_stage(0.82, 10), 8);
        assert_eq!(frac_to_stage(0.10, 10), 1);
        assert_eq!(frac_to_stage(0.45, 10), 5);
    }

    #[test]
    fn frac_to_stage_endpoints() {
        assert_eq!(frac_to_stage(0.0, 10), 0);
        assert_eq!(frac_to_stage(1.0, 10), 9);
    }

    #[test]
    fn frac_to_stage_clamps_overflow() {
        // frac > 1 clamps to n-1.
        assert_eq!(frac_to_stage(1.5, 10), 9);
        assert_eq!(frac_to_stage(2.0, 10), 9);
    }

    #[test]
    fn frac_to_stage_clamps_negative() {
        assert_eq!(frac_to_stage(-0.5, 10), 0);
    }

    #[test]
    fn frac_to_stage_degenerate() {
        assert_eq!(frac_to_stage(0.5, 0), 0);
        assert_eq!(frac_to_stage(0.5, 1), 0);
    }

    // ── RelocatePair::to_ops ───────────────────────────────────────────────

    #[test]
    fn late_early_to_ops_10_stages() {
        let [op_a, op_b] = RelocatePair::LateEarly.to_ops(10, /* anchor */ 3);
        // (0.82, 0.45): src=⌊8.2⌉=8, dst=⌊4.5⌉=5
        assert_eq!(op_a.src_stage, 8);
        assert_eq!(op_a.dst_stage, 5);
        assert_eq!(op_a.anchor_token_idx, 3);
        // (0.10, 0.45): src=⌊1.0⌉=1, dst=⌊4.5⌉=5
        assert_eq!(op_b.src_stage, 1);
        assert_eq!(op_b.dst_stage, 5);
        assert_eq!(op_b.anchor_token_idx, 3);
    }

    #[test]
    fn late_early_to_ops_shares_dst() {
        // Both ops must target the same dst_stage (paper's shared ~0.45L).
        let [op_a, op_b] = RelocatePair::LateEarly.to_ops(20, 0);
        assert_eq!(op_a.dst_stage, op_b.dst_stage);
    }

    #[test]
    fn custom_to_ops() {
        let pair = RelocatePair::Custom {
            src_a: 0.5,
            src_b: 0.25,
            dst: 0.75,
        };
        let [op_a, op_b] = pair.to_ops(8, 1);
        // 0.5 * 7 = 3.5 → round = 4 (round half up); actually 3.5 rounds to 4
        // in rust's .round() (rounds half away from zero).
        assert_eq!(op_a.src_stage, frac_to_stage(0.5, 8));
        assert_eq!(op_b.src_stage, frac_to_stage(0.25, 8));
        assert_eq!(op_a.dst_stage, frac_to_stage(0.75, 8));
        assert_eq!(op_b.dst_stage, op_a.dst_stage);
        assert_eq!(op_a.anchor_token_idx, 1);
    }

    #[test]
    fn default_is_late_early() {
        assert_eq!(RelocatePair::default(), RelocatePair::LateEarly);
    }

    // ── RelocateOp::apply_into (synthetic 4-stage host) ────────────────────

    /// Synthetic 4-stage residual stream where stage 1 holds the answer but
    /// the readout (stage 3) reads from stage 2 (empty). This is the
    /// canonical "stranded representation" toy domain from Plan 431 T2.5.
    ///
    /// Layout: `residuals[stage][token * width + dim]`. Each stage has its
    /// own buffer; the host's "forward pass" is just reading from a chosen
    /// stage (we don't model attention/MLP — only the relocation mechanic).
    #[derive(Debug)]
    #[allow(dead_code)]
    struct StrandedHost {
        residuals: Vec<Vec<f32>>,
        n_tokens: usize,
        width: usize,
    }

    impl StrandedHost {
        fn new(n_stages: usize, n_tokens: usize, width: usize) -> Self {
            Self {
                residuals: vec![vec![0.0; n_tokens * width]; n_stages],
                n_tokens,
                width,
            }
        }

        fn set(&mut self, stage: usize, token: usize, values: &[f32]) {
            let base = token * self.width;
            self.residuals[stage][base..base + self.width].copy_from_slice(values);
        }

        fn read(&self, stage: usize, token: usize) -> Vec<f32> {
            let base = token * self.width;
            self.residuals[stage][base..base + self.width].to_vec()
        }
    }

    impl RelocatingForward for StrandedHost {
        fn snapshot_anchor_at(&self, stage: usize, anchor_idx: usize, out: &mut [f32]) {
            let base = anchor_idx * self.width;
            let src = &self.residuals[stage][base..base + self.width];
            let w = self.width.min(out.len());
            out[..w].copy_from_slice(&src[..w]);
        }

        fn overwrite_anchor_at(&mut self, stage: usize, anchor_idx: usize, state: &[f32]) {
            let base = anchor_idx * self.width;
            let dst = &mut self.residuals[stage][base..base + self.width];
            let w = self.width.min(state.len());
            dst[..w].copy_from_slice(&state[..w]);
        }

        fn n_stages(&self) -> usize {
            self.residuals.len()
        }
    }

    #[test]
    fn relocate_recovers_stranded_representation() {
        // 4-stage host, 1 token, width 4. Stage 1 holds the answer
        // [1.0, 0.5, -0.5, 0.25]; stages 0, 2, 3 are zero. The readout
        // (stage 3) reads zero. Relocate {src:1, dst:2} recovers it into
        // stage 2; then a further relocate {src:2, dst:3} would recover it
        // into the readout. This test verifies the snapshot+overwrite
        // mechanic on the simpler single-hop case.
        let mut host = StrandedHost::new(4, 1, 4);
        let answer = [1.0, 0.5, -0.5, 0.25];
        host.set(1, 0, &answer);

        // Sanity: stage 2 is zero before the relocate.
        assert!(host.read(2, 0).iter().all(|&v| v == 0.0));

        // Apply the relocation: snapshot stage 1, overwrite stage 2.
        let op = RelocateOp {
            src_stage: 1,
            dst_stage: 2,
            anchor_token_idx: 0,
        };
        let mut scratch = vec![0.0; 4];
        op.apply_into(&mut host, &mut scratch);

        // Stage 2 now holds the answer (recovered from stage 1).
        let recovered = host.read(2, 0);
        for (i, (&a, &b)) in answer.iter().zip(recovered.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "dim {i}: expected {a}, got {b}");
        }

        // Stage 1 is unchanged (snapshot, not move).
        let still_there = host.read(1, 0);
        for (i, (&a, &b)) in answer.iter().zip(still_there.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "stage 1 dim {i}: expected {a}, got {b}"
            );
        }
    }

    #[test]
    fn relocate_chain_reaches_readout() {
        // Two-hop relocate: stage 1 → stage 2 → stage 3. Verifies that the
        // operator composes (each apply_into is independent; the host's
        // state carries forward).
        let mut host = StrandedHost::new(4, 1, 2);
        let answer = [0.7, -0.3];
        host.set(1, 0, &answer);

        let mut scratch = vec![0.0; 2];

        // Hop 1: 1 → 2
        RelocateOp {
            src_stage: 1,
            dst_stage: 2,
            anchor_token_idx: 0,
        }
        .apply_into(&mut host, &mut scratch);

        // Hop 2: 2 → 3
        RelocateOp {
            src_stage: 2,
            dst_stage: 3,
            anchor_token_idx: 0,
        }
        .apply_into(&mut host, &mut scratch);

        let final_state = host.read(3, 0);
        for (i, (&a, &b)) in answer.iter().zip(final_state.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "readout dim {i}: expected {a}, got {b}"
            );
        }
    }

    #[test]
    fn relocate_late_early_pair_recovers() {
        // Apply both LateEarly ops in sequence on a 10-stage host. Stage 8
        // holds the answer; after both ops, stage 4 (the shared dst) holds
        // the answer (twice-overwritten, bit-identical).
        let n_stages = 10;
        let mut host = StrandedHost::new(n_stages, 1, 3);
        let answer = [0.1, 0.2, 0.3];
        host.set(8, 0, &answer); // late src (0.82L = stage 8)
        host.set(1, 0, &answer); // early src (0.10L = stage 1)

        let [op_a, op_b] = RelocatePair::LateEarly.to_ops(n_stages, 0);
        let mut scratch = vec![0.0; 3];
        op_a.apply_into(&mut host, &mut scratch);
        op_b.apply_into(&mut host, &mut scratch);

        // Both ops target dst_stage 4 (0.45L on L=10 → stage 4).
        let dst = host.read(op_a.dst_stage, 0);
        for (i, (&a, &b)) in answer.iter().zip(dst.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "dst dim {i}: expected {a}, got {b}");
        }
    }

    #[test]
    fn relocate_is_zero_alloc() {
        // The apply_into path uses only the caller-supplied scratch buffer;
        // no Vec growth inside the operator. We verify by running many
        // applies on a fixed-size host + scratch and checking no allocation
        // occurs (the trait impls themselves are pure memcpy).
        let mut host = StrandedHost::new(4, 2, 4);
        host.set(0, 0, &[1.0; 4]);
        let mut scratch = vec![0.0; 4];

        // Run 100 applies; if any allocated, the Vec would have grown (but
        // we can't directly assert no allocation here — the G4 alloc-free
        // gate is the dedicated benchmark). This test verifies correctness
        // across many iterations.
        for _ in 0..100 {
            RelocateOp {
                src_stage: 0,
                dst_stage: 1,
                anchor_token_idx: 0,
            }
            .apply_into(&mut host, &mut scratch);
        }
        // After 100 applies, stage 1 token 0 still holds [1.0; 4].
        let v = host.read(1, 0);
        assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn relocate_uses_scratch_not_internal_buffer() {
        // Verify the snapshot goes through `scratch`, not an internal host
        // buffer. We do this by checking that scratch holds the snapshot
        // after apply_into (the host impl writes into scratch first, then
        // reads from scratch to overwrite).
        let mut host = StrandedHost::new(2, 1, 3);
        host.set(0, 0, &[0.5, 0.6, 0.7]);
        let mut scratch = vec![-1.0; 3];

        RelocateOp {
            src_stage: 0,
            dst_stage: 1,
            anchor_token_idx: 0,
        }
        .apply_into(&mut host, &mut scratch);

        // scratch should now hold the snapshot of stage 0.
        assert_eq!(scratch, vec![0.5, 0.6, 0.7]);
    }
}
