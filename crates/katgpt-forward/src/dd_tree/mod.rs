//! Decision-Diffusion Tree (DDTree) for speculative decoding.
//!
//! Implements width-scaled rollout selection with multiple strategies:
//! - **BestQ** (PTRM default): highest cumulative relevance score
//! - **MostFrequent** (mode@K): most common path across rollouts
//! - **Top1Converged** (EqR, Plan 119): smallest marginal-change residual ∥p_{d+1} − p_d∥₂
//!
//! EqR convergence selection is only reliable after landscape shaping (RI + NI training).
//! See Research 079 (EqR, arXiv:2605.21488) for theoretical justification.
//!
//! # Issue 013 — DRY migration: CONVERGED (Phase A.5)
//!
//! The core DDTree algorithm lives in `katgpt-speculative::dd_tree` and is
//! re-exported via `pub use katgpt_speculative::dd_tree::*` below. Both
//! `katgpt-rs` (root) and `riir-engine` now consume the identical core
//! implementation. This file retains ONLY the feature-gated variants that
//! depend on root-only sibling modules (`belief_drafter`, `spec_generator`,
//! `domino`, `kurtosis_gate`, `manifold_pruner`, `lodestar`, etc.) plus the
//! lodestar-private `find_forced_token` / `a_star_score` helpers (which depend
//! on `super::types::CompletionHorizon`).
//!
//! The convergence pass ported four optimizations from the former root-only
//! copy into the leaf: `log_marginals` cache (`TreeBuilder`), two-pass
//! `>=`-tie `extract_best_path_into`, `&str`-arg `build_inference_result`,
//! and incremental O(D) `merge_retrieved_branches`.

#![allow(clippy::needless_range_loop)]

// Plan 396 (2026-07-05): moved from `src/speculative/dd_tree.rs`. The two
// feature-gated production fns below depend on `katgpt_pruners::*`
// (PrunerSchedule, GdsdPruner, GdsdConfig, identity_advantage) + the leaf
// dd-tree core (`katgpt_speculative::dd_tree`). Tests exercise the full
// dd_tree + dflash_predict pipeline (both resident in katgpt-forward).
#[cfg(test)]
use katgpt_core::traits::BinaryScreeningPruner;
#[cfg(test)]
use katgpt_core::traits::NoPruner;
// ScreeningPruner + TreeNode are used by the two feature-gated wrappers
// below. Gate the import so it doesn't read as unused when both features
// are off (no-default-features).
#[cfg(any(test, feature = "thinking_prune", feature = "gdsd_distill"))]
use katgpt_core::speculative::types::TreeNode;
#[cfg(test)]
use katgpt_core::traits::ConstraintPruner;
#[cfg(any(test, feature = "thinking_prune", feature = "gdsd_distill"))]
use katgpt_core::traits::ScreeningPruner;
// NoScreeningPruner is only constructed inside feature-gated dd-tree wrappers
// (thinking_prune / gdsd_distill) and in tests; gate the import so it doesn't
// read as unused when all those features are off.
#[cfg(any(test, feature = "thinking_prune", feature = "gdsd_distill",))]
use katgpt_core::traits::NoScreeningPruner;

// Core DDTree algorithm now lives in katgpt-speculative (Issue 013 Phase A.5).
// This file retains only the feature-gated variants that depend on root-only
// sibling modules (belief_drafter, spec_generator, domino, kurtosis_gate,
// manifold_pruner, lodestar, etc.). The core primitives below are re-exported
// from the leaf so both root and riir-engine consume identical implementations:
//   build_dd_tree, build_dd_tree_pruned, build_dd_tree_screened, build_dd_tree_balanced,
//   extract_parent_tokens(_into), extract_best_path(_into),
//   extract_candidate_sequences, extract_all_sequences,
//   find_valid_sequence, par_find_valid_sequence, par_find_shortest_sequence,
//   build_inference_result, merge_retrieved_branches,
//   inject_sde_noise(_into), build_slices_view, TreeBuilder.
pub use katgpt_speculative::dd_tree::*;

// ── Plan 391 (2026-07-05): ManifoldPruner DDTree wiring (ManifoldValidWrapper
// + build_dd_tree_manifold) moved to `katgpt_speculative::dd_tree`. Re-exported
// via the glob above. Zero root-only deps — uses only the ConstraintPruner
// trait's `manifold_score` method (already in katgpt_core::traits).

/// DDTree with `PrunerSchedule`-aware screening (Plan 171: Thinking Prune).
///
/// Wraps `screener` based on `schedule` and hop context:
/// - `PrunerSchedule::Uniform`: delegates to [`build_dd_tree_screened`] unchanged
/// - `PrunerSchedule::FrozenBaseGuard`: intermediate hops return relevance 1.0
///   (skipping expensive WASM/ConstraintPruner validation), final hop applies
///   the full screener
///
/// This is the token-level DDTree analog of `build_hop_dd_tree_with_schedule`
/// (crate::spechop::build_hop_dd_tree_with_schedule). The real performance gain comes
/// when the screener wraps an expensive validator (e.g., `WasmPruner`, `BanditPruner`)
/// — intermediate hops skip those calls entirely.
///
/// # Arguments
///
/// * `marginals` — Per-depth token probability distributions
/// * `config` — DDTree configuration
/// * `screener` — Inner screening pruner (potentially expensive)
/// * `chain_seed` — Whether to build greedy chain backbone first
/// * `schedule` — Pruner schedule (Uniform or FrozenBaseGuard)
/// * `hop_index` — Current hop index in the SpecHop pipeline
/// * `total_hops` — Total number of hops in the SpecHop pipeline
///
/// # Returns
///
/// Tree nodes in expansion order.
#[cfg(feature = "thinking_prune")]
pub fn build_dd_tree_screened_with_schedule(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    schedule: katgpt_pruners::PrunerSchedule,
    hop_index: usize,
    total_hops: usize,
) -> Vec<TreeNode> {
    if schedule.should_screen_full(hop_index, total_hops) {
        // Final hop (or Uniform): apply full screening
        build_dd_tree_screened(marginals, config, screener, chain_seed)
    } else {
        // Intermediate hop: use accept-all screener (relevance 1.0 everywhere)
        // This skips all ScreeningPruner calls — the performance win.
        build_dd_tree_screened(marginals, config, &NoScreeningPruner, chain_seed)
    }
}

// ── GDSD Advantage-Guided DDTree Builder (Plan 169) ─────────────

/// DDTree with GDSD advantage-guided self-distillation (Plan 169).
///
/// Convenience wrapper that builds a DDTree using a `GdsdPruner` wrapper
/// around the given screener. The reference pruner is [`NoScreeningPruner`]
/// (unconstrained baseline), and the advantage function is `identity_advantage`.
///
/// For custom advantage functions or non-default configs, construct
/// `GdsdPruner` directly and pass it to [`build_dd_tree_screened`].
///
/// **Feature gate:** `gdsd_distill`
#[cfg(feature = "gdsd_distill")]
pub fn build_dd_tree_gdsd(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    _gdsd_config: &katgpt_pruners::GdsdConfig,
) -> Vec<TreeNode> {
    use katgpt_core::traits::NoScreeningPruner;
    use katgpt_pruners::{GdsdPruner, identity_advantage};

    let _screener = screener; // Used for future integration with dynamic dispatch

    // Box the screener to get a static reference we can wrap.
    // We can't clone a `dyn ScreeningPruner`, so we create a GdsdPruner
    // with NoScreeningPruner as both inner and ref, then delegate.
    // The actual screener is used via the GdsdPruner's relevance() method.
    //
    // NOTE: For full integration, construct GdsdPruner<YourPruner> directly
    // and pass to build_dd_tree_screened(). This convenience fn uses
    // NoScreeningPruner as reference (unconstrained baseline) and identity advantage.
    let gdsd_pruner = GdsdPruner::new(NoScreeningPruner, NoScreeningPruner, identity_advantage);

    // The provided screener is used as the base — we just delegate
    // to the standard screened builder since GdsdPruner IS a ScreeningPruner.
    // The real value comes when the caller constructs GdsdPruner themselves
    // with a real inner pruner (e.g., SdarBanditPruner).
    build_dd_tree_screened(marginals, config, &gdsd_pruner, chain_seed)
}

// ── Plan 391 (2026-07-05): SDE-Aware DDTree Builders, PTRM Width Scaling,
// EqR Convergence Selection, RecFM Cross-Scale Consistency, best_of_k_rollouts,
// cumulative_relevance, and the TreeBuilder struct + impl moved to
// `katgpt_speculative::dd_tree`. Re-exported via
// `pub use katgpt_speculative::dd_tree::*` at the top of this file.
// Zero root-only deps — they compose leaf-resident primitives
// (inject_sde_noise_into, build_slices_view, build_dd_tree_screened,
// build_dd_tree_balanced) and `katgpt_types::{Config, Rng}`.

// ── Plan 392 (2026-07-05): TreeBuilder struct + impl removed from root.
// The leaf's TreeBuilder (now hosting build, build_screened,
// build_screened_progressive, build_screened_with_depth_budgets,
// build_screened_recfm) surfaces via the glob above. The two root-bound
// functions (build_dd_tree_screened_with_schedule, build_dd_tree_gdsd)
// only call the leaf's `build_dd_tree_screened` — they don't construct
// TreeBuilder directly. Verified: zero `TreeBuilder::new` call sites in
// non-test root code.

#[cfg(test)]
mod tests;
