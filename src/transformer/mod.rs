//! Transformer forward-pass variants, generators, raven routing, and quantized paths.
//!
//! Split from the historical monolithic `src/transformer.rs` (Issue 162 C1,
//! 2026-07-17). The public API surface is unchanged — every item that was
//! accessible via `crate::transformer::*` before the split resolves identically
//! after it, via the `pub use` re-exports below and in each sub-module.
//!
//! ## Module layout
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`variants`] | `forward_batched`, `forward_with_domain_latent`, `forward_looped` |
//! | [`tf_loop`] | `forward_training_free_loop`, `depth_route_weights` (+ private helpers) |
//! | [`prefill`] | `forward_prefill` |
//! | [`generators`] | `generate_with_prefill`, `generate_with_collapse_detection`, `generate`, … |
//! | [`paged`] | `forward_paged` |
//! | [`raven`] | `raven_compute_router`, `raven_readout`, `forward_raven`, `tokens_to_string` |
//! | [`quantized`] | `forward_quantized`, `forward_turboquant` |
//! | `tests` | In-module test suite |

use crate::types::*;
use rayon::prelude::*;

// Plan 008 Step 2: substrate types now live in `katgpt-transformer`.
// Re-export so historical `crate::transformer::TransformerWeights` / `KVCache`
// / `MultiLayerKVCache` / `PagedKVCache` / `RavenKVCache` / `PrefillContext`
// / `WallPrefixState` / `GateStatistics` / `MtpProjection` / `load_mtp_projection`
// / `project_target_activation` / `preload_kv_cache` / `ContiguousWeights`
// / `load_ternary_bits` / `load_binary_bits` (Issue 145) / `DecodeStage`
// callers resolve unchanged.
#[cfg(feature = "binary_plasma")]
pub use katgpt_transformer::load_binary_bits;
pub use katgpt_transformer::{
    ContiguousWeights, DecodeStage, GateStatistics, KVCache, KVLayerSnapshot, KVSnapshot,
    LayerWeights, MtpProjection, MultiLayerKVCache, PagedKVCache, PrefillContext, RavenKVCache,
    TransformerWeights, WallPrefixState, load_mtp_projection, load_ternary_bits, preload_kv_cache,
    project_target_activation,
};
// Page size in tokens for PagedKVCache — re-exported so root's tests can drive
// `paged.ensure_pages(0, PAGE_SIZE - 1)` without restating the literal.
pub use katgpt_transformer::PAGE_SIZE;

// ---------------------------------------------------------------------------
// RiM Reasoning Buffer Slots — Plan 172 helpers
// ---------------------------------------------------------------------------

/// Extend a token sequence with RiM reasoning buffer tokens (Plan 172).
/// Appends K×M buffer token IDs after the original prompt tokens.
/// Returns the extended token Vec. No-op when rim is disabled.
#[cfg(feature = "rim_slots")]
pub fn rim_extend_tokens(tokens: &[usize], config: &Config) -> Vec<usize> {
    if !config.rim_enabled() {
        return tokens.to_vec();
    }
    let buf_count = config.rim_total_buffer_tokens();
    let mut extended = Vec::with_capacity(tokens.len() + buf_count);
    extended.extend_from_slice(tokens);
    let buf_token = if config.rim_buffer_token == 0 {
        config.bos_token // fallback to BOS when rim_buffer_token unset
    } else {
        config.rim_buffer_token
    };
    extended.resize(tokens.len() + buf_count, buf_token);
    extended
}

/// Returns the index from which to read logits when RiM buffer slots are active.
/// When enabled, readout is at the LAST buffer position.
/// When disabled, readout is at the last prompt token position.
#[cfg(feature = "rim_slots")]
#[inline]
pub fn rim_readout_index(prompt_len: usize, config: &Config) -> usize {
    if config.rim_enabled() {
        prompt_len + config.rim_total_buffer_tokens() - 1
    } else {
        prompt_len - 1
    }
}

// ---------------------------------------------------------------------------
// ForwardContext + depth_route_with_indices MOVED to `katgpt-forward` crate
// (Issue 007 Phase F, 2026-07-02). The struct was the composition-layer pin —
// it references katgpt-transformer buffer types AND katgpt-pruners handle types
// (CnaModulator/SubstrateMask/HydraSkipPlan), and pruners already depends on
// transformer, so the struct could not live in either leaf without a cycle.
// `katgpt-forward` sits above both. Re-exported here so every historical
// `crate::transformer::ForwardContext` call site resolves unchanged.
//
// Fields are now `pub` in the leaf crate (they were `pub(crate)` in root).
// The forward-pass functions below (forward/forward_looped/forward_batched/…)
// stay in root and access ctx.<field> directly — pub visibility is required for
// the cross-crate access. This is safe: ForwardContext is a pre-allocated
// scratch buffer, not an invariant-guarded type.
// ---------------------------------------------------------------------------
pub use katgpt_forward::ForwardContext;
// `DepthRouteIndicesArgs` + `depth_route_with_indices` are gated behind the
// `delta_routing` feature in `katgpt-forward`. Gate the re-export to match —
// otherwise consumers that depend on katgpt-rs with `default-features = false`
// hit `unresolved import` when the feature is off (Issue 364 T1 wiring hit this).
#[cfg(feature = "delta_routing")]
pub use katgpt_forward::{DepthRouteIndicesArgs, depth_route_with_indices};

// Plan 385 (2026-07-05): forward-pass composition trio + helpers moved to
// katgpt-forward. `forward` is re-exported as pub for historical
// `katgpt_rs::transformer::forward` callers. `forward_base` / `forward_coda`
// are imported privately because root's remaining forward variants
// (`forward_with_domain_latent`, `generate_with_prefill`,
// `generate_with_collapse_detection`) call them. The helpers (`attention_head`,
// `standard_lm_head`, `clustered_lm_head`, `select_topk_indices*`,
// `cluster_map_*`) are also imported because they're called by the remaining
// forward variants AND by tests inside this file. Public re-exports preserve
// the historical API surface (`katgpt_rs::transformer::select_topk_indices`,
// etc.).
//
// Plan 393 (2026-07-05): `forward_decode_stage` + `forward_draft` +
// `forward_verify` also moved to katgpt-forward (they only dispatch to
// `forward_base`). Re-exported below at the `forward_decode_stage` site.
#[cfg(feature = "coda_fusion")]
pub use katgpt_forward::forward_coda;
pub use katgpt_forward::{
    cluster_map_from_embeddings, cluster_map_round_robin, clustered_lm_head, forward, forward_base,
    select_topk_indices, select_topk_indices_into_buf, standard_lm_head,
};
// `attention_head` is `unsafe fn` — re-export publicly for root's other
// forward variants and tests that call it inside `unsafe { ... }` blocks.
pub use katgpt_forward::attention_head;

// ── Stage-specialized forward pass (moved to katgpt-forward, Plan 393) ──
// `forward_decode_stage` + `forward_draft` + `forward_verify` moved to
// `katgpt_forward::forward` because they only dispatch to `forward_base`
// (which has lived there since Plan 385). Re-exported here so
// `crate::transformer::forward_decode_stage` call sites continue to resolve.
#[cfg(feature = "decode_specialize")]
pub use katgpt_forward::forward::forward_decode_stage;

// ---------------------------------------------------------------------------
// Sub-modules (Issue 162 C1 split, 2026-07-17)
// ---------------------------------------------------------------------------

mod generators;
mod variants;
mod paged;
mod prefill;
mod quantized;
mod raven;
mod tf_loop;

// Re-export all public items so `crate::transformer::*` callers resolve
// unchanged. Each sub-module's public API is preserved 1:1.
pub use variants::{forward_batched, forward_looped, forward_with_domain_latent};
pub use generators::{
    generate, generate_batch, generate_into, generate_with_collapse_detection,
    generate_with_prefill, generate_with_prefill_and_domain_latent,
};
pub use paged::forward_paged;
pub use prefill::forward_prefill;
pub use quantized::{forward_quantized, forward_turboquant};
pub use raven::{
    forward_raven, raven_compute_router, raven_compute_router_into, raven_readout,
    raven_readout_into, raven_update, tokens_to_string,
};
pub use tf_loop::{depth_route_weights, forward_training_free_loop};

// `depth_route` is a test-only helper (private impl, exercised by the norm-
// stability test). Re-exported at `pub(crate)` visibility so the in-module test
// suite can call it via `use super::*;` without leaking it into the public API.
// Gated on both `delta_routing` (the feature that compiles the fn) and `test`
// (only tests reference it) so non-test builds never see an unused import.
#[cfg(all(feature = "delta_routing", test))]
pub(crate) use tf_loop::depth_route;

#[cfg(test)]
mod tests;
