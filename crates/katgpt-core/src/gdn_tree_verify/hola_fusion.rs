//! Dual-Path Rollback-Free Tree Verification — GDN Recurrent State × HOLA
//! Hippocampal Cache (Plan 430, Research 407 §2.2).
//!
//! Fuses Plan 424 (GDN rollback-free tree verify) with Plan 395 (HOLA
//! hippocampal exact KV cache) into a **dual-path tree verifier** that scores
//! speculative draft tree nodes against BOTH the GDN recurrent state (via
//! masked triangular solve) AND the HOLA hippocampal cache (via ancestor-masked
//! softmax read) — with **zero rollback on either path**.
//!
//! # Why
//!
//! GDN2's fixed-size recurrent state compresses context but loses exact
//! long-range recall. HOLA's hippocampal cache recovers exact recall for
//! high-surprise tokens but has no tree-verification story. Fusing them at the
//! speculative tree-verify layer gives both exact-recall recovery (HOLA) AND
//! rollback-free tree scoring (GDN masked solve) — the hippocampal complement
//! to the compressive recurrent state, extended to branching speculative drafts.
//!
//! # Algorithm
//!
//! For each node `i` in topological order:
//!
//! 1. **GDN path** (Plan 424): masked triangular solve produces `O_gdn[i]`.
//! 2. **HOLA path** (this module): build `block_kv_i` = ancestor `(k_j, v_j)`
//!    pairs from the tree, then `cache.read_cache_into_fast_block(q_i, block_kv_i)`
//!    → `O_hola[i]`. Ancestor masking is **by construction** — only ancestor
//!    tokens are passed.
//! 3. **Residual-add**: `O[i] = O_gdn[i] + O_hola[i]` (HOLA §3.5 — the cache
//!    *complements* the recurrent state, not replaces it).
//!
//! Both paths are read-only. Zero rollback.
//!
//! # Design: ancestor masking by construction, not by bitmask
//!
//! The GDN masked solve needs explicit ancestor bitmasks because the recurrent
//! state couples all positions multiplicatively (global linear system). The
//! HOLA cache read is a local softmax attention — ancestor masking is enforced
//! by only passing ancestor tokens in `block_kv_i`. No bitmask needed on the
//! cache path. This asymmetry is correct: global solve vs local attention.

use super::{
    GdnLayerParams, GdnMultiHeadParams, GdnTreeVerifier, TreeTopology, verify_gdn_tree_into,
};
use crate::hippocampal_cache_dyn::HippocampalCacheDyn;
use crate::types::rmsnorm_with_gamma;

// ── Dual-path verifier (pre-allocated scratch) ─────────────────

/// Pre-allocated scratch buffers for the dual-path tree verifier.
///
/// Extends [`GdnTreeVerifier`] with a scratch buffer for the HOLA output and a
/// reusable ancestor-path buffer for building `block_kv` per node.
///
/// The hot path (`verify_gdn_hola_tree_into`) performs **zero heap allocations**
/// after construction (G4 gate): the `block_kv_ptrs` buffer is pre-sized to
/// `max_depth` and reused across all nodes via `clear()` + `push()` within
/// capacity. The raw pointers are safe because they are derived from the
/// caller's params slices (which outlive each verify call) and cleared before
/// each node's use.
pub struct GdnHolaTreeVerifier {
    /// Inner GDN verifier (Plan 424 scratch).
    inner: GdnTreeVerifier,
    /// HOLA output buffer: `[max_t × d_v]` (topo-indexed).
    scratch_o_hola: Vec<f32>,
    /// Raw (k_ptr, v_ptr) pairs for building block_kv without lifetime issues.
    /// All ancestor slices have the same length `d` (= d_k = d_v), so we only
    /// store pointers (not lengths). Cleared + refilled per node.
    /// Safe because the params slices outlive each verify call.
    block_kv_ptrs: Vec<(*const f32, *const f32)>,
    /// Typed block_kv buffer for the cache read call. Pre-allocated to
    /// `max_depth`; reused across all nodes via `clear()` + `push()` within
    /// capacity. This is the G4 (alloc-free) fix: avoids per-node Vec allocation.
    block_kv_typed: Vec<(&'static [f32], &'static [f32])>,
    /// Reusable ancestor path buffer (topo indices, root first).
    /// Pre-sized to `max_depth`; `clear()` + `push()` within capacity is alloc-free.
    ancestor_path_scratch: Vec<usize>,
    /// Pre-normalized tree keys `[max_t × d]`, topo-indexed. Computed once per
    /// `verify_gdn_hola_tree_into` call using the cache's gamma. Avoids
    /// redundant per-node RMSNorm of ancestor keys (a chain tree at T=N
    /// normalizes the root key N times without this). The G2 perf fix.
    tree_keys_norm: Vec<f32>,
    /// `max_t` from construction (for bounds checking).
    max_t: usize,
}

impl GdnHolaTreeVerifier {
    /// Construct a verifier sized for trees up to `max_t` nodes with head
    /// dimensions `d_k` (key/query) and `d_v` (value), and max tree depth
    /// `max_depth` (for the ancestor path buffer).
    ///
    /// `max_depth` should be the maximum number of ancestors any node can have
    /// (= tree depth). For typical speculative trees this is ≤ 16.
    pub fn new(max_t: usize, _d_k: usize, d_v: usize, max_depth: usize) -> Self {
        Self {
            inner: GdnTreeVerifier::new(max_t, _d_k, d_v),
            scratch_o_hola: vec![0.0; max_t.saturating_mul(d_v)],
            block_kv_ptrs: Vec::with_capacity(max_depth.max(1)),
            block_kv_typed: Vec::with_capacity(max_depth.max(1)),
            ancestor_path_scratch: Vec::with_capacity(max_depth.max(1)),
            tree_keys_norm: vec![0.0; max_t.saturating_mul(d_v)],
            max_t,
        }
    }

    /// Access the inner GDN-only verifier (for G3 no-regression: when the cache
    /// is disabled, callers can fall back to the Plan 424 API directly).
    pub fn inner(&mut self) -> &mut GdnTreeVerifier {
        &mut self.inner
    }
}

// ── HOLA cache read path (T1.3) ────────────────────────────────

/// Compute the HOLA hippocampal output for each tree node.
///
/// For each node `i` (topo order), builds `block_kv_i` = ancestor `(k_j, v_j)`
/// pairs from the tree, then calls `cache.read_cache_into_fast_block(q_i, block_kv_i)`
/// → `o_hola[i]`.
///
/// Writes to `o_hola_buf[0..t*d_v]` (topo-indexed). Uses `block_kv_ptrs` and
/// `ancestor_path_scratch` — **zero heap allocation** after construction.
///
/// # Arguments
/// * `o_hola_buf` — output buffer `[max_t × d_v]`, only `[0..t*d_v]` written.
/// * `block_kv_ptrs` — raw (k_ptr, v_ptr) pairs, cleared + refilled per node.
/// * `ancestor_path_scratch` — topo indices of ancestors, cleared + refilled per node.
/// * `tree_keys_norm` — pre-normalized tree keys buffer `[max_t × d]`,
///   topo-indexed. Filled once at the start of this call using the cache's
///   gamma. Avoids redundant per-node RMSNorm of ancestor keys.
/// * `topo` — tree topology.
/// * `params` — GDN layer params (original-indexed; used for keys/values/queries).
/// * `cache` — the per-layer HOLA cache (read-only; `&mut` for scratch only).
/// * `d` — head dimension (d_k = d_v = cache.d).
#[allow(clippy::too_many_arguments)]
fn compute_out_hola(
    o_hola_buf: &mut [f32],
    block_kv_ptrs: &mut Vec<(*const f32, *const f32)>,
    block_kv_typed: &mut Vec<(&'static [f32], &'static [f32])>,
    ancestor_path_scratch: &mut Vec<usize>,
    tree_keys_norm: &mut [f32],
    topo: &TreeTopology,
    params: &GdnLayerParams,
    cache: &mut HippocampalCacheDyn,
    d: usize,
) {
    let t = topo.n_nodes;

    // Pre-normalize all tree keys once using the cache's gamma. This avoids
    // redundant per-node RMSNorm of ancestor keys: a chain tree at T=N would
    // normalize the root key N times without this. The pre-normalized keys are
    // topo-indexed (node i's key is at tree_keys_norm[i*d..(i+1)*d]).
    let gamma = cache.gamma();
    for i in 0..t {
        let orig_i = topo.topo_order[i];
        let k_src = &params.keys[orig_i * d..(orig_i + 1) * d];
        let k_dst = &mut tree_keys_norm[i * d..(i + 1) * d];
        k_dst.copy_from_slice(k_src);
        rmsnorm_with_gamma(k_dst, gamma);
    }

    for i in 0..t {
        let orig_i = topo.topo_order[i];
        let q_i = &params.queries[orig_i * d..(orig_i + 1) * d];

        // Build ancestor path (topo indices, root first) by walking parent chain.
        ancestor_path_scratch.clear();
        let mut cur = i;
        while cur != usize::MAX {
            ancestor_path_scratch.push(cur);
            cur = topo.parent[cur];
        }
        ancestor_path_scratch.reverse();

        // Collect raw pointers: keys from tree_keys_norm (pre-normalized,
        // topo-indexed), values from params (orig-indexed).
        // Safe because tree_keys_norm and params slices outlive this function
        // call and we clear() before each use.
        block_kv_ptrs.clear();
        for &ancestor_k in ancestor_path_scratch.iter() {
            let k_norm_j = &tree_keys_norm[ancestor_k * d..(ancestor_k + 1) * d];
            let orig_j = topo.topo_order[ancestor_k];
            let v_j = &params.values[orig_j * d..(orig_j + 1) * d];
            block_kv_ptrs.push((k_norm_j.as_ptr(), v_j.as_ptr()));
        }

        // Reconstruct typed slice references from raw pointers for the cache read.
        // Safe: pointers come from tree_keys_norm / params which outlive this call;
        // all slices have length d. The typed buffer is pre-allocated in the verifier
        // struct and reused via clear() + push() — zero allocation.
        block_kv_typed.clear();
        for &(k_ptr, v_ptr) in block_kv_ptrs.iter() {
            // SAFETY: k_ptr points into tree_keys_norm; v_ptr into params.values.
            // Both outlive this call; length d is correct (all slices are d-wide).
            let k_slice = unsafe { core::slice::from_raw_parts(k_ptr, d) };
            let v_slice = unsafe { core::slice::from_raw_parts(v_ptr, d) };
            block_kv_typed.push((k_slice, v_slice));
        }

        // Read cache into o_hola[i] using the pre-normalized block_kv path.
        let out_i = i * d;
        let o_hola_i = &mut o_hola_buf[out_i..out_i + d];
        cache.read_cache_into_fast_block_prenorm(q_i, block_kv_typed, o_hola_i);
    }
}

// ── Top-level dual-path API (T1.4) ─────────────────────────────

/// Verify a speculative draft tree using the dual-path (GDN + HOLA) algorithm.
///
/// Convenience wrapper — allocates the output `Vec`. For the zero-alloc hot
/// path, use [`verify_gdn_hola_tree_into`].
///
/// # Returns
/// Per-node outputs `O`: `T × d_v`, row-major, **topo-indexed** (row k = topo node k).
/// `O[i] = O_gdn[i] + O_hola[i]` (residual-add complement, HOLA §3.5).
pub fn verify_gdn_hola_tree(
    verifier: &mut GdnHolaTreeVerifier,
    topo: &TreeTopology,
    params: &GdnLayerParams,
    cache: &mut HippocampalCacheDyn,
    s0: &[f32],
    d_k: usize,
    d_v: usize,
) -> Vec<f32> {
    let out = verify_gdn_hola_tree_into(verifier, topo, params, cache, s0, d_k, d_v);
    out.to_vec()
}

/// Zero-alloc dual-path verify. Returns a reference to the verifier's internal
/// output buffer (topo-indexed). The reference is valid until the next
/// `verify_gdn_hola_tree*` call on the same verifier.
///
/// # Algorithm
/// 1. GDN path: `verify_gdn_tree_into` (Plan 424) → `O_gdn[i]`.
/// 2. HOLA path: `compute_out_hola` → `O_hola[i]`.
/// 3. Residual-add: `O[i] = O_gdn[i] + O_hola[i]`.
///
/// Both paths are read-only. Zero rollback.
///
/// # Constraint
/// `d_k` must equal `d_v` and both must equal the cache's head dimension `d`.
/// The HOLA cache operates on a single head dim (D = d_k = d_v).
pub fn verify_gdn_hola_tree_into<'a>(
    verifier: &'a mut GdnHolaTreeVerifier,
    topo: &TreeTopology,
    params: &GdnLayerParams,
    cache: &mut HippocampalCacheDyn,
    s0: &[f32],
    d_k: usize,
    d_v: usize,
) -> &'a [f32] {
    debug_assert_eq!(
        d_k, d_v,
        "dual-path requires d_k == d_v (HOLA cache uses single head dim)"
    );
    debug_assert_eq!(cache.d(), d_k, "cache head dim must equal d_k");
    let t = topo.n_nodes;
    debug_assert!(
        t <= verifier.max_t,
        "tree size {t} > max_t {}",
        verifier.max_t
    );

    // Step 1: GDN path (Plan 424 masked triangular solve) → writes to inner scratch_out.
    verify_gdn_tree_into(&mut verifier.inner, topo, params, s0, d_k, d_v);

    // Step 2: HOLA path → writes to scratch_o_hola.
    let d = d_k;
    {
        let o_hola = &mut verifier.scratch_o_hola[..t * d];
        let block_kv_ptrs = &mut verifier.block_kv_ptrs;
        let block_kv_typed = &mut verifier.block_kv_typed;
        let ancestor_path = &mut verifier.ancestor_path_scratch;
        let tree_keys_norm = &mut verifier.tree_keys_norm[..t * d];
        compute_out_hola(
            o_hola,
            block_kv_ptrs,
            block_kv_typed,
            ancestor_path,
            tree_keys_norm,
            topo,
            params,
            cache,
            d,
        );
    }

    // Step 3: residual-add O[i] = O_gdn[i] + O_hola[i] into the inner scratch_out.
    for idx in 0..t * d {
        verifier.inner.scratch_out[idx] += verifier.scratch_o_hola[idx];
    }

    &verifier.inner.scratch_out[..t * d]
}

// ── Dual-path commit (T2.1) ────────────────────────────────────

/// Commit the accepted path for both the GDN state AND the HOLA cache.
///
/// # GDN commit
/// Replay the delta-rule recurrence along the accepted path. Updates `S₀` in place.
///
/// # HOLA commit
/// For each token on the accepted path, call `cache.observe(k_t, v_t, beta_t,
/// residual_norm_t)`. The `(beta_t, residual_norm_t)` come from the GDN delta-rule
/// update at each step — `beta_t` is `params.betas[t]`, and `residual_norm_t` is
/// `‖v_t − α_t · k_tᵀS_{t-1}‖` (the delta-rule residual before the write).
///
/// Both commits are append-only / in-place-update — no rollback needed.
///
/// # Arguments
/// * `topo` — tree topology.
/// * `accepted_leaf` — **topo** index of the accepted leaf node.
/// * `params` — GDN layer params (original-indexed).
/// * `s0` — committed prefix state `[d_k × d_v]`, updated in place.
/// * `cache` — per-layer HOLA cache, updated in place (observe high-surprise tokens).
pub fn commit_accepted_dual(
    topo: &TreeTopology,
    accepted_leaf: usize,
    params: &GdnLayerParams,
    s0: &mut [f32],
    cache: &mut HippocampalCacheDyn,
    d_k: usize,
    d_v: usize,
) {
    // Reconstruct the path from root to accepted_leaf (topo indices, root first).
    let mut path: Vec<usize> = Vec::with_capacity(topo.n_nodes);
    let mut cur = accepted_leaf;
    while cur != usize::MAX {
        path.push(cur);
        cur = topo.parent[cur];
    }
    path.reverse();

    commit_path_dual(topo, &path, params, s0, cache, d_k, d_v);
}

/// Replay the delta-rule recurrence along a given path AND observe tokens into
/// the HOLA cache.
///
/// Updates `S₀` in place (GDN delta-rule) and observes each token into the cache
/// (HOLA hippocampal). The residual `‖e‖ = ‖v − α·kᵀS_{prev}‖` is computed for
/// free during the GDN commit — piped to the cache as a side effect.
#[allow(clippy::too_many_arguments)]
pub fn commit_path_dual(
    topo: &TreeTopology,
    path: &[usize],
    params: &GdnLayerParams,
    s0: &mut [f32],
    cache: &mut HippocampalCacheDyn,
    d_k: usize,
    d_v: usize,
) {
    let mut residual = vec![0.0f32; d_v];

    for &node_k in path {
        let orig = topo.topo_order[node_k];
        let k = &params.keys[orig * d_k..(orig + 1) * d_k];
        let v = &params.values[orig * d_v..(orig + 1) * d_v];
        let alpha = params.alphas[orig];
        let beta = params.betas[orig];

        // r = kᵀS (before decay): r[d] = Σ_m k[m] · S[m*d_v + d]
        residual.fill(0.0);
        for m in 0..d_k {
            let km = k[m];
            if km != 0.0 {
                for d in 0..d_v {
                    residual[d] += s0[m * d_v + d] * km;
                }
            }
        }

        // Compute the residual norm ‖v − α·r‖ BEFORE the state update.
        let mut residual_norm_sq = 0.0f32;
        for d in 0..d_v {
            let e_d = v[d] - alpha * residual[d];
            residual_norm_sq += e_d * e_d;
        }
        let residual_norm = residual_norm_sq.sqrt();

        // Observe into the HOLA cache: score = beta · ‖e‖.
        cache.observe(k, v, beta, residual_norm);

        // S = α · S (decay).
        for val in s0[..d_k * d_v].iter_mut() {
            *val *= alpha;
        }
        // S += β · k ⊗ (v − α·r).
        for m in 0..d_k {
            let beta_km = beta * k[m];
            if beta_km != 0.0 {
                for d in 0..d_v {
                    s0[m * d_v + d] += beta_km * (v[d] - alpha * residual[d]);
                }
            }
        }
    }
}

// ── Multi-head dual-path (T2.3) ────────────────────────────────

/// GDN multi-head params + per-head HOLA caches.
///
/// Mirrors [`GdnMultiHeadParams`] but carries one cache per KV head. The cache
/// is shared across query heads within a KV head group (paper form: one cache
/// per layer, not per Q head). For standard MHA (H_k == H_v), each head has its
/// own cache.
pub struct GdnHolaMultiHeadParams<'a> {
    /// The GDN multi-head params.
    pub gdn: GdnMultiHeadParams<'a>,
    /// Per-KV-head HOLA caches. Length must equal `gdn.n_kv_heads`.
    pub caches: &'a mut [&'a mut HippocampalCacheDyn],
}

/// Verify a speculative draft tree using the dual-path algorithm for all heads.
///
/// Loops over heads, reusing the verifier's scratch buffers. Returns a `Vec` of
/// `[H * T * d_v]` (head-major, topo-indexed within each head).
///
/// `s0_per_head[h]` is the committed prefix state for head h (not modified — use
/// [`commit_accepted_dual_multihead`] to write back).
pub fn verify_gdn_hola_tree_multihead(
    verifier: &mut GdnHolaTreeVerifier,
    topo: &TreeTopology,
    params: &mut GdnHolaMultiHeadParams,
    s0_per_head: &[&[f32]],
    d_k: usize,
    d_v: usize,
) -> Vec<f32> {
    let t = topo.n_nodes;
    let h = params.gdn.n_kv_heads;
    debug_assert_eq!(params.caches.len(), h, "caches.len must equal n_kv_heads");
    let mut out = vec![0.0f32; h * t * d_v];
    for head in 0..h {
        let hp = params.gdn.head_params(head, t, d_k, d_v);
        let s0 = s0_per_head[head];
        let cache = &mut params.caches[head];
        let head_out = verify_gdn_hola_tree_into(verifier, topo, &hp, cache, s0, d_k, d_v);
        out[head * t * d_v..(head + 1) * t * d_v].copy_from_slice(head_out);
    }
    out
}

/// Commit the accepted path for all heads: replay the delta-rule + observe into
/// cache for each head's S₀ and cache, updating in place.
#[allow(clippy::too_many_arguments)]
pub fn commit_accepted_dual_multihead(
    topo: &TreeTopology,
    accepted_leaf: usize,
    params: &GdnMultiHeadParams,
    s0_per_head: &mut [&mut [f32]],
    caches: &mut [&mut HippocampalCacheDyn],
    d_k: usize,
    d_v: usize,
) {
    let t = topo.n_nodes;
    debug_assert_eq!(s0_per_head.len(), params.n_kv_heads);
    debug_assert_eq!(caches.len(), params.n_kv_heads);
    for (head, (s0, cache)) in s0_per_head.iter_mut().zip(caches.iter_mut()).enumerate() {
        let hp = params.head_params(head, t, d_k, d_v);
        commit_accepted_dual(topo, accepted_leaf, &hp, s0, cache, d_k, d_v);
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn_tree_verify::{GdnLayerParams, build_topology};

    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32) / (u32::MAX as f32) * 2.0 - 1.0
        }
    }

    // ── T1.5: chain tree matches sequential GDN+HOLA ──

    #[test]
    fn test_dual_path_chain_matches_sequential() {
        // Chain tree: 0 ← 1 ← 2 ← ... ← T-1 (linear).
        let (t, d) = (8, 16);
        let parents: Vec<usize> = (0..t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut rng = rng(42);
        let keys: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let values: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let queries: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.8 + 0.15 * rng()).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.5 + 0.4 * rng()).collect();
        let s0: Vec<f32> = (0..d * d).map(|_| 0.1 * rng()).collect();

        // Populate a HOLA cache with some prior context (not from the tree).
        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(d, 32);
        for _ in 0..16 {
            let k_prior: Vec<f32> = (0..d).map(|_| rng()).collect();
            let v_prior: Vec<f32> = (0..d).map(|_| rng()).collect();
            cache.observe(&k_prior, &v_prior, 0.5, 0.5 + 0.5 * rng().abs());
        }

        let params = GdnLayerParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
        };
        let topo = build_topology(&parents, &alphas);
        let max_depth = t; // chain has depth t-1
        let mut verifier = GdnHolaTreeVerifier::new(t, d, d, max_depth);

        let dual_out =
            verify_gdn_hola_tree_into(&mut verifier, &topo, &params, &mut cache, &s0, d, d);

        // Reference: GDN-only output (Plan 424) + per-node HOLA read.
        let gdn_ref = {
            let mut verifier_gdn = GdnTreeVerifier::new(t, d, d);
            verify_gdn_tree_into(&mut verifier_gdn, &topo, &params, &s0, d, d).to_vec()
        };

        let mut hola_ref = vec![0.0f32; t * d];
        let mut ancestor_path: Vec<usize> = Vec::with_capacity(t);
        let mut block_kv: Vec<(&[f32], &[f32])> = Vec::with_capacity(t);
        for i in 0..t {
            ancestor_path.clear();
            let mut cur = i;
            while cur != usize::MAX {
                ancestor_path.push(cur);
                cur = topo.parent[cur];
            }
            ancestor_path.reverse();

            block_kv.clear();
            for &ak in &ancestor_path {
                let orig_j = topo.topo_order[ak];
                block_kv.push((
                    &keys[orig_j * d..(orig_j + 1) * d],
                    &values[orig_j * d..(orig_j + 1) * d],
                ));
            }

            let orig_i = topo.topo_order[i];
            let q_i = &queries[orig_i * d..(orig_i + 1) * d];
            cache.read_cache_into_fast_block(q_i, &block_kv, &mut hola_ref[i * d..(i + 1) * d]);
        }

        // Dual = GDN + HOLA (residual-add).
        let tol = 1e-3f32;
        let mut max_err = 0.0f32;
        for i in 0..t * d {
            let expected = gdn_ref[i] + hola_ref[i];
            max_err = max_err.max((dual_out[i] - expected).abs());
        }
        assert!(
            max_err < tol,
            "dual-path chain: max error {max_err:.6} >= {tol}"
        );
    }

    // ── T2.2: dual-path commit matches sequential ──

    #[test]
    fn test_dual_path_commit_matches_sequential() {
        let (t, d) = (6, 8);
        let parents: Vec<usize> = (0..t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut rng = rng(99);
        let keys: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let values: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let queries: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.8 + 0.15 * rng()).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.5 + 0.4 * rng()).collect();
        let s0_init: Vec<f32> = (0..d * d).map(|_| 0.1 * rng()).collect();

        let topo = build_topology(&parents, &alphas);

        let params = GdnLayerParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
        };

        // Dual commit: path = root to last node.
        let mut s0_dual = s0_init.clone();
        let mut cache_dual = HippocampalCacheDyn::new_with_ones_gamma(d, 64);
        let accepted_leaf = t - 1;
        commit_accepted_dual(
            &topo,
            accepted_leaf,
            &params,
            &mut s0_dual,
            &mut cache_dual,
            d,
            d,
        );

        // Reference: sequential GDN2 commit + HOLA observe.
        let mut s0_ref = s0_init.clone();
        let mut cache_ref = HippocampalCacheDyn::new_with_ones_gamma(d, 64);
        let mut residual = vec![0.0f32; d];
        for node in 0..t {
            let k = &keys[node * d..(node + 1) * d];
            let v = &values[node * d..(node + 1) * d];
            let alpha = alphas[node];
            let beta = betas[node];
            residual.fill(0.0);
            for m in 0..d {
                for dd in 0..d {
                    residual[dd] += s0_ref[m * d + dd] * k[m];
                }
            }
            let mut res_sq = 0.0f32;
            for dd in 0..d {
                let e = v[dd] - alpha * residual[dd];
                res_sq += e * e;
            }
            let res_norm = res_sq.sqrt();
            cache_ref.observe(k, v, beta, res_norm);
            for val in s0_ref[..d * d].iter_mut() {
                *val *= alpha;
            }
            for m in 0..d {
                let beta_km = beta * k[m];
                for dd in 0..d {
                    s0_ref[m * d + dd] += beta_km * (v[dd] - alpha * residual[dd]);
                }
            }
        }

        // Compare S₀ states.
        let tol = 1e-4f32;
        let mut max_s_err = 0.0f32;
        for i in 0..d * d {
            max_s_err = max_s_err.max((s0_dual[i] - s0_ref[i]).abs());
        }
        assert!(
            max_s_err < tol,
            "S₀ mismatch after dual commit: {max_s_err:.6}"
        );

        // Compare cache: same tokens observed with same scores → same cache contents.
        assert_eq!(cache_dual.len(), cache_ref.len(), "cache length mismatch");
        let (min_dual, min_ref) = (cache_dual.min_score(), cache_ref.min_score());
        assert!(
            min_dual.is_some() && min_ref.is_some(),
            "caches should not be empty after commit"
        );
        let (min_dual, min_ref) = (min_dual.unwrap(), min_ref.unwrap());
        assert!(
            (min_dual - min_ref).abs() < tol,
            "cache min_score mismatch: dual={min_dual:.6} ref={min_ref:.6}"
        );
    }

    // ── T4.3 (G3): GDN component is unperturbed by the HOLA path ──

    #[test]
    fn test_dual_path_gdn_component_unperturbed() {
        // G3 no-regression: the GDN component of dual-path verify must be
        // byte-identical to standalone GDN verify. The HOLA path is a pure
        // addition — it does not modify S₀ or the GDN scratch.
        //
        // (The compile-time G3 gate — "with hippocampal_cache OFF, the dual-path
        // function doesn't exist" — is trivially true by the feature flag.)
        let (t, d) = (4, 8);
        let parents: Vec<usize> = (0..t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut rng = rng(7);
        let keys: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let values: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let queries: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.9).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.7).collect();
        let s0: Vec<f32> = (0..d * d).map(|_| 0.1 * rng()).collect();

        let params = GdnLayerParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
        };
        let topo = build_topology(&parents, &alphas);

        // GDN-only reference.
        let gdn_ref = {
            let mut v = GdnTreeVerifier::new(t, d, d);
            verify_gdn_tree_into(&mut v, &topo, &params, &s0, d, d).to_vec()
        };

        // Dual-path with populated cache.
        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(d, 32);
        for _ in 0..8 {
            let kp: Vec<f32> = (0..d).map(|_| rng()).collect();
            let vp: Vec<f32> = (0..d).map(|_| rng()).collect();
            cache.observe(&kp, &vp, 0.6, 0.4 + 0.6 * rng().abs());
        }
        let mut verifier = GdnHolaTreeVerifier::new(t, d, d, t);
        let dual_out =
            verify_gdn_hola_tree_into(&mut verifier, &topo, &params, &mut cache, &s0, d, d);

        // Compute expected HOLA contribution per node.
        let mut hola_ref = vec![0.0f32; t * d];
        let mut ancestor_path: Vec<usize> = Vec::with_capacity(t);
        let mut block_kv: Vec<(&[f32], &[f32])> = Vec::with_capacity(t);
        for i in 0..t {
            ancestor_path.clear();
            let mut cur = i;
            while cur != usize::MAX {
                ancestor_path.push(cur);
                cur = topo.parent[cur];
            }
            ancestor_path.reverse();
            block_kv.clear();
            for &ak in &ancestor_path {
                let orig_j = topo.topo_order[ak];
                block_kv.push((
                    &keys[orig_j * d..(orig_j + 1) * d],
                    &values[orig_j * d..(orig_j + 1) * d],
                ));
            }
            let orig_i = topo.topo_order[i];
            let q_i = &queries[orig_i * d..(orig_i + 1) * d];
            cache.read_cache_into_fast_block(q_i, &block_kv, &mut hola_ref[i * d..(i + 1) * d]);
        }

        // G3: dual = GDN + HOLA. If GDN is perturbed, this fails.
        let tol = 1e-4f32;
        let mut max_err = 0.0f32;
        for i in 0..t * d {
            let expected = gdn_ref[i] + hola_ref[i];
            max_err = max_err.max((dual_out[i] - expected).abs());
        }
        assert!(
            max_err < tol,
            "GDN component perturbed: max_err={max_err:.6}"
        );
    }

    // ── T4.1 (G1): correctness on random branching trees ──

    #[test]
    fn test_dual_path_random_tree_correctness() {
        // Branching tree:
        //         0 (root)
        //        / \
        //       1   2
        //      / \   \
        //     3   4   5
        //    /
        //   6
        let parents = [usize::MAX, 0, 0, 1, 1, 2, 3];
        let t = parents.len();
        let d = 8;
        let mut rng = rng(2024);
        let keys: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let values: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let queries: Vec<f32> = (0..t * d).map(|_| rng()).collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.85).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.6).collect();
        let s0: Vec<f32> = (0..d * d).map(|_| 0.1 * rng()).collect();

        // Populate cache with prior context.
        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(d, 32);
        for _ in 0..12 {
            let kp: Vec<f32> = (0..d).map(|_| rng()).collect();
            let vp: Vec<f32> = (0..d).map(|_| rng()).collect();
            cache.observe(&kp, &vp, 0.5, 0.3 + 0.7 * rng().abs());
        }

        let params = GdnLayerParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
        };
        let topo = build_topology(&parents, &alphas);

        let mut verifier = GdnHolaTreeVerifier::new(t, d, d, t);
        let dual_out =
            verify_gdn_hola_tree_into(&mut verifier, &topo, &params, &mut cache, &s0, d, d);

        // Reference: GDN-only + per-node HOLA read.
        let gdn_ref = {
            let mut v = GdnTreeVerifier::new(t, d, d);
            verify_gdn_tree_into(&mut v, &topo, &params, &s0, d, d).to_vec()
        };

        let mut hola_ref = vec![0.0f32; t * d];
        let mut ancestor_path: Vec<usize> = Vec::with_capacity(t);
        let mut block_kv: Vec<(&[f32], &[f32])> = Vec::with_capacity(t);
        for i in 0..t {
            ancestor_path.clear();
            let mut cur = i;
            while cur != usize::MAX {
                ancestor_path.push(cur);
                cur = topo.parent[cur];
            }
            ancestor_path.reverse();

            block_kv.clear();
            for &ak in &ancestor_path {
                let orig_j = topo.topo_order[ak];
                block_kv.push((
                    &keys[orig_j * d..(orig_j + 1) * d],
                    &values[orig_j * d..(orig_j + 1) * d],
                ));
            }

            let orig_i = topo.topo_order[i];
            let q_i = &queries[orig_i * d..(orig_i + 1) * d];
            cache.read_cache_into_fast_block(q_i, &block_kv, &mut hola_ref[i * d..(i + 1) * d]);
        }

        let tol = 1e-3f32;
        let mut max_err = 0.0f32;
        for i in 0..t * d {
            let expected = gdn_ref[i] + hola_ref[i];
            max_err = max_err.max((dual_out[i] - expected).abs());
        }
        assert!(
            max_err < tol,
            "dual-path random tree: max error {max_err:.6} >= {tol}"
        );
    }

    // ── T4.1 (G1): correctness at T=16,32,64,128 ──

    #[test]
    fn test_dual_path_g1_scaled_trees() {
        // Test correctness on larger chain trees (T=16,32,64,128).
        // Chain trees are the worst case for the GDN masked solve (deepest paths).
        for &t in &[16usize, 32, 64, 128] {
            let d = 8;
            let parents: Vec<usize> = (0..t)
                .map(|i| if i == 0 { usize::MAX } else { i - 1 })
                .collect();
            let mut rng = rng(t as u32 * 7 + 13);
            // Small-magnitude keys to avoid forward-sub overflow on deep chains.
            let keys: Vec<f32> = (0..t * d).map(|_| 0.05 * rng()).collect();
            let values: Vec<f32> = (0..t * d).map(|_| 0.05 * rng()).collect();
            let queries: Vec<f32> = (0..t * d).map(|_| 0.05 * rng()).collect();
            let alphas: Vec<f32> = vec![0.95; t];
            let betas: Vec<f32> = vec![0.1; t];
            let s0: Vec<f32> = (0..d * d).map(|_| 0.05 * rng()).collect();

            let mut cache = HippocampalCacheDyn::new_with_ones_gamma(d, 32);
            for _ in 0..16 {
                let kp: Vec<f32> = (0..d).map(|_| 0.05 * rng()).collect();
                let vp: Vec<f32> = (0..d).map(|_| 0.05 * rng()).collect();
                cache.observe(&kp, &vp, 0.5, 0.1 + 0.9 * rng().abs());
            }

            let params = GdnLayerParams {
                keys: &keys,
                values: &values,
                queries: &queries,
                alphas: &alphas,
                betas: &betas,
            };
            let topo = build_topology(&parents, &alphas);
            let mut verifier = GdnHolaTreeVerifier::new(t, d, d, t);
            let dual_out =
                verify_gdn_hola_tree_into(&mut verifier, &topo, &params, &mut cache, &s0, d, d);

            // Reference: GDN-only + HOLA read.
            let gdn_ref = {
                let mut v = GdnTreeVerifier::new(t, d, d);
                verify_gdn_tree_into(&mut v, &topo, &params, &s0, d, d).to_vec()
            };
            let mut hola_ref = vec![0.0f32; t * d];
            let mut ancestor_path: Vec<usize> = Vec::with_capacity(t);
            let mut block_kv: Vec<(&[f32], &[f32])> = Vec::with_capacity(t);
            for i in 0..t {
                ancestor_path.clear();
                let mut cur = i;
                while cur != usize::MAX {
                    ancestor_path.push(cur);
                    cur = topo.parent[cur];
                }
                ancestor_path.reverse();
                block_kv.clear();
                for &ak in &ancestor_path {
                    let orig_j = topo.topo_order[ak];
                    block_kv.push((
                        &keys[orig_j * d..(orig_j + 1) * d],
                        &values[orig_j * d..(orig_j + 1) * d],
                    ));
                }
                let orig_i = topo.topo_order[i];
                let q_i = &queries[orig_i * d..(orig_i + 1) * d];
                cache.read_cache_into_fast_block(q_i, &block_kv, &mut hola_ref[i * d..(i + 1) * d]);
            }

            let tol = 1e-3f32;
            let mut max_err = 0.0f32;
            for i in 0..t * d {
                let expected = gdn_ref[i] + hola_ref[i];
                max_err = max_err.max((dual_out[i] - expected).abs());
            }
            assert!(max_err < tol, "G1 T={t}: max error {max_err:.6} >= {tol}");
        }
    }

    // ── T4.4 (G4): alloc-free determinism ──

    #[test]
    fn test_dual_path_g4_alloc_free_determinism() {
        // G4: verify_gdn_hola_tree_into must be deterministic (scratch reuse
        // without corruption) and all-finite on repeated calls at max tree size.
        // This mirrors the Plan 424 test_verify_alloc_free_hot_path pattern:
        // since scratch buffers are private, we verify determinism + finiteness
        // (which proves no realloc corruption).
        let (t, d) = (32, 8);
        let parents: Vec<usize> = (0..t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut rng = rng(77);
        let keys: Vec<f32> = (0..t * d).map(|_| 0.05 * rng()).collect();
        let values: Vec<f32> = (0..t * d).map(|_| 0.05 * rng()).collect();
        let queries: Vec<f32> = (0..t * d).map(|_| 0.05 * rng()).collect();
        let alphas: Vec<f32> = vec![0.95; t];
        let betas: Vec<f32> = vec![0.1; t];
        let s0: Vec<f32> = (0..d * d).map(|_| 0.05 * rng()).collect();

        let mut cache = HippocampalCacheDyn::new_with_ones_gamma(d, 32);
        for _ in 0..16 {
            let kp: Vec<f32> = (0..d).map(|_| 0.05 * rng()).collect();
            let vp: Vec<f32> = (0..d).map(|_| 0.05 * rng()).collect();
            cache.observe(&kp, &vp, 0.5, 0.1 + 0.9 * rng().abs());
        }

        let params = GdnLayerParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
        };
        let topo = build_topology(&parents, &alphas);
        let mut verifier = GdnHolaTreeVerifier::new(t, d, d, t);

        let out1 = verify_gdn_hola_tree_into(&mut verifier, &topo, &params, &mut cache, &s0, d, d)
            .to_vec();
        let out2 = verify_gdn_hola_tree_into(&mut verifier, &topo, &params, &mut cache, &s0, d, d)
            .to_vec();

        // Determinism: repeated calls produce identical output (scratch reuse
        // does not leak stale state).
        assert_eq!(
            out1, out2,
            "repeated dual-path verify must be deterministic"
        );

        // All finite (no NaN/Inf from corrupted scratch).
        for &v in &out1 {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
