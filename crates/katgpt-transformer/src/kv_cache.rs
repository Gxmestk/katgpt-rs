//! KV cache types for autoregressive generation, paged branching, and Raven routing.
//!
//! All types here are pure data + allocator helpers — no forward logic.

use katgpt_core::types::{self, Config};

/// KV cache for a single layer (autoregressive generation).
pub struct KVCache {
    pub key: Vec<f32>,   // [block_size, kv_dim] where kv_dim = n_kv_head * head_dim
    pub value: Vec<f32>, // [block_size, kv_dim]
}

impl KVCache {
    pub fn new(config: &Config) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            key: vec![0.0; config.block_size * kvd],
            value: vec![0.0; config.block_size * kvd],
        }
    }

    pub fn reset(&mut self) {
        // Eager zeroing — safe default for a shared substrate crate. The no-op
        // optimization (relying on write-before-read invariant) is a consumer-
        // specific perf decision; consumers that provably maintain the invariant
        // can override locally. The conservative behavior avoids stale-KV leaks
        // for consumers that reset between sequences without re-writing every
        // position (e.g. dflash speculative rollback paths).
        self.key.fill(0.0);
        self.value.fill(0.0);
    }

    /// Invalidate only a single position in the KV cache — O(kv_dim) instead of
    /// O(block_size × kv_dim). Used by dflash when only one position is dirty per
    /// step (Issue 053). Also used by consumers that need to clear a rejected
    /// speculative token's KV before the next draft iteration.
    #[inline]
    pub fn invalidate_position(&mut self, pos: usize, kv_dim: usize) {
        let off = pos * kv_dim;
        if off + kv_dim <= self.key.len() {
            self.key[off..off + kv_dim].fill(0.0);
            self.value[off..off + kv_dim].fill(0.0);
        }
    }
}

/// Multi-layer KV cache: one KVCache per transformer layer.
pub struct MultiLayerKVCache {
    pub layers: Vec<KVCache>,
    /// Highest position written + 1 across all layers, for efficient snapshot.
    fill_pos: usize,
    /// Per-layer sliding-window capacity. `0` = unbounded (uses full
    /// `block_size` allocation, the original behavior for all non-Gemma-4 models).
    /// `> 0` = sliding-bounded ring buffer with mirrored allocation (capacity
    /// `2 * sliding_window` per layer, see [`new_gemma4_sliding_bounded`]).
    ///
    /// When bounded, the forward code writes K/V at `pos % capacity` AND mirrors
    /// to `pos % capacity + capacity`, so any window of ≤ `capacity` positions is
    /// always contiguous in the buffer — no wrap-around handling needed in the
    /// attention code. Reads remap `t_start → t_start % capacity`.
    ///
    /// [`new_gemma4_sliding_bounded`]: Self::new_gemma4_sliding_bounded
    pub sliding_capacity: Vec<usize>,
}

impl MultiLayerKVCache {
    pub fn new(config: &Config) -> Self {
        let mut layers = Vec::with_capacity(config.n_layer);
        layers.extend((0..config.n_layer).map(|_| KVCache::new(config)));
        Self {
            layers,
            fill_pos: 0,
            sliding_capacity: vec![0; config.n_layer],
        }
    }

    /// Construct with per-layer KV dimensions. Used by models where each layer
    /// has a different kv_dim (e.g. Gemma 4's alternating sliding/full layers —
    // Issue 577). The vec length MUST equal `config.n_layer`.
    ///
    /// Each cache layer is allocated as `[block_size * per_layer_kv_dim[i]]`.
    pub fn new_with_per_layer_kv_dim(config: &Config, per_layer_kv_dim: &[usize]) -> Self {
        debug_assert_eq!(
            per_layer_kv_dim.len(),
            config.n_layer,
            "per_layer_kv_dim length must equal n_layer"
        );
        let layers = per_layer_kv_dim
            .iter()
            .map(|&kvd| KVCache {
                key: vec![0.0; config.block_size * kvd],
                value: vec![0.0; config.block_size * kvd],
            })
            .collect();
        Self {
            layers,
            fill_pos: 0,
            sliding_capacity: vec![0; config.n_layer],
        }
    }

    /// Construct a sliding-bounded cache for Gemma-4's alternating
    /// Sliding/Full layer pattern (Plan 320 Phase D3 production fix).
    ///
    /// Sliding layers get a **mirrored ring buffer** of capacity
    /// `2 * sliding_window * kvd` (logical capacity `sliding_window`, mirrored
    /// for contiguous-window reads). Full layers get `block_size * kvd`
    /// (unbounded, same as `new_with_per_layer_kv_dim`).
    ///
    /// This reduces the 256K-context KV cache from ~168 GiB (naive, all layers
    /// at `block_size`) to ~8 GiB (f32) or ~1.08 GiB (Q4) — a 20–150× reduction.
    /// See the D3 test (`plan320_d3_kv_cache_256k_budget.rs`) for the budget
    /// analysis.
    ///
    /// # Arguments
    ///
    /// * `config` — for `n_layer` + `block_size`.
    /// * `per_layer_kv_dim` — per-layer KV dimension (same as
    ///   `new_with_per_layer_kv_dim`).
    /// * `sliding_layers` — boolean mask: `true` = this layer is sliding-window
    ///   (bounded allocation), `false` = full attention (unbounded). The caller
    ///   determines this from the model's layer-type pattern (e.g.
    ///   `config.gemma4_layer_types` in katgpt-core, which isn't visible at this
    ///   substrate layer).
    /// * `sliding_window` — the sliding-window size (Gemma-4 = 1024). Must be > 0
    ///   for sliding layers to be bounded; 0 falls back to unbounded for all layers.
    pub fn new_gemma4_sliding_bounded(
        config: &Config,
        per_layer_kv_dim: &[usize],
        sliding_layers: &[bool],
        sliding_window: usize,
    ) -> Self {
        debug_assert_eq!(
            per_layer_kv_dim.len(),
            config.n_layer,
            "per_layer_kv_dim length must equal n_layer"
        );
        debug_assert_eq!(
            sliding_layers.len(),
            config.n_layer,
            "sliding_layers length must equal n_layer"
        );
        let sw = sliding_window;
        let layers = per_layer_kv_dim
            .iter()
            .enumerate()
            .map(|(i, &kvd)| {
                let is_sliding = sliding_layers.get(i).copied().unwrap_or(false);
                let capacity = if is_sliding && sw > 0 {
                    // Mirrored: 2× sliding_window for contiguous-window reads.
                    2 * sw * kvd
                } else {
                    config.block_size * kvd
                };
                KVCache {
                    key: vec![0.0; capacity],
                    value: vec![0.0; capacity],
                }
            })
            .collect();
        // Record per-layer sliding capacity (0 = unbounded, sw = sliding-bounded).
        let sliding_capacity: Vec<usize> = (0..config.n_layer)
            .map(|i| {
                let is_sliding = sliding_layers.get(i).copied().unwrap_or(false);
                if is_sliding && sw > 0 { sw } else { 0 }
            })
            .collect();
        Self {
            layers,
            fill_pos: 0,
            sliding_capacity,
        }
    }

    /// Get the sliding-window capacity for a layer (0 = unbounded/full block_size).
    #[inline]
    pub fn sliding_capacity(&self, layer_idx: usize) -> usize {
        self.sliding_capacity.get(layer_idx).copied().unwrap_or(0)
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.fill_pos = 0;
    }

    /// Invalidate only a single position across all layers — O(n_layer × kv_dim).
    /// Much cheaper than full reset O(n_layer × block_size × kv_dim) when only 1
    /// position is dirty. Used by dflash speculative decoding (Issue 053).
    #[inline]
    pub fn invalidate_position(&mut self, pos: usize, kv_dim: usize) {
        for layer in &mut self.layers {
            layer.invalidate_position(pos, kv_dim);
        }
    }

    /// Update fill_pos tracker. Call after writing to the cache at a position.
    pub fn advance_pos(&mut self, pos: usize) {
        self.fill_pos = self.fill_pos.max(pos + 1);
    }

    /// Get the tracked fill position (highest position written + 1).
    #[inline]
    pub fn fill_pos(&self) -> usize {
        self.fill_pos
    }

    /// Set the fill_pos tracker directly, WITHOUT touching the K/V buffers.
    ///
    /// Use this when a caller has already reshaped the buffer contents in-place
    /// (e.g. sliding-window eviction's `copy_within` shift) and only needs to
    /// advance/shrink the logical fill marker. Distinct from `reset()`, which
    /// also zeroes the K/V data — calling `reset()` after an in-place shift
    /// would wipe the just-copied entries (Issue: sleep eviction
    /// sliding_window_retains_recent failure, 2026-06-29).
    #[inline]
    pub fn set_fill_pos(&mut self, pos: usize) {
        self.fill_pos = pos;
    }

    /// Snapshot KV cache state up to position `pos`.
    /// Copies only filled slots [0..pos) per layer — cheap at our model scale.
    ///
    /// **Allocating:** makes `1 + 2*n_layer` heap allocations per call. For the
    /// per-speculation-step hot path, prefer [`snapshot_into`](Self::snapshot_into)
    /// with a hoisted scratch buffer (zero alloc in steady state). This variant
    /// remains for cold paths and convenience wrappers that don't have a
    /// reusable buffer.
    pub fn snapshot(&self, pos: usize, config: &Config) -> KVSnapshot {
        let kd = types::kv_dim(config);
        let end = pos * kd;
        // Pre-allocate outer Vec to avoid collect() reallocation jitter.
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(KVLayerSnapshot {
                key: layer.key[..end].to_vec(),
                value: layer.value[..end].to_vec(),
            });
        }
        KVSnapshot { pos, layers }
    }

    /// Zero-alloc variant of [`snapshot`](Self::snapshot) that refills a reusable
    /// [`KVSnapshot`] in place. The snapshot's per-layer `key`/`value` buffers
    /// are `resize`d to the new length (reusing their existing allocation when
    /// possible) and overwritten — no new `Vec` is allocated in steady state.
    ///
    /// # Allocation
    ///
    /// On the first call (or when `out` was previously shorter), the inner
    /// Vecs grow. On every subsequent call with the same or smaller `pos`,
    /// the existing allocations are reused — zero new heap allocations. This
    /// is the variant to use on the per-speculation-step hot path.
    ///
    /// # Layer-count changes
    ///
    /// If `out.layers.len() != self.layers.len()`, the outer Vec is resized.
    /// In steady state (same model), this branch is never taken.
    pub fn snapshot_into(&self, pos: usize, config: &Config, out: &mut KVSnapshot) {
        let kd = types::kv_dim(config);
        let end = pos * kd;
        out.pos = pos;
        if out.layers.len() != self.layers.len() {
            out.layers
                .resize_with(self.layers.len(), || KVLayerSnapshot {
                    key: Vec::new(),
                    value: Vec::new(),
                });
        }
        for (src, dst) in self.layers.iter().zip(out.layers.iter_mut()) {
            dst.key.resize(end, 0.0);
            dst.value.resize(end, 0.0);
            dst.key[..end].copy_from_slice(&src.key[..end]);
            dst.value[..end].copy_from_slice(&src.value[..end]);
        }
    }

    /// Restore KV cache from a snapshot.
    /// Writes snapshot data back and zeros out positions [snapshot.pos..block_size)
    /// to prevent stale data leaking into the next sequence. The tail zeroing is
    /// the conservative default for a shared substrate crate.
    pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
        let kd = types::kv_dim(config);
        // Hoist loop-invariant `end` out of the per-layer loop.
        let end = snapshot.pos * kd;
        for (layer, snap_layer) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
            layer.key[..end].copy_from_slice(&snap_layer.key);
            layer.value[..end].copy_from_slice(&snap_layer.value);
            // Zero out positions [snapshot.pos..block_size) to prevent stale data
            // from a previous sequence leaking into the restored state.
            layer.key[end..].fill(0.0);
            layer.value[end..].fill(0.0);
        }
    }
}

/// Cheap snapshot of KV cache state up to position `pos`.
/// Only copies filled slots [0..pos) per layer, not the entire block_size buffer.
///
/// `Default` is derived so callers can construct an empty snapshot for use as
/// a reusable scratch buffer with [`MultiLayerKVCache::snapshot_into`] — the
/// zero-alloc variant on the per-speculation-step hot path.
#[derive(Default)]
pub struct KVSnapshot {
    pub pos: usize,
    pub layers: Vec<KVLayerSnapshot>,
}

/// Per-layer snapshot of KV cache data.
#[derive(Default)]
pub struct KVLayerSnapshot {
    pub key: Vec<f32>,   // [pos * kv_dim]
    pub value: Vec<f32>, // [pos * kv_dim]
}

/// Preload drafter's KV cache with target's pre-computed key/value pairs.
///
/// Copies target's KV for positions [0..pos) into drafter's cache.
/// This enables cross-attention: the drafter attends to the target's past KV
/// instead of computing its own from scratch.
///
/// Only active when `target_kv_dim == draft_kv_dim` (dimensions must match).
/// When dimensions don't match, silently returns (drafter computes its own KV).
///
/// Hybrid behavior after preload:
/// - Past positions [0..pos): read from preloaded target KV
/// - New positions [pos..]: computed by drafter during forward pass
pub fn preload_kv_cache(
    draft_cache: &mut MultiLayerKVCache,
    target_cache: &MultiLayerKVCache,
    pos: usize,
    target_config: &Config,
    draft_config: &Config,
) {
    let target_kv_dim = types::kv_dim(target_config);
    let draft_kv_dim = types::kv_dim(draft_config);

    // Dimension guard: can only share when kv_dim matches
    if target_kv_dim != draft_kv_dim {
        return;
    }

    // Layer guard: can only share layers that exist in both caches
    let min_layers = draft_cache.layers.len().min(target_cache.layers.len());

    // Copy KV for positions [0..pos) for each shared layer
    let copy_len = pos * target_kv_dim;
    if copy_len > 0 {
        for layer_idx in 0..min_layers {
            let draft_layer = &mut draft_cache.layers[layer_idx];
            let target_layer = &target_cache.layers[layer_idx];
            draft_layer.key[..copy_len].copy_from_slice(&target_layer.key[..copy_len]);
            draft_layer.value[..copy_len].copy_from_slice(&target_layer.value[..copy_len]);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Paged KV cache — DDTree branch exploration (copy-on-write fork)
// ──────────────────────────────────────────────────────────────────────────

/// Page size in tokens (tuneable, must be power of 2).
pub const PAGE_SIZE: usize = 16;

/// Paged KV cache for DDTree branch exploration.
/// Allocates memory in fixed-size pages with copy-on-write fork.
///
/// Page layout per page: `[K_data | V_data]` where each segment is `PAGE_SIZE * kv_dim` floats.
/// This enables sharing prefix pages between branches without cloning data.
///
/// Fields are `pub` because katgpt-rs root tests inspect them directly
/// (`layer_page_tables`, `free_pages`). Consumers should prefer the method API.
/// Field order groups heap pointers first, then `usize` scalars, to minimize padding.
pub struct PagedKVCache {
    /// Pool of pages. Each page: `[PAGE_SIZE * kv_dim * 2]` floats (K then V).
    pub pages: Vec<Vec<f32>>,
    /// Per-layer page tables. `layer_page_tables[layer][seq_idx]` = vec of page indices.
    pub layer_page_tables: Vec<Vec<Vec<usize>>>,
    /// Free list of page indices for reuse.
    pub free_pages: Vec<usize>,
    /// Reference count per page index. Page is free when ref_count == 0.
    /// Maintained on fork (increment shared pages) and rollback (decrement removed pages).
    /// Enables O(1) exclusive-page detection instead of O(N×P×L) HashSet scan (Issue 053).
    pub page_ref_counts: Vec<usize>,
    /// Dimension of each KV entry (`n_kv_head * head_dim`).
    pub kv_dim: usize,
    /// Cached `PAGE_SIZE * kv_dim` — avoids recomputing on every write/read.
    pub kv_page_size: usize,
    /// Total pages ever allocated (monotonically increasing).
    pub total_pages: usize,
}

impl PagedKVCache {
    /// Create a new paged KV cache.
    /// `max_sequences`: initial number of sequence slots (can grow via fork).
    ///
    /// All initial pages start in the free list (ref_count == 0) so they can be
    /// reused immediately by `alloc_page` without growing the pool. This is the
    /// memory-efficient initialization ported from riir-engine.
    pub fn new(config: &Config, max_sequences: usize) -> Self {
        let kvd = types::kv_dim(config);
        let initial_pages_per_layer = config.block_size / PAGE_SIZE + 1;
        let initial_total = initial_pages_per_layer * config.n_layer;

        Self {
            pages: (0..initial_total)
                .map(|_| vec![0.0; PAGE_SIZE * kvd * 2])
                .collect(),
            layer_page_tables: (0..config.n_layer)
                .map(|_| (0..max_sequences).map(|_| Vec::new()).collect())
                .collect(),
            // All initial pages start as free; preallocate to avoid first-grow realloc.
            free_pages: (0..initial_total).collect(),
            page_ref_counts: vec![0; initial_total],
            kv_dim: kvd,
            kv_page_size: PAGE_SIZE * kvd,
            total_pages: initial_total,
        }
    }

    /// Allocate a new page. Reuse from free list or grow the pool.
    fn alloc_page(&mut self) -> usize {
        let idx = match self.free_pages.pop() {
            Some(idx) => {
                self.pages[idx].fill(0.0);
                idx
            }
            None => {
                self.pages.push(vec![0.0; PAGE_SIZE * self.kv_dim * 2]);
                let idx = self.total_pages;
                self.total_pages += 1;
                self.page_ref_counts.push(0);
                idx
            }
        };
        self.page_ref_counts[idx] += 1;
        idx
    }

    /// Ensure sequence `seq_idx` has enough pages to cover position `pos` for all layers.
    ///
    /// Uses stack-allocated `ArrayVec` for scratch (bounded to 128 layers) — zero
    /// heap allocation in the hot path. Ported from riir-engine.
    pub fn ensure_pages(&mut self, seq_idx: usize, pos: usize) {
        use arrayvec::ArrayVec;
        let pages_needed = pos / PAGE_SIZE + 1;

        // Grow sequence slots if needed (no page allocation, just empty vecs)
        for layer_tables in &mut self.layer_page_tables {
            while seq_idx >= layer_tables.len() {
                layer_tables.push(Vec::new());
            }
        }

        // Collect deficits into a stack-allocated array.
        // Most models have <= 128 layers; ArrayVec avoids per-call heap allocation.
        let mut deficits = ArrayVec::<usize, 128>::new();
        for lt in &self.layer_page_tables {
            deficits.push(pages_needed.saturating_sub(lt[seq_idx].len()));
        }
        let total_new: usize = deficits.iter().copied().sum();

        // Distribute newly-allocated page indices back into the layer tables.
        //
        // Fast path (the common case — autoregressive decode advances `pos` by
        // 1, so each layer's deficit is 0 or 1, total_new ≤ n_layers ≤ 128):
        // allocate into a flat stack `ArrayVec<usize, 128>` and `extend_from_slice`
        // per layer. Zero heap allocation regardless of deficit distribution.
        //
        // Slow path (prefill / large position jump where total_new > 128): fall
        // back to one heap `Vec<usize>` per layer-with-deficit. Matches the
        // previous behavior; the cost is dominated by the page-data allocation
        // itself, not the index Vec.
        if total_new <= 128 {
            let mut flat_new_pages = ArrayVec::<usize, 128>::new();
            for _ in 0..total_new {
                flat_new_pages.push(self.alloc_page());
            }
            let mut cursor = 0usize;
            for (layer_tables, &deficit) in self.layer_page_tables.iter_mut().zip(&deficits) {
                if deficit > 0 {
                    layer_tables[seq_idx]
                        .extend_from_slice(&flat_new_pages[cursor..cursor + deficit]);
                    cursor += deficit;
                }
            }
            debug_assert_eq!(cursor, total_new, "distributed all allocated pages");
        } else {
            // Slow path: per-layer heap Vecs (original behavior).
            let mut new_pages = ArrayVec::<Vec<usize>, 128>::new();
            for &deficit in &deficits {
                let pages: Vec<usize> = (0..deficit).map(|_| self.alloc_page()).collect();
                new_pages.push(pages);
            }
            for (layer_tables, pages) in self.layer_page_tables.iter_mut().zip(new_pages) {
                layer_tables[seq_idx].extend(pages);
            }
        }
    }

    /// Write K and V for a token position in a specific layer.
    /// Layout per page: `[K_data | V_data]` where each is `PAGE_SIZE * kv_dim` floats.
    #[inline]
    pub fn write_kv(&mut self, layer_idx: usize, seq_idx: usize, pos: usize, k: &[f32], v: &[f32]) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &mut self.pages[pidx];
        let k_off = page_local * self.kv_dim;
        let v_off = self.kv_page_size + page_local * self.kv_dim;
        page[k_off..k_off + self.kv_dim].copy_from_slice(k);
        page[v_off..v_off + self.kv_dim].copy_from_slice(v);
    }

    /// Read K and V for a token position in a specific layer.
    #[inline]
    pub fn read_kv(
        &self,
        layer_idx: usize,
        seq_idx: usize,
        pos: usize,
        k: &mut [f32],
        v: &mut [f32],
    ) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &self.pages[pidx];
        let k_off = page_local * self.kv_dim;
        let v_off = self.kv_page_size + page_local * self.kv_dim;
        k.copy_from_slice(&page[k_off..k_off + self.kv_dim]);
        v.copy_from_slice(&page[v_off..v_off + self.kv_dim]);
    }

    /// Fork a sequence with copy-on-write semantics.
    /// Shares prefix pages up to `fork_at_pos`, allocates new pages on demand after fork.
    /// Returns the new sequence index.
    pub fn fork(&mut self, seq_idx: usize, fork_at_pos: usize) -> usize {
        let fork_page = fork_at_pos / PAGE_SIZE;
        let new_seq = self.layer_page_tables[0].len();

        for layer_tables in &mut self.layer_page_tables {
            let source = &layer_tables[seq_idx];
            let shared_pages = source[..fork_page.min(source.len())].to_vec();
            // Increment ref counts for shared pages (Issue 053)
            for &pidx in &shared_pages {
                self.page_ref_counts[pidx] += 1;
            }
            layer_tables.push(shared_pages);
        }

        new_seq
    }

    /// Rollback a sequence to a given position, freeing exclusive pages.
    ///
    /// Truncates page tables to keep only pages covering positions `[0..rollback_to_pos)`.
    /// Pages that are exclusively owned by this sequence (not referenced by any other
    /// sequence in any layer) are returned to the free list for reuse.
    ///
    /// This is the "page table CoW rollback" — no data is copied, only page table
    /// entries are manipulated and exclusive pages are recycled.
    pub fn rollback(&mut self, seq_idx: usize, rollback_to_pos: usize) {
        let keep_count = rollback_to_pos / PAGE_SIZE;

        // Issue 053: use ref counts for O(1) exclusive-page detection instead of
        // building a HashSet by scanning all sequences across all layers (O(N×P×L)).
        // Decrement ref count for each removed page; if count reaches 0, it's exclusive.
        //
        // Pop from the end (no intermediate Vec) — the previous form allocated
        // a `Vec<usize>` per layer per rollback just to iterate it once.
        for layer_tables in &mut self.layer_page_tables {
            if seq_idx >= layer_tables.len() {
                continue;
            }
            let table = &mut layer_tables[seq_idx];
            while table.len() > keep_count {
                // SAFETY: we just checked `table.len() > keep_count`, so the
                // table is non-empty; `pop` returns the last element.
                let pidx = table.pop().expect("checked non-empty above");
                self.page_ref_counts[pidx] -= 1;
                if self.page_ref_counts[pidx] == 0 {
                    self.free_pages.push(pidx);
                }
            }
        }
    }

    /// Reset all sequences and free all pages.
    pub fn reset(&mut self) {
        for layer_tables in &mut self.layer_page_tables {
            for table in layer_tables.iter_mut() {
                self.free_pages.append(table);
            }
        }
        // Zero all ref counts since all page tables are cleared
        self.page_ref_counts.fill(0);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Raven RSM — Routing Slot Memory (O(1) KV replacement for draft model)
// ──────────────────────────────────────────────────────────────────────────

/// Raven Routing Slot Memory — O(1) KV replacement for the draft model.
///
/// Fixed-size `[num_slots × kv_dim]` memory updated via sparse Top-K routing.
/// Unselected slots are completely frozen — perfect for preserving struct
/// definitions and imports while churning through syntax tokens.
pub struct RavenKVCache {
    // ── Vec fields first (ptr+len+cap = 24 bytes, 8-byte aligned) ──
    /// Key memory: [num_slots × kv_dim]
    pub keys: Vec<f32>,
    /// Value memory: [num_slots × kv_dim]
    pub values: Vec<f32>,
    // Pre-allocated buffers for zero-alloc router computation
    pub router_scored: Vec<(usize, f32)>, // [num_slots]
    pub router_r_t: Vec<f32>,             // [num_slots]
    /// Pre-allocated score buffer for raven_readout_into `[num_slots]`
    pub readout_scores: Vec<f32>,
    /// Pre-allocated output buffer for raven_readout_into `[kv_dim]`
    pub readout_output: Vec<f32>,
    // ── usize fields (8-byte aligned, no padding after Vecs) ──
    /// Number of memory slots
    pub num_slots: usize,
    /// Dimension of each KV entry (= kv_dim = n_kv_head × head_dim)
    pub kv_dim: usize,
    /// Top-K slots to update per token
    pub top_k: usize,
    // ── f32 field last (4-byte aligned, no trailing padding on 64-bit) ──
    /// Forget rate for gated update (negative = slower decay)
    pub forget_rate: f32,
}

impl RavenKVCache {
    pub fn new(config: &Config, num_slots: usize, top_k: usize) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            num_slots,
            kv_dim: kvd,
            top_k,
            keys: vec![0.0; num_slots * kvd],
            values: vec![0.0; num_slots * kvd],
            router_scored: vec![(0usize, 0.0f32); num_slots],
            router_r_t: vec![0.0f32; num_slots],
            readout_scores: vec![0.0; num_slots],
            readout_output: vec![0.0; kvd],
            forget_rate: -1.0,
        }
    }

    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        // Use fill() instead of clear() to preserve pre-allocated capacity.
        // clear() drops len to 0, forcing reallocation on next use via resize.
        self.router_scored.fill((0, 0.0));
        self.router_r_t.fill(0.0);
        self.readout_scores.fill(0.0);
        self.readout_output.fill(0.0);
    }

    /// Export the current routing vector `r_t` (post-router, pre-update).
    /// Returns the normalized Top-K routing weights for all slots.
    /// Used by Phase 3 routed speculation to feed slot selection into anyrag.
    #[inline]
    pub fn r_t(&self) -> &[f32] {
        &self.router_r_t
    }
}

// ── Plan 320 Phase D3: sliding-bounded cache tests ──────────────────────────

#[cfg(test)]
mod sliding_bounded_tests {
    use super::*;

    /// Build a tiny config for testing: 3 layers, block_size=32, kvd=4.
    fn tiny_config() -> Config {
        Config {
            n_layer: 3,
            block_size: 32,
            n_embd: 16,
            n_head: 2,
            head_dim: 4,
            n_kv_head: 1,
            ..Config::micro()
        }
    }

    #[test]
    fn sliding_bounded_allocates_less_than_naive() {
        // G1: sliding-bounded cache should allocate less memory than the naive
        // all-full allocation for sliding layers.
        let config = tiny_config();
        let kvd = 4;
        let sliding_window = 8; // much less than block_size=32

        // Layer 0 and 2 are sliding; layer 1 is full.
        let sliding_layers = vec![true, false, true];
        let per_layer_kvd = vec![kvd; 3];

        // Naive allocation: all layers at block_size.
        let naive = MultiLayerKVCache::new_with_per_layer_kv_dim(&config, &per_layer_kvd);
        let naive_bytes: usize = naive
            .layers
            .iter()
            .map(|l| (l.key.len() + l.value.len()) * std::mem::size_of::<f32>())
            .sum();

        // Sliding-bounded allocation.
        let bounded = MultiLayerKVCache::new_gemma4_sliding_bounded(
            &config,
            &per_layer_kvd,
            &sliding_layers,
            sliding_window,
        );
        let bounded_bytes: usize = bounded
            .layers
            .iter()
            .map(|l| (l.key.len() + l.value.len()) * std::mem::size_of::<f32>())
            .sum();

        // Sliding layers: 2 * sw * kvd * 2 (mirrored, key+value)
        // Full layer: block_size * kvd * 2
        // Total bounded should be less than naive.
        // Sliding layers: 2 * sw * kvd elements per buffer (key+value) → 2 * 2 * sw * kvd
        // Full layer: block_size * kvd elements per buffer → 2 * block_size * kvd
        // Total elements × sizeof(f32) = bytes.
        let sliding_layer_elems = 2 * sliding_window * kvd * 2; // key+value
        let full_layer_elems = config.block_size * kvd * 2;
        let expected_elems = 2 * sliding_layer_elems + full_layer_elems; // 2 sliding + 1 full
        let expected_bytes = expected_elems * std::mem::size_of::<f32>();
        assert_eq!(
            bounded_bytes, expected_bytes,
            "bounded bytes should match expected: 2 sliding + 1 full layer"
        );

        // The claim in the test's name — `naive_bytes` was computed above but
        // never checked, so G1's actual premise went unasserted.
        assert!(
            bounded_bytes < naive_bytes,
            "sliding-bounded ({bounded_bytes} B) should allocate less than naive ({naive_bytes} B)"
        );
    }

    #[test]
    fn sliding_capacity_reports_correct_values() {
        let config = tiny_config();
        let sliding_layers = vec![true, false, true];
        let per_layer_kvd = vec![4; 3];
        let sw = 8;

        let cache = MultiLayerKVCache::new_gemma4_sliding_bounded(
            &config,
            &per_layer_kvd,
            &sliding_layers,
            sw,
        );

        assert_eq!(cache.sliding_capacity(0), sw, "layer 0 is sliding");
        assert_eq!(cache.sliding_capacity(1), 0, "layer 1 is full");
        assert_eq!(cache.sliding_capacity(2), sw, "layer 2 is sliding");
    }

    #[test]
    fn unbounded_cache_has_zero_sliding_capacity() {
        // The standard new_with_per_layer_kv_dim should have all-zero capacities.
        let config = tiny_config();
        let per_layer_kvd = vec![4; 3];
        let cache = MultiLayerKVCache::new_with_per_layer_kv_dim(&config, &per_layer_kvd);

        for i in 0..config.n_layer {
            assert_eq!(
                cache.sliding_capacity(i),
                0,
                "layer {i} should be unbounded"
            );
        }
    }

    #[test]
    fn sliding_bounded_mirror_write_correct() {
        // G2: writing to a sliding-bounded cache at position `pos` should write
        // to both `pos % sw` and `(pos % sw) + sw` (mirror).
        let config = tiny_config();
        let sw = 8;
        let kvd = 4;
        let sliding_layers = vec![true, false, false];
        let per_layer_kvd = vec![kvd; 3];

        let mut cache = MultiLayerKVCache::new_gemma4_sliding_bounded(
            &config,
            &per_layer_kvd,
            &sliding_layers,
            sw,
        );

        // Write a recognizable value at position 3 on layer 0 (sliding).
        let pos = 3usize;
        let test_val = 42.0f32;
        let ring_pos = pos % sw; // = 3
        let layer = &mut cache.layers[0];

        // Simulate the write pattern from forward_gemma4.
        let off = ring_pos * kvd;
        let mirror_off = (ring_pos + sw) * kvd;
        layer.key[off..off + kvd].fill(test_val);
        layer.key[mirror_off..mirror_off + kvd].fill(test_val);
        layer.value[off..off + kvd].fill(test_val);
        layer.value[mirror_off..mirror_off + kvd].fill(test_val);

        // Verify both primary and mirror slots have the value.
        assert_eq!(layer.key[off], test_val, "primary slot should have value");
        assert_eq!(
            layer.key[mirror_off], test_val,
            "mirror slot should have value"
        );

        // The buffer is 2 * sw * kvd = 2 * 8 * 4 = 64 elements.
        assert_eq!(layer.key.len(), 2 * sw * kvd);
    }

    #[test]
    fn sliding_bounded_window_contiguous() {
        // G3: a window of positions [t_start, pos] where pos - t_start < sw should
        // map to a contiguous region in the mirrored buffer.
        let sw = 8;
        let kvd = 4;

        // Simulate positions 0..15 (wraps around the 8-capacity ring).
        // At pos=15, t_start = 15 - 7 = 8. Window = [8, 15], length = 8.
        // In the ring: t_start % sw = 8 % 8 = 0, pos % sw = 15 % 8 = 7.
        // The window maps to [0, 7] in the primary region — contiguous.
        let pos = 15;
        let t_start = 8;
        let t_n = pos - t_start + 1;
        assert_eq!(t_n, 8);
        assert!(t_n <= sw, "window must fit in sliding_window");

        let read_start = (t_start % sw) * kvd; // = 0
        let read_end = (pos % sw) * kvd + kvd; // = 7*4+4 = 32
        assert_eq!(read_start, 0);
        assert_eq!(read_end, sw * kvd);
        // The window [0, 32) is contiguous within the primary region [0, sw*kvd).
    }

    #[test]
    fn sliding_bounded_window_contiguous_at_boundary() {
        // G4: window that straddles a ring boundary but fits within sw.
        // pos=9, t_start=3, window=[3,9], length=7.
        // ring: t_start%8=3, pos%8=1. In the mirrored buffer, [3*4, 2*4) spans
        // [12, 8) — wrapping! But because of the mirror at [sw, 2*sw), the
        // read from position 3 at offset 3*kvd through the mirror region is
        // contiguous: primary[3..8) + mirror[0..2) = primary[3..8) + primary[8..10).
        // Wait — that's NOT contiguous unless we read from the mirror.
        //
        // Actually, the key insight: in the mirrored buffer of size 2*sw,
        // positions t_start..=pos map to (t_start%sw)..(t_start%sw + t_n - 1).
        // Since t_n ≤ sw, this range is within [0, 2*sw). And we read from
        // (t_start % sw) * kvd, which puts us at the start of the window.
        // The mirror ensures that positions that wrapped around are available
        // at sw + their ring index.
        let sw = 8;
        let kvd = 4;
        let pos = 9;
        let t_start = 3;
        let t_n = pos - t_start + 1; // = 7

        // Read starts at t_start % sw = 3, continues for t_n = 7 positions.
        // In the mirrored buffer: positions 3,4,5,6,7,0(mirror@8),1(mirror@9).
        // But we read from offset (3%8)*4 = 12, for 7 positions = 28 floats.
        // The buffer at [12, 40) covers ring positions 3,4,5,6,7,8(mirror of 0),9(mirror of 1).
        // This is contiguous and correct!
        let read_start = (t_start % sw) * kvd; // = 12
        let read_len = t_n * kvd; // = 28
        let read_end = read_start + read_len; // = 40

        // The buffer is 2*sw*kvd = 64 elements. read_end = 40 ≤ 64. ✓
        assert!(read_end <= 2 * sw * kvd, "read must fit in mirrored buffer");
        assert_eq!(read_start, 12);
        assert_eq!(read_len, 28);
    }
}
