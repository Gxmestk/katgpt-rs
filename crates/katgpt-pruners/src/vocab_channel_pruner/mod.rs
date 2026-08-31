//! VocabChannel Pruner — ROTATE-Derived ConstraintPruner for Speculative Decoding (Plan 228).
//!
//! At model load time, decomposes MLP output weights into vocabulary channels using
//! kurtosis-maximizing Householder reflections (ROTATE method, arXiv:2606.03990).
//! Builds per-neuron token reachability maps. At inference time, acts as a
//! `ConstraintPruner` to reject unreachable tokens in DDTree speculative decoding.
//!
//! **Expected gain:** 30-60% DDTree branch reduction, quality-neutral.
//!
//! # Architecture
//!
//! ```text
//! Load Time:
//!   Wout[l][i] → ROTATE → channels {v₁...vₖ}
//!   Each channel → top-K tokens → per-neuron reachability set
//!   Aggregate → per-layer union reachability
//!
//! Inference Time:
//!   ConstraintPruner::is_valid(depth, token_idx, ...) = reachability.contains(token_idx)
//! ```
//!
//! # Phases
//!
//! 1. Core math: skewness, Householder reflection, vocab projection, token masking
//! 2. ROTATE decomposition: kurtosis-maximizing channel discovery
//! 3. Reachability map: per-layer, per-neuron token sets with binary serialization
//! 4. ConstraintPruner: integrates with DDTree via `ConstraintPruner` trait

use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use katgpt_core::traits::ConstraintPruner;
// TransformerWeights lives in katgpt-transformer (leaf); Config lives in
// katgpt-types (leaf). Both are mandatory deps of this crate.
use katgpt_transformer::TransformerWeights;
use katgpt_types::Config;

#[cfg(feature = "lattice_operad")]
use crate::lattice_operad::PrunerExpr;

// ── Phase 1: Core Math ──────────────────────────────────────────────

/// Population skewness (γ₁) of a distribution in a single O(n) pass.
///
/// Returns 0.0 for degenerate inputs (n < 3 or zero variance).
/// No allocation — operates entirely on the input slice.
#[inline]
pub fn skewness(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n < 3.0 {
        return 0.0;
    }

    let mean: f32 = values.iter().copied().sum::<f32>() / n;

    // Separate accumulators (not a tuple fold) so LLVM can vectorize m2 and m3
    // independently. The cross-iteration data dependency in the tuple-fold form
    // blocked auto-vectorization. FP semantics preserved: `d*d*d` left-to-right.
    let mut m2 = 0.0f32;
    let mut m3 = 0.0f32;
    for &x in values {
        let d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
    }

    if m2 < 1e-10 {
        return 0.0;
    }

    // Population skewness: γ₁ = n²·m₃ / (n·m₂)^{3/2} = √n · m₃ / m₂^{3/2}
    let m2_32 = m2 * m2.sqrt();
    if m2_32 < 1e-10 {
        return 0.0;
    }

    n.sqrt() * m3 / m2_32
}

/// Excess kurtosis (κ) of a distribution in a single O(n) pass.
///
/// Reuses the same formula as `kurtosis_gate::excess_kurtosis` for self-containment.
/// Returns 0.0 for degenerate inputs (n < 4 or zero variance).
#[inline]
pub fn excess_kurtosis(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n < 4.0 {
        return 0.0;
    }

    let mean: f32 = values.iter().copied().sum::<f32>() / n;

    // Separate accumulators for independent SIMD vectorization (see skewness).
    // FP semantics preserved: `d*d*d*d` left-to-right matches original fold.
    let mut m2 = 0.0f32;
    let mut m4 = 0.0f32;
    for &x in values {
        let d = x - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }

    if m2 < 1e-10 {
        return 0.0;
    }

    (m4 * n) / (m2 * m2) - 3.0
}

/// Apply a Householder reflection to vector `x` given Householder vector `h`.
///
/// Computes `(I - 2·h·hᵀ / ‖h‖²) @ x = x - 2·h·(hᵀ·x) / ‖h‖²`.
///
/// O(d) time, O(d) allocation for the output.
/// Returns a zero vector if `h` has zero norm (degenerate reflection).
///
/// Fused single-pass: `h_norm_sq` and `dot_hx` computed together to halve memory
/// reads vs the previous two-pass iterator chain.
#[inline]
pub fn householder_apply(h: &[f32], x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(h.len(), x.len());

    let mut h_norm_sq = 0.0f32;
    let mut dot_hx = 0.0f32;
    for i in 0..h.len() {
        h_norm_sq += h[i] * h[i];
        dot_hx += h[i] * x[i];
    }
    if h_norm_sq < 1e-12 {
        return x.to_vec();
    }

    let scale = 2.0 * dot_hx / h_norm_sq;
    let mut out = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        out.push(x[i] - scale * h[i]);
    }
    out
}

/// Sigmoid function. Delegates to `katgpt_core::simd::fast_sigmoid` (Cephes polynomial).
#[cfg(test)]
#[inline]
fn sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
}

/// Project a single neuron weight vector through the LM head to get vocabulary logits.
///
/// Given:
/// - `neuron_weight`: size `n_embd` — the neuron's output weight in residual space
/// - `lm_head`: flat `[vocab_size * n_embd]` — LM head weights, row-major
///
/// Computes `logits[t] = dot(lm_head[t*n_embd..(t+1)*n_embd], neuron_weight)`
/// for each token `t` in `0..vocab_size`.
///
/// Returns a vector of size `vocab_size`.
pub fn vocab_project(
    neuron_weight: &[f32],
    lm_head: &[f32],
    vocab_size: usize,
    n_embd: usize,
) -> Vec<f32> {
    debug_assert_eq!(neuron_weight.len(), n_embd);
    debug_assert!(lm_head.len() >= vocab_size * n_embd);

    let mut logits = vec![0.0f32; vocab_size];
    // Direct indexed loop enables LLVM auto-vectorization (FMA on AVX2/NEON).
    // Inner iterator chain (zip+map+sum) prevented vectorization. This is the
    // inner loop of ROTATE decomposition: called 2× per coordinate per iteration
    // × 20 iterations × 3 coords = 120 times per channel discovery.
    // stride math: `t` drives both `logits[t]` write and `base = t * n_embd`
    #[allow(clippy::needless_range_loop)]
    for t in 0..vocab_size {
        let base = t * n_embd;
        let mut sum = 0.0f32;
        for i in 0..n_embd {
            sum += lm_head[base + i] * neuron_weight[i];
        }
        logits[t] = sum;
    }
    logits
}

/// Apply token exclusion mask in-place. Sets masked positions to 0.0.
#[inline]
fn apply_mask(logits: &mut [f32], mask: &[bool]) {
    for (i, m) in mask.iter().enumerate() {
        if *m {
            logits[i] = 0.0;
        }
    }
}

/// Mark tokens whose channel logits are outliers as excluded (True in mask).
///
/// Tokens with `|z_i - μ| > k_sigma * σ` are marked True.
/// Existing True values in the mask are preserved (union).
pub fn iterative_token_mask(logits: &[f32], mask: &mut [bool], k_sigma: f32) {
    let n = logits.len() as f32;
    if n < 2.0 {
        return;
    }

    let mean: f32 = logits.iter().sum::<f32>() / n;
    let variance: f32 = logits.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let sigma = variance.sqrt();

    if sigma < 1e-10 {
        return;
    }

    let threshold = k_sigma * sigma;
    for (i, &z) in logits.iter().enumerate() {
        if (z - mean).abs() > threshold {
            mask[i] = true;
        }
    }
}

/// Find the top-K indices by value using partial selection.
///
/// Returns at most `k` indices sorted by descending value.
fn topk_indices(values: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(values.len());
    if k == 0 {
        return Vec::new();
    }

    // For small k, use partial selection via a min-heap approach
    let mut top: Vec<(usize, f32)> = (0..k).map(|i| (i, values[i])).collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));

    for (i, &val) in values.iter().enumerate().skip(k) {
        if val > top.last().map_or(f32::NEG_INFINITY, |&(_, v)| v) {
            // Find insertion point
            let pos = top.partition_point(|&(_, v)| v > val);
            top.insert(pos, (i, val));
            top.pop(); // keep only k elements
        }
    }

    top.into_iter().map(|(idx, _)| idx).collect()
}

/// Cosine similarity between two vectors.
#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    // Three SIMD dot-product reductions — LLVM cannot vectorize the fused
    // tuple-fold (cross-iteration data dependency).
    let dot = katgpt_core::simd::simd_dot_f32(a, b, n);
    let na = katgpt_core::simd::simd_dot_f32(a, a, n);
    let nb = katgpt_core::simd::simd_dot_f32(b, b, n);
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    dot / denom
}

// ── Phase 2: ROTATE Decomposition ───────────────────────────────────

/// A single vocabulary channel discovered by ROTATE decomposition.
#[derive(Debug, Clone)]
pub struct VocabChannel {
    /// Direction vector in weight space (size n_embd)
    pub direction: Vec<f32>,
    /// Top-K token indices promoted by this channel
    pub top_tokens: Vec<usize>,
    /// Kurtosis of this channel's vocabulary projection
    pub kurtosis: f32,
    /// Skewness of this channel's vocabulary projection
    pub skewness: f32,
}

/// Configuration for the ROTATE decomposition pipeline.
#[derive(Debug, Clone, Copy)]
pub struct VocabChannelConfig {
    /// Number of channels to extract per neuron
    pub max_channels: usize,
    /// Top-K tokens per channel for reachability
    pub top_k_tokens: usize,
    /// Kurtosis threshold for accepting a channel
    pub kurtosis_threshold: f32,
    /// Regularization weight for cosine similarity preservation
    pub lambda: f32,
    /// Learning rate for Householder vector optimization
    pub eta: f32,
    /// Maximum optimization iterations per channel
    pub max_iterations: usize,
    /// Standard deviation multiplier for token masking
    pub sigma_mask: f32,
    /// Number of random coordinate descent dimensions per iteration
    pub coords_per_iter: usize,
    /// Finite difference epsilon for gradient estimation
    pub fd_epsilon: f32,
}

impl Default for VocabChannelConfig {
    fn default() -> Self {
        Self {
            max_channels: 5,
            top_k_tokens: 50,
            kurtosis_threshold: 1.0,
            lambda: 0.1,
            eta: 0.01,
            max_iterations: 20,
            sigma_mask: 2.0,
            coords_per_iter: 3,
            fd_epsilon: 1e-3,
        }
    }
}

/// ROTATE decomposition engine for discovering vocabulary channels in MLP neurons.
pub struct VocabChannelDecomposer {
    config: VocabChannelConfig,
}

impl VocabChannelDecomposer {
    pub fn new(config: VocabChannelConfig) -> Self {
        Self { config }
    }

    /// Decompose a single neuron weight into vocabulary channels.
    ///
    /// Uses iterative ROTATE: Householder-reflected directions that maximize
    /// kurtosis of the vocabulary projection, with cosine similarity regularization.
    pub fn decompose_neuron(
        &self,
        neuron_weight: &[f32],
        lm_head: &[f32],
        vocab_size: usize,
        n_embd: usize,
    ) -> Vec<VocabChannel> {
        let mut channels = Vec::with_capacity(self.config.max_channels);
        let mut mask = vec![false; vocab_size];

        for _ in 0..self.config.max_channels {
            let Some(ch) = self.discover_channel(neuron_weight, lm_head, vocab_size, n_embd, &mask)
            else {
                break;
            };

            if channel.kurtosis < self.config.kurtosis_threshold {
                break;
            }

            // Update mask: exclude outlier tokens for the next iteration
            let z = vocab_project(&channel.direction, lm_head, vocab_size, n_embd);
            iterative_token_mask(&z, &mut mask, self.config.sigma_mask);

            channels.push(channel);
        }

        channels
    }

    /// Discover a single channel via Householder optimization.
    fn discover_channel(
        &self,
        w: &[f32],
        lm_head: &[f32],
        vocab_size: usize,
        n_embd: usize,
        mask: &[bool],
    ) -> Option<VocabChannel> {
        // Initialize Householder vector as small random perturbation
        let mut h: Vec<f32> = (0..n_embd).map(|_| (fastrand::f32() - 0.5) * 0.1).collect();

        // Optimize h to maximize kurtosis - lambda*(1 - cos(v, w))
        self.optimize_householder(&mut h, w, lm_head, vocab_size, n_embd, mask);

        // Compute final channel
        let v = householder_apply(&h, w);
        let mut z = vocab_project(&v, lm_head, vocab_size, n_embd);
        apply_mask(&mut z, mask);

        let k = excess_kurtosis(&z);
        let s = skewness(&z);
        let top_tokens = topk_indices(&z, self.config.top_k_tokens);

        Some(VocabChannel {
            direction: v,
            top_tokens,
            kurtosis: k,
            skewness: s,
        })
    }

    /// Optimize Householder vector via random coordinate descent on kurtosis.
    fn optimize_householder(
        &self,
        h: &mut [f32],
        w: &[f32],
        lm_head: &[f32],
        vocab_size: usize,
        n_embd: usize,
        mask: &[bool],
    ) {
        let eps = self.config.fd_epsilon;

        for _ in 0..self.config.max_iterations {
            // Pick random dimensions for coordinate descent
            for _ in 0..self.config.coords_per_iter {
                let dim = fastrand::usize(0..n_embd);

                // Compute objective at h + eps*e_dim
                h[dim] += eps;
                let v_plus = householder_apply(h, w);
                let mut z_plus = vocab_project(&v_plus, lm_head, vocab_size, n_embd);
                apply_mask(&mut z_plus, mask);
                let obj_plus =
                    excess_kurtosis(&z_plus) + self.config.lambda * cosine_sim(&v_plus, w);

                // Compute objective at h - eps*e_dim
                h[dim] -= 2.0 * eps;
                let v_minus = householder_apply(h, w);
                let mut z_minus = vocab_project(&v_minus, lm_head, vocab_size, n_embd);
                apply_mask(&mut z_minus, mask);
                let obj_minus =
                    excess_kurtosis(&z_minus) + self.config.lambda * cosine_sim(&v_minus, w);

                // Restore h
                h[dim] += eps;

                // Gradient ascent
                let gradient = (obj_plus - obj_minus) / (2.0 * eps);
                h[dim] += self.config.eta * gradient;
            }
        }
    }
}

/// Decompose all neurons in a single layer's MLP down-projection weights.
///
/// # Arguments
/// * `mlp_w2` - Flat `[n_embd * mlp_hidden]` row-major MLP weights
/// * `lm_head` - Flat `[vocab_size * n_embd]` row-major LM head weights
/// * `n_embd` - Embedding dimension
/// * `mlp_hidden` - MLP hidden dimension
/// * `vocab_size` - Vocabulary size
/// * `config` - Decomposition configuration
///
/// # Returns
/// Per-neuron sorted token reachability sets (union of top-K tokens from each channel).
pub fn decompose_layer_channels(
    mlp_w2: &[f32],
    lm_head: &[f32],
    n_embd: usize,
    mlp_hidden: usize,
    vocab_size: usize,
    config: &VocabChannelConfig,
) -> Vec<Vec<usize>> {
    let decomposer = VocabChannelDecomposer::new(*config);
    let mut result = Vec::with_capacity(mlp_hidden);

    for neuron_idx in 0..mlp_hidden {
        // Extract column neuron_idx from mlp_w2 (row-major [n_embd, mlp_hidden])
        // neuron weight: w[i] = mlp_w2[i * mlp_hidden + neuron_idx] for i in 0..n_embd
        let mut neuron_weight = Vec::with_capacity(n_embd);
        for i in 0..n_embd {
            neuron_weight.push(mlp_w2[i * mlp_hidden + neuron_idx]);
        }

        let channels = decomposer.decompose_neuron(&neuron_weight, lm_head, vocab_size, n_embd);

        // Union top-K tokens from each channel into a sorted set.
        // collect-all → sort_unstable → dedup is O(N log N) vs the old
        // per-element binary_search + Vec::insert which was O(N²) due to shifts.
        let total_tokens: usize = channels.iter().map(|ch| ch.top_tokens.len()).sum();
        let mut token_set: Vec<usize> = Vec::with_capacity(total_tokens);
        for ch in &channels {
            token_set.extend_from_slice(&ch.top_tokens);
        }
        token_set.sort_unstable();
        token_set.dedup();

        result.push(token_set);
    }

    result
}

// ── Phase 3: Reachability Map ───────────────────────────────────────

/// Per-layer, per-neuron token reachability map.
///
/// Uses compact sorted `Vec<usize>` sets per neuron for O(log n) lookup.
#[derive(Debug, Clone)]
pub struct VocabChannelMap {
    /// Per-layer reachability: `layers[layer_idx][neuron_idx]` = sorted token set
    layers: Vec<Vec<Vec<usize>>>,
    /// Vocabulary size for bounds checking
    vocab_size: usize,
}

impl VocabChannelMap {
    /// Build the map from per-neuron channel results.
    ///
    /// `channels_per_neuron[layer][neuron]` = sorted token reachability set.
    pub fn from_channels(channels_per_neuron: &[Vec<Vec<usize>>], vocab_size: usize) -> Self {
        Self {
            layers: channels_per_neuron.to_vec(),
            vocab_size,
        }
    }

    /// Number of layers in the map.
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Number of neurons in a given layer.
    #[inline]
    pub fn neuron_count(&self, layer: usize) -> usize {
        match self.layers.get(layer) {
            Some(l) => l.len(),
            None => 0,
        }
    }

    /// Check if a token is reachable by a specific neuron in a specific layer.
    ///
    /// Uses binary search on the sorted token set: O(log n).
    pub fn is_reachable(&self, layer: usize, neuron: usize, token: usize) -> bool {
        match self.layers.get(layer).and_then(|l| l.get(neuron)) {
            Some(tokens) => tokens.binary_search(&token).is_ok(),
            None => false,
        }
    }

    /// Get the sorted token set for a specific neuron.
    ///
    /// Returns an empty slice if layer or neuron is out of bounds.
    #[inline]
    pub fn neuron_tokens(&self, layer: usize, neuron: usize) -> &[usize] {
        match self.layers.get(layer).and_then(|l| l.get(neuron)) {
            Some(tokens) => tokens,
            None => &[],
        }
    }

    /// Compute the union of all neurons' token sets for a given layer.
    ///
    /// Returns a sorted Vec. Useful for layer-level pruning without activation info.
    pub fn layer_union(&self, layer: usize) -> Vec<usize> {
        match self.layers.get(layer) {
            Some(neurons) => {
                // O(N log N): collect-all with exact capacity → sort_unstable → dedup.
                // Was O(N²): per-token binary_search + Vec::insert (shifts elements).
                let total: usize = neurons.iter().map(|t| t.len()).sum();
                let mut union_set: Vec<usize> = Vec::with_capacity(total);
                for tokens in neurons {
                    union_set.extend_from_slice(tokens);
                }
                union_set.sort_unstable();
                union_set.dedup();
                union_set
            }
            None => Vec::new(),
        }
    }

    /// Compute the global union of all tokens across all layers and neurons.
    ///
    /// Returns a sorted Vec.
    pub fn global_union(&self) -> Vec<usize> {
        // O(N log N): collect-all with exact capacity → sort_unstable → dedup.
        // Was O(N²): per-token binary_search + Vec::insert across all layers.
        let total: usize = self
            .layers
            .iter()
            .flat_map(|neurons| neurons.iter())
            .map(|t| t.len())
            .sum();
        let mut union_set: Vec<usize> = Vec::with_capacity(total);
        for neurons in &self.layers {
            for tokens in neurons {
                union_set.extend_from_slice(tokens);
            }
        }
        union_set.sort_unstable();
        union_set.dedup();
        union_set
    }

    /// Serialize to a simple binary format.
    ///
    /// Format:
    /// - u32 LE: num_layers
    /// - Per layer:
    ///   - u32 LE: num_neurons
    ///   - Per neuron:
    ///     - u32 LE: num_tokens
    ///     - u32 LE × num_tokens: token indices
    pub fn serialize(&self) -> Vec<u8> {
        // Pre-compute exact byte size to avoid repeated Vec growth reallocs.
        // Layout: header (4) + per layer (4) + per neuron (4 + 4 * token_count).
        let total_bytes: usize = 4 + self
            .layers
            .iter()
            .map(|neurons| {
                4 + neurons
                    .iter()
                    .map(|tokens| 4 + tokens.len() * 4)
                    .sum::<usize>()
            })
            .sum::<usize>();

        let mut buf = Vec::with_capacity(total_bytes);

        let num_layers = self.layers.len() as u32;
        buf.extend_from_slice(&num_layers.to_le_bytes());

        for neurons in &self.layers {
            let num_neurons = neurons.len() as u32;
            buf.extend_from_slice(&num_neurons.to_le_bytes());

            for tokens in neurons {
                let num_tokens = tokens.len() as u32;
                buf.extend_from_slice(&num_tokens.to_le_bytes());
                for &tok in tokens {
                    buf.extend_from_slice(&(tok as u32).to_le_bytes());
                }
            }
        }

        buf
    }

    /// Deserialize from binary format.
    pub fn deserialize(data: &[u8]) -> Result<Self, String> {
        if data.len() < 4 {
            return Err("data too short for header".to_string());
        }

        let mut offset = 0;

        let read_u32 = |data: &[u8], off: &mut usize| -> Result<u32, String> {
            if *off + 4 > data.len() {
                return Err("unexpected end of data reading u32".to_string());
            }
            let val = u32::from_le_bytes(
                data[*off..*off + 4]
                    .try_into()
                    .map_err(|e: std::array::TryFromSliceError| e.to_string())?,
            );
            *off += 4;
            Ok(val)
        };

        let num_layers = read_u32(data, &mut offset)? as usize;
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let num_neurons = read_u32(data, &mut offset)? as usize;
            let mut neurons = Vec::with_capacity(num_neurons);

            for _ in 0..num_neurons {
                let num_tokens = read_u32(data, &mut offset)? as usize;
                let mut tokens = Vec::with_capacity(num_tokens);

                for _ in 0..num_tokens {
                    let tok = read_u32(data, &mut offset)? as usize;
                    tokens.push(tok);
                }

                neurons.push(tokens);
            }

            layers.push(neurons);
        }

        Ok(Self {
            layers,
            vocab_size: 0, // Unknown from serialized data; caller sets via builder
        })
    }

    /// Get the stored vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Set the vocabulary size (used after deserialization).
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }
}

// ── Phase 4: ConstraintPruner ───────────────────────────────────────

/// ROTATE-derived ConstraintPruner for speculative decoding.
///
/// Stores per-layer, per-neuron token reachability maps. At inference time,
/// checks if a candidate token is in the reachability set of the current layer.
///
/// # Active Context
///
/// The pruner uses a thread-safe "active context" — the inference loop sets
/// the current layer and active neurons before each speculative tree build.
/// If no active context is set, falls back to the per-layer union (all neurons).
///
/// # Thread Safety
///
/// - `active_layer`: `AtomicUsize` for lock-free reads
/// - `active_neurons`: `RwLock<Vec<usize>>` for concurrent read/write
/// - `ConstraintPruner` impl is `Send + Sync`
pub struct VocabChannelPruner {
    /// Per-layer, per-neuron token reachability
    map: VocabChannelMap,
    /// Currently active layer (set by inference loop)
    active_layer: AtomicUsize,
    /// Currently active neurons (set by inference loop)
    active_neurons: RwLock<Vec<usize>>,
    /// Per-layer union of all neuron reachability (fallback when no neurons set)
    per_layer_union: Vec<Vec<usize>>,
}

impl VocabChannelPruner {
    /// Create a new pruner from a pre-built VocabChannelMap.
    pub fn new(map: VocabChannelMap) -> Self {
        let per_layer_union: Vec<Vec<usize>> =
            (0..map.layer_count()).map(|l| map.layer_union(l)).collect();

        Self {
            map,
            active_layer: AtomicUsize::new(0),
            active_neurons: RwLock::new(Vec::new()),
            per_layer_union,
        }
    }

    /// Set the current active context for the pruner.
    ///
    /// Call this before each speculative tree build to inform the pruner
    /// which layer and neurons are currently active.
    pub fn set_active_context(&self, layer: usize, neurons: &[usize]) {
        self.active_layer.store(layer, AtomicOrdering::Relaxed);
        if let Ok(mut guard) = self.active_neurons.write() {
            guard.clear();
            guard.extend_from_slice(neurons);
        }
    }

    /// Get the number of layers.
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.map.layer_count()
    }

    /// Check if a token is reachable by a specific set of neurons in a layer.
    ///
    /// A token is reachable if ANY of the given neurons has it in their reachability set.
    pub fn is_valid_with_neurons(
        &self,
        layer: usize,
        active_neurons: &[usize],
        token_idx: usize,
    ) -> bool {
        for &neuron in active_neurons {
            if self.map.is_reachable(layer, neuron, token_idx) {
                return true;
            }
        }
        false
    }

    /// Check if a token is in the layer's union reachability set.
    ///
    /// This is the fallback check when neuron activations are not available.
    pub fn is_valid_layer_union(&self, layer: usize, token_idx: usize) -> bool {
        match self.per_layer_union.get(layer) {
            Some(tokens) => tokens.binary_search(&token_idx).is_ok(),
            None => true, // Unknown layer → don't prune
        }
    }

    /// Get a reference to the underlying map.
    #[inline]
    pub fn map(&self) -> &VocabChannelMap {
        &self.map
    }
}

impl ConstraintPruner for VocabChannelPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        let layer = self.active_layer.load(AtomicOrdering::Relaxed);

        // Try neuron-specific check first
        if let Ok(neurons) = self.active_neurons.read()
            && !neurons.is_empty()
        {
            return self.is_valid_with_neurons(layer, &neurons, token_idx);
        }

        // Fallback: per-layer union
        self.is_valid_layer_union(layer, token_idx)
    }

    fn batch_is_valid(
        &self,
        _depth: usize,
        candidates: &[usize],
        _parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let layer = self.active_layer.load(AtomicOrdering::Relaxed);

        // Try neuron-specific check first
        if let Ok(neurons) = self.active_neurons.read()
            && !neurons.is_empty()
        {
            let len = candidates.len().min(results.len());
            for i in 0..len {
                results[i] = self.is_valid_with_neurons(layer, &neurons, candidates[i]);
            }
            return;
        }

        // Fallback: per-layer union batch check
        let len = candidates.len().min(results.len());
        if let Some(tokens) = self.per_layer_union.get(layer) {
            for i in 0..len {
                results[i] = tokens.binary_search(&candidates[i]).is_ok();
            }
        } else {
            // Unknown layer → don't prune
            results[..len].fill(true);
        }
    }
}

// ── Phase 4: ComposedPruner (DDTree Integration) ────────────────────

/// Compose multiple `ConstraintPruner`s — token is valid only if ALL pruners agree.
///
/// This enables stacking `VocabChannelPruner` with any existing pruner
/// (e.g., `SudokuPruner`, `EpisodePruner`) without modifying either.
///
/// # Example
///
/// ```ignore
/// let composed = ComposedPruner::new(vec![
///     Box::new(vocab_pruner),
///     Box::new(other_pruner),
/// ]);
/// build_dd_tree_pruned(&marginals, &config, &composed, false);
/// ```
pub struct ComposedPruner {
    pruners: Vec<Box<dyn ConstraintPruner>>,
    /// Cached balanced AND-tree over the sub-pruner indices.
    ///
    /// `pruners` is only ever written in the constructors below (no push/pop/
    /// mutation anywhere), so the tree shape is invariant after construction —
    /// building it once here removes an O(n) boxed-node allocation from every
    /// `batch_is_valid` call (which runs per DDTree node expansion, i.e. per
    /// token).
    #[cfg(feature = "lattice_operad")]
    and_tree: PrunerExpr,
}

impl ComposedPruner {
    /// Create a composed pruner from a list of pruners.
    ///
    /// An empty list acts as `NoPruner` (all tokens valid).
    pub fn new(pruners: Vec<Box<dyn ConstraintPruner>>) -> Self {
        #[cfg(feature = "lattice_operad")]
        let and_tree = build_and_tree(pruners.len());
        Self {
            pruners,
            #[cfg(feature = "lattice_operad")]
            and_tree,
        }
    }

    /// Create a single-pruner composition (no overhead if only one).
    pub fn single(pruner: Box<dyn ConstraintPruner>) -> Self {
        Self::new(vec![pruner])
    }

    /// Number of composed pruners.
    pub fn len(&self) -> usize {
        self.pruners.len()
    }

    /// True if no pruners are composed.
    pub fn is_empty(&self) -> bool {
        self.pruners.is_empty()
    }
}

impl ConstraintPruner for ComposedPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // ALL must agree — short-circuit on first rejection
        self.pruners
            .iter()
            .all(|p| p.is_valid(depth, token_idx, parent_tokens))
    }

    #[cfg(not(feature = "lattice_operad"))]
    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        if len == 0 {
            return;
        }

        let mut iter = self.pruners.iter();
        let Some(first) = iter.next() else {
            // No pruners → all valid.
            results[..len].fill(true);
            return;
        };

        // First pruner writes straight into `results` (removes the `fill(true)`
        // pass plus one AND-reduce pass). Sub-pruners clamp to
        // `candidates.len().min(results.len())`, so handing them `results[..len]`
        // covers exactly the same indices the old `buf` of length `len` did.
        first.batch_is_valid(depth, candidates, parent_tokens, &mut results[..len]);

        // Remaining pruners AND-reduce through a scratch buffer. Small batches
        // (the common case — DDTree fanout) stay on the stack, so the per-token
        // `vec![false; len]` heap allocation disappears.
        const STACK_CAP: usize = 128;
        let mut stack_buf = [false; STACK_CAP];
        let mut heap_buf: Vec<bool> = if len > STACK_CAP {
            vec![false; len]
        } else {
            Vec::new()
        };
        let buf: &mut [bool] = if len > STACK_CAP {
            &mut heap_buf
        } else {
            &mut stack_buf[..len]
        };

        for pruner in iter {
            pruner.batch_is_valid(depth, candidates, parent_tokens, &mut *buf);
            // `&=` on bool is the same value as `&&` (both operands already
            // evaluated) but branch-free.
            for i in 0..len {
                results[i] &= buf[i];
            }
        }
    }

    /// When `lattice_operad` feature is on, use canonical PrunerExpr composition
    /// for batch evaluation. This builds a balanced AND-tree expression,
    /// canonicalizes it (eliminating redundant evaluations via absorption/idempotency),
    /// and evaluates per-candidate. For pure AND composition, the result is identical
    /// to the ad-hoc loop, but the canonical form enables future OR/AND mixtures.
    #[cfg(feature = "lattice_operad")]
    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        use crate::lattice_operad::PrunerResult;

        let len = candidates.len().min(results.len());
        let n_p = self.pruners.len();
        if len == 0 || n_p == 0 {
            return;
        }

        // Runs the exact algorithm of `lattice_operad::ComposedPruner::
        // batch_is_valid` (same per-pruner call order, same flat P×N atom
        // buffer, same per-candidate `expr.eval`) against the *cached*
        // AND-tree. This drops three per-call allocations: the boxed
        // expression tree, the `Vec<&dyn ConstraintPruner>`, and the
        // `LatticeComposedPruner` wrapper. Results are bit-identical.
        let mut atom_batch = vec![false; n_p * len];
        for (pi, pruner) in self.pruners.iter().enumerate() {
            let row = &mut atom_batch[pi * len..pi * len + len];
            pruner.batch_is_valid(depth, &candidates[..len], parent_tokens, row);
        }

        // Per-candidate atom row — stack-resident for the realistic pruner
        // counts, so no heap traffic per token.
        const STACK_ATOMS: usize = 32;
        let mut stack_atoms = [false; STACK_ATOMS];
        let mut heap_atoms: Vec<bool> = if n_p > STACK_ATOMS {
            vec![false; n_p]
        } else {
            Vec::new()
        };
        let atom_results: &mut [bool] = if n_p > STACK_ATOMS {
            &mut heap_atoms
        } else {
            &mut stack_atoms[..n_p]
        };

        for i in 0..len {
            for pi in 0..n_p {
                atom_results[pi] = atom_batch[pi * len + i];
            }
            results[i] = matches!(self.and_tree.eval(atom_results), PrunerResult::Accept);
        }
    }
}

/// Build a balanced AND-tree PrunerExpr from N atoms.
///
/// For N=1: Atom(0)
/// For N=2: And(Atom(0), Atom(1))
/// For N=4: And(And(Atom(0), Atom(1)), And(Atom(2), Atom(3)))
///
/// Balanced trees give better short-circuit behavior than left-chained.
#[cfg(feature = "lattice_operad")]
fn build_and_tree(n: usize) -> PrunerExpr {
    match n {
        0 => PrunerExpr::Atom(0), // degenerate: single atom, always valid
        1 => PrunerExpr::Atom(0),
        _ => build_and_tree_range(0, n),
    }
}

#[cfg(feature = "lattice_operad")]
fn build_and_tree_range(start: usize, end: usize) -> PrunerExpr {
    let len = end - start;
    if len == 1 {
        PrunerExpr::Atom(start)
    } else if len == 2 {
        PrunerExpr::and(PrunerExpr::Atom(start), PrunerExpr::Atom(start + 1))
    } else {
        let mid = start + len / 2;
        PrunerExpr::and(
            build_and_tree_range(start, mid),
            build_and_tree_range(mid, end),
        )
    }
}

// ── Phase 5: Load-Time Pipeline ─────────────────────────────────────

/// Result of load-time ROTATE decomposition for a single model.
pub struct DecompositionResult {
    /// Per-layer decomposition timing.
    pub layer_timings_ms: Vec<f64>,
    /// Total decomposition time.
    pub total_ms: f64,
    /// BLAKE3 hash of the weight bytes used for decomposition.
    pub weight_hash: [u8; 32],
    /// The constructed pruner (ready for use).
    pub pruner: VocabChannelPruner,
}

/// Decompose all layers of a model's MLP output weights into vocab channels.
///
/// Runs ROTATE decomposition on each layer's `mlp_w2` matrix against `lm_head`,
/// builds a `VocabChannelMap`, and wraps it in a `VocabChannelPruner`.
///
/// Returns timing per layer, total time, and the BLAKE3 hash of all weight bytes.
pub fn decompose_model_channels(
    weights: &TransformerWeights,
    config: &Config,
    channel_config: &VocabChannelConfig,
) -> DecompositionResult {
    let mut hasher = blake3::Hasher::new();

    // Hash lm_head once
    hasher.update(bytemuck::cast_slice::<f32, u8>(&weights.lm_head));

    let n_layer = weights.layers.len();
    let mut layer_channels = Vec::with_capacity(n_layer);
    let mut layer_timings_ms = Vec::with_capacity(n_layer);

    let total_start = std::time::Instant::now();

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        // Hash this layer's mlp_w2
        hasher.update(bytemuck::cast_slice::<f32, u8>(&layer.mlp_w2));

        let layer_start = std::time::Instant::now();
        let channels = decompose_layer_channels(
            &layer.mlp_w2,
            &weights.lm_head,
            config.n_embd,
            config.mlp_hidden,
            config.vocab_size,
            channel_config,
        );
        let layer_elapsed = layer_start.elapsed();
        layer_timings_ms.push(layer_elapsed.as_secs_f64() * 1000.0);

        log::info!(
            "[vocab_channel] Layer {}/{}: {} neurons, {:.1}ms",
            layer_idx,
            n_layer,
            channels.len(),
            layer_elapsed.as_secs_f64() * 1000.0,
        );

        layer_channels.push(channels);
    }

    let total_elapsed = total_start.elapsed();
    let weight_hash = *hasher.finalize().as_bytes();

    let map = VocabChannelMap::from_channels(&layer_channels, config.vocab_size);
    let pruner = VocabChannelPruner::new(map);

    log::info!(
        "[vocab_channel] Decomposition complete: {} layers, {:.1}ms total, hash={:.16}",
        n_layer,
        total_elapsed.as_secs_f64() * 1000.0,
        weight_hash[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );

    DecompositionResult {
        layer_timings_ms,
        total_ms: total_elapsed.as_secs_f64() * 1000.0,
        weight_hash,
        pruner,
    }
}

/// Cache file header for serialized `VocabChannelMap`.
#[derive(serde::Serialize, serde::Deserialize)]
struct CacheHeader {
    /// BLAKE3 hash of the weight bytes used for decomposition.
    weight_hash: [u8; 32],
    /// Vocab size at decomposition time.
    vocab_size: usize,
    /// Layer count at decomposition time.
    layer_count: usize,
}

/// Try to load a cached `VocabChannelMap` from disk.
///
/// Returns `Some(pruner)` if the cache exists, the weight hash matches,
/// and the dimensions are compatible. Returns `None` otherwise.
pub fn load_cached_pruner(
    cache_path: &std::path::Path,
    expected_hash: &[u8; 32],
    expected_vocab_size: usize,
    expected_layer_count: usize,
) -> Option<VocabChannelPruner> {
    let data = std::fs::read(cache_path).ok()?;
    if data.len() < 4 + std::mem::size_of::<CacheHeader>() {
        log::info!("[vocab_channel] Cache file too small, skipping");
        return None;
    }

    // Header: 4 bytes JSON length (big-endian) + JSON + binary map
    let json_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if data.len() < 4 + json_len {
        log::info!("[vocab_channel] Cache file truncated, skipping");
        return None;
    }

    let header: CacheHeader = serde_json::from_slice(&data[4..4 + json_len]).ok()?;

    if &header.weight_hash != expected_hash {
        log::info!("[vocab_channel] Cache hash mismatch, skipping");
        return None;
    }
    if header.vocab_size != expected_vocab_size || header.layer_count != expected_layer_count {
        log::info!(
            "[vocab_channel] Cache dimension mismatch (expected vocab={expected_vocab_size}, layers={expected_layer_count}), skipping",
        );
        return None;
    }

    let map_bytes = &data[4 + json_len..];
    let map = VocabChannelMap::deserialize(map_bytes).ok()?;
    log::info!(
        "[vocab_channel] Loaded cached map: {} layers, {:.1}KB",
        map.layer_count(),
        map_bytes.len() as f64 / 1024.0,
    );
    Some(VocabChannelPruner::new(map))
}

/// Save a `VocabChannelMap` to disk with BLAKE3 hash verification header.
///
/// Format: `[4-byte JSON len][JSON header][binary map]`
pub fn save_pruner_cache(
    cache_path: &std::path::Path,
    weight_hash: &[u8; 32],
    pruner: &VocabChannelPruner,
    vocab_size: usize,
    layer_count: usize,
) -> std::io::Result<()> {
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let header = CacheHeader {
        weight_hash: *weight_hash,
        vocab_size,
        layer_count,
    };
    let header_json = serde_json::to_vec(&header)?;

    let map_bytes = pruner.map().serialize();

    let mut out = Vec::with_capacity(4 + header_json.len() + map_bytes.len());
    out.extend_from_slice(&(header_json.len() as u32).to_be_bytes());
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&map_bytes);

    std::fs::write(cache_path, out)?;
    log::info!(
        "[vocab_channel] Saved cache: {:.1}KB to {}",
        (4 + header_json.len() + map_bytes.len()) as f64 / 1024.0,
        cache_path.display(),
    );
    Ok(())
}

#[cfg(test)]
mod tests;

// TL;DR: ROTATE-derived ConstraintPruner for speculative decoding (Plan 228).
// - Phase 1: `skewness()`, `excess_kurtosis()`, `householder_apply()`, `vocab_project()`,
//   `iterative_token_mask()` — core math, O(n) single-pass, no alloc on hot paths.
// - Phase 2: `VocabChannelDecomposer` — kurtosis-maximizing Householder coordinate descent
//   discovers vocabulary channels per neuron; `decompose_layer_channels()` extracts column
//   weights from row-major mlp_w2 and runs per-neuron decomposition.
// - Phase 3: `VocabChannelMap` — per-layer, per-neuron sorted token sets with O(log n) binary
//   search lookup, layer/global unions, binary serialization round-trip.
// - Phase 4: `VocabChannelPruner` implements `ConstraintPruner` — thread-safe active context
//   (AtomicUsize + RwLock) for neuron-specific pruning, falls back to per-layer union.
// Feature-gated behind `vocab_channel_pruner`.
