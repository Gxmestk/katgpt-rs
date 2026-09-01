//! Native Rust port of Moka v1 (`github.com/millionco/moka`, MIT license) —
//! a real 105,353-parameter 9×9 Go policy/value network, run natively in
//! Rust instead of the browser JS/WASM runtime. Plan 563.
//!
//! Weights are vendored at `crates/katgpt-pruners/assets/moka/` (see
//! `NOTICE.md` there for license text + provenance) and embedded into the
//! binary via `include_bytes!`/`include_str!` — no runtime download, no
//! Python, no Node, no WASM.
//!
//! Architecture (`MokaGlobalResidualNetwork` in the upstream
//! `python/go_model/model.py`, confirmed against the shipped
//! `go-model.json` manifest): a 3×3 stem conv (12→32 channels) followed by
//! 12 `NestedBottleneckBlock`s (32→16→16→16→32, residual), three of which
//! (indices 3, 7, 11) add a global mean+max-pool branch before the second
//! spatial conv. Policy head: 1×1 conv (32→4) + linear (324→82, 81 board
//! points + pass). Value head: 1×1 conv (32→2) + linear (162→32) + linear
//! (32→1) + tanh.
//!
//! Quantization: symmetric per-output-channel int8 (`scale[c] =
//! max(abs(weight[c,..])) / 127`), dequantized once at load time. Biases
//! are plain float32.
//!
//! Feature encoding mirrors `python/go_model/features.py::encode_moka_features`
//! — 12 planes over the 9×9 grid: own stones, opponent stones, own/opponent
//! atari groups, own/opponent 2-liberty groups, ko point, last two plies
//! (including pass-fill planes), and a constant komi-perspective plane.
//! `GoState` has no built-in move history, so [`MokaPlayer`] tracks its own
//! (see [`MokaPlayer::observe_external_move`]).

use std::any::Any;
use std::collections::HashMap;

use fastrand::Rng;
use serde::Deserialize;

use super::players::GoPlayer;
use super::state::GoState;
use super::types::{GoAction, GoCell};
use super::utils::flood_group;
use crate::game_state::{GameState, mcts_search};

const BOARD_SIZE: usize = 9;
const BOARD_AREA: usize = 81;
const INPUT_PLANES: usize = 12;
const TRUNK_CHANNELS: usize = 32;
const BOTTLENECK_CHANNELS: usize = 16;
const NUM_BLOCKS: usize = 12;
const GLOBAL_BLOCK_INTERVAL: usize = 4;
const POLICY_CHANNELS: usize = 4;
const POLICY_MOVES: usize = 82;
const VALUE_CHANNELS: usize = 2;
/// Value head's hidden width (`scoreHiddenChannelCount` in the manifest).
const SCORE_HIDDEN_CHANNELS: usize = 32;
/// Moka's own training-time komi convention (`KOMI_POINTS` in upstream
/// `config.py`) — deliberately NOT this repo's `GoState::komi` default
/// (7.5) or the self-play-converged 42 (Plan 091). The komi feature plane
/// must match what the network was trained on, independent of what komi
/// the surrounding match is actually played under.
const MOKA_KOMI: f32 = 7.0;
const KOMI_NORMALIZATION: f32 = 15.0;

static MANIFEST_JSON: &str = include_str!("../../assets/moka/go-model.json");
static WEIGHTS_BIN: &[u8] = include_bytes!("../../assets/moka/go-model.bin");

// ── Manifest parsing ────────────────────────────────────────────

#[derive(Deserialize)]
struct Manifest {
    tensors: HashMap<String, TensorMeta>,
}

#[derive(Deserialize)]
struct TensorMeta {
    #[serde(rename = "dataOffset")]
    data_offset: usize,
    dtype: String,
    shape: Vec<usize>,
    #[serde(rename = "scaleOffset", default)]
    scale_offset: Option<usize>,
}

fn read_f32(bytes: &[u8], offset: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let o = offset + i * 4;
            f32::from_le_bytes(bytes[o..o + 4].try_into().expect("4-byte slice"))
        })
        .collect()
}

/// Dequantize a per-output-channel symmetric int8 weight tensor: `value =
/// int8 * scale[out_channel]`, `out_channel = flat_index / (count / shape[0])`.
fn load_dequantized(tensors: &HashMap<String, TensorMeta>, bytes: &[u8], name: &str) -> Vec<f32> {
    let meta = tensors
        .get(name)
        .unwrap_or_else(|| panic!("moka manifest missing tensor {name}"));
    assert_eq!(meta.dtype, "int8", "expected int8 weight tensor {name}");
    let out_channels = meta.shape[0];
    let count: usize = meta.shape.iter().product();
    let per_channel = count / out_channels;
    let scale_offset = meta
        .scale_offset
        .unwrap_or_else(|| panic!("{name} missing scaleOffset"));
    let scales = read_f32(bytes, scale_offset, out_channels);
    let mut out = Vec::with_capacity(count);
    for (oc, &scale) in scales.iter().enumerate() {
        let base = meta.data_offset + oc * per_channel;
        for k in 0..per_channel {
            out.push((bytes[base + k] as i8) as f32 * scale);
        }
    }
    out.shrink_to_fit();
    out
}

fn load_bias(tensors: &HashMap<String, TensorMeta>, bytes: &[u8], name: &str) -> Vec<f32> {
    let meta = tensors
        .get(name)
        .unwrap_or_else(|| panic!("moka manifest missing tensor {name}"));
    assert_eq!(meta.dtype, "float32", "expected float32 bias tensor {name}");
    let count: usize = meta.shape.iter().product();
    read_f32(bytes, meta.data_offset, count)
}

// ── Weight storage ───────────────────────────────────────────────

/// Weight + bias pair, used for both conv (`w` shape `[out,kh,kw,in]`
/// flattened, out slowest / in fastest) and linear (`w` shape `[out,in]`)
/// layers — the compute functions differ, the storage doesn't.
struct Wb {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct GlobalBranch {
    hidden: Wb, // 32 -> 8
    output: Wb, // 8 -> 16 (no activation on output)
}

struct ResidualBlock {
    reduce: Wb, // 32 -> 16, 1x1
    first: Wb,  // 16 -> 16, 3x3
    global: Option<GlobalBranch>,
    second: Wb, // 16 -> 16, 3x3
    expand: Wb, // 16 -> 32, 1x1
}

pub struct MokaWeights {
    stem: Wb, // 12 -> 32, 3x3
    blocks: Vec<ResidualBlock>,
    policy_conv: Wb,   // 32 -> 4, 1x1
    policy_linear: Wb, // 324 -> 82
    value_conv: Wb,    // 32 -> 2, 1x1
    value_hidden: Wb,  // 162 -> 32
    value_output: Wb,  // 32 -> 1
}

impl MokaWeights {
    pub fn load() -> Self {
        let manifest: Manifest =
            serde_json::from_str(MANIFEST_JSON).expect("vendored moka manifest is valid JSON");
        let tensors = &manifest.tensors;
        let get = |prefix: &str| -> Wb {
            Wb {
                w: load_dequantized(tensors, WEIGHTS_BIN, &format!("{prefix}.weight")),
                b: load_bias(tensors, WEIGHTS_BIN, &format!("{prefix}.bias")),
            }
        };

        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            let prefix = format!("residual.{i}");
            let has_global = (i + 1) % GLOBAL_BLOCK_INTERVAL == 0;
            blocks.push(ResidualBlock {
                reduce: get(&format!("{prefix}.reduce")),
                first: get(&format!("{prefix}.first")),
                global: has_global.then(|| GlobalBranch {
                    hidden: get(&format!("{prefix}.global.hidden")),
                    output: get(&format!("{prefix}.global.output")),
                }),
                second: get(&format!("{prefix}.second")),
                expand: get(&format!("{prefix}.expand")),
            });
        }

        Self {
            stem: get("stem"),
            blocks,
            policy_conv: get("policy.convolution"),
            policy_linear: get("policy.linear"),
            value_conv: get("value.convolution"),
            value_hidden: get("value.hidden"),
            value_output: get("value.output"),
        }
    }
}

// ── Forward-pass primitives ─────────────────────────────────────
//
// Layout facts that the hot loop depends on: activations are HWC (channels
// contiguous innermost), and conv weights are `[out, kh, kw, in]` flattened —
// so for a fixed output channel the entire `k*k*in` kernel is ONE contiguous
// span, in the same element order as a gathered input patch.
//
// Latency rework (Plan 563 follow-up). The original implementation cost
// ~3.4 ms/forward for a 105K-param net (~5.8M MACs ⇒ only ~1.7 GMAC/s) for
// three reasons, all fixed here:
//   1. `for oc` was the OUTERMOST loop, so the input was re-read `out_ch`
//      times (32× for the stem) — cache-hostile. Now position is outermost:
//      each position's `k*k*in` neighbourhood is gathered ONCE into a
//      contiguous patch and reused across every output channel.
//   2. ~50 `vec![]` allocations per forward. Now every layer writes into a
//      caller-supplied [`MokaScratch`] buffer — zero allocation per forward.
//   3. Dot products accumulated into a single sequential `f32`, which LLVM
//      cannot auto-vectorize (FP addition is not associative). Now delegated
//      to `katgpt_types::simd::simd_dot_f32` — this workspace's hand-written
//      NEON / AVX2-FMA kernel (4 accumulators + scalar fallback), the same
//      primitive the rest of this crate already uses.
//
// (3) changes summation ORDER (and uses fused multiply-add, which rounds once
// instead of twice), so results differ from a strict sequential sum in the low
// bits. That is pinned by `optimized_conv_matches_naive_reference`, which keeps
// the original naive implementations in the test module as the equivalence
// oracle.

/// `init + dot(a, b)`, delegating to this workspace's SIMD kernel.
///
/// `katgpt_types::simd::simd_dot_f32` is hand-written NEON / AVX2-FMA
/// intrinsics with 4 independent accumulators and a scalar fallback — strictly
/// better than relying on LLVM to auto-vectorize a hand-rolled accumulator
/// loop here, and it is the primitive the rest of this crate already uses
/// (`symbolic_expression.rs`, `step_attribution_qualifier.rs`).
///
/// Length discipline: the kernel indexes via `get_unchecked`/raw pointers up
/// to `len`, so `len` MUST NOT exceed either slice. Passing
/// `a.len().min(b.len())` is the convention used at the other call sites in
/// this crate and is what keeps this off the known `dot.rs` OOB footgun.
#[inline]
fn dot_lanes(a: &[f32], b: &[f32], init: f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    init + katgpt_types::simd::simd_dot_f32(a, b, a.len().min(b.len()))
}

/// Zero-padded stride-1 convolution, HWC in/out, weight `[out,kh,kw,in]`.
/// Writes `h*w*out_ch` floats into `out`. `patch` is scratch of length
/// ≥ `k*k*in_ch`. Allocation-free.
#[allow(clippy::too_many_arguments)]
fn conv2d_into(
    input: &[f32],
    h: usize,
    w: usize,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    weight: &[f32],
    bias: &[f32],
    patch: &mut [f32],
    out: &mut [f32],
) {
    let patch_len = k * k * in_ch;
    debug_assert!(patch.len() >= patch_len);
    debug_assert!(out.len() >= h * w * out_ch);

    // 1×1 fast path (over half the convs in this net): the "patch" IS the
    // input's channel span at that position, so skip the gather entirely.
    if k == 1 {
        for pos in 0..h * w {
            let pslice = &input[pos * in_ch..pos * in_ch + in_ch];
            let obase = pos * out_ch;
            for oc in 0..out_ch {
                let wbase = oc * in_ch;
                out[obase + oc] = dot_lanes(pslice, &weight[wbase..wbase + in_ch], bias[oc]);
            }
        }
        return;
    }

    let pad = k / 2;
    for y in 0..h {
        for x in 0..w {
            // Gather the k×k×in_ch neighbourhood once, in (ky,kx,c) order —
            // matching the weight layout — so each output channel becomes a
            // single contiguous dot product. Zero-fill covers the padding.
            patch[..patch_len].fill(0.0);
            for ky in 0..k {
                let iy = y + ky;
                if iy < pad || iy >= h + pad {
                    continue;
                }
                let iy = iy - pad;
                for kx in 0..k {
                    let ix = x + kx;
                    if ix < pad || ix >= w + pad {
                        continue;
                    }
                    let ix = ix - pad;
                    let src = (iy * w + ix) * in_ch;
                    let dst = (ky * k + kx) * in_ch;
                    patch[dst..dst + in_ch].copy_from_slice(&input[src..src + in_ch]);
                }
            }

            let obase = (y * w + x) * out_ch;
            let pslice = &patch[..patch_len];
            for oc in 0..out_ch {
                let wbase = oc * patch_len;
                out[obase + oc] = dot_lanes(pslice, &weight[wbase..wbase + patch_len], bias[oc]);
            }
        }
    }
}

/// `weight` shape `[out_dim, in_dim]` flattened (row-major, out slowest).
/// Writes `out_dim` floats into `out`. Allocation-free.
fn linear_into(
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert!(out.len() >= out_dim);
    for o in 0..out_dim {
        let base = o * in_dim;
        out[o] = dot_lanes(&input[..in_dim], &weight[base..base + in_dim], bias[o]);
    }
}

fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// `concat(mean(x, axis=HW), max(x, axis=HW))` into `out` (length `2*ch`).
fn global_mean_max_into(x: &[f32], h: usize, w: usize, ch: usize, out: &mut [f32]) {
    debug_assert!(out.len() >= ch * 2);
    let (mean, max) = out[..ch * 2].split_at_mut(ch);
    mean.fill(0.0);
    max.fill(f32::MIN);
    for pos in 0..h * w {
        let row = &x[pos * ch..pos * ch + ch];
        for c in 0..ch {
            let v = row[c];
            mean[c] += v;
            if v > max[c] {
                max[c] = v;
            }
        }
    }
    let n = (h * w) as f32;
    for m in mean.iter_mut() {
        *m /= n;
    }
}

/// Reusable per-forward working set. Sized for the fixed 9×9 Moka topology,
/// allocated once and reused across every forward pass — this is what makes
/// the search players affordable (they run one forward per visited node).
pub struct MokaScratch {
    trunk: Vec<f32>,    // 81*32 — residual stream
    expand: Vec<f32>,   // 81*32 — block expand output
    hidden_a: Vec<f32>, // 81*16
    hidden_b: Vec<f32>, // 81*16
    head4: Vec<f32>,    // 81*4  — policy conv
    head2: Vec<f32>,    // 81*2  — value conv
    patch: Vec<f32>,    // 3*3*32 — conv gather scratch (max over all layers)
    pooled: Vec<f32>,   // 32 — global mean+max concat
    gh: Vec<f32>,       // 8  — global hidden
    gbias: Vec<f32>,    // 16 — global bias output
    value_h: Vec<f32>,  // 32 — value hidden
    policy: Vec<f32>,   // 82 — policy logits
}

impl MokaScratch {
    pub fn new() -> Self {
        Self {
            trunk: vec![0.0; BOARD_AREA * TRUNK_CHANNELS],
            expand: vec![0.0; BOARD_AREA * TRUNK_CHANNELS],
            hidden_a: vec![0.0; BOARD_AREA * BOTTLENECK_CHANNELS],
            hidden_b: vec![0.0; BOARD_AREA * BOTTLENECK_CHANNELS],
            head4: vec![0.0; BOARD_AREA * POLICY_CHANNELS],
            head2: vec![0.0; BOARD_AREA * VALUE_CHANNELS],
            patch: vec![0.0; 3 * 3 * TRUNK_CHANNELS],
            pooled: vec![0.0; BOTTLENECK_CHANNELS * 2],
            gh: vec![0.0; 8],
            gbias: vec![0.0; BOTTLENECK_CHANNELS],
            value_h: vec![0.0; SCORE_HIDDEN_CHANNELS],
            policy: vec![0.0; POLICY_MOVES],
        }
    }
}

impl Default for MokaScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Stem conv (12→32, 3×3, pad 1) + ReLU, into `scratch.trunk`. Split out so
/// the parity test (Plan 563 T7) can check it against a hand-derived value
/// without running the full 12-block network.
fn stem_forward_into(
    weights: &MokaWeights,
    features: &[f32],
    patch: &mut [f32],
    trunk: &mut [f32],
) {
    conv2d_into(
        features,
        BOARD_SIZE,
        BOARD_SIZE,
        INPUT_PLANES,
        TRUNK_CHANNELS,
        3,
        &weights.stem.w,
        &weights.stem.b,
        patch,
        trunk,
    );
    relu_inplace(&mut trunk[..BOARD_AREA * TRUNK_CHANNELS]);
}

/// Run the full network using a caller-owned scratch (zero allocation).
/// Returns (82 policy logits — 81 board points row-major + pass at index 81,
/// current-player-perspective value in `[-1,1]`).
pub fn forward_with_scratch(
    weights: &MokaWeights,
    features: &[f32],
    scratch: &mut MokaScratch,
) -> ([f32; POLICY_MOVES], f32) {
    // Destructure so each buffer is an independent &mut — avoids aliasing
    // complaints when several are live across one call.
    let MokaScratch {
        trunk,
        expand,
        hidden_a,
        hidden_b,
        head4,
        head2,
        patch,
        pooled,
        gh,
        gbias,
        value_h,
        policy,
    } = scratch;

    stem_forward_into(weights, features, patch, trunk);

    for block in &weights.blocks {
        // reduce 32→16 (1×1), first 16→16 (3×3)
        conv2d_into(
            trunk,
            BOARD_SIZE,
            BOARD_SIZE,
            TRUNK_CHANNELS,
            BOTTLENECK_CHANNELS,
            1,
            &block.reduce.w,
            &block.reduce.b,
            patch,
            hidden_a,
        );
        relu_inplace(hidden_a);
        conv2d_into(
            hidden_a,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            BOTTLENECK_CHANNELS,
            3,
            &block.first.w,
            &block.first.b,
            patch,
            hidden_b,
        );
        relu_inplace(hidden_b);

        if let Some(g) = &block.global {
            global_mean_max_into(
                hidden_b,
                BOARD_SIZE,
                BOARD_SIZE,
                BOTTLENECK_CHANNELS,
                pooled,
            );
            linear_into(
                pooled,
                BOTTLENECK_CHANNELS * 2,
                g.hidden.b.len(),
                &g.hidden.w,
                &g.hidden.b,
                gh,
            );
            relu_inplace(&mut gh[..g.hidden.b.len()]);
            // No activation on the global bias output.
            linear_into(
                gh,
                g.hidden.b.len(),
                BOTTLENECK_CHANNELS,
                &g.output.w,
                &g.output.b,
                gbias,
            );
            for pos in 0..BOARD_AREA {
                let row = &mut hidden_b[pos * BOTTLENECK_CHANNELS..(pos + 1) * BOTTLENECK_CHANNELS];
                for c in 0..BOTTLENECK_CHANNELS {
                    row[c] += gbias[c];
                }
            }
        }

        // second 16→16 (3×3) — reuse hidden_a — then expand 16→32 (1×1)
        conv2d_into(
            hidden_b,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            BOTTLENECK_CHANNELS,
            3,
            &block.second.w,
            &block.second.b,
            patch,
            hidden_a,
        );
        relu_inplace(hidden_a);
        conv2d_into(
            hidden_a,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            TRUNK_CHANNELS,
            1,
            &block.expand.w,
            &block.expand.b,
            patch,
            expand,
        );

        // Residual add + ReLU, in place on the trunk.
        for i in 0..BOARD_AREA * TRUNK_CHANNELS {
            let v = trunk[i] + expand[i];
            trunk[i] = if v < 0.0 { 0.0 } else { v };
        }
    }

    conv2d_into(
        trunk,
        BOARD_SIZE,
        BOARD_SIZE,
        TRUNK_CHANNELS,
        POLICY_CHANNELS,
        1,
        &weights.policy_conv.w,
        &weights.policy_conv.b,
        patch,
        head4,
    );
    relu_inplace(head4);
    linear_into(
        head4,
        POLICY_CHANNELS * BOARD_AREA,
        POLICY_MOVES,
        &weights.policy_linear.w,
        &weights.policy_linear.b,
        policy,
    );

    conv2d_into(
        trunk,
        BOARD_SIZE,
        BOARD_SIZE,
        TRUNK_CHANNELS,
        VALUE_CHANNELS,
        1,
        &weights.value_conv.w,
        &weights.value_conv.b,
        patch,
        head2,
    );
    relu_inplace(head2);
    let value_hidden_dim = weights.value_hidden.b.len();
    linear_into(
        head2,
        VALUE_CHANNELS * BOARD_AREA,
        value_hidden_dim,
        &weights.value_hidden.w,
        &weights.value_hidden.b,
        value_h,
    );
    relu_inplace(&mut value_h[..value_hidden_dim]);
    let mut value_out = [0f32; 1];
    linear_into(
        value_h,
        value_hidden_dim,
        1,
        &weights.value_output.w,
        &weights.value_output.b,
        &mut value_out,
    );

    let mut logits = [0f32; POLICY_MOVES];
    logits.copy_from_slice(&policy[..POLICY_MOVES]);
    (logits, value_out[0].tanh())
}

/// Convenience wrapper that allocates a scratch per call. Fine for one-off
/// use and tests; hot paths should hold a [`MokaScratch`] and call
/// [`forward_with_scratch`] instead.
pub fn forward(weights: &MokaWeights, features: &[f32]) -> ([f32; POLICY_MOVES], f32) {
    let mut scratch = MokaScratch::new();
    forward_with_scratch(weights, features, &mut scratch)
}

// ── Feature encoding (mirrors `features.py::encode_moka_features`) ────────

/// Encode `state` + the last two plies (`history`, most-recent last; `None`
/// = pass) into Moka's 12-plane `9*9*12` HWC feature tensor.
pub fn encode_features(state: &GoState, history: &[Option<(usize, usize)>]) -> Vec<f32> {
    debug_assert_eq!(state.size, BOARD_SIZE, "moka_net is a 9x9-only network");
    let size = state.size;
    let idx = |row: usize, col: usize, plane: usize| (row * size + col) * INPUT_PLANES + plane;
    let mut feats = vec![0f32; size * size * INPUT_PLANES];

    let mut visited = vec![false; size * size];
    for pos in 0..size * size {
        let color = state.board[pos];
        if color == GoCell::Empty {
            continue;
        }
        let is_current = color == state.to_play;
        let (row, col) = (pos / size, pos % size);
        feats[idx(row, col, if is_current { 0 } else { 1 })] = 1.0;

        if visited[pos] {
            continue;
        }
        let (stones, liberties) = flood_group(&state.board, pos, size);
        for &s in &stones {
            visited[s] = true;
        }
        let liberty_count = liberties.len();
        if liberty_count != 1 && liberty_count != 2 {
            continue;
        }
        let plane = match (is_current, liberty_count) {
            (true, 1) => 2,
            (false, 1) => 3,
            (true, 2) => 4,
            (false, 2) => 5,
            _ => unreachable!(),
        };
        for &s in &stones {
            let (r, c) = (s / size, s % size);
            feats[idx(r, c, plane)] = 1.0;
        }
    }

    if let Some(ko) = state.ko_point {
        let (r, c) = (ko / size, ko % size);
        feats[idx(r, c, 6)] = 1.0;
    }

    for offset in 1..=2usize {
        if history.len() < offset {
            continue;
        }
        if let Some((r, c)) = history[history.len() - offset] {
            let plane = if offset == 1 { 7 } else { 8 };
            feats[idx(r, c, plane)] = 1.0;
        } else {
            // Whole-plane fill marking "the move `offset` plies ago was a pass".
            let plane = 8 + offset;
            for row in 0..size {
                for col in 0..size {
                    feats[idx(row, col, plane)] = 1.0;
                }
            }
        }
    }

    // Black never receives komi (perspective -komi), White does (+komi) —
    // matches upstream's `next_color` sign convention (+1 first player).
    let next_color: f32 = if state.to_play == GoCell::Black {
        1.0
    } else {
        -1.0
    };
    let komi_value = (-MOKA_KOMI * next_color) / KOMI_NORMALIZATION;
    for row in 0..size {
        for col in 0..size {
            feats[idx(row, col, 11)] = komi_value;
        }
    }

    feats
}

// ── GoPlayer wiring ──────────────────────────────────────────────

/// Plays real Moka v1 weights via a native Rust forward pass — greedy
/// (argmax) policy, no search, matching Moka's own `temperature=0.0` arena
/// convention. Tracks its own last-two-plies history (`GoState` doesn't
/// carry one) — callers running a bespoke match loop MUST call
/// [`observe_external_move`](Self::observe_external_move) after every ply
/// the opponent makes, or the last-move feature planes will silently go
/// stale.
pub struct MokaPlayer {
    weights: MokaWeights,
    /// Held across moves so the forward pass allocates nothing per move.
    scratch: MokaScratch,
    history: Vec<Option<(usize, usize)>>,
}

impl MokaPlayer {
    pub fn new() -> Self {
        Self {
            weights: MokaWeights::load(),
            scratch: MokaScratch::new(),
            history: Vec::new(),
        }
    }

    /// Record a ply made by the OTHER player in a bespoke match loop (this
    /// player's own moves are recorded automatically inside `select_move`).
    pub fn observe_external_move(&mut self, action: &GoAction) {
        self.push_history(action);
    }

    fn push_history(&mut self, action: &GoAction) {
        self.history.push(match *action {
            GoAction::Place(r, c) => Some((r, c)),
            GoAction::Pass => None,
        });
    }
}

impl Default for MokaPlayer {
    fn default() -> Self {
        Self::new()
    }
}

impl GoPlayer for MokaPlayer {
    fn select_move(
        &mut self,
        state: &GoState,
        legal_moves: &[(usize, usize)],
        _rng: &mut Rng,
    ) -> GoAction {
        let features = encode_features(state, &self.history);
        let (policy, _value) = forward_with_scratch(&self.weights, &features, &mut self.scratch);

        let mut best_logit = policy[BOARD_AREA]; // pass logit
        let mut best_move: Option<(usize, usize)> = None;
        for &(r, c) in legal_moves {
            let logit = policy[r * BOARD_SIZE + c];
            if logit > best_logit {
                best_logit = logit;
                best_move = Some((r, c));
            }
        }

        let action = match best_move {
            Some((r, c)) => GoAction::Place(r, c),
            None => GoAction::Pass,
        };
        self.push_history(&action);
        action
    }

    fn name(&self) -> &'static str {
        "Moka"
    }

    fn reset(&mut self) {
        self.history.clear();
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Neuro-symbolic composition: Moka value head as an MCTS leaf evaluator ──
//
// go_11_moka_head_to_head (Plan 563) showed pure hand-crafted heuristics lose
// every game to Moka's real weights. Rather than out-heuristic a trained net,
// compose it with search: our generic `mcts_search` already accepts any
// `Fn(&GoState, u8) -> f32` as its non-terminal-state evaluator (see
// `GoMctsPlayer` in `players.rs`, which plugs in `GoHeuristic` the same way).
// `MokaHeuristic` plugs Moka's real value head into that exact slot instead —
// symbolic UCB1 tree search + a frozen neural evaluator, zero additional
// training.
//
// Caveat (honest, not hidden): `mcts_search`'s rollout nodes are hypothetical
// future states with no tracked move history (unlike `MokaPlayer`, which
// tracks its own history across real plies). `MokaHeuristic::evaluate` always
// encodes with an empty history, so the last-two-plies feature planes (7–10)
// read as "game just started" at every rollout leaf. This degrades Moka's own
// accuracy somewhat (it was trained conditioned on recent moves) but is an
// honest, bounded approximation — not a silent bug — and still gives MCTS a
// vastly better leaf evaluator than random-playout-to-terminal or a linear
// hand-tuned formula.

/// Moka's value head as a [`GameState`] rollout/leaf evaluator. Terminal
/// states use exact Tromp-Taylor scoring (via [`GameState::reward`], same as
/// `GoHeuristic`); non-terminal states use Moka's real forward pass.
pub struct MokaHeuristic {
    weights: MokaWeights,
    /// `mcts_search` hands the heuristic out as `&dyn Fn(&GoState, u8) -> f32`,
    /// so `evaluate` only gets `&self` — but the forward pass needs a mutable
    /// scratch buffer. `RefCell` keeps the buffer reusable (no per-node
    /// allocation) without changing the shared search API. Single-threaded by
    /// construction: `mcts_search` evaluates one leaf at a time.
    scratch: std::cell::RefCell<MokaScratch>,
}

impl MokaHeuristic {
    pub fn new() -> Self {
        Self {
            weights: MokaWeights::load(),
            scratch: std::cell::RefCell::new(MokaScratch::new()),
        }
    }

    /// Evaluate `state` for `player_id` on the same `[0.0, 1.0]` scale as
    /// `GameState::reward` (1.0 = win, 0.5 = neutral/draw, 0.0 = loss).
    pub fn evaluate(&self, state: &GoState, player_id: u8) -> f32 {
        if state.is_terminal() {
            return state.reward(player_id);
        }
        // No rollout history available here (see module note above) —
        // encode with an empty history rather than a stale/wrong one.
        let features = encode_features(state, &[]);
        // value: to_play's perspective, [-1,1]
        let (_, value) =
            forward_with_scratch(&self.weights, &features, &mut self.scratch.borrow_mut());
        let value_for_player = if GoCell::from_player_id(player_id) == state.to_play {
            value
        } else {
            -value
        };
        (value_for_player + 1.0) / 2.0
    }
}

impl Default for MokaHeuristic {
    fn default() -> Self {
        Self::new()
    }
}

/// `GoMctsPlayer`'s exact shape, but with [`MokaHeuristic`] in place of
/// `GoHeuristic` — MCTS search over Moka's own value head. Every heuristic
/// call runs a full native forward pass (~3 ms, see Plan 563 latency table),
/// so keep `budget` modest relative to plain `GoMctsPlayer` or per-move cost
/// grows fast (worst case ~`budget` forward passes if rollouts terminate
/// early; typically far fewer since each rollout only evaluates once, at
/// its depth cutoff or terminal state).
pub struct GoMctsMokaPlayer {
    budget: usize,
    rollout_depth: usize,
    heuristic: MokaHeuristic,
}

impl GoMctsMokaPlayer {
    pub fn new(budget: usize, rollout_depth: usize) -> Self {
        Self {
            budget,
            rollout_depth,
            heuristic: MokaHeuristic::new(),
        }
    }
}

impl GoPlayer for GoMctsMokaPlayer {
    fn select_move(
        &mut self,
        state: &GoState,
        legal_moves: &[(usize, usize)],
        rng: &mut Rng,
    ) -> GoAction {
        if legal_moves.is_empty() {
            return GoAction::Pass;
        }
        if legal_moves.len() == 1 {
            let (r, c) = legal_moves[0];
            return GoAction::Place(r, c);
        }

        let player_id = state.to_play.player_id();
        let heuristic = &self.heuristic;
        let heuristic_fn = |s: &GoState, pid: u8| heuristic.evaluate(s, pid);

        mcts_search(
            state,
            player_id,
            self.budget,
            self.rollout_depth,
            &heuristic_fn,
            rng,
        )
    }

    fn name(&self) -> &'static str {
        "MCTS-Moka"
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Policy-pruned negamax: Moka policy prior + Moka value leaves ──────────
//
// Diagnosis from the two failed MCTS attempts (see `.docs/06_game_arenas/
// go_arena.md`): plain UCB1 with budget ≈ branching factor (~50-80 legal
// moves on 9x9) gives each candidate move ~1 visit — far too few to
// differentiate arms, regardless of leaf-eval quality. Both attempts also
// ignored Moka's *policy* head entirely, using only its value head.
//
// This is the missing half, done directly: Moka's policy logits order and
// prune the move list (top-K beam), Moka's value head evaluates leaves, and
// a standard alpha-beta negamax does the lookahead. That is the same
// "policy prunes the tree, value judges the leaves" structure PUCT provides,
// without rewriting the generic UCB1 `mcts_search` into PUCT.
//
// Why this *could* beat greedy Moka (policy improvement): raw `MokaPlayer`
// picks argmax over its policy head with zero lookahead. Searching D plies
// deep and backing up real value estimates is a strict refinement of that
// same policy — the classic reason AlphaZero-with-search outplays its own
// raw policy network. It is not guaranteed (a weak/miscalibrated value head
// can make search actively worse), which is exactly what the benchmark
// settles.
//
// It also fixes the fidelity flaw in `MokaHeuristic`: history IS threaded
// through the search here, so the last-two-plies feature planes (7-10) are
// correct at every node instead of always reading "game just started".

/// Last-two-plies move history in the layout `encode_features` expects
/// (`[two_plies_ago, one_ply_ago]`, most recent last), with an explicit
/// length so a 1-move game passes a 1-element slice. Passing a fixed
/// 2-element slice with a `None` filler would be read as "that ply was a
/// pass" — a silent feature corruption this type exists to prevent.
#[derive(Clone, Copy)]
struct SearchHistory {
    buf: [Option<(usize, usize)>; 2],
    len: usize,
}

impl SearchHistory {
    fn from_slice(history: &[Option<(usize, usize)>]) -> Self {
        match history.len() {
            0 => Self {
                buf: [None, None],
                len: 0,
            },
            1 => Self {
                buf: [history[0], None],
                len: 1,
            },
            n => Self {
                buf: [history[n - 2], history[n - 1]],
                len: 2,
            },
        }
    }

    fn push(&self, action: &GoAction) -> Self {
        let m = match *action {
            GoAction::Place(r, c) => Some((r, c)),
            GoAction::Pass => None,
        };
        match self.len {
            0 => Self {
                buf: [m, None],
                len: 1,
            },
            _ => Self {
                buf: [self.buf[self.len - 1], m],
                len: 2,
            },
        }
    }

    #[inline]
    fn as_slice(&self) -> &[Option<(usize, usize)>] {
        &self.buf[..self.len]
    }
}

/// Alpha-beta negamax over Moka's own policy-ordered top-K moves, with
/// Moka's value head at the leaves. `depth` is in plies; `top_k` is the
/// branching factor per node. Cost is ~1 forward pass per visited node
/// (bounded by `top_k^depth`, reduced by alpha-beta pruning), so keep both
/// modest — at ~3 ms/forward, `depth=2, top_k=8` is ~75 passes ≈ 220 ms/move
/// worst case.
pub struct GoMokaSearchPlayer {
    weights: MokaWeights,
    depth: usize,
    top_k: usize,
    /// Reused across every visited node — this is the difference between
    /// ~50 allocations per node and zero.
    scratch: MokaScratch,
    history: Vec<Option<(usize, usize)>>,
    /// Diagnostic: total leaf/internal nodes evaluated across the game.
    nodes_evaluated: usize,
}

impl GoMokaSearchPlayer {
    pub fn new(depth: usize, top_k: usize) -> Self {
        Self {
            weights: MokaWeights::load(),
            depth: depth.max(1),
            top_k: top_k.max(1),
            scratch: MokaScratch::new(),
            history: Vec::new(),
            nodes_evaluated: 0,
        }
    }

    pub fn observe_external_move(&mut self, action: &GoAction) {
        self.history.push(match *action {
            GoAction::Place(r, c) => Some((r, c)),
            GoAction::Pass => None,
        });
    }

    #[inline]
    pub fn nodes_evaluated(&self) -> usize {
        self.nodes_evaluated
    }

    /// Terminal value for `state.to_play`, rescaled from `reward`'s
    /// `[0,1]` (1=win, 0.5=draw, 0=loss) to negamax's `[-1,1]`.
    #[inline]
    fn terminal_value(state: &GoState) -> f32 {
        2.0 * state.reward(state.to_play.player_id()) - 1.0
    }

    /// Candidate actions at a node: Moka's top-`top_k` legal placements by
    /// policy logit, plus `Pass` (whose logit lives at index 81) when it
    /// ranks among them. Pass must stay a real candidate — it is how the
    /// endgame is played, and search correctly evaluates its consequence
    /// (two passes ⇒ terminal ⇒ exact score).
    fn candidates(&self, state: &GoState, policy: &[f32; POLICY_MOVES]) -> Vec<GoAction> {
        let legal = state.legal_moves();
        let mut scored: Vec<(f32, GoAction)> = legal
            .iter()
            .map(|&(r, c)| (policy[r * BOARD_SIZE + c], GoAction::Place(r, c)))
            .collect();
        scored.push((policy[BOARD_AREA], GoAction::Pass));
        // Descending by policy logit; NaN-safe (treats NaN as lowest).
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(self.top_k);
        scored.into_iter().map(|(_, a)| a).collect()
    }

    /// Returns the value of `state` from `state.to_play`'s perspective.
    fn negamax(
        &mut self,
        state: &GoState,
        hist: SearchHistory,
        depth: usize,
        mut alpha: f32,
        beta: f32,
    ) -> f32 {
        if state.is_terminal() {
            return Self::terminal_value(state);
        }

        let features = encode_features(state, hist.as_slice());
        let (policy, value) = forward_with_scratch(&self.weights, &features, &mut self.scratch);
        self.nodes_evaluated += 1;

        if depth == 0 {
            return value;
        }

        let mut best = f32::NEG_INFINITY;
        for action in self.candidates(state, &policy) {
            let child = state.advance(&action, state.to_play.player_id());
            let child_hist = hist.push(&action);
            // Child's value is from the opponent's perspective — negate.
            let score = -self.negamax(&child, child_hist, depth - 1, -beta, -alpha);
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break; // opponent won't allow this line
            }
        }

        // No candidate produced a value (shouldn't happen — `candidates`
        // always yields at least `Pass`), fall back to the static eval.
        if best == f32::NEG_INFINITY {
            value
        } else {
            best
        }
    }
}

impl GoPlayer for GoMokaSearchPlayer {
    fn select_move(
        &mut self,
        state: &GoState,
        legal_moves: &[(usize, usize)],
        _rng: &mut Rng,
    ) -> GoAction {
        if legal_moves.is_empty() {
            let action = GoAction::Pass;
            self.observe_external_move(&action);
            return action;
        }

        let root_hist = SearchHistory::from_slice(&self.history);
        let features = encode_features(state, root_hist.as_slice());
        let (policy, _) = forward_with_scratch(&self.weights, &features, &mut self.scratch);
        self.nodes_evaluated += 1;

        let mut best_action = GoAction::Pass;
        let mut best_score = f32::NEG_INFINITY;
        let mut alpha = f32::NEG_INFINITY;

        for action in self.candidates(state, &policy) {
            let child = state.advance(&action, state.to_play.player_id());
            let child_hist = root_hist.push(&action);
            let score = -self.negamax(
                &child,
                child_hist,
                self.depth - 1,
                f32::NEG_INFINITY,
                -alpha,
            );
            if score > best_score {
                best_score = score;
                best_action = action;
            }
            if best_score > alpha {
                alpha = best_score;
            }
        }

        self.observe_external_move(&best_action);
        best_action
    }

    fn name(&self) -> &'static str {
        "Moka-Search"
    }

    fn reset(&mut self) {
        self.history.clear();
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Opening-book wrapper (research: does star-point opening beat pure search?) ──
//
// Wraps `GoMokaSearchPlayer`. For the first `opening_moves` plies where a
// star point is available and empty, plays it deterministically; otherwise
// delegates to the inner search player. The star-point set mirrors
// `OpeningBookStrategy::compute_star_points` from riir-router (corner 4-4,
// 3-3, side stars, center), adapted to GoPlayer's (row, col) shape.
//
// Hypothesis under test: on 9×9, Moka's policy already plays good corner
// openings (it was trained on 9×9), so forcing star points may not help —
// could even hurt by overriding the policy's contextual preferences. Null
// result is a valid research outcome.
pub struct GoOpeningBookSearchPlayer {
    inner: GoMokaSearchPlayer,
    opening_moves: usize,
    star_points: Vec<(usize, usize)>,
}

impl GoOpeningBookSearchPlayer {
    /// Create with the same search config as `GoMokaSearchPlayer`, plus an
    /// opening phase of `opening_moves` plies where star points are forced.
    pub fn new(depth: usize, top_k: usize, opening_moves: usize) -> Self {
        Self {
            inner: GoMokaSearchPlayer::new(depth, top_k),
            opening_moves,
            star_points: Self::compute_star_points(BOARD_SIZE),
        }
    }

    /// Forward history observations to the inner player.
    pub fn observe_external_move(&mut self, action: &GoAction) {
        self.inner.observe_external_move(action);
    }

    /// Star points for an N×N board, as (row, col) pairs. Mirrors
    /// `OpeningBookStrategy::compute_star_points` from riir-router.
    fn compute_star_points(n: usize) -> Vec<(usize, usize)> {
        let mut pts = Vec::with_capacity(13);
        if n < 5 {
            return pts;
        }
        // Corner 4-4 points
        pts.extend([(3, 3), (3, n - 4), (n - 4, 3), (n - 4, n - 4)]);
        // Corner 3-3 points
        pts.extend([(2, 2), (2, n - 3), (n - 3, 2), (n - 3, n - 3)]);
        // Side star points (larger boards only)
        if n >= 13 {
            let mid = n / 2;
            pts.extend([(3, mid), (mid, 3), (n - 4, mid), (mid, n - 4)]);
        }
        // Center for odd N
        if n % 2 == 1 {
            pts.push((n / 2, n / 2));
        }
        pts
    }

    /// Count stones on the board to determine opening phase. Same heuristic
    /// as `OpeningBookStrategy::is_opening`: stones < opening_moves * 2.
    fn is_opening(&self, state: &GoState) -> bool {
        let threshold = self.opening_moves * 2;
        let mut stones = 0;
        for &cell in &state.board {
            if cell != GoCell::Empty {
                stones += 1;
                if stones >= threshold {
                    return false;
                }
            }
        }
        stones < threshold
    }

    /// First available empty star point that is also a legal move.
    fn first_legal_star(&self, state: &GoState, legal: &[(usize, usize)]) -> Option<GoAction> {
        for &(r, c) in &self.star_points {
            if state.board[r * BOARD_SIZE + c] == GoCell::Empty && legal.contains(&(r, c)) {
                return Some(GoAction::Place(r, c));
            }
        }
        None
    }
}

impl GoPlayer for GoOpeningBookSearchPlayer {
    fn select_move(
        &mut self,
        state: &GoState,
        legal_moves: &[(usize, usize)],
        rng: &mut Rng,
    ) -> GoAction {
        if legal_moves.is_empty() {
            let action = GoAction::Pass;
            self.observe_external_move(&action);
            return action;
        }

        if self.is_opening(state)
            && let Some(star) = self.first_legal_star(state, legal_moves)
        {
            self.observe_external_move(&star);
            return star;
        }

        // Out of opening or no star point available — delegate to search.
        self.inner.select_move(state, legal_moves, rng)
    }

    fn name(&self) -> &'static str {
        "Moka-OpeningBook-Search"
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── PUCT search (AlphaZero-style: policy prior + value head + MCTS) ──────
//
// The last unexplored lever for >70% win rate vs Moka greedy. Unlike the
// existing `GoMctsMokaPlayer` (UCB1 + value head — negative result, see
// go_arena.md), this uses BOTH of Moka's heads: the POLICY head provides
// the exploration prior P(s,a), and the VALUE head evaluates leaves. This
// is the AlphaZero recipe, known to extract more strength from small
// policy+value networks than fixed-depth alpha-beta.
//
// PUCT formula: a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a)) ]
// where Q(s,a) = mean action value, P(s,a) = policy prior, N = visit counts.

struct PuctNode {
    /// Move that led to this node. `Pass` for root.
    action: GoAction,
    /// Board state at this node (cloned on expansion).
    state: GoState,
    /// Visit count.
    visits: u32,
    /// Accumulated value from the perspective of the player who MOVED INTO
    /// this node (i.e., the parent's to_play). Negamax: negate at each level.
    total_value: f32,
    /// Policy prior P(s,a) from the parent's policy head evaluation.
    prior: f32,
    /// Arena indices of children.
    children: Vec<usize>,
    /// Arena index of parent. None for root.
    parent: Option<usize>,
    /// Whether this node has been expanded (policy+value evaluated, children created).
    expanded: bool,
}

impl PuctNode {
    fn new_root(state: GoState) -> Self {
        Self {
            action: GoAction::Pass,
            state,
            visits: 0,
            total_value: 0.0,
            prior: 1.0,
            children: Vec::new(),
            parent: None,
            expanded: false,
        }
    }

    #[inline]
    fn mean_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f32
        }
    }
}

pub struct GoPuctMokaPlayer {
    weights: MokaWeights,
    scratch: MokaScratch,
    history: Vec<Option<(usize, usize)>>,
    budget: usize,
    c_puct: f32,
    top_k: usize,
    /// Reused across select_move calls — avoids re-allocating the arena.
    arena: Vec<PuctNode>,
    nodes_evaluated: usize,
}

impl GoPuctMokaPlayer {
    pub fn new(budget: usize, c_puct: f32, top_k: usize) -> Self {
        Self {
            weights: MokaWeights::load(),
            scratch: MokaScratch::new(),
            history: Vec::new(),
            budget: budget.max(1),
            c_puct,
            top_k: top_k.max(1),
            arena: Vec::new(),
            nodes_evaluated: 0,
        }
    }

    pub fn observe_external_move(&mut self, action: &GoAction) {
        self.history.push(match *action {
            GoAction::Place(r, c) => Some((r, c)),
            GoAction::Pass => None,
        });
    }

    #[inline]
    pub fn nodes_evaluated(&self) -> usize {
        self.nodes_evaluated
    }

    /// Expand a node: run policy+value, create children for top_k legal moves.
    /// Returns the value head evaluation [-1,1] from this node's to_play perspective.
    fn expand(&mut self, node_idx: usize) -> f32 {
        // Collect parent chain BEFORE mutable borrow (Moka needs last-2-plies history).
        let mut hist: Vec<Option<(usize, usize)>> = Vec::with_capacity(2);
        {
            let mut chain_actions: Vec<GoAction> = Vec::with_capacity(2);
            let mut cur = Some(node_idx);
            while let Some(idx) = cur {
                if chain_actions.len() >= 2 {
                    break;
                }
                let n = &self.arena[idx];
                if n.parent.is_some() {
                    chain_actions.push(n.action);
                }
                cur = n.parent;
            }
            for a in chain_actions.iter().rev() {
                hist.push(match *a {
                    GoAction::Place(r, c) => Some((r, c)),
                    GoAction::Pass => None,
                });
            }
        }

        let node = &mut self.arena[node_idx];
        node.expanded = true;

        if node.state.is_terminal() {
            return 2.0 * node.state.reward(node.state.to_play.player_id()) - 1.0;
        }

        let features = encode_features(&node.state, &hist);
        let (policy, value) = forward_with_scratch(&self.weights, &features, &mut self.scratch);
        self.nodes_evaluated += 1;

        // Snapshot everything we need from node, then drop the mutable borrow.
        let player = node.state.to_play;
        let parent_state = node.state.clone();
        let legal = parent_state.legal_moves();
        let mut scored: Vec<(f32, GoAction)> = legal
            .iter()
            .map(|&(r, c)| (policy[r * BOARD_SIZE + c], GoAction::Place(r, c)))
            .collect();
        scored.push((policy[BOARD_AREA], GoAction::Pass));
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(self.top_k);

        // Softmax the top_k priors for normalized P(s,a).
        let max_logit = scored
            .iter()
            .map(|(l, _)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scored.iter().map(|(l, _)| (l - max_logit).exp()).sum();
        let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 1.0 };

        // Mutable borrow is now over (scored is owned). Safe to push children.
        let children_start = self.arena.len();
        for (logit, action) in &scored {
            let prior = (logit - max_logit).exp() * inv_exp_sum;
            let child_state = parent_state.advance(action, player.player_id());
            self.arena.push(PuctNode {
                action: *action,
                state: child_state,
                visits: 0,
                total_value: 0.0,
                prior,
                children: Vec::new(),
                parent: Some(node_idx),
                expanded: false,
            });
        }
        let children_end = self.arena.len();
        self.arena[node_idx]
            .children
            .extend(children_start..children_end);

        value
    }

    /// Selection: traverse from root to first unexpanded leaf using PUCT.
    /// Returns the leaf node index.
    fn select(&self, root: usize) -> usize {
        let mut cur = root;
        loop {
            let node = &self.arena[cur];
            if !node.expanded || node.children.is_empty() {
                return cur;
            }
            // Pick child with highest PUCT score.
            // Q is negated because child.total_value is from child.to_play's
            // perspective, but we're selecting from parent's perspective.
            let parent_visits = node.visits.max(1) as f32;
            let sqrt_parent = parent_visits.sqrt();
            let mut best_idx = node.children[0];
            let mut best_score = f32::NEG_INFINITY;
            for &child_idx in &node.children {
                let child = &self.arena[child_idx];
                let q = -child.mean_value(); // negate: child's perspective → parent's
                let u = self.c_puct * child.prior * sqrt_parent / (1.0 + child.visits as f32);
                let score = q + u;
                if score > best_score {
                    best_score = score;
                    best_idx = child_idx;
                }
            }
            cur = best_idx;
        }
    }

    /// Backpropagate value from leaf to root. Negamax: negate at each level.
    /// `total_value` stores from the node's own `to_play` perspective (standard
    /// MCTS negamax convention). PUCT selection negates Q when comparing children
    /// because the parent's to_play is opposite to the child's.
    fn backprop(&mut self, leaf_idx: usize, mut value: f32) {
        let mut cur = Some(leaf_idx);
        while let Some(idx) = cur {
            let node = &mut self.arena[idx];
            node.visits += 1;
            node.total_value += value;
            value = -value; // negate for parent (opponent's perspective)
            cur = node.parent;
        }
    }
}

impl GoPlayer for GoPuctMokaPlayer {
    fn select_move(
        &mut self,
        state: &GoState,
        legal_moves: &[(usize, usize)],
        _rng: &mut Rng,
    ) -> GoAction {
        if legal_moves.is_empty() {
            let action = GoAction::Pass;
            self.observe_external_move(&action);
            return action;
        }

        // Reset arena for this move's search.
        self.arena.clear();
        self.arena.push(PuctNode::new_root(state.clone()));
        let root = 0;

        for _ in 0..self.budget {
            // 1. Selection
            let leaf = self.select(root);

            // 2. Expansion + Evaluation
            let value = self.expand(leaf);

            // 3. Backpropagation (negamax)
            self.backprop(leaf, value);
        }

        // Pick most-visited child at root.
        let root_node = &self.arena[root];
        let mut best_action = GoAction::Pass;
        let mut best_visits = 0u32;
        for &child_idx in &root_node.children {
            let child = &self.arena[child_idx];
            if child.visits > best_visits {
                best_visits = child.visits;
                best_action = child.action;
            }
        }

        self.observe_external_move(&best_action);
        best_action
    }

    fn name(&self) -> &'static str {
        "Moka-PUCT"
    }

    fn reset(&mut self) {
        self.history.clear();
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Issue 564: ANE residency probe (stem + one residual block) ──────────
//
// Scoped probe, NOT the full 40-layer graph. Builds the stem conv + one
// non-global residual block as a CoreML NeuralNetwork spec, loads it via
// `coreml-native`, times a real prediction, and checks output against the
// CPU reference (`stem_forward_into` + `residual_block_forward`) for
// correctness. Answers two questions before committing to authoring the
// full topology: (1) does a conv net this small actually land on the ANE,
// or does CoreML silently fall back to CPU; (2) is the weight/activation
// layout transpose (Moka's `[out,kh,kw,in]` HWC-native vs CoreML's
// `[out,in,kh,kw]` CHW-native) implemented correctly.
#[cfg(all(target_os = "macos", feature = "moka_ane"))]
pub mod ane_probe {
    use coreml_native as coreml;
    use coreml_proto::proto::{
        ActivationParams, ActivationReLu, AddLayerParams, ArrayFeatureType, ConvolutionLayerParams,
        FeatureDescription, FeatureType, Model, ModelDescription, NeuralNetwork,
        NeuralNetworkLayer, SamePadding, WeightParams,
        activation_params::NonlinearityType as ActivationKind, array_feature_type::ArrayDataType,
        convolution_layer_params::ConvolutionPaddingType, feature_type::Type as FeatureTypeKind,
        model::Type as ModelType, neural_network_layer::Layer as LayerKind,
    };
    use prost::Message;

    use super::{
        BOARD_AREA, BOARD_SIZE, BOTTLENECK_CHANNELS, INPUT_PLANES, MokaWeights, TRUNK_CHANNELS,
        conv2d_into, relu_inplace, stem_forward_into,
    };

    /// HWC (Moka's native layout) → CHW (CoreML's expected layout).
    fn hwc_to_chw(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
        let mut out = vec![0f32; h * w * c];
        for y in 0..h {
            for xx in 0..w {
                for ch in 0..c {
                    out[(ch * h + y) * w + xx] = x[(y * w + xx) * c + ch];
                }
            }
        }
        out
    }

    /// CHW (CoreML's output layout) → HWC, for comparison against the CPU
    /// reference (which is HWC throughout).
    fn chw_to_hwc(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
        let mut out = vec![0f32; h * w * c];
        for ch in 0..c {
            for y in 0..h {
                for xx in 0..w {
                    out[(y * w + xx) * c + ch] = x[(ch * h + y) * w + xx];
                }
            }
        }
        out
    }

    /// Moka's conv weight layout `[out,kh,kw,in]` (in fastest) → CoreML's
    /// documented `[out,in,kh,kw]` (kw fastest). Degenerates to the identity
    /// permutation when `k == 1`, so this single function correctly handles
    /// both the 3×3 convs (stem/first/second) and the 1×1 convs
    /// (reduce/expand) — no special-casing needed.
    fn transpose_conv_weight_to_coreml(
        src: &[f32],
        out_ch: usize,
        k: usize,
        in_ch: usize,
    ) -> Vec<f32> {
        let mut dst = vec![0f32; out_ch * in_ch * k * k];
        for o in 0..out_ch {
            for ky in 0..k {
                for kx in 0..k {
                    for c in 0..in_ch {
                        let src_idx = ((o * k + ky) * k + kx) * in_ch + c;
                        let dst_idx = ((o * in_ch + c) * k + ky) * k + kx;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
        dst
    }

    fn multi_array_type(shape: &[i64]) -> FeatureType {
        FeatureType {
            r#type: Some(FeatureTypeKind::MultiArrayType(ArrayFeatureType {
                shape: shape.to_vec(),
                data_type: ArrayDataType::Float32 as i32,
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    fn nn_layer(
        name: &str,
        input: &[&str],
        output: &[&str],
        layer: LayerKind,
    ) -> NeuralNetworkLayer {
        NeuralNetworkLayer {
            name: name.into(),
            input: input.iter().map(|s| (*s).into()).collect(),
            output: output.iter().map(|s| (*s).into()).collect(),
            layer: Some(layer),
            ..Default::default()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_layer(
        name: &str,
        input: &str,
        output: &str,
        out_ch: usize,
        in_ch: usize,
        k: usize,
        weight_native: &[f32],
        bias: &[f32],
    ) -> NeuralNetworkLayer {
        nn_layer(
            name,
            &[input],
            &[output],
            LayerKind::Convolution(ConvolutionLayerParams {
                output_channels: out_ch as u64,
                kernel_channels: in_ch as u64,
                n_groups: 1,
                kernel_size: vec![k as u64, k as u64],
                stride: vec![1, 1],
                has_bias: true,
                weights: Some(WeightParams {
                    float_value: transpose_conv_weight_to_coreml(weight_native, out_ch, k, in_ch),
                    ..Default::default()
                }),
                bias: Some(WeightParams {
                    float_value: bias.to_vec(),
                    ..Default::default()
                }),
                convolution_padding_type: Some(
                    ConvolutionPaddingType::Same(SamePadding::default()),
                ),
                ..Default::default()
            }),
        )
    }

    fn relu_layer(name: &str, input: &str, output: &str) -> NeuralNetworkLayer {
        nn_layer(
            name,
            &[input],
            &[output],
            LayerKind::Activation(ActivationParams {
                nonlinearity_type: Some(ActivationKind::ReLu(ActivationReLu {})),
            }),
        )
    }

    /// Build the CoreML spec for stem + residual block 0 (block 0 has no
    /// global-pooling branch — `GLOBAL_BLOCK_INTERVAL=4` means the first
    /// global block is index 3 — so this probe stays within the "no
    /// Pooling layer needed" subset by construction).
    fn build_probe_spec(weights: &MokaWeights) -> Model {
        let block = &weights.blocks[0];
        debug_assert!(block.global.is_none(), "probe assumes a non-global block");

        let layers = vec![
            conv_layer(
                "stem",
                "input",
                "stem_out",
                TRUNK_CHANNELS,
                INPUT_PLANES,
                3,
                &weights.stem.w,
                &weights.stem.b,
            ),
            relu_layer("stem_relu", "stem_out", "trunk"),
            conv_layer(
                "reduce",
                "trunk",
                "reduce_out",
                BOTTLENECK_CHANNELS,
                TRUNK_CHANNELS,
                1,
                &block.reduce.w,
                &block.reduce.b,
            ),
            relu_layer("reduce_relu", "reduce_out", "reduce_act"),
            conv_layer(
                "first",
                "reduce_act",
                "first_out",
                BOTTLENECK_CHANNELS,
                BOTTLENECK_CHANNELS,
                3,
                &block.first.w,
                &block.first.b,
            ),
            relu_layer("first_relu", "first_out", "first_act"),
            conv_layer(
                "second",
                "first_act",
                "second_out",
                BOTTLENECK_CHANNELS,
                BOTTLENECK_CHANNELS,
                3,
                &block.second.w,
                &block.second.b,
            ),
            relu_layer("second_relu", "second_out", "second_act"),
            conv_layer(
                "expand",
                "second_act",
                "expand_out",
                TRUNK_CHANNELS,
                BOTTLENECK_CHANNELS,
                1,
                &block.expand.w,
                &block.expand.b,
            ),
            nn_layer(
                "residual_add",
                &["expand_out", "trunk"],
                &["sum_out"],
                LayerKind::Add(AddLayerParams { alpha: 1.0 }),
            ),
            relu_layer("final_relu", "sum_out", "output"),
        ];

        Model {
            specification_version: 7,
            description: Some(ModelDescription {
                input: vec![FeatureDescription {
                    name: "input".into(),
                    short_description: "Moka features, CHW".into(),
                    r#type: Some(multi_array_type(&[
                        INPUT_PLANES as i64,
                        BOARD_SIZE as i64,
                        BOARD_SIZE as i64,
                    ])),
                }],
                output: vec![FeatureDescription {
                    name: "output".into(),
                    short_description: "Trunk after stem + block 0, CHW".into(),
                    r#type: Some(multi_array_type(&[
                        TRUNK_CHANNELS as i64,
                        BOARD_SIZE as i64,
                        BOARD_SIZE as i64,
                    ])),
                }],
                ..Default::default()
            }),
            is_updatable: false,
            r#type: Some(ModelType::NeuralNetwork(NeuralNetwork {
                layers,
                ..Default::default()
            })),
        }
    }

    /// Result of the residency + correctness probe.
    #[derive(Debug)]
    pub struct AneProbeResult {
        pub latency_us: u64,
        /// CPU latency for the identical 9-layer slice — a matched-workload
        /// comparison, not a proportional estimate from the full network.
        pub cpu_latency_us: u64,
        pub max_abs_diff: f32,
        pub mean_abs_diff: f32,
    }

    /// Compile stem+block0 to CoreML, run one real prediction, and diff
    /// against the CPU reference for the identical sub-graph. No hard
    /// residency threshold is asserted here — the transformer backend's
    /// 1 ms `lm_head` threshold (a single matvec) doesn't transfer to a
    /// multi-layer 9×9 conv workload; read `latency_us` and judge by eye
    /// (tens of µs to low-single-digit ms → plausibly ANE; tens of ms →
    /// plausibly CPU fallback).
    pub fn run_probe(
        weights: &MokaWeights,
        features_hwc: &[f32],
    ) -> Result<AneProbeResult, String> {
        let spec = build_probe_spec(weights);
        let bytes = spec.encode_to_vec();
        let model = coreml::Model::load_from_bytes(&bytes, coreml::ComputeUnits::All)
            .map_err(|e| format!("load_from_bytes: {e}"))?
            .block_on()
            .map_err(|e| format!("load_from_bytes block_on: {e}"))?;

        let input_chw = hwc_to_chw(features_hwc, BOARD_SIZE, BOARD_SIZE, INPUT_PLANES);
        let tensor =
            coreml::BorrowedTensor::from_f32(&input_chw, &[INPUT_PLANES, BOARD_SIZE, BOARD_SIZE])
                .map_err(|e| format!("tensor create: {e}"))?;

        let start = std::time::Instant::now();
        let prediction = model
            .predict(&[("input", &tensor)])
            .map_err(|e| format!("predict: {e}"))?;
        let latency_us = start.elapsed().as_micros() as u64;

        let (output_chw, _shape) = prediction
            .get_f32("output")
            .map_err(|e| format!("get output: {e}"))?;
        let output_hwc = chw_to_hwc(&output_chw, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS);

        // CPU reference for the IDENTICAL sub-graph (stem + non-global block),
        // built from the same standalone primitives `forward_with_scratch`
        // uses, since the block loop itself is inlined there rather than a
        // separately callable function.
        let mut patch = vec![0f32; 3 * 3 * TRUNK_CHANNELS];
        let mut trunk = vec![0f32; BOARD_AREA * TRUNK_CHANNELS];
        stem_forward_into(weights, features_hwc, &mut patch, &mut trunk);

        let block = &weights.blocks[0];
        let mut hidden_a = vec![0f32; BOARD_AREA * BOTTLENECK_CHANNELS];
        let mut hidden_b = vec![0f32; BOARD_AREA * BOTTLENECK_CHANNELS];
        let mut expand_out = vec![0f32; BOARD_AREA * TRUNK_CHANNELS];

        conv2d_into(
            &trunk,
            BOARD_SIZE,
            BOARD_SIZE,
            TRUNK_CHANNELS,
            BOTTLENECK_CHANNELS,
            1,
            &block.reduce.w,
            &block.reduce.b,
            &mut patch,
            &mut hidden_a,
        );
        relu_inplace(&mut hidden_a);
        conv2d_into(
            &hidden_a,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            BOTTLENECK_CHANNELS,
            3,
            &block.first.w,
            &block.first.b,
            &mut patch,
            &mut hidden_b,
        );
        relu_inplace(&mut hidden_b);
        conv2d_into(
            &hidden_b,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            BOTTLENECK_CHANNELS,
            3,
            &block.second.w,
            &block.second.b,
            &mut patch,
            &mut hidden_a,
        );
        relu_inplace(&mut hidden_a);
        conv2d_into(
            &hidden_a,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            TRUNK_CHANNELS,
            1,
            &block.expand.w,
            &block.expand.b,
            &mut patch,
            &mut expand_out,
        );

        let mut cpu_ref = trunk.clone();
        for (t, e) in cpu_ref.iter_mut().zip(&expand_out) {
            *t = (*t + *e).max(0.0);
        }

        let mut max_abs_diff = 0f32;
        let mut sum_abs_diff = 0f64;
        for (a, b) in output_hwc.iter().zip(&cpu_ref) {
            let d: f32 = (a - b).abs();
            max_abs_diff = max_abs_diff.max(d);
            sum_abs_diff += f64::from(d);
        }
        let mean_abs_diff = (sum_abs_diff / cpu_ref.len() as f64) as f32;

        // CPU latency for the IDENTICAL 9-layer slice (same conv/relu calls
        // as the correctness check above), so the ANE number has a fair,
        // matched-workload comparison instead of a proportional estimate
        // from the full network's latency. Warmed up, then timed.
        let cpu_iters = 200;
        for _ in 0..20 {
            stem_forward_into(weights, features_hwc, &mut patch, &mut trunk);
        }
        let cpu_start = std::time::Instant::now();
        for _ in 0..cpu_iters {
            stem_forward_into(weights, features_hwc, &mut patch, &mut trunk);
            conv2d_into(
                &trunk,
                BOARD_SIZE,
                BOARD_SIZE,
                TRUNK_CHANNELS,
                BOTTLENECK_CHANNELS,
                1,
                &block.reduce.w,
                &block.reduce.b,
                &mut patch,
                &mut hidden_a,
            );
            relu_inplace(&mut hidden_a);
            conv2d_into(
                &hidden_a,
                BOARD_SIZE,
                BOARD_SIZE,
                BOTTLENECK_CHANNELS,
                BOTTLENECK_CHANNELS,
                3,
                &block.first.w,
                &block.first.b,
                &mut patch,
                &mut hidden_b,
            );
            relu_inplace(&mut hidden_b);
            conv2d_into(
                &hidden_b,
                BOARD_SIZE,
                BOARD_SIZE,
                BOTTLENECK_CHANNELS,
                BOTTLENECK_CHANNELS,
                3,
                &block.second.w,
                &block.second.b,
                &mut patch,
                &mut hidden_a,
            );
            relu_inplace(&mut hidden_a);
            conv2d_into(
                &hidden_a,
                BOARD_SIZE,
                BOARD_SIZE,
                BOTTLENECK_CHANNELS,
                TRUNK_CHANNELS,
                1,
                &block.expand.w,
                &block.expand.b,
                &mut patch,
                &mut expand_out,
            );
            std::hint::black_box((&trunk, &expand_out));
        }
        let cpu_latency_us = (cpu_start.elapsed().as_micros() as f64 / f64::from(cpu_iters)) as u64;

        Ok(AneProbeResult {
            latency_us,
            cpu_latency_us,
            max_abs_diff,
            mean_abs_diff,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::go::moka_net::encode_features;
        use crate::go::state::GoState;

        /// Runs the real probe against real Moka weights on a mid-game
        /// position. Prints latency for manual eyeballing (`--nocapture`)
        /// and asserts the ANE output matches the CPU reference — a
        /// residency finding is only trustworthy if correctness holds too.
        #[test]
        fn ane_probe_matches_cpu_reference() {
            let weights = MokaWeights::load();
            let state = GoState::new(BOARD_SIZE);
            let features = encode_features(&state, &[]);

            let result = run_probe(&weights, &features).expect("ANE probe failed to run");
            println!(
                "Issue 564 ANE probe: ane_latency={} us, cpu_latency={} us (same 9-layer slice), ratio={:.2}x, max_abs_diff={}, mean_abs_diff={}",
                result.latency_us,
                result.cpu_latency_us,
                result.latency_us as f64 / result.cpu_latency_us as f64,
                result.max_abs_diff,
                result.mean_abs_diff
            );

            assert!(
                result.max_abs_diff < 1e-2,
                "ANE output diverges from CPU reference (max_abs_diff={}) — layout transpose is likely wrong",
                result.max_abs_diff
            );
        }
    }
}

// ── Parity tests (Plan 563 T7) ────────────────────────────────────
//
// No golden input→output fixture exists upstream (`tests/model-smoke.mjs`
// only asserts shape/finiteness, not exact values) and per project policy
// this port must be validated without shelling out to Python/MLX or Node —
// so parity here means: (1) an independent hand-derived closed-form check
// of the stem layer on an all-zero-input-plane board, computed by
// re-reading the manifest via a generic `serde_json::Value` walk (NOT the
// `TensorMeta`/`load_dequantized` code under test — a transcription bug in
// those wouldn't be caught if the check reused them), and (2) upstream's
// own finiteness/shape smoke test, ported from JS to Rust.
#[cfg(test)]
mod tests {
    use super::*;

    /// Re-parses the manifest + dequantizes ONE tensor by hand, independent
    /// of `load_dequantized`/`TensorMeta`, to give the parity test below an
    /// oracle that isn't just calling the code it's supposed to catch bugs in.
    fn hand_dequantize(name: &str) -> (Vec<f32>, Vec<usize>) {
        let manifest: serde_json::Value =
            serde_json::from_str(MANIFEST_JSON).expect("manifest parses as generic JSON");
        let meta = &manifest["tensors"][name];
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let data_offset = meta["dataOffset"].as_u64().unwrap() as usize;
        let scale_offset = meta["scaleOffset"].as_u64().unwrap() as usize;
        let out_channels = shape[0];
        let count: usize = shape.iter().product();
        let per_channel = count / out_channels;

        let read_f32_at = |off: usize| -> f32 {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&WEIGHTS_BIN[off..off + 4]);
            f32::from_le_bytes(buf)
        };

        let mut values = vec![0f32; count];
        for oc in 0..out_channels {
            let scale = read_f32_at(scale_offset + oc * 4);
            for k in 0..per_channel {
                let byte = WEIGHTS_BIN[data_offset + oc * per_channel + k];
                let signed = i8::from_le_bytes([byte]);
                values[oc * per_channel + k] = signed as f32 * scale;
            }
        }
        (values, shape)
    }

    /// On an all-empty 9×9 board, every input plane is exactly zero EXCEPT
    /// plane 11 (the constant komi-perspective value) — no stones, no ko, no
    /// move history yet. At any interior position (padding never triggers),
    /// the stem conv reduces to a closed form:
    /// `bias[oc] + komi_value * sum_{ky,kx} weight[oc,ky,kx,11]`. This is
    /// computed here via `hand_dequantize` (independent of the module under
    /// test) and compared against `stem_forward`'s actual output.
    #[test]
    fn stem_conv_matches_hand_derived_value_on_empty_board() {
        let weights = MokaWeights::load();
        let state = GoState::new(BOARD_SIZE);
        let features = encode_features(&state, &[]);

        // Sanity: confirm the empty-board assumption the closed form relies on.
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                for plane in 0..INPUT_PLANES {
                    let v = features[(row * BOARD_SIZE + col) * INPUT_PLANES + plane];
                    if plane == 11 {
                        assert!(v != 0.0, "komi plane should be non-zero");
                    } else {
                        assert_eq!(v, 0.0, "plane {plane} should be all-zero on an empty board");
                    }
                }
            }
        }
        let komi_value = features[11]; // any position — plane 11 is constant

        let (hand_weight, shape) = hand_dequantize("stem.weight");
        let (hand_bias, _) = {
            let manifest: serde_json::Value = serde_json::from_str(MANIFEST_JSON).unwrap();
            let meta = &manifest["tensors"]["stem.bias"];
            let offset = meta["dataOffset"].as_u64().unwrap() as usize;
            let count = meta["shape"][0].as_u64().unwrap() as usize;
            let mut b = Vec::with_capacity(count);
            for i in 0..count {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&WEIGHTS_BIN[offset + i * 4..offset + i * 4 + 4]);
                b.push(f32::from_le_bytes(buf));
            }
            (b, ())
        };
        assert_eq!(shape, vec![TRUNK_CHANNELS, 3, 3, INPUT_PLANES]);

        let mut patch = vec![0f32; 3 * 3 * INPUT_PLANES];
        let mut trunk = vec![0f32; BOARD_AREA * TRUNK_CHANNELS];
        stem_forward_into(&weights, &features, &mut patch, &mut trunk);
        let (y, x) = (4usize, 4usize); // interior — all 3x3 taps in bounds

        for oc in 0..TRUNK_CHANNELS {
            let mut expected = hand_bias[oc];
            for ky in 0..3 {
                for kx in 0..3 {
                    let weight_idx = ((oc * 3 + ky) * 3 + kx) * INPUT_PLANES + 11;
                    expected += hand_weight[weight_idx] * komi_value;
                }
            }
            // ReLU (stem_forward applies it).
            expected = expected.max(0.0);
            let actual = trunk[(y * BOARD_SIZE + x) * TRUNK_CHANNELS + oc];
            assert!(
                (actual - expected).abs() < 1e-4,
                "stem output mismatch at oc={oc}: actual={actual}, expected={expected}"
            );
        }
    }

    /// Port of upstream `tests/model-smoke.mjs`: play a short opening, run
    /// inference, assert finite policy logits (82) + finite value. This is
    /// the shape/sanity check upstream itself relies on (they don't assert
    /// exact values either — only shape and finiteness).
    #[test]
    fn forward_produces_finite_output_after_opening_moves() {
        let weights = MokaWeights::load();
        let mut state = GoState::new(BOARD_SIZE);
        let mut history: Vec<Option<(usize, usize)>> = Vec::new();
        // Arbitrary legal opening (row, col) pairs, corner/side plays.
        let opening = [(2usize, 2usize), (2, 6), (6, 2), (6, 6)];
        for &(r, c) in &opening {
            state = state.advance(&GoAction::Place(r, c), state.to_play.player_id());
            history.push(Some((r, c)));
        }

        let features = encode_features(&state, &history);
        let (policy, value) = forward(&weights, &features);

        assert_eq!(policy.len(), POLICY_MOVES);
        assert!(
            policy.iter().all(|v| v.is_finite()),
            "policy logits must be finite"
        );
        assert!(value.is_finite(), "value must be finite");
        assert!(
            (-1.0..=1.0).contains(&value),
            "tanh output must be in [-1,1], got {value}"
        );
    }

    // ── Optimized-kernel equivalence oracle ───────────────────────
    //
    // The latency rework reorders FP summation (multi-accumulator dot) and
    // restructures loops, so it CANNOT be assumed bit-identical. These are
    // the original naive implementations, kept verbatim as the oracle: if an
    // optimization silently changes results, `optimized_conv_matches_naive`
    // fails instead of the strength benchmark quietly drifting.

    /// Original `for oc` outermost, single-accumulator convolution.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_naive(
        input: &[f32],
        h: usize,
        w: usize,
        in_ch: usize,
        out_ch: usize,
        k: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Vec<f32> {
        let pad = (k / 2) as isize;
        let mut out = vec![0f32; h * w * out_ch];
        for oc in 0..out_ch {
            for y in 0..h {
                for x in 0..w {
                    let mut sum = bias[oc];
                    for ky in 0..k {
                        let iy = y as isize + ky as isize - pad;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..k {
                            let ix = x as isize + kx as isize - pad;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            let wb = ((oc * k + ky) * k + kx) * in_ch;
                            let ib = (iy as usize * w + ix as usize) * in_ch;
                            for c in 0..in_ch {
                                sum += input[ib + c] * weight[wb + c];
                            }
                        }
                    }
                    out[(y * w + x) * out_ch + oc] = sum;
                }
            }
        }
        out
    }

    fn linear_naive(
        input: &[f32],
        in_dim: usize,
        out_dim: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Vec<f32> {
        (0..out_dim)
            .map(|o| {
                let base = o * in_dim;
                let mut sum = bias[o];
                for i in 0..in_dim {
                    sum += input[i] * weight[base + i];
                }
                sum
            })
            .collect()
    }

    /// Deterministic pseudo-random filler — avoids a dev-dependency and keeps
    /// the test reproducible.
    fn fill_pseudo(buf: &mut [f32], seed: u64) {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for v in buf.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to roughly [-1, 1).
            *v = ((s >> 33) as f32 / (1u64 << 30) as f32) - 1.0;
        }
    }

    /// The optimized `conv2d_into` must match the naive reference at every
    /// layer shape this network actually uses — including the 1×1 fast path
    /// and the 3×3 padded path.
    #[test]
    fn optimized_conv_matches_naive_reference() {
        // (in_ch, out_ch, k) for every distinct conv in the Moka topology.
        let shapes = [
            (INPUT_PLANES, TRUNK_CHANNELS, 3),             // stem
            (TRUNK_CHANNELS, BOTTLENECK_CHANNELS, 1),      // reduce
            (BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, 3), // first / second
            (BOTTLENECK_CHANNELS, TRUNK_CHANNELS, 1),      // expand
            (TRUNK_CHANNELS, POLICY_CHANNELS, 1),          // policy conv
            (TRUNK_CHANNELS, VALUE_CHANNELS, 1),           // value conv
        ];

        for (idx, &(in_ch, out_ch, k)) in shapes.iter().enumerate() {
            let mut input = vec![0f32; BOARD_AREA * in_ch];
            let mut weight = vec![0f32; out_ch * k * k * in_ch];
            let mut bias = vec![0f32; out_ch];
            fill_pseudo(&mut input, 11 + idx as u64);
            fill_pseudo(&mut weight, 101 + idx as u64);
            fill_pseudo(&mut bias, 1009 + idx as u64);

            let expected = conv2d_naive(
                &input, BOARD_SIZE, BOARD_SIZE, in_ch, out_ch, k, &weight, &bias,
            );

            let mut patch = vec![0f32; k * k * in_ch];
            let mut got = vec![0f32; BOARD_AREA * out_ch];
            conv2d_into(
                &input, BOARD_SIZE, BOARD_SIZE, in_ch, out_ch, k, &weight, &bias, &mut patch,
                &mut got,
            );

            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                    "conv mismatch at shape {in_ch}->{out_ch} k{k}, idx {i}: got {g}, want {e}"
                );
            }
        }
    }

    #[test]
    fn optimized_linear_matches_naive_reference() {
        // Both linear shapes used: policy (324->82) and value (162->32).
        for (idx, &(in_dim, out_dim)) in [
            (POLICY_CHANNELS * BOARD_AREA, POLICY_MOVES),
            (VALUE_CHANNELS * BOARD_AREA, 32),
        ]
        .iter()
        .enumerate()
        {
            let mut input = vec![0f32; in_dim];
            let mut weight = vec![0f32; out_dim * in_dim];
            let mut bias = vec![0f32; out_dim];
            fill_pseudo(&mut input, 21 + idx as u64);
            fill_pseudo(&mut weight, 211 + idx as u64);
            fill_pseudo(&mut bias, 2011 + idx as u64);

            let expected = linear_naive(&input, in_dim, out_dim, &weight, &bias);
            let mut got = vec![0f32; out_dim];
            linear_into(&input, in_dim, out_dim, &weight, &bias, &mut got);

            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                    "linear mismatch {in_dim}->{out_dim} idx {i}: got {g}, want {e}"
                );
            }
        }
    }

    /// `forward` (allocating wrapper) and `forward_with_scratch` (reused
    /// buffers) must agree exactly — a stale-buffer bug in the scratch path
    /// would otherwise be invisible.
    #[test]
    fn scratch_reuse_matches_fresh_forward() {
        let weights = MokaWeights::load();
        let mut scratch = MokaScratch::new();
        let mut state = GoState::new(BOARD_SIZE);
        let mut history: Vec<Option<(usize, usize)>> = Vec::new();

        // Run several distinct positions through the SAME scratch — this is
        // what catches leftover state between calls.
        for &(r, c) in &[(2usize, 2usize), (4, 4), (6, 3), (1, 7)] {
            state = state.advance(&GoAction::Place(r, c), state.to_play.player_id());
            history.push(Some((r, c)));
            let features = encode_features(&state, &history);

            let (p_fresh, v_fresh) = forward(&weights, &features);
            let (p_reuse, v_reuse) = forward_with_scratch(&weights, &features, &mut scratch);

            assert_eq!(
                v_fresh.to_bits(),
                v_reuse.to_bits(),
                "value differs on reused scratch"
            );
            for i in 0..POLICY_MOVES {
                assert_eq!(
                    p_fresh[i].to_bits(),
                    p_reuse[i].to_bits(),
                    "policy[{i}] differs between fresh and reused scratch"
                );
            }
        }
    }

    /// The negamax search must never surface an illegal move, and its
    /// threaded `SearchHistory` must stay consistent across plies.
    #[test]
    fn moka_search_player_returns_legal_moves_and_threads_history() {
        let mut player = GoMokaSearchPlayer::new(2, 4);
        let mut state = GoState::new(BOARD_SIZE);
        let mut rng = Rng::with_seed(7);

        for ply in 0..6 {
            let legal = state.legal_moves();
            let action = player.select_move(&state, &legal, &mut rng);
            match &action {
                GoAction::Place(r, c) => {
                    assert!(
                        legal.contains(&(*r, *c)),
                        "search chose illegal move {r},{c}"
                    );
                }
                GoAction::Pass => {}
            }
            assert_eq!(
                player.history.len(),
                ply + 1,
                "history must grow one entry per ply"
            );
            let mover = state.to_play;
            state = state.advance(&action, mover.player_id());
        }
        assert!(
            player.nodes_evaluated() > 6,
            "search should evaluate more nodes than plies"
        );
    }

    /// `SearchHistory` must produce a 1-element slice for a 1-move game —
    /// a `None` filler would be misread as "that ply was a pass".
    #[test]
    fn search_history_length_distinguishes_missing_from_pass() {
        let empty = SearchHistory::from_slice(&[]);
        assert_eq!(empty.as_slice().len(), 0);

        let one = empty.push(&GoAction::Place(3, 3));
        assert_eq!(one.as_slice(), &[Some((3, 3))]);

        let two = one.push(&GoAction::Place(4, 4));
        assert_eq!(two.as_slice(), &[Some((3, 3)), Some((4, 4))]);

        // Rolls the window; a real pass is preserved as None.
        let three = two.push(&GoAction::Pass);
        assert_eq!(three.as_slice(), &[Some((4, 4)), None]);
    }

    #[test]
    fn moka_player_always_returns_legal_or_pass_moves() {
        let mut player = MokaPlayer::new();
        let mut state = GoState::new(BOARD_SIZE);
        let mut rng = Rng::with_seed(42);

        for _ in 0..10 {
            let legal = state.legal_moves();
            let action = player.select_move(&state, &legal, &mut rng);
            match &action {
                GoAction::Place(r, c) => {
                    assert!(
                        legal.contains(&(*r, *c)),
                        "MokaPlayer chose an illegal move {r},{c}"
                    );
                }
                GoAction::Pass => {}
            }
            let mover = state.to_play;
            state = state.advance(&action, mover.player_id());
        }
    }
}
