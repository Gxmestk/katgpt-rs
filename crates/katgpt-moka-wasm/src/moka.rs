//! Moka v1 forward pass, ported from `katgpt_pruners::go::moka_net` (Plan 563)
//! for a dependency-free WASM build (Plan 565). Same architecture, same
//! vendored weights, same kernel — only the board type changed (this crate's
//! minimal `board::Board` instead of the full `GoState` engine, so this
//! crate carries no `katgpt-core` dependency and therefore no ahash/getrandom
//! wasm32 backend friction the full engine's dependency tree runs into).
//!
//! Kept in sync by construction: this is the same forward-pass code, not a
//! reimplementation — see Plan 563/Issue 564 for the parity/equivalence
//! testing this logic already passed in its native home.

use std::collections::HashMap;

use serde::Deserialize;

use crate::board::{AREA as BOARD_AREA, Board, Cell, SIZE as BOARD_SIZE, flood_group};

const INPUT_PLANES: usize = 12;
const TRUNK_CHANNELS: usize = 32;
const BOTTLENECK_CHANNELS: usize = 16;
const NUM_BLOCKS: usize = 12;
const GLOBAL_BLOCK_INTERVAL: usize = 4;
const POLICY_CHANNELS: usize = 4;
pub const POLICY_MOVES: usize = 82;
/// Flat feature-tensor length (`9*9*12`) — the input size `wasmi_infer`'s
/// raw FFI boundary expects.
pub const INPUT_ELEMENT_COUNT: usize = BOARD_AREA * INPUT_PLANES;
const VALUE_CHANNELS: usize = 2;
const SCORE_HIDDEN_CHANNELS: usize = 32;
/// Moka's own training-time komi convention — not this crate's board
/// convention, whichever that ends up being. The feature plane must match
/// what the network was trained on.
const MOKA_KOMI: f32 = 7.0;
const KOMI_NORMALIZATION: f32 = 15.0;

static MANIFEST_JSON: &str = include_str!("../../katgpt-pruners/assets/moka/go-model.json");
static WEIGHTS_BIN: &[u8] = include_bytes!("../../katgpt-pruners/assets/moka/go-model.bin");

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

fn load_dequantized(tensors: &HashMap<String, TensorMeta>, bytes: &[u8], name: &str) -> Vec<f32> {
    let meta = tensors.get(name).unwrap_or_else(|| panic!("moka manifest missing tensor {name}"));
    assert_eq!(meta.dtype, "int8", "expected int8 weight tensor {name}");
    let out_channels = meta.shape[0];
    let count: usize = meta.shape.iter().product();
    let per_channel = count / out_channels;
    let scale_offset = meta.scale_offset.unwrap_or_else(|| panic!("{name} missing scaleOffset"));
    let scales = read_f32(bytes, scale_offset, out_channels);
    let mut out = Vec::with_capacity(count);
    for (oc, &scale) in scales.iter().enumerate() {
        let base = meta.data_offset + oc * per_channel;
        for k in 0..per_channel {
            out.push((bytes[base + k] as i8) as f32 * scale);
        }
    }
    out
}

fn load_bias(tensors: &HashMap<String, TensorMeta>, bytes: &[u8], name: &str) -> Vec<f32> {
    let meta = tensors.get(name).unwrap_or_else(|| panic!("moka manifest missing tensor {name}"));
    assert_eq!(meta.dtype, "float32", "expected float32 bias tensor {name}");
    let count: usize = meta.shape.iter().product();
    read_f32(bytes, meta.data_offset, count)
}

struct Wb {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct GlobalBranch {
    hidden: Wb,
    output: Wb,
}

struct ResidualBlock {
    reduce: Wb,
    first: Wb,
    global: Option<GlobalBranch>,
    second: Wb,
    expand: Wb,
}

pub struct MokaWeights {
    stem: Wb,
    blocks: Vec<ResidualBlock>,
    policy_conv: Wb,
    policy_linear: Wb,
    value_conv: Wb,
    value_hidden: Wb,
    value_output: Wb,
}

impl MokaWeights {
    pub fn load() -> Self {
        let manifest: Manifest = serde_json::from_str(MANIFEST_JSON).expect("vendored moka manifest is valid JSON");
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

#[inline]
fn dot_lanes(a: &[f32], b: &[f32], init: f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    init + katgpt_types::simd::simd_dot_f32(a, b, a.len().min(b.len()))
}

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

fn linear_into(input: &[f32], in_dim: usize, out_dim: usize, weight: &[f32], bias: &[f32], out: &mut [f32]) {
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

fn global_mean_max_into(x: &[f32], h: usize, w: usize, ch: usize, out: &mut [f32]) {
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

pub struct MokaScratch {
    trunk: Vec<f32>,
    expand: Vec<f32>,
    hidden_a: Vec<f32>,
    hidden_b: Vec<f32>,
    head4: Vec<f32>,
    head2: Vec<f32>,
    patch: Vec<f32>,
    pooled: Vec<f32>,
    gh: Vec<f32>,
    gbias: Vec<f32>,
    value_h: Vec<f32>,
    policy: Vec<f32>,
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

pub fn forward_with_scratch(weights: &MokaWeights, features: &[f32], scratch: &mut MokaScratch) -> ([f32; POLICY_MOVES], f32) {
    let MokaScratch {
        trunk, expand, hidden_a, hidden_b, head4, head2, patch, pooled, gh, gbias, value_h, policy,
    } = scratch;

    conv2d_into(features, BOARD_SIZE, BOARD_SIZE, INPUT_PLANES, TRUNK_CHANNELS, 3, &weights.stem.w, &weights.stem.b, patch, trunk);
    relu_inplace(&mut trunk[..BOARD_AREA * TRUNK_CHANNELS]);

    for block in &weights.blocks {
        conv2d_into(trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, BOTTLENECK_CHANNELS, 1, &block.reduce.w, &block.reduce.b, patch, hidden_a);
        relu_inplace(hidden_a);
        conv2d_into(hidden_a, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, 3, &block.first.w, &block.first.b, patch, hidden_b);
        relu_inplace(hidden_b);

        if let Some(g) = &block.global {
            global_mean_max_into(hidden_b, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, pooled);
            linear_into(pooled, BOTTLENECK_CHANNELS * 2, g.hidden.b.len(), &g.hidden.w, &g.hidden.b, gh);
            relu_inplace(&mut gh[..g.hidden.b.len()]);
            linear_into(gh, g.hidden.b.len(), BOTTLENECK_CHANNELS, &g.output.w, &g.output.b, gbias);
            for pos in 0..BOARD_AREA {
                let row = &mut hidden_b[pos * BOTTLENECK_CHANNELS..(pos + 1) * BOTTLENECK_CHANNELS];
                for c in 0..BOTTLENECK_CHANNELS {
                    row[c] += gbias[c];
                }
            }
        }

        conv2d_into(hidden_b, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, 3, &block.second.w, &block.second.b, patch, hidden_a);
        relu_inplace(hidden_a);
        conv2d_into(hidden_a, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, TRUNK_CHANNELS, 1, &block.expand.w, &block.expand.b, patch, expand);

        for i in 0..BOARD_AREA * TRUNK_CHANNELS {
            let v = trunk[i] + expand[i];
            trunk[i] = if v < 0.0 { 0.0 } else { v };
        }
    }

    conv2d_into(trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, POLICY_CHANNELS, 1, &weights.policy_conv.w, &weights.policy_conv.b, patch, head4);
    relu_inplace(head4);
    linear_into(head4, POLICY_CHANNELS * BOARD_AREA, POLICY_MOVES, &weights.policy_linear.w, &weights.policy_linear.b, policy);

    conv2d_into(trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, VALUE_CHANNELS, 1, &weights.value_conv.w, &weights.value_conv.b, patch, head2);
    relu_inplace(head2);
    let value_hidden_dim = weights.value_hidden.b.len();
    linear_into(head2, VALUE_CHANNELS * BOARD_AREA, value_hidden_dim, &weights.value_hidden.w, &weights.value_hidden.b, value_h);
    relu_inplace(&mut value_h[..value_hidden_dim]);
    let mut value_out = [0f32; 1];
    linear_into(value_h, value_hidden_dim, 1, &weights.value_output.w, &weights.value_output.b, &mut value_out);

    let mut logits = [0f32; POLICY_MOVES];
    logits.copy_from_slice(&policy[..POLICY_MOVES]);
    (logits, value_out[0].tanh())
}

/// Encode `board` + last-two-plies `history` (`None` = pass) into Moka's
/// 12-plane `9*9*12` HWC feature tensor, written into `out` (length ≥
/// [`INPUT_ELEMENT_COUNT`]). Logic identical to
/// `katgpt_pruners::go::moka_net::encode_features`, adapted to this crate's
/// standalone `Board`/`Cell`; `_into` here (unlike the native version, which
/// still returns an owned `Vec`) because this crate's `WasmGame` holds a
/// persistent buffer specifically so its address stays stable across calls —
/// letting JS wrap ONE `Float32Array` view over it instead of marshalling a
/// fresh array out of wasm on every encode.
pub fn encode_features_into(board: &Board, history: &[Option<(usize, usize)>], out: &mut [f32]) {
    let size = BOARD_SIZE;
    let idx = |row: usize, col: usize, plane: usize| (row * size + col) * INPUT_PLANES + plane;
    let feats = &mut out[..size * size * INPUT_PLANES];
    feats.fill(0.0);

    let mut visited = vec![false; size * size];
    for pos in 0..size * size {
        let color = board.cells[pos];
        if color == Cell::Empty {
            continue;
        }
        let is_current = color == board.to_play;
        let (row, col) = (pos / size, pos % size);
        feats[idx(row, col, if is_current { 0 } else { 1 })] = 1.0;

        if visited[pos] {
            continue;
        }
        let (stones, liberties) = flood_group(&board.cells, pos);
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

    if let Some(ko) = board.ko_point {
        let (r, c) = (ko / size, ko % size);
        feats[idx(r, c, 6)] = 1.0;
    }

    for offset in 1..=2usize {
        if history.len() < offset {
            continue;
        }
        match history[history.len() - offset] {
            Some((r, c)) => {
                let plane = if offset == 1 { 7 } else { 8 };
                feats[idx(r, c, plane)] = 1.0;
            }
            None => {
                let plane = 8 + offset;
                for row in 0..size {
                    for col in 0..size {
                        feats[idx(row, col, plane)] = 1.0;
                    }
                }
            }
        }
    }

    let next_color: f32 = if board.to_play == Cell::Black { 1.0 } else { -1.0 };
    let komi_value = (-MOKA_KOMI * next_color) / KOMI_NORMALIZATION;
    for row in 0..size {
        for col in 0..size {
            feats[idx(row, col, 11)] = komi_value;
        }
    }
}
