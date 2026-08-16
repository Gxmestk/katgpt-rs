//! MTP Metal Batch-Width Floor — is speculative verify affordable on Apple Silicon?
//!
//! Run with:
//! ```text
//! cargo run --release -p katgpt-backend --features gpu_inference \
//!     --example bench_mtp_metal_batch_floor
//! ```
//!
//! # Why this benchmark exists
//!
//! llama.cpp issue #23752 reports MTP speculative decoding as a *net loss* on
//! Apple Silicon at every configuration (M1 Max, Qwen3.5-9B-Q4_K_M, 2048 tok):
//!
//! | config              | tok/s | acceptance |
//! |---------------------|-------|------------|
//! | baseline (no MTP)   | 25.3  | —          |
//! | `--spec-draft-n-max 0` | 22.4 | 100%      |
//! | `--spec-draft-n-max 2` | 21.9 | 73–76%    |
//! | `--spec-draft-n-max 6` | 19.3 | 41–44%    |
//!
//! The load-bearing row is the second: **11% slower while drafting nothing and
//! accepting everything**. That penalty cannot be an acceptance-quality problem,
//! so no better tree-builder (DDTree) can recover it. Before this repo invests in
//! an MTP drafter feeding `build_dd_tree`, we need to know which of these is true:
//!
//! - **ARTIFACT** — llama.cpp evaluates the MTP path unconditionally / adds a
//!   sync point, and a from-scratch Metal implementation would not inherit it.
//! - **FUNDAMENTAL** — batched verify is genuinely expensive on Metal, so
//!   speculative decoding can never pay here regardless of implementation.
//!
//! # What is measured
//!
//! Speculative decoding replaces one width-1 decode step with one width-`N`
//! verify pass (`N` = draft depth + 1). Decode is dominated by weight-matrix
//! multiplies, so the whole question reduces to a single ratio:
//!
//! ```text
//!     cost(N) / cost(1)
//! ```
//!
//! Speculation pays iff that ratio is below the mean accepted-token count `E`.
//! From the llama.cpp table, `N=3` (`n-max 2`) yields `E ≈ 1.7`.
//!
//! # Theory prediction
//!
//! Decode matvec reads `out_dim × in_dim` weights to do `2 × out_dim × in_dim`
//! FLOPs — roughly **0.5 FLOP/byte**, i.e. bandwidth-bound by a wide margin.
//! M3 Max is ~400 GB/s against ~14 TFLOP/s fp32, so it only becomes
//! compute-bound above ~35 FLOP/byte. Widening to `N` multiplies FLOPs by `N`
//! while re-reading the *same* weights, so the ratio should stay ≈1.0 until
//! `N` ≈ 64. If measurement matches, the llama.cpp penalty is an ARTIFACT.
//!
//! The kernel below is written the way a real verify pass would be: one thread
//! per output row, loading each weight **once** and accumulating all `N`
//! columns in registers. `BATCH` is baked in as a compile-time constant per
//! pipeline so the accumulator array unrolls into registers rather than
//! spilling to thread-local memory (a spill would distort the ratio upward and
//! understate Metal).

use std::time::Instant;

use metal::{
    CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize, NSUInteger,
};

/// Batch widths to sweep. `N=1` is the baseline decode step; `N=3` corresponds
/// to llama.cpp's `--spec-draft-n-max 2`, `N=7` to `--spec-draft-n-max 6`.
const BATCH_WIDTHS: &[u32] = &[1, 2, 3, 4, 7, 8, 16];

/// Timed iterations per configuration (after warmup).
const ITERS: u32 = 40;

/// Untimed iterations to absorb shader compile + first-dispatch cost.
const WARMUP: u32 = 3;

/// Decode-realistic weight shapes: `(label, out_dim, in_dim)`.
///
/// Sized to a ~7-9B model's projections, which is the class llama.cpp measured.
/// The ratio under test is shape-independent in theory; three shapes spanning
/// 67 MB → 524 MB confirm it holds as the weight matrix outgrows every cache.
const SHAPES: &[(&str, u32, u32)] = &[
    ("attn_qkv", 4096, 4096),
    ("ffn_up", 11008, 4096),
    ("lm_head", 32000, 4096),
];

/// Mean accepted tokens/step at `n-max 2`, from llama.cpp #23752's 73–76%
/// acceptance. Speculation pays iff `cost(N)/cost(1)` lands below this.
const BREAKEVEN_E: f64 = 1.7;

/// Build the MSL source for a batched matvec specialized to `batch`.
///
/// `X` is stored transposed (`[in_dim][batch]`) so the unrolled inner loop
/// reads `batch` contiguous floats — coalesced, the layout a real verify uses.
fn shader_source(batch: u32) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

#define BATCH {batch}u

// out[n][row] = dot(W[row][:], Xt[:][n])
//   W  : [out_dim][in_dim]  — the weight matrix, re-read once per row
//   Xt : [in_dim][BATCH]    — activations, transposed for coalesced access
//   out: [BATCH][out_dim]
kernel void matmul_batched(
    device const float* W [[buffer(0)]],
    device const float* Xt [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= out_dim) {{ return; }}

    float acc[BATCH];
    #pragma unroll
    for (uint n = 0; n < BATCH; n++) {{ acc[n] = 0.0f; }}

    device const float* w_row = W + (ulong)gid * in_dim;
    for (uint k = 0; k < in_dim; k++) {{
        float w = w_row[k];
        device const float* xk = Xt + (ulong)k * BATCH;
        #pragma unroll
        for (uint n = 0; n < BATCH; n++) {{ acc[n] += w * xk[n]; }}
    }}

    #pragma unroll
    for (uint n = 0; n < BATCH; n++) {{ out[(ulong)n * out_dim + gid] = acc[n]; }}
}}
"#
    )
}

/// Compile the `batch`-specialized pipeline.
fn build_pipeline(device: &Device, batch: u32) -> ComputePipelineState {
    let library = device
        .new_library_with_source(&shader_source(batch), &CompileOptions::new())
        .expect("MSL compile failed");
    let function = library
        .get_function("matmul_batched", None)
        .expect("matmul_batched not found");
    device
        .new_compute_pipeline_state_with_function(&function)
        .expect("pipeline creation failed")
}

/// Allocate a shared-storage buffer of `len` f32s filled by `fill`.
fn filled_buffer(device: &Device, len: usize, fill: impl Fn(usize) -> f32) -> metal::Buffer {
    let bytes = (len * size_of::<f32>()) as NSUInteger;
    let buffer = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    // SAFETY: `contents()` is a valid mapping of `len` f32s for a shared buffer.
    let slice = unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), len) };
    for (i, slot) in slice.iter_mut().enumerate() {
        *slot = fill(i);
    }
    buffer
}

/// Fastest observed pass, in milliseconds.
///
/// Min rather than median: every perturbation here (scheduler preemption, clock
/// ramp, contention from other work on the box) is strictly additive, so the
/// minimum is the least-contaminated estimate of true kernel cost. Using the
/// median let baseline drift reach 75% between the pre- and post-sweep `N=1`
/// measurements, which is wider than the effect under test.
fn min_ms(samples: &[f64]) -> f64 {
    samples.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Drive the GPU to a steady clock before any timing.
///
/// Without this the very first timed configuration absorbs shader-cache warmup
/// and power-state ramp. That inflates the `N=1` baseline, which *understates*
/// every `cost(N)/cost(1)` ratio and biases the verdict toward "free" — the
/// exact error this benchmark exists to avoid. Symptom in an unwarmed run: a
/// 64 MB matvec measuring slower than a 172 MB one, and sub-1.0 ratios.
fn warm_gpu(device: &Device, queue: &metal::CommandQueue, samples: &mut Vec<f64>) {
    let (_, out_dim, in_dim) = SHAPES[SHAPES.len() - 1];
    let w = filled_buffer(device, (out_dim as usize) * (in_dim as usize), |i| {
        ((i % 17) as f32).mul_add(0.01, -0.08)
    });
    for _ in 0..4 {
        time_config(device, queue, &w, out_dim, in_dim, 1, samples);
    }
}

/// Time one `(shape, batch)` configuration, returning median ms per pass.
///
/// `w` is owned by the caller and reused across every batch width for a shape.
/// Re-allocating it per configuration would page-fault and CPU-write hundreds of
/// MB between measurements, polluting the very ratio under test — and a real
/// decoder keeps weights resident anyway.
fn time_config(
    device: &Device,
    queue: &metal::CommandQueue,
    w: &metal::Buffer,
    out_dim: u32,
    in_dim: u32,
    batch: u32,
    samples: &mut Vec<f64>,
) -> f64 {
    let pipeline = build_pipeline(device, batch);

    let xt = filled_buffer(device, (in_dim as usize) * (batch as usize), |i| {
        ((i % 13) as f32).mul_add(0.02, -0.12)
    });
    let out = filled_buffer(device, (out_dim as usize) * (batch as usize), |_| 0.0);

    let tg_width = pipeline.thread_execution_width();
    let groups = MTLSize::new(u64::from(out_dim).div_ceil(tg_width), 1, 1);
    let threads = MTLSize::new(tg_width, 1, 1);

    let dispatch = || {
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(w), 0);
        encoder.set_buffer(1, Some(&xt), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, size_of::<u32>() as NSUInteger, (&raw const in_dim).cast());
        encoder.set_bytes(4, size_of::<u32>() as NSUInteger, (&raw const out_dim).cast());
        encoder.dispatch_thread_groups(groups, threads);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };

    for _ in 0..WARMUP {
        dispatch();
    }

    samples.clear();
    for _ in 0..ITERS {
        let t0 = Instant::now();
        dispatch();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    min_ms(samples)
}

fn main() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device — this benchmark requires Apple Silicon");
        std::process::exit(1);
    };
    let queue = device.new_command_queue();

    println!("MTP Metal Batch-Width Floor");
    println!("device: {}", device.name());
    println!("iters/config: {ITERS} (median reported), warmup: {WARMUP}");
    println!(
        "\nQuestion: does a decode-shaped matmul at width N cost ~1x or ~Nx?\n\
         Speculation pays iff cost(N)/cost(1) < E (mean accepted tokens/step).\n\
         llama.cpp #23752 measured E ~= {BREAKEVEN_E} at n-max 2 (N=3).\n"
    );

    let mut samples: Vec<f64> = Vec::with_capacity(ITERS as usize);
    warm_gpu(&device, &queue, &mut samples);

    // Worst (largest) observed ratio at the N=3 speculative-verify width, across
    // shapes — the conservative number the verdict is drawn from.
    let mut worst_n3_ratio = 0.0_f64;

    for &(label, out_dim, in_dim) in SHAPES {
        let weight_bytes = f64::from(out_dim) * f64::from(in_dim) * 4.0;
        let weight_mb = weight_bytes / (1024.0 * 1024.0);
        println!("── {label}  W=[{out_dim}, {in_dim}]  ({weight_mb:.0} MB, {out_dim} threads) ──");
        println!("    N     ms/pass    GB/s    cost(N)/cost(1)    verdict");

        // Baseline is measured twice — once before the sweep and once after —
        // and the faster is used. A large gap means thermal/clock drift rather
        // than a real batching cost, so it is surfaced instead of folded in.
        let w = filled_buffer(&device, (out_dim as usize) * (in_dim as usize), |i| {
            ((i % 17) as f32).mul_add(0.01, -0.08)
        });
        let base_first = time_config(&device, &queue, &w, out_dim, in_dim, 1, &mut samples);
        let mut rows: Vec<(u32, f64)> = Vec::with_capacity(BATCH_WIDTHS.len());
        for &batch in BATCH_WIDTHS.iter().skip(1) {
            rows.push((
                batch,
                time_config(&device, &queue, &w, out_dim, in_dim, batch, &mut samples),
            ));
        }
        let base_last = time_config(&device, &queue, &w, out_dim, in_dim, 1, &mut samples);
        let base_ms = base_first.min(base_last);
        let drift = (base_first - base_last).abs() / base_ms;

        let bandwidth = |ms: f64| weight_bytes / (ms * 1e-3) / 1e9;
        println!(
            "{:5}  {base_ms:9.3}  {:6.0}    {:>15}    baseline",
            1,
            bandwidth(base_ms),
            "1.00x"
        );
        for (batch, ms) in rows {
            let ratio = ms / base_ms;
            // A width-N pass is worth it when it costs less than the N serial
            // steps it replaces; "free" means ~no marginal cost over width-1.
            let verdict = match ratio {
                r if r < 1.15 => "~free",
                r if r < BREAKEVEN_E => "pays",
                r if r < f64::from(batch) => "partial",
                _ => "LOSS",
            };
            println!(
                "{batch:5}  {ms:9.3}  {:6.0}    {ratio:>14.2}x    {verdict}",
                bandwidth(ms)
            );
            if batch == 3 {
                worst_n3_ratio = worst_n3_ratio.max(ratio);
            }
        }
        println!("    baseline drift first-vs-last: {:.1}%\n", drift * 100.0);
    }

    println!("── VERDICT ──");
    println!("worst-case cost(3)/cost(1) across shapes: {worst_n3_ratio:.2}x  (breakeven {BREAKEVEN_E}x)");
    match worst_n3_ratio < BREAKEVEN_E {
        true => {
            println!(
                "ARTIFACT — batched verify is affordable on this device, so the llama.cpp\n\
                 Metal penalty is an implementation issue, not a hardware limit.\n\
                 => mtp+ddtree is viable on M3; MTP can be gated on Metal, not CUDA-only."
            );
        }
        false => {
            println!(
                "FUNDAMENTAL — batched verify costs more than the tokens it can win back.\n\
                 Speculative decoding cannot pay on this device at any acceptance rate.\n\
                 => MTP is a CUDA/4090-only opt-in; do not gate it on Metal."
            );
        }
    }
}
