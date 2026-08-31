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
//!
//! # Two protocols, reported side by side
//!
//! The original run measured configurations **sequentially** (`N=1,2,3,4,7,8,16`
//! within each shape) with the `N=1` baseline taken once before and once after
//! the sweep. That is the pattern riir-ai's Metal measurements have twice been
//! burned by — a "1.19× win" that was a 0.87× loss (Bench 666), a "1.24× win"
//! that was a 0.95× loss (Issue 658) — because monotonic thermal drift over the
//! sweep is charged entirely to the later configurations.
//!
//! Benchmark 656 flagged one specific number as un-settled on those grounds: the
//! **`N ≤ 4` viable band**. It is set by `attn_qkv`, the shape that carried
//! 13–17% baseline drift (vs 0.4–4.0% for the bandwidth-saturated shapes), so
//! it is exactly the boundary the sequential protocol is least able to place.
//!
//! This harness therefore runs **both**:
//!
//! - **sequential** — the original, kept so the protocol change is attributable
//!   rather than asserted.
//! - **interleaved** — riir-ai's house protocol. Each `(N=1, N=k)` pair is
//!   measured back to back, alternating which side runs first, and the primary
//!   statistic is the median of **per-pair** ratios. Drift that is monotonic
//!   within a pair cancels instead of accumulating across the sweep.
//!
//! The verdict is drawn from the interleaved numbers. If the two disagree, the
//! disagreement is the result.

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

/// Interleaved-protocol pairs discarded before measurement begins.
const PAIR_WARMUP: u32 = 2;

/// Interleaved-protocol pairs that count. Each yields one ratio; the median of
/// those is the reported statistic.
const PAIR_MEASURE: u32 = 7;

/// Timed dispatches per side within one pair. Lower than [`ITERS`] because the
/// protocol's robustness comes from repeating the *pair*, not from grinding a
/// single side — and 9 pairs × 2 sides already dominates the sequential run.
const PAIR_ITERS: u32 = 10;

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

/// A compiled, ready-to-dispatch `(shape, batch)` configuration.
///
/// Built once and reused. The interleaved protocol alternates between two
/// configurations 18 times per comparison; rebuilding the pipeline each time
/// would recompile MSL on every switch and charge shader-compile latency to
/// whichever side happened to run first — an ordering artifact inside the very
/// mechanism meant to remove ordering artifacts.
struct Prepared {
    pipeline: ComputePipelineState,
    xt: metal::Buffer,
    out: metal::Buffer,
    groups: MTLSize,
    threads: MTLSize,
    in_dim: u32,
    out_dim: u32,
}

impl Prepared {
    fn new(device: &Device, out_dim: u32, in_dim: u32, batch: u32) -> Self {
        let pipeline = build_pipeline(device, batch);
        let xt = filled_buffer(device, (in_dim as usize) * (batch as usize), |i| {
            ((i % 13) as f32).mul_add(0.02, -0.12)
        });
        let out = filled_buffer(device, (out_dim as usize) * (batch as usize), |_| 0.0);
        let tg_width = pipeline.thread_execution_width();
        Self {
            groups: MTLSize::new(u64::from(out_dim).div_ceil(tg_width), 1, 1),
            threads: MTLSize::new(tg_width, 1, 1),
            pipeline,
            xt,
            out,
            in_dim,
            out_dim,
        }
    }

    fn dispatch(&self, queue: &metal::CommandQueue, w: &metal::Buffer) {
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(w), 0);
        encoder.set_buffer(1, Some(&self.xt), 0);
        encoder.set_buffer(2, Some(&self.out), 0);
        encoder.set_bytes(3, size_of::<u32>() as NSUInteger, (&raw const self.in_dim).cast());
        encoder.set_bytes(4, size_of::<u32>() as NSUInteger, (&raw const self.out_dim).cast());
        encoder.dispatch_thread_groups(self.groups, self.threads);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Fastest of `iters` dispatches, after `warmup` untimed ones.
    fn time(
        &self,
        queue: &metal::CommandQueue,
        w: &metal::Buffer,
        warmup: u32,
        iters: u32,
        samples: &mut Vec<f64>,
    ) -> f64 {
        for _ in 0..warmup {
            self.dispatch(queue, w);
        }
        samples.clear();
        for _ in 0..iters {
            let t0 = Instant::now();
            self.dispatch(queue, w);
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        min_ms(samples)
    }
}

/// Time one `(shape, batch)` configuration under the **sequential** protocol.
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
    Prepared::new(device, out_dim, in_dim, batch).time(queue, w, WARMUP, ITERS, samples)
}

/// Result of one interleaved A/B comparison.
struct PairStats {
    /// Median of the per-pair ratios — the reported statistic.
    ratio: f64,
    /// Spread across pairs. A wide spread means only the direction is
    /// established, not the magnitude.
    lo: f64,
    hi: f64,
    /// Median absolute timings, for the bandwidth column.
    base_ms: f64,
    wide_ms: f64,
}

/// Interleaved `cost(N)/cost(1)` — riir-ai's house protocol.
///
/// [`PAIR_WARMUP`] pairs are discarded, then [`PAIR_MEASURE`] pairs are timed
/// with the A/B order **alternating** between pairs, so a within-pair drift
/// penalises each side equally often instead of always the second-placed one.
/// The primary statistic is the median of per-pair ratios, not
/// `median(A) / median(B)` — the latter lets drift accumulated between the two
/// halves of the run leak straight into the answer.
fn paired_ratio(
    queue: &metal::CommandQueue,
    w: &metal::Buffer,
    base: &Prepared,
    wide: &Prepared,
    samples: &mut Vec<f64>,
) -> PairStats {
    let mut ratios = Vec::with_capacity(PAIR_MEASURE as usize);
    let mut base_all = Vec::with_capacity(PAIR_MEASURE as usize);
    let mut wide_all = Vec::with_capacity(PAIR_MEASURE as usize);

    for pair in 0..PAIR_WARMUP + PAIR_MEASURE {
        let (b_ms, w_ms) = if (pair % 2) == 0 {
                let b = base.time(queue, w, 1, PAIR_ITERS, samples);
                let x = wide.time(queue, w, 1, PAIR_ITERS, samples);
                (b, x)
            } else {
                let x = wide.time(queue, w, 1, PAIR_ITERS, samples);
                let b = base.time(queue, w, 1, PAIR_ITERS, samples);
                (b, x)
            };
        if pair >= PAIR_WARMUP {
            ratios.push(w_ms / b_ms);
            base_all.push(b_ms);
            wide_all.push(w_ms);
        }
    }

    PairStats {
        ratio: median(&mut ratios),
        lo: ratios.iter().copied().fold(f64::INFINITY, f64::min),
        hi: ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        base_ms: median(&mut base_all),
        wide_ms: median(&mut wide_all),
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
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
    // shapes — the conservative number the verdict is drawn from. Tracked under
    // both protocols so a disagreement is visible rather than averaged away.
    let mut worst_n3_seq = 0.0_f64;
    let mut worst_n3_int = 0.0_f64;
    // Largest N that still pays, per shape, under the interleaved protocol —
    // this is the `N <= 4` band Benchmark 656 left un-settled.
    let mut bands: Vec<(&str, u32)> = Vec::with_capacity(SHAPES.len());

    for &(label, out_dim, in_dim) in SHAPES {
        let weight_bytes = f64::from(out_dim) * f64::from(in_dim) * 4.0;
        let weight_mb = weight_bytes / (1024.0 * 1024.0);
        println!("── {label}  W=[{out_dim}, {in_dim}]  ({weight_mb:.0} MB, {out_dim} threads) ──");

        let w = filled_buffer(&device, (out_dim as usize) * (in_dim as usize), |i| {
            ((i % 17) as f32).mul_add(0.01, -0.08)
        });
        let bandwidth = |ms: f64| weight_bytes / (ms * 1e-3) / 1e9;

        // ── Protocol A: sequential (the original) ──
        //
        // Baseline is measured twice — once before the sweep and once after —
        // and the faster is used. A large gap means thermal/clock drift rather
        // than a real batching cost, so it is surfaced instead of folded in.
        let base_first = time_config(&device, &queue, &w, out_dim, in_dim, 1, &mut samples);
        let mut seq_rows: Vec<(u32, f64)> = Vec::with_capacity(BATCH_WIDTHS.len());
        for &batch in BATCH_WIDTHS.iter().skip(1) {
            seq_rows.push((
                batch,
                time_config(&device, &queue, &w, out_dim, in_dim, batch, &mut samples),
            ));
        }
        let base_last = time_config(&device, &queue, &w, out_dim, in_dim, 1, &mut samples);
        let base_ms = base_first.min(base_last);
        let drift = (base_first - base_last).abs() / base_ms;

        // ── Protocol B: interleaved (the house protocol) ──
        let base_prep = Prepared::new(&device, out_dim, in_dim, 1);
        let mut int_rows: Vec<(u32, PairStats)> = Vec::with_capacity(BATCH_WIDTHS.len());
        for &batch in BATCH_WIDTHS.iter().skip(1) {
            let wide = Prepared::new(&device, out_dim, in_dim, batch);
            int_rows.push((batch, paired_ratio(&queue, &w, &base_prep, &wide, &mut samples)));
        }

        // A width-N pass is worth it when it costs less than the N serial steps
        // it replaces; "free" means ~no marginal cost over width-1.
        let verdict = |ratio: f64, batch: u32| match ratio {
            r if r < 1.15 => "~free",
            r if r < BREAKEVEN_E => "pays",
            r if r < f64::from(batch) => "partial",
            _ => "LOSS",
        };

        println!("    {:>3}  {:>18}  {:>30}", "N", "sequential", "interleaved (median of pairs)");
        println!(
            "    {:>3}  {:>9} {:>8}  {:>9} {:>7} {:>12}",
            "", "ms", "ratio", "ms", "ratio", "spread"
        );
        // The interleaved baseline is the one the ratios were actually computed
        // against — the median of the `N=1` side across every pair — not a fresh
        // standalone measurement, which would not share their thermal context.
        let mut int_bases: Vec<f64> = int_rows.iter().map(|(_, s)| s.base_ms).collect();
        println!(
            "    {:>3}  {base_ms:9.3} {:>8}  {:9.3} {:>7} {:>12}",
            1,
            "1.00x",
            median(&mut int_bases),
            "1.00x",
            "—"
        );
        for ((batch, seq_ms), (_, stats)) in seq_rows.iter().zip(&int_rows) {
            let seq_ratio = seq_ms / base_ms;
            println!(
                "    {batch:>3}  {seq_ms:9.3} {:>7.2}x  {:9.3} {:>6.2}x {:>12}   {} / {}",
                seq_ratio,
                stats.wide_ms,
                stats.ratio,
                format!("{:.2}–{:.2}", stats.lo, stats.hi),
                verdict(seq_ratio, *batch),
                verdict(stats.ratio, *batch),
            );
            if *batch == 3 {
                worst_n3_seq = worst_n3_seq.max(seq_ratio);
                worst_n3_int = worst_n3_int.max(stats.ratio);
            }
        }

        // The band: largest swept N whose interleaved ratio still clears
        // breakeven. `int_rows` is in ascending batch order, and cost is
        // monotone in N, so the last passing row is the boundary.
        let band = int_rows
            .iter()
            .filter(|(_, s)| s.ratio < BREAKEVEN_E)
            .map(|(b, _)| *b)
            .next_back()
            .unwrap_or(1);
        bands.push((label, band));

        println!(
            "    GB/s @ N=1: {:.0}   sequential baseline drift: {:.1}%   viable band: N <= {band}\n",
            bandwidth(base_ms),
            drift * 100.0
        );
    }

    println!("── VERDICT ──");
    println!("worst-case cost(3)/cost(1) across shapes  (breakeven {BREAKEVEN_E}x)");
    println!("  sequential:  {worst_n3_seq:.2}x");
    println!("  interleaved: {worst_n3_int:.2}x   <- the verdict is drawn from this");
    println!("\nviable band per shape (interleaved):");
    for (label, band) in &bands {
        println!("  {label:<10} N <= {band}");
    }
    let overall_band = bands.iter().map(|(_, b)| *b).min().unwrap_or(1);
    println!("  {:<10} N <= {overall_band}   <- binding constraint", "OVERALL");

    if worst_n3_int < BREAKEVEN_E {
            println!(
                "\nARTIFACT — batched verify is affordable on this device, so the llama.cpp\n\
                 Metal penalty is an implementation issue, not a hardware limit.\n\
                 => mtp+ddtree is viable on M3; MTP can be gated on Metal, not CUDA-only."
            );
        } else {
            println!(
                "\nFUNDAMENTAL — batched verify costs more than the tokens it can win back.\n\
                 Speculative decoding cannot pay on this device at any acceptance rate.\n\
                 => MTP is a CUDA/4090-only opt-in; do not gate it on Metal."
            );
        }
}
