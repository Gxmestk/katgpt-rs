//! Plan 439 Phase 3 Gate Check — T3.3: Compute-Heavy Fused Conv Chain
//!
//! This benchmark is the **trigger-condition check** for Phase 3 (tile-level
//! cross-op overlap). It tests whether `ane_fused_estimate` accurately
//! predicts real ANE latency for a **memory-bound conv chain with substantial
//! intermediates** — a meaningfully different regime from Phase 2.5's
//! dispatch-bound GEMV(256×256).
//!
//! ## Verdict logic
//!
//! - If `ane_fused_estimate` matches measurement (ratio ≤ 2×) → Phase 1 is
//!   sufficient → **close Plan 439**.
//! - If `ane_fused_estimate` over-predicts by > 2× → tile-level overlap is
//!   happening → **Phase 3 justified** (trigger fires).
//!
//! ## Test case
//!
//! 3× Conv2d(3×3, SAME) with Cin=Cout=192, H=W=32, F32:
//! - Per-op FLOPs = 679M, bytes = 2.9 MB (memory-bound, 40% above dispatch floor)
//! - Intermediate = 768 KB; 2 intermediates = 1.5 MB < 2 MB working set → fit
//! - Cost model predicts: unfused 0.97 ms, fused 0.79 ms, savings 18%
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example bench_439_phase3_gate_check --features ane
//! ```

#[cfg(target_os = "macos")]
use coreml_proto::proto::{
    ArrayFeatureType, ConvolutionLayerParams, FeatureDescription, FeatureType, Model,
    ModelDescription, NeuralNetwork, NeuralNetworkLayer, SamePadding, WeightParams,
    convolution_layer_params::ConvolutionPaddingType, feature_type::Type as FeatureTypeKind,
    model::Type as ModelType, neural_network_layer::Layer as LayerKind,
};

/// Conv spatial height (module-level so builder helpers can access it).
#[cfg(target_os = "macos")]
const H: i64 = 32;
/// Conv spatial width.
#[cfg(target_os = "macos")]
const W: i64 = 32;
/// Kernel height.
#[cfg(target_os = "macos")]
const KH: u64 = 3;
/// Kernel width.
#[cfg(target_os = "macos")]
const KW: u64 = 3;

#[cfg(target_os = "macos")]
fn main() {
    use coreml_native::{BorrowedTensor, ComputeUnits, Model as NativeModel};
    use katgpt_core::ane_roofline::{
        AneDataDep, AneFamily, AneOpShape, AnePeaks, Dtype, ane_estimate, ane_fused_estimate,
    };
    use prost::Message;

    // ── Conv shape: 3×3 SAME, Cin=Cout=192, H=W=32, F32 ──────────────────
    // Memory-bound: per-op memory_ms (0.322) > compute_ms (0.209) > dispatch (0.23 on M1).
    // Intermediates (768 KB × 2 = 1.5 MB) fit the 2 MB working set → fusion eliminates them.
    const C: usize = 192; // channels (multiple of 16 for ANE alignment)
    const N_OPS: usize = 3; // 3-op fused chain
    const N_ITERS: usize = 200;
    const WARMUP: usize = 5;

    const SPATIAL: usize = C * (H as usize) * (W as usize); // elements per feature map
    const WEIGHT_ELEMS: usize = C * C * (KH * KW) as usize; // per conv layer

    // BorrowedTensor::from_f32 takes &[usize], not &[i64]
    // CoreML NeuralNetwork requires 3D input [C, H, W] (not 4D NCHW)
    const SHAPE: [usize; 3] = [C, H as usize, W as usize];

    // ── Detect chip ────────────────────────────────────────────────────────
    let chip = AneFamily::detect().unwrap_or(AneFamily::A13);
    let peaks =
        AnePeaks::for_family(chip).expect("detected chip family must have calibrated peaks");
    eprintln!(
        "Chip family: {chip:?} (dispatch floor: {} ms)",
        peaks.dispatch_floor_ms
    );

    // ── Deterministic non-zero weights ─────────────────────────────────────
    let w1: Vec<f32> = (0..WEIGHT_ELEMS)
        .map(|i| ((i as f32) * 0.0001) - 0.5)
        .collect();
    let w2: Vec<f32> = (0..WEIGHT_ELEMS)
        .map(|i| (((i + 10_000) as f32) * 0.0001) - 0.5)
        .collect();
    let w3: Vec<f32> = (0..WEIGHT_ELEMS)
        .map(|i| (((i + 20_000) as f32) * 0.0001) - 0.5)
        .collect();

    // ── Build CoreML model specs ───────────────────────────────────────────
    let spec_a = build_single_conv("conv_a", &w1, C);
    let spec_b = build_single_conv("conv_b", &w2, C);
    let spec_c = build_single_conv("conv_c", &w3, C);
    let spec_fused = build_fused_3conv("fused_3conv", &w1, &w2, &w3, C);

    let compute = ComputeUnits::CpuAndNeuralEngine;

    eprintln!("Compiling 4 CoreML models (CpuAndNeuralEngine)...");
    eprintln!("  Conv shape: {}×{}×{} (CHW), 3×3 SAME, F32", C, H, W);
    eprintln!(
        "  Weight per layer: {} floats ({} KB)",
        WEIGHT_ELEMS,
        WEIGHT_ELEMS * 4 / 1024
    );
    eprintln!(
        "  Intermediate: {} floats ({} KB)",
        SPATIAL,
        SPATIAL * 4 / 1024
    );

    let model_a = NativeModel::load_from_bytes(&spec_a.encode_to_vec(), compute)
        .expect("load spec_a")
        .block_on()
        .expect("compile model_a");
    let model_b = NativeModel::load_from_bytes(&spec_b.encode_to_vec(), compute)
        .expect("load spec_b")
        .block_on()
        .expect("compile model_b");
    let model_c = NativeModel::load_from_bytes(&spec_c.encode_to_vec(), compute)
        .expect("load spec_c")
        .block_on()
        .expect("compile model_c");
    let model_fused = NativeModel::load_from_bytes(&spec_fused.encode_to_vec(), compute)
        .expect("load spec_fused")
        .block_on()
        .expect("compile model_fused");
    eprintln!("Compilation complete.\n");

    // ── Prepare input [1, C, H, W] ─────────────────────────────────────────
    let input: Vec<f32> = (0..SPATIAL).map(|i| ((i as f32) * 0.001) - 0.5).collect();
    let input_tensor = BorrowedTensor::from_f32(&input, &SHAPE).expect("input tensor");

    // Intermediate buffers for unfused DRAM round-trips
    let mut buf1 = vec![0.0f32; SPATIAL];
    let mut buf2 = vec![0.0f32; SPATIAL];

    // ── Warmup ─────────────────────────────────────────────────────────────
    eprintln!("Warming up ({WARMUP} iterations each)...");
    for _ in 0..WARMUP {
        let pa = model_a
            .predict(&[("input", &input_tensor)])
            .expect("warmup a");
        let (oa, _) = pa.get_f32("output").expect("warmup a output");
        let t1 = BorrowedTensor::from_f32(&oa, &SHAPE).unwrap();
        let pb = model_b.predict(&[("input", &t1)]).expect("warmup b");
        let (ob, _) = pb.get_f32("output").expect("warmup b output");
        let t2 = BorrowedTensor::from_f32(&ob, &SHAPE).unwrap();
        let _ = model_c.predict(&[("input", &t2)]).expect("warmup c");
        let _ = model_fused
            .predict(&[("input", &input_tensor)])
            .expect("warmup fused");
    }
    eprintln!("Warmup complete.\n");

    // ── Measure unfused: 3 dispatches + 2 DRAM round-trips ─────────────────
    let unfused_start = std::time::Instant::now();
    for _ in 0..N_ITERS {
        let pa = model_a
            .predict(&[("input", &input_tensor)])
            .expect("predict a");
        let (oa, _) = pa.get_f32("output").expect("output a");
        buf1.copy_from_slice(&oa);
        let t1 = BorrowedTensor::from_f32(&buf1, &SHAPE).unwrap();
        let pb = model_b.predict(&[("input", &t1)]).expect("predict b");
        let (ob, _) = pb.get_f32("output").expect("output b");
        buf2.copy_from_slice(&ob);
        let t2 = BorrowedTensor::from_f32(&buf2, &SHAPE).unwrap();
        let pc = model_c.predict(&[("input", &t2)]).expect("predict c");
        let (_oc, _) = pc.get_f32("output").expect("output c");
    }
    let unfused_elapsed = unfused_start.elapsed();
    let unfused_us = unfused_elapsed.as_secs_f64() * 1e6 / N_ITERS as f64;

    // ── Measure fused: 1 dispatch, 3 ops internally ────────────────────────
    let fused_start = std::time::Instant::now();
    for _ in 0..N_ITERS {
        let pf = model_fused
            .predict(&[("input", &input_tensor)])
            .expect("predict fused");
        let (_of, _) = pf.get_f32("output").expect("output fused");
    }
    let fused_elapsed = fused_start.elapsed();
    let fused_us = fused_elapsed.as_secs_f64() * 1e6 / N_ITERS as f64;

    // ── Cost model predictions ─────────────────────────────────────────────
    let op_shape = AneOpShape::conv_3x3(C as u64, C as u64, H as u64, W as u64, Dtype::F32);
    let single = ane_estimate(op_shape, Dtype::F32, &peaks);

    let ops = [op_shape; N_OPS];
    let intermediate_bytes = (SPATIAL * 4) as u64;
    let deps = [
        AneDataDep {
            from_op: 0,
            to_op: 1,
            intermediate_bytes,
        },
        AneDataDep {
            from_op: 1,
            to_op: 2,
            intermediate_bytes,
        },
    ];
    let fused_cost = ane_fused_estimate(&ops, &deps, Dtype::F32, &peaks);

    let pred_unfused_us = fused_cost.sequential_runtime_ms * 1000.0;
    let pred_fused_us = fused_cost.base.runtime_ms * 1000.0;
    let pred_savings_us = fused_cost.fusion_savings_ms * 1000.0;

    let measured_savings_us = unfused_us - fused_us;
    let measured_savings_pct = (measured_savings_us / unfused_us) * 100.0;
    let pred_savings_pct =
        (fused_cost.fusion_savings_ms / fused_cost.sequential_runtime_ms) * 100.0;

    let savings_ratio = if pred_savings_us.abs() > 1e-9 {
        measured_savings_us / pred_savings_us
    } else {
        0.0
    };

    // THE KEY GATE: fused measured/predicted ratio.
    // If real fused latency << predicted → tile-level overlap happening.
    let fused_pred_ratio = if pred_fused_us > 0.0 {
        fused_us / pred_fused_us
    } else {
        0.0
    };

    // ── Report ─────────────────────────────────────────────────────────────
    println!();
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 439 Phase 3 Gate Check — T3.3 Compute-Heavy Conv Chain      ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Hardware: {} (chip detect: {chip:?})", {
        let devs = coreml_native::available_devices();
        devs.iter()
            .map(|d| format!("{d}"))
            .collect::<Vec<_>>()
            .join(", ")
    });
    println!("Shape:    Conv2d(3×3, SAME) Cin=Cout={C}, H=W={H}, F32 × {N_OPS} ops");
    println!("Iters:    {N_ITERS} (after {WARMUP} warmup each)");
    println!();

    // ANE residency heuristic: compare fused latency against single-op
    // compute time on ANE FP16 peaks. If fused >> 3× single-op compute,
    // CoreML likely fell back to CPU (ANE would be faster).
    // Correct units: TFLOP/s = 1e12 FLOP/s → µs = flops / (TFLOPS * 1e6)
    let single_compute_us = op_shape.flops as f64 / (peaks.compute_tflops_fp16 * 1e6);
    let fused_compute_us = single_compute_us * N_OPS as f64;
    let ane_likely = fused_us < fused_compute_us * 4.0;
    println!(
        "ANE residency: {} (fused {} µs vs 3×single-op compute {} µs on FP16 peaks)",
        if ane_likely {
            "LIKELY ✅"
        } else {
            "CPU FALLBACK ⚠️"
        },
        fused_us as u64,
        fused_compute_us as u64,
    );
    println!();

    println!("─── Measured (wall-clock, CpuAndNeuralEngine) ─────────────────────");
    println!("  Unfused ({N_OPS} dispatches + 2 DRAM round-trips): {unfused_us:>9.1} µs/iter");
    println!("  Fused   (1 dispatch, {N_OPS} ops internally):      {fused_us:>9.1} µs/iter");
    println!("  Measured savings:   {measured_savings_us:>9.1} µs ({measured_savings_pct:.1}%)");
    println!();

    println!("─── Cost model predictions (ane_fused_estimate) ───────────────────");
    println!(
        "  Single op ({:?}):  {:>9.1} µs",
        single.bound,
        single.runtime_ms * 1000.0
    );
    println!("  Unfused predicted:  {pred_unfused_us:>9.1} µs ({N_OPS} × single)");
    println!(
        "  Fused predicted:    {pred_fused_us:>9.1} µs ({:?})",
        fused_cost.base.bound
    );
    println!("  Predicted savings:  {pred_savings_us:>9.1} µs ({pred_savings_pct:.1}%)");
    println!(
        "  Eliminated bytes:   {} ({} bytes × {} deps, n_fused={})",
        fused_cost.eliminated_bytes,
        intermediate_bytes,
        N_OPS - 1,
        fused_cost.n_fused_deps,
    );
    println!();

    println!("─── Phase 3 Gate Check ────────────────────────────────────────────");
    println!();

    // G1: fusion never hurts
    let g1_pass = fused_us <= unfused_us * 1.05;
    println!(
        "  G1 (fusion never hurts): fused ≤ unfused → {}",
        if g1_pass { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "     fused={fused_us:.1}µs  unfused={unfused_us:.1}µs  ratio={:.3}",
        fused_us / unfused_us
    );

    // G2: measured/predicted savings ratio
    let g2_pass = (0.5..=2.0).contains(&savings_ratio);
    println!(
        "  G2 (savings ratio): measured/predicted = {savings_ratio:.2}× → {}",
        if g2_pass {
            "PASS ✅ (0.5×–2.0×)"
        } else {
            "FAIL ❌"
        }
    );
    println!("     measured={measured_savings_us:.1}µs  predicted={pred_savings_us:.1}µs");

    // G3: fused measured/predicted ratio — THE PHASE 3 TRIGGER
    // If fused_pred_ratio < 0.5, the real ANE is >2× faster than predicted.
    // That means tile-level overlap is happening that the model doesn't capture.
    let g3_pass = (0.5..=2.0).contains(&fused_pred_ratio);
    println!(
        "  G3 (fused latency ratio): measured/predicted = {fused_pred_ratio:.2}× → {}",
        if g3_pass {
            "PASS ✅ — Phase 1 is accurate"
        } else if fused_pred_ratio < 0.5 {
            "TRIGGER 🔥 — Phase 3 justified"
        } else {
            "CHECK ⚠️ — model under-predicts"
        }
    );
    println!("     measured_fused={fused_us:.1}µs  predicted_fused={pred_fused_us:.1}µs");
    println!();

    // ── Verdict ────────────────────────────────────────────────────────────
    let all_pass = g1_pass && g2_pass && g3_pass;
    println!("══════════════════════════════════════════════════════════════════");
    if all_pass {
        println!("  VERDICT: PHASE 1 SUFFICIENT ✅");
        println!("  ane_fused_estimate matches reality within 2× for this regime.");
        println!("  Phase 3 (tile-level overlap) is NOT justified.");
        println!("  → Close Plan 439 (Phase 3 permanently deferred).");
    } else if !g3_pass && fused_pred_ratio < 0.5 {
        println!("  VERDICT: PHASE 3 JUSTIFIED 🔥");
        println!("  Real ANE is >2× faster than predicted for fused conv chain.");
        println!("  Tile-level cross-op overlap is happening.");
        println!("  → Phase 3 trigger fires. Begin T3.1 (tile-graph DAG).");
    } else {
        println!("  VERDICT: INCONCLUSIVE ⚠️");
        println!("  Some gates failed but not in the Phase 3 direction.");
        println!("  Investigate model accuracy before proceeding.");
    }
    println!("══════════════════════════════════════════════════════════════════");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Plan 439 Phase 3 Gate Check requires macOS with Apple Neural Engine.");
    eprintln!("This binary is a no-op on non-macOS targets.");
}

// ── CoreML spec builders ───────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn multi_array_type(shape: &[i64]) -> FeatureType {
    use coreml_proto::proto::array_feature_type::ArrayDataType;
    FeatureType {
        r#type: Some(FeatureTypeKind::MultiArrayType(ArrayFeatureType {
            shape: shape.to_vec(),
            data_type: ArrayDataType::Float32 as i32,
            ..Default::default()
        })),
        ..Default::default()
    }
}

#[cfg(target_os = "macos")]
fn build_single_conv(name: &str, weights: &[f32], channels: usize) -> Model {
    Model {
        specification_version: 7,
        description: Some(ModelDescription {
            input: vec![FeatureDescription {
                name: "input".into(),
                short_description: "Input tensor (NCHW)".into(),
                r#type: Some(multi_array_type(&[channels as i64, H, W])),
            }],
            output: vec![FeatureDescription {
                name: "output".into(),
                short_description: "Output tensor (NCHW)".into(),
                r#type: Some(multi_array_type(&[channels as i64, H, W])),
            }],
            ..Default::default()
        }),
        is_updatable: false,
        r#type: Some(ModelType::NeuralNetwork(NeuralNetwork {
            layers: vec![conv_layer(
                &format!("{name}_conv"),
                &["input".to_string()],
                &["output".to_string()],
                channels,
                weights,
            )],
            ..Default::default()
        })),
    }
}

#[cfg(target_os = "macos")]
fn build_fused_3conv(name: &str, w1: &[f32], w2: &[f32], w3: &[f32], channels: usize) -> Model {
    let layers = vec![
        conv_layer(
            &format!("{name}_conv_0"),
            &["input".to_string()],
            &["hidden1".to_string()],
            channels,
            w1,
        ),
        conv_layer(
            &format!("{name}_conv_1"),
            &["hidden1".to_string()],
            &["hidden2".to_string()],
            channels,
            w2,
        ),
        conv_layer(
            &format!("{name}_conv_2"),
            &["hidden2".to_string()],
            &["output".to_string()],
            channels,
            w3,
        ),
    ];

    Model {
        specification_version: 7,
        description: Some(ModelDescription {
            input: vec![FeatureDescription {
                name: "input".into(),
                short_description: "Input tensor (NCHW)".into(),
                r#type: Some(multi_array_type(&[channels as i64, H, W])),
            }],
            output: vec![FeatureDescription {
                name: "output".into(),
                short_description: "Output tensor (3-conv fused)".into(),
                r#type: Some(multi_array_type(&[channels as i64, H, W])),
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

#[cfg(target_os = "macos")]
fn conv_layer(
    name: &str,
    inputs: &[String],
    outputs: &[String],
    channels: usize,
    weights: &[f32],
) -> NeuralNetworkLayer {
    NeuralNetworkLayer {
        name: name.into(),
        input: inputs.to_vec(),
        output: outputs.to_vec(),
        layer: Some(LayerKind::Convolution(ConvolutionLayerParams {
            output_channels: channels as u64,
            kernel_channels: channels as u64,
            n_groups: 1,
            kernel_size: vec![KH, KW],
            stride: vec![1, 1],
            dilation_factor: vec![],
            is_deconvolution: false,
            has_bias: false,
            weights: Some(WeightParams {
                float_value: weights.to_vec(),
                ..Default::default()
            }),
            bias: None,
            output_shape: vec![],
            // SAME padding: output spatial dims == input spatial dims
            convolution_padding_type: Some(ConvolutionPaddingType::Same(SamePadding {
                ..Default::default()
            })),
        })),
        ..Default::default()
    }
}
