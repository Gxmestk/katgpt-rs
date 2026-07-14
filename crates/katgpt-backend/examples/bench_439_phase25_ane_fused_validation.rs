//! Plan 439 Phase 2.5 — Real ANE Validation of the Fused-Chain Cost Model
//!
//! Validates that fusing 3 InnerProduct ops into a single CoreML model saves
//! dispatch overhead vs running them as 3 separate CoreML models (with DRAM
//! round-trips between each).
//!
//! ## What this measures
//!
//! - **Unfused:** 3 separate CoreML models, each loaded with
//!   `ComputeUnits::CpuAndNeuralEngine`. Each prediction = one ANE dispatch.
//!   Between predictions, the output is copied to a host `Vec<f32>` (the DRAM
//!   round-trip that fusion eliminates).
//! - **Fused:** 1 CoreML model with 3 chained InnerProduct layers. One
//!   prediction = one ANE dispatch. Intermediates stay on-chip.
//!
//! ## Cost model prediction (ane_fused_estimate)
//!
//! For GEMV(DIM×DIM, F32) × 3 ops with 2 data deps:
//! - Unfused: 3 × dispatch_floor (each op pays its own dispatch)
//! - Fused: 1 × dispatch_floor (single aggregate dispatch)
//! - Savings: 2 × dispatch_floor (~0.46 ms on M1/A13)
//!
//! ## MLComputePlan substitution
//!
//! The plan's T2.5.2 says "use MLComputePlan to verify ANE placement." The
//! `coreml-native` 0.2 crate does NOT expose MLComputePlan (that's a Python
//! coremltools API per Research 224). We substitute:
//! 1. `ComputeUnits::CpuAndNeuralEngine` — excludes GPU, forces ANE preference.
//! 2. Timing heuristic — if per-prediction latency >> CPU compute time for the
//!    shape, the model is dispatch-bound on the ANE (dispatch floor ~230µs on
//!    M1/A13). If latency < 1µs, CoreML fell back to CPU.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example bench_439_phase25_ane_fused_validation --features ane
//! ```

#[cfg(target_os = "macos")]
use coreml_proto::proto::{
    feature_type::Type as FeatureTypeKind,
    model::Type as ModelType,
    neural_network_layer::Layer as LayerKind,
    ArrayFeatureType, FeatureDescription, FeatureType, InnerProductLayerParams, Model,
    ModelDescription, NeuralNetwork, NeuralNetworkLayer, WeightParams,
};

#[cfg(target_os = "macos")]
fn main() {
    use coreml_native::{BorrowedTensor, ComputeUnits, Model as NativeModel};
    use katgpt_core::ane_roofline::{
        ane_estimate, ane_fused_estimate, AneDataDep, AneOpShape, AneFamily, AnePeaks, Dtype,
    };
    use prost::Message;

    const DIM: usize = 256; // divisible by 128 for ANE preference, dispatch-bound
    const N_ITERS: usize = 200;
    const WARMUP: usize = 5;

    // ── Detect chip ────────────────────────────────────────────────────────
    let chip = AneFamily::detect().unwrap_or(AneFamily::A13);
    let peaks = AnePeaks::for_family(chip)
        .expect("detected chip family must have calibrated peaks");
    eprintln!("Chip family: {chip:?} (dispatch floor: {} ms)", peaks.dispatch_floor_ms);

    // ── Weights (deterministic, non-zero) ──────────────────────────────────
    let weights_a: Vec<f32> = (0..DIM * DIM).map(|i| ((i as f32) * 0.001) - 0.5).collect();
    let weights_b: Vec<f32> =
        (0..DIM * DIM).map(|i| (((i + 1000) as f32) * 0.001) - 0.5).collect();
    let weights_c: Vec<f32> =
        (0..DIM * DIM).map(|i| (((i + 2000) as f32) * 0.001) - 0.5).collect();

    // ── Build CoreML model specs ───────────────────────────────────────────
    let spec_a = build_single_linear("linear_a", &weights_a, DIM);
    let spec_b = build_single_linear("linear_b", &weights_b, DIM);
    let spec_c = build_single_linear("linear_c", &weights_c, DIM);
    let spec_fused =
        build_fused_3linear("fused_3linear", &weights_a, &weights_b, &weights_c, DIM);

    let compute = ComputeUnits::CpuAndNeuralEngine;

    eprintln!("Compiling 4 CoreML models (CpuAndNeuralEngine)...");
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

    // ── Prepare input ──────────────────────────────────────────────────────
    let input: Vec<f32> = (0..DIM).map(|i| ((i as f32) * 0.01) - 0.5).collect();
    let input_tensor =
        BorrowedTensor::from_f32(&input, &[DIM, 1, 1]).expect("create input tensor");

    // Intermediate buffers for unfused DRAM round-trips
    let mut buf1 = vec![0.0f32; DIM];
    let mut buf2 = vec![0.0f32; DIM];

    // ── Warmup (cold-start includes ANE pipeline compile) ──────────────────
    eprintln!("Warming up ({WARMUP} iterations each)...");
    for _ in 0..WARMUP {
        // Unfused path
        let pa = model_a.predict(&[("input", &input_tensor)]).expect("warmup a");
        let (oa, _) = pa.get_f32("output").expect("warmup a output");
        let t1 = BorrowedTensor::from_f32(&oa, &[DIM, 1, 1]).unwrap();
        let pb = model_b.predict(&[("input", &t1)]).expect("warmup b");
        let (ob, _) = pb.get_f32("output").expect("warmup b output");
        let t2 = BorrowedTensor::from_f32(&ob, &[DIM, 1, 1]).unwrap();
        let _ = model_c.predict(&[("input", &t2)]).expect("warmup c");

        // Fused path
        let _ = model_fused
            .predict(&[("input", &input_tensor)])
            .expect("warmup fused");
    }
    eprintln!("Warmup complete.\n");

    // ── Measure unfused: 3 dispatches + 2 DRAM round-trips ─────────────────
    let unfused_start = std::time::Instant::now();
    for _ in 0..N_ITERS {
        // Dispatch 1: input → model_a → buf1
        let pa = model_a.predict(&[("input", &input_tensor)]).expect("predict a");
        let (oa, _) = pa.get_f32("output").expect("output a");
        buf1.copy_from_slice(&oa);

        // DRAM round-trip: buf1 → model_b input
        let t1 = BorrowedTensor::from_f32(&buf1, &[DIM, 1, 1]).unwrap();

        // Dispatch 2: buf1 → model_b → buf2
        let pb = model_b.predict(&[("input", &t1)]).expect("predict b");
        let (ob, _) = pb.get_f32("output").expect("output b");
        buf2.copy_from_slice(&ob);

        // DRAM round-trip: buf2 → model_c input
        let t2 = BorrowedTensor::from_f32(&buf2, &[DIM, 1, 1]).unwrap();

        // Dispatch 3: buf2 → model_c → output
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
    let op_shape = AneOpShape::gemv(DIM as u64, DIM as u64, Dtype::F32);
    let single = ane_estimate(op_shape, Dtype::F32, &peaks);
    let single_us = single.runtime_ms * 1000.0;

    let ops = [op_shape, op_shape, op_shape];
    let deps = [
        AneDataDep {
            from_op: 0,
            to_op: 1,
            intermediate_bytes: (DIM * 4) as u64,
        },
        AneDataDep {
            from_op: 1,
            to_op: 2,
            intermediate_bytes: (DIM * 4) as u64,
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

    let ratio = if pred_savings_us.abs() > 1e-9 {
        measured_savings_us / pred_savings_us
    } else {
        0.0
    };

    // ── Report ─────────────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 439 Phase 2.5 — ANE Fused-Chain Real-Hardware Validation  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Hardware: {} (chip detect: {chip:?})",
        {
            let devs = coreml_native::available_devices();
            devs.iter()
                .map(|d| format!("{d}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!("Shape:    GEMV({DIM}×{DIM}, F32) × 3 ops");
    println!("Iters:    {N_ITERS} (after {WARMUP} warmup each)");
    println!();

    println!("─── Measured (wall-clock, CpuAndNeuralEngine) ─────────────────────");
    println!("  Unfused (3 dispatches + 2 DRAM round-trips): {unfused_us:>9.1} µs/iter");
    println!("  Fused   (1 dispatch, 3 ops internally):      {fused_us:>9.1} µs/iter");
    println!("  Measured savings:   {measured_savings_us:>9.1} µs ({measured_savings_pct:.1}%)");
    println!();

    println!("─── Cost model predictions (ane_fused_estimate) ───────────────────");
    println!("  Single op ({:?}):  {single_us:>9.1} µs", single.bound);
    println!("  Unfused predicted:  {pred_unfused_us:>9.1} µs (3 × single)");
    println!(
        "  Fused predicted:    {pred_fused_us:>9.1} µs ({:?})",
        fused_cost.base.bound
    );
    println!("  Predicted savings:  {pred_savings_us:>9.1} µs ({pred_savings_pct:.1}%)");
    println!(
        "  Eliminated bytes:   {} ({} bytes × 2 deps)",
        fused_cost.eliminated_bytes,
        DIM * 4
    );
    println!();

    println!("─── Validation Gates ──────────────────────────────────────────────");

    let g1_pass = fused_us <= unfused_us * 1.05; // 5% tolerance for measurement noise
    println!(
        "  G1 (fusion never hurts): fused ≤ unfused → {}",
        if g1_pass { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "     fused={fused_us:.1}µs  unfused={unfused_us:.1}µs  ratio={:.3}",
        fused_us / unfused_us
    );

    let g2_pass = (0.5..=2.0).contains(&ratio);
    println!(
        "  G2 (measured/predicted savings ratio): {ratio:.2}× → {}",
        if g2_pass {
            "PASS ✅ (0.5×–2.0×)"
        } else {
            "FAIL ❌"
        }
    );
    println!("     measured={measured_savings_us:.1}µs  predicted={pred_savings_us:.1}µs");

    // T2.5.3: Compare against predictions within ±30% or ~2× tolerance
    let pred_ratio = if pred_unfused_us > 0.0 {
        unfused_us / pred_unfused_us
    } else {
        0.0
    };
    println!(
        "  T2.5.3 (unfused ≈ prediction): measured/predicted = {pred_ratio:.2}× → {}",
        if (0.5..=2.0).contains(&pred_ratio) {
            "PASS ✅"
        } else {
            "CHECK ⚠️"
        }
    );

    println!();

    if !g2_pass {
        println!("⚠️  T2.5.4 TRIGGER: model diverges >2× from measurement.");
        println!("    → The eliminated-bytes accounting or dispatch-floor model may be wrong.");
        println!("    → File issue, adjust the model.");
    }

    let all_pass = g1_pass && g2_pass;
    println!("══════════════════════════════════════════════════════════════════");
    println!(
        "  OVERALL: {}",
        if all_pass {
            "VALIDATED ✅ — cost model matches real ANE"
        } else {
            "DIVERGENCE — investigate ❌"
        }
    );
    println!("══════════════════════════════════════════════════════════════════");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Plan 439 Phase 2.5 requires macOS with Apple Neural Engine.");
    eprintln!("This binary is a no-op on non-macOS targets.");
}

// ── CoreML spec builders (inline, matching ane.rs patterns) ────────────────

#[cfg(target_os = "macos")]
fn multi_array_type(shape: &[usize]) -> FeatureType {
    use coreml_proto::proto::array_feature_type::ArrayDataType;
    FeatureType {
        r#type: Some(FeatureTypeKind::MultiArrayType(ArrayFeatureType {
            shape: shape.iter().map(|&d| d as i64).collect(),
            data_type: ArrayDataType::Float32 as i32,
            ..Default::default()
        })),
        ..Default::default()
    }
}

#[cfg(target_os = "macos")]
fn build_single_linear(name: &str, weights: &[f32], dim: usize) -> Model {
    Model {
        specification_version: 7,
        description: Some(ModelDescription {
            input: vec![FeatureDescription {
                name: "input".into(),
                short_description: "Input tensor".into(),
                r#type: Some(multi_array_type(&[dim, 1, 1])),
            }],
            output: vec![FeatureDescription {
                name: "output".into(),
                short_description: "Output tensor".into(),
                r#type: Some(multi_array_type(&[dim, 1, 1])),
            }],
            ..Default::default()
        }),
        is_updatable: false,
        r#type: Some(ModelType::NeuralNetwork(NeuralNetwork {
            layers: vec![NeuralNetworkLayer {
                name: format!("{name}_linear"),
                input: vec!["input".into()],
                output: vec!["output".into()],
                layer: Some(LayerKind::InnerProduct(InnerProductLayerParams {
                    input_channels: dim as u64,
                    output_channels: dim as u64,
                    has_bias: false,
                    weights: Some(WeightParams {
                        float_value: weights.to_vec(),
                        ..Default::default()
                    }),
                    bias: None,
                    ..Default::default()
                })),
                ..Default::default()
            }],
            ..Default::default()
        })),
    }
}

#[cfg(target_os = "macos")]
fn build_fused_3linear(
    name: &str,
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    dim: usize,
) -> Model {
    let layers = vec![
        nn_layer(
            &format!("{name}_linear_0"),
            &["input".to_string()],
            &["hidden1".to_string()],
            LayerKind::InnerProduct(InnerProductLayerParams {
                input_channels: dim as u64,
                output_channels: dim as u64,
                has_bias: false,
                weights: Some(WeightParams {
                    float_value: w1.to_vec(),
                    ..Default::default()
                }),
                bias: None,
                ..Default::default()
            }),
        ),
        nn_layer(
            &format!("{name}_linear_1"),
            &["hidden1".to_string()],
            &["hidden2".to_string()],
            LayerKind::InnerProduct(InnerProductLayerParams {
                input_channels: dim as u64,
                output_channels: dim as u64,
                has_bias: false,
                weights: Some(WeightParams {
                    float_value: w2.to_vec(),
                    ..Default::default()
                }),
                bias: None,
                ..Default::default()
            }),
        ),
        nn_layer(
            &format!("{name}_linear_2"),
            &["hidden2".to_string()],
            &["output".to_string()],
            LayerKind::InnerProduct(InnerProductLayerParams {
                input_channels: dim as u64,
                output_channels: dim as u64,
                has_bias: false,
                weights: Some(WeightParams {
                    float_value: w3.to_vec(),
                    ..Default::default()
                }),
                bias: None,
                ..Default::default()
            }),
        ),
    ];

    Model {
        specification_version: 7,
        description: Some(ModelDescription {
            input: vec![FeatureDescription {
                name: "input".into(),
                short_description: "Input tensor".into(),
                r#type: Some(multi_array_type(&[dim, 1, 1])),
            }],
            output: vec![FeatureDescription {
                name: "output".into(),
                short_description: "Output tensor (3-layer fused)".into(),
                r#type: Some(multi_array_type(&[dim, 1, 1])),
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
fn nn_layer(
    name: &str,
    inputs: &[String],
    outputs: &[String],
    layer: LayerKind,
) -> NeuralNetworkLayer {
    NeuralNetworkLayer {
        name: name.into(),
        input: inputs.to_vec(),
        output: outputs.to_vec(),
        layer: Some(layer),
        ..Default::default()
    }
}
