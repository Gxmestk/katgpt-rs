//! Loop Stability PoC — Defend-Wrong Benchmark
//!
//! Tests 6 parameter-free architectural fixes for looped transformer stability
//! at high loop counts (T=12). Self-contained: std-only, no katgpt-rs deps.
//!
//! Papers:
//! - Fully Looped Transformer (arXiv:2605.18797): FLA + Attention Injection
//! - Readout Blind Spot (arXiv:2606.24898): Inter-loop RMSNorm
//!
//! Run:
//!   CARGO_TARGET_DIR=/tmp/loop_stability_poc cargo run --release --example loop_stability_poc

use std::hint::black_box;
use std::time::Instant;

// ── Config ────────────────────────────────────────────────────
const D_MODEL: usize = 256;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const VOCAB: usize = 256;
const N_LAYERS: usize = 6;
const MLP_HIDDEN: usize = 1024;
const T_LOOPS: usize = 12;
const SEED: u32 = 42;
const INIT_STD: f32 = 0.02;
const TOKEN_ID: usize = 42;

// ── PRNG (xorshift32 + Box-Muller) ────────────────────────────
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed | 1 }
    }
    fn u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
    fn f32(&mut self) -> f32 {
        (self.u32() as f32 / u32::MAX as f32).clamp(1e-8, 1.0 - 1e-8)
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.f32();
        let u2 = self.f32();
        (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos()
    }
}

// ── Weight structs ────────────────────────────────────────────
struct LayerW {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
}

struct Weights {
    wte: Vec<f32>,
    layers: Vec<LayerW>,
    lm_head: Vec<f32>,
}

fn init_mat(rng: &mut Rng, rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|_| rng.gauss() * INIT_STD).collect()
}

impl Weights {
    fn new() -> Self {
        let mut rng = Rng::new(SEED);
        Self {
            wte: init_mat(&mut rng, VOCAB, D_MODEL),
            lm_head: init_mat(&mut rng, VOCAB, D_MODEL),
            layers: (0..N_LAYERS)
                .map(|_| LayerW {
                    wq: init_mat(&mut rng, D_MODEL, D_MODEL),
                    wk: init_mat(&mut rng, D_MODEL, D_MODEL),
                    wv: init_mat(&mut rng, D_MODEL, D_MODEL),
                    wo: init_mat(&mut rng, D_MODEL, D_MODEL),
                    w1: init_mat(&mut rng, MLP_HIDDEN, D_MODEL),
                    w2: init_mat(&mut rng, D_MODEL, MLP_HIDDEN),
                })
                .collect(),
        }
    }
}

// ── Math primitives ───────────────────────────────────────────

/// In-place RMSNorm: normalizes x to unit RMS norm.
fn rmsnorm(x: &mut [f32]) {
    let n = x.len() as f32;
    let ss = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ss + 1e-8).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Row-major matvec: y[m] = W[m,k] @ x[k]
fn matvec(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for i in 0..m {
        let row = &w[i * k..(i + 1) * k];
        let mut s = 0.0f32;
        for j in 0..k {
            s += row[j] * x[j];
        }
        y[i] = s;
    }
    y
}

fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

fn rms_val(x: &[f32]) -> f32 {
    let n = x.len() as f32;
    (x.iter().map(|v| v * v).sum::<f32>() / n + 1e-8).sqrt()
}

fn l2_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let mx = x.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - mx).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// KL(p || q) = sum p * ln(p/q)
fn kl_div(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q)
        .map(|(p, q)| {
            if *p > 1e-12 && *q > 1e-12 {
                p * (p / q).ln()
            } else {
                0.0
            }
        })
        .sum()
}

// ── Competitor config ─────────────────────────────────────────
struct Comp {
    name: &'static str,
    inter_norm: bool,
    fla_res: bool,
    attn_inj: bool,
    decay: Option<f32>,
}

struct Res {
    name: &'static str,
    norms: Vec<f32>,
    steps: Vec<f32>,
    kls: Vec<f32>,
    us: f64,
}

// ── Run one competitor ────────────────────────────────────────
fn run(w: &Weights, c: &Comp) -> Res {
    let mut x = w.wte[TOKEN_ID * D_MODEL..(TOKEN_ID + 1) * D_MODEL].to_vec();
    let mut prev_h: Vec<f32> = Vec::new();
    let mut prev_logits: Vec<f32> = Vec::new();
    let mut norms = vec![0.0f32; T_LOOPS];
    let mut steps = vec![0.0f32; T_LOOPS];
    let mut kls = vec![0.0f32; T_LOOPS];
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let t0 = Instant::now();

    for tau in 0..T_LOOPS {
        // Capture input to this loop (for step-size metric)
        let h_in = x.clone();

        // Inter-loop RMSNorm: normalize between loops (tau > 0)
        if c.inter_norm && tau > 0 {
            rmsnorm(&mut x);
        }

        // Run all layers (weight-shared block)
        for layer in &w.layers {
            // FLA-res: add prev_h at start of each layer (tau > 0)
            if c.fla_res && tau > 0 && !prev_h.is_empty() {
                for i in 0..D_MODEL {
                    x[i] += prev_h[i];
                }
            }

            // Pre-attn: save residual, normalize
            let res1 = x.clone();
            rmsnorm(&mut x);

            // QKV projections.
            // Attention Injection: Q from prev_h instead of x (tau > 0).
            let q_src: &[f32] = if c.attn_inj && tau > 0 && !prev_h.is_empty() {
                &prev_h
            } else {
                &x
            };
            let q = matvec(&layer.wq, q_src, D_MODEL, D_MODEL);
            let k = matvec(&layer.wk, &x, D_MODEL, D_MODEL);
            let v = matvec(&layer.wv, &x, D_MODEL, D_MODEL);

            // Single-position attention: softmax(1 element) = 1.0, so attn_out = V.
            // Compute Q·K score for realism (prevents dead-code elimination of Q/K).
            let score: f32 = (0..N_HEADS)
                .map(|h| {
                    let off = h * HEAD_DIM;
                    (0..HEAD_DIM).map(|d| q[off + d] * k[off + d]).sum::<f32>()
                })
                .sum::<f32>()
                * scale;
            black_box(score);

            let attn = matvec(&layer.wo, &v, D_MODEL, D_MODEL);
            for i in 0..D_MODEL {
                x[i] = res1[i] + attn[i];
            }

            // Pre-MLP: save residual, normalize
            let res2 = x.clone();
            rmsnorm(&mut x);

            // MLP: w2 @ ReLU(w1 @ x)
            let mut h = matvec(&layer.w1, &x, MLP_HIDDEN, D_MODEL);
            relu(&mut h);
            let mlp = matvec(&layer.w2, &h, D_MODEL, MLP_HIDDEN);
            for i in 0..D_MODEL {
                x[i] = res2[i] + mlp[i];
            }
        }

        // Fixed decay gate: h^tau = h_tilde + decay * h^(tau-1)
        if let Some(d) = c.decay
            && tau > 0
            && !prev_h.is_empty()
        {
            for i in 0..D_MODEL {
                x[i] += d * prev_h[i];
            }
        }

        // Capture prev_h for next loop
        prev_h = x.clone();

        // Metrics
        norms[tau] = rms_val(&x);
        steps[tau] = if tau > 0 { l2_diff(&x, &h_in) } else { 0.0 };

        let logits = matvec(&w.lm_head, &x, VOCAB, D_MODEL);
        if tau > 0 && !prev_logits.is_empty() {
            kls[tau] = kl_div(&softmax(&logits), &softmax(&prev_logits));
        }
        prev_logits = logits;
    }

    Res {
        name: c.name,
        norms,
        steps,
        kls,
        us: t0.elapsed().as_micros() as f64,
    }
}

// ── Main ──────────────────────────────────────────────────────
fn main() {
    let w = Weights::new();
    let comps = [
        Comp {
            name: "Baseline",
            inter_norm: false,
            fla_res: false,
            attn_inj: false,
            decay: None,
        },
        Comp {
            name: "InterNorm",
            inter_norm: true,
            fla_res: false,
            attn_inj: false,
            decay: None,
        },
        Comp {
            name: "FLA-res",
            inter_norm: false,
            fla_res: true,
            attn_inj: false,
            decay: None,
        },
        Comp {
            name: "AttnInj",
            inter_norm: false,
            fla_res: false,
            attn_inj: true,
            decay: None,
        },
        Comp {
            name: "Combined",
            inter_norm: true,
            fla_res: true,
            attn_inj: false,
            decay: None,
        },
        Comp {
            name: "DecayGate",
            inter_norm: false,
            fla_res: false,
            attn_inj: false,
            decay: Some(0.8),
        },
    ];

    // Run all competitors with identical weights
    let results: Vec<_> = comps.iter().map(|c| run(&w, c)).collect();

    // ── Verdict table ──
    println!("=== Loop Stability PoC — Defend-Wrong Benchmark ===");
    println!(
        "Toy transformer: {} layers, d_model={}, {} heads, vocab={}, T={} loops",
        N_LAYERS, D_MODEL, N_HEADS, VOCAB, T_LOOPS
    );
    println!();
    println!("┌──────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│ Competitor           │ G1 Norm  │ G1 Ratio │ G2 KL    │ G3 Time  │ G4 Step  │");
    println!("│                      │ @τ=12    │ τ12/τ0   │ @τ=12    │ (µs)     │ @τ=12    │");
    println!("├──────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤");
    for r in &results {
        let g1 = r.norms[T_LOOPS - 1];
        let g1r = r.norms[T_LOOPS - 1] / r.norms[0].max(1e-8);
        let g2 = r.kls[T_LOOPS - 1];
        let g3 = r.us;
        let g4 = r.steps[T_LOOPS - 1];
        println!(
            "│ {:<20} │ {:>8.3} │ {:>8.3} │ {:>8.4} │ {:>8.1} │ {:>8.4} │",
            r.name, g1, g1r, g2, g3, g4
        );
    }
    println!("└──────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘");
    println!();

    // ── Per-loop norm trajectory ──
    println!("Per-loop norm trajectory:");
    print!("τ     ");
    for r in &results {
        print!("{:>10} ", r.name);
    }
    println!();
    for tau in 0..T_LOOPS {
        print!("{:<5} ", tau);
        for r in &results {
            print!("{:>10.4} ", r.norms[tau]);
        }
        println!();
    }
    println!();

    // ── Per-loop step size trajectory ──
    println!("Per-loop step size trajectory:");
    print!("τ     ");
    for r in &results {
        print!("{:>10} ", r.name);
    }
    println!();
    for tau in 0..T_LOOPS {
        print!("{:<5} ", tau);
        for r in &results {
            print!("{:>10.4} ", r.steps[tau]);
        }
        println!();
    }
    println!();

    // ── Per-loop KL divergence trajectory ──
    println!("Per-loop KL divergence trajectory:");
    print!("τ     ");
    for r in &results {
        print!("{:>10} ", r.name);
    }
    println!();
    for tau in 0..T_LOOPS {
        print!("{:<5} ", tau);
        for r in &results {
            print!("{:>10.4} ", r.kls[tau]);
        }
        println!();
    }
    println!();

    // ── Verdict ──
    let init = results[0].norms[0];
    let base_us = results[0].us;

    println!("Verdict:");
    println!("- G1 (norm control, keep ratio < 10x initial {:.4}):", init);
    for r in &results {
        let ratio = r.norms[T_LOOPS - 1] / r.norms[0].max(1e-8);
        let pass = ratio < 10.0;
        println!(
            "  {} {} (ratio={:.2})",
            if pass { "✅" } else { "❌" },
            r.name,
            ratio
        );
    }

    println!("- G2 (output stability, keep KL < 1.0):");
    for r in &results {
        let pass = r.kls[T_LOOPS - 1] < 1.0;
        println!(
            "  {} {} (KL={:.4})",
            if pass { "✅" } else { "❌" },
            r.name,
            r.kls[T_LOOPS - 1]
        );
    }

    println!("- G3 (latency, keep overhead < 5%):");
    for r in &results {
        let oh = if base_us > 0.0 {
            (r.us - base_us) / base_us * 100.0
        } else {
            0.0
        };
        let pass = oh < 5.0;
        println!(
            "  {} {} (overhead={:.1}%)",
            if pass { "✅" } else { "❌" },
            r.name,
            oh
        );
    }

    println!("- G4 (convergence, final step < 0.1):");
    for r in &results {
        let pass = r.steps[T_LOOPS - 1] < 0.1;
        println!(
            "  {} {} (step={:.4})",
            if pass { "✅" } else { "❌" },
            r.name,
            r.steps[T_LOOPS - 1]
        );
    }
    println!();

    // ── Overall assessment ──
    println!("Overall assessment:");

    // Check which pass all 4 gates
    let all_pass: Vec<&str> = results
        .iter()
        .filter(|r| {
            let ratio = r.norms[T_LOOPS - 1] / r.norms[0].max(1e-8);
            let oh = if base_us > 0.0 {
                (r.us - base_us) / base_us * 100.0
            } else {
                0.0
            };
            ratio < 10.0 && r.kls[T_LOOPS - 1] < 1.0 && oh < 5.0 && r.steps[T_LOOPS - 1] < 0.1
        })
        .map(|r| r.name)
        .collect();

    if all_pass.is_empty() {
        println!("  No competitor passes all 4 gates at T=12 with this init scale.");
    } else {
        println!("  Passes all 4 gates: {}", all_pass.join(", "));
    }

    // Note AttnInj = Baseline finding
    let base = &results[0];
    let inj = &results[3];
    if (base.norms[T_LOOPS - 1] - inj.norms[T_LOOPS - 1]).abs() < 1e-6 {
        println!("  NOTE: AttnInj == Baseline (norms identical). For single-position attention,");
        println!("        Q does not affect the output (softmax of 1 element = 1.0, attn = V).");
        println!("        Attention Injection is a no-op in this single-token regime.");
    }

    // Note which fixes help with norm control
    let norm_pass: Vec<&str> = results
        .iter()
        .filter(|r| r.norms[T_LOOPS - 1] / r.norms[0].max(1e-8) < 10.0)
        .map(|r| r.name)
        .collect();
    println!("  Norm control (G1 pass): {}", norm_pass.join(", "));

    let kl_pass: Vec<&str> = results
        .iter()
        .filter(|r| r.kls[T_LOOPS - 1] < 1.0)
        .map(|r| r.name)
        .collect();
    println!("  Output stability (G2 pass): {}", kl_pass.join(", "));

    let step_pass: Vec<&str> = results
        .iter()
        .filter(|r| r.steps[T_LOOPS - 1] < 0.1)
        .map(|r| r.name)
        .collect();
    println!("  Convergence (G4 pass): {}", step_pass.join(", "));
}
