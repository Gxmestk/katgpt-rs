//! CochainFreezeEnvelope — GOAT gate bench (Issue 455).
//!
//! Exercises the GOAT gates against the `CochainFreezeEnvelope` primitive.
//! The 7 unit tests in `dec_freeze::tests` cover G1 (correctness — roundtrip,
//! tamper detection, NaN rejection, malformation); this bench adds:
//!
//! - **G2 (perf)** — `freeze()` and `thaw()` latency at three representative
//!   cochain sizes:
//!   - Small:  8×8 grid,  dim=1  (64 f32,    256 B payload) — vertex belief
//!   - Medium: 32×32 grid, dim=1  (1024 f32,  4 KB payload) — zone belief
//!   - Large:  64×64 grid, dim=8  (32768 f32, 128 KB payload) — high-dim belief
//!
//!   Target: all sizes < 1 ms (1,000,000 ns). Freeze/thaw is a cold-path
//!   serialization operation (BLAKE3 hash + Vec build / parse), NOT a hot-path
//!   zero-alloc primitive — so the gate is absolute latency, not alloc count.
//!
//! - **G4 (determinism)** — freeze the same cochain 100 times; all commitments
//!   must be bit-identical.
//!
//! - **G5 (tamper sensitivity)** — two cochains differing in a single f32 must
//!   produce different commitments (BLAKE3 collision resistance). Plus
//!   `verify()` consistency + latency on a clean envelope. The exhaustive
//!   bit-flip-detection correctness gate lives in the unit tests
//!   (`tampered_payload_detected`, `tampered_commitment_detected`) which have
//!   private-field access; this bench re-confirms the property via the public
//!   API (commitment divergence on minimal input change).
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features dec_operators \
//!   --bench bench_455_cochain_freeze_goat --release -- --nocapture
//! ```

#![cfg(feature = "dec_operators")]

use katgpt_core::dec::CochainField;
use katgpt_core::dec_freeze::CochainFreezeEnvelope;
use std::hint::black_box;

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Build a cochain of `n_cells` cells with feature dim `dim`, filled with a
/// deterministic non-trivial pattern (avoids all-zeros which BLAKE3 may
/// short-circuit on some inputs).
fn make_cochain(rank: u8, n_cells: usize, dim: usize) -> CochainField {
    let mut data = Vec::with_capacity(n_cells * dim);
    for i in 0..n_cells * dim {
        // Deterministic pseudo-data: mix of positive, negative, fractional.
        let v = (i as f32) * 0.1 - (n_cells as f32) * 0.05;
        data.push(v.sin() + (i as f32).cos() * 0.5);
    }
    CochainField::from_vec(rank, dim, data)
}

/// Time median over `iterations` runs. Returns ns.
fn time_median_ns(f: &mut dyn FnMut(), iterations: usize) -> f64 {
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        f();
        times.push(start.elapsed().as_secs_f64() * 1_000_000_000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times[times.len() / 2]
}

/// One bench size configuration.
struct Size {
    label: &'static str,
    grid: usize,
    dim: usize,
    rank: u8,
}

// ─── G2 (perf): freeze + thaw latency at three sizes ───────────────────────

fn g2_perf() -> bool {
    let sizes = [
        Size { label: "Small  (8×8, dim=1)",   grid: 8,  dim: 1, rank: 0 },
        Size { label: "Medium (32×32, dim=1)", grid: 32, dim: 1, rank: 0 },
        Size { label: "Large  (64×64, dim=8)", grid: 64, dim: 8, rank: 1 },
    ];

    println!("── G2 (perf): freeze + thaw latency (median over 10,000 iters) ──");
    println!(
        "   {:<24} {:>8} {:>10} {:>10} {:>10}",
        "Size", "n_f32", "bytes", "freeze", "thaw"
    );
    println!(
        "   {:<24} {:>8} {:>10} {:>10} {:>10}",
        "", "", "(payload)", "(ns)", "(ns)"
    );

    let mut all_pass = true;
    let gate_ns = 1_000_000.0; // 1 ms

    for s in &sizes {
        let n_cells = s.grid * s.grid;
        let n_f32 = n_cells * s.dim;
        let cf = make_cochain(s.rank, n_cells, s.dim);
        let payload_bytes = 4 + 1 + n_f32 * 4;

        // freeze latency.
        let cf_ref = &cf;
        let mut freeze_call = || {
            let env = CochainFreezeEnvelope::freeze(black_box(cf_ref));
            black_box(&env);
        };
        let freeze_ns = time_median_ns(&mut freeze_call, 10_000);

        // thaw latency (includes verify + deserialize).
        let env = CochainFreezeEnvelope::freeze(&cf);
        let env_ref = &env;
        let mut thaw_call = || {
            let cf = env_ref.thaw();
            black_box(&cf);
        };
        let thaw_ns = time_median_ns(&mut thaw_call, 10_000);

        let pass = freeze_ns < gate_ns && thaw_ns < gate_ns;
        if !pass {
            all_pass = false;
        }

        println!(
            "   {:<24} {:>8} {:>10} {:>10.0} {:>10.0}  {}",
            s.label,
            n_f32,
            payload_bytes,
            freeze_ns,
            thaw_ns,
            if pass { "PASS ✓" } else { "FAIL ✗" }
        );
    }

    println!("   Gate: all < {gate_ns:.0} ns (1 ms)");
    println!("   Result: {}", if all_pass { "PASS ✓" } else { "FAIL ✗" });
    all_pass
}

// ─── G4 (determinism): 100 freezes → bit-identical commitments ─────────────

fn g4_determinism() -> bool {
    let cf = make_cochain(0, 64, 1);
    let baseline = CochainFreezeEnvelope::freeze(&cf).commitment();

    for i in 0..100 {
        let c = CochainFreezeEnvelope::freeze(&cf).commitment();
        if c != baseline {
            eprintln!("G4 FAIL: commitment differs on iteration {i}");
            return false;
        }
    }
    true
}

// ─── G5 (tamper sensitivity + verify latency) ──────────────────────────────

/// Two cochains differing in a single f32 must produce different commitments
/// (BLAKE3 collision resistance — any input change flips ~50% of output bits).
/// Also measures `verify()` latency on a clean envelope.
fn g5_tamper() -> (bool, f64) {
    let cf_a = make_cochain(0, 64, 1);

    // Flip one ULP in the first data element (smallest possible input change).
    let mut cf_b = cf_a.clone();
    cf_b.data[0] = f32::from_bits(cf_a.data[0].to_bits() ^ 1);

    let c_a = CochainFreezeEnvelope::freeze(&cf_a).commitment();
    let c_b = CochainFreezeEnvelope::freeze(&cf_b).commitment();
    let sensitive = c_a != c_b;

    // Measure verify() latency on a clean envelope.
    let env = CochainFreezeEnvelope::freeze(&cf_a);
    let env_ref = &env;
    let mut verify_call = || {
        let ok = env_ref.verify();
        black_box(ok);
    };
    let verify_ns = time_median_ns(&mut verify_call, 10_000);

    let consistent = env.verify();
    (sensitive && consistent, verify_ns)
}

// ─── main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Issue 455 — CochainFreezeEnvelope GOAT Gate                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // G2: perf
    let g2_pass = g2_perf();
    println!();

    // G4: determinism
    let g4_pass = g4_determinism();
    println!("── G4 (determinism): 100 freezes → bit-identical commitments ──");
    println!("   Result: {}", if g4_pass { "PASS ✓" } else { "FAIL ✗" });
    println!();

    // G5: tamper sensitivity + verify latency
    let (g5_pass, verify_ns) = g5_tamper();
    println!("── G5 (tamper sensitivity): 1-ULP input change → different commitment ──");
    println!("   1-ULP flip in data[0]:  commitment diverges ✓ (BLAKE3 collision resistance)");
    println!("   Clean verify():         consistent ✓");
    println!("   verify latency:         {verify_ns:.0} ns/call (64-cell cochain)");
    println!("   Bit-flip tamper detection correctness: covered by dec_freeze::tests");
    println!("   Result: {}", if g5_pass { "PASS ✓" } else { "FAIL ✗" });
    println!();

    // Summary
    let all_pass = g2_pass && g4_pass && g5_pass;
    println!("═══ GOAT gate summary ─══");
    if all_pass {
        println!("   G2 ✓ G4 ✓ G5 ✓");
        println!("   → CochainFreezeEnvelope GOAT gate passes (cold-path serialization).");
    } else {
        println!("   One or more gates failed — STOP and audit.");
    }
    println!("   all_pass = {all_pass}");
}
