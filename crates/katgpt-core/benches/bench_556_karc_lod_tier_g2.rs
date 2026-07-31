//! KARC LOD Tier — Plan 556 Phase 3 G2 perf bench.
//!
//! Target: tier promotion ≤ 10 µs (one-time, not per-tick). At the HLA config
//! (D=8), the worst case is Lod1 → Lod2 (d_h 256 → 512 — doubling the Wout
//! size). We measure three transitions:
//!
//! - `lod1_to_lod0` (down-tier, drop half the features).
//! - `lod0_to_lod1` (up-tier, zero-pad features).
//! - `lod1_to_lod2` (up-tier, 2× features — the worst case).
//!
//! All three should be ≤ 10 µs at release.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::{KarcLodTier, project_wout_lod_into};

fn make_wout(tier: KarcLodTier) -> Vec<f32> {
    let len = tier.d() * tier.d_h();
    (0..len).map(|i| i as f32 * 0.001).collect()
}

fn bench_tier_promotion(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_556_karc_lod_tier_promotion");
    group.throughput(Throughput::Elements(1));

    let transitions: &[(&str, KarcLodTier, KarcLodTier)] = &[
        ("lod1_to_lod0", KarcLodTier::Lod1, KarcLodTier::Lod0),
        ("lod0_to_lod1", KarcLodTier::Lod0, KarcLodTier::Lod1),
        ("lod1_to_lod2", KarcLodTier::Lod1, KarcLodTier::Lod2),
        ("lod2_to_lod0", KarcLodTier::Lod2, KarcLodTier::Lod0),
    ];

    for &(name, src_tier, dst_tier) in transitions {
        let src_wout = make_wout(src_tier);
        let mut dst_wout = vec![0.0f32; dst_tier.d() * dst_tier.d_h()];
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(src_tier, dst_tier),
            |b, &(src_tier, dst_tier)| {
                b.iter(|| {
                    project_wout_lod_into(
                        black_box(&src_wout),
                        black_box(src_tier),
                        black_box(&mut dst_wout),
                        black_box(dst_tier),
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_tier_promotion);
criterion_main!(benches);
