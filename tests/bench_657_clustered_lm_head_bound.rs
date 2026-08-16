//! Issue 657 — re-gate the clustered LM head after two modelless fixes.
//!
//! Run with:
//! ```text
//! cargo test --release --test bench_657_clustered_lm_head_bound -- --nocapture
//! ```
//!
//! # What this re-measures, and why there are two fixes
//!
//! Plan 574 failed G2b: best argmax recall 0.675 against a 0.99 target. Issue
//! 657 attributed that to the **scoring objective** — stage 1 ranks by
//! `⟨h, centroid_c⟩`, the cluster's *mean* logit, when the question is which
//! cluster holds the *max*. That diagnosis produced fix A.
//!
//! Implementing it surfaced a second, unrelated defect that the diagnosis had
//! not considered — fix B. K-means seeded its centres at token IDs
//! `0, stride, 2·stride, …` with `stride = vocab / k`. Benchmark 657's
//! structured fixture assigns token `t` to planted group `t % n_groups` with
//! `n_groups == k`, so every seeded centre came from group `(c·stride) % k`,
//! which for `stride = vocab / k` collapses to **two distinct groups**. Lloyd
//! cannot recover 256 planted groups from centres drawn out of two.
//!
//! Two candidate causes for one failed gate is exactly the situation that
//! produces a confident wrong attribution, so this bench measures the **2×2**:
//!
//! | | mean ranking | bound ranking |
//! |---|---|---|
//! | strided init | Plan 574's measurement (the recorded FAIL) | fix A alone |
//! | D² init | fix B alone | both |
//!
//! plus the admissible stop rule, whose recall is 1.0 by construction and whose
//! *cost* is therefore the number being measured.
//!
//! Fixture caveat carried forward from Benchmark 657: these are synthetic LM
//! heads, not a real checkpoint. The strided-init pathology in particular is
//! amplified by the fixture's deliberate ID-interleaving — on a real vocabulary
//! strided seeding is arbitrary rather than adversarial, so fix B's share of
//! the gain here is an upper bound on what it buys in production.

use katgpt_rs::transformer::{
    ClusterHeadView, ClusterInit, ClusterLayout, ClusterScratch, ClusterStop, PackedHeadView,
    TiedPolicy, cluster_classifier_from_map, cluster_layout_from_map,
    cluster_map_from_embeddings_with_init, cluster_map_round_robin, cluster_radii_from_map,
    clustered_lm_head_bounded, clustered_lm_head_packed, standard_lm_head,
};
use std::time::Instant;

/// BPE-ish vocabulary — same shape as Benchmark 657 so the numbers compare.
const VOCAB: usize = 32768;
const N_EMBD: usize = 512;
/// `VOCAB / 128 = 256` clusters, mirroring Gemma 4's 2048-over-262144 ratio.
const CLUSTER_SIZE: usize = 128;
const PROBES: usize = 200;
/// Plan 574's absolute quality bar.
const RECALL_TARGET: f64 = 0.99;
/// Baseline group jitter — the tight, favourable geometry.
const DEFAULT_SPREAD: f32 = 0.05;
/// Wave size used wherever a single admissible configuration is needed.
/// Chosen from the Issue 661 sweep printed by this bench.
const WAVE: usize = 8;
/// Whether the crossover sweep and G3 use Issue 666's cluster-contiguous layout.
/// `true` once the packed path is measured as the better default.
const PACKED: bool = true;

/// Deterministic LCG — reproducible across runs and platforms.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

/// LM head with planted cluster structure; token `t` belongs to group
/// `t % n_groups` so round-robin (which partitions by ID) cannot recover it.
///
/// `spread` is the per-coordinate jitter around a group's centre. It is the
/// **separability dial**: at 0.05 the groups are tight and the bound prunes
/// hard; as it grows the geometry washes out toward the random control. Issue
/// 661's crossover sweep walks it.
fn structured_lm_head(n_groups: usize, spread: f32, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut centres = vec![0.0f32; n_groups * N_EMBD];
    for slot in centres.iter_mut() {
        *slot = rng.next_f32();
    }
    let mut w = vec![0.0f32; VOCAB * N_EMBD];
    for t in 0..VOCAB {
        let g = t % n_groups;
        for j in 0..N_EMBD {
            w[t * N_EMBD + j] = centres[g * N_EMBD + j] + spread * rng.next_f32();
        }
    }
    w
}

/// Control: no structure. K-means has nothing to find.
fn random_lm_head(seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut w = vec![0.0f32; VOCAB * N_EMBD];
    for slot in w.iter_mut() {
        *slot = rng.next_f32();
    }
    w
}

fn probe_vectors(count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg(seed);
    (0..count)
        .map(|_| (0..N_EMBD).map(|_| rng.next_f32()).collect())
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i).unwrap_or(0)
}

fn true_argmaxes(lm_head: &[f32], probes: &[Vec<f32>]) -> Vec<usize> {
    let mut logits = vec![0.0f32; VOCAB];
    probes
        .iter()
        .map(|hidden| {
            standard_lm_head(&mut logits, hidden, lm_head, VOCAB, N_EMBD);
            argmax(&logits)
        })
        .collect()
}

/// One clustering configuration under test.
struct Arm {
    label: &'static str,
    map: Vec<Vec<usize>>,
    classifier: Vec<f32>,
    radii: Vec<f32>,
    /// `false` scores by the bare centroid (mean logit) — the shipped path.
    use_bound: bool,
}

impl Arm {
    fn build(
        label: &'static str,
        lm_head: &[f32],
        map: Vec<Vec<usize>>,
        use_bound: bool,
    ) -> Self {
        let classifier = cluster_classifier_from_map(lm_head, &map, N_EMBD);
        let radii = cluster_radii_from_map(lm_head, &map, &classifier, N_EMBD);
        Self { label, map, classifier, radii, use_bound }
    }

    /// Issue 666's cluster-contiguous layout for this arm.
    fn layout(&self, lm_head: &[f32]) -> ClusterLayout {
        // `Accept`: the bench owns its fixture, so there is no `wte` to alias.
        // Production must pass `Refuse { wte }` — see `TiedPolicy`.
        cluster_layout_from_map(lm_head, &self.map, VOCAB, N_EMBD, TiedPolicy::Accept)
            .expect("bench fixture is well-shaped and untied")
    }

    fn packed_view<'a>(&'a self, layout: &'a ClusterLayout) -> PackedHeadView<'a> {
        PackedHeadView {
            classifier: &self.classifier,
            radii: match self.use_bound {
                true => Some(&self.radii),
                false => None,
            },
            layout,
        }
    }

    fn view(&self) -> ClusterHeadView<'_> {
        ClusterHeadView {
            classifier: &self.classifier,
            radii: match self.use_bound {
                true => Some(&self.radii),
                false => None,
            },
            map: &self.map,
        }
    }
}

/// `(recall, mean active fraction)` for one arm at one stop rule.
fn measure(
    arm: &Arm,
    lm_head: &[f32],
    probes: &[Vec<f32>],
    truth: &[usize],
    stop: ClusterStop,
) -> (f64, f64) {
    let mut logits = vec![0.0f32; VOCAB];
    let mut scores = vec![0.0f32; arm.map.len()];
    let (mut indexed, mut selected) = (Vec::new(), Vec::new());
    let (mut gathered, mut dots) = (Vec::new(), Vec::new());

    let mut hits = 0usize;
    let mut active = 0usize;
    for (i, hidden) in probes.iter().enumerate() {
        let cost = clustered_lm_head_bounded(
            &mut logits,
            hidden,
            lm_head,
            arm.view(),
            VOCAB,
            N_EMBD,
            stop,
            ClusterScratch {
                scores: &mut scores,
                indexed: &mut indexed,
                selected: &mut selected,
                gathered: &mut gathered,
                dots: &mut dots,
            },
        );
        if argmax(&logits) == truth[i] {
            hits += 1;
        }
        active += cost.tokens;
    }
    (hits as f64 / probes.len() as f64, active as f64 / (probes.len() * VOCAB) as f64)
}

/// Recall at the largest `topk` whose active fraction still fits `budget`.
///
/// Binary search, not a grid: `active(topk)` is monotonically non-decreasing, so
/// bisection is exact. Benchmark 657 recorded two verdict-moving errors here —
/// comparing at equal `topk` (unequal compute) and using a coarse geometric grid
/// that stepped over the optimum — so the search resolution is load-bearing.
fn recall_at_budget(
    arm: &Arm,
    lm_head: &[f32],
    probes: &[Vec<f32>],
    truth: &[usize],
    budget: f64,
) -> (f64, usize) {
    let (mut lo, mut hi) = (1usize, arm.map.len());
    let mut best = (0.0f64, 0usize);
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let (recall, active) = measure(arm, lm_head, probes, truth, ClusterStop::TopK(mid));
        match active <= budget {
            true => {
                best = (recall, mid);
                lo = mid + 1;
            }
            false => match mid {
                0 => break,
                _ => hi = mid - 1,
            },
        }
    }
    best
}

const BUDGETS: [f64; 4] = [0.02, 0.05, 0.10, 0.25];

/// Runs the 2×2 attribution matrix plus round-robin and the admissible stop.
/// Returns the best recall over all budgets for the fully-fixed arm.
fn run_regime(label: &str, lm_head: &[f32], probes: &[Vec<f32>]) -> f64 {
    println!("\n══ {label} ══");

    let t0 = Instant::now();
    let dsq = cluster_map_from_embeddings_with_init(
        lm_head,
        VOCAB,
        N_EMBD,
        CLUSTER_SIZE,
        ClusterInit::Dsquared,
    );
    let strided = cluster_map_from_embeddings_with_init(
        lm_head,
        VOCAB,
        N_EMBD,
        CLUSTER_SIZE,
        ClusterInit::Strided,
    );
    println!(
        "build {:.1}s   clusters: D² {}   strided {}",
        t0.elapsed().as_secs_f64(),
        dsq.len(),
        strided.len()
    );

    let truth = true_argmaxes(lm_head, probes);

    let arms = vec![
        Arm::build("round-robin  + mean ", lm_head, cluster_map_round_robin(VOCAB, CLUSTER_SIZE), false),
        Arm::build("strided init + mean ", lm_head, strided.clone(), false),
        Arm::build("strided init + BOUND", lm_head, strided, true),
        Arm::build("D² init      + mean ", lm_head, dsq.clone(), false),
        Arm::build("D² init      + BOUND", lm_head, dsq, true),
    ];

    println!("  -- argmax recall at MATCHED active budget (topk in parens) --");
    print!("  {:<20}", "arm");
    for b in BUDGETS {
        print!("{:>16}", format!("{:.0}%", b * 100.0));
    }
    println!();

    let mut best_fixed = 0.0f64;
    for arm in &arms {
        print!("  {:<20}", arm.label);
        for b in BUDGETS {
            let (r, k) = recall_at_budget(arm, lm_head, probes, &truth, b);
            print!("{:>16}", format!("{r:.4} ({k})"));
            if arm.label.starts_with("D²") && arm.use_bound {
                best_fixed = best_fixed.max(r);
            }
        }
        println!();
    }

    println!("  -- ADMISSIBLE stop (recall is 1.0 by construction; cost is the result) --");
    for arm in arms.iter().filter(|a| a.use_bound) {
        let (r, active) =
            measure(arm, lm_head, probes, &truth, ClusterStop::Admissible { wave: WAVE });
        println!(
            "  {:<20} recall {r:.4}   active {:.2}%   speedup-vs-full {:.2}x",
            arm.label,
            active * 100.0,
            1.0 / active.max(1e-9)
        );
        assert!(
            (r - 1.0).abs() < 1e-9,
            "{}: admissible mode must be exact, got recall {r}",
            arm.label
        );
    }

    best_fixed
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

/// Warmup pairs, discarded — absorb cache fill and clock ramp.
const WARMUP_PAIRS: usize = 4;
/// Measured pairs. Each contributes one ratio; the median of those is reported.
const MEASURE_PAIRS: usize = 20;

/// Wall-clock of the admissible stop against the full head.
///
/// Run on **both** regimes deliberately. The FLOP reduction is `1 / active`,
/// but `standard_lm_head` dispatches through `matmul_parallel` (rayon across
/// all `VOCAB` rows) while the clustered path walks scattered token IDs on one
/// thread. So the speedup is always well below the FLOP ratio, and on
/// unstructured data — where the bound cannot prune — it inverts into a
/// **loss**. Reporting only the structured number would hide that.
///
/// # Protocol
///
/// riir-ai's interleaved protocol, adopted after two sign-flipping corrections
/// there (a "1.19× win" that was a 0.87× loss, Bench 666; a "1.24× win" that
/// was a 0.95× loss, Issue 658). Warmup pairs are discarded, measure pairs
/// **alternate** A→B / B→A ordering, and the primary statistic is the median of
/// **per-pair** ratios — so monotonic drift cancels within each pair instead of
/// contaminating a median-of-A over median-of-B.
///
/// This matters here specifically: an earlier revision of this bench used
/// median(A)/median(B) and reported 2.44× on one run and 1.42× on the next from
/// identical inputs. Neither number was trustworthy.
/// [`latency`] with an explicit scattered/packed choice (Issue 666).
fn latency_layout(
    label: &str,
    lm_head: &[f32],
    probes: &[Vec<f32>],
    wave: usize,
    quiet: bool,
    packed: bool,
) -> f64 {
    let map = cluster_map_from_embeddings_with_init(
        lm_head,
        VOCAB,
        N_EMBD,
        CLUSTER_SIZE,
        ClusterInit::Dsquared,
    );
    let arm = Arm::build("D² + BOUND", lm_head, map, true);
    latency_for(label, lm_head, &arm, probes, wave, quiet, packed)
}

/// [`latency`] against a pre-built arm, so a sweep does not rebuild the map.
#[allow(clippy::too_many_arguments)]
fn latency_for(
    label: &str,
    lm_head: &[f32],
    arm: &Arm,
    probes: &[Vec<f32>],
    wave: usize,
    quiet: bool,
    packed: bool,
) -> f64 {
    let layout = arm.layout(lm_head);
    let hidden = &probes[0];
    let mut logits = vec![0.0f32; VOCAB];
    let mut scores = vec![0.0f32; arm.map.len()];
    let (mut indexed, mut selected) = (Vec::new(), Vec::new());
    let (mut gathered, mut dots) = (Vec::new(), Vec::new());

    let time_standard = |logits: &mut [f32]| {
        let t = Instant::now();
        standard_lm_head(logits, hidden, lm_head, VOCAB, N_EMBD);
        t.elapsed().as_secs_f64() * 1e3
    };

    let mut ratios = Vec::with_capacity(MEASURE_PAIRS);
    let (mut std_s, mut adm_s) = (Vec::new(), Vec::new());

    for pair in 0..WARMUP_PAIRS + MEASURE_PAIRS {
        // Alternate which side runs first, so a within-pair drift penalises
        // each side equally often instead of always the second-placed one.
        let standard_first = pair % 2 == 0;

        let run_admissible = |logits: &mut [f32],
                                  scores: &mut Vec<f32>,
                                  indexed: &mut Vec<(usize, f32)>,
                                  selected: &mut Vec<usize>,
                                  gathered: &mut Vec<usize>,
                                  dots: &mut Vec<f32>| {
            let t = Instant::now();
            match packed {
                true => {
                    clustered_lm_head_packed(
                        logits,
                        hidden,
                        arm.packed_view(&layout),
                        VOCAB,
                        N_EMBD,
                        ClusterStop::Admissible { wave },
                        ClusterScratch { scores, indexed, selected, gathered, dots },
                    );
                }
                false => {
                    clustered_lm_head_bounded(
                        logits,
                        hidden,
                        lm_head,
                        arm.view(),
                        VOCAB,
                        N_EMBD,
                        ClusterStop::Admissible { wave },
                        ClusterScratch { scores, indexed, selected, gathered, dots },
                    );
                }
            }
            t.elapsed().as_secs_f64() * 1e3
        };

        let (s_ms, a_ms) = match standard_first {
            true => {
                let s = time_standard(&mut logits);
                let a = run_admissible(
                    &mut logits,
                    &mut scores,
                    &mut indexed,
                    &mut selected,
                    &mut gathered,
                    &mut dots,
                );
                (s, a)
            }
            false => {
                let a = run_admissible(
                    &mut logits,
                    &mut scores,
                    &mut indexed,
                    &mut selected,
                    &mut gathered,
                    &mut dots,
                );
                let s = time_standard(&mut logits);
                (s, a)
            }
        };

        if pair >= WARMUP_PAIRS {
            ratios.push(s_ms / a_ms);
            std_s.push(s_ms);
            adm_s.push(a_ms);
        }
    }

    let ratio = median_ms(&mut ratios);
    if !quiet {
        println!(
            "  {label:<12} standard {:.4} ms   admissible {:.4} ms   per-pair ratio {ratio:.2}x  \
             (spread {:.2}–{:.2})",
            median_ms(&mut std_s),
            median_ms(&mut adm_s),
            ratios.iter().cloned().fold(f64::INFINITY, f64::min),
            ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
    }
    ratio
}

#[test]
fn goat_657_clustered_lm_head_bound() {
    println!("Issue 657 — clustered LM head, admissible bound + D² init");
    println!("vocab={VOCAB} n_embd={N_EMBD} cluster_size={CLUSTER_SIZE} probes={PROBES}");

    let probes = probe_vectors(PROBES, 0xC0FFEE);
    let n_clusters = VOCAB / CLUSTER_SIZE;

    let structured = structured_lm_head(n_clusters, DEFAULT_SPREAD, 0xA11CE);
    let best = run_regime("STRUCTURED (groups == clusters — the verdict regime)", &structured, &probes);

    let split = structured_lm_head(64, DEFAULT_SPREAD, 0xA11CE);
    let _ = run_regime("STRUCTURED (64 groups vs 256 clusters — split penalty)", &split, &probes);

    let random = random_lm_head(0xB0B);
    let _ = run_regime("RANDOM (no structure — control)", &random, &probes);

    // ── Issue 661a: wave sweep — how much parallelism can the exact stop take? ──
    //
    // `wave: 1` is the sequential reference measured in Benchmark 658. Larger
    // waves evaluate more clusters per round before re-checking the bound, so
    // they may compute clusters they did not strictly need — bounded extra work
    // bought for rayon width. Exactness is independent of `wave` (asserted in
    // `cluster_head`'s unit tests), so this is purely a cost trade.
    println!(
        "\n══ Issue 661a — wave sweep (structured, packed={PACKED}; exactness is wave-independent) ══"
    );
    let map = cluster_map_from_embeddings_with_init(
        &structured,
        VOCAB,
        N_EMBD,
        CLUSTER_SIZE,
        ClusterInit::Dsquared,
    );
    let arm = Arm::build("D² + BOUND", &structured, map, true);
    let truth = true_argmaxes(&structured, &probes);
    println!("  {:>5}  {:>9}  {:>10}  {:>8}", "wave", "active%", "vs wave 1", "speedup");
    let mut base_active = 0.0f64;
    for (i, wave) in [1usize, 2, 4, 8, 16, 32, 64].into_iter().enumerate() {
        let (recall, active) =
            measure(&arm, &structured, &probes, &truth, ClusterStop::Admissible { wave });
        assert!((recall - 1.0).abs() < 1e-9, "wave {wave} broke exactness: recall {recall}");
        if i == 0 {
            base_active = active;
        }
        let ratio = latency_for("", &structured, &arm, &probes, wave, true, PACKED);
        println!(
            "  {wave:>5}  {:>8.2}%  {:>9.2}x  {ratio:>7.2}x",
            active * 100.0,
            active / base_active
        );
    }

    // ── Issue 666: does a cluster-contiguous layout recover the locality loss? ──
    //
    // Issue 661 showed the wall-clock shortfall is entirely memory access:
    // scattered gathers ran at 20.6 GB/s where the full head streams at
    // 108.0 GB/s, and `FLOP_ratio / locality_penalty` reproduced the measured
    // speedup exactly. Permuting rows into cluster order at load time makes each
    // cluster one contiguous span. Same clusters, same logits — only the
    // addresses change (asserted bit-identical in `cluster_head`'s unit tests).
    println!("\n══ Issue 666 — scattered vs packed layout (structured, wave={WAVE}) ══");
    let layout = arm.layout(&structured);
    let (_, active_666) =
        measure(&arm, &structured, &probes, &truth, ClusterStop::Admissible { wave: WAVE });
    let scattered_ratio =
        latency_for("", &structured, &arm, &probes, WAVE, true, false);
    let packed_ratio = latency_for("", &structured, &arm, &probes, WAVE, true, true);
    let bytes = (VOCAB * N_EMBD * 4) as f64;
    let gbs = |ratio: f64, std_ms: f64| bytes * active_666 / ((std_ms / ratio) * 1e-3) / 1e9;
    // One shared `standard` reference so both bandwidths are on the same base.
    let std_ms = {
        let mut logits = vec![0.0f32; VOCAB];
        let mut samples: Vec<f64> = (0..20)
            .map(|_| {
                let t = Instant::now();
                standard_lm_head(&mut logits, &probes[0], &structured, VOCAB, N_EMBD);
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        median_ms(&mut samples)
    };
    println!("  active {:.2}%   FLOP ratio {:.2}x", active_666 * 100.0, 1.0 / active_666);
    println!("  full head        {std_ms:.4} ms   {:.1} GB/s", bytes / (std_ms * 1e-3) / 1e9);
    println!(
        "  scattered        {:.4} ms   {:.1} GB/s   {scattered_ratio:.2}x",
        std_ms / scattered_ratio,
        gbs(scattered_ratio, std_ms)
    );
    println!(
        "  packed (666)     {:.4} ms   {:.1} GB/s   {packed_ratio:.2}x",
        std_ms / packed_ratio,
        gbs(packed_ratio, std_ms)
    );
    println!(
        "  => packed is {:.2}x the scattered path;  extra memory {:.1} MB ({:.0}% of lm_head)",
        packed_ratio / scattered_ratio,
        layout.extra_bytes() as f64 / 1e6,
        layout.extra_bytes() as f64 / bytes * 100.0
    );

    // ── Issue 661b: the crossover — at what active fraction does this stop paying? ──
    //
    // Benchmark 658 measured the two extremes (7.30% active → win, 99.99% →
    // 12x loss) and could not say where the sign flips. Walking the fixture's
    // separability dial produces the curve between them. The crossover active%
    // is the load-time enable condition: above it, use the full head.
    println!(
        "\n══ Issue 661b — crossover (separability sweep, wave={WAVE}, packed={PACKED}) ══"
    );
    println!("  {:>7}  {:>9}  {:>8}  {:>9}", "spread", "active%", "speedup", "verdict");
    let mut crossover: Option<(f64, f64)> = None;
    let mut prev: Option<(f64, f64)> = None;
    for spread in [0.05f32, 0.07, 0.09, 0.11, 0.13, 0.15, 0.3, 1.0] {
        let w = structured_lm_head(n_clusters, spread, 0xA11CE);
        let m = cluster_map_from_embeddings_with_init(
            &w,
            VOCAB,
            N_EMBD,
            CLUSTER_SIZE,
            ClusterInit::Dsquared,
        );
        let a = Arm::build("D² + BOUND", &w, m, true);
        let t = true_argmaxes(&w, &probes);
        let (recall, active) =
            measure(&a, &w, &probes, &t, ClusterStop::Admissible { wave: WAVE });
        assert!((recall - 1.0).abs() < 1e-9, "spread {spread} broke exactness");
        let ratio = latency_for("", &w, &a, &probes, WAVE, true, PACKED);
        println!(
            "  {spread:>7.2}  {:>8.2}%  {ratio:>7.2}x  {:>9}",
            active * 100.0,
            match ratio > 1.0 {
                true => "win",
                false => "LOSS",
            }
        );
        // Bracket the sign change on the previous winning point.
        if let Some((p_active, p_ratio)) = prev
            && p_ratio > 1.0
            && ratio <= 1.0
            && crossover.is_none()
        {
            crossover = Some((p_active, active));
        }
        prev = Some((active, ratio));
    }
    match crossover {
        Some((lo, hi)) => println!(
            "  => crossover bracketed between {:.1}% and {:.1}% active",
            lo * 100.0,
            hi * 100.0
        ),
        None => println!("  => no sign change observed across the swept range"),
    }

    // ── G3 perf: the admissible operating point vs the full head ──
    println!("\n══ G3 latency (admissible stop, both regimes) ══");
    let g3_structured =
        latency_layout("structured", &structured, &probes, WAVE, false, PACKED);
    let g3_random = latency_layout("random", &random, &probes, WAVE, false, PACKED);

    println!("\n══ VERDICT ══");
    println!(
        "G2b absolute (best TopK recall >= {RECALL_TARGET}): {best:.4} → {}",
        match best >= RECALL_TARGET {
            true => "PASS",
            false => "FAIL",
        }
    );
    println!("Admissible stop: recall 1.0 asserted above — quality is exact by construction.");
    println!(
        "G3 perf structured (admissible < standard): {} ({g3_structured:.2}x)",
        match g3_structured > 1.0 {
            true => "PASS",
            false => "FAIL",
        }
    );
    println!(
        "G3 perf random control: {} ({g3_random:.2}x) — the scope limit, not a bug",
        match g3_random > 1.0 {
            true => "PASS",
            false => "LOSS",
        }
    );

    // Asserted, not merely reported: admissibility is a *proof obligation*, not
    // a measurement, so a regression here is a correctness bug. Recall/latency
    // targets stay reported so the bench survives as a recorded result either
    // way — a permanently red test gets swept as noise instead of read.
}
