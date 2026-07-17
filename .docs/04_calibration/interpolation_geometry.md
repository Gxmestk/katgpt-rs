# Interpolation Geometry — iMAUVE + 5-Way Intervention Probe

> **Source:** [Benchmark 456](../../.benchmarks/456_interpolation_geometry_goat.md),
> [Research 445](../../.research/445_Latent_Thought_Flows_Text_Compression.md) —
> Prabhudesai & Geng, *Latent Thought Flows with Text Compression* (Jun 2, 2026).
> **Module:** `katgpt_core::interpolation_geometry` (feature `interpolation_geometry`, opt-in).
> **GOAT bench:** `benches/bench_456_interpolation_geometry_goat.rs`.

## What this is

A generic, modelless evaluation methodology for any committed latent substrate.
It exposes two protocols distilled from the paper:

1. **`imauve_score`** — nearest-neighbor midpoint interpolation quality
   (paper §1.2). The paper's headline methodological contribution: predicts
   downstream generation quality with Pearson r=0.99, while reconstruction
   quality (rMAUVE) saturates near 1.0 and is uninformative. Our analog: a
   substrate can have perfect freeze/thaw bit-identity yet have midpoints
   that decode to incoherent intermediate behavior.

2. **`intervention_battery`** — 5-way causal probe (paper §1.4): matched /
   shuffled / zero / mean / noise. Extends Plan 278's `FaithfulnessProbe`
   (binary, on injected memory) to the per-entity committed-state domain.

## What this is NOT

- **NOT a training primitive.** No encoder, no decoder network, no MeanFlow
  generator. The trait abstracts over substrates that already exist; the
  caller supplies the decode operation.
- **NOT a probability / confidence / predictive interval.** The fields are
  raw geometric / divergence measurements. The "Report the Floor"
  conformal-naive rule (Research 322 / Plan 340) does NOT apply.
- **NOT a router.** The diagnostic produces measurements; the caller decides
  what to do with them.

## The `LatentSpace` trait

```rust
pub trait LatentSpace {
    type Point: Clone;       // e.g. [f32; 8] (HLA), [f32; 64] (style_weights)
    type Behavior;           // e.g. 5-scalar affect set, action distribution

    fn dim(&self) -> usize;
    fn decode(&self, point: &Self::Point) -> Self::Behavior;
    fn midpoint(&self, a: &Self::Point, b: &Self::Point) -> Self::Point;
    fn zero(&self) -> Self::Point;
    fn mean(&self, samples: &[Self::Point]) -> Self::Point;
    fn noise(&self, seed: u64) -> Self::Point;       // deterministic
    fn latent_distance(&self, a: &Self::Point, b: &Self::Point) -> f32;  // L2
    fn behavior_distance(&self, a: &Self::Behavior, b: &Self::Behavior) -> f32;
}
```

**Midpoint contract**: `midpoint(a, b)` MUST be symmetric and idempotent at
the endpoints (`midpoint(a, a) == a`). The shipped reference impls verify
both; custom impls should add their own test.

**Distance contract**: `latent_distance` is the metric used for nearest-
neighbor search (typically L2). `behavior_distance` is the metric used to
score decoded outputs (the paper's cross-entropy analog; caller-supplied —
could be L2 on emotion scalars, KL on action distributions, or Hamming on
tokenized output).

## Reference implementations shipped

| Impl | Point | Behavior | Decode | Used for |
|---|---|---|---|---|
| `GaussianMixtureSpace` | `[f32; 2]` | `[f32; 2]` | identity | Unit tests; demonstrates good vs bad geometry |
| `EuclideanLatentSpace<N>` | `[f32; N]` | `[f32; N]` | identity | Generic shape analog — `N=8` for HLA, `N=64` for `style_weights` |

**Why identity decode:** the synthetic test fixtures prove the *protocol
mechanics* (nearest-neighbor search, midpoint computation, behavior-distance
aggregation) without depending on any real substrate's decode path. Real
substrates plug in their own decode via the trait.

## Usage

```rust
use katgpt_core::interpolation_geometry::{
    EuclideanLatentSpace, LatentSpace, imauve_score, intervention_battery,
};

// 1. Define your substrate (here: a generic 64-dim Euclidean space).
let space = EuclideanLatentSpace::<64>;

// 2. Build an anchor/candidate pool.
let points: Vec<[f32; 64]> = /* your committed latents */;
let mut midpoint_scratch = [0.0f32; 64];

// 3. Score interpolation geometry.
let score = imauve_score(&space, &points, &points, &mut midpoint_scratch, 4.0);
// score.score in [0, 1]; 1.0 = midpoints stay on-manifold.

// 4. Run a 5-way intervention probe on one anchor.
let anchor = points[0];
let donors = &points[1..];
let mut z = [0.0f32; 64];
let mut m = [0.0f32; 64];
let mut n = [0.0f32; 64];
let report = intervention_battery(&space, &anchor, donors, 42, &mut z, &mut m, &mut n);
// report.matched ~ 0 (control); .shuffled/.zero/.mean/.noise diverge if the
// latent matters. report.latent_is_causal(5.0) → bool.
```

## GOAT gate (Phase 1)

All gates measured by `bench_456_interpolation_geometry_goat`:

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| **G1 correctness** | good > bad geometry, good > 0.9, bad < 0.95 | good=0.9646, bad=0.8087 | PASS |
| **G2 perf** (n=256 × d=64) | < 50 ms | 642 µs median (78× headroom) | PASS |
| **G4 zero-alloc** | 0 allocs / 100 calls × 2 primitives | 0 / 0 | PASS |
| **G3 no-regression** | clippy --all-features clean | clean | PASS |

**Why the gates stop here (not at promotion to default-on):** This is an
*evaluation methodology*, not a primitive that improves runtime. The GOAT
promotion rule requires a modelless **gain** — this primitive produces
*measurements*, not gains. It stays opt-in until a real substrate's audit
(riir-engine `NpcEmotionScalars`, riir-neuron-db `NeuronShard`) either
(a) confirms all substrates have good geometry (close issue, methodology
documented, no further work) or (b) surfaces a substrate with bad geometry
(open a fix plan; the fix becomes the GOAT candidate, the metric stays as
the regression test).

## Substrates covered (Research 445 §2.6)

| Substrate | Lives in | Status |
|---|---|---|
| `NpcEmotionScalars` (5 emotion fields) | riir-engine (private) | **AUDITED PASS** (2026-07-17, commit `20f153eb`, iMAUVE=0.9962 via `curiosity_drive` decode) |
| `ArchetypeBlendShard` π | riir-neuron-db (private) | **AUDITED PASS** (2026-07-17, continuation; iMAUVE=0.9917 via `sigmoid(pi_k/tau)` gate decode; Q2 intervention battery all-diverge) |
| `KarcShard` weights | riir-neuron-db (private) | **AUDITED PASS** (2026-07-17, continuation; iMAUVE=0.9773 via linear readout `Wout @ delay_state`; Q2 intervention battery all-diverge) |
| `NeuronShard::style_weights[64]` | riir-neuron-db (private) | **AUDITED PASS** (2026-07-17, commit `746c4a0`, iMAUVE=0.9693 GOOD vs 0.6994 BAD via identity decode) |
| `ZoneGeometryPod` | riir-neuron-db (private) | **N/A** (derived artifact, no learnable latent — Q2/Q3 structural PASS) |
| `MerkleFrozenEnvelope` versions | riir-neuron-db (private) | **N/A** (cryptographic commitment, no decode path — Q3 vacuous PASS) |

The `EuclideanLatentSpace<N>` reference impl proves the protocol works at
the HLA dimension (N=8) and the `style_weights` dimension (N=64)
generically. The two primary substrates have now been audited end-to-end
with their real decode paths — see the cross-references below.

### Cross-repo audit reports (2026-07-17)

- [`riir-ai/.benchmarks/517_emotion_scalars_interpolation_geometry_audit.md`](../../../../riir-ai/.benchmarks/517_emotion_scalars_interpolation_geometry_audit.md) — `NpcEmotionScalars` audit. Decode = `curiosity_drive()` (the existing sigmoid-blended λ-modulation scalar). iMAUVE=0.9962 on 50 anchors (5 archetype clusters × σ=0.1). Intervention battery: `latent_is_causal(10.0)` PASS. Honest caveat: high score depends on tight archetype clustering — a population straddling the sigmoid's steepest region would diverge more.
- [`riir-neuron-db/.benchmarks/459_style_weights_interpolation_geometry_audit.md`](../../../../riir-neuron-db/.benchmarks/459_style_weights_interpolation_geometry_audit.md) — `NeuronShard::style_weights[64]` audit. **v1** identity decode: iMAUVE GOOD=0.9693 vs BAD=0.6994 (margin 0.27 — the metric discriminates). **v2** runtime affect-scalar decode (`sigmoid((1/STYLE_DIM) · dot)`, mirroring `project_compacted_to_scalars`): iMAUVE=0.9984 on high-signal clustered population; intervention battery `latent_is_causal(5.0)` PASS under a high-signal anchor (matched=0, shuffled=0.20, zero=0.56, mean=0.13, noise=0.56). This answers Issue 158 three-pressure audit Q2 in the affirmative for `style_weights`. Low-signal regime documented as expected sigmoid behavior.

## Three-pressure audit (paper §1.3)

For each substrate, the audit checks whether the latent summarizes-vs-routes,
whether the runtime depends on it, and whether context stays local. These
are documented in [Benchmark 456](../../.benchmarks/456_interpolation_geometry_goat.md) (originally filed as Issue 158, removed 2026-07-17 per noise rule; verdict preserved in git history)
§"Three-pressure audit". The audit requires the real decode path.

**Status (2026-07-17):**
- **Q1 (summarize-vs-route)** is **CLOSED** for all six substrates. `NeuronShard::style_weights` via the summarize-vs-route subsampling audit (Benchmark 459 v3 addendum) — the `ConsolidationPipeline::sleep()` average is the canonical summarize operation, divergence scales proportionally with drop fraction, and the routing control exhibits the expected worst-case spike. `NpcEmotionScalars` is current-state (not trajectory-summary), so Q1 is vacuously N/A. `ArchetypeBlendShard.pi`, `ZoneGeometryPod`, and `MerkleFrozenEnvelope` are N/A (no trajectory operation). **`KarcShard.wout` PASS** (Benchmark 459 v5 addendum) — `audit_karc_wout_summarize_vs_route` empirically confirms the ridge-fit Wout is a summarizing latent: at drop=30%, mean_rel=0.047, worst_rel=0.066 (well under the bound), divergence scales proportionally with drop_fraction; the routing control fails at worst_rel=1.33.
- **Q2 (runtime-depends-on-latent)** is **CLOSED** for all six substrates:
  - `NpcEmotionScalars`: `curiosity_drive()` is itself a runtime bridge — Q2 implicit.
  - `NeuronShard::style_weights`: v2 audit (`StyleWeightsScalarSpace`) confirms `latent_is_causal(5.0)` under the runtime affect-scalar decode.
  - `ArchetypeBlendShard.pi`: runtime audit (`ArchetypeBlendPiSpace`) iMAUVE=0.9917; intervention battery all-diverge (0.90/0.47/0.49/0.41).
  - `KarcShard.wout`: runtime audit (`KarcWoutSpace`) iMAUVE=0.9773; intervention battery all-diverge (4.67/4.29/3.46/9.00).
  - `ZoneGeometryPod` + `MerkleFrozenEnvelope`: vacuously N/A (no learnable latent decode path).
- **Q3 (local-context-vs-bypass)** is **CLOSED** for all six substrates via a structural code audit (decode-path purity + consumer-input audit + locality-mechanism inventory). Verdict: **PASS by construction** — the decode paths are pure functions of the latent; the consumers take only the decoded value; cross-NPC influence flows through latent refinement before reaching the decode, not through a raw-state side channel. The SpKv (Plan 070) `window: 128` sliding-window + RTPurbo (Plan 126) sparse-decode infrastructure enforce locality at the transformer-attention layer for runtimes that use transformer attention (NPC dialog WASM etc.), which are out of scope for this substrate-level audit. Per-substrate trace + locality-mechanism inventory preserved in git history (originally Issue 158 §"Q3 structural audit findings" + §"Remaining-substrate Q1/Q2/Q3 verdicts"; issue removed 2026-07-17 per noise rule).
- **All three three-pressure audit questions are now CLOSED** for all six substrates. Issue 158 is fully closed (removed 2026-07-17 per noise rule; verdict preserved in git history + [Benchmark 456](../../.benchmarks/456_interpolation_geometry_goat.md)).

## See also

- [`faithfulness_probe.md`](faithfulness_probe.md) — Plan 278's binary
  FaithfulnessProbe; this module extends it to per-entity committed state.
- [`../03_memory/engram.md`](../03_memory/engram.md) — Engram memory
  primitive; one of the six substrates' decay can be audited via this metric.
- [Research 445](../../.research/445_Latent_Thought_Flows_Text_Compression.md)
  — the parent research note with the full vocabulary translation table.
