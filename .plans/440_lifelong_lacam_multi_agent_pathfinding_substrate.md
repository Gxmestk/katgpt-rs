# Plan 440: Lifelong LaCAM Multi-Agent Pathfinding Substrate

**Date:** 2026-07-15
**Research:** [katgpt-rs/.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md)
**Private guide:** [riir-ai/.research/318_lifelong_lacam_multi_agent_physical_coordination_guide.md](../../riir-ai/.research/318_lifelong_lacam_multi_agent_physical_coordination_guide.md)
**Private runtime plan:** [riir-ai/.plans/489_lifelong_lacam_crowd_coordination_runtime.md](../../riir-ai/.plans/489_lifelong_lacam_crowd_coordination_runtime.md)
**Source paper:** [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, "Lifelong LaCAM with Local Guidance for Lifelong MAPF", AAAI 2026.
**Target:** `katgpt-rs/crates/katgpt-core/src/multi_agent_path/` (new module) + Cargo feature `multi_agent_path`
**Status:** Active — Phase 1 COMPLETE, Phase 2 (GOAT gate) not started

---

## Goal

Ship a generic, modelless, training-free multi-agent pathfinding substrate
distilled from LLLG (arXiv:2605.16855). The substrate is paper-faithful LLLG
with four pluggable seams (cost function, guidance source, warm-start
scheme, hindrance estimator) so private consumers (riir-ai/489) can fuse it
with HLA, Crowd MCGS, and the warm-path stack without forking.

The substrate scales to 10,000 agents at real-time per-step planning
(paper-reported; our G4 gate measures actual Rust impl performance). It is
**entirely heuristic** — no training, no backprop, no gradient descent.
Modelless mandate trivially satisfied.

**GOAT gate:** G1 (paper reproduction within 10%) + G2 (congestion
mitigation heatmap) + G3 (no-regression) + G4 (latency). Promotion to
default-on allowed once G1–G4 pass (the substrate is modelless). The
riir-ai fusion gates G5–G7 (HLA projection, Crowd MCGS physical layer, P350
multi-agent closure) run in riir-ai/489 and are NOT blocking for this plan.

## Non-Goals

- **HLA projection / per-NPC personality modulation.** That's the private
  fusion (riir-ai/489 Extension A). This plan ships paper-faithful uniform-α
  LLLG only.
- **Crowd MCGS / Sheaf / P350 integration.** Private runtime concerns
  (riir-ai/489 Extensions B, C).
- **Learned guidance.** None — modelless mandate. If a future tuning need
  arises, exhaust modelless paths (per-personality `α` table) before
  deferring to riir-train.
- **Bit-identical reproduction of the paper's C++ impl.** We aim for "same
  order of magnitude, same qualitative rankings" (G1 within 10%), not
  tick-by-tick trajectory identity.
- **Anytime refinement (LaCAM*).** The paper §4.C Fig. 9 shows anytime
  refinement *degrades* lifelong throughput. We do NOT add it by default.

## Phase 1 — Substrate Skeleton (paper-faithful LLLG)

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/multi_agent_path/` module.
      Add `mod multi_agent_path;` to `lib.rs`. Add feature
      `multi_agent_path = []` to `Cargo.toml` (opt-in).
- [x] **T1.2** `mod.rs` — `LifelongLaCam<P, C, G>` struct + `tick()`
      orchestrator. Generic over `P: Position` (trait:
      `neighbors() -> &[P]`, `is_passable() -> bool`), `C: CostFn<P>`,
      `G: LocalGuidanceSource<P>`. Fields: `w_pi`, `w_phi`, `m`,
      `guidance_source`, `cost`, `hindrance_enabled`, `prev_plan_suffix`.
- [x] **T1.3** `config.rs` — `Config<P>` (joint configuration = `Vec<P>`),
      `AgentId(u32)` newtype, `JointAction<P>` (the executed first step:
      `Vec<P>` of next positions, one per agent).
- [x] **T1.4** `pibt.rs` — PIBT one-step generator (Okumura et al. 2022).
      Lexicographic cost `⟨Ind[Φ[i][1] ≠ u], dist(u, g_i), hindrance(i→u),
      ε⟩`. Priority inheritance with backtracking. Returns the first
      collision-free joint configuration or `None` (deadlock — caller
      handles via higher-level LaCAM search).
- [x] **T1.5** `local_guidance.rs` — default `LocalGuidanceSource`
      implementation: space-time A* on paper Eq. 1 cost. Sequential
      per-agent refinement, `m` rounds (Algorithm 1). `α` is a config
      scalar (default 1.0). Returns `Φ: Vec<Vec<P>>` (per-agent `w_Φ`-step
      path).
- [x] **T1.6** `warm_start.rs` — `WarmStartScheme` enum:
      `LllgPi` (suffix of prev solution, paper default), `LllgPhi` (inherit
      prev guidance), `LllgEmpty` (recompute from scratch). `Π_{t-1}[2:w_Φ]`
      suffix cache with padding for `w_Φ > w_Π` case (paper §3).
- [x] **T1.7** `hindrance.rs` — one-step blocking-count estimator (Okumura
      & Nagai 2025). For agent `i` considering move to `u`, count siblings
      `j ≠ i` whose neighborhood at `t+1` would include `u`. O(neighbors²)
      per agent, near-zero cost in practice.
- [x] **T1.8** `position.rs` — `Position` trait. Default impl for `(usize,
      usize)` grid cells. Document the extension point for 3D / NavMesh
      positions (seal-core `NavMesh` integration is a consumer concern, not
      substrate).
- [x] **T1.9** Feature flag wiring. `Cargo.toml`:
      ```toml
      [features]
      multi_agent_path = []
      ```
      No heavy deps. Pure Rust. (Rayon parallelism is a consumer concern —
      the substrate is single-threaded per LLLG instance, matching the
      paper.)
- [x] **T1.10** Unit tests in `tests.rs`:
      - 2 agents, 3×3 map, vertex collision (agents swap → edge collision).
      - Deadlock (two agents in a 1-wide corridor).
      - Throughput sanity (10 agents, 10×10 map, 100 ticks, all reach goals).
      - Warm-start cache correctness (`Π_{t-1}` suffix → `Φ` init).
      - Hindrance estimator correctness (known blocking count).

### Acceptance

`cargo test -p katgpt-core --features multi_agent_path --lib` passes.
`cargo clippy -p katgpt-core --features multi_agent_path --lib` clean.

## Phase 2 — GOAT Gate (paper reproduction, G1–G4)

### Tasks

- [ ] **T2.1** Benchmark harness: `benches/bench_440_lllg_paper_repro.rs`.
      4 paper maps (empty-48-48, random-64-64-10, warehouse-10-20-10-2-2,
      ht_chantry), 800 agents, 500 steps. Reproduce paper Fig. 4 throughput
      numbers.
- [ ] **T2.2** **G1 — correctness gate.** Our LLLG throughput within 10%
      of paper-reported numbers on all 4 maps at 800 agents. Qualitative
      ranking preserved (LLLG > RHCR > PIBT in dense settings).
- [ ] **T2.3** **G2 — congestion mitigation gate.** Heatmap of stop-counts
      (paper Fig. 3 visual comparator). LLLG max-stops-per-cell < 0.5 ×
      PIBT max-stops-per-cell on empty-48-48 at 1000 agents.
- [ ] **T2.4** **G3 — no-regression gate.** `cargo check --all-features`
      clean. `cargo test -p katgpt-core --lib` passes. Existing single-agent
      `find_path` tests unaffected (the new module is feature-gated, off by
      default).
- [ ] **T2.5** **G4 — latency gate.** Per-tick planning time at 1000 agents
      on commodity hardware. Target: < 500 ms (generous; paper reports
      210–260 ms on M1 Ultra at 1000 agents). Stretch: < 100 ms.
- [ ] **T2.6** File GOAT results in
      `katgpt-rs/.benchmarks/440_lllg_paper_repro_goat.md`. Promote
      `multi_agent_path` from opt-in to default-on if G1–G4 pass AND the
      substrate is modelless (it is). **OR** keep opt-in if there are
      perf concerns — the riir-ai fusion (riir-ai/489) can consume it
      either way.

### Acceptance

G1–G4 results documented in `.benchmarks/440_*.md`. Honest about any gate
that fails (e.g., if G4 latency is 2× over target, document it and propose
zone-sharding as the mitigation — don't silently ship a slow impl).

## Phase 3 — Fusion Hooks (pluggable seams, documented extension points)

### Tasks

- [ ] **T3.1** Document `LocalGuidanceSource<P>` trait as the primary
      extension point. Include a stub `HlaProjectedGuidance` example in
      doc comments (the actual impl lives in riir-ai/489).
- [ ] **T3.2** Document `CostFn<P>` trait. Include stub examples for
      heightfield slope, threat cochain, faction zone penalty.
- [ ] **T3.3** Document `WarmStartScheme` enum and how a consumer can add
      a custom scheme (e.g., personality-weighted blend).
- [ ] **T3.4** Document `HindranceEstimator` trait (extract from the
      default impl). Include stub for affect-aware hindrance.

### Acceptance

The four traits are documented with examples. A consumer reading the
`multi_agent_path::mod` doc comment understands how to plug in a custom
guidance source without reading the paper.

## Phase 4 — Documentation

### Tasks

- [ ] **T4.1** Add LLLG entry to `katgpt-rs/README.md` feature showcase
      (following the existing format — see e.g. the Set Attention entry).
- [ ] **T4.2** Cross-ref from `katgpt-rs/.research/219` (DEC substrate —
      note the `χ`-as-codifferential reframing) and `katgpt-rs/.research/354`
      (Set Attention — note the latent-domain analog).
- [ ] **T4.3** Update `katgpt-rs/.docs/01_orientation/overview.md` feature
      flag table.

### Acceptance

README + docs reflect the new feature. Cross-refs are bidirectional.

## Phase 5 — Promotion Decision (after Phase 2)

### Tasks

- [ ] **T5.1** If G1–G4 pass: decide promote-to-default vs keep-opt-in.
      Considerations:
      - The substrate is modelless → promotion is allowed by AGENTS.md.
      - The substrate is heavy (multi-agent pathfinding is not a leaf-clean
        primitive) → keeping it opt-in may be preferable to avoid bloating
        the default build.
      - The riir-ai fusion (G5–G7) hasn't run yet → the Super-GOAT claim is
        unconfirmed. Promoting the substrate to default before the fusion
        is validated is premature.
      - **Recommendation:** keep opt-in until riir-ai/489 G5–G7 pass. The
        substrate ships opt-in; consumers opt in via feature flag.
- [ ] **T5.2** Record the decision in `.benchmarks/440_*.md` with
      reasoning.

## References

- [Research 424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md) — the open primitive research note.
- [riir-ai Research 318](../../riir-ai/.research/318_lifelong_lacam_multi_agent_physical_coordination_guide.md) — the private selling-point guide.
- [riir-ai Plan 489](../../riir-ai/.plans/489_lifelong_lacam_crowd_coordination_runtime.md) — the private runtime fusion plan.
- [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — the source paper.
- <https://github.com/allegorywrite/lllg> — paper reference implementation (C++).

## TL;DR

Ship paper-faithful LLLG as a generic, opt-in, modelless multi-agent
pathfinding substrate in `katgpt-core/src/multi_agent_path/`. Four pluggable
seams (cost, guidance, warm-start, hindrance). GOAT gate G1–G4 (paper
reproduction within 10%, congestion mitigation, no-regression, latency).
Stays opt-in until the riir-ai fusion (G5–G7) validates the Super-GOAT
claim.
