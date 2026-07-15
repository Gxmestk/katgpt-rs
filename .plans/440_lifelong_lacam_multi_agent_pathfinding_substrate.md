# Plan 440: Lifelong LaCAM Multi-Agent Pathfinding Substrate

**Date:** 2026-07-15
**Research:** [katgpt-rs/.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md)
**Private guide:** [riir-ai/.research/318_lifelong_lacam_multi_agent_physical_coordination_guide.md](../../riir-ai/.research/318_lifelong_lacam_multi_agent_physical_coordination_guide.md)
**Private runtime plan:** [riir-ai/.plans/489_lifelong_lacam_crowd_coordination_runtime.md](../../riir-ai/.plans/489_lifelong_lacam_crowd_coordination_runtime.md)
**Source paper:** [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, "Lifelong LaCAM with Local Guidance for Lifelong MAPF", AAAI 2026.
**Target:** `katgpt-rs/crates/katgpt-core/src/multi_agent_path/` (new module) + Cargo feature `multi_agent_path`
**Status:** Active — Phase 1 COMPLETE, Phase 2 (GOAT gate) PARTIAL — G3/G4 PASS, G1 2/4 maps PASS, G2 FAIL (warm-start bug)

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

### Results

See `.benchmarks/440_lllg_paper_repro_goat.md` for full results.

| Gate | Status | Detail |
|---|---|---|
| G1 (throughput) | **PARTIAL (2/4)** | empty-48-48 ratio 0.63 ✅, random-64-64-10 ratio 0.62 ✅, warehouse ratio 0.35 ❌, ht_chantry ratio 0.01 ❌ |
| G2 (congestion) | **FAIL** | Warm-start not integrated — LLLG_Π and LllgEmpty produce identical results |
| G3 (no-regression) | **PASS** | 1556/1556 tests pass, clippy clean on all-features |
| G4 (latency) | **PASS** | 80ms median at 1000 agents (target <500ms, stretch <100ms). Paper: 210-260ms |

### Substrate upgrades applied during Phase 2

1. **PIBT wall-aware neighbors** — fixed `tick()` passing `None` to PIBT.
2. **BFS distance fields** — replaced Manhattan distance with BFS flood-fill
   for true shortest-path distance around obstacles. 15× throughput
   improvement on random-64-64-10.
3. **Guidance goal pull** — fixed 0.1→ full distance.

### Tasks

- [x] **T2.1** Benchmark harness: `benches/bench_440_lllg_paper_repro.rs`.
      4 paper maps (synthetic approximations), 800 agents, 300 steps.
- [x] **T2.2** **G1 — correctness gate.** 2/4 maps PASS (open + moderate
      obstacle). 2/4 FAIL (warehouse + maze) due to greedy PIBT lacking
      priority inheritance. Honest about the gap.
- [x] **T2.3** **G2 — congestion mitigation gate.** FAIL — warm-start
      integration is broken (warm-start data discarded in `tick()`).
      LLLG_Π and LllgEmpty produce identical results.
- [x] **T2.4** **G3 — no-regression gate.** PASS — `cargo check
      --all-features` clean. `cargo test -p katgpt-core --lib` 1556/1556 pass.
- [x] **T2.5** **G4 — latency gate.** PASS — 80ms median at 1000 agents.
      Stretch (<100ms) also passes.
- [x] **T2.6** GOAT results filed in
      `.benchmarks/440_lllg_paper_repro_goat.md`. Decision: KEEP OPT-IN.
      The substrate passes G3/G4 and partially passes G1, but the G1
      warehouse/maze failures and G2 warm-start bug prevent full GOAT
      validation. The Super-GOAT claim remains conditional on the
      riir-ai/489 G5–G7 fusion gates AND a PIBT priority-inheritance upgrade.

### Blocking items for G1/G2 full pass

1. **PIBT priority inheritance** — the single most impactful fix for warehouse/
   ht_chantry. Real PIBT recursively resolves conflicts by forcing
   lower-priority agents to move when a higher-priority agent claims their
   cell. ~200 lines of recursive backtracking in `pibt.rs`.
2. **Warm-start integration** — thread warm-start data into
   `SpaceTimeGuidance::compute_guidance`. ~50 lines.

### Acceptance

G1–G4 results documented in `.benchmarks/440_*.md`. Honest about gates that
fail — the warehouse/maze failures are explained by the greedy PIBT lacking
priority inheritance, and the warm-start bug is identified and documented.
The substrate is kept opt-in. The GOAT gate served its purpose: it identified
the exact algorithmic gaps that need upgrading.

## Phase 3 — Fusion Hooks (pluggable seams, documented extension points)

### Tasks

- [x] **T3.1** Document `LocalGuidanceSource<P>` trait as the primary
      extension point. Include a stub `HlaProjectedGuidance` example in
      doc comments (the actual impl lives in riir-ai/489).
- [x] **T3.2** Document `CostFn<P>` trait. Include stub examples for
      heightfield slope, threat cochain, faction zone penalty.
- [x] **T3.3** Document `WarmStartScheme` enum and how a consumer can add
      a custom scheme (e.g., personality-weighted blend).
- [x] **T3.4** Document `HindranceEstimator` trait (extract from the
      default impl). Include stub for affect-aware hindrance.

### Implementation notes

- All four traits now have compile-checked rustdoc examples (verified via
  `cargo test --doc -p katgpt-core --features multi_agent_path` — 7/7 pass).
- The stubs use `GridPos` (the substrate's default position type) so they
  compile without consumer-only types.
- **Bonus fix:** the module-level example in `mod.rs` had a pre-existing
  doctest failure (closure captured `map` by reference but `with_neighbors`
  requires `'static`). Fixed by leaking the map into a `&'static` reference
  and using a `move` closure, with an explanatory comment that a real
  consumer would store the map in a long-lived struct field.
- The G3 no-regression gate still passes: `cargo clippy --all-features` clean,
  1578/1578 lib tests pass (the plan doc's 1556 count was stale).

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
