# Opt-In & Gated Features — Full Detail

> These features are proven and tested but opt-in (not in default feature set).
> See main README for the default GOAT stack. Each feature is behind a feature flag.

## 1. D2F: Discrete Diffusion Forcing (Plan 066)

Block-parallel decoding via iterative denoising — a third decode strategy alongside autoregressive and speculative. Feature-gated behind `dllm`.

- **Block-causal attention**: bidirectional within block, causal across blocks → existing KV cache works
- **`D2fContext`**: pre-allocated flat buffers, zero `Vec<Vec<f32>>` per denoising step
- **`D2fPipeline`**: multi-block sequential decode with KV cache commit across blocks
- **`DecodeStrategy::DiscreteDiffusion`**: config-driven auto-switch heuristic (AR → Speculative → D2F)

📖 See [`.docs/02_inference/speculative_decoding.md`](../02_inference/speculative_decoding.md) for D2F API details and [`.research/034_D2F_Discrete_Diffusion_Forcing.md`](../../.research/034_D2F_Discrete_Diffusion_Forcing.md) for experimental results.

### Tri-Mode: D2F+AR Self-Speculation (Plan 089)

D2F drafts in parallel → AR verifies causally → accept longest prefix match. Feature-gated behind `tri_mode` (requires `dllm`).

- **`D2fDrafterVerifier`**: `d2f_decode_block()` drafts → `forward()` verifies → prefix accept + bonus token
- **`DecodeStrategy::SelfSpeculation`**: D2F+AR mode, auto-selected by `recommend()` when draft model available
- **Global Loss Averaging**: `LossAveraging::Global` (Nemotron +2.12% accuracy vs per-sequence)
- **`DiffusionSampler`**: per-position correctness predictor replaces fixed confidence threshold — Logistic (AUC 0.765) / MLP (AUC 0.781) vs fixed baseline 0.343 (Plan 116, Bench 019)
- **GOAT 9/9 passed**: Tri-Mode 4/4 (Bench 018) + DiffusionSampler 5/5 (Bench 019) + Natsukaze validation 100.0% accuracy

📖 See [`.benchmarks/018_d2f_verifier_goat.md`](../../.benchmarks/018_d2f_verifier_goat.md) and [`.benchmarks/019_diffusion_sampler_goat.md`](../../.benchmarks/019_diffusion_sampler_goat.md) for full GOAT proof results.

## 2. SR²AM Configurator Bandit (Plan 112)

Distilled from [SR²AM: Self-Regulated Simulative Reasoning](https://arxiv.org/pdf/2605.22138) (Deng, Hou, Sá Neves et al., 2026). Bandit-based per-turn planning regulation — learns when to plan deep, extend, or skip entirely.

### Adaptive Planning Decisions

| Decision | When | Effect |
|----------|------|--------|
| `PlanNew` | High uncertainty, new sub-problem | Reset tree, full budget allocation |
| `PlanExtend` | Moderate uncertainty, continuing | Keep tree, +1 depth level |
| `PlanSkip` | Low uncertainty, confident | Bypass tree, direct token sampling |

### Context-Aware UCB1 Selection

```text
Context: (domain, entropy_bin)
  → ConfiguratorBandit selects arm via UCB1
  → Reward: quality_gain − β × token_cost
```

Entropy binning (10 bins via `floor(entropy * 10.0)`) provides coarse context — low entropy → `PlanSkip`, high → `PlanNew`.

### Uncertainty-Aware Horizon Truncation

High-uncertainty states cap `draft_lookahead` at 2 (SR²AM finding: web tasks benefit from short horizons). Configurable via `max_plan_horizon` override.

### Feature Gate

`sr2am_configurator = ["bandit"]` — default-on. All new code behind feature flag. `InferenceResult` extended with `planning_decision` and `plan_horizon_used` metrics.

🧪 `tests/test_sr2am_configurator_goat.rs` — 29 integration tests (arm selection, context isolation, entropy truncation, pipeline wiring)

📖 See [`.plans/112_sr2am_configurator_bandit.md`](../../.plans/112_sr2am_configurator_bandit.md) for full plan.

## 3. FeedbackBandit — Harness + Weight Co-Evolution (Plan 178)

Distilled from [SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/pdf/2605.27276). Extends the SR²AM ConfiguratorBandit (4 arms) with 2 new arms that close the model-based/modelless loop. The bandit learns when to trigger harness hot-swaps and weight updates based on trajectory dynamics, not a fixed schedule.

### Six Arms

| Arm | Behavior | When It Helps |
|-----|----------|---------------|
| `PlanNew` | Discard tree, build fresh | High entropy / novel situations |
| `PlanExtend` | Keep tree, +1 depth | Moderate uncertainty / continuing |
| `PlanSkip` | Early exit, zero tokens | Low entropy / confident |
| `SpecHop { k }` | Continuous speculation, k threads | Fast speculator + tool-bound workload |
| `HarnessUpdate` | AbsorbCompress promote + HotSwapPruner reload | Trajectory stalled, new heuristic needed |
| `WeightUpdate` | Trigger DPO/GRPO on TrialLog buffer | Persistent plateau, model refinement needed |

### Architecture

```text
FeedbackBandit extends ConfiguratorBandit:
  Base arms (SR²AM):      PlanNew, PlanExtend, PlanSkip, SpecHop
  New arms (SIA):         HarnessUpdate, WeightUpdate
  Selection:              UCB1 over (domain, entropy_bin) context
  Exploration:            FB_UCB1_C = 0.5 (reduced) for faster feedback arm convergence
  Reward:                 quality_gain − β × cost
  Stall detection:        Δ reward < ε for N episodes → triggers feedback arm exploration
```

### Bomber Arena GOAT — ✅ PASS

**Setup:** 4 matchups × 1000 games = 4000 total, `Sr2amPlayer` with `sia_feedback` (6 arms) vs baselines.

| Matchup | Opponents | FB Wins | Win% | Top Arm |
|---------|-----------|--------:|-----:|--------|
| Easy Baselines | Random, Greedy, Validator | 147 | 14.7% | PlanNew |
| vs HL | Random, HL, Validator | 144 | 14.4% | PlanNew |
| vs GZero | Random, HL, GZero | 402 | 40.2% | PlanExtend |
| Championship | HL, GZero, Validator | 290 | 29.0% | PlanExtend |

**Aggregate:** 983W / 4000 games (24.6% win rate, ELO -9125). FB arms explored: 20 (HarnessUpdate=16, WeightUpdate=4).

### Feature Gate

`sia_feedback = ["sr2am_configurator"]` — **opt-in**. FeedbackBandit manages own 6-arm UCB1; ConfiguratorBandit remains unchanged at 4 arms when feature is off. All new code behind feature flag. 10 FeedbackBandit tests + 15 ConfiguratorBandit tests pass independently.

🧪 `examples/bomber_17_feedback_goat.rs` — 4000-game arena GOAT regression proof

📖 See [`riir-ai/.plans/178_sia_feedback_bandit.md`](../../../riir-ai/.plans/178_sia_feedback_bandit.md) for full plan.

## 4. SpecHop — Continuous Multi-Hop Speculation (Plan 131)

Hop-level speculative execution for multi-step tool-use agents. Based on [arXiv:2605.21965](https://arxiv.org/pdf/2605.21965) — continuous speculation at trajectory granularity (not token level).

### How It Works

```text
Agent trajectory:  [hop₁] → [hop₂] → [hop₃] → [hop₄]
                        ↘ spec    ↘ spec    ↘ spec
                     Thread k=1   k=2       k=3       k=4
                        ↓          ↓          ↓          ↓
                  Verify earliest pending → Commit ✓ or Rollback ✗
```

The pipeline maintains **k speculative threads** that predict tool-call observations ahead of actual tool responses. When the target tool returns, a verifier checks equivalence → commit correct branch, rollback incorrect ones.

### Theoretical Cost Model

| Parameter | Meaning | Formula |
|-----------|---------|---------|
| α | Speculator latency ratio | `E[T_spec] / E[T_target]` |
| β | Decode-to-tool ratio | `E[T_seg] / E[T_target]` |
| p | Speculator hit rate | Fraction of correct predictions |
| k* | Optimal threads | `⌈(1+β)/(α+β)⌉` (Theorem 2) |
| RelLat* | Oracle latency | `1 − p(1−α)/(1+β)` (Theorem 3) |

Example: α=0.2, β=0.15, p=0.7 → k*=4, RelLat*=0.513 (1.95× speedup).

### SR²AM Integration

`PlanningDecision::SpecHop { k }` arm added to the configurator bandit (Plan 112). Auto-activated when:
- α < 0.3 (fast speculator)
- β < 0.5 (tool-bound workload)
- `reward = latency_reduction / α > 1.0`

### Hop-Level DDTree

`build_hop_dd_tree()` extends the token-level DDTree concept to hop granularity. Each node is an (action, observation) pair scored by speculator confidence. `verify_hop_tree()` wires `ObservationVerifier` for branch accept/reject.

### Module Structure

```text
src/spechop/
├── mod.rs              # Module index, re-exports, feature gate
├── types.rs            # SpecHopConfig, HopObservation, SpecOutcome, HopState
├── cost_model.rs       # α/β/p → k*, RelLat, starvation probability
├── verifier.rs         # ObservationVerifier trait + RuleBasedVerifier
├── speculator.rs       # HopSpeculator trait + CacheSpeculator + BanditSpeculator
├── window.rs           # SpecWindow k-bounded thread manager
├── pipeline.rs         # SpecHopPipeline continuous loop (Algorithm 1)
├── hop_tree.rs         # Hop-level DDTree integration
└── segment_match.rs    # Rolling hash sub-sequence matching (Plan 140 T19, behind cache_prune+spechop)
```

### Examples

```bash
cargo run --example spechop_01_pipeline --features spechop   # 4-hop continuous speculation
cargo run --example spechop_02_cost_model --features spechop  # α/β/p → k* and RelLat
```

🔧 Feature flag: `spechop = ["bandit"]` (**opt-in** — requires GOAT proof before default-on promotion)

📖 See [`.plans/131_spechop_continuous_spec_pipeline.md`](../../.plans/131_spechop_continuous_spec_pipeline.md) for full plan (T1–T32, T40–T41 complete).

## 5. Parallel-Probe 2D (Plan 133)

Training-free 2D probing controller for N parallel reasoning branches. Based on [arXiv:2602.03845](https://arxiv.org/pdf/2602.03845) — monitors branches via periodic answer extraction, uses **consensus-based early stopping** + **deviation-based branch pruning** to reduce sequential tokens by ~30%.

The key insight: **answer-level consensus across parallel branches is O(N) per probe step** — uniquely cheap compared to EqR distribution residuals (O(N×V)) or trajectory bandit scores (requires reward signal).

```text
Parallel Branch 0: ...think...think... → "42"
Parallel Branch 1: ...think...think... → "42"  ← consensus!
Parallel Branch 2: ...think...think... → "17"  ← deviant, prune after k steps
                     ↑
              Probe every Δ tokens
              → majority vote → stop if stable for u steps
              → prune branches that disagree for k steps
```

### Components

| Component | Purpose |
|-----------|----------|
| `ParallelProbeController<A>` | Generic controller: probe(), majority_vote(), should_stop(), should_prune() |
| `ProbeDecision` | Continue / Stop / Prune / StopAndPrune |
| `AnswerExtractor` trait | Pluggable answer extraction (regex, think-token, game actions) |
| `RegexAnswerExtractor` | `\boxed{...}`, "The answer is ...", numeric patterns |
| `ThinkTokenExtractor` | `</think`> boundary detection |
| `DiscreteActionExtractor` | Game domain actions (Bomber, Go moves) |
| `ParallelProbeVerifier<V>` | Wraps any `SpeculativeVerifier` with probe control |

26 unit tests covering: consensus detection, deviation pruning, warmup suppression, all answer formats, integer/generic answer types.

🔧 Feature flag: `parallel_probe` (**default-on**)

📖 See [`.plans/133_parallel_probe_2d_probing.md`](../../.plans/133_parallel_probe_2d_probing.md) for full plan.

## 6. GFlowNet Modelless Distillation (Plan 052)

Distills the GFlowNet shortest-path theorem — **minimize flow = shortest paths** — into the existing ScreeningPruner + BanditPruner + DDTree stack **without any neural network training**.

**Core insight:** The paper proves that minimizing expected trajectory length `E[nτ]` forces the backward policy `P_B` to assign zero probability to all non-shortest paths. Our stack already computes forward marginals (LoRA logits = P_F), backward relevance (WASM validator = P_B), and flow proxy (BanditPruner Q-values = F(s)). We harmonize these signals.

### Four Additive Distillations

| Distillation | Component | What It Does |
|-------------|-----------|-------------|
| **D1: FlowPruner** | `FlowPruner<P: ScreeningPruner>` | Wraps any screener, adds `λ × (1 - stop_prob[depth])` flow bonus |
| **D2: Balanced DDTree** | `build_dd_tree_balanced()` | Scores beams with `ln(P_llm) + w × ln(R) + λ × flow_bonus` |
| **D3: Flow-weighted bandit** | `observe_delta_with_flow()` | Adds `λ_length / prefix_len` trajectory length bonus to δ reward |
| **D4: Backward replay** | `ReplayBackwardWalker` | Walks winning replays backward, finds safe alternatives = P_B data |

### Benchmark Results (NoScreeningPruner baseline)

| Metric | Result |
|--------|--------|
| FlowPruner node delta | **+0.0%** ✅ |
| Balanced DDTree backward compat | **Identical to `build_screened`** ✅ |
| Flow-weighted bandit reward delta | **+0.0%** ✅ |
| Backward replay alternatives | **4.0 avg/tick** (target: ≥2) ✅ |

Run: `cargo test --features "bandit,g_zero,bomber" --test bench_gflownet_modelless -- --nocapture`

📖 See [`.plans/052_gflownet_modelless_distillation.md`](../../.plans/052_gflownet_modelless_distillation.md) for full plan, [`.research/023_GFlowNet_Shortest_Paths.md`](../../.research/023_GFlowNet_Shortest_Paths.md) for paper analysis.

## 7. ROPD Rubric Modelless Distillation (Plan 071)

Distills ROPD's rubric-based scoring into our modelless stack. Replaces scalar [`HintDelta`](#-g-zero-verifier-free-self-play-plan-049) with structured [`RubricVector`] — multi-criteria reward without LLM judges. Template rubrics + pattern scorers provide per-criterion scoring at inference speed (~µs).

### Key Innovation: Per-Criterion Gap Targeting

- **Scalar δ**: `gate = mean_delta > threshold` (blind — *why* did it trigger?)
- **Rubric**: `gate = any(high_weight_criterion_gap > threshold)` (targeted — "constraint #2 failed")

### Multi-Reference Requirement

ROPD ablation (Table 6): m=4→m=1 costs **−17.94 pts** — the single biggest impact. Single reference over-anchors rubric to one trajectory. Always use M ≥ 2 references.

### Benchmark Results (`.benchmarks/007_ropd_rubric_modelless.md`)

| Method | Throughput | Hot-path overhead |
|--------|-----------|-------------------|
| `observe_rubric()` (bomber) | 4.9M/sec | — |
| `observe_rubric()` (generic) | 5.3M/sec | — |
| `RubricBanditPruner::observe_rubric()` | 14.1M/sec | — |
| `relevance()` (absorb) | — | ~0% (inlined) |
| `relevance()` (bandit) | — | -2.7% (inlined) |

| Targeting | Detected | Expected |
|-----------|----------|----------|
| High-weight gaps (w=4.0) | 20/20 | ✅ All |
| Low-weight gaps (w=1.0) | 0/10 | ✅ Filtered |
| No-gap arms | 0/55 | ✅ Excluded |

**Feature gate:** `ropd_rubric = ["bandit"]` — off by default.

## 8. VPD — Variational Policy Distillation

EM-style co-evolutionary teacher-student distillation that actively trains the feedback-conditioned teacher via BCO (Binary Cross-Entropy Optimization).

- **E-step (every F=5 rounds)**: BCO refines teacher Q-values from unpaired outcome preferences
- **M-step (every round)**: KL-gated distillation of teacher → student with dynamic prior
- **Dynamic prior**: Student Q tracks teacher Q via soft update (η=0.2), breaking SDAR plateau
- **+6.3% win rate over SDAR** in fixed-seed bomber tournament (38.0% vs 31.7%)
- **Non-degrading** in varied-seed arena (within 2.3% of SDAR over 1000 games)

Feature gate: `vpd_em_distill` (requires `sdar_gate`, `bandit`)

```rust
use katgpt_rs::pruners::vpd_em::{VpdConfig, VpdEmCycle};
use katgpt_rs::pruners::bomber::VpdPlayer;

// Create VPD player with paper defaults
let player = VpdPlayer::new(0);

// Or customize: F=5, β=0.1, λ=0.1, dynamic prior
let config = VpdConfig::default();
let player = VpdPlayer::with_config(0, config);
```

Paper: arXiv:2605.15113 — Variational Policy Distillation (Salesforce AI Research, 2026)

## 9. Committee Boost (Plan 132)

Four diagnostics from the [boosting committee paper](https://arxiv.org/pdf/2605.14163) that our DDTree + BtRank + ScreeningPruner stack already supports conceptually but lacked as measurable metrics:

| Diagnostic | What It Measures | Our Stack Mapping |
|------------|-----------------|-------------------|
| **Oracle-gap recovery** `Rec = (p_system - p1) / (p_oracle - p1)` | How much latent capability the selector recovers | `ConstraintPruner` measures selection vs coverage failure |
| **Position-swap debiasing** | Eliminates lead-position bias in BtRank | `DebiasedComparator` wraps pairwise comparison |
| **Budget sizing** (Theorem 3) | Given (α₀, β₀, σ₀, L, δ) → optimal (k, m, r) | Sizes DDTree width, ScreeningPruner depth, BtRank votes |
| **Blind-spot floor** `B = 1 - lim_{k→∞} p_oracle(k)` | Proposer diversity ceiling | CoverageDiagnostic recommends action |

The paper proves our stack IS the committee protocol Π_{k,m,r}. These additions make the theoretical guarantees **measurable and actionable**.

### GOAT Proof Results (`.benchmarks/020_committee_boost_goat.md`)

Run: `cargo test --features committee_boost --test bench_committee_boost_goat -- --nocapture`

| Proof | Description | Verdict |
|-------|-------------|--------|
| G1 | Oracle-gap recovery: Rec within ±0.01 for 6 known cases | ✅ |
| G2 | Debiased comparison: 100% Tie rate for biased comparator | ✅ |
| G2b | Debiasing catches lead-position bias (false rankings eliminated) | ✅ |
| G3 | Budget sizing: Theorem 3 monotonicity + determinism | ✅ |
| G3b | Budget rejects all invalid parameters | ✅ |
| G4 | Blind-spot floor: 8 cases verified (B estimation, convergence, diagnostics) | ✅ |
| G5 | End-to-end: committee improves ≥5% over single-shot | ✅ |

### Key API

```rust,ignore
use katgpt_rs::pruners::committee_boost::{
    OracleGapRecovery, FailureMode, DebiasedComparator, CommitteeBudget,
    committee_budget, estimate_blind_spot_floor, coverage_diagnostic,
};

// Oracle-gap recovery
let r = OracleGapRecovery::new(0.5, 0.8, 0.74);
let rec = r.recovery(); // Some(0.8)
let mode = r.failure_mode(); // CoverageLimited
let diag = r.diagnostic(); // "Recovery=80.0% (coverage-limited); ..."

// Debiased BtRank comparison
let comparator = DebiasedComparator::new(|i, j| biased_compare(i, j));
let comparisons = comparator.tournament(4); // Vec<BtComparison>

// Budget sizing (Theorem 3)
let budget = committee_budget(10, 0.05, 0.3, 0.2, 0.4, 2)?;
println!("k={}, m={}, r={}", budget.k, budget.m, budget.r);

// Blind-spot floor
let rates = vec![(1, 0.5), (2, 0.65), (4, 0.75), (8, 0.8)];
let b = estimate_blind_spot_floor(&rates); // 0.2
let diag = coverage_diagnostic(&rates);
println!("B={:.3}, action={}", diag.blind_spot_floor, diag.action);
```

### Module Structure

```
src/pruners/committee_boost/
    mod.rs               ← Module index, re-exports
    types.rs             ← OracleGapRecovery, FailureMode
    debiased_compare.rs  ← DebiasedComparator, debiased_compare
    budget.rs            ← CommitteeBudget, committee_budget
    blind_spot.rs        ← BlindSpotEstimate, coverage_diagnostic
tests/
    bench_committee_boost_goat.rs  ← 7-proof GOAT benchmark
```

**Feature gate:** `committee_boost = ["bt_rank", "bandit"]` — **opt-in**.

📖 See [`.research/093_Boosting_Weak_Reasoning_Committee_Search.md`](../../.research/093_Boosting_Weak_Reasoning_Committee_Search.md) for the paper distillation.

## 10. Induced CWM (Plan 296)

Open half of the Code World Models Super-GOAT, distilled from [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542) (Lehrach et al., DeepMind Oct 2025). A generic, IP-free trait surface for LLM-induced forward-model implementations that are **verifiable** (transition unit tests), **committable** (BLAKE3 over canonical bytes), and **hot-swappable** (atomic slot). The kernel primitive is shipped open in `katgpt-core`; the LLM-induction pipeline itself is private (riir-ai Plan 326).

The primitive exists to let downstream consumers (Bomber, Go, NPC domains, custom IIGs) plug in induced forward models behind a stable trait boundary — `InducedCwmKernel: GameState` — without coupling to any specific induction recipe.

- **`induced_cwm`** — `InducedCwmKernel: GameState` marker + `CwmCommitment` (BLAKE3) + `BeliefInferenceFn<S>` + `TransitionUnitTest` + `verify_transition` (Plan 296 Phase 1).
- **`induced_cwm_ismcts`** (requires `induced_cwm`) — Information-Set MCTS over an induced CWM + belief fn: `ismcts_search_with_inference<S, B>` + `InformationSet` + `NodeStats` (Plan 296 Phase 2).
- **`induced_cwm_tournament`** (requires `induced_cwm`) — Value Function Tournament: round-robin arena-play selector over `StateHeuristic` candidates, `ValueFnTournament<S, V>` + `PlayerStats` + `TournamentWinner` (Plan 296 Phase 3).

Phase 4 ships `InducedCwmSlot<K>` — lock-free atomic hot-swap slot for live kernel replacement (under the `induced_cwm` feature).

**GOAT 4/4 PASS** (all gates green, see [`.benchmarks/296_induced_cwm_primitive_goat.md`](../../.benchmarks/296_induced_cwm_primitive_goat.md)):

| Gate | Target | Verdict |
|------|--------|--------|
| **G1** Verifiability | 100% pass on known-correct transitions; correct diff on mutation | ✅ PASS |
| **G2** Play strength | ISMCTS picks non-fold ≥ 70% when P(strong) ≥ 0.6 | ✅ PASS |
| **G3** Latency | `advance()` ≤ 10 µs/call on mock CWM | ✅ PASS (~1–5 ns, ~3 orders of magnitude under budget) |
| **G4** Commitment integrity | Same logical kernel → identical BLAKE3 across 10 re-runs | ✅ PASS |

The primitive stays **opt-in by design** — it's a primitive surface, not a default-on capability; downstream pipelines opt in by enabling the feature. **Ready for downstream consumption** (riir-ai Plan 326).

### Examples

```bash
cargo run --example induced_cwm_01_mock_iig            --features induced_cwm_ismcts        # Phase 2: mock Leduc IIG + ISMCTS
cargo run --example induced_cwm_02_value_tournament    --features induced_cwm_tournament     # Phase 3: value-fn tournament arena
```

🔧 Feature flags: `induced_cwm`, `induced_cwm_ismcts` (deps `induced_cwm`), `induced_cwm_tournament` (deps `induced_cwm`) — all **opt-in**.

📖 See [`.plans/296_induced_cwm_kernel_primitive.md`](../../.plans/296_induced_cwm_kernel_primitive.md) for the plan, [`.research/275_Code_World_Model_Induced_Forward_Model.md`](../../.research/275_Code_World_Model_Induced_Forward_Model.md) for the paper distillation, [`.benchmarks/296_induced_cwm_primitive_goat.md`](../../.benchmarks/296_induced_cwm_primitive_goat.md) for the GOAT proof (G1–G4 all PASS).

## 11. HLA Windowed Eigenbasis Recovery (Issue 001)

Per-NPC eigenbasis recovery from a windowed HLA activation matrix — **modelless**, no LAPACK, no training. Power iteration with deflation on the D×D Gram `W^T W` (D = HLA dim, 8 today) recovers the top-`k` orthogonal principal directions of a single NPC's recent affective trajectory. Those eigenvectors are the right singular vectors `V` of `W`; their eigenvalues are `σ²`. The recovered basis is a per-NPC rotation/projection matrix usable for emotion routing, zone attention, or adapter selection — every NPC currently shares the same hand-tuned universal basis (Research 032); this exposes individualized affective geometry from the NPC's *own* experience.

The deterministic seed is `1/sqrt(D)` (no RNG), mirroring `stable_rank_update_into` — the same cross-platform determinism surface.

Three entry points serve three operating points:

| Entry point | Path | When to use |
|------------|------|-------------|
| `recover_eigenbasis_from_window` | cold (BLAKE3 + `Uuid::now_v7` provenance) | freeze/thaw cache validation |
| `recover_eigenbasis_from_window_fast` | cold-start (no provenance, rebuilds Gram) | first-time recovery from a stored window |
| `EigenbasisTracker` | plasma-tier hot path (incremental Gram, O(D²)/tick) | live NPC, one push + one recover per tick |

**GOAT gate PASS (synthetic, 2026-06-30)** — see [`.benchmarks/001_hla_eigenbasis_recovery_goat.md`](../../.benchmarks/001_hla_eigenbasis_recovery_goat.md):

| Gate | Target | Verdict |
|------|--------|--------|
| **G1** Latency (`EigenbasisTracker` hot path) | ≤ 2 µs/tick, T=512 D=8 k=4 | ✅ PASS (613.9 ns/tick, 3.25× margin) |
| **G2** Determinism (same-binary) | 0 bit diffs | ✅ PASS (cross-platform protocol in `tests/hla_eigenbasis_determinism.rs`) |
| **G3** Quality (reconstruction error) | < 0.10, k=4, rank-3 ground truth | ✅ PASS (0.0003, 333× margin) |
| **G4** Behavioral divergence | > 50% of 1000-NPC pairs cos < 0.7 | ✅ PASS (87.8%) |
| **G5** Memory (per-NPC) | ≤ 256 bytes at D=8, k=4 | ✅ PASS (144 bytes, 1.78× margin) |

**Opt-in by design.** The issue's GOAT outcome requires a head-to-head against Research 032's hand-tuned axes + a private `riir-ai` architectural guide before promotion to default — both cross the repo boundary and are tracked as `riir-ai` follow-ups. The stateless path (~9 µs) and full provenance path (~17 µs) are reported for transparency; only the `EigenbasisTracker` hot path is the G1 budget path.

**Sync-boundary compliant** (per AGENTS.md): the recovered eigenbasis stays local to the NPC — never synced, never crosses `LatCalFixed`/`SyncBlock`, never used for anti-cheat. `EigenbasisProvenance.window_hash` is a cache key, not a synced value.

🔧 Feature flag: `hla_eigenbasis_recovery` — **opt-in**.

📖 See [`.benchmarks/001_hla_eigenbasis_recovery_goat.md`](../../.benchmarks/001_hla_eigenbasis_recovery_goat.md) for the full GOAT proof and the G1 three-path latency breakdown.

## 12. Canvas Schema Compiler (Plan 419)

A typed `CanvasSchema` compiler that lowers a declared region layout + directed topology into a sparse `AttentionMaskSpec` (consumable by AC-Prefix / VortexFlow / any sparse-attention path), a per-position `LossWeightMask`, and a **reachability** primitive proving **exact marginal independence for binary masks** — absent edge ⟹ no influence, by construction. Plus a `transfer_distance` semantic-type compatibility scalar (`1 − cosine` of frozen embeddings, schema-ABI check from paper §2.4 Table 1).

**Modelless by construction** (Plan 419, Research 398, Valdez *Canvas Engineering* July 2026): every primitive is a pure function over index sets + graphs. Zero backprop, zero weight mutation. The compiler ships on **structural / correctness** merits — the reachability guarantee is provable by construction (like the DEC `d∘d=0` identity, Plan 251), NOT on the paper's behavioral headline (1.73× parameter efficiency, cortical R²=0.825), which is **training-dependent** (`.issues/043` fusion PoC resolved-and-removed 2026-07-09, inconclusive; see Research 398 §7–8).

**Direction convention (paper §2.2):** `Connection(src, dst)` licenses `src` to query `dst` keys/values; information flows `dst → src`; the information-flow graph `G` has arc `dst → src`; `can_reach(from, to)` therefore reads as "`from` influences `to`". `causal_chain([A,B,C])` emits each region querying its predecessor → info arcs `A → B → C` → `can_reach(A, C, 2) == true` (Plan 419 T3.6).

**GOAT gate — ✅ PASS (all G1–G6)**, see [`.benchmarks/419_canvas_schema_goat.md`](../../.benchmarks/419_canvas_schema_goat.md):

| Gate | Target | Verdict |
|------|--------|--------|
| **G1** Reachability soundness (LOAD-BEARING) | absent edge ⟹ `can_reach == false` ∀ horizons | ✅ PASS (exact marginal independence by construction) |
| **G2** Horizon bound (T3.6) | `can_reach(A,C,1)=false`, `can_reach(A,C,2)=true` | ✅ PASS |
| **G3** No-regression | `--all-features` + `--no-default-features` clean | ✅ PASS |
| **G4** Alloc-free hot path | `TransitiveClosure::reaches` + `reachability_horizon` = 0 allocs/call | ✅ PASS (0/1000 reaches, 0/1000 horizon) |
| **G5** Perf | `compile_schema` (199-region ICU schema) < 10 ms; `reaches` p50 < 100 ns | ✅ PASS (compile = **1515 ns** (6600× under); reaches p50 = **0 ns**) |
| **G6** Feature isolation | `canvas_schema` gates all symbols; 0 bytes when disabled | ✅ PASS |

**What the GOAT does NOT claim** (the honesty): behavioral parity with the paper's training-dependent results. Applying a declared-topology mask to a frozen untrained-for-it backbone is a documented 19% loss (paper §5 calibration #2). The modelless primitive ships the *compilation* + the *guarantee*; the *behavioral gain* requires riir-train (`.issues/043` fusion PoC resolved-and-removed 2026-07-09, inconclusive).

Module split (AGENTS.md `< 2048` line rule): `canvas/{mod,types,mask,reachability,transfer}.rs`.

🔧 Feature flag: `canvas_schema` — **opt-in** (promotion deferred; `.issues/043` fusion PoC resolved inconclusively, constituents already default-on with runtime consumers — see Research 398 §8).

📖 See [`.benchmarks/419_canvas_schema_goat.md`](../../.benchmarks/419_canvas_schema_goat.md) for the full GOAT proof + the direction-convention derivation.

## 13. Multi-scale V-cycle on Cell Complexes (Plan 413)

Fills the multi-scale composition gap in the shipped single-complex DEC operators (`exterior_derivative`, `codifferential`, `hodge_laplacian`, `hodge_decompose`). Those handle one resolution level; `htno_v_cycle` composes two (fine → coarse → fine): restrict a fine vertex cochain to a coarse complex, apply a caller-supplied coarse operator, prolongate back — the classic multigrid V-cycle on DEC cochains.

**GOAT gate:** G1 (commutativity) `dₖKc ∘ Rₖ = Rₖ₊₁ ∘ dₖK` verified on induced sub-complexes; G2 (perf) restrict/prolongate cheaper than rebuilding the complex; G3 (no-regression) clean with/without feature; G4 (alloc-free) `htno_v_cycle_into` zero bytes beyond pre-allocated scratch.

The 2×2 aggregation coarsening is documented as **non-commuting** (its coarse edges are long-range, not fine edges) — the V-cycle still provides coarse smoothing, but it is a smoother, not a d-commuting transfer.

🔧 Feature flag: `htno_v_cycle` — **opt-in** (in `katgpt-dec`). Forwarded through `katgpt_core::dec::htno_v_cycle`.

📖 Plan: [`.plans/413_multiscale_v_cycle_primitive.md`](../../.plans/413_multiscale_v_cycle_primitive.md)

## 14. HLA Committed-Belief π-Sensitivity Probe (Plan 414)

A modelless diagnostic that perturbs the committed `π` weights of a `CommittedFieldBlend`, re-evaluates the blend map, and measures output drift against an on-the-fly theoretical **π-sensitivity Lipschitz bound** (`L_π = max_j (1/τ)·σ_j·(1−σ_j)·‖f_j(z)‖`). A bound violation flags a numerics bug in the committed blend.

**Key design correction:** the cached `CommittedFieldBlend::lipschitz_bound` computes the **z-sensitivity** bound, not the π-sensitivity bound. The F4 probe computes its own on-the-fly π-bound using the actual `‖f_j(z)‖` — so it catches bugs even when a field under-reports its Lipschitz constant.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Lipschitz bound holds | 1000/1000 random configs, 0 violations | ✅ PASS |
| **G2** | Bug detection (NaN → reject) | NaN in π → `accepted=false` | ✅ PASS |
| **G3** | No regression | 13/13 existing tests pass | ✅ PASS |
| **G4** | Zero-alloc hot path | 0 allocs/1000 calls | ✅ PASS |
| **G5** | Latency | p50 = 3.042µs (target <5µs) | ✅ PASS |

DRY refactor extracts `apply_blended_with_pi` free function shared by production + probe.

🔧 Feature flag: `hla_committed_belief_probe` — **opt-in** (diagnostic/self-verifier, no runtime consumer yet). F4 fusion follow-up from Plan 406 (renoise-CE).

📖 Plan: [`.plans/414_hla_committed_belief_lipschitz_probe.md`](../../.plans/414_hla_committed_belief_lipschitz_probe.md)

## 15. Within-Class Effective Rank (Plan 415)

Class-conditioned collapse diagnostic: the entropy-based effective rank of the **within-class residual** covariance matrix (arXiv:2412.19419 §5.3.1). Fusion of two shipped halves never combined: `effective_rank` (class-agnostic) + `within_class_adjacency` / `between_class_adjacency` (class-conditioning from `riir-ai/crates/riir-engine/src/latent_functor/quality_gate.rs`).

Fills the gap where the class-agnostic `effective_rank` cannot distinguish "between-class variance dominates, within-class collapsed" from "all variance is healthy and isotropic". The existing Dirichlet-energy quality gate measures *separation* (between > within) but not *within-class subspace health*.

**Key insight:** effective rank is scale-invariant — tiny-but-isotropic within-class variance still gives high rank; the low-rank signal requires rank-deficient within-class structure, not just small-magnitude variance.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | `r_WC ∈ [1, min(d, n−C)]`, monotone | 3 tests pass | ✅ PASS |
| **G2** | Non-redundancy vs global `effective_rank` (load-bearing) | within ≈ 0, global ≈ 3 | ✅ PASS |
| **G3** | No regression | 1385 tests pass | ✅ PASS |
| **G4** | Latency | within-class 232µs (0.485× of global) | ✅ PASS |

Not UQ-bearing, not Super-GOAT (Q2 fails — better diagnostic for existing class, not new class).

🔧 Feature flag: inherits `sink_aware_attn` (same gate as sibling `effective_rank`). **Opt-in** — stays alongside its sibling.

📖 Plan: [`.plans/415_within_class_effective_rank.md`](../../.plans/415_within_class_effective_rank.md)

## 16. Cochain Point Sampler (Plan 422)

Continuous intra-primitive cochain field sampler that answers "what is the cochain value at continuous point `p` inside cell `Ω`?" with local-coordinate conditioning. The modelless LPPN *input* computation — Whitney/de-Rham reconstruction turning a discrete `CochainField` into a continuously-queryable field.

Ships quad (2D grid, bilinear λ-weights) and triangle (mesh, barycentric sort + CDF remap) samplers with local-coordinate augmentation (`sin/cos` harmonics for quad, barycentric sort-CDF for tri). The barycentric sort enforces C⁰ continuity across triangle edges (vertices listed in arbitrary order per face).

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Linear-precision exactness | 1250 points, all < 1e-5 | ✅ PASS |
| **G2** | Partition-of-unity (Σλ = 1, λ ≥ 0) | both quad + tri | ✅ PASS |
| **G3** | C⁰ continuity across boundaries | 0 diff | ✅ PASS |
| **G4** | Zero-alloc steady state | 0 allocs on `*_into` paths | ✅ PASS |
| **G5** | Latency | 11.2 ns/call on 64×64 grid | ✅ PASS |

🔧 Feature flag: `cochain_point_sampler` — **opt-in** (in `katgpt-dec`). Gain-tier — substrate-completeness primitive, not a default-path improvement.

📖 Plan: [`.plans/422_cochain_point_sampler_primitive.md`](../../.plans/422_cochain_point_sampler_primitive.md), Research: [`.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md`](../../.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md), Paper: [arXiv:2506.22899](https://arxiv.org/abs/2506.22899)

## 17. Spectral Rewiring (Plan 423)

The modelless SAR kernel: project a weight delta onto the base matrix's SVD subspace, extract the compact rewiring matrix M, reconstruct the purified on-manifold delta ΔW*. Reuses `thin_svd_into` from `subspace_phase_gate` (Plan 301).

**Stays opt-in** because the spectral concentration assumption (G1b) is unvalidated without real training deltas — a generic delta is NOT concentrated (0.12–0.18). Promotion to default is blocked on Issue 123 (real-delta test). The SVD 64-col cap (Issue 124) blocks 128×128/512×512. The cached-index path (`SpectralRewireIndex`) is the recommended hot-loop API.

**LLM-scale escape hatch CLOSED (Issue 151, 2026-07-15).** The SAR concentration phenomenon was the foundational assumption for the SAR × QuasiMoTTo Pass@k fusion (Issue 151) — the hypothesis was that concentration holds at LLM scale (4096×4096 weight matrices) even though it failed at NPC scale (Issue 123). A 1.5B-scale Phase 1 PoC **refuted** concentration: 0/196 layers exceeded the 0.8 `on_manifold_fraction` threshold. The LLM-scale regime does NOT rescue the concentration assumption. See negative_results §10.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1a** | SVD recovery | ~8e-6 | ✅ PASS |
| **G1b** | Spectral concentration at NPC-scale | 0.12–0.18 (NOT concentrated) | ❌ UNVALIDATED |
| **G3** | Determinism | bit-identical | ✅ PASS |
| **G4** | Zero-alloc | 0 allocs | ✅ PASS |
| **G5** | Latency | 0.41µs NPC-scale (cached-index) | ✅ PASS |

Cross-repo applications (freeze/thaw purification, spectral LoRA, spectral TIES) are noted as follow-ups but NOT implemented in this plan.

🔧 Feature flag: `spectral_rewire` — **opt-in** (blocked on Issue 123 real-delta validation).

📖 Plan: [`.plans/423_spectral_rewire_primitive.md`](../../.plans/423_spectral_rewire_primitive.md)

## 18. GDN Rollback-Free Tree Verification (Plan 424)

Verifies speculative draft trees against GDN (Gated DeltaNet) recurrent layers **without rolling back the recurrent state**. The algorithm (arXiv:2607.06763 §3.4) extends the chunked delta-rule recurrence to tree-structured drafts via a partial order (ancestor relation), reducing verification to a masked triangular solve `(I + X)U = βV` followed by an ancestor-masked output read.

Fills a confirmed gap: katgpt-rs ships GDN2 (Plan 105, default-on) and KV-cache snapshot/rollback tree verification for attention models (Plan 012), but has **no tree verification for GDN/delta-rule recurrent layers**. Includes multi-head batching + QwenDeltaNet hybrid integration (attention layers use per-branch sequential KV-rollback; DeltaNet layers use tree verify).

**Chain tree speedup matches paper's B200 GPU numbers on CPU SIMD**:

| Tree size T | Speedup | Paper B200 |
|---|---|---|
| T=16 | **1.93×** | 1.5× |
| T=32 | **2.79×** | 2.7× |
| T=64 | **4.66×** | 4.6× |
| T=128 | **7.09×** | 7.1× |

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Bit-exact vs per-branch sequential verify | within 1e-3 (f32 accumulation) | ✅ PASS |
| **G2** | ≥2× faster at T=32, ≥4× at T=64 | 2.79× / 4.66× / 7.09× | ✅ PASS |
| **G3** | No regression | 1429 tests pass | ✅ PASS |
| **G4** | Alloc-free hot path | 0 allocs steady-state | ✅ PASS |

Phase 6 (DDTree argmax-of-marginal tuning) produced a **negative result** — the paper's §3.5 insight does not transfer to best-first tree building (best-first search already prioritizes the argmax path naturally).

🔧 Feature flag: `gdn_tree_verify` — **opt-in** (complement to Plan 012's attention verify; only relevant for `QwenDeltaNet` / GDN-layer configs).

📖 Plan: [`.plans/424_gdn_tree_verification_primitive.md`](../../.plans/424_gdn_tree_verification_primitive.md), Research: [`.research/407_Trees_from_Marginals_GDN_Tree_Verify.md`](../../.research/407_Trees_from_Marginals_GDN_Tree_Verify.md), Benchmark: [`.benchmarks/424_gdn_tree_verify_goat.md`](../../.benchmarks/424_gdn_tree_verify_goat.md), Paper: [arXiv:2607.06763](https://arxiv.org/abs/2607.06763)

## 19. Interpolation Geometry — iMAUVE + Intervention Battery (Research 445)

Modelless evaluation methodology for committed latent substrates — answers "does the *midpoint* of two committed latents decode to a coherent intermediate behavior?" Distilled from Prabhudesai & Geng, *Latent Thought Flows with Text Compression* (Jun 2026). The paper's headline metric **iMAUVE** (nearest-neighbor midpoint interpolation quality) predicts downstream generation quality with Pearson r=0.99; the **5-way intervention probe** (matched/shuffled/zero/mean/noise) extends Plan 278's binary FaithfulnessProbe to per-entity committed state.

Generic `LatentSpace` trait abstracts over the **six committed-latent substrates** cataloged in Research 445 §2.6: HLA `[f32;8]`, `NeuronShard::style_weights[64]`, `ArchetypeBlendShard.pi`, `KarcShard.wout`, `ZoneGeometryPod`, `MerkleFrozenEnvelope`-versioned states. Pure evaluation methodology — NOT a training primitive.

**Three-pressure audit (all six substrates PASS):**
- **Q1 (summarize-vs-route)** — does the latent summarize the underlying trajectory, or is it a lookup key? Subsample the trajectory, recompute the latent, measure divergence. A summarizing latent is stable under subsampling.
- **Q2 (runtime-depends-on-latent)** — does runtime behavior actually use the committed latent, or bypass via raw state? Zero/shuffle the latent (intervention battery), measure behavior delta.
- **Q3 (local-context-vs-bypass)** — does the runtime's attention to the latent stay local? Structural code audit (decode-path purity + consumer-input audit + locality-mechanism inventory). SpKv (Plan 070) + RTPurbo (Plan 126) enforce locality at the transformer-attention layer.

🔧 Feature flag: `interpolation_geometry` (in `katgpt-core`) — **opt-in**. Pure evaluation methodology, no runtime consumer.

📖 Research: [`.research/445_Latent_Thought_Flows_Text_Compression.md`](../../.research/445_Latent_Thought_Flows_Text_Compression.md), Benchmark: [`.benchmarks/456_interpolation_geometry_goat.md`](../../.benchmarks/456_interpolation_geometry_goat.md), Doc: [`.docs/04_calibration/interpolation_geometry.md`](../04_calibration/interpolation_geometry.md), Source: [latent-thought.vercel.app](https://latent-thought.vercel.app) (Prabhudesai & Geng 2026 blog + MeanFlow arXiv:2601.22158)

## 20. GRAPE-M Rank-2 Rodrigues Exponential (Research 446)

Closed-form application of `exp(n·ω·L)` for an arbitrary rank-2 skew-symmetric generator `L = abᵀ − baᵀ ∈ so(d)` (arXiv:2512.07805 §2.3). Uses the Rodrigues formula `I + (sin s / s)·L + ((1 − cos s) / s²)·L²` with `s = ω·‖a∧b‖`, evaluated as `O(d)` work via two inner products `⟨a,x⟩`, `⟨b,x⟩` (no materialized `L` or `L²`).

Generalizes `phase_rotation`'s scalar-broadcast 2D rotation (canonical basis special case where `a = e_i`, `b = e_{i+D/2}`). Pure modelless float arithmetic on a user-supplied plane `(a, b)`.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Bit-identical to materialized `expm(L)` on random `(a,b,ω)` | 0.0 diff | ✅ PASS |
| **G2** | Latency `< 2× phase_rotation_gate_into` | within bound | ✅ PASS |
| **G4** | Alloc-free | 0 allocs | ✅ PASS |

🔧 Feature flag: `grapem_rodrigues` (in `katgpt-core`) — **opt-in**. `Rank2Plane` retains `a, b` as `Box<[f32]>` (not just the 4 scalars) — mathematically necessary for the projections.

📖 Research: [`.research/446_GRAPE_Group_Representational_Position_Encoding.md`](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md), Benchmark: [`.benchmarks/457_grapem_rodrigues_goat.md`](../../.benchmarks/457_grapem_rodrigues_goat.md), Paper: [arXiv:2512.07805](https://arxiv.org/abs/2512.07805)

## 21. Unified PositionGroupAction Trait (Research 446)

Abstract trait unifying five position-encoding families under one `G(n) = exp(n·ω·L)` interface (arXiv:2512.07805 §2.2 + §4.1):

- **RoPE** (`SO(d)` multiplicative action) — wraps `PositionFreeCompactor`'s math
- **ALiBi / FoX / Wall** (`GL(d+2)` unipotent lift — additive bias family)
- **NoPE** (trivial `L = 0`)
- **GRAPE-M** (rotary generalization — wraps `Rank2Plane`)

All obey the exact relative law `G(t−s) = G(s)^T·G(t)`, enabling position-encoding-agnostic tooling (KV compaction, attention matching). Hot-path code keeps using `PositionFreeCompactor` / `WallDiagonalGate` directly; the trait is for cold-path interop.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G3** | No-regression — existing RoPE/Wall paths unchanged when feature off | bit-identical | ✅ PASS |

🔧 Feature flag: `position_group_action` (in `katgpt-core`, implies `grapem_rodrigues`) — **opt-in**. 19 unit tests in-crate.

📖 Research: [`.research/446_GRAPE_Group_Representational_Position_Encoding.md`](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md), Benchmark: [`.benchmarks/458_position_group_action_goat.md`](../../.benchmarks/458_position_group_action_goat.md), Paper: [arXiv:2512.07805](https://arxiv.org/abs/2512.07805)

## 22. GRAPE-AP Vector-Similarity Gates (Research 446)

Content-aware extension of Wall Attention's scalar prefix-sum gates (arXiv:2512.07805 §5). For each head `h` and decoding step `t`, the bias from key position `j` to query `t` is a path integral of edge potentials:

```
b_h(t, j) = Σ_{ℓ=j+1}^{t} ψ_h(t, ℓ)
ψ_h(t, ℓ) = α · g(⟨p_t, R_ℓ · p_ℓ⟩ / d)
```

with `g = log_sigmoid` (the paper's choice) and `R_ℓ = exp(ℓ·J)` a cached rotation schedule. Tokens whose positional embedding matches the query's decay slower. **Wall is the scalar special case** (endpoint-independent embeddings).

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G2** | Latency `< 1.5×` Wall's scalar path | within bound | ✅ PASS |
| **G4** | Alloc-free after scratch init | 0 steady-state allocs | ✅ PASS |
| **G5** | Direction-check per paper's 1/d normalization | verified | ✅ PASS |

🔧 Feature flag: `grape_ap_vector` (in `katgpt-core`) — **opt-in**. 15 unit tests in-crate.

📖 Research: [`.research/446_GRAPE_Group_Representational_Position_Encoding.md`](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md), Benchmark: [`.benchmarks/459_grape_ap_vector_goat.md`](../../.benchmarks/459_grape_ap_vector_goat.md), Paper: [arXiv:2512.07805](https://arxiv.org/abs/2512.07805)

## 23. GRAPE Joint Lift — GL(d+2) Block-Diagonal Composition (Research 446)

Composes rotary (GRAPE-M) + additive (GRAPE-A) into a single block-diagonal group action per Appendix E of arXiv:2512.07805. One-pass `score_into`:

```
score(q, k) = q^T · exp(m · ω_rot · L) · k / √d  +  m · ω_add · (softplus(v · q / √d) + softplus(u · k / √d))
```

Closes the GRAPE composition story: today Wall *replaces* RoPE in our stack; this primitive proves they *compose* into a single one-parameter subgroup of `GL(d+2)` while preserving the exact relative law. The decoupled `omega_rot` / `omega_add` is a strict generalization of the paper's shared `ω`.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Bit-identical to manual composition + relativity law | 0.0 diff | ✅ PASS |
| **G2** | Latency smoke | within bound | ✅ PASS |
| **G4** | Alloc-free after `new` | 0 steady-state allocs | ✅ PASS |

🔧 Feature flag: `grape_joint_lift` (in `katgpt-core`, implies `grapem_rodrigues`) — **opt-in**.

📖 Research: [`.research/446_GRAPE_Group_Representational_Position_Encoding.md`](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md), Benchmark: [`.benchmarks/460_grape_joint_lift_goat.md`](../../.benchmarks/460_grape_joint_lift_goat.md), Paper: [arXiv:2512.07805](https://arxiv.org/abs/2512.07805)

## 24. KARC Family — Delay-Basis Ridge Forecaster + Mitigations + Eigensolver (Plan 308 + Plan 556 + Issues 186/187)

The KARC family is the open-primitive surface for Kolmogorov-Arnold Reservoir Computing (Huang/Kurths/Tang 2026, arXiv:2606.19984). One core forecaster + one large-d_h eigensolver path (Issues 186/187) + three Plan 556 mitigations (regime gate, batched matvec, LOD tier) that address KARC's structural cons (periodic-blindness, crowd-scale per-NPC cost, tiered compute) without changing the core algorithm. All six features are opt-in; the core forecaster's promotion is blocked on the G1 threshold leg (10% short at K=8/M=8/R=2 d_h=18_720).

### 24.1 Core Forecaster — `karc_forecaster` (Plan 308)

`KarcForecaster<D, M, K>` × sealed `KarcBasis` trait (Fourier/Chebyshev/BSpline shipped) × closed-form ridge readout `Wout = YH^T(HH^T + λI)^{-1}`. Phase 2 ships higher-order R=2 (pair-product features, paper Eq. 32) + chunked Gram + ALS low-rank `Wout ≈ A·B` (the form that persists into a `KarcShard` in riir-neuron-db).

| Gate | Target | Result (Phase 5.1, 2026-07-20) | Verdict |
|------|--------|------------------------------|---------|
| **G1 NRMSE** | double-scroll Table I ≤ 1.0×10⁻³ (paper: 5.3×10⁻⁴) | **9.43e-4** (K=8/M=8/R=2 d_h=18_720, λ=5e-2 λ-sweep) | ✅ PASS |
| **G1 threshold** | ≥ 8 Lyapunov times | 2.85 LT (K=4) / **7.23 LT** (K=8/M=8/R=2, 10% short) / 8.16 LT (Phase 1 K=8/M=24 first-order) | ❌ FAIL |
| **G2** | ≤ 500 ns/call forecast (HLA config) | 381 ns | ✅ PASS |
| **G3** | zero-alloc `forecast_into` | 0 allocs | ✅ PASS |
| **G4** | bit-reproducibility | byte-identical `Wout` | ✅ PASS |

**Phase 5 history (load-bearing negative result):** Phase 4 *interpolated* (without measuring) that K=8/M=8/R=2 d_h=18_720 would be the smallest config to pass both G1 legs. Phase 5 measured it directly: BOTH legs FAILED at λ=5e-3 — NRMSE 6.68e-3 (6.7× miss) because the K=8 system is heavily underdetermined (N=4050 samples, d_h=18_720 features → ≥14_670 zero eigenvalues; λ=5e-3 tuned for K=4 is too small to regularize K=8). Phase 5.1 ran the λ-sweep and recovered NRMSE at λ=5e-2 (10× larger). The threshold leg remains flat across λ (~7.0–7.2 LT) — confirming it's a capacity/delay problem, not a regularization problem.

**Compute blocker resolved.** Before Issue 186 Path B, d_h=18_720 was projected infeasible (~6 h via Jacobi eigendecomp). Householder+QL + full-rank direct Cholesky brought it to ~29 min wall. Any future config sweep is now cheap to test.

**Promotion paths (all open, all cheap to test):**
1. **K=10/M=8/R=2 at λ=5e-2** (~28 min Cholesky). Linear K-extrapolation from K=4=2.85 LT, K=8=7.23 LT predicts ~8.5 LT — PASS.
2. **Issue 186 Path D gate re-spec.** Promote on two-config evidence (Phase 5.1 K=8 NRMSE 9.43e-4 + Phase 1 K=8/M=24 threshold 8.16 LT — same K=8 delay length).
3. **More training data** (N=20_000+) — would make the Gram full-rank. Compute cost scales linearly.

🔧 Feature flag: `karc_forecaster` — **opt-in**.

📖 Plan: [`.plans/308_karc_delay_basis_ridge_forecaster.md`](../../.plans/308_karc_delay_basis_ridge_forecaster.md), Research: [`.research/288_KARC_Delay_Basis_Ridge_Forecaster.md`](../../.research/288_KARC_Delay_Basis_Ridge_Forecaster.md), Benchmark: [`.benchmarks/308_karc_goat.md`](../../.benchmarks/308_karc_goat.md), Paper: [arXiv:2606.19984](https://arxiv.org/abs/2606.19984)

### 24.2 Large-d_h Eigensolver — `karc_householder_eig` + `karc_householder_eig_par` (Issues 186 + 187)

The ALS B-step's original eigendecomp (`karc::jacobi_eigen`, O(d_h³·n_sweeps)) is infeasible at d_h > ~5000 — blocking Plan 308's K=8/M=24/R=2 config (d_h=18_720). Issue 186 Path B swaps in `linalg::symmetric_eig` (Householder tridiag + implicit-shift QL), which is ~5-10× faster at d_h ≥ 256 and feasible at d_h=18_720. The eigensolver is always compiled as a generic `linalg` primitive; the feature gates only the wiring in `karc::large_dh`.

**Measured speedup** (single-threaded, release build):

| n | Householder+QL | Jacobi | Speedup |
|---|---|---|---|
| 64 | 310 µs | 2.5 ms | 7.92× |
| 128 | 3.4 ms | 36.6 ms | 10.62× |
| 256 | 73.5 ms | 687 ms | 9.35× |
| 512 | 794 ms | 10.9 s | 13.69× |

`karc_householder_eig_par` (Issue 187) adds a row-parallel rayon variant. Four row-parallel hot loops (Householder matvec, rank-2 update, Q accumulation, QL eigenvector rotation) parallelize across rows via `par_chunk_mut(n)`; each row's work is fully sequential so the result is bit-identical to the serial path. **Landed a critical QL convergence fix** for near-singular Grams (the NR-local check `|e[m]| + dd == dd` cannot deflate tiny-eigenvalue matrices; added the LAPACK `dsteqr` global-scale criterion — affects both serial and parallel paths).

**Why both stay opt-in despite T1-T6 PASS.** The Phase 5 G1 measurement at d_h=18_720 showed the full-rank direct Cholesky path is BOTH faster and more accurate than Householder+QL for the actual G1 measurement (direct Cholesky ~22 min vs parallel eigendecomp ~87 min; NRMSE 6.68e-3 vs ALS-rank-8 4.71e-3 — 28× worse). The parallel path landed a critical bug fix that ships regardless (it affects the serial Householder path too), but there is no passing G1 gate to promote against. The serial `karc_householder_eig` path stays opt-in for the same reason.

🔧 Feature flags: `karc_householder_eig` (implies `karc_forecaster`) — **opt-in**; `karc_householder_eig_par` (implies `karc_householder_eig`) — **opt-in**.

📖 Issue: [`.issues/187_karc_householder_eig_parallel.md`](../../.issues/187_karc_householder_eig_parallel.md), Benchmark: [`.benchmarks/308_karc_goat.md`](../../.benchmarks/308_karc_goat.md) §Phase 5.

### 24.3 Regime Gate — `karc_regime_gate` (Plan 556 Phase 1)

Closed-form residual-MSE mux between `KarcForecaster` (chaotic-regime specialist) and `SeasonalNaiveForecaster` (periodic-regime floor). Directly fixes KARC's structural periodic-blindness documented in [`.benchmarks/010_report_the_floor_consolidated.md`](../../.benchmarks/010_report_the_floor_consolidated.md) §T7 (K-sweep 2026-07-20 refuted the "K=4 too shallow" hypothesis: KARC's Chebyshev basis can't fit periodic data regardless of K).

Two `WelfordMse` accumulators + sigmoid confidence + cold-start floor. **Revised from variance-only to MSE (variance + bias²)** after Plan 514 surfaced the failure mode where a consistently-biased forecaster has variance 0 but large error. Implies `karc_forecaster` (the gate routes to KARC) + `conformal_predictive_intervals` (the floor the gate routes to in the periodic regime and during cold-start).

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | KARC ≥95% ticks on Lorenz-63; Seasonal ≥95% on period=12; mix ≤5% | PASS (Plan 514 Phase 1) | ✅ PASS |
| **G2** | `decide()` ≤ 50 ns/call | 37 ns median | ✅ PASS |
| **G3** | enabling gate does not perturb KARC forecasts | bit-identical (conformal_karc_no_regression.rs) | ✅ PASS |
| **G4** | 0 allocs/100 calls | 0 allocs | ✅ PASS |

**Runtime integration gain (riir-ai Plan 514 Phase 1):** G1 PASS — **92.45% MAE reduction** on mixed-regime NPC corpus (synthetic). G2 ~at-budget — 89 ns/tick. Pure modelless (two Welford accumulators + sigmoid).

**Why opt-in.** Primitive-level GOAT PASS + positive synthetic-corpus runtime gain. Stays opt-in pending a production-corpus gain measurement.

🔧 Feature flag: `karc_regime_gate` (implies `karc_forecaster` + `conformal_predictive_intervals`) — **opt-in**.

📖 Plan: [`.plans/556_karc_mitigations_open_primitives.md`](../../.plans/556_karc_mitigations_open_primitives.md), Benchmark: [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md), Companion runtime: [`riir-ai/.plans/514_karc_mitigations_runtime.md`](../../riir-ai/.plans/514_karc_mitigations_runtime.md).

### 24.4 Batched MatVec — `karc_batched_matvec` (Plan 556 Phase 2)

SIMD-batched forecast across N forecasters of identical (D, M, K) shape. Crowd-scale perf primitive: amortizes memory bandwidth by laying out N `Wout` matrices contiguously and hoisting the per-output-row `simd::simd_matvec` call across the batch. Ships `KarcBatchForecaster` + `karc_batched_matvec_into`.

**G2 partial PASS (architectural finding).** Pure-matvec amortizes well, but full-forecast amortization does NOT materialize because `feature_expand` dominates the per-forecast cost.

| N | pure_matvec | batched_forecast_full | sequential_baseline | matvec amortization | full amortization |
|---|---|---|---|---|---|
| 1 | 104 ns | 408 ns | 411 ns | 1.0× | 1.0× |
| 8 | **815 ns** | **3.42 µs** | **3.33 µs** | **4.0×** | **0.97×** |
| 32 | 3.77 µs | 13.6 µs | 14.8 µs | **7.0×** | 1.09× |

The original G2 target (5.3× full-forecast amortization at N=8) assumed the matvec was the dominant cost. Measurement showed it's only ~25% — `feature_expand` is ~75% (delay state → ψ basis expansion, per-NPC, not amortizable by the batched matvec).

**Architectural redirect (Plan 514 Phase 3).** The right consumer for this primitive is cell-shared-KARC + per-NPC latent_functor deviation — ONE feature_expand per cell, batched matvec across N NPC Wouts — not per-NPC-Wout batching. A future `feature_expand_batched` primitive could also close the gap.

🔧 Feature flag: `karc_batched_matvec` (implies `karc_forecaster`) — **opt-in**.

📖 Plan: [`.plans/556_karc_mitigations_open_primitives.md`](../../.plans/556_karc_mitigations_open_primitives.md), Benchmark: [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md), Companion runtime: [`riir-ai/.plans/514_karc_mitigations_runtime.md`](../../riir-ai/.plans/514_karc_mitigations_runtime.md).

### 24.5 LOD Tier — `karc_lod_tier` (Plan 556 Phase 3)

Config tag + tier-promotion Wout projection. Three nested tiers (LOD0 background D=8/M=4/K=2 d_h=64 / LOD1 midground D=8/M=8/K=4 d_h=256 / LOD2 hero D=8/M=8/K=8 d_h=512) map to different `KarcForecaster` const-generic monomorphizations. The nested-subset structure (LOD0 features are a strict prefix of LOD1; LOD1 of LOD2) makes tier promotion a pure index remap — down-tier preserves surviving Wout columns bit-identically; up-tier zero-fills new columns.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | tier promotion preserves surviving Wout columns bit-identically | bit-identical | ✅ PASS |
| **G2** | tier promotion ≤ 10 µs (one-time) | worst-case 831 ns (Lod1→Lod2 release) | ✅ PASS |
| **G3** | default (LOD1) path unchanged | bit-identical | ✅ PASS |
| **G4** | per-tick dispatch zero-alloc | 0 allocs/tick | ✅ PASS |

**Config revision (load-bearing).** Lod2 ships as (D=8, M=8, K=8, R=1) → d_h=512, NOT the plan's original (8, 8, 8, 2) → d_h=18_720. The plan's figure doesn't math out (8·8·8·2 = 1024, not 18_720). R=2 promotion-gate config (the real d_h=18_720 from Issue 185/186/187) deferred — pair-product features break the nested-subset invariant.

**Runtime integration — honest split verdict (riir-ai Plan 514 Phase 2).** The primitive itself is correct (G1-G4 all PASS). The runtime integration's G2 has a split verdict:

| Scale | Savings | Verdict |
|---|---|---|
| **1k NPCs (production scale)** | 14.7% (re-validated 2026-07-20), 5.3× headroom | ✅ PASS |
| **10k NPCs (crowd scale)** | 4.9% | ❌ FAIL |

**Root cause of the 10k FAIL.** 10k-NPC state (~20 MB) exceeds L3 cache, so memory bandwidth dominates and the compute savings vanish. The dormant-Lod1 memory overhead cancels Lod0's 4× compute advantage at crowd scale. LOD is a **per-node compute optimization, not a per-cluster one** — 10k+ NPC scale belongs in a **sharding layer** (across game-server nodes) that does NOT exist yet in this stack (tracked at `riir-ai/.issues/556_npc_sharding_for_crowd_scale.md`). Plan 514 Phase 3/4 G2 targets revised from "10k NPCs on a single node" to "1k NPCs per shard".

**Why opt-in.** Primitive-level GOAT PASS + 1k-scale runtime PASS. Stays opt-in until either a pure-enum redesign (breaks `forecaster()` API) or a positive gain on a smaller-scale corpus.

🔧 Feature flag: `karc_lod_tier` (implies `karc_forecaster`) — **opt-in**.

📖 Plan: [`.plans/556_karc_mitigations_open_primitives.md`](../../.plans/556_karc_mitigations_open_primitives.md), Benchmark: [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md), Runtime bench: [`riir-ai/.benchmarks/514_karc_lod_dispatch_goat.md`](../../riir-ai/.benchmarks/514_karc_lod_dispatch_goat.md), Sharding gap: [`riir-ai/.issues/556_npc_sharding_for_crowd_scale.md`](../../riir-ai/.issues/556_npc_sharding_for_crowd_scale.md).
