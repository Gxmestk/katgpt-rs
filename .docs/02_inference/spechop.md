# katgpt-rs: SpecHop — Continuous Multi-Hop Speculation Architecture

> **Plan 131** · **Feature gate:** `spechop` (opt-in, requires `bandit`)
> **Reference:** arXiv:2605.21965 — Continuous speculation for multi-hop retrieval agents

## 1. Overview

SpecHop extends speculative execution from **token-level** to **hop-level** (tool-call granularity). Instead of predicting individual tokens, it predicts entire tool-call observations while the LLM continues reasoning ahead. When the target tool returns, a verifier checks equivalence → commit correct branches, rollback incorrect ones.

**Target:** 25–40% wall-clock latency reduction on multi-hop tool-use trajectories, lossless under verifier.

### Key Parameters

| Symbol | Name | Meaning |
|--------|------|---------|
| α | Relative speculator latency | `E[T_spec] / E[T_target]` — must be < 1.0 |
| β | Decode-to-tool ratio | `E[T_seg] / E[T_target]` |
| p | Speculator accuracy | `P(speculator prediction correct)` |
| k* | Optimal thread count | `⌈(1+β) / (α+β)⌉` |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SpecHopPipeline                              │
│                     (src/spechop/pipeline.rs)                       │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐    │
│  │   Config     │    │  Speculator │    │      Verifier        │    │
│  │ (α, β, p, k)│    │     (S)     │    │  ObservationVerifier │    │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬───────────┘    │
│         │                  │                      │                 │
│         ▼                  ▼                      ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SpecWindow (FIFO, k threads)              │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐  │   │
│  │  │ HopObserv│  │ HopObserv│  │ HopObserv│ ... │ HopObserv│  │   │
│  │  │  #0 ✓   │  │  #1 …   │  │  #2 …   │       │  #k-1 … │  │   │
│  │  │Committed│  │Pending  │  │Pending  │       │Pending  │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘       └─────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Loop (Algorithm 1)

```
    ┌──────────────┐
    │   START      │
    │ Trajectory   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐     capacity < k
    │  Next Hop?   │──────────────────┐
    │              │                  │
    └──────┬───────┘                  │
           │ has hop                  │
           ▼                          ▼
    ┌──────────────┐          ┌──────────────┐
    │ Speculator   │          │  Wait for    │
    │ .speculate() │          │  target tool │
    │ o_spec for   │          │  to return   │
    │ next action  │          └──────┬───────┘
    └──────┬───────┘                 │
           │                         │
           ▼                         │
    ┌──────────────┐                 │
    │ Push to      │                 │
    │ SpecWindow   │                 │
    └──────┬───────┘                 │
           │                         │
           ▼                         │
    ┌──────────────┐                 │
    │ Target tool  │◄────────────────┘
    │ returned?    │
    └──────┬───────┘
           │ yes
           ▼
    ┌──────────────┐
    │ Verifier     │
    │ .verify()    │
    │ o_target vs  │
    │ o_spec       │
    └──┬───────┬───┘
       │       │
  match│       │mismatch
       ▼       ▼
  ┌────────┐ ┌──────────┐
  │ COMMIT │ │ ROLLBACK │
  │ branch │ │ + retry  │
  └───┬────┘ └────┬─────┘
      │           │
      │     commit real
      │     observation
      │           │
      ▼           ▼
  ┌──────────────┐     ┌──────────────┐
  │ Early stop?  │─no─►│  Continue    │──► Next Hop?
  │ (final ans)  │     │  pipeline    │
  └──────┬───────┘     └──────────────┘
         │ yes
         ▼
    ┌──────────┐
    │  DONE    │
    │Pipeline  │
    │Result    │
    └──────────┘
```

---

## 3. Module Structure

```
src/spechop/
├── mod.rs              Module index, re-exports, feature gate
├── types.rs            SpecHopConfig, HopObservation, SpecOutcome, HopState, SpecError
├── cost_model.rs       α/β/p → k* computation, RelLat formulas (Theorems 2–4)
├── verifier.rs         ObservationVerifier trait + RuleBasedVerifier
├── speculator.rs       HopSpeculator trait + CacheSpeculator + BanditSpeculator
├── window.rs           SpecWindow thread pool manager (FIFO commit/rollback)
├── pipeline.rs         SpecHopPipeline continuous loop (Algorithm 1)
├── hop_tree.rs         Hop-level DDTree integration (Phase 6)
└── segment_match.rs    Rolling-hash segment index for hop observations (requires spechop + cache_prune)
```

---

## 4. Key Types & Relationships

```
                    ┌──────────────────┐
                    │  SpecHopConfig   │
                    │  α, β, p, k, ν  │
                    └────────┬─────────┘
                             │ configures
                             ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│HopSpeculator │◄───│ SpecHopPipeline  │───►│ ObservationVerifier  │
│  trait       │    │                  │    │      trait           │
├──────────────┤    ├──────────────────┤    ├──────────────────────┤
│CacheSpeculator│   │ SpecWindow       │    │ RuleBasedVerifier    │
│BanditSpeculator│  │  (FIFO, k cap)  │    │  - exact match       │
└──────────────┘    │                  │    │  - refusal detect    │
                    │  PipelineResult  │    │  - numeric check     │
                    │  - hits/misses   │    │  - Jaccard ≥ 0.55    │
                    │  - accuracy      │    │  - substring match   │
                    │  - early_stop    │    └──────────────────────┘
                    └──────────────────┘
                             │
                             │ feeds
                             ▼
                    ┌──────────────────────┐
                    │   Hop-level DDTree   │
                    ├──────────────────────┤
                    │ HopVerifyState       │
                    │  Pending|Committed   │
                    │  |RolledBack         │
                    │ HopTreeNode          │
                    │  - score (cum.)      │
                    │  - depth             │
                    │  - action            │
                    │  - observation       │
                    │  - parent_idx        │
                    │  - verified          │
                    │ HopCandidate         │
                    │  - observation       │
                    │  - confidence        │
                    │ HopMarginal          │
                    │  - action            │
                    │  - candidates[]      │
                    │ HopTreeConfig        │
                    │  - tree_budget       │
                    │  - confidence_floor  │
                    │  - chain_seed        │
                    │ VerifiedHopPath      │
                    │  - path[]            │
                    │  - commits/rollbacks │
                    │  - direct_commits    │
                    ├──────────────────────┤
                    │ build_hop_dd_tree    │
                    │ verify_hop_tree      │
                    │ extract_best_hop_path│
                    │ extract_deepest_hop_ │
                    │  path                │
                    │ build_and_verify_hop │
                    │  _tree               │
                    └──────────────────────┘
```

---

## 5. State Machine

Each hop transitions through these states:

```
  ┌───────────────┐
  │ AwaitingTarget│ ◄── initial state: action sent, waiting for tool
  └───────┬───────┘
          │ speculator predicts o_spec
          ▼
  ┌───────────────┐
  │  Speculating  │ ◄── prediction made, LLM continues ahead
  └───┬───────┬───┘
      │       │
 verify│       │verify
 match │       │mismatch
      ▼       ▼
┌──────────┐ ┌──────────┐
│ Committed│ │RolledBack│
│          │ │          │
│ o_spec   │ │ discard  │
│ matches  │ │ o_spec,  │
│ target   │ │ use real │
└──────────┘ └──────────┘
```

---

## 6. Cost Model (Theorems 2–4)

### Thread Sizing

```
k* = ⌈(1 + β) / (α + β)⌉

Examples:
  α=0.2, β=0.15 → k*=⌈1.15/0.35⌉=4  (cheap speculator, short decode)
  α=0.3, β=0.75 → k*=⌈1.75/1.05⌉=2  (moderate speculator, long decode)
```

### Latency Bounds

```
Oracle:     RelLat* = 1 − p(1−α)/(1+β)
Bounded:    RelLat_k = RelLat* + (1−α)/(1+β) × (1−p)^(k−1)
Starvation: P_starve ≈ Φ((1+β − k(α+β)) / (ν√(kα² + (k−1)β² + 1)))

Where:
  RelLat = 1.0 means no speedup (sequential baseline)
  RelLat < 1.0 means speedup (lower = faster)
  As k→∞, RelLat_k → RelLat* (oracle bound)
  As p→1.0, RelLat_k → RelLat* (perfect speculator)
```

### Activation Criteria (SR²AM Integration)

```
SpecHop activates when ALL of:
  1. observations ≥ 10  (enough data to estimate parameters)
  2. α < 0.3           (speculator is fast enough)
  3. β ≤ 0.8           (not decode-bound)
  4. reward > 1.0       where reward = latency_reduction / α

SpecHop SKIPS when β > 0.8 (decode-bound, speculation won't help)
SpecHop SKIPS when α ≥ 0.3 (speculator too slow relative to target tool)
```

---

## 7. Integration with Existing Systems

### SR²AM Configurator (Plan 112)

```
┌────────────────────┐      ┌───────────────────┐
│ ConfiguratorBandit │      │ InferenceStats    │
│                    │      │  - avg_spec_      │
│  Arms:            │      │    latency_ns     │
│  0: Baseline      │      │  - avg_target_    │
│  1: Speculative   │      │    latency_ns     │
│  2: MTP           │      │  - avg_decode_    │
│  3: SpecHop       │      │    latency_ns     │
│                    │      │  - avg_hit_rate   │
│                    │      │  - observations   │
│                    │◄─────┤  auto_k()         │
│                    │      └─────────┬─────────┘
└────────────────────┘                │
                                      ▼
                           ┌───────────────────┐
                           │ PlanningDecision   │
                           │ ::SpecHop { k }    │
                           └───────────────────┘
```

### DDTree Comparison

```
┌─────────────────┬─────────────────────────┬──────────────────────────────┐
│ Aspect          │ Token-level DDTree      │ Hop-level DDTree (SpecHop)   │
├─────────────────┼─────────────────────────┼──────────────────────────────┤
│ Node payload    │ token_idx: usize        │ action + observation: String │
│ Score source    │ ln(P_llm) marginals     │ ln(confidence) from speculat │
│ Parent tracking │ parent_path: u128       │ parent_idx: Option<usize>    │
│ Verification    │ Exact logit match       │ ObservationVerifier (fuzzy)  │
│ Granularity     │ Single token            │ Entire tool-call hop         │
│ Module          │ src/speculative/        │ crates/katgpt-speculative/src/spechop/hop_tree.rs      │
└─────────────────┴─────────────────────────┴──────────────────────────────┘
```

### Speculator Implementations

```
┌────────────────────┬──────────────────────────────────────┐
│ CacheSpeculator    │ HashMap<action, observation> lookup   │
│                    │ Cache hit rate = effective p          │
│                    │ Feature: always available             │
├────────────────────┼──────────────────────────────────────┤
│ BanditSpeculator   │ Uses ScreeningPruner relevance score  │
│ (requires bandit)  │ Modelless→model-based bridge         │
│                    │ Feature: requires "bandit"            │
└────────────────────┴──────────────────────────────────────┘
```

---

## 8. Verification Pipeline (Appendix D.4)

The `RuleBasedVerifier` checks observations in order of increasing cost, with early exit on pass:

```
  ┌─────────────────┐
  │ Normalize text  │
  │ (lowercase,     │
  │  trim)          │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐──pass──► COMMIT
  │ Exact match?    │
  └────────┬────────┘
           │ fail
           ▼
  ┌─────────────────┐──pass──► COMMIT
  │ Short answer    │ (< 10 chars)
  │ exact match?    │
  └────────┬────────┘
           │ fail
           ▼
  ┌─────────────────┐
  │ Refusal pattern │──both refused──► COMMIT
  │ check           │──one refused───► ROLLBACK
  └────────┬────────┘
           │ neither refused
           ▼
  ┌─────────────────┐──pass──► COMMIT
  │ Numeric         │ (same number sets)
  │ consistency?    │
  └────────┬────────┘
           │ fail
           ▼
  ┌─────────────────┐──pass──► COMMIT
  │ Substring       │
  │ containment?    │
  └────────┬────────┘
           │ fail
           ▼
  ┌─────────────────┐──≥ 0.55──► COMMIT
  │ Token-set       │ (stopwords removed)
  │ Jaccard sim?    │
  └────────┬────────┘
           │ < 0.55
           ▼
         ROLLBACK
```

---

## 9. Feature Gate

```toml
# Cargo.toml
[features]
spechop = ["bandit"]  # Continuous multi-hop speculation pipeline (Plan 131)
```

```rust
// lib.rs
#[cfg(feature = "spechop")]
pub mod spechop;

// spechop/mod.rs — segment_match is gated on both features
#[cfg(all(feature = "spechop", feature = "cache_prune"))]
pub mod segment_match;
```

**Not in default features** until GOAT 6/6 proved (T33–T38).

### Compatibility Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| `bandit` | ✅ Required | BanditPruner feeds into speculator decisions |
| `cache_prune` | ✅ Compatible | `segment_match` requires both `spechop` + `cache_prune` |
| `bt_rank` | ✅ Compatible | Bradley-Terry ranking for branch selection |
| `spectral_quant` | ✅ Compatible | KV cache compression orthogonal |
| `dash_attn` | ✅ Compatible | Sparse attention + hop speculation complementary |
| `rt_turbo` | ✅ Compatible | Retrieval heads can serve as hop speculators |
| `sr2am_configurator` | ✅ Compatible | Configurator decides k (thread count) |
| `data_gate` | ✅ Compatible | Data gating for training, spechop for inference |
| `lt2_looped` | ⚠️ Needs test | Looped inference may interact with hop-level speculation |
| `dllm` / `dmax_spd` | ⚠️ Needs test | Diffusion speculation + hop speculation may conflict |
| `game_state` | ✅ Compatible | Game forward model as "target tool" for hop speculation |

---

## 10. Examples

| Example | Location | Demonstrates |
|---------|----------|-------------|
| `spechop_01_pipeline` | `examples/spechop_01_pipeline.rs` | 4-hop continuous speculation with cache speculator, commit/rollback, DDTree integration |
| `spechop_02_cost_model` | `examples/spechop_02_cost_model.rs` | α/β/p → k* computation, RelLat prediction, configurator reward, auto-k from measured stats |

---

## 11. References

- **SpecHop paper:** arXiv:2605.21965
- **Speculative Actions (predecessor):** arXiv:2510.04371
- **Speculative Decoding (Leviathan et al.):** arXiv:2302.01318
- **Token-level DDTree:** `speculative_decoding.md`
- **SR²AM Configurator:** Plan 112, Research 076
- **Plan 131:** `.plans/131_spechop_continuous_spec_pipeline.md`
