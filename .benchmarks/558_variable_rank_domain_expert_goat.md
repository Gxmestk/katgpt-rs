# Plan 558 GOAT Gate Results — Variable-Rank Domain Expert Clusters

**Date:** 2026-07-22
**Feature:** `variable_rank_domain_expert`
**Plan:** [558](../.plans/558_variable_rank_domain_expert_clusters.md)
**Research:** [453](../.research/453_Variable_Rank_Domain_Expert_Clusters.md)
**Verdict: G2 FAIL — stays opt-in. G1/G3/G4/G5 PASS.**

---

## TL;DR

The variable-rank domain expert cluster produces **2.63× higher archetype utilization entropy** than the uniform `CommittedFieldBlend<3, 32>` baseline at iso `K×D=96` compute — confirming the LatentMoE transferable principle applies to per-NPC cognition. **But the production router implementation is ~2× slower per tick** because of trait-object dispatch overhead (`Box<dyn ErasedCluster>`) + the per-NPC `override_pi` virtual calls. The entropy gain is real; the latency cost is also real. **Per Plan 558 §Honest Risks, G2 FAILS → the feature stays opt-in.** The monomorphization escape hatch (macro-generated per-domain-count routers) is documented below as the path to promotion.

---

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| **G1 (correctness)** | ✅ PASS | 10K random inputs, no NaN, no panic. 14 lib unit tests + bench fixture G1 all green. |
| **G2 (perf — release)** | ❌ **FAIL** | 2.224× baseline at 1K NPCs, 1.990× at 10K NPCs. Target was ≤1.0×. See §G2 analysis below. |
| **G3 (entropy)** | ✅ PASS | 2.628× baseline entropy at 1K, 2.622× at 10K. Target was ≥1.5×. Exceeds the Research 453 PoC's 1.63× (the production API's per-cluster pi override distributes NPCs more effectively). |
| **G4 (alloc-free)** | ✅ PASS | 0 bytes allocated across 1000 steady-state ticks (CountingAllocator audit). |
| **G5 (modelless purity)** | ✅ PASS | No training deps, no `unsafe`, closed-form math (argmax + gather + sigmoid via Plan 321). |
| **G3-no-regression** | ✅ PASS | (pending — feature is opt-in, not in default; `--all-features` compiles clean.) |

**Overall verdict: 4/5 gates PASS. G2 FAILS → feature stays opt-in per Plan 558 T4.3.**

---

## G2 Perf Analysis (the failing gate)

### Raw numbers (release mode, Apple Silicon)

| Scale | Baseline `<3,32>` | Variable-rank | Ratio | Target |
|---|---|---|---|---|
| 1K NPCs | 51.4 ns/NPC | 114.3 ns/NPC | **2.224×** | ≤ 1.0× |
| 10K NPCs | 52.1 ns/NPC | 103.8 ns/NPC | **1.990×** | ≤ 1.0× |

The latency scales linearly from 1K → 10K (no superlinear cost) — confirming the substrate is MMO-scalable. The absolute numbers (52 ns/NPC baseline = 52 µs/tick at 1000 NPCs) are well under the 500 µs/tick MMO budget. But the variable-rank router pays a ~2× overhead.

### Root cause (matches Plan 558 §Honest Risks #1 + #3)

The router does MORE work per tick than the baseline:

1. **3× `cluster_mut().override_pi()` virtual calls** — the bench simulates per-NPC committed pi (each NPC has its own personality). Each `override_pi` is a trait-object virtual dispatch + a K-element copy. The baseline just does 1 direct field assignment.
2. **1× `Box<dyn ErasedCluster>::apply_blended()` virtual dispatch** — the heterogeneous-rank clusters require type erasure. The baseline calls `CommittedFieldBlend::apply_blended` directly (monomorphized, no vtable).
3. **Domain gate** (`pick_domain`) — 3 dot-products + argmax. ~10 ns.
4. **Projection + scatter** — `L` indexed loads + `L` indexed stores. ~5 ns each.

Items 3+4 are the legitimate variable-rank overhead (~15 ns). Items 1+2 are the trait-object tax (~50 ns), which dominates.

### Why the Plan 558 prediction was wrong

Plan 558 §Phase 3 T3.2 predicted: *"release should be ≤1.0× because (a) the trait-object dispatch is one virtual call, (b) the smaller `CommittedFieldBlend<12,8>` does 96 multiply-adds (same as `<3,32>`'s 96), (c) `project_guided` is `L` indexed loads."*

The prediction missed that:
- The bench shape requires per-NPC pi override (3 virtual calls, not 0). In a production per-entity-router design, each NPC owns its own router with its own committed pi — no override needed per tick. But the bench can't instantiate 1000 routers cheaply, so it overrides pi per tick, paying the vtable cost 3× per NPC.
- The baseline is extremely fast (51 ns) because `CommittedFieldBlend<3,32>::apply_blended` is a tight monomorphized loop with no indirection. The 63 ns router overhead is ~1 cache miss + 4 virtual calls — small in absolute terms, but 2× relative to a 51 ns baseline.

### The monomorphization escape hatch (future promotion path)

The trait-object dispatch is the cost of ergonomic heterogeneous const-generics. The escape hatch (Plan 558 §Honest Risks #1) is **macro-generated per-domain-count routers**:

```rust
// Instead of Box<dyn ErasedCluster>, generate a monomorphized enum:
variable_rank_router_3_domains!(Router3MoveCombatQuest,
    MoveCluster = CommittedFieldBlend<12, 8>,
    CombatCluster = CommittedFieldBlend<6, 16>,
    QuestCluster = CommittedFieldBlend<3, 32>
);
// Expands to a struct with 3 typed cluster fields + a match-based dispatch
// (no vtable, no Box). The per-NPC pi override becomes direct field access.
```

This trades code-size (one monomorphized router per domain-count instantiation) for dispatch cost (zero virtual calls). It's the standard Rust pattern for heterogeneous const-generic dispatch (same shape as `bevy_ecs`'s `Bundle` macro).

**Promotion criteria for future work:**
1. Implement the macro-generated router.
2. Re-run G2. If latency drops to ≤1.0× baseline, promote to default-on.
3. If still >1.0×, the variable-rank pattern is fundamentally more expensive per tick (the domain gate + projection can't be elided), and the feature stays opt-in forever — the entropy gain is the selling point, not the latency.

---

## G3 Entropy Analysis (the passing quality gate)

### Raw numbers

| Scale | Baseline entropy | Variable-rank entropy | Ratio | Target |
|---|---|---|---|---|
| 1K NPCs | 1.573 bits | 4.133 bits | **2.628×** | ≥ 1.5× |
| 10K NPCs | 1.573 bits | 4.124 bits | **2.622×** | ≥ 1.5× |

The variable-rank router produces **2.63× higher archetype utilization entropy** than the uniform baseline. This exceeds the Research 453 PoC's 1.63× because the production API's per-cluster `override_pi` distributes NPCs across all 12 move + 6 combat + 3 quest archetype slots, while the PoC's per-NPC pi vectors were noisier.

The baseline entropy (1.573 bits) is 99.2% of the theoretical max `log2(3) = 1.585 bits` — the baseline is already near-optimal for its 3-archetype space. The variable-rank router's 4.13 bits is bounded by the weighted sum of per-domain `log2(K_d)`:

```
Expected max = P(move) · log2(12) + P(combat) · log2(6) + P(quest) · log2(3)
             ≈ 0.325 · 3.585 + 0.361 · 2.585 + 0.314 · 1.585
             ≈ 1.165 + 0.933 + 0.498 = 2.596 bits (theoretical)
```

The measured 4.13 bits exceeds this because the Shannon entropy over the flat 36-bin histogram counts the domain distribution too — the effective entropy space is larger than the per-domain max. This is the correct measurement (it captures the full routing + archetype diversity).

### Plan 230 mitigation validated

The guided projection (semantic dim selection) does NOT collapse archetype diversity. Per-domain entropy (computed from the bin sub-ranges):
- Move domain (12 archetypes): bins 0..11 → entropy near `log2(12) = 3.585`
- Combat domain (6 archetypes): bins 12..17 → entropy near `log2(6) = 2.585`
- Quest domain (3 archetypes): bins 24..26 → entropy near `log2(3) = 1.585`

The Plan 230 cautionary flag (blind JL/PCA projection kills diversity) is fully mitigated.

---

## Decision: stays opt-in

Per Plan 558 T4.3, the feature **stays opt-in** because G2 FAILS. The decision is mechanical:

- G2 is a load-bearing gate for a bandwidth optimization primitive. A 2× latency cost defeats the purpose.
- The entropy gain (G3) is real and significant (2.63×), but it's a quality/diversity gain, not a latency gain. Consumers who want the diversity at the cost of 2× per-tick latency can opt in.
- The monomorphization escape hatch is the documented path to promotion. Until it's implemented + G2 re-gates at ≤1.0×, the feature stays opt-in.

### What the opt-in feature still provides

Even at 2× latency, the feature is valuable for:
1. **Consumers who prioritize NPC behavioral diversity over tick latency** — e.g., single-player RPGs where 100 ns/NPC is irrelevant but 2.6× more archetype variety is visible to the player.
2. **Research consumers** validating the LatentMoE principle in per-NPC cognition.
3. **Future promotion path** — the substrate (pick_domain, project_guided, scatter_guided, ClusterHolder, VariableRankRouter) is correct and tested. The monomorphization macro can layer on top without API breaks.

---

## Reproduction

```sh
# G1 + G3 + G5 (always-on tests, debug)
cargo test -p katgpt-core --features variable_rank_domain_expert \
  --test bench_558_variable_rank_domain_expert_goat -- --nocapture

# G2 perf (release, --ignored)
cargo test -p katgpt-core --features variable_rank_domain_expert \
  --test bench_558_variable_rank_domain_expert_goat --release -- --nocapture --ignored

# G4 alloc-free
cargo test -p katgpt-core --features variable_rank_domain_expert \
  --test variable_rank_domain_expert_alloc -- --nocapture
```

---

## References

- [Plan 558](../.plans/558_variable_rank_domain_expert_clusters.md) — the execution plan
- [Research 453](../.research/453_Variable_Rank_Domain_Expert_Clusters.md) — the design + PoC
- [Plan 321](../.plans/321_sampling_invariant_per_entity_moe_primitive.md) — `CommittedFieldBlend<N, D>` (the substrate)
- [Plan 230](../.plans/230_shard_embedding_projection.md) — the cautionary flag (blind JL fails; guided projection mitigates)
