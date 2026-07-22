# Issue 189 — Variable-Rank Domain Expert: Monomorphization Escape Hatch (Path to Promotion)

**Filed:** 2026-07-22
**Priority:** P3 (promotion-unblocker for `variable_rank_domain_expert`; not on any critical path — feature works opt-in)
**Related:** Plan 558, Research 453, [Benchmark 558](../.benchmarks/558_variable_rank_domain_expert_goat.md), Plan 321 (`CommittedFieldBlend` substrate), Plan 230 (guided-projection mitigation)
**Origin:** Benchmark 558 §"monomorphization escape hatch" + Plan 558 §Honest Risks #1 — the documented promotion path after G2 FAILS at 2.0× baseline latency.

## TL;DR

Plan 558 shipped the `variable_rank_domain_expert` primitive behind an opt-in feature flag. The GOAT gate ran honestly: **G1/G3/G4/G5 PASS, G2 FAIL.** The entropy gain is real (2.63× over the uniform `CommittedFieldBlend<3,32>` baseline — exceeding the Research 453 PoC's 1.63×); the latency cost is also real (~2.0× per tick).

The G2 failure root cause is **trait-object dispatch overhead** — the `Box<dyn ErasedCluster>` vtable + 3× per-NPC `override_pi()` virtual calls cost ~50 ns on top of a 51 ns baseline. This is the cost of ergonomic heterogeneous const-generics (different `K_d`, `L_d` per domain can't share an array without type erasure).

The escape hatch is **macro-generated per-domain-count routers** that monomorphize the dispatch into a `match` over typed cluster fields — no `Box`, no vtable. This issue tracks implementing that macro and re-running G2. **If latency drops to ≤1.0× baseline, promote `variable_rank_domain_expert` to default-on.** If still >1.0×, the variable-rank pattern is fundamentally more expensive per tick (the domain gate + projection work can't be elided) and the feature stays opt-in forever — the entropy gain is the selling point, not the latency.

## Context — why the current approach fell short

### What shipped (Plan 558, commit `d5fe574a`)

`crates/katgpt-core/src/variable_rank_domain_expert.rs` ships:

- `pick_domain<const N, const A>` — deterministic argmax routing on `activity · domain_directions`.
- `project_guided<const D, const L>` / `scatter_guided` — zero-cost dimension gather (the Plan 230 mitigation: guided projection, not blind JL/PCA, which had failed by 200× the JL lower bound).
- `VariableRankRouter<const DOMAINS, const D_FULL, const A>` — heterogeneous-rank dispatch via `Box<dyn ErasedCluster>` + `ClusterHolder<K, L>` wrapper.
- `ErasedCluster::override_pi()` trait method — simulates per-NPC committed personalities.

14 lib unit tests + GOAT gate (G1/G2/G3/G5) + G4 alloc-free audit. All green except G2.

### The G2 failure

Release-mode latency at 1K + 10K NPCs (Apple Silicon):

| Scale | Baseline `<3,32>` | Variable-rank | Ratio | Target |
|---|---|---|---|---|
| 1K NPCs | 51.4 ns/NPC | 114.3 ns/NPC | **2.224×** | ≤ 1.0× |
| 10K NPCs | 52.1 ns/NPC | 103.8 ns/NPC | **1.990×** | ≤ 1.0× |

Latency scales linearly (1K → 10K) — the substrate is MMO-scalable. The absolute numbers (~52 µs/tick at 1K NPCs) are well under the 500 µs/tick MMO budget. But the router pays a ~2× overhead vs the monomorphized baseline.

### Root cause (matches Plan 558 §Honest Risks #1 + #3)

Per-tick work breakdown:

1. **3× `cluster_mut().override_pi()` virtual calls** — the bench simulates per-NPC committed pi (each NPC has its own personality). Each call is a trait-object virtual dispatch + a K-element copy. ~30 ns. The baseline just does 1 direct field assignment.
2. **1× `Box<dyn ErasedCluster>::apply_blended()` virtual dispatch** — the heterogeneous-rank clusters require type erasure. ~20 ns. The baseline calls `CommittedFieldBlend::apply_blended` directly (monomorphized, no vtable).
3. **Domain gate** (`pick_domain`) — 3 dot-products + argmax. ~10 ns. Legitimate.
4. **Projection + scatter** — `L` indexed loads + `L` indexed stores. ~5 ns each. Legitimate.

Items 3+4 (~15 ns) are the irreducible variable-rank overhead. Items 1+2 (~50 ns) are the trait-object tax, which dominates.

### Why the Plan 558 prediction was wrong

Plan 558 §Phase 3 T3.2 predicted release overhead would be ≤1.0× because (a) the trait-object dispatch is "one virtual call", (b) the smaller `CommittedFieldBlend<12,8>` does 96 multiply-adds (same as `<3,32>`'s 96), (c) `project_guided` is just `L` indexed loads.

The prediction missed two things, both honestly recorded in [Benchmark 558](../.benchmarks/558_variable_rank_domain_expert_goat.md) §"Why the Plan 558 prediction was wrong":

- The bench shape requires **per-NPC pi override** (3 virtual calls, not 0). In a production per-entity-router design, each NPC owns its own router with its own committed pi — no override needed per tick. But the bench can't instantiate 1000 routers cheaply, so it overrides pi per tick, paying the vtable cost 3× per NPC.
- The baseline is **extremely fast** (51 ns) — a tight monomorphized loop with no indirection. The 63 ns router overhead is small in absolute terms (~1 cache miss + 4 virtual calls) but 2× relative to a 51 ns baseline.

Research 453 PoC §4 finding #3 prediction ("release overhead negligible") was honestly updated as wrong in the research note.

## Goal

Implement a **macro-generated per-domain-count router** that eliminates the `Box<dyn ErasedCluster>` vtable dispatch, then re-run G2.

The macro should expand to a struct with N typed cluster fields (no `Box`) + a `match`-based dispatch (no vtable). Per-NPC pi override becomes direct field access. This trades code-size (one monomorphized router per domain-count instantiation) for dispatch cost (zero virtual calls) — the standard Rust pattern for heterogeneous const-generic dispatch (same shape as `bevy_ecs`'s `Bundle` macro).

### Sketch

```rust
// Instead of:
//   clusters: [Box<dyn ErasedCluster<A>>; DOMAINS]
// Generate:
variable_rank_router_3_domains!(Router3MoveCombatQuest,
    MoveCluster   = CommittedFieldBlend<12, 8>,
    CombatCluster = CommittedFieldBlend<6, 16>,
    QuestCluster  = CommittedFieldBlend<3, 32>
);
// Expands to a struct with 3 typed cluster fields + match-based dispatch
// (no vtable, no Box). Per-NPC pi override becomes direct field access.
```

The existing `pick_domain` / `project_guided` / `scatter_guided` primitives stay — they're already monomorphized and fast. Only the `VariableRankRouter` + `ErasedCluster`/`ClusterHolder` layer needs replacing.

## Acceptance Criteria

- [ ] **T1 — Macro design.** Decide between (a) a single macro that takes a domain count + cluster-type list, or (b) per-count specialized macros (`router_2_domains!`, `router_3_domains!`, ...). Document the trade-off (generality vs macro complexity). The minimum useful count is 3 (move/combat/quest — the Research 453 configuration); higher counts are nice-to-have.
- [ ] **T2 — Implementation.** Ship the macro in `crates/katgpt-core/src/variable_rank_domain_expert.rs` (or a sibling `macro.rs` module if line-count warrants). The macro-generated router MUST:
  - Expose the same public API as `VariableRankRouter` (route + apply + override_pi).
  - Preserve G1 correctness (10K inputs, no NaN) — port the existing G1 test.
  - Preserve G4 alloc-free (0 bytes / 1000 ticks) — port the existing alloc audit.
  - NOT break the existing `VariableRankRouter` (keep it as the ergonomic opt-in path for consumers who don't want to commit to a fixed domain count at compile time).
- [ ] **T3 — G2 re-gate (release).** Re-run the G2 bench at 1K + 10K NPCs with the monomorphized router. Capture the new ratio in [`.benchmarks/558_variable_rank_domain_expert_goat.md`](../.benchmarks/558_variable_rank_domain_expert_goat.md) §"Monomorphization re-gate".
- [ ] **T4 — Promotion decision.**
  - **If G2 ≤ 1.0× baseline:** promote `variable_rank_domain_expert` to the `default = [...]` list in `crates/katgpt-core/Cargo.toml`. Update Plan 558 + Research 453 + Benchmark 558 with the promotion verdict. Run `cargo check --all-features` + `--no-default-features` to confirm G3 no-regression. Close this issue.
  - **If G2 still > 1.0×:** the variable-rank pattern is fundamentally more expensive per tick. Document the verdict honestly. The feature stays opt-in forever — the 2.63× entropy gain (G3) is the selling point for diversity-prioritizing consumers, not latency. Close this issue with the negative result recorded.

## Out of scope

- **Octree spatial indexing integration** (Research 453 §2.4 "mixture of octree experts") — separate future plan; Plan 558 is the substrate primitive, not the spatial wiring.
- **Quest-grammar archetype construction** (Research 453 §2.5) — separate future plan; Plan 558 uses host-supplied archetype direction fields.
- **riir-ai runtime wiring** (per-NPC `DomainExpertBundle` resource) — separate future plan; Plan 558 ships the generic primitive, not the consumer integration.
- **Private selling-point guide** in `riir-ai/.research/` — only created if the feature promotes to GOAT/Super-GOAT. Currently opt-in, so NOT created.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Macro-generated router still >1.0× baseline | Medium | Low (feature stays opt-in, no regression) | The legitimate variable-rank work (domain gate + projection + scatter, ~15 ns) is irreducible. If the vtable elimination doesn't recover the full 50 ns, the floor is ~66 ns vs 51 ns baseline = 1.3×. Acceptable outcome: stays opt-in, honest negative result. |
| Macro complexity balloons (per-count specialization explosion) | Low | Medium (maintenance burden) | Cap at domain-count = 3 (move/combat/quest) for v1. Higher counts deferred until a concrete consumer needs them. |
| API divergence between `VariableRankRouter` (dynamic) + macro router (static) | Medium | Medium (consumer confusion) | Both expose the same trait surface; document the trade-off (dynamic = ergonomic, static = fast). |
| G1 regression from monomorphization (different codegen → different float rounding) | Low | High (correctness) | Port the existing G1 test bit-for-bit; the math is identical, only the dispatch path changes. |

## Dependencies

- Plan 558 (shipped, commit `d5fe574a`) — the substrate primitives stay as-is.
- Plan 321 (`CommittedFieldBlend<N, D>`) — the per-cluster substrate, unchanged.

## Re-evaluation triggers

Reopen or escalate this issue if:

- A concrete consumer (riir-ai runtime, riir-game-sdk, seal-online-remaster) requests default-on `variable_rank_domain_expert` and is willing to absorb the 2× latency — re-prioritize.
- A new modelless technique (e.g., const-generic specialization, `min_specialization`) makes the vtable elimination possible without a macro — re-evaluate the implementation path.
- The entropy gain (G3) becomes load-bearing for a downstream benchmark — re-prioritize.

## Related

- [Plan 558](../.plans/558_variable_rank_domain_expert_clusters.md) — the execution plan (all 21 tasks done; G2 FAIL honest).
- [Research 453](../.research/453_Variable_Rank_Domain_Expert_Clusters.md) — the design + PoC (prediction honestly updated post-Plan-558).
- [Benchmark 558](../.benchmarks/558_variable_rank_domain_expert_goat.md) — full GOAT results + root-cause analysis + escape hatch sketch.
- Plan 321 — `CommittedFieldBlend<N, D>` (the substrate).
- Plan 230 — the cautionary flag (blind JL fails; guided projection mitigates).

## TL;DR

`variable_rank_domain_expert` works (G1/G3/G4/G5 PASS, 2.63× entropy gain) but costs 2× per-tick latency (G2 FAIL) because of `Box<dyn ErasedCluster>` vtable dispatch. The monomorphization macro eliminates the vtable. Implement it, re-run G2, promote if ≤1.0× — otherwise the feature stays opt-in forever with the entropy gain as the selling point.
