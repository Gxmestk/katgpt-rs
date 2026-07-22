# Variable-Rank Domain Expert — Monomorphization Escape Hatch (T1 Macro Design)

**Date:** 2026-07-22
**Issue:** [189](../../.issues/189_variable_rank_domain_expert_monomorphization_escape_hatch.md)
**Plan:** [558](../../.plans/558_variable_rank_domain_expert_clusters.md)
**Benchmark:** [558](../../.benchmarks/558_variable_rank_domain_expert_goat.md)
**Status:** T1 DECISION — Option B (single generic `macro_rules!` with explicit indices) recommended. **T2 IMPLEMENTED** — macro shipped, G1+G4 gates pass. **T3-T4 DONE** — G2 re-gate ran, G2 still FAILS (~1.7× macro shared, ~1.95× production-shape). Feature stays opt-in forever.

## TL;DR

**Recommendation: Option B — a single generic `macro_rules!` macro that takes an explicit `(index, name, cluster_type, projection_indices)` list per domain.** No per-count specialization, no proc macro, no TT-munching. Works on stable Rust, matches the codebase's existing `macro_rules!` style, and eliminates all 4 vtable dispatches per tick (3× `override_pi` + 1× `apply_blended`) by generating typed struct fields + a `match`-based dispatch.

The explicit-index cost (user must keep `0..N` contiguous) is trivial for the domain counts in play (max ~5; Research 453 uses 3) and debug_assertable. The macro router keeps the `VariableRankRouter` API shape (route + apply + override_pi) so the existing GOAT bench runs unmodified.

## Problem statement

Plan 558's `VariableRankRouter<DOMAINS, D_FULL, A>` stores domains as `[Box<dyn ErasedCluster>; DOMAINS]`. The bench hot path pays 4 virtual calls per NPC per tick:

| # | Call | Cost | Why virtual |
|---|---|---|---|
| 1 | `router.cluster_mut(0).override_pi(pi_move)` | ~10 ns | `&mut dyn ErasedCluster` |
| 2 | `router.cluster_mut(1).override_pi(pi_combat)` | ~10 ns | same |
| 3 | `router.cluster_mut(2).override_pi(pi_quest)` | ~10 ns | same |
| 4 | `cluster.apply_blended(...)` inside `tick()` | ~20 ns | `&dyn ErasedCluster` |

Total vtable tax: ~50 ns on top of a 51 ns baseline → **2.0×** (G2 FAIL).

The baseline pays ZERO vtable cost — it owns `CommittedFieldBlend<3, 32>` directly (monomorphic) and writes `self.blend.pi = *pi_override` (direct field access). The macro router must reach the same zero-vtable shape.

## Options considered

### Option A — Per-count specialized macros

```rust
variable_rank_router_2_domains!(Router2, Move, Combat);
variable_rank_router_3_domains!(Router3, Move, Combat, Quest);
variable_rank_router_4_domains!(Router4, Move, Combat, Quest, Social);
```

Each macro is a separate `macro_rules!` hardcoded for that count.

**Pros:**
- Each macro is simple to write + read (no repetition logic).
- The `match` arms are hand-written per macro — no index-generation problem.
- Zero TT-munching, zero recursion.

**Cons:**
- **N macros for N domain counts** — combinatorial growth. Need one per supported count.
- Code duplication across macros (the struct/impl body is 90% identical, only the field count + match arms differ).
- Adding domain count 5 requires writing a 5th macro — maintenance burden scales linearly with supported counts.
- Doesn't match the "generic over count" spirit of the existing `VariableRankRouter<const DOMAINS>`.

### Option B — Single generic `macro_rules!` with explicit indices *(RECOMMENDED)*

```rust
variable_rank_router_static! {
    /// Per-NPC cognition router: move (K=12, L=8) + combat (K=6, L=16) + quest (K=3, L=32).
    pub struct Router3MoveCombatQuest<3, 32, 3>;

    domain_directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    0 => move:   ClusterHolder<12, 8>  => [0, 1, 2, 3, 4, 5, 6, 7];
    1 => combat: ClusterHolder<6, 16>  => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    2 => quest:  ClusterHolder<3, 32>  => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
}
```

The user supplies the literal domain index (`0 =>`, `1 =>`, `2 =>`). The macro generates:

1. A struct with typed `ClusterHolder<K_d, L_d>` fields (no `Box<dyn>`).
2. A `new()` that accepts the clusters + validates indices.
3. A `tick()` that `match`es on the domain index and calls the typed cluster directly (no vtable).
4. An `override_cluster_pi(domain, pi)` that `match`es on the domain index and writes `self.$field.blend.pi[..K_d]` directly (no vtable).

**Pros:**
- **One macro, any count.** No per-count duplication.
- Works on stable Rust — no TT-munching, no recursion. The explicit index is the trick: `macro_rules!` repetition `$( $idx:literal => $name:ident : $ty:ty => $indices:tt );*` generates `match` arms trivially.
- Matches the codebase's existing `macro_rules!` style (serde_via_u32, gate_check, counting_allocator, collect_proj).
- The explicit index (`0 =>`, `1 =>`, ...) is not a real degree of freedom — it's `0..N` bookkeeping. A `debug_assert!` in `new()` verifies contiguity.
- Eliminates ALL 4 vtable calls (3 override_pi + 1 apply_blended) — the full G2 win.

**Cons:**
- User must type the index explicitly. Trivial cost; `debug_assert` catches mistakes.
- The macro body is denser than Option A (repetition logic), but still ~60 lines of `macro_rules!`.
- The projection indices array (`[0,1,2,3,4,5,6,7]`) is verbose. Acceptable — these are compile-time-known semantic dimension selections (the whole point of guided projection).

### Option C — TT-munching counter (auto-assigned indices)

```rust
variable_rank_router_static! {
    pub struct Router3<3, 32, 3>;
    domain_directions: [...];
    move:   ClusterHolder<12, 8>  => [...];
    combat: ClusterHolder<6, 16>  => [...];
    quest:  ClusterHolder<3, 32>  => [...];
}
```

The macro uses internal TT-munching rules to auto-assign indices 0, 1, 2...

**Pros:**
- Cleanest user-facing syntax (no explicit indices).

**Cons:**
- **TT-munching is notoriously hard to read/maintain.** The recursion rules (`@count` accumulator pattern) are opaque to anyone who didn't write them.
- Recursion-depth limits (~128 levels on stable) — not a concern at ≤5 domains, but a latent footgun.
- **Diminishing returns:** the only benefit over Option B is not typing `0 =>`. The index is `0..N` and is `debug_assert`-checked. Not worth the macro complexity.

### Option D — Proc macro

Cleanest syntax (can parse arbitrary structure), but:
- Adds a `proc-macro` crate dependency for a single macro.
- Overkill — the problem doesn't need AST-level parsing.
- Violates the codebase's `macro_rules!`-only convention (zero proc macros in `katgpt-core` today).

**Rejected.**

### Option E — Array of function pointers / closures

Store `[fn(&Self, ...) -> usize; DOMAINS]` and call `dispatch[domain](self, ...)`. This reintroduces indirect dispatch (function pointer call ≈ vtable cost). **Defeats the purpose.** Rejected.

### Option F — `min_specialization`

```rust
impl RouterTrait<Domain<0>> for Router3 { ... }
impl RouterTrait<Domain<1>> for Router3 { ... }
```

Requires `min_specialization` or full specialization — **unstable**, not viable on stable Rust. Rejected.

## Why Option B wins

| Criterion | A (per-count) | B (generic+explicit idx) | C (TT-munch) | D (proc-macro) |
|---|---|---|---|---|
| Stable Rust | ✅ | ✅ | ✅ | ✅ |
| One macro, any count | ❌ (N macros) | ✅ | ✅ | ✅ |
| Readable macro body | ✅ (simple) | ✅ (repetition) | ❌ (opaque) | ✅ |
| Matches codebase convention | ✅ | ✅ | ✅ | ❌ (no proc-macros) |
| Eliminates all 4 vtables | ✅ | ✅ | ✅ | ✅ |
| User effort (low) | medium | **low** | lowest | lowest |
| Maintenance burden (low) | high (N macros) | **low** | medium | medium |

Option B is the Pareto-optimal choice: lowest combined user + maintenance effort, single macro, stable Rust, matches codebase conventions.

## The explicit-index contract

The macro takes `$idx:literal => $name:ident : $ty:ty => $indices:tt` per domain. The `new()` constructor `debug_assert`s:

1. Indices are `0..N` contiguous (the first domain is `0`, each subsequent is `prev + 1`).
2. Each domain's `projection_indices` length matches the cluster's `L`.
3. Each domain's indices are `< D_FULL` and strictly ascending.

These are the same checks the existing `VariableRankRouter::new()` already does. Release builds skip them (zero cost); debug builds catch mistakes at construction time.

## Generated code shape (the actual perf target)

```rust
// What variable_rank_router_static!{...} generates for the 3-domain case:

pub struct Router3MoveCombatQuest {
    move_cluster: ClusterHolder<12, 8>,
    combat_cluster: ClusterHolder<6, 16>,
    quest_cluster: ClusterHolder<3, 32>,
    domain_directions: [[f32; 3]; 3],
}

impl Router3MoveCombatQuest {
    pub fn tick(
        &self,
        z_full: &[f32; 32],
        activity: &[f32; 3],
        scratch_full: &mut [f32],
        dz_out_full: &mut [f32; 32],
    ) -> RoutingVerdict {
        let domain = pick_domain::<3, 3>(activity, &self.domain_directions);
        let winner = match domain {
            0 => {
                // L=8 — no vtable, direct ClusterHolder method call
                let z_proj = &scratch_full[..8];
                let blend_scratch = &mut scratch_full[8..16];
                let dz_proj = &mut scratch_full[16..24];
                self.move_cluster.apply_direct(z_full, MOVE_IDX, z_proj, blend_scratch, dz_proj);
                self.move_cluster.scatter(MOVE_IDX, dz_proj, dz_out_full);
                self.move_cluster.winner()
            }
            1 => { /* L=16 */ }
            2 => { /* L=32 */ }
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        RoutingVerdict { domain, winner }
    }

    pub fn override_cluster_pi(&mut self, domain: usize, pi: &[f32]) {
        match domain {
            0 => self.move_cluster.blend.pi[..12].copy_from_slice(&pi[..12]),
            1 => self.combat_cluster.blend.pi[..6].copy_from_slice(&pi[..6]),
            2 => self.quest_cluster.blend.pi[..3].copy_from_slice(&pi[..3]),
            _ => unreachable!(),
        }
    }
}
```

**Every call is a direct, monomorphized method call.** Zero `Box<dyn>`, zero vtables. The match is a jump table predicted by the CPU branch predictor (3 arms, heavily skewed by the gate — typically >80% to one domain).

## Prerequisite: inherent methods on `ClusterHolder`

`ClusterHolder` currently only exposes `apply_blended` through the `ErasedCluster` trait. The macro router must call it WITHOUT going through `dyn ErasedCluster`. Two clean options:

1. **Add inherent methods** `apply_direct(...)`, `winner()`, `scatter(...)` on `ClusterHolder<K, L>`. These wrap the existing `blend.apply_blended(...)` + winner-scan logic. No trait object involved. **RECOMMENDED** — keeps the logic in one place (DRY) and the macro router stays thin.

2. **Macro owns the blend directly.** The macro-generated struct owns `CommittedFieldBlend<K, L>` + `[Box<dyn ArchetypeFieldSource<L>>; K]` per domain (not wrapped in `ClusterHolder`). Duplicates `ClusterHolder`'s shape. **REJECTED** — DRY violation.

Option 1 is the T2 implementation detail. The macro assumes `ClusterHolder` gains 2-3 inherent helper methods.

## Bench impact (no restructuring needed)

The existing G2 bench hot path:

```rust
router.cluster_mut(0).override_pi(pi_move);      // 3 virtual calls
router.cluster_mut(1).override_pi(pi_combat);
router.cluster_mut(2).override_pi(pi_quest);
router.tick(state, activity, &mut scratch, &mut dz_out);  // 1 virtual call inside
```

becomes, with the macro router:

```rust
macro_router.override_cluster_pi(0, pi_move);    // 3 direct field writes
macro_router.override_cluster_pi(1, pi_combat);
macro_router.override_cluster_pi(2, pi_quest);
macro_router.tick(state, activity, &mut scratch, &mut dz_out);  // match dispatch, no vtable
```

The bench shape (shared router + per-NPC pi override) is UNCHANGED. The macro router supports the same API. All 4 vtables eliminated. **No goalpost-moving.**

**Honest caveat:** even with the macro, the per-NPC `override_cluster_pi` calls cost ~5 ns each (3× `copy_from_slice` on K-element arrays). This is ~15 ns vs the baseline's 0 ns (baseline writes pi as a single `[f32; 3]` field assignment). The legitimate variable-rank work (domain gate + projection + scatter) is another ~15 ns. So the realistic floor is `51 + 15 + 15 = 81 ns` vs `51 ns` baseline = **~1.6×**, not ≤1.0×.

**This means G2 might still FAIL even after monomorphization.** The vtable tax (~50 ns) is recoverable, but the `override_pi` per-NPC cost (~15 ns) and the irreducible variable-rank work (~15 ns) are not. If G2 re-gates at ~1.6×, Issue 189 T4's fallback applies: the feature stays opt-in forever, the 2.63× entropy gain is the selling point.

The honest path forward: implement the macro (T2), re-run G2 (T3), report the actual number (T4). Do NOT pre-emptively declare victory.

## Relationship to production usage

The bench's per-NPC `override_pi` is a **bench artifact**, not a production cost. In production (per Issue 189 §"Why the Plan 558 prediction was wrong"):

- Each NPC owns its own router with its own committed pi baked in at construction.
- Zero `override_pi` calls per tick.
- Only 1 vtable call (the `apply_blended` inside `tick`) — which the macro also eliminates.

So in production, the macro router reaches `51 + 15 = 66 ns` vs `51 ns` baseline = **~1.3×**. Still not ≤1.0×, but much closer. The ≤1.0× target may be structurally unreachable for variable-rank (the domain gate + projection is real work the baseline doesn't do).

**Recommendation for T3:** add a SECOND bench variant (`g2_perf_production_shape`) that measures the per-NPC-owned-router case (no `override_pi`). Report both numbers. The shared-router bench stays as the conservative bound; the production-shape bench is the realistic case. This is honest measurement, not goalpost-moving — both shapes are documented.

## T1 verdict

**DECISION: Option B — single generic `macro_rules!` with explicit indices.**

T2 implements:
1. `variable_rank_router_static!` macro in `variable_rank_domain_expert.rs` (or a sibling `macro.rs` if line count exceeds 2048).
2. 2-3 inherent helper methods on `ClusterHolder` (`apply_direct`, `winner`, `scatter`) — no trait object.
3. G1 test ported to the macro router (10K inputs, no NaN).
4. G4 alloc-free audit ported to the macro router.
5. Keep existing `VariableRankRouter` (dynamic) as the ergonomic opt-in path.

T3 re-runs G2 with both shapes:
- `g2_perf_shared_router` (existing shape — conservative bound)
- `g2_perf_production_shape` (per-NPC-owned-router — realistic case)

T4 promotion decision based on T3 results, per Issue 189 acceptance criteria.

## T2 verdict — IMPLEMENTED (2026-07-22)

**DONE.** All 5 T2 deliverables shipped:

1. **`variable_rank_router_static!` macro** — `#[macro_export]` at crate root.
   Syntax: `pub struct Name<DOMAINS, D_FULL, A>;` + per-domain entries
   (`$idx => $field: ClusterHolder<K, L> => [indices];`). Generates a
   non-generic struct with typed fields + `new()` / `tick()` /
   `override_cluster_pi()` methods. The `tick()` body is a `match` on the
   domain index — zero vtable dispatch.
2. **Inherent helper methods on `ClusterHolder`**: `LATENT_DIM` +
   `EXPERT_COUNT` (associated consts) + `apply_direct()` +
   `override_pi_direct()` (inherent methods). The `ErasedCluster` impl
   delegates to these (DRY — one logic path).
3. **G1 correctness ported**: 7 new lib tests — dispatch (move/combat),
   scatter-back zeros, `override_cluster_pi` winner-change, 10K-input
   no-NaN, and the **bit-identical-to-dynamic parity gate** (500 random
   inputs, verdict + dz_out byte-for-byte identical to the dynamic
   `VariableRankRouter`).
4. **G4 alloc-free ported**: `variable_rank_domain_expert_macro_alloc.rs`
   (2-phase audit: plain tick + override_pi path, both 0 bytes / 1000
   ticks).
5. **Dynamic `VariableRankRouter` preserved** — unchanged, still the
   ergonomic opt-in path.

**Validation**: 1786 lib tests + 2 alloc tests pass. `--all-features` +
`--no-default-features` + `clippy --all-targets` all clean. File is 1071
lines (well under the 2048 limit).

**Next**: T3 re-runs the G2 bench with the macro router to measure the
vtable-elimination gain.

## T3+T4 verdict — G2 still FAILS (2026-07-22)

**DONE.** Re-ran G2 in release mode with warm-up + min-of-5 methodology.
Two bench shapes measured:

| Shape | Baseline | Variable-rank | Ratio |
|---|---|---|---|
| Dynamic shared (original) — 1K | ~50 ns | ~114 ns | ~2.2× |
| **Macro shared** — 1K | 49.8 ns | 83.1 ns | **1.668×** |
| **Macro shared** — 10K | 49.1 ns | 82.7 ns | **1.682×** |
| **Macro production-shape** — 1K | 49.1 ns | 93.1 ns | **1.896×** |
| **Macro production-shape** — 10K | 47.8 ns | 91.2 ns | **1.908×** |

The vtable elimination recovered ~25% of the overhead (dynamic ~2.2× → macro
shared ~1.7×), but the variable-rank pattern is structurally more expensive —
the ~35 ns irreducible overhead (domain gate + projection + scatter +
override_pi) over a ~50 ns baseline = 1.7×.

The production-shape was **SLOWER** than shared (~1.95× vs ~1.7×) due to cache
thrashing from per-NPC boxed fields — a bench artifact, not a fundamental
cost. In real production, fields would be shared + only pi would be per-NPC.

**T4 verdict: stays opt-in forever.** The ≤1.0× target is structurally
unreachable for variable-rank. The 2.63× entropy gain is the selling point.
See [Benchmark 558](../../.benchmarks/558_variable_rank_domain_expert_goat.md)
§"Monomorphization re-gate" for the full analysis.
