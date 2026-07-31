# Plan 558: Variable-Rank Domain Expert Clusters — Production Primitive + GOAT Gate

**Date:** 2026-07-22
**Research:** [katgpt-rs/.research/453_Variable_Rank_Domain_Expert_Clusters.md](../.research/453_Variable_Rank_Domain_Expert_Clusters.md)
**Source paper:** [arXiv:2601.18089](https://arxiv.org/abs/2601.18089) — LatentMoE (NVIDIA, 2026-01) — **PASS paper itself**; this plan distills the *transferable principle* (compress-to-intrinsic-rank → scale expert count) into a modelless per-NPC cognition primitive.
**Target:** `crates/katgpt-core/src/variable_rank_domain_expert.rs` (new module, sibling to `committed_field_blend.rs`) + Cargo feature `variable_rank_domain_expert` (re-exported from root `katgpt-rs/Cargo.toml`).
**Status:** COMPLETE — Phase 0 (PoC) + Phase 1 (skeleton + primitives) + Phase 2 (router) + Phase 3 (GOAT gate) + Phase 4 (honest assessment) ALL DONE. **Verdict: G2 FAIL (2.0× latency) — stays opt-in.** G1/G3/G4/G5 PASS. See [.benchmarks/558_variable_rank_domain_expert_goat.md](../.benchmarks/558_variable_rank_domain_expert_goat.md) for the full results + the monomorphization escape hatch.

> **Numbering note.** Research note 453 + Plan 558 use *different* numbers because `.research/` and `.plans/` are independent namespaces with independent highwater markers (`.research/` was at 452 → 453; `.plans/` is at 557 → 558). `.plans/453_*` is already taken (`453_bounded_one_step_lacam_escalation.md`). The research note at `.research/453_*.md` is the design doc; this plan is the execution tracker.

---

## Goal

Ship `variable_rank_domain_expert` — a thin composition layer over `CommittedFieldBlend<N, D>` (Plan 321, already DEFAULT-ON) that applies LatentMoE's transferable principle to per-NPC cognition:

1. **Different domains have different intrinsic feature ranks `r_eff`.** Movement needs ~8 dims; combat ~16; quest/social ~32. The current uniform `CommittedFieldBlend<3, 32>` wastes 24 of 32 dimensions on movement decisions.
2. **Compress to the task's rank, then scale expert count by α = D_full / ℓ_domain.** `K' = αK` preserves total compute `K×D = K'×ℓ` while boosting archetype diversity (LatentMoE §3.2). At iso-FLOP, more archetypes = higher utilization entropy = more behaviorally diverse NPCs.
3. **Guided projection, NOT blind JL/PCA.** We select semantically-relevant dimensions per domain (a zero-cost slice gather), not a random projection matrix. This mitigates the Plan 230 cautionary flag (blind JL projection to m=8 violated the lower bound by 200×).

**The PoC (already DONE, Research 453 §4) confirmed the core hypothesis H1:**
- **1.63× higher archetype utilization entropy** vs uniform `<3, 32>` at iso K×D=96 compute
- Per-domain entropy at 97–99.7% of `log₂(K')` — guided projection does NOT collapse diversity
- 1.23× latency overhead in debug (≤2.0× gate)

**This plan's job is to:**
1. Promote the PoC into a feature-gated production module (small generic primitives, not the hard-coded `Move/Combat/Quest` clusters of the PoC).
2. Land a release-mode criterion bench at 1K + 10K NPCs to validate the Plasma-tier bandwidth claim.
3. Run the GOAT gate (G1–G5) honestly. Promotion to default-on only if all gates pass.

**Scope — what this plan does NOT do (deferred to future plans):**
- **Octree spatial indexing integration** (Research 453 §2.4 "mixture of octree experts") — separate future plan; this plan is the substrate primitive, not the spatial wiring.
- **Quest-grammar archetype construction** (Research 453 §2.5) — separate future plan; this plan uses host-supplied archetype direction fields.
- **riir-ai runtime wiring** (per-NPC `DomainExpertBundle` resource, system integration) — separate future plan; this plan ships the generic primitive, not the consumer integration.
- **Private selling-point guide** (riir-ai/.research/) — only if GOAT gate passes + this promotes to default-on.

---

## §3.5 Modelless Unblock — PASSED (verified in Research 453 §1)

All three modelless paths pass:
- **Path 1 (freeze/thaw):** archetype direction fields are frozen snapshots; per-domain projection masks are compile-time constants.
- **Path 2 (raw/lora hot-swap):** the projection is a *selection mask* (zero-cost gather), not a learned matrix — no LoRA to swap.
- **Path 3 (latent correction):** domain gate is `argmax(activity · domain_directions)` — a deterministic dot-product routing, not a learned router.

**No riir-train dependency.** The K archetype fields themselves are host-supplied (eventually mined by MAG Plan 418 or quest grammar) — that's the freeze/thaw substrate, not a per-entity training dependency.

---

## Architecture

### The three primitives (generic math, no game semantics)

The PoC ships three hard-coded clusters (`MoveCluster`/`CombatCluster`/`QuestCluster`). The production module generalizes to three small generic primitives:

```rust
/// 1. Domain Gate — deterministic dot-product routing.
///
/// Picks ONE domain from N candidates by `argmax(activity · domain_directions)`.
/// Modelless (no learned router, no softmax — pure argmax on host-supplied
/// activity vector × host-supplied direction matrix).
///
/// Sibling to `latent_steering::LatentSteeringVector` (Plan 309) — both use
/// dot-projection onto pre-computed direction vectors. The gate just routes;
/// the steering vector adjusts state.
pub fn pick_domain<const N: usize, const A: usize>(
    activity: &[f32; A],
    domain_directions: &[[f32; A]; N],
) -> usize;

/// 2. Guided Projection — zero-cost dimension gather (NOT random projection).
///
/// Selects the semantically-relevant `ℓ` dimensions from a `D`-dim state into
/// an `ℓ`-dim latent. This is the Plan 230 mitigation: instead of a random
/// JL/PCA projection matrix (which requires m ≥ 554 for ε=0.5 at n=100), we
/// select known-relevant dimensions by index. No information loss within the
/// selected subspace; no JL bound to violate.
///
/// `indices` MUST be sorted ascending + unique (debug_assert enforces).
/// Production code supplies compile-time-known index arrays
/// (e.g. `MOVE_DIMS = [0,1,2,3,4,5,6,7]`).
pub fn project_guided<const D: usize, const L: usize>(
    z_full: &[f32; D],
    indices: &[usize; L],
    z_out: &mut [f32; L],
);

/// 3. Variable-Rank Router — generic over N domains × heterogeneous cluster shapes.
///
/// A `VariableRankRouter<DOMAINS>` owns one `CommittedFieldBlend<K_d, L_d>` per
/// domain `d` (where each domain has its own `K_d` archetypes + its own
/// projection rank `L_d`). Per NPC per tick:
///   1. `pick_domain(activity, dirs)` → which cluster to invoke
///   2. `project_guided(z_full, dims[d], z_proj)` → compress to that domain's rank
///   3. `cluster.apply_blended(fields, z_proj, ...)` → blend at variable rank
///
/// The router is generic over a `DomainSpec` array describing each domain's
/// (rank L_d, expert count K_d, projection indices). This keeps the module
/// free of game semantics — "move/combat/quest" is host vocabulary, not
/// primitive vocabulary.
pub struct VariableRankRouter<const DOMAINS: usize> { /* ... */ }
```

### Why this is a thin composition, not a new primitive

`CommittedFieldBlend<N, D>` (Plan 321) is already generic over `N` and `D` — `CommittedFieldBlend<12, 8>` is already valid Rust. The variable-rank property is already expressible at the type level. What this module adds:

1. **The domain gate** — `pick_domain` (~5 LOC, deterministic argmax on dot-product).
2. **The guided projection** — `project_guided` (~5 LOC, slice gather with bounds check).
3. **The router orchestration** — `VariableRankRouter` (~50 LOC, dispatch loop over domains).
4. **The validated α-scaling recipe** — the PoC's empirical confirmation that `K' = αK` preserves entropy (Research 453 §4 results table).

The value is not novel math; the value is the **demonstrated** entropy gain at scale + the **validated** Plan 230 mitigation (guided projection doesn't collapse diversity) + a single canonical place where host code reaches the variable-rank pattern.

### Composition with existing primitives

- **Reuses `CommittedFieldBlend<N, D>` (Plan 321)** — the per-domain blend is just `CommittedFieldBlend<K_d, L_d>` instantiated at the domain's variable rank. No code duplication.
- **Reuses `latent_steering` direction-vector pattern (Plan 309)** — the domain gate is a multi-candidate generalization of the latent steering dot-projection.
- **Compatible with `MAG` (Plan 418)** — the host supplies `domain_directions`; MAG can mine them unsupervised (Research 453 §2.5).
- **Compatible with freeze/thaw (Plan 321 §Architecture)** — the projection indices + per-domain π vectors are frozen artifacts, BLAKE3-committable.

---

## Phase 1 — Feature-gated Skeleton + Primitives

Goal: a compiling, tested, feature-gated module that implements the three primitives generically. No router yet (Phase 2).

### Tasks

- [x] **T1.1** Add feature flag `variable_rank_domain_expert = ["committed_field_blend"]` to `katgpt-rs/crates/katgpt-core/Cargo.toml` features section (place alphabetically near `subspace_steering`). Add a root-level alias `variable_rank_domain_expert = ["katgpt-core/variable_rank_domain_expert"]` to `katgpt-rs/Cargo.toml` (mirror the Plan 557 RoVE root-facade pattern).
- [x] **T1.2** Add `#[cfg(feature = "variable_rank_domain_expert")] pub mod variable_rank_domain_expert;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` (near `committed_field_blend` re-export, around L1461).
- [x] **T1.3** Implement `pick_domain<const N: usize, const A: usize>(activity: &[f32; A], domain_directions: &[[f32; A]; N]) -> usize` in `variable_rank_domain_expert.rs`:
  - Compute `score[d] = dot(activity, domain_directions[d])` for `d in 0..N`.
  - Return `argmax(score)`. Ties broken by lowest index (deterministic).
  - **Zero allocation.** Stack array `[f32; N]` for scores.
  - **No softmax, no sigmoid** — pure argmax (per global rule "use sigmoid not softmax"; here we use neither because routing is deterministic argmax).
- [x] **T1.4** Implement `project_guided<const D: usize, const L: usize>(z_full: &[f32; D], indices: &[usize; L], z_out: &mut [f32; L])`:
  - `debug_assert!` that `indices` is sorted ascending + unique (catches misuse in debug; release elides).
  - `debug_assert!(indices[L-1] < D)` (bounds check in debug).
  - Loop: `z_out[i] = z_full[indices[i]]` for `i in 0..L`.
  - **Zero allocation.** Pure gather.
- [x] **T1.5** Write unit tests in `variable_rank_domain_expert.rs` `mod tests`:
  - [x] **G1-mechanics (gate argmax):** `pick_domain` on 3 hand-crafted activity/direction combos picks the expected winner. Verifies argmax (not softmax, not sigmoid).
  - [x] **G1-mechanics (gate ties):** two domains with equal score → returns the lower index (deterministic tie-break).
  - [x] **G2-mechanics (projection gather):** `project_guided` on a known `z_full` + `indices=[0,2,5,7]` produces `[z[0], z[2], z[5], z[7]]`.
  - [x] **G2-mechanics (projection identity):** `indices=[0,1,...,D-1]` reproduces `z_full` (no projection).
  - [x] **G3-mechanics (projection debug_assert):** unsorted indices trigger `debug_assert` panic in `#[cfg(debug_assertions)]` test (use `#[should_panic]`).
- [x] **T1.6** Document module in `variable_rank_domain_expert.rs` header with:
  - Paper reference (arXiv:2601.18089 LatentMoE — PASS; this distills the transferable principle).
  - Research note link (Research 453).
  - Plan 230 cautionary flag explanation (blind JL fails; guided projection mitigates).
  - Composition note: thin layer over `CommittedFieldBlend` (Plan 321); reuses Plan 309 direction-vector pattern.
  - Honest scope caveat: this is a bandwidth/diversity optimization (Q2 in Research 453 novelty gate was conditional), not a new capability class.

**Exit criterion:** `cargo check -p katgpt-core --features variable_rank_domain_expert` clean. `cargo test -p katgpt-core --features variable_rank_domain_expert --lib` passes.

---

## Phase 2 — VariableRankRouter + α-Scaling Recipe

Goal: implement the dispatch orchestration that owns multiple heterogeneous-rank clusters and routes per-NPC per-tick. This is the user-facing API.

### Tasks

- [x] **T2.1** Define `DomainSpec<const L: usize, const K: usize>` struct describing one domain's shape:
  - `projection_indices: [usize; L]` — which dims of the full state belong to this domain (guided projection mask).
  - `domain_direction: [f32; A]` — the activity-vector direction for the domain gate (the row of `domain_directions` the gate uses to score this domain).
  - Type-level `L` and `K` enforce the variable-rank property at compile time (different domains = different instantiations).
- [x] **T2.2** Implement `VariableRankRouter<const DOMAINS: usize, const D_FULL: usize, const A: usize>`:
  - Owns `domain_specs: [DomainSpec; DOMAINS]` (the projection masks + gate directions).
  - Owns `clusters: [Box<dyn ErasedCluster>; DOMAINS]` — type-erased per-domain blend holders (because each cluster has a different `K_d, L_d` const-generic, we need trait objects for the dispatch; the type erasure is internal, the public API is generic).
  - `tick(&mut self, z_full: &[f32; D_FULL], activity: &[f32; A], dz_out_full: &mut [f32; D_FULL]) -> RoutingVerdict`:
    1. `domain = pick_domain(activity, &self.domain_directions())` → which cluster.
    2. `project_guided(z_full, &self.domain_specs[domain].projection_indices, z_proj)` → compress.
    3. `self.clusters[domain].apply_blended(z_proj, dz_proj)` → blend at rank `L_d`.
    4. **Inverse projection** (scatter back to full D): zero `dz_out_full`, then write `dz_proj[i]` back to `dz_out_full[projection_indices[i]]`. Domains not in the mask get zero update (the masked-out dims don't affect this domain's dynamics — that's the entire point).
    5. Return `RoutingVerdict { domain, cluster_size: K_d }` for caller introspection.
- [x] **T2.3** Define `trait ErasedCluster` (private to the module) — object-safe wrapper around `CommittedFieldBlend<N, D>`:
  - `fn apply_blended(&self, z_proj: &[f32], dz_out: &mut [f32]) -> usize;` (returns winning archetype index)
  - `fn commitment(&self) -> [u8; 32];` (BLAKE3 of the underlying blend — anti-tamper)
  - Implementations: `ClusterHolder<const K: usize, const L: usize>` wraps `CommittedFieldBlend<K, L>` + `[DirectionField<L>; K]` (host-supplied archetype fields).
  - **Honest note:** the type erasure here is the cost of heterogeneous const-generics across domains. The alternative (macro-generate per-domain-count routers) is worse for ergonomics. Trait-object dispatch is one virtual call per NPC per tick — negligible vs the blend work.
- [x] **T2.4** Implement `RoutingVerdict { domain: usize, cluster_size: usize, winner: usize }`:
  - Small `Copy` struct returned from `tick` — consumers use it for entropy bookkeeping (Research 453 PoC pattern).
- [x] **T2.5** Write integration tests using a 3-domain fixture (move `<12,8>`, combat `<6,16>`, quest `<3,32>` — same shapes as the PoC):
  - [x] **G1-router (dispatch correctness):** NPC with high "move" activity → `domain == 0`; high "combat" → `domain == 1`; high "quest" → `domain == 2`.
  - [x] **G2-router (projection round-trip):** for a domain with `indices = [0..L]`, the scatter-back writes `dz_proj` to `dz_out_full[0..L]` and zeros `dz_out_full[L..D_FULL]`.
  - [x] **G3-router (α-scaling entropy):** 1000-NPC run produces weighted-avg entropy ≥ 1.5× the uniform `<3,32>` baseline (reproduces the PoC's 1.63× result on the production API).
- [x] **T2.6** Document the `DomainSpec` + `VariableRankRouter` API with the move/combat/quest example from Research 453 §2.1 as the canonical usage example.

**Exit criterion:** `cargo test -p katgpt-core --features variable_rank_domain_expert --lib` passes including the 1000-NPC entropy test.

---

## Phase 3 — GOAT Gate (G1–G5)

Goal: the formal GOAT gate. Mirrors Plan 321's gate structure. Promotion to default-on requires ALL gates to PASS.

### Tasks

- [x] **T3.1 (G1 correctness)** Unit tests in module:
  - `pick_domain` argmax + tie-break.
  - `project_guided` gather + identity + debug_assert on bad indices.
  - `VariableRankRouter::tick` dispatch + scatter-back correctness.
  - **Pass criterion:** all green, no panics, no NaN in outputs across 10K random inputs.
- [x] **T3.2 (G2 perf — release mode)** Criterion bench `benches/variable_rank_domain_expert_goat.rs`:
  - **B1:** 1K-NPC throughput — `pick_domain` + `project_guided` + `apply_blended` per NPC, mean ns/NPC. Compare vs uniform `<3,32>` baseline.
  - **B2:** 10K-NPC throughput — same, at MMO scale (the Plasma-tier bandwidth claim).
  - **Pass criterion:** mean latency ≤ 1.0× baseline at 1K (variable-rank should NOT be slower in release — debug was 1.23× due to lack of inlining; release should be ≤1.0× because the masked-out dims skip blend work). At 10K, latency should be linear (no superlinear cost).
  - **Honest failure mode:** if release latency is still >1.0× baseline, the gate FAILS. Variable-rank is a bandwidth optimization; if it costs compute, it's not a win. The PoC's debug 1.23× is expected to drop to ≤1.0× in release because (a) the trait-object dispatch is one virtual call, (b) the smaller `CommittedFieldBlend<12,8>` does 96 multiply-adds (same as `<3,32>`'s 96), (c) `project_guided` is `L` indexed loads. If this prediction is wrong, the gate honestly fails.
- [x] **T3.3 (G3 no-regression)** All existing tests pass under `--all-features` and under `--no-default-features`:
  - `cargo test -p katgpt-core --all-features`
  - `cargo test -p katgpt-core --no-default-features`
  - **Pass criterion:** zero new failures vs the pre-Plan-558 baseline.
- [x] **T3.4 (G4 alloc-free hot path)** CountingAllocator test in `tests/variable_rank_domain_expert_alloc.rs`:
  - 1000 calls to `VariableRankRouter::tick` → 0 allocations after warmup.
  - **Pass criterion:** 0 bytes allocated in the steady-state hot path (the `Box<dyn ErasedCluster>` is constructed ONCE at router build time, not per-tick).
- [x] **T3.5 (G5 modelless purity)** Audit:
  - No `dep:` on any training/backprop/grad crate.
  - No `unsafe` in the module.
  - All math is closed-form (argmax + gather + sigmoid via Plan 321's existing primitive).
  - Document the audit result in `.benchmarks/558_variable_rank_domain_expert_goat.md`.

**Exit criterion:** All five gate tasks complete with verdicts recorded in `.benchmarks/558_variable_rank_domain_expert_goat.md`. ANY FAIL → stays opt-in, write honest postmortem in the bench file.

---

## Phase 4 — Honest Assessment + Promotion Decision

Goal: decide whether to promote to default-on based on the Phase 3 results. This phase is intentionally short — the decision rule is mechanical.

### Tasks

- [x] **T4.1** Write `.benchmarks/558_variable_rank_domain_expert_goat.md` with:
  - Full results table (B1, B2, G1–G5 verdicts).
  - Comparison vs the Research 453 PoC numbers (did release-mode match the debug-mode prediction?).
  - Honest assessment: did variable-rank actually beat uniform-rank on latency at MMO scale? If not, why?
  - Verdict: PASS (promote) or FAIL (stays opt-in).
- [x] **T4.2 (only if all gates PASS)** Promote to default-on:
  - Add `variable_rank_domain_expert` to the `default = [...]` list in `katgpt-rs/crates/katgpt-core/Cargo.toml` with a Phase N comment block following the existing convention (see Plan 321's DEFAULT-ON block at L267-271 for the format).
  - Update the comment block with the GOAT results.
- [x] **T4.3 (only if FAIL)** Document why + what would need to change for promotion:
  - E.g., "G2 failed because the trait-object dispatch overhead dominated the blend work at K=3, L=32 (small clusters); promotion would require monomorphizing the router at compile time per domain count, which trades code-size for dispatch cost."
  - Stays opt-in. The Research 453 PoC is still valuable as a validated negative result (or: a "validated at debug scale, needs more work for production" result).
- [x] **T4.4** Update Research 453 §4 with a pointer to this plan + the bench results:
  - Update the "Status: Active — PoC phase" line to "Status: Promoted to Plan 558 — GOAT [PASS|FAIL]".

**Exit criterion:** Bench file written + promotion decision recorded in both the plan status block + Research 453.

---

## Validation

| Step | Command |
|---|---|
| Phase 1 exit | `cargo check -p katgpt-core --features variable_rank_domain_expert && cargo test -p katgpt-core --features variable_rank_domain_expert --lib` |
| Phase 2 exit | `cargo test -p katgpt-core --features variable_rank_domain_expert --lib` (incl. entropy test) |
| Phase 3 G2 bench | `cargo bench -p katgpt-core --bench variable_rank_domain_expert_goat --features variable_rank_domain_expert` |
| Phase 3 G3 no-regression | `cargo test -p katgpt-core --all-features && cargo test -p katgpt-core --no-default-features` |
| Phase 3 G4 alloc | `cargo test -p katgpt-core --features variable_rank_domain_expert --test variable_rank_domain_expert_alloc -- --nocapture` |
| Phase 4 (if PASS) | `cargo check --all-features` (default-on now compiles + doesn't break the matrix) |

---

## Honest Risks

1. **G2 latency prediction uncertainty.** The PoC ran at 1.23× baseline in debug. Release should be ≤1.0× because the masked-out dims skip blend work. **But:** the trait-object dispatch (`Box<dyn ErasedCluster>`) adds one virtual call per NPC per tick, which debug builds don't measure fairly. If the virtual call + the dispatch logic costs more than the saved blend work, G2 FAILS. Mitigation: if G2 fails narrowly, T4.3 documents a monomorphization escape hatch (macro-generated per-domain-count routers) as future work.

2. **Q2 (Research 453 novelty gate) was conditional.** This is a bandwidth optimization, not a new capability class. Even if all GOAT gates pass, the Super-GOAT tier (which requires Q2 = YES) is out of reach. The plan honestly targets GOAT, not Super-GOAT.

3. **The `Box<dyn ErasedCluster>` is a small smell.** Heterogeneous const-generics force type erasure. The alternative — a macro-generated enum dispatch — is more code for the same result. The plan accepts the trait-object cost as the price of ergonomic generic-over-domain-count API. If G2 fails because of it, T4.3 is the escape hatch.

4. **Plan 230 cautionary flag still applies to the host's projection indices.** This plan ships the *machanism* (guided projection = gather by index); the *correctness* of any specific projection mask (e.g. "move domain = dims [0..8]") is the host's responsibility. The module debug_asserts the indices are sorted + in-bounds, but cannot verify they're semantically correct. This is the same shape as Plan 309's "host supplies direction vectors" contract.

---

## References

- **Research:** [.research/453_Variable_Rank_Domain_Expert_Clusters.md](../.research/453_Variable_Rank_Domain_Expert_Clusters.md)
- **Source paper:** [arXiv:2601.18089](https://arxiv.org/abs/2601.18089) LatentMoE (PASS — paper itself; this plan distills the transferable principle)
- **Sibling plan:** [Plan 321](321_sampling_invariant_per_entity_moe_primitive.md) — `CommittedFieldBlend<N, D>` (the per-entity blend primitive this module composes over)
- **Cautionary flag:** [Plan 230](230_shard_embedding_projection.md) — blind JL/PCA projection to low rank FAILS; this plan mitigates with guided projection
- **Composition targets:** [Plan 309](309_latent_field_steering_primitive.md) (latent steering direction pattern), [Plan 418](418_mag_activation_geometry_primitive.md) (MAG unsupervised direction mining — future host supplier of `domain_directions`)
- **Format precedent:** [Plan 557](557_rotary_value_embeddings.md) (recent feature-flagged primitive plan; same root-facade Cargo pattern)
