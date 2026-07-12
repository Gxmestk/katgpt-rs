# Plan 429: Elasticity-Gated Update — DSOM Error-Scaled Neighborhood Primitive

**Date:** 2026-07-12
**Research:** [katgpt-rs/.research/415_Dynamic_SOM_Elasticity_Gated_Latent_Update.md](../.research/415_Dynamic_SOM_Elasticity_Gated_Latent_Update.md)
**Source paper:** Rougier & Boniface, "Dynamic Self-Organising Map" (Neurocomputing 2011, ⟨inria-00495827⟩) + Guérin et al. survey (arXiv:2501.08416)
**Target:** `katgpt-rs/crates/katgpt-core/src/` (new module) + `riir-neuron-db/src/neighbor_heal.rs` (consumer)
**Status:** Active — Phase 1–3 complete (T1.1–T1.4, T2.1–T2.7, T3.1–T3.3 done). G5 PASS.

---

## Goal

Ship the **elasticity-gated update** as an open primitive in `katgpt-core` — a generic, modelless function that computes a DSOM-style error-scaled neighborhood update on any latent-state vector. The primitive takes a current state, an observation (or target centroid), a set of neighbors with lattice distances, and an elasticity parameter η, and produces an update delta where:
1. The step size scales with the local error (`‖observation − state‖`)
2. The neighborhood weights are error-gated (`exp(−1/η² · d²/error²)`)

The primary consumer is `riir-neuron-db`'s `neighbor_heal` (error-scaled shard heal). Secondary consumers: `evolve_hla` (error-scaled belief update), `graph_laplacian` (non-uniform DEC diffusion).

**GOAT gate**: G1 (error-scaled step correctness), G2 (neighborhood expansion), G3 (no-regression on fallback), G4 (quorum bit-identity), G5 (structure-matching PoC — the §3.6 defend-wrong gate), G6 (freeze gate compatibility).

---

## Phase 1 — Open Primitive Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/elasticity_gated_update.rs` — the open primitive module.
  - `ElasticityConfig { eta: f32, epsilon: f32, support_diameter: f32 }` — config POD.
  - `elasticity_gated_update_into(state: &[f32], target: &[f32], neighbors: &[(&[f32], f32)], config: &ElasticityConfig, out: &mut [f32])` — the core function.
  - Pure closed-form math: error = normalized L2 distance, step = ε · error, weight = exp(−d²/(η²·error²)), delta = step · Σ(weight · (neighbor − state)) / Σweight.
  - Zero-allocation: all scratch on stack, `&mut [f32]` output buffer.
  - Feature gate: `elasticity_gated_update` (opt-in).

- [x] **T1.2** Add `elasticity_gated_update` feature to `katgpt-core/Cargo.toml` (opt-in, no default).

- [x] **T1.3** Unit tests (20 tests, all PASS):
  - Error-scaled step: `step(δ=0.5) / step(δ=0.01) ≈ 50` (within 5%).
  - Neighborhood expansion: `effective_k(error=0.01) ≤ 2`, `effective_k(error=0.5) ≥ 5`.
  - Zero-error guard: when `error < 1e-8`, output is all zeros (no heal needed).
  - Determinism: same input → bit-identical output (100/100 runs).

- [x] **T1.4** `cargo clippy` clean (default+feature, --all-features, --no-default-features+feature all clean). `cargo test -p katgpt-core --features elasticity_gated_update --lib` passes (1506 passed, 0 failed, 3 ignored).

---

## Phase 2 — riir-neuron-db Consumer Wiring

### Tasks

- [x] **T2.1** Add `eta: Option<f32>` + `support_diameter: f32` fields to `NeighborHealConfig` in `riir-neuron-db/src/neighbor_heal.rs`. Default `None` / `1.0` (backward-compatible).

- [x] **T2.2** Implement `dsom_heal_into` + `plan_dsom_heal_into` — calls `katgpt_core::elasticity_gated_update::compute_error` + `neighborhood_weight` (DRY: reuses `neighbor_heal_delta_into` for the weighted centroid). Feature gate: `elasticity_gated_heal` (implies `neighbor_heal` + `katgpt-core/elasticity_gated_update`).

- [x] **T2.3** Wire into `MapeKLoop::plan_with_index`: when `config.eta.is_some()`, uses `plan_dsom_heal_into` (returns step as alpha); when `None`, uses the existing `plan_neighbor_heal_into` (backward-compatible). Added `neighbor_eta` + `support_diameter` fields to `MapeKLoop` with `with_neighbor_eta` / `with_support_diameter` builders.

- [x] **T2.4** GOAT gate G3 (no-regression): with `eta = None`, the heal produces bit-identical results to the current `neighbor_heal_delta`. Verified: 309 default-feature tests pass (0 failures), `test_config_default` updated to check new fields default to `None`/`1.0`.

- [x] **T2.5** GOAT gate G4 (quorum bit-identity): 100/100 bit-identical runs of `plan_dsom_heal_into` with the same input (`g4_dsom_determinism` test).

- [x] **T2.6** GOAT gate G6 (freeze gate compatibility): `g6_dsom_step_positive_on_error` (step > 0 on non-trivial error) + `g6_dsom_zero_error_guard` (step = 0 when state equals centroid) PASS. The DSOM step scales with error, so low error → small step → state stabilizes → compatible with `can_freeze`.

- [x] **T2.7** `cargo clippy` clean (default / `--features elasticity_gated_heal` / `--all-features`). `cargo test --features elasticity_gated_heal --lib` passes (303 passed, 0 failed). Default-feature tests pass (309 passed, 0 failed).

---

## Phase 3 — Structure-Matching PoC (§3.6 Defend-Wrong Gate)

### Tasks

- [x] **T3.1** Created `riir-neuron-db/tests/dsom_structure_matching_poc.rs` — 90%/10% safe/frontier population, 20 heal cycles, measures coverage ratio.
- [x] **T3.2** GOAT gate G5 (structure-matching): `frontier_coverage / safe_coverage = 23.31 ≥ 0.5` → **PASS**. The DSOM does not under-represent rare regions. Verdict remains GOAT (not auto-elevated to Super-GOAT — the structure-matching is confirmed but not a novel capability class).
- [x] **T3.3** PoC result recorded in Research 415 §"PoC Addendum" (with honest caveat: ratio > 1.0 means frontier heals BETTER, not exactly equal; full structure-matching would need a SOM training run).

---

## Phase 4 — Benchmark + Promotion Decision

### Tasks

- [ ] **T4.1** Benchmark: `criterion` bench comparing `dsom_heal_delta` vs `plan_neighbor_heal_into` on a realistic shard population (1000 shards, k=5, STYLE_DIM=64). Measure latency per heal.

- [ ] **T4.2** GOAT gate G2 (latency): `dsom_heal_delta` latency < 2× the current `plan_neighbor_heal_into` (the error computation + exponential adds overhead). Target: < 500 ns per heal (same budget as the current heal).

- [ ] **T4.3** Promotion decision:
  - If G1–G6 all PASS → promote `eta` to default-on in `NeighborHealConfig` (with `eta = Some(1.0)` as default, configurable).
  - If G5 FAILS (structure-matching doesn't hold) → keep `eta` opt-in, note the PoC result, the error-scaled step (G1) is still a GOAT gain.
  - If G6 FAILS (freeze gate breaks) → keep `eta` opt-in, investigate the freeze-gate interaction before promotion.

- [ ] **T4.4** Update README feature table + AGENTS.md with the promotion/demotion result.

---

## Phase 5 — Cross-Repo Fusion (Optional, P3)

### Tasks

- [ ] **T5.1** Error-weighted `graph_laplacian` in `katgpt-core/src/dec/` — add an `error_weighted_graph_laplacian` variant that takes per-edge error signals and uses the DSOM neighborhood function as edge weights. Feature gate: `dec_operators` + `elasticity_gated_update`.

- [ ] **T5.2** Three-tier heal routing in `MapeKLoop::plan_with_index`: frozen backup (O(1), stability) → DSOM neighborhood (O(n), plasticity) → global mean (O(1), fallback). Error signal routes between tiers. Depends on Research 298 frozen LOD backup.

---

## Notes

- **Modelless**: pure closed-form math (exponential + weighted average). No training, no backprop. Satisfies the modelless-first mandate.
- **Backward-compatible**: `eta = None` falls back to the current fixed-alpha `neighbor_heal`. Zero behavior change unless opted in.
- **Raw vs latent boundary**: the error and neighborhood weights are computed in latent space (cosine on HLA). Only the post-heal `style_weights` + BLAKE3 cross the sync boundary. Same as today.
- **The §3.6 PoC (Phase 3) is mandatory** — the structure-matching claim is a quality claim that needs a head-to-head PoC. The error-scaled step (G1) is provable by construction; the structure-matching (G5) is not.
