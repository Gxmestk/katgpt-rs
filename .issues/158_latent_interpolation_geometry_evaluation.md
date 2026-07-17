# Issue 158 — Latent Interpolation Geometry Evaluation (iMAUVE for Committed Latents)

> **Source:** [Research 445](../.research/445_Latent_Thought_Flows_Text_Compression.md) — Prabhudesai & Geng, *Latent Thought Flows with Text Compression* (Jun 2, 2026).
> **Opened:** 2026-07-17
> **Type:** POC / proof task (per AGENTS.md — issue, not plan)
> **Verdict that opened it:** Gain (Research 445 §3). Not Super-GOAT (Q2/Q3 fail). Not GOAT (no provable gain until a defect is found).

---

## TL;DR

The paper's headline methodological contribution is the **iMAUVE metric** (nearest-neighbor midpoint interpolation quality predicts generation quality; Pearson r=0.99 with gMAUVE; rMAUVE saturates and is uninformative). We have **six** committed-latent substrates across the 7-repo stack (HLA states, `ArchetypeBlendShard` π, `KarcShard` weights, `NeuronShard::style_weights`, `ZoneGeometryPod`, `MerkleFrozenEnvelope`-versioned states). Each has a reconstruction/integrity test (the rMAUVE analog — "thaw produces bit-identical behavior"). **None has an interpolation-quality test (the iMAUVE analog — "midpoint of two latents decodes to a coherent intermediate behavior")**. This issue tracks a PoC that closes that gap uniformly.

The paper also ships a **5-way intervention probe battery** (matched/shuffled/zero/mean/noise) that extends our `FaithfulnessProbe` (Plan 278, binary, on injected memory) to the per-entity committed-state domain. Bundled into the same PoC.

---

## The PoC deliverable

A new katgpt-rs primitive + benchmark pair:

1. **`interpolation_geometry` module** (probably `katgpt-rs/crates/katgpt-core/src/eval/interpolation_geometry.rs` or under the existing `faithfulness_probe/` module) exposing:
   - `imauve_score<S: LatentSpace>(space: &S, samples: &[S::Point], held_out: &[S::Point]) -> f32` — for each real `point`, find nearest neighbor in latent space, decode `(point + nn) / 2`, compare decoded midpoint against `held_out` distribution.
   - `intervention_battery<S: LatentSpace>(space: &S, anchor: &S::Point, donors: &[S::Point]) -> InterventionReport` — 5 interventions (matched, shuffled, zero, mean, noise), each returning a divergence metric (CE analog or behavior delta).
   - The `LatentSpace` trait abstracts over HLA / ArchetypeBlendShard π / KarcShard weights / NeuronShard style_weights / ZoneGeometryPod / MerkleFrozenEnvelope versions — callers supply the encode/decode/midpoint operations.

2. **A benchmark report** applying the primitive to a representative subset of substrates (start with HLA and `NeuronShard::style_weights` — both already in katgpt-rs's purview; private substrates evaluated via their katgpt-rs-visible traits). Report which substrates have good interpolation geometry (midpoint stays on-manifold) vs bad (midpoint collapses to token-soup analog).

3. **Decision**:
   - If **all substrates pass** → close issue, document the methodology in a `.docs/` note, no further work.
   - If **any substrate fails** (midpoint is incoherent) → open a plan for a modelless fix (likely a regularization, projection, or commitment cadence change). The fix becomes the GOAT candidate; the metric stays as the regression test.

## Phase breakdown

### Phase 1 — `LatentSpace` trait + synthetic test fixture
- [ ] **T1.1** Define `trait LatentSpace { type Point; fn encode(&self, ...) -> Point; fn decode(&self, p: &Point) -> ...; fn midpoint(&self, a: &Point, b: &Point) -> Point; fn zero(&self) -> Point; fn mean(&self, samples: &[Point]) -> Point; fn noise(&self, rng) -> Point; fn divergence(&self, a: &Point, b: &Point) -> f32; }` — generic over the substrate.
- [ ] **T1.2** Implement a synthetic `LatentSpace` for unit testing (e.g., a 2D Gaussian mixture with known interpolation geometry).
- [ ] **T1.3** Unit-test `imauve_score` and `intervention_battery` on the synthetic space — verify the protocol correctly distinguishes a "good" synthetic space from a deliberately constructed "bad" one (length-clustering, like the paper's failure mode).

### Phase 2 — Apply to HLA (the strongest single substrate)
- [ ] **T2.1** Implement `LatentSpace` for HLA `[f32; 8]` (or whatever the public katgpt-core HLA type is). Midpoint = element-wise average. Decode = the existing `evolve_hla` / bridge-to-5-scalars path.
- [ ] **T2.2** Build a synthetic HLA population (e.g., sample N random HLA states from plausible distributions of the 8 dims).
- [ ] **T2.3** Run `imauve_score` and `intervention_battery` on the HLA population.
- [ ] **T2.4** Report: does HLA midpoint decode to a coherent intermediate emotion-scalar set, or does it collapse? Do the 5 interventions have the expected ordering (matched < shuffled ≈ zero ≈ mean ≈ noise)?

### Phase 3 — Apply to `NeuronShard::style_weights` (via the lock-free ShardIndex API)
- [ ] **T3.1** Implement `LatentSpace` for `NeuronShard` (likely in `riir-neuron-db`, with katgpt-rs consuming via trait). Midpoint = element-wise average of `style_weights[64]`. Decode = whatever reconstruction path the shard exposes.
- [ ] **T3.2** Build a synthetic shard population (e.g., sample N shards with random style_weights, or load from a benchmark fixture).
- [ ] **T3.3** Run `imauve_score` and `intervention_battery`.
- [ ] **T3.4** Report.

### Phase 4 — Decision
- [ ] **T4.1** If both HLA and NeuronShard pass: extend to ArchetypeBlendShard π, KarcShard weights, ZoneGeometryPod (in their respective private repos).
- [ ] **T4.2** If any substrate fails: open a plan for the fix. Document the failure mode in the related `.docs/` note.
- [ ] **T4.3** Either way, write up the methodology in `.docs/04_calibration/interpolation_geometry.md` (or extend `faithfulness_probe.md`).

## Three-pressure audit (bundled)

For each substrate, run the audit checklist derived from Research 445 §1.3:

- [ ] **Audit Q1 — summarize or route?** Does the latent summarize the underlying trajectory, or is it a lookup key? Test: subsample the trajectory (MAE-drop analog — sparse observations under fog-of-war), recompute the latent, measure latent divergence. A summarizing latent is stable under subsampling; a routing latent diverges.
- [ ] **Audit Q2 — runtime depends on latent?** Does the runtime behavior actually use the committed latent, or does it bypass via raw state? Test: zero/shuffle the latent (intervention battery), measure behavior delta. FaithfulnessProbe (Plan 278) already does this for injected memory; extend to per-entity committed state.
- [ ] **Audit Q3 — local context or full bypass?** Does the runtime's attention to the latent stay local, or does raw context bypass it? Already addressed by SpKv (Plan 070) and RTPurbo (Plan 126) sliding-window infrastructure; audit confirms no substrate accidentally bypasses via global attention.

## Non-goals

- **NOT** implementing the text autoencoder (Stage 1) or MeanFlow generator (Stage 2) — both are training-side → riir-train. The PoC is **evaluation only**, applied to our existing committed latent substrates.
- **NOT** training any new artifacts. The PoC consumes whatever is already shipped.
- **NOT** a quality-parity claim against the paper's TinyStories results (different domain, different substrate). The PoC's claim is **architectural**: does the interpolation-geometry property hold for our substrates?

## Acceptance criteria

- The `LatentSpace` trait + `imauve_score` + `intervention_battery` primitives ship in katgpt-rs (generic, modelless, MIT-eligible).
- A benchmark report documents the interpolation geometry of at least HLA + `NeuronShard::style_weights`.
- A decision is made: close (all pass) or open a fix plan (any fail).

## Cross-references

- [Research 445](../.research/445_Latent_Thought_Flows_Text_Compression.md) — the parent research note.
- [Plan 238](../.plans/238_mux_latent_context_compression.md) — MUX-Latent (closest compression cousin).
- [Plan 278](../.plans/278_cognitive_integrity_layer.md) — FaithfulnessProbe (closest intervention-probe cousin).
- [Plan 321](../.plans/321_sampling_invariant_per_entity_moe_primitive.md) — CommittedFieldBlend.
- [Plan 308](../.plans/308_karc_delay_basis_ridge_forecaster.md) — KARC.
- [Plan 276](../.plans/276_micro_recurrent_belief_state.md) — MicroRecurrentBeliefState / LatentThoughtKernel.
- `riir-neuron-db/.plans/316_nca_neighborhood_heal_primitive.md` — neighbor_heal (closest shard-interpolation cousin).
- `riir-ai/.research/158_per_npc_committed_personality_blend_guide.md` — committed personality runtime (entropy-relocation analog).
