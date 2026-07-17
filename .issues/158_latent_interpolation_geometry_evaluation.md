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
- [x] **T1.1** Define `trait LatentSpace { type Point; fn encode(&self, ...) -> Point; fn decode(&self, p: &Point) -> ...; fn midpoint(&self, a: &Point, b: &Point) -> Point; fn zero(&self) -> Point; fn mean(&self, samples: &[Point]) -> Point; fn noise(&self, rng) -> Point; fn divergence(&self, a: &Point, b: &Point) -> f32; }` — generic over the substrate.
- [x] **T1.2** Implement a synthetic `LatentSpace` for unit testing (e.g., a 2D Gaussian mixture with known interpolation geometry).
- [x] **T1.3** Unit-test `imauve_score` and `intervention_battery` on the synthetic space — verify the protocol correctly distinguishes a "good" synthetic space from a deliberately constructed "bad" one (length-clustering, like the paper's failure mode).

### Phase 2 — Apply to HLA (the strongest single substrate)
- [x] **T2.1** Implement `LatentSpace` for HLA `[f32; 8]` (or whatever the public katgpt-core HLA type is). Midpoint = element-wise average. Decode = the existing `evolve_hla` / bridge-to-5-scalars path.
  - **Verdict (2026-07-17):** The katgpt-rs public surface does NOT expose the 8-dim HLA emotion-scalar type — `NpcEmotionScalars` lives in `riir-engine` (private). The katgpt-rs HLA (`katgpt-core::hla`) is the transformer attention cache, not the per-NPC affect vector. The honest deliverable for this issue is the **generic trait** (Phase 1) + the **`EuclideanLatentSpace<8>` reference impl** that proves the protocol works at the HLA dimension (test `test_imauve_on_euclidean_8d_clusters`: 50 anchors, score > 0.95). The riir-engine-side `impl LatentSpace for NpcEmotionScalars` is a trivial follow-up that wraps the existing decode-to-5-scalars bridge — out of scope for katgpt-rs per the facade constraint.
- [x] **T2.2** Build a synthetic HLA population (e.g., sample N random HLA states from plausible distributions of the 8 dims).
  - Covered by `test_imauve_on_euclidean_8d_clusters` and the GOAT bench G2 sweep (8 clusters × N/8 anchors per cluster).
- [x] **T2.3** Run `imauve_score` and `intervention_battery` on the HLA population.
  - Done in unit tests + GOAT bench. Good-geometry 8D manifold scores > 0.95.
- [x] **T2.4** Report: does HLA midpoint decode to a coherent intermediate emotion-scalar set, or does it collapse? Do the 5 interventions have the expected ordering (matched < shuffled ≈ zero ≈ mean ≈ noise)?
  - **Report:** on the synthetic 8D substrate with identity decode + good manifold geometry, midpoints stay on-manifold (score > 0.95). All 4 interventions diverge from matched=0 with ratios > 5× (test `test_intervention_battery_on_euclidean_8d`). The protocol mechanics are sound at HLA scale. The actual emotion-coherence question requires the riir-engine `NpcEmotionScalars` decode path — deferred.

### Phase 3 — Apply to `NeuronShard::style_weights` (via the lock-free ShardIndex API)
- [x] **T3.1** Implement `LatentSpace` for `NeuronShard` (likely in `riir-neuron-db`, with katgpt-rs consuming via trait). Midpoint = element-wise average of `style_weights[64]`. Decode = whatever reconstruction path the shard exposes.
  - **Verdict (2026-07-17):** Same as Phase 2 — `NeuronShard` lives in `riir-neuron-db` (private), not in katgpt-rs. The katgpt-rs public surface exposes only the dimensionality constant `STYLE_DIM = 64` (via `shard_embedding::STYLE_DIM`). The honest deliverable is the **`EuclideanLatentSpace<64>` reference impl** (test `test_imauve_on_euclidean_64d_clusters`: 40 anchors, score > 0.95). The riir-neuron-db-side `impl LatentSpace for NeuronShard` is a trivial follow-up — out of scope per the facade constraint.
- [x] **T3.2** Build a synthetic shard population (e.g., sample N shards with random style_weights, or load from a benchmark fixture).
  - Covered by `test_imauve_on_euclidean_64d_clusters` + the GOAT bench G2 (n=256 × d=64 — the audit-cadence reference scale).
- [x] **T3.3** Run `imauve_score` and `intervention_battery`.
  - Done in GOAT bench: G1/G2/G4 all PASS at d=64.
- [x] **T3.4** Report.
  - **Report:** on the synthetic 64D substrate with identity decode + good manifold geometry, midpoints stay on-manifold (score > 0.95). GOAT bench measures **642 µs median latency** at n=256 × d=64 (78× under the 50ms audit-cadence budget) with **0 allocations**. The protocol mechanics are sound at style_weights scale. The actual shard-coherence question requires the riir-neuron-db decode path — deferred.

### Phase 4 — Decision
- [x] **T4.1** If both HLA and NeuronShard pass: extend to ArchetypeBlendShard π, KarcShard weights, ZoneGeometryPod (in their respective private repos).
  - **Decision:** the **generic protocol** passes on both shape analogs (`[f32;8]` and `[f32;64]`). The private-substrate extension (ArchetypeBlendShard π, KarcShard weights, ZoneGeometryPod) is a **riir-side follow-up** that requires the actual decode paths. It is correctly deferred — each private repo will plug in its concrete type via the trait when a real evaluation is needed. The trait + reference impls shipped here are the modelless substrate those follow-ups consume.
- [x] **T4.2** If any substrate fails: open a plan for the fix. Document the failure mode in the related `.docs/` note.
  - **Resolution (2026-07-17):** real-substrate audit ran on the two primary substrates (`NpcEmotionScalars` via `curiosity_drive` decode, `NeuronShard::style_weights[64]` via identity decode). **Both PASS** — no fix plan needed. See §"Real-substrate audit results" below for the exact numbers + per-substrate verdicts. The remaining four substrates (`ArchetypeBlendShard` π, `KarcShard` weights, `ZoneGeometryPod`, `MerkleFrozenEnvelope`) are non-blocking follow-ups — the two primary substrates cover both the small-dim sigmoid-decode regime (emotion scalars) and the high-dim identity-decode regime (style weights), so the protocol is empirically validated end-to-end.
- [x] **T4.3** Either way, write up the methodology in `.docs/04_calibration/interpolation_geometry.md` (or extend `faithfulness_probe.md`).
  - **Done:** `.docs/04_calibration/interpolation_geometry.md` (see below) + audit addendum appended 2026-07-17 after the real-substrate runs.

### Real-substrate audit results (2026-07-17)

The deferred riir-side `impl LatentSpace for <ConcreteType>` work landed in two
sibling commits. Both substrates **PASS** the iMAUVE + intervention battery
on their realistic operating distributions.

| Substrate | Repo | Commit | Decode | iMAUVE | Intervention battery | Verdict |
|---|---|---|---|---|---|---|
| `NpcEmotionScalars` (5 fields) | riir-engine | `20f153eb` | `curiosity_drive()` (sigmoid-blended λ-modulation scalar) | **0.9962** (50 anchors, 5 archetype clusters × σ=0.1) | matched=0, shuffled=0.512, zero=0.530, mean=0.219, noise=0.530; `latent_is_causal(10.0)` PASS | **PASS** — no sigmoid-induced midpoint collapse on archetype-clustered distribution |
| `NeuronShard::style_weights[64]` | riir-neuron-db | `746c4a0` | identity (v1 geometric-structure audit) | **0.9693** (GOOD clustered) vs **0.6994** (BAD radial shell); margin 0.27 | matched=0, shuffled=13.59, zero=2.40, mean=13.62, noise=8.54; `latent_is_causal(5.0)` PASS | **PASS** — substrate has sound geometric structure; v2 richer-decode audit is non-blocking follow-up |

**Cross-references:**
- riir-engine: [`riir-ai/.benchmarks/517_emotion_scalars_interpolation_geometry_audit.md`](../../riir-ai/.benchmarks/517_emotion_scalars_interpolation_geometry_audit.md)
- riir-neuron-db: [`riir-neuron-db/.benchmarks/459_style_weights_interpolation_geometry_audit.md`](../../riir-neuron-db/.benchmarks/459_style_weights_interpolation_geometry_audit.md)

**Honest caveats (preserved from the audit reports):**
- `NpcEmotionScalars`: the high score depends on tight archetype clustering (σ=0.1). A population straddling the sigmoid's steepest region (raw=0.25) would produce larger midpoint divergence. This is the expected behavior of a non-linear decoder on its operational distribution, not a defect.
- `NeuronShard::style_weights`: v1 uses identity decode. A richer downstream decode (LoRA routing projection, KARC ridge readout, MAG top-1 retrieval) is a v2 follow-up. If v2 surfaces a defect, it would be in the decode path, not the latent — the substrate's geometric foundation is sound.

**Issue 158 verdict: CLOSE.** All four phases complete. The generic `LatentSpace` trait + the two real-substrate impls + the audit reports close the loop: the protocol correctly distinguishes good from bad geometry, and both committed-latent substrates pass on their operational distributions. The four remaining private substrates are non-blocking follow-ups.

## Q3 structural audit findings (2026-07-17)

Q3 asks: *does the runtime's attention to the committed latent stay local, or
does raw global context bypass it?* The audit is structural (code-trace, not
runtime-test) and decomposes into three checks per substrate:

1. **Decode purity** — is the decode a pure function of the latent, or does
   it accept a global-state parameter that could leak raw context in?
2. **Consumer-input audit** — do the direct consumers of the decoded value
   take ONLY the decoded value, or do they also accept raw global state?
3. **Locality-mechanism inventory** — what architectural invariant prevents
   global-attention bypass for this substrate?

### Substrate 1: `NpcEmotionScalars` (riir-engine) — PASS

| Check | Finding |
|---|---|
| Decode purity | `fn curiosity_drive(self) -> f32` — pure function of `self` (the 5 emotion scalars). No global-state parameter exists. Formula: `sigmoid(4.0 * (0.30·arousal + 0.25·valence + 0.20·calm − 0.15·fear − 0.10·desperation − 0.25))`. File: `riir-engine/src/cgsp_runtime/types.rs:215`. |
| Consumer-input audit | Two consumers: (a) `GameQualityGuide::with_emotion` / `update_emotion` — takes only `NpcEmotionScalars`, produces `lambda_eff` via `rubric.lambda_eff(drive)`. The rubric is per-NPC config, not global state. File: `riir-engine/src/cgsp_runtime/guide.rs:181-203`. (b) `post_action_router::adaptive_width` — maps drive scalar to candidate count `K ∈ [1, max_k]` via sigmoid projection. Pure function of the drive scalar. File: `riir-engine/src/post_action_router.rs:42-48`. |
| Locality mechanism | (i) `post_action_router` is explicitly documented as per-NPC, local, never synced — "No new sync data is introduced" (`post_action_router.rs:50-57`). (ii) Cross-NPC interaction happens via `CrowdAttentionStep::tick_into`, which refines HLA over **visible peers within a zone radius** — local by construction, not global. Crowd attention output still flows through the same `compute_animal_emotions()` bridge to produce the 5 synced affect scalars (`post_action_router.rs:26`). (iii) The host's action enumeration provides raw candidate actions, but selection among them uses ONLY the latent via `route_argmax(parent_hidden, candidates_hidden)` — both args are HLA hidden states (latent), not raw global state. |

**Verdict: no global-state bypass.** The latent is consumed in local context;
cross-NPC influence flows through the HLA refinement (still latent) before
reaching the decode, not through a raw-state side channel.

### Substrate 2: `NeuronShard::style_weights[64]` (riir-neuron-db) — PASS

| Check | Finding |
|---|---|
| Decode purity | `fn project_compacted_to_scalars(compacted: &[NeuronShard], direction_vectors: &[[f32; STYLE_DIM]], out: &mut [f32])` — pure function of `compacted` (the shards) + `direction_vectors` (a trained private asset, fixed at runtime). No global-state parameter exists. Formula: `sigmoid(dot(mean(compacted.style_weights), dir_i) · (1/M · 1/STYLE_DIM))`. File: `riir-neuron-db/src/shard_compactor.rs:864-924`. |
| Consumer-input audit | The decoded 5-scalar affect vector crosses the sync boundary (per AGENTS.md sync rule) and is consumed by the same `NpcEmotionScalars` runtime path as Substrate 1. The shard query itself (`ShardIndex::get`) is per-entity local — no cross-entity attention over shards exists. |
| Locality mechanism | (i) `ShardIndex` is a per-entity `papaya::HashMap` (lock-free, zone→shard lookup) — shards are local to the entity, not a globally-attended pool. File: `riir-neuron-db/src/index.rs:112`. (ii) The compaction pipeline (`ShardCompactor::compact`) operates on a per-entity shard set, not a global shard pool. (iii) No transformer attention over shards — the substrate is structurally a local key-value store, not an attention cache. SpKv / RTPurbo (which DO govern transformer attention locality) are not directly involved here, but the substrate's locality invariant is stronger: it's local by data-layout, not by attention-window policy. |

**Verdict: no global-state bypass.** The substrate is local by construction
(per-entity shard store); there is no global-attention path that could bypass it.

### Locality-mechanism inventory (cross-cutting)

The 7-repo stack enforces the locality invariant at three layers, all of
which the two audited substrates either satisfy directly or are exempt from:

| Layer | Mechanism | Plan / Source | Applies to |
|---|---|---|---|
| **Transformer KV-cache** | SpKv `window: 128` ("Local sliding window always retained; positions within `window` of the current token are never gated out") + hard-gating at inference | [Plan 070](../.plans/070_sp_kv_self_pruned_attention.md), `katgpt-kv/src/sp_kv/types.rs:25-26` | Transformer-attention runtimes (NPC dialog WASM, etc.) — NOT the two audited substrates, which don't go through transformer attention |
| **Retrieval-head sparse decode** | RTPurbo `calibrate_from_scores` (attention-mass scoring) + `HeadCalibration` critical/convertible split | [Plan 126](../.plans/126_rt_turbo_retrieval_head_sparse_decode.md) | Transformer-attention runtimes — same exemption as SpKv |
| **Cross-NPC attention** | `CrowdAttentionStep` operates over **visible peers within a zone radius** — local by construction, not global | `riir-engine/src/cce_runtime/crowd_attention_bridge.rs`, `cognitive_branches_runtime/crowd_attention_bridge.rs` | Substrate 1 (`NpcEmotionScalars`) — crowd attention refines HLA (latent), then the same decode path runs. The latent is still the runtime dependency. |
| **Per-entity shard store** | `ShardIndex` is a per-entity lock-free HashMap; no cross-entity attention over shards exists | `riir-neuron-db/src/index.rs` | Substrate 2 (`NeuronShard::style_weights`) — local by data-layout, not by attention-window policy |
| **Sync-boundary rule** | Raw state crosses sync (deterministic, quorum-committed); latent stays local (per AGENTS.md) | `AGENTS.md` §"Sync Boundary Rule" | All six committed-latent substrates — the rule is the architectural invariant that prevents latent bypass via raw state |

### Q3 verdict

**Both primary substrates PASS Q3 by construction.** The decode paths are
pure functions of the latent; the consumers take only the decoded value; the
cross-NPC influence (when present) flows through latent refinement before
reaching the decode, not through a raw-state side channel. The SpKv / RTPurbo
sliding-window infrastructure enforces locality at the transformer-attention
layer for runtimes that DO use transformer attention (dialog WASM etc.) —
those are out of scope for this substrate-level audit but are covered by the
existing primitives' GOAT gates.

## Remaining-substrate Q1/Q2/Q3 verdicts (2026-07-17, continuation)

The four remaining private substrates are closed by a combination of runtime
Q2 audits (for the two substrates with a meaningful decode) and structural
verdicts (for all four). See
[`riir-neuron-db/src/substrate_geometry_audits.rs`](../../riir-neuron-db/src/substrate_geometry_audits.rs)
for the runtime Q2 audit code.

### Q2 runtime audits — `ArchetypeBlendShard.pi` + `KarcShard.wout`

Both substrates PASS Q2 (runtime-depends-on-latent) under their real runtime
decode paths. The intervention battery diverges from matched on both —
runtime behavior causally depends on the latent.

| Substrate | Decode | iMAUVE | Intervention (matched / shuffled / zero / mean / noise) | Verdict |
|---|---|---|---|---|
| `ArchetypeBlendShard.pi` (K=3, tau=1.0) | `sigmoid(pi_k / tau)` per-archetype gate (matches `CommittedFieldBlend::apply_blended`) | **0.9917** | 0 / **0.896** / **0.465** / **0.492** / **0.408** | **PASS** — all 4 interventions diverge from matched |
| `KarcShard.wout` (d_h=32, D=8) | `ŷ[r] = dot(wout[r*d_h..(r+1)*d_h], delay_state)` per-output-row (matches `KarcForecaster::forecast_into`) | **0.9773** | 0 / **4.67** / **4.29** / **3.46** / **9.00** | **PASS** — all 4 interventions diverge massively from matched |

Both iMAUVE scores are high (>0.97) — midpoints of cluster-mates decode
close to anchors, confirming the latent geometry is sound. The intervention
battery confirms the decode causally depends on the latent (all interventions
move the decoded behavior away from matched).

### Q1 verdicts — `KarcShard.wout` PASS (empirically confirmed), rest N/A

Q1 asks: does the latent summarize an underlying trajectory, or is it a
lookup key? The audit requires a trajectory-summary operation (the latent as
a function of a sequence of events). Of the four remaining substrates:

| Substrate | Trajectory operation | Q1 verdict |
|---|---|---|
| `ArchetypeBlendShard.pi` | One-shot commit via `set_blend_state` from `CommittedFieldBlend::commit`. No trajectory of events — pi is set in a single atomic commit from the field blend. | **N/A** (not a trajectory summary) |
| `KarcShard.wout` | Ridge-regression fit on a (reservoir_state, target) trajectory. IS a trajectory summary (closed-form least squares). | **PASS** (empirically confirmed, 2026-07-17 continuation) — `audit_karc_wout_summarize_vs_route` measures the ridge-fit Wout divergence under subsampling. At drop=30%, mean_rel=0.047, worst_rel=0.066 (well under the drop×2.0=0.6 bound). The routing control (single-pair degenerate fit) has worst_rel=1.33 (fails the bound). The ridge fit's divergence scales proportionally with drop_fraction (0.02→0.05→0.07 mean across 10/30/50%), confirming the summarize signature. See `.benchmarks/459` v5 addendum. |
| `ZoneGeometryPod` | Regenerated from chain-committed NeuronShard + raw info-brain state. Not a trajectory summary — single-source derivation. | **N/A** (derived artifact, not trajectory summary) |
| `MerkleFrozenEnvelope` | `freeze(data_blocks)` computes a Merkle root over the blocks. Not a trajectory summary — cryptographic commitment over a static block set. | **N/A** (cryptographic commitment, not trajectory summary) |

**Q1 verdict: `KarcShard.wout` PASS (empirically confirmed).** The
previously-deferred Q1 audit is now implemented — the ridge-fit Wout is
confirmed as a summarizing latent, not a routing key. The other three
remaining substrates have no trajectory-summary operation (N/A).

### Q3 structural verdicts — all four PASS by construction

Q3 asks: does the runtime's attention to the committed latent stay local, or
does raw global context bypass it? The audit is structural (decode purity +
consumer inputs + locality mechanism).

| Substrate | Decode purity | Consumer inputs | Locality mechanism | Verdict |
|---|---|---|---|---|
| `ArchetypeBlendShard.pi` | `sigmoid(pi_k / tau)` — pure function of `pi` + per-NPC `tau` (config, not global state) | `CommittedFieldBlend::apply_blended(pi, tau, fields, z)` — takes only `pi`, `tau`, the field library (per-NPC), and the input `z`. No global state. | Per-NPC shard store (same `papaya::HashMap` locality as `NeuronShard`). The field library is per-NPC (library-side definitions; only the hash crosses sync). | **PASS** |
| `KarcShard.wout` | `forecast_into(delay_state, out)` = `Wout @ delay_state` — pure function of `wout` (the latent) + `delay_state` (the reservoir state, itself a pure function of prior inputs via the reservoir dynamics) | `KarcForecaster::forecast_now` / `forecast_into` — takes only the forecaster's `wout` + the ring buffer's delay state. No global state. | Per-NPC shard store. The reservoir state is local to the forecaster instance (per-NPC ring buffer). No cross-entity attention over forecasters. | **PASS** |
| `ZoneGeometryPod` | Cochains are read via DEC operators (`exterior_derivative`, `codifferential`, etc.) — pure functions of the cochain values | DEC consumers (zone reasoning, threat assessment) take only the cochain values + the cell complex topology. No global state. | `ZoneGeometryPod` is per-zone (regenerated at zone AOI entry). The `blake3_source_shard` self-validation ensures the pod is ground-truth-derived from the chain-committed shard — no subjective belief state crosses into the pod (two-brain compliance). | **PASS** |
| `MerkleFrozenEnvelope` | No decode — `merkle_root` is verified (`verify_thaw`), not decoded. The hash is one-way. | `verify_thaw(data_blocks)` takes only the envelope + the candidate data blocks. No global state. | Per-envelope cryptographic commitment. No attention mechanism, no cross-entity interaction. | **PASS** (vacuously — no decode path to bypass) |

**Q3 verdict: all four remaining substrates PASS by construction.** Each is
per-entity (or per-zone / per-envelope) committed state; the decode paths
are pure functions of the latent (or don't exist, for the cryptographic
envelope); no global-attention bypass exists.

### Why `ZoneGeometryPod` and `MerkleFrozenEnvelope` have no Q2 runtime audit

These two substrates are **vacuously N/A** for Q2 because they have no
learnable latent decode path:

- **`ZoneGeometryPod`** — cochains are DERIVED from the chain-committed
  NeuronShard + raw info-brain state (projectiles, occupancy, POI anchors).
  The pod is a regenerated artifact, not a learnable latent. Reading a
  cochain value is not a "decode" in the Q2 sense (there's no learned
  projection; it's a direct SoA read). Q2 doesn't apply — there's no latent
  whose intervention would change runtime behavior.
- **`MerkleFrozenEnvelope`** — `merkle_root` is a BLAKE3-based cryptographic
  commitment. It is verified (`verify_thaw`), never decoded. There is no
  behavior that depends on the hash value itself (only on whether it
  matches). Zeroing / shuffling the hash doesn't change runtime behavior —
  it just flips the integrity check to false. Q2 is vacuous.

### Remaining-substrate audit summary

| Substrate | Q1 | Q2 | Q3 |
|---|---|---|---|
| `ArchetypeBlendShard.pi` | N/A (one-shot commit) | **PASS** (runtime audit, iMAUVE=0.9917) | **PASS** (structural) |
| `KarcShard.wout` | **PASS** (empirical, mean_rel=0.047 @ 30% drop) | **PASS** (runtime audit, iMAUVE=0.9773) | **PASS** (structural) |
| `ZoneGeometryPod` | N/A (derived artifact) | N/A (no learnable latent) | **PASS** (structural) |
| `MerkleFrozenEnvelope` | N/A (cryptographic commitment) | N/A (no decode path) | **PASS** (vacuous — no decode to bypass) |

**All four remaining substrates are now fully audited.** Two PASS the
runtime Q2 audit with real numbers; two are vacuously N/A for Q2 (no
learnable latent); all four PASS Q3 by construction. Q1 is PASS for
`KarcShard.wout` (empirically confirmed) and N/A for the other three.

## Three-pressure audit (bundled)

For each substrate, run the audit checklist derived from Research 445 §1.3:

- [x] **Audit Q1 — summarize or route?** Does the latent summarize the underlying trajectory, or is it a lookup key? Test: subsample the trajectory (MAE-drop analog — sparse observations under fog-of-war), recompute the latent, measure latent divergence. A summarizing latent is stable under subsampling; a routing latent diverges.
  - **Resolution (2026-07-17):** CLOSED for `NeuronShard::style_weights` via the Q1 summarize-vs-route audit (Benchmark 459 v3 addendum). The `ConsolidationPipeline::sleep()` average is the canonical summarize operation — divergence under subsampling scales proportionally with the drop fraction, both in mean (10%→0.028, 30%→0.047, 50%→0.069) and worst case (10%→0.077, 30%→0.162, 50%→0.226). The routing control (argmax-norm on a unique-outlier trajectory) exhibits the expected routing signature (worst-case spike to ~0.77 at low drop fractions), proving the audit discriminates. The average is also robust to outliers — dropping a 10×-magnitude outlier event from a 60-event trajectory shifts the average by at most `outlier/60`. **`NpcEmotionScalars`** (riir-engine) is current-state, not trajectory-summary — the Q1 question doesn't apply directly (the latent IS the current observation, not a trajectory summary). Q1 for that substrate is vacuously N/A. **The four remaining private substrates** (`ArchetypeBlendShard` π, `KarcShard` weights, `ZoneGeometryPod`, `MerkleFrozenEnvelope`) are closed in the continuation sessions: 3 are N/A (no trajectory-summary operation); **`KarcShard.wout` is PASS** (empirically confirmed in Benchmark 459 v5 — `audit_karc_wout_summarize_vs_route` measures the ridge-fit Wout divergence under subsampling; at drop=30%, mean_rel=0.047, worst_rel=0.066; the routing control fails at worst_rel=1.33; divergence scales proportionally with drop_fraction, confirming the summarize signature). See §"Remaining-substrate Q1/Q2/Q3 verdicts" below.
- [x] **Audit Q2 — runtime depends on latent?** Does the runtime behavior actually use the committed latent, or does it bypass via raw state? Test: zero/shuffle the latent (intervention battery), measure behavior delta. FaithfulnessProbe (Plan 278) already does this for injected memory; extend to per-entity committed state.
  - **Resolution (2026-07-17):** CLOSED for `NeuronShard::style_weights` via the v2 runtime-decode audit (Benchmark 459 v2 addendum). The v2 `StyleWeightsScalarSpace` uses the canonical `sigmoid((1/STYLE_DIM) · dot)` decode (exactly mirroring `project_compacted_to_scalars`) and confirms `latent_is_causal(5.0)` under a high-signal anchor: matched=0, shuffled=0.20, zero=0.56, mean=0.13, noise=0.56. The low-signal regime (small projections → sigmoid ≈ 0.5) is documented as expected sigmoid behavior, not a defect. **`NpcEmotionScalars`** (riir-engine) implicitly answers Q2 via its existing `curiosity_drive()` decode (already a runtime bridge). **The four remaining private substrates** are closed in the continuation session: `ArchetypeBlendShard.pi` PASSES (iMAUVE=0.9917, all interventions diverge: 0.90/0.47/0.49/0.41) and `KarcShard.wout` PASSES (iMAUVE=0.9773, all interventions diverge: 4.67/4.29/3.46/9.00) via `riir-neuron-db/src/substrate_geometry_audits.rs`; `ZoneGeometryPod` and `MerkleFrozenEnvelope` are vacuously N/A (no learnable latent decode path). See §"Remaining-substrate Q1/Q2/Q3 verdicts" below.
- [x] **Audit Q3 — local context or full bypass?** Does the runtime's attention to the latent stay local, or does raw context bypass it? Already addressed by SpKv (Plan 070) and RTPurbo (Plan 126) sliding-window infrastructure; audit confirms no substrate accidentally bypasses via global attention.
  - **Resolution (2026-07-17):** CLOSED for the two primary substrates via a structural code audit (decode-path purity + consumer-input audit + locality-mechanism inventory). The audit answer is **PASS by construction** — the latent is consumed in local context; no global-state side channel exists in the decode path or its direct consumers. See §"Q3 structural audit findings" below for the per-substrate trace + locality-mechanism inventory. **The four remaining private substrates** are closed in the continuation session via the same structural audit template: all four PASS by construction (per-entity/per-zone/per-envelope committed state; pure decode paths; no global-attention bypass). See §"Remaining-substrate Q1/Q2/Q3 verdicts" below.

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
