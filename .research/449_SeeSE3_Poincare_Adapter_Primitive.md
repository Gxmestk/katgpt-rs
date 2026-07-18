# Research 449: SeeSE3 — Poincaré Adapter & Closed-Form Latent Navigation Primitive

> **Source:** Chen, Ebrahimi, Kitashov, Yang, Guibas, Pătrăucean, Ovsjanikov — *SeeSE3: Emergence of 3D Space in Vision Features* — [arXiv:2607.14228](https://arxiv.org/abs/2607.14228) (Google DeepMind, 15 Jul 2026).
> **Date:** 2026-07-18
> **Status:** Active — Super-GOAT (4/4 novelty gate PASS). Open primitive + private game-runtime guide + plan all in this session.
> **Related Research:** 290 (Latent Field Steering — additive injection; this is the *closed-form inverse* for `Δz → ΔP`), 294 (Viable Manifold Graph — graph-based latent navigation; this is the *linear-algebra* complement), 235 (SLoD — already ships `poincare_distance`/`log_map`/`exp_map` for KG abstraction), 271 (MIT 6S184 Diffusion/Flow Vocabulary Crosswalk), 382 (Spherical Geodesic Steering — Slerp geodesic on sphere)
> **Related Plans:** 449 (this primitive — open), 309 (Latent Field Steering — companion), 312 (Viable Manifold Graph — graph cousin), 235 (SLoD — Poincaré ball substrate), 405 (Spherical Geodesic Steering)
> **Cross-ref (riir-ai):** Research 319 — `SeeSE3_Latent_Imagination_Game_Runtime_Guide.md` (private Super-GOAT moat)
> **Classification:** Public — generic closed-form linear navigator math. No game IP (the NPC "imagine observation" selling point is private, lives in riir-ai/.research/319).

---

## TL;DR

The paper proves that vision foundation models *passively* encode the geometry of SE(3) (the group of rigid 3D transformations) inside their latent features — but only after a lightweight Siamese "Poincaré Adapter" `φ` "unrolls" the curved feature manifold into a flat homogeneous coordinate patch. Once unrolled, **camera motion becomes linear in latent displacement**: `ΔP ≈ W·(φ(z_{t+s}) − φ(z_t))`, where `W` is a fixed 6×d linear readout and `ΔP ∈ ℝ⁶` is the SE(3) Lie-algebra twist. This linearization is so tight that **inverse navigation is closed-form and training-free at inference**: given a current feature `g_t` and a desired physical displacement `ΔP`, the destination feature is `ĝ_{t+s} = g_t + W†·ΔP` (Eq. 4 of the paper). Multi-step open-loop trajectories of 4+ steps stay coherent.

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is **a closed-form linear bridge between latent displacement and an externally-defined target space** (camera pose in the paper; for us: SE(2) game-map pose, HLA target state, or any other low-dim physical/semantic target). It has three pieces, all inference-time and training-free after one offline fit:

1. **Poincaré Adapter** — a Siamese `φ: ℝᵈ → ℝ^{d'}` (small MLP, ~2-layer, ~64 hidden) such that, after the adapter, the latent displacement is **linearly readable** as the target: `target_diff ≈ W·(φ(z₂) − φ(z₁))`. The adapter "unrolls" the curved latent manifold into a flat homogeneous coordinate chart.
2. **Inverse Poincaré Navigator** — `z_dest = z_src + W†·Δtarget`. Zero-training, zero-allocation, SIMD-acceleratable. The pseudo-inverse `W† ∈ ℝ^{d'×6}` is precomputed once at adapter-fit time. This is the inference-time primitive.
3. **Multi-step open-loop integration** — split `Δtarget` into `S` sub-steps, iterate `z ← z + W†·(Δtarget/S)` `S` times. Recovers coherent trajectories for moderate displacements; drift accumulates on extreme paths.

This is the **closed-form-completion companion** to Latent Field Steering (R290): LFS pushes latent state top-down via an additive direction vector; Poincaré answers the inverse question "given the desired target movement, what latent step realizes it?". It is also the **linear-algebra cousin** of Viable Manifold Graph (R294): VMG navigates a discrete graph of viable latent nodes via A*; Poincaré navigates a continuous flat chart via a single matvec. The two are complementary — VMG for highly-non-linear manifolds (NPC affect), Poincaré for manifolds with a clean linear structure after unrolling (raw pose ↔ latent displacement).

**Theorems** (paper §A.7, distilled):

- **Theorem 3 (local decodability always exists).** For any smooth encoder `f` with rank-6 Jacobian at `g₀`, a linear left-inverse `W` exists locally such that `W·(f(g) − f(g₀)) = log(g₀⁻¹·g) + O(‖·‖²)`. Trivial by Taylor. **The hard question is global.**
- **Theorem 5 (global linear decodability obstruction).** A single `W` works across a region `K` iff the Jacobian field `J(g)` is constant modulo `ker W`. The obstruction is **manifold curvature** (Theorem 5b): error scales with `diam(K) · sup ‖∇J‖`. The nonlinear adapter `φ` unrolls this curvature entirely (Theorem 5c), leaving only an intrinsic residual `ε_Lie = ½[ξ, X] + …` from the non-commutativity of SE(3) — bounded by `O(‖ξ‖ · R_K)`, vanishes for pure translations.
- **Theorem 7 (rotation vs translation asymmetry).** Rotational optical flow is **depth-independent** (depends only on pixel coords + focal length); translational flow scales as `1/Z`. So translation magnitude is harder to decode than rotation (geometric, not algorithmic). Implication: Poincaré navigators for "facing/rotation" targets will be tighter than for "translation magnitude" targets.

---

## 1. Paper Core Findings

### 1.1 The "Poincaré Task" — can a motionless observer discover 3D space?

The paper poses (and answers YES to) Poincaré's hypothesis: *a motionless being cannot construct the concept of space*. The trick: vision foundation models (DINOv2, DINOv3, V-JEPA 2.1, DUSt3R) trained only on passive observation *nevertheless* organize their latent features such that a lightweight adapter can recover SE(3) camera motion from feature differences alone — bypassing explicit 3D reconstruction entirely.

### 1.2 Four progressively-restrictive probes (Metrics M1–M4)

| Probe | Question | Training-free? |
|---|---|---|
| **M1 Mutual k-nn Alignment** | Do feature nearest-neighbors match pose nearest-neighbors? | ✅ Yes |
| **M2 Intrinsic Dimensionality** | Is the latent manifold locally 6D? (MLE / TwoNN) | ✅ Yes |
| **M3 Linear Equivariance** | Does a single global `W` satisfy `ΔP ≈ W·Δz`? | ✅ Yes (Ridge regression) |
| **M4 Poincaré Adapter** | After a Siamese φ unroll, does `ΔP ≈ W·(φ(z₂) − φ(z₁))` hold? | ❌ φ needs offline fit |

**M1 + M2 pass without any training** for DINOv2/v3, V-JEPA, CroCo, DUSt3R — the topology is *natively* present. **M3 fails for every model** (negative R²) — no encoder is natively linearly equivariant; the manifold is curved. **M4 (the adapter) closes the gap**: DINOv2 reaches R² ≈ 0.61 on ScanNet, V-JEPA 2.1 reaches 0.803 on 12-Scenes. The adapter is small (2-layer MLP, 64-dim hidden, 20-dim bottleneck, ~10K params).

### 1.3 Inverse Poincaré for Latent-Space Navigation (the headline)

Given an adapted feature `g_t = φ(z_t)` and a desired pose change `ΔP`, predict the destination feature:
```
ĝ_{t+s} = g_t + W† · ΔP      (paper Eq. 4)
```
This is **training-free at inference** — `W†` is the pre-computed pseudo-inverse of `W`. On ScanNet (DINOv2 backbone), this beats a learned Attention corrector on topological retrieval (Hit@0.5: 0.741 vs 0.422) despite worse MSE. **The linear structure dominates at the neighborhood level.**

Multi-step: split `ΔP` into `S` equal sub-steps, iterate `ĝ ← ĝ + W†·(ΔP/S)`. 4-step paths to moderate displacements (0.68m, 19°) reach the target frame exactly; extreme displacements (2.45m, 37°) stay rotation-coherent (<5° error) with translation drift.

### 1.4 Generalization (the Ridge-Refit trick)

Zero-shot `W†` succeeds on ~60% of unseen test rooms. **Ridge-Refit** — re-fit only the linear `W` on the new room given a frozen φ adapter — pushes success to **90–94% across DINOv2, DUSt3R, V-JEPA**. This is the proof that the adapter discovers a *transferable* geometric manifold, not a room-specific fit.

### 1.5 Difficulty hierarchy (geometric, not algorithmic)

Across 38,929 adapter runs (DINOv2-B Layer 7, ScanNet):
- Rotation R² > 0 in **88%** of configurations, seed σ = 0.17
- Translation direction R² > 0 in **60%**, σ = 0.30
- Translation magnitude R² > 0 in only **27%**, σ = 1.03 (6× rotation variance)

Theorem 7 proves this is geometric: rotational optical flow is depth-independent; translational flow scales as `1/Z`. Translation magnitude is *unidentifiable without metric depth priors* — the encoder cannot disambiguate `(v, Z) ~ (λv, λZ)`. **Implication: Poincaré navigators for facing/rotation targets will be tighter than for translation-magnitude targets** — a design constraint that propagates to every consumer.

### 1.6 What does NOT transfer (training-only)

- The adapter φ itself must be **fit once offline** (Ridge regression + 10-epoch AdamW, ~minutes). The *fitting* is a training-method question → **riir-train** (or, for modelless fitting, a deterministic closed-form ridge solver — see §3.5 below).
- Foundation-model pretraining (DINOv2's 142M images) is the substrate; the adapter is the technique. We have no foundation vision model. Our substrate is **HLA's 8-dim per-NPC latent state**, the **SenseModule 8-dim projection**, and any **frozen NeuronShard embedding** — all of which are candidates for the "z" the adapter unrolls.

---

## 2. Distillation

### 2.1 What's already in katgpt-rs (verified by notes + code grep, 2026-07-18)

| Paper concept | Existing codebase analogue | Status |
|---|---|---|
| Poincaré ball geometry (distance, log/exp map, Fréchet mean) | `poincare_distance`, `log_map_into`, `exp_map`, `frechet_mean` in `katgpt-core/src/slod.rs` (Plan 235) | ✅ Shipped DEFAULT-ON — used for KG abstraction-level selection. **Different use case** (KG LOD, not latent navigation). |
| Top-down additive direction injection | `apply_latent_steering(state, field)` (Plan 309, R290) | ✅ Shipped DEFAULT-ON, Super-GOAT. **The forward direction**: given a direction vector `v`, push state by `s + α·v`. Poincaré is the **inverse**: given a desired target movement `ΔP`, find the latent step `Δz` that achieves it. |
| Linear-algebra navigation on a graph | `manifold_geodesic`, `manifold_random_walk` in `viable_manifold_graph.rs` (Plan 312, R294) | ✅ Shipped (DEFAULT-ON as part of latent_functor). **Graph-based A* on discrete viable nodes**. Poincaré is **continuous closed-form** — one matvec vs path search. Complementary, not duplicate. |
| Multi-axis fixed-strength subspace projection | `subspace_steering` (Plan 412) | ✅ Shipped DEFAULT-ON. Projects a delta onto orthogonal basis axes. Different shape — Poincaré's `W` is **non-orthogonal** (it's a left-inverse of the Jacobian, not an orthonormalization). |
| Alignment-gated invariant-subspace correction | `tilr_invariant_subspace` (Plan 425) | ✅ Shipped DEFAULT-ON. Step-size `η = η_base · γ` where `γ = ‖Πd‖/‖d‖`. **The TILR `γ` is the modelless analog of Poincaré's "Jacobian rank-6" check** — both gate on "the latent step lives in the decodable subspace". |
| Rotation-equivariant perception | `se2_equivariant` in `riir-engine/src/equivariant/` (Plan 354, R166, DEFAULT-ON in riir-ai) | ✅ Shipped private. Lift → group-conv → project pipeline for SE(2) equivariance. **Poincaré is the action→latent inverse**; SE(2) is the perception-equivariance forward. They compose: SE(2) gives equivariant features; Poincaré navigates inside them. |
| Pseudoinverse `W†` as a deterministic navigator | None shipped. `katgpt-core/src/linear.rs` has matmul helpers; `thin_svd_into` (Plan 301) gives the SVD from which `W† = V·Σ⁻¹·Uᵀ` is one line. | ❌ **MISSING.** The closed-form inverse map is the gap. |
| Frozen `MerkleFrozenEnvelope` for the fitted adapter | `MerkleFrozenEnvelope` (riir-neuron-db, Plan 007) | ✅ Shipped in neuron-db (not katgpt-rs — chain-bridge concern). The fitted `φ + W + W†` triple is a frozen artifact. **Pattern exists; Poincaré-specific wrapper does not.** |

### 2.2 What's NOT in katgpt-rs (the gap — three missing primitives)

1. **`PoincareAdapter` Pod** — `#[repr(C)]` frozen struct `{ phi_weights: [f32; ...], phi_biases: [f32; ...], W: [f32; 6·d'], W_pinv: [f32; d'·6], target_kind: u8, blake3: [u8; 32] }`. Holds the offline-fit triple `(φ, W, W†)`. BLAKE3-committed over the weights. Zero-copy mmap-able. **No shipped equivalent** — this is a *new* Pod kind (frozen linearizing chart).
2. **`poincare_navigate_into(z_src, delta_target, adapter, z_dest_out)`** — the closed-form navigator. Computes `φ(z_src) + W†·Δtarget`, then optionally projects back to the original latent space (via `φ⁻¹` if the adapter is invertible, or via nearest-neighbor retrieval in the adapted chart). **No shipped equivalent.**
3. **`poincare_multi_step_into(z_src, delta_target, S, adapter, z_out)`** — split `Δtarget` into `S` sub-steps, iterate the navigator. Open-loop integrator. **No shipped equivalent.**

### 2.3 Closest cousins (3)

1. **Latent Field Steering (R290 / Plan 309)** — `apply_latent_steering(state, field) = s + α·v`. **Sibling, not duplicate**: LFS pushes state in a designer-chosen direction; Poincaré pushes state to realize a designer-chosen *target movement*. LFS answers "go that way"; Poincaré answers "what latent step corresponds to this physical movement?". The two compose: LFS for steering without a target; Poincaré for steering toward a known target in a different space.
2. **Viable Manifold Graph (R294 / Plan 312)** — `manifold_geodesic(g, src, dst)` (A* on safe subgraph). **Structural cousin, different substrate**: VMG is discrete graph search over viable latent nodes (handles highly curved manifolds); Poincaré is continuous closed-form over a flat chart (handles manifolds that admit linear unrolling). **Selection rule**: if the target space is low-dim (≤6) and the encoder's Jacobian has rank ≥ target_dim → Poincaré (closed-form, ~µs). If the manifold is highly curved or the target is high-dim → VMG (graph search, ~ms). **They compose**: Poincaré navigates *within* a chart; VMG navigates *between* charts.
3. **TILR alignment-gated correction (R408 / Plan 425)** — `η = η_base · ‖Πd‖/‖d‖`. **Methodological cousin**: TILR gates the correction step by how much the direction lives in the invariant subspace; Poincaré's `W†·Δtarget` is *only* the in-subspace component by construction (it's a projection onto `row(W)`). TILR's `γ` is a soft version of Poincaré's hard projection. **Compose**: TILR can gate Poincaré's output by the local alignment ratio for a safety margin.

### 2.4 Fusion — what novel combination does this enable?

**Fusion A (PRIMARY — Super-GOAT, see riir-ai/.research/319): Poincaré × two-brain model × sleep_time anticipator → "NPCs imagine observations at unvisited positions"**

Today the two-brain model (per AGENTS.md) has an asymmetry: the **info brain** has real `MapPos` (synced, ground truth); the **think brain** has `SpatialBelief { zone KG triple + stale last_known_pos + confidence }`. The think brain can *remember* and *forget* but **cannot imagine** — it has no mechanism for "what would I see/feel if I were at position (x', y')?". This forces every planning step to either (a) commit to a real movement (info brain) or (b) rely on stale memory.

Poincaré closes this gap. With an adapter fit offline over `(MapPos_t, HLA_state_t)` pairs from observed trajectories, an NPC can predict `HLA_state_{t+s}` at an unvisited position `MapPos_{t+s}` via a single matvec: `ĥ' = h_t + W†·(MapPos_{t+s} − MapPos_t)`. This is the **"imagine" half** of the two-brain model. Sleep-time anticipation (Plan 341, today anticipates *dialog queries*) extends to *anticipate HLA states across the spatial map* during idle cycles. The imagined HLA states seed plan-tree rollouts (MCTS over imagined affects) without ever moving the NPC.

**Fusion B (open, secondary): Poincaré × Spherical Geodesic Steering (R382/Plan 405) × Committed Field Blend (R302/Plan 321) → "linear-chart navigation across personality snapshots"**

A NeuronShard's `style_weights[64]` is a frozen latent direction. Each shard defines a personality. Today, blending two personalities uses `CommittedFieldBlend`'s convex sum `Σ sigmoid(π_k)·f_k`. Poincaré gives an alternative: fit an adapter `φ_shard` such that `ΔPersonalityIdx ≈ W·(φ(s₂) − φ(s₁))`. Then "shift personality 0.3 toward archetype B" becomes a closed-form step `s + W†·0.3`. Composes with Spherical Steering's Slerp for the geodesic correction.

**Fusion C (open, secondary): Poincaré × InducedCwmKernel × Motor-Gated DEC (R168/Plan 357) → "frozen imagination kernel"**

InducedCwmKernel (Plan 296) is a frozen world model with `advance(state, action) → next_state`. Today `advance` is game-rule transitions. With Poincaré, `advance` can be a *learned linear chart* over `(state, action)` pairs — closing the loop between Motor-Gated DEC's field evolution and InducedCwm's frozen advance. **The frozen `φ + W` triple IS the InducedCwm's parameter set**, BLAKE3-committed via the existing `canonical_bytes()` pattern.

---

## 3. Verdict

### 3.1 Novelty Gate (Q1–Q4) — RUN 2026-07-18

### Q1: No prior art? — **PASS** (zero hits for the closed-form linear navigator)

**Paper-vocabulary grep** (`poincare.adapter|latent.navigation|visual.odometry|imagine.observation|inverse.poincare|W†·ΔP|closed-form.navigator`): zero hits across all 7 repos, both layers (notes + code).

**Codebase-vocabulary grep** (`pseudoinverse|pinv|W†|left.inverse|jacobian.inverse|manifold.unroll|homogeneous.coordinate|linear.chart`): zero hits in latent-navigation context. Closest matches are unrelated:
- `katgpt-core/src/slod.rs` — Poincaré ball distance/log/exp (KG abstraction LOD, **not** latent navigation).
- `thin_svd_into` (Plan 301) — gives the SVD; `W† = V·Σ⁻¹·Uᵀ` is one line of post-processing that's **not exposed** as a navigator.
- `pathfinder.rs` (Plan 017/018) — raw-grid A* (graph search, not closed-form).
- `manifold_geodesic` (Plan 312) — graph A* on viable latent nodes (graph search, not closed-form).
- `apply_latent_steering` (Plan 309) — additive push (forward direction; Poincaré is the inverse).

**Verdict Q1:** GENUINELY NOT SHIPPED. The closed-form `z + W†·Δtarget` linear navigator is a new primitive. Existing pieces cover *either* the forward direction (LFS) *or* graph-based navigation (VMG) *or* the Poincaré-ball metric (SLoD); none ships the linearizing adapter + inverse navigator pair.

### Q2: New class of behavior? — **PASS**

Not better numbers — a new capability. Today no shipped mechanism answers "given a desired target movement in an externally-defined space (pose, affect, personality), what latent step realizes it?" — closed-form, training-free at inference. LFS gives "push by a direction"; VMG gives "find a path"; SE(2) gives "rotate-equivariantly perceive"; Motor-Gated DEC gives "evolve a field". **None gives "imagine the latent state at a target without visiting it".** Poincaré adds this.

### Q3: Product selling point? — **PASS**

Finish the sentence: *"Our NPCs imagine what they would see and feel at unvisited positions — closing the two-brain model's imagination gap — by a single matvec per step, no full cognition pipeline, no training at inference, with the adapter frozen and BLAKE3-committed so the imagination is quorum-verifiable across nodes."* Defensible, demoable, no competitor ships it.

### Q4: Force multiplier? — **PASS** (≥8 systems)

Connects to:
1. **HLA kernel** (`katgpt-core/src/sense/reconstruction.rs`) — `evolve_hla` becomes *imagined-able*; the NPC can predict its own HLA at unvisited positions.
2. **Two-brain model** (per AGENTS.md) — closes the "imagine" gap.
3. **Latent Field Steering** (Plan 309) — Poincaré is the inverse direction.
4. **Viable Manifold Graph** (Plan 312) — complementary (continuous vs discrete navigation).
5. **SE(2) equivariant maps** (R166/Plan 354) — compose: SE(2) gives equivariant features; Poincaré navigates inside them.
6. **InducedCwmKernel** (Plan 296) — the adapter triple *is* a frozen Cwm parameter set.
7. **Motor-Gated DEC** (Plan 357) — closes the loop between field evolution and imagination.
8. **Sleep-time anticipator** (Plan 341) — extends from dialog-query anticipation to HLA-at-position anticipation.
9. **Committed Field Blend** (Plan 321) + **Spherical Steering** (Plan 405) — Poincaré gives the linear-chart alternative for personality drift.
10. **MerkleFrozenEnvelope** (riir-neuron-db) — the adapter triple freezes via existing commitment pattern.

**Novelty gate verdict: 4/4 YES → Super-GOAT.** Per the research skill's mandatory outputs, this triggers (a) this open primitive note, (b) private game-runtime guide in riir-ai/.research/319 (created in same session), (c) plan in katgpt-rs/.plans/449 (created in same session).

### 3.2 MOAT Gate — Domain Fit

| Concern | Verdict |
|---|---|
| Is this a `katgpt-rs`-tier primitive? | **YES** — generic closed-form linear-algebra bridge between latent displacement and target space. No game IP, no chain IP, no shard IP. Target space is generic (any low-dim physical/semantic coordinate). |
| Public adoption value | **YES** — closed-form linear navigation after offline adapter fit is a fundamental inference primitive. Any consumer of latent state (LLM embeddings, robot pose, agent state) can use it. |
| Risk of leaking private IP | **LOW** — the open primitive is `poincare_navigate_into(z_src, delta_target, adapter, z_out)`. The private IP (the NPC "imagine observation" selling point, the HLA ↔ MapPos adapter fit, the two-brain imagination loop) stays in `riir-ai/.research/319`. |
| Strengthens katgpt-rs moat? | **YES** — adds a new capability class (closed-form latent navigation) to the engine's modelless-inference stack. Pairs with LFS (forward) and VMG (graph) to cover all three navigation regimes. |

**Verdict:** open primitive stays in katgpt-rs. Private guide in riir-ai. Adapter freezing reuses MerkleFrozenEnvelope pattern (cross-repo dependency via riir-neuron-db's existing re-export).

---

## 4. Latent-space reframings (mandatory per workflow §1.5 step 3)

The paper operates on vision features + SE(3) camera pose. Our reframings across the 7 Super-GOAT factory modules:

### 4.1 HLA per-NPC latent state (`katgpt-rs/crates/katgpt-core/src/sense/`)

HLA is `R⁸` (valence, arousal, desperation, calm, fear + 3 reserved). The "target space" is the game map `MapPos2D ∈ R²` (or `MapPos3D ∈ R³` if height matters). Fit a Poincaré adapter `(φ_hla, W_hla)` over `(MapPos_t, HLA_state_t)` pairs from observed NPC trajectories. Then an NPC at `(x, y)` with HLA state `h` can imagine its HLA at unvisited `(x', y')` via `ĥ' = φ_hla⁻¹(φ_hla(h) + W_hla†·((x',y') − (x,y)))`. This is the **missing think-brain imagination primitive**.

**Difficulty hierarchy (per Theorem 7)**: facing-direction changes are easier to predict than position-magnitude changes. NPCs will imagine their affect at "the same spot but rotated 90°" more reliably than at "5 meters north". Designers should author imagination queries that lean on the easier component.

### 4.2 `latent_functor/` (`zone_gating`, `reestimation`, `arithmetic`, ...)

A functor application `apply_functor(src, dst, f)` is a map `Rᵈ → Rᵈ`. Today functors are point-to-point — they don't support "imagine the functor's output at a hypothetical source". With a Poincaré adapter fit over `(source, f-application)` pairs, the functor becomes *imagined-able*: predict the output at an unobserved source. This is a **new capability** for the latent functor runtime — closes the gap between `reestimation`'s reactive re-derivation and proactive imagination.

### 4.3 `cgsp_runtime/` (curiosity-guided self-play)

Curiosity drives exploration. Today, exploration is a free Gaussian step in latent space — undirected. With a Poincaré adapter over `(position, curiosity_state)`, an NPC can imagine its curiosity at unvisited positions and **direct exploration toward high-curiosity regions** — a closed-form curiosity map. This is the curiosity analog of Fusion A.

### 4.4 LatCal fixed-point commitment (`riir-chain/src/encoding/latcal*.rs`)

The adapter triple `(φ, W, W†)` is a frozen linear operator (W) plus a small nonlinear unroller (φ). LatCal bridges: commit `W` as a LatCal matrix (deterministic fixed-point); commit `φ`'s weights as a separate frozen artifact. **The navigator is then chain-verifiable**: any node can replay `z + W†·Δtarget` and check the BLAKE3 commitment of the result. **Cross-ref for riir-chain follow-up**: a `poincare_navigator_commit` recipe (Plan 002 chain-bridge class).

### 4.5 `NeuronShard` (`riir-neuron-db/src/shard/mod.rs`)

A shard's `style_weights[64]` is a frozen latent direction. The Poincaré adapter triple `(φ, W, W†)` fits naturally as a **new Pod kind** alongside the shard: a `PoincareNavigatorPod` with the same BLAKE3/Merkle commitment discipline. Freeze/thaw versions the adapter — each personality snapshot has its own imagination chart. Cross-ref for riir-neuron-db follow-up: a `poincare_navigator` feature gate, sibling of `KarcShard` / `ArchetypeBlendShard`.

### 4.6 DEC Stokes operators (`katgpt-rs/crates/katgpt-dec/`)

`hodge_decompose` decomposes a flow field into exact + coexact + harmonic components. The harmonic component is **curvature-free** by construction. **Theorem 5b** says Poincaré's obstruction *is* curvature — so the harmonic component is the part that doesn't need unrolling. They compose: hodge gives the harmonic navigation chart (free); Poincaré unrolls the exact + coexact residual. **Speculative fusion** — out of scope for the initial primitive.

### 4.7 SE(2) equivariant maps (`riir-engine/src/equivariant/`)

SE(2) lift → group-conv → project gives rotation-equivariant features. **These are the ideal input to a Poincaré adapter**: the adapter's job (unroll curvature) is half-done by the SE(2) lift (which already moves to the group manifold where the stabilizer is trivial). The fusion: SE(2) features as `z`, Poincaré adapter recovers `(Δx, Δy, Δθ)` from `z` differences. The adapter should fit tighter on SE(2) features than on raw features. **Strong fusion candidate** — Phase 5 of the plan.

---

## 5. Validation Protocol (G1–G7)

**Run in `katgpt-rs` (open primitive, Plan 449 Phase 2):**
- **G1 — Local decodability (Theorem 3 analog).** Construct a known smooth map `f: R⁶ → R^d` with rank-6 Jacobian (e.g., a random linear projection plus a tanh warp). Fit adapter. Assert `W·(φ(z₂) − φ(z₁)) ≈ log(g₁⁻¹·g₂)` to within `O(‖·‖²)`. PASS threshold: max abs diff `< 1e-3` for small displacements.
- **G2 — Global unrolling (Theorem 5c analog).** Construct a deliberately-curved manifold (e.g., `f(g) = MLP(g)` with a known arc). Fit adapter over a region `K`. Assert R² > 0.5 over `K`. Assert the linear-only baseline (no φ) achieves R² < 0. This proves the adapter unrolls curvature.
- **G3 — Inverse navigation round-trip.** Pick `(z_src, Δtarget)`, compute `z_dest = z_src + W†·Δtarget`. Assert nearest-neighbor retrieval of `z_dest` in a held-out embedding table recovers the ground-truth target within Hit@ε. PASS: Hit@0.3 > 0.5 on a synthetic target space.
- **G4 — Zero-alloc steady state.** `TrackingAllocator` audit on `poincare_navigate_into`: 0 allocations after warmup.
- **G5 — Latency.** Single navigator call (d=64, target_dim=6) under 1µs SIMD.
- **G6 — Multi-step coherence.** 4-step open-loop trajectory stays within the chart's valid region (no blow-up). R² of retrieved vs ground-truth > 0.3 at step 4.
- **G7 — Latent-vs-raw boundary.** The navigator operates only on `&[f32]` slices + a `PoincareAdapter` Pod. Never touches sync. The adapter Pod's BLAKE3 commitment is the only chain-crossing concern (delegated to riir-chain follow-up).

**Run in riir-ai (private, post-G1–G7):**
- **G8 — HLA imagination quality.** Fit adapter over `(MapPos, HLA_state)` pairs from a crowd simulation. Assert imagined HLA at held-out positions matches ground-truth evolved HLA within `R² > 0.4` on rotation queries, `R² > 0.2` on translation queries (per Theorem 7's asymmetry prediction).

**Promotion ladder:** G1–G7 PASS → feature ships opt-in in katgpt-rs. G8 + quality gate → riir-ai integration (per `riir-ai/.research/319`).

---

## 6. What stays open vs private

| Piece | Location | Why |
|---|---|---|
| `PoincareAdapter` Pod, `poincare_navigate_into`, `poincare_multi_step_into`, offline-fit helpers (closed-form ridge + small MLP) | `katgpt-rs/crates/katgpt-core/src/poincare.rs` (new, feature `poincare_navigator`) | Generic closed-form linear-algebra navigator. No game/chain/shard semantics. |
| HLA ↔ MapPos adapter fit, two-brain imagination loop, sleep-time imagination pipeline | `riir-ai/crates/riir-engine/src/` (private) | Game IP — the selling point. |
| Chain commitment of the adapter triple | `riir-chain/src/` (follow-up, gated by riir-chain decision) | Chain IP — only if the adapter crosses the sync boundary as a committed artifact. |
| `PoincareNavigatorPod` as a NeuronShard sibling | `riir-neuron-db/src/` (follow-up) | Shard IP — only if the adapter is stored as a shard-side artifact. |

---

## 7. Implementation Priority

| Phase | Task | Repo | Effort |
|---|---|---|---|
| 0 | This research note + riir-ai/.research/319 guide | katgpt-rs + riir-ai | ✅ THIS SESSION |
| 1 | `PoincareAdapter` Pod + `poincare_navigate_into` + `poincare_multi_step_into` (Plan 449 Phase 1) | katgpt-rs | 2 sessions |
| 2 | GOAT G1–G7 gates | katgpt-rs | 1 session |
| 3 | Offline-fit helper (deterministic ridge + 2-layer MLP via existing `katgpt-train`-free path) | katgpt-rs | 1 session |
| 4 | riir-ai HLA ↔ MapPos adapter fit (per R319) | riir-ai | 2 sessions |
| 5 | Fusion: SE(2) features as adapter input | riir-ai + katgpt-rs | speculative |
| 6 | Chain commitment of adapter triple (only if sync boundary crossed) | riir-chain | speculative |

---

## 8. Out of Scope (do not bundle)

- **The offline adapter fit's gradient-descent recipe** — the paper uses AdamW + 10 epochs. The *recipe* is a riir-train concern. **Modelless-first: the open primitive ships a closed-form ridge solver for `W` (one-line `W = (ZᵀZ + αI)⁻¹·ZᵀY`) and a deterministic closed-form φ fit (PCA-then-tanh)** as the default; gradient fit is a riir-train follow-up IF the modelless fit fails the G2 gate. Per §3.5 modelless-unblock protocol.
- **Multi-chart MoE adapter** (paper Appendix A.11) — `K` local charts with prototype routing. Speculative; single-chart adapter first.
- **SE(3) full 6D target** — the open primitive supports arbitrary `target_dim` (1–8 typical). SE(3) specifically is a 6D instantiation; the math is dimension-agnostic.
- **Application to actual vision features** — the paper uses DINOv2/V-JEPA. We have no vision foundation model. Our substrates are HLA, SenseModule projections, frozen shard embeddings — all valid adapter inputs.
- **Backprop through the adapter** — verboten per modelless mandate (constraint #1). Adapter is fit once offline, frozen, BLAKE3-committed, atomic hot-swap.

---

## 9. PO-Caveat — Quality Parity with the Paper

Per research skill §3.6, any "parity" claim with the paper needs a defend-wrong PoC. The claims in this note:

| Claim | Type | Proof required |
|---|---|---|
| The adapter unrolls manifold curvature (Theorem 5c holds on our substrate) | Architectural | Grep + G1/G2 gate (closed-form ridge + MLP fit) |
| The navigator is sub-µs at inference | Latency | G5 criterion bench |
| HLA imagination achieves R² > 0.4 on rotation queries (paper achieves 0.59 on DINOv2 rotation) | **Quality** | **PoC in `riir-ai/crates/riir-poc/`** — head-to-head vs (a) no-imagination baseline (use `last_known_pos`), (b) Motor-Gated DEC field evolution (R168), (c) Poincaré adapter. On a controlled synthetic NPC-trajectory benchmark. |

**The PoC's job is to defend OR refute.** If Poincaré imagination underperforms Motor-Gated DEC on rotation queries, the verdict still stands on architectural + latency axes; the quality axis becomes a tracked follow-up in `.issues/`. The canonical failure mode to watch for: **translation-magnitude R² collapses to near-zero on HLA** (per Theorem 7, this is expected — HLA's "depth analog" is the NPC's distance to the affective stimulus, which is not encoded in `MapPos` alone). If this happens, the HLA adapter needs `(MapPos, distance_to_stimulus)` as the target, not `MapPos` alone.

---

## 10. References

- **Paper:** [arXiv:2607.14228](https://arxiv.org/abs/2607.14228) — Chen et al., *SeeSE3: Emergence of 3D Space in Vision Features* (DeepMind, 15 Jul 2026).
- **Theorems 3, 5, 7** — paper Appendix A.7.
- **Sibling research notes:** R290 (Latent Field Steering — forward), R294 (Viable Manifold Graph — graph cousin), R382 (Spherical Geodesic Steering — sphere geodesic), R166 (riir-ai, SE(2) equivariant maps — composes), R168 (riir-ai, Motor-Gated DEC — field evolution cousin).
- **Private guide:** `riir-ai/.research/319_SeeSE3_Latent_Imagination_Game_Runtime_Guide.md`.
- **Plan:** `katgpt-rs/.plans/449_poincare_latent_navigation_primitive.md`.

---

## TL;DR (one sentence)

SeeSE3's Poincaré Adapter is the **closed-form linear-navigator primitive** our stack is missing: an offline-fit Siamese `φ` unrolls the curved latent manifold into a flat chart where `Δtarget ≈ W·Δz` holds globally, so inverse navigation `z + W†·Δtarget` becomes a single training-free matvec at inference — closing the two-brain model's "imagination gap" (NPCs can predict their HLA at unvisited positions without ever moving there) and composing with LFS (forward), VMG (graph), SE(2) (equivariant features), InducedCwm (frozen advance), and Motor-Gated DEC (field evolution) into a Super-GOAT moat: 4/4 novelty gate PASS, 10 systems touched, open primitive in katgpt-rs + private game-runtime guide in riir-ai/.research/319 + plan in katgpt-rs/.plans/449.
