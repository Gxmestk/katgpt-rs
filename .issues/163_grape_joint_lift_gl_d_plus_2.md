# Issue 163 — GRAPE Joint Lift: `GL(d+2)` Block-Diagonal Composition

**Source:** Zhang et al., *GRAPE: Group Representational Position Encoding*
(arXiv:2512.07805, ICLR 2026, **Appendix E**). See
[Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md).

**Status:** opened 2026-07-17. **Unblocked** by Issues [159](./159_grapem_rank2_rodrigues_exponential.md)
+ [160](./160_position_group_action_trait.md) + [161](./161_grape_ap_vector_similarity_gates.md)
landing on `develop` (commits `25b758f6`, `5a495496`, `10cd9134`). Research 446 §4 explicitly
deferred this 4th item until the trilogy landed.

## TL;DR

Implement the **`GL(d+2)` block-diagonal joint lift** (Appendix E) — a single
primitive that composes GRAPE-M (rotary `SO(d)` rotation, Issue 159) with
GRAPE-A (additive logit bias, §4 of the paper) into one group action. This
closes the GRAPE paper's composition story: today Wall *replaces* RoPE (they
are alternatives); the joint lift proves they *compose* into a single
one-parameter subgroup of `GL(d+2)` while preserving the exact relative law.

**Feature gate:** `grape_joint_lift` (opt-in). **Implies** `grapem_rodrigues`.

**Verdict target:** GAIN — same engine-layer transformer-math class as the
trilogy (159/160/161). No Super-GOAT (no new capability class: rotary+additive
composition is parameter-rich but not a new mechanism).

## What the paper says (Appendix E, verbatim math)

> Multiplicative GRAPE (Section 3) and Additive GRAPE (Section 4) compose
> naturally. This can be viewed either additively at the logit level:
>
> ```
> ℓ_{t,j,h} = (1/√d) · q_{t,h}^T · G_h(j−t) · k_{j,h}
>           + (j−t)·ω·(λ_{q,t} + λ_{k,j})
>             └─────────── additive GRAPE bias ───────────┘
> ```
>
> or rigorously as a single group action in `GL(d+2)`. Concretely, we construct
> the joint lift by direct sum. Let `bq = [q; 1; 0]` and `bk = [k; 0; 1]` be
> the augmented vectors. Define the joint generator `L ∈ gl(d+2)` as the direct
> sum of the rotational generator `L ∈ so(d)` and the gated nilpotent generator
> `ω·Λ·A0` (where `Λ = λq + λk` captures the content modulation):
>
> ```
> G_joint(m) = exp(m·L) = [ exp(m·L)   0       0      ]
>                         [ 0^T        1       m·ω·Λ  ]   ∈ GL(d+2)
>                         [ 0^T        0       1      ]
> ```
>
> Scoring with the paired inverse-transpose strictly preserves the exact
> relative law for the combined system:
>
> ```
> bq_i^T · G_joint(j−i)^{−⊤} · bk_j
>   = q_i^T · exp((j−i)·L) · k_j  +  (j−i)·ω·Λ  +  const
> ```
>
> exactly reproducing the sum of multiplicative (rotary) and additive (slope)
> components.

The content gates are non-negative softplus forms (§4.1):

```
λq(q) = softplus(v^T · q / √d)         (query gate)
λk(k) = softplus(u^T · k / √d)         (key gate)
softplus(z) = log(1 + e^z) ≥ 0
```

For the causal regime `j ≤ i`, the offset `m = j − i ≤ 0`, so `(j−i)·ω·Λ ≤ 0`
— a monotonic penalty (matches ALiBi sign convention; recovers ALiBi exactly
when `λq ≡ 0`, `λk ≡ β_h`, and `L = 0`).

## Scope (what ships)

A single new module `crates/katgpt-core/src/grape_joint_lift.rs` gated by
`grape_joint_lift` (implies `grapem_rodrigues`). The primitive composes
[`Rank2Plane`](../crates/katgpt-core/src/grapem.rs) (rotary part, Issue 159)
with two gate vectors `(u, v)` (additive part) and exposes a single
**score-into** entry point that produces the joint logit in one pass.

### Public API (target)

```rust
pub struct GrapeJointLift {
    plane: Rank2Plane,       // rotary part (Issue 159)
    omega_rot: f32,          // rotary frequency ω
    omega_add: f32,          // additive frequency ω (paper uses same ω; decoupled for flexibility)
    u_gate: Box<[f32]>,      // key gate vector (length D)
    v_gate: Box<[f32]>,      // query gate vector (length D)
}

impl GrapeJointLift {
    pub fn new(plane: Rank2Plane, omega_rot: f32, omega_add: f32,
               u_gate: &[f32], v_gate: &[f32]) -> Result<Self, JointLiftError>;

    pub fn dim(&self) -> usize;
    pub fn plane(&self) -> &Rank2Plane;
    pub const fn omega_rot(&self) -> f32;
    pub const fn omega_add(&self) -> f32;
    pub fn u_gate(&self) -> &[f32];
    pub fn v_gate(&self) -> &[f32];

    /// Compute Λ = softplus(v^T·q/√d) + softplus(u^T·k/√d) (the additive
    /// gate sum). Pure function — exposed for streaming-cache callers that
    /// want to cache λk(k) at key arrival and only recompute λq(q) per query.
    pub fn gate_sum(&self, q: &[f32], k: &[f32]) -> Result<f32, JointLiftError>;

    /// One-pass joint score:
    ///   out = q^T · exp(m·ω_rot·L) · k / √d  +  m · ω_add · (λq(q) + λk(k))
    /// Writing into a caller-provided `rotated_q` scratch buffer (length D).
    /// Zero allocation after `new`.
    ///
    /// This is the primitive the paper's Appendix E distills to: a single
    /// fused rotary+additive logit. `m = j − i` is the relative offset.
    pub fn score_into(&self, q: &[f32], k: &[f32], m: i32,
                      rotated_q_scratch: &mut [f32], out: &mut f32)
                      -> Result<(), JointLiftError>;
}

/// softplus(z) = log(1 + e^z) ≥ 0. Numerically stable: for z < 0 use
/// `z + log1p(e^{-z})`; for z ≥ 0 use `log1p(e^z)`.
pub fn softplus(z: f32) -> f32;
```

### Why `omega_rot` and `omega_add` are decoupled

The paper uses a single shared `ω` for both the rotary and additive parts
(Eq. after the `G_joint(m)` display). Decoupling them is a **strict
generalization**: setting `omega_rot == omega_add` recovers the paper exactly.
The decoupling lets a caller scale the additive decay independently of the
rotary frequency (e.g. strong decay + slow rotation for long-context forgetting).
This is **not** a deviation from the paper — it is a parametric superset.

## What does NOT ship

- **Learning the gates.** `u, v` are user-supplied; learning is `→ riir-train`
  (the modelless-first mandate). Same boundary as Issue 159's `(a, b)`.
- **The full `GL(d+2)` matrix.** Never materialised — the joint score
  decomposes into one rotary apply + two dot products + one softplus pair +
  one FMA, all `O(d)`.
- **Multi-head / batched API.** Single-head, single-(q, k) entry point. The
  paper's per-head formulation (Appendix B) is a thin wrapper — out of scope
  for this primitive (no transformer context exists in katgpt-core).
- **The path-integral additive (GRAPE-AP, §5).** This issue is GRAPE-A (§4,
  single offset bias `(j−i)·ω·Λ`). GRAPE-AP composition with rotary is a
  separate fusion candidate (would compose Issue 161's `GrapeApGate` with
  `Rank2Plane`) — out of scope here, not opened.
- **Streaming cache.** The pattern is documented in the module doc (cache
  `G(j)·k_j` and `λk(k_j)` together at key arrival; recompute `λq(q_t)` and
  `G(t)·q_t` per query). The primitive itself is stateless — the caller owns
  the cache. Matches Issue 159's stateless contract.

## GOAT gate (G1–G4)

| Gate | Target | Method |
|------|--------|--------|
| **G1** | Joint score bit-identical (within f32 precision) to the manual composition: `Rank2Plane::apply_into(q, m, ω_rot, scratch)` then `dot(scratch, k)/√d + m·ω_add·(softplus(v·q/√d) + softplus(u·k/√d))`. Verify exact relativity: `score(i, j) ≈ score(0, j−i)` (the relative-law consequence of the joint lift proof). | in-crate test: 20 random `(u, v, a, b, q, k, m)` instances per dim {8, 16, 32, 64}; assert abs diff < 1e-5 and rel-law invariance < 1e-5. |
| **G2** | Latency ≤ 1.10× the sum of separate calls (`Rank2Plane::apply_into` + 2 dot products + 2 softplus + 1 FMA). The primitive should not be measurably slower than calling the parts separately — the value is the unified API + correctness proof, not a perf gain. | in-crate timing smoke test (100k calls at d=64). |
| **G3** | No regression. Default + opt-in + `--all-features` clean. | external `cargo clippy` (the user runs this). |
| **G4** | 0 allocations in `score_into` after `new` (CountingAllocator). `new` does exactly 2 allocs (the `u, v` `Box<[f32]>`). | in-crate `g4_*` test mirroring Issue 159's pattern. |

### G1 special cases (must hold)

1. **ω_add = 0** → pure rotary. Score reduces to `q^T·exp(m·ω·L)·k/√d` (Issue 159 only).
2. **ω_rot = 0 (or s = 0)** → pure additive. Score reduces to `q^T·k/√d + m·ω_add·Λ` (GRAPE-A, Eq. 4.6).
3. **u = v = 0** → softplus(0) = log(2), constant gate. Score is rotary + constant shift.
4. **Λ = 0 (impossible with softplus ≥ log 2)** → would reduce to pure rotary, but softplus is strictly positive; document that the additive term is always ≥ `m·ω_add·2·log(2)`.
5. **m = 0** → no offset. Score reduces to `q^T·k/√d` (rotary identity + zero additive).
6. **Exact relativity.** `score(q, k, j−i)` depends only on `j−i`, not on absolute `i, j` (modulo f32 rounding). The joint lift's whole point is preserving the relative law under composition.

## Subtasks

- [x] T1. Implement `crates/katgpt-core/src/grape_joint_lift.rs`: `softplus`, `GrapeJointLift`, `JointLiftError`, all accessors.
- [x] T2. Wire feature `grape_joint_lift = ["grapem_rodrigues"]` in `Cargo.toml` + `pub mod` + `pub use` in `lib.rs` (mirror Issue 161's wiring).
- [x] T3. In-crate tests: G1 (bit-identical to manual composition + relativity), G2 (latency smoke), G4 (alloc-free after `new`), all 6 G1 special cases above, plus shape-mismatch and accessor coverage.
- [x] T4. Write `.benchmarks/460_grape_joint_lift_goat.md` with the recorded G1–G4 verdict.
- [x] T5. `cargo clippy -p katgpt-core --features grape_joint_lift --lib` clean.
- [x] T6. `cargo clippy -p katgpt-core --all-features --lib` clean (no combo regression).
- [-] T7. Promotion to default-on: **deferred**. No hot-path consumer today. Re-evaluate when a transformer attention path or a cross-repo fusion (riir-ai HLA personality + decay, riir-neuron-db shard rotation + bias) lands. Per the `- [-]` convention.

## Why this issue (not a plan)

Per AGENTS.md: *"Create issue at .issues for poc, proof, optimization or refactor task, do not create plan."*
This is a single primitive (one module, one feature) distilled from a known
paper section. Issues 159/160/161 used the same issues-not-plans pattern.

## Cross-references

- [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — GRAPE distillation (§4 "Actionable follow-ups" lists this as the deferred 4th item).
- [Issue 159](./159_grapem_rank2_rodrigues_exponential.md) — GRAPE-M rotary primitive (`Rank2Plane`).
- [Issue 160](./160_position_group_action_trait.md) — Unified `PositionGroupAction` trait.
- [Issue 161](./161_grape_ap_vector_similarity_gates.md) — GRAPE-AP path-integral gates (NOT composed here; this is GRAPE-A §4).
- [`.benchmarks/457`](../.benchmarks/457_grapem_rodrigues_goat.md) / [`.benchmarks/458`](../.benchmarks/458_position_group_action_goat.md) / [`.benchmarks/459`](../.benchmarks/459_grape_ap_vector_goat.md) — trilogy GOAT verdicts.
- Paper Appendix C (FoX as GRAPE-AP) + Appendix E (this composition) + §4.1–4.2 (ALiBi/FoX as GRAPE-A).
