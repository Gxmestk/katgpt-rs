# Proposal 005 — BridgeCertified: continuous determinism contract for raw↔latent bridges at the sync boundary

Status: **draft**
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Fusion of: Plan 385 T3.4 (`CogSlashEvidence` / `NonDeterminism` slashing) × `katgpt-core/src/closure/bridge.rs`
(raw↔latent bridge substrate) × `katgpt-core/src/engram/architecture_root.rs` (commitment pattern)
Related: [Plan 385 — Think-Brain-as-WASM Sidecar GOAT Gate](../../riir-ai/.benchmarks/385_think_brain_wasm_goat.md),
[Research 148 — Think-Brain WASM Vessel](../../riir-ai/.research/148_think_brain_wasm_vessel.md)

## TL;DR

Plan 385 (2026-07-06, shipped) closed the per-module cognition determinism gap: a third-party
`cog_*` WASM module must pass G3 (x86↔ARM bit-identical output) before registration, and is
slashable post-hoc via `CogSlashEvidence { kind: NonDeterminism }` if it diverges in production.
**That wall does not extend to the bridges in `katgpt-core` that translate raw↔latent at the sync
boundary** — `ptg_to_motif_embedding`, `motif_embedding_to_tar_score`, the HLA projection
(`SenseModule::project`), and any future `raw → latent` / `latent → raw` function. If a client
runs one of these with a different `libm` sigmoid, a different `style_weights` freeze version, or
a different FPU rounding mode than the pillar node replaying it, the client commits a sync scalar
that the pillar cannot bit-reconstruct — and the system has no continuous invariant to catch it.

This proposal ships three things: (1) a `BridgeCertified` marker trait in `katgpt-core` that
bridge functions opt into, asserting determinism + zero-allocation + freeze-version-pinning
contract; (2) a **bridge determinism test harness** that mirrors Plan 385's G3 for every certified
bridge (x86_64 + aarch64, same freeze snapshot, assert bit-identical output); (3) a
**`BridgeDriftEvidence`** evidence kind that extends the existing slashing pipeline so continuous
production drift becomes slashable end-to-end. Freeze-version skew is handled by extending the
`architecture_root` commitment to cover the direction-vector table the bridge projects onto —
mismatched versions fail at commitment check before any replay.

**This is a `katgpt-rs` invention.** No paper proposes a determinism contract for ML-inference
bridge functions in a quorum-replay setting. The closest prior art — consensus-based cheat
detection (Biró 2021) and the arxiv systematic review on anti-cheat defenses (Alangari &
Alharbi 2025, arXiv:2512.21377) — both operate at the *game-move* layer (validate that a player's
claimed move is legal), not at the *cognitive-bridge* layer (validate that a sigmoid projection
reconstructs bit-identically on a peer node). The contribution here is taking Plan 385's
module-level determinism pattern and lifting it to the function-level bridges that the modules
themselves depend on.

## The problem this solves

The threat model has three vectors today, each partially defended:

1. **Cognition WASM module non-determinism** — *fully defended* by Plan 385 (`CogSlashEvidence {
   kind: NonDeterminism }`, G3 gate, slashing pipeline in `riir-chain/src/asset_lifecycle/`).
2. **Cognition WASM module contract violation** (affect out of range, KG triple for unobserved
   target) — *fully defended* by Plan 385 (`DeterministicViolation` evidence kind).
3. **Bridge non-determinism in `katgpt-core`** — *undefended*. The WASM module is sandboxed and
   fog-of-war gated, but its outputs flow through `closure/bridge.rs`,
   `SenseModule::project`, the HLA kernel, and the latent-functor math before becoming the 5
   synced affect scalars. None of those bridges carry a determinism contract.

Vector 3 is the one this proposal fills. Concrete failure modes:

- **`libm` divergence.** `sigmoid(x)` is `1.0 / (1.0 + (-x).exp())`. glibc's `expf` and
  macOS-libsystem's `expf` differ by 1 ULP on a small fraction of inputs. The HLA projection
  saturates near `|x| > 17` to exactly 0.0 or 1.0 (the spec-match test in
  `riir-engine/tests/hla_bounds_spec_match.rs` already accepts this), but in the linear regime
  (`|x| ≤ 5`), 1-ULP divergence accumulates across a 64-dim dot product into ≥1-ULP scalar
  divergence. A client on aarch64-macOS commits a `fear = 0.421...` scalar that the pillar on
  x86_64-linux computes as `0.422...` → quorum replay fails → the validator either silently
  accepts the divergence (anti-cheat degraded) or rejects the block (liveness degraded).

- **Freeze-version skew.** A bridge projects onto a `MotifDirections` table (`closure/bridge.rs`)
  or a `SenseModule` direction vector. These are freeze-versioned (`MerkleFrozenEnvelope`,
  `KarcShard`). If the client's frozen snapshot is at version N and the pillar's is at N-1
  (because the pillar hasn't thawed yet, or the client is racing ahead), the projection produces
  different scalars for the same raw input. The current sync layer has no field that pins the
  direction version the scalar was computed against — the scalar is just a number.

- **Continuous drift.** Plan 385's G3 runs once at gate-time. A bridge can be bit-identical on
  the gate corpus and diverge on a production input that wasn't in the corpus. There is no
  continuous sampler that re-runs bridges on the pillar and emits evidence when drift appears.

Without this proposal, the wall between adaptive client computation and the pillar's deterministic
replay holds *only* for the WASM module surface. The bridge surface — which is where the actual
adaptive math lives — is undefended.

## The proposed design

### Piece 1 — `BridgeCertified` marker trait (ships in `katgpt-core`)

```rust
/// Marker trait: the implementing function is a certified raw↔latent bridge.
///
/// Certified bridges MUST satisfy, by construction:
/// 1. **Deterministic** — same inputs → same outputs, bit-identical, across all
///    target platforms (x86_64, aarch64). No `libm` calls that diverge; no
///    `HashMap` iteration order; no thread-local state; no `Instant::now()`.
/// 2. **Zero-allocation** on the hot path (per AGENTS.md "Allocation" rule).
///    Callers pass pre-allocated scratch buffers via `&mut [T]` parameters.
/// 3. **Freeze-version-pinned** — the bridge's output is a pure function of
///    (raw inputs, freeze snapshot, feature-flag config). The freeze snapshot
///    MUST be content-addressed (BLAKE3) and the hash carried with any synced
///    scalar computed by the bridge.
/// 4. **Feature-gateable** — never on by default; promotion requires the
///    determinism harness (Piece 2) to pass.
///
/// Bridges that violate any property MUST NOT impl this trait. Lint rule
/// `bridge_certified_invariants` (Piece 4) audits the impls.
pub trait BridgeCertified {
    /// The BLAKE3 hash of the freeze snapshot the bridge projects onto.
    /// Carried alongside any synced scalar computed by this bridge so the
    /// pillar can reject version-skewed outputs at commitment check.
    fn freeze_version_hash(&self) -> [u8; 32];

    /// Re-run the bridge on the given inputs + snapshot, returning the output
    /// bytes. Used by Piece 3 (drift sampler) for pillar-side replay.
    fn replay(&self, raw_inputs: &[u8], freeze_snapshot: &[u8]) -> Vec<u8>;
}
```

The trait lives in `katgpt-core/src/closure/bridge.rs` next to the existing `MotifDirections`
and `ptg_to_motif_embedding`. Initial impls: `MotifDirections`-backed bridges. Extensible to
`SenseModule::project`, HLA kernel, latent-functor ops.

### Piece 2 — Bridge determinism test harness (ships in `katgpt-core`)

Mirrors Plan 385's G3 cross-arch check. For each `BridgeCertified` impl:

1. Generate a deterministic input corpus (≥10⁴ inputs, fixed seed).
2. Run the bridge on `x86_64-apple-darwin` (native on Intel Macs, Rosetta on Apple Silicon).
3. Run the bridge on `aarch64-apple-darwin` (native on Apple Silicon).
4. Assert bit-identical outputs across architectures for all inputs.

The harness reuses Plan 385's `cog_determinism_check` example pattern — a standalone binary in
`katgpt-core/examples/bridge_determinism_check.rs` that loads a corpus, runs both arches, and
exits non-zero on divergence. CI runs it on both architectures.

**Non-goal:** this is NOT a proof. It's an empirical gate. The Lean proof instances in the 5-repo
quintet (`KatgptProof`, `RiirAiProof`) prove properties over `ℝ`; f32 bridge determinism cannot
be proven in Lean because `ℝ` has no NaN and no ULP. The harness is the sole validator.

### Piece 3 — `BridgeDriftEvidence` evidence kind (extends Plan 385 slashing pipeline)

Adds a new evidence kind to the slashing pipeline alongside `CogSlashEvidence`:

```rust
/// Evidence that a bridge produced a synced scalar that the pillar could not
/// bit-reconstruct. Slashable via the existing `SlashNft` path.
pub struct BridgeDriftEvidence {
    /// BLAKE3 of the bridge function's impl (content-addressed identity).
    pub bridge_impl_blob_id: AssetBlobId,
    /// BLAKE3 of the freeze snapshot the client claims to have used.
    pub client_freeze_hash: [u8; 32],
    /// BLAKE3 of the freeze snapshot the pillar replayed against.
    pub pillar_freeze_hash: [u8; 32],
    /// The raw inputs to the bridge (recoverable on the verifier side).
    pub raw_inputs: Vec<u8>,
    /// The client's claimed output scalar(s).
    pub client_output: Vec<u8>,
    /// The pillar's recomputed output scalar(s).
    pub pillar_output: Vec<u8>,
}
```

If `client_freeze_hash == pillar_freeze_hash` AND `client_output != pillar_output` → the bridge
is non-deterministic. Slash.

If `client_freeze_hash != pillar_freeze_hash` → the client ran against a different snapshot.
This is a *valid* divergence (the bridge IS deterministic, just against a different version) —
NOT slashable, but the pillar MUST reject the sync scalar and re-request against the canonical
snapshot. This is the freeze-version-skew recovery path.

The drift sampler lives in `riir-ai` runtime — samples 1 in N bridge calls, sends
`(bridge_impl_blob_id, freeze_hash, raw_inputs, output)` to the pillar for replay. N is tunable
per bridge (default N=1024 for hot paths, N=1 for cold paths).

### Piece 4 — Lint rule `bridge_certified_invariants` (ships in `katgpt-core`)

A `tests/bridge_certified_invariants.rs` audit that grep-style asserts: every fn that produces a
value flowing into a `SyncBlock` field either (a) is `BridgeCertified`, or (b) is itself a pure
function over `BridgeCertified` outputs + raw inputs. Failure blocks promotion. This is the
structural guard that prevents risk #1 from the prior conversation (a careless `cog_*` export
funneling adaptive state into a sync field).

## Honest caveats — READ BEFORE IMPLEMENTING

1. **f32 cross-arch bit-identical sigmoid may be impossible without software emulation.** glibc
   vs macOS-libsystem `expf` divergence is real and not under our control. The honest path is
   (a) detect divergence on the corpus in Piece 2, (b) if it exists, ship a software-emulated
   sigmoid (e.g., a rational-polynomial approximation with fixed coefficients) gated behind a
   `bridge_soft_sigmoid` feature flag, and require `BridgeCertified` impls to use it. **The
   proposal is worthless if the bridge cannot actually be made bit-identical.** G1 = corpus
   divergence = 0 across architectures, with the soft sigmoid if needed.

2. **The drift sampler adds per-tick overhead.** Plan 385's wasmtime budget headroom is 6.7×
   (7.51ms / 1 CPU-sec at 10K NPCs × 20Hz). The sampler must fit in the remaining 92.49% of
   budget. Sending `(blob_id, freeze_hash, raw_inputs, output)` to the pillar is one network
   round-trip per sampled bridge call — at N=1024 with ~10 bridges per tick, that's
   ~200 samples/sec across all NPCs. Probably fine; benchmarked in G2 of the GOAT gate. If the
   overhead pushes wasmtime over the 5% target, N must increase (sampling sparser) and the
   detection latency grows accordingly.

3. **The freeze-version-pinning commitment is a sync-boundary field.** The naive version adds
   32 bytes per synced scalar (one hash per scalar). The honest version commits to a single
   root freeze hash per NPC per tick — the `architecture_root` already exists in
   `engram/architecture_root.rs` and commits over the full cognitive architecture. The
   proposal extends `architecture_root` to cover the `MotifDirections` table + `SenseModule`
   direction vectors, so a single 32-byte field on `SyncBlock` pins the entire version state.
   **This is a `SyncBlock` layout change and requires riir-chain coordination.**

4. **The proposal depends on Plan 385's slashing pipeline being live in production.** If no
   validator actually triggers `CogSlashEvidence` slashing, this proposal's
   `BridgeDriftEvidence` is theoretical. The RIIR-chain validator set MUST include a node that
   (a) samples bridge calls, (b) replays them, (c) emits `BridgeDriftEvidence` on divergence.
   Without that validator, the wall exists on paper but not in production. This is an
   operational dependency, not a code dependency.

5. **The lint rule (Piece 4) is a soft guard, not a hard compiler error.** Rust's type system
   cannot encode "this value flows into a sync field"; the audit is a test, not a trait bound.
   A determined contributor can bypass it. The structural fix would be a `SyncSafe` newtype that
   wraps any value allowed in a `SyncBlock`, with a sealed-constructor that only
   `BridgeCertified::replay` can produce — but that's a much larger refactor across all
   `SyncBlock` consumers and is out of scope for this proposal (see Out of scope).

## Fusion lineage

| Source | What it contributes |
|---|---|
| Plan 385 T3.4 (`CogSlashEvidence`, `CogSlashEvidenceKind::NonDeterminism`) | The slashing pipeline + evidence schema pattern. `BridgeDriftEvidence` mirrors `CogSlashEvidence`'s shape exactly — different identity (bridge impl vs module blob), same BLAKE3-commitment + forensic-replay contract. |
| `katgpt-core/src/closure/bridge.rs` (`MotifDirections`, `ptg_to_motif_embedding`) | The substrate being certified. The trait lives next to these because they're the canonical example of a raw↔latent bridge. |
| `katgpt-core/src/engram/architecture_root.rs` (`CognitiveArchitectureRoot::from_parts`) | The commitment pattern. Extending the root to cover `MotifDirections` + `SenseModule` direction vectors is what makes the freeze-version-pin a single 32-byte field instead of N hashes. |

**This is not a new capability class.** Plan 385 already ships determinism verification +
slashing for cognition modules. This proposal extends the *same pattern* to the *substrate the
modules depend on*. Verdict ceiling: **GOAT** (the gain is "anti-cheat wall extends to bridges",
which is a provable invariant extension, not a new mechanism). The promotion condition is G1
(corpus divergence = 0 cross-arch) — if it fails, the soft-sigmoid fallback must land first.

## GOAT gate

| Gate | Target | How measured |
|---|---|---|
| **G1 correctness** | Bridge output corpus bit-identical across `x86_64-apple-darwin` and `aarch64-apple-darwin`, for all `BridgeCertified` impls, on ≥10⁴ inputs each. | `cargo run --example bridge_determinism_check --release`. If non-zero divergence on `libm` path, the soft-sigmoid feature flag MUST be on for the gate to pass. |
| **G2 perf** | Drift sampler adds ≤ 0.5% CPU at 10K NPCs × 20Hz (within Plan 385 wasmtime headroom). N=1024 sampling default; if G2 fails, N=4096 and document the latency tradeoff. | Bench in `riir-wasm/benches/bridge_drift_sampler.rs`, mirroring `think_brain_call_overhead.rs`. |
| **G3 no-regression** | All existing cognition slashing tests (`cog_slash_*` in `riir-chain`) still pass with the new evidence kind added. The `architecture_root` extension does not invalidate any existing commitment hash (the extension is additive — the root now commits over MORE state, not different state). | `cargo test -p riir-chain --features chain_asset_fingerprinting --lib cog_slash_evidence::tests::cog`. Plus a new `bridge_drift_slash_round_trip` test. |
| **G4 alloc-free** | `BridgeCertified::replay` does not allocate on the hot path. Scratch buffers are caller-supplied via `&mut [T]`. | `cargo test --features bridge_certified --benches -- alloc_check` (dhat or alloc-counter). |

**No UQ-bearing primitive is added** (no probability distribution, no quantile, no coverage
claim), so the "Report the Floor" conformal-naive baseline rule does not apply.

## What ships now (`katgpt-rs`) vs deferred (`riir-ai` / `riir-chain`)

### Ships now — open primitive, leaf-clean (`katgpt-rs`)

- `BridgeCertified` trait in `katgpt-core/src/closure/bridge.rs` (gated `bridge_certified`).
- `MotifDirections` impl of `BridgeCertified` (the canonical example).
- Bridge determinism test harness: `katgpt-core/examples/bridge_determinism_check.rs`.
- Lint rule: `katgpt-core/tests/bridge_certified_invariants.rs`.
- Optional soft-sigmoid: `katgpt-core/src/closure/soft_sigmoid.rs` (gated
  `bridge_soft_sigmoid`, default-off, on only if G1 needs it).

### Deferred — runtime wiring (`riir-ai`)

- `BridgeDriftSampler` in `riir-engine` (samples 1 in N bridge calls, sends to pillar).
- Plan 385 host-side `ThinkBrainHost` extension to also sample bridge calls, not just
  `cog_*` calls.
- Phase-2 G2 perf bench (`riir-wasm/benches/bridge_drift_sampler.rs`).

### Deferred — slashing pipeline extension (`riir-chain`)

- `BridgeDriftEvidence` schema in `riir-chain/src/asset_lifecycle/bridge_drift_evidence.rs`,
  mirroring `cog_slash_evidence.rs` layout.
- `architecture_root` extension to cover `MotifDirections` + `SenseModule` direction vectors
  (the freeze-version-pin field). Layout change to `SyncBlock`; requires quorum-aware migration.
- Validator-side replay hook that consumes `BridgeDriftEvidence` and re-runs the bridge.

### Explicitly NOT shipped by this proposal

- **TEE / SGX enclave.** The arxiv systematic review (Alangari & Alharbi 2025) notes hardware
  TEEs as the strongest anti-cheat defense. This proposal does not require or recommend TEEs —
  the threat model is quorum replay divergence, not memory introspection. TEEs are orthogonal
  and could compose with this proposal if a future threat model demands it.
- **Kernel-level anti-cheat.** Out of scope by design — the modelless mandate and the WASM
  sandbox already provide the isolation that kernel anti-cheat would provide. We do not ship
  kernel drivers.
- **`SyncSafe` newtype refactor.** The structural guard (Piece 4's lint rule) is a soft test,
  not a type-system enforcement. A full `SyncSafe` newtype that the compiler enforces is a
  larger refactor across every `SyncBlock` consumer and is explicitly deferred.

## Phased rollout (sketch — a plan would expand this)

### Phase 1 — Open primitive skeleton (`katgpt-rs`, ships now)
- [ ] T1.1 `BridgeCertified` trait + `MotifDirections` impl in `katgpt-core/src/closure/bridge.rs`
- [ ] T1.2 Bridge determinism test harness (`bridge_determinism_check.rs`)
- [ ] T1.3 Lint rule (`bridge_certified_invariants.rs`)
- [ ] T1.4 Soft-sigmoid module (gated `bridge_soft_sigmoid`, default-off)

### Phase 2 — G1 validation (`katgpt-rs`, ships now)
- [ ] T2.1 Run G1 on the corpus with `libm` sigmoid → measure divergence
- [ ] T2.2 If divergence > 0: enable soft-sigmoid, re-run G1 → measure divergence
- [ ] T2.3 If divergence still > 0: STOP, file issue, proposal cannot ship

### Phase 3 — Runtime + slashing wiring (`riir-ai` + `riir-chain`, deferred)
- [ ] T3.1 `BridgeDriftSampler` in `riir-engine`
- [ ] T3.2 `BridgeDriftEvidence` schema in `riir-chain`
- [ ] T3.3 `architecture_root` extension (sync-boundary layout change)
- [ ] T3.4 Validator-side replay hook
- [ ] T3.5 G2 perf bench + G3 no-regression

### Phase 4 — Promotion decision (`katgpt-rs`)
- [ ] T4.1 If G1 + G2 + G3 + G4 all pass → promote `bridge_certified` to default-on
- [ ] T4.2 If soft-sigmoid required → promote `bridge_soft_sigmoid` to default-on alongside
- [ ] T4.3 If any gate fails → keep opt-in, document the gap, file follow-up

## Risks

1. **f32 cross-arch determinism may require software-emulated transcendentals.** This is the
   highest-impact risk — if `libm` cannot be made bit-identical across platforms, the soft-sigmoid
   fallback is mandatory, not optional. The soft-sigmoid adds ~5–20 ns per call (polynomial
   approximation vs hardware `expf`), which may eat into the G2 budget. **Mitigation:** Phase 2
   measures before Phase 3 commits.

2. **The `architecture_root` extension is a `SyncBlock` layout change.** Extending the commitment
   to cover direction-vector tables changes the root hash for every existing NPC. Quorum nodes
   running the old root will reject blocks committed under the new root. **Mitigation:** version
   the root (`architecture_root_v1`, `architecture_root_v2`) and coordinate the upgrade via the
   chain's existing fork mechanism. This is a hard fork and must be scheduled.

3. **The drift sampler's network round-trip is a new sync dependency.** Plan 385's cognition
   slashing is *post-hoc* — the violation is detected and the module is slashed, but the tainted
   state may have already propagated for a tick or two. This proposal's drift sampler has the
   same latency. **Mitigation:** for hot-path bridges (HP/wallet-related, if any ever become
   bridge-driven), set N=1 (sample every call). For cold-path bridges (affect scalars,
   zone-attention), N=1024 is acceptable.

4. **Cognitive coupling between `katgpt-rs` and `riir-chain`'s slashing pipeline.** Today
   `katgpt-core` ships primitives with no chain dependency. This proposal's
   `BridgeDriftEvidence` schema lives in `riir-chain` but the `BridgeCertified` trait lives in
   `katgpt-core`. The link is the `bridge_impl_blob_id` content-addressing — `katgpt-core` must
   produce a stable impl BLAKE3 that `riir-chain` can reference. **Mitigation:** the
   `bridge_impl_blob_id` is computed by a `katgpt-core`-side helper (`impl_blob_id()` on the
   trait) and treated as opaque by `riir-chain`. No `riir-chain` import in `katgpt-core`.

5. **The continuous drift sampler is a new attack surface.** A malicious validator could sample
   every bridge call, learn the raw inputs, and infer latent state. **Mitigation:** the sampler
   sends only `(blob_id, freeze_hash, raw_inputs, output)` — never the latent embedding. Raw
   inputs are already in the sync layer (they're raw by definition). The output is a sync scalar
   (also already synced). No new information crosses the wire.

## Out of scope

- **TEE / SGX / kernel anti-cheat.** See "Explicitly NOT shipped" above.
- **`SyncSafe` newtype refactor.** See "Explicitly NOT shipped" above.
- **Multi-NPC batched bridge replay.** The drift sampler replays one bridge call at a time.
  Batching across NPCs is a perf optimization deferred to a follow-up plan if G2 fails.
- **Bridge determinism for non-`BridgeCertified` functions.** The trait is opt-in. Functions
  that don't impl it are not covered. The lint rule (Piece 4) catches functions that *should*
  impl it but don't; it does not retroactively certify arbitrary functions.
- **A Lean proof of f32 bridge determinism.** f32 has NaN, ULP, and platform-dependent
  transcendentals — none of which Lean's `ℝ` models. The harness is the sole validator. The
  `KatgptProof` instance in `.proofs/` is not extended by this proposal.

## References

1. **Alangari & Alharbi (2025), "A Systematic Review of Technical Defenses Against Software-Based
   Cheating in Online Multiplayer Games," arXiv:2512.21377.** Cited-only. Surveys four defense
   categories: server-side detection, client-side anti-tamper, kernel-level drivers, hardware
   TEEs. This proposal fits the "server-side detection" category with a novel sub-mechanism
   (bridge-level determinism contract rather than game-move-level validation). The review does
   not consider cognitive-bridge determinism — its server-side-detection category covers game
   state (position, HP, velocity), not latent cognitive projections.

2. **Biró (2021), "Consensus-Based Cheat Detection for Multiplayer Games."** Cited-only. The
   peer-voting model — clients subscribe to each other's moves, evaluate against rules, vote on
   violations — is structurally what `ChainConsensus` quorum already does. The 51% attack risk
   Biró identifies is the standard Sybil concern; `riir-chain`'s slashing + stake economics is
   the existing mitigation. This proposal extends the consensus mechanism from "validate the
   game move" to "validate the bridge output that produced the synced scalar."

3. **Plan 385 (riir-ai), "Think-Brain-as-WASM Sidecar GOAT Gate."** Substrate. The G3
   cross-arch determinism check and `CogSlashEvidence` slashing pipeline are the direct
   predecessors. This proposal lifts both from the module surface to the bridge surface.

4. **Research 148 (riir-ai), "Think-Brain WASM Vessel."** Substrate context. Section 6.3
   defines the two misbehavior classes (`DeterministicViolation`, `NonDeterminism`) that
   `CogSlashEvidenceKind` encodes. This proposal's `BridgeDriftEvidence` is a third class —
   it's structurally `NonDeterminism` but scoped to bridges, not modules, because the bridge's
   `freeze_version_hash` introduces a new failure mode (version skew) that the module-level
   evidence schema cannot represent.

## TL;DR

**Verdict: ship Phase 1+2 (the open primitive + determinism harness) in `katgpt-rs` now; defer
Phase 3 (runtime + slashing wiring) to `riir-ai` + `riir-chain` after G1 passes.** The win is
extending Plan 385's anti-cheat wall from the WASM module surface to the bridge substrate. The
cost is one `SyncBlock` layout change (architecture_root extension) and possibly one soft-sigmoid
fallback if `libm` cannot be made cross-arch bit-identical. **Next action: open Plan NNN for
Phase 1 in `katgpt-rs`, run G1 on the bridge corpus, decide on soft-sigmoid before Phase 3.**
