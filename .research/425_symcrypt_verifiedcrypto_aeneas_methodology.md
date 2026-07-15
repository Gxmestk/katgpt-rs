# Research 425: SymCrypt Verified-Crypto — Aeneas Rust→Lean Methodology Cross-Check

> **Source:** [microsoft/SymCrypt `feature/verifiedcrypto` README-VERIFIEDCRYPTO.md](https://github.com/microsoft/SymCrypt/blob/feature/verifiedcrypto/README-VERIFIEDCRYPTO.md) (Microsoft, 2026 branch)
> **Date:** 2026-07-15
> **Status:** Done
> **Related Research:** 351 (cross-repo Lean 4 FV pattern — the canonical counterpart), 292 (gap analysis), 198 (Lean4Agent — different FV angle)
> **Classification:** Public

---

## TL;DR

Microsoft's SymCrypt `feature/verifiedcrypto` branch ships Lean 4 proofs of **functional correctness and panic freedom** for selected Rust crypto primitives (SHA-3/SHAKE, ML-KEM, hardware intrinsics). Three primitives, ~58K LOC of Lean proofs over ~5.5K LOC of Rust, full re-verification in ~15 minutes via `lake build`. **Verdict: Gain.** It is a methodology cross-check against our own cross-repo FV pattern (Research 351), not a new primitive. Three genuinely new techniques surface that our C1–C6 conventions don't yet name: (1) **Aeneas/Charon Rust→Lean extraction** (vs our hand-translated `Basic.lean`), (2) **external-spec-authority model** (FIPS/NIST spec independent of implementation interpretation), (3) **`"verify"` feature shims** for verification-only intrinsics models + C-FFI exclusion. All three are YAGNI for us today (79 theorems in 7 days without them), but Aeneas is the credible future tool if hand-translation becomes a bottleneck.

**Distilled for katgpt-rs (process refinement, not a primitive):**
The SymCrypt work validates our Research 351 pattern — same Lean 4 toolchain (`v4.31.0`, matching three of our four instances), same `lean-toolchain` pin, same `lake build` invocation, same axiom-budget discipline (only standard foundation), same "spec-match test catches what Lean's ℝ can't" intuition (their Intrinsics/Axioms vs our spec-match tests on f32/NaN/edge-cases), same "Lean kernel independently re-checks" framing. The delta is the *extraction* approach: SymCrypt auto-extracts; we hand-translate. Our pattern is cheaper for abstract proofs (the freeze/thaw proof abstracts matrices to a `Matrix : Type` token — Aeneas would extract concrete `Vec<f32>`, hurting that proof); SymCrypt's pattern is cheaper for concrete-algorithm proofs (SHA-3, ML-KEM) where the spec IS a published standard.

---

## 1. What SymCrypt Verified-Crypto actually does

| Axis | SymCrypt verifiedcrypto | Our quintet (per Research 351) |
|---|---|---|
| **Spec author** | FIPS/NIST/IETF standard (external authority) | Our own design (internal authority) |
| **Spec↔code bridge** | Aeneas extracts Rust → Lean `Code/`; spec in `Spec/`; proof in `Properties/` | Hand-translated `Basic.lean`; Rust checked by spec-match `#[test]` |
| **Toolchain** | Lean 4 `v4.31.0` + Aeneas `d71d2e3f` + Charon `5a501733` + nightly Rust `nightly-2026-06-01` | Lean 4 `v4.31.0` (3 instances) or `v4.32.0-rc1` (1 instance, Mathlib-required); stable Rust |
| **Tactics** | `bv_decide` (bitvector decision), Mathlib (selectively) | `omega`, `ring`, lattice tactics; Mathlib only when transcendental (sigmoid) |
| **Verification-only Rust** | `#[cfg(feature = "verify")]` shims for intrinsics + C-FFI exclusion | None — spec-match tests under `#[cfg(test)]` |
| **Intrinsics model** | `Intrinsics/Axioms/` = "trusted silicon semantics" transcribed from vendor docs | Implicit — handled by the spec-match test on real f32 hardware paths |
| **Axiom budget** | Standard foundation (propext, choice, quotient) + intrinsics axioms | `{propext, Classical.choice, Quot.sound}` only — all 79 theorems verified by `#print axioms` |
| **Trust assumptions** | Explicit 7-point list: toolchain, Charon/Aeneas, runtime, Lean kernel + native tactics, intrinsics model, spec accuracy, pre/post-condition capture | C2 (axiom policy) + C3 (spec-match) + C6 (README regeneration) — implicit in the convention set |
| **Spec testing** | Spec itself runs on standard test vectors (CAVP/ACVP) inside Lean | Spec-match tests run Rust against spec on f32/NaN/edge-case corpora |
| **Build time** | ~15 min full re-verify | Seconds-to-minutes per instance (Mathlib cache one-time ~5 min) |
| **Coverage** | 3 primitives (SHA-3 15K LOC proofs, ML-KEM 38K LOC, intrinsics 5K LOC) | 79 theorems across 4 instances (sigmoid ranking, LatCal round-trip, quorum, slashing, split-key, shard layout, freeze gate, Merkle tamper-evidence, HLA boundedness, freeze/thaw reader invariant, SSMax dilution-bound) |

## 2. Distillation — three genuinely new techniques

### 2.1 Aeneas/Charon Rust→Lean extraction (the substantive delta)

SymCrypt's `Code/` directory is **extracted** from `src/` via `make extract` (Aeneas + Charon). The reviewer does not write Lean models of the Rust; Aeneas produces them. The reviewer writes `Spec/` (the standard) and `Properties/` (the theorems linking extracted code to spec).

**Where this would help us:** for any future FV instance where the spec is large or the Rust-side data structures are complex, extraction eliminates the hand-translation step. Our `Basic.lean` files are short (the freeze/thaw spec is ~30 lines) because we deliberately *abstract away* irrelevant detail (matrix contents → `Matrix : Type` token). For a future instance that genuinely needs to model concrete Rust structures, Aeneas could save effort.

**Where this would hurt us (the honest caveat):** three of our four proof instances depend on **deliberate abstraction**:
- The freeze/thaw proof (`Runtime/FreezeThaw.lean`) abstracts matrices to a token `Matrix : Type` because the theorem is about *structural atomicity*, not matrix data. Aeneas extracting concrete `LoRAWeightSnapshot { a: Vec<f32>, b: Vec<f32> }` would force the proof to reason about `Vec<f32>` contents, which is irrelevant and would balloon proof state.
- The Merkle tamper-evidence proof parameterizes over an abstract injective `hashFn`. Aeneas would extract the concrete BLAKE3 FFI binding, which is exactly what the proof wants to *abstract away*.
- The LatCal round-trip proof models the math (`round`, `abs`, integer arithmetic). Aeneas would extract the Rust fixed-point struct layout — useful for layout-match, but not for the round-trip theorem.

**Net:** Aeneas is the right tool when the spec is an external standard and the proof is "Rust matches the standard." Our pattern (hand-translated abstract spec + spec-match test) is the right tool when the spec is our own design and the proof is about structural/algebraic properties. **YAGNI for now** — hand-translation has produced 79 theorems in 7 days; the bottleneck has not been translation effort, it has been scoping the right invariant (Issue 354 case study).

### 2.2 External-spec-authority axis (a methodology framing, not a tool)

SymCrypt's `Spec/` is the **FIPS/NIST/IETF standard**, transcribed line-by-line from the published pseudocode. The reviewer can ask "is this the right formal specification?" and check against an *independent* document.

Our `Basic.lean` specs are **self-authored** — we wrote the Lean spec to match our intent, then proved the Rust matches our intent. The spec-match test (C3) catches Rust↔Lean drift, but it does **not** catch intent↔spec drift (we wrote both). If our intent is wrong, the proof is wrong, and no test catches it.

**This is a real methodology gap, not a tooling gap.** For our existing 79 theorems, the specs are either (a) public math (sigmoid monotonicity — Mathlib is the external authority) or (b) structural invariants asserted in doc comments (the freeze/thaw doc comment WAS the spec — and Issue 354 proved the doc comment was *wrong*, which is exactly the failure mode external authority would have caught earlier). For future theorems where the property comes from a paper or a standard, citing the paper/standard as the authority (the way Research notes cite arxiv IDs) is the cheap mitigation. **No new tool required; just a documentation discipline addition to C6.**

### 2.3 `"verify"` feature shims pattern (a Rust-side convention)

SymCrypt gates verification-only code with `#[cfg(feature = "verify")]`:
- Rust executable models of intrinsics (used only inside Lean, never in production)
- Exclusion of C FFI code that's out of scope for verification

Our equivalent is implicit: spec-match tests under `#[cfg(test)]` exercise the production code paths; we don't ship separate verification-only models. The SymCrypt pattern would be useful if we ever needed to ship a *deliberate simplification* of a production function for verification purposes (e.g., a non-SIMD reference implementation of `fast_sigmoid` that Lean can extract, paired with a SIMD production version that the spec-match test proves is bit-equivalent on f32 inputs). **YAGNI** — our spec-match tests already exercise the actual SIMD production path.

## 3. What SymCrypt's framing confirms about our pattern

Reading SymCrypt's README against Research 351 confirms our C1–C6 conventions are **industry-aligned, not idiosyncratic**:

- **C1 (toolchain pin)** — SymCrypt pins `lean-toolchain` to `v4.31.0`. Identical to three of our four instances.
- **C2 (axiom budget)** — SymCrypt's trust-assumptions section explicitly cites "soundness of the Lean kernel" and uses only standard foundational axioms (their intrinsics axioms are analogous to our `arcswap_store_atomicity` documentation-only axiom in `RiirAiProof/Runtime/Basic.lean` — both are honestly labeled as modeling assumptions, not proven facts).
- **C3 (spec-match test)** — SymCrypt's "spec tested on standard vectors" is the same two-way gate: Lean proves the math; executable tests catch drift. Our spec-match tests cover f32/NaN/edge-cases that Lean's ℝ cannot express; SymCrypt's spec-tests cover CAVP/ACVP vectors that their abstract spec must reproduce.
- **C5 (build isolation)** — SymCrypt's `Code/` is committed (extraction artifacts), analogous to our `lake-manifest.json` pinning. Both keep the proof build hermetic.
- **C6 (README discipline)** — SymCrypt's per-primitive `VERIFIED.md` documents the trust footprint of each top-level theorem. Identical in spirit to our per-instance `.proofs/README.md` regeneration protocol.

The SymCrypt framing that **the reviewer need not re-check the bulk of Lean proofs** (only spec, properties, and trust assumptions) is the same insight behind our C2: the Lean kernel is the trust anchor; the human review surface is the spec + axiom inventory. Their phrasing — "This enables us to delegate proof work to agents without trusting the AI machinery" — is a useful articulation: **Lean proofs are AI-delegable by construction** because the kernel re-checks regardless of author. This is relevant to our multi-agent workflow: a sub-agent can write Lean proof tactics and the kernel verifies them. We have not explicitly framed this as a workflow capability; SymCrypt does.

## 4. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier | — |
| **GOAT** | Provable gain over existing approach, promotes to default if it wins | — |
| **Gain** | Incremental improvement, useful but not headline-worthy | **← THIS** |
| **Pass** | Not relevant, OR training-only, OR LLM-orchestration class | — |

**Gain.** Useful process refinements, not a new primitive. The three genuinely-new techniques (Aeneas extraction, external-spec-authority framing, `"verify"` feature shims) are documented here for future reference; none warrants a plan today.

**One-line reasoning:** Our Research 351 pattern (C1–C6 conventions, 79 theorems across 4 instances, spec-match test as the two-way gate) already covers the load-bearing methodology; SymCrypt's contributions are (a) extraction tooling we don't currently need, (b) a spec-authority framing that is a documentation-discipline addition rather than a tool, (c) a Rust-side convention that is YAGNI given our spec-match test pattern.

**Routing:** This note stays in `katgpt-rs/.research/` (public process IP). No private guide — there is no Super-GOAT selling point. No plan — YAGNI on all three techniques. No issue — nothing to track as a PoC.

**MOAT gate (§1.6):** Neutral. This is process refinement to an existing moat (Research 351), not a new pillar candidate. Stays in `katgpt-rs/.research/` as a sibling note to 351.

## 5. When to revisit (the trigger conditions)

Re-open this note and consider a plan IF any of these triggers fire:

1. **A future FV instance requires modeling concrete Rust structures** (not abstract tokens) AND the hand-translation effort exceeds ~1 day → evaluate Aeneas extraction. The freeze/thaw, Merkle, and LatCal proofs would NOT have benefited; a future "verify the SIMD matmul kernel" proof might.
2. **A future FV instance cites a paper or external standard as the property source** → add the citation to the spec README as the external authority (cheap C6 extension, no new tool).
3. **A future proof needs bitvector reasoning** (bit-packing, fixed-point arithmetic with explicit bit-width) → evaluate `bv_decide` tactic. Our current proofs use `omega`/`ring`/lattice; bitvector reasoning has not been needed.
4. **A future proof needs to exclude a complex subsystem** (analogous to SymCrypt excluding C FFI) → evaluate the `"verify"` feature shim pattern.

None of these is currently on the roadmap. The note is a marker for the future, not a commitment.

## 6. Cross-references

- **`katgpt-rs/.research/351_cross_repo_lean4_fv_pattern.md`** — the canonical FV pattern (C1–C6 conventions, 79 theorems, Issue 354 bug-finding case study). This note is a methodology cross-check against it.
- **`katgpt-rs/.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md`** — the original gap analysis (zero machine-checked proofs → tiered plan).
- **`katgpt-rs/.research/198_Lean4Agent_Formal_Workflow_Verification.md`** — a different FV angle (agent workflow verification, deliberately avoided Lean for the agent layer; Tier 3 reversed that decision for the bridge math only).
- **`riir-chain/.research/004_LatCal_Fixed_Point_Bridge_Lean4_Proof_Guide.md`** — Tier 1 instance (LatCal round-trip, the sync-boundary bridge proof).
- **`riir-neuron-db/.research/005_Shard_Snapshot_Atomicity_Iris_Proof_Crossref.md`** — Tier 2 instance (atomicity, deferred — the freeze/thaw reader invariant in RiirAiProof partially covers this).
- **`katgpt-rs/.proofs/README.md`** + **`riir-ai/.proofs/README.md`** + `riir-chain/.proofs/README.md` + `riir-neuron-db/.proofs/README.md` — the four shipped instances.
- **SymCrypt source:** [microsoft/SymCrypt `feature/verifiedcrypto`](https://github.com/microsoft/SymCrypt/blob/feature/verifiedcrypto/README-VERIFIEDCRYPTO.md). Related: [Aeneas project](https://github.com/AeneasVerif/aeneas), [Charon](https://github.com/AeneasVerif/charon).

---

## TL;DR

SymCrypt's verified-crypto branch is a methodology cross-check against our Research 351 FV pattern. Same Lean 4 toolchain, same axiom discipline, same spec-match intuition, same "Lean kernel is the trust anchor" framing. The three genuinely-new techniques (Aeneas Rust→Lean extraction, external-spec-authority model, `"verify"` feature shims) are documented but YAGNI — our hand-translation + spec-match test pattern produced 79 theorems in 7 days, including one that found a real concurrency bug (Issue 354). Aeneas would *hurt* three of our four existing proofs because they depend on deliberate abstraction (token `Matrix : Type`, abstract injective `hashFn`); it would only help for a future "verify concrete Rust matches an external standard" instance, which is not on the roadmap. **Gain — useful marker, no plan, no guide, no commitment.**
