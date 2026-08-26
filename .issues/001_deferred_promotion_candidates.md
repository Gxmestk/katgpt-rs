# Issue 001 — Deferred crate-promotion candidates

Status: **RESOLVED** (2026-08-26). All candidates landed. Both "genuinely
OPEN" T2 decisions were executed by Proposal 003 Phase 12 commits shortly
after the 2026-07-04 audit — the header below was stale from then until
this resolution note:
  - **Candidate C — `dash_attn/`**: ✅ landed via Proposal 002 + Proposal 003
    Phase 2 (`katgpt-attn` crate).
  - **Candidate A — `mux_latent/`**: ✅ **RESOLVED** — moved to
    `crates/katgpt-core/src/mux_latent/` as its **own module** in commit
    `348347bd` (Phase 12 T4.3, "move 4 folders to katgpt-core"). NOT merged
    into `katgpt-core/src/mux/` — the audit's "they are unrelated
    subsystems" separation holds (verified 2026-08-26: the two modules are
    siblings with zero cross-imports). Verified healthy: 35/35 lib tests +
    7/7 `bench_238_mux_latent_goat` under `mux_latent_context`.
  - **Candidate B — `proof_cert/`**: ✅ **RESOLVED** — extracted to
    `crates/katgpt-proof-cert/` in commit `cf23050a` (Proposal 003 Phase 12
    T1+T4.1), exactly the audit's T2 recommendation (katgpt-rs-local crate;
    NOT riir-chain). Root re-exports preserved (`katgpt_rs::proof_cert::*`).
    Verified healthy: compiles default, 8/8 tests with `wasm_proof_witness`.
The earlier "all candidates scheduled in Proposal 003" claim was aspirational
and inaccurate at the time; Phase 12's final sweep subsequently covered both.
Created: 2026-07-01
Status corrected: 2026-07-04
Resolved: 2026-08-26 (verification pass: compile + tests, evidence above)
Related proposal: `.proposals/003_src_consolidation_master.md` (Phase 12 executed A + B)

## Context

Promotion analysis of `katgpt-rs/src/` surfaced four candidates beyond the
quant family. The quant family is being promoted first (Proposal 001) because
it has the cleanest boundary. The three below are deferred — each has a
boundary or coupling issue that needs untangling before promotion is
worth the churn.

This issue tracks the deferred work so it isn't lost.

---

## Candidate A — `mux_latent/` (12 files)

**Status:** deferred — fuzzy MUX dependency boundary. **(DEFERRAL PREMISE DISPROVED 2026-07-04 — see T1 audit result below.)**

**What it is:** Inference-time context compression via vocabulary
superposition (distilled from LCLM, arXiv:2606.09659). Pipeline: input
tokens → MUX superposition encoder → latent slots → domain_latent
mid-layer injection. Modelless (uses existing MUX infrastructure).

```
mux_latent/
├── buffer.rs          # LatentContextBuffer, EvictionPolicy
├── config.rs          # MuxLatentConfig, CompressionRatio
├── context.rs         # CompressedContext, LatentSegment
├── encoder.rs         # MuxLatentEncoder
├── expand.rs          # segment expansion
├── inject.rs          # LatentPrefillAdapter, MixedPrefillSequence
├── mod.rs
├── prefill.rs         # forward_prefill_with_compression
├── spectral_lod.rs    # SpectralLOD
├── octree_bridge.rs   # gated mux_latent_wire
├── patcher.rs         # gated mux_latent_wire
└── wire.rs            # gated mux_latent_wire
```

**Why deferred (ORIGINAL rationale — now disproved at T1):** `mux_latent`
depends on existing MUX infrastructure that already lives in
`katgpt-core/src/mux/`. Promoting `mux_latent` alone would create a
circular or awkward dep (new crate → katgpt-core::mux, while katgpt-core
may want to re-export the new crate). The MUX substrate needs to be split
out *first* (or `mux_latent` needs to be folded into the existing mux
module rather than its own crate).

**⚠ AUDIT RESULT (T1, 2026-07-04): the deferral premise is FALSE.** Grep
of `src/mux_latent/` shows:
  - **Zero** `use katgpt_core::...` imports (the entire subsystem is
    std-only internally; cross-refs are all `crate::mux_latent::*`).
  - **Zero** references to `katgpt_core::mux` or `core::mux`.
  - Only one external consumer: `src/lib.rs:320` (`pub mod mux_latent;`
    under `#[cfg(feature = "mux_latent_context")]`).

  The name collision is misleading: `katgpt-core/src/mux/` holds
  speculative-decoding multi-token drafting primitives (`dd_tree`, `demux`,
  `span_pruner`, `top_k`, `bfs`, `bandit_width`, `freeze_thaw`) — i.e.
  "multiplexed draft tokens". `src/mux_latent/` is LCLM-style context
  compression via vocabulary superposition into latent slots. They are
  unrelated subsystems that happen to share the "mux" prefix; there is no
  primitive/application relationship and no dependency edge in either
  direction.

  **Consequence:** the circular-dep concern that drove the deferral does
  not exist. mux_latent is fully self-contained and could be lifted to its
  own crate without any katgpt-core entanglement. T2's option (b) "fold
  into katgpt-core::mux" is also wrong — they don't belong together.

**Unblock criteria:**
- [x] **T1 — Audit `katgpt-core/src/mux/` vs `src/mux_latent/` — is the
      split "mux primitive" (core) vs "mux application" (mux_latent) clean?**
      **ANSWER: there is no split — they are unrelated. mux_latent has
      zero katgpt-core deps. Deferral premise disproved.** (Audit
      performed 2026-07-04: grep `use katgpt_core` in `src/mux_latent/`
      returns zero hits; grep `katgpt_core::mux` returns zero hits; only
      consumer is `src/lib.rs:320`.)
- [x] **T2 — Decide promotion target. Original options reframed by T1
      finding:**
      - ~~(a) promote mux primitive + mux_latent together into `katgpt-mux`~~ —
        **INVALID**: they are unrelated; bundling them would create a false
        semantic grouping.
      - ~~(b) fold mux_latent into `katgpt-core::mux`~~ — **INVALID**:
        they are different concerns; `katgpt-core::mux` is spec-decode
        multi-token drafting, mux_latent is LCLM context compression.
      - **(c) keep as-is in root `src/mux_latent/`** — rejected: leaves
        ~104 KB of LCLM code in the root crate.
      - **(d) promote to its own crate** — superseded by the landed decision.
      - **(e) fold into katgpt-sleep / katgpt-context** — not taken.
      **DECISION (executed, commit `348347bd`, Phase 12 T4.3): fold into
      `katgpt-core` as its own top-level module**
      `katgpt-core/src/mux_latent/` — a hybrid none of the original options
      named exactly: consolidated into core (removing it from the root
      crate, option (c)'s complaint) while preserving the module boundary
      the audit demanded (NOT inside `katgpt-core::mux`, option (b)'s
      invalid grouping). Feature `mux_latent_context` ("DEFAULT-ON in root"
      per katgpt-core Cargo.toml L578). Verified 2026-08-26: zero
      cross-imports between `mux/` and `mux_latent/`; 35/35 lib tests +
      7/7 bench_238 GOAT pass.
- [-] **T3 — If (a): write Proposal 003 with the full MUX closure.**
      **DEFERRED — moot.** T1 disproved the premise that made (a) an
      option. T2's landed decision (fold into katgpt-core) was executed
      directly as Phase 12 T4.3 — no separate proposal was needed.
      CLOSED as moot with the T2 resolution.

---

## Candidate B — `proof_cert/` (7 files)

**Status:** deferred — cross-cuts chain/WASM runtime. **(DEFERRAL PREMISE LARGELY DISPROVED 2026-07-04 — see T1 audit result below.)**

**What it is:** Proof certificate chain — verification/integrity substrate.
Emits and validates certificates for runtime artifacts. Origin: Plan 145
("Hierarchical GOAT Proof Certificates") — standalone, serializable proof
certificates with dependency chains, topological verification, and blake3
checksum integrity.

```
proof_cert/
├── certificate.rs
├── chain.rs
├── macros.rs
├── mod.rs
├── serde_impls.rs
├── wasm_certificates.rs
└── wasm_proof_witness.rs   # gated: feature wasm_proof_witness
```

**Why deferred (ORIGINAL rationale — now largely disproved at T1):**
`proof_cert` cross-cuts the chain runtime (riir-chain has its own proof
concerns per its AGENTS.md) and the WASM runtime (`wasm_certificates.rs`,
`wasm_proof_witness.rs`). Promoting it into a `katgpt-proof-cert` crate
risks duplicating or conflicting with riir-chain's proof envelope
(`riir-neuron-db` owns `freeze.rs` / `FreezeGateReport`; riir-chain owns
`catchup/merkle.rs`). The boundary across the 5-repo quintet needs design,
not just a local lift.

**⚠ AUDIT RESULT (T1, 2026-07-04): the deferral premise is largely FALSE.**
Grep of `src/proof_cert/` shows:
  - **Zero** `use crate::...` imports and **zero** `use katgpt_core::...`
    imports — the subsystem is fully self-contained (only `super::` refs).
  - **Zero** runtime deps on a WASM engine. Despite the misleading names
    (`wasm_certificates.rs`, `wasm_proof_witness.rs`), grep for
    `wasmi|wasmtime|wasm_bindgen` returns **zero hits**. The "wasm" in the
    names refers to *certificates that describe wasm-validator outcomes*
    (e.g. `lora_wasm_delta: i32` as a metric value, `challenger: "wasm"`
    as a tag) — the module produces certificates *about* wasm validation,
    it does not *execute* wasm. The entire module compiles std-only.
  - Only one external consumer: `src/lib.rs:184` (`pub mod proof_cert;`
    under `#[cfg(feature = "proof_cert")]`). No internal katgpt-rs consumers
    at all.

  **Consequence:** the "cross-cuts WASM runtime" concern is wrong — there
  is no wasm runtime dep. The "cross-cuts chain runtime" concern is
  *semantic*, not technical: proof_cert (Plan 145 GOAT proof certificates)
  and riir-neuron-db's freeze/Merkle (shard integrity envelopes) serve
  different proof domains but might overlap conceptually. That's a
  design/semantic question, not a dep-graph blocker — proof_cert can be
  lifted cleanly on dep-graph grounds alone.

  **Cross-repo quintet proof surface (audit, 2026-07-04):**
    - `katgpt-rs/crates/katgpt-core/src/merkle.rs` + `content_store/merkle.rs`
      — content-addressed blob storage (local).
    - `katgpt-rs/crates/katgpt-core/src/mux/freeze_thaw.rs` — spec-decode
      freeze/thaw (unrelated to proof_cert).
    - `riir-neuron-db/src/freeze.rs` — `FreezeGateReport`, freeze/thaw
      integrity envelope (shard-level, committed).
    - `riir-neuron-db/src/merkle.rs` — generic BLAKE3 binary Merkle tree
      (shard-level proofs).
    - `riir-chain/src/catchup/merkle.rs` (per riir-chain AGENTS.md) — chain
      block commitment.

  None of these implement "GOAT gate proof certificates with dependency
  chains + topological verification" (proof_cert's actual concern). The
  overlap is at the word "proof" / "certificate", not at the algorithm.

**Unblock criteria:**
- [x] **T1 — Map the proof surface across the quintet: what does
      katgpt-rs's `proof_cert` prove that riir-chain's merkle proofs and
      riir-neuron-db's `FreezeGateReport` don't?**
      **ANSWER: proof_cert (Plan 145) implements hierarchical GOAT proof
      certificates — dependency chains + topological verification of GOAT
      gate outcomes (ProofProperty/ProofResult/ProofEvidence). The
      quintet's other proof surfaces are shard-integrity (freeze/Merkle)
      or chain-commitment (catchup/Merkle) — they do NOT implement GOAT
      gate dependency chains. The surfaces are disjoint at the algorithm
      level; the "overlap" was lexical (the word "proof"), not technical.**
      (Audit 2026-07-04: grep `use katgpt_core` and `use crate::` in
      `src/proof_cert/` returns zero; grep `wasmi|wasmtime|wasm_bindgen`
      returns zero; quintet proof-surface map above.)
- [x] **T2 — Decide: is this a katgpt-rs-local crate, or does it belong
      in a different repo (riir-chain)?**
      **DECISION: katgpt-rs-local `katgpt-proof-cert` crate — the audit's
      recommendation, taken verbatim.** Executed in commit `cf23050a`
      (Proposal 003 Phase 12 T1+T4.1, 2026-07-04): extracted to
      `crates/katgpt-proof-cert/` (serde + postcard + blake3 always-on;
      `wasm_proof_witness` feature gates the witness subset), root
      re-exports preserved (`katgpt_rs::proof_cert::*` + the
      `conditional_proof!` macro). riir-chain consumes nothing here —
      it doesn't own the algorithm. Verified 2026-08-26: compiles
      default-clean; 8/8 tests under `wasm_proof_witness` (0/0 default,
      matching the original module's gated-test coverage).
- [x] **T3 — If katgpt-rs-local: confirm the WASM coupling can be
      feature-gated so the crate compiles without a WASM runtime.**
      **ANSWER: yes — it already is.** `wasm_proof_witness` is gated by
      `#[cfg(feature = "wasm_proof_witness")]` in mod.rs:6,11. The other
      "wasm" file (`wasm_certificates.rs`) has no wasm-runtime dep at all
      (T1 finding) — it compiles std-only. So the entire crate compiles
      with zero wasm deps under the default feature set; only the
      opt-in `wasm_proof_witness` feature adds the witness generator.
- [-] **T4 — Write Proposal 004 with the cross-repo boundary decision.**
      **DEFERRED — CLOSED as moot with T2.** T2 was decided + executed
      inside Proposal 003 Phase 12 (`cf23050a`); a separate cross-repo
      boundary proposal became unnecessary — the T1 audit had already
      established the surfaces are disjoint at the algorithm level, and
      the extraction landed katgpt-rs-local with no riir-chain involvement.
      (Note: Proposal 004 number was already taken; a new proposal was
      never filed because none was needed.)

---

## Candidate C — `dash_attn/` (16 files) — NOT deferred, separate proposal

**Status:** strong candidate, deserves own proposal (002).

Listed here only for cross-reference. `dash_attn` is the biggest single
module in `src/` (adaptive sparse hierarchical attention via α-entmax
routing, Plan 106/196 lineage, vortex_flow/msa_*/cache_prune feature
surface). It is *not* deferred — it's the natural Proposal 002 follow-up
to the quant promotion. Distinct from both `katgpt-core`'s base attention
primitives and `katgpt-attn-match` (KV compaction, Plan 271).

- [x] Write Proposal 002 — `katgpt-dash-attn` crate promotion. See
      `.proposals/002_dash_attn_crate_promotion.md`.
      **Key nuance captured:** unlike the quant family, `dash_attn` is NOT
      a clean leaf — `forward.rs` + `tests.rs` are hard-coupled to
      `crate::transformer::ForwardContext` (which lives in root, not in
      the types-only `katgpt-transformer` crate). Proposal 002 splits the
      module: 13 primitive/routing files promote to the crate; 2
      transformer-integration files stay in root. Mirrors the
      `katgpt-attn-match` (Plan 271 / Issue 359) precedent.

---

## Non-candidate — `src/sleep/` vs `katgpt-sleep` crate (clarification)

**Not a promotion candidate — they are different things. Do not merge.**

| | `src/sleep/` | `crates/katgpt-sleep/` |
|---|---|---|
| Paper | Plan 154 (GDN2 fast-weight consolidation) | arXiv:2504.13171 (Lin et al., Sleep-Time Query Anticipator) |
| Concern | offline recursive memory consolidation *at eviction* | offline query *anticipation*, wake-time consume() |
| Mechanism | N recurrent passes into GDN2 fast weights → evict KV | per-direction sleep-time compute → AnticipatedQuerySet (c' artifact) |
| Feature gate | `sleep_consolidation` (deps `lt2_looped`, `gdn2_attention`) | `sleep_time_anticipation` (forwards to `dep:katgpt-sleep`) |

They share the word "sleep" but are unrelated substrates. `src/sleep/` is
*not* stale source for the `katgpt-sleep` crate. No action.

---

## Priority order when revisiting

All three landed — nothing left to revisit.

1. ~~**Proposal 002 — `dash_attn`**~~ — DONE (`katgpt-attn` crate, Proposals
   002 + 003 Phase 2).
2. ~~**Candidate A — `mux_latent`**~~ — DONE (`katgpt-core/src/mux_latent/`,
   commit `348347bd`).
3. ~~**Candidate B — `proof_cert`**~~ — DONE (`katgpt-proof-cert` crate,
   commit `cf23050a`).

## TL;DR

`mux_latent` and `proof_cert` were real promotion candidates whose deferral
premises the 2026-07-04 audit disproved; both landed in Proposal 003 Phase
12 (2026-07-04): mux_latent as its own `katgpt-core` module (NOT inside
`mux/`), proof_cert as the `katgpt-proof-cert` crate. `dash_attn` landed via
Proposal 002. `src/sleep/` was never a duplicate of the `katgpt-sleep` crate
— different papers, left alone (still true: `src/sleep/` remains the
`sleep_consolidation` GDN2 substrate). Issue fully RESOLVED 2026-08-26.
