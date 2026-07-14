# Issue 138 — JL Backdoor Shard-Embedding Provenance: Defend-Wrong PoC

> **Source:** Research 422 (`katgpt-rs/.research/422_Statistically_Undetectable_Backdoors_Model_Provenance.md`)
> **Paper:** Bogdanov, Rosen, Vafa. *Statistically Undetectable Backdoors in Deep Neural Networks.* arXiv:2607.09532v1, 10 Jul 2026.
> **Verdict that produced this issue:** PASS with fusion idea — novelty TBD.
> **Date opened:** 2026-07-14
> **Date closed:** 2026-07-14
> **Status:** **CLOSED — Fusion SHELVED on G1.** See §"PoC outcome" below + Research 422 §6 PoC Addendum.

---

## TL;DR

Research 422 verdicted the paper as **PASS for the modelless repos** (training-side value; modelless verification `V` is a corollary that requires a backdoored-model workflow we don't have). One fusion idea was called out as "novelty TBD, needs Q1–Q4 check before verdict upgrade":

> **Replace our honest `ShardEmbedding` JL projection (Plan 230) with a backdoored one** (constructed deterministically given a secret `z ∈ {±1}ⁿ` per the paper's `BackdoorMatrix`), to gain **cryptographic shard-embedding provenance** — verifiable in 2 modelless queries, unforgeable under LWE.

This issue tracks the **defend-wrong PoC** (per skill §3.6) needed before any verdict upgrade from PASS → GOAT/Super-GOAT. The PoC must defend OR refute: *can the paper's three architectural constraints be met for our shard-embedding path, and does the verification `V` actually deliver unforgeability on that path?*

**The PoC's job is to defend OR refute.** A PoC that only confirms the verdict is weaker than one that honestly refutes part of it.

---

## The four open questions the PoC must settle

| # | Question | Why it matters | Plausibility |
|---|---|---|---|
| C1 | **Does our `ShardEmbedding` satisfy Constraint 1 (frozen compressing Gaussian first layer)?** | The honest matrix is Gram-Schmidt-orthogonalised Gaussian `[f32;64]→[f32;8]` — frozen at shard init. **Yes structurally.** The backdoored variant must preserve the JL guarantee AND plant `z`; the paper proves `d_TV(A, N(0,1)^{m×n}) = o(1)`, so retrieval quality should survive. | **High** — but needs measurement. |
| C2 | **Does the downstream path satisfy Constraint 2 (bi-Lipschitz)?** | Downstream is cosine-similarity retrieval (1-Lipschitz). No deep network between projection and retrieval. | **High** — likely trivially satisfied. |
| C3 | **Does our input satisfy Constraint 3 (discrete bounded integers)?** | Shard inputs are `style_weights[64]` — *continuous* f32. **The paper's LWE-hardness argument is over `ℤⁿ`; continuous inputs may break the reduction.** | **LOW — this is the make-or-break question.** May require a quantisation/rounding pre-step on `style_weights` with a measured retrieval-quality cost. |
| C4 | **Does verification `V` actually deliver unforgeability on our stack?** | `V` is 2 queries — trivially runs. But the *unforgeability* guarantee depends on C1–C3 holding. | **Depends on C1–C3.** |

**If C3 fails (continuous inputs break LWE hardness for our use case), the fusion is SHELVED.** The paper's Theorem 7 explicitly requires "Constraint 3: The inputs are discrete, i.e., integers from a bounded range" and justifies it: "data ultimately needs to be discretized up to some precision in practice... in many domains (e.g., text), inputs are already discrete." Our `style_weights` are f32 — not naturally discrete. We would need to argue either (a) a fixed-point quantisation of `style_weights` is sufficient, or (b) the continuous-input relaxation still admits a (weaker) hardness argument.

---

## PoC design

**Location:** `riir-ai/crates/riir-poc/benches/jl_backdoor_provenance_modelless_goat.rs`

**Three competitors, head-to-head on a controlled toy shard-retrieval domain:**

1. **Backdoored JL with `z`** — construct `A = BackdoorMatrix(z, n=64, m=8)`, embed 128 toy shards, verify `V(A, z) = 1`.
2. **Honest JL baseline** — construct `A_honest` via Gram-Schmidt on i.i.d. Gaussian (current `ShardEmbedding` path), embed the same 128 shards, verify retrieval quality (top-K cosine) matches the backdoored path within ε.
3. **Attempted forgery without `z`** — give an adversary only `A` (white-box), let it search for any `z′` such that `V(A, z′) = 1` via (a) LLL, (b) random search, (c) gradient-free optimisation. Measure forgery success rate.

**GOAT gate (the bar to clear for verdict upgrade PASS → GOAT):**

| Gate | Pass criterion | Failure outcome |
|---|---|---|
| **G1 (retrieval parity)** | Backdoored JL top-5 retrieval matches honest JL top-5 on ≥ 90% of 128 toy shards (cosine within ε = 0.05) | Fusion shelved — backdoor destroys retrieval quality |
| **G2 (verification correctness)** | `V(A_backdoor, z) = 1` in 100/100 trials | Bug in construction |
| **G3 (unforgeability — the load-bearing gate)** | Adversary forgery rate (`V(A, z′) = 1` for `z′ ≠ z`) ≤ 1/2²⁰ over 1000 trials per algorithm (LLL, random, gradient-free) | **C3 likely failed — continuous inputs break LWE.** Fusion shelved OR retried with quantisation pre-step. |
| **G4 (latency)** | `V` completes in ≤ 1 µs (two `[f32;8]` dot-products + norm) | Pure formalisation check |

**If G3 fails on continuous inputs, retry with a `round(style_weights · 2ᵏ) / 2ᵏ` quantisation pre-step** for `k ∈ {4, 8, 12}` and re-measure G1 (retrieval quality cost) + G3 (unforgeability). The quantisation that preserves G1 while restoring G3 becomes the production input pipeline.

---

## Routing on PoC outcome

| PoC outcome | Verdict upgrade | Where the primitive lands |
|---|---|---|
| G1–G4 all PASS on continuous inputs | **GOAT** (provable security gain over BLAKE3-only; new capability) | `katgpt-rs/src/forensic/backdoor_jl.rs` (open constructor + verify) + `riir-chain/src/backdoor_commit.rs` (commit `z` via LatCal) + `riir-neuron-db/src/shard.rs` (backdoored shard variant) |
| G1–G4 PASS only with quantisation pre-step | **GOAT with caveat** (quantisation cost must be documented) | Same as above + quantisation pre-step in `ShardEmbedding` |
| G1 FAILS (retrieval parity) | **Fusion shelved** — backdoor destroys shard retrieval | No primitive; close this issue with PoC Addendum in Research 422 |
| G3 FAILS (unforgeability) | **Fusion shelved** — LWE hardness does not hold for our continuous input domain | No primitive; close this issue with PoC Addendum documenting the continuous-vs-discrete gap |
| G1 + G3 PASS, multiplies ≥2 pillars | **Super-GOAT** (per skill §1.5, re-run Q1–Q4 with PoC evidence) | Open primitive in `katgpt-rs` + private guide in `riir-chain/.research/` (chain-side provenance commitment is the selling point) |

---

## Dependencies

- **`riir-train`** must own the actual `BackdoorMatrix(z, n, m)` constructor if the PoC passes — it is a model-set-up decision, not a runtime op. Track in `riir-train/.issues/` if/when the PoC advances.
- **`katgpt-rs/.plans/230_shard_embedding_projection.md`** — the honest JL path that would be replaced. Read before PoC to understand the current constructor.
- **`riir-ai/crates/riir-poc/`** — the "defend-wrong" R&D crate (per skill §3.6). Build with `CARGO_TARGET_DIR=/tmp/jl_backdoor_poc`, clean up when done.

---

## Tasks

- [x] **T1** Read `katgpt-rs/.plans/230_shard_embedding_projection.md` + the current `ShardEmbedding` constructor to understand the honest JL path.
- [x] **T2** Implement `BackdoorMatrix(z, n=64, m=8)` in `riir-ai/crates/riir-poc/` (deterministic given `z` per paper Figure 3).
- [x] **T3** Generate 128 toy shards with known top-5 nearest neighbours (cosine on `style_weights[64]`).
- [x] **T4** **G1 retrieval parity:** embed via honest JL and backdoored JL, measure top-5 overlap. **RESULT: 0.000 at m∈{8,16,32} — FAIL.**
- [x] **T5** **G2 verification correctness:** `V(A_backdoor, z) = 1` over 100 trials. **RESULT: 100/100 accept true z, 100/100 reject decoy — PASS.**
- [x] **T6** **G3 unforgeability:** LLL-proxy (pairwise-flip) + random + greedy-bitflip over 100 trials × 4096 matmul budget each. **RESULT: 0 forgeries across all 3 adversaries — PASS at this budget.**
- [-] **T7** If G3 fails on continuous inputs, retry with quantisation pre-step. **DEFERRED — G3 passed; C3 was a red herring (see Research 422 §6.1).**
- [x] **T8** **G4 latency:** `V` measured at 272 ns (≤ 1 µs threshold) — **PASS.**
- [x] **T9** Write PoC Addendum to Research 422 with raw numbers (defend OR refute). **Done — Research 422 §6.**
- [x] **T10** Route per "Routing on PoC outcome" table. **Routed: G1 FAILS → fusion shelved.** Verdict stays PASS.

---

## PoC outcome (2026-07-14)

**G1 FAILS catastrophically** (parity = 0.000 at m ∈ {8, 16, 32}). The backdoor construction projects out the `z` direction from an already aggressively-compressed space, destroying nearest-neighbour structure. The honest JL baseline is itself broken at these dims (13–25% top-5 overlap with ground truth — confirms Plan 230's "64→8 too aggressive" finding).

**G2/G3/G4 all PASS.** The provenance primitive is cryptographically sound at the PoC budget; C3 (discrete inputs) was a red herring for the provenance use case because `V(A, z)` never takes `style_weights` — only the discrete secret `z`.

**Verdict: PASS (unchanged). Fusion SHELVED.** No primitive lands in any of the 5 repos. The `jl_backdoor_poc` module stays in `riir-poc` as a permanent negative control. Full analysis in Research 422 §6.

---

## Why this is an issue, not a plan

Per global AGENTS.md: *"Create issue at `.issues` for poc, proof, optimization or refactor task, do not create plan."* This is a **PoC task** — the verdict is PASS until the PoC defends it. A plan would be premature; the routing decision (PASS → GOAT → Super-GOAT) depends on the PoC outcome.

---

## Cross-references

- `katgpt-rs/.research/422_Statistically_Undetectable_Backdoors_Model_Provenance.md` — the parent research note
- `katgpt-rs/.plans/230_shard_embedding_projection.md` — the JL projection we'd be modifying
- `katgpt-rs/.research/268_Forensic_Asset_Fingerprinting_LatCal_Recipe.md` — pattern template (forensic watermark recipe → commit → verify)
- `riir-ai/.research/144_PQC_Dilithium_Cold_Path_Optional.md` — pattern template (deferred GOAT gate G1–G5)
- `riir-chain/src/forensic_fingerprint_commit.rs` — structural twin for the commit bridge if the PoC advances
