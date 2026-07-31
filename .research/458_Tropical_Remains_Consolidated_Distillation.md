# Research 458: Tropical `(max,+)` Remains — Consolidated Distillation (4 Candidates)

> **Source:** Smets, *Mathematics of Neural Networks* [arXiv:2403.04807] Ch.3 §3.5 — follow-up to Research 321 §2.4 de-deferral.
> **Date:** 2026-07-25
> **Status:** Done
> **Related Research:** 321 (Tropical Semiring parent), 299 (Clifford Geometric Product), 457 (SE(2)-equivariant lift, sibling).
> **Related Plans:** 337 (tropical_algebra Super-GOAT, shipped).
> **Classification:** Public

---

## TL;DR

User instruction ("dont skip remains") required evaluating **all 4 deferred Tropical-fusion candidates** from Research 321 §2.4, not just #1 (SE(2), shipped as Research 457 / Plan 560). This note records the consolidated verdicts.

**Headline:** **2 of 4 PASS (shipped / not-applicable), 2 of 4 PASS (no modelless gain after honest evaluation). Zero new primitives ship.** Research 321's "speculative" labels were correct on all 3 — none rises to Super-GOAT/GOAT. The strongest candidate (TropicalFunctor) had **already shipped silently** under the `tropical_algebra` feature flag, which the §2.4 note missed.

**Why nothing ships:** Tropical algebra shines as a **bottleneck/worst-case aggregation** of already-computed signals. For all 4 candidates, the existing primitive either already does the right thing (linear sum over cosines is correct for *expected coherence*; cosine retrieval is correct for *nearest-match retrieval*) or breaks the host primitive's contract (LatCal homomorphism). The tropical variant adds a second-best aggregation of the same data without adding a new capability the consumer needs.

**Distilled for katgpt-rs (modelless, inference-time):** nothing new. The existing `tropical_*` primitives in `katgpt-core/src/algebra/tropical.rs` + `riir-engine/src/latent_functor/arithmetic/tropical_extract_functor_into` are the right scope.

---

## 1. Candidate-by-Candidate Verdict

### 1.1 TropicalFunctor — `max_k cos(target_k − source_k, f)` (latent_functor × tropical)

**Verdict: PASS — already shipped silently.**

Research 321 §2.4 described this as "strong; new signal: max-pair displacement coherence vs mean-pair displacement coherence". A vocabulary-translated grep for `tropical_extract_functor` finds it: `riir-ai/crates/riir-engine/src/latent_functor/arithmetic/mod.rs:316` ships `tropical_extract_functor_into`, gated by `feature = "tropical_algebra"` (the same flag Plan 337 promoted to DEFAULT-ON).

The implementation does exactly what §2.4 described:
- Pass 1: linear mean-displacement direction `f = (1/N) Σ_k (target_k − source_k)` — identical to `extract_functor_into` (the direction is a stable central-tendency estimate; replacing it with a single max-displacement would be noise-dominated).
- Pass 2: tropical coherence `max_k cos(disp_k, f)` — best-pair alignment, vs the linear `mean_k cos(disp_k, f)`.
- 6 unit tests pin the contract in `latent_functor/arithmetic/tropical_tests.rs`.

This is the canonical **#3 failure mode** the research skill warns about: a notes-only grep missed a primitive that shipped without a dedicated `.research/` note. The §2.4 note should have included a codebase-vocabulary grep (`tropical_extract`, not just `tropical`); it would have hit `latent_functor/arithmetic/` immediately.

**Follow-up (optional):** No action needed. The primitive is opt-in (gated by `tropical_algebra`) — if a consumer needs the tropical coherence scalar, it's already wired through the latent_functor substrate. The fact that no consumer calls it today is itself evidence that the tropical coherence signal is not a product-grade capability gap.

---

### 1.2 TropicalShardRetrieval — `max_d (w_d + q_d)` (shard retrieval × tropical)

**Verdict: PASS — no modelless gain over cosine retrieval.**

Research 321 §2.4 labeled this *speculative* and noted "may be redundant with max-wedge-span diverse retrieval." Grepping `riir-neuron-db/src/item_index.rs` confirms the existing retrieval is **cosine-similarity top-k**: `cosine_sim_ranking_mul` ranks candidates by `dot(q, w) / (|q|·|w|)`. There is no `(max, +)` retrieval.

**The honest analysis:** Tropical retrieval `max_d (q_d + w_d)` over index dimensions would compute "the dimension where the query + the weight align most strongly" — a *bottleneck* match, not an *average* match. Whether this is useful depends on the retrieval task:

| Retrieval task | Linear cosine wins | Tropical max wins |
|---|---|---|
| Nearest-match (semantic) | ✓ (small angle → high cos) | ✗ (one lucky dim doesn't mean semantic match) |
| Worst-case survival path (game map threat) | ✗ (averages over terrain) | ✓ (bottleneck dominates) — but this is `tropical_line_integral`, already shipped |
| Diverse subset (max-wedge-span) | ✗ (linear wedge is structural) | ✗ (tropical wedge is non-sensical — wedge is anti-symmetric) |

The retrieval use case for shards is nearest-match (semantic), which is exactly where cosine wins and tropical loses. The bottleneck match use case is already served by `tropical_line_integral` on cochain fields — a different substrate. Forcing `max_d (w_d + q_d)` into `ItemEmbedIndex::query_top_k` would compute a different scalar that nobody asked for.

**No Super-GOAT criteria met:**
- Q1 (no prior art): N/A — mechanism is trivially derivable from any `(max,+)` tutorial.
- Q2 (new class?): ✗ — bottleneck retrieval is not a new capability; it's a different aggregation of the same query/weight data.
- Q3 (selling point?): ✗ — cannot finish the sentence "Our NPCs retrieve shards via max-plus and that's better than cosine because..." without contriving a use case.
- Q4 (force multiplier?): ✗ — no existing pillar consumes it.

---

### 1.3 TropicalLatCal — `(max,+)` commitment arithmetic in `riir-chain/src/encoding/latcal.rs`

**Verdict: PASS — not applicable; breaks the LatCal homomorphism contract.**

Research 321 §2.4 labeled this *speculative* and noted "No clear modelless unblock. Flag for riir-chain follow-up." Reading `LatCalMatrix::add` at `riir-chain/src/encoding/latcal.rs:265` confirms the intuition:

```rust
pub fn add(&self, other: &Self) -> Self {
    Self {
        value: self.value + other.value,
        overflow: self.overflow + other.overflow,
        sign: self.sign + other.sign,
        precision: self.precision + other.precision,
    }
}
```

LatCal is a **fixed-point arithmetic obfuscation primitive** whose entire value is that it **preserves linear arithmetic structure** across a deterministic commitment: `decode(encode(a) + encode(b)) = a + b`. This is the sync-boundary bridge — raw numeric commitments must be bit-identical across nodes (per AGENTS.md §Sync Boundary Rule).

**Replacing `+` with `max` breaks the homomorphism:**
- `decode(max(encode(a), encode(b)))` ≠ `max(decode(encode(a)), decode(encode(b)))` in general.
- Worse: `max` is not even defined across the 4-component `LatCalMatrix` `{value, overflow, sign, precision}` — which component's max wins? Per-component max is a different matrix entirely, not a max of the encoded value.
- Quorum commitment becomes non-deterministic: 5 nodes applying `max` over 5 commitment fields could disagree on which component's max applies per slot.

This is the same lesson as the AGENTS.md anti-pattern "Never use latent similarity to validate movement claims — deterministic replay needs exact (x, y) at exact tick." LatCal is the deterministic-replay substrate; making it tropical breaks determinism.

**Where (max,+) *could* fit LatCal-adjacently:** aggregating commitment *metadata* (e.g., "max pending commit tick") — but that's plain `f64::max` over raw tick values, not a tropical extension of the LatCal algebra. No new primitive needed.

---

### 1.4 TropicalGeometricProduct — `max_s W_s` instead of `Σ_s W_s` (Clifford × tropical)

**Verdict: PASS — no modelless gain; the wedge is anti-symmetric, so tropical aggregation is information-lossy.**

Research 321 §2.4 labeled this *speculative* and noted "Max-plus wedge; unclear added value over default-on `geometric_product`." Reading `katgpt-rs/crates/katgpt-core/src/linalg/geometric_product.rs` confirms:

The Clifford wedge at shift `s` is:
```text
W_s(u, v)[c] = u[c] · v[(c+s) mod D]  −  u[(c+s) mod D] · v[c]
```
It is **anti-symmetric** in `(u, v)` and **per-channel** (indexed by `c`). The accumulation across shifts is linear: `wedge_out[c] = Σ_s W_s[c]`.

A tropical variant `wedge_out_max[c] = max_s W_s[c]` would compute "the strongest cross-channel divergence per channel". Sounds useful — but:

1. **Sign cancellation lost.** The linear sum has the property that two equally-strong divergences of opposite sign cancel (information-preserving). The tropical max picks one and silently discards the other (information-destroying).
2. **Anti-symmetry + max is non-invertible.** Replacing `Σ` with `max` over an anti-symmetric field produces a field that is *no longer* anti-symmetric — `max_s W_s(u,v) ≠ -max_s W_s(v,u)` in general (the maxima can be at different `s`). This breaks the rotor-algebra property that consumers downstream might rely on.
3. **The Plan 319 G8c bench already showed wedge + greedy-min selection is the right consumer pattern** — it uses `Σ_s` to compute a single L1 norm, then selects for min-pairwise. `max_s` would break this aggregation in a way that no consumer benefits from.

The legitimate bottleneck-shaped use of the wedge (e.g., "max structural divergence across the spectral shifts") is already achievable by the consumer by calling `geometric_product_wedge_into` per-shift and taking their own `max`. No new primitive is needed.

**No Super-GOAT criteria met:**
- Q1 (no prior art): ✗ — `(max,+)` over an existing accumulation is the trivial tropical lift.
- Q2 (new class?): ✗ — bottleneck-wedge is a different aggregation of the same anti-symmetric cross-terms; loses information rather than gains capability.
- Q3 (selling point?): ✗ — "Our NPCs find the worst spectral divergence channel" has no concrete use case.
- Q4 (force multiplier?): ✗ — no existing pillar needs bottleneck-wedge.

---

## 2. Why the Tropical Pattern Doesn't Generalize Indefinitely

The Research 321 §2.4 list implied that "tropical-ify every aggregation" would be a force multiplier. The honest verdict after evaluating all 4 candidates is that the Super-GOAT-tier tropical primitives already shipped (Plan 337 G1 passed 3/3 substrates), and the remaining candidates are all in one of three categories:

1. **Already shipped** (TropicalFunctor) — a codebase grep miss, not a real gap.
2. **Wrong substrate** (TropicalShardRetrieval, TropicalLatCal) — the substrate's contract is `linear + deterministic`, not `aggregation + bottleneck`. Tropical aggregation here is either redundant (cosine retrieval already does the right thing) or contract-breaking (LatCal homomorphism).
3. **Anti-symmetric / information-lossy** (TropicalGeometricProduct) — tropical max is asymmetric under sign, breaking properties the host primitive's consumers depend on.

**General rule (records the lesson):** Tropical `(max,+)` is the right tool when:
- The aggregation is *symmetric* and *monotone* (worst-case path, bottleneck flux, max-pair coherence).
- The host primitive's value is already a *bottleneck/worst-case aggregation* (DEC max-flux, path bottleneck, best-pair alignment).

It is the **wrong tool** when:
- The host primitive's value is *expected/averaged coherence* (cosine retrieval, mean coherence).
- The aggregation must preserve *sign information* (anti-symmetric wedge).
- The host primitive's contract is *linear-homomorphic* (LatCal fixed-point arithmetic).

---

## 3. References

- Parent: `katgpt-rs/.research/321_Tropical_Semiring_Equivariant_Operators.md`
- Shipped primitives: `katgpt-rs/crates/katgpt-core/src/algebra/tropical.rs`, `riir-ai/crates/riir-engine/src/latent_functor/arithmetic/mod.rs:316` (`tropical_extract_functor_into`).
- SE(2) sibling: `katgpt-rs/.research/457_SE2_Equivariant_Lift_Game_Maps.md` + Plan 560.
- LatCal source: `riir-chain/src/encoding/latcal.rs`.
- Item retrieval source: `riir-neuron-db/src/item_index.rs`.
- Geometric product source: `katgpt-rs/crates/katgpt-core/src/linalg/geometric_product.rs`.

---

## 4. Honest Caveat

The pre-existing §2.4 line *"TropicalFunctor — strong. New signal: max-pair displacement coherence vs mean-pair displacement coherence"* was wrong-by-omission: the primitive had already shipped under the `tropical_algebra` feature flag at the time §2.4 was written. Research 321 was correct to label the other 3 *speculative*; this note confirms the speculation.

**Lesson encoded:** the research skill's "codebase-vocabulary grep + read-the-hits" rule (#3 failure mode) is the defense. A paper-vocabulary grep (`tropical`) followed by codebase-vocabulary grep (`tropical_extract`) would have found `latent_functor/arithmetic/tropical_tests.rs` immediately. Future fusion-candidate evaluations must include both.
