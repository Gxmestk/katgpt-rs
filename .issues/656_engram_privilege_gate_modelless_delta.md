# Issue 656: Counterfactual privilege gating for engram fusion (modelless δ from LOPD)

> **Source:** [Research 419](../riir-train/.research/419_LOPD_Latent_Privileged_Context_OPSD.md) (riir-train) — arXiv:2608.13040 LOPD, modelless corollary §5.2 (upgraded 2026-08-16 after verification)
> **Repo:** katgpt-rs (public substrate — generic math, no game semantics)
> **Priority:** P2 · **Type:** optimization (gate extension) + PoC

## Problem

The engram kernel gate is **similarity-only**:

```
gate = σ(dot(q_norm, k_norm) / τ)        // kernel.rs
out[j] = gate * v[j]                      // residual-add into hidden state
```

It answers *"is this memory relevant to the query?"* — never *"does fusing this memory improve the consumer's prediction?"* The shipped test `fuse_zero_query_does_not_corrupt_hidden_state` documents the blind spot: with q=0, `dot=0` → `gate = sigmoid(0) = 0.5`, so every populated slot fuses at **half strength into the hidden state regardless of utility**. An anti-useful or drifted entry cannot be vetoed — only scaled by similarity.

LOPD's δ quantity is the missing signal, and it is **modelless-computable**:

```
δ = score(f(state + fuse(mem))) − score(f(state))    // counterfactual advantage
Δ = EMA( A_outcome · δ )                              // outcome-weighted, per slot
fuse_strength = base_gate · σ((Δ − m) / s)            // privilege gate, m = margin
```

Two evaluations + a comparison. Zero GD. LOPD's F2 finding is the empirical justification: **the same context helps in one regime and hurts in another** — so utility must be measured at use time, conditioned on the current query. Neither the similarity gate (query-only) nor riir-clippy's EvidenceTier (history-only, 3-tier) is query-conditional.

## Proposed design (scoping, not final)

1. **Per-slot privilege EMA** in the engram table: `Δ_slot ← (1−α)·Δ_slot + α·A·δ` updated whenever the host reports an outcome for a fused prediction. Fixed-size `Vec<f32>` alongside values — table layout change, versioned (freeze/thaw bump).
2. **Privilege gate on fusion:** multiply the kernel gate by `σ((Δ_slot − m)/s)`. Cold start: Δ₀ = 0 → gate ≈ σ(−m/s) < 0.5 — new slots fuse weakly and *earn* strength through verified outcomes (mirrors LOPD's cold-start + margin).
3. **Dual-β accumulator (optional host-facing):** `β ← [β + η(m − Δ)]₊` exposed as a table health metric — ramping pressure when a table's aggregate advantage decays. The generic "must keep earning ≥ m" runtime contract.
4. **Cost control (the honest constraint):** counterfactual δ needs the consumer scored twice at retrieval events. Gate-on-gate: only score the counterfactual when the similarity gate passes a floor, or amortize via the per-slot EMA (score sparsely, decay between). Must stay zero-alloc in the hot path.

## Tasks

- [x] T1 PoC (riir-poc or katgpt-core bench): planted-drift test — table with x% poisoned entries; measure consumer error with/without privilege gate; falsifiable target: gate recovers ≥ half the poisoned-entry penalty at ≤ 2× retrieval-event cost
- [x] T2 `PrivilegeGatedFuse` extension behind feature flag `engram_privilege` (opt-in) — kernel change is multiplicative, existing gate math untouched
- [x] T3 GOAT gate: G1 relevance-ranking preserved on clean tables (gate ≈ no-op when Δ high), G2 overhead budget (amortized ≤ +20% at retrieval events), G3 existing engram tests pass both feature states, G4 zero-alloc hot path (counterfactual scoring is amortized, not per-fuse)
- [x] T4 Cross-ref: EvidenceTier (riir-clippy Issues 018/021) as the discrete predecessor; note the query-conditional upgrade path for LatentFixMemory retrieval weighting as a NON-goal here (separate repo, no demonstrated binding)

## Verdict rule

T1 falsifies or defends. If planted-drift shows no recoverable penalty (the similarity gate already suffices at realistic poisoning rates), close as negative result — the similarity gate is enough, privilege gating is training-world machinery only.

## Verdict (2026-08-16) — **DEFENDED, opt-in, scope-limited**

Full results: [`.benchmarks/656_engram_privilege_goat.md`](../.benchmarks/656_engram_privilege_goat.md).
Shipped: `crates/katgpt-core/src/engram/privilege.rs` + `sigmoid_fuse_scaled_into` (cfg-gated
sibling in `kernel.rs`), feature `engram_privilege` (opt-in).

**T1 defends the mechanism.** On sign-opposed drift — poison with *identical* cosine to the
query as the good patterns but the opposite utility projection, where the similarity gate is
provably blind — recovery is **100.0%** (rel_err 0.3332 → 0.0001, purity 0.500 → 1.000) at
**1.149×** retrieval-event cost. Bar was ≥50% at ≤2×.

**GOAT: G1–G4 all PASS.** G1 clean-table rel_err 0.0 (exact) plus two bit-level anchors;
G2 amortized **1.180×** ≤ 1.20× at update period 64 while holding 100% recovery; G3 142 (off)
/ 167 (on) engram tests pass, build matrix clean; G4 0 allocs on fuse / update / read / trace.

**Four measured scope limits — read these before consuming:**

1. **Control B falsifies the general case.** Where poison is similarity-*separable*, the
   shipped gate already handles it: `err_naive` 0.0036 vs. regime A's 0.3332 — **93× smaller**.
   Privilege gating polishes a negligible penalty for ~18% cost. This primitive earns its keep
   *only* where relevance and utility genuinely diverge.
2. **The design is NOT query-conditional in its state.** δ is measured query-conditionally, but
   `Δ_slot` is one scalar averaged over every query that touched the slot — structurally the
   same as `EvidenceTier`, just continuous. On the real LOPD F2 shape (regime D: an entry that
   helps query class 0 and hurts class 1) the EMA **oscillates rather than converges**:
   `recency_latch = 0.6622` — one extra training round swings `p_poison` by 0.66. The 83%
   headline recovery there is an artifact of where training stopped. True query-conditional
   gating needs the query in the ledger key — a different primitive.
3. **The cheap aggregate-δ path fails exactly where the gate is needed.** `CreditAssignment`
   splits one scalar δ by *unsigned* weights, so on sign-opposed slots (aggregate δ ≈ 0)
   recovery is **−0.0% vs. 100% exact**. Shipped and documented as same-sign-only, not removed
   — it is the first thing a cost-sensitive host reaches for.
4. **"7 updates suffice" is noise-free.** The cadence sweep is flat to 7 updates only because
   the fixture's δ is exact; at 8× outcome noise the same cadence reads 51.6% and sparser/denser
   cadences go negative. Budget updates against real outcome-label quality.

**Not promoted to default** despite passing, and the gain being modelless (two evaluations and
a comparison; ledger is runtime latent state, no gradients): the win is regime-specific (1),
`margin`/`scale` are in host score units with no sensible default, and (2) is a real
instability to inflict on hosts who never opted in. **Promote when** a consumer demonstrates
the regime-A shape on real data *and* can supply outcome labels at ≥1× noise quality.

**Design deviation from §1 above:** `Δ_slot` ships as a **side-car `PrivilegeLedger`**, not a
table field. `InMemoryEngramTable` is frozen + BLAKE3-committed; mutable per-slot state would
either poison the commitment or be excluded from it. The side-car needs no trait change, no
freeze/thaw version bump, and composes with every `EngramTable` impl. The §1 "table layout
change, versioned" scoping is superseded.

## References

- [Research 419 §5.2](../riir-train/.research/419_LOPD_Latent_Privileged_Context_OPSD.md) · [Plan 340](../riir-train/.plans/340_lopd_latent_privileged_context_sdar_fusion.md) (training-track sibling)
- `crates/katgpt-core/src/engram/kernel.rs` (similarity gate) · `forward.rs::fuse_into_hidden_state` (zero-query 0.5-strength fusion)
- LOPD F2 (context utility reverses across regimes) + F3 (margin necessity) — the empirical basis
