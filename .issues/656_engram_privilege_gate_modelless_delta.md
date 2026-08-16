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

- [ ] T1 PoC (riir-poc or katgpt-core bench): planted-drift test — table with x% poisoned entries; measure consumer error with/without privilege gate; falsifiable target: gate recovers ≥ half the poisoned-entry penalty at ≤ 2× retrieval-event cost
- [ ] T2 `PrivilegeGatedFuse` extension behind feature flag `engram_privilege` (opt-in) — kernel change is multiplicative, existing gate math untouched
- [ ] T3 GOAT gate: G1 relevance-ranking preserved on clean tables (gate ≈ no-op when Δ high), G2 overhead budget (amortized ≤ +20% at retrieval events), G3 existing engram tests pass both feature states, G4 zero-alloc hot path (counterfactual scoring is amortized, not per-fuse)
- [ ] T4 Cross-ref: EvidenceTier (riir-clippy Issues 018/021) as the discrete predecessor; note the query-conditional upgrade path for LatentFixMemory retrieval weighting as a NON-goal here (separate repo, no demonstrated binding)

## Verdict rule

T1 falsifies or defends. If planted-drift shows no recoverable penalty (the similarity gate already suffices at realistic poisoning rates), close as negative result — the similarity gate is enough, privilege gating is training-world machinery only.

## References

- [Research 419 §5.2](../riir-train/.research/419_LOPD_Latent_Privileged_Context_OPSD.md) · [Plan 340](../riir-train/.plans/340_lopd_latent_privileged_context_sdar_fusion.md) (training-track sibling)
- `crates/katgpt-core/src/engram/kernel.rs` (similarity gate) · `forward.rs::fuse_into_hidden_state` (zero-query 0.5-strength fusion)
- LOPD F2 (context utility reverses across regimes) + F3 (margin necessity) — the empirical basis
