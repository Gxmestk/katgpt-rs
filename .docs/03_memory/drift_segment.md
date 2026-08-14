# DriftSegmentStore: Training-Free Drift-Segmented Multi-State Memory

> **Status:** Opt-in (`drift_segment`, root forwards to `katgpt-kv/drift_segment`) — Issue 652 / Research 482, modelless adaptation of "Dynamic Linear Attention" ([arXiv:2606.10650](https://arxiv.org/abs/2606.10650)). **Bench 635 GOAT PASS** (2026-08-15, all four gates). Consumers landed (see below); promotion candidate at the next re-gate.

## The selling point

**Adaptive-resolution memory at fixed budget.** A per-token drift score opens a
new memory state at semantic transitions; a capacity-K cache merges the
adjacent lowest-density pair when full. Surprising spans (needles, regime
heads) get their own slot and survive; boring neighbors fuse first. At matched
budget this beats fixed tiling + LFU eviction by **+46pp needle recall** on
change-point streams — with zero training, 12 ns/token, and zero steady-state
allocations.

The paper proves the *co-trained* variant (soft gating during pre-training,
50B tokens, 4×A100). The published open lane (Research 482 §2.5, Area 7) is
the **training-free post-hoc** variant — this primitive is that measurement,
and it validates the *policy* (not the training) as the load-bearing part.

## Composition (three shipped primitives — nothing new is authored here)

| DLA component | Substrate consumed | Origin |
|---|---|---|
| `I_t` drift score | `TemporalDerivativeKernel` dual-EMA on the key stream | Plan 277 / Research 435 |
| capacity-K state cache | `SegmentStore`-shaped bounded slots | Plan 223b / Research 199 |
| adjacent pair-merge | `hope_compactor`'s pair-merge pattern (online + adjacency-restricted) | riir-neuron-db Plan 321 |
| query-gated readout | GRM sigmoid gating (`segment_checkpoint::gating`) | Plan 223b |

## Deltas over the paper (Research 482 §2.3)

1. **Score in key space, not state space.** The paper's `I_t` is the relative
   Frobenius drift of the d×d state (cost ≈ the update itself). We score the
   relative drift of the dual-EMA key estimate — O(d) per token, same signal
   class (regime change ⇒ key-distribution shift).
2. **Rising-edge boundary.** `surprise_norm()` stays elevated for the slow
   EMA's timescale after a transition; firing on the rising edge (score ≥ τ
   while armed, disarm until below τ) recovers single-fire-per-transition
   semantics.
3. **Modelless readout.** Learned per-slot λ → sigmoid-gated signed dot
   (`w_i = σ(β·(q·k̄ᵢ))·(q·k̄ᵢ)`) — the gated-linear-attention shape,
   sigmoid never softmax.

## Density accounting

Each slot carries `info_sum` (Σ per-token drift score) and `n_tokens`; slot
density = `info_sum / n_tokens`. At capacity, the **adjacent** pair with the
lowest pair density `(info_i + info_j)/(n_i + n_j)` is merged in place
(additive linear-attention states compose cheaply — unlike softmax
attention). Chronological order is preserved by construction.

**`reset_slots` (warmup-discard, 2026-08-15):** discards accumulated segments
+ boundary/merge counters while keeping the converged drift kernel and
monotonic token position. The kernel's slow EMA converges over ~2/α_slow
tokens with relative scores elevated by construction — those early scores
land in slot 0's `info_sum` and make the startup transient look like a
high-density needle. Consumers using density as a "moment that mattered"
signal (recognition/salience feeds) discard the warmup prefix with this
method.

## GOAT gate (Bench 635 — M3 Max, CPU-only, 16 seeds, matched budget)

| Gate | Target | Result |
|---|---|---|
| G1 change-point recall | (c)−(b) ≥ +10pp | **+46.09pp** (0.578 vs 0.117) PASS |
| G1 stationary recall | (c)−(b) ≥ −2pp | **+75.00pp** (0.930 vs 0.180) PASS |
| G2 latency | O(d+K)/token, reported | **12 ns/token** (d=32, K=30); 307 ns/query PASS |
| G3 no-regression | segment_checkpoint + all-features unchanged | 38 + 253 lib tests PASS |
| G4 alloc-free | 0 steady-state allocs | 0 across 1000 tokens PASS |

Reference arms: (a) single-state accumulator 0.172/0.133 (chance-level
identification — needle diluted to ~32/8000); (b) fixed 128-token tiles +
LFU 0.117/0.180 (tiles straddle change points, write-only LFU degenerates to
content-blind FIFO). Not UQ-bearing (recall metric, no
probability/interval claim) — conformal floor N/A per the
Report-the-Floor rule.

## Honest caveats

1. **Synthetic-only.** Same limitation class as segment_checkpoint's blocked
   NIAH (needs real model KV). Validated claim: the **policy** comparison at
   matched budget on controlled streams — not end-to-end LM quality.
2. **Hysteresis trap.** A regime whose noise floor exceeds τ permanently
   disarms the rising-edge detector. τ must sit above the haystack noise
   floor — a deployment calibration constraint (documented in
   `drift_segment/mod.rs` tests); the downstream `recommend_tau` helper
   probes it.
3. **G2 ratio artifact.** The 0.69× vs single-state ratio reflects the
   single-state arm's unoptimized scalar accumulate; the meaningful numbers
   are 12 ns/token absolute and 307 ns/query.
4. **Multi-token needles (span 32) are paper-faithful**; single-token
   needles are an adversarial case the paper does not claim (pair-density is
   n-weighted).

## Promotion state + consumers

Stays opt-in per Issue 652 T5: the bench is synthetic-only. Both follow-up
consumers filed at GOAT time have since **landed**, so promotion is a
candidate at the next re-gate:

| Consumer | Repo / record | Shape |
|---|---|---|
| Per-NPC episodic belief memory (F2) | riir-ai Issue 677 → Bench 675 | Type-level: `NpcEpisodicState<K,D>` wraps `DriftSegmentStore` (feature `npc_episodic_memory`, DEFAULT-OFF — promotion there is a product decision) |
| Salience-gate warmup discard | riir-ai Issue 680 → Bench 677 | `reset_slots` via `with_warmup_ticks` — discards the startup transient before the civ salience gate reads density |
| Consolidation density-aware merge (F3) | riir-neuron-db Issue 595 → Bench 480 | Policy transfer (no type dep): `capture_wake_scored` adjacent-density merge in the wake buffer |
| Scored-wake runtime collector | riir-ai Issue 681 (in flight) | Feeds temporal-deriv info scores into the F3 capture path from the ARG offline loop |

## Code locations

| File | Content |
|---|---|
| `crates/katgpt-kv/src/drift_segment/mod.rs` | `DriftSegmentStore<K,D>`, `DriftSlot`, `sigmoid_gated_readout`, `reset_slots`, density/merge machinery + tests |
| `benches/bench_635_drift_segment_goat.rs` (root crate) | The three-arm GOAT bench (feature-gated) |
| `.benchmarks/635_drift_segment_goat.md` | Full gate record + arm design + caveats |

## Run

```bash
cargo bench --bench bench_635_drift_segment_goat --features drift_segment
```

## See also

- [Research 482 — DLA drift-segmented memory distillation](../../.research/482_Dynamic_Linear_Attention_Drift_Segmented_Memory.md)
- [`product_key_memory.md`](product_key_memory.md) — the O(√N) factored-retrieval sibling
- [`sleep_consolidation.md`](sleep_consolidation.md) — eviction-time consolidation (the offline cousin)
- riir-ai `crates/riir-engine/src/npc_episodic.rs` — the per-NPC think-brain consumer
