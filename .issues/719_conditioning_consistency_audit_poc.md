# Issue 719: Conditioning-consistency audit PoC — per-junction KL + Pinsker TV gate for semantic context compression

**Status:** T1 LANDED (opt-in `cond_audit`, T2–T4 trigger-gated, no consumer, no GOAT claim) — `.benchmarks/700_cond_audit_poc.md`, commit `995dea6d`

**Source:** arXiv:2609.00865 "MemoryWalker" (Research 528, 2026-09-03). The modelless extract: at any serving site that conditions on a *semantically compressed* context (window, eviction, summarization, budget packing), a two-forward pair (compressed-conditioned vs full-context teacher) yields a per-junction forward-KL whose **unconditional** Pinsker bound `TV <= sqrt(eps_KL/2)` is a proven behavioral-gap verdict between context regimes. No instrument in the stack computes a KL/TV distance between conditioning regimes today (grepped 2026-09-03, Research 528 §4).

**Why not bit-identity:** every shipped *numeric* compression surface (f16 KV, q8kv, PoT scales) is already gated at bit-identity — stronger than any KL bound. The audit matters only where compression is semantic by construction, i.e. where bit-identity is impossible:
- Gemma-4 sliding-window ring (`kv_cache.rs` ctors "do-not-pass-yet"; designated consumer riir-train Plan 343 T1.6) — Issue 752 pins truncation-differ existence-only; the audit would quantify bounded-vs-unbounded across `sw`.
- `rt_turbo` window + sink decode (katgpt-speculative).
- `TokenBudgetPacker` (riir-rag) — budget levels rankable by measured audit KL instead of byte count.
- H2O eviction (Research 523, deferred) — if un-deferred, hit-rate alone is insufficient; this is the gate.

## Tasks

- [x] **T1 — audit core in katgpt-core** (opt-in feature `cond_audit`, zero new deps): `audited pair = (student forward, teacher no-grad forward)` → per-junction forward-KL, total `eps_KL`, verdict `sqrt(eps_KL/2)`, greedy-stream flip counter, calibrated-zero arm (compression-off control; quiet-box discipline per Bench 649). Deterministic ×3. — LANDED 2026-09-03: `src/cond_audit.rs` delegates the KL to `stale_residual::kl_logits` (substrate composition — the §4 no-KL-instrument claim refined: a stable categorical KL existed behind `stale_residual`; the seam + verdict chain + calibrated-zero discipline are the new half); report carries both the issue's `sqrt(eps_KL/2)` form AND the K-aware chain form `sqrt(K·eps_KL/2)` (Pinsker is pairwise; documented). Bit-identical ×3, and bit-identical across debug AND release profiles. Bench 700.
- [-] **T2 — first consumer: Gemma-4 sliding ring** — blocked on the ctors' designated consumer (Plan 343 T1.6) or an explicit un-pin; sweep `sw` and report the TV-gap curve. `[-]` until then.
- [-] **T3 — TokenBudgetPacker budget ranking** (riir-rag consumer, opt-in) — `[-]` until a packed-context consumer exists.
- [-] **T4 — H2O un-defer gate** — wire as Research 523's behavioral gate before any eviction PR lands. `[-]`.
- [x] **G8 non-vacuity (part of T1):** planted logit corruption must flip the verdict — an audit that cannot fail proves nothing (Bench 804 gate-9 lesson). — LANDED: 12-nat argmax deficit → eps_kl 49.33 nats, tv_bound 4.97 ≫ 0.05 threshold, flips 8/8, verdict FAIL, while the calibrated control arm measures exactly 0.0 in the same run (`g8_planted_corruption_exceeds_threshold_calibrated_stays_zero`).
- [x] G2 cost: audit overhead ≤ a measured budget of the paired forward (assert measured, never asserted-in-prose). — MEASURED: median audit/forward ratio **1.487** (15 interleaved reps × 25 calls, release-gated test) vs the 4.0 budget; fixture models the minimal real forward (16-dim vocab projection — a table-lookup draft measured 5.42× and was replaced as a fixture artifact: zero-compute forwards do not exist in serving). Real transformer forwards make the true fraction ≈ noise.

## Reopen triggers (any one un-defers T2–T4)

1. Any PR introducing semantic context eviction/windowing into a serving path.
2. The Gemma-4 sliding-ring consumer lands (Plan 343 T1.6) or the ctors un-pin.
3. L4 fixer lane revival with a train/serve context-regime divergence (Research 528 §5 recipe becomes a riir-train plan at the same time).
4. Research 523 H2O un-defers.

## Discipline

Opt-in, no default promotion, no GOAT claim until a consumer exists. Do not cite the paper's 26x drift figure for our surfaces — it is their harness, their densities; Issue 719 T1+T2 is what produces our number.
