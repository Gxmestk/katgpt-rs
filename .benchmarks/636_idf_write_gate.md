# Bench 636 — TF-IDF Write-Gate GOAT (Issue 650 / Research 481)

**Date:** 2026-08-15
**Machine:** M3 Max (CPU-only bench; no GPU used — exclusivity check N/A)
**Source paper:** "Continual Learning via Sparse Memory Finetuning" ([arXiv:2510.15103](https://arxiv.org/abs/2510.15103), Meta FAIR + Berkeley, Oct 2025)
**Feature:** `product_key_memory_episodic` (katgpt-core) — **opt-in, stays opt-in** (T5: documented as the recommended write path; module-level promotion is a separate scheduled re-gate)
**Run:** `CARGO_TARGET_DIR=/tmp/pkm650 cargo bench -p katgpt-core --features product_key_memory_episodic --bench bench_481_idf_write_gate`

> Bench-numbering note: the issue text named this artifact `.benchmarks/481_idf_write_gate.md`,
> but `.benchmarks/` numbers are their own monotonic sequence (highwater was 635) — this is 636.
> The bench **target** keeps the research-numbered name `bench_481_idf_write_gate.rs`.

## Verdict: GOAT PASS (G1 + G2 + G3 + G4)

```
G1 interference/retention (write fact set A, then B; recall(A) after B;
   matched candidate pool 16, matched write width t=4; gate swept per arm so
   final recall(B) matches; target IDF − TF ≥ +10pp absolute)

  norm-ramp regime (trained-analog — hot-slot pathology present):
    IDF @ gate=0.60: recall(B)=0.852  recall(A)_post=0.719
    TF  @ gate=0.90: recall(B)=0.852  recall(A)_post=0.594
    → margin +12.5pp at recall(B) gap 0.000 — PASS

  organic regime (untrained from_random table — no-harm control):
    IDF @ gate=0.60: recall(B)=0.852  recall(A)_post=0.734
    TF  @ gate=0.60: recall(B)=0.820  recall(A)_post=0.719
    → margin +1.6pp — IDF does not damage recall when the pathology
      it fixes is absent (safety property; informational)

G2 latency (release, 1000 writes):
  write_idf (pool=16, t=4):  1278 ns/write
  write     (k=4 plain):      656 ns/write
  idf-fold overhead:          ~620 ns (1.9× plain; O(k) multiplies +
                              one O(t·k) selection sort) — PASS (µs-scale)

G3 no-regression:
  bench_408_pkm_episodic_fusion output bit-identical to HEAD@adcb92e6
    (worktree run: MSE=0.205308 ratio=0.8005, same k=1-arm PASS verdict —
     the k=4-arm FAILs are pre-existing at HEAD, not caused by this change)
  bench_408_pkm_goat: ✅ PROMOTE verdict unchanged
  1922 lib tests pass with --features product_key_memory_episodic
  clippy clean (lib + benches + tests)

G4 alloc-free: 0 allocations across 1000 steady-state write_idf calls
  (stats table fixed-size [u32; N]; selection reuses dead PkmScratch::scores_1)
```

## What shipped (T1/T2)

- **T1** `BackgroundAccessStats<const N: usize>` — static background access-count
  table (`new` / `record_batch` / `build_background_stats` / `idf` /
  `n_batches` / `slot_batch_count`). `idf(i) = ln((|B|+1)/(1+count[i]))` — the
  same smoothed log-ratio riir-neuron-db `Bm25Index` applies to terms, applied
  to slots. Zero-alloc build (sort + compaction dedup on a caller buffer).
- **T2** `PkmEpisodicStore::write_idf` + `write_weighted_idf` — retrieve
  top-`k` pool, re-rank by `weight × idf(slot)`, apply the **unchanged**
  δ-rule to the top-`t`. Plain `write`/`write_weighted` untouched (baseline
  arms). `write_selected` — the escape hatch for custom selection policies
  (also the bench's random-t control). Selection score is a ranking statistic,
  NOT a probability — not UQ-bearing, conformal floor N/A (per the Report-the-
  Floor rule; a slot-selection ranking has no coverage/interval semantics).

## Why two regimes (the honest design)

The paper's Fig 6 measures forgetting on a PKM **trained by pretraining** —
which develops activation concentration on generally-hot slots. Our
`from_random` table (untrained) has only chi-fluctuation norm spread: the
measured histogram shows no slot retrieved by >62% of background batches
(idf floor 0.501, never 0). The bench therefore measures both:

1. **NormRamp** — key rows scaled by a linear norm ramp (1.0 → 1.8). This
   models the trained-table precondition via the substrate's *documented*
   `ScoreFn::Dot` magnitude-sensitivity (the exact pathology `ScoreFn::Idw`
   exists to avoid on the read side). Background counts concentrate (idf
   range [0.278, 3.497]); TF-only writes pile onto hot rows; IDF downweights
   them. **This is the gated regime** — the paper's claim, re-measured in our
   substrate.
2. **Organic** — plain `from_random`. The no-harm control: IDF ≈ TF (+1.6pp)
   when the pathology is absent.

## Honest findings (the defend-wrong record)

1. **The mechanism is real in our substrate**: +12.5pp retention at *exactly*
   matched learning (recall(B) 0.852 vs 0.852, gap 0.000). TF needs gate=0.90
   to reach the learning that IDF reaches at gate=0.60 — and still forgets
   more. This exceeds the paper's own ~11pp (their t=500, 1.3B LLM) —
   consistent with their "gap widens as the write set shrinks" (our t=4 is
   the smallest write set in the stack).
2. **The random-t control OUTPERFORMS IDF on retention** (ramp: 0.789 @
   gate=0.60 vs IDF 0.719). Uniform-random-within-pool is maximal-spread
   selection; it beats relevance-aware spread at this scale because the read
   pool and the write pool have the same width (16) — every random pick is
   still readable back. IDF is the best policy that *respects activation
   ranking* (relevant slots only); pure spread discards ranking. Scope note
   for consumers: if your read width equals your write pool, consider
   random-within-pool; if writes may land outside the read neighborhood
   (write pool > read pool, the paper's t=50-500 regime), relevance-aware
   selection is the safe policy. Not gated (the issue's PASS criterion is
   IDF vs TF-only).
3. **G2 overhead is 1.9× plain write** (~620ns) — dominated by 16 `ln`
   evaluations (one per pool candidate). Fine at episodic cadence (13µs/s at
   20Hz × 1 write/tick); a precomputed idf LUT would cut it if a consumer
   ever needs it hot (deferred — no such consumer today).
4. **Pre-existing bench debt surfaced by G3**: `bench_408_pkm_episodic_fusion`
   k=4 arms FAIL at HEAD (both before and after this change, bit-identical
   outputs — verified in a clean worktree at `adcb92e6`). The bench's overall
   G4 verdict still PASSes via the k=1 arm. Not caused by, and not in scope
   of, this issue.

## Tasks

- [x] T1 — `BackgroundAccessStats` (6 unit tests)
- [x] T2 — `write_idf` + `write_weighted_idf` + `write_selected` (6 unit tests;
      23 episodic tests total pass)
- [x] T3 — GOAT bench `bench_481_idf_write_gate.rs` (G1 +12.5pp / G2 1.9× /
      G4 0 allocs)
- [x] T4 — G3 no-regression (bit-identical fusion bench, pkm_goat PROMOTE,
      1922 lib tests, clippy clean)
- [x] T5 — G1–G4 PASS → `write_idf` documented as the recommended write path
      in the module docs (paper + PoC evidence + the random-control scope
      note). `product_key_memory_episodic` stays opt-in; module-level default
      promotion is deferred to its next scheduled re-gate (the G4-fusion-gate
      open question from Plan 408 Phase 5 still stands, and the k=4-arm
      pre-existing FAILs above should be looked at in that re-gate).

## Out of scope (follow-ups, Research 481 §2.5)

- F3: consolidation-level batch aggregation — riir-neuron-db
- F2: hierarchical branch(RIZZ)×slot(IDF) — riir-ai
- F4: background-relative SPEFT salience — riir-train
