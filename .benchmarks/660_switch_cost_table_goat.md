# Bench 660 — `SwitchCostTable` modelless primitive GOAT (Issue 663)

**Feature:** `switch_cost` (opt-in — NOT promoted; promotion requires a
riir-ai consumer A/B, see verdict)
**Source:** Research 484 — Skill Entropy (arXiv:2608.05139, He et al. 2026-08-06)
**Date:** 2026-08-17
**Machine:** M3 Max (Apple Silicon), release-mode timing best-of-3 × 1M;
counting-allocator G4 in an isolated test binary.
**GPU exclusivity:** N/A (pure CPU primitive; a sibling riir-train probe was
running at ~1.2 cores during the run — no GPU contention, timing headroom
~3× on every gate).

## What shipped

| Piece | Notes |
|---|---|
| `SwitchCostTable<const N>` | `#[repr(C)]` u32 counters + α; `Copy` + manual `Pod`/`Zeroable`. Directed `ske(a,b)`, `sequence_entropy` (Eq. 4, mean over consecutive pairs), `record_solo`/`record_switch`, `ske_if_armed` warm-up gate. |
| `SwitchCostSnapshot<const N>` | `#[repr(transparent)]` read-only freeze/thaw view — no `record_*` methods; bitwise-committable. |
| `FactorizedSwitchCost<const N, const F>` | Eq. 7 factorization: `ske(a,b) ≈ ske(a, fam_b)·ske(fam_a, b)`; O(N·F) counters; `record_switch` routes into leave cell `(a, fam_b)` + land cell `(fam_a, b)` in one call. |
| `cdf_rank(value, sample)` | Empirical-CDF rank (paper §4) — scale-free normalization for the riir-train Gap-6 reward shape. |
| Formula | `SkE(a,b) = (½·Acc(a) + ½·Acc(b) + α) / (Acc(a,b) + α)`, α = 0.1. Zero-trial accuracies read `0.5` (neutral prior) ⇒ a cold table is **exactly 1.0** everywhere (0.6/0.6) — evidence-free ⇒ no assumed difficulty. |

## Gate results

| Gate | Target | Result |
|---|---|---|
| G1 formula | hand-computed 3.0 / 0.667 fixture | **PASS** — `ske(0,1)=3.0`, `ske(1,0)=0.6667` (tol 1e-6) |
| G1 directionality | `ske(a,b) ≠ ske(b,a)` constructible + pinned | **PASS** — 3.0 vs 0.667; `>` gap asserted >1.0 |
| G1 determinism | same counters → bit-identical; **record order independent** | **PASS** — forward vs reverse replay: `to_bits()` equal on all pairs + sequence entropies (u32 counters commute exactly) |
| G1 cold semantics | cold = exactly 1.0 | **PASS** — all pairs, empty/1-elem/2-elem sequences |
| G1 monotone warm-up | fixed solos, accumulating pair failures ⇒ ske non-decreasing | **PASS** — 20-step ladder, monotone, ends >1.0 |
| G1-A/B factorized | paper's own fidelity bar (82–86% overlap) → Spearman ≥ 0.75 + identical argmax | **PASS** — Spearman over all 36 ordered pairs on a multiplicative ground truth (600 trials/pair); argmax pair identical (mode-1→fam-1 leave × mode-4 land); cross-seed Spearman ≥ 0.75 |
| G2 lookup | single-digit ns | **PASS** — **3.08 ns/op** `ske` (N=8, best of 3 × 1M, black_box) |
| G2 sequence | F1 per-tick budget headroom | **PASS** — **34.04 ns** `sequence_entropy` over a 16-mode seq (target <300 ns) |
| G3 no-regression | default build untouched; clippy 0 | **PASS** — `cargo check -p katgpt-core` default clean; `--no-default-features --features switch_cost` clean; `cargo clippy --features switch_cost --all-targets` 0 warnings |
| G4 alloc-free | 0 steady-state | **PASS** — 0 allocations across 1M `ske` (exact + factorized) + 100k `sequence_entropy` ×2 + 100k `record_*` + snapshot + `cdf_rank` (CountingAllocator, isolated binary, single serial test) |
| Consumer shape | `cdf_rank` reward scale-free | **PASS** — `r_ent` identical under a 10× corpus rescale (tol 1e-6) |

## T7 demo (the F1 trigger preview)

`examples/switch_cost_demo.rs` — 5-mode FSM (Idle/Hunt/Flee/Tame/Sleep) with
a **designed hard Flee→Hunt switch** (carry-over, the Issue-054 border-piling
shape), 400 trials/pair:

- The directed table reads **SkE(Flee→Hunt) = 3.30** vs ~1.0–1.4 everywhere
  else — the designed structure is recovered from telemetry alone.
- "Hardest incoming switch" per mode finds `Hunt: hardest from Flee` —
  exactly the proactive trigger F1 keys on.
- Sequence entropy separates a calm routine (1.16) from a panic routine
  (1.53) — the quest-difficulty dial (Eq. 4).
- Warm-up gate (`ske_if_armed`, 50-trial floor) armed on the measured pair.

## Honest findings

1. **The factorized variant under-estimates hard pairs on heterogeneous
   families** — demo: factorized 1.99 vs exact 3.30 on Flee→Hunt. The leave
   cell `(Flee, fam_execution)` aggregates Flee→{Idle, Flee, Hunt} and the
   two easy members dilute the one hard member. The paper's 82–86% fidelity
   is a *ranking* fidelity, and the ranking holds (G1-A/B), but magnitudes
   from the factorized form should be treated as diluted when a family mixes
   easy and hard members. **Guidance: use the exact table for bounded mode
   sets (N ≤ ~64); reach for the factorized form only when N² measurement is
   genuinely impractical.**
2. **Cold-start neutrality is a design choice, not the paper's text.** The
   paper always measures under a warm reference model; our zero-trial `0.5`
   prior (⇒ cold SkE exactly 1.0) is the minimal-assumption analog and keeps
   un-armed cells from reading as infinitely hard. `ske_if_armed` exists
   because 1.0 ≠ "measured easy" (Research 484 §6.1).

## Verdict

**Stays opt-in (`switch_cost`), as the issue pre-registered.** The primitive
is modelless (u32 counters + f32 arithmetic), fast (3 ns), alloc-free, and
freeze/thaw-able — but the GOAT promotion rule requires a measured consumer:
the falsifiable A/B is riir-ai **F1** (SkE-gated preemptive re-estimation vs
the coherence-only arm on the Issue-054 stuck-rate scenario). The
healer-consumer branch was already measured dead
(riir-clippy Bench 032: fix-ordering refuted by mechanism — the gap was
retrieval pool crowding, not fix interference); remaining consumers are
riir-ai F1/F3 and riir-train Plan 319 Gap 6.

## Files

- `crates/katgpt-core/src/switch_cost.rs` — module + 11 unit tests
- `crates/katgpt-core/tests/switch_cost_663_poc.rs` — 6 tests (2 release-only G2)
- `crates/katgpt-core/tests/switch_cost_alloc_check.rs` — G4
- `crates/katgpt-core/examples/switch_cost_demo.rs` — T7
- `crates/katgpt-core/src/lib.rs` + `Cargo.toml` — registration + feature
