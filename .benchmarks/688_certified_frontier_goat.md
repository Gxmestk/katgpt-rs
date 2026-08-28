# Bench 688: Certified Frontier — Phase 3 GOAT gate

**Status:** RECORD — G1/G2/G3/G4 **PASS**; T3.4 floor gate **SPLIT** (PASS on calibration, FAIL on the plan's stated product metric, which is measured here to be degenerate). Decision: **stays opt-in**, promotion deferred to a re-gate on a corrected metric. 2026-08-28.

**Plan:** [580](../.plans/580_certified_frontier_primitive.md) Phase 3 (T3.1–T3.5)
**Phase 0:** [Bench 687](687_certified_frontier_phase0_poc.md) — 0 violations, monotone, 51.4× frontier-vs-passive
**Code:** `crates/katgpt-core/src/certified_frontier.rs`, feature `certified_frontier` (opt-in)
**Gates:** `tests/certified_frontier_correctness.rs` (22), `tests/bench_688_certified_frontier_goat.rs` (3), `tests/bench_688_certified_frontier_alloc_check.rs` (1)

Repro:

```sh
cargo test --release -p katgpt-core --features certified_frontier \
  --test certified_frontier_correctness
cargo test --release -p katgpt-core --features certified_frontier \
  --test bench_688_certified_frontier_goat -- --nocapture --test-threads=1
cargo test --release -p katgpt-core --features certified_frontier \
  --test bench_688_certified_frontier_alloc_check
```

Box: M3 Max, `--release`, rustc 1.97.0.

---

## G1 correctness — PASS (22/22)

Detailed in the Phase 2 commit. Headlines: incremental Cholesky agrees with an
independent dense f64 Gaussian-elimination solve to **max abs 1.161e-6 / max
rel 7.252e-6** at N=64, D=8; **zero** unsound certifications across 1000
adversarial random-order seeds and under the deployed frontier policy;
certified set and every per-cell `cb` monotone under label-corrupted sequences;
halting law fires at 185 / 728 / 2961 observations for ε = 0.2 / 0.1 / 0.05
(the predicted 1/ε² scaling, ratios 3.9× and 4.1×).

## G2 perf — PASS (0.264 µs/query vs a 1 µs budget)

Pool 1024 cells, D=8, 875 certified at warmup. The deployed shape: 1000 NPCs
each acquire + observe, then ONE expansion pass folds the batch in, amortised
over the batch.

| Operator | Time |
|---|---|
| acquire + observe + amortised expand | **0.264 µs/query** |
| `posterior_variance_linear`, N=256, D=8 | 16.3 µs/call |
| `reachability_dilation`, 1 hop, 1024 cells | 222 µs/pass |

**It did not start there.** The first measurement was **3.428 µs/query — a
FAIL** — and the two fixes are the interesting part of this gate:

| Revision | µs/query | What changed |
|---|---:|---|
| initial | 3.428 | acquisition rescanned `cells` for a certified neighbour per query, `O(cells·certified)` |
| + cached candidacy + Beta sd | 1.080 | `near_certified` stamped once per certification; sd recomputed on `observe`, not per scan |
| + SoA lane, branch-free argmax | **0.264** | one contiguous `f32` lane; 8-wide max reduction then `position` |

**13.2× total.** The last step is the one worth generalising: `acquire` reads
four fields out of a ~56-byte cell, so scanning the AoS array streams ~57 KiB
through L1 to use ~5 KiB of it, and tracking the argmax inline adds a
loop-carried dependency on an unpredictable branch. Splitting the hot fields
into `acq_sigma: [f32; MAX_CELLS]` and replacing the scalar argmax with a
branch-free 8-wide max reduction plus a short-circuiting `position` gave 4.1×
on its own.

The lane is maintained incrementally at every mutation point, which is exactly
the kind of derived state that rots silently — a missed refresh biases the
query sequence without failing anything. `acquisition_lane_matches_a_full_rescan`
pins it against a reference argmax over the public cell view at **every step**
of a 4000-round run, plus after a radius change and `rebuild_neighborhoods`.

`reachability_dilation` at 222 µs is `O(certified · uncertified · D)` and is a
periodic op, not a per-query one; it is off the hot path by construction.

## G3 no-regression — PASS

Feature is opt-in and absent from both `default` lists. `cargo check -p
katgpt-core` (default features) clean; `cargo metadata` parses. `clippy -D
warnings` clean on the module and all three gate binaries. (The two pre-existing
`prover_selection.rs` lint errors under `--tests` are untouched and predate this
work.)

## G4 alloc-free — PASS (0 allocs / 0 deallocs over 1000 cycles)

Full operator set per cycle: acquire, observe, `expand_certified`, `lcb`/`ucb`/
`sigma`, `query_is_decision_relevant`, `dilation_feasibility`,
`reachability_dilation`, `append_observation`, `posterior_variance_linear`,
`ridge_mean`, `refresh_kernel_sigma`, every closed form, and both scoreboards.

**The instrument was revert-probed**, not trusted: inserting a single
`vec![1u8; 8]` into the loop produced `1000 allocs / 1000 deallocs`, and
removing it returned to 0. A green alloc gate that cannot go red is not a gate.

## T3.4 Report-the-Floor — SPLIT VERDICT

The primitive claims a coverage guarantee, so it is UQ-bearing and owes a floor
comparison. Floor = **adjacency-only expansion**: certify any 4-neighbour of a
cell whose tally leans valid; no posterior, no β, no Lipschitz. Both arms
consume the **identical query sequence** per seed, so they differ only in the
certification rule. World 48×48 = 2304 cells, 888 truly valid at h=0.6, 200 000
queries/seed, δ=0.05, 5 seeds, paired per-seed ratios.

| seed | prim cert | viol | rate | product | floor cert | viol | rate | product | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 298 | 0 | 0.0000 | 298.0 | 1296 | 408 | 0.3148 | 888.0 | 0.336 |
| 1 | 309 | 0 | 0.0000 | 309.0 | 1290 | 402 | 0.3116 | 888.0 | 0.348 |
| 2 | 322 | 0 | 0.0000 | 322.0 | 1281 | 393 | 0.3068 | 888.0 | 0.363 |
| 3 | 303 | 0 | 0.0000 | 303.0 | 1284 | 396 | 0.3084 | 888.0 | 0.341 |
| 4 | 312 | 0 | 0.0000 | 312.0 | 1245 | 357 | 0.2867 | 888.0 | 0.351 |

Mean paired product ratio **0.348**, primitive wins **0/5**.
Pooled violation rate: primitive **0.00000**, floor **0.30582** — the floor is
**6.1× over its own δ**.

### The stated product metric is degenerate

`growth · (1 − violation_rate)` expands algebraically to
`certified − violations` — the **true-positive count**. It carries *no penalty
for a false positive* beyond not counting it, so it is maximised by certifying
everything. The data shows this directly: the floor scores **exactly 888.0 on
every seed**, which is exactly `n_valid`. It "wins" by certifying 55% of the
entire grid and being wrong about a third of it.

That is not a metric a safety primitive can be gated on. The one axis that
matches the claim is calibration, and there the result is unambiguous: the
primitive holds δ with room to spare, the floor breaches it by 6×. **The
primitive is the only deployable arm, and it loses the stated gate.**

### Where δ actually binds (T3.4b)

The floor comparison raises a fair question — is 300 cells a *loose bound* or a
*budget limit*? Sweeping the confidence width (scale < 1 is unsound; this is a
diagnostic, not a shipping mode):

| β scale | certified | violations | rate | calibrated? |
|---:|---:|---:|---:|---|
| 1.00 | 308 | 0 | 0.00000 | yes |
| 0.75 | 407 | 0 | 0.00000 | yes |
| 0.50 | 534 | 0 | 0.00000 | yes |
| 0.25 | 725 | 85 | 0.02344 | yes |
| 0.10 | 1001 | 818 | 0.16331 | **NO** |

**The shipped schedule spends 0.000 of a 0.05 budget while certifying 35% of
the valid region.** A 4× narrower width still holds δ. So the deficit is a loose
bound, not the query budget — the paper's Eq 31/37 is derived for a *kernel
logistic* model where information pools through the RKHS norm, and this module's
default posterior is a per-cell Beta-Bernoulli with no pooling at all. The
schedule is answering a harder question than the one being asked.

### What shipped as a result: `beta_union_bound`

A width derived from the comparison count instead of an RKHS norm:
`sqrt(2 ln(cells · rounds / δ))`.

| width | value | certified | violations | rate |
|---|---:|---:|---:|---:|
| `confidence_schedule` (paper) | 9.135 | 308 | 0 | 0.00000 |
| `beta_union_bound` | 6.774 | **410** | **0** | 0.00000 |

**+33% certified growth at zero measured violations.** Its doc comment states
the assumption plainly rather than burying it: the `sqrt(2 ln(1/δ'))` z-score is
a sub-Gaussian tail applied to a posterior that is only approximately Gaussian,
which makes it *derived-but-approximate* where the paper's is
worst-case-rigorous. It is **offered, not defaulted**, and a caller who adopts
it owes the calibration check this bench runs on their own field. For a rigorous
small-`n` bound the answer is an exact Clopper-Pearson / Beta quantile; this is
the closed-form, allocation-free middle.

---

## Decision: stays opt-in (Plan 580 T5.2, not T5.1)

G1–G4 all PASS, and the primitive is the only arm that can be deployed where the
guarantee matters. It is still **not promoted to default**:

1. **The stated floor gate FAILS.** It fails on a metric this bench shows to be
   degenerate — but redefining a gate's metric after seeing the result, and then
   promoting on the redefinition, is how a gate stops being a gate. The
   corrected metric is proposed below; the re-gate runs against it.
2. **No in-tree consumer.** The plan's consumers are riir-ai `cgsp_runtime` and
   riir-games `swarm/coverage_curiosity`. Phase 4 T4.1/T4.3 are open. Promoting
   a primitive nothing calls buys default-build cost for zero delivered value.
3. **Opt-in costs nothing.** std-only, zero deps, feature-gated.

### Proposed corrected gate for the re-gate

Replace `growth · (1 − violation_rate)` with a **two-stage** gate, because a
certification primitive has a hard constraint and a soft objective, not one
blended score:

1. **Admissibility (hard):** measured violation rate ≤ δ. An arm that breaches
   δ is excluded outright — it has not produced a certified set. This alone
   eliminates the adjacency floor (0.306 vs 0.05).
2. **Growth (soft, among admissible arms only):** certified count, or recall
   against `n_valid`. Compare only arms that cleared stage 1.

Under that gate the floor is inadmissible and the primitive passes unopposed —
which is the honest reading of the measurement, and is what a caller deploying
a coverage guarantee actually needs to know.

## Open

- **T4.1** `CertifiedFrontier → SafeManifoldGraph` adapter (grow-then-navigate).
- **T4.3** riir-poc four-arm gate (riir-ai side).
- **Re-gate** on the corrected two-stage metric, once a consumer exists.
- **`beta_union_bound` rigour**: an exact Clopper-Pearson / Beta-quantile variant
  would make the tighter width worst-case sound rather than approximate.
