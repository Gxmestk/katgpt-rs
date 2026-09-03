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

---

## T5.3 addendum — the regime-conditional dual form (2026-09-03)

**Status:** LANDED + measured in-tree. `DualPosteriorBuffer<D>` ships beside
`PosteriorBuffer<MAX_OBS, D>` behind the same `certified_frontier` feature,
with a `LinearPosterior<D>` trait so one caller can drive either arm and
`prefer_dual(expected_obs, d)` naming the crossover.

### Why a second factorisation and not a replacement

`PosteriorBuffer` factorises the `n × n` Gram matrix, which is correct for this
plan's own setting (`n < D`: observations scarce, latent wide) — and it is now
also the **oracle** the dual is gated against, since T2.1 already gated it
against an independent dense f64 solve. The first external consumer inverts the
regime (riir-train Plan 357 T1.2 — a 4096-sample warm-up against a 32-D
projected feature), which is where the primal is the wrong end. For a linear
kernel the two are the same number exactly, by Woodbury:

```text
k(x,x) − k(x,X)(X Xᵀ + λI)⁻¹k(X,x)  ==  λ · xᵀ(XᵀX + λI)⁻¹x
```

### G2-dual — measured on this box, not quoted from the consumer

`cargo test --release -p katgpt-core --features certified_frontier --test
bench_688_certified_frontier_goat -- t53 --nocapture`, M3 Max, D = 32,
2 000 queries/point, both arms warmed:

| n | dual ns/query | primal ns/query | speedup |
|---|---|---|---|
| 16 | 253 | **195** | 0.8× (primal wins — `n < D`) |
| 64 | — | 1 210 | — |
| 256 | 208–256 | 17 530–17 788 | **69–84×** |
| 4096 | 252 | — (64 MiB of state) | — |

- **dual scaling 16 → 4096: 0.967–0.997×** across four runs — `O(1)` in `n`.
- **primal scaling 16 → 256: 88.9–94.5×** — the `O(n²)` it is.
- State: `size_of::<DualPosteriorBuffer<32>>()` = **4 368 B at any `n`**, pinned
  by test, versus 291 KiB at `MAX_OBS = 256` and >64 MiB at 4096.

The n = 16 row is the point of `prefer_dual`, not a defect: below the crossover
the primal is genuinely faster, so the rule `expected_obs > d` is validated by
measurement rather than asserted. Bars are on the **scaling class** (a property
of the algorithm) rather than a microsecond budget (a property of the machine).

### G1-dual — equivalence, and what kind of equivalence

`t53_primal_dual_equivalence_across_n_and_lambda`: every `n` in 0..=48 (spanning
both regimes) × λ ∈ {1, 1e-1, 1e-2} × 16 fixed probes.

| quantity | worst relative deviation | bound |
|---|---|---|
| `posterior_variance_linear` | **3.229e-6** (at n=8, λ=0.01) | 2e-3 |
| `ridge_mean` | **1.614e-4** | 2e-3 |

**Tolerance, not bit-identity — and that is a fact about the identity, not a
weakened bar.** Two different f32 expression trees for the same real number
cannot agree bit-for-bit; the primal in particular computes `k(x,x) − quad` as
a difference of two possibly-large terms and loses digits exactly where `σ²` is
small, which is why the bar is relative-or-absolute. Neither arm is on a sync
surface, so no quorum claim depends on this. Bit-identity *is* asserted where it
is available: `t53_dual_is_bit_identical_across_runs` compares one arm with
itself over 200 observations and 32 probes.

Five more gates in `certified_frontier_correctness` (31 tests total, all green):
`n = 0` parity with `k(x,x)` in both forms; NaN/inf rejection that cannot poison
the factor (one NaN in a Cholesky factor poisons *every later query about an
unrelated direction*, so this is the failure mode being ruled out, not
defensive noise); 500 identical observations needing no numerical floor —
the dual's pivots are bounded below by `√λ` by construction, so the primal's
`rem.max(λ·1e-6)` has no analogue; the regime rule and the state-size pin; and
a generic caller driving both arms through the trait off `scratch_len()`.

### Two things this addendum does NOT claim

- **It is not a promotion.** `certified_frontier` stays opt-in. The blocker is
  unchanged and is not perf: the T3.4 floor gate's stated product metric is
  degenerate and the re-gate awaits a default-path consumer.
- **It does not make riir-train's copy redundant on its own.** riir-train wrote
  the dual independently and the two agree; folding its copy onto this one is a
  riir-train call, and the boundary direction (katgpt-rs is upstream) says that
  is the eventual home. Filed as a note, not done here.
