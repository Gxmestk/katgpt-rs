# Bench 687 — Certified Frontier Phase 0 PoC: acquisition PASSES, dilation is conditional

Status: **RECORD — measured 2026-08-28.** Verdict: Plan 580's three stated
Phase 0 exit criteria all **PASS** (zero violations, monotone growth, 51.4×
passive-vs-frontier separation). A fourth question the plan does not ask —
*does the Lipschitz dilation contribute anything?* — returns **conditional,
with a measured feasibility law**: at coarse resolution the plan's core op
(`expand_certified`, Eq 32) certifies **nothing at all**.

Repo: katgpt-rs · Example: `crates/katgpt-core/examples/certified_frontier_01_basic.rs`
Plan: [580](../.plans/580_certified_frontier_primitive.md) T0.1 / T0.2 (+ T0.3, added)
Research: [510](../.research/510_ActFlow_Certified_Frontier_Expansion.md) · arXiv:2606.08802

---

## Setup

Std-only, zero-dep, LCG-seeded. Smooth latent field `g`, validity probability
`p = sigmoid(g)`, threshold `h = 0.60`. The verifier is **binary and
stochastic** — one Bernoulli(`p`) draw per query; the algorithm never reads
`p`. Certified lower bound `cb` is a Beta-Bernoulli LCB (`μ − β_t·σ`, union
bound over cells × rounds, monotone in `t`) relaxed outward by the Lipschitz
rule. Budget 6 000 queries, δ = 0.05.

`L` is measured **exactly on the grid** (max adjacent `|Δp|` / spacing) rather
than derived as `L_s·L_g`. That isolates the acquisition question from
Lipschitz-*estimation* error — and it means the dilation is sound by
construction here, so the violation count tests the confidence bound, not the
constant. A real deployment must bound `L` a priori; a too-small `L` is
unsound and that failure mode is **invisible in this harness**. Flagged, not
measured.

## T0.1 — dense world (smooth block checkerboard, 32×32)

```
certified 30/400 truly-valid | violations 0 | by-cause: direct 30 dilated 0 | monotone true
```

Zero violations, monotone. Note a literal single-cell checkerboard was *not*
used: alternating cells have unbounded Lipschitz constant, so no dilation
could ever be sound and the certified set could only ever equal the queried
set — the setup would answer the question by construction. A smooth field,
thresholded, is the paper's actual illustrative shape.

## T0.2 — sparse corridor, passive vs frontier at equal budget

| seed | passive certified | frontier certified |
|---|---|---|
| 1 | 1 | 52 |
| 2 | 1 | 49 |
| 3 | 1 | 53 |
| 4 | 1 | 53 |
| 5 | 1 | 50 |

**mean 1.0 vs 51.4 → 51.4× separation, 0 violations on both arms.**

Passive certifies exactly the seed cell and never more, at every seed: uniform
sampling spreads 6 000 queries over 1 024 cells (~6 each), which is nowhere
near enough for any single cell's LCB to clear `h`. That is the Prop-1
mechanism in miniature — not a marginal effect.

## T0.3 (ADDED) — does the Lipschitz dilation contribute anything?

The plan does not ask this, but it decides whether functions **4**
(`reachability_dilation`) and **5** (`expand_certified`, labelled "THE core
op") are worth building: if every certified cell had to be queried anyway,
a plain per-cell Beta bound *is* the whole primitive and the frontier
machinery is decoration.

**First measurement was wrong and said so.** Counting end-state "certified but
never queried" returned 0 everywhere — confounded, because the frontier policy
hands maximum posterior σ to any freshly-certified cell, so it gets queried
moments later regardless of how it was certified. The fix is to attribute each
certification **at the moment `cb` first crosses `h`**, by cause.

| grid | hop decrement `L·spacing` | headroom `best_cb − h` | certified | direct | **dilated** | viol |
|---|---|---|---|---|---|---|
| 16×16 | 0.2942 | 0.2083 | 9 | 9 | **0** | 0 |
| 32×32 | 0.1496 | 0.1437 | 30 | 30 | **0** | 0 |
| 64×64 | 0.0745 | 0.0884 | 74 | 68 | **6** | 0 |
| 96×96 | 0.0495 | 0.0833 | 113 | 83 | **30** | 0 |

### The law

A hop is admissible iff the achievable certified lower bound clears the
threshold by at least one hop's cost:

```
best_cb − h  ≥  L · spacing        where  L · spacing = max adjacent |Δp|
```

The crossover lands exactly where the two columns cross — between 32×32
(0.1437 < 0.1496, **0 dilated**) and 64×64 (0.0884 > 0.0745, dilation fires).
By 96×96 dilation does **27%** of all certifications. Predicted and observed
agree on all four points.

### Why it bites

`L·spacing` is the largest adjacent `|Δp|` **anywhere on the grid**, but the
cells that want to dilate sit on the field's *plateau*, where the local
gradient is near zero. A single **global** Lipschitz constant therefore
charges plateau hops the steepest-cliff price. The paper's `L = L_s·L_g` is
global in exactly the same way, so this is not an artifact of the Beta
substitute.

## Consequences for Plan 580 Phase 1

1. **`FrontierConfig` must expose the feasibility check.** A caller who picks
   a coarse cell grid gets an expensive, silent no-op: `expand_certified`
   runs, allocates, relaxes, and certifies nothing. The primitive should
   compute `L·spacing` vs the attainable headroom and **report** it — a
   `dilation_feasible()` predicate or a warning field, not a silent zero.
2. **A local/anisotropic Lipschitz estimate is where the value is.** The
   global constant is what makes rows 1–2 dead. This is a design change to
   function 4, worth deciding before T1.4 writes it.
3. **The acquisition half stands on its own.** Functions 6 (`acquire_frontier_target`)
   and 7 (`should_advance`) delivered the entire 51.4× at 32×32 — a grid where
   dilation contributed nothing. If Phase 1 needs to be cut, cut from the
   dilation side, not the acquisition side.
4. **The plan's stated exit criteria are met**, so Phase 1 is justified — with
   the scope note above rather than as-written.

## Honest scope

- Beta-Bernoulli per-cell posterior, **not** the Eq 10 kernel posterior with
  incremental Cholesky (Phase 1 T1.1/T2.1). Cells share no information except
  through the Lipschitz hop, which if anything *under*-states the real
  primitive's reach.
- `β_t` is a union bound, not Eq 31/37. Monotone in `t`, which is the property
  T2.4 pins, but the constants are not the paper's.
- Zero violations is an **empirical** result over these runs, not the δ-level
  soundness proof — that is T2.2 (≥1 000 seeds, adversarial orderings).
- 2D grid only. The primitive's target consumers are higher-dimensional
  latents, where "adjacent cell" and this grid-exact `L` both need rethinking.
