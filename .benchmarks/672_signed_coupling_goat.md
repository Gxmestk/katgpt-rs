# Bench 672 — `signed_coupling_dynamics` GOAT gate

**Issue:** [680](https://github.com/) `.issues/680_signed_coupling_dynamics_primitive.md` (resolved by this bench)
**Research:** [.research/497](../.research/497_Signed_Coupling_Opinion_Phase_Forecast.md)
**Source:** El, Paeng, Dinc, Su, Erdogan, Pappu, Ye, Zhao, Ganguli & Zou,
"Physics of Agents: Statistical Mechanics Predicts Collective Behavior of AI
Agents", [arXiv:2608.16578](https://arxiv.org/abs/2608.16578), 17 Aug 2026.
**Date:** 2026-08-22 · **Box:** M3 Max (16 cores), release profile, sibling
RLVR training running at ~900% CPU throughout (which is why G2 uses an
interleaved A/B protocol — see below).
**Verdict:** **G1–G4 ALL PASS.** Stays **opt-in** — promotion waits on a
production consumer, the `clr_weighted_set_attention` precedent.

Harnesses:
- `crates/katgpt-core/tests/bench_680_signed_coupling_goat.rs` (G1a–G1d, G2)
- `crates/katgpt-core/tests/signed_coupling_alloc_check.rs` (G4, isolated
  binary — a `CountingAllocator` global in the timing binary would perturb the
  `Instant::now()` loops)
- 12 in-module unit tests (`--lib signed_coupling`)

```bash
cargo test -p katgpt-core --no-default-features --features signed_coupling_dynamics --lib signed_coupling
cargo test -p katgpt-core --no-default-features --features signed_coupling_dynamics --test bench_680_signed_coupling_goat --release -- --nocapture
cargo test -p katgpt-core --no-default-features --features signed_coupling_dynamics --test signed_coupling_alloc_check --release -- --nocapture
```

---

## G1a — the three regimes, deterministic mean-field rollout · PASS

Same kernel, same 256-entity crowd, same near-neutral start (`±0.02` random
init), 200 steps of `m ← 2σ(h) − 1`. Only the graph family and the couplings
move.

| regime | graph family | couplings | \|n\| | c | gate |
|---|---|---|---|---|---|
| indifference | random signed (deg 8, 30% discordant) | default @ T=40 | **0.0000** | **0.0000** | \|n\|<0.10, c<0.10 |
| consensus | random, 5% discordant | default @ T=0.5 | **1.0000** | **1.0000** | \|n\|>0.90, c>0.80 |
| polarization | frustrated two-block (rank-2) | repulsive corner @ T=0.5 | **0.0000** | **0.9998** | \|n\|<0.25, c>0.80 |
| *lattice quench* | 16×16 square lattice, all concordant | default @ T=0.5 | 0.2714 | 0.9778 | c>0.80, \|n\|<0.60 |
| *lattice + shared field* | same, `g_i = +0.2` | default @ T=0.5 | 1.0000 | 1.0000 | n>0.90, c>0.80 |

`c_polarization − c_indifference = 0.9998` (gate ≥ 0.70). **This is the whole
reason `crowd_conviction` had to ship**: to `net_opinion` alone, a deadlocked
two-faction standoff and an apathetic crowd are the same reading (`|n| ≈ 0`);
to `mean(s²)` they are opposite (0.9998 vs 0.0000).

### Two honest findings the gate forced out

1. **`β₀ > β⁻` makes rivals attractive.** A discordant tie's net weight is
   `β₀ − β⁻`. At the paper's fitted-range *midpoints* (which is what
   `Couplings::default()` is) that value is `0.8 − 0.65 = +0.15` — still
   positive. Mere connection outweighs rivalry, so a perfectly frustrated
   two-block graph **converges** rather than polarizing. That is faithful to
   the paper's own §6 consensus bias (`β⁺ > β⁻` in every model × dataset
   cell), not a defect — but it means "cold ⇒ polarized" is false at the
   defaults. Polarization needs `β⁻ > β₀`, which is reachable *inside* the
   fitted ranges at the corner `β⁺=0.9` (lo), `β⁻=1.1` (hi), `β₀=0.6` (lo) →
   discordant weight `−0.5`. The gate uses exactly that corner; nothing was
   invented to make it pass. Now recorded on the `Couplings` type docs.
2. **"Cold ⇒ consensus" is a statement about the graph, not only the
   temperature.** A cold *short-range* lattice quenched from a near-neutral
   start freezes into domains (`|n|=0.27`, `c=0.98`) — committed everywhere,
   leaning nowhere. Consensus on that same lattice needs a shared disposition
   (`g_i = +0.2` → `n=1.0`). The first draft of this gate asserted lattice
   consensus at zero field and failed; the physics was right and the gate was
   wrong.

## G1b — the three regimes, seeded stochastic rollout (discrete ±1) · PASS

400 steps, splitmix64 uniforms through `sample_states_into`, conviction read
off each entity's back-half time-averaged stance.

| regime | \|n\| (last tick) | c (time-avg magnitude) | gate |
|---|---|---|---|
| consensus (allied random) | **1.0000** | **1.0000** | \|n\|>0.90, c>0.80 |
| polarization (frustrated) | **0.0000** | **0.9994** | \|n\|<0.30, c>0.70 |
| indifference (random, T=40) | **0.0391** | **0.0053** | \|n\|<0.20, c<0.15 |

**Reproducibility:** two rollouts at the same seed are equal element-for-element
on both the final states and the magnitudes (`assert_eq!`, not a tolerance).
The kernel holds no RNG — the caller supplies the uniforms — so a replayable
rollout is exactly a replayable uniform stream.

## G1c — `β⁺ > β⁻` orders a frustrated crowd (the paper's mechanism) · PASS

Same frustrated graph, same start, only the coupling ratio moves:

| couplings | \|n\| |
|---|---|
| symmetric (`β⁺ = β⁻ = 1.0`, `β₀ = 0`) @ T=0.5 | 0.0000 (deadlock) |
| ally-dominant (`β⁺ = 2.0`, `β⁻ = 0.2`, `β₀ = 0.6`) @ T=0.5 | 1.0000 |

Gate: `|n|_symmetric < 0.25` and `|n|_ally > |n|_symmetric + 0.30`. This is the
paper's universal finding reproduced as a mechanism, not a fit.

## G1d — χ locates an interior critical temperature · PASS

41 log-spaced temperatures over `[0.1, 10]` (the paper's sweep resolution),
128 entities, 200-step burn-in + 600 sampled ticks per point, χ = N·Var_t(|n|)
from `SusceptibilityAccumulator`.

- χ peak at **T_c = 6.31**, `χ_max = 3.497` — **interior** (not at either endpoint — an endpoint peak would mean
  the sweep missed the transition).
- Peak dominates both tails by >4×: `χ(T=10) = 0.7266`, `χ(T=0.1) = 0.0000`
  (frozen crowd, no fluctuation at all).

Offline by construction, as the primitive's docs promise: 41 points × 800 ticks
is a bench workload, not a tick-rate path. What ships at runtime is the
accumulator — a live "how twitchy is this crowd" reading.

## G2 — latency vs a hand-rolled explicit three-sum baseline · PASS

Baseline = what a consumer writes reading the paper's equation literally: three
separate accumulators (`J⁺`, `J⁻`, `|J|`) and a `match` on the tie sign inside
the edge loop. Degree 8, 2000 iterations per round, **9 interleaved rounds**,
verdict = **median of per-round ratios** (not a ratio of medians — the box was
running sibling compute, and a load spike hitting one arm must not decide the
gate; the first draft used single-shot per-arm timing and flaked 1 run in 3).

| N | entries | kernel ns/update | ns/entry | baseline ns | median pairwise ratio |
|---|---|---|---|---|---|
| 32 | 254 | 443–529 | 1.74–2.08 | 455–544 | 0.969–0.982× |
| 256 | 2036 | 3575–4539 | 1.76–2.23 | 3649–4464 | 0.971–1.016× |
| 1024 | 8176 | 14885–17144 | 1.82–2.10 | 15005–17366 | 0.969–0.990× |

Five consecutive runs, all PASS (gate: ratio ≤ 1.15, kernel < 200 µs Plasma
budget). ~1.8–2.1 ns/edge, flat from N=32 to N=1024. The absolute spread
(±15%) is box contention from the sibling RLVR run — which is exactly why the
gate is on the **pairwise ratio** (spread 0.969–1.016×, i.e. ±2.4%) and not on
either arm's absolute number.

### The optimization that mattered, and the one that didn't

The paper's `|J|` channel is **not independent**: with `P = Σ_concordant s_j`
and `D = Σ_discordant s_j`, the third sum is just `P + D`, so

```text
h_i = (β⁺ + β₀)·P + (β₀ − β⁻)·D + g_i
```

— two channel sums, weighted **once per node**. The edge loop is then two
conditional adds with **no multiply and no table load at all**.

The first draft instead collapsed the couplings into a 2-entry sign-indexed
weight table and multiplied *per edge* (`h += w[sign_bit] * s_j`). Branch-free,
and **1.52× slower than the naive baseline** at N=32 — an indexed load plus a
multiply per edge costs more than the branch it removed. The per-node hoist is
what recovered parity. Recorded on the module docs so the next person to
"optimize" this loop doesn't re-walk it.

## G3 — no-regression · PASS

- `cargo test -p katgpt-core --lib` (default features): **1904 passed, 0
  failed**, 7 ignored — the feature is opt-in and adds nothing to the default
  surface.
- `cargo clippy -p katgpt-core --all-features`: clean.
- `cargo clippy -p katgpt-core --no-default-features --features
  signed_coupling_dynamics`: clean.
- `--lib signed_coupling`: 12 unit tests pass.

## G4 — alloc-free steady state · PASS

`CountingAllocator`, isolated binary, one `#[test]` (the counter is a process
global), N=256 ring graph, 1000 calls per path:

| path | allocs / 1000 calls |
|---|---|
| `signed_coupling_update_into` | **0** |
| `signed_coupling_update_informed_into` | **0** |
| `sample_states_into` | **0** |
| `net_opinion` + `crowd_conviction` + `SusceptibilityAccumulator::observe`/`susceptibility` | **0** |
| `SignedGraph::from_edges` (construction — reported, deliberately not gated) | 14 |

The contract is "no heap **after** construction", so the builder's 14
allocations are in scope of the measurement and out of scope of the gate.

---

## What is NOT claimed

- **No calibrated-forecast claim.** `σ(h_i)` is a Bernoulli parameter of a
  dynamics rule. The paper's 75–86% balanced accuracy is a claim about
  predicting *real LLM crowds*; nothing here is measured against it. Any future
  prediction-quality claim on this primitive is UQ-bearing and owes the
  conformal-naive floor (`AGENTS.md` §"Report the Floor", Issue 010).
- **No consumer.** Promotion to default-on requires a production consumer, per
  the CLR precedent. Research 497 §7 lists the candidates (swarm emotions is
  the natural first); `goat-audit` runs before any riir-ai plan consumes this.
- **No prior-art novelty claim on the framing.** Research 497 §3 already scored
  Q1 **NO** — "statistical mechanics predicts LLM agent collective behavior" is
  published (De Marzo arXiv:2605.10721, De Nobili arXiv:2608.02178). Gain, not
  Super-GOAT. What this bench measures is that the *kernel + reducers* work and
  cost nothing, not that the idea is ours.
