# Issue 151: SAR × QuasiMoTTo — Purified-Weights × Low-Discrepancy Sampling Pass@k Fusion PoC

**Opened:** 2026-07-15
**Origin:** Research 406 §8.3 Fusion F (addendum)
**Related Research:** [406 (SAR)](../.research/406_Spectral_Rewiring_Weight_Delta_Purification.md), [367 (QuasiMoTTo)](../.research/367_QuasiMoTTo.md)
**Related Plans:** [423 (spectral_rewire)](../.plans/423_spectral_rewire_primitive.md), [367 (qmc_sampling)](../.plans/367_quasimotto_qmc_sampling.md)
**Status:** Open — needs PoC before plan/super-GOAT commitment

## TL;DR

**Fusion hypothesis:** SAR (weight-delta purification, widens the reachable
problem set) × QuasiMoTTo (QMC lattice sampling, covers a fixed set with 50%
fewer samples) produces a **compound Pass@k gain neither alone provides**.
SAR acts on weights (deterministic, cold-tier), QMC acts on sampling
(stochastic, inference-time) — orthogonal axes.

This is the strongest surviving fusion from the Research 406 §8 addendum. It
does **not** depend on the NPC-scale concentration that G1b refuted — it
depends on the LLM-scale path (4096×4096 weight matrices), which is where
Pass@k matters anyway (reasoning models).

## The compound mechanism

```
Reachable-set widening (SAR, weight-space, deterministic, one-time cold op)
  ×
Coverage efficiency   (QMC, sample-space, stochastic, per-inference)
  =
Compound Pass@k gain (wider set covered more efficiently)
```

- **SAR alone** — removes off-manifold drift from a trained delta → wider set
  of problems reachable by repeated sampling. But each sample is i.i.d.
  (potentially redundant coverage of the widened set).
- **QMC alone** — covers a *fixed* reachable set with 50% fewer samples
  (K_qmc=8 vs K_iid=16 at pass@k≥0.5, bench 367). Cannot reach problems
  outside the set.
- **SAR + QMC** — widen the set AND cover the widened set efficiently.

## Why this survives the G1b concentration failure

`spectral_rewire` G1b FAILED at NPC scale (≤64×64, `on_manifold_fraction` in
[0.27, 0.58] vs threshold >0.8, Issue 123 CLOSED). But the SAR concentration
phenomenon IS real at LLM scale (the paper proves it for 1.5B–32B models). The
Pass@k gain in the paper's Fig 2 is measured on AIME 2024 with DeepScaleR-1.5B
and OLMo-3.1-32B — exactly the LLM-scale regime where concentration holds.

So the fusion operates at LLM scale, where:
- SAR concentration is real (paper's Fig 2, Table 5: +1 problem covered @256 rollouts)
- QMC sample reduction is real (bench 367: 50% reduction at pass@k≥0.5)

The compound gain is the open question.

## The PoC (the gate before any Super-GOAT commitment)

Per skill §1.5 "no candidate escape hatch" + §3.6 defend-wrong PoC rule, this
fusion stays an **issue** until a PoC validates the compound gain is
**super-additive** (SAR+QMC > max(SAR_alone_gain, QMC_alone_gain) + ε).

### PoC design

Four competitors on a controlled reasoning benchmark (e.g., AIME-style
synthetic math, or riir-train's existing reasoning setup):

| Competitor | Weights | Sampler | Expected |
|---|---|---|---|
| Baseline | unpurified W_RL | i.i.d. | reference Pass@k curve |
| SAR-only | W_SAR (purified) | i.i.d. | widened reachable set (paper Fig 2) |
| QMC-only | unpurified W_RL | QMC lattice | 50% sample reduction (bench 367) |
| **SAR+QMC** | W_SAR (purified) | QMC lattice | **compound — super-additive?** |

**Pass criterion:** SAR+QMC Pass@k at fixed K strictly exceeds the
super-additive bound `max(SAR-only, QMC-only) + ε` at ≥2 K values.

**Fail criterion:** SAR+QMC ≈ max(SAR-only, QMC-only) → the mechanisms are
substitutes, not complements; the fusion has no compound value.

### Where the PoC lives

- **Delta production:** `riir-train` (Issue 374 infrastructure already produces
  trained LoRA deltas; needs scale-up to a real reasoning model — this is the
  expensive part, GPU-hours).
- **SAR purification:** `katgpt-spectral::spectral_rewire` (shipped, opt-in).
  LLM-scale (4096×4096) requires the SVD col cap (Issue 124) to hold — it does
  (resolved), but one-sided Jacobi at 4096 cols is minutes/call (cold-tier
  only, acceptable for one-shot purification).
- **QMC sampling:** `katgpt-core::speculative::qmc` (shipped, default-on).
- **Pass@k measurement:** reuse bench 367's harness (fresh-draw-per-K, the G2
  bug-fix lesson).

## Routing if the PoC passes

If super-additive gain is confirmed:
1. **Open primitive** (already shipped): `spectral_rewire` + `qmc` — no new
   katgpt-rs primitive needed; the fusion is a *composition*.
2. **Architectural guide:** `riir-ai/.research/` (Reasoning Pack pillar
   amplifier — "our reasoning pack covers more problems with fewer samples").
3. **Plan:** `riir-ai/.plans/` to wire the composition into the Reasoning Pack
   runtime (the modelless residue — QMC sampling on SAR-purified weights — is
   inference-time; the SAR purification is a cold-tier pre-process).

## Routing if the PoC fails

Close this issue with the negative result. The fusion is substitutes, not
complements. Document in Research 406 §8 addendum as a refuted fusion.

## Open questions

- [ ] Does riir-train have a reasoning-model training setup that produces a
      W_RL we can SAR-purify at LLM scale? (Issue 374 was ≤64×64 synthetic; we
      need real 4096×4096 weights, or at minimum a 512×512 reasoning toy.)
- [x] **Is the one-sided Jacobi SVD at 4096 cols tractable for one-shot
      cold-tier purification?** — **ANSWERED: INTRACTABLE. See §"SVD
      tractability analysis" below.** Needs randomized SVD (Halko et al. 2011)
      as the LLM-scale algorithm; Jacobi stays for NPC-scale.

## SVD tractability analysis (2026-07-15)

**Source:** `katgpt-core/src/subspace_phase_gate.rs::one_sided_jacobi_svd_into`
(L678–910), called via `thin_svd_into` from `spectral_rewire_into`.

### Algorithm complexity

The one-sided Jacobi SVD iterates up to `max_sweeps = 60` sweeps (L732, L765).
Per sweep, it visits O(n²) column pairs (L774: `for p in 0..n, for q in
(p+1)..n`), each doing O(m) work for the dot products (L780–786) + O(m) for
the A-column rotation (L808–813) + O(n) for the V-column rotation (L814–819).

**Total: O(60 · n² · (m + n))** for an m×n matrix.

### Measured baseline (bench 423)

| Scale (m×n) | SVD path latency | Implied throughput |
|---|---|---|
| 64×64 | 2ms | ~4.6 GFLOP/s (cached-scale) |
| 512×64 | 14ms | ~3.9 GFLOP/s (cached-scale) |

Bench 423 explicitly warns: "512×512 would take minutes per call (O(n²) pairs
× 60 sweeps). 128×128 is the practical cold-tier bound."

### Extrapolation to 4096×4096 (LLM scale)

**Per-sweep flop count:** 4096²/2 · (5·4096 + 2·4096) ≈ **241 billion flops**.

**Memory footprint:** `work.a` (m·n = 64MB) + `work.v` (n² = 64MB) = **128MB
working set** — far exceeds L3 cache (8–32MB typical). Effective throughput
drops to **~0.5–1 GFLOP/s** (memory-bound, ~5–10× worse than cached-scale).

**Wall-clock estimate:**

| Sweeps to converge | At 1 GFLOP/s | At 0.5 GFLOP/s |
|---|---|---|
| 10 (optimistic) | ~40 min | ~80 min |
| 60 (worst case) | ~4 hr | ~8 hr |

**Verdict: INTRACTABLE.** Even the optimistic estimate (40 min for a single
SVD of one 4096×4096 weight matrix) is unacceptable for a cold-tier
purification step that runs per-layer per-model. A 32-layer transformer would
need ~21 hours (optimistic) to ~11 days (pessimistic) just for the SVDs.

### The unblocker: randomized SVD (Halko-Martinsson-Tropp 2011)

Randomized truncated SVD computes only the top-k singular triples in
**O(m·n·k)** where k = target rank + oversampling:

1. Draw n×(k+p) Gaussian Ω (p ≈ 10 oversampling)
2. Y = A·Ω — O(m·n·(k+p))
3. QR: Y = QR — O(m·(k+p)²)
4. B = QᵀA — O(m·n·(k+p))
5. Small SVD of (k+p)×n B — O((k+p)²·n)
6. U = Q·U_B — O(m·(k+p)²)

**For 4096×4096 with r=32, k+p=42:** ~704M flops → **~140ms (cached) to
~700ms (memory-bound)**. That's a **100–1000× speedup** over Jacobi at this
scale. A 32-layer transformer purifies in ~5–22 seconds (one-shot cold-tier).

### Recommendation

**Two-track approach for the Issue 151 PoC:**

1. **PoC validation track (fast):** Use an external SVD (numpy/scipy
   `scipy.sparse.linalg.svds`) for the cold-tier purification step. This lets
   the PoC validate the compound Pass@k gain without blocking on a new
   katgpt-spectral primitive. The PoC is about the *fusion*, not the SVD
   algorithm.

2. **Primitive track (if PoC passes):** Implement randomized SVD in
   `katgpt-spectral` as a feature-gated (`randomized_svd`) primitive — the
   LLM-scale complement to the NPC-scale Jacobi SVD. Same `SvdResult`
   output shape so `spectral_rewire_into` can use either backend. This is a
   pure linear-algebra primitive (modelless — no training, no gradients), so
   it belongs in the public engine. **File as a sub-issue of 151 if/when the
   PoC passes; not worth implementing until the compound gain is confirmed.**

The Jacobi SVD stays the default for NPC-scale (≤128×128) — it's deterministic,
zero-alloc, and correct at that scale. Randomized SVD is the LLM-scale path.
