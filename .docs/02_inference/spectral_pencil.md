# spectral_pencil — the affine matrix pencil scalar gate

> Issue 676 / Research 495 / [Bench 671](../../.benchmarks/671_spectral_pencil_goat.md) ·
> Source: arXiv:2608.08003 "The Spectral Neuron" (Shtoff, TII 2026) ·
> Feature: `spectral_pencil` (opt-in; implies `hebbian_kernel_memory` for the
> `SeedRng` substrate) · Code: `katgpt-core/src/spectral_pencil/`

The scalar decision function `f(x) = λk(A₀ + Σ xᵢAᵢ)` — the input enters
**linearly** into a symmetric matrix; the nonlinearity is reading **one
ordered eigenvalue**. Expressivity grows with matrix dimension d while
retaining linear-model-style transparency:

- **Shape by construction** — k=1 concave, k=d convex (Rayleigh–Ritz);
  `Aᵢ ⪰ 0` ⇒ non-decreasing in `xᵢ` (Loewner) — mixable per feature
  (`shape.rs`).
- **Global influence bounds** — `|f(x+δ)−f(x)| ≤ Σ|δᵢ|·‖Aᵢ‖₂` (Weyl), closed
  form from coefficients (`bounds.rs`).
- **Exact local attribution** — `∂f/∂xᵢ = vᵀAᵢv` at simple eigenvalues
  (Hellmann–Feynman), `‖VᵀAᵢV‖₂` bound at repeats (`attribution.rs`).
- **Canonical gauge** — `{Λ, VᵀAᵢV}` with sign-fixed vectors: stable
  commitment bytes per-binary (`gauge.rs`).
- **Invertible monotone warp** — `g(x) = λk(A + xB)`, `B ≻ 0`, closed-form
  inverse via the mirrored index (`warp.rs`).

## Cost table (measured, Bench 671)

| Path | d=8 | d=16 | d=32 | Theory (paper §7.3) |
|---|---|---|---|---|
| dense eval (pinned Jacobi) | 3.95 µs | 21.1 µs | 166.7 µs | ~4/3·d³ ≈ 8K FLOPs @ d=16 |
| tridiag eval (Sturm bisection) | 748 ns | 2.05 µs | 3.71 µs | ≈ 60·d ops/eigenvalue |
| Sturm `count_below` | — | **51 ns** | — | O(d), exact integer |

The wall-clock gap vs the FLOP arithmetic is the determinism policy's price
(f64 accumulation + pinned full sweeps + per-eval materialization). At the
10k NPC × 20 Hz production shape the **tridiagonal family is the per-tick
path** (~41% of one P-core at d=16; ~15% at d=8); dense is the
low-cardinality path (spawn-time construction, GM inspection, canonical
gauge) — see Bench 671 §G2 for the full arithmetic.

## The γk ≥ ½ initialization lemma (paper Lemma 2, proof sketch)

**Construction.** `A₀ = Qᵀ·diag(−1,…,−1, 0@k, 1,…,1)·Q` with Q from the QR
of a Gaussian matrix (Householder, positive-R sign fix); `Aᵢ = αᵢI +
diag(εᵢ)` with `αᵢ ~ U(±1/√n)`, `εᵢⱼ ~ U(±1/(20n))` — the R=5 baking of the
lemma's `1/(4Rn)`.

**Claim.** γk(A(x)) ≥ ½ whenever ‖x‖∞ ≤ 5.

**Sketch.** `Σᵢ xᵢαᵢ·I` is a multiple of the identity — it shifts every
eigenvalue equally and never moves a gap. The only gap-moving term is the
diagonal jitter `E(x) = Σᵢ xᵢ·diag(εᵢ)`:

```text
‖E(x)‖₂ = max_j |Σᵢ xᵢ εᵢⱼ| ≤ R·n·max|εᵢⱼ| = 5n · 1/(20n) = ¼
γk(A(x)) ≥ γk(A₀) − 2‖E‖₂ = 1 − ½ = ½        (Weyl)
```

The tridiagonal family rescales ε to `1/(12Rn)`: Gershgorin's row radius
covers 3 entries per row (sparsity-blind bound), so `3·R·n·ε ≤ ¼`.

Gate: `seeded_dense_eigengap_ge_half_on_box` /
`seeded_tridiag_eigengap_ge_half_on_box` (frozen seeds, box corners +
interior sweep) — pinned in `tests.rs`.

## Determinism policy — no library eigensolver on committed paths

Committed readouts (chain predicates, canonical-gauge bytes, committed
floats) must be bit-reproducible **per binary**. Library eigensolvers vary
rotation order, blocking, and vectorization across versions and targets.
This module pins everything:

- **Dense**: cyclic Jacobi, strict `p<q` row-major schedule, convergence at
  `off² ≤ (1e-7·‖A‖_F)²` with a 30-sweep cap, selection sort with
  eigenvector-column permutation (`dense.rs`).
- **Tridiagonal**: LDLᵀ Sturm count with the zero-pivot convention
  (`d == 0 → +ε·max(1, b²)`), 60-step Gershgorin bisection, early exit only
  on bracket underflow (`tridiag.rs`).
- **QR**: Householder with `α = −sign(x₀)·‖x‖` + positive-R sign fix
  (`init.rs`).
- **RNG**: `SeedRng` (splitmix64 + Box-Muller) seeded from
  `BLAKE3(seed_bytes)[0..8]`.

**Integer Sturm counts are the platform-stable exact class** — the only
quorum-grade readout with zero float cross-platform drift (the chain-seam
composition stays deliberately unfiled until a Glacial-rate consumer
exists; Research 495 §3.1 P2).

## UQ scope-limit (T12, the "Report the Floor" rule)

The eigengap-confidence sigmoid `σ(−γk/τ)` and the monotone box→interval
readout are UQ-bearing and **not shipped** — both must beat the conformal
floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`,
`conformal/floor_harness.rs`) first. The raw γk ships as a structural
certificate (trust-flag threshold), not a calibrated probability. KARC
precedent: scope-limit honestly.

## Consumers

- riir-ai Issue 736 — personality gates + certificate Lipschitz +
  exact-attribution PoC (the armed Super-GOAT re-gate converter).
- riir-train Issue 472 — sym packing, squareplus, eigengap init, hero
  4th-fusion-arm training recipes.
- `CommittedFieldBlend` — certificate-backed `lipschitz_bound()` upgrade
  (Research 495 §5 fusion).

## Layout

| File | Contents |
|---|---|
| `sym.rs` | isometric `1/√2` packing, `SymPacked`/`Tridiagonal` |
| `dense.rs` | pinned cyclic Jacobi + `DenseScratch` |
| `tridiag.rs` | Sturm count + bisection + `TriScratch` |
| `init.rs` | seeded γk≥½ constructors, squareplus |
| `bounds.rs` | spectral norms, growth envelope, Weyl budgets |
| `attribution.rs` | Hellmann–Feynman influence + trust flag |
| `shape.rs` | PSD/NSD/rank-one DSL, `Temperament` |
| `gauge.rs` | canonical form (commitment bytes) |
| `warp.rs` | invertible monotone warp + closed-form inverse |
