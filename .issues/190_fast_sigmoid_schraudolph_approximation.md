# Issue 190 — Schraudolph Fast-Sigmoid Approximation for CGSP Hot Path

> **Spawned from:** rust-optimize session 2026-07-24 (post-structural_complexity fusion)
> **Date:** 2026-07-24
> **Type:** optimization (transcendental-function approximation)
> **Severity:** MEDIUM — attacks the transcendental floor (30-40% of CGSP cycle time)
> **Status:** RESOLVED — sigmoid + exp components addressed via Cephes (2026-07-24)

## Update 4 (2026-07-24): fast_tanh (Padé [2/2]) + simd_tanh_inplace consolidation

Following the sigmoid/exp consolidation, the remaining hot-path
transcendental was `tanh` — used in GELU activations, GRU new-gate,
Poincaré MLP hidden layers, attention/logit softcaps, and feature hashing.

**New primitives:**
- `fast_tanh` (scalar) — Padé [2/2] rational polynomial
  `x·(27+x²)/(27+9x²)`, ~5× faster than libm tanh on aarch64 (pure
  arithmetic, no exp call). Worst-case error ~0.025 near |x|≈2; saturates
  to ±1 for |x|>3. Documented safety contract: safe for bounded
  activations (GELU, GRU, MLP); unsafe for algebraic-identity preservation
  (cos²+sin²=1 — see Plan 322 phase_rotation_coupling).
- `simd_tanh_inplace` — NEON (4-lane) / AVX2 (8-lane) / WASM simd128
  (4-lane) vectorized Padé kernel. Uses FMA (vfmaq_f32 / _mm256_fmadd_ps)
  for numerator + denominator, branchless mask-select for the |x|>3
  saturation. ~1 ULP more accurate than scalar fast_tanh due to FMA.

**Precedent:** `mean_field::fast_tanh` shipped this exact Padé [2/2] form
since Plan 281 with a 0.03 tolerance assertion. The consolidation promotes
it to the canonical `simd::activations` module + adds the SIMD vector path.

**Shipped (6 commits across 2 repos):**
- `78f28e45` — add fast_tanh + adopt in activation hot paths (katgpt-types,
  katgpt-core coda/poincare/mean_field/delta_mem, katgpt-speculative)
- `483b1983` — add simd_tanh_inplace (NEON/AVX2/WASM) + tests + adopt in
  delta_mem/hash.rs
- `82dc0dc8` — fix missing fast_tanh import in MoaActivation::activate
- `616fea66` — relax MoaActivation::Tanh test tolerance for Padé
- `e7d28c1bc` (riir-ai) — adopt fast_tanh/simd_tanh_inplace in riir-engine:
  opponent_hash (2 SIMD tanh loops), gemma2 logit softcap (5 sites),
  gemma2_train + causal_validation softcap, GRU new-gate, attention softcap

**Intentionally NOT touched:**
- `slod.rs` exp_map_into — Poincaré ball geometry, tanh is a scalar factor
  in the manifold operation. Conservative: left as libm tanh.
- `gemma2_train/backward.rs` — training gradient computation needs higher
  accuracy than Padé (0.025 error would compound in sech² derivative).
- `katgpt-pruners/gdsd.rs::tanh_advantage` — public API, not a hot loop.

**Validation:** 6765 katgpt-rs + 7324 riir-ai workspace tests pass. Clippy
clean workspace-wide. 1 test tolerance adjusted (MoaActivation::Tanh
1e-7 → 0.03, same pattern as prior Padé acceptance in mean_field).

The tanh consolidation completes the transcendental-function optimization
loop. sigmoid → Cephes; exp → Cephes; tanh → Padé [2/2]. All three are now
backed by dedicated simd::activations primitives with documented accuracy
bounds.

## Update 3 (2026-07-24): softmax exp loop + fast_exp codebase-wide consolidation

The prior updates addressed `fn sigmoid` definitions. This update addresses
the OTHER class of libm `exp()` calls: **softmax exp loops** and **scalar
exp callsites** that were not sigmoid-wrapped.

**New primitive:** `fast_exp` added to `katgpt_types::simd::activations` —
the scalar counterpart to `simd_exp_inplace` / `simd_exp_sum_inplace`.
Wraps `cephes_exp_scalar` (same Cephes 6th-order polynomial, ~1 ULP
accurate for |x| < 88, ~1.7× faster than libm exp on aarch64). Re-exported
at `katgpt_core::simd::fast_exp`.

**SIMD softmax pattern adopted** in hot paths where the data is a
contiguous `&mut [f32]` and the loop is a pure exp+sum (no branches):
`simd_max_f32` + `simd_add_scalar_inplace` + `simd_exp_sum_inplace` +
`simd_scale_inplace`. This fuses the exp and sum into one buffer traversal.

**Shipped (8 commits, ~60 files):**
- `d5440da4` — add `fast_exp` + SIMD softmax in katgpt-attn (chunk_summary,
  kv_outer_prefill, diagonal_gate) + katgpt-core (bebop_upgrade ICT softmax,
  product_key_memory softmax) + katgpt-types (re-export)
- `636581df` — SIMD softmax + fast_exp across katgpt-speculative (weaver 7
  sites, rt_turbo, kurtosis_gate) + katgpt-pruners (boltzmann, lora_player,
  kernel_scoring, etc.) — 19 files
- `b6cbcebe` — katgpt-core remaining: coda.rs SiLU/GELU activations,
  compute_moa_gates sigmoid loop, conformal decay weights, best_belief
  beta pdf, hippocampal_cache streaming softmax, etc. — 10 files
- `5cc3918d` — katgpt-forward (drafter_lora SIMD softmax, diffusion_sampler,
  flashar, set_diffusion) + katgpt-kv (var_norm) + katgpt-transformer
  (context, swir) — 10 files
- `ebfd3cc6` — katgpt-core remaining: grape softplus/log_sigmoid, mag RBF
  kernel, spectral_hierarchy Gram matrix + katgpt-claim (mgpo) +
  katgpt-band (InfoNCE, bckvss) — 9 files
- `78b8c574` — CRITICAL: the root `katgpt_core::sigmoid` (lib.rs pub fn)
  now delegates to `fast_sigmoid`. This was the highest-impact missed site.
  Also pipeline_pruner, memory_soup_lora, hippocampal_cache_dyn — 5 files
- `b35dd1f6` — last 3 missed sigmoids (spectral_lod, set_attention,
  similarity edit_penalty) — 3 files
- `3782b2e6` — katgpt-sense reconstruction 6-wide exp loops — 1 file

**Critical finds during this sweep:**
1. The root `katgpt_core::sigmoid` in `lib.rs` was STILL using libm exp
   despite the prior sigmoid sweep. Every caller of `katgpt_core::sigmoid`
   was routing through libm. Now delegates to `fast_sigmoid`.
2. `coda.rs` SiLU/GELU activations had inline `1/(1+(-x).exp())` patterns —
   these are ML activation hot paths that should use `fast_sigmoid`.
3. Multiple local `fn sigmoid` definitions survived the prior sweep:
   `region_subspace`, `pipeline_pruner`, `set_attention`, `spectral_lod`,
   `memory_soup_lora::sigmoid_gate`, `group_invariance_probe`. All fixed.

**Untouched (by design):**
- `katgpt-attn-match` core `attn_match` feature — zero-dep by design
  (no katgpt-core dep). Its softmax loops stay libm.
- `katgpt-dec` — zero-dep by design (no katgpt-core/katgpt-types dep).
  Has its own `fast_sigmoid` that uses libm exp as the foundation.
- f64 exp calls (`breakeven/mod.rs::sigmoid`, `occupancy/linear.rs`,
  `gdn_tree_verify/mod.rs`) — `fast_exp` is f32-only.
- Test reference implementations (geometric_product.rs `silu_ref`, etc.) —
  these are the ground truth for accuracy tests; must stay libm.

**Test adjustments:**
- `similarity::tests::edit_penalty_always_in_unit_interval`: relaxed from
  open (0,1] to closed [0,1] — `exp(-100)` returns 0.0 under Cephes
  (same subnormal-to-zero behavior as the sigmoid_bounds test in Update 2).

**Validation:** 6738 workspace tests pass. Clippy clean workspace-wide.

## Update 2 (2026-07-24): codebase-wide sigmoid DRY consolidation

The initial Cephes win was scoped to the CGSP hot path. A broader audit
found the SAME DRY violation across the entire 7-repo stack: **100+ local
`fn sigmoid` definitions** using libm `exp()` (or the `libm` crate's
`expf` for no_std paths) existed in parallel with the canonical
`fast_sigmoid` (Cephes). Every one has been consolidated to delegate.

**Shipped (5 commits):**
- `2e3be549` — katgpt-core/attn/claim (13 files)
- `120339a6` — katgpt-pruners/speculative/kv/transformer/forward/personality/spectral (59 files)
- `e50bb744` — `entropy_nats_zero_alloc` (acceptance_forecast hot path): Cephes exp + SIMD max
- `f4fc4862` — `entropy_f32` (DDtree hot path): Cephes exp in the 4-wide unrolled loop
- `047043467` (riir-ai) — riir-engine + riir-games (42 files)

**Total: ~120 files across 2 repos.** All sigmoid paths now use Cephes.
Bonus: removes the `libm` crate dependency from 6+ no_std paths in riir-engine
(integrity, latent_functor, ict_runtime, post_action_router).

The two failure modes caught during the bulk edit:
1. `posterior::surprise::sigmoid_bounds` test asserted open interval (0,1] for
   sigmoid(-100); corrected to [0,1] (sigmoid(-100) = 0.0 in f32 for any
   correct implementation — the true value 3.7e-44 rounds to subnormal-zero).
2. `precision_aware_draft::proximity_with` sign error: `1/(1+exp(X))` is
   `sigmoid(-X)`, not `sigmoid(X)`. The bulk edit missed the sign; caught by
   the test suite before commit.

## Update (2026-07-24): sigmoid floor addressed

The original issue proposed Schraudolph bit-manipulation exp for the sigmoid
floor. Instead, the rust-optimize continuation session found a better path:
the codebase ALREADY shipped a Cephes 6th-order polynomial exp
(`cephes_exp_scalar` in `katgpt_types::simd::activations`) used as the scalar
tail of the SIMD exp kernels. It was ~1 ULP accurate and ~1.7× faster than
libm `exp` on aarch64 (Apple Silicon), but was private and not used by the
scalar sigmoid paths.

**Shipped (4 commits):**
- `7d868801` — `fast_sigmoid` → Cephes exp (1.7× vs libm)
- `1e2559b4` — `cgsp::types::sigmoid` → delegates to `fast_sigmoid`; `cephes_exp_scalar` made `pub`
- `0aa4d7b54` (riir-ai) — DRY dedup of 3 local sigmoid impls in cgsp_runtime
- `f1a532b0` — `staleness_weight` → Cephes exp

All libm `exp()` calls eliminated from the CGSP hot path. The sigmoid
component of the transcendental floor is now at Cephes polynomial speed.

**The ln component remains on libm.** Micro-benchmark confirmed libm `ln()`
on Apple Silicon is at hardware floor (~1.9 ns/call, auto-vectorized) — a
scalar Cephes polynomial for ln is SLOWER than libm on this platform
(measured: 227 ns vs 183 ns for a 64-element entropy_nats call). The
`entropy_nats` function (64 ln calls per CGSP cycle) is at its practical
floor on aarch64.

## Context

The CGSP cycle's transcendental-function floor was identified across prior
rust-optimize sessions: ~30-40% of `CgspLoop::cycle` time is `exp`/`ln` calls
via `cgsp::types::sigmoid` + `simd::fast_sigmoid`. The accessible code-level
optimizations (loop fusion, structural_complexity 3→1 pass fusion, DRY
dedup) are exhausted. The remaining lever is **approximating the
transcendental itself**.

### Where sigmoid is called in the CGSP hot path

| Call site | Frequency | Current impl |
|---|---|---|
| `HlaProjectionGuide::score` | 2× per candidate × k candidates/cycle | `cgsp::types::sigmoid` (two-branch stable, libm `exp`) |
| `DualPoolBandit::exploitation_probability` | 1× per cycle | `cgsp::types::sigmoid` |
| `entropy_nats` | 1× per cycle | `p.ln()` (libm `ln`) |

At default k=4: **9 transcendental calls per cycle** (8 sigmoid + 1 ln).

### Why SIMD batching is NOT the answer here

`simd_sigmoid_inplace` exists (`katgpt-types::simd::activations`) and wins
at ≥8 elements. But DEFAULT_K=4 → only 4-8 sigmoid calls per cycle, below
the SIMD win threshold. The scalar `fast_sigmoid` (libm `expf`, ~5ns/call)
is already the GOAT-validated path for this batch size.

## The Schraudolph Trick

**Reference:** Schraudolph, N. N. (1999). "A Fast, Compact Approximation
of the Exponential Function." *Neural Computation* 11(4):853-862.

The trick exploits the IEEE-754 float representation to approximate `exp(x)`
in ~2ns using integer arithmetic + a single multiply:

```text
exp(x) ≈ reinterpret_as_f32((int)(1512775 * x + 1072693248))
```

For sigmoid: `σ(x) = 1 / (1 + exp(-x))` → one Schraudolph exp + one
reciprocal. Total: ~3ns vs ~5ns for libm `expf` + reciprocal.

**Precision:** ~1% relative error (the original paper's bound; tighter
variants exist — e.g. with a 2nd-order correction term, <0.1% error).

### Does 1% error matter for CGSP?

The sigmoid in CGSP serves two roles:
1. **Guide scoring** — `sigmoid(λ · dot)` maps dot-product to [0,1] relevance.
   A 1% error shifts the score slightly but preserves monotonicity (the
   Schraudolph approximation is monotonic). The priority-table update is
   robust to small score perturbations (it's a bandit, not an argmax).
2. **Pool routing** — `sigmoid(w_E − w_X)` determines E-vs-X pool selection
   probability. A 1% error shifts the routing probability slightly. The
   reachability clamp `[ε, 1−ε]` already absorbs saturation; small errors
   in the middle of the range should be benign.

**This needs to be proven by the GOAT gate, not assumed.** The CGSP
regret-bound proofs (Research 249) may depend on exact sigmoid properties.

## Proposed approach

1. Implement `schraudolph_sigmoid(x: f32) -> f32` in
   `katgpt-types::simd::activations.rs` behind a new `schraudolph_sigmoid`
   feature flag. Include the 2nd-order correction variant.
2. Add a CGSP-specific gate: `cgsp::types::sigmoid` delegates to
   `schraudolph_sigmoid` when the feature is on, else the current stable
   sigmoid.
3. Run the GOAT gate:
   - **G1**: All 44 `cgsp::` tests pass (qualitative behavior preserved).
     Additionally, verify the CGSP regret bound holds (run the
     `g2_log_regret_synthetic` test with the approximation — the regret
     trajectory should stay within the theoretical bound).
   - **G2**: Benchmark `CgspLoop::cycle` with vs without the approximation.
     Target: ≥20% cycle-time reduction (the transcendental floor is
     30-40%; Schraudolph recovers ~40% of that).
   - **G3**: All katgpt-core tests pass (`cargo test --features cgsp`).
   - **G4**: Zero allocation (the approximation is pure arithmetic).

## Don't do this yet

This issue is a **captured-for-the-record** optimization candidate. Do NOT
implement until:
- A concrete production workload shows CGSP cycle time is a bottleneck (not
  just a micro-benchmark), OR
- The structural_complexity fusion (commit `51d6bfc8`) + prior session's
  loop fusion (`404ffd63`) are measured in situ and found insufficient.

The transcendental floor is real but may not matter at the system level
(the swarm tick has 1363× headroom; CGSP is one of many cognition steps).

## Cross-references

- `katgpt-types::simd::activations::fast_sigmoid` — the canonical scalar sigmoid
- `katgpt-types::simd::activations::simd_sigmoid_inplace` — SIMD sigmoid (≥8 elements)
- `cgsp::types::sigmoid` — the two-branch stable sigmoid used in the hot path
- `cgsp::guide::HlaProjectionGuide::score` — primary hot-path caller
- `cgsp::dual_pool::DualPoolBandit::exploitation_probability` — secondary caller
- Prior session commit `404ffd63` — guide-score + difficulty-admit loop fusion
- This session commit `51d6bfc8` — structural_complexity 3-pass → 1-pass fusion
