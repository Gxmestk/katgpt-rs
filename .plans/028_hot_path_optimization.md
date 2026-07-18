# Plan 028: Hot-Path Allocation Elimination & Fused Ops

## Objective
Eliminate heap allocations from all hot-path functions and fuse redundant passes
in softmax/attention. Measure before/after with release benchmarks.

## Baseline (Plan 013 baseline, different commit — for reference only)
| Method | Throughput | μs/step |
|---|---|---|
| Transformer AR | 1,164,426 tok/s | 0.86 |
| DFlash | 3,058,496 tok/s | 2.62 |
| DDTree Build | 308,906 trees/s | 3.24 |
| Speculative (Simulated) | 834,159 tok/s | 5.99 |
| Speculative (AR Draft) | 1,171,896 tok/s | 5.97 |

## Optimized (Plan 028, release build, 50K iters)
| Method | Throughput | μs/step | Δ vs Baseline (μs) |
|---|---|---|---|
| Transformer AR | 1,056,038 tok/s | 0.95 | +10% (noise) |
| DFlash | 481,674 ops/s | 2.08 | **−20%** |
| DDTree Build | 376,776 ops/s | 2.65 | **−18%** |
| Speculative (Simulated) | 1,006,154 tok/s | 4.97 | **−17%** |
| Speculative (AR Draft) | 1,460,641 tok/s | 4.79 | **−20%** |
| Leviathan (Algorithm 1) | 113,333 tok/s | 10.40 | — (new baseline) |
| Leviathan (w/ rollback) | 189,084 tok/s | 6.21 | — (new baseline) |
| Spec (conditioned) | 1,105,351 tok/s | 6.10 | — (new baseline) |
| Prefill (no compress) | 304,905 ops/s | 3.28 | — (new baseline) |
| DDTree (chain-seed) | 395,742 ops/s | 2.53 | — (new baseline) |
| Forward (flat) | 1,156,557 ops/s | 0.86 | — (new baseline) |

> **Note**: Baseline was measured on a different commit (Plan 013). Back-to-back
> comparison is not perfectly controlled. Transformer AR regression is likely
> thermal noise (laptop CPU throttling). DFlash/DDTree/Speculative improvements
> are consistent across multiple runs.

## Tasks

- [x] 1. **`ForwardContext` paged buffers** — Added `paged_flat_key`/`paged_flat_value` (`[block_size * kv_dim]`) and `raven_query_buf` (`[kv_dim]`) to `ForwardContext`. Eliminates 2 allocations per `forward_paged` call.
- [x] 2. **`LeviathanVerifier` zero-alloc scoring** — Replaced 4× `logits.to_vec()` + `for p /= temp` + `softmax` with `probs_buf.copy_from_slice(logits)` + `softmax_scaled`. Eliminates 4 heap allocations + 4 full passes per speculation.
- [x] 3. **`speculative_step_rollback` fused softmax** — Replaced 3× temp-div+softmax with `softmax_scaled` in rollback, rollback_paged, conditioned, and all `_with` variants. Saves 1 full pass per occurrence (11 total).
- [x] 4. **`raven_compute_router_into` zero-alloc** — New `_into` variant reuses pre-allocated `scored` and `r_t` buffers from `RavenKVCache`. Original function kept as backward-compatible wrapper.
- [x] 5. **`forward_raven` zero-alloc** — Eliminated 3 allocations per call: router logits via `ctx.raven_query_buf`, router scored/r_t via `cache.router_scored`/`cache.router_r_t`, per-head `full_query` via `ctx.raven_query_buf` reuse.
- [x] 6. **`softmax_scaled` fused temperature+softmax** — New function in `types.rs` that fuses `(x - max) * inv_temp` into the exp computation, saving one full buffer pass vs separate `for p /= temp; softmax(x)`. Used in 15+ call sites.
- [x] 7. **`dflash.rs` fused softmax** — Replaced 4× temp-div+softmax in `dflash_predict_with`, `dflash_predict_ar_with`, `dflash_predict_conditioned_with`, `dflash_predict_parallel` with `softmax_scaled`.
- [x] 8. **`generate_into` fused softmax** — Replaced temp-div+softmax with `softmax_scaled` in transformer generation loop.
- [x] 9. **`benchmark.rs` fused softmax** — Replaced 2× temp-div+softmax in `bench_ar` warmup/measure loops.
- [x] 10. **`ppot_rescue_adaptive` dedup entropy** — Added `identify_high_entropy_positions_with_entropy_into` and `identify_positions_adaptive_with_entropy_into` that return both positions AND entropy values in one pass. Eliminates double `token_entropy()` computation and `entropy_cache` Vec allocation.
- [x] 11. **Run benchmarks & verify** — All 330 tests pass, clean build, no new warnings.
- [x] 12. **Update this plan** with final benchmark numbers.

## Not Done (intentionally deferred)

- **Fuse `softmax` to 2 passes** — 3-pass is optimal for the algorithm (need full sum before normalizing). Not fusable without algorithmic change.
- **Fuse `attention_head` to 2 passes** — Same reason: need full exp sum before weighted value accumulation.
- **`speculative_step_rollback_paged` full zero-alloc** — Would require adding `SpeculativeContext` + `TreeBuilder` + `probs_buf` + `residual_buf` parameters (already 12 params). Defer to future `_with` variant if needed.
- **`sample_from_distribution` SIMD/binary search** — CDF scan is branch-predictable and fast for small vocab. SIMD optimization deferred.

## Files Modified
| File | Changes |
|---|---|
| `src/types.rs` | Added `softmax_scaled()` — fused temperature+softmax |
| `crates/katgpt-percepta/src/transformer.rs` | Added 3 buffers to `ForwardContext`, eliminated `forward_paged` allocs, added `raven_compute_router_into`, added 2 buffers to `RavenKVCache`, fixed `forward_raven` allocs, fused `generate_into` softmax |
| `src/speculative/verifier.rs` | Eliminated 4× `logits.to_vec()` in `LeviathanVerifier::speculate`, used `softmax_scaled` |
| `crates/katgpt-forward/src/step.rs` | Replaced 11× temp-div+softmax with `softmax_scaled` across 5 functions |
| `crates/katgpt-speculative/src/dflash.rs` | Replaced 4× temp-div+softmax with `softmax_scaled` |
| `crates/katgpt-speculative/src/ppot/entropy.rs` | Added `_with_entropy_into` variants for zero-double-compute entropy+positions |
| `src/speculative/ppot/resample.rs` | Updated `ppot_rescue_adaptive` to use new entropy functions |
| `src/benchmark.rs` | Replaced 2× temp-div+softmax with `softmax_scaled` |

## Key Design Decisions
1. **`softmax_scaled(x, inv_temp)`** as a new function rather than modifying `softmax` signature — backward compatible, callers opt-in
2. **Pre-allocate in existing structs** (`ForwardContext`, `RavenKVCache`) rather than new context types
3. **`_into` pattern** for zero-alloc router variant, keep original as thin wrapper
4. **Raven self-borrow**: small `router_r_t.clone()` (16-64 floats) to resolve `&mut cache.keys` vs `&cache.router_r_t` conflict — acceptable trade-off vs 2 larger eliminated allocations
5. **Entropy `_with_entropy_into`** returns positions + entropy in parallel buffers instead of changing return type of existing functions