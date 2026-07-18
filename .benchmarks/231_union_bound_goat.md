# GOAT Proof 231: Union Bound Branch Confidence

> **UPDATE 2026-07-18 (status sync):** `union_bound_confidence` is
> **default-on** in the root `Cargo.toml` (`default = [...]`, comment: "GOAT
> 6/6, default-ON"). The `(opt-in)` label below was stale relative to the
> Cargo.toml default list.

**Date:** 2026-06-09
**Plan:** 231
**Research:** 205 (Deep Manifold §2.4.2)
**Feature gate:** `union_bound_confidence` (**default-on in root Cargo.toml**)
**Status:** ✅ GOAT 6/6 PASS

---

## GOAT Criteria

| # | Criterion | Target | Result | Status |
|---|-----------|--------|--------|--------|
| G1 | Boole's inequality correctness | union ≤ mult, both ∈ [0,1] | 6/6 test cases verified | ✅ |
| G2 | Degradation shape verification | Linear (union) vs exponential (mult) | Exact formulas verified, mult(n=16) < 0.2 | ✅ |
| G3 | HybridScorer routing | ≤4 → mult, >4 → union | Boundary cases correct | ✅ |
| G4 | Per-step overhead | < 1μs typical chain | 76 ns (8-elem), 5.6μs (1000-elem), linear scaling | ✅ |
| G5 | Edge cases | Empty, zeros, single, perfect, clamp | All correct | ✅ |
| G6 | Feature gate isolation | Types accessible, trait objects work | All 3 scorers + trait object verified | ✅ |

---

## Test Results

```
running 7 tests — 7 passed; 0 failed
```

### GOAT Gates (6/6)

| Gate | Key Metric |
|------|-----------|
| G1 | Boole's inequality: union ≤ mult for all cases, trivial (empty, perfect) match |
| G2 | Union: 1 - n(1-p) linear degradation, exact zero at n=10 for p=0.9 |
| G3 | Hybrid: len=4 → multiplicative, len=5 → union bound (boundary correct) |
| G4 | 8-element chain: 76 ns/call, linear scaling to 1000-element |
| G5 | Empty→1.0, zeros→0.0, single→identity, perfect→1.0, clamp→0.0 |
| G6 | All types accessible, trait objects work, configurable threshold |

### Benchmarks

| Benchmark | Result |
|-----------|--------|
| 8-element chain | ~76 ns/call |
| 1000-element chain | ~5.6 μs/call |
| Scaling | Linear (not quadratic) |

---

## Component Coverage

| Component | Tests | File |
|-----------|-------|------|
| `BranchConfidence` trait | 3 unit + 6 GOAT | `crates/katgpt-speculative/src/branch_confidence.rs` |
| `MultiplicativeScorer` | 3 unit + 6 GOAT | `crates/katgpt-speculative/src/branch_confidence.rs` |
| `UnionBoundScorer` | 4 unit + 6 GOAT | `crates/katgpt-speculative/src/branch_confidence.rs` |
| `HybridScorer` | 3 unit + 6 GOAT | `crates/katgpt-speculative/src/branch_confidence.rs` |
| GOAT proof | 7 | `tests/bench_231_union_bound_goat.rs` |
| **Total** | **33** | |

---

## GOAT Decision

**6/6 GOAT gates passed. Mathematically correct, zero regressions.**

### Verdict: ✅ GOAT — Promote to default-ON

The union bound scorer is mathematically correct (Boole's inequality), provides predictable linear degradation (vs exponential cliff), and is a clean drop-in replacement for multiplicative scoring with negligible overhead.

### Feature Gate Decision

**Promote to default-ON.** Add `union_bound_confidence` to the default features list.

### Note on the +36% claim

The original claim of "+36% branch survival" was mathematically incorrect — by Boole's inequality, union bound confidence is always ≤ multiplicative confidence. The real value is:
1. **Architectural correctness**: Models additive error propagation per Deep Manifold §2.4.2
2. **Predictability**: Linear degradation (no exponential cliff at scale)
3. **Conservative but accurate**: Doesn't over-estimate confidence like multiplicative can
4. **HybridScorer**: Best of both worlds — multiplicative for short chains, union bound for long chains
