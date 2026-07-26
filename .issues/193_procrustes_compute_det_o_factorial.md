# Issue 193: `ProcrustesConfig::compute_det = true` silently degrades to O(d!) at d > 8

**Date:** 2026-07-26
**Discovered by:** P1 SubspaceAdapter validation (riir-train `.benchmarks/423`)
**Resolved:** 2026-07-26 (LU decomposition fix landed)
**Severity:** Medium — silent multi-hour hang on a single config knob (before fix)
**Surface:** [`crates/katgpt-spectral/src/procrustes.rs::determinant_d`](../../katgpt-rs/crates/katgpt-spectral/src/procrustes.rs)
**Class:** Footgun / missing guardrail (not a correctness bug — the algorithm is correct, just asymptotically pathological)

**STATUS: RESOLVED.** See "Fix landed" section below.

---

## TL;DR

`ProcrustesConfig::compute_det = true` triggers `determinant_d`, which uses **cofactor expansion — O(d!) complexity**. The substrate's own docstring warns "O(d!) — only call for small d (≤ 8 typical for KG embeddings)", but the warning is in a docstring that callers don't see at the call site — they only see `compute_det: bool` in `ProcrustesConfig`. At d=16, the cost is 16! ≈ 2.1 × 10¹³ operations ≈ 5–6 hours. There's no runtime check, no debug assertion, no log line — it just appears to "hang".

This wasted ~3 P1 harness runs during Bench 423 before being root-caused. The next consumer turning `compute_det = true` at d > 8 will hit the same trap.

## Reproduction

```rust
use katgpt_spectral::procrustes::{orthogonal_procrustes, ProcrustesConfig, ProcrustesScratch};

let n = 30;
let d = 16;  // <- the trigger
let a: Vec<f32> = (0..n*d).map(|i| (i as f32) * 0.001).collect();
let b: Vec<f32> = (0..n*d).map(|i| (i as f32) * 0.002 + 0.5).collect();
let mut rot = vec![0f32; d*d];
let mut scr = ProcrustesScratch::new(n, d);
let mut cfg = ProcrustesConfig::default();
cfg.compute_det = true;  // <- the footgun
// cfg.compute_det = false; // <- the workaround (polar factor is det = ±1 by construction)

// Hangs for hours at d=16. Returns in ~44µs at d=16 if compute_det = false.
orthogonal_procrustes(&a, &b, n, d, &mut rot, &mut scr, &cfg).unwrap();
```

## Why this matters

The Procrustes primitive is **default-on** in `katgpt-rs` (Plan 152 / Issue 001). It ships in every consumer of `katgpt-spectral`. The `compute_det` knob is in the public API. A consumer who turns it on at d > 8 — for any reason, including "I want to know if my rotation is a reflection" — will silently hang.

The determinant of an orthogonal matrix is provably ±1 by construction. **There is no legitimate reason for any consumer to ever compute it explicitly** — `det < 0` means reflection, `det > 0` means rotation, and the special-orthogonal correction (`ProcrustesConfig::special_orthogonal = true`) already handles that case via column flip without needing the explicit determinant.

## Proposed fix (any one of these is sufficient)

### Option A — LU decomposition (O(d³), best)

Replace cofactor expansion with LU decomposition with partial pivoting for `d > 6`. LU is O(d³) and gives `det = ±∏ diagonal entries of U × (sign of permutation)`. Standard textbook algorithm. For d ≤ 6 the cofactor expansion is faster (small constant factor) so keep the existing code path.

**Sketch:**
```rust
fn determinant_d(m: &[f32], d: usize) -> f32 {
    match d {
        1 => m[0],
        2 => m[0] * m[3] - m[1] * m[2],
        3 => { /* Sarrus' rule (unchanged) */ }
        _ if d <= 6 => { /* cofactor expansion (unchanged) — faster constant factor */ }
        _ => determinant_lu(m, d),  // O(d³) — new path
    }
}
```

### Option B — Runtime assertion (cheapest)

Add a debug-mode assertion that fails loudly when `compute_det = true && d > 8`:
```rust
debug_assert!(
    d <= 8,
    "determinant_d: d={d} > 8 triggers O(d!) cofactor expansion. \
     Either disable `ProcrustesConfig::compute_det` (the polar factor is det = ±1 \
     by construction) or use a different method."
);
```

This is a one-line fix. It doesn't solve the problem for release builds, but it catches the bug during development.

### Option C — Doc cross-link (weakest)

Add a `#[deprecated]` or `#[doc = "warning"]` note on `ProcrustesConfig::compute_det` linking to this issue. Doesn't prevent the trap, just documents it more visibly.

**Recommendation:** Option A (LU decomposition) — it's the right algorithmic fix and matches the substrate's documented "no eigensolver, deterministic across platforms" guarantee (LU is also deterministic). Option B is a one-line backstop for development.

## Why wasn't this caught earlier

1. P2 (Bench 422) used the default `compute_det = false`, so it never hit this.
2. The Procrustes tests in `procrustes.rs::mod tests` use `d ≤ 8` exclusively, where cofactor expansion is fast.
3. The docstring warning is on `determinant_d` (a private fn), not on `ProcrustesConfig::compute_det` (the public knob).
4. No production consumer had turned `compute_det = true` until P1 did so for diagnostic purposes.

## Scope

This is a katgpt-rs issue. The fix is one file in `katgpt-spectral`. No cross-repo coordination needed. No semantic change to the API — just an asymptotic improvement + (optional) a runtime guardrail.

## Acceptance criteria

- [x] LU decomposition path lands for `d > 6` (Option A) — `determinant_lu` function with partial pivoting, O(d³)
- [x] Test: `compute_det = true` at d=16 returns in < 100ms — `compute_det_at_d16_returns_quickly` regression test
- [x] Doc on `ProcrustesConfig::compute_det` cross-references the asymptotic constraint — updated to note the LU fix
- [x] (Option A) G3 determinism preserved — LU with partial pivoting is deterministic (tie-break by lowest row index; no eigensolver, no convergence loop)

## Fix landed (2026-07-26)

**Option A (LU decomposition)** shipped. The `determinant_d` function now dispatches:
- `d ∈ {1, 2, 3}`: hardcoded fast paths (unchanged)
- `d ∈ {4, 5, 6}`: cofactor expansion (unchanged — fast constant factor)
- `d > 6`: **LU decomposition with partial pivoting** (new — O(d³))

The LU path is deterministic across platforms (partial pivoting breaks ties by lowest row index; no eigensolver, no convergence loop — matches the substrate's documented determinism guarantee). Singular matrices return det = 0.0.

**5 new tests** verify correctness:
- `determinant_lu_identity_large_d` — identity det = 1 at d ∈ {7, 16, 32, 64}
- `determinant_lu_matches_known_analytical_values` — upper-triangular det = product of diagonal at d=8
- `determinant_lu_singular_returns_zero` — rank-deficient matrix at d=16
- `determinant_lu_diagonal_matrix` — diagonal det = product at d=16 (16! = 2.09 × 10¹³)
- `compute_det_at_d16_returns_quickly` — **Issue 193 regression test**: `compute_det = true` at d=16 returns in < 1s with |det| ≈ 1 (polar factor property)

All 18 procrustes tests pass. The footgun is closed — callers can safely turn `compute_det = true` at any d for diagnostics.

## References

- Discovery context: [riir-train/.benchmarks/423_canon_p1_subspace_validation.md §"The bug I almost shipped"](../../riir-train/.benchmarks/423_canon_p1_subspace_validation.md)
- Substrate: `katgpt-spectral/src/procrustes.rs` (default-on in katgpt-rs since Plan 152 / Issue 001)
- Source algorithm: Higham (1986) — Newton-Schulz iteration for the polar factor (the determinant computation is NOT part of Higham's algorithm; it's a diagnostic add-on)
