# Issue 669 — `FeatureHasher::hash_value` OOB read: test passed 3-element vectors to a `feature_dim=5` hasher (silent UB in release, SIGABRT in debug)

**Found:** 2026-08-18, during the katgpt-rs root-tests/ clippy sweep slice 6 (cycle 8
baseline — `bench_delta_mem_modelless` failed at BASELINE, pre-heal; the heal did not
touch the failing path).

**Status:** RESOLVED same day (fix in the slice-6 commit).

## Symptom

```
thread 'test_phase1_state_interference' panicked at crates/katgpt-types/src/simd/dot.rs:149:45:
unsafe precondition(s) violated: slice::get_unchecked requires that the index is within the slice
thread caused non-unwinding panic. aborting. (signal: 6, SIGABRT)
```

The abort kills the whole test binary — the other 15 tests in `bench_delta_mem_modelless`
never ran. `bench_delta_mem_modelless` sits behind the `delta_mem` feature, which is not
in any pinned CI suite (the gate-coverage lesson), so this never fired in CI.

## Root cause (two halves)

1. **Test bug (the trigger):** `tests/bench_delta_mem_modelless.rs` line 114 constructs
   ONE hasher — `FeatureHasher::new(8, 5, 42)` (feature_dim=5) — and uses it for BOTH
   keys and values. Keys are 5-element ✓; values are 3-element ✗ (lines 118 + 126:
   `hash_value(&[0.0, 1.0, 0.0])`). Production uses TWO hashers
   (`katgpt-pruners/src/delta_mem/pruner.rs:99-100`: key dim=feature_dim, val dim=3
   because `OutcomeFeatures` is 3-dim) — the test author followed the value convention
   against the single dim=5 hasher.
2. **Missing guard (the blast radius):** `FeatureHasher::project_into`
   (`crates/katgpt-core/src/delta_mem/hash.rs`) asserted `out.len() == rank` but NOT
   `features.len() == feature_dim`. The row length comes from the projection layout
   (`chunks_exact(fd)`), so `simd_dot_f32(row, features, fd)` read `features[fd..]`
   past the slice — **silent UB in release** (garbage floats folded into the hash),
   debug-precondition abort in debug.

## Fix

- `tests/bench_delta_mem_modelless.rs`: pad `val_a`/`val_b` to 5 elements
  (`[0.0, 1.0, 0.0, 0.0, 0.0]` / `[1.0, 0.0, 0.0, 0.0, 0.0]`). Semantics: the test
  asserts interference patterns, not specific hash values; zero-padding is a valid
  feature vector of the correct dimension.
- `crates/katgpt-core/src/delta_mem/hash.rs` `project_into`: added
  `assert_eq!(features.len(), self.feature_dim, ...)` mirroring the existing
  `out.len() == rank` assert — an OOB read is otherwise silent UB in release.

## Validation

- `bench_delta_mem_modelless`: 16/16 (was: abort at test 1)
- `katgpt-core --lib delta_mem`: 47/47
- `katgpt-core --lib` default: 1893 passed / 0 failed / 6 ignored
- `katgpt-pruners --features delta_mem --lib`: 217/217

## Lessons

- The OOB lived at the **API seam** (caller-supplied `len` vs slice lengths in
  `simd_dot_f32`) but the right guard is at the **typed boundary** (`project_into`),
  where the intended dimension is known. `simd_dot_f32` itself cannot distinguish
  caller bugs from intent — see the sibling `is_finite` screen debate in
  riir-chain Issue 026 (guard where the value becomes semantic, not at the wire).
- Release-mode "passing" tests that read OOB are evidence of nothing: this test
  passed its assertions for its whole life while folding 2 stack-garbage floats into
  every `hash_value` call.
- Feature-gated test binaries with no pinned CI suite rot silently — this abort could
  have landed any time and nothing would have flagged it.
