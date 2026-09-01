# Issue 700 — two red gates on `develop`, both pre-existing

Status: OPEN — found in passing during Bench 695 (SIMD soundness guard), NOT
caused by it. Both reproduced at clean `HEAD` in a detached worktree before
being reported. Neither is owned by this session; filed so they are not
rediscovered as "your diff broke it".

## R1 — `cargo test --workspace --lib` does not compile

```
error[E0425]: cannot find value `backend` in this scope   (×2)
error: could not compile `katgpt-backend` (lib test) due to 2 previous errors
```

**Only under `--workspace`.** `cargo test -p katgpt-backend --lib` compiles
cleanly on its own. The break is a **feature-unification** artifact: building the
whole workspace together activates a feature combination that leaves `backend`
undefined in a cfg'd test path. This is the documented feature-variant trap —
the invocation *is* the claim, and a per-crate green says nothing about the
unified build.

Consequence: the natural whole-repo gate is unavailable, so no-regression
evidence has to be assembled per crate (Bench 695 G3 did exactly that, 21 crates
/ 3304 tests). Worth fixing precisely because its absence silently degrades every
future validation.

Repro: `cargo test --workspace --lib --no-run` at HEAD.

## R2 — `transformer::tests::test_cluster_map_from_embeddings_collapses_identical_rows` fails

```
src/transformer/tests.rs:2389
assertion `left == right` failed: identical rows cannot be separated
  left: 4, right: 1
```

Four clusters produced where identical embedding rows must collapse to one. This
is a behavior failure, not a compile error, and it is in the default `--lib`
suite (`200 passed; 1 failed`), i.e. the default gate is red on `develop`.

Not investigated here beyond confirming it is pre-existing (identical failure at
clean HEAD). Whether the bug is in the clustering threshold or in the test's
premise is unknown — a fix should establish which before changing either.

Repro: `cargo test --lib test_cluster_map_from_embeddings_collapses_identical_rows`.

## Closing conditions

- [ ] R1: `cargo test --workspace --lib` compiles; note the feature that caused
      the unification break so the class is recognisable next time.
- [ ] R2: identical rows collapse to one cluster, or the test's premise is
      corrected with a recorded rationale.
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `.benchmarks/695_simd_len_guard_goat.md` §G3 (where both surfaced).
