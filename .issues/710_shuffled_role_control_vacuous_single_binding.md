# Issue 710: `shuffled_role_control` cannot fail on a single-binding corpus

**Status:** OPEN, found by the primitive's first external consumer
(riir-clippy Bench 063, 2026-09-02). Not a wrong answer — a **control that
cannot report a negative**, which is worse, because its pass is silent.

## The defect

`crates/katgpt-core/src/tpr/validate.rs::shuffled_role_control` permutes each
state's role list **within that state**:

```rust
let mut roles = b.roles.clone();
let n = roles.len();
for i in (1..n).rev() {
    let j = (next() % (i as u64 + 1)) as usize;
    roles.swap(i, j);
}
```

At `n == 1` the loop range `(1..1)` is empty. The "shuffled" bindings are
**bit-identical** to the input, both fits see the same data, and the report
comes back

```
r_true == r_shuffled  (exactly)   ratio == 1.0000   degraded == false
```

which is **indistinguishable from the real negative result** the control exists
to detect ("the specific role assignment is not load-bearing").

Its own doc comment states the contract it then cannot honour: *"A fit that
scores the same on shuffled roles was never using them, whatever its residual
says."* On a single-binding corpus every fit scores the same, so the control
condemns every artifact it is handed.

## Why single-binding corpora are the common case, not a corner

The Issue 707 tests exercise multi-binding states (planted structures with
`m` constituents per state), which is why this never fired in-repo. But a
**retrieval** consumer binds one role to one filler per item: one span carries
one rule and one shape-class. That is riir-clippy Issue 062's entire corpus
(113 clippy + 212 rustc states, every one of them `len() == 1`), and it is the
shape of any (key, value) retrieval index. The first real consumer hit it
immediately.

## Measured

riir-clippy Bench 063 §7.4, `clippy_lints/tokenbag256`, `d=1`:

| control | ratio | degraded | reading |
|---|---|---|---|
| `shuffled_role_control` (upstream) | 1.0000 | false | "roles carry nothing" — **wrong** |
| `bow_router` (upstream) | 1.1773 | structured=true | "roles carry something" |
| `cross_state_role_control` (consumer-local) | 1.0812 | true | "the pairing is load-bearing" |

The upstream control contradicted the upstream router on the same artifact.
Read naively, that reads as a defect in the consumer's wiring — it took
inspecting the permutation loop to see the control was the identity map.

## The fix that actually applies

Permute the role assignment **across** states (fillers untouched), which is
what "are the roles load-bearing?" means when no state has two bindings to
swap. Implemented consumer-side as
`riir-clippy/src/draft/tpr_fit.rs::cross_state_role_control` — deterministic
Fisher-Yates over the flattened role vector, same `SplitMix64` generator shape,
plus a `moved` count so an identity draw cannot masquerade as a no-structure
verdict either.

## Tasks

- [ ] **T1** Make the vacuity **impossible to read as a verdict**: return an
  explicit vacuity signal from `shuffled_role_control` (e.g. a
  `vacuous: bool` on `ShuffledRoleReport`, set when
  `bindings.iter().all(|b| b.len() <= 1)`) rather than a `degraded: false` that
  looks like data. A consumer must not have to read the permutation loop to
  learn the control did nothing. Cheapest correct fix; do this one first.
- [ ] **T2** Ship the cross-state permutation upstream as the single-binding
  arm — either a second fn or a mode on the existing one — so every consumer
  gets a control that can fail. Port from the riir-clippy implementation
  referenced above rather than re-deriving it.
- [ ] **T3** Add a single-binding case to `tpr/tests.rs`. Every existing
  role-control test uses multi-binding states, which is exactly why this
  survived the Issue 707 GOAT gate (G8 included `shuffled_role_control` and it
  passed — on multi-binding synthetic data).
- [-] **T4** Audit the other validators for the same shape.
  `AtomicNull`/`withheld_pair_top1` were already designed against vacuity
  (`AtomicNull::coverage` exists precisely as the vacuity check, and Issue 062
  Phase 1 found a vacuous null with it), and `bow_router` is unaffected (it
  changes `m`, not a within-state order). **Unblock:** owner pass — the two
  known-good cases suggest the discipline was applied unevenly rather than
  absent, so this is a review, not a build.

## The generalizable lesson

`AtomicNull` ships a `coverage()` method **whose entire purpose** is to detect
that the null is vacuous, and its doc says *"a null at 0% ID is vacuous and its
OOD 0% certifies nothing."* The same discipline was not applied one function
down to the role control, which has the identical failure mode and no
equivalent method. A control's report should carry whether the control could
have failed — otherwise the consumer cannot tell a pass from a no-op.
