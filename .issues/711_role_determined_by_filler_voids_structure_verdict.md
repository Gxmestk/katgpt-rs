# Issue 711: a structure verdict is unreadable when `role = f(filler)`, and nothing says so

**Status:** OPEN, found by the same consumer that found Issue 710, one level
up. Not a wrong number — the probes move and both `vacuous` flags stay false.
What is void is the *interpretation*, and nothing in the report carries that.

## The defect

Issue 710 made vacuity a reported quantity on four instruments, on the
principle that **a control's report must carry whether the control could have
failed**. Two vacuity modes are now covered:

| mode | detected by |
|---|---|
| the permutation is an identity | `ShuffledRoleReport::vacuous` / `moved` |
| the null IS the caller's fit | `BowRouterReport::vacuous` |
| a truth is absent from the pool | `candidate_pool_coverage` |

There is a third, and it is invisible to all of them: **every filler is seen
with exactly one role**, so the role is a deterministic function of the filler
and adds nothing to what the filler already determines.

In that regime the probes still *move* — the `m`-role model has `m` core
blocks against the null's 1, so it has more capacity and will generally fit
differently. Neither `vacuous` flag fires. But TPR's claim is **systematicity**
— generalizing to unseen `(role, filler)` pairs — and when `role = f(filler)`
there are **no unseen pairs to generalize to**. The corpus is outside the
primitive's domain of applicability, not inside it and unstructured. A
`structured = false` there is unreadable, and so is a `structured = true`.

`withheld_pair_top1` is affected the hardest: withholding a pair withholds the
filler entirely, so its OOD arm is not a harder version of the ID arm — it is a
different question.

## Measured

riir-clippy Bench 063 §12.4, the `kernel_opt` arm (445 GPU-optimization rules,
each belonging to exactly one declared `KernelCategory`, so `n_fillers ==
n_states == 445`):

| arm | states/filler | `bow_router` | cross-state control | reported |
|---|---|---|---|---|
| `kernel_opt` | **1.00** | 0.9307 (structured=false) | 1.0147 (degraded=false) | `structured = false` |
| `clippy_lints` | 2.51 | 1.1417 (true) | 1.0693 (true) | `structured = true` |
| `rustc_errors` | 26.50 | 2.2096 (true) | 2.6661 (true) | `structured = true` |

Both probes are **monotone in states-per-filler across all three arms**, which
is the shape to worry about: it is what a capacity effect looks like, and the
degenerate arm sits at exactly 1.00 where the structure question is not even
posed. Read without the covariate, `kernel_opt` reads as "the largest corpus we
have carries no binding structure" — a corpus finding — when it is a statement
about the scheme's applicability to it.

## The fix

- [ ] **T1** Ship the predicate: `role_determined_by_filler(bindings) -> bool`
  (every filler id appears with `<= 1` distinct role id) beside
  `role_shuffle_is_vacuous`. Cheapest correct step; a consumer must not have to
  derive it from `n_fillers == n_states`, which is only a proxy and is wrong
  whenever a filler legitimately repeats within one role.
- [ ] **T2** Report the covariate, not only its threshold: max and mean
  distinct-roles-per-filler on `BowRouterReport` and `ShuffledRoleReport`, so a
  *near*-degenerate corpus (1.02 roles/filler) is visible too. The threshold
  alone would have called `kernel_opt` and a healthy corpus the same until the
  last filler tipped over.
- [ ] **T3** Test with a planted corpus at both extremes — `planted_retrieval`
  already produces the healthy case (roles drawn independently of the filler);
  the degenerate case is the same fixture with `role = filler % m`.
- [-] **T4** Decide whether `withheld_pair_top1` should REFUSE rather than
  report on such a corpus. Refusing is the honest answer and is also a breaking
  change to a gate the Issue 707 GOAT depends on. **Unblock:** owner call on
  whether G8 keeps a number it cannot interpret. Not urgent — G8's corpus is
  planted and healthy; this is about future consumers.

## The generalizable lesson, restated

Issue 710's rule was "a control's report should carry whether the control could
have failed." This case sharpens it: **a control can be perfectly capable of
failing and still be measuring the wrong thing.** `moved = 383/445` on the
`kernel_opt` cross-state arm — the control worked, permuted most of the corpus,
and reported a real number. The number just does not answer the question the
consumer is asking it. Reporting the covariate the interpretation depends on is
the general form; `vacuous` was the special case.

**Cross-ref:** consumer-side implementation shipped first as
`riir-clippy/src/draft/tpr_fit.rs::FitCorpus::role_is_a_function_of_filler`,
with `SpaceReport::structured()` returning `None` on it. Port from there rather
than re-deriving, as Issue 710 T2 did.
