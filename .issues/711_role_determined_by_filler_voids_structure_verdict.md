# Issue 711: a structure verdict is unreadable when `role = f(filler)`, and nothing says so

**Status:** T1–T3 LANDED upstream 2026-09-02 — `FillerRoleSpread` +
`filler_role_spread` + `role_determined_by_filler` ship in `katgpt_core::tpr`,
riding on `BowRouterReport::spread` / `ShuffledRoleReport::spread`, with
`verdict() -> Option<bool>` withholding an unreadable answer. 26 tpr lib tests
(was 23). **T4 alone is open** and is an owner call, not engineering. Found by
the same consumer that found Issue 710, one level up: not a wrong number — the
probes move and both `vacuous` flags stay false. What was void is the
*interpretation*, and nothing in the report carried it.

## The defect

Issue 710 made vacuity a reported quantity on four instruments, on the
principle that **a control's report must carry whether the control could have
failed**. Three vacuity modes are now covered:

| mode | detected by |
|---|---|
| the permutation is an identity | `ShuffledRoleReport::vacuous` / `moved` |
| the null IS the caller's fit | `BowRouterReport::vacuous` |
| a truth is absent from the pool | `candidate_pool_coverage` |

There is a fourth, and it is invisible to all of them: **every filler is seen
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

- [x] **T1** Ship the predicate: `role_determined_by_filler(bindings) -> bool`
  (every filler id appears with `<= 1` distinct role id) beside
  `role_shuffle_is_vacuous`. Cheapest correct step; a consumer must not have to
  derive it from `n_fillers == n_states`, which is only a proxy and is wrong
  whenever a filler legitimately repeats within one role.
- [x] **T2** Report the covariate, not only its threshold: max and mean
  distinct-roles-per-filler on `BowRouterReport` and `ShuffledRoleReport`, so a
  *near*-degenerate corpus (1.02 roles/filler) is visible too. The threshold
  alone would have called `kernel_opt` and a healthy corpus the same until the
  last filler tipped over.
- [x] **T3** Test with a planted corpus at both extremes — `planted_retrieval`
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

## Landed — what the upstream shape is, and the canary that proves it fires

`FillerRoleSpread { max, mean, fillers }` is measured by one
`sort_unstable + dedup` over the `(filler, role)` pairs — no hashing, one
allocation, and the deduped list length IS the numerator of `mean`, so the
covariate costs one pass and no second data structure. `fillers` counts the
fillers that **appear**, deliberately not `AlsInput::n_fillers`: an unused
filler id would drag the mean toward 0 and make a degenerate corpus read as
worse than degenerate.

The predicate is exposed three ways so no consumer has to re-derive it —
`FillerRoleSpread::role_determined_by_filler` (have the covariate),
`role_determined_by_filler(bindings)` (want only the answer), and
`{Bow,Shuffled}Report::verdict() -> Option<bool>` (want the gated verdict).
`structured` / `degraded` keep their raw `bool`, so **nothing breaks** — which
is what keeps T4 a separable decision rather than a consequence of T1–T3.

Measured, on the same planted corpus with only the roles rewritten as
`role = filler % m` (`the_composition_covariate…`, `a_structure_verdict_is_withheld…`):

| corpus | max | mean | `bow` ratio | shuffle ratio | `moved` | either `vacuous` | `verdict()` |
|---|---|---|---|---|---|---|---|
| healthy (roles drawn independently) | 6 | 6.000 | — | — | >0 | false | `Some(..)` |
| degenerate (`role = filler % m`) | 1 | 1.000 | 1.0109 | 0.9505 | 199 | **false** | **`None`** |

The degenerate row reproduces the Bench 063 §12.4 signature in a unit test:
both probes move, 199 role slots permuted, **neither vacuity flag fires**, and
the two numbers are still uninterpretable. That is the canary — a dead guard
would leave a confident `structured = false` exactly there.

Pinned **two-sided**, per the lesson that a guard which never fires and a
guard which always fires fail identically from the outside: the healthy arm
asserts `verdict() == Some(structured)`, so an always-on guard reds the suite
as loudly as a dead one. A third test pins the *near*-degenerate case
(one filler tipped to two roles): `max = 2` clears the threshold while
`mean < 1.2` still says how thin the structure is — the reason the covariate is
reported and not only its boolean.
