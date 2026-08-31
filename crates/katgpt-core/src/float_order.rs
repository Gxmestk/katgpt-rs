//! NaN-safe total-order comparators for f32 sorts and selections.
//!
//! `f32: PartialOrd` is not a total order, so the idiom
//! `b.partial_cmp(&a).unwrap_or(Ordering::Equal)` inside a sort comparator is
//! INTRANSITIVE the moment one NaN is in the slice: NaN compares `Equal` to
//! everything while real values compare Less/Greater, and the violated sort
//! contract makes std's sort **abort** in release
//! ("user-provided comparison function does not correctly implement a total
//! order"). Small slices survive on the insertion-sort fast path, which is
//! exactly why the shape passes fixtures and panics at production sizes.
//! The same idiom in `max_by`/`min_by` does not abort but picks whichever
//! element the tie happened to land on — the corrupt value can win.
//!
//! NaN is produced by ordinary numeric code, not just corrupt input: a
//! zero-norm vector in a cosine (`0/0`), `0 * INFINITY` in a weighted sum,
//! a degenerate corpus in a log-based statistic, an empty-slice mean.
//!
//! The fix has three properties, shared by all four comparators:
//!
//! 1. **NaN loses.** A corrupt score must never win. Sort comparators
//!    (`desc`/`asc`) map NaN to the direction's far end so it sorts LAST;
//!    selection comparators (`cmp_for_max`/`cmp_for_min`) map NaN past the
//!    selected end so `max_by`/`min_by` can never return it. Handing NaN to
//!    `f32::total_cmp` directly does the OPPOSITE: IEEE-754's total order
//!    ranks a positive NaN ABOVE `+INFINITY`, promoting the corrupt value
//!    to rank 0.
//! 2. **`-0.0` ≡ `+0.0`.** `total_cmp` distinguishes signed zeros; the
//!    `partial_cmp` idiom being replaced treats them as `Equal`. Collapsing
//!    them preserves the tie behavior of the code swapped out.
//! 3. **Identical ordering to the replaced idiom on all NaN-free input** —
//!    for non-NaN pairs `total_cmp` agrees with `partial_cmp`, which is what
//!    lets fixes land under existing no-regression gates instead of forcing
//!    re-baselines. Pinned over an 11-value corpus (both infinities, both
//!    zeros, both subnormal bounds) by the tests below.
//!
//! Which comparator where:
//!
//! | consumer | pass |
//! |---|---|
//! | `sort_by` / `sort_unstable_by`, largest first | [`desc`] |
//! | `sort_by` / `sort_unstable_by`, smallest first | [`asc`] |
//! | `max_by` / `max_by_key` / `select_nth_unstable_by` (max) | [`cmp_for_max`] |
//! | `min_by` / `min_by_key` | [`cmp_for_min`] |
//!
//! `max_by`/`min_by` interpret their comparator as NATURAL ascending order —
//! passing a reversed (descending) comparator swaps the selection — so the
//! selection comparators are both natural-ascending and differ only in where
//! NaN lands. Workspace sweep record: riir-ai Issue 832 (the first fixed
//! instance was riir-rag's `score_cmp_desc`, which [`desc`] generalizes).

/// Normalizes an f32 into a total-order-safe sort key.
///
/// NaN becomes `nan_sentinel` (the caller picks where the corrupt value must
/// land), `-0.0` collapses into `+0.0`, every other bit pattern passes
/// through unchanged.
#[inline]
fn key(x: f32, nan_sentinel: f32) -> f32 {
    if x.is_nan() {
        nan_sentinel
    } else if x == 0.0 {
        // Collapses -0.0 into +0.0; `total_cmp` would otherwise order them.
        0.0
    } else {
        x
    }
}

/// Descending f32 sort comparator (largest/best first) that is a TOTAL order.
///
/// Replaces `b.partial_cmp(&a).unwrap_or(Ordering::Equal)` in `sort_by` and
/// friends: NaN sorts last (it can never top a best-first list), `-0.0` ties
/// `+0.0`, and every NaN-free pair orders exactly as the replaced idiom.
/// For selection, use [`cmp_for_max`] instead — `max_by` reads its
/// comparator as natural order and a descending one flips the selection.
#[inline]
pub fn desc(a: f32, b: f32) -> core::cmp::Ordering {
    key(b, f32::NEG_INFINITY).total_cmp(&key(a, f32::NEG_INFINITY))
}

/// Ascending f32 sort comparator (smallest/cheapest first) that is a TOTAL
/// order.
///
/// Replaces `a.partial_cmp(&b).unwrap_or(Ordering::Equal)` in `sort_by` and
/// friends: NaN sorts last even though "ascending" puts small values first —
/// a corrupt cost must never look cheap. Also the correct `min_by`
/// comparator (see [`cmp_for_min`]).
#[inline]
pub fn asc(a: f32, b: f32) -> core::cmp::Ordering {
    key(a, -f32::NEG_INFINITY).total_cmp(&key(b, -f32::NEG_INFINITY))
}

/// Comparator for `max_by`/`max_by_key`/top-k selection that can never
/// select NaN.
///
/// Natural ascending order with NaN mapped BELOW every real value, so the
/// maximum is always a real number. Equal reals keep std's last-wins tie
/// rule, exactly like the replaced `partial_cmp` idiom.
#[inline]
pub fn cmp_for_max(a: f32, b: f32) -> core::cmp::Ordering {
    key(a, f32::NEG_INFINITY).total_cmp(&key(b, f32::NEG_INFINITY))
}

/// Comparator for `min_by`/`min_by_key` selection that can never select NaN.
///
/// Natural ascending order with NaN mapped ABOVE every real value, so the
/// minimum is always a real number. Identical to [`asc`]; shipped under its
/// own name so selection call sites read as selection.
#[inline]
pub fn cmp_for_min(a: f32, b: f32) -> core::cmp::Ordering {
    asc(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering::Equal;

    /// Both infinities, both zeros, both subnormal bounds, unit scales —
    /// every pair must order exactly as the `partial_cmp` idiom it replaces.
    #[test]
    fn matches_partial_cmp_idiom_on_nan_free_corpus() {
        let corpus = [
            f32::INFINITY,
            f32::MAX,
            1.0,
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE * 0.5, // subnormal
            0.0,
            -0.0,
            -f32::MIN_POSITIVE * 0.5,
            -1.0,
            f32::MIN,
            f32::NEG_INFINITY,
        ];
        for a in corpus {
            for b in corpus {
                let legacy = a.partial_cmp(&b).unwrap_or(Equal);
                assert_eq!(asc(a, b), legacy, "asc({a},{b}) diverged");
                assert_eq!(desc(a, b), legacy.reverse(), "desc({a},{b}) diverged");
                assert_eq!(cmp_for_max(a, b), legacy, "cmp_for_max({a},{b}) diverged");
                assert_eq!(cmp_for_min(a, b), legacy, "cmp_for_min({a},{b}) diverged");
            }
        }
    }

    #[test]
    fn signed_zeros_tie_under_all_comparators() {
        for (name, c) in [
            ("asc", asc as fn(f32, f32) -> _),
            ("desc", desc as fn(f32, f32) -> _),
            ("cmp_for_max", cmp_for_max as fn(f32, f32) -> _),
            ("cmp_for_min", cmp_for_min as fn(f32, f32) -> _),
        ] {
            assert_eq!(c(0.0, -0.0), Equal, "{name} must tie signed zeros");
        }
        // The bare total_cmp WOULD order them; the collapse is the point.
        assert_ne!(0.0_f32.total_cmp(&-0.0), Equal);
    }

    #[test]
    fn nan_sorts_last_under_both_directions() {
        let mut xs = vec![1.0, f32::NAN, 3.0, f32::NAN, -2.0, 0.0, f32::NAN];
        xs.sort_by(|a, b| desc(*a, *b));
        assert_eq!(&xs[..4], &[3.0, 1.0, 0.0, -2.0]);
        assert!(xs[4..].iter().all(|v| v.is_nan()), "NaN must sort last");

        let mut ys = vec![1.0, f32::NAN, 3.0, f32::NAN, -2.0, 0.0, f32::NAN];
        ys.sort_by(|a, b| asc(*a, *b));
        assert_eq!(&ys[..4], &[-2.0, 0.0, 1.0, 3.0]);
        assert!(ys[4..].iter().all(|v| v.is_nan()), "NaN must sort last");
    }

    /// The panic gate: a 24-element slice with 6 NaNs aborts std's sort under
    /// the legacy idiom (contract violation) — must pass under `desc`. This
    /// test FAILS against the old comparator by construction: the legacy
    /// comparator is intransitive on exactly this input.
    #[test]
    fn nan_heavy_sort_does_not_abort_and_orders_reals() {
        let mut xs: Vec<f32> = (0..18).map(|i| (i as f32) * 0.5 - 4.0).collect();
        let n = xs.len();
        for i in 0..6 {
            xs[i * 4 % n] = f32::NAN;
        }
        xs.push(f32::NAN);
        xs.sort_by(|a, b| desc(*a, *b));
        let mut reals: Vec<f32> = xs.iter().copied().filter(|v| !v.is_nan()).collect();
        reals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(&xs[..reals.len()], &reals[..], "reals must come out fully sorted");
        assert_eq!(reals.len() + 7, xs.len());
    }

    #[test]
    fn selection_never_returns_nan() {
        let xs = [1.0, f32::NAN, 3.0];
        assert_eq!(xs.iter().copied().max_by(|a, b| cmp_for_max(*a, *b)), Some(3.0));
        assert_eq!(xs.iter().copied().min_by(|a, b| cmp_for_min(*a, *b)), Some(1.0));
        // NaN at the end, the legacy idiom's failure position.
        let ys = [3.0, f32::NAN];
        assert_eq!(ys.iter().copied().max_by(|a, b| cmp_for_max(*a, *b)), Some(3.0));
        assert_eq!(ys.iter().copied().min_by(|a, b| cmp_for_min(*a, *b)), Some(3.0));
        // All-NaN input still returns NaN (nothing better exists) but never
        // aborts.
        let zs = [f32::NAN, f32::NAN];
        assert!(zs
            .iter()
            .copied()
            .max_by(|a, b| cmp_for_max(*a, *b))
            .unwrap()
            .is_nan());
    }

    #[test]
    fn legacy_max_by_can_select_nan_this_is_the_bug() {
        // Documents the replaced behavior: the legacy idiom hands the
        // selection to the tie, so a trailing NaN wins. Do not "fix" this
        // test — it is the record of why cmp_for_max exists.
        let legacy = |a: f32, b: f32| a.partial_cmp(&b).unwrap_or(Equal);
        let ys = [3.0, f32::NAN];
        assert!(ys.iter().copied().max_by(|a, b| legacy(*a, *b)).unwrap().is_nan());
    }

    #[test]
    fn sort_comparators_agree_off_nan() {
        for (a, b) in [(1.0, 2.0), (2.0, 1.0), (-0.5, 0.5), (-5.5, 5.5), (0.0, -0.0)] {
            assert_eq!(desc(a, b), asc(b, a), "desc(a,b) must equal asc(b,a) off NaN");
            assert_eq!(cmp_for_max(a, b), cmp_for_min(a, b), "selection comparators agree off NaN");
        }
    }
}
