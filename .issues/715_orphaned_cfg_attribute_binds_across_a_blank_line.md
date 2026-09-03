# Issue 715 — a `#[cfg]` separated from its item by a blank line still applies to it

**Status:** **CLOSED, fixed + gated 2026-09-03.** Bug fixed in `a08376a0`;
the class is gated at zero by `scripts/orphaned_attr_gate.py`, a
`docs_gate.sh` check (now 6/6). Recorded rather than removed because the
*measurement* — 2,044 candidate sites narrowing to 0 — is the reusable part.

## The defect

Rust binds an attribute to the next item **across a blank line**. So this
compiles, and makes the import debug-only:

```rust
#[cfg(debug_assertions)]

use crate::absorb_compress::{AbsorbCompress, AbsorbCompressLayer};
```

`crates/katgpt-pruners/src/sdar/sdar_absorb.rs:47` was in exactly that state
for two days. In every build with `debug_assertions` OFF — i.e. **every release
build** — the import vanished while its five usages stayed unconditional (the
`inner` struct field, `new()`'s signature, `inner()`, `inner_mut()`, and the
`impl AbsorbCompress`). Result: 5 × `E0425` and
`could not compile katgpt-pruners (lib)`.

`sdar_gate = ["bandit"]` guarantees `absorb_compress` exists whenever this
module compiles, so the attribute was simply wrong. Removed.

## Provenance, exactly

| commit | what it did |
|---|---|
| `26d055c6` | dropped `use std::cmp::Ordering;` — **what that attribute correctly applied to** — and left the attribute behind, where it silently re-bound to the next import |
| `7e34ccef` | deleted the blank line, making the wrong binding look deliberate |
| `a08376a0` | removed the attribute; verified in **both** profiles |

**Why it survived two days.** `26d055c6`'s own validation line reads
*"katgpt-pruners lib 597/0 under g_zero,ropd_rubric,sdar_gate,go"* — a **debug**
run, where `debug_assertions` is on and the import exists. That is
`.issues/713` T2b's lesson with the sign flipped: there, debug **manufactured**
four false perf reds; here, debug **hid** a real build break. Neither profile is
the safe default. The profile is part of the claim.

`7e34ccef` is the more interesting one: it did not cause the bug, it **erased
the evidence**. A future reader sees an attribute attached to an import and has
no reason to doubt it.

## Blast radius: audited, not assumed

`26d055c6` dropped **seven** imports. `git show 26d055c6 -U4` over every
removal site: only this one had an attribute above it. The other six are clean.

## The gate, and why the narrowing IS the instrument

`scripts/orphaned_attr_gate.py`, pinned at **0**, no floor to negotiate.

| shape | sites, 19 repos | gateable? |
|---|---|---|
| **any** attribute + blank line + item | **2,044** | no |
| **outer `#[cfg]`/`#[cfg_attr]`** + blank line + item | **0** | **yes** |

The naive shape is dominated by whole-file **inner** attributes
(`#![cfg(...)]`), which bind to the *enclosing module* rather than to the next
item, and which are conventionally followed by a blank line. Restricting to
outer `cfg` attributes is what takes 2,044 to 0. A first measurement that
reported 2,044 would have been filed as "too common to gate" — and the
narrowing was found by reading the sample, not by reasoning about the regex.

Also of note: the broader 2,044 shape is *not* a defect and must not be
reported as one. A gate that flags it would be ignored within a day.

### Canaried against the real bug, not a synthetic one

The pre-`a08376a0` state was reconstructed in the working tree; the gate
reported the offender at the exact line (`sdar_absorb.rs:47`), exit 1, and
returned to green on restore. `selftest()` additionally pins five negatives —
the inner-attribute case being the single one worth naming, since it is the
whole difference between 2,044 and 0 — and asserts that the walk **prunes**
`target/` rather than filtering after `rglob` (the documented 556 s trap, and
the `find -not -path` trap one level over).

## How it was found

Not by looking for it. Measuring `.issues/713` **T6** (targets that print a
green `0 passed` because every test is `#[ignore]`d) required *executing* the
exemplar `test_120_vpd_arena_goat` under its own features to confirm its
`3 ignored` output — and it would not compile. The T6 measurement is in
`.issues/713`.

Worth stating plainly: this was reachable only because the T6 work insisted on
running a target rather than reading it. Two of the three load-bearing T6 rows
disagreed with my auditor's count on first measurement, and chasing those
disagreements is what put a release build of `sdar_gate` on the command line.

## Related

- `.issues/713` — the parent measurement (T6), and T2b's debug/release lesson
  in the opposite direction.
- `.issues/709` — shared-worktree sweeps. `26d055c6`'s change was correct in
  itself; the defect was the residue it left.
