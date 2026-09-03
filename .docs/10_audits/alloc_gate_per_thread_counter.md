# Alloc gates counted their SIBLING tests' allocations (Issue 714, closed 2026-09-03, file removed 2026-09-03)

Status: **historical record, CLOSED.** T1 measured + root-caused, T2 landed,
T3 ran over all 37 consumers (35 pass / 0 fail / 2 `fn main()` non-results).
Recover the narrative with `git log --all -- '.issues/714_*.md'` (last revision
`36a07567`). **Code:** `crates/katgpt-core/tests/common/mod.rs` (`ThreadCounter`,
`assert_counter_is_live`), wired first into
`tests/plan414_hla_committed_belief_probe_goat.rs`.

**Commits:** `711ea571` (filed) · **`62911111`** (T2 code — landed inside a
sibling session's doc-sync commit, see below) · `48b9740e` (two-sided
verification record) · `05ee3dd5` (T3 sweep).

## The defect

`counting_allocator!()` installed a `#[global_allocator]` incrementing a
**process-global** `ALLOC_COUNT`. `cargo test` runs a binary's tests on
parallel threads by default, so every allocation a *sibling* test made inside a
gate's `after - before` window landed in the gate's count.

| `plan414 … g4_zero_alloc` | result |
|---|---|
| default (parallel) | **FAILED — 6 allocs in 1000 calls, expected 0** |
| `-- --test-threads=1` | ok, 3 of 3 |
| pre-fix, release, 3 runs | 8 / 3 / 3 — a *varying* count is a concurrent contributor's signature |

Note the direction: 6, not 1000. A handful is exactly what a concurrent
contributor looks like, and it is what makes such a gate pass most of the time.
37 katgpt-core binaries use the macro; 14 have two or more tests and were
exposed. The gate had never run before Issue 713 armed it — a gate nobody could
run is a gate nobody debugged.

## The fix, and its constraints

The counter is **per-thread**, which is what an alloc gate means: *did this
code path, on this thread, allocate*. Constraints respected:

- **Call-site API kept** — `ThreadCounter` wears `load` / `fetch_add` / `store`
  with an accepted-and-ignored `Ordering`, so all 37 call sites compiled
  untouched.
- **The allocator hook must not allocate** — a `const`-initialised
  `Cell<usize>` read with `try_with`, so TLS teardown degrades to a dropped
  count rather than a panic inside `alloc`.
- **`--test-threads=1` is NOT the fix** — invisible at the call site, carried
  by no CI invocation, and would have to be remembered forever.

## Two-sided verification — the negative arm is the one that matters

| arm | result |
|---|---|
| plan414 G4, parallel | **ok, 3 of 3** (was 6 allocs / 1000) |
| same gate, body deliberately allocating | **FAILED — exactly 1000 allocs in 1000 calls** |

A counter that had silently become a no-op would have turned all 37 alloc gates
green over zero measurement — Issue 713's shape one level down, and strictly
worse. `assert_counter_is_live()` exists so each gate proves its instrument can
still fail before measuring.

T3's sweep drew the distinction worth keeping: the 13 at-risk gates never
observed failing were silently **unverified**, not silently broken.

## Provenance note — the code landed in a sibling's commit

T2's two files were staged when a concurrent session committed the whole index
(`62911111`, "docs: doc-sync run 2026-09-03"). Nothing was lost, but `git log`
attributes an allocator change to a docs run. Deliberately not repaired by
rewriting history (already on `origin/develop`, four sessions active). This is
`.issues/709`'s shape; `scripts/staged_set_audit.py` audits what YOU are about
to commit, not what someone else is about to commit around you.

Related memory: *"alloc-budget gates conflate two populations"* — a budget
asserted over a window that contains more than the thing being budgeted.
