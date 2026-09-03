# Issue 714 — alloc gates count their SIBLING tests' allocations

**Status:** **CLOSED 2026-09-03 — all three tasks done.** T1 MEASURED +
ROOT-CAUSED, T2 LANDED (code in `62911111`, see the note below), T3 **RAN**
over all 37 binaries on a quiet box: **35 pass / 0 fail**, the 2 non-results
being `fn main()` program-style targets. Kept rather than removed because the
reusable part is the *distinction* T3 drew — the 13 at-risk gates were
silently **unverified**, not silently broken.

> **T2's code landed inside a SIBLING's commit, `62911111` ("docs: doc-sync
> run 2026-09-03"), not under a message describing it.** My two files were
> staged when a concurrent session in this worktree committed, and a
> whole-index commit takes the whole index. Nothing was lost — HEAD carries
> `ThreadCounter`, `assert_counter_is_live`, and plan414's canary wiring
> exactly as verified — but `git log` now attributes an allocator change to a
> docs run, and the two-sided verification below would have been unrecorded.
> This issue is that record.
>
> Deliberately **not** repaired by rewriting history: `62911111` is already on
> `origin/develop` and four sessions are active in this worktree. This is
> `.issues/709`'s shape, and `staged_set_audit.py` is the instrument for it —
> it ran clean on my own staged pair minutes before, which is the honest
> limit of that tool: it audits what YOU are about to commit, not what someone
> else is about to commit around you.
Found by Issue 713's sweep, which is the first time several of these gates had
ever executed.

## The defect

`crates/katgpt-core/tests/common/mod.rs`'s `counting_allocator!()` installs a
`#[global_allocator]` incrementing a **process-global** `ALLOC_COUNT`. An
alloc gate then measures:

```rust
let before = ALLOC_COUNT.load(SeqCst);
for _ in 0..1000 { hot_path(); }
let after = ALLOC_COUNT.load(SeqCst);
assert_eq!(after - before, 0);
```

`cargo test` runs a binary's tests **on parallel threads by default**. Every
allocation any *sibling* test makes during that window lands inside `after -
before`. The gate is not measuring the hot path; it is measuring the hot path
**plus whatever else the binary happened to be doing**.

## Measured

`plan414_hla_committed_belief_probe_goat::g4_zero_alloc`:

| invocation | result |
|---|---|
| default (parallel) | **FAILED — 6 allocs in 1000 probe calls, expected 0** |
| `-- --test-threads=1` | **ok, 3 runs of 3** |

The sibling is `g5_latency`, which times the same probe and allocates while
doing it. The product path allocates **zero** times; the gate said otherwise.

Note the direction: 6, not 1000. A per-call allocation would be obvious. A
handful is exactly what a *concurrent* contributor looks like, and it is what
makes this pass most of the time.

## Exposure — 37 binaries, ~14 at risk

`counting_allocator!()` is used by **37** test binaries in `katgpt-core`.
A binary with a **single** `#[test]` has no sibling and is safe. A binary with
**two or more** can race, and 14 do:

`bench_406_renoise_ce_goat` (5 tests), `bench_412_subspace_steering_goat` (4),
`bench_656_privilege_alloc_check` (3), and 2 tests each in
`sterling_alloc_check`, `switch_cost_alloc_check`,
`engram_tripwire_alloc_check`, `signed_coupling_alloc_check`,
`contrastive_scope_alloc_check`, `bench_680_kinematic_alloc_check`,
`conformal_alloc_check`, `effective_degree_alloc_check`,
`recirculation_alloc_check`, `group_invariance_probe_g4`,
`plan414_hla_committed_belief_probe_goat`.

**Being at risk is not the same as failing.** Whether a given gate trips
depends on whether a sibling allocates inside the measured window — which is
scheduling, so it is flaky, so a gate can pass for months and then fail once.
Only `plan414` was observed failing; the other 13 are unmeasured and this issue
does not claim they are broken.

## Why it went unnoticed

`plan414_hla_committed_belief_probe_goat` is one of the 39 targets Issue 713
armed. Before `180be9c5` it carried `#![cfg(feature =
"hla_committed_belief_probe")]` with **no** `required-features` row, so
`cargo test --test plan414_...` compiled it to nothing and printed
`ok. 0 passed` with exit 0. **This is the first time the gate ran.** The
two issues are one story: a gate nobody could run is a gate nobody debugged.

## The fix (T2)

Make the counter **thread-local**. That is not a workaround for the harness —
it is what the gate actually means: *did this code path, on this thread,
allocate*. A process-global count answers a question no alloc gate has ever
wanted to ask.

Constraints the fix must respect:

- **Keep the call-site API.** 37 files do `ALLOC_COUNT.load(Ordering::SeqCst)`.
  Changing the type breaks all of them, so the replacement must keep a `load`
  (and `fetch_add`) shaped like the atomic's.
- **The allocator hook must not allocate.** A lazily-initialised
  `thread_local!` can allocate on first touch, from inside `alloc` — infinite
  recursion. Use a `const`-initialised `Cell<usize>` and `try_with`, so TLS
  destruction during thread teardown degrades to a dropped count rather than a
  panic in the allocator.
- **Two-sided verification.** The fix must make `plan414` pass *in parallel*,
  and must still FAIL on a deliberately-allocating body. A counter that has
  silently become a no-op passes every alloc gate in the repo — which is the
  Issue 713 shape again, one level down, and strictly worse because it would
  make 37 gates vacuous at once.
- `-- --test-threads=1` is **not** the fix. It is invisible at the call site,
  no CI invocation carries it, and it would have to be remembered forever.

## Tasks

- [x] **T1** Root-cause + measure exposure. DONE 2026-09-03.
- [x] **T2 LANDED 2026-09-03** (code in `62911111`, see the status note).
  `ThreadCounter` wears the `AtomicUsize` API (`load` / `fetch_add` / `store`,
  `Ordering` accepted and ignored) so all 37 call sites compile untouched;
  the TLS slot is `const`-initialised and read with `try_with`.
  `assert_counter_is_live()` added and wired into plan414's G4.

  **Verified two-sided, by execution:**

  | arm | result |
  |---|---|
  | plan414 `g4_zero_alloc`, PARALLEL (default) | **ok, 3 runs of 3** (was 6 allocs/1000) |
  | same gate, body deliberately allocating | **FAILED — exactly 1000 allocs in 1000 calls** |

  The negative arm is the one that matters: it proves the counter is still
  capable of failing. A counter that had silently become a no-op would have
  turned all 37 alloc gates green over zero measurement.
- [x] **T3 RAN 2026-09-03 — all 37 binaries, on a quiet box.** Every one that
  is a test target passes: **35 pass, 0 fail.** The two non-results are
  `bench_331_babel_codec_goat` and `bench_360_engram_staging_goat`, which carry
  `fn main()` and no `#[test]` — program-style targets that emit no
  `test result:` line. Not a defect and not affected by this change.

  So the T2 refactor is confirmed non-breaking across every consumer of the
  macro, and the 13 previously-unmeasured at-risk gates now have a verdict:
  **all green.** None of them was silently broken; they were silently
  *unverified*, which is the distinction this issue exists to draw and the
  reason T2's negative arm mattered more than its positive one.

## Related

- `.issues/713` — the silent-0-pass class that hid this gate. Same story.
- The memory note *"alloc-budget gates conflate two populations"* is this
  defect's general form: a budget asserted over a window that contains more
  than the thing being budgeted.
