# Issue 714 — alloc gates count their SIBLING tests' allocations

**Status:** T1 **MEASURED + ROOT-CAUSED 2026-09-03.** T2 (the fix) open.
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
- [ ] **T2** Thread-local counter behind the existing `ALLOC_COUNT` surface,
  plus the liveness canary above. Verify `plan414` parallel-green 3×, and
  re-run the other 13 at-risk binaries.
- [ ] **T3** Once T2 lands, the 13 unmeasured at-risk gates get a real verdict
  for the first time. Any that then fail are separate findings, not T2
  regressions — the same rule Issue 713 T2 set.

## Related

- `.issues/713` — the silent-0-pass class that hid this gate. Same story.
- The memory note *"alloc-budget gates conflate two populations"* is this
  defect's general form: a budget asserted over a window that contains more
  than the thing being budgeted.
