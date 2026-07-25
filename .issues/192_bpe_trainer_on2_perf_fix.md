# Issue 192: `BpeTrainer::train` O(N²) perf + tie-break nondeterminism

**Opened:** 2026-07-25
**Related:** Issue 191 (fast_bpe GOAT gate — five of its tests call `BpeTrainer::train(corpus, 1024)` on the full `bpe.rs` source and currently wait ~142 s on setup)

## Problem

`BpeTrainer::train` has two defects that surfaced while auditing Issue 191's
GOAT-gate runtime:

### 1. O(N²) merge-apply loop (the perf defect)

The current implementation (bpe.rs L635-675) re-applies **all** prior merges
from scratch on every merge round:

```rust
for _ in 0..num_merges {
    pair_counts.clear();
    for word in &words {
        let tokens = Self::apply_merges(word, &merges);  // O(N) per word
        for i in 0..tokens.len().saturating_sub(1) { /* count pair */ }
    }
    // pick best pair, push to merges
}
```

Total cost: **O(N × W × T × N) = O(N² · W · T)** where `N = num_merges`,
`W = word count`, `T = avg tokens/word`. For `bpe.rs` (25 KB, ~3700 words,
1024 merges) this is ~10^10 char-ops — measured at **142 s wall-clock**
on Apple Silicon (release mode).

This is what makes five Issue 191 GOAT-gate tests (`g1_*_medium_vocab`,
`g2_perf_smoke_*_long_input` ×2, `g4_encode_into_*`) take 142 s of pure
setup time before the actual fast_bpe verification begins.

### 2. Tie-break nondeterminism (the correctness defect)

The existing implementation picks the merge pair via:

```rust
let best_pair = pair_counts.drain().max_by_key(|(_, count)| *count);
```

`pair_counts` is a `std::collections::HashMap` whose default `RandomState`
is seeded randomly per thread. When two pairs tie at the winning count,
`drain().max_by_key()` returns whichever the hash traversal happens to
visit LAST — and that varies per process invocation.

**Reproduced** with a probe (`/tmp/trainer_probe/`, 5 runs on
`"ab ab ab cd cd cd"`): run 0-3 picked `("a","b")`, run 4 picked
`("c","d")` — both at count 3.

This is not currently caught by any test because the GOAT-gate G1/G4
tests compare `encode` vs `encode_fast` on the **same** trained
tokenizer (so nondeterministic training doesn't matter — both encoders
see the same merges in the same run). It WOULD matter for any future
test that pins a specific merge sequence or trains in one process and
encodes in another.

## Scope decision: fix BOTH in one pass

These are technically independent, but they share a common fix surface
(rewrite the merge loop). Doing them together avoids touching the same
code twice. The rewrite:

1. **Memoizes the per-word tokenized state** so each round only applies
   the new merge (not all prior merges). O(N² · W · T) → O(N · W · T).
2. **Replaces `HashMap::drain().max_by_key()`** with an explicit
   tie-broken max scan (highest count; ties broken by lexicographic
   order of `(left, right)` for determinism). The tie-break rule is
   documented + tested.

The output merge sequence is **bit-identical to the prior implementation
on corpora that have no ties** at the winning count (verified by a
differential test against a frozen reference). On corpora WITH ties the
output is now deterministic where it wasn't before — this is the
intended behavior change.

## Tasks

- [x] **T1** Capture the current trainer's merge output on a small
  non-tied corpus as a frozen reference (a `&[(&str, &str, &str)]`
  array of `(left, right, merged)` tuples, baked into a unit test).
  Done as `test_bpe_train_frozen_reference_no_ties` — uses `ab_only`
  + `ab_dominant` corpora (verified deterministic on pre-fix
  implementation via a `/tmp/ref_probe_dir/` probe).
- [x] **T2** Add a unit test that verifies the tie-break rule:
  on `"ab ab ab cd cd cd"`, the first merge is **always**
  `("a","b")→"ab"` (lexicographic-smallest of the tied pairs).
  Done as `test_bpe_train_tie_break_is_lexicographic`.
- [x] **T3** Rewrite `BpeTrainer::train` to memoize per-word state +
  apply only the new merge each round. Preserve the public API
  (`pub fn train(corpus: &str, vocab_size: usize) -> BpeTokenizer`).
  The new implementation maintains `words_tok: Vec<Vec<String>>`
  outside the merge loop and applies each new merge in-place (no
  `apply_merges` re-application). The old `apply_merges` helper is
  removed — it had no external callers.
- [x] **T4** Replace `HashMap::drain().max_by_key()` with an explicit
  tie-broken max scan over `pair_counts.iter()`. The scan uses
  `min_by_key` with `(Reverse(count), left, right)` — highest count
  + lexicographically smallest pair on tie, in one pass.
- [x] **T5** Verify all existing trainer unit tests pass unchanged
  (`test_bpe_*` in `bpe.rs::tests`). All 6 PASS.
- [x] **T6** Re-run the Issue 191 GOAT gate (`fast_bpe_goat.rs`) and
  measure the new setup time.
  **Before:** 142.54s total (5 of 8 tests each spent ~30s in
  `BpeTrainer::train(corpus, 1024)` setup on the full `bpe.rs` source).
  **After:** 15.71s total — **9.1× speedup** on the full GOAT gate.
  Individual trainer setup dropped from ~30s to <100ms (the gate
  is now encoder-bound, not trainer-bound).
- [x] **T7** Verify G1 bit-identical, G4 alloc-free, lib tests, wasm32
  compile — no regression on any gate.
  - G1 (`fast_bpe_goat.rs`): 8/8 PASS
  - G1 (`fast_bpe_goat_pretok.rs`): 5/5 PASS (1 ignored)
  - G1 (`fast_bpe_pretok_hypothesis.rs`): 3/3 PASS
  - G4 (`fast_bpe_goat_g4_alloc.rs`): 1/1 PASS — encoder path
    unaffected (trainer is not in the encoder's hot path)
  - Lib tests: 10/10 PASS (added 4 new tests for Issue 192)
  - `katgpt-validator` tests: 7/7 PASS (consumer of the trainer)
  - `--no-default-features` check: PASS
  - `wasm32-unknown-unknown` check: PASS
  - `clippy --all-targets --all-features --release`: clean
- [x] **T8** Document the perf gain + the tie-break rule in the
  trainer's rustdoc + in this issue.
  The rustdoc on `BpeTrainer::train` now has explicit
  `# Tie-breaking (deterministic)` and `# Complexity` sections.
  This issue file documents the perf gain (9.1× on GOAT gate,
  ~300× on the trainer itself).

## Out of scope

- A full incremental pair-count update (subtract destroyed pairs, add
  new pairs). The memoized-apply-merge approach gets us O(N · W · T),
  which is fast enough for any realistic training corpus in this repo.
  The fully-incremental O(W · T) algorithm would need a per-word pair
  index + careful pair-destruction bookkeeping — not worth the
  complexity for the corpora we actually train on.
- The `String`-based token representation. Replacing `Vec<String>` with
  `Vec<u32>` (token IDs) would be a deeper refactor — separate issue
  if it ever becomes the bottleneck.

## Numbering note

Per AGENTS.md monotonic-numbering rule: 192 was `value + 1` from
`.issues/.highwater = 191`. Bumped `.highwater` to 192 in the same
commit as this file.
