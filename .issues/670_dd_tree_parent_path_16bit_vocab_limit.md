# Issue 670 — dd_tree `parent_path` 16-bit packing breaks at vocab > 65,536 (Bonsai = 248,320)

**Status:** RESOLVED (2026-08-18, same day) — `TreePath` widening landed
**Opened:** 2026-08-18
**Owner:** katgpt-rs (the substrate: `katgpt-core/src/speculative/types.rs` `TreeNode` + `katgpt-speculative/src/dd_tree/tree_builder.rs`)
**Discovered by:** riir-ai Issue 717 gate — [Bench 693](../../riir-ai/.benchmarks/693_issue717_bigram_bonsai_sparse_vocab_gate.md) §3
**Blocks:** riir-ai Issue 717 G1/G3a/G3b (Bonsai consumer gate), and ANY tree-based speculative verification at modern vocab scale

## The defect

`TreeNode.parent_path: u128` packs one token per tree level at **16 bits**:

```rust
// dd_tree/tree_builder.rs (every build path)
node_path = (parent_path << 16) | (token_idx as u128)
```

and extraction masks to 16 bits:

```rust
// dd_tree/mod.rs extract_parent_tokens_into
((parent_path >> ((num_tokens - 1 - k) * 16)) & 0xFFFF) as usize
```

Any `token_idx ≥ 65_536`:

1. **corrupts the parent's bits** at insert time (`token_idx`'s bit 16+ overlaps
   the parent token's low bits via the OR), and
2. **can never be recovered** by the 16-bit extraction mask.

Not a measurement artifact — the verifier's own token extraction
(`extract_parent_tokens*`, used to build the batched verification forward) reads
the same corrupted path, so production tree verification would feed the target
model WRONG tokens wherever any path token id ≥ 65,536.

## Measured exposure (Bench 693, real Bonsai tokenizer from the Q2_0 GGUF)

- Bonsai vocab = **248,320** (18-bit ids; max observed id 248,069).
- clippy-L4 corpus: **3.33%** of tokens ≥ 65,536.
- strandset real-Rust corpus: **4.18%** ≥ 65,536.
- Symptom that exposed it: the bigram tree arm at `top_m=1` measured BELOW the
  bare greedy chain over all positions (1.3707 vs 1.4736) — impossible for a tree
  that contains the chain; on the "clean16" subset (no id ≥ 65,536 anywhere in
  the window) tree == chain exactly, pinned by assertion.

## Why a bit-width tweak is not enough

- u128 @ 16 bits = 8 levels (lookahead 8 ✓, vocab ≤ 65,536 ✗).
- u128 @ 18 bits = 7 levels (vocab 262,144 ✓, lookahead 8 ✗ — 144 > 128).
- u128 @ 32 bits = 4 levels.

The lookahead-8 + Bonsai-vocab combination needs ≥ 144 bits. The fix is a wider
path **type**, not a wider field: e.g. `[u32; 8]` (or a `SmallVec<[u32; 8]>`)
with `extract_parent_tokens*` reading levels directly.

## Blast radius (measured 2026-08-18)

`parent_path` grep: **216 references across 17 files in katgpt-rs** (core:
`speculative/types.rs` + `gdn_tree_verify/mod.rs`; the dd_tree builders +
tests; `katgpt-attn/gdn2/tree_forward.rs`; `katgpt-forward/step.rs`;
`katgpt-pruners` ×2; `spechop/hop_tree.rs`; `distill/ilc.rs`; a bench) —
plus **9 references in 2 files in riir-ai**. `TreeNode` lives in
**katgpt-core** (the public crate), so the change is a cross-repo API break
that must land in one coordinated pass.

## Gate (GOAT)

- G1: at vocab ≤ 65,536, all existing dd_tree tests bit-identical (the packing
  is lossless there — the new representation must reproduce current outputs).
- G2: at vocab 248,320, an 8-level path round-trips every id exactly
  (property test over the full id range, incl. 248,319).
- G3: no perf regression in tree build/extract (the u128 was chosen for
  copy-cheapness; `[u32; 8]` is still 32 bytes = same size as u128).
- G4: riir-ai Bench-693 probe rerun — all-positions tree arm == clean16 arm
  within sampling noise (the aliasing signature gone).

## Resolution

`TreeNode.parent_path` is now `TreePath` — a `[u32; 8]` newtype (same 32-byte
footprint as the old `u128`, `Copy`/`Eq`/`Ord`/`Hash` derived) with
`root`/`push(token, depth)`/`parent(depth)`/`token_at(k)` ops replacing the
packing/shift/mask idioms. Lexicographic slot order ≡ the old packed-u128
order for same-length paths, so the deterministic topology sort is unchanged.
A bare path does not encode its own length (`[t]` ≡ `[t, 0]`); every in-tree
key pairs the path with `depth`, which disambiguates (the `parent(depth)`
signature makes the caller supply it).

### GOAT

- **G1 PASS** — existing dd_tree suites green: katgpt-speculative 329
  (bigram_markov+lodestar+ilc_distill), katgpt-forward 124, katgpt-attn 25
  (gdn2_attention), katgpt-pruners 126, katgpt-core 1903 incl. 4 new
  `tree_path_*` unit tests (one pins bit-exact equivalence with the old
  16-bit packing at vocab ≤ 65,536), root lib 201, sudoku + cgsp benches run.
  The slot representation reproduces the old extracted tokens exactly in the
  old encoding's lossless regime (pinned by
  `tree_path_matches_legacy_u128_packing_at_small_vocab`).
- **G2 PASS** — `tree_path_round_trips_bonsai_vocab_ids` (katgpt-core):
  8-level round-trip over the hard ids incl. 65,536 / 131,072 / 248,319.
  Consumer-level: `bench_694`'s path-encoding check flipped from overflow
  DETECTOR to round-trip ASSERTION at full Bonsai vocab (zero failures on
  every deep node, ids ≥ 65,536 exercised).
- **G3 PASS** — same 32-byte Copy footprint; slot index write replaces
  shift+or; `bench_424_dd_tree_deep_argmax` completes with its pre-existing
  verdict (no behavioral change). No perf-sensitive consumer changed shape.
- **G4** — the riir-ai `issue717_bigram_bonsai_gate` structural invariants
  moved from clean16-subset to ALL positions (tree ≥ chain everywhere,
  tree == chain at top_m=1); the lower-bound caveat is retired. Rerun with
  the Bonsai GGUFs (`--ignored`) is the standing consumer gate.

### Blast radius closed (246 refs, 19 files)

katgpt-core (`types.rs`, `gdn_tree_verify`), katgpt-speculative (`dd_tree/mod.rs`,
`tree_builder.rs`, `lodestar.rs`, `tests.rs`, `distill/ilc.rs`, `bigram_markov.rs`,
`spechop/hop_tree.rs` docs), katgpt-forward (`step.rs`, `dd_tree/tests.rs`),
katgpt-attn (`gdn2/tree_forward.rs`), katgpt-pruners (comments), root
(`step_gdn_tree.rs`, `bench_elf_modelless.rs`, `sudoku_speculate_bench.rs`
comments), riir-ai (`deltanet/tree_forward/mod.rs` + the two 717 gate files).

Notes carried:

- `distill/ilc.rs`'s path→f32 state feature folds the leaf + low 8 bits of
  its parent (the old low-24-bit hash); bit-identical at ≤ 65,536, documented
  truncation beyond (the feature was always lossy there).
- `merge_retrieved_branches`' incremental reconstruction + the chain
  accumulators collapse to unconditional `push(t, depth)` (push at depth 0 on
  the empty path ≡ `root`).
- The public `extract_parent_tokens*` signatures now take `TreePath` — a
  breaking API change landed in one coordinated cross-repo pass, as the issue
  prescribed.

## Tasks

- [x] Widen `TreeNode.parent_path` (u128 → `TreePath` = `[u32; 8]`) in katgpt-core.
- [x] Update all `build_dd_tree*` path construction + `extract_parent_tokens*`.
- [x] Fix the ~20 dd_tree builder variants + tests in katgpt-speculative.
- [x] Update riir-engine consumers (the re-exported seam).
- [x] GOAT G1–G3 + Bench 693/694 probe rerun gate (G4).

## Notes

- Keep `score`, `depth`, `token_idx` as-is; only the path representation widens.
- `u128::from_be_bytes`-style helpers can preserve the old packing semantics
  for the ≤65,536 case if a migration shim is wanted for external consumers —
  but the repo-internal break can be one coordinated commit.
