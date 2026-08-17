# Issue 670 — dd_tree `parent_path` 16-bit packing breaks at vocab > 65,536 (Bonsai = 248,320)

**Status:** Open
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

## Tasks

- [ ] Widen `TreeNode.parent_path` (u128 → `[u32; 8]` or equivalent) in katgpt-core.
- [ ] Update all `build_dd_tree*` path construction + `extract_parent_tokens*`.
- [ ] Fix the ~20 dd_tree builder variants + tests in katgpt-speculative.
- [ ] Update riir-engine consumers (the re-exported seam).
- [ ] GOAT G1–G4 + Bench 693 rerun.

## Notes

- Keep `score`, `depth`, `token_idx` as-is; only the path representation widens.
- `u128::from_be_bytes`-style helpers can preserve the old packing semantics
  for the ≤65,536 case if a migration shim is wanted for external consumers —
  but the repo-internal break can be one coordinated commit.
