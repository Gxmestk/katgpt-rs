# katgpt-tokenizer

BPE, ToaST split-tree, and ConvexTok LP vocabulary optimization — standalone
modelless tokenizer crate extracted from `katgpt-rs/src/tokenizer/` (Issue
014, 2026-06-29). Leaf crate — no `katgpt-*` dependencies.

## Overview

Three tokenizers covering different points on the quality/speed/vocab-size
trade-off:

- **BPE** *(default)* — byte-pair-encoding encoder/decoder + trainer. The
  baseline, always compiled.
- **ToaST** (`toast_tokenizer`) — split-tree tokenization (Plan 122,
  Research 081) + Double-Array Trie vocab lookup, auto-routed above
  `DATRIE_VOCAB_THRESHOLD` (Plan 137).
- **ConvexTok** (`convex_tok`) — LP vocabulary optimizer (Plan 127, Research
  087). Implies `toast_tokenizer` and pulls `good_lp` (HiGHS + microlp).

## Key types / modules

- `bpe` — `BpeTokenizerImpl`, `BpeTrainer` (the encoder/decoder pair).
- `types` — `BpeTokenizer`, `MergeRule`.
- `toast_builder` *(gated `toast_tokenizer`)* — `SplitTreeBuilder`.
- `toast_inference` *(gated `toast_tokenizer`)* — `ToastTokenizerImpl`.
- `toast_types` *(gated `toast_tokenizer`)* — `SplitTree`, `SplitNode`,
  `ToastTokenizer`, `DATRIE_VOCAB_THRESHOLD`.
- Double-Array Trie vocab lookup *(gated `toast_tokenizer`)* — auto-built
  when vocab > `DATRIE_VOCAB_THRESHOLD`. Threshold-routed: no separate
  feature gate needed.
- ConvexTok LP optimizer *(gated `convex_tok`)* — vocabulary optimization
  via the LP solver.

## Feature flags

`default = []`.

| Feature | Description |
|---|---|
| `toast_tokenizer` | ToaST split-tree tokenization (Plan 122) + Double-Array Trie vocab lookup (Plan 137). |
| `convex_tok` | ConvexTok LP vocabulary optimizer (Plan 127, Research 087). Implies `toast_tokenizer` and pulls `good_lp` (HiGHS + microlp). |
| `datrie_vocab` | Alias for `toast_tokenizer` (kept for back-compat with the katgpt-rs feature surface). |

## Dependencies

- `serde` — tokenizer struct serialization (always-on).
- `good_lp` *(optional)* — ConvexTok LP solver (HiGHS primary + microlp
  fallback). Gated by `convex_tok`.

No `katgpt-*` dependencies — this is a leaf crate.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
