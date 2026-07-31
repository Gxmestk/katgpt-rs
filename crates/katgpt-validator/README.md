# katgpt-validator

Two-tier syntax pruner for inference-time code validation — `PartialParser`
(bracket-balancer DFA) + `SynPruner` (syn parse). Extracted from
`katgpt-rs/src/validator/` per Proposal 003 Phase 11 (2026-07-04).

## Overview

A cheap-then-accurate pipeline that filters drafted code tokens before they
reach the verifier:

- **Tier 0 — `PartialParser`**: an `O(n)` bracket-balancer DFA. Rejects clearly
  broken code cheaply. Never false-accepts unbalanced brackets.
- **Tier 1 — `SynPruner`**: `syn::parse_str::<syn::Stmt>` accurate parse.
  Only called if Tier 0 passes. Implements `katgpt_core::ConstraintPruner`.

## Key types / modules

- `partial_parser` — `PartialParser` (the Tier-0 bracket-balancer DFA).
- `syn_pruner` — `SynPruner` (the Tier-1 accurate parse). Implements
  `katgpt_core::ConstraintPruner`.
- `types` — `CompilerFeedback`, `ErrorKind`, `PruneResult`.

## Feature flags

`default = []`.

| Feature | Description |
|---|---|
| `validator` | Back-compat feature name (root forwards to it). The crate's modules compile unconditionally; this feature exists for parity with the historical root feature surface. |
| `hoare_pruner` | Gates the optional `ConstraintPruner::propagate` impl on `SynPruner` that does Hoare-style predicate checking during DDTree expansion. Root's `hoare_pruner` feature forwards here so the method compiles when both `validator` and `hoare_pruner` are on. NOTE: this is a separate concept from root's predicate-propagation `hoare_pruner` (which gates `llmexec_guard` + `katgpt-pruners/hoare_pruner`) — they share the feature name but not the code path. The shared name is preserved for back-compat. |

## Dependencies

- `katgpt-core` — `traits::ConstraintPruner` (the trait `SynPruner`
  implements). Always-on.
- `katgpt-tokenizer` — `BpeTokenizer` (struct field) +
  `BpeTokenizerImpl::decode` (prod decode path). Always-on — the validator's
  `SynPruner::is_valid` decodes the token sequence back to source code before
  parsing.
- `syn` — third-party Rust parser, used for the Tier-1 accurate parse
  (`syn::parse_str::<syn::Stmt>`). Always-on.
- `proc-macro2` — transitive requirement of `syn` for span/token tracking.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
