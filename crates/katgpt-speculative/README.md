# katgpt-speculative

Shared speculative decoding + DDTree substrate. Issue 013 (2026-06-29)
collapsed the fork between `katgpt-rs/src/speculative/` and
`riir-engine/src/{dd_tree, dflash}.rs`. The core DDTree algorithm lives here
so improvements propagate to both consumers.

## What lives here

- **DDTree core** (`dd_tree`) — `build_dd_tree`, `build_dd_tree_pruned`,
  `build_dd_tree_screened`, `build_dd_tree_balanced`, `TreeBuilder`,
  `extract_parent_tokens`, `extract_best_path`, `merge_retrieved_branches`,
  `find_valid_sequence`, etc. Pure algorithm over pre-computed marginals. No
  `forward` dependency.
- **DFlash zero-alloc cores** (`dflash`) — the three `_with` cores
  (Issue 013 Phase B). Generic over a `DflashCtx` + `DflashCache` backend
  trait pair and a `forward_fn` closure, because the underlying
  `ForwardContext` / `MultiLayerKVCache` / `TransformerWeights` types are
  crate-specific. The thin wrappers (`dflash_predict`, `_ar`, `_conditioned`,
  `_parallel`) and feature-gated variants (`_domino`, `_routing`, `_fusion`)
  stay in each consumer.
- **NF-Flow cluster** (`nf_flow`, `nf_flow_budget`, `nf_flow_fold`,
  `nf_flow_gate`, `nf_flow_mux`) — normalizing-flow routing substrate.
- **Pathway tracking / scheduling / vocab** — `pathway_tracker`,
  `prefix_scheduler`, `vocab_coreset`, `correlation_budget`,
  `branch_confidence`, `blueprint`, `decomp_reviewer`.

## What does NOT live here

- **Types** (`TreeNode`, `DraftResult`, `ConstraintPruner`, …) → already in
  `katgpt_core::speculative::types` + `katgpt_core::traits` (Plan 008 Phase
  2.5).
- **Sampling** (`sample_from_distribution`) → already in
  `katgpt_core::speculative::sampling` (Plan 008 Phase 2.6).
- **Feature-gated DDTree variants** (`build_dd_tree_belief`, `_speculative`,
  `_kurtosis`, `_domino`, `_manifold`, `_lodestar`, `_gdsd`, …) → stay in
  `katgpt-rs/src/speculative/dd_tree.rs` because they reference root-only
  sibling modules (`super::belief_drafter`, `super::spec_generator`, etc.).

## Feature flags

`default = []`. The crate has many opt-in features mirroring the historical
root speculative feature surface. Headline entries:

| Feature | Description |
|---|---|
| `sr2am_configurator` | SR²AM Configurator context types (Plan 112). |
| `lodestar` | Lodestar speculative drafter. |
| `ppot` | PPOT prompt-prediction-over-time drafter. |
| `ilc_distill` | Iterated Learning Curriculum distillation. |
| `spechop` | SpecHop stage-specialized decode paths (Plan 102). |
| `bandit` | Bandit-based drafter. |
| `recfm` | RecFM (receptive-field model) drafter. |
| `adaptive_causal_calibration` | Adaptive causal calibration. |
| `cache_prune` | Forwards `katgpt-kv/cache_prune` for speculative pruning. |
| `rt_turbo` | RT-Turbo drafter. |
| `precision_aware_draft` | Precision-aware drafting. |
| `spec_reconciliation` | Speculative reconciliation. |
| `thinking_prune` | Thinking-mode pruning. |
| `progressive_mcgs` | Progressive Monte-Carlo graph search. |
| `weaver_runtime` | Weaver runtime (speculative cycle). |

See `Cargo.toml` for the full feature list (additional tracking flags consumed
by the moved modules).

## Dependencies

- `katgpt-core` — shared traits + speculative types.
- `katgpt-types` — shared types.
- `rayon` — parallel kernels.
- `serde` *(optional)* — speculative state serialization.
- `postcard` *(optional)* — binary persistence format.
- `blake3` *(optional)* — integrity envelopes.
- `katgpt-kv` *(optional)* — KV-cache substrate (forwards `cache_prune`).
- `half` / `fastrand` / `papaya` / `bytemuck` / `safetensors` *(optional)* —
  feature-gated.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
