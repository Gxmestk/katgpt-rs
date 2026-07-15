# Issue 153 — Numbering Collision Inventory (research/plans)

**Status:** OPEN
**Scope:** Mechanical refactor — renumber colliding `.research/` and `.plans/` files + update all references.
**Prior fix:** `.research/423_GPU_Tile_Sim_*` → `427_GPU_Tile_Sim_*` (resolved 2026-07-15, this session — the only collision flagged by the Spectral Rewiring duplicate-research session).

## Context

The repo-wide numbering discipline rule (per `katgpt-rs/AGENTS.md`):
> "Issue, plan, doc, benchmark, and research numbers are monotonic and never reused."

A scan on 2026-07-15 found **24 remaining number collisions** (13 in `.research/`, 11 in `.plans/`, 0 in `.issues/`). Each collision is a past violation where two files were created with the same number. The fix is deterministic: the file created FIRST (per `git log --diff-filter=A`) keeps the number; the interloper is renumbered to the next available number above the current `.highwater`, and all inbound references are updated.

## Collision inventory

### `.research/` (13 collisions, `.highwater` = 427)

| # | File A | File B |
|---|---|---|
| 050 | `050_LDT_Lattice_Deduction_Transformer.md` | `050_PFlash_Compression_Adaptive_Decode_Budget.md` |
| 119 | `119_KPop_Adaptive_KL_Masking_RL_Training.md` | `119_PiD_Pixel_Diffusion_Decoder.md` |
| 131 | `131_DiffusionBlocks_Block_Wise_Training.md` | `131_UNSL_Unified_Neural_Scaling_Laws.md` |
| 145 | `145_SIA_Harness_Weight_Co_Evolution.md` | `145_Wall_Attention_Diagonal_Gate_RoPE_Replacement.md` |
| 156 | `156_Speculative_Reconciliation_Engine.md` | `156_Weight_Isolate_Extension_Protocol.md` |
| 211 | `211_Bayesian_Agent_Posterior_Guided_Skill_Evolution.md` | `211_LCLM_Latent_Context_Language_Model_Distillation.md` |
| 228 | `228_RCD_Residual_Context_Diffusion.md` | `228_TwinProp_Dendritic_Inference_Compute.md` |
| 243 | `243_Bebop_Entropy_Bounded_MTP_Acceptance_Adaptive_Gamma.md` | `243_Temporal_Derivative_Kernel_Neocortical_Learning.md` |
| 258 | `258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md` | `258_FlashMemory_Lookahead_Periodic_Sparse_Attention.md` |
| 265 | `265_CoFRe_FP_MGM_Three_State_Reuse.md` | `265_b_posit_tapered_precision_format.md` |
| 384 | `384_RePo_Context_Repositioning_Pass.md` | `384_Sheaf_ADMM_Multi_Agent_Coordination.md` |
| 410 | `410_NVFP4_RL_4over6_Adaptive_Block_Scaling_Pass.md` | `410_vllm_dynamic_speculative_decoding_pass.md` |
| 425 | `425_AIDE2_Recursive_Self_Improvement_PASS.md` | `425_symcrypt_verifiedcrypto_aeneas_methodology.md` |

### `.plans/` (11 collisions, `.highwater` = 441)

| # | File A | File B |
|---|---|---|
| 001 | `001_pruners_optimization.md` | `001_sudoku_9x9_example.md` |
| 003 | `003_perf_optimization.md` | `003_still_kv_heuristic_beta_optimization.md` |
| 054 | `054_and_or_flow_mux_optimization.md` | `054_stepcode_reasoner_modelless.md` |
| 135 | `135_hl_regularization_principles.md` | `135_parallax_attn.md` |
| 164 | `164_gepa_reflective_config_evolution.md` | `164_phrase_boost_context_trie.md` |
| 189 | `189_freq_bandit_phase1.md` | `189_oscillatory_state_space_modelless.md` |
| 272 | `272_chunked_asset_merkle_store.md` | `272_progressive_mcgs.md` |
| 293 | `293_action_bridge_lean4_monotonicity_proof.md` | `293_forensic_watermark_recipe_primitive.md` |
| 381 | `381_phase10_core_absorption.md` | `381_step_attribution_delta_qualification_primitive.md` |
| 390 | `390_mcts_state_action_cache_unmaskfork.md` | `390_speculative_phase5_prefill_substrate.md` |
| 431 | `431_cross_stage_residual_relocation_primitive.md` | `431_simd_lut_dequant.md` |

## Resolution protocol (per collision)

For each collision, the deterministic fix is:

1. `git log --diff-filter=A --format=%ai -- <fileA>` vs `<fileB>` → determine which was created first.
2. First-created keeps the number; interloper renumbers to `++highwater`.
3. Update the renumbered file's title line (`# Research NNN:` / `# Plan NNN:`).
4. `grep -rn` for all references to the old number/filename across `.research/`, `.plans/`, `.benchmarks/`, `README.md`, and `.rs` module doc-comments.
5. Update all references.
6. Write the new `.highwater` back.
7. Verify with `grep -rn "<old_filename>"` returning zero hits.

## Acceptance criteria

- [ ] T1: Resolve all 13 `.research/` collisions (renumber + update references)
- [ ] T2: Resolve all 11 `.plans/` collisions (renumber + update references)
- [ ] T3: Update `.research/.highwater` and `.plans/.highwater` to final values
- [ ] T4: `grep -rn` verification: zero stale references to any renumbered file
- [ ] T5: `cargo clippy` clean (in case any `.rs` doc-comments reference old numbers)
- [ ] T6: Update `.issues/.highwater` (done: 153)

## Priority

LOW — mechanical noise reduction. No functional impact. Batch-process when convenient; each collision is independent and can be resolved in isolation. The 423 collision (the only one flagged by research workflow) is already resolved.

## Risk

Each renumber requires updating all inbound references. The blast radius varies: some files have 0 inbound references (safe), others have 5-10 (e.g., the 423 collision had 6 reference sites across 2 plans + 1 benchmark). Missing a reference leaves a stale doc link — annoying but not functionally breaking.
